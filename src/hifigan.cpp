#include "pc_nsf_hifigan/hifigan.h"

#include <mininsf/mininsf.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace pc_nsf_hifigan {

namespace {

// F32 im2col + mul_mat 1D conv: avoids ggml_conv_1d's F16 im2col on CPU and
// is also much faster on Vulkan (ggml mul_mat is properly optimized there).
// Returns the conv output reshaped to [OL, OC].
ggml_tensor * conv1d_f32(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x,
                         int pad, int dil) {
    ggml_tensor * im = ggml_im2col(ctx, w, x, 1, 0, pad, 0, dil, 0, false, GGML_TYPE_F32);
    ggml_tensor * m0 = ggml_reshape_2d(ctx, im, im->ne[0], (im->ne[2] * im->ne[1]));
    ggml_tensor * wm = ggml_reshape_2d(ctx, w, (w->ne[0] * w->ne[1]), w->ne[2]);
    ggml_tensor * r = ggml_mul_mat(ctx, m0, wm);
    r = ggml_reshape_3d(ctx, r, im->ne[1], w->ne[2], im->ne[2]);
    return ggml_reshape_2d(ctx, r, r->ne[0], r->ne[1]);
}

// Default policy: use conv1d_f32 on every backend except CUDA (where the stock
// ggml conv1d kernel is already optimized).  PCNSF_MANUAL_CONV=1/0 overrides.
inline bool use_manual_conv(const GGUFModel & m) {
    const char * pcnsf_manual = getenv("PCNSF_MANUAL_CONV");
    const char * backend_name = ggml_backend_name(m.backend);
    const bool is_cuda = backend_name && std::strstr(backend_name, "CUDA");
    return pcnsf_manual ? pcnsf_manual[0] != '0' : !is_cuda;
}

// Conv1d "same" (pad = (K/2)*dilation, matching torch get_padding for odd K),
// kernel layout [K, IC, OC], data layout [T, IC] -> [T', OC].
// `ctype` selects the storage precision of the kernel: GGML_TYPE_F32 (exact
// golden line) or GGML_TYPE_F16 (fp16 line, reserved for future fp16/bf16
// trained checkpoints).  ggml's 1D conv computes in fp32 on CPU, so
// activations/bias stay fp32 regardless; the fp16 line therefore means
// "weights fp16, compute fp32" (the same contract as the fp16/bf16 line).
ggml_tensor * conv1d_same(ggml_context * ctx, const GGUFModel & m, ggml_type ctype,
                          const std::string & prefix, ggml_tensor * x, int dilation = 1) {
    ggml_tensor * w = gguf_get(m, prefix + ".weight");
    if (w->type != ctype) w = ggml_cast(ctx, w, ctype);
    const int pad = (int) (w->ne[0] / 2) * dilation;
    // F32 im2col+mul_mat is faster than ggml's F16-im2col conv1d on CPU/Vulkan
    // (CPU 21.8->17.7s, Vulkan 18.7->2.56s for a 20s clip); CUDA keeps stock.
    ggml_tensor * y;
    if (use_manual_conv(m)) {
        y = conv1d_f32(ctx, w, x, pad, dilation);
    } else {
        y = ggml_conv_1d(ctx, w, x, 1, pad, dilation);
        y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    }
    ggml_tensor * b = ggml_reshape_2d(ctx, gguf_get(m, prefix + ".bias"), 1, y->ne[1]);
    return ggml_add(ctx, y, b);
}

// Conv1d 1x1 via mul_mat (source_conv): kernel [1, IC, OC] -> [OC, T].
// ggml mul_mat requires B (activations) fp32; the fp16 line keeps fp32
// activations and only casts the kernel weights to fp16.
ggml_tensor * conv1d_k1(ggml_context * ctx, const GGUFModel & m, ggml_type ctype,
                        const std::string & prefix, ggml_tensor * x) {
    ggml_tensor * w = gguf_get(m, prefix + ".weight");
    if (w->type != ctype) w = ggml_cast(ctx, w, ctype);
    ggml_tensor * w2 = ggml_reshape_2d(ctx, w, w->ne[1], w->ne[2]);  // [IC, OC]
    ggml_tensor * xt = ggml_cont(ctx, ggml_transpose(ctx, x));       // [IC, T]
    ggml_tensor * y = ggml_mul_mat(ctx, w2, xt);                     // [OC, T]
    y = ggml_cont(ctx, ggml_transpose(ctx, y));                      // [T, OC]
    ggml_tensor * b = ggml_reshape_2d(ctx, gguf_get(m, prefix + ".bias"), 1, y->ne[1]);
    return ggml_add(ctx, y, b);
}

// ConvTranspose1d with torch padding=(K-stride)/2: ggml conv_transpose_1d
// asserts p0 == 0, so run p0=0 and crop `pad` samples from the left.
ggml_tensor * conv_transpose1d_crop(ggml_context * ctx, const GGUFModel & m,
                                    ggml_type ctype, const std::string & prefix,
                                    ggml_tensor * x, int stride, int kernel) {
    ggml_tensor * w = gguf_get(m, prefix + ".weight");
    if (w->type != ctype) w = ggml_cast(ctx, w, ctype);
    ggml_tensor * y = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);
    y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    const int pad = (kernel - stride) / 2;
    const int64_t target_len = x->ne[0] * stride;
    if (pad > 0) {
        y = ggml_view_2d(ctx, y, target_len, y->ne[1], y->nb[1], pad * y->nb[0]);
        y = ggml_cont(ctx, y);
    }
    ggml_tensor * b = ggml_reshape_2d(ctx, gguf_get(m, prefix + ".bias"), 1, y->ne[1]);
    return ggml_add(ctx, y, b);
}

// Sub-pixel upsample: an EXACT, backend-portable replacement for
// ggml_conv_transpose_1d (conv1d + phase interleave).  The converter emits
// `hifigan.upsub.N.weight` with channels ordered PHASE-MAJOR (c = r*Cout + o)
// and TILED bias (bias.repeat(s)); the interleave below yields
//   z    = conv1d(W_sub, x, pad = M-1)
//   out[t*s+r, o] = z[t, r*Cout + o]          via permute/cont/reshape
// This matches torch ConvTranspose1d bit-for-bit (fp32), and works on every
// backend that supports ggml_conv_1d / cont (CPU/CUDA/Vulkan/Metal).
ggml_tensor * upsample_subpixel(ggml_context * ctx, const GGUFModel & m, ggml_type ctype,
                                int idx, ggml_tensor * x, int stride, int kernel) {
    char pre[64];
    std::snprintf(pre, sizeof pre, "hifigan.upsub.%d", idx);
    ggml_tensor * w = gguf_get(m, std::string(pre) + ".weight");
    if (w->type != ctype) w = ggml_cast(ctx, w, ctype);
    const int M  = (int) w->ne[0];         // phase-kernel taps
    const int Cin = (int) w->ne[1];        // input channels
    const int CS = (int) w->ne[2];         // Cout * stride, phase-fast
    const int Cout = CS / stride;
    const int L0 = (int) x->ne[0];

    ggml_tensor * y = use_manual_conv(m) ? conv1d_f32(ctx, w, x, M - 1, 1)
                                         : ggml_conv_1d(ctx, w, x, 1, M - 1, 1);
    y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    ggml_tensor * b = ggml_reshape_2d(ctx, gguf_get(m, std::string(pre) + ".bias"), 1, CS);
    y = ggml_add(ctx, y, b);
    const int Lt = (int) y->ne[0];

    // phase interleave: out[(t*s+r), o] = z[t, r*Cout + o]   (phase-major)
    ggml_tensor * y3 = ggml_reshape_3d(ctx, y, Lt, Cout, stride); // [Lt, Cout, s]
    ggml_tensor * yp = ggml_permute(ctx, y3, 1, 2, 0, 3);         // [s, Lt, Cout]
    ggml_tensor * yc = ggml_cont(ctx, yp);                        // dense copy
    const int64_t new_len = (int64_t) Lt * stride;
    ggml_tensor * y2 = ggml_reshape_2d(ctx, yc, new_len, Cout);   // [Lt*s, Cout]

    // same crop as the legacy convT path
    const int crop = (kernel - stride) / 2;
    const int64_t target_len = (int64_t) L0 * stride;
    ggml_tensor * out = y2;
    if (crop > 0) {
        out = ggml_view_2d(ctx, out, target_len, Cout, out->nb[1], crop * out->nb[0]);
        out = ggml_cont(ctx, out);
    }
    return out;
}

// ResBlock1: for each of 3 (k, d) pairs: lrelu -> conv(d) -> lrelu -> conv(1) -> add.
// convs1 dilations [1,3,5]; convs2 always dilation 1.
ggml_tensor * resblock(ggml_context * ctx, const GGUFModel & m, ggml_type ctype,
                       ggml_tensor * x, int rb_index) {
    ggml_tensor * h = x;
    for (int k = 0; k < 3; ++k) {
        const int dil = k == 0 ? 1 : (k == 1 ? 3 : 5);
        char pre[96];
        std::snprintf(pre, sizeof pre, "hifigan.resblocks.%d.convs1.%d", rb_index, k);
        ggml_tensor * xt = ggml_leaky_relu(ctx, h, 0.1f, false);
        xt = conv1d_same(ctx, m, ctype, pre, xt, dil);
        xt = ggml_leaky_relu(ctx, xt, 0.1f, false);
        std::snprintf(pre, sizeof pre, "hifigan.resblocks.%d.convs2.%d", rb_index, k);
        xt = conv1d_same(ctx, m, ctype, pre, xt, 1);
        h = ggml_add(ctx, xt, h);
    }
    return h;
}

}  // namespace

// ---------------------------------------------------------------------------

HifiganModel::HifiganModel(const std::string & path, int n_threads, const char * precision)
    : gguf(gguf_load(path, n_threads)) {
    const GGUFModel & gm = *gguf;
    compute_type = (precision && std::string(precision) == "F16")
                       ? GGML_TYPE_F16 : GGML_TYPE_F32;
    int64_t iv;
    float fv;
    if (gguf_meta_int(gm, "audio.n_mels", iv)) num_mels = (int) iv;
    if (gguf_meta_int(gm, "hifigan.num_upsamples", iv)) num_upsamples = (int) iv;
    if (gguf_meta_int(gm, "hifigan.num_resblocks", iv)) num_resblocks = (int) iv;
    if (gguf_meta_int(gm, "hifigan.upsample_initial_channel", iv)) upsample_initial_channel = (int) iv;
    if (gguf_meta_int(gm, "audio.sample_rate", iv)) sampling_rate = (int) iv;
    if (gguf_meta_int(gm, "audio.hop", iv)) hop_size = (int) iv;
    if (gguf_meta_int(gm, "hifigan.upp", iv)) upp = (int) iv;
    if (gguf_meta_float(gm, "hifigan.source_sr", fv)) source_sr = fv;
    if (gguf_meta_float(gm, "hifigan.noise_sigma", fv)) noise_sigma = fv;
    gguf_meta_bool(gm, "hifigan.mini_nsf", mini_nsf);
    std::vector<float> tmp;
    if (gguf_meta_floats(gm, "hifigan.upsample_rates", tmp))
        for (float v : tmp) upsample_rates.push_back((int) v);
    if (gguf_meta_floats(gm, "hifigan.upsample_kernels", tmp))
        for (float v : tmp) upsample_kernels.push_back((int) v);
    if (gguf_meta_floats(gm, "hifigan.resblock_kernels", tmp))
        for (float v : tmp) resblock_kernels.push_back((int) v);
    if (gguf_meta_floats(gm, "hifigan.resblock_dilations", tmp))
        for (float v : tmp) resblock_dilations.push_back((int) v);
    if (upp == 0 && sampling_rate > 0 && !upsample_rates.empty()) {
        // fallback: upp = prod(rates[:2]), source_sr = sr / prod(rates[2:])
        upp = 1;
        for (size_t i = 0; i < 2 && i < upsample_rates.size(); ++i) upp *= upsample_rates[i];
        int tail = 1;
        for (size_t i = 2; i < upsample_rates.size(); ++i) tail *= upsample_rates[i];
        source_sr = (float) sampling_rate / (float) tail;
    }
    std::fprintf(stderr, "[pc-nsf-hifigan] prec=%s mel=%d ups=%d resblocks=%d upp=%d source_sr=%.1f noise=%.4g\n",
                 compute_type == GGML_TYPE_F16 ? "F16" : "F32",
                 num_mels, num_upsamples, num_resblocks, upp, source_sr, noise_sigma);
}

HifiganModel::~HifiganModel() = default;
HifiganModel::HifiganModel(HifiganModel &&) noexcept = default;
HifiganModel & HifiganModel::operator=(HifiganModel &&) noexcept = default;

void hifigan_run(const HifiganModel & m, const float * mel, const float * f0, int T,
                 std::vector<float> & wav) {
    const GGUFModel & gm = *m.gguf;
    const ggml_type ctype = m.compute_type;
    const auto t_run_start = std::chrono::steady_clock::now();
    if (T <= 0 || !m.mini_nsf) {
        throw std::runtime_error(m.mini_nsf ? "hifigan: frames must be positive"
                                            : "hifigan: non-mini NSF source not implemented");
    }

    // source excitation [T*upp] from libmininsf (exact sinf path == torch)
    std::vector<float> source((std::size_t) T * m.upp);
    MiniNsfConfig cfg;
    cfg.source_sample_rate = m.source_sr;
    cfg.upsample = m.upp;
    if (mininsf_fastsinegen_f32(f0, 1, T, &cfg, source.data()) != MININSF_OK) {
        throw std::runtime_error("hifigan: libmininsf fastsinegen failed");
    }
    const int64_t source_len = (int64_t) source.size();

    // mel row-major [T, C] -> column-major [C, T] (ggml ne0 = T)
    std::vector<float> mel_cf((std::size_t) m.num_mels * T);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < m.num_mels; ++c)
            mel_cf[(std::size_t) c * T + t] = mel[(std::size_t) t * m.num_mels + c];

    // one shared ctx for inputs + graph (gallocr allocates the leaves)
    ggml_context * ctx = ggml_init({ 1024ULL * 1024 * 1024, nullptr, true });
    ggml_tensor * mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, m.num_mels);
    ggml_tensor * src_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, source_len, 1);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE * 64, false);

    // new-format GGUFs carry phase-split sub-pixel upsample tensors
    const bool subpixel = gm.tensors.find("hifigan.upsub.0.weight") != gm.tensors.end();

    ggml_tensor * x = conv1d_same(ctx, gm, ctype, "hifigan.conv_pre", mel_in, 1);
    const int num_kernels = (int) m.resblock_kernels.size();
    for (int i = 0; i < m.num_upsamples; ++i) {
        x = ggml_leaky_relu(ctx, x, 0.1f, false);
        char pre[64];
        std::snprintf(pre, sizeof pre, "hifigan.ups.%d", i);
        const int s = m.upsample_rates[i];
        const int K = m.upsample_kernels[i];
        x = subpixel ? upsample_subpixel(ctx, gm, ctype, i, x, s, K)
                     : conv_transpose1d_crop(ctx, gm, ctype, pre, x, s, K);
        if (i == 1) {
            ggml_tensor * xs = conv1d_k1(ctx, gm, ctype, "hifigan.source_conv", src_in);
            x = ggml_add(ctx, x, xs);
        }
        ggml_tensor * sum = nullptr;
        for (int j = 0; j < num_kernels; ++j) {
            ggml_tensor * rb = resblock(ctx, gm, ctype, x, i * num_kernels + j);
            sum = sum ? ggml_add(ctx, sum, rb) : rb;
        }
        x = ggml_scale(ctx, sum, 1.0f / (float) num_kernels);
    }
    // NOTE: torch Generator.forward ends with F.leaky_relu(x) whose DEFAULT
    // negative_slope is 0.01 (not 0.1).  We match torch here.
    x = ggml_leaky_relu(ctx, x, 0.01f, false);
    x = conv1d_same(ctx, gm, ctype, "hifigan.conv_post", x, 1);
    x = ggml_tanh(ctx, x);
    ggml_set_output(x);

    ggml_build_forward_expand(gf, x);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gm.backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        throw std::runtime_error("hifigan: gallocr alloc failed");
    }

    ggml_backend_tensor_set(mel_in, mel_cf.data(), 0, mel_cf.size() * sizeof(float));
    ggml_backend_tensor_set(src_in, source.data(), 0, source.size() * sizeof(float));
    ggml_status st = ggml_backend_graph_compute(gm.backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        throw std::runtime_error("hifigan: graph compute failed");
    }

    const int samples = (int) x->ne[0];
    std::vector<float> tmp((std::size_t) x->ne[0] * x->ne[1]);
    ggml_backend_tensor_get(x, tmp.data(), 0, tmp.size() * sizeof(float));
    wav.resize((std::size_t) samples);
    std::copy(tmp.begin(), tmp.begin() + samples, wav.begin());

    if (getenv("PCNSF_TIMING")) {
        const auto t_end = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[timing] hifigan_run %.1f ms (T=%d samples=%d)\n",
                     std::chrono::duration<double, std::milli>(t_end - t_run_start).count(),
                     T, samples);
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

}  // namespace pc_nsf_hifigan
