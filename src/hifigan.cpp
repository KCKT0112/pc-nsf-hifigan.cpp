#include "pc_nsf_hifigan/hifigan.h"

#include <mininsf/mininsf.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>

namespace pc_nsf_hifigan {

namespace {

// F32 im2col + mul_mat 1D conv: avoids ggml_conv_1d's F16 im2col on CPU and
// is also much faster on Vulkan (ggml mul_mat is properly optimized there).
// Uses the IM2COL_FAST_1D-tagged im2col (ggml-audio-patch): on CPU the kernel
// tiles for L2 residency and splits threads over output positions.  dst stays
// F32 to keep the exact-golden precision line.
// The im2col output length is rounded up to a multiple of 16 (ow_align) so
// every GEMM hits the CPU tinyBLAS m%16 packing (the llamafile sgemm falls
// back to the slow vec_dot path otherwise); the padding rows reproduce conv
// zero-padding, so cropping them with a free view keeps bit-exact output.
// Returns the conv output reshaped to [OL, OC].
ggml_tensor * conv1d_f32(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x,
                         int pad, int dil) {
    const int64_t K  = w->ne[0];
    const int64_t IC = w->ne[1];
    const int64_t T  = x->ne[0];
    const int64_t OL = T + 2*pad - dil*(K - 1);   // stride 1
    ggml_tensor * im = ggml_im2col_fast_1d(ctx, w, x, 1, pad, dil, GGML_TYPE_F32, 16);
    ggml_tensor * m0 = ggml_reshape_2d(ctx, im, im->ne[0], (im->ne[2] * im->ne[1]));
    ggml_tensor * wm = ggml_reshape_2d(ctx, w, (w->ne[0] * w->ne[1]), w->ne[2]);
    ggml_tensor * r = ggml_mul_mat(ctx, m0, wm);
    r = ggml_reshape_3d(ctx, r, im->ne[1], w->ne[2], im->ne[2]);
    r = ggml_reshape_2d(ctx, r, r->ne[0], r->ne[1]);
    if (r->ne[0] != OL) {
        // drop align pad rows; materialize so downstream reshapes keep their
        // contiguity asserts (only conv_pre + the 5 upsub convs ever pad,
        // ~200MB of copies total for a 20s clip — noise vs the GEMMs)
        r = ggml_view_2d(ctx, r, OL, r->ne[1], r->nb[1], 0);
        r = ggml_cont(ctx, r);
    }
    return r;
}

// Default policy: use conv1d_f32 on every backend except CUDA (where the stock
// ggml conv1d kernel is already optimized).  PCNSF_MANUAL_CONV=1/0 overrides.
inline bool use_manual_conv(const GGUFModel & m) {
    const char * pcnsf_manual = getenv("PCNSF_MANUAL_CONV");
    const char * backend_name = ggml_backend_name(m.backend);
    const bool is_cuda = backend_name && std::strstr(backend_name, "CUDA");
    return pcnsf_manual ? pcnsf_manual[0] != '0' : !is_cuda;
}

// Fused add+leaky-ReLU op availability: CPU backend only (the ggml-audio-patch
// ADD_LEAKY_RELU kernel).  PCNSF_FUSED_ADD=0 disables; p/c select single
// fusion sites for bisect/debug (p=conv_pre, c=resblock convs1 bias).
inline bool use_fused_add_leaky(const GGUFModel & m) {
    const char * pcnsf_fused = getenv("PCNSF_FUSED_ADD");
    if (pcnsf_fused) return pcnsf_fused[0] != '0';
    const char * backend_name = ggml_backend_name(m.backend);
    return backend_name && std::strcmp(backend_name, "CPU") == 0;
}

// which fusion site group is active under the current PCNSF_FUSED_ADD value
inline bool fused_site(const GGUFModel & m, char site) {
    const char * v = getenv("PCNSF_FUSED_ADD");
    if (!v || !*v) return use_fused_add_leaky(m);   // default: CPU => all sites
    return v[0] == '1' || v[0] == 'y' || v[0] == site;
}

// Conv1d "same" (pad = (K/2)*dilation, matching torch get_padding for odd K),
// kernel layout [K, IC, OC], data layout [T, IC] -> [T', OC].
// `ctype` selects the storage precision of the kernel: GGML_TYPE_F32 (exact
// golden line) or GGML_TYPE_F16 (fp16 line, reserved for future fp16/bf16
// trained checkpoints).  ggml's 1D conv computes in fp32 on CPU, so
// activations/bias stay fp32 regardless; the fp16 line therefore means
// "weights fp16, compute fp32" (the same contract as the fp16/bf16 line).
// When `leaky_slope` is nonzero (and the backend has the fused op), the bias
// add and the following leaky ReLU collapse into one ADD_LEAKY_RELU pass —
// the caller then MUST NOT emit its own ggml_leaky_relu on the result.
ggml_tensor * conv1d_same(ggml_context * ctx, const GGUFModel & m, ggml_type ctype,
                          const std::string & prefix, ggml_tensor * x, int dilation = 1,
                          float leaky_slope = 0.0f) {    ggml_tensor * w = gguf_get(m, prefix + ".weight");
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
    if (leaky_slope != 0.0f && use_fused_add_leaky(m)) {
        return ggml_add_leaky_relu(ctx, y, b, leaky_slope);
    }
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
// On CPU the convs1 bias add fuses with the leaky that follows it (bit-identical).
// The residual add is NOT fused with the next sub-block's leaky: the residual
// sum itself stays the residual operand of the NEXT sub-block (unactivated),
// so it has two consumers and the fused (leaky'd) tensor cannot replace it.
ggml_tensor * resblock(ggml_context * ctx, const GGUFModel & m, ggml_type ctype,
                       ggml_tensor * x, int rb_index) {
    const bool fuse_c1 = fused_site(m, 'c');
    ggml_tensor * h = x;
    for (int k = 0; k < 3; ++k) {
        const int dil = k == 0 ? 1 : (k == 1 ? 3 : 5);
        char pre[96];
        std::snprintf(pre, sizeof pre, "hifigan.resblocks.%d.convs1.%d", rb_index, k);
        ggml_tensor * xt = ggml_leaky_relu(ctx, h, 0.1f, false);
        xt = conv1d_same(ctx, m, ctype, pre, xt, dil, fuse_c1 ? 0.1f : 0.0f);
        if (!fuse_c1) xt = ggml_leaky_relu(ctx, xt, 0.1f, false);
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

    ggml_tensor * x = conv1d_same(ctx, gm, ctype, "hifigan.conv_pre", mel_in, 1,
                                  fused_site(gm, 'p') ? 0.1f : 0.0f);
    const int num_kernels = (int) m.resblock_kernels.size();
    for (int i = 0; i < m.num_upsamples; ++i) {
        // i==0's leaky was fused into conv_pre's bias add; every later level
        // leaky-izes the previous level's resblock mean instead
        if (i > 0 || !fused_site(gm, 'p')) x = ggml_leaky_relu(ctx, x, 0.1f, false);
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
    // mark graph inputs so the allocator assigns them first and never recycles
    // their storage mid-graph (required by the backend_sched eval-callback path,
    // correct for the raw gallocr path too)
    ggml_set_input(mel_in);
    ggml_set_input(src_in);

    // PCNSF_PROFILE=1: per-node timing via a single-backend backend_sched eval
    // callback (computes the graph node-by-node; adds small per-node overhead).
    const bool profile = getenv("PCNSF_PROFILE") != nullptr;
    static const bool dump_nodes = getenv("PCNSF_DUMP") != nullptr;  // captured via static
    struct NodeProf {
        std::chrono::steady_clock::time_point t0;
        std::map<std::string, std::pair<int, double>> acc;  // op -> (count, ms)
        int idx = 0;
    } np;
    auto prof_cb = [](struct ggml_tensor * t, bool ask, void * ud) -> bool {
        NodeProf * p = (NodeProf *) ud;
        if (ask) { p->t0 = std::chrono::steady_clock::now(); return true; }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - p->t0).count();
        fprintf(stderr, "[prof %4d] %-18s [%6lld %6lld %4lld] %9.3f ms\n", p->idx++,
                ggml_op_name(t->op), (long long) t->ne[0], (long long) t->ne[1],
                (long long) t->ne[2], ms);
        if (dump_nodes && t->type == GGML_TYPE_F32 && t->ne[0] > 0) {
            const int64_t n = ggml_nelements(t);
            const float * d = (const float *) t->data;
            double sum = 0, mx = 0;
            for (int64_t i = 0; i < n; i += 997) {
                const double v = d[i];
                sum += v;
                const double a = v < 0 ? -v : v;
                if (a > mx) mx = a;
            }
            fprintf(stderr, "[dump %4d] %-18s [%6lld] sum=%.6f max=%.6f\n",
                    p->idx - 1, ggml_op_name(t->op), (long long) n, sum, mx);
        }
        auto & e = p->acc[ggml_op_name(t->op)];
        e.first++; e.second += ms;
        return true;
    };

    ggml_backend_sched_t sched = nullptr;
    ggml_gallocr_t alloc = nullptr;
    if (profile) {
        sched = ggml_backend_sched_new((ggml_backend_t *) &gm.backend, nullptr, 1,
                                       GGML_DEFAULT_GRAPH_SIZE, false, false);
        if (!sched) throw std::runtime_error("hifigan: sched init failed");
        ggml_backend_sched_set_eval_callback(sched, prof_cb, &np);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            ggml_backend_sched_free(sched);
            throw std::runtime_error("hifigan: sched alloc failed");
        }
    } else {
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gm.backend));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) {
            ggml_gallocr_free(alloc);
            throw std::runtime_error("hifigan: gallocr alloc failed");
        }
    }

    ggml_backend_tensor_set(mel_in, mel_cf.data(), 0, mel_cf.size() * sizeof(float));
    ggml_backend_tensor_set(src_in, source.data(), 0, source.size() * sizeof(float));
    ggml_status st = profile ? ggml_backend_sched_graph_compute(sched, gf)
                             : ggml_backend_graph_compute(gm.backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        if (alloc) ggml_gallocr_free(alloc);
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        throw std::runtime_error("hifigan: graph compute failed");
    }
    if (profile) {
        fprintf(stderr, "[prof] ---- per-op totals ----\n");
        double total_ms = 0.0;
        for (const auto & kv : np.acc) {
            fprintf(stderr, "[prof] %-18s n=%4d  total=%9.1f ms  avg=%7.3f ms\n",
                    kv.first.c_str(), kv.second.first, kv.second.second,
                    kv.second.second / kv.second.first);
            total_ms += kv.second.second;
        }
        fprintf(stderr, "[prof] node total %.1f ms\n", total_ms);
        ggml_backend_sched_free(sched);
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

    if (alloc) ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

}  // namespace pc_nsf_hifigan
