#include "pc_nsf_hifigan/hifigan.h"
#include "backend_policy.h"

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
    // Keep the established F32 orientation. F16 weights must be operand A:
    // the CPU mul_mat implementation requires an F32 activation operand B.
    ggml_tensor * r = w->type == GGML_TYPE_F32
        ? ggml_mul_mat(ctx, m0, wm)
        : ggml_cont(ctx, ggml_transpose(ctx, ggml_mul_mat(ctx, wm, m0)));
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

// like fused_site, but the default (env unset) folds: with the direct conv
// the post-bias leaky rides the conv epilogue for free (CPU and Vulkan),
// so no separate kernel/op is needed
inline bool fused_site_conv(const GGUFModel & m, char site) {
    const char * v = getenv("PCNSF_FUSED_ADD");
    if (!v || !*v) return true;
    return v[0] == '1' || v[0] == 'y' || v[0] == site;
}

// Direct fused conv (ggml-audio-patch CONV_DIRECT_1D): packs the kernel and
// runs output tiles with no im2col buffer; the bias add (and the leaky
// that follows convs1) fold into the epilogue.  AVX2 CPU + Vulkan/Metal + F32 line.
// The CPU kernel's non-AVX2 implementation is a correctness-only scalar
// fallback, so ARM64 defaults to im2col + mul_mat instead.
// The Vulkan and Metal CONV_DIRECT_1D shaders are F32 implicit GEMMs that also
// fold bias/residual/leaky/input-scale into the same device pass.  Metal uses
// simdgroup matrices and never materializes the very large im2col tensors.
// PCNSF_DIRECT_CONV=0 falls back to the im2col path for A/B.
inline bool use_direct_conv(const GGUFModel & m, ggml_type ctype) {
    return detail::direct_conv_enabled(ctype, ggml_backend_name(m.backend),
                                      m.supports_direct_conv, ggml_cpu_has_avx2(),
                                      getenv("PCNSF_DIRECT_CONV"));
}

// The Vulkan conv_direct_1d shader (vulkan-shaders/conv_direct_1d.comp:78-84)
// requires K >= 3: a BK=32 tap chunk spans floor((BK-1)/K)+1 new input rows,
// which must fit the XS_ROWS=12 circular x-window; K=2 spans 16 rows > 11 and
// xs_store() silently overwrites rows the current chunk still reads
// (deterministic wrong results, K=2 subpixel upsampling convs).  The CPU
// direct conv has no such restriction (verified bit-exact on the CPU line).
// Default gate: Vulkan K>=3, other backends K>=1; PCNSF_DIRECT_MIN_K=N
// overrides both (set 1 to reproduce the old Vulkan K=2 corruption for A/B).
inline int direct_conv_min_k(const GGUFModel & m) {
    const char * v = getenv("PCNSF_DIRECT_MIN_K");
    if (v && *v) return atoi(v);
    const char * backend_name = ggml_backend_name(m.backend);
    return (backend_name && std::strstr(backend_name, "Vulkan") != nullptr) ? 3 : 1;
}

// Sub-3 kernels fall back to the im2col/ggml_conv_1d path (+ explicit
// bias/leaky/residual nodes, exactly as on non-direct backends).  The caller
// must treat this like "direct conv unavailable": fuse_io folds pre-scale /
// pre-leaky / residual only hold when this returns true, so compute it
// BEFORE deciding any fusion.
inline bool direct_conv_k_ok(const GGUFModel & m, ggml_type ctype, int64_t kernel) {
    return use_direct_conv(m, ctype) && kernel >= (int64_t) direct_conv_min_k(m);
}

// Direct-conv producer-side fusion sites (ggml-audio-patch CONV_DIRECT_1D
// in_scale/in_slope/res params).  Each site removes a whole elementwise node
// bit-identically by folding it into the conv's X-pad copy or epilogue:
//   i: input leaky (resblock leaky-A and the cross-level leaky after scale)
//   s: input scale (the 1/num_kernels resblock-mean scale)
//   r: residual add (convs2 epilogue adds the running h)
// PCNSF_FUSE_IO=0 disables all; i/s/r select single sites for bisect.
inline bool use_fuse_io(const GGUFModel & m, ggml_type ctype) {
    if (!use_direct_conv(m, ctype)) return false;
    const char * v = getenv("PCNSF_FUSE_IO");
    if (v) return v[0] != '0';
    return true;
}

inline bool fuse_io_site(const GGUFModel & m, char site) {
    const char * v = getenv("PCNSF_FUSE_IO");
    if (!v || !*v) return true;   // checked by the caller via use_fuse_io
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
// `res` (residual), `in_scale` and `in_slope` (producer-side fusions) only
// apply on the direct-conv path; on fallbacks the caller must have emitted
// the equivalent explicit nodes (pass res=nullptr, in_scale=1, in_slope=0).
ggml_tensor * conv1d_same(ggml_context * ctx, const GGUFModel & m, ggml_type ctype,
                          const std::string & prefix, ggml_tensor * x, int dilation = 1,
                          float leaky_slope = 0.0f, ggml_tensor * res = nullptr,
                          float in_scale = 1.0f, float in_slope = 0.0f) {
    ggml_tensor * w = gguf_get(m, prefix + ".weight");
    if (w->type != ctype) w = ggml_cast(ctx, w, ctype);
    const int pad = (int) (w->ne[0] / 2) * dilation;
    // F32 im2col+mul_mat is faster than ggml's F16-im2col conv1d on CPU/Vulkan
    // (CPU 21.8->17.7s, Vulkan 18.7->2.56s for a 20s clip); CUDA keeps stock.
    ggml_tensor * y;
    if (direct_conv_k_ok(m, ctype, w->ne[0])) {
        // direct fused conv: bias (+ optional leaky) in the epilogue;
        // optional residual add and input scale/leaky folded in as well
        const bool fio = use_fuse_io(m, ctype);
        ggml_tensor * rres = (res && fio) ? res : nullptr;
        const float fsc = (in_scale != 1.0f && fio) ? in_scale : 1.0f;
        const float fsl = (in_slope != 0.0f && fio) ? in_slope : 0.0f;
        if (rres || fsc != 1.0f || fsl != 0.0f) {
            return ggml_conv_direct_1d_fused(ctx, w, x,
                                             gguf_get(m, prefix + ".bias"), rres,
                                             pad, dilation, leaky_slope, fsc, fsl);
        }
        return ggml_conv_direct_1d(ctx, w, x, gguf_get(m, prefix + ".bias"),
                                   pad, dilation, leaky_slope);
    }
    // A caller may defer producers into this layer before applying the kernel
    // threshold. If this convolution falls back, materialize those producers.
    if (in_scale != 1.0f) x = ggml_scale(ctx, x, in_scale);
    if (in_slope != 0.0f) x = ggml_leaky_relu(ctx, x, in_slope, false);
    if (use_manual_conv(m)) {
        y = conv1d_f32(ctx, w, x, pad, dilation);
    } else {
        y = ggml_conv_1d(ctx, w, x, 1, pad, dilation);
        y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    }
    ggml_tensor * b = ggml_reshape_2d(ctx, gguf_get(m, prefix + ".bias"), 1, y->ne[1]);
    if (leaky_slope != 0.0f && use_fused_add_leaky(m)) {
        y = ggml_add_leaky_relu(ctx, y, b, leaky_slope);
    } else {
        y = ggml_add(ctx, y, b);
    }
    if (res) y = ggml_add(ctx, y, res);
    return y;
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
                                int idx, ggml_tensor * x, int stride, int kernel,
                                float in_scale = 1.0f, float in_slope = 0.0f) {
    char pre[64];
    std::snprintf(pre, sizeof pre, "hifigan.upsub.%d", idx);
    ggml_tensor * w = gguf_get(m, std::string(pre) + ".weight");
    if (w->type != ctype) w = ggml_cast(ctx, w, ctype);
    const int M  = (int) w->ne[0];         // phase-kernel taps
    const int Cin = (int) w->ne[1];        // input channels
    const int CS = (int) w->ne[2];         // Cout * stride, phase-fast
    const int Cout = CS / stride;
    const int L0 = (int) x->ne[0];

    // producer-side scale/leaky fusion (cross-level leaky + resblock-mean
    // scale) folds into the direct conv's X-pad copy; K<3 (subpixel M=2)
    // fails the Vulkan shader's K>=3 precondition, so the gate applies to
    // BOTH the folding and the direct-conv branch — on the fallback the
    // scale/leaky are emitted explicitly here
    const bool direct = direct_conv_k_ok(m, ctype, w->ne[0]);
    const bool fio = direct && use_fuse_io(m, ctype) && (in_scale != 1.0f || in_slope != 0.0f);
    if (in_scale != 1.0f && !fio) x = ggml_scale(ctx, x, in_scale);
    if (in_slope != 0.0f && !fio) x = ggml_leaky_relu(ctx, x, in_slope, false);

    ggml_tensor * y;
    if (direct) {
        // direct fused conv: the tiled bias folds into the epilogue
        if (fio) {
            y = ggml_conv_direct_1d_fused(ctx, w, x, gguf_get(m, std::string(pre) + ".bias"),
                                          nullptr, M - 1, 1, 0.0f, in_scale, in_slope);
        } else {
            y = ggml_conv_direct_1d(ctx, w, x, gguf_get(m, std::string(pre) + ".bias"),
                                    M - 1, 1, 0.0f);
        }
    } else {
        y = use_manual_conv(m) ? conv1d_f32(ctx, w, x, M - 1, 1)
                               : ggml_conv_1d(ctx, w, x, 1, M - 1, 1);
        ggml_tensor * b = ggml_reshape_2d(ctx, gguf_get(m, std::string(pre) + ".bias"), 1, CS);
        y = ggml_add(ctx, y, b);
    }
    y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
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
// With fuse_io (CPU direct conv): the leaky on h folds into convs1's X-pad
// (in_slope), and the residual add folds into convs2's epilogue (res=h) — the
// convs2 output IS the new h.  h itself stays un-activated for the residual
// chain; each conv reads its own padded copy, so the two consumers of h
// (convs1 input and convs2 residual) both see exactly the right values.
ggml_tensor * resblock(ggml_context * ctx, const GGUFModel & m, ggml_type ctype,
                       ggml_tensor * x, int rb_index) {
    // convs1 post-bias leaky: rides the direct-conv epilogue (CPU/Vulkan);
    // on the im2col fallback the fold needs the CPU ADD_LEAKY_RELU op
    const bool fio     = use_fuse_io(m, ctype);
    ggml_tensor * h = x;
    for (int k = 0; k < 3; ++k) {
        const int dil = k == 0 ? 1 : (k == 1 ? 3 : 5);
        char pre[96];
        std::snprintf(pre, sizeof pre, "hifigan.resblocks.%d.convs1.%d", rb_index, k);
        const bool dc1 = direct_conv_k_ok(m, ctype, gguf_get(m, std::string(pre) + ".weight")->ne[0]);
        const bool fuse_c1 = dc1 ? fused_site_conv(m, 'c') : fused_site(m, 'c');
        const bool fio_i = dc1 && fio && fuse_io_site(m, 'i');
        ggml_tensor * xt = fio_i ? h : ggml_leaky_relu(ctx, h, 0.1f, false);
        xt = conv1d_same(ctx, m, ctype, pre, xt, dil, fuse_c1 ? 0.1f : 0.0f,
                         nullptr, 1.0f, fio_i ? 0.1f : 0.0f);
        if (!fuse_c1) xt = ggml_leaky_relu(ctx, xt, 0.1f, false);
        std::snprintf(pre, sizeof pre, "hifigan.resblocks.%d.convs2.%d", rb_index, k);
        const bool dc2 = direct_conv_k_ok(m, ctype, gguf_get(m, std::string(pre) + ".weight")->ne[0]);
        const bool fio_r = dc2 && fio && fuse_io_site(m, 'r');
        if (fio_r) {
            // residual folded into the convs2 epilogue: out = conv + bias + h
            h = conv1d_same(ctx, m, ctype, pre, xt, 1, 0.0f, h);
        } else {
            xt = conv1d_same(ctx, m, ctype, pre, xt, 1);
            h = ggml_add(ctx, xt, h);
        }
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

    const bool pre_direct = direct_conv_k_ok(gm, ctype, gguf_get(gm, "hifigan.conv_pre.weight")->ne[0]);
    const bool p_folded = pre_direct ? fused_site_conv(gm, 'p') : fused_site(gm, 'p');
    ggml_tensor * x = conv1d_same(ctx, gm, ctype, "hifigan.conv_pre", mel_in, 1,
                                p_folded ? 0.1f : 0.0f);
    const int num_kernels = (int) m.resblock_kernels.size();
    const float mean_scale = 1.0f / (float) num_kernels;
    const bool fio   = use_fuse_io(gm, ctype);
    const bool fio_s = fio && fuse_io_site(gm, 's');
    const bool fio_i = fio && fuse_io_site(gm, 'i');
    for (int i = 0; i < m.num_upsamples; ++i) {
        // Entering level i>0, x is the previous level's resblock mean
        // (scale(sum, 1/nk) then leaky).  With fuse_io both the scale and the
        // leaky fold into this level's upsub conv X-pad (in_scale/in_slope);
        // on the fallback path (or per-site bisect) they stay explicit nodes.
        // i==0's leaky was already fused into conv_pre's bias add instead.
        const bool need_leaky = (i > 0 || !p_folded);
        const bool need_scale = (i > 0);
        if (need_scale && !fio_s) x = ggml_scale(ctx, x, mean_scale);
        if (need_leaky && !fio_i) x = ggml_leaky_relu(ctx, x, 0.1f, false);
        char pre[64];
        std::snprintf(pre, sizeof pre, "hifigan.ups.%d", i);
        const int s = m.upsample_rates[i];
        const int K = m.upsample_kernels[i];
        const float up_sc = (need_scale && fio_s) ? mean_scale : 1.0f;
        const float up_sl = (need_leaky && fio_i) ? 0.1f : 0.0f;
        if (up_sc != 1.0f || up_sl != 0.0f) {
            x = subpixel ? upsample_subpixel(ctx, gm, ctype, i, x, s, K, up_sc, up_sl)
                         : (x = ggml_scale(ctx, x, up_sc),
                            x = ggml_leaky_relu(ctx, x, up_sl, false),
                            conv_transpose1d_crop(ctx, gm, ctype, pre, x, s, K));
        } else {
            x = subpixel ? upsample_subpixel(ctx, gm, ctype, i, x, s, K)
                         : conv_transpose1d_crop(ctx, gm, ctype, pre, x, s, K);
        }
        if (i == 1) {
            ggml_tensor * xs = conv1d_k1(ctx, gm, ctype, "hifigan.source_conv", src_in);
            x = ggml_add(ctx, x, xs);
        }
        ggml_tensor * sum = nullptr;
        for (int j = 0; j < num_kernels; ++j) {
            ggml_tensor * rb = resblock(ctx, gm, ctype, x, i * num_kernels + j);
            sum = sum ? ggml_add(ctx, sum, rb) : rb;
        }
        // the mean scale is deferred: on the fused path it folds into the
        // next level's upsub (or conv_post) X-pad; otherwise it is applied
        // explicitly at the top of the next iteration
        x = sum;
    }
    // NOTE: torch Generator.forward ends with F.leaky_relu(x) whose DEFAULT
    // negative_slope is 0.01 (not 0.1).  We match torch here.  With fuse_io
    // the final mean-scale + this leaky fold into conv_post's X-pad.
    if (!fio_s) x = ggml_scale(ctx, x, mean_scale);
    if (fio_i) {
        x = conv1d_same(ctx, gm, ctype, "hifigan.conv_post", x, 1, 0.0f,
                        nullptr, fio_s ? mean_scale : 1.0f, 0.01f);
    } else {
        x = ggml_leaky_relu(ctx, x, 0.01f, false);
        x = conv1d_same(ctx, gm, ctype, "hifigan.conv_post", x, 1);
    }
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
        std::vector<char> stage;   // host staging for cross-backend node dumps
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
            // GPU tensors are not host-readable through t->data; stage the
            // whole node through the backend's tensor_get path instead.
            p->stage.resize((size_t) n * sizeof(float));
            ggml_backend_tensor_get(t, p->stage.data(), 0, (size_t) n * sizeof(float));
            const float * d = (const float *) p->stage.data();
            double sum = 0, mx = 0;
            for (int64_t i = 0; i < n; i += 997) {
                const double v = d[i];
                sum += v;
                const double a = v < 0 ? -v : v;
                if (a > mx) mx = a;
            }
            fprintf(stderr, "[dump %4d] %-18s [%6lld] sum=%.6f max=%.6f\n",
                    p->idx - 1, ggml_op_name(t->op), (long long) n, sum, mx);
            // raw dump: PCNSF_DUMP=<dir> writes node_<idx>_<op>.f32 + manifest
            if (const char * dir = getenv("PCNSF_DUMP")) {
                char path[512];
                snprintf(path, sizeof path, "%s/node_%04d.f32", dir, p->idx - 1);
                FILE * f = ggml_fopen(path, "wb");
                if (f) {
                    fwrite(p->stage.data(), 1, p->stage.size(), f);
                    fclose(f);
                    snprintf(path, sizeof path, "%s/manifest.txt", dir);
                    f = ggml_fopen(path, "a");
                    if (f) {
                        fprintf(f, "%d %s %lld %lld %lld\n", p->idx - 1,
                                ggml_op_name(t->op), (long long) t->ne[0],
                                (long long) t->ne[1], (long long) t->ne[2]);
                        fclose(f);
                    }
                }
            }
        }
        auto & e = p->acc[ggml_op_name(t->op)];
        e.first++; e.second += ms;
        return true;
    };

    ggml_backend_sched_t sched = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_backend_t prof_cpu = nullptr;   // extra CPU backend appended for sched
    if (profile) {
        // ggml_backend_sched_new requires the backend list to END with a CPU
        // backend; when profiling a GPU backend (e.g. Vulkan), append one.
        // The graph itself still lands on gm.backend: every op in it is
        // supported there, so the CPU entry only satisfies the API contract.
        std::vector<ggml_backend_t> prof_backends;
        prof_backends.push_back(gm.backend);
        ggml_backend_dev_t dev = ggml_backend_get_device(gm.backend);
        if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (cpu_dev) {
                prof_cpu = ggml_backend_dev_init(cpu_dev, nullptr);
                prof_backends.push_back(prof_cpu);
            }
        }
        sched = ggml_backend_sched_new(prof_backends.data(), nullptr,
                                       (int) prof_backends.size(),
                                       GGML_DEFAULT_GRAPH_SIZE, false, false);
        if (!sched) {
            if (prof_cpu) ggml_backend_free(prof_cpu);
            ggml_free(ctx);
            throw std::runtime_error("hifigan: sched init failed");
        }
        ggml_backend_sched_set_eval_callback(sched, prof_cb, &np);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            ggml_backend_sched_free(sched);
            if (prof_cpu) ggml_backend_free(prof_cpu);
            ggml_free(ctx);
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
        if (prof_cpu) ggml_backend_free(prof_cpu);
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
    }

    const int samples = (int) x->ne[0];
    std::vector<float> tmp((std::size_t) x->ne[0] * x->ne[1]);
    ggml_backend_tensor_get(x, tmp.data(), 0, tmp.size() * sizeof(float));
    wav.resize((std::size_t) samples);
    std::copy(tmp.begin(), tmp.begin() + samples, wav.begin());

    // The scheduler owns x's allocation in profile mode, so keep it alive
    // through the device-to-host copy above.
    if (sched) ggml_backend_sched_free(sched);
    if (prof_cpu) ggml_backend_free(prof_cpu);

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
