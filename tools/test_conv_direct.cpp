// SPDX-License-Identifier: MPL-2.0
// test_conv_direct.cpp — isolate the Vulkan CONV_DIRECT_1D shader against
// the CPU kernel with deterministic data.  Usage: [backend] (default vulkan)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>

static uint32_t lcg_state = 0x12345678u;
static float frand() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((lcg_state >> 8) & 0xffffff) / 16777216.0f - 0.5f;
}

struct Case {
    int K, IC, OC, T, dil;
    bool bias, res;
    float slope, in_scale, in_slope;
};

// returns false on mismatch
static bool run_case(const Case & cs, ggml_backend_t bk, const char * bk_name, bool check_only) {
    const int pad = (cs.K / 2) * cs.dil;   // "same" padding like the model
    const int64_t OL = (int64_t) cs.T + 2 * pad - cs.dil * (cs.K - 1);

    ggml_init_params ip = { 64ull * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cs.K, cs.IC, cs.OC);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cs.T, cs.IC);
    ggml_tensor * b = cs.bias ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cs.OC) : nullptr;
    ggml_tensor * r = cs.res ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, OL, cs.OC) : nullptr;

    // deterministic fill (host side; uploaded after allocation)
    lcg_state = 0xdeadbeefu;
    std::vector<float> wv(ggml_nbytes(w) / 4), xv(ggml_nbytes(x) / 4), bv(cs.bias ? cs.OC : 0), rv(cs.res ? (size_t) (OL * cs.OC) : 0);
    for (auto & v : wv) v = frand() * 0.2f;
    for (auto & v : xv) v = frand() * 2.0f;
    for (auto & v : bv) v = frand() * 0.5f;
    for (auto & v : rv) v = frand() * 0.5f;

    ggml_tensor * y = ggml_conv_direct_1d_fused(ctx, w, x, b, r, pad, cs.dil,
                                                cs.slope, cs.in_scale, cs.in_slope);
    ggml_set_input(w); ggml_set_input(x);
    if (b) ggml_set_input(b);
    if (r) ggml_set_input(r);
    ggml_set_output(y);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(bk));
    if (!ggml_gallocr_alloc_graph(ga, gf)) { printf("alloc failed\n"); return false; }

    ggml_backend_tensor_set(w, wv.data(), 0, wv.size() * sizeof(float));
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
    if (b) ggml_backend_tensor_set(b, bv.data(), 0, bv.size() * sizeof(float));
    if (r) ggml_backend_tensor_set(r, rv.data(), 0, rv.size() * sizeof(float));

    const ggml_status st = ggml_backend_graph_compute(bk, gf);
    if (st != GGML_STATUS_SUCCESS) { printf("compute failed\n"); return false; }

    // read back
    std::vector<float> out(OL * cs.OC);
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));

    ggml_gallocr_free(ga);
    ggml_free(ctx);

    if (check_only) return true;   // warmup / driver check

    // reference: plain fp64 conv in the same k order (ic-major, kw within ic
    // ascending — k = ic*K + kw) to mirror the shader accumulation order
    std::vector<double> ref(OL * cs.OC, 0.0);
    const float * wd = wv.data();
    const float * xd = xv.data();
    const float * bd = bv.empty() ? nullptr : bv.data();
    const float * rd = rv.empty() ? nullptr : rv.data();
    for (int oc = 0; oc < cs.OC; oc++) {
        for (int t = 0; t < OL; t++) {
            double acc = bd ? bd[oc] : 0.0;
            for (int ic = 0; ic < cs.IC; ic++) {          // ic-major, k ascending
                for (int kw = 0; kw < cs.K; kw++) {
                    const int xi = t + kw * cs.dil - pad;
                    if (xi >= 0 && xi < cs.T) {
                        double v = xd[(int64_t) ic * cs.T + xi];
                        if (cs.in_scale != 1.0f) v *= cs.in_scale;
                        if (cs.in_slope != 0.0f) v = v > 0 ? v : cs.in_slope * v;
                        acc += (double) wd[((int64_t) oc * cs.IC + ic) * cs.K + kw] * v;
                    }
                }
            }
            if (rd) acc += rd[(int64_t) oc * OL + t];
            if (cs.slope != 0.0f) acc = acc > 0 ? acc : cs.slope * acc;
            ref[(int64_t) oc * OL + t] = acc;
        }
    }

    double max_abs = 0, sum_sq = 0, ref_sq = 0;
    int worst_t = -1, worst_oc = -1;
    for (int oc = 0; oc < cs.OC; oc++) {
        for (int t = 0; t < OL; t++) {
            const double d = fabs(out[(int64_t) oc * OL + t] - ref[(int64_t) oc * OL + t]);
            sum_sq += d * d;
            ref_sq += ref[(int64_t) oc * OL + t] * ref[(int64_t) oc * OL + t];
            if (d > max_abs) { max_abs = d; worst_t = t; worst_oc = oc; }
        }
    }
    const bool ok = max_abs < 5e-4;
    printf("  [%s] K=%2d IC=%3d OC=%3d T=%4d dil=%d b=%d r=%d sl=%.2f isc=%.3f isl=%.2f -> max_abs=%.3e rms=%.3e %s\n",
           bk_name, cs.K, cs.IC, cs.OC, cs.T, cs.dil, cs.bias, cs.res, cs.slope, cs.in_scale, cs.in_slope,
           max_abs, std::sqrt(sum_sq / (OL * cs.OC)), ok ? "OK" : "FAIL");
    if (!ok) {
        printf("      worst at t=%d oc=%d: got %f want %f\n", worst_t, worst_oc,
               out[(int64_t) worst_oc * OL + worst_t], ref[(int64_t) worst_oc * OL + worst_t]);
    }
    return ok;
}

// two-node chain mimicking the resblock: convs1 (x=h, in_slope, slope)
// -> convs2 (x=out1, res=h).  Catches allocator/lifetime issues with the
// 4th source tensor (res) across chained custom ops.
// N-pair resblock in ONE graph (production structure reproduction).
// PCNSF_PAIRS sets N (default 3).  Weight magnitudes kept small (x0.1) so the
// residual chain does not numerically explode the way the real generator does
// when a node reads garbage — that keeps the test a *correctness* probe.
static bool run_chain_n(ggml_backend_t bk, const char * bk_name) {
    const int K = 3, IC = 256, OC = 256, T = 13776;
    const int dils[3] = {1, 3, 5};
    const int64_t OL = T;
    const int N = getenv("PCNSF_PAIRS") ? atoi(getenv("PCNSF_PAIRS")) : 3;

    ggml_init_params ip = { 1024ull * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    // initial h
    ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, IC);
    ggml_set_input(h);
    std::vector<float> hv(T * IC);
    lcg_state = 0xC0FFEEu;
    for (auto & v : hv) v = frand() * 1.0f;

    std::vector<std::vector<float>> store;  // keep host copies alive
    auto mk_w = [&](std::vector<float> & outv, int K_, int IC_, int OC_) {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K_, IC_, OC_);
        outv.assign(ggml_nbytes(t) / 4, 0.0f);
        for (auto & v : outv) v = frand() * 0.05f;
        return t;
    };
    auto mk_b = [&](std::vector<float> & outv, int OC_) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OC_);
        outv.assign(ggml_nbytes(t) / 4, 0.0f);
        for (auto & v : outv) v = frand() * 0.1f;
        return t;
    };

    ggml_tensor * hh = h;
    std::vector<ggml_tensor *> y1s, y2s;
    for (int k = 0; k < N; ++k) {
        const int dil = dils[k % 3];
        const int pad = (K / 2) * dil;
        std::vector<float> w1v, w2v, b1v, b2v;
        ggml_tensor * w1 = mk_w(w1v, K, IC, OC);
        ggml_tensor * w2 = mk_w(w2v, K, IC, OC);
        ggml_tensor * b1 = mk_b(b1v, OC);
        ggml_tensor * b2 = mk_b(b2v, OC);
        store.push_back(w1v); store.push_back(w2v); store.push_back(b1v); store.push_back(b2v);
        // must mark every leaf we feed as INPUT, otherwise gallocr treats them
        // as temporary nodes and may recycle their storage mid-graph
        // (production does exactly this, hifigan.cpp:456-458).
        ggml_set_input(w1); ggml_set_input(w2); ggml_set_input(b1); ggml_set_input(b2);
        ggml_tensor * y1 = ggml_conv_direct_1d_fused(ctx, w1, hh, b1, nullptr, pad, dil, 0.1f, 1.0f, 0.1f);
        ggml_tensor * y2 = ggml_conv_direct_1d_fused(ctx, w2, y1, b2, hh, (K / 2) * 1, 1, 0.0f, 1.0f, 0.0f);
        y1s.push_back(y1); y2s.push_back(y2);
        hh = y2;
    }
    ggml_set_output(hh);
    // y1s must also be outputs: gallocr recycles a non-output tensor's storage
    // after its last consumer runs, so reading y1 back after compute returns
    // whatever tensor reused that memory (the stale readback is what made the
    // harness look corrupt while convs2(res) was fine).
    for (auto y : y1s) ggml_set_output(y);
    for (auto y : y2s) ggml_set_output(y);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, hh);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(bk));
    if (!ggml_gallocr_alloc_graph(ga, gf)) { printf("chainN alloc failed\n"); return false; }

    ggml_backend_tensor_set(h, hv.data(), 0, hv.size() * sizeof(float));
    // upload by walking the store in the same creation order (w1,w2,b1,b2 per pair),
    // resolving the actual tensors from the graph's src pointers
    for (int k = 0; k < N; ++k) {
        ggml_tensor * y1 = y1s[k];
        ggml_tensor * w1 = y1->src[0];
        ggml_tensor * b1 = y1->src[2];
        ggml_tensor * y2 = y2s[k];
        ggml_tensor * w2 = y2->src[0];
        ggml_tensor * b2 = y2->src[2];
        ggml_backend_tensor_set(w1, store[k * 4 + 0].data(), 0, store[k * 4 + 0].size() * 4);
        ggml_backend_tensor_set(w2, store[k * 4 + 1].data(), 0, store[k * 4 + 1].size() * 4);
        ggml_backend_tensor_set(b1, store[k * 4 + 2].data(), 0, store[k * 4 + 2].size() * 4);
        ggml_backend_tensor_set(b2, store[k * 4 + 3].data(), 0, store[k * 4 + 3].size() * 4);
    }

    if (ggml_backend_graph_compute(bk, gf) != GGML_STATUS_SUCCESS) { printf("chainN compute failed\n"); return false; }

    // fp64 reference, pair by pair (accumulator in fp64, k-order matches shader)
    std::vector<double> href(hv.begin(), hv.end());
    double worst = 0.0; int worst_pair = -1;
    for (int k = 0; k < N; ++k) {
        const int dil = dils[k % 3];
        const int pad = (K / 2) * dil;
        auto convref = [&](const std::vector<float> & wv, const std::vector<float> & bv,
                           const std::vector<double> & xd, float slope, float in_slope,
                           bool addr, const std::vector<double> & res,
                           int cdil, int cpad) {
            std::vector<double> r((size_t) OL * OC, 0.0);
            for (int oc = 0; oc < OC; oc++)
                for (int t = 0; t < OL; t++) {
                    double acc = bv[oc];
                    for (int ic = 0; ic < IC; ic++)
                        for (int kw = 0; kw < K; kw++) {
                            const int xi = t + kw * cdil - cpad;
                            if (xi >= 0 && xi < T) {
                                double v = xd[(int64_t) ic * T + xi];
                                if (in_slope != 0.0f) v = v > 0 ? v : in_slope * v;
                                acc += (double) wv[((int64_t) oc * IC + ic) * K + kw] * v;
                            }
                        }
                    if (addr) acc += res[(int64_t) oc * OL + t];
                    if (slope != 0.0f) acc = acc > 0 ? acc : slope * acc;
                    r[(int64_t) oc * OL + t] = acc;
                }
            return r;
        };
        auto r1 = convref(store[k * 4 + 0], store[k * 4 + 2], href, 0.1f, 0.1f, false, href, dil, pad);
        std::vector<double> r1f = r1;
        auto r2 = convref(store[k * 4 + 1], store[k * 4 + 3], r1f, 0.0f, 0.0f, true, href, 1, K / 2);

        std::vector<float> o1(OL * OC), o2(OL * OC);
        ggml_backend_tensor_get(y1s[k], o1.data(), 0, ggml_nbytes(y1s[k]));
        ggml_backend_tensor_get(y2s[k], o2.data(), 0, ggml_nbytes(y2s[k]));
        double m1 = 0, m2 = 0;
        for (int64_t i = 0; i < OL * OC; i++) {
            double e1 = fabs(o1[i] - (float) r1[i]); if (e1 > m1) m1 = e1;
            double e2 = fabs(o2[i] - (float) r2[i]); if (e2 > m2) m2 = e2;
        }
        printf("  [%s] chainN pair %d dil=%d convs1 max_abs=%.3e convs2(res) max_abs=%.3e %s\n",
               bk_name, k, dil, m1, m2, (m1 < 5e-4 && m2 < 5e-4) ? "OK" : "FAIL");
        if (m2 > worst) { worst = m2; worst_pair = k; }
        if (m1 >= 5e-4 || m2 >= 5e-4) { ggml_gallocr_free(ga); ggml_free(ctx); return false; }
        href = r2;
    }
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    printf("  [%s] chainN ALL %d pairs OK (worst convs2 max_abs=%.3e at pair %d)\n", bk_name, N, worst, worst_pair);
    return true;
}

static bool run_chain(ggml_backend_t bk, const char * bk_name) {
    const int K = 3, IC = 256, OC = 256, T = 13776, dil = 1;
    const int pad = (K / 2) * dil;
    const int64_t OL = T;   // same padding

    ggml_init_params ip = { 256ull * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, IC, OC);
    ggml_tensor * w2 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, IC, OC);
    ggml_tensor * b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OC);
    ggml_tensor * b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OC);
    ggml_tensor * h  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, IC);

    lcg_state = 0xcafebabeu;
    std::vector<float> w1v(ggml_nbytes(w1) / 4), w2v(ggml_nbytes(w2) / 4),
                       b1v(OC), b2v(OC), hv(ggml_nbytes(h) / 4);
    for (auto & v : w1v) v = frand() * 0.2f;
    for (auto & v : w2v) v = frand() * 0.2f;
    for (auto & v : b1v) v = frand() * 0.5f;
    for (auto & v : b2v) v = frand() * 0.5f;
    for (auto & v : hv)  v = frand() * 2.0f;

    ggml_tensor * y1 = ggml_conv_direct_1d_fused(ctx, w1, h, b1, nullptr, pad, dil, 0.1f, 1.0f, 0.1f);
    ggml_tensor * y2 = ggml_conv_direct_1d_fused(ctx, w2, y1, b2, h, pad, dil, 0.0f, 1.0f, 0.0f);
    ggml_set_input(w1); ggml_set_input(w2); ggml_set_input(b1); ggml_set_input(b2); ggml_set_input(h);
    ggml_set_output(y2);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y2);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(bk));
    if (!ggml_gallocr_alloc_graph(ga, gf)) { printf("chain alloc failed\n"); return false; }

    ggml_backend_tensor_set(w1, w1v.data(), 0, w1v.size() * sizeof(float));
    ggml_backend_tensor_set(w2, w2v.data(), 0, w2v.size() * sizeof(float));
    ggml_backend_tensor_set(b1, b1v.data(), 0, b1v.size() * sizeof(float));
    ggml_backend_tensor_set(b2, b2v.data(), 0, b2v.size() * sizeof(float));
    ggml_backend_tensor_set(h, hv.data(), 0, hv.size() * sizeof(float));

    if (ggml_backend_graph_compute(bk, gf) != GGML_STATUS_SUCCESS) { printf("chain compute failed\n"); return false; }

    std::vector<float> out(OL * OC);
    ggml_backend_tensor_get(y2, out.data(), 0, ggml_nbytes(y2));
    ggml_gallocr_free(ga);

    // reference
    auto convref = [&](const std::vector<float> & wv, const std::vector<float> & bv,
                       const float * xd, float slope, float in_slope) {
        std::vector<double> r(OL * OC, 0.0);
        for (int oc = 0; oc < OC; oc++)
            for (int t = 0; t < OL; t++) {
                double acc = bv[oc];
                for (int ic = 0; ic < IC; ic++)
                    for (int kw = 0; kw < K; kw++) {
                        const int xi = t + kw * dil - pad;
                        if (xi >= 0 && xi < T) {
                            double v = xd[(int64_t) ic * T + xi];
                            if (in_slope != 0.0f) v = v > 0 ? v : in_slope * v;
                            acc += (double) wv[((int64_t) oc * IC + ic) * K + kw] * v;
                        }
                    }
                if (slope != 0.0f) acc = acc > 0 ? acc : slope * acc;
                r[(int64_t) oc * OL + t] = acc;
            }
        return r;
    };
    auto r1 = convref(w1v, b1v, hv.data(), 0.1f, 0.1f);
    std::vector<float> y1f(r1.begin(), r1.end());
    auto r2 = convref(w2v, b2v, y1f.data(), 0.0f, 0.0f);
    for (int64_t i = 0; i < OL * OC; i++) r2[i] += hv[i];   // res = h

    double max_abs = 0;
    for (int64_t i = 0; i < OL * OC; i++) {
        const double d = fabs(out[i] - r2[i]);
        if (d > max_abs) max_abs = d;
    }
    const bool ok = max_abs < 5e-4;
    printf("  [%s] chain convs1->convs2(res=h) K=3 IC=OC=256 T=13776 -> max_abs=%.3e %s\n",
           bk_name, max_abs, ok ? "OK" : "FAIL");
    ggml_free(ctx);
    return ok;
}

// Full resblock replica: 3 (convs1 -> convs2 with res=h) pairs in sequence,
// matching the real level-0 block (K=3, dils {1,3,5}, IC=OC=256, T=13776).
// h is carried across the pairs; each convs2 folds res=h in its epilogue.
// Every conv output is checked against an fp64 reference so the FIRST bad
// node in the chain is identified, reproducing the graph-mode failure shape.
static bool run_resblock(ggml_backend_t bk, const char * bk_name) {
    const int K = 3, IC = 256, OC = 256, T = 13776;
    const int dils[3] = {1, 3, 5};
    const int64_t OL = T;

    ggml_init_params ip = { 512ull * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    lcg_state = 0x12344321u;
    std::vector<float> hv(ggml_nbytes(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, IC)) / 4);
    for (auto & v : hv) v = frand() * 2.0f;

    ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, IC);
    bool fails = false;

    // host reference state (fp64), mirrors h
    std::vector<double> href(hv.begin(), hv.end());

    for (int kpi = 0; kpi < 3; ++kpi) {
        const int dil = dils[kpi];
        const int pad = (K / 2) * dil;
        // two weight/bias sets: convs1 (in_slope + slope), convs2 (res)
        ggml_tensor * w1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, IC, OC);
        ggml_tensor * w2 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, IC, OC);
        ggml_tensor * b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OC);
        ggml_tensor * b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OC);
        std::vector<float> w1v(ggml_nbytes(w1) / 4), w2v(ggml_nbytes(w2) / 4), b1v(OC), b2v(OC);
        for (auto & v : w1v) v = frand() * 0.2f;
        for (auto & v : w2v) v = frand() * 0.2f;
        for (auto & v : b1v) v = frand() * 0.5f;
        for (auto & v : b2v) v = frand() * 0.5f;

        ggml_tensor * y1 = ggml_conv_direct_1d_fused(ctx, w1, h, b1, nullptr,
                                                     pad, dil, 0.1f, 1.0f, 0.1f);
        ggml_tensor * y2 = ggml_conv_direct_1d_fused(ctx, w2, y1, b2, h,
                                                     pad, dil, 0.0f, 1.0f, 0.0f);
        ggml_set_input(w1); ggml_set_input(w2); ggml_set_input(b1); ggml_set_input(b2);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y2);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(bk));
        if (!ggml_gallocr_alloc_graph(ga, gf)) { printf("rb alloc failed\n"); return false; }
        ggml_backend_tensor_set(w1, w1v.data(), 0, w1v.size() * sizeof(float));
        ggml_backend_tensor_set(w2, w2v.data(), 0, w2v.size() * sizeof(float));
        ggml_backend_tensor_set(b1, b1v.data(), 0, b1v.size() * sizeof(float));
        ggml_backend_tensor_set(b2, b2v.data(), 0, b2v.size() * sizeof(float));
        if (kpi == 0) ggml_backend_tensor_set(h, hv.data(), 0, hv.size() * sizeof(float));
        if (ggml_backend_graph_compute(bk, gf) != GGML_STATUS_SUCCESS) { printf("rb compute failed\n"); return false; }

        std::vector<float> o1(OL * OC), o2(OL * OC);
        ggml_backend_tensor_get(y1, o1.data(), 0, ggml_nbytes(y1));
        ggml_backend_tensor_get(y2, o2.data(), 0, ggml_nbytes(y2));
        ggml_gallocr_free(ga);

        auto convref = [&](const std::vector<float> & wv, const std::vector<float> & bv,
                           const std::vector<double> & xd, float slope, float in_slope, bool addr) {
            std::vector<double> r(OL * OC, 0.0);
            for (int oc = 0; oc < OC; oc++)
                for (int t = 0; t < OL; t++) {
                    double acc = bv[oc];
                    for (int ic = 0; ic < IC; ic++)
                        for (int kw = 0; kw < K; kw++) {
                            const int xi = t + kw * dil - pad;
                            if (xi >= 0 && xi < T) {
                                double v = xd[(int64_t) ic * T + xi];
                                if (in_slope != 0.0f) v = v > 0 ? v : in_slope * v;
                                acc += (double) wv[((int64_t) oc * IC + ic) * K + kw] * v;
                            }
                        }
                    if (addr) acc += href[(int64_t) oc * OL + t];
                    if (slope != 0.0f) acc = acc > 0 ? acc : slope * acc;
                    r[(int64_t) oc * OL + t] = acc;
                }
            return r;
        };
        auto r1 = convref(w1v, b1v, href, 0.1f, 0.1f, false);
        std::vector<double> r1f(r1.begin(), r1.end());
        auto r2 = convref(w2v, b2v, r1f, 0.0f, 0.0f, true);

        double m1 = 0, m2 = 0;
        for (int64_t i = 0; i < OL * OC; i++) {
            const double d1 = fabs(o1[i] - (float) r1[i]);
            const double d2 = fabs(o2[i] - (float) r2[i]);
            if (d1 > m1) m1 = d1;
            if (d2 > m2) m2 = d2;
        }
        printf("  [%s] resblock pair %d dil=%d  convs1 max_abs=%.3e  convs2(res) max_abs=%.3e  %s\n",
               bk_name, kpi, dil, m1, m2, (m1 < 5e-4 && m2 < 5e-4) ? "OK" : "FAIL");
        if (m1 >= 5e-4 || m2 >= 5e-4) fails = true;
        // advance the running h to the fp64 reference of y2 for the next pair
        href = r2;
        // reload h tensor with the reference so the next pair starts clean
        std::vector<float> hf(r2.begin(), r2.end());
        ggml_backend_tensor_set(h, hf.data(), 0, ggml_nbytes(h));
    }
    ggml_free(ctx);
    return !fails;
}

// Replicates the exact production sub-graph node_5..node_10 using the REAL
// dumped data (dump_cpu2): node5 CONT [8,1723,256] -> reshape [13784,256]
// -> view [13776,256] -> cont -> convs1(w1,h,in_slope=0.1,leaky=0.1) ->
// convs2(w2, y, res=h).  Compares against dump_cpu2 node_0009/0010.  PCNSF_REPRO=1.
static bool run_repro(ggml_backend_t bk, const char * bk_name) {
    const char * dir = getenv("PCNSF_DUMP_DIR");
    if (!dir) dir = "dump_cpu2";
    const int K = 3, C = 256, T = 13776;
    const int64_t A5 = 8ll * 1723 * 256;   // node5 elements
    const bool full_rb = getenv("PCNSF_REPRO_FULL") != nullptr;  // full level-0: 3 parallel resblocks + MRF sum

    auto loadf = [&](const char * name, std::vector<float> & out) -> bool {
        char path[512]; snprintf(path, sizeof path, "%s/%s", dir, name);
        FILE * f = fopen(path, "rb");
        if (!f) { printf("repro: cannot open %s\n", path); return false; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        out.resize(sz / 4);
        const size_t rd = fread(out.data(), 1, sz, f); fclose(f);
        return rd == (size_t) sz;
    };
    std::vector<float> n5, n9ref, n10ref, w1v, w2v, b1v, b2v;
    if (!loadf("node_0005.f32", n5) || !loadf("node_0009.f32", n9ref) ||
        !loadf("node_0010.f32", n10ref) || n5.size() != (size_t) A5) return false;

    // weights from GGUF via a tiny ad-hoc env-free loader would be heavy;
    // instead accept pre-extracted raw bins: w_rbs0c10.f32 etc.
    if (!loadf("repro_w1.f32", w1v) || !loadf("repro_w2.f32", w2v) ||
        !loadf("repro_b1.f32", b1v) || !loadf("repro_b2.f32", b2v)) return false;

    const size_t voff_b = (size_t) (getenv("PCNSF_VOFF") ? atoi(getenv("PCNSF_VOFF")) : 4) * sizeof(float);

    ggml_init_params ip = { 512ull * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    // node5 data layout is contiguous [8,1723,256] == flat (13784,256)
    ggml_tensor * a5  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 13784, 256);
    ggml_tensor * vw  = ggml_view_2d(ctx, a5, T, 256, a5->nb[1], voff_b);   // node7 view
    ggml_tensor * h   = ggml_cont(ctx, vw);                                 // node8 cont

    ggml_tensor * w1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, C, C);
    ggml_tensor * b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
    ggml_tensor * w2 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, C, C);
    ggml_tensor * b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
    ggml_set_input(a5); ggml_set_input(w1); ggml_set_input(b1);
    ggml_set_input(w2); ggml_set_input(b2);

    ggml_tensor * y1 = ggml_conv_direct_1d_fused(ctx, w1, h, b1, nullptr, 1, 1, 0.1f, 1.0f, 0.1f);
    ggml_tensor * y2 = ggml_conv_direct_1d_fused(ctx, w2, y1, b2, h, 1, 1, 0.0f, 1.0f, 0.0f);
    ggml_set_output(y1); ggml_set_output(y2);

    // full level-0 replica: all 3 resblocks (nodes 11..26 follow the same
    // (convs1 dil in {1,3,5} -> convs2 res) pattern with real weights)
    std::vector<ggml_tensor *> extra_w, extra_b, extra_out;
    std::vector<std::vector<float>> extra_wv, extra_bv, extra_ref;
    ggml_tensor * hh = y2;
    // true MRF: rb0/rb1/rb2 all consume the SAME h; outputs summed pairwise.
    // y1/y2 above already computed rb0 pair k0 (nodes 9,10) feeding off h.
    ggml_tensor * mrf_sum = y2;   // rb0 out so far
    if (full_rb) {
        // resblock kernel per rb (HiFiGAN kernel_sizes = [3,7,11]); dils {1,3,5}
        const int rbK[3]  = {3, 7, 11};
        const int dils[3] = {1, 3, 5};
        // rb0 pairs k1,k2 still hang off h (nodes 11..14)
        ggml_tensor * hrbs[3];
        hrbs[0] = y2;  // rb0 k0 done
        for (int rb = 0; rb < 3; ++rb) {
            // rb0 already has k0 = y2, i.e. hrbs[0] after k0; we must add k1,k2
            // For rb0 we continue from hrbs[0]; for rb1/rb2 we start from h.
            ggml_tensor * hh_k = (rb == 0) ? hrbs[0] : h;
            ggml_tensor * res_carry = (rb == 0) ? nullptr : h;  // residual at each pair = the input h of the rb
            const int kstart = (rb == 0) ? 1 : 0;
            ggml_tensor * hrb = h;   // h inside this rb (running)
            if (rb == 0) hrb = hrbs[0];
            ggml_tensor * hres = h;
            for (int k = kstart; k < 3; ++k) {
                const int Kw = rbK[rb];
                const int dil = dils[k];
                const int pad1 = (Kw / 2) * dil;
                const int pad2 = (Kw / 2) * 1;
                char nm[64];
                snprintf(nm, sizeof nm, "repro_w_r%d_c1_%d.f32", rb, k);
                std::vector<float> wv1; if (!loadf(nm, wv1)) return false;
                snprintf(nm, sizeof nm, "repro_w_r%d_c2_%d.f32", rb, k);
                std::vector<float> wv2; if (!loadf(nm, wv2)) return false;
                snprintf(nm, sizeof nm, "repro_b_r%d_c1_%d.f32", rb, k);
                std::vector<float> bv1; if (!loadf(nm, bv1)) return false;
                snprintf(nm, sizeof nm, "repro_b_r%d_c2_%d.f32", rb, k);
                std::vector<float> bv2; if (!loadf(nm, bv2)) return false;
                if (wv1.size() != (size_t) Kw * C * C) { printf("repro: bad w size %zu for %s\n", wv1.size(), nm); return false; }
                ggml_tensor * ww1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Kw, C, C);
                ggml_tensor * bb1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
                ggml_tensor * ww2 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Kw, C, C);
                ggml_tensor * bb2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
                ggml_set_input(ww1); ggml_set_input(bb1); ggml_set_input(ww2); ggml_set_input(bb2);
                ggml_tensor * yy1 = ggml_conv_direct_1d_fused(ctx, ww1, hrb, bb1, nullptr, pad1, dil, 0.1f, 1.0f, 0.1f);
                ggml_tensor * yy2 = ggml_conv_direct_1d_fused(ctx, ww2, yy1, bb2, hrb, pad2, 1, 0.0f, 1.0f, 0.0f);
                ggml_set_output(yy1); ggml_set_output(yy2);
                extra_w.push_back(ww1); extra_b.push_back(bb1);
                extra_w.push_back(ww2); extra_b.push_back(bb2);
                extra_wv.push_back(wv1); extra_wv.push_back(wv2);
                extra_bv.push_back(bv1); extra_bv.push_back(bv2);
                // expected node index: rb*(2*3) + k*2 + 9 base offset
                const int node1 = 9 + rb * 6 + k * 2;      // convs1
                snprintf(nm, sizeof nm, "node_%04d.f32", node1);
                std::vector<float> rref1; if (!loadf(nm, rref1)) return false;
                snprintf(nm, sizeof nm, "node_%04d.f32", node1 + 1);  // convs2
                std::vector<float> rref2; if (!loadf(nm, rref2)) return false;
                extra_ref.push_back(rref1);
                extra_ref.push_back(rref2);
                extra_out.push_back(yy1);
                extra_out.push_back(yy2);
                hrb = yy2;
                (void) res_carry;
                (void) hres;
            }
            hrbs[rb] = hrb;
        }
        mrf_sum = ggml_add(ctx, ggml_add(ctx, hrbs[0], hrbs[1]), hrbs[2]);
    }
    hh = mrf_sum;

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, hh);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(bk));
    if (!ggml_gallocr_alloc_graph(ga, gf)) { printf("repro alloc failed\n"); return false; }

    ggml_backend_tensor_set(a5, n5.data(), 0, n5.size() * sizeof(float));
    ggml_backend_tensor_set(w1, w1v.data(), 0, w1v.size() * sizeof(float));
    ggml_backend_tensor_set(b1, b1v.data(), 0, b1v.size() * sizeof(float));
    ggml_backend_tensor_set(w2, w2v.data(), 0, w2v.size() * sizeof(float));
    ggml_backend_tensor_set(b2, b2v.data(), 0, b2v.size() * sizeof(float));
    for (size_t i = 0; i < extra_w.size(); i++) {
        ggml_backend_tensor_set(extra_w[i], extra_wv[i].data(), 0, extra_wv[i].size() * sizeof(float));
        ggml_backend_tensor_set(extra_b[i], extra_bv[i].data(), 0, extra_bv[i].size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(bk, gf) != GGML_STATUS_SUCCESS) { printf("repro compute failed\n"); return false; }

    std::vector<float> o1(T * C), o2(T * C);
    ggml_backend_tensor_get(y1, o1.data(), 0, ggml_nbytes(y1));
    ggml_backend_tensor_get(y2, o2.data(), 0, ggml_nbytes(y2));
    double m1 = 0, m2 = 0;
    for (int64_t i = 0; i < (int64_t) T * C; i++) {
        double e1 = fabs((double) o1[i] - n9ref[i]);  if (e1 > m1) m1 = e1;
        double e2 = fabs((double) o2[i] - n10ref[i]); if (e2 > m2) m2 = e2;
    }
    bool ok = (m1 < 5e-4 && m2 < 5e-4);
    printf("  [%s] repro node6..10: convs1(vs n9) max_abs=%.3e  convs2(vs n10) max_abs=%.3e  %s\n",
           bk_name, m1, m2, ok ? "OK" : "FAIL");
    if (full_rb) {
        // vs CPU reference dumps where our model is confirmed (rb0 k1,k2 / rb1
        // all pairs = nodes 11..20); beyond that compare the FINAL MRF output
        // only when running both backends (main() runs them separately, so
        // instead the MRF result is validated cross-backend via its own dump:
        // here we just report max_abs vs the CPU dump node for the ADD at
        // node_0028 when PCNSF_DUMP_DIR points at dump_cpu2).
        std::vector<float> oo(T * C);
        for (size_t i = 0; i < extra_out.size() && i < 12; i++) {
            ggml_backend_tensor_get(extra_out[i], oo.data(), 0, ggml_nbytes(extra_out[i]));
            double mm = 0;
            for (int64_t j = 0; j < (int64_t) T * C; j++) {
                double e = fabs((double) oo[j] - extra_ref[i][j]); if (e > mm) mm = e;
            }
            printf("  [%s] repro full: stage %zu max_abs=%.3e %s\n",
                   bk_name, i, mm, mm < 5e-4 ? "OK" : "FAIL");
            ok = ok && (mm < 5e-4);
        }
        // always dump the MRF sum for cross-backend side-by-side comparison
        std::vector<float> ms(T * C);
        ggml_backend_tensor_get(mrf_sum, ms.data(), 0, ggml_nbytes(mrf_sum));
        char outp[512];
        snprintf(outp, sizeof outp, "mrf_sum_%s.f32", bk_name);
        FILE * f = fopen(outp, "wb");
        if (f) { fwrite(ms.data(), 4, ms.size(), f); fclose(f); }
        printf("  [%s] repro full: mrf_sum -> %s\n", bk_name, outp);
    }
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    // unbuffer stdout: ggml-vk teardown can crash the process on exit
    // (known 0xC0000005) and full buffering would swallow the case table
    setvbuf(stdout, nullptr, _IONBF, 0);
    ggml_time_init();
    const char * want = argc > 1 ? argv[1] : "vulkan";

    // enumerate vulkan device
    std::vector<ggml_backend_t> backends;
    if (strcmp(want, "vulkan") == 0) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name("Vulkan0");
        if (!dev) {
            // fall back: first non-CPU device
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_CPU) { dev = d; break; }
            }
        }
        if (!dev) { printf("no vulkan device\n"); return 1; }
        backends.push_back(ggml_backend_dev_init(dev, nullptr));
    } else if (strcmp(want, "cpu") == 0) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!dev) { printf("no cpu device\n"); return 1; }
        backends.push_back(ggml_backend_dev_init(dev, nullptr));
    } else {
        printf("usage: %s [vulkan|cpu]\n", argv[0]);
        return 1;
    }
    const char * bk_name = want;

    std::vector<Case> cases = {
        // K, IC, OC, T, dil, bias, res, slope, in_scale, in_slope
        { 3,  16,  16,  70, 1, true,  false, 0.0f, 1.0f, 0.0f },
        { 3,  32,  64, 200, 1, true,  false, 0.0f, 1.0f, 0.0f },
        { 3, 128, 128, 300, 1, true,  false, 0.0f, 1.0f, 0.0f },
        { 7,  16,  16,  70, 1, true,  false, 0.0f, 1.0f, 0.0f },
        { 7,  64, 128, 300, 3, true,  false, 0.0f, 1.0f, 0.0f },
        { 7, 128, 256, 300, 1, true,  false, 0.0f, 1.0f, 0.0f },
        {11,  16,  16,  70, 1, true,  false, 0.0f, 1.0f, 0.0f },
        {11,  32,  32, 200, 5, true,  false, 0.0f, 1.0f, 0.0f },
        {11, 128, 128, 300, 5, true,  false, 0.0f, 1.0f, 0.0f },
        {16, 512,1024, 100, 1, true,  false, 0.0f, 1.0f, 0.0f },
        { 8, 256, 512, 100, 1, true,  false, 0.0f, 1.0f, 0.0f },
        { 3,  16,  16,  70, 1, true,  true, 0.1f, 0.5f, 0.1f },
        { 7,  64, 128, 300, 3, true,  true, 0.1f, 0.25f, 0.1f },
        {11, 128, 128, 300, 5, false, true, 0.01f, 1.0f, 0.0f },
        {16, 512,1024, 100, 1, true,  false, 0.0f, 0.333f, 0.1f },
        // exact graph node shapes (level-0 resblock, w128 line)
        { 3, 256, 256, 13776, 1, true, false, 0.1f, 1.0f, 0.1f },  // convs1.0
        { 3, 256, 256, 13776, 1, true, true, 0.0f, 1.0f, 0.0f },   // convs2.0 shape
        { 7, 256, 256, 13776, 3, true, false, 0.1f, 1.0f, 0.1f },  // convs1.1
        {11, 256, 256, 13776, 5, true, false, 0.1f, 1.0f, 0.1f },  // convs1.2
    };

    // production sub-graph replica: PCNSF_REPRO=1 (uses dump_cpu2 + repro_*.f32)
    if (getenv("PCNSF_REPRO")) {
        run_case(cases[0], backends[0], bk_name, true);   // warmup / JIT
        const bool ok = run_repro(backends[0], bk_name);
        printf("%s: repro %s\n", bk_name, ok ? "OK" : "FAIL");
        ggml_backend_free(backends[0]);
        return ok ? 0 : 1;
    }

    // chain-only mode: PCNSF_CHAIN=1 runs ONLY run_chain + run_resblock
    // (the multi-conv h-carry structure that mirrors the full-graph resblock)
    if (getenv("PCNSF_CHAIN")) {
        run_case(cases[0], backends[0], bk_name, true);   // warmup / JIT
        int fails = 0;
        if (getenv("PCNSF_CHAINN")) {
            if (!run_chain_n(backends[0], bk_name)) fails++;
        } else {
            if (!run_chain(backends[0], bk_name)) fails++;
            if (getenv("PCNSF_RESBLOCK") && !run_resblock(backends[0], bk_name)) fails++;
        }
        printf("%s: chain %s\n", bk_name, fails ? "FAIL" : "OK");
        ggml_backend_free(backends[0]);
        return fails ? 1 : 0;
    }

    // focused mode: PCNSF_FOCUS=K,IC,OC,T,dil[,bias,res,slope,in_scale,in_slope]
    if (const char * f = getenv("PCNSF_FOCUS")) {
        Case c{3,256,256,13776,1,true,false,0.1f,1.0f,0.1f};
        if (sscanf(f, "%d,%d,%d,%d,%d", &c.K, &c.IC, &c.OC, &c.T, &c.dil) >= 5) {
            const char * p = f; int nb=0; while(*p && nb<5){ if(*p==',') nb++; p++; }
            // optional tail: bias,res,slope,in_scale,in_slope
            int bi, re; float sl, isc, isl;
            if (sscanf(p, "%d,%d,%f,%f,%f", &bi, &re, &sl, &isc, &isl) == 5) {
                c.bias=bi; c.res=re; c.slope=sl; c.in_scale=isc; c.in_slope=isl;
            }
        }
        // bench mode: PCNSF_BENCH=N → time N iterations (hw clock) after warmup,
        // using the variant forced by GGML_VK_CONV_DIRECT_VARIANT (or default pick)
        const int n_bench = getenv("PCNSF_BENCH") ? atoi(getenv("PCNSF_BENCH")) : 0;
        if (n_bench > 0) {
            run_case(c, backends[0], bk_name, true);                 // warmup / JIT
            double best = 1e300, worst = 0.0, sum = 0.0;
            for (int it = 0; it < n_bench; it++) {
                const auto t0 = std::chrono::steady_clock::now();
                run_case(c, backends[0], bk_name, true);             // compute-only (check_only skips fp64 ref)
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                best = ms < best ? ms : best;  worst = ms > worst ? ms : worst;  sum += ms;
            }
            const char * vf = getenv("GGML_VK_CONV_DIRECT_VARIANT");
            printf("[bench] variant=%s K=%d IC=%d OC=%d T=%d dil=%d  n=%d  mean=%.3f ms  min=%.3f  max=%.3f\n",
                   vf ? vf : "def", c.K, c.IC, c.OC, c.T, c.dil, n_bench, sum / n_bench, best, worst);
            ggml_backend_free(backends[0]);
            return 0;
        }
        run_case(c, backends[0], bk_name, true);    // warmup
        const bool ok = run_case(c, backends[0], bk_name, false);
        printf("%s: focus %s\n", bk_name, ok ? "OK" : "FAIL");
        ggml_backend_free(backends[0]);
        return ok ? 0 : 1;
    }

    // warmup (JIT)
    run_case(cases[0], backends[0], bk_name, true);

    int fails = 0;
    for (const auto & cs : cases) {
        if (!run_case(cs, backends[0], bk_name, false)) fails++;
    }
    if (!run_chain(backends[0], bk_name)) fails++;
    if (!run_resblock(backends[0], bk_name)) fails++;

    printf("%s: %d/%d failed\n", bk_name, fails, (int) cases.size() + 2);
    ggml_backend_free(backends[0]);
    return fails ? 1 : 0;
}
