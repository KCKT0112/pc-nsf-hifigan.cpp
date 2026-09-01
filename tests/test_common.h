// SPDX-License-Identifier: MPL-2.0
// Small golden-comparison helpers shared by the C++ tests.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace pcnsf_test {

inline int failures = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++pcnsf_test::failures;                                         \
        }                                                                  \
    } while (0)

inline std::string golden_path(const char * name) {
    return std::string(PCNSF_GOLDEN_DIR) + "/" + name;
}

inline bool read_raw(const std::string & path, std::vector<float> & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    if (sz <= 0 || sz % 4) { std::fclose(f); return false; }
    std::fseek(f, 0, SEEK_SET);
    out.resize(sz / 4);
    size_t got = std::fread(out.data(), 4, out.size(), f);
    std::fclose(f);
    return got == out.size();
}

// max abs + rmse comparison with a relative tolerance per element.
inline bool rmse_ok(const std::vector<float> & a, const std::vector<float> & b,
                    float rmse_max, float abs_max) {
    if (a.size() != b.size()) return false;
    if (a.empty()) return true;
    double se = 0; float mx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = (double) a[i] - (double) b[i];
        se += d * d;
        mx = std::max(mx, (float) std::fabs(d));
    }
    double rmse = std::sqrt(se / (double) a.size());
    return (float) rmse <= rmse_max && mx <= abs_max;
}

// Floor-aware mel comparison.  log(clip) means any float difference on the
// near-clip (acc ~= clip) region is amplified non-linearly, so the absolute
// bound is split in two: a tight bound where ref > ref_floor (real signal),
// and a loose absolute bound below it (the log-amplified near-floor zone).
// rmse covers all elements -> gross wiring errors still fail hard.
inline bool rmse_floor_ok(const std::vector<float> & a, const std::vector<float> & b,
                          float rmse_max, float abs_tight, float ref_floor,
                          float abs_loose) {
    if (a.size() != b.size()) return false;
    if (a.empty()) return true;
    double se = 0; float mx = 0, mx_loose = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = (double) a[i] - (double) b[i];
        se += d * d;
        if (b[i] > ref_floor) mx = std::max(mx, (float) std::fabs(d));
        else                  mx_loose = std::max(mx_loose, (float) std::fabs(d));
    }
    double rmse = std::sqrt(se / (double) a.size());
    return (float) rmse <= rmse_max && mx <= abs_tight && mx_loose <= abs_loose;
}

}  // namespace pcnsf_test
