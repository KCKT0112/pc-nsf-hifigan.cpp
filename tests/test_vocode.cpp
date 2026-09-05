// SPDX-License-Identifier: MPL-2.0
// test_vocode: optional end-to-end vocode (requires a real GGUF).
// Compiles only when PCNSF_MODEL_GGUF is set at configure time; the CI path
// skips it.  Expected wav length == T*hop_size, finite, non-silent for vocal.
#include "test_common.h"
#include "pc_nsf_hifigan/hifigan.h"
#include "pc_nsf_hifigan/mel.h"

#include <cstdint>
#include <cmath>
#include <vector>

int main(int argc, char ** argv) {
    pc_nsf_hifigan::HifiganModel m(PCNSF_MODEL_GGUF, 4, argc > 1 ? argv[1] : "F32");

    const bool compare_min_k = argc > 2 && std::string(argv[2]) == "compare-min-k";
    const int T = compare_min_k ? 32 : (m.hop_size > 512 ? 128 : 96);
    std::vector<float> mel(T * m.num_mels, compare_min_k ? -4.0f : -11.0f);
    std::vector<float> f0(T, 261.6f);            // C4 constant pitch
    std::vector<float> wav;
    pc_nsf_hifigan::hifigan_run(m, mel.data(), f0.data(), T, wav);

    if (compare_min_k) {
        // Force different mixtures of direct and composed convolutions.
        for (const char * minimum : {"3", "8", "64"}) {
#ifdef _WIN32
            _putenv_s("PCNSF_DIRECT_MIN_K", minimum);
#else
            setenv("PCNSF_DIRECT_MIN_K", minimum, 1);
#endif
            std::vector<float> fallback;
            pc_nsf_hifigan::hifigan_run(m, mel.data(), f0.data(), T, fallback);
            CHECK(fallback.size() == wav.size());
            for (size_t i = 0; i < std::min(wav.size(), fallback.size()); ++i)
                CHECK(std::isfinite(fallback[i]) && std::fabs(wav[i] - fallback[i]) < 1e-4f);
        }
    }

    const int expect = T * m.hop_size;
    CHECK((int) wav.size() == expect);

    float e = 0, mx = 0;
    for (float v : wav) { e += v * v; mx = std::max(mx, std::fabs(v)); }
    CHECK(std::isfinite(e) && e > 0.0f && mx <= 1.0f + 1e-2f);
    std::fprintf(stderr, "vocode produced %d samples (rms=%.4f peak=%.4f)\n",
                 (int) wav.size(), std::sqrt(e / wav.size()), mx);

    if (pcnsf_test::failures) return 1;
    std::printf("test_vocode OK\n");
    return 0;
}
