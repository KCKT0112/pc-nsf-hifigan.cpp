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

int main() {
    pc_nsf_hifigan::HifiganModel m(PCNSF_MODEL_GGUF, 4);

    const int T = m.hop_size > 512 ? 128 : 96;   // keep short for CI speed
    std::vector<float> mel(T * m.num_mels, -11.0f);
    std::vector<float> f0(T, 261.6f);            // C4 constant pitch
    std::vector<float> wav;
    pc_nsf_hifigan::hifigan_run(m, mel.data(), f0.data(), T, wav);

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
