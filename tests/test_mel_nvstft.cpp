// SPDX-License-Identifier: MPL-2.0
// test_mel_nvstft: mel_nvstft (44.1k, slaney, nvSTFT config) vs golden.
#include "test_common.h"
#include "pc_nsf_hifigan/mel.h"

#include <vector>

#include <cstdlib>

using namespace pcnsf_test;

namespace {
std::vector<float> make_signal_44k() {
    const int n = 22050;
    std::vector<float> wav(n);
    for (int i = 0; i < n; ++i) {
        const float t = (float) i / 44100.0f;
        wav[i] = 0.3f * std::sin(2.0f * 3.14159265f * 440.0f * t)
               + 0.15f * std::sin(2.0f * 3.14159265f * 880.0f * t)
               + 0.05f * std::sin(2.0f * 3.14159265f * 1320.0f * t);
    }
    return wav;
}
}  // namespace

int main() {
    std::vector<float> wav = make_signal_44k();
    std::vector<float> got = pc_nsf_hifigan::mel_nvstft(
        wav.data(), wav.size(), 44100, 2048, 2048, 512, 128, 40.0f, 16000.0f, 1e-5f);

    std::vector<float> ref;
    CHECK(read_raw(pcnsf_test::golden_path("mel_nvstft_44k.bin"), ref));
    if (ref.empty() && pcnsf_test::failures == 0) {
        std::fprintf(stderr, "  golden file missing — did gen_golden run?\n");
        return 1;
    }
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "  size mismatch: got %zu ref %zu\n", got.size(), ref.size());
        return 1;
    }
    if (std::getenv("MEL_DUMP_GOT")) {
        FILE * f = std::fopen("got_nvstft.bin", "wb");
        if (f) { std::fwrite(got.data(), 4, got.size(), f); std::fclose(f); }
        FILE * g = std::fopen("ref_nvstft.bin", "wb");
        if (g) { std::fwrite(ref.data(), 4, ref.size(), g); std::fclose(g); }
        std::fprintf(stderr, "  dumped got/ref\n");
    }
    // log(clip) amplifies near-floor diffs non-linearly; use the floor-aware
    // helper: tight abs above ref_floor, loose abs in the log-amplified zone.
    // Measured on the 3-component 44.1k tone: tight max 0.0023, floor max 1.18.
    CHECK(pcnsf_test::rmse_floor_ok(got, ref, 0.15f, 0.05f, -5.0f, 1.5f));

    if (pcnsf_test::failures) return 1;
    std::printf("test_mel_nvstft OK (%zu frames)\n", got.size() / 128);
    return 0;
}
