// test_mel: MelExtractor (htk=True, 16k RMVPE-style config) vs golden.
#include "test_common.h"
#include "pc_nsf_hifigan/mel.h"

#include <vector>

#include <cstdlib>

using namespace pcnsf_test;

namespace {
std::vector<float> make_signal_16k() {
    const int n = 3200;
    std::vector<float> wav(n);
    for (int i = 0; i < n; ++i) {
        const float t = (float) i / 16000.0f;
        wav[i] = 0.3f * std::sin(2.0f * 3.14159265f * 440.0f * t)
               + 0.15f * std::sin(2.0f * 3.14159265f * 880.0f * t)
               + 0.05f * std::sin(2.0f * 3.14159265f * 1320.0f * t);
    }
    return wav;
}
}  // namespace

int main() {
    pc_nsf_hifigan::MelConfig cfg;
    cfg.sample_rate = 16000;
    cfg.n_fft = 1024;
    cfg.win_length = 1024;
    cfg.hop_length = 160;
    cfg.n_mels = 128;
    cfg.fmin = 30.0f;
    cfg.fmax = 8000.0f;
    cfg.clip_val = 1e-5f;
    cfg.htk = true;

    pc_nsf_hifigan::MelExtractor ext(cfg);
    std::vector<float> wav = make_signal_16k();
    if (std::getenv("MEL_DUMP_FB2")) {
        // re-derive htk filterbank C++-side and dump (via public config only
        // we cannot; instead recompute in test copy identical to mel.cpp)
        for (int m = 0; m < 3; ++m) {
            std::fprintf(stderr, "dump not available via public API\n");
        }
    }
    std::vector<float> got = ext.forward(wav.data(), wav.size());

    CHECK(ext.num_frames(wav.size()) == (int) (got.size() / (size_t) cfg.n_mels));
    std::vector<float> ref;
    CHECK(read_raw(pcnsf_test::golden_path("mel_ng_16k.bin"), ref));
    if (ref.empty() && pcnsf_test::failures == 0) {
        std::fprintf(stderr, "  golden file missing — did gen_golden run?\n");
        return 1;
    }
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "  size mismatch: got %zu ref %zu\n", got.size(), ref.size());
        return 1;
    }
    if (std::getenv("MEL_DUMP_GOT")) {
        FILE * f = std::fopen("got_ng.bin", "wb");
        if (f) { std::fwrite(got.data(), 4, got.size(), f); std::fclose(f); }
        FILE * g = std::fopen("ref_ng.bin", "wb");
        if (g) { std::fwrite(ref.data(), 4, ref.size(), g); std::fclose(g); }
        std::fprintf(stderr, "  dumped got/ref\n");
    }
    // log(clip) amplifies near-floor diffs non-linearly; use the floor-aware
    // helper: tight abs above ref_floor, loose abs in the log-amplified zone.
    // Measured on the 3-component 16k tone: tight max 0.0012, floor max 0.22.
    CHECK(pcnsf_test::rmse_floor_ok(got, ref, 0.01f, 0.02f, -5.0f, 0.30f));

    if (pcnsf_test::failures) return 1;
    std::printf("test_mel OK (%zu frames)\n", got.size() / (size_t) cfg.n_mels);
    return 0;
}
