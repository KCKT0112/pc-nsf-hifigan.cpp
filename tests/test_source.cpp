// test_source: mininsf_fastsinegen_f32 (exact sin path) vs numpy golden.
#include "test_common.h"

#include <mininsf/mininsf.h>

#include <vector>

using namespace pcnsf_test;

int main() {
    const float f0[4] = {196.0f, 261.6f, 329.6f, 392.0f};
    MiniNsfConfig cfg;
    cfg.source_sample_rate = 5512.5f;
    cfg.upsample = 64;
    std::vector<float> src(4 * 64);
    CHECK(mininsf_fastsinegen_f32(f0, 1, 4, &cfg, src.data()) == MININSF_OK);

    std::vector<float> ref;
    CHECK(read_raw(pcnsf_test::golden_path("source_64.bin"), ref));
    if (ref.empty() && pcnsf_test::failures == 0) {
        std::fprintf(stderr, "  golden file missing — did gen_golden run?\n");
        return 1;
    }
    if (src.size() != ref.size()) {
        std::fprintf(stderr, "  size mismatch: got %zu ref %zu\n", src.size(), ref.size());
        return 1;
    }
    CHECK(pcnsf_test::rmse_ok(src, ref, 1e-3f, 5e-3f));

    if (pcnsf_test::failures) return 1;
    std::printf("test_source OK (%zu samples)\n", src.size());
    return 0;
}
