// SPDX-License-Identifier: MPL-2.0
// test_gguf_meta: GGUF load + metadata accessors on the tiny synthetic GGUF.
#include "test_common.h"
#include "pc_nsf_hifigan/gguf_model.h"

#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    const std::string path = pcnsf_test::golden_path("tiny.gguf");
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { std::printf("tiny.gguf absent — gguf pkg missing in gen_golden, skip\n"); return 0; }
    std::fclose(f);

    try {
        auto m = pc_nsf_hifigan::gguf_load(path, 2);

        int64_t sr = 0, hop = 0;
        float nf = 0;
        bool mini = false;
        std::string type;
        std::vector<float> rates;
        CHECK(pc_nsf_hifigan::gguf_meta_int(*m, "audio.sample_rate", sr) && sr == 44100);
        CHECK(pc_nsf_hifigan::gguf_meta_int(*m, "audio.hop", hop) && hop == 512);
        CHECK(pc_nsf_hifigan::gguf_meta_float(*m, "audio.n_mels_f", nf) && nf == 128.5f);
        CHECK(pc_nsf_hifigan::gguf_meta_bool(*m, "hifigan.mini_nsf", mini) && mini);
        CHECK(pc_nsf_hifigan::gguf_meta_str(*m, "audio.type", type) && type == "hifigan");
        CHECK(pc_nsf_hifigan::gguf_meta_floats(*m, "hifigan.upsample_rates", rates) &&
              rates.size() == 5 && rates[0] == 8.0f && rates[4] == 2.0f);

        ggml_tensor * w = pc_nsf_hifigan::gguf_get(*m, "hifigan.tiny_w");
        // gguf-py writes numpy (3,2) with ggml's reversed shape -> ne = (2,3)
        CHECK(w != nullptr && w->ne[0] == 2 && w->ne[1] == 3);
        // tensor is F16 (12 bytes); read exactly ggml_nbytes and decode
        std::vector<uint8_t> raw(ggml_nbytes(w));
        ggml_backend_tensor_get(w, raw.data(), 0, ggml_nbytes(w));
        bool all_ok = true;
        for (int i = 0; i < 6; ++i) {
            ggml_fp16_t h;
            std::memcpy(&h, raw.data() + 2 * i, 2);
            float v = ggml_fp16_to_fp32(h);
            all_ok = all_ok && std::fabs(v - 0.25f) < 1e-2f;
        }
        CHECK(all_ok);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "unexpected exception: %s\n", e.what());
        return 1;
    }

    if (pcnsf_test::failures) return 1;
    std::printf("test_gguf_meta OK\n");
    return 0;
}
