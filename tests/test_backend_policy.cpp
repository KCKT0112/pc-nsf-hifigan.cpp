#include "../src/backend_policy.h"
#include "test_common.h"

int main() {
    using pc_nsf_hifigan::detail::direct_conv_enabled;
    for (const char * name : {"CPU", "Vulkan0", "MTL0", "Metal", "CUDA0"}) {
        CHECK(!direct_conv_enabled(GGML_TYPE_F16, name, true, true, "1"));
        CHECK(!direct_conv_enabled(GGML_TYPE_BF16, name, true, true, "1"));
        CHECK(!direct_conv_enabled(GGML_TYPE_F32, name, true, true, "0"));
    }
    CHECK(!direct_conv_enabled(GGML_TYPE_F32, "MTL0", false, false, nullptr));
    CHECK(!direct_conv_enabled(GGML_TYPE_F32, "Metal", false, false, "1"));
    CHECK(direct_conv_enabled(GGML_TYPE_F32, "MTL0", true, false, nullptr));
    CHECK(direct_conv_enabled(GGML_TYPE_F32, "MTL0", true, false, "1"));
    CHECK(direct_conv_enabled(GGML_TYPE_F32, "Vulkan0", false, false, nullptr));
    CHECK(direct_conv_enabled(GGML_TYPE_F32, "CPU", true, true, nullptr));
    CHECK(!direct_conv_enabled(GGML_TYPE_F32, "CPU", true, false, nullptr));
    CHECK(direct_conv_enabled(GGML_TYPE_F32, "CPU", true, false, "1"));
    CHECK(!direct_conv_enabled(GGML_TYPE_F32, "CUDA0", false, false, nullptr));
    return pcnsf_test::failures ? 1 : 0;
}
