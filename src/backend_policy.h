#pragma once

#include <ggml.h>
#include <cstring>

namespace pc_nsf_hifigan {
namespace detail {

// Precision and Metal capability are requirements, even for a forced opt-in.
inline bool direct_conv_enabled(ggml_type type, const char * name,
                                bool supports_op, bool cpu_avx2,
                                const char * override_value) {
    if (type != GGML_TYPE_F32 || !name || !supports_op) return false;
    const bool metal = std::strstr(name, "Metal") || std::strncmp(name, "MTL", 3) == 0;
    if (override_value) return override_value[0] != '0';
    return metal || std::strstr(name, "Vulkan") ||
           (std::strcmp(name, "CPU") == 0 && cpu_avx2);
}

} // namespace detail
} // namespace pc_nsf_hifigan
