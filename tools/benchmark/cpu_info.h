#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace bench {

struct CpuInfo {
    std::string vendor;
    std::string brand;
    int logical_cores = 0;
    int physical_cores = 0;
    int numa_nodes = 0;
    double max_mhz = 0.0;
    double current_mhz = 0.0;
    bool sse41 = false;
    bool sse42 = false;
    bool avx = false;
    bool avx2 = false;
    bool avx512f = false;
    bool fma = false;

    std::string to_string() const;
};

CpuInfo get_cpu_info();

} // namespace bench
