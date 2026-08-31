#include "cpu_info.h"
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <windows.h>
#include <intrin.h>

namespace bench {

namespace {

std::string trim(const std::string &s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end = s.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return {};
    return s.substr(start, end - start + 1);
}

} // namespace

std::string CpuInfo::to_string() const {
    std::ostringstream oss;
    oss << "CPU: " << (brand.empty() ? vendor : brand) << "\n";
    oss << "Logical cores: " << logical_cores
        << ", Physical cores: " << physical_cores
        << ", NUMA nodes: " << numa_nodes << "\n";
    oss << "Base clock: " << std::fixed << std::setprecision(1) << max_mhz << " MHz\n";
    oss << "Instruction sets:";
    if (sse41)  oss << " SSE4.1";
    if (sse42)  oss << " SSE4.2";
    if (avx)    oss << " AVX";
    if (avx2)   oss << " AVX2";
    if (avx512f) oss << " AVX-512F";
    if (fma)    oss << " FMA";
    oss << "\n";
    return oss.str();
}

CpuInfo get_cpu_info() {
    CpuInfo info{};

    int cpu_info[4] = {0};
    auto cpuid = [&](int func) {
        __cpuid(cpu_info, func);
    };

    cpuid(0);
    char vendor[13] = {0};
    memcpy(vendor + 0, &cpu_info[1], 4);
    memcpy(vendor + 4, &cpu_info[3], 4);
    memcpy(vendor + 8, &cpu_info[2], 4);
    info.vendor = vendor;

    cpuid(0x80000000);
    const uint32_t max_ext = static_cast<uint32_t>(cpu_info[0]);

    if (max_ext >= 0x80000004) {
        char brand[49] = {0};
        for (int i = 0x80000002; i <= 0x80000004; ++i) {
            cpuid(i);
            size_t offset = (i - 0x80000002) * 16;
            memcpy(brand + offset + 0, &cpu_info[0], 4);
            memcpy(brand + offset + 4, &cpu_info[1], 4);
            memcpy(brand + offset + 8, &cpu_info[2], 4);
            memcpy(brand + offset + 12, &cpu_info[3], 4);
        }
        info.brand = trim(brand);
    }

    cpuid(1);
    info.sse41 = (cpu_info[2] & (1 << 19)) != 0;
    info.sse42 = (cpu_info[2] & (1 << 20)) != 0;
    info.avx   = (cpu_info[2] & (1 << 28)) != 0;
    info.fma   = (cpu_info[2] & (1 << 12)) != 0;

    if (max_ext >= 0x80000001) {
        cpuid(0x80000001);
        info.avx2   = (cpu_info[2] & (1 << 5)) != 0;
        info.avx512f = (cpu_info[2] & (1 << 16)) != 0;
    }

    SYSTEM_INFO si;
    GetSystemInfo(&si);
    info.logical_cores = static_cast<int>(si.dwNumberOfProcessors);

    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationAll, nullptr, &length);
    if (length > 0) {
        std::vector<char> buf(length);
        auto *lpli = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data());
        if (GetLogicalProcessorInformationEx(RelationAll, lpli, &length)) {
            DWORD offset = 0;
            do {
                switch (lpli->Relationship) {
                case RelationProcessorCore:
                    info.physical_cores++;
                    break;
                case RelationNumaNode:
                    info.numa_nodes++;
                    break;
                default:
                    break;
                }
                offset += lpli->Size;
                lpli = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                    reinterpret_cast<char*>(lpli) + offset);
            } while (offset < length);
        }
    }

    if (info.physical_cores == 0) {
        info.physical_cores = info.logical_cores;
    }

    HKEY key;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      R"(HARDWARE\DESCRIPTION\System\CentralProcessor\0)",
                      0, KEY_READ, &key) == ERROR_SUCCESS) {
        DWORD mhz = 0;
        DWORD size = sizeof(mhz);
        RegQueryValueExA(key, "~MHz", nullptr, nullptr,
                         reinterpret_cast<LPBYTE>(&mhz), &size);
        info.max_mhz = static_cast<double>(mhz);
        RegCloseKey(key);
    }

    return info;
}

} // namespace bench
