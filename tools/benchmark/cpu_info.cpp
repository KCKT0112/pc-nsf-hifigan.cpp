// SPDX-License-Identifier: MPL-2.0
#include "cpu_info.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#  include <windows.h>
#  if defined(_MSC_VER)
#    include <intrin.h>
#  endif
#elif defined(__APPLE__)
#  include <sys/sysctl.h>
#  include <sys/types.h>
#else
#  include <array>
#endif

#if (defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)) && \
    !defined(__APPLE__) && !defined(_WIN32)
// Linux x86: use gcc/clang cpuid helpers
#  include <cpuid.h>
#endif

namespace bench {

namespace {

std::string trim(const std::string &s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end = s.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return {};
    return s.substr(start, end - start + 1);
}

// Read a single "key : value" from /proc/cpuinfo (first match wins).
std::string proc_cpuinfo_value(const std::string &key) {
    std::ifstream f("/proc/cpuinfo");
    if (!f) return {};
    std::string line;
    while (std::getline(f, line)) {
        auto pos = line.find(':');
        if (pos == std::string::npos) continue;
        std::string k = trim(line.substr(0, pos));
        if (k == key) return trim(line.substr(pos + 1));
    }
    return {};
}

} // namespace

std::string CpuInfo::to_string() const {
    std::ostringstream oss;
    oss << "CPU: " << (brand.empty() ? vendor : brand) << "\n";
    oss << "Logical cores: " << logical_cores
        << ", Physical cores: " << physical_cores
        << ", NUMA nodes: " << numa_nodes << "\n";
    if (max_mhz > 0.0) {
        oss << "Base clock: " << std::fixed << std::setprecision(1) << max_mhz << " MHz\n";
    }
    oss << "Instruction sets:";
    if (sse41)  oss << " SSE4.1";
    if (sse42)  oss << " SSE4.2";
    if (avx)    oss << " AVX";
    if (avx2)   oss << " AVX2";
    if (avx512f) oss << " AVX-512F";
    if (fma)    oss << " FMA";
    if (neon)   oss << " NEON";
    oss << "\n";
    return oss.str();
}

CpuInfo get_cpu_info() {
    CpuInfo info{};

#if defined(__aarch64__) || defined(_M_ARM64)
    // ARM64 (Apple Silicon / Linux aarch64) — no cpuid; NEON is mandatory.
    info.neon = true;

#  if defined(__APPLE__)
    {
        char buf[256] = {0};
        size_t len = sizeof(buf);
        if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0) {
            info.brand = trim(buf);
        }
        len = sizeof(buf);
        if (sysctlbyname("machdep.cpu.vendor", buf, &len, nullptr, 0) == 0) {
            info.vendor = trim(buf);
        }
        int64_t n = 0;
        len = sizeof(n);
        if (sysctlbyname("hw.physicalcpu", &n, &len, nullptr, 0) == 0) info.physical_cores = static_cast<int>(n);
        len = sizeof(n);
        if (sysctlbyname("hw.logicalcpu", &n, &len, nullptr, 0) == 0) info.logical_cores = static_cast<int>(n);
        uint64_t hz = 0;
        len = sizeof(hz);
        if (sysctlbyname("hw.cpufrequency", &hz, &len, nullptr, 0) == 0) info.max_mhz = hz / 1e6;
    }
#  else
    // Linux aarch64
    {
        std::string b = proc_cpuinfo_value("model name");
        if (b.empty()) b = proc_cpuinfo_value("Processor");
        info.brand = b;
        info.physical_cores = 0;
        info.logical_cores = 0;
        std::ifstream f("/proc/cpuinfo");
        if (f) {
            std::string line;
            while (std::getline(f, line)) {
                auto pos = line.find(':');
                if (pos == std::string::npos) continue;
                std::string k = trim(line.substr(0, pos));
                std::string v = trim(line.substr(pos + 1));
                if (k == "processor") info.logical_cores++;
                else if (k == "core id") info.physical_cores++;
            }
        }
        if (info.physical_cores == 0) info.physical_cores = info.logical_cores;
    }
#  endif

#else
    // x86 / x86_64
    unsigned int cpu_info[4] = {0};

#  if defined(_MSC_VER) || defined(__INTEL_COMPILER)
    auto cpuidex = [&](unsigned int func, unsigned int sub) { __cpuidex(reinterpret_cast<int*>(cpu_info), static_cast<int>(func), static_cast<int>(sub)); };
    auto cpuid0  = [&](unsigned int func)                 { __cpuid(reinterpret_cast<int*>(cpu_info), static_cast<int>(func)); };
#  elif defined(__APPLE__) || defined(_WIN32)
    // clang/gcc on these platforms with cpuid.h available:
    auto cpuidex = [&](unsigned int func, unsigned int sub) {
        if (__get_cpuid_count(func, sub, &cpu_info[0], &cpu_info[1], &cpu_info[2], &cpu_info[3]) == 0) {
            cpu_info[0] = cpu_info[1] = cpu_info[2] = cpu_info[3] = 0;
        }
    };
    auto cpuid0 = [&](unsigned int func) {
        if (__get_cpuid(func, &cpu_info[0], &cpu_info[1], &cpu_info[2], &cpu_info[3]) == 0) {
            cpu_info[0] = cpu_info[1] = cpu_info[2] = cpu_info[3] = 0;
        }
    };
#  else
    // Linux / generic gcc-clang: use <cpuid.h>
    auto cpuidex = [&](unsigned int func, unsigned int sub) {
        if (__get_cpuid_count(func, sub, &cpu_info[0], &cpu_info[1], &cpu_info[2], &cpu_info[3]) == 0) {
            cpu_info[0] = cpu_info[1] = cpu_info[2] = cpu_info[3] = 0;
        }
    };
    auto cpuid0 = [&](unsigned int func) {
        if (__get_cpuid(func, &cpu_info[0], &cpu_info[1], &cpu_info[2], &cpu_info[3]) == 0) {
            cpu_info[0] = cpu_info[1] = cpu_info[2] = cpu_info[3] = 0;
        }
    };
#  endif

    cpuid0(0);
    {
        char vendor[13] = {0};
        std::memcpy(vendor + 0, &cpu_info[1], 4);
        std::memcpy(vendor + 4, &cpu_info[3], 4);
        std::memcpy(vendor + 8, &cpu_info[2], 4);
        info.vendor = vendor;
    }

    unsigned int max_ext = 0;
    cpuid0(0x80000000);
    max_ext = cpu_info[0];

    if (max_ext >= 0x80000004) {
        char brand[49] = {0};
        for (unsigned int i = 0x80000002; i <= 0x80000004; ++i) {
            unsigned int t[4] = {0};
            cpuid0(i);
            std::memcpy(t, cpu_info, sizeof(t));
            size_t offset = (i - 0x80000002) * 16;
            std::memcpy(brand + offset + 0, &t[0], 4);
            std::memcpy(brand + offset + 4, &t[1], 4);
            std::memcpy(brand + offset + 8, &t[2], 4);
            std::memcpy(brand + offset + 12, &t[3], 4);
        }
        info.brand = trim(brand);
    }

    cpuid0(1);
    info.sse41 = (cpu_info[2] & (1u << 19)) != 0;
    info.sse42 = (cpu_info[2] & (1u << 20)) != 0;
    info.avx   = (cpu_info[2] & (1u << 28)) != 0;
    info.fma   = (cpu_info[2] & (1u << 12)) != 0;  // bit 12 of ECX = FMA

    // AVX2 / AVX-512F live in leaf 7, sub-leaf 0 (EBX)
    {
        unsigned int l7[4] = {0};
        cpuidex(7, 0);
        std::memcpy(l7, cpu_info, sizeof(l7));
        info.avx2    = (l7[1] & (1u << 5)) != 0;
        info.avx512f = (l7[1] & (1u << 16)) != 0;
    }

#  if defined(_WIN32)
    {
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
        if (info.physical_cores == 0) info.physical_cores = info.logical_cores;

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
    }
#  elif defined(__APPLE__)
    {
        int64_t n = 0;
        size_t len = sizeof(n);
        if (sysctlbyname("hw.physicalcpu", &n, &len, nullptr, 0) == 0) info.physical_cores = static_cast<int>(n);
        len = sizeof(n);
        if (sysctlbyname("hw.logicalcpu", &n, &len, nullptr, 0) == 0) info.logical_cores = static_cast<int>(n);
        if (info.physical_cores == 0) info.physical_cores = info.logical_cores;
        uint64_t hz = 0;
        len = sizeof(hz);
        if (sysctlbyname("hw.cpufrequency", &hz, &len, nullptr, 0) == 0) info.max_mhz = hz / 1e6;
    }
#  else
    {
        // Linux: physical cores from unique "core id" + "physical id" combos.
        info.logical_cores = 0;
        info.physical_cores = 0;
        std::ifstream f("/proc/cpuinfo");
        std::string line;
        int cur_phys = 0, cur_core = 0;
        bool have_phys = false, have_core = false;
        std::vector<std::pair<int,int>> sockets;
        while (std::getline(f, line)) {
            auto pos = line.find(':');
            if (pos == std::string::npos) continue;
            std::string k = trim(line.substr(0, pos));
            std::string v = trim(line.substr(pos + 1));
            if (k == "processor") {
                info.logical_cores++;
                if (have_phys && have_core) {
                    sockets.emplace_back(cur_phys, cur_core);
                    have_phys = have_core = false;
                }
            } else if (k == "physical id") {
                cur_phys = std::stoi(v); have_phys = true;
            } else if (k == "core id") {
                cur_core = std::stoi(v); have_core = true;
            }
        }
        if (have_phys && have_core) sockets.emplace_back(cur_phys, cur_core);
        if (!sockets.empty()) {
            std::sort(sockets.begin(), sockets.end());
            sockets.erase(std::unique(sockets.begin(), sockets.end()), sockets.end());
            info.physical_cores = static_cast<int>(sockets.size());
        }
        if (info.physical_cores == 0) info.physical_cores = info.logical_cores;

        std::string mhz = proc_cpuinfo_value("cpu MHz");
        if (!mhz.empty()) {
            try { info.max_mhz = std::stod(mhz); } catch (...) {}
        }
    }
#  endif
#endif

    if (info.logical_cores == 0) {
        info.logical_cores = static_cast<int>(std::thread::hardware_concurrency());
    }
    if (info.physical_cores == 0) {
        info.physical_cores = info.logical_cores;
    }
    return info;
}

} // namespace bench
