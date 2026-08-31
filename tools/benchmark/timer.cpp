#include "timer.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <windows.h>

namespace bench {

void Timer::start() {
    QueryPerformanceFrequency(&freq_);
    QueryPerformanceCounter(&start_);
    running_ = true;
}

void Timer::stop() {
    QueryPerformanceCounter(&stop_);
    running_ = false;
}

double Timer::elapsed_ms() const {
    LARGE_INTEGER end = running_ ? LARGE_INTEGER{} : stop_;
    if (!running_) {
        return static_cast<double>(end.QuadPart - start_.QuadPart)
               / freq_.QuadPart * 1000.0;
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return static_cast<double>(now.QuadPart - start_.QuadPart)
           / freq_.QuadPart * 1000.0;
}

RunStats measure_latency_ms(void (*fn)(void*), void* arg,
                            size_t warmup, size_t runs) {
    RunStats stats{};
    if (runs == 0) return stats;

    std::vector<double> samples;
    samples.reserve(runs);

    Timer timer;
    for (size_t i = 0; i < warmup + runs; ++i) {
        timer.start();
        fn(arg);
        timer.stop();
        double ms = timer.elapsed_ms();
        if (i >= warmup) samples.push_back(ms);
    }

    if (samples.empty()) return stats;

    std::sort(samples.begin(), samples.end());
    stats.min_ms = samples.front();
    stats.max_ms = samples.back();

    if (samples.size() % 2 == 1) {
        stats.median_ms = samples[samples.size() / 2];
    } else {
        double a = samples[samples.size() / 2 - 1];
        double b = samples[samples.size() / 2];
        stats.median_ms = (a + b) / 2.0;
    }

    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    stats.mean_ms = sum / samples.size();

    double sq_sum = 0.0;
    for (double v : samples) {
        double d = v - stats.mean_ms;
        sq_sum += d * d;
    }
    stats.stddev_ms = std::sqrt(sq_sum / samples.size());

    return stats;
}

} // namespace bench
