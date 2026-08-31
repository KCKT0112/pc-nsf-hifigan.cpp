#pragma once

#include <cstddef>
#include <windows.h>

namespace bench {

class Timer {
public:
    void start();
    void stop();
    double elapsed_ms() const;

private:
    LARGE_INTEGER freq_{};
    LARGE_INTEGER start_{};
    LARGE_INTEGER stop_{};
    bool running_ = false;
};

struct RunStats {
    double median_ms = 0.0;
    double mean_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double stddev_ms = 0.0;
    double samples_per_sec = 0.0;
};

RunStats measure_latency_ms(void (*fn)(void*), void* arg,
                            size_t warmup, size_t runs);

} // namespace bench
