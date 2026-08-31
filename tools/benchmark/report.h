#pragma once

#include <string>
#include <vector>
#include "accuracy.h"
#include "cpu_info.h"

namespace bench {

struct BenchmarkResult {
    std::string backend;
    double median_ms = 0.0;
    double mean_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double stddev_ms = 0.0;
    double throughput = 0.0;
    bool ok = false;
    std::string error;
};

struct LengthResult {
    int frames = 0;
    std::vector<float> ggml_wav;
    std::vector<float> ort_wav;
    BenchmarkResult ggml;
    BenchmarkResult ort;
    AccuracyReport accuracy;
};

struct Report {
    CpuInfo cpu;
    std::string gguf_path;
    std::string onnx_path;
    int ggml_threads = 0;
    int ort_threads = 0;
    std::vector<LengthResult> lengths;
    std::string extra;
};

std::string render_report(const Report& r);

} // namespace bench
