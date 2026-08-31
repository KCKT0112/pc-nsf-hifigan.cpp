#include "report.h"
#include "cpu_info.h"
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace bench {

namespace {

std::string now_string() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_MSC_VER)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

std::string backend_label(const std::string& backend) {
    if (backend == "ggml") return "ggml (GGUF)";
    if (backend == "ort") return "ORT CPU (ONNX)";
    return backend;
}

} // namespace

std::string render_report(const Report& r) {
    std::ostringstream out;
    out.setf(std::ios::fixed);

    out << "# pc-nsf-hifigan CPU Benchmark Report\n\n";
    out << "Generated: " << now_string() << "\n\n";

    out << "## Machine\n\n";
    out << r.cpu.to_string() << "\n";

    out << "## Configuration\n\n";
    out << "GGUF: " << (r.gguf_path.empty() ? "(none)" : r.gguf_path) << "\n";
    out << "ONNX: " << (r.onnx_path.empty() ? "(none)" : r.onnx_path) << "\n";
    out << "ggml threads: " << r.ggml_threads << "\n";
    out << "ort threads: " << r.ort_threads << "\n\n";

    out << "## Results by sequence length\n\n";

    for (const auto& len : r.lengths) {
        out << "### Frames: " << len.frames << "\n\n";
        out << std::left << std::setw(20) << "Backend"
            << std::right << std::setw(12) << "Median(ms)"
            << std::setw(12) << "Mean(ms)"
            << std::setw(12) << "Min(ms)"
            << std::setw(12) << "Max(ms)"
            << std::setw(14) << "Throughput" << "\n";
        out << std::string(92, '-') << "\n";

        auto print_row = [&](const std::string& label, const BenchmarkResult& b) {
            out << std::left << std::setw(20) << label
                << std::right << std::setprecision(3) << std::setw(12) << b.median_ms
                << std::setw(12) << b.mean_ms
                << std::setw(12) << b.min_ms
                << std::setw(12) << b.max_ms;
            if (b.throughput > 0.0) {
                out << std::setw(14) << static_cast<int>(b.throughput);
            } else {
                out << std::setw(14) << "-";
            }
            out << "\n";
            if (!b.error.empty()) {
                out << "  error: " << b.error << "\n";
            }
        };

        print_row(backend_label(len.ggml.backend), len.ggml);
        print_row(backend_label(len.ort.backend), len.ort);
        out << "\n";

        if (len.accuracy.length_a > 0 || len.accuracy.length_b > 0) {
            out << "**Accuracy (ggml vs ORT)**\n\n";
            out << accuracy_to_string(len.accuracy) << "\n";
        }
    }

    if (!r.extra.empty()) {
        out << "## Notes\n\n";
        out << r.extra << "\n";
    }

    return out.str();
}

} // namespace bench
