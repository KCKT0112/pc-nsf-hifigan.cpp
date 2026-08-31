#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "pc_nsf_hifigan/hifigan.h"

#include "cpu_info.h"
#include "timer.h"
#include "ggml_runner.h"
#include "ort_runner.h"
#include "accuracy.h"
#include "report.h"

namespace {

struct Args {
    std::string gguf = "hifigan_f32.gguf";
    std::string onnx = "nsf_hifigan.onnx";
    int threads = 0;
    int warmup = 3;
    int runs = 10;
    std::string out = "benchmark_report.md";
    std::vector<int> lengths = {256, 512, 1024, 1722, 2048, 4096};
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << "\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--gguf") a.gguf = next();
        else if (arg == "--onnx") a.onnx = next();
        else if (arg == "--threads") a.threads = std::atoi(next().c_str());
        else if (arg == "--warmup") a.warmup = std::atoi(next().c_str());
        else if (arg == "--runs") a.runs = std::atoi(next().c_str());
        else if (arg == "--out") a.out = next();
        else if (arg == "-h" || arg == "--help") {
            std::cout <<
R"(pc-nsf-hifigan CPU benchmark
Usage: benchmark [options]
  --gguf PATH   GGUF model path (default: hifigan_f32.gguf)
  --onnx PATH   ONNX model path (default: nsf_hifigan.onnx)
  --threads N   Worker threads (0 = auto = logical cores)
  --warmup N    Warmup runs per case (default: 3)
  --runs N      Timed runs per case (default: 10)
  --out PATH    Report file (default: benchmark_report.md)
)";
            std::exit(0);
        }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::exit(1);
        }
    }
    return a;
}

void generate_inputs(size_t frames, int num_mels,
                     std::vector<float>& mel,
                     std::vector<float>& f0) {
    static thread_local std::mt19937_64 rng((std::random_device{}()));
    std::uniform_real_distribution<float> mel_dist(-11.0f, 0.0f);
    std::uniform_real_distribution<float> f0_dist(50.0f, 500.0f);

    mel.resize(static_cast<size_t>(frames) * num_mels);
    f0.resize(frames);

    for (size_t i = 0; i < mel.size(); ++i) mel[i] = mel_dist(rng);
    for (size_t i = 0; i < frames; ++i) f0[i] = f0_dist(rng);
}

struct BenchContext {
    BenchContext(pc_nsf_hifigan::HifiganModel* m, std::vector<float>* o,
                 const std::vector<float>& mel, const std::vector<float>& f0, int t)
        : model(m), output(o), mel(mel), f0(f0), frames(t) {}
    pc_nsf_hifigan::HifiganModel* model;
    std::vector<float>* output;
    const std::vector<float>& mel;
    const std::vector<float>& f0;
    int frames;
};

struct OrtBenchContext {
    OrtBenchContext(bench::OrtRunner* r, std::vector<float>* o,
                    const std::vector<float>& mel, const std::vector<float>& f0, int t)
        : runner(r), output(o), mel(mel), f0(f0), frames(t) {}
    bench::OrtRunner* runner;
    std::vector<float>* output;
    const std::vector<float>& mel;
    const std::vector<float>& f0;
    int frames;
};

} // namespace

#ifdef _WIN32
#include <windows.h>
LONG WINAPI seh_filter(EXCEPTION_POINTERS* ep) {
    std::cerr << "SEH exception: 0x" << std::hex << ep->ExceptionRecord->ExceptionCode << std::dec << "\n";
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    SetUnhandledExceptionFilter(seh_filter);
#endif
    Args args = parse_args(argc, argv);
    if (args.threads <= 0) {
#ifdef _WIN32
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        args.threads = static_cast<int>(si.dwNumberOfProcessors);
#else
        args.threads = static_cast<int>(std::thread::hardware_concurrency());
#endif
    }

    std::cout << "pc-nsf-hifigan CPU benchmark\n";
    std::cout << "GGUF: " << args.gguf << "\n";
    std::cout << "ONNX: " << args.onnx << "\n";
    std::cout << "Threads: " << args.threads << "\n";
    std::cout << "Warmup: " << args.warmup << "\n";
    std::cout << "Runs: " << args.runs << "\n\n";

    bench::CpuInfo cpu = bench::get_cpu_info();
    std::cout << cpu.to_string() << "\n";

    bench::Report report{};
    report.cpu = cpu;
    report.gguf_path = args.gguf;
    report.onnx_path = args.onnx;
    report.ggml_threads = args.threads;
    report.ort_threads = args.threads;

    bench::GgmlRunnerPtr ggml_runner;
    try {
        ggml_runner = bench::make_ggml_runner(args.gguf, args.threads);
        if (!ggml_runner->valid()) {
            throw std::runtime_error("GGUF model loaded but appears empty.");
        }
    } catch (const std::exception& e) {
        std::cerr << "GGML init failed: " << e.what() << "\n";
    }

    bench::OrtRunnerPtr ort_runner;
    try {
        ort_runner = bench::make_ort_runner(args.onnx, args.threads);
    } catch (const std::exception& e) {
        std::cerr << "ORT init failed: " << e.what() << "\n";
    }

    if (!ggml_runner && !ort_runner) {
        std::cerr << "Both backends failed to initialize. Exiting.\n";
        std::ofstream f(args.out);
        f << bench::render_report(report);
        return 1;
    }

    int num_mels = ggml_runner ? ggml_runner->model.num_mels : 128;

    for (int frames : args.lengths) {
        std::cout << "=== Frames: " << frames << " ===\n";
        std::cout.flush();

        std::vector<float> mel;
        std::vector<float> f0;
        generate_inputs(frames, num_mels, mel, f0);

        bench::LengthResult lr{};
        lr.frames = frames;

        if (ggml_runner) {
            ggml_runner->mel = mel;
            ggml_runner->f0 = f0;
            BenchContext ctx(&ggml_runner->model, &lr.ggml_wav, ggml_runner->mel, ggml_runner->f0, frames);

            auto fn = [](void* p) {
                auto* c = static_cast<BenchContext*>(p);
                c->output->clear();
                pc_nsf_hifigan::hifigan_run(*c->model, c->mel.data(), c->f0.data(), c->frames, *c->output);
            };

            bench::RunStats stats = bench::measure_latency_ms(fn, &ctx, args.warmup, args.runs);
            lr.ggml.backend = "ggml";
            lr.ggml.median_ms = stats.median_ms;
            lr.ggml.mean_ms = stats.mean_ms;
            lr.ggml.min_ms = stats.min_ms;
            lr.ggml.max_ms = stats.max_ms;
            lr.ggml.stddev_ms = stats.stddev_ms;
            if (!lr.ggml_wav.empty()) {
                lr.ggml.throughput = (lr.ggml_wav.size() / stats.median_ms) * 1000.0;
            }
            std::cout << "  ggml: " << stats.median_ms << " ms\n";
            std::cout.flush();
        }

        if (ort_runner) {
            OrtBenchContext ctx(ort_runner.get(), &lr.ort_wav, mel, f0, frames);

            auto fn = [](void* p) {
                auto* c = static_cast<OrtBenchContext*>(p);
                c->output->clear();
                *c->output = c->runner->run(c->mel.data(), c->f0.data(), c->frames);
            };

            bench::RunStats stats = bench::measure_latency_ms(fn, &ctx, args.warmup, args.runs);
            lr.ort.backend = "ort";
            lr.ort.median_ms = stats.median_ms;
            lr.ort.mean_ms = stats.mean_ms;
            lr.ort.min_ms = stats.min_ms;
            lr.ort.max_ms = stats.max_ms;
            lr.ort.stddev_ms = stats.stddev_ms;
            if (!lr.ort_wav.empty()) {
                lr.ort.throughput = (lr.ort_wav.size() / stats.median_ms) * 1000.0;
            }
            std::cout << "  ort:  " << stats.median_ms << " ms\n";
            std::cout.flush();
        }

        if (!lr.ggml_wav.empty() && !lr.ort_wav.empty()) {
            lr.accuracy = bench::compare_audio(lr.ggml_wav, lr.ort_wav, "ggml", "ort");
        }

        report.lengths.push_back(std::move(lr));
    }

    std::string report_text = bench::render_report(report);
    std::cout << "\n" << report_text << "\n";

    std::ofstream f(args.out);
    if (!f) {
        std::cerr << "Failed to write report to " << args.out << "\n";
        return 1;
    }
    f << report_text;
    std::cout << "Report written to " << args.out << "\n";
    return 0;
}
