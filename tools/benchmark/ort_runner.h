// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

struct OrtApi;
struct OrtEnv;
struct OrtSession;
struct OrtSessionOptions;
struct OrtMemoryInfo;
struct OrtAllocator;
struct OrtValue;
struct OrtTensorTypeAndShapeInfo;

namespace bench {

struct OrtRunner {
    const OrtApi* api = nullptr;
    OrtEnv* env = nullptr;
    OrtSessionOptions* session_options = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* memory_info = nullptr;
    OrtAllocator* allocator = nullptr;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    OrtRunner() = default;
    ~OrtRunner();
    OrtRunner(const OrtRunner&) = delete;
    OrtRunner& operator=(const OrtRunner&) = delete;
    OrtRunner(OrtRunner&& other) noexcept;
    OrtRunner& operator=(OrtRunner&& other) noexcept;
    static OrtRunner create(const std::string& model_path, int intra_op_num_threads);
    std::vector<float> run(const float* mel, const float* f0, int frames);
};

using OrtRunnerPtr = std::unique_ptr<OrtRunner>;

OrtRunnerPtr make_ort_runner(const std::string& model_path, int intra_op_num_threads);

} // namespace bench
