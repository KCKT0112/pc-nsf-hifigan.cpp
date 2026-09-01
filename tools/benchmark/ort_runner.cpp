// SPDX-License-Identifier: MPL-2.0
#include "ort_runner.h"
#include <stdexcept>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#endif

// Official ONNX Runtime flat header, fetched by CI into ${ONNX_RUNTIME_ROOT}/include.
// ORT_API_VERSION is provided by this header; do NOT hard-code it (the installed
// ORT release may differ from the one this file was originally written against).
#include <onnxruntime_c_api.h>

namespace bench {

namespace {

#ifdef _WIN32
std::wstring widen_path(const std::string& path) {
    if (path.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, NULL, 0);
    std::wstring ws(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &ws[0], n);
    return ws;
}
#endif

void check_ort_status(const OrtApi* api, OrtStatus* status) {
    if (!status) return;
    const char* msg = api->GetErrorMessage(status);
    throw std::runtime_error(std::string("ONNX Runtime error: ") + msg);
}

} // namespace

OrtRunner::~OrtRunner() {
    if (allocator && api) {
        for (const char* name : input_names) {
            api->AllocatorFree(allocator, const_cast<void*>(static_cast<const void*>(name)));
        }
        for (const char* name : output_names) {
            api->AllocatorFree(allocator, const_cast<void*>(static_cast<const void*>(name)));
        }
    }
    if (session && api) api->ReleaseSession(session);
    if (session_options && api) api->ReleaseSessionOptions(session_options);
    if (env && api) api->ReleaseEnv(env);
}

OrtRunner::OrtRunner(OrtRunner&& other) noexcept
    : api(other.api),
      env(other.env),
      session_options(other.session_options),
      memory_info(other.memory_info),
      session(other.session),
      allocator(other.allocator),
      input_names(std::move(other.input_names)),
      output_names(std::move(other.output_names)) {
    other.api = nullptr;
    other.env = nullptr;
    other.session_options = nullptr;
    other.memory_info = nullptr;
    other.session = nullptr;
    other.allocator = nullptr;
}

OrtRunner& OrtRunner::operator=(OrtRunner&& other) noexcept {
    if (this != &other) {
        if (allocator && api) {
            for (const char* name : input_names) {
                api->AllocatorFree(allocator, const_cast<void*>(static_cast<const void*>(name)));
            }
            for (const char* name : output_names) {
                api->AllocatorFree(allocator, const_cast<void*>(static_cast<const void*>(name)));
            }
        }
        if (session && api) api->ReleaseSession(session);
        if (session_options && api) api->ReleaseSessionOptions(session_options);
        if (env && api) api->ReleaseEnv(env);
        api = other.api;
        env = other.env;
        session_options = other.session_options;
        memory_info = other.memory_info;
        session = other.session;
        allocator = other.allocator;
        input_names = std::move(other.input_names);
        output_names = std::move(other.output_names);
        other.api = nullptr;
        other.env = nullptr;
        other.session_options = nullptr;
        other.memory_info = nullptr;
        other.session = nullptr;
        other.allocator = nullptr;
    }
    return *this;
}

OrtRunner OrtRunner::create(const std::string& model_path, int intra_op_num_threads) {
    OrtRunner runner{};
    const OrtApiBase* base = OrtGetApiBase();
    const OrtApi* api = base ? base->GetApi(ORT_API_VERSION) : nullptr;
    if (!api) {
        throw std::runtime_error("Failed to obtain ONNX Runtime C API table.");
    }
    runner.api = api;

    check_ort_status(api, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "pcnsf_bench", &runner.env));
    check_ort_status(api, api->CreateSessionOptions(&runner.session_options));
    check_ort_status(api, api->SetIntraOpNumThreads(runner.session_options, intra_op_num_threads));
    check_ort_status(api, api->SetSessionExecutionMode(runner.session_options, ORT_SEQUENTIAL));
    check_ort_status(api, api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &runner.memory_info));

#ifdef _WIN32
    std::wstring model_path_wide = widen_path(model_path);
    check_ort_status(api, api->CreateSession(runner.env, model_path_wide.c_str(),
                                              runner.session_options, &runner.session));
#else
    check_ort_status(api, api->CreateSession(runner.env, model_path.c_str(),
                                              runner.session_options, &runner.session));
#endif

    check_ort_status(api, api->GetAllocatorWithDefaultOptions(&runner.allocator));

    size_t input_count = 0;
    check_ort_status(api, api->SessionGetInputCount(runner.session, &input_count));
    runner.input_names.reserve(input_count);
    for (size_t i = 0; i < input_count; ++i) {
        char* name = nullptr;
        check_ort_status(api, api->SessionGetInputName(runner.session, static_cast<size_t>(i),
                                                 runner.allocator, &name));
        runner.input_names.push_back(name);
    }

    size_t output_count = 0;
    check_ort_status(api, api->SessionGetOutputCount(runner.session, &output_count));
    runner.output_names.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i) {
        char* name = nullptr;
        check_ort_status(api, api->SessionGetOutputName(runner.session, static_cast<size_t>(i),
                                                  runner.allocator, &name));
        runner.output_names.push_back(name);
    }

    return runner;
}

std::vector<float> OrtRunner::run(const float* mel, const float* f0, int frames) {
    std::vector<float> output;

    if (input_names.size() < 2 || output_names.empty()) {
        return output;
    }

    std::vector<int64_t> mel_shape = {1, static_cast<int64_t>(frames), static_cast<int64_t>(128)};
    std::vector<int64_t> f0_shape = {1, static_cast<int64_t>(frames)};

    OrtValue* mel_val = nullptr;
    check_ort_status(api, api->CreateTensorWithDataAsOrtValue(
        memory_info,
        const_cast<void*>(reinterpret_cast<const void*>(mel)),
        static_cast<size_t>(frames * 128 * sizeof(float)),
        mel_shape.data(), static_cast<int64_t>(mel_shape.size()),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &mel_val));

    OrtValue* f0_val = nullptr;
    check_ort_status(api, api->CreateTensorWithDataAsOrtValue(
        memory_info,
        const_cast<void*>(reinterpret_cast<const void*>(f0)),
        static_cast<size_t>(frames * sizeof(float)),
        f0_shape.data(), static_cast<int64_t>(f0_shape.size()),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &f0_val));

    OrtValue* out_val = nullptr;
    OrtValue* inputs[] = {mel_val, f0_val};
    check_ort_status(api, api->Run(session, nullptr,
                             input_names.data(), inputs, 2,
                             output_names.data(), static_cast<size_t>(output_names.size()),
                             &out_val));

    if (out_val) {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        api->GetTensorTypeAndShape(out_val, &info);
        size_t n = 1;
        api->GetDimensionsCount(info, &n);
        std::vector<int64_t> shape(n);
        api->GetDimensions(info, shape.data(), n);
        size_t total = 1;
        for (auto s : shape) total *= static_cast<size_t>(s);
        float* ptr = nullptr;
        api->GetTensorMutableData(out_val, reinterpret_cast<void**>(&ptr));
        output.assign(ptr, ptr + total);
        api->ReleaseTensorTypeAndShapeInfo(info);
        api->ReleaseValue(out_val);
    }

    api->ReleaseValue(mel_val);
    api->ReleaseValue(f0_val);
    return output;
}

OrtRunnerPtr make_ort_runner(const std::string& model_path, int intra_op_num_threads) {
    return std::make_unique<OrtRunner>(OrtRunner::create(model_path, intra_op_num_threads));
}

} // namespace bench



