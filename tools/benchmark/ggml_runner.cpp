// SPDX-License-Identifier: MPL-2.0
#include "ggml_runner.h"
#include <stdexcept>

namespace bench {

GgmlRunner::GgmlRunner(const std::string& gguf_path, int n_threads)
    : model(gguf_path, n_threads) {}

bool GgmlRunner::valid() const {
    return !model.gguf->tensors.empty();
}

GgmlRunnerPtr make_ggml_runner(const std::string& gguf_path, int n_threads) {
    return std::make_unique<GgmlRunner>(gguf_path, n_threads);
}

} // namespace bench
