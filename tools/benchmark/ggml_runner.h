// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <vector>
#include <string>
#include "pc_nsf_hifigan/hifigan.h"

namespace bench {

struct GgmlRunner {
    pc_nsf_hifigan::HifiganModel model;
    std::vector<float> mel;
    std::vector<float> f0;
    std::vector<float> output;

    explicit GgmlRunner(const std::string& gguf_path, int n_threads);
    bool valid() const;
};

using GgmlRunnerPtr = std::unique_ptr<GgmlRunner>;

GgmlRunnerPtr make_ggml_runner(const std::string& gguf_path, int n_threads);

} // namespace bench
