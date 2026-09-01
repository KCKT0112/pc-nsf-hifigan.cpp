// SPDX-License-Identifier: MPL-2.0
#pragma once

// GGUF model loading (weights in meta context) + backend setup.
// Wraps ggml v0.19.0: gguf_init_from_file -> tensors in a ggml context,
// compute backend (CPU by default; GPU backends upload weights via
// ggml_backend_tensor_alloc/set so device shaders never read CPU memory).
// CPU backends additionally get a persistent threadpool (created once at
// load, reused by every graph_compute) so per-call thread spawn cost is
// paid once instead of on every run.

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <gguf.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace pc_nsf_hifigan {

struct GGUFModel {
    gguf_context * gguf   = nullptr;
    ggml_context * meta   = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_threadpool_t tpool = nullptr;  // CPU backends only; borrowed by backend
    std::unordered_map<std::string, ggml_tensor *> tensors;

    ~GGUFModel() {
        if (buf)    ggml_backend_buffer_free(buf);
        if (tpool)  ggml_threadpool_free(tpool);  // before backend: pool threads must not outlive backend use
        if (backend) ggml_backend_free(backend);
        if (gguf)   gguf_free(gguf);
        if (meta)   ggml_free(meta);
    }
    GGUFModel() = default;
    GGUFModel(const GGUFModel &) = delete;
    GGUFModel & operator=(const GGUFModel &) = delete;
};

// Load a .gguf file into a GGUFModel (tensors stay in the meta context,
// already in a backend buffer so they are directly usable as graph leaves).
std::unique_ptr<GGUFModel> gguf_load(const std::string & path, int n_threads = 4);

// Convenience: fetch a tensor by name or die.
ggml_tensor * gguf_get(const GGUFModel & m, const std::string & name);

// Read an int/float/string metadata value (returns false if absent/wrong type).
bool gguf_meta_int(const GGUFModel & m, const std::string & key, int64_t & out);
bool gguf_meta_float(const GGUFModel & m, const std::string & key, float & out);
bool gguf_meta_bool(const GGUFModel & m, const std::string & key, bool & out);
bool gguf_meta_str(const GGUFModel & m, const std::string & key, std::string & out);
bool gguf_meta_floats(const GGUFModel & m, const std::string & key, std::vector<float> & out);

}  // namespace pc_nsf_hifigan
