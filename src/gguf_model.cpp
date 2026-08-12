#include "pc_nsf_hifigan/gguf_model.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#    include <windows.h>
#endif

namespace pc_nsf_hifigan {

std::unique_ptr<GGUFModel> gguf_load(const std::string & path, int n_threads) {
    auto m = std::make_unique<GGUFModel>();

    struct gguf_init_params params = {
        /*.no_alloc   =*/ false,  // gguf_init allocates tensor data in the meta ctx
        /*.ctx        =*/ &m->meta,
    };
    m->gguf = gguf_init_from_file(path.c_str(), params);
    if (!m->gguf) {
        throw std::runtime_error("gguf_init_from_file failed: " + path);
    }

    m->backend = ggml_backend_init_best();
    if (!m->backend) {
        throw std::runtime_error("ggml_backend_init_best failed");
    }
    // PCNSF_BACKEND=cpu forces the CPU backend (Vulkan/others may use fp16 math).
    if (const char * be = std::getenv("PCNSF_BACKEND"); be && std::string(be) == "cpu") {
        ggml_backend_free(m->backend);
        m->backend = ggml_backend_init_by_name("CPU", nullptr);
        if (!m->backend) {
            throw std::runtime_error("PCNSF_BACKEND=cpu but CPU backend init failed");
        }
    }
    std::fprintf(stderr, "ggml backend: %s\n", ggml_backend_name(m->backend));
    if (ggml_backend_is_cpu(m->backend)) {
        ggml_backend_cpu_set_n_threads(m->backend, n_threads);
    }

    const int n = gguf_get_n_tensors(m->gguf);
    for (int i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(m->gguf, i);
        ggml_tensor * t = ggml_get_tensor(m->meta, name);
        if (t) m->tensors[name] = t;
    }

    // GPU backends (Vulkan/CUDA/Metal): weights live in the meta context (CPU
    // memory) after gguf_init, but a bare ggml_backend_graph_compute will not
    // upload them -- the device shader would read CPU memory and crash.  Copy
    // every weight tensor into one device buffer so they are directly usable
    // as graph leaves.  After this, tensor->data is a device-side pointer
    // (Vulkan: fake base 0x1000 + offset) and MUST only be accessed through
    // ggml_backend_tensor_get/set.
    if (!ggml_backend_is_cpu(m->backend)) {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(m->backend);
        const size_t align = ggml_backend_buft_get_alignment(buft);
        size_t total = 0;
        for (auto & kv : m->tensors) {
            total += ggml_backend_buft_get_alloc_size(buft, kv.second);
            total = (total + align - 1) & ~(align - 1);  // vk alloc_size has no alignment
        }
        m->buf = ggml_backend_alloc_buffer(m->backend, total);
        if (!m->buf) {
            throw std::runtime_error("ggml_backend_alloc_buffer failed (device weights)");
        }
        char * base = (char *) ggml_backend_buffer_get_base(m->buf);
        char * off  = base;
        for (auto & kv : m->tensors) {
            ggml_tensor * t = kv.second;
            const size_t nb = ggml_nbytes(t);
            std::vector<uint8_t> tmp(nb);
            memcpy(tmp.data(), t->data, nb);  // capture CPU copy before re-pointing
            t->data   = nullptr;  // ggml_backend_tensor_alloc asserts data == NULL
            t->buffer = nullptr;  // gguf_init tensors have no backend buffer set
            if (ggml_backend_tensor_alloc(m->buf, t, off) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("ggml_backend_tensor_alloc failed for " + kv.first);
            }
            ggml_backend_tensor_set(t, tmp.data(), 0, nb);  // upload to device
            off += ggml_backend_buft_get_alloc_size(buft, t);
            off = (char *) (((uintptr_t) off + align - 1) & ~(uintptr_t)(align - 1));
        }
        std::fprintf(stderr, "uploaded %u tensors (%.1f MB) to device buffer\n",
                     (unsigned) m->tensors.size(), total / 1e6);
    }
    return m;
}

ggml_tensor * gguf_get(const GGUFModel & m, const std::string & name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) {
        throw std::runtime_error("tensor not found: " + name);
    }
    return it->second;
}

namespace {
template <typename F> bool gguf_meta(const GGUFModel & m, const std::string & key, F fn) {
    const int kid = gguf_find_key(m.gguf, key.c_str());
    if (kid < 0) return false;
    fn(kid);
    return true;
}

// gguf-py writes scalars as INT32/FLOAT32/BOOL; the strict gguf_get_val_i64/f32
// getters only accept their exact type.  Read via the raw data pointer instead.
}  // namespace

bool gguf_meta_int(const GGUFModel & m, const std::string & key, int64_t & out) {
    return gguf_meta(m, key, [&](int kid) {
        const enum gguf_type t = gguf_get_kv_type(m.gguf, kid);
        const void * p = gguf_get_val_data(m.gguf, kid);
        if (t == GGUF_TYPE_INT32)        out = *(const int32_t *) p;
        else if (t == GGUF_TYPE_INT16)   out = *(const int16_t *) p;
        else if (t == GGUF_TYPE_INT8)    out = *(const int8_t *) p;
        else if (t == GGUF_TYPE_INT64)   out = *(const int64_t *) p;
        else if (t == GGUF_TYPE_FLOAT32) out = (int64_t) *(const float *) p;
        else                             out = 0;
    });
}

bool gguf_meta_float(const GGUFModel & m, const std::string & key, float & out) {
    return gguf_meta(m, key, [&](int kid) {
        const enum gguf_type t = gguf_get_kv_type(m.gguf, kid);
        const void * p = gguf_get_val_data(m.gguf, kid);
        if (t == GGUF_TYPE_FLOAT32)      out = *(const float *) p;
        else if (t == GGUF_TYPE_FLOAT64) out = (float) *(const double *) p;
        else if (t == GGUF_TYPE_INT32)   out = (float) *(const int32_t *) p;
        else if (t == GGUF_TYPE_INT64)   out = (float) *(const int64_t *) p;
        else                             out = 0.0f;
    });
}

bool gguf_meta_bool(const GGUFModel & m, const std::string & key, bool & out) {
    return gguf_meta(m, key, [&](int kid) {
        const enum gguf_type t = gguf_get_kv_type(m.gguf, kid);
        const void * p = gguf_get_val_data(m.gguf, kid);
        if (t == GGUF_TYPE_BOOL)         out = *(const bool *) p;
        else if (t == GGUF_TYPE_UINT8)   out = *(const uint8_t *) p;
        else if (t == GGUF_TYPE_INT32)   out = *(const int32_t *) p != 0;
        else                             out = false;
    });
}

bool gguf_meta_str(const GGUFModel & m, const std::string & key, std::string & out) {
    return gguf_meta(m, key, [&](int kid) {
        const char * s = gguf_get_val_str(m.gguf, kid);
        out = s ? s : "";
    });
}

bool gguf_meta_floats(const GGUFModel & m, const std::string & key, std::vector<float> & out) {
    return gguf_meta(m, key, [&](int kid) {
        const int n = (int) gguf_get_arr_n(m.gguf, kid);
        const void * data = gguf_get_arr_data(m.gguf, kid);
        const float * f = (const float *) data;
        out.assign(f, f + n);
    });
}

}  // namespace pc_nsf_hifigan
