// SPDX-License-Identifier: MPL-2.0
#pragma once

// NSF-HiFiGAN (mini_nsf) vocoder on ggml native ops (ggml_conv_1d /
// ggml_conv_transpose_1d).  The sine source generation is delegated to
// KakaruHayate/libmininsf (D1 decision); this repo owns the ggml vocoder body.
//
// Server-style API: load model once, vocode many (mel, f0) inputs.

#include <cstddef>
#include <memory>
#include <vector>

#include "pc_nsf_hifigan/gguf_model.h"

namespace pc_nsf_hifigan {

struct HifiganModel {
    std::unique_ptr<GGUFModel> gguf;

    // config (from GGUF meta)
    int num_mels = 0;
    int upsample_initial_channel = 0;
    int num_upsamples = 0;
    int num_resblocks = 0;
    bool mini_nsf = true;
    float noise_sigma = 0.0f;
    int sampling_rate = 0;
    int hop_size = 0;
    int upp = 0;             // source upsample factor (prod rates[:2]) = 64
    float source_sr = 0.0f;  // 44100 / prod(rates[2:]) = 5512.5
    ggml_type compute_type = GGML_TYPE_F32;   // engine precision line
    std::vector<int> upsample_rates;
    std::vector<int> upsample_kernels;
    std::vector<int> resblock_kernels;
    std::vector<int> resblock_dilations;

    // precision: "F32" (default, weights+compute fp32) or "F16" (reserved for
    // future fp16/bf16 training pilots).  The F16 line reads weights as fp16
    // but activations stay fp32 (ggml 1D conv requires fp32 activations on
    // CPU; bias is always stored F32).  See docs/hifigan.md §5.
    HifiganModel(const std::string & path, int n_threads, const char * precision = "F32");
    ~HifiganModel();
    HifiganModel(HifiganModel &&) noexcept;
    HifiganModel & operator=(HifiganModel &&) noexcept;
    HifiganModel(const HifiganModel &) = delete;
    HifiganModel & operator=(const HifiganModel &) = delete;
};

// Vocode mel [T, num_mels] (row-major, natural-log) + f0 [T] (Hz) into a
// mono waveform of T*hop_size samples.  noise_sigma disabled by default.
void hifigan_run(const HifiganModel & m, const float * mel, const float * f0, int T,
                 std::vector<float> & wav);

}  // namespace pc_nsf_hifigan
