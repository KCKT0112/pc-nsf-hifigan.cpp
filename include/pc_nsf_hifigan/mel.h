// SPDX-License-Identifier: MPL-2.0
#pragma once

// Mel spectrogram front-end matching torch.stft(center=True, pad_mode='reflect')
// + librosa mel filterbank (htk or slaney) + log(clamp).
// Used by RMVPE (16k, htk) and VR (44.1k, slaney) in the sibling repos.
// pc-nsf-hifigan only needs mel_nvstft (DiffSinger NSF-HiFiGAN mel front-end);
// the MelExtractor class is kept for API parity across the ecosystem.

#include <cstddef>
#include <memory>
#include <vector>

namespace pc_nsf_hifigan {

struct MelConfig {
    int   sample_rate = 16000;
    int   n_fft       = 1024;
    int   win_length  = 1024;
    int   hop_length  = 160;
    int   n_mels      = 128;
    float fmin        = 30.0f;
    float fmax        = 8000.0f;
    float clip_val    = 1e-5f;
    bool  htk         = true;   // librosa mel scale: htk=True (RMVPE) / False (VR)
};

class MelExtractor {
public:
    explicit MelExtractor(const MelConfig & cfg);
    ~MelExtractor();
    MelExtractor(MelExtractor &&) noexcept;
    MelExtractor & operator=(MelExtractor &&) noexcept;
    MelExtractor(const MelExtractor &) = delete;
    MelExtractor & operator=(const MelExtractor &) = delete;

    const MelConfig & config() const noexcept;

    // Frame count matching torch.stft(center=True): 1 + n/hop (with n_fft//2 pad each side).
    int num_frames(std::size_t n_samples) const noexcept;

    // Row-major [T, n_mels] log-mel.  Input is mono float in [-1, 1].
    std::vector<float> forward(const float * wav, std::size_t n) const;

    // Magnitude spectrogram [T, n_fft/2+1] (pre-filterbank, pre-log) for VR.
    std::vector<float> magnitude(const float * wav, std::size_t n) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// DiffSinger NSF-HiFiGAN mel front-end (modules/nsf_hifigan/nvSTFT.py STFT.get_mel,
// htk=False/slaney, log_e):
//   reflect pad (win-hop)//2 on each side  (NOT n_fft//2, and NOT constant)
//   torch.stft(center=False, win_length=win, hop, hann, onesided) -> |.|   (magnitude)
//   mel_basis[slaney] @ spec  ->  log(clamp(clip_val))
// Returns row-major [T, n_mels].  Input is mono float in [-1, 1].
std::vector<float> mel_nvstft(const float * wav, std::size_t n,
                              int sample_rate, int n_fft, int win_length,
                              int hop_length, int n_mels, float fmin, float fmax,
                              float clip_val = 1e-5f);

}  // namespace pc_nsf_hifigan
