#include "pc_nsf_hifigan/mel.h"

#include "pocketfft_hdronly.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace pc_nsf_hifigan {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// --- librosa mel scales -------------------------------------------------
float hz_to_mel_htk(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}
float mel_to_hz_htk(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}
float hz_to_mel_slaney(float hz) {
    const float f_sp = 200.0f / 3.0f, min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp, logstep = std::log(6.4f) / 27.0f;
    return hz >= min_log_hz ? min_log_mel + std::log(hz / min_log_hz) / logstep : hz / f_sp;
}
float mel_to_hz_slaney(float mel) {
    const float f_sp = 200.0f / 3.0f, min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp, logstep = std::log(6.4f) / 27.0f;
    return mel >= min_log_mel ? min_log_hz * std::exp(logstep * (mel - min_log_mel)) : mel * f_sp;
}

// [n_mels, n_bins] filterbank (row-major).
std::vector<float> make_mel_filterbank(const MelConfig & cfg) {
    const int n_bins = cfg.n_fft / 2 + 1;
    std::vector<float> fb(static_cast<std::size_t>(cfg.n_mels) * n_bins, 0.0f);

    auto h2m = cfg.htk ? hz_to_mel_htk : hz_to_mel_slaney;
    auto m2h = cfg.htk ? mel_to_hz_htk : mel_to_hz_slaney;

    std::vector<float> fft_freqs(n_bins);
    for (int k = 0; k < n_bins; ++k)
        fft_freqs[k] = static_cast<float>(k) * cfg.sample_rate / static_cast<float>(cfg.n_fft);

    const float mel_min = h2m(cfg.fmin), mel_max = h2m(cfg.fmax);
    std::vector<float> hz_points(cfg.n_mels + 2);
    for (int i = 0; i < cfg.n_mels + 2; ++i)
        hz_points[i] = m2h(mel_min + (mel_max - mel_min) * i / (cfg.n_mels + 1));

    for (int m = 0; m < cfg.n_mels; ++m) {
        const float lower = hz_points[m], center = hz_points[m + 1], upper = hz_points[m + 2];
        const float enorm = 2.0f / (upper - lower);
        for (int k = 0; k < n_bins; ++k) {
            const float f = fft_freqs[k];
            float w = 0.0f;
            if (f >= lower && f <= center)      w = (f - lower) / (center - lower);
            else if (f > center && f <= upper)  w = (upper - f) / (upper - center);
            fb[m * n_bins + k] = w * enorm;
        }
    }
    if (std::getenv("MEL_DUMP_FB")) {
        FILE * mf = std::fopen("nv_fb.bin", "wb");
        if (mf) { std::fwrite(fb.data(), 4, fb.size(), mf); std::fclose(mf); }
    }
    return fb;
}

// torch.hann_window (periodic).
std::vector<float> make_hann_window(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i) w[i] = 0.5f - 0.5f * std::cos(2.0f * kPi * i / n);
    return w;
}

// F.pad(mode='reflect') semantics (edge not duplicated): out[i] = in[pad - i] for i < pad.
void reflect_pad(const float * src, std::size_t n, int pad_l, int pad_r,
                 std::vector<float> & dst) {
    dst.resize(n + pad_l + pad_r);
    for (int i = 0; i < pad_l; ++i) dst[i] = src[pad_l - i];
    std::memcpy(dst.data() + pad_l, src, n * sizeof(float));
    for (int i = 0; i < pad_r; ++i) dst[pad_l + n + i] = src[n - 2 - i];
}

}  // namespace

struct MelExtractor::Impl {
    MelConfig          cfg;
    std::vector<float> window;
    std::vector<float> mel_fb;   // [n_mels, n_bins]
};

MelExtractor::MelExtractor(const MelConfig & cfg) : impl_(std::make_unique<Impl>()) {
    impl_->cfg    = cfg;
    impl_->window = make_hann_window(cfg.win_length);
    impl_->mel_fb = make_mel_filterbank(cfg);
}

MelExtractor::~MelExtractor() = default;
MelExtractor::MelExtractor(MelExtractor &&) noexcept = default;
MelExtractor & MelExtractor::operator=(MelExtractor &&) noexcept = default;

const MelConfig & MelExtractor::config() const noexcept { return impl_->cfg; }

int MelExtractor::num_frames(std::size_t n_samples) const noexcept {
    const auto & cfg = impl_->cfg;
    // torch.stft(center=True): pad n_fft//2 each side, hop from index 0.
    const std::int64_t pad = cfg.n_fft / 2;
    const std::int64_t frames = (static_cast<std::int64_t>(n_samples) + 2 * pad - cfg.win_length)
                                    / cfg.hop_length + 1;
    return frames > 0 ? static_cast<int>(frames) : 0;
}

std::vector<float> MelExtractor::magnitude(const float * wav, std::size_t n) const {
    const auto & cfg = impl_->cfg;
    const auto & window = impl_->window;
    const int n_fft = cfg.n_fft, win = cfg.win_length, hop = cfg.hop_length;
    const int n_bins = n_fft / 2 + 1;
    const int pad = n_fft / 2;

    std::vector<float> padded;
    reflect_pad(wav, n, pad, pad, padded);

    const int T = num_frames(n);
    if (T <= 0) return {};

    std::vector<float>               frame(n_fft, 0.0f);
    std::vector<std::complex<float>> spec(n_bins);
    std::vector<float>               mag(static_cast<std::size_t>(T) * n_bins);

    pocketfft::shape_t   shape      = {static_cast<std::size_t>(n_fft)};
    pocketfft::stride_t  stride_in  = {sizeof(float)};
    pocketfft::stride_t  stride_out = {sizeof(std::complex<float>)};
    pocketfft::shape_t   axes       = {0};

    for (int t = 0; t < T; ++t) {
        const std::size_t off = static_cast<std::size_t>(t) * hop;
        std::fill(frame.begin(), frame.end(), 0.0f);
        for (int k = 0; k < win; ++k) frame[k] = padded[off + k] * window[k];
        pocketfft::r2c(shape, stride_in, stride_out, axes, pocketfft::FORWARD,
                       frame.data(), spec.data(), 1.0f);
        for (int k = 0; k < n_bins; ++k)
            mag[static_cast<std::size_t>(t) * n_bins + k] = std::hypot(spec[k].real(), spec[k].imag());
    }
    return mag;
}

std::vector<float> MelExtractor::forward(const float * wav, std::size_t n) const {
    const auto & cfg = impl_->cfg;
    const auto & mel_fb = impl_->mel_fb;
    const int n_bins = cfg.n_fft / 2 + 1;
    const int n_mels = cfg.n_mels;
    const int T = num_frames(n);
    if (T <= 0) return {};

    std::vector<float> mag = magnitude(wav, n);
    std::vector<float> out(static_cast<std::size_t>(T) * n_mels);
    if (std::getenv("MEL_DUMP_ACC")) {
        FILE * mf = std::fopen("acc_ng.bin", "wb");
        if (mf) {
            for (int t = 0; t < T; ++t) {
                const float * mrow = mag.data() + static_cast<std::size_t>(t) * n_bins;
                for (int mm = 0; mm < n_mels; ++mm) {
                    const float * fbrow = mel_fb.data() + static_cast<std::size_t>(mm) * n_bins;
                    float acc = 0.0f;
                    for (int k = 0; k < n_bins; ++k) acc += fbrow[k] * mrow[k];
                    std::fwrite(&acc, 4, 1, mf);
                }
            }
            std::fclose(mf);
            std::fprintf(stderr, "dumped acc_ng.bin\n");
        }
    }
    for (int t = 0; t < T; ++t) {
        const float * mrow = mag.data() + static_cast<std::size_t>(t) * n_bins;
        for (int m = 0; m < n_mels; ++m) {
            const float * fbrow = mel_fb.data() + static_cast<std::size_t>(m) * n_bins;
            float acc = 0.0f;
            for (int k = 0; k < n_bins; ++k) acc += fbrow[k] * mrow[k];
            out[static_cast<std::size_t>(t) * n_mels + m] = std::log(std::max(acc, cfg.clip_val));
        }
    }
    return out;
}

// DiffSinger nvSTFT mel (hifigan vocoder front-end).
std::vector<float> mel_nvstft(const float * wav, std::size_t n,
                              int sample_rate, int n_fft, int win_length,
                              int hop_length, int n_mels, float fmin, float fmax,
                              float clip_val) {
    MelConfig cfg;
    cfg.sample_rate = sample_rate;
    cfg.n_fft       = n_fft;
    cfg.win_length  = win_length;
    cfg.hop_length  = hop_length;
    cfg.n_mels      = n_mels;
    cfg.fmin        = fmin;
    cfg.fmax        = fmax;
    cfg.clip_val    = clip_val;
    cfg.htk         = false;  // slaney

    const int n_bins = n_fft / 2 + 1;
    // reflect pad (win-hop)//2 each side (nvSTFT: F.pad((win-hop)//2, (win-hop+1)//2))
    const int pad_l = (win_length - hop_length) / 2;
    const int pad_r = (win_length - hop_length + 1) / 2;
    std::vector<float> padded;
    reflect_pad(wav, n, pad_l, pad_r, padded);

    // torch.stft(center=False): frames start at 0, no extra padding.
    const int64_t frames = (static_cast<int64_t>(padded.size()) - win_length) / hop_length + 1;
    if (frames <= 0) return {};

    std::vector<float> window = make_hann_window(win_length);
    std::vector<float> fb = make_mel_filterbank(cfg);  // [n_mels, n_bins]

    std::vector<float>               frame(n_fft, 0.0f);
    std::vector<std::complex<float>> spec(n_bins);
    std::vector<float>               mag(static_cast<std::size_t>(frames) * n_bins);
    pocketfft::shape_t      shape      = {static_cast<std::size_t>(n_fft)};
    pocketfft::stride_t     stride_in  = {sizeof(float)};
    pocketfft::stride_t     stride_out = {sizeof(std::complex<float>)};
    pocketfft::shape_t      axes       = {0};

    for (int64_t t = 0; t < frames; ++t) {
        const std::size_t off = static_cast<std::size_t>(t) * hop_length;
        std::fill(frame.begin(), frame.end(), 0.0f);
        for (int k = 0; k < win_length; ++k) frame[k] = padded[off + k] * window[k];
        pocketfft::r2c(shape, stride_in, stride_out, axes, pocketfft::FORWARD,
                       frame.data(), spec.data(), 1.0f);
        for (int k = 0; k < n_bins; ++k)
            mag[static_cast<std::size_t>(t) * n_bins + k] = std::hypot(spec[k].real(), spec[k].imag());
    }

    std::vector<float> out(static_cast<std::size_t>(frames) * n_mels);
    if (std::getenv("MEL_DUMP_MAG")) {
        FILE * mf = std::fopen("nv_mag.bin", "wb");
        if (mf) { std::fwrite(mag.data(), 4, mag.size(), mf); std::fclose(mf); }
    }
    for (int64_t t = 0; t < frames; ++t) {
        const float * mrow = mag.data() + static_cast<std::size_t>(t) * n_bins;
        for (int m = 0; m < n_mels; ++m) {
            const float * fbrow = fb.data() + static_cast<std::size_t>(m) * n_bins;
            float acc = 0.0f;
            for (int k = 0; k < n_bins; ++k) acc += fbrow[k] * mrow[k];
            out[static_cast<std::size_t>(t) * n_mels + m] = std::log(std::max(acc, clip_val));
        }
    }
    return out;
}

}  // namespace pc_nsf_hifigan
