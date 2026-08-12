#!/usr/bin/env python3
"""Diagnose mel test failure: compare numpy golden vs DiffSinger torch nvSTFT.

Uses the diffsinger conda env (torch + librosa optional).  Writes a dump the
C++ side can also read back.
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

# --- torch mirror of generate_signal wave (used by both C++ and numpy) -------
def make_signal_wav(n, sr):
    t = np.arange(n) / sr
    return (0.3 * np.sin(2 * np.pi * 440 * t)
            + 0.15 * np.sin(2 * np.pi * 880 * t)
            + 0.05 * np.sin(2 * np.pi * 1320 * t)).astype(np.float32)


def torch_nv_mel(wav, sr=44100, n_fft=2048, win=2048, hop=512, n_mels=128,
                 fmin=40.0, fmax=16000.0, clip=1e-5):
    """Mirror of modules/nsf_hifigan/nvSTFT.py STFT.get_mel (slaney)."""
    wav_t = torch.from_numpy(wav).unsqueeze(0)  # [1, L]
    pad_l = (win - hop) // 2
    pad_r = (win - hop + 1) // 2
    wav_t = F.pad(wav_t, (pad_l, pad_r), mode="reflect")       # [1, L+pad]
    spec = torch.stft(wav_t.squeeze(0), n_fft, hop_length=hop, win_length=win,
                      window=torch.hann_window(win), center=False, return_complex=True)
    mag = spec.abs().T                                           # [T, n_bins] f32
    # mel filterbank slaney (mirror librosa.filters.mel)
    n_bins = n_fft // 2 + 1
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    def hz2mel(h):
        h = np.asarray(h, dtype=np.float64)
        return np.where(h >= min_log_hz, min_log_mel + np.log(h / min_log_hz) / logstep, h / f_sp)
    def mel2hz(m):
        m = np.asarray(m, dtype=np.float64)
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), m * f_sp)
    fft_freqs = np.arange(n_bins) * sr / n_fft
    mel_min, mel_max = hz2mel(fmin), hz2mel(fmax)
    pts = mel2hz(mel_min + np.arange(n_mels + 2) * (mel_max - mel_min) / (n_mels + 1.0))
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        lo, c, up = pts[m], pts[m + 1], pts[m + 2]
        w = np.zeros(n_bins, dtype=np.float64)
        inx = (fft_freqs >= lo) & (fft_freqs <= c)
        w[inx] = (fft_freqs[inx] - lo) / (c - lo)
        inx2 = (fft_freqs > c) & (fft_freqs <= up)
        w[inx2] = (up - fft_freqs[inx2]) / (up - c)
        fb[m] = (w * (2.0 / (up - lo))).astype(np.float32)
    mel = fb @ mag.T.numpy()
    return np.log(np.clip(mel, clip, None)).T  # sloppy: needed .astype later


if __name__ == "__main__":
    import importlib.util
    specp = importlib.util.spec_from_file_location(
        "gg", "J:/GGML-GAME/pc-nsf-hifigan.cpp/tests/gen_golden.py")
    gg = importlib.util.module_from_spec(specp)
    specp.loader.exec_module(gg)

    wav44 = make_signal_wav(22050, 44100)
    npt = gg.mel_nvstft_np(wav44)
    tr = torch_nv_mel(wav44).astype(np.float32)
    print("numpy ref shape", npt.shape)
    print("torch ref shape", tr.shape)
    d = tr.ravel() - npt.ravel()
    print("max abs diff torch-numpy:", np.abs(d).max())
    print("rmse:", np.sqrt(np.mean(d * d)))

    # also mel_ng (htk, center=True path)
    w16 = make_signal_wav(3200, 16000)
    # C++ uses MelExtractor (htk=True, center=True); numpy uses mel_center_htk_np
    ng_np = gg.mel_center_htk_np(w16, 16000, 1024, 1024, 160, 128, 30.0, 8000.0, 1e-5)
    # torch htk center=True mirror
    pad = 1024 // 2
    wt = torch.from_numpy(w16).unsqueeze(0)
    wt = F.pad(wt, (pad, pad), mode="reflect").squeeze(0)
    spec = torch.stft(wt, 1024, hop_length=160, win_length=1024,
                      window=torch.hann_window(1024), center=False, return_complex=True)
    mag = spec.abs().T
    n_bins = 1024 // 2 + 1
    fft_freqs = np.arange(n_bins) * 16000 / 1024
    mel_min = 2595.0 * np.log10(1 + 30.0 / 700.0)
    mel_max = 2595.0 * np.log10(1 + 8000.0 / 700.0)
    pts = 700.0 * (10.0 ** ((mel_min + np.arange(130) * (mel_max - mel_min) / 129) / 2595.0) - 1.0)
    fb = np.zeros((128, n_bins), dtype=np.float32)
    for m in range(128):
        lo, c, up = pts[m], pts[m + 1], pts[m + 2]
        w = np.zeros(n_bins, dtype=np.float64)
        inx = (fft_freqs >= lo) & (fft_freqs <= c)
        w[inx] = (fft_freqs[inx] - lo) / (c - lo)
        inx2 = (fft_freqs > c) & (fft_freqs <= up)
        w[inx2] = (up - fft_freqs[inx2]) / (up - c)
        fb[m] = (w * (2.0 / (up - lo))).astype(np.float32)
    mel_t = (fb @ mag.T.numpy()).T
    tr16 = np.log(np.clip(mel_t, 1e-5, None)).astype(np.float32)
    print("mel_ng numpy shape", ng_np.shape, "torch shape", tr16.shape)
    d16 = tr16.ravel() - ng_np.ravel()
    print("max abs diff torch-numpy(16k):", np.abs(d16).max(), "rmse:", np.sqrt(np.mean(d16 * d16)))

    # dump both numpy goldens to where the C++ reads, for visual compare later
    npt.astype("<f4").tofile("J:/GGML-GAME/pc-nsf-hifigan.cpp/build/dbg/mel_nvstft_44k.bin")
    ng_np.astype("<f4").tofile("J:/GGML-GAME/pc-nsf-hifigan.cpp/build/dbg/mel_ng_16k.bin")
    print("dumped numpy goldens to build/dbg")
