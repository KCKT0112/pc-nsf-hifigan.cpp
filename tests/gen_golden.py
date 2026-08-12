#!/usr/bin/env python3
"""Golden-file generator for pc-nsf-hifigan C++ tests.

Produces deterministic .bin goldens from a pure numpy re-derivation of the
DiffSinger NSF-HiFiGAN graph's deterministic pieces:
  * mel_nvstft  — the nvSTFT.py mel front-end (hann win, slaney fb, reflect pad)
  * fastsinegen — the mini-NSF sine source formula (libmininsf mirrors it)
  * a tiny .gguf with int/float/bool/string/array meta + a single weight

Only numpy is required (the tiny GGUF needs the gguf package, and is skipped
if unavailable).  These goldens let the C++ tests verify the graph wiring
end-to-end WITHOUT a model asset (the vocode itself needs PCNSF_MODEL_GGUF,
see BUILDING.md).
"""
import argparse
import math
import os

import numpy as np

try:
    from gguf import GGUFWriter
    from gguf.constants import GGMLQuantizationType as QT
    HAVE_GGUF = True
except Exception:  # pragma: no cover - gguf pkg optional
    HAVE_GGUF = False


# --------------------------------------------------------------------------
# mel_nvstft reference (params for the 44.1k vocoder front-end)
# --------------------------------------------------------------------------
def mel_nvstft_np(wav, sr=44100, n_fft=2048, win=2048, hop=512,
                  n_mels=128, fmin=40.0, fmax=16000.0, clip=1e-5, htk=False):
    # htk selects the librosa mel scale: True -> htk formula (MelExtractor),
    # False -> slaney (nvSTFT/mel_nvstft).  Padding differs (see below): the
    # MelExtractor path pads n_fft//2 and uses hop from index 0 (center=True),
    # while nvSTFT pads (win-hop)//2 and uses center=False.
    if htk:
        # separate reference for the MelExtractor (center=True, htk) case
        return mel_center_htk_np(wav, sr, n_fft, win, hop, n_mels, fmin, fmax, clip)

    wav = np.asarray(wav, dtype=np.float32)

    # 1) reflect pad (win-hop)//2 each side, right side uses (win-hop+1)//2
    pad_l = (win - hop) // 2
    pad_r = (win - hop + 1) // 2
    padded = np.pad(wav, (pad_l, pad_r), mode="reflect")

    # 2) frames, center=False
    T = (len(padded) - win) // hop + 1
    idx = np.arange(win)[None, :] + hop * np.arange(T)[:, None]
    frames = padded[idx]
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)
    spec = np.fft.rfft(frames * hann, n=n_fft, axis=1)          # [T, n_bins]
    mag = np.abs(spec).astype(np.float32)

    # 3) mel filterbank (slaney scale) — mirrors make_mel_filterbank(htk=False)
    n_bins = n_fft // 2 + 1
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    def hz2mel2(h):
        h = np.asarray(h, dtype=np.float64)
        return np.where(h >= min_log_hz, min_log_mel + np.log(h / min_log_hz) / logstep,
                        h / f_sp)

    def mel2hz2(m):
        m = np.asarray(m, dtype=np.float64)
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)),
                        m * f_sp)

    fft_freqs = np.arange(n_bins) * sr / n_fft
    mel_min, mel_max = hz2mel2(fmin), hz2mel2(fmax)
    pts = mel2hz2(mel_min + np.arange(n_mels + 2) * (mel_max - mel_min) / (n_mels + 1.0))
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        lo, c, up = pts[m], pts[m + 1], pts[m + 2]
        w = np.zeros(n_bins)
        inx = (fft_freqs >= lo) & (fft_freqs <= c)
        w[inx] = (fft_freqs[inx] - lo) / (c - lo)
        inx2 = (fft_freqs > c) & (fft_freqs <= up)
        w[inx2] = (up - fft_freqs[inx2]) / (up - c)
        fb[m] = w * (2.0 / (up - lo))
    mel = (fb.astype(np.float32) @ mag.T).T          # [T, n_mels]
    return np.log(np.maximum(mel, clip)).astype(np.float32)


def mel_center_htk_np(wav, sr, n_fft, win, hop, n_mels, fmin, fmax, clip):
    """Reference for MelExtractor path: center=True, htk mel scale."""
    wav = np.asarray(wav, dtype=np.float32)
    pad = n_fft // 2
    padded = np.pad(wav, (pad, pad), mode="reflect")
    T = (wav.shape[0] + 2 * pad - win) // hop + 1
    idx = np.arange(win)[None, :] + hop * np.arange(T)[:, None]
    frames = padded[idx]
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)
    spec = np.fft.rfft(frames * hann, n=n_fft, axis=1)
    mag = np.abs(spec).astype(np.float32)
    n_bins = n_fft // 2 + 1

    def hz2mel_htk(h):
        return 2595.0 * np.log10(1.0 + h / 700.0)

    def mel2hz_htk(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    fft_freqs = np.arange(n_bins) * sr / n_fft
    mel_min, mel_max = hz2mel_htk(fmin), hz2mel_htk(fmax)
    pts = mel2hz_htk(mel_min + np.arange(n_mels + 2) * (mel_max - mel_min) / (n_mels + 1.0))
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        lo, c, up = pts[m], pts[m + 1], pts[m + 2]
        w = np.zeros(n_bins)
        inx = (fft_freqs >= lo) & (fft_freqs <= c)
        w[inx] = (fft_freqs[inx] - lo) / (c - lo)
        inx2 = (fft_freqs > c) & (fft_freqs <= up)
        w[inx2] = (up - fft_freqs[inx2]) / (up - c)
        fb[m] = w * (2.0 / (up - lo))
    mel = (fb.astype(np.float32) @ mag.T).T
    return np.log(np.maximum(mel, clip)).astype(np.float32)


def make_signal_wav(n=22050, sr=44100.0):
    t = np.arange(n) / sr
    return (0.3 * np.sin(2 * np.pi * 440 * t)
            + 0.15 * np.sin(2 * np.pi * 880 * t)
            + 0.05 * np.sin(2 * np.pi * 1320 * t)).astype(np.float32)


# --------------------------------------------------------------------------
# fastsinegen reference (mini-nsf formula; libmininsf exact sin path)
# --------------------------------------------------------------------------
def fastsinegen_np(f0, source_sr=5512.5, upsample=64):
    f0 = np.asarray(f0, dtype=np.float32)
    n = f0.shape[0]
    out = np.zeros(n * upsample)       # float64 accumulation, cast at end
    phase_sum = 0.0
    for t in range(n):
        s0 = f0[t] / source_sr
        ds0 = (f0[t + 1] - f0[t]) / source_sr if t + 1 < n else 0.0
        offset = math.fmod(phase_sum, 1.0)
        for k in range(upsample):
            nk = k + 1
            rad = s0 * nk + 0.5 * ds0 * nk * (nk - 1.0) / upsample + offset
            out[t * upsample + k] = math.sin(2 * math.pi * rad)
        nk = upsample
        rad_last = s0 * nk + 0.5 * ds0 * nk * (nk - 1.0) / upsample
        phase_sum += (math.fmod(rad_last + 0.5, 1.0) - 0.5)
    return out.astype(np.float32)


def save_raw(path, arr):
    arr.astype("<f4").tofile(path)
    print(f"  {path}: {arr.size} floats")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print("generating goldens in", args.out)

    # (1) mel_nvstft golden — 22050 samples (~0.5 s)
    wav = make_signal_wav()
    mel = mel_nvstft_np(wav)
    save_raw(os.path.join(args.out, "mel_nvstft_44k.bin"), mel)

    # (2) mel_ng golden — MelExtractor path (16k, htk=True, RMVPE-style config)
    #     NOTE make_signal_wav sr must match the C++ test's timebase exactly:
    #     t = i / 16000.0f → sr=16000.0 (librosa/stft phase alignment matters).
    wav_16k = make_signal_wav(3200, sr=16000.0)
    mel_ng = mel_nvstft_np(wav_16k, sr=16000, n_fft=1024, win=1024, hop=160,
                           n_mels=128, fmin=30.0, fmax=8000.0, clip=1e-5, htk=True)
    save_raw(os.path.join(args.out, "mel_ng_16k.bin"), mel_ng)

    # (3) fastsinegen golden — 4 frames at 5512.5 Hz, upsample 64
    f0 = np.array([196.0, 261.6, 329.6, 392.0], dtype=np.float32)
    src = fastsinegen_np(f0)
    save_raw(os.path.join(args.out, "source_64.bin"), src)

    # (3) tiny GGUF with mixed metadata (requires gguf pkg; else note skip)
    if HAVE_GGUF:
        w = GGUFWriter(os.path.join(args.out, "tiny.gguf"), "tiny")
        w.add_int32("audio.sample_rate", 44100)
        w.add_int32("audio.hop", 512)
        w.add_float32("audio.n_mels_f", 128.5)
        w.add_bool("hifigan.mini_nsf", True)
        w.add_string("audio.type", "hifigan")
        w.add_array("hifigan.upsample_rates", [8.0, 8.0, 2.0, 2.0, 2.0])
        w.add_tensor("hifigan.tiny_w", np.full((3, 2), 0.25, np.float16), raw_dtype=QT.F16)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        print("  tiny.gguf written")
    else:
        print("(gguf pkg unavailable — tiny.gguf skipped; test_gguf_meta will skip)")

    print("golden generation done")


if __name__ == "__main__":
    main()
