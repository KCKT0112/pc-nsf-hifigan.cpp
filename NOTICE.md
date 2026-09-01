# NOTICE

The pc-nsf-hifigan code itself is licensed under the **Mozilla Public License
2.0** (see `LICENSE`), except for the `patches/` directory, which is
dual-licensed **MIT OR Apache-2.0** (see `patches/LICENSE`) so the diffs remain
acceptable to upstream ggml.

The notices below are retained under MPL-2.0 §3.4 and must be preserved in
redistributions.

## Third-party components

| Component | License | Notes | Source |
|---|---|---|---|
| ggml | MIT | tensor engine | https://github.com/ggerganov/ggml |
| libmininsf | MPL-2.0 | mini-nsf sine source generator | KakaruHayate/libmininsf |
| pocketfft | BSD-3-Clause | FFT / filterbank | mreineck/pocketfft |
| `third_party/pocketfft_hdronly.h` | BSD-3-Clause | vendored single header — **not** covered by this repository's MPL-2.0 license | same |
| dr_libs (dr_wav) | Unlicense / Public Domain | WAV I/O (CLI) | mackron/dr_libs |
| Apache License, Version 2.0 | — | full text shipped at `licenses/Apache-2.0.txt` | — |

## Reference implementation

| Project | License | Relationship |
|---|---|---|
| [openvpi/DiffSinger](https://github.com/openvpi/DiffSinger) | **Apache-2.0** | Numerical/architectural reference for the vocoder body and the `mel_nvstft` front-end (`modules/nsf_hifigan/nvSTFT.py`). |

This repository is an **independent C++ implementation** on ggml. It reproduces
DiffSinger's numerics so that exported vocoder weights produce matching output,
but no DiffSinger source code is copied or vendored here.

## patches/ directory

`patches/` carries ggml operator patches. Attribution for the upstream projects
each operator was adopted from is recorded in
[ggml-audio-patch's NOTICE.md](https://github.com/KakaruHayate/ggml-audio-patch/blob/main/NOTICE.md),
in particular:

- `IM2COL_FAST_1D` — [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp), **Apache-2.0**
- `ggml_conv_transpose_1d_ext` — [mmwillet/TTS.cpp](https://github.com/mmwillet/TTS.cpp), MIT
- `REL_POS_BIAS`, `SCATTER_ELEMENTS` — [ggmlR](https://CRAN.R-project.org/package=ggmlR), MIT
- the ten qvac fused ops — [tetherto/qvac-ext-ggml](https://github.com/tetherto/qvac-ext-ggml), MIT

The MIT / Apache-2.0 grants issued by those authors are irrevocable and are
**not** superseded by this repository's license.

## Model assets

| Asset | License | Notes |
|---|---|---|
| DiffSinger official NSF-HiFiGAN weights | CC BY-NC-SA 4.0 | non-commercial only; derived GGUF inherits the same license |

> GGUF / WAV files produced from the DiffSinger official hifigan weights are for non-commercial research and personal use only. For commercial use or redistribution, follow CC BY-NC-SA 4.0 and provide attribution.

## Trademarks

"DiffSinger" and "OpenUTAU" are trademarks of their respective owners; this project is not affiliated with them.
