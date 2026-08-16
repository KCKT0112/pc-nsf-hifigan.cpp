# pc-nsf-hifigan.cpp

> **Languages:** [English](README.md) | [中文](README_CN.md)

**NSF-HiFiGAN (mini_nsf) vocoder on ggml**

An independent vocoder library for the DiffSinger `diffsinger.cpp` pipeline. This repository owns only the **ggml vocoder body**; the mini-nsf sine source generator is delegated to [KakaruHayate/libmininsf](https://github.com/KakaruHayate/libmininsf).

---

## Feature highlights

- **Native ggml ops** — sub-pixel upsample (phase-major, exact) via F32 `im2col + mul_mat` + graph interleave (default); `ggml_conv_transpose_1d` kept as legacy fallback; `ggml_mul_mat` for the source conv
- **Multi-backend** — weights auto-uploaded to backend buffers (CPU/Vulkan/CUDA/Metal); device shaders never read CPU memory
- **F16/F32 dual precision (no quantization)** — F32 line (weights+compute fp32, exact baseline) and F16 line (fp16 weights, reserved for future fp16/bf16 training pilots)
- **Mel front-ends** — `mel_nvstft` (DiffSinger hifigan front-end) + `MelExtractor` (ecosystem API parity)
- **CLI + CTest** — single/batch vocode + CTest golden checks (pure numpy reference, no model assets)

## Quick start

```bash
# 1. convert (PT checkpoint -> GGUF; needs torch + gguf)
python converter/convert_hifigan.py --ckpt model.ckpt --config config.json --out hifigan.gguf

# 2. build (see BUILDING.md)
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. vocode (mel.bin + f0.bin -> out.wav)
build/bin/hifigan_cli hifigan.gguf mel.bin f0.bin out.wav
#    fp16 line (needs a --dtype F16 GGUF):
# HF_PRECISION=F16 build/bin/hifigan_cli hifigan_f16.gguf mel.bin f0.bin out_f16.wav
```

## Ecosystem positioning

Full reference chain: see `ggml-patch` `docs/ECOSYSTEM.md`. Training-side
repos never use ggml — these C++ repos exist for edge deployment.

| Repository | Component | Relationship |
|---|---|---|
| **pc-nsf-hifigan.cpp** | NSF-HiFiGAN vocoder (this repo) | depends on `libmininsf`; pulled by `hifisampler` as a build dependency |
| libmininsf | mini-nsf sine source | base component (this repo's dependency) |
| hifisampler.cpp | VR vocal remover | builds VR + vocoder |
| hachitune.cpp | RMVPE pitch + cuFCPE | consumes `ggml-patch` `ext/rnn` |
| ggml-patch | patch set + ext/ (no ggml fork) | ecosystem foundation |
| game_ggml_cli | GAME (DiffSinger V3, score recognition) | independent opudep |

## Model release

This repo is the **quantization exception** in the ecosystem: the vocoder is
numerically sensitive, so only two precisions are ever produced
(`converter/convert_hifigan.py --dtype F32|F16`); there is **no** `rec`/Q8
tier.

| Asset | Precision | When to use |
|---|---|---|
| `hifigan_f16.gguf` | F16 (fp16 line) | normal deployments (weights fp16) |
| `hifigan_f32.gguf` | F32 (exact line) | golden-grade regression baseline |

- Publish via GitHub **Releases**, never commit weights to the repo.
- Names always carry the precision suffix (`f16` / `f32`) — consumers glob by
  size (e.g. OpenUtau), so a bare `hifigan.gguf` would silently shadow the
  other precision.
- Release checklist: convert → F16 wav vs torch reference (CTest t01/t02 +
  `tests/gen_hifigan_golden.py`) → attach both assets to the release notes.
- Future fp16 *training* pilots are validated torch-side first; the inference
  interface stays F16/F32 only.

## Deployment note (speed baseline = ONNX/DML)

The **efficient deployment path for this vocoder is the ONNX export running on
DirectML** (`J:\0608\backend_matrix_outputs\metrics_*.md`: ~340 ms for a 20 s
clip, about 59× realtime).  This ggml engine is maintained as the **reference /
pure-inference implementation** — it is backend-portable (CPU/Vulkan/CUDA/Metal)
and numerically validated against torch (corr > 0.9999), but it is **not** the
latency baseline and should not be expected to beat ONNX/DML (see
`docs/verification_real_hifigan.md` for the measured gap).  Decision recorded:
leave hifigan on ONNX, keep the ggml build for correctness/no-GPU/edge cases
and ecosystem integration (pulled by `hifisampler`/`hachitune`).

## Development guide

### For contributors

- **No quantization path** — this repository deliberately **does not quantize**: the vocoder is numerically sensitive; F16/F32 only. Future fp16 training pilots are validated torch-side first.
- **Dependency policy** — ggml changes not accepted upstream go to `ggml-patch` (patch set), applied by consumers that enable CUDA; this repo's CPU/F16 path needs no patch.
- **Numeric gate before commit** — run `tests/` golden comparison (`gen_hifigan_golden.py` + CTest t01/t02) before committing; wav output must match the torch reference.
- **mininsf source sync** — source-generator changes live in `libmininsf`; this repo only consumes (FetchContent).

### Known pitfalls

- `ggml_conv_transpose_1d` output-length semantics differ from torch ConvTranspose1d padding conventions — verify lengths.
- Two mel front-ends, don't mix: `mel_nvstft` (DiffSinger-specific, reflect pad + center=False) vs `MelExtractor` (htk/slaney general).

## License

Code MIT (this repo). Model weights retain their own licenses: the DiffSinger official NSF-HiFiGAN weights are CC BY-NC-SA 4.0 (see NOTICE.md); note the non-commercial restriction.

## Project layout

```
include/pc_nsf_hifigan/  public headers (gguf_model.h, mel.h, hifigan.h)
src/                     impl (gguf_model.cpp, mel.cpp, hifigan.cpp)
tools/                   CLI (hifigan_cli.cpp)
converter/               weight conversion scripts (convert_hifigan.py)
tests/                   golden comparison tests (numpy reference, no model assets)
examples/external_consumer  third-party integration example
third_party/             pocketfft_hdronly.h (vendored single header)
cmake/Dependencies.cmake dependency management (FetchContent)
```

[Building](BUILDING.md) · [Architecture](docs/hifigan.md) · [Tests](tests/README.md)
