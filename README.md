# pc-nsf-hifigan.cpp

> **Languages:** [English](README.md) | [中文](README_CN.md)

**NSF-HiFiGAN (mini_nsf) vocoder on ggml**

An independent vocoder library for the DiffSinger `diffsinger.cpp` pipeline. This repository owns only the **ggml vocoder body**; the mini-nsf sine source generator is delegated to [KakaruHayate/libmininsf](https://github.com/KakaruHayate/libmininsf).

---

## Feature highlights

- **Native ggml ops** — `ggml_conv_1d` / `ggml_conv_transpose_1d` / `ggml_mul_mat` (source conv)
- **Multi-backend** — weights auto-uploaded to backend buffers (CPU/Vulkan/CUDA/Metal); device shaders never read CPU memory
- **F16/F32 only** — no quantization interface (fp16 training pilot reserved)
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
```

## Ecosystem positioning

| Repository | Component | Status |
|---|---|---|
| **pc-nsf-hifigan.cpp** | NSF-HiFiGAN vocoder (this repo, libmininsf source) | maintained |
| hachitune.cpp | RMVPE pitch + cuFCPE | maintained |
| hifisampler.cpp | VR vocal remover | maintained |
| game_ggml_cli | GAME (DiffSinger V3, score recognition) | independent opudep |
| ggml-patch | patch set + ext/ (no ggml fork) | maintained |
| libmininsf | mini-nsf sine source generator (this repo's dependency) | maintained |

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
