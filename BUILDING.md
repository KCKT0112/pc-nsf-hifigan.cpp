# BUILDING

All dependencies are fetched by CMake (FetchContent); no manual third-party installs.

## Dependencies

| Component | Source | Version |
|---|---|---|
| ggml | https://github.com/ggerganov/ggml | v0.11.0 (pin) |
| libmininsf | KakaruHayate/libmininsf | main |
| pocketfft | mreineck/pocketfft | cpp pin |
| dr_libs (dr_wav) | mackron/dr_libs | master pin |

> Patch policy: this repository **does not maintain a ggml fork**; patches not accepted upstream go to `KakaruHayate/ggml-patch`, applied by consumers that enable CUDA. The CPU/F16 path needs no patch (see docs/hifigan.md).

## Steps

```bash
# Release build
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D PCNSF_BUILD_CLI=ON
cmake --build build --config Release -j
# binaries in build/bin/
```

## Key CMake options

| Option | Default | Meaning |
|---|---|---|
| `PCNSF_CUDA` | OFF | CUDA backend |
| `PCNSF_VULKAN` | OFF | Vulkan backend |
| `PCNSF_METAL` | OFF (Apple: ON) | Metal backend (auto on Apple) |
| `PCNSF_BUILD_CLI` | ON | hifigan_cli |
| `PCNSF_BUILD_TESTS` | ON | golden comparison tests (needs Python3 + numpy) |
| `PCNSF_BUILD_EXAMPLES` | OFF | examples/external_consumer |

## Vulkan / CUDA

Libraries and executables auto-detect and link. Runtime requirements:

- Vulkan: a Vulkan-capable GPU driver; the < 1 GB net buffer requirement fits default single-GPU memory (this vocoder peaks far below 1 GB).
- CUDA: this repo does not carry ggml-patch's CUDA kernels itself; when enabling CUDA, verify the target platform has the `ggml-patch` conv_transpose_1d fix applied, otherwise results may be inaccurate.

## Tests

```bash
cmake --build build --config Release -j
ctest --test-dir build         # runs gen_golden first (numpy reference)
```

- `t01/t02`: mel front-end golden comparison
- `t03`: libmininsf source golden comparison
- `t04`: gguf_model load + meta accessors (tiny.gguf, needs gguf pkg)
- `t05_vocode` (optional, set `PCNSF_MODEL_GGUF` at configure): real-model end-to-end vocode smoke

CI builds core only and runs t01–t04.

## Generating a model

```bash
python converter/convert_hifigan.py --ckpt model.ckpt --config config.json \
    --out hifigan.gguf --dtype F16   # default (recommended)
# or debug/golden: --dtype F32
```

`model.ckpt` must have the DiffSinger hifigan structure (incl. `generator.ups.*.weight`; weight_norm `weight_g`/`weight_v` pairs are materialized automatically).
