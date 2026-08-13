# pc-nsf-hifigan.cpp

**NSF-HiFiGAN (mini_nsf) vocoder on ggml** · 基于 ggml 的 NSF-HiFiGAN 声码器推理库

An independent vocoder library for the DiffSinger `diffsinger.cpp` pipeline. This repository owns only the **ggml vocoder body**; the mini-nsf sine source generator is delegated to [KakaruHayate/libmininsf](https://github.com/KakaruHayate/libmininsf).

面向 DiffSinger `diffsinger.cpp` 链路的独立 vocoder 库。本仓库只负责 **ggml vocoder body**；mini-nsf 正弦源生成委托给 [KakaruHayate/libmininsf](https://github.com/KakaruHayate/libmininsf)。

---

## Feature highlights · 特性

- **Native ggml ops** · 全部 ggml 原生算子：`ggml_conv_1d` / `ggml_conv_transpose_1d` / `ggml_mul_mat`（source conv）
- **Multi-backend** · 权重自动上传到后端 buffer（CPU/Vulkan/CUDA/Metal 通用），设备 shader 不读 CPU 内存
- **F16/F32 only** · 无量化接口（fp16 训练试点预留）
- **Mel front-ends** · `mel_nvstft`（DiffSinger hifigan 前端）+ `MelExtractor`（生态 API parity）
- **CLI + CTest** · 单条/批量 vocode + CTest golden 校验（纯 numpy 参考，无需模型资产）

## Quick start · 快速上手

```bash
# 1. convert（PT checkpoint -> GGUF；需 torch + gguf）
python converter/convert_hifigan.py --ckpt model.ckpt --config config.json --out hifigan.gguf

# 2. build（见 BUILDING.md）
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. vocode（mel.bin + f0.bin -> out.wav）
build/bin/hifigan_cli hifigan.gguf mel.bin f0.bin out.wav
```

## Ecosystem positioning · 生态定位

| Repository · 仓库 | Component · 组件 | Status · 状态 |
|---|---|---|
| **pc-nsf-hifigan.cpp** | NSF-HiFiGAN vocoder（本仓库，libmininsf 源） | maintained |
| hachitune.cpp | RMVPE pitch + cuFCPE | maintained |
| hifisampler.cpp | VR vocal remover | maintained |
| game_ggml_cli | GAME（DiffSinger V3, score recognition） | independent opudep |
| ggml-patch | patch set + ext/（无 ggml fork） | maintained |
| libmininsf | mini-nsf sine source generator（本仓库依赖） | maintained |

## Development guide · 开发指引

### For contributors · 贡献者

- **No quantization path** · 本仓库**不引入量化**：vocoder 对数值敏感，仅保留 F16/F32；未来 fp16 训练试点在 torch 侧先验证
- **Dependency policy** · 上游不接受对 ggml 的修改 → 进 `ggml-patch`（patch 集），由启用 CUDA 的消费者应用；本仓库 CPU/F16 路径无需 patch
- **Numeric gate before commit** · 提交前跑 `tests/` golden 对比（`gen_hifigan_golden.py` + CTest t01/t02），wav 输出与 torch 参考一致
- **mininsf source sync** · source generator 改动在 `libmininsf` 仓库，本仓库只消费（FetchContent）

### Known pitfalls · 已知坑

- `ggml_conv_transpose_1d` 需注意输出长度语义（与 torch ConvTranspose1d 的 padding 约定）
- mel 前端二选一：`mel_nvstft`（DiffSinger 专用，reflect pad + center=False）vs `MelExtractor`（htk/slaney 通用）；不要混用

## License · 许可

代码 MIT（本仓库）。模型权重视各自许可：DiffSinger 官方 NSF-HiFiGAN 权重为 CC BY-NC-SA 4.0（见 NOTICE.md）。开源社区 / non-commercial 使用请注意。

## Project layout · 目录结构

```
include/pc_nsf_hifigan/  公共头文件（gguf_model.h, mel.h, hifigan.h）
src/                     实现（gguf_model.cpp, mel.cpp, hifigan.cpp）
tools/                   CLI（hifigan_cli.cpp）
converter/               权重转换脚本（convert_hifigan.py）
tests/                   golden 对比测试（numpy 参考，无需模型资产）
examples/external_consumer  第三方集成示例
third_party/             pocketfft_hdronly.h（vendored 单头文件）
cmake/Dependencies.cmake 依赖管理（FetchContent）
```

[构建](BUILDING.md) · [架构](docs/hifigan.md) · [测试](tests/README.md)
