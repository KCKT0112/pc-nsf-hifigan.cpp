# pc-nsf-hifigan.cpp

NSF-HiFiGAN (mini_nsf) 原生 ggml 推理引擎 —— 面向 DiffSinger `diffsinger.cpp` 的独立 vocoder 库。

本仓库只负责 **ggml vocoder body**；mini-nsf 正弦源生成委托给
[KakaruHayate/libmininsf](https://github.com/KakaruHayate/libmininsf)（D1 决策）。

## 特性

- 全部 ggml 原生算子：`ggml_conv_1d` / `ggml_conv_transpose_1d` / `ggml_mul_mat`（source conv）
- 权重自动上传到后端 buffer（CPU/Vulkan/CUDA/Metal 通用），设备 shader 不读 CPU 内存
- 无量化接口：仅 F16/F32（fp16 训练试点预留）
- `mel_nvstft`（DiffSinger hifigan 前端）+ `MelExtractor`（ecosystem API parity）
- CLI 单条/批量 vocode + CTest golden 校验（纯 numpy 参考无需模型资产）

## 快速上手

```bash
# 1. 转换（PT checkpoint -> GGUF），需要 torch + gguf（diffsinger env）
python converter/convert_hifigan.py --ckpt model.ckpt --config config.json --out hifigan.gguf

# 2. 构建（参考 BUILDING.md）
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. 单条 vocode（mel.bin + f0.bin -> out.wav）
build/bin/hifigan_cli hifigan.gguf mel.bin f0.bin out.wav
```

## 生态定位

| 仓库 | 内容 | 状态 |
|------|------|------|
| `pc-nsf-hifigan.cpp` | NSF-HiFiGAN vocoder（本仓库，libmininsf 源） | 活跃开发 |
| `game_ggml_cli` | GAME 模型（DiffSinger V3，score recognition） | OpenUtau opudep 独立 |
| `hachitune.cpp` | RMVPE pitch + (规划) cuFCPE | 待重构 |
| `hifisampler.cpp` | VR 声码器 + 全功能 hifigan 采样 | 待重构 |
| `ggml-patch` | 方案 B：ggml fork → patch 合集（不维护 fork） | 待初始化 |
| `libmininsf` | mini-nsf 正弦源生成器（本仓库依赖） | main |

详见 [docs/hifigan.md](docs/hifigan.md) 与 [NOTICE.md](NOTICE.md)。

## 许可

代码 MIT（本仓库），模型权重视各自许可（DiffSinger 官方 NSF-HiFiGAN 权重
CC BY-NC-SA 4.0，见 [NOTICE.md](NOTICE.md)）。开源社区/non-commercial 使用注意。

## 目录

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
