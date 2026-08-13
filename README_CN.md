# pc-nsf-hifigan.cpp

> **语言：** [English](README.md) | [中文](README_CN.md)

**基于 ggml 的 NSF-HiFiGAN 声码器推理库**

面向 DiffSinger `diffsinger.cpp` 链路的独立 vocoder 库。本仓库只负责 **ggml vocoder body**；mini-nsf 正弦源生成委托给 [KakaruHayate/libmininsf](https://github.com/KakaruHayate/libmininsf)。

---

## 特性

- **原生 ggml 算子** — `ggml_conv_1d` / `ggml_conv_transpose_1d` / `ggml_mul_mat`（source conv）
- **多后端** — 权重自动上传到后端 buffer（CPU/Vulkan/CUDA/Metal 通用），设备 shader 不读 CPU 内存
- **仅 F16/F32** — 无量化接口（fp16 训练试点预留）
- **Mel 前端** — `mel_nvstft`（DiffSinger hifigan 前端）+ `MelExtractor`（生态 API parity）
- **CLI + CTest** — 单条/批量 vocode + CTest golden 校验（纯 numpy 参考，无需模型资产）

## 快速上手

```bash
# 1. 转换（PT checkpoint -> GGUF；需 torch + gguf）
python converter/convert_hifigan.py --ckpt model.ckpt --config config.json --out hifigan.gguf

# 2. 构建（见 BUILDING.md）
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. vocode（mel.bin + f0.bin -> out.wav）
build/bin/hifigan_cli hifigan.gguf mel.bin f0.bin out.wav
```

## 生态定位

| 仓库 | 组件 | 状态 |
|---|---|---|
| **pc-nsf-hifigan.cpp** | NSF-HiFiGAN vocoder（本仓库，libmininsf 源） | maintained |
| hachitune.cpp | RMVPE 音高 + cuFCPE | maintained |
| hifisampler.cpp | VR 人声分离 | maintained |
| game_ggml_cli | GAME（DiffSinger V3, score recognition） | independent opudep |
| ggml-patch | 补丁集 + ext/（无 ggml fork） | maintained |
| libmininsf | mini-nsf 正弦源生成器（本仓库依赖） | maintained |

## 开发指引

### 贡献者

- **无量化路径** — 本仓库**刻意不量化**：vocoder 对数值敏感，仅保留 F16/F32；未来 fp16 训练试点先在 torch 侧验证
- **依赖策略** — 上游不接受的 ggml 修改 → 进 `ggml-patch`（补丁集），由启用 CUDA 的消费者应用；本仓库 CPU/F16 路径无需补丁
- **提交前过数值门槛** — 提交前跑 `tests/` golden 对比（`gen_hifigan_golden.py` + CTest t01/t02），wav 输出须与 torch 参考一致
- **mininsf 源同步** — source generator 改动在 `libmininsf` 仓库，本仓库只消费（FetchContent）

### 已知坑

- `ggml_conv_transpose_1d` 的输出长度语义与 torch ConvTranspose1d 的 padding 约定不同——注意核对长度
- 两个 mel 前端勿混用：`mel_nvstft`（DiffSinger 专用，reflect pad + center=False）vs `MelExtractor`（htk/slaney 通用）

## 许可

代码 MIT（本仓库）。模型权重视各自许可：DiffSinger 官方 NSF-HiFiGAN 权重为 CC BY-NC-SA 4.0（见 NOTICE.md）；请注意 non-commercial 限制。

## 目录结构

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
