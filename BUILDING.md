# BUILDING

pc-nsf-hifigan 依赖均通过 FetchContent 获取，无需手动安装第三方库。

## 依赖

| 组件 | 来源 | 版本 |
|------|------|------|
| ggml | https://github.com/ggerganov/ggml | v0.11.0 (pin) |
| libmininsf | KakaruHayate/libmininsf | main |
| pocketfft | mreineck/pocketfft | cpp pin |
| dr_libs (dr_wav) | mackron/dr_libs | master pin |

> D2 决策：本仓库**不维护 ggml fork**，任何上游不接受的 patch 进入
> `KakaruHayate/ggml-patch`，由启用 CUDA 的消费者应用。CPU/F16 路径无需
> patch（见 docs/hifigan.md）。

## 步骤

```bash
# Release 构建
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D PCNSF_BUILD_CLI=ON
cmake --build build --config Release -j
# 二进制在 build/bin/
```

## 关键 CMake 选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `PCNSF_CUDA` | OFF | CUDA 后端 |
| `PCNSF_VULKAN` | OFF | Vulkan 后端 |
| `PCNSF_METAL` | OFF (Apple: ON) | Metal 后端（Apple 自动） |
| `PCNSF_BUILD_CLI` | ON | hifigan_cli |
| `PCNSF_BUILD_TESTS` | ON | golden 对比测试（需 Python3 + numpy） |
| `PCNSF_BUILD_EXAMPLES` | OFF | examples/external_consumer |
| `PCNSF_METAL` | Apple 上 ON | Metal 后端 |

## Vulkan / CUDA

库和可执行文件会自动检测并链接。运行需要：

- Vulkan：支持 Vulkan 的 GPU 驱动；超 1GB buffer 净需求默认能满足单块显存（本
  vocoder 峰值 << 1GB）。
- CUDA：本仓库自身不携带 ggml-patch 的 CUDA kernel；若启用 CUDA，先检查目标
  平台上 `ggml-patch` 的 conv_transpose_1d 修复是否已应用，否则结果可能不精确。

## 测试

```bash
cmake --build build --config Release -j
ctest --test-dir build         # 会先运行 gen_golden（numpy 参考）
```

- `t01/t02`：mel 前端 golden 对比
- `t03`：libmininsf source golden 对比
- `t04`：gguf_model 加载与 meta 访问器（tiny.gguf，需 gguf pkg）
- `t05_vocode`（可选，仅在配置时设 `PCNSF_MODEL_GGUF`）：真实模型端到端
  vocode 冒烟测试

CI 中仅构建 core + 运行 t01–t04。

## 生成模型

```bash
python converter/convert_hifigan.py --ckpt model.ckpt --config config.json \
    --out hifigan.gguf --dtype F16   # 默认（推荐）
# 或 debug/golden: --dtype F32
```

要求 `model.ckpt` 为 DiffSinger hifigan 结构（含 `generator.ups.*.weight`、
weight_norm 的 `weight_g`/`weight_v` 对会自动 materialize）。
