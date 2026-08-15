# 真实 hifigan 权重 + libmininsf 的 CPU 链路验证报告

- 日期：2026-08（本会话）
- 权重：`J:\0608\pc_nsf_hifigan_torch\pc_nsf_hifigan_44.1k_hop512_128bin_2025.02`
- 参考管线：`J:\0608\DiffSinger\inference\val_nsf_hifigan.py`（mel 前端 nvSTFT + f0 估计 + vocoder）

## 方法与输入

- 音频：`J:\0608\backend_matrix_outputs\input_segment.wav`（44.1k 单声道，882000 采样 = 20.0s）
- mel：nvSTFT log-mel（n_fft=2048, win=2048, hop=512, bins=128, fmin=40, fmax=16000）→ `real_mel.bin`（1722×128）
- f0：librosa.pyin（40–800Hz，静音置 0）→ `real_f0.bin`（1722 点，1436 帧有声）
- 对照：torch Generator 用**同一真实权重**对同一 mel/f0 前向，得 `torch_real_ref.raw`（881664 样本）

## 转换产物（pc-nsf-hifigan.cpp/build-v019/ab/）

| 文件 | 说明 |
|---|---|
| `real_f32_sub.gguf` | 新 sub-pixel（相位快）F32 |
| `real_f16_sub.gguf` | sub-pixel F16（权重 fp16） |
| `real_f32_convt.gguf` | legacy `ggml_conv_transpose_1d` F32（对照） |

## 精度（vs torch 参考，881664 样本；引擎加载真实权重 + libmininsf 生成 source）

| 路径 | max | rms | corr |
|---|---|---|---|
| legacy convT F32（CPU） | 0.00179 | 6.38e-05 | 0.9999994 |
| legacy convT F32（Vulkan） | 0.00983 | 2.40e-04 | 0.999991 |
| legacy convT F32（CUDA） | 0.01106 | 2.37e-04 | 0.999991 |
| sub-pixel F32（CPU） | 0.237 | 0.0414 | 0.696 |

> 早期用小常量 mel 的 stage-0 冒烟可达 ~1e-3；但真实变长 mel/f0 全链路下
> **sub-pixel 路径数值明显错误（corr 0.696）**，不能用于发布。convT 路径
> 与 torch 高度一致（corr>0.9999），仍是当前可靠默认。

## 效率（本机，T=1722，约 19.99s 音频；PCNSF_TIMING 只计 `hifigan_run`）

| 后端 | 耗时 | RTF | 备注 |
|---|---|---|---|
| ggml CPU F32 sub-pixel（4 线程） | 54681 ms | ≈2.7x 慢于实时 | **CPU 性能不满足实时/加速** |
| ggml CPU F32 convT（4 线程） | 55096 ms | ≈2.8x 慢于实时 | 与 sub-pixel 相当，非 sub-pixel 特有 |
| ggml CPU F32 convT（16 线程） | 21809 ms | ≈1.09x（勉强实时） | `HF_THREADS=16` 后约 2.4x 提升 |
| ggml CPU F16 convT（16 线程） | 22815 ms | ≈1.14x | 权重 fp16 但计算仍 fp32，无额外提速 |
| ggml Vulkan F32 convT（RTX 2070） | 18742 ms | ≈0.94x | Vulkan 后端比 16 线程 CPU 略快一点，仍远不及 torch CUDA |
| ggml CUDA F32 convT（RTX 2070） | 1378 ms | 0.069（≈14.5x 实时） | CUDA 后端真正可用，仍约为 torch CUDA 的 2.5x |
| torch CUDA（同参数参考） | 557 ms | 0.028（36x 实时） | CUDA 正常加速 |

> CLI 现在支持 `HF_THREADS`（默认 4）显式控制 ggml CPU 线程数。

CPU 与 torch CUDA 差距巨大；CPU 路径是当前最大瓶颈，需按 conv/resblock 逐层
profile（怀疑 conv1d 在 v0.19 CPU 上未并行 + 大中间 tensor 的
`ggml_cont`/im2col 开销）。

## Vulkan / CUDA 构建状态（本机均通过）

- **Vulkan（✅）**：VS 内置 **Ninja** 生成器（`-G Ninja`）+ 本地 FetchContent，
  `PCNSF_VULKAN=ON` 构建成功并 E2E 跑通（RTX 2070）。
- **CUDA（✅，已修复根因）**：`nvcc "A single input file is required"` 的真因是
  项目对 MSVC 全局 `add_compile_options(/utf-8)` 泄漏给 NVCC（裸 `/utf-8` 被
  当作第二个输入文件）。修复为 `$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>` 后，
  `PCNSF_CUDA=ON` + Ninja + `-DCMAKE_CUDA_ARCHITECTURES=75-real`
  （RTX 2070 只需 sm_75，单架构显著缩短编译）构建成功并 E2E 跑通
  （`build-cuda-ninja/bin/hifigan_cli.exe`）。提交 `0d71cf4`。

## 结论 / 建议

1. **可靠默认已是 legacy convT**：真实权重下与 torch corr>0.9999。
   本会话已把 converter 默认切回 convT（不再输出 `hifigan.upsub.*`）；
   sub-pixel 保留为实验中格式，修复后再启用。
2. **CPU 目前只到“勉强实时”**（16 线程 F32 21.8s/20s≈RTF 1.09）：已加
   `HF_THREADS` 控制线程数；要进一步需 profile
   conv1d/resblock 与 `ggml_cont`，评估多线程与算子替换（如预留
   im2col+fp32 mul_mat、避免每层大 cont copy）。
3. **三后端均已验证**：CPU/Vulkan/CUDA 精度 corr>0.9999；
   CUDA 效率最佳（RTF 0.069，≈14.5x 实时），Vulkan≈0.94x，CPU≈1.09x（16 线程）。
   CUDA 是当前可部署的加速后端。

## 产物路径（本机，build 目录内）

- GGUF：`pc-nsf-hifigan.cpp/build-v019/ab/real_f32_sub.gguf`、`real_f16_sub.gguf`、`real_f32_convt.gguf`
- 输入：`real_mel.bin`、`real_f0.bin`
- 参考/输出 raw：`torch_real_ref.raw`、`real_cpu_f32.raw`、`real_cpu_convt.raw`、`real_default_f32.raw`、`real_vk_f32.raw`、`real_cuda_f32.raw`
- 脚本：`prepare_inputs.py`、`torch_ref_real.py`、`torch_time.py`、`gen_convt.py`
