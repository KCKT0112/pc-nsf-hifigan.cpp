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
| legacy convT F32 | 0.00179 | 6.38e-05 | 0.9999994 |
| sub-pixel F32 | 0.237 | 0.0414 | 0.696 |

> 早期用小常量 mel 的 stage-0 冒烟可达 ~1e-3；但真实变长 mel/f0 全链路下
> **sub-pixel 路径数值明显错误（corr 0.696）**，不能用于发布。convT 路径
> 与 torch 高度一致（corr>0.9999），仍是当前可靠默认。

## 效率（本机，T=1722，约 19.99s 音频；PCNSF_TIMING 只计 `hifigan_run`）

| 后端 | 耗时 | RTF | 备注 |
|---|---|---|---|
| ggml CPU F32 sub-pixel（4 线程） | 54681 ms | ≈2.7x 慢于实时 | **CPU 性能不满足实时/加速** |
| ggml CPU F32 convT（4 线程） | 55096 ms | ≈2.8x 慢于实时 | 与 sub-pixel 相当，非 sub-pixel 特有 |
| torch CUDA（同参数参考） | 557 ms | 0.028（36x 实时） | CUDA 正常加速 |

CPU 与 torch CUDA 差距巨大；CPU 路径是当前最大瓶颈，需按 conv/resblock 逐层
profile（怀疑 conv1d 在 v0.19 CPU 上未并行 + 大中间 tensor 的
`ggml_cont`/im2col 开销）。

## Vulkan / CUDA 构建状态（本机受阻，非代码回归）

- **CUDA**：`build-cuda-v019`（`PCNSF_CUDA=ON`，VS16/MSVC v142 + CUDA 13.0）编译
  ggml-cuda 模板实例即失败：
  `nvcc fatal: A single input file is required for a non-link phase when an outputfile is specified`
  —— 已知的 MSVC v160 与 CUDA 13.0 MSBuild 集成不兼容（需 VS2022 v143 或
  Ninja/直调 nvcc）。此前报告同样结论，属环境工具链问题。
- **Vulkan**：`PCNSF_VULKAN=ON` configure 已找到
  `C:/VulkanSDK/1.4.350.0`，但 ggml-vulkan 的
  `add_custom_command(DEPFILE ...)` 在 **Visual Studio 16 2019** 生成器不受支持，
  configure 失败（需 Ninja 或更新 VS 生成器）。

## 结论 / 建议

1. **可靠默认仍是 legacy convT**：真实权重下与 torch corr>0.9999。
   仓库当前 converter 默认已改为 sub-pixel——这是**发布阻断项**，建议
   在修复 sub-pixel 数值前把 converter 默认切回 convT（或仅 F16 用 sub-pixel）。
2. **CPU 目前无法达到“加速”目标**（RTF≈2.7），下一步应 profile
   conv1d/resblock 与 `ggml_cont`，评估多线程与算子替换（如提前
   im2col+fp32 mul_mat）。
3. **CUDA/Vulkan 未在本机验证**：工具链阻碍明确，建议在带 VS2022/Ninja 的 CI
   或机器上补跑 E2E 数值与 RTF。

## 产物路径（本机，build 目录内）

- GGUF：`pc-nsf-hifigan.cpp/build-v019/ab/real_f32_sub.gguf`、`real_f16_sub.gguf`、`real_f32_convt.gguf`
- 输入：`real_mel.bin`、`real_f0.bin`
- 参考/输出 raw：`torch_real_ref.raw`、`real_cpu_f32.raw`、`real_cpu_convt.raw`
- 脚本：`prepare_inputs.py`、`torch_ref_real.py`、`torch_time.py`、`gen_convt.py`
