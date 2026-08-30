# 真实权重 + libmininsf 的端到端验证报告(历史基线 · 2026-08-30)

> **superseded**:本文件是 patch-1~4 落地**之前**的端到端验证快照,记录真实权重、真实 mel/f0 输入下本
> 引擎与 torch 参考的对贴方法。其"效率"章今已被 `benchmarks.md` §1 / §6 / §7 / §8 全面取代(那里才是
> 裁决后的速度/精度终态与论证链);此处只保留方法学价值。GGUF/输入资产不入库(见 §产物)。

## 1. 方法与输入

- 权重: DiffSinger NSF-HiFiGAN 44.1k / hop512 / 128-bin 权重(2025.02 训练,权重办法见 `NOTICE.md`,
  不入库)
- 参考管线: DiffSinger 推理栈 `inference/val_nsf_hifigan.py` 的 nvSTFT + f0 估计 + vocoder
- 音频输入: 44.1k 单声道 882 000 采样 = 20.0 s(裁剪自一条真实歌声)
- mel: nvSTFT log-mel(`n_fft=2048, win=2048, hop=512, bins=128, fmin=40, fmax=16000`,slaney、自然对数)→
  `[T=1722, 128]`
- f0: `librosa.pyin`(40–800 Hz,静音置 0)→ 1722 点、其中 1436 帧有声
- 对照: torch `Generator` 用**同一真实权重**对同一 mel/f0 前向,得 golden(881 664 采样 = 17 采样裁尾)

## 2. 精度(vs torch 参考,881 664 样本;引擎加载真实权重 + libmininsf 生成 source)

| 路径 | max\|Δ\| | rms | corr |
|---|---:|---:|---:|
| legacy convT F32(CPU,当时基线) | 1.79e-03 | 6.38e-05 | 0.9999994 |
| **sub-pixel F32(CPU,相位主序/tile-bias,今默认)** | 4.88e-04 | 6.38e-05 | 0.99999976 |
| **sub-pixel F32(Vulkan)** | 1.61e-02 | 2.60e-04 | 0.999990 |
| **sub-pixel F32(CUDA)** | 1.62e-02 | 2.72e-04 | 0.999989 |

> sub-pixel 早期全链路 corr 0.696 的根因是 **bias 复制错**——应 tile(`bias.repeat(s)`)而非
> `repeat_interleave`,修复后真实 20 s 输入下三后端 corr>0.9999。legacy convT 仅用于旧 GGUF 回退
> (见 `hifigan.md` §2)。**注意:本表 GPU 行是在 fp32 合同未被强制(张量核未被禁用)的构建上取得,
> 与现主线"Vulkan fp32 直卷积 + coopmat2 关"(见 `benchmarks.md` §1/§7)不可横比。**

## 3. 效率(当时快照;今已被 `benchmarks.md` §1 取代)

| 后端 | 耗时(T=1722,≈20 s 音频) | RTF |
|---|---:|---:|
| ggml CPU F32 convT(16 线程,旧 im2col,提速前) | 21 809 ms | ≈1.09x |
| **ggml CPU F32 sub-pixel(16 线程)** | 16 818 ms | ≈0.84x |
| **ggml Vulkan F32 sub-pixel(RTX 2070)** | 2 010 ms | ≈0.100 |
| **ggml CUDA F32 sub-pixel(RTX 2070)** | 1 576 ms | ≈0.079 |
| torch CUDA(同参数参考) | 557 ms | ≈0.028 |

当时的 ONNX 参考(同输入):ONNX+DML ≈ 340 ms / ONNX+CPU ≈ 5 187 ms。**今(`benchmarks.md` §1 重锚
后)同一引擎上 Vulkan 433–480 ms、CPU 4 484 ms**; ONNX+DML 重锚 281.6 ms(见 §6)。下一段的
"差距"与"下一步瞄准关闭 GPU 差距"导语以旧基线写成,已被 §6 终态裁决否定(刻意止步,不再继续)。

## 4. 当时的构建状态(CPU 通过 + 历史上 Vulkan/CUDA 各踩一次坑)

- **Vulkan**:VS 内置 Ninja 生成器(`-G Ninja`)+ 本地 FetchContent 可构建并 E2E 跑通(RTX 2070)。
- **CUDA**:曾踩 `nvcc "A single input file is required"` —— 真因是项目 MSVC 全局
  `add_compile_options(/utf-8)` 泄漏给 NVCC;修为 `$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>` 后
  `PCNSF_CUDA=ON` + `-DCMAKE_CUDA_ARCHITECTURES=75-real`(RTX 2070 只需 sm_75)构建成功并 E2E
  跑通。修复见提交 `0d71cf4`。**这两个构建开关仍可用,见 `BUILDING.md`。**

## 5. 当时的结论 / 建议(历史)

当时结论(仅存档):真实权重下 corr>0.9999;CUDA 当时效率最佳。**今已迭**:sub-pixel 上采样为默认
(convT 仅回退),Vocord 主战场在 `benchmarks.md` §1/§6(CPU/Vulkan 已赶超当时的"可部署后端",
且与 ONNX-DML 的差距已定性为不再继续);f16 权重路径未启用。

## 6. 产物

- GGUF、输入(`real_mel.bin` / `real_f0.bin`)、golden 与各后端输出 raw:均**不入库**;转换产物放置
  在 `build-*/` 私有构建目录,运行 `tests/` golden 重新生成;历史资产归档说明见根 README"模型发布"
  小节。
- 输入/参考产出脚本(`prepare_inputs.py`、`torch_ref_real.py`、…)保留在 `tests/` 历史快照,见
  `tests/README.md`。
