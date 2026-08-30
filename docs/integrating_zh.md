# 集成开发文档 (mel + f0 → wav)

面向把本仓库作为歌唱合成末段声码器接入的开发者：上游（DiffSinger 类推理管线或离线渲染器）产出**对数 mel 谱 + f0 基频**两个条件数组，本仓库把它声码为单声道波形。典型场景里帧数对齐、mel 提取与 f0 估计都由上游完成，本文只定义**本仓库的输入契约、模型参数与输出保证**，并给出 mel/f0 预处理的重合点提示。

本模型是 **Pitch Controllable NSF-HiFiGAN**（名称里的 `pc` 即 Pitch Controllable）：f0 输入**完全独立于 mel 所对应的原始音频基频**。你可以把任意 Hz 序列（例如按半音 / cent 平移后的 f0、或逐帧手动指定音高曲线）喂给模型，输出会严格跟随该 f0。这使本仓库天然适合做歌声变调、音高编辑等场景。

## 测量环境说明

本文引用的性能 / 精度数字除非另注，均基于以下测量平台：
- **CPU**：Xeon E5-2675 v3 @ 16 线程（AVX2）
- **GPU**：RTX 2070 (Turing)
- **工具链**：MSVC 2019 14.29 / Ninja
- **ggml 基线**：v0.19.0 + 本仓库四枚补丁（learned-ops / qvac / metal / vulkan-conv-direct-1d）
- **commit 锚定**：见 `docs/benchmarks.md` 各节引用（如 `f94c1d1`、`3ff3998` 等）

你的本地结果可能因微架构、驱动版本、线程调度而异。

## 1. 最小接入（C++，两步）

```cpp
#include "pc_nsf_hifigan/hifigan.h"

// 1) 模型加载一次，常驻
pc_nsf_hifigan::HifiganModel model("hifigan_f32.gguf", /*n_threads=*/16, "F32");

// 2) 每条推理：mel[T,128] 行主序 + f0[T] Hz
std::vector<float> wav;
pc_nsf_hifigan::hifigan_run(model, mel.data(), f0.data(), T, wav);
// wav.size() == T * 512，单声道 float32，44100 Hz
```

- **Server 式 API**：模型 load 一次、可反复 vocode；`hifigan_run` 逐条无状态（内部逐条重建 ggml graph）。连接方式：add_subdirectory 或 install/export（模板见 `examples/external_consumer/`）。
- **CLI**：单条 `hifigan_cli <gguf> <mel.bin> <f0.bin> <out.wav>`；批量 `hifigan_cli --batch <gguf> <list.txt> <outdir> [--warmup N]`（list.txt 每行 `mel.bin f0.bin 名字`）。

## 2. C# / .NET 接入

本仓库是纯 C++ 库，没有原生 C# 绑定。常见接入方式有三种：

### 2.1 C++/CLI 包装（推荐，延迟最低）

在 .NET 解决方案里加一个 C++/CLI 项目，引用本仓库的 `pc_nsf_hifigan` target，把 `HifiganModel` 与 `hifigan_run` 封装成 `ref class`。关键点：用 `pin_ptr<float>` 或 `GCHandle::AddrOfPinnedArray` 固定住托管数组首地址再传给 native 层，避免一次 copy。

```cpp
// Wrapper.h (C++/CLI)
#pragma once
#include "pc_nsf_hifigan/hifigan.h"

using namespace System;

namespace PcNsfHifigan {

public ref class HifiganModelWrapper {
private:
    std::unique_ptr<pc_nsf_hifigan::HifiganModel> model_;
public:
    HifiganModelWrapper(String^ ggufPath, int threads, String^ precision);
    array<float>^ Vocode(array<float>^ mel, array<float>^ f0);
    ~HifiganModelWrapper() { this->!HifiganModelWrapper(); }
    !HifiganModelWrapper() { model_.reset(); }
};

} // namespace PcNsfHifigan
```

```cpp
// Wrapper.cpp (C++/CLI)
#include "Wrapper.h"

namespace PcNsfHifigan {

HifiganModelWrapper::HifiganModelWrapper(String^ ggufPath, int threads, String^ precision) {
    msclr::interop::marshal_context ctx;
    model_ = std::make_unique<pc_nsf_hifigan::HifiganModel>(
        ctx.marshal_as<const char*>(ggufPath), threads,
        ctx.marshal_as<const char*>(precision));
}

array<float>^ HifiganModelWrapper::Vocode(array<float>^ mel, array<float>^ f0) {
    pin_ptr<float> pMel = &mel[0];
    pin_ptr<float> pF0  = &f0[0];
    int T = f0->Length;
    std::vector<float> wav;
    pc_nsf_hifigan::hifigan_run(*model_, pMel, pF0, T, wav);
    array<float>^ ret = gcnew array<float>((int)wav.size());
    Marshal::Copy(IntPtr(wav.data()), ret, 0, (int)wav.size());
    return ret;
}

} // namespace PcNsfHifigan
```

C# 侧调用：
```csharp
using PcNsfHifigan;

var model = new HifiganModelWrapper("hifigan_f32.gguf", 16, "F32");
float[] mel = ...; // [T, 128] row-major, ln/slaney
float[] f0 = ...;  // [T] Hz
float[] wav = model.Vocode(mel, f0);
```

### 2.2 P/Invoke（若不愿引入 C++/CLI）

如果下游只想用纯 C# 项目，可以给本仓库补一个薄 `extern "C"` 包装层（不提交回上游也没关系），然后 `DllImport`：

```c
#ifdef __cplusplus
extern "C" {
#endif

typedef void* hifigan_handle;
hifigan_handle hifigan_load(const char* gguf_path, int n_threads, const char* precision);
void hifigan_run_f32(hifigan_handle h, const float* mel, const float* f0, int T,
                     float* wav_out, int wav_cap, int* wav_len);
void hifigan_free(hifigan_handle h);

#ifdef __cplusplus
}
#endif
```

C# 侧把 `float[]` 固定后传入，或改用 `SafeBuffer` / `Span<float>`（.NET 5+）。

### 2.3 CLI 进程（最简，零胶水）

集成预算有限时，直接 `Process.Start` 调 `hifigan_cli.exe <gguf> <mel.bin> <f0.bin> <out.wav>`，再把 wav 读回。适合离线渲染、脚本链路；实时流式需要自己写 named-pipe / stdin 适配。

## 3. 输入契约与 Pitch Controllable 特性

本模型是 **Pitch Controllable NSF-HiFiGAN**（名称里的 `pc` 即 Pitch Controllable）。f0 输入**完全独立于 mel 所对应的原始音频基频**：你可以把任意 Hz 序列（例如按半音 / cent 平移后的 f0、或逐帧手动指定音高曲线）喂给模型，输出会严格跟随该 f0。

| 项 | 形状 / 布局 | 语义 | 说明 |
|---|---|---|---|
| `mel` | float32,行主序 **`[T, num_mels]`** | **自然对数** (ln) mel,clip ≥ 1e-5 | 必须是 DiffSinger NSF-HiFiGAN 前端的输出分布（slaney 刻度、ln；不是 log10，不是归一化） |
| `f0` | float32,`[T]` | **Hz**；可任意指定（Pitch Controllable） | 与 mel **逐帧同索引**；可以是原始基频、平移后半音、或任何你想要的曲线 |
| `T` | int(由 `f0` 长度决定) | 帧数 | `mel` 的行数必须等于 `f0` 的元素数，不等则报错（`mel frames != f0 frames`） |

- **帧数 / 时间已由上游处理的情况（主流场景）**：mel 与 f0 与各帧一帧对一个采样点 = hop 512 采样 ≈ 11.61 ms @ 44.1 kHz。若你的上游 mel/f0 帧率不同（如 10 ms hop 的 f0 提取器），需先按 vocoder 的帧栅（512 采样）把 f0 重采样对齐，使 `mel.shape[0] == f0.shape[0]`，否则会报长度错。
- **mel 必须是"自然对数 + slaney 刻度"**：本仓库自带的 `mel_nvstft()` 正好是 DiffSinger NSF-HiFiGAN 的前端拿法（reflect pad `(win-hop)//2`、`center=False`、slaney、ln）。若复用上游提取好的 mel，确认其对数底与刻度一致——**log10 vs ln 混用是本类集成最常见的错**。`MelExtractor`（通用生态 API）默认 htk，**不要**拿来喂本模型。
- **f0 取值范围**：模型训练覆盖约 **40–800 Hz**（验证管线用 librosa.pyin 此范围）。在此范围内变调效果自然；超出范围的极端值可能导致音质下降或出现金属声。
- **清音语义**：`f0 = 0.0` 的帧驱动 mini-nsf 走清音分支（noise 源），输出"哑"段；不要把静音帧填成平滑插值。

## 4. 模型参数（当前适配模型，GGUF 元数据一致）

| 参数 | 值 |
|---|---|
| sampling_rate | **44100 Hz** |
| hop_size（帧移） | **512**（≈ 11.61 ms / 帧） |
| n_mels | **128** |
| upsample_rates | [8, 8, 2, 2, 2]（总升采样 512） |
| upsample_kernels | [16, 16, 4, 4, 4] |
| upsample_initial_channel | 512 |
| num_resblocks | 15（= 5 个 upsample 段 × 3，ResBlock1 dilation [1,3,5]） |
| resblock_kernels | [3, 7, 11] |
| 上采样结构 | sub-pixel（相位主序，默认）；legacy convT 仅旧 GGUF 回退 |
| mini-nsf 源 | 内部由 f0 生成（fastsinegen）；`upp` = 64、`source_sr` = 5512.5 Hz |
| noise_sigma | 0（禁噪声） |
| lrelu_slope | 0.1（主干）；末层 conv_post 前 0.01（见 `hifigan.md` §1） |

音频侧约定（上游 mel 提取若复现）一致参数为 n_fft=2048 / win=2048 / hop=512 / n_mels=128 / fmin=40 / fmax=16000（详见 `verification_real_hifigan.md` §1）。

## 5. 输出契约与精度保证

- **wav**：单声道、float32（IEEE float WAV via dr_wav）、采样率 44100，长度 = `T * 512`；de-facto 波形近似在 `[-1, 1]`（tanh 头）。`HF_RAW_OUT` 环境变量可额外落一份裸 f32 数组用于比对。
- **精度**：fp32 主线下，引擎输出对 torch CPU golden 的 corr **0.9999998460**（max|Δ| 6.13e-04、rms 3.43e-05），且**位于两个已接收 EP（ORT CPU / DML）的合法 EP 噪声带内，双黄金帧下皆然**（详见 `benchmarks.md` §7）。这就是"可以放心替代 torch 末段"的精度合同。
- **无状态保证**：`hifigan_run` 内部逐条重建 graph、不持有跨条状态，线程安全前提是对同一 `HifiganModel` 只读、多个调用方各自申请返回缓冲（本项目内部如此使用）。

## 6. 运行时开关（环境变量）

| 变量 | 取值 | 默认 | 语义 |
|---|---|---|---|
| `HF_PRECISION` | `F32` / `F16` | `F32` | 精度线路；F16 = 权重 fp16、激活仍 fp32（为未来 fp16 训练试点预留） |
| `HF_THREADS` | int | 16 | CPU 线程数 |
| `HF_RAW_OUT` | 路径 | 关 | 额外把裸 f32 波形写到该路径（供 cmp 比对） |
| `PCNSF_BACKEND` | `cpu` | — | 强制 CPU 后端（默认 `ggml_backend_init_best()`）；进 debug / 非 GPU 环境用 |
| `GGML_VK_DISABLE_COOPMAT2` | `1` | 1（CLI 自动设） | Vulkan 上关张量核 fp16 舍入（fp32 合同）；留空又走库外则须自理 |

> 引擎级调试开关（`PCNSF_TIMING` / `PCNSF_PROFILE` / `PCNSF_FUSED_ADD` / `PCNSF_DIRECT_MIN_K` 等）面向引擎开发，不是输入契约，量产接入不应设。

## 7. 性能数量级（部署先有一个数感）

- **目标场景 fp32 主线**：`T=1722` / 20 s 音频，Vulkan 433–480 ms、CPU 16 线程 4484 ms（ORT CPU 级、比 ORT CPU 还略紧），CUDA 1576 ms（详见 `benchmarks.md` §1）。
- RTF：Vulkan ≈ 0.022–0.024、CUDA ≈ 0.079、CPU ≈ 0.22；**纯 CPU 即满足实时**（≈ 4.4× 实时）。
- **NOT 部署基线**：延迟敏感生产首选仍是 ONNX + DirectML（同机 281.6 ms）；本 ggml 定位 = 参考 / 纯推理实现 + 无 GPU / 边缘后端（§6 终态裁决：与 DML 的剩余 1.5× 差距已定，不再继续）。
- 冷启动 ~1.5 s（JIT/pipeline warm-up）；长会话或常驻进程下均摊可忽略。

## 8. 从源码构建与依赖获取

本仓库通过 CMake FetchContent 自动拉取两个外部依赖，clean checkout 无需手工克隆：

- **libmininsf**（mini-nsf 正弦源）：[KakaruHayate/libmininsf](https://github.com/KakaruHayate/libmininsf)
- **ggml-audio-patch**（ggml 补丁集 + 音频后端）：[KakaruHayate/ggml-audio-patch](https://github.com/KakaruHayate/ggml-audio-patch)

CMake 首次配置时会自动下载并应用四枚补丁（learned-ops / qvac / metal / vulkan-conv-direct-1d）。如果你需要修改依赖源码（例如调试 sine 生成器或 Vulkan shader），可以在 `cmake/Dependencies.cmake` 里把对应 `FetchContent_Declare` 的 `URL` 替换为本地路径，或设置 `FETCHCONTENT_SOURCE_DIR_LIBMININSF` / `FETCHCONTENT_SOURCE_DIR_GGML-AUDIO-PATCH` 环境变量指向 fork。

构建步骤与平台注意事项见 `BUILDING.md`（以及 patch 仓的 `docs/building_zh.md`）。

## 9. 常见坑（踩过）

1. **log10 vs ln 混用** → 看似小误差，实际 corr 掉到 ~0.99 以下、听感发浑。本模型是 ln。
2. **mel / f0 帧长不一致** → CLI 直接报长度错（`mel frames != f0 frames`）；库 API 同约。
3. **把清音帧 f0 填成平滑插值** → 失去清音分支，"哑"段变成无声。保持 0.0。
4. **拿 `MelExtractor` 当 DiffSinger 前端** → 刻度错（htk vs slaney）；只有 `mel_nvstft()` 与本模型对齐。
5. **fp16 当成 f16 激活** → 不存在。F16 线路权重 fp16、激活仍 fp32（ggml conv1d 要求）。
6. **Vulkan 上不设 `GGML_VK_DISABLE_COOPMAT2=1`** → 张量核 fp16 舍入污染 fp32 合同（corr ~0.999990 量级，见 `benchmarks.md` §8）。CLI 已自动设，库调用者须自理。
7. **f0 超出训练范围** → 40–800 Hz 外的值音质下降。变调尽量保持在此区间。
