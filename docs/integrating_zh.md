# 集成开发文档(mel + f0 → wav)

面向把本仓库作为歌唱合成末段声码器接入的开发者:上游(DiffSinger 类推理管线或离线渲染器)产出
**对数 mel 谱 + f0 基频**两个条件数组,本仓库把它声码为单声道波形。典型场景里帧数对齐、mel
提取与 f0 估计都由上游完成,本文只定义**本仓库的输入契约、模型参数与输出保证**,并给出 mel/f0
预处理的重合点提示。

阅读顺序建议:本页(集成契约)→ `hifigan.md`(内部架构)→ `benchmarks.md`(性能与精度论证链)。

## 1. 最小接入(两步)

```cpp
#include "pc_nsf_hifigan/hifigan.h"

// 1) 模型加载一次,常驻
pc_nsf_hifigan::HifiganModel model("hifigan_f32.gguf", /*n_threads=*/16, "F32");

// 2) 每条推理:mel[T,128] 行主序 + f0[T] Hz
std::vector<float> wav;
pc_nsf_hifigan::hifigan_run(model, mel.data(), f0.data(), T, wav);
// wav.size() == T * 512,单声道 float32,44100 Hz
```

- **Server 式 API**:模型 load 一次、可反复 vocode;`hifigan_run` 逐条无状态(内部逐条重建 ggml
  graph)。连接方式:add_subdirectory 或 install/export(模板见 `examples/external_consumer/`)。
- **CLI**:单条 `hifigan_cli <gguf> <mel.bin> <f0.bin> <out.wav>`;批量 `hifigan_cli --batch
  <gguf> <list.txt> <outdir> [--warmup N]`(list.txt 每行 `mel.bin f0.bin 名字`)。

## 2. 输入契约(关键节)

| 项 | 形状 / 布局 | 语义 | 说明 |
|---|---|---|---|
| `mel` | float32,行主序 **`[T, num_mels]`** | **自然对数** (ln) mel,clip ≥ 1e-5 | 必须是 DiffSinger NSF-HiFiGAN 前端的输出分布(slaney 刻度、ln;不是 log10,不是归一化) |
| `f0` | float32,`[T]` | **Hz**;**清音/休止帧填 0.0** | 每一帧一个基频值,与 mel **逐帧同索引** |
| `T` | int(由 `f0` 长度决定) | 帧数 | `mel` 的行数必须等于 `f0` 的元素数,不等则报错(`mel frames != f0 frames`) |

- **帧数 / 时间已由上游处理的情况(主流场景)**:mel 与 f0 与各帧一帧对一个采样点 = hop 512
  采样 ≈ 11.61 ms @ 44.1 kHz。若你的上游 mel/f0 帧率不同(如 10 ms hop 的 f0 提取器),需先按
  vocoder 的帧栅(512 采样)把 f0 重采样对齐,使 `mel.shape[0] == f0.shape[0]`,否则会报长度错。
- **mel 必须是"自然对数 + slaney 刻度"**:本仓库自带的 `mel_nvstft()` 正好是
  DiffSinger NSF-HiFiGAN 的前端拿法(reflect pad `(win-hop)//2`、`center=False`、slaney、ln)。
  若复用上游提取好的 mel,确认其对数底与刻度一致——**log10 vs ln 混用是本类集成最常见的错**。
  `MelExtractor`(通用生态 API)默认 htk,**不要**拿来喂本模型。
- **清音语义**:f0=0 的帧驱动 mini-nsf 走清音分支(noise 源),输出"哑"段;不要给 0 帧填伪基频。

## 3. 模型参数(当前适配模型,GGUF 元数据一致)

| 参数 | 值 |
|---|---|
| sampling_rate | **44100 Hz** |
| hop_size(帧移) | **512**(≈ 11.61 ms / 帧) |
| n_mels | **128** |
| upsample_rates | [8, 8, 2, 2, 2](总升采样 512) |
| upsample_kernels | [16, 16, 4, 4, 4] |
| upsample_initial_channel | 512 |
| num_resblocks | 15(= 5 upsample 段 × 3,ResBlock1 dilation [1,3,5]) |
| resblock_kernels | [3, 7, 11] |
| 上采样结构 | sub-pixel(相位主序,默认);legacy convT 仅旧 GGUF 回退 |
| mini-nsf 源 | 内部由 f0 生成(fastsinegen);`upp`=64、`source_sr`=5512.5 Hz |
| noise_sigma | 0(禁噪声) |
| lrelu_slope | 0.1(主干);末层 conv_post 前 0.01(见 `hifigan.md` §1) |

音频侧约定(上游 mel 提取若复现)一致参数为 n_fft=2048 / win=2048 / hop=512 / n_mels=128 /
fmin=40 / fmax=16000(详见 `verification_real_hifigan.md` §方法与输入)。

## 4. 输出契约与精度保证

- **wav**:单声道、float32(IEEE float WAV via dr_wav)、采样率 44100,长度 = `T * 512`;de-facto
  波形近似在 `[-1, 1]`(tanh 头)。`HF_RAW_OUT` 环境变量可额外落一份裸 f32 数组用于比对。
- **精度**:fp32 主线下,引擎输出对 torch CPU golden 的 corr **0.9999998460**(max|Δ| 6.13e-04、
  rms 3.43e-05),且**位于两个已接收 EP(ORT CPU / DML)的合法 EP 噪声带内,双黄金帧下皆然**
  (详见 `benchmarks.md` §7)。这就是"可以放心替代 torch 末段"的精度合同。
- **无状态保证**:`hifigan_run` 内部逐条重建 graph、不持有跨条状态,线程安全前提是对同一
  `HifiganModel` 只读、多个调用方各自申请返回缓冲(本项目内部如此使用)。

## 5. 运行时开关(环境变量)

| 变量 | 取值 | 默认 | 语义 |
|---|---|---|---|
| `HF_PRECISION` | `F32` / `F16` | `F32` | 精度线路;F16 = 权重 fp16、激活仍 fp32(为未来 fp16 训练试点预留) |
| `HF_THREADS` | int | 16 | CPU 线程数 |
| `HF_RAW_OUT` | 路径 | 关 | 额外把裸 f32 波形写到该路径(供 cmp 比对) |
| `PCNSF_BACKEND` | `cpu` | — | 强制 CPU 后端(默认 `ggml_backend_init_best()`);进 debug / 非 GPU 环境用 |
| `GGML_VK_DISABLE_COOPMAT2` | `1` | 1(CLI 自动设) | Vulkan 上关张量核 fp16 舍入(fp32 合同);留空又走库外则须自理 |

> 引擎级调试开关(`PCNSF_TIMING`/`PCNSF_PROFILE`/`PCNSF_FUSED_ADD`/`PCNSF_DIRECT_MIN_K` 等)
> 面向引擎开发,不是输入契约,量产接入不应设。

## 6. 性能数量级(部署先有一个数感)

- **目标场景 fp32 主线**:`T=1722` / 20 s 音频,Vulkan 433–480 ms、CPU 16 线程 4484 ms
  (ORT CPU 级、比 ORT CPU 还略紧),CUDA 1576 ms(详见 `benchmarks.md` §1)。
- RTF:Vulkan ≈ 0.022–0.024、CUDA ≈ 0.079、CPU ≈ 0.22;**纯 CPU 即满足实时**(≈ 4.4× 实时)。
- **NOT 部署基线**:延迟敏感生产首选仍是 ONNX + DirectML(同机 281.6 ms);本 ggml 定位 =
  参考 / 纯推理实现 + 无 GPU / 边缘后端(§6 终态裁决:与 DML 的剩余 1.5× 差距已定,不再继续)。
- 冷启动 ~1.5 s(JIT/pipeline warm-up);长会话或常驻进程下均摊可忽略。

## 7. 常见坑(踩过)

1. **log10 vs ln 混用** → 看似小误差,实际 corr 掉到 ~0.99 以下、听感发浑。本模型是 ln。
2. **mel / f0 帧长不一致** → CLI 直接报长度错(`mel frames != f0 frames`);库 API 同约。
3. **把清音帧 f0 填成平滑插值** → 失去清音分支,"哑"段变成无声。保持 0.0。
4. **拿 `MelExtractor` 当 DiffSinger 前端** → 刻度错(htk vs slaney);只有
   `mel_nvstft()` 与本模型对齐。
5. **fp16 当成 f16 激活** → 不存在。F16 线路权重 fp16、激活仍 fp32(ggml conv1d 要求)。
6. **Vulkan 上不设 `GGML_VK_DISABLE_COOPMAT2=1`** → 张量核 fp16 舍入污染 fp32 合同
   (corr ~0.999990 量级,见 `benchmarks.md` §8)。CLI 已自动设,库调用者须自理。
