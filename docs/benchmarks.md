# NSF-HiFiGAN 性能与精度测量（ggml CPU / Vulkan vs ONNX Runtime）

本文记录 mini_nsf 声码器在本仓库当前实现下的端到端测量。所有数字只声称“本机本次测量”，数字一律绑定源码状态与运行命令；音频用 20 s 参考输入（`T=1722` 帧 mel，输出 881 675 samples，与 torch golden 差 11 个样本的 convT K−s 对齐差，比较用 `cmp_align.py` 做 offset 对齐）。

- 测量提交：`f8c16ba`（`wip/profile-nodes`）。其后与本提交仅改文档/工具开关，不改变数字。
- 本机环境：Windows，Xeon E5-2675 v3（16C/32T，AVX2，无 AVX512），RTX 2070（驱动 32.0.16.2002），MSVC 2019（14.29），Python 3 / onnxruntime 1.23.0。
- ggml 源码基线：v0.19.0（`30bf868`）+ [ggml-audio-patch](https://github.com/KakaruHayate/ggml-audio-patch) 补丁一、二、四（Vulkan 直接卷积后端为补丁四）。
- 运行姿态：raw 模式（`HF_RAW_OUT=<out>.f32` 直写原始 f32 波形），fp32 全程；Vulkan 侧程序内默认置 `GGML_VK_DISABLE_COOPMAT2=1`（不走张量核）。

## 1. 端到端耗时

中位数原则：每条配置至少 3 次取中位（`PCNSF_TIMING=1`，stderr `[timing] hifigan_run`）。

| 实现 | 时间（中位） | 命令姿态 |
|---|---:|---|
| ggml CPU，原生 conv 路径（`PCNSF_DIRECT_CONV=0`） | 80 002 ms | `build-cpu` 姿态，见 §3 |
| ggml CPU，直接卷积、无生产侧融合（`PCNSF_FUSE_IO=0`） | 31 772 ms | 同上 |
| **ggml CPU，直接卷积 + 生产侧融合（默认）** | **23 651 ms**（中位，`HF_THREADS=24`） | 同上；旧窗 28 030 ms 的勘误见 §1.1 |
| ggml CPU 同上、`HF_THREADS` 扫描 | 见 §1.1 | T1→T24 全矩阵（本次会话重测）；线程数自 `wip/cpu-threads-default` 起默认 16 |
| **ggml Vulkan（补丁四，全卷积过 `supports_op`）** | **433.2–435.0 ms** | `build-vk` 姿态，见 §3；历史上限值 |
| ggml Vulkan，同二进制清理调试钩子后复测 | 458.4–478.8 ms | 输出与清理前逐位一致（MD5 同）、SPIR-V 未变；记入偶发整机负载噪声，本文采用 **≈430–480 ms** 区间话术 |
| ggml Vulkan，全部卷积强制回落 CPU（`PCNSF_DIRECT_MIN_K` 实验姿态） | 571–581 ms | 精度 0.99999956，验证回落路径正确 |
| ONNX Runtime CPU EP（fp32，`onnxruntime` 1.23.0） | 5 155.0 ms | `work/bench_ort.py`（资产不入库） |
| ONNX Runtime DML EP（fp32） | 325.8 ms | 同上 |

诚实陈述：CPU 侧相对原生 ggml 提速 2.85×，但仍落后 ORT CPU EP 约 5.4×；ORT 的线程化 GEMM 在该 Haswell 级 CPU 上调校得更好。差距定位于 level-0 `[881664, 256]` 级大激活的分块与并行划分，是后续工作。**勘误**：本文档/上游补丁文档旧版曾引用 10 038 / 5 973 / 4 764 ms 并称与 ORT CPU 持平——该记录对应一次陈旧/错误构建配置，不可复现；上表取而代之（同机同模型重测）。

### 1.1 CPU 线程扩展矩阵（同 binary 交错 A/B 重测，2026-08-30）

绑定：consumer `f8c16ba`；`build-cpu\bin\hifigan_cli.exe`（2026-08-29 22:58:57 构建，Release/AVX2/MSVC 14.29）；命令姿态 `HF_THREADS=<N>` + `PCNSF_TIMING=1`，每档 ≥3 跑取中位。

| HF_THREADS | 中位 (ms) | 相对 T1 | 效率 |
|---:|---:|---:|---:|
| 1 | 291 938 | 1.00× | 1.00 |
| 4 | 75 800 | 3.85× | 0.96 |
| 8 | 42 131 | 6.93× | 0.87 |
| 12 | 32 096 | 9.09× | 0.76 |
| 16 | 28 113 | 10.39× | 0.65 |
| 24 | 23 651（n=4：22 701.8 / 23 079.7 / 24 221.7 / 25 368.0） | 12.34× | 0.51 |

判读：12 线程附近进入访存墙；16 → 24 只再换 +19%（SMT 兄弟收益）。**自 `wip/cpu-threads-default` 起 `HF_THREADS` 默认 4 → 16**：默认姿态端到端从 ≈80 s 档直接进入 ≈28 s 档；低核数机器的超额线程订阅对本访存型负载基本无害。**勘误（§1 首表）**：旧行“28 030 ms（T24）”来自更早带调试钩子的窗口；同 binary 干净复测中位 23 651 ms（上表档内），原值按噪声上限保留记录、不再作为基准引用。

## 2. 精度（对 torch CPU golden，`cmp_align.py` offset 对齐）

| 实现 | corr | max\|Δ\| |
|---|---:|---:|
| ORT CPU EP | 0.9999999873 | — |
| ORT DML EP | 0.99999697 | 2.09e-03 |
| **ggml Vulkan（补丁四，验收锚点）** | **0.99999985** | **6.1314e-04** |
| ggml Vulkan 全回落 CPU 对照 | 0.99999956 | — |
| 门控反例：`PCNSF_DIRECT_MIN_K=1`（K=2 亚像素卷积被错误放行） | 0.363 | —（证明 `K ≥ 3` 门控有效） |

Vulkan 路径对 torch CPU golden 的贴合比 DML 紧一个数量级（max|Δ| 6.1e-4 对 2.1e-3）。判定命令：`python cmp_align.py <out>.f32`（工作目录为含 `golden_f32.bin` 的私有数据目录）。

## 3. 复现命令

构建：

```bash
# CPU（Release）
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --config Release -j

# Vulkan（ggml 源码需先按序应用 ggml-audio-patch 补丁一、二、四）
cmake -S . -B build-vk -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build-vk --config Release -j
```

ggml 补丁应用（仓库：ggml-audio-patch，基线 `30bf868`）：

```bash
git apply patches/learned-ops-ggml0190.patch            # 补丁一
git apply patches/qvac-ops-ggml0190.patch               # 补丁二
# git apply patches/metal-ops-ggml0190.patch            # 补丁三（Metal 可选，与补丁四正交）
git apply patches/vulkan-conv-direct-1d-ggml0190.patch  # 补丁四
```

运行（示意为 Windows cmd 风格；资产 `hifigan_f32.gguf` / `mel.bin` / `f0.bin` / `golden_f32.bin` / `cmp_align.py` 体积原因不入库）：

```bat
:: CPU 计时（线程数由 HF_THREADS 控制，默认 16）
set HF_THREADS=24
set PCNSF_TIMING=1
build-cpu\bin\hifigan_cli.exe hifigan_f32.gguf mel.bin f0.bin cpu.wav

:: Vulkan 原始波形 + 精度判定
set HF_RAW_OUT=vk.f32
build-vk\bin\hifigan_cli.exe hifigan_f32.gguf mel.bin f0.bin vk.wav
python cmp_align.py vk.f32
```

## 4. Vulkan 后端的正确性边界（重要）

补丁四的 shader 为隐式 GEMM stride-1 直接 1D 卷积，fp32 专用。共享内存 halo 上限决定门控为 **`K ≥ 3` 且 `(K−1)·dilation ≤ 72`**；其他组合 `supports_op` 返回 false，由调度器回落 CPU 内核（回落实测见 §1/§2 对应行）。

⚠️ 以 raw graph 模式直接调 `ggml_vk_build_graph` 的消费方会绕过调度器的 `supports_op` 检查，必须自行执行同一门控：本模型的 K=2 亚像素上采样卷积超出 shader 暂存行窗口，不加门控会静默产生错误输出（对照实验 corr 掉到 0.363）。pc-nsf-hifigan 用 `PCNSF_DIRECT_MIN_K` 实现，Vulkan 默认 3、其余后端默认 1。

## 5. 分辨率说明

- Vulkan side-by-side 验收另有逐阶 dump（offset=0、外部输入一致）于私有工作目录档案；公开数字以本表为准。
- 同一 Vulkan 二进制的清理调试钩子前/后输出 MD5 相同、SPIR-V 未变；458–479 ms 复测出现在清理后窗口，按整机状态噪声如实记录并给出区间，不混入 433–435 ms 的锚点测量。
- ORT baseline 由私有脚本 `bench_ort.py` 在本机产出，用于横向参照；模型权重与输入资产与前述私有数据目录一致。
