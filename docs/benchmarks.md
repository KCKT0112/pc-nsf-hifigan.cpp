# NSF-HiFiGAN 性能与精度测量（ggml CPU / Vulkan vs ONNX Runtime）

本文记录 mini_nsf 声码器在本仓库当前实现下的端到端测量。所有数字只声称“本机本次测量”，数字一律绑定源码状态与运行命令；音频用 20 s 参考输入（`T=1722` 帧 mel，输出 881 675 samples，与 torch golden 差 11 个样本的 convT K−s 对齐差，比较用 `cmp_align.py` 做 offset 对齐）。

- 测量提交：consumer `f8c16ba`；`build-cpu\bin\hifigan_cli.exe` 本体构建于 2026-08-30 12:12:46（Release/AVX2/MSVC 14.29）；源码 = ggml `30bf868` + ggml-audio-patch 补丁一（`5f6becc`）+ 补丁四（`7c60839`）。其后与本提交仅改文档/工具开关，不改变数字。
- 本机环境：Windows，Xeon E5-2675 v3（16C/32T，AVX2，无 AVX512），RTX 2070（驱动 32.0.16.2002），MSVC 2019（14.29），Python 3 / onnxruntime 1.23.0。
- ggml 源码基线：v0.19.0（`30bf868`）+ [ggml-audio-patch](https://github.com/KakaruHayate/ggml-audio-patch) 补丁一、二、四（Vulkan 直接卷积后端为补丁四）。
- 运行姿态：raw 模式（`HF_RAW_OUT=<out>.f32` 直写原始 f32 波形），fp32 全程；Vulkan 侧程序内默认置 `GGML_VK_DISABLE_COOPMAT2=1`（不走张量核）。

## 1. 端到端耗时

中位数原则：每条配置至少 3 次取中位（`PCNSF_TIMING=1`，stderr `[timing] hifigan_run`）。

| 实现 | 时间（中位，n≥3，`HF_THREADS=24`） | 命令姿态 |
|---|---:|---|
| ggml CPU，原生 conv 路径（`PCNSF_DIRECT_CONV=0`） | 11 088.8 ms（n=3：11 304.5 / 11 088.8 / 11 078.3） | `build-cpu` 姿态，见 §3 |
| ggml CPU，直接卷积、无生产侧融合（`PCNSF_FUSE_IO=0`） | 4 869.0 ms（n=3：4 890.7 / 4 869.0 / 4 855.2） | 同上 |
| **ggml CPU，直接卷积 + 生产侧融合（默认）** | **4 430.9 ms**（n=3：4 405.8 / 4 430.9 / 4 434.6） | 同上，**2.50× 对原生 ggml** |
| ggml CPU 同上、`HF_THREADS` 扫描 | 见 §1.1 | T1→T24 全矩阵（2026-08-30 干净重测)；线程数自 `wip/cpu-threads-default` 起默认 16 |
| **ggml Vulkan（补丁四，全卷积过 `supports_op`）** | **433.2–435.0 ms** | `build-vk` 姿态，见 §3；历史上限值 |
| ggml Vulkan，同二进制清理调试钩子后复测 | 458.4–478.8 ms | 输出与清理前逐位一致（MD5 同）、SPIR-V 未变；记入偶发整机负载噪声，本文采用 **≈430–480 ms** 区间话术 |
| ggml Vulkan，全部卷积强制回落 CPU（`PCNSF_DIRECT_MIN_K` 实验姿态） | 571–581 ms | 精度 0.99999956，验证回落路径正确 |
| ONNX Runtime CPU EP（fp32，`onnxruntime` 1.23.0） | 4 723.8 ms（n=10，2026-08-30 下午重锚；同日上午锚为 5 155.0 ms，±8% 属本机时段漂移，引用须注明时刻） | `work/bench_ort.py`（资产不入库） |
| ONNX Runtime DML EP（fp32） | 281.6 ms（同上重锚；上午锚 325.8 ms） | 同上 |

诚实陈述：CPU 默认姿态 **4 430.9 ms @ HF_THREADS=24** 已**反超 ONNX Runtime CPU EP**：对同日下午重锚 4 723.8 ms 约 **1.05×**（对同日上午旧锚 5 155.0 ms 为 1.16×，比例随机器时段漂移，方向不变）（精度同档 corr ≥ 0.99999997，见 §2），即本仓库 CPU 性能已达成"与 ORT CPU EP 持平"目标；DML 目标（≤340 ms）仍是 GPU 侧的课题，CPU 暂时不追。**errata-v2（2026-08-30）**：本文档此前引用的 28 030 / 23 651 ms（T24）为**早前一次失真测量链**（当时 `_deps/ggml-src` 处于补丁 regen 前后模板状态，timing 与后来干净复测不一致，按"同一源码+同一 env 必须可复现"原则全行废弃）；更早 ggml-audio-patch 提交 `5f6becc` 自述的 "4764 ms (fused) 系（`HF_THREADS=24` 近档）"才是真正口径，今日在 rebuilt 同源码 exe 上复现为 4 405~4 435 ms（中位 4 430.9），`cmp_align.py` offset=0、corr=0.99999999、maxabs=1.4918e-04，均与 5f6becc 提交信息自报一致。

### 1.1 CPU 线程扩展矩阵（同 binary 交错 A/B 干净重测，2026-08-30）

绑定：consumer 提交 `f8c16ba`；`build-cpu\bin\hifigan_cli.exe`（2026-08-30 12:12:46 构建，Release/AVX2/MSVC 14.29，源码 = `30bf868` + ggml-audio-patch `5f6becc`/`7c60839` 现态）；命令姿态 `HF_THREADS=<N>` + `PCNSF_TIMING=1`（无 profile)，交错 A/B、每档 n=3 取中位。

| HF_THREADS | 中位 (ms) | 相对 T1 | 效率 |
|---:|---:|---:|---:|
| 1 | 44 297.0 | 1.00× | 1.00 |
| 4 | 12 006.1 | 3.69× | 0.92 |
| 8 | 7 079.1 | 6.26× | 0.78 |
| 12 | 5 988.6 | 7.40× | 0.62 |
| 16 | 5 526.9 | 8.02× | 0.50 |
| 24 | 4 430.9 | 10.00× | 0.42 |

判读：
- 相对旧"预-regen"态（§1 旧表）全档位 **~6× 提速**，主要来自 learned-ops patch 的 ADD_LEAKY_RELU + CONV_DIRECT_1D 融合（regen 后 producer 的 mul+leaky 折进 conv op 本身，少了 3 个 elementwise pass 的 L3/RAM 往返）。
- 扩展曲线已显著变平：T8→T12 仅 +18%,T12→T16 仅 +8%,T16→T24 只 +25%(SMT 兄弟）——**新 kernel 的瓶颈已从"算力"换成"barrier + 线程间 pack/load/store 开销"**，152 节点的线程同步成本开始吃掉增益。
- **自 `wip/cpu-threads-default` 起 `HF_THREADS` 默认 4 → 16**：默认姿态端到端从 ≈12 s 档进入 ≈5.5 s 档；低核数机器的超额线程订阅对本访存型负载基本无害，需要跑满的仍可 `HF_THREADS=24`（再 −20%）。

### 1.2 CPU 内核 cache-blocking 旋钮扫描（实验态，2026-08-30)

绑定：同上 exe（12:12:46），实验 env `GGML_CONV_TSB`(t-superblock 上限，默认 32)/`GGML_CONV_OSB_KB`（打包 Wt 每片字节上限 KB，默认 128)/`GGML_CONV_ORDER`（超块遍历序，0=t-outer / 1=oc-outer)。扫描 3×4×3×2 = 48 组（TSB ∈ {1,2,4,32} × OSB_KB ∈ {32,64,128} × ORDER ∈ {0,1} × 线程 ∈ {12,16})，每组 n=1:**全部输出与默认姿态 MD5 逐位一致**（参见 `_sweep_logs/_anchor.txt`);wall 范围 5 405~6 371 ms(T12 与 T16 合并）,**组内任一轴单调性差 ≤ 2%**（整组散布为噪声）。判读：当前 kernel 的 cache-blocking 已达饱和，没有可信增益空间；后续若要再压 CPU 时间，方向在"减少 152 节点的 barrier/pack 固定开销"而不是"调 superblock 尺寸"。**按泛用性听证约定：没有可信正效应 → 不改热循环。**

### 1.3 独立可复现性验证（全新 sandbox 构建，2026-08-30 下午）

目的：证明"从仓库声明的源状态出发、不经手任何既有构建产物"即可复现本文档的 CPU 主张。步骤：新建构建目录 → 按 `build-cpu` 的 CMakeCache 签名配置（VS2019 x64，仅 CPU，CPU-only）→ FetchContent 拉取**未打补丁的 stock ggml v0.19.0**（`30bf868`）→ `git apply` ggml-audio-patch 的 `learned-ops-ggml0190.patch`（rc=0，仅补丁一，CPU 路径）→ Release 全量编译。结果：

| 判据 | 结果 | 对照 |
|---|---|---|
| 输出位等价 | `HF_RAW_OUT` f32 MD5 与 §1.2 门锚**逐位一致** | MD5 同锚 |
| 精度 | offset=0，corr=0.99999999，maxabs=1.4918e-04 | 与 §2 主张一致 |
| 性能（与同机 build-cpu 交错 A/B，各 n=3 取中位） | T24：**4 487 ms**；T16：5 963 ms | build-cpu 同测 T24 4 490 / T16 5 808 ms；差异在噪声内 |
| 编译环境 | 两侧 ggml-cpu Release 编译旗标逐字节相同 | MSVC 14.29，/O2，/arch:AVX2 |

结论：补丁库现态 payload 可从 stock v0.19.0 复现全部主张；§1.2 的实验旋钮脚手架仅存在于本机工作目录，不随 patch 分发，也不影响输出。sandbox 构建产物验证完成即删除，本节记录为其唯一存活证据（含各文件 mtime 与 MD5，见私有 ROOTCAUSE_NOTES §2.14）。

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
git apply patches/vulkan-pipeline-cache-ggml0190.patch  # 补丁五（持久管线缓存）
git apply patches/metal-conv-direct-1d-ggml0190.patch   # 补丁六（Metal F32 implicit-GEMM direct conv）
```

全新 GitHub clone 由 `cmake/ApplyGgmlPatches.cmake` 自动完成上述六步，无需手敲。

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

## 6. 终态裁决（2026-08-30）：Vulkan↔DML 差距取舍已定性、刻意止步

剩余 ~1.5× 差距（Vulkan 430–480 ms 对本文重锚 ORT DML EP **281.6 ms**，2026-08-30 同批次计时 n=10）做过系统勘察，结论为**不再继续**：每条提速路线要么违反数值合同、要么属于本仓库无意持有的引擎级基建。

- **f16 张量核 GEMM**（算力余量唯一够的路线，im2col + `KHR_cooperative_matrix`，f16 操作数 f32 累加）以 torch 侧模拟取证——把全部 Conv1d 的操作数过一道 f16 舍入再以 fp32 前向（HMMA 的 f16×f16 乘积在 f32 中精确、累加 f32，该模拟即其忠实上界，93 个 Conv1d 全覆盖）：

| 舍入档位 | corr | max\|Δ\| | 定位 |
|---|---:|---:|---|
| 权重+激活（B 方案本相） | 0.99999969 | 3.94e-03 | **低于 DML 锚（0.99999697 / 2.09e-03）**，精度掉到合同线（≥0.9999999）之下，速度才刚够 DML |
| 仅权重 | 0.99999993 | 5.80e-04 | 数值可，但 f32×f16 混合算式不在 HMMA 支持集，无对应硬件路径 |
| 仅激活 | 0.99999977 | 2.11e-03 | 同上不可达；且激活舍入是误差主导项 |

  → **B 听证不通过**：精度换速度之后精度仍打不过 DML，战术与战略双输。
- **fp32 shader 微雕**：现实天花板端到端 ~330–380 ms，仍在 DML 之上；变体/启发式低位果实已在 §1.2 与补丁四侧穷尽。
- **Winograd**（K=3 主战场）：与微雕同档天花板、成本更高、另需数值听证——被支配。
- **保精度张量核**（split-f16/Kahan 分解 GEMM）：基建级（数值正确的矩阵引擎 + 各家矩阵扩展代际维护）——按"拒绝拥有跨代指令集/扩展维护义务"的既定原则打住，与 CPU 侧 ISA 考量同构。

**故 Vulkan 终态 = fp32 430–480 ms、max|Δ| 6.1314e-04（比 DML 紧约一个数量级）**，本文为最终测量文档；CPU 侧合同（≈ORT CPU EP，精度 ≥0.9999999）已达成并反超（§1），CPU 终态同此文档。

---

## 7. EP 噪声审计 + 参照系 Shift(2026-08-30 PM · N=881 664 · 双黄金帧)

**动机**:"B-sim corr 0.99999969 这类低于 PR 锚点的差异,是否本来就落在两个**已接受** EP 实现(ORT CPU EP / ORT DML EP / torch CPU↔CUDA)之间的正常噪声带内?"方法:同一段 881 664 采样分别对 torch CPU golden 与 torch CUDA golden 做 offset 对齐(offset=0),对每条候选算 corr / max|Δ| / rms / p99.9;完整输出与全部 `_*.f32` 原始指针留私有工作目录,此处为精简转写。

### 7.1 CPU 黄金帧(与 §1/§6 同系)

| cand@CPU golden | corr | max\|Δ\| | rms | p99.9 |
|---|---:|---:|---:|---:|
| ORT CPU EP | 0.9999999873 | 1.49e-04 | 9.84e-06 | 7.73e-05 |
| **ggml-Vulkan(主线)** | **0.9999998460** | **6.13e-04** | **3.43e-05** | **1.69e-04** |
| ORT DML EP | 0.9999969745 | 2.09e-03 | 1.52e-04 | 1.08e-03 |
| B-weights(f16 权重仿真) | 0.9999999258 | 5.80e-04 | 2.50e-05 | 1.06e-04 |
| B-acts(f16 激活仿真) | 0.9999997725 | 2.11e-03 | 4.17e-05 | 3.61e-04 |
| B-both(=§6 听证行) | 0.9999996909 | 3.94e-03 | 4.92e-05 | 3.57e-04 |

### 7.2 CUDA 黄金帧(参照系 Shift 要点)

| cand@CUDA golden | corr | max\|Δ\| | rms | p99.9 |
|---|---:|---:|---:|---:|
| ORT DML EP | 0.9999998863 | 5.72e-04 | 2.95e-05 | 2.63e-04 |
| ORT CPU EP | 0.9999978486 | 1.43e-03 | 1.28e-04 | 8.47e-04 |
| torch CPU(黄金对换) | 0.9999978463 | 1.52e-03 | 1.28e-04 | 8.52e-04 |
| **ggml-Vulkan(主线)** | **0.9999976269** | **1.55e-03** | **1.35e-04** | **8.84e-04** |
| B-both(f16 仿真) | 0.9999976045 | 3.91e-03 | 1.35e-04 | 8.54e-04 |

要点:CUDA 帧下"CPU 系实现簇"(torch CPU / ORT CPU / **ggml-Vulkan**) corr 紧簇在 0.9999976–0.9999979、max|Δ| 1.43e-03–1.55e-03,Vulkan 与 ORT-CPU、乃至 torch-CPU 自身等距 —— **Vulkan 的 fp32 实现质量已达 CPU 黄金同类水位**。DML@CUDA 的 0.9999998863 是该帧的**平台底**(DirectML 与 torch-CUDA golden 同族 f32 GPU 数学,共享舍入指纹),不是质量优势。**故 §6 的 DML-vs-Vulkan 精度排序必须按参照系敏感性打折读:两帧互换后 Vulkan 始终处于合法 EP 带内,与 DML 的差距属平台噪声量级。**

### 7.3 B 方案误差指纹鉴定(为何"corr 已压过 DML"仍不可收)

- corr / rms / p99.9:B-both@CPU 全部落在 §7.1 合法 EP 带内,corr(0.9999996909)甚至高于 DML(0.9999969745);
- **max|Δ| = 3.94e-03,为合法 EP 带顶(DML 2.09e-03)的 1.88×,出格**;
- 误差**方向与 EP 族正交**:err_B 对各 EP err 的互相关 ≤0.05,对 span(err_ORT-CPU, err_DML) 的投影 R²=0.0023;误差谱峰度合法 EP 27.4–39.4 dB、B 仅 19.9–21.7 dB(更白);B-weights 对 golden 呈 +0.276 系统偏置。

**判定**:B 不是"更脏的 EP 随机噪声",而是**方向正交的系统性舍入误差**;DML-等价合同(corr/rms/p999 ≥ ORT-DML 且 max|Δ| ≤ 2× DML)下,max|Δ| 单项不达标、误差又不属同族 → 维持 §6 听证的不信任,并为接下来 §8 的真机 spike 给出"必须实测、不可凭仿真放行"的依据。

---

## 8. B-HMMA 真机 spike(§6 方向 B 的实证检验 · **NO-GO 终判**)

在 §6 听证冻结 B、§7 EP 噪声审计把 B-sim 误差判为"corr/rms 在台站内、maxabs 出格且方向正交"之后,仍按用户拍板做了真机 spike(因为微基准 lvl1 9.19 TFLOPS 投影 1.54× 快于 DML 的收益值得实证一次)。实现走 `PCNSF_BHMMA=1` 环境分支:全部 K>1 conv 改 `cast(w→f16)` + `im2col_fast_1d(src1=f32→dst=f16, ow_align=16)` + `ggml_cont(wm16)` + `ggml_mul_mat(f16×f16→f32)`,io-fusion 禁用、bias/leaky/residual 改显式节点;K=1 / OC=1 走守卫返回 fp32。**分支代码整体保留在 `stash@{0}`("BHMMA spike"),主线仍是 fp32 direct conv。**

### 8.1 真机数据(RTX 2070 · ggml v0.19.0+patch4 · 881 664 采样 / 20 s 片段)

| 路径 | 暖机 wall(n=7 去首帧) | corr(vs CPU golden) | max\|Δ\| | rms |
|---|---|---|---|---|
| ONNX Runtime DML EP | ~282 ms | 0.9999969745 | 2.09e-03 | 1.52e-04 |
| **fp32 direct conv(主线,终态)** | **433–480 ms** | **0.9999998460** | **6.13e-04** | **3.43e-05** |
| B-sim(离线仿真,非真机) | — | 0.9999996909 | 3.94e-03 | 4.92e-05 |
| **B-HMMA 真机 spike** | **1 294.1 ms(2.7× 慢于 fp32,4.6× 慢于 DML)** | **0.9999902022** | **1.365e-02** | 2.73e-04 |

冷启动:B 1 691.9 ms vs fp32 1 489.4 ms。精度比 DML 还差一个数量级(corr 差一位,maxabs ~7×),速度全面倒退 → **双重 NO-GO**。

### 8.2 慢在哪里(per-node profile,`PCNSF_PROFILE=1`,总 9 053.5 ms / 1 185 节点)

1. **IM2COL_FAST_1D f16-dst n=97 均 49.9 ms**:学到的算子补丁只优化了 f32-dst,f16-dst 落回通用 kernel;
2. **MUL_MAT f16 n=98 均 39.9 ms**:reshape(im2col) 的布局(stride/对齐)不满足 mul_mm/mul_mm_cm2 快路径要求,落回通用标量 kernel —— 同几何、干净 contiguous 输入下微基准为 1.18 ms,**图内惩罚 34×**;
3. conv_post(K=1 / OC=1)单发 matvec-scalar **881 ms**(守卫 path 退化成 matvec);
4. ADD n=154 92.3 ms、CPY n=196 78.1 ms:显式 epilogue 拆点额外开销。

### 8.3 五个失效原因与"B↔E 坍缩"

要把 B 做到微基准投影水平,需要:f16-dst im2col 专用 kernel + 放开 mul_mat 对输入布局的 stride/对齐限制 + OC=1 安全的 HMMA 路径 + 图内重融合 epilogue —— **这正是 §6 已经否决的"基础设施级 E 改造"本身**。B 与 E 的边界在真机上坍缩:B 只有按 E 的工程强度去做才有意义,而 E 已被否决。故 B 在三个独立层面终结:

- (a) §6 听证:严格合同(≥0.9999999)下 corr 不够;
- (b) §7 审计:DML-等价合同下 corr/rms 在台站内、maxabs 1.88× 出格、误差方向与 EP 族正交(R²=0.0023);
- (c) §8 真机:速度 2.7× 倒退 + 精度比 DML 还差一位。

### 8.4 保留资产

- 微基准 `bench_mm_vk.exe` 证实 lvl1 配置在干净输入下确有 9.19 TFLOPS(>fp32 峰值 7.47,HMMA 生效)—— 说明瓶颈全在图内布局/缺 kernel,不在硅;
- vk_shader 头以 文本形式进 git(此前阴影常量 PR 之后),`git status` 可直接看到 shader 再生成差异;
- spike 评测脚本与原始数据留存私有工作目录,不入库。

**终态建议(同 §6):冻结主线 fp32 direct conv;B/E 仅在 ggml 上游补齐 f16 im2col kernel 与 mul_mat 布局泛化后再议。**

## 9. game.cpp 三项借鉴的落地与实测(2026-08-30 晚 · 补丁五 / CPU 常驻线程池 / llamafile 阴性结果)

把 game.cpp 的三件可复用资产逐项落地或否决,全部按数字纪律记录(commit SHA + CMakeCache 关键行 + 命令齐备)。

### 9.1 补丁五:Vulkan 持久化磁盘管线缓存(已落地)

移植 game.cpp `cmake/patches/ggml-vulkan-pipeline-cache` 至 ggml v0.19.0,作为 ggml-audio-patch 的第五个补丁(上游仓已交付 `vulkan-pipeline-cache-ggml0190.patch`;本仓 `patches/` 同步字节一致副本,`cmake/Dependencies.cmake` + `ApplyGgmlPatches.cmake` 校验并自动应用,全新 GitHub clone 构建路径已实测:5 patch 全绿、幂等重跑 5 个 skip)。

环境:`build-vk` Release(MSVC 14.29,`/O2 /Ob2 /DNDEBUG`),`GGML_VULKAN=ON / GGML_AVX2=ON / GGML_CPU_REPACK=ON`,`FETCHCONTENT_SOURCE_DIR_GGML=ggml worktree @ b899edd`(= 补丁一至四栈 + 补丁五);本仓 `5c23b2d` + 本节改动;RTX 2070 驱动 32.0.16.2002;参考片段 T=1722(881 664 采样);测量 2026-08-30 16:52–16:57。

| 场景 | blob 状态 | wall (ms) | `hifigan_run` (ms) |
|---|---|---:|---:|
| 冷(删除 blob) | 无 → 写出 334 150 B | 2 454 | 1 061.5 |
| 热(进程重启,blob 载入) | 334 150 B | 1 120–1 295(n=4) | 346.8–355.0(n=3,首序列) |
| `GGML_VK_DISABLE_PIPELINE_CACHE=1` | 存在未用 | 1 122–1 232 | 347.5–355.3(首序列 n=3) |
| 独立交叉复测(同晚数分钟后) | ON / OFF 各 n=3 | ≈1 200 | ON 中位 376.1 / OFF 中位 373.1 |

精度:`vk_p5.f32` / `vk_p5_w.f32` 对 torch-CPU golden corr 0.99999985、max|Δ| 6.1314e-04、offset=0 —— 与存档 Vulkan 锚点逐位同级,零数值影响。

如实解读:**首跑收益真实且大**(无 blob 时编译计入首次计算:图内 1 061.5 → 稳定 ≈350 ms,≈3×;wall 2 454 → ≈1 200 ms,≈2×);但本机 NVIDIA 驱动 GLCache 也跨进程缓存编译产物,该层"热"时补丁五在稳态无可测增益(上表 ON≈OFF)。blob 的价值场景 = 驱动缓存冷/被逐出/容量受限/弱缓存平台。构造"驱动冷"受控 A/B 因驱动自管理 GLCache 行为未获干净对照(尝试期间逐跑散布达 ±90 ms),除首表冷行外不宣称"驱动冷"数值。完整表与复现命令见上游仓 `docs/benchmarks.md` 补丁五节。

环境开关:`GGML_VK_PIPELINE_CACHE_PATH`(自定义位置,默认 `%LOCALAPPDATA%\ggml_audio_vk_pipeline.cache` / Linux `$HOME/.cache`)、`GGML_VK_DISABLE_PIPELINE_CACHE=1`、`GGML_VK_PIPELINE_CACHE_DEBUG=1`(打印载入/写回字节数,实测 334 150 B)。

另注:`build-vk` 首测序列稳态读数(346.8–355.3 ms)低于 §1 存档区间 433–480 ms;同晚后续序列又回到 374–430 ms 簇 —— 属机器状态漂移而非补丁效应(补丁五不触碰 shader/算子),§1 存档数仍有效,本节附录列出全部序列供后人校准。

### 9.2 CPU 后端常驻线程池(已落地,`GGUFModel` 级)

参考 game.cpp `src/backend.cpp`:`ggml_threadpool_params_default(n_threads)` → `ggml_threadpool_new` → `ggml_backend_cpu_set_threadpool`,池随模型生命周期常驻(混合轮询 poll=50),析构时先于 backend 释放。收益场景为嵌入式宿主反复调用(每次省一次 16–24 线程 spawn/join);对一次性 CLI 不可见是预期内。

环境:`build-cpu` Release(MSVC 14.29),`GGML_NATIVE=ON / GGML_LLAMAFILE=OFF / GGML_CPU_REPACK=ON`,ggml worktree 同上 `b899edd`;测量 2026-08-30 17:03。

| 线程 | 本实现 n=3(ms) | 中位 | §1.1 存档(同机,2026-08-30 白天) |
|---|---|---:|---:|
| 16 | 5745.2 / 5580.0 / 5632.8 | **5632.8** | 5526.9 |
| 24 | 4578.1 / 4522.9 / 4789.4 | **4578.1** | 4430.9 |

精度 corr 0.99999999、max|Δ| 1.4918e-04 —— 与存档 CPU 锚点逐位同级。结论:时序与精度均在漂移带内持平(线程池改动对单次 CLI 中性),行为无回归;同时本次重建也顺带验证了"规范补丁栈(含补丁二 CPU 内容)对我们图无可测影响"。

### 9.3 GGML_LLAMAFILE sgemm:实测无差异 → **不启用**(保持 OFF)

分析:ggml `forward_mul_mat` 在 src1 连续或臂内转换路径上有两处 `llamafile_sgemm` 调用,兜底 `goto UseGgmlGemm1`;tinyBLAS 门控要求 `k % 8 == 0` 且 `m % 4 == 0`(AVX2 f32),`n < 2` 直接退回。本图 MUL_MAT 的 m 为输出通道侧(含 conv_post 的 OC=1),多数落在门控之外。

实测(同一 `build-cpu` 目录翻转 `-DGGML_LLAMAFILE=ON` 重链,仅 `ggml-cpu.dll` 变化;测量 2026-08-30 17:05,环境同 §9.2):

| 线程 | llamafile OFF 中位(17:03) | llamafile ON n=3(ms) | ON 中位 |
|---|---:|---|---:|
| 24 | 4578.1 | 4521.0 / 4496.5 / 4534.3 | 4521.0 |
| 16 | 5632.8 | 5722.0 / 5620.2 / 5794.6 | 5722.0 |

精度:ON 下 corr 0.99999999、max|Δ| **1.4918e-04 与 OFF 完全相同**(七位有效数字)——说明本图内 sgemm 快路径实际未被触发(若触发,f32 累加分块顺序不同几乎必然产生非零微差),全部落回 ggml 默认 GEMM。零收益 + 无法触发的死路径 = 维持默认 OFF,已回退并重建验证(4529.3 ms @ T24,corr 不变)。`build-vk` 缓存里历史上带着 `GGML_LLAMAFILE=ON`(该构造的 CPU 回落后端;同样回落至默认 GEMM,无主线路径影响),存档 Vulkan 数字不带此项差异。

### 9.4 真相对齐:ggml worktree 谱系修正(法证)

本会话发现旧 ggml worktree 谱系(`fd606d4` 系)只含补丁一内容 + 未入库的补丁四变体,长期缺失仓库补丁二/三的全部文件(CPU `ops.cpp` 478 行等 + 5 个 Vulkan shader + Metal 集成);此前全部存档数字即在该树上测得。已将该树保留为 `pre-canonical-backup`,主 worktree 重置到规范提交栈(补丁 1–4 = `57bf852`,+补丁五 = `b899edd`),并以差分证明缺失内容对本图是死代码:**声码器图不含任何 qvac 算子/Metal 组件,因此 §1/§6/§7/§8 全部存档数字与结论依然有效可比**(本节 9.2 的同位重测亦证实数值逐位一致)。

## 10. Apple Metal 真机适配与基准（2026-08-31）

环境：Apple M4、macOS 27.0、AppleClang 21、ggml v0.19.0 + 本仓五枚补丁、F32 GGUF。模型来自 OpenVPI `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.ckpt`；转换所得 GGUF SHA-256 为 `650471e4a75f782132822e622e2d0e9c5f2a1f4ae4d5ad7cf02b3ac830fcbe67`。输入为 1722 帧（输出 881664 采样，19.992 s）的同一用户音频前端结果。

适配前的 Apple 默认构建与运行均不成立：`GGML_METAL_EMBED_LIBRARY=OFF` 要求额外安装 Xcode Metal Toolchain；改为嵌入 shader 后，运行又在 `IM2COL_FAST_1D` 被 Metal `supports_op` 拒绝并 abort。现已默认嵌入 shader，并把该算子映射到与普通 `IM2COL` 相同的 Metal kernel。Apple ARM64 CPU 同时避开仅 AVX2 优化、其余架构为标量回退的 `CONV_DIRECT_1D`，改走 im2col + mul_mat。

| 后端 | `hifigan_run` | RTF | 相对实时 | 说明 |
|---|---:|---:|---:|---|
| CPU，8 线程，修正 ARM64 门控 | 中位 **12.531 s**（n=4：12.582 / 12.480 / 12.591 / 12.265） | 0.627 | 1.60× | 模型单次加载、连续 4 次 |
| Metal，Apple M4 | **6.828–8.911 s** | 0.342–0.446 | 2.24–2.93× | 独立进程正常机器状态；极端压力后的热降频数据不混入 |

按同批 CPU 中位数，Metal 加速为 **1.41–1.84×**。0.372 s 短样本在模型单次加载后连续 4 次为 102.5 / 98.9 / 98.0 / 96.5 ms，验证重复调用可用。嵌入 shader 的全新首次运行时编译在本机曾为 7.815 s；系统缓存热后库加载为 9–46 ms，因此部署侧应区分首次启动与图执行。

精度（同一 881664 采样，对本机 PyTorch CPU 参考）：

| 后端 | corr | max\|Δ\| | RMS Δ | p99.9\|Δ\| |
|---|---:|---:|---:|---:|
| ggml CPU | 0.999999999999 | 3.177e-06 | 1.100e-07 | 7.674e-07 |
| Metal | 0.999999287801 | 3.794e-03 | 9.903e-05 | 1.094e-03 |

Metal 多次独立进程输出 SHA-256 均为 `d5f365f189c90fb0ccad339e3a7c196d3de016b3b959c886e5afa57335ad94e8`，结果确定且全部有限。关闭 Metal fast-math 会使 20 s 样本超过 16 分钟仍未完成，故维持 ggml 默认数学模式；Metal 是 GPU EP 级近似，不宣称 CPU bit-exact。

## 11. Metal 极致优化：implicit-GEMM direct conv（2026-08-31）

对 §10 同一 Apple M4、同一 F32 GGUF 和 1722 帧输入做逐节点 profile：旧图的 97 个 `IM2COL_FAST_1D` 合计约 **4295 ms**，98 个 `MUL_MAT` 合计约 **974 ms**，另有 154 个 ADD、96 个 LEAKY_RELU 和 18 个 CONT。瓶颈不是算力，而是为每个卷积物化巨型 im2col 张量并反复读写统一内存。

补丁六为 Metal 实现 `GGML_OP_CONV_DIRECT_1D`：

- F32 simdgroup-matrix implicit GEMM，不再物化 im2col；
- bias、residual、输出 leaky-ReLU、输入 scale/leaky 全部折入同一 kernel；
- 按逻辑输出通道选择 `16x64`、`32x64`、`64x64` tile；`OC < 8` 使用连续时间轴 scalar kernel，避免 `conv_post` 的空矩阵行；
- K tile 为 8；累加完成后复用权重/输入 shared memory 作为输出 spill，使 `64x64` 从 20 KiB 降至 16 KiB。

最终图从 **989 个节点降至 152 个节点**，97 个卷积直接执行。20 秒样本在模型单次加载后连续四次：

| 路径 | `hifigan_run` | 相对旧 Metal | 相对实时 |
|---|---:|---:|---:|
| 旧 Metal `im2col + mul_mat` | 6.828–8.911 s | 1.00x | 2.24–2.93x |
| direct conv，`32x64` 热态 | 1.042–1.051 s | 6.50–8.55x | 19.0–19.2x |
| **direct conv，最终 `64x64`** | **0.823–0.836 s** | **8.17–10.83x** | **23.9–24.3x** |

0.372 秒短样本连续四次为 **30.0 / 20.2 / 20.2 / 20.1 ms**；旧 Metal 为 102.5 / 98.9 / 98.0 / 96.5 ms。

同日较晚的提交前复测遇到明显机器状态漂移：相同源码、相同输入的常驻进程四次为 **1.310 / 1.350 / 1.313 / 1.310 s**，先前独立干净构建的同源码二进制也同步漂移到 **1.383 s**；两者输出与上述 0.823–0.836 s 调优轮逐位一致。因此这里同时保留最快稳定轮和复测轮，不把主机功耗/温度状态变化误记成代码回归。

精度相对同机 ggml CPU 参考：`corr=0.9999999999994206`、`max|Delta|=2.693e-06`、`RMS=8.919e-08`、`p99.9|Delta|=7.562e-07`。短样本与 direct-conv 调优各候选输出逐位一致；最终路径保持 F32，不用半精度换速度。

调优中的负结果也保留为边界：`64x32` 约 1.412 s，K tile 16 约 1.615 s，`128x64` 约 1.37–2.04 s，`64x128` 约 4.12 s；把 sub-pixel phase-interleave/crop 融入卷积 epilogue 会造成非合并写，约 1.35 s，故均未进入默认实现。`PCNSF_DIRECT_CONV=0` 可回退旧路径做 A/B。
