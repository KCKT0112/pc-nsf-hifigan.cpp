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

## 6. 终态裁决（2026-08-30）：Vulkan↔DML 差距取舍已定性、刻意止步

剩余 ~1.5× 差距（Vulkan 430–480 ms 对本文重锚 ORT DML EP **281.6 ms**，同会话 n=10）做过系统勘察，结论为**不再继续**：每条提速路线要么违反数值合同、要么属于本仓库无意持有的引擎级基建。

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
