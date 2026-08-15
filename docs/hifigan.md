# NSF-HiFiGAN ggml 推理架构

本文档描述 pc-nsf-hifigan 的图构建、权重布局、正确性保证和设计决策。

## 1. 模型结构

```
source = fastsinegen(f0)                     [libmininsf, 布局见 §3]
x      = conv_pre  (Conv1d 128->512 k7       pad 3)
for i in ups (rates [8,8,2,2,2], kernels [16,16,4,4,4]):
    x = LReLU(x)
    x = ups[i](ConvTranspose1d)               # i==1 时叠加 source_conv(source)
    x = mean(resblocks[i*3..i*3+2](x))        # ResBlock1: 3 组 dilated Conv1d 对
x = LReLU(x, 0.01); x = conv_post(Conv1d 16->1 k7 pad3); x = tanh(x)
```

- `resblock1` 的 3 个分支 dilation = [1,3,5]，第二个 conv 恒为 dilation 1。
- 最终 `Generator.forward` 用 `F.leaky_relu(x)`（默认 slope 0.01），不是 0.1。

## 2. ggml 算子与布局

| 组件 | torch 布局 | ggml kernel ne | ggml op |
|------|------------|----------------|---------|
| convs  | [OC,IC,K]  | [K,IC,OC]      | `ggml_conv_1d(s=1,p=K/2*dil,dil)` |
| ups    | [Cin,Cout,K]| 默认 `hifigan.ups.N` [K,Cout,Cin]（legacy convT）；可选 sub-pixel `hifigan.upsub.N` [Cout*s,Cin,M]（相位快 `c=r+s*o`） | `ggml_conv_transpose_1d` + crop，或 `ggml_conv_1d` + graph interleave |
| source | [1,1,256]  | [1,1,256]      | `ggml_mul_mat`（1x1 conv） |

- **conv 层**：`ggml_conv_1d` 的最终 reshape 假定 `c_out==OW`（官方实现），本
  模型大多数 conv c_out≠OW，因此对每个 conv 都显式 `reshape_2d` 到
  `[ne0, ne1]`。实测 CPU 正确（DOUBLE_CONV check；CUDA 需 ggml-patch）。
- **ups（默认 = legacy convT）**：converter 默认输出 `hifigan.ups.N`，
  引擎走 `ggml_conv_transpose_1d(p0=0)` + crop `(K-s)/2`。这条路径在真实
  权重下与 torch 高度一致（corr>0.9999，见 `verification_real_hifigan.md`）。
- **ups（可选 sub-pixel）**：引擎检测到 `hifigan.upsub.0.weight` 时走
  sub-pixel（`ggml_conv_1d` + graph interleave），converter 尚未产出该格式
  （处于实验修复中）。
- **source conv 1x1**：`mul_mat` 路径（B 必须 F32，权重在 A），输入
  `[IC,T]` transpose 后 `mul_mat(W[IC,OC], xt)` → `[OC,T]` → transpose +
  bias，等价 torch conv1d(k1)。

## 3. mini-nsf 源

由 `libmininsf.mininsf_fastsinegen_f32` 在 host 生成浮点正弦源，长度
`T*upp`（upp = prod(rates[:2]) = 64），采样率 `src_sr = sr/prod(rates[2:])
= 5512.5Hz`。与 torch 的 `fastsinegen` 数学等价（取整/相位累加见
libmininsf README），C++ 走精确 `sinf` 路径（`_fast_` 变体为近似 poly
sin，误差 1e-3 量级）。

`upp` 与 `src_sr` 既是 GGUF meta（`hifigan.upp` / `hifigan.source_sr`），
也能从 rates 推算回退，保持 torch 原样。

## 4. 权重上传与后端

ggml v0.11.0 的 `backend` 在 gguf load 后需要设备 buffer。CPU 后端：
gguf_init(no_alloc=false) 自带内存。GPU 后端：遍历 meta ctx tensor，
缓存在 device buffer 中（`tsl_alloc`），行走 `ggml_backend_tensor_set`。

关键坑（Vulkan 已验证）：
- **必须**在 `tsl_alloc` 前 `t->data=nullptr`、`t->buffer=nullptr`。
- **绝不能解引用** `t->data`（Vulkan data 是固定假基址 0x1000+offset），
  全部走 `tensor_get/set`。

本仓库的 CPU/F16 路径本身无量化，GPU 上按设备 F32/F16 精度计算，无精度
损失（vocoder 峰值 buffer 通常 << 1GB）。

## 5. 精度策略（双线路）

引擎通过 `HifiganModel(path, n_threads, precision)` 选择线路（CLI 用
`HF_PRECISION` 环境变量），converter 用 `--dtype` 产出对应 GGUF：

- **F32 线路（默认）**：`--dtype F32`；权重读取 F32、所有卷积/矩阵计算在
  fp32，用于 golden/debug（全精度）。
- **F16 线路（预留）**：`--dtype F16` + `HF_PRECISION=F16`；权重视为
  fp16（为未来 fp16/bf16 训练 checkpoint 预留），但 ggml 的 1D 卷积/上采样
  在 CPU 上以 fp32 计算，因此实际语义为 **fp16 权重 + fp32 计算**
  （bias 恒为 F32）。本实现不启动 fp16 激活计算——ggml conv1d 要求激活
  为 fp32，未来若上游支持 fp16 激活再升级。
- **sub-pixel 实验说明**：sub-pixel 走 `ggml_conv_1d`，其 im2col 内部在 CPU
  以 **fp16** 累积（ggml v0.19 行为）；stage 冒烟可达 rms 5e-4/max 1e-3，
  但 **真实变长输入全链路仍不正确（corr 0.696）**——在修复前 converter
  默认不输出 `hifigan.upsub.*`，引擎仍用 legacy convT（`ggml_conv_transpose_1d`，
  与 torch corr>0.9999）。
- **无量化**：GB 由「hifigan 不量化」策略固定（与 VR/RMVPE 不同，那两者
  才会 Q8/Q4 消融）。未来 fp16 训练后仍保持此接口。

## 6. 正确性保证（测试链）

| 层 | 验证 | 方法 |
|----|------|------|
| mel 前端 | t01/t02 与 numpy 参考比对 | gen_golden.py |
| source | t03 与 numpy 参考比对 | gen_golden.py |
| GGUF 元数据 | t04 与 tiny.gguf（gguf pkg 生成）比对 | gen_golden.py |
| 端到端 vocode | t05（optional） | 真实 GGUF 冒烟 |

golden 由 `tests/gen_golden.py` 在 CTest fixture 内重新生成，无需提交
二进制 golden。

## 7. 性能说明

- 每帧 512 样本输出，~1.1s 音频/帧成本由卷积决定（卷积带来的 upsample
  是 hop 方向的）；单条 CLI 中等曲目规模在 CPU 多线程下实时可承受，
  Vulkan 后端 <1GB buffer 稳定。
- 复用 `gguf_get` 缓存 tensor，`hifigan_run` 每次重建 graph（server 无状态
  API）。

## 8. 参考

- DiffSinger `modules/nsf_hifigan/models.py`（Generator）
- DiffSinger `modules/nsf_hifigan/nvSTFT.py`（mel 前端）
- `sts`：pocketfft（stft），slaney mel scale
- ggml v0.11.0 `ggml-conv.h` / `ggml-backend.h`
