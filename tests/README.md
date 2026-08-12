# Tests

golden 对比测试：`gen_golden.py`（纯 numpy 参考）在 CTest fixture 阶段
生成 .bin golden，随后各 C++ 测试比对输出。

## 运行

```bash
cmake --build build --config Release -j
ctest --test-dir build --output-on-failure
```

## 用例

| 测试 | 验证 | 比对精度目标 |
|------|------|--------------|
| t01 test_mel | `MelExtractor.forward` (16k, htk=True) vs numpy | rmse<=1e-2, max<=0.1 |
| t02 test_mel_nvstft | `mel_nvstft` (44.1k, slaney, nvSTFT config) vs numpy | rmse<=5e-2, max<=0.25 |
| t03 test_source | `mininsf_fastsinegen_f32` (5512.5Hz upsample64) vs numpy | rmse<=1e-3, max<=5e-3 |
| t04 test_gguf_meta | GGUF 加载 + 元数据访问器 | 需要 gguf pkg，缺则跳过 |
| t05 test_vocode | 真实模型端到端 vocode | 仅配置时设 `PCNSF_MODEL_GGUF` |

## 新增 golden 流程

修改 `gen_golden.py` 重新生成，修改/新增 C++ test 比对，提交时不含 .bin。

## 是否可用

- t01–t04 可在 CI 中稳定跑（无模型资产）
- t05 手动场景：`PCNSF_MODEL_GGUF=<path> cmake -S . -B build ...`
