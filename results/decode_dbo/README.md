# Decode DBO 测试结果

本目录包含 Decode 阶段 DBO (Dual Batch Overlap) 的基准测试结果和性能分析。

## 文件说明

### 📊 测试数据

| 文件 | 说明 |
|------|------|
| `summary.csv` | **测试结果汇总**（12 个测试用例） |
| `analysis.md` | **详细性能分析**（根因分析、优化方向） |
| `decode_comparison.md` | DBO ON vs OFF 对比分析 |

### 📋 测试日志

#### 基准测试日志（DBO ON vs OFF）

- `Qwen2-1.5B_batch2_seq128_dbo_on.log` / `*_off.log`
- `Qwen2-1.5B_batch2_seq512_dbo_on.log` / `*_off.log`
- `Qwen2-1.5B_batch4_seq128_dbo_on.log` / `*_off.log`
- `Qwen2-1.5B_batch4_seq512_dbo_on.log` / `*_off.log`
- `Qwen2-1.5B_batch8_seq128_dbo_on.log` / `*_off.log`
- `Qwen2-1.5B_batch8_seq512_dbo_on.log` / `*_off.log`

#### 其他测试日志

- `decode_batch2_test.log` - Batch=2 初步测试
- `decode_dbo_batch1.log` - DBO ON, batch=1
- `decode_nodbo_batch1.log` - DBO OFF, batch=1
- `decode_benchmark_run.log` - 基准测试运行日志

## 核心发现

### 🔴 **Decode DBO 性能严重倒退**

| Batch | Seq Len | DBO ON (tok/s) | DBO OFF (tok/s) | 差异 |
|-------|---------|----------------|-----------------|------|
| 2 | 128 | 6.6 | 6.9 | **-4.3%** |
| 2 | 512 | 6.1 | 6.9 | **-11.6%** |
| **4** | **128** | **3.8** | **7.0** | **-45.7%** ❌ |
| **4** | **512** | **3.8** | **6.8** | **-44.1%** ❌ |
| **8** | **128** | **3.8** | **6.9** | **-44.9%** ❌ |
| **8** | **512** | **3.8** | **6.7** | **-43.3%** ❌ |

### 关键特征

1. **固定瓶颈**: batch >= 4 时，DBO ON 性能固定在 3.8 tok/s
2. **与 seq_len 无关**: 增加序列长度不改善性能
3. **batch=2 时相对正常**: 性能下降较小（-4% ~ -12%）

### 根因分析

**主要瓶颈**: KV Cache 对象创建开销

```python
# 当前实现 - 每个 decode step 都创建新对象
def _slice_kv_cache(self, kv_cache, batch_start, batch_end):
    sliced_cache = DynamicCache()  # 新对象！
    for layer_idx in range(28):
        sliced_cache.update(k_slice, v_slice, layer_idx)
    return sliced_cache
```

**开销估算**:
- 每个 token: 2 micro-batches × 28 layers × 对象创建
- 50 tokens: **约 2800 次对象创建**
- 加上 Python 对象管理开销 → 性能瓶颈

## 测试配置

- **模型**: Qwen2-1.5B (28 layers)
- **环境**: 单机 4 GPU
- **测试矩阵**:
  - Batch sizes: 2, 4, 8
  - Sequence lengths: 128, 512
  - Configurations: DBO ON vs OFF
  - 总计: **12 个测试用例**
- **生成长度**: 50 tokens per test

## 使用方法

### 运行基准测试

```bash
# 完整测试（12 个用例，10-15 分钟）
./scripts/benchmark_decode_dbo.sh

# 查看结果
cat results/decode_dbo/summary.csv
cat results/decode_dbo/analysis.md
```

### 单独测试

```bash
# DBO ON
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m src.main \
  --local-test --model-name Qwen/Qwen2-1.5B \
  --batch-size 4 --prompt "Hello" --max-new-tokens 50

# DBO OFF
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m src.main \
  --local-test --model-name Qwen/Qwen2-1.5B \
  --batch-size 4 --prompt "Hello" --max-new-tokens 50 --no-dbo
```

## 结论与建议

### 当前建议

❌ **不推荐在生产环境使用 Decode DBO**（当前实现）

**原因**: 性能倒退 -44% ~ -46%，远超任何潜在收益

### 优化方向

**P0 - KV Cache 切片优化**:
1. 缓存 MB cache 对象（推荐）
2. 直接传递 tensor 切片
3. C++/CUDA 层切片

**P1 - 动态启用策略**:
- batch < 4 时禁用 Decode DBO
- 单机环境禁用 Decode DBO
- 仅在高延迟多机环境启用

### 多机测试

**状态**: 暂停，等待优化完成

**原因**: 单机已严重倒退 -44%，即使多机通信延迟高（10-100ms），也无法弥补这个损失

## 相关文档

- [../reports/FINAL_REPORT.md](../reports/FINAL_REPORT.md) - 最终综合报告
- [../reports/DECODE_DBO_STATUS.md](../reports/DECODE_DBO_STATUS.md) - 状态总结
- [../prefill_dbo/](../prefill_dbo/) - Prefill DBO 测试（推荐使用✅）
- [../../doc/01-architecture.md](../../doc/01-architecture.md) - Decode DBO 实现细节
