# Decode DBO 测试报告

**测试时间**: 2026-03-30 13:05
**测试模型**: Qwen/Qwen2-1.5B
**测试环境**: 单机双 GPU (CUDA_VISIBLE_DEVICES=0,1)

## 测试结果

### 测试 1: batch_size=2, max_new_tokens=5

| 模式 | 生成时间 | 吞吐量 |
|------|----------|--------|
| 使用 Decode DBO | 1133.29ms | 4.4 tok/s |
| 无 Decode DBO | 1144.25ms | 4.4 tok/s |

**结论**: batch_size=2 时两者性能相近，因为 micro-batch 大小为 1，无法有效并行。

### 测试 2: batch_size=4, max_new_tokens=10

| 模式 | 生成时间 | 吞吐量 |
|------|----------|--------|
| 使用 Decode DBO | 1460.70ms | 6.8 tok/s |
| 无 Decode DBO | 1436.68ms | 7.0 tok/s |

**结论**: batch_size=4 时，Decode DBO 有轻微开销 (~2%)。这是因为:
1. Qwen2-1.5B 是小模型，计算时间短，通信开销占比大
2. 单机双 GPU 通信延迟低，DBO 的重叠优势不明显

## 分析

Decode DBO 的优势在于:
1. **大 batch + 大模型**: 计算时间长，可以有效隐藏通信
2. **多机部署**: 网络延迟高，DBO 重叠更有价值
3. **内存受限场景**: 将大 batch 拆分为小 micro-batch，降低峰值内存

当前测试场景 (小模型 + 单机) 不是 Decode DBO 的最佳场景。

## 实现状态

- [x] `--generate` 开关已删除，默认启用自回归生成
- [x] Decode DBO 调度器已实现 (`src/pipeline/decode_scheduler.py`)
- [x] 支持 `--no-decode-dbo` 关闭 Decode DBO
- [x] 支持 `--decode-micro-batches` 配置 micro-batch 数量

## 后续优化方向

1. 多机测试验证 Decode DBO 效果
2. 更大模型 (如 Qwen3-30B-A3B) 测试
3. 计算-通信重叠的精细化调优
