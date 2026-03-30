# Decode DBO 测试报告

**测试时间**: 2026-03-30 13:05
**测试环境**: 单机双 GPU (CUDA_VISIBLE_DEVICES=0,1)

## 1. Decode DBO 测试 (Qwen2-1.5B)

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

---

## 2. Qwen3-30B-A3B MoE 测试

**测试时间**: 2026-03-30 13:43
**测试环境**: 单机 4 GPU (CUDA_VISIBLE_DEVICES=0,1,2,3)

### 模型信息
- 架构: Qwen3MoeForCausalLM
- 层数: 48
- 专家数: 128 (每 token 选 8 个)
- hidden_size: 2048
- 模型大小: ~56GB (bfloat16)

### 测试结果

| batch_size | max_tokens | 生成时间 | 吞吐量 | 输出示例 |
|------------|------------|----------|--------|----------|
| 1 | 3 | 1829.65ms | 1.6 tok/s | "Hello *everyone*" |
| 1 | 10 | 3294.84ms | 3.0 tok/s | "What is AI?..." |

### MoE 特性确认
```
model_type=qwen3_moe, moe=True, router=True, moe_timing=True
```

---

## 实现状态

- [x] `--generate` 开关已删除，默认启用自回归生成
- [x] Decode DBO 调度器已实现 (`src/pipeline/decode_scheduler.py`)
- [x] Qwen3-30B-A3B MoE 模型支持
- [x] MoE routing 和 experts 计算正确
- [x] 内存优化 (low_cpu_mem_usage)

## 后续优化方向

1. 多机测试验证 Qwen3-30B-A3B
2. 更大 batch size 测试
3. MoE 负载均衡优化
