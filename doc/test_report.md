# AFD Demo 测试报告

## 测试环境

### 硬件配置
| 节点 | GPU | 内存 | IP |
|------|-----|------|-----|
| 本地机器 | V100-32GB × 4 | 300GB | 10.244.64.179 |
| 远程机器 | V100-32GB × 4 | 300GB | 192.168.5.32 (SSH: port 31310) |

### 软件环境
- Python 3.12
- PyTorch 2.7.0 (CUDA 12.6)
- Transformers 5.4.0
- Accelerate 1.13.0

---

## 1. Qwen3-30B-A3B MoE 测试

### 模型信息
| 参数 | 值 |
|------|-----|
| 架构 | Qwen3MoeForCausalLM |
| 总参数 | ~30B |
| 激活参数 | ~3B (每 token) |
| 层数 | 48 |
| 专家数 | 128/层, 每 token 选 8 个 |
| hidden_size | 2048 |
| 数据类型 | bfloat16 |
| 模型大小 | ~56GB |

### 资源分配
- **Attention 部分**: ~2.86 GB (6.87% 参数)
- **FFN 部分**: ~38.68 GB (93.13% 参数,含 MoE 专家)
- **每个节点需要**: 2 × 32GB GPU

### 单机测试 (4 GPU)

```bash
./scripts/test_local.sh
```

| 指标 | 值 |
|------|-----|
| GPU 分配 | Attention: 0,1 / FFN: 2,3 |
| 生成 tokens | 3 |
| 生成时间 | 1321.20ms |
| 吞吐量 | **2.3 tok/s** |
| 输出示例 | "Hello *everyone*" |

### 多机测试 (2+2 GPU)

```bash
./scripts/test_multinode.sh
```

| 指标 | 值 |
|------|-----|
| Attention 节点 | 本地 GPU 0,1 |
| FFN 节点 | 远程 GPU 0,1 |
| 生成 tokens | 3 |
| 生成时间 | 1374.72ms |
| 吞吐量 | **2.2 tok/s** |
| 网络开销 | ~4% |

### MoE 特性确认
```
model_type=qwen3_moe, moe=True, router=True, moe_timing=True
```

---

## 2. Decode DBO 测试 (Qwen2-1.5B)

### batch_size=2, max_new_tokens=5

| 模式 | 生成时间 | 吞吐量 |
|------|----------|--------|
| 使用 Decode DBO | 1133.29ms | 4.4 tok/s |
| 无 Decode DBO | 1144.25ms | 4.4 tok/s |

**结论**: batch_size=2 时两者性能相近 (micro-batch=1 无法有效并行)

### batch_size=4, max_new_tokens=10

| 模式 | 生成时间 | 吞吐量 |
|------|----------|--------|
| 使用 Decode DBO | 1460.70ms | 6.8 tok/s |
| 无 Decode DBO | 1436.68ms | 7.0 tok/s |

---

## 3. Prefill DBO 测试

### 同步 vs 异步对比 (Qwen2-1.5B)

```bash
# 使用 --scheduler compare 选项
```

| 模式 | 时间 | 加速比 |
|------|------|--------|
| Sync | 268.94ms | baseline |
| Async (DBO) | 237.60ms | **1.13x** |

---

## 4. 已知问题

1. **NCCL 退出警告**: PyTorch 2.7 + NCCL 2.26 在进程退出时可能触发清理警告,不影响运行结果
2. **torch_dtype 弃用警告**: 建议使用 `dtype` 参数替代

---

## 5. 功能状态

| 功能 | 状态 |
|------|------|
| Prefill DBO | ✅ 已实现 |
| Decode DBO | ✅ 已实现 |
| KV Cache | ✅ 已实现 |
| MoE 支持 | ✅ 已实现 |
| 单机部署 | ✅ 测试通过 |
| 多机部署 | ✅ 测试通过 |

---

*测试日期: 2026-03-30*
