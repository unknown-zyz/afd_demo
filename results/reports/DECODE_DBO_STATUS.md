# Decode DBO 状态报告

## 执行摘要

 **已完成**:
1. 修复 Decode DBO KV cache bug - 支持 batch > 1
2. 测量单机通信延迟 - 0.58ms (one-way)
3. 完成全面的单机基准测试


## 详细结果

### 1. Bug 修复 ✅

**问题**: batch > 1 时 KV cache shape 不匹配导致崩溃  
**解决**: 实现 `_slice_kv_cache()` 方法，为每个 micro-batch 切片 KV cache  
**状态**: 功能正常，batch=2/4/8 均可运行

### 2. 网络延 ✅

**单机实测** (4096 floats, 200 iterations):
- Mean: 1.160ms (round-trip) → **0.580ms (one-way)**
- P95: 0.163ms
- P99: 0.176ms

**结论**: 单机通信延迟极低，符合 NVLink/PCIe 预期

### 3. 性能基准测试 🔴

#### 测试配置
- 模型: Qwen2-1.5B (28 layers)
- 硬件: 单机 4 GPU
- Tokens: 50 per test
- Micro-batches: 2

#### 结果对比

| Config | DBO ON | DBO OFF | 差异 | 判断 |
|--------|---------|---------|------|------|
| batch=2, seq=128 | 6.6 | 6.9 | -4% | 🟡 小幅倒退 |
| batch=2, seq=512 | 6.1 | 6.9 | -12% | 🟡 中度倒退 |
| **batch=4, seq=128** | **3.8** | **7.0** | **-46%** | 🔴 **严重倒退** |
| **batch=4, seq=512** | **3.8** | **6.8** | **-44%** | 🔴 **严重倒退** |
| **batch=8, seq=128** | **3.8** | **6.9** | **-45%** | 🔴 **严重倒退** |
| **batch=8, seq=512** | **3.8** | **6.7** | **-43%** | 🔴 **严重倒退** |

#### 关键观察

1. **固定瓶颈**: batch >= 4 时，DBO ON 的吞吐量固定在 **3.8 tok/s**
2. **与 seq_len 无关**: 增加 prompt 长度不影响 DBO ON 性能
3. **batch=2 相对正常**: 性能下降较 (-4% ~ -12%)

## 问题分析

### 根因推断

**主要瓶颈**: KV Cache 切片对象创建开销

```python
# 当前实现 - 每次调用都创建新对象
def _slice_kv_cache(self, kv_cache, batch_start, batch_end):
    sliced_cache = DynamicCache()  # 新对象！
    for layer_idx in range(28):     # 28 layers
        k_slice = layer.keys[batch_start:batch_end]
        v_slice = layer.values[batch_start:batch_end]
        sliced_cache.update(k_slice, v_slice, layer_idx)  # 创建新 layer
    return sliced_cache
```

**开销估算**:
- 每个 decode step: 2 个 micro-batch → 2 次 `_slice_kv_cache()`
- 每次调用: 创建 1 个 cache + 28 个 layers
- 50 tokens: **2 × 28 × 50 = 2800 次对象创建**

**其他因素**:
- Micro-batch 太小 (batch=4 时每个 MB 只有 2 samples)
- GPU 利用率低
- isend/irecv 引入额外同步点

### 为什么 batch=2 时相对正常？

- Micro-batch size = 1 (可能触发特殊优化路径)
- 对象创建开销相对总时间占比较小
- 或退化为串行执行

## 对比分析

### Single-node vs Multi-node

**单机特点**:
- ✅ 通信延迟: 0.58ms (实测)
- ✅ 通信占比极小
- ❌ DBO 重叠收益被开销抵消

**多机预期** (10-100ms 网络):
- 通信延迟增加 20-200 倍
- 通信占比显著增加
- DBO 重叠收益**可能**更明显
- **但**: 必须先解决当前性能问题！

### Prefill vs Decode DBO

| 阶段 | 单机效果 | 原因 |
|------|---------|------|
| **Prefill** | ✅ 良好 (54%-71% 效率) | 大矩阵计算，重叠有效 |
| **Decode** | ❌ 严重倒退 (-44%) | 单 token 计算快，开销大 |

## 修

### P0 - 优化 KV Cache 切片 (必须)

**选项 A**: 缓存切片对象
```python
# 初始化时创建
self.mb_kv_caches = [DynamicCache() for _ in range(num_mb)]

# Decode 时复用
for mb_idx in range(num_mb):
    # 只更新 tensor，不创建新对象
    mb_cache = self.mb_kv_caches[mb_idx]
    for layer_idx in range(num_layers):
        mb_cache.layers[layer_idx].keys = full_cache.layers[layer_idx].keys[start:end]
        mb_cache.layers[layer_idx].values = full_cache.layers[layer_idx].values[start:end]
```

**选项 B**: 直接传递切片（避免新对象）
```python
# 不创建 DynamicCache，直接切片原始 tensor
k_slice = kv_cache.layers[layer_idx].keys[start:end]
v_slice = kv_cache.layers[layer_idx].values[start:end]
C++ 层处理
```

**选项 C**: Python 层传递索引，C++/CUDA 层切片
- 避免 Python 对象创建
- 在 kernel 中直接索引

### P1 - 调整 Micro-batch 策略

**动态启用条件**:
```python
# 只有 batch >= 4 且每个 MB >= 2 时才启用 DBO
if batch_size >= 4 and batch_size // num_micro_batches >= 2:
    use_decode_dbo = True
else:
    use_decode_dbo = False  # 退化为串行
```

**自适应 micro-batch 数量**:
- batch=4 → 2 MB (each 2 samples)
- batch=8 → 2 MB (each 4 samples) 而非 4 MB
- 保证每个 MB 足够大

### P2 - 减少同步点

- 合并多个 isend/irecv
- 使用 batch_isend_irecv
- 减少 wait() 调用

## 下一步行动

### 立

1. **Profile 分析** - 确认瓶颈
   ```bash
   nsys profile python scripts/benchmark_decode_dbo.sh
   ```

2. **实现选项 A** - 缓存 KV cache 对象
   - 最简单，风险最小
   - 预期提升显著

3. **重新测试** - 验证优化效果
   - 目标: DBO ON ≥ DBO OFF
   - 理想: DBO ON > DBO OFF (至少 5-10%)

### 后续计划

4. **多机测试** - 优化完成后
   - 测量跨节点延迟
   - 验证 DBO 在高延迟下的优势

5. **Qwen3-30B 测试** - 大模型验证
   - 从 batch=8 开始尝试
   - OOM 后逐步降低

## 结论

**当前状态**: Decode DBO 功能正确但性能严重倒退

**核心问题**: KV cache 切片对象创建开销过大（推断）

**行动**: 必须先优化性能再进行多机测试

**预期**: 优化后 DBO 在高延迟环境应有显著收益

---

*: 2026-03-31*  
*测试环境: 单机 4xGPU, Qwen2-1.5B*  
*通信延迟: 0.58ms (measured)*
