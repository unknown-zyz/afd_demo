# Decode DBO 单机基准测试结果 - 性能倒退分析

## 测试环境
- 模型: Qwen2-1.5B
- 硬件: 单机 4 GPU
- 通信延迟: 0.58ms (one-way, 实测)

## 测试结果

| Batch | Seq Len | DBO ON (tok/s) | DBO OFF (tok/s) | 差异 | 时间 (ON/OFF) |
|-------|---------|----------------|-----------------|------|---------------|
| 2 | 128 | 6.6 | 6.9 | -4.3% | 7550ms / 7222ms |
| 2 | 512 | 6.1 | 6.9 | -11.6% | 8258ms / 7249ms |
| 4 | 128 | **3.8** | **7.0** | **-45.7%** ❌ | 13017ms / 7177ms |
| 4 | 512 | **3.8** | **6.8** | **-44.1%** ❌ | 13153ms / 7342ms |
| 8 | 128 | **3.8** | **6.9** | **-44.9%** ❌ | 13016ms / 7267ms |
| 8 | 512 | **3.8** | **6.7** | **-43.3%** ❌ | 13316ms / 7427ms |

## 关键发现

### 🔴 **严重性能倒退**
- Batch=2: DBO ON 性能略差 (-4% ~ -12%)
- **Batch>=4: DBO ON 性能崩溃 (-44% ~ -46%)**
- DBO ON 反而比 OFF **慢了近 2 倍**！

### 📊 性能模式分析

**奇怪的现象**：
1. Batch=4/8 时，DBO ON 的吞吐量都是 **3.8 tok/s**（完全一致）
2. DBO OFF 的吞吐量保持稳定 6.7-7.0 tok/s
3. Seq length 对 DBO ON 几乎无影响（都是 3.8 tok/s）

echo .git .github .gitignore .pytest_cache README.md config doc requirements.txt results scripts src tests venv **固定的瓶颈**限制了 DBO ON 的吞吐量。

## 问题根因分析

### 可能原因 1: KV 切片开 Cache ⚠️

性能倒退分析 decode step、每个 micro-batch 都创建新的 `DynamicCache`:

```python
def _slice_kv_cache(self, kv_cache, batch_start, batch_end):
    sliced_cache = DynamicCache()  # 新对象！
    for layer_idx in range(len(kv_cache.layers)):
        k_slice = layer.keys[batch_start:batch_end]
        v_slice = layer.values[batch_start:batch_end]
        sliced_cache.update(k_slice, v_slice, layer_idx)  # 创建新 layer
    return sliced_cache
```

**开销估算**：
- 每个 token: 创建 2 个 micro-batch cache
- 每个 cache: 创建 28 layers (Qwen2-1.5B)
- 50 tokens: 2 × 28 × 50 = **2800 次对象创建**

### 可能原因 2: 同步开销

DBO 引入了额外的 `isend`/`irecv`/`wait` 操作，可能导致：
- 过多的同步点
- GPU kernel launch 开销
- 通信调度开销

### 可能原因 3: Micro-batch 太小

 batch=4, micro_batches=2 时：
- 每个 micro-batch 只有 2 个样本
-  GPU
- 切分带来的开销 > 重叠带来的收益

## 为什么 Batch=2 时相对正常？

Batch=2 时性能差异较小 (-4%):
- Micro-batch size = 1 (batch=2 / 2 MB)
- 可能触发了优化路径或退
- 开销相对于总计算时间更小

## 单机 vs 多机预期

**单机环境特点**：
- 通信延迟：0.58ms（实测）
- 计算时间主导，通信占比极小
- DBO 重叠收益有限

**多机环境预期**：
ls
- 通信占比显著增加
- DBO 重叠收益可能更明显
- **但前提是要解决当前的性能问题！**

## 修复方向

### 优先级 P0: 优化 KV Cache 切片
1. **缓存切片对象**：避免每次都创建新 cache
2. **直接传递 slice 而非创建新对象**
3. **考虑在 Python 层传递索引，C++ 层切片**

### 优先级 P1: 调整 Micro-batch 策略
- Batch >= 4 时才启用 DBO
- 每个 micro-batch 至少 2-4 个样本
- 动态调整 micro-batch 数量

### 优先级 P2: 减少同步点
- 合并通信操作
- Overlap more aggressively

## 结论

**当前状态**: Decode DBO 在单机环境性能严重倒退 (-44% ~ -46%)，主要原因可能是 KV cache 切片开销过大。

**建议**:
1. **暂时禁用 Decode DBO** 直到优化完成
2. 优化 KV cache 切片实现
3. 多机测试前必须先解决性能问题
4. 考虑 Decode 阶段是否真的需 DBO（单 token 计算本就很快）

**下一步**:
- Profile 找出确切瓶颈
- 实现 cache slice 优化
- 重新测试验证
