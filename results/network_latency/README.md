# 网络延迟测试结果

本目录包含通信延迟测量结果。

## 文件说明

| 文件 | 说明 |
|------|------|
| `comm_latency_single_node.log` | 单机通信延迟测试结果 |

## 测试结果

### 单机通信延迟（NVLink/PCIe）

**测试配置**:
- 环境: 4 GPU (CUDA_VISIBLE_DEVICES=0,1,2,3)
- 数据大小: 4096 floats (~16KB)
- 迭代次数: 200
- 测试方法: Ping-pong (双向收发)

**实测结果**:
- Round-trip: **1.160ms** (mean)
- **One-way: 0.580ms**
- P95: 0.163ms
- P99: 0.176ms

**结论**: 单机通信延迟极低（<1ms），验证了之前的推断。

## 使用方法

### 运行延迟测试

```bash
# 默认配置
python scripts/measure_comm_latency.py

# 自定义参数
python scripts/measure_comm_latency.py --size 8192 --iterations 500
```

### 参数说明

- `--size` - 传输数据大小（float 数量，默认: 4096）
- `--iterations` - 测试迭代次数（默认: 200）

### 输出示例

```
Communication Latency Test
==========================
Configuration:
  Tensor size: 4096 floats
  Iterations: 200
  
Results:
  Round-trip latency:
    Mean: 1.160 ms
    Std:  0.089 ms
    P95:  0.163 ms
    P99:  0.176 ms
  
  One-way latency (estimated):
    Mean: 0.580 ms
```

## 多机测试

**状态**: 未测试（Decode DBO 性能问题优先）

**预期延迟**:
- 千兆以太网: 10-50ms
- InfiniBand: 1-5ms
- 跨数据中心: 50-200ms

### 运行多机测试

```bash
# 机器 1 (Attention 节点)
CUDA_VISIBLE_DEVICES=0 python scripts/measure_comm_latency.py \
  --rank 0 \
  --world-size 2 \
  --master-addr 10.244.64.179 \
  --master-port 29500

# 机器 2 (FFN 节点)
CUDA_VISIBLE_DEVICES=0 python scripts/measure_comm_latency.py \
  --rank 1 \
  --world-size 2 \
  --master-addr 10.244.64.179 \
  --master-port 29500
```

## 分析

### 单机环境（0.58ms）

- **Prefill DBO**: ✅ 有效（54.8%-71.2% 效率）
- **Decode DBO**: ❌ 负收益（-44% 性能倒退）

**原因**: 通信延迟极低，DBO 重叠效果依赖计算时间 >> 通信时间。Prefill 阶段计算密集，效果好；Decode 阶段计算时间短，反而被 Python 对象创建开销拖累。

### 多机环境（10-100ms 预期）

- **Prefill DBO**: ✅ 预期效果更好（>70% 效率）
- **Decode DBO**: ⚠️ 需优化后验证

**预期**: 通信延迟增加 20-200 倍，DBO 重叠收益应显著提升。但 Decode DBO 当前性能问题（-44%）需先优化。

## 相关文档

- [../decode_dbo/analysis.md](../decode_dbo/analysis.md) - 性能瓶颈分析
- [../reports/FINAL_REPORT.md](../reports/FINAL_REPORT.md) - 综合评估
- [../../doc/01-architecture.md](../../doc/01-architecture.md) - DBO 实现原理
