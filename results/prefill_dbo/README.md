# Prefill DBO 测试结果

本目录包含 Prefill 阶段 DBO (Dual Batch Overlap) 的测试结果和分析。

## 文件说明

### 📊 可视化结果

| 文件 | 说明 |
|------|------|
| `dbo_summary.png` | **综合总结图**（6 个子图）- 最重要的可视化 |
| `dbo_prefill_timeline.png` | 详细时间线（每层每个 micro-batch） |
| `dbo_prefill_timeline_analysis.png` | 计算 vs 等待时间汇总 |

### 📝 原始数据

| 文件 | 说明 |
|------|------|
| `timing_attention.json` | Attention 节点 Prefill 阶段详细时间（30KB） |
| `timing_ffn.json` | FFN 节点 Prefill 阶段详细时间（49KB） |

### 📋 测试日志

| 文件 | 说明 |
|------|------|
| `attn_prefill.log` | Attention 节点测试日志 |
| `ffn_prefill.log` | FFN 节点测试日志 |
| `attn_bench.log` | Attention 基准测试日志 |
| `ffn_bench.log` | FFN 基准测试日志 |

### 📄 分析报告

| 文件 | 说明 |
|------|------|
| `timing_analysis.md` | Prefill 阶段详细分析 |

## 核心发现

### Prefill DBO 效率

| 节点 | 端到端时间 | 计算时间 | 计算效率 |
|------|----------|---------|---------|
| **Attention** | 1278ms | 701ms | **54.8%** |
| **FFN** | 1550ms | 1103ms | **71.2%** |

**关键结论**:
- ✅ FFN 节点达到 71.2% 计算利用率，DBO 重叠有效
- ⚠️ FFN 计算时间是 Attention 的 1.57x，成为瓶颈
- 📊 MoE 开销：router 171ms + experts 702ms (79% FFN compute)

## 测试配置

- **模型**: Qwen2-1.5B (28 layers)
- **环境**: 单机 4 GPU (CUDA_VISIBLE_DEVICES=0,1,2,3)
- **Prompt**: "Hello, how are you today? I'm testing the DBO system."
- **Micro-batches**: 2

## 使用方法

### 重新生成可视化

```bash
# 从 timing JSON 生成 timeline
MPLBACKEND=Agg python scripts/visualize_dbo.py results/prefill_dbo/ \
  --output results/prefill_dbo/dbo_prefill_timeline.png \
  --max-layers 8

# 生成综合总结图
python scripts/plot_dbo_summary.py
```

### 运行新测试

```bash
# Prefill 测试（不生成 tokens）
./scripts/benchmark_dbo.sh 50 4 on   # DBO ON
./scripts/benchmark_dbo.sh 50 4 off  # DBO OFF
```

## 相关文档

- [../reports/FINAL_REPORT.md](../reports/FINAL_REPORT.md) - 最终综合报告
- [../decode_dbo/](../decode_dbo/) - Decode DBO 测试结果
- [../../doc/01-architecture.md](../../doc/01-architecture.md) - DBO 实现原理
