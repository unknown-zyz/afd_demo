# 实验结果目录

 AFD (Attention-FFN Disaggregation) + DBO (Dual Batch Overlap) 系统的所有实验结果和分析报告。

## 📁 目录

| 目录 | 说明 | 重要文件 |
|------|------|---------|
| [`prefill_dbo/`](prefill_dbo/) | Prefill DBO 测试 | `dbo_summary.png`, `timing_*.json` |
| [`decode_dbo/`](decode_dbo/) | Decode DBO 测试 | `summary.csv`, `analysis.md` |
| [`network_latency/`](network_latency/) | 网络延迟测量 | `comm_latency_single_node.log` |
| [`reports/`](reports/) | 综合报告 | `FINAL_REPORT.md` ⭐ |

## 🎯 核心结论

### Prefill DBO（推荐使用✅）

- **单机效率**: Attention 54.8%, FFN 71.2%
- **多机预期**: >70% (Attention), >80% (FFN)
- **推荐**: ✅ 推荐在生产环境使用

### Decode DBO（当前不推荐❌）

- **性能影响**: batch=2 时 -4% ~ -12%, batch>=4 时 **-44% ~ -46%**
- **根因**: KV Cache 对象创建开销（2800+ 次/50 tokens）
- **推荐**: ❌ 需优化后才能使用

 [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md)。

## 📊 快速访问

### 最重要的文件

1. **[reports/FINAL_REPORT.md](reports/FINAL_REPORT.md)** - 最终综合评估报告
2. **[prefill_dbo/dbo_summary.png](prefill_dbo/dbo_summary.png)** - Prefill DBO 可视化总结
3. **[decode_dbo/summary.csv](decode_dbo/summary.csv)** - Decode DBO 测试结果汇总
4. **[decode_dbo/analysis.md](decode_dbo/

### 按任务查找

**了解 Prefill DBO 效果**:
- 查看 `prefill_dbo/dbo_summary.png`
- 阅读 `prefill_dbo/README.md`

**了解 Decode DBO 问题**:
- 查看 `decode_dbo/summary.csv`
- 阅读 `decode_dbo/analysis.md`

**查看网络延迟数据**:
- 查看 `network_latency/comm_latency_single_node.log`
- 阅读 `network_latency/README.md`

**获取完整评估**:
- 阅读 `reports/FINAL_REPORT.md`

## 📈 测试数据总结

### Prefill DBO 测试

- **测试配置**: Qwen2-1.5B, 单机 4 GPU, 2 micro-batches
- **测试文件**: 12+ 个日志、2 个 JSON、3 个 PNG
- **核心指标**: 54.8% (Attention), 71.2% (FFN) 计算效率

### Decode DBO 测试

- **测试配置**: Qwen2-1.5B, batch=2/4/8, seq=128/512, DBO ON/OFF
- **测试文件**: 12 个基准测试日志 + summary.csv
- **核心指标**: batch>=4 时性能倒退 -44% ~ -46%

### 网络延迟测试

- **测试配置**: 单机 4 GPU, ping-pong, 200 iterations
- **核心指标**: 0.58ms (one-way), P95=0.16ms

## 🔄 更新结果

### 运行新测试

```bash
# Prefill DBO
./scripts/test_local.sh 50 4 --timing          # DBO ON
./scripts/test_local.sh 50 4 --timing --no-dbo # DBO OFF

# Decode DBO
./scripts/benchmark_decode_dbo.sh

# 网络延迟
python scripts/measure_comm_latency.py
```

### 生成可视化

```bash
# Prefill timeline
python scripts/visualize_dbo.py results/prefill_dbo/

# 综合总结
python scripts/plot_dbo_summary.py
```

## 📝 文件命名规范

### Prefill DBO

- 日: `{attn|ffn}_{prefill|bench}.log`
- JSON: `timing_{attention|ffn}.json`
- 图表: `dbo_*.png`

### Decode DBO

- 日志: `Qwen2-1.5B_batch{N}_seq{N}_dbo_{on|off}.log`
- 汇总: `summary.csv`
- 分析: `analysis.md`

### 网络延迟

- 日志: `comm_latency_{single_node|multi_node}.log`

### 报告

- 综合: `FINAL_REPORT.md`
- 状态: `DECODE_DBO_STATUS.md`
- 会话: `README_SESSION.md`

## 🔍 相关文档

- [../doc/01-architecture.md](../doc/01-architecture.md) - DBO 实现原理
- [../doc/02-usage.md](../doc/02-usage.md) - 如何运行测试
- [../scripts/README.md](../scripts/README.md) - 测试脚本说明
