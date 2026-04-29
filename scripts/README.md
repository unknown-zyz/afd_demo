# Scripts

| 脚本 | 用途 |
|---|---|
| `run_single.sh` | 运行单个配置：local/multinode、serial/prefill-dbo/decode-dbo/crosslayer |
| `run_experiment_matrix.sh` | 批量扫描 batch × seq × mode，并生成 summary CSV |
| `run_node.sh` | 手动启动单个 Attention 或 FFN 节点 |
| `gen_experiment_report.py` | 从 attention/ffn timing JSON 生成 markdown 报告 |
| `visualize_dbo_pipeline.py` | 从一组 timing JSON 生成 4 泳道 pipeline Gantt 图 |
| `plot_all_pipelines.py` | 批量扫描结果目录并生成所有可视化图 |
| `audit_experiment_baselines.py` | 扫描 DBO timing 是否有 mode-matched serial baseline |
| `capture_serial_split.py` | 为 GPU serial cache 补充 `prefill_ms` / `decode_tpot_ms` |
| `capture_serial_prefill.sh` | 旧版 GPU-only split 补采脚本 |

## 常用命令

```bash
# 单次 prefill DBO
./scripts/run_single.sh local 8 128 --tokens 20

# serial baseline
./scripts/run_single.sh local 8 128 --tokens 20 --no-dbo

# decode DBO / cross-layer decode
./scripts/run_single.sh local 8 128 --tokens 20 --generate
./scripts/run_single.sh local 8 128 --tokens 20 --generate --crosslayer

# 批量矩阵
./scripts/run_experiment_matrix.sh --modes serial,prefill-dbo --batches 2,4,8 --seqs 128

# 批量画图
python scripts/plot_all_pipelines.py --root results

# 审计图表 speedup 是否有可信 serial baseline
python scripts/audit_experiment_baselines.py --root results

# 为已有 serial cache 补 TTFT/TPOT split
python scripts/capture_serial_split.py --root results --skip-existing
```

所有运行脚本默认写入 `results/`；详细参数可用 `--help` 查看。
