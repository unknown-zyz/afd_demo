# 脚本目录

## 文件

| 脚本 | 功能 |
|------|------|
| `run_node.sh` | 手动启动单个节点（attention / ffn） |
ls DBO/Serial、本地/多机、可视化 |
| `run_experiment_matrix.sh` | 实验矩阵扫描：batch 到 OOM × seq × 模式 |
| `gen_experiment_report.py` | 从 timing JSON 生成 markdown 报告 |
| `visualize_dbo_pipeline.py` | DBO Pipeline 4 泳道甘特图可视化 |

## 快速用法

```bash
# 单次实验
./scripts/run_single.sh local 8 128 --tokens 20 --generate --warmup-p2p

# 带可视化
./scripts/run_single.sh local 4 128 --visualize

# 实验矩阵
./scripts/run_experiment_matrix.sh --mode decode-dbo --seq 128
```

 `--help` 或 `.github/skills/run-experiments/SKILL.md`。
