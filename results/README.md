# Results

本目录保存 GPU 实验产物。历史 JSON/PNG/report 作为实验归档保留，不在常规代码清理中删除。

## 目录结构

| 目录/文件 | 说明 |
|---|---|
| `experiment_matrix_summary.csv` | 最近一次矩阵实验摘要 |
| `serial/` | 串行 baseline；`cache/` 中的 JSON 供 DBO 报告计算 speedup |
| `prefill-dbo/` | Prefill DBO timing JSON、报告和 pipeline PNG |
| `decode-dbo/` | Decode DBO（crosslayer off）timing JSON、报告和 pipeline PNG |
| `decode-dbo-crosslayer/` | Decode DBO + cross-layer pipeline 实验产物 |
| `archive/` | 早期实验归档 |

## 命名规范

单次配置使用 `b{B}_s{S}_t{T}` 表示 batch、prefill sequence length、decode tokens。

| 文件 | 说明 |
|---|---|
| `timing_attention_<mode>_b{B}_s{S}_t{T}.json` | Attention 节点 timing events |
| `timing_ffn_<mode>_b{B}_s{S}_t{T}.json` | FFN 节点 timing events |
| `report_<mode>_b{B}_s{S}_t{T}.md` | 自动报告 |
| `pipeline_<mode>_b{B}_s{S}_t{T}.png` | 4 泳道 pipeline Gantt 图 |

## 复现实验

```bash
./scripts/run_experiment_matrix.sh
python scripts/plot_all_pipelines.py --root results
```

如果只需要单个配置：

```bash
./scripts/run_single.sh local 8 128 --tokens 20
./scripts/run_single.sh local 8 128 --tokens 20 --no-dbo
```
