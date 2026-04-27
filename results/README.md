# 实验结果目录

 AFD + DBO 系统的实验结果。

## 目录结构

| 目录 | 说明 |
|------|------|
ls  |
| `serial/` | Serial baseline 结果（含 `cache/` 供后续实验复用） |
| `prefill-dbo/` | Prefill DBO 实验 |
| `decode-dbo/` | Decode DBO 实验（crosslayer OFF） |
| `decode-dbo-crosslayer/` | Decode DBO + 跨层流水线实验 |
| `summary/` | 跨实验对比报告 |

## 文件命名规范

系统的实验结果。 3 个文件，命名为 `<mode>_<config>.<ext>`，其中 `<config>` = `b{B}_s{S}_t{T}`：

- `timing_attention_<config>.json` — Attention 节点 timing events
- `timing_ffn_<config>.json` — FFN 节点 timing events
- `report_<config>.md` — 自动生成的报告（E2E、per-step 表、per-layer 四段时间表）

## 运行实验

 `scripts/run_experiment_matrix.sh` 与 `.github/skills/run-experiments/SKILL.md`。
