# 11. 代码审阅与实验脚本使用指南

本文档面向需要快速理解 AFD-DBO（Attention/FFN Disaggregation + Dual-Batch
Overlap）代码与实验流程的审阅者。涵盖：

- 仓库目录速览
- 1A7F (1 Attention + 7 FFN EP) NPU 拓扑
- 三条关键执行路径（Serial / Prefill-DBO / Decode-DBO）
- 实验脚本手册（`run_npu.sh`、`run_experiment_matrix_npu.sh`、聚合脚本）
- 指标定义（TTFT / TPOT / TBT mean/p50/p99 / decode_loop / send-recv wait）
- 结果目录指南
- 常见排查（HCCL EJ0003、OOM、Host2 切换）

若只关心"如何 reproduce 一组结果"，直接跳到 §5。
若要审 PR diff，先看 §3 + §6 中风险点列表。

---

## 1. 仓库目录速览

```
afd_demo/
├── src/
│   ├── main.py                    # CLI 入口（解析 --attn-size/--ffn-size/--generate 等）
│   ├── distributed/               # HCCL/NCCL 初始化、warmup（p2p/prefill）
│   ├── model/
│   │   ├── disaggregated.py       # 顶层 Qwen3 模型 + generate 循环（含 TBT 计时）
│   │   ├── attention_worker.py    # 单 rank Attention layer 0-X
│   │   └── ffn_worker.py          # 单 rank FFN/MoE + EP dispatch
│   ├── pipeline/
│   │   ├── scheduler.py           # Serial scheduler
│   │   ├── async_scheduler.py     # Prefill-DBO
│   │   └── decode_scheduler.py    # Decode-DBO（含 EP broadcast_reduce_overlap）
│   └── utils/
│       ├── timing.py              # PipelineTiming dataclass（JSON 持久化）
│       └── sampling.py
├── scripts/
│   ├── run_npu.sh                 # 单次实验入口（NPU），自动拆 ASCEND_VISIBLE_DEVICES
│   ├── run_experiment_matrix_npu.sh   # 矩阵实验入口
│   ├── aggregate_npu_ep7.py       # EP7 矩阵聚合 + heatmap/曲线图
│   ├── aggregate_mb4_vs_mb2.py    # MB=2 vs MB=4 对比
│   ├── gen_experiment_report.py   # 单次实验 Markdown 报告
│   ├── visualize_dbo_pipeline.py  # 4-lane pipeline 图
│   └── plot_all_pipelines.py
├── doc/                           # 本目录
├── results_npu_ep7/               # EP7 主基线结果
├── results_npu_ep7_mb4_v2/        # fused dispatch / MB4 最终对比
└── tests/
```

---

## 2. 1A7F (EP7) NPU 拓扑

当前默认验证拓扑：1 个 Attention rank + 7 个 FFN EP rank = **8 颗 910C chip**。
preset 定义在 `scripts/run_npu.sh:52`：

```bash
npu-ep7) ATTN_SIZE=1; FFN_SIZE=7; FFN_TP_SIZE=1; FFN_EP_SIZE=7 ;;
```

`run_npu.sh` 会把宿主 `ASCEND_VISIBLE_DEVICES`（如 `0,1,2,3,4,5,6,7`）按角色拆分：

| 角色 | NPU 索引 | 内容 |
|---|---|---|
| ATTN rank 0 | 第 1 颗 | 所有 attention layer + embedding/head |
| FFN rank 0..6 | 后 7 颗 | MoE experts 按 EP=7 切分（每 rank 持有 18 个 expert，共 128） |

attention/FFN 之间走 HCCL p2p 传输 hidden states；FFN 内部走 EP all-to-all
（`broadcast_reduce_overlap` 模式可与 expert 计算 overlap）。

**HBM 占用经验值**：每颗 chip 需要 **≥ 25 GB** 空闲 HBM 才能装下 ATTN
（≈18 GB）或单 FFN EP shard（≈11 GB）+ 激活。Host1 共享 16 chips，常被其
他租户占用；调度前先 `npu-smi info` 确认。

---

## 3. 三条执行路径与代码地图

| 模式 | CLI 触发 | Scheduler 类 | 关键文件 |
|---|---|---|---|
| Serial | `--no-dbo --generate` | `SimplePipelineScheduler` | `src/pipeline/scheduler.py` |
| Prefill-DBO | 默认（无 `--generate`） | `AsyncPipelineScheduler` | `src/pipeline/async_scheduler.py` |
| Decode-DBO | `--generate`（默认 DBO 开） | `DecodeDBOScheduler` | `src/pipeline/decode_scheduler.py` |

### 3.1 Serial decode 循环
`disaggregated.py: _generate_attention_node` (no `decode_scheduler`)：
1. Prefill 一次，sample 第一个 token。
2. 循环 `max_new_tokens - 1` 次：单 token forward → recv FFN logits → sample。

### 3.2 Decode-DBO 循环
同一 generate 函数，但每步调用
`decode_scheduler.forward_decode_dbo(input_ids, position_ids, attention_mask)`。
内部把 batch 切成 `num_micro_batches` 个 micro-batch，attention 与 FFN
之间 hidden states 交替 send/recv，与 compute overlap。

### 3.3 FFN EP dispatch backend
`src/pipeline/decode_scheduler.py` 中根据 `--ffn-ep-backend` 选择：
- `broadcast`：朴素 all-gather（基线）
- `broadcast_reduce_overlap`：dispatch/combine 与 grouped MoE 计算 overlap（推荐）

---

## 4. 指标定义

JSON 字段（每次 run 写入 `results*/<mode>/timing_*.json`）：

| 字段 | 含义 | 计算位置 |
|---|---|---|
| `prefill_ms` | TTFT，等价于 first-token 延迟 | `disaggregated.py` prefill 计时点 |
| `decode_loop_ms` | 整个 decode 循环的 wall time | `decode_start` 到循环结束 |
| `decode_steps` | `max_new_tokens - 1` | — |
| `decode_tpot_ms` | TPOT = `decode_loop_ms / decode_steps`（batch 级聚合） | — |
| **`tbt_mean_ms`** | per-step latency 的平均 | `_tbt_stats()` |
| **`tbt_p50_ms`** | per-step latency 中位数 | `_tbt_stats()` |
| **`tbt_p99_ms`** | per-step latency p99（线性插值） | `_tbt_stats()` |
| `decode_step_times_ms` | 每个 decode step 的 wall time 数组 | 见 §4.1 |
| `send_wait_ms` / `recv_wait_ms` | DBO p2p 等待 | `decode_scheduler` |
| `compute_ratio` | compute / (compute+comm) | 报告里派生 |

### 4.1 TBT 计时实现
`src/model/disaggregated.py`：在 attention 与 FFN 两条 generate 路径的
decode loop 内部，每个 step 前 `step_start = time.perf_counter()`，step 后
`devmod.synchronize() + perf_counter`。每步 sync 引入的开销远小于 step 本身
（典型 step ≥ 30 ms）；同时 sync 不在 DBO overlap 区间内（overlap 发生
在 `forward_decode_dbo` 内部），不影响 DBO 性能。

数组 `decode_step_times_ms` 也写入 JSON，可用于绘制 per-step latency 曲线。
对极长 decode（`tokens > 200`）数据量可能膨胀；如有需要可后续加
`--no-tbt-steps` flag。

### 4.2 TBT vs TPOT
- `decode_tpot_ms` 是 batch wall-time 均摊到 step 上的全局视角；
- `tbt_mean_ms` 是 per-step 序列的均值；
- 二者数值上接近但不同：TPOT 包含循环外 sync 与 barrier 时间，TBT 不含
  step 之间的 Python overhead。

---

## 5. 实验脚本手册

### 5.1 单次实验：`scripts/run_npu.sh`

```bash
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/run_npu.sh \
  --preset npu-ep7 \
  --batch 16 --seq 512 --tokens 20 \
  --ffn-ep-backend broadcast_reduce_overlap \
  --num-micro-batches 2 \
  --generate --timing
```

关键 flag：
- `--preset {npu-ep7,npu-2rank}`：拓扑选择
- `--ffn-ep-backend {broadcast,broadcast_reduce_overlap}`
- `--num-micro-batches {2,4}`：MB；MB4 多数情况下负收益
- `--prefill-warmup-rounds N`：prefill JIT 预热轮数
- `--warmup-p2p`：开启 p2p 通信预热

### 5.2 矩阵实验：`scripts/run_experiment_matrix_npu.sh`

```bash
./scripts/run_experiment_matrix_npu.sh \
  --modes serial,prefill-dbo,decode-dbo \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 --tokens 20 \
  --preset npu-ep7 \
  --ffn-ep-backend broadcast_reduce_overlap \
  --num-micro-batches 2 \
  --no-cache
```

输出 `experiment_matrix_summary.csv`（run 状态），各 run 的 `timing_*.json`
保存在 `results/{prefill_dbo,decode_dbo}/` 与 `results_npu/...`。

### 5.3 聚合脚本
- `aggregate_npu_ep7.py`：扫 `results_npu_ep7/`，产出
  `*_summary.csv`（含 TBT mean/p50/p99 三列）+ 热力图。
- `aggregate_mb4_vs_mb2.py`：对比 MB2/MB4/serial 的 TPOT 与
  TBT p50/p99。
- `aggregate_warmup_ablation_npu.py`：warmup 开关消融。
- `gen_experiment_report.py`：单 run Markdown 报告。

---

## 6. 审阅重点（PR 风险表）

| 模块 | 高风险变更 | 检查方法 |
|---|---|---|
| `disaggregated.py` generate | TBT 计时点位置变化 | 确认 `step_start` 在 forward 调用前；sync 在 sample 后 |
| `decode_scheduler.py` | EP dispatch backend 切换 | serial 与 DBO 必须使用同一 sample 结果对比 |
| `distributed/warmup.py` | warmup 轮数 | 矩阵 run 第一行（L0）TPOT 不应 ≫ 其他行 |
| `utils/timing.py` | 新增字段 | JSON schema 兼容；下游脚本必须 `.get()` 取 |
| 聚合脚本 | CSV header | 不能改既有列顺序，只追加 |

### 7. 常见排查

- **HCCL EJ0003 "IP port bound already"**：容器内残留 `python -m src.main`
  进程，`ps -ef | grep src.main` → `kill <PID>`。彻底情况需 `docker
  restart afd-npu-test`（已授权）。
- **OOM**：Host1 chip 不够 25 GB 时切 Host2（参考
  `.github/skills/npu_910c_env_setup_and_run/SKILL.md` 决策树）。
- **Decode L0 异常慢**：通常是 JIT 编译；提高 `--prefill-warmup-rounds`
  或开 `--warmup-p2p`；可参考 `results_npu_ep7_mb4_v2/` 的 no-L0 图。
- **TBT 数组缺失**：旧 JSON 没有 `decode_step_times_ms`；聚合脚本应用
  `.get()` 兼容。

---

## 8. 修改 / 新增的关键 commit 索引

- TBT 三值指标：`src/model/disaggregated.py: _tbt_stats()`，两条 decode
  loop 中加 `step_start/perf_counter + sync`；`src/utils/timing.py` 新增 4
  字段；`src/main.py` 日志行与 JSON 输出新增同名字段；
  `scripts/aggregate_npu_ep7.py`、`scripts/aggregate_mb4_vs_mb2.py` CSV
  新增列。

历史归档与负结果详见 `doc/experiment_archive.md` 与
`doc/branch_consolidation.md`。
