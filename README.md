# AFD Demo: Attention-FFN Disaggregation + DBO

AFD Demo 是一个用于研究 **Attention-FFN 分离推理** 和 **Dual Batch Overlap (DBO)** 流水线调度的实验仓库。当前主模型是 Qwen3-30B-A3B；本地 4 GPU 模式用两组 GPU 模拟 Attention 节点和 FFN 节点。

## 当前能力

| 能力 | 状态 | 入口 |
|---|---|---|
| Attention/FFN 分离推理 | 可用 | `src/main.py` |
| Prefill DBO | 可用，主要优化路径 | `AsyncPipelineScheduler` |
| Decode DBO | 可用，需按结果判断是否启用 | `DecodeDBOScheduler` |
| Decode cross-layer pipeline | 可用，实验性 | `--crosslayer` |
| KV cache / 自回归生成 | 可用 | HuggingFace `DynamicCache` |
| Pipeline timing / Gantt 图 | 可用 | `scripts/visualize_dbo_pipeline.py` |

## 环境

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

主要依赖见 `requirements.txt`：Python 3.10+、PyTorch 2.7.0、Transformers 5.4.0、Accelerate、matplotlib、pytest。

默认模型路径由脚本环境变量控制：

```bash
export MODEL_PATH=/data/Qwen/Qwen3-30B-A3B/
```

## 快速验证

```bash
source venv/bin/activate
pytest tests/ -q

# 本地 4 GPU，prefill DBO，batch=4，seq=128
./scripts/run_single.sh local 4 128 --tokens 20

# 串行 baseline
./scripts/run_single.sh local 4 128 --tokens 20 --no-dbo

# decode DBO / cross-layer decode
./scripts/run_single.sh local 4 128 --tokens 20 --generate
./scripts/run_single.sh local 4 128 --tokens 20 --generate --crosslayer
```

`run_single.sh local` 默认使用：

| 角色 | GPU |
|---|---|
| Attention | `CUDA_VISIBLE_DEVICES=0,1` |
| FFN | `CUDA_VISIBLE_DEVICES=2,3` |

## 实验矩阵

```bash
# 默认扫 serial / prefill-dbo / decode-dbo / decode-dbo-crosslayer
./scripts/run_experiment_matrix.sh

# 只跑一个小矩阵
./scripts/run_experiment_matrix.sh --modes serial,prefill-dbo --batches 2,4 --seqs 128

# 只打印命令
./scripts/run_experiment_matrix.sh --dry-run --modes serial --batches 2 --seqs 128
```

输出目录：

| 目录 | 内容 |
|---|---|
| `results/serial/` | 串行 baseline timing/report/cache |
| `results/prefill-dbo/` | prefill DBO timing/report/PNG |
| `results/decode-dbo/` | decode DBO timing/report/PNG |
| `results/decode-dbo-crosslayer/` | cross-layer decode timing/report/PNG |

## 可视化

```bash
# 单张图
python scripts/visualize_dbo_pipeline.py \
  --attn-timing results/prefill-dbo/timing_attention_prefill-dbo_b4_s128_t20.json \
  --ffn-timing results/prefill-dbo/timing_ffn_prefill-dbo_b4_s128_t20.json \
  --output /tmp/pipeline.png \
  --start-layer 1 --num-layers 4

# 批量重画 results 下已有 timing JSON
python scripts/plot_all_pipelines.py --root results
```

## 多机运行

自动方式：

```bash
./scripts/run_single.sh multinode 4 128 --tokens 20
```

手动方式：

```bash
# FFN 节点
./scripts/run_node.sh ffn <master_ip> 29500

# Attention 节点
./scripts/run_node.sh attention <master_ip> 29500 --batch-size 4 --prefill-seq-len 128
```

默认远程节点信息在 `scripts/run_single.sh` 中配置，当前 SSH 命令为：

```bash
ssh zyz@192.168.5.32 -p 31310 -i ~/.ssh/id_rsa_second
```

## 项目结构

```text
src/
  main.py                    # CLI 入口；prefill/generate 分发；timing 落盘
  distributed/               # 分布式上下文、NCCL P2P、warmup
  model/                     # DisaggregatedQwenModel、AttentionWorker、FFNWorker
  pipeline/                  # Simple / prefill DBO / decode DBO 调度器
  utils/                     # timing、sampling、profiling、tensor validation
scripts/
  run_single.sh              # 单配置运行器
  run_experiment_matrix.sh   # 批量矩阵运行器
  gen_experiment_report.py   # timing JSON -> markdown report
  visualize_dbo_pipeline.py  # 单张 Gantt 图
  plot_all_pipelines.py      # 批量重画 Gantt 图
  capture_serial_prefill.sh  # 补充 serial cache 的 prefill_ms
doc/
  01-architecture.md
  02-usage.md
  03-api-reference.md
  04-deployment.md
  05-code-review-guide.md
results/
  README.md                  # 实验产物说明
```

## 文档

- `doc/01-architecture.md`：系统结构和调度模型
- `doc/02-usage.md`：CLI、脚本和实验运行
- `doc/03-api-reference.md`：当前代码 API
- `doc/04-deployment.md`：本地/多机部署
- `doc/05-code-review-guide.md`：面向 review 的关键路径说明

## 已知注意事项

1. `--warmup-p2p --warmup-rounds N` 用于消除 NCCL P2P 冷启动；已删除历史 keepalive 路径。
2. Decode DBO 的收益依赖 batch/seq/网络和 cross-layer 设置，必须与 serial baseline 对比。
3. PyTorch + NCCL 退出阶段可能有 `destroy_process_group` 警告；代码选择让进程退出时自然释放 NCCL 资源。

## 参考

- [vLLM AFD #22799](https://github.com/vllm-project/vllm/issues/22799)
- [vLLM DBO #23693](https://github.com/vllm-project/vllm/pull/23693)
