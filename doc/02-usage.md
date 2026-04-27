# 使用指南

## 1. 环境准备

```bash
cd /path/to/afd_demo
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
```

默认模型：

```bash
export MODEL_PATH=/data/Qwen/Qwen3-30B-A3B/
```

## 2. 单配置运行

统一入口：

```bash
./scripts/run_single.sh <local|multinode> <batch> <seq> [options]
```

常用 options：

| 参数 | 说明 |
|---|---|
| `--tokens N` | decode token 数，默认 5 |
| `--no-dbo` | 关闭 DBO，作为 serial baseline |
| `--generate` | 跑 prefill + autoregressive decode；默认只跑 prefill |
| `--crosslayer` | decode DBO 启用 cross-layer pipeline |
| `--visualize` | 同步生成单张 pipeline PNG |
| `--warmup-p2p --warmup-rounds N` | 预热 NCCL P2P，降低冷启动污染 |
| `--verbose` | 输出详细日志 |

示例：

```bash
# Prefill DBO
./scripts/run_single.sh local 8 128 --tokens 20

# Serial prefill baseline
./scripts/run_single.sh local 8 128 --tokens 20 --no-dbo

# Decode DBO
./scripts/run_single.sh local 8 128 --tokens 20 --generate

# Decode DBO + cross-layer pipeline
./scripts/run_single.sh local 8 128 --tokens 20 --generate --crosslayer

# 自动画图
./scripts/run_single.sh local 4 128 --tokens 20 --visualize
```

本地模式默认将 Attention 放在 GPU 0,1，FFN 放在 GPU 2,3。

## 3. 矩阵实验

```bash
# 默认全模式：serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer
./scripts/run_experiment_matrix.sh

# 指定子集
./scripts/run_experiment_matrix.sh \
  --modes serial,prefill-dbo \
  --batches 2,4,8 \
  --seqs 128,256 \
  --tokens 20

# dry-run
./scripts/run_experiment_matrix.sh --dry-run --modes serial --batches 2 --seqs 128
```

输出：

| 目录 | 内容 |
|---|---|
| `results/serial/` | serial timing/report/cache |
| `results/prefill-dbo/` | prefill DBO timing/report |
| `results/decode-dbo/` | decode DBO timing/report |
| `results/decode-dbo-crosslayer/` | cross-layer decode timing/report |
| `results/experiment_matrix_summary.csv` | 实验摘要 |

## 4. 可视化和报告

```bash
# timing JSON -> 单张 Gantt 图
python scripts/visualize_dbo_pipeline.py \
  --attn-timing results/prefill-dbo/timing_attention_prefill-dbo_b8_s128_t20.json \
  --ffn-timing results/prefill-dbo/timing_ffn_prefill-dbo_b8_s128_t20.json \
  --output results/prefill-dbo/pipeline_prefill-dbo_b8_s128_t20.png \
  --start-layer 1 --num-layers 4

# 批量重画
python scripts/plot_all_pipelines.py --root results
```

`scripts/gen_experiment_report.py` 由运行脚本自动调用，也可手动从一对 timing JSON 生成报告。

## 5. 多机运行

自动模式：

```bash
./scripts/run_single.sh multinode 8 128 --tokens 20
```

手动模式：

```bash
# 远程 / FFN 节点
./scripts/run_node.sh ffn <master_ip> 29500

# 本地 / Attention 节点
./scripts/run_node.sh attention <master_ip> 29500 \
  --batch-size 8 --prefill-seq-len 128 --max-new-tokens 20
```

当前远程机器登录命令：

```bash
ssh zyz@192.168.5.32 -p 31310 -i ~/.ssh/id_rsa_second
```

## 6. 直接调用 Python 入口

脚本最终都会调用：

```bash
python -u -m src.main [options]
```

关键参数：

| 参数 | 说明 |
|---|---|
| `--role attention|ffn|auto` | 当前 rank 角色 |
| `--master-addr` / `--master-port` | 分布式初始化地址 |
| `--batch-size` | batch size |
| `--prefill-seq-len` | 固定 prefill 长度 |
| `--num-micro-batches` | DBO micro-batch 数 |
| `--no-dbo` | 关闭 DBO |
| `--no-generate` | 只跑 prefill |
| `--timing --timing-suffix S` | 保存 timing JSON |
| `--timing-mode cuda_events|sync` | timing 方式，默认 `cuda_events` |
| `--warmup-p2p --warmup-rounds N` | NCCL P2P 预热 |

## 7. 故障排查

| 现象 | 处理 |
|---|---|
| CUDA OOM | 降低 batch/seq；检查 `results/experiment_matrix_summary.csv` 中的 OOM 边界 |
| NCCL 首次通信很慢 | 使用 `--warmup-p2p --warmup-rounds 5` |
| 多机连接失败 | 检查 SSH、`MASTER_ADDR` 可达性、端口 29500-29600 |
| 退出时 NCCL warning | 已知退出清理问题，结果已写出即可忽略 |
| speedup 为 N/A | serial cache 缺 baseline；先跑对应 `--no-dbo` 配置 |

## 8. 相关文档

- [架构设计](01-architecture.md)
- [API 参考](03-api-reference.md)
- [部署指南](04-deployment.md)
- [实验结果](../results/README.md)
