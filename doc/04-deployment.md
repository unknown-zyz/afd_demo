# 部署指南

## 1. 资源要求

| 场景 | 推荐资源 |
|---|---|
| 本地开发 | 4 × V100-32GB 或同级 GPU |
| 最小功能验证 | 2 × 具备足够显存的 GPU |
| 多机实验 | 2 节点，每节点 2 × GPU，节点间网络互通 |

Qwen3-30B-A3B 的 FFN/MoE 权重占主要显存；本地脚本默认把 Attention 放在 GPU 0,1，FFN 放在 GPU 2,3。

## 2. 软件环境

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
```

模型路径：

```bash
export MODEL_PATH=/data/Qwen/Qwen3-30B-A3B/
```

## 3. 本地部署

```bash
# prefill DBO
./scripts/run_single.sh local 4 128 --tokens 20

# serial baseline
./scripts/run_single.sh local 4 128 --tokens 20 --no-dbo

# decode / cross-layer decode
./scripts/run_single.sh local 4 128 --tokens 20 --generate
./scripts/run_single.sh local 4 128 --tokens 20 --generate --crosslayer
```

建议测试前检查资源：

```bash
bash .github/skills/testing-workflow/check_resources.sh
```

## 4. 多机部署

### 4.1 网络与 SSH

确保本地 Attention 节点和远程 FFN 节点网络互通，端口 29500-29600 可用。

当前远程机器：

```bash
ssh zyz@192.168.5.32 -p 31310 -i ~/.ssh/id_rsa_second
```

两台机器都需要相同代码、虚拟环境和模型路径。

### 4.2 自动启动

在本地执行：

```bash
./scripts/run_single.sh multinode 4 128 --tokens 20
```

`run_single.sh` 会通过 SSH 启动远程 FFN 节点，再在本地运行 Attention 节点，并把 FFN timing JSON 拉回本地。

### 4.3 手动启动

远程 / FFN 节点：

```bash
cd /path/to/afd_demo
source venv/bin/activate
./scripts/run_node.sh ffn <local_master_ip> 29500
```

本地 / Attention 节点：

```bash
cd /path/to/afd_demo
source venv/bin/activate
./scripts/run_node.sh attention <local_master_ip> 29500 \
  --batch-size 4 \
  --prefill-seq-len 128 \
  --max-new-tokens 20
```

## 5. 实验矩阵

```bash
./scripts/run_experiment_matrix.sh --deployment local
./scripts/run_experiment_matrix.sh --deployment multinode --modes serial,prefill-dbo
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--modes` | `serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer` 子集 |
| `--batches` | 逗号分隔 batch 列表 |
| `--seqs` | 逗号分隔 prefill seq 列表 |
| `--tokens` | decode tokens |
| `--no-cache` | 强制重跑 serial baseline |
| `--dry-run` | 只打印命令 |

## 6. 性能稳定性建议

```bash
export NCCL_BUFFSIZE=33554432
export NCCL_NCHANNELS_PER_NET_PEER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

这些变量已在 `run_single.sh` 中默认设置。长矩阵实验建议保留 `--warmup-p2p --warmup-rounds 5`，`run_experiment_matrix.sh` 会自动加上。

## 7. 故障排查

| 问题 | 建议 |
|---|---|
| OOM | 降低 batch/seq；矩阵脚本遇到 OOM 会停止该 seq 的更大 batch |
| NCCL connection timeout | 检查 `MASTER_ADDR`、端口、防火墙、SSH 隧道 |
| 首次 P2P 慢 | 使用 warmup；不要恢复历史 keepalive 路径 |
| 远程 timing 缺失 | 检查远程 `results/prefill_dbo/logs/ffn_*.log` |
| 退出 warning | NCCL 退出清理 warning 不影响已写出的结果 |

## 8. 结果同步与画图

```bash
python scripts/plot_all_pipelines.py --root results
```

结果目录结构见 `results/README.md`。
