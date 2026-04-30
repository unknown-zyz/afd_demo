# 04. 部署说明

本文说明 GPU local、GPU multinode 和 Ascend 910C NPU 的部署 / 运行方式。

## 1. 通用准备

```bash
git clone <repo>
cd afd_demo
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

确认模型路径：

```bash
# GPU
export MODEL_PATH=/data/Qwen/Qwen3-30B-A3B/

# NPU
export MODEL_NAME=/models/Qwen3-30B-A3B
```

## 2. GPU 本地运行

本地 GPU 单机使用 `scripts/run_single.sh local`。典型命令：

```bash
# 串行基线
./scripts/run_single.sh local 4 128 --tokens 20 --no-dbo --generate

# 预填充 DBO
./scripts/run_single.sh local 4 128 --tokens 20

# 解码 DBO
./scripts/run_single.sh local 4 128 --tokens 20 --generate

# 解码跨层流水
./scripts/run_single.sh local 4 128 --tokens 20 --generate --crosslayer
```

运行前建议检查资源：

```bash
nvidia-smi
free -h
df -h
```

## 3. GPU 矩阵

```bash
./scripts/run_experiment_matrix.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 \
  --tokens 20
```

矩阵脚本会：

1. 为 serial cache 去重；
2. 自动加入 P2P warmup；
3. 在同一 `(mode, seq)` 遇到 OOM 后停止更大 batch；
4. 写入 `results/experiment_matrix_summary.csv`。

## 4. GPU 多机

多机 GPU 使用 `run_node.sh` 手动启动 Attention / FFN 节点。远端机器登录命令：

```bash
ssh zyz@192.168.5.32 -p 31310 -i ~/.ssh/id_rsa_second
```

示例流程：

```bash
# FFN 节点
bash scripts/run_node.sh ffn <master_ip> <port>

# Attention 节点
bash scripts/run_node.sh attention <master_ip> <port> --prompt "Hello"
```

注意：

- 两端代码版本、模型路径、Python 环境需要一致。
- `MASTER_ADDR` / port 必须互通。
- 如果只做单机矩阵实验，优先使用 `run_single.sh local`。

## 5. Ascend 910C NPU 工作流

NPU 使用长期容器：

```bash
ssh schedTeam@1.95.114.229 -p 22 -i ~/.ssh/id_rsa_second
docker exec -it afd-npu-test bash
```

不要删除 `afd-npu-test`，它是共享的长期测试容器。

容器内：

```bash
cd /workspace/afd_demo_npu_rerun_20260429
export MODEL_NAME=/models/Qwen3-30B-A3B
```

单配置：

```bash
# 串行
bash scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 4 --seq 128 --tokens 20 --model-name "$MODEL_NAME" --no-dbo

# 预填充 DBO
bash scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 4 --seq 128 --tokens 20 --model-name "$MODEL_NAME" --no-generate

# 解码 DBO / 跨层流水
bash scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 4 --seq 128 --tokens 20 --model-name "$MODEL_NAME"
bash scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 4 --seq 128 --tokens 20 --model-name "$MODEL_NAME" --crosslayer
```

NPU matrix：

```bash
./scripts/run_experiment_matrix_npu.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64,128,256 \
  --seqs 128,256,512 \
  --tokens 20 \
  --no-cache
```

关键环境变量：

| 变量 | 含义 |
|---|---|
| `ASCEND_VISIBLE_DEVICES` / `--visible-devs` | 可见 NPU chip pool。 |
| `ATTN_DEVICES` / `--attn-devs` | Attention rank 使用的 device pool。 |
| `FFN_DEVICES` / `--ffn-devs` | FFN rank 使用的 device pool。 |
| `HCCL_BUFFSIZE` | HCCL 通信 buffer，单位 MB。 |
| `HCCL_CONNECT_TIMEOUT`、`HCCL_EXEC_TIMEOUT` | HCCL 连接 / 执行超时。 |

当前验证拓扑：

```text
attn_size=1
ffn_size=1
ffn_tp_size=1
active_world_size=2
```

## 6. 结果复制与画图

NPU 容器中可能没有 `matplotlib`。可以把 `results_npu/` 拷回本地后画图：

```bash
python scripts/plot_all_pipelines.py --root results_npu
python scripts/audit_experiment_baselines.py --root results_npu --output-csv results_npu/baseline_audit.csv
```

## 7. 故障处理

| 现象 | 处理 |
|---|---|
| 启动很慢 | Qwen3-30B-A3B 权重加载、进程启动和 warmup 不计入 scheduler timing。 |
| CUDA / NPU OOM | 记录配置边界；矩阵脚本会停止同一 `(mode, seq)` 更大 batch。 |
| NPU rank 挂住 | 先确认对端日志是否 OOM，再只 kill 对应 stuck peer PID。 |
| Speedup 缺失或可疑 | 运行 `audit_experiment_baselines.py`，检查 `prefill_ms` / `decode_tpot_ms`。 |
| HCCL timeout | 检查可见设备、旧 rank 进程和 `MASTER_PORT` 冲突。 |
