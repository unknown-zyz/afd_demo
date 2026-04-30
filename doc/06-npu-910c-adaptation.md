# 06. Ascend NPU-910C 适配

本文说明 `npu` 分支上的 NPU/HCCL 适配。最新 910C 结果位于 `results_npu/`；
覆盖率、speedup 和 OOM 边界见 [08-gpu-npu-experiment-summary.md](08-gpu-npu-experiment-summary.md)。

## 1. 已验证环境

使用长期容器：

```bash
ssh schedTeam@1.95.114.229 -p 22 -i ~/.ssh/id_rsa_second
docker exec -it afd-npu-test bash
```

不要删除 `afd-npu-test`。

容器内已验证：

```text
torch 2.6.0+cpu
torch_npu 2.6.0
torch_npu.npu.is_available() == True
MODEL_NAME=/models/Qwen3-30B-A3B
```

## 2. 后端差异

| 事项 | CUDA 分支 | NPU 分支 |
|---|---|---|
| Accelerator API | `torch.cuda` | `torch.npu` / `torch_npu` |
| Distributed backend | NCCL | HCCL |
| 单次脚本 | `scripts/run_single.sh` | `scripts/run_npu.sh` |
| 矩阵脚本 | `scripts/run_experiment_matrix.sh` | `scripts/run_experiment_matrix_npu.sh` |
| 结果目录 | `results/` | `results_npu/` |

设备抽象位于 `src/utils/device.py`。`src/main.py` 会在模型与 scheduler 构造前
初始化 backend。

## 3. 单配置命令

```bash
# 串行
bash scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 4 --seq 128 --tokens 20 --model-name "$MODEL_NAME" --no-dbo

# 预填充 DBO
bash scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 4 --seq 128 --tokens 20 --model-name "$MODEL_NAME" --no-generate

# 解码 DBO
bash scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 4 --seq 128 --tokens 20 --model-name "$MODEL_NAME"

# 解码跨层流水
bash scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 4 --seq 128 --tokens 20 --model-name "$MODEL_NAME" --crosslayer
```

矩阵：

```bash
./scripts/run_experiment_matrix_npu.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64,128,256 \
  --seqs 128,256,512 \
  --tokens 20 \
  --no-cache
```

脚本会把 `run_npu.sh` 的中间 timing 从 `results/prefill_dbo/` 移动到
`results_npu/{serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer}/`，并写入
`results_npu/experiment_matrix_summary.csv`。

## 4. HCCL 与设备变量

| 变量 / 参数 | 含义 |
|---|---|
| `ASCEND_VISIBLE_DEVICES` / `--visible-devs` | 可见 NPU chip pool。 |
| `ATTN_DEVICES` / `--attn-devs` | Attention rank 的设备池。 |
| `FFN_DEVICES` / `--ffn-devs` | FFN rank 的设备池。 |
| `HCCL_BUFFSIZE` | HCCL 通信 buffer，单位 MB。 |
| `HCCL_CONNECT_TIMEOUT` | HCCL rendezvous 超时。 |
| `HCCL_EXEC_TIMEOUT` | HCCL 执行超时。 |

`scripts/run_npu.sh` 会清理 NCCL-only 环境变量，因为它们不适用于 HCCL。

## 5. 验证拓扑与限制

Fresh matrix 使用：

```text
attn_size=1
ffn_size=1
ffn_tp_size=1
active_world_size=2
visible_chip_pool=16
```

已知限制：

1. 旧的 4-rank preset（如 `attn_size=2`、`ffn_size=2`、`ffn_tp_size=2`）仍是
   scaffolding；除非单独修复 device mapping / memory 问题，否则不是当前验证路径。
2. `ffn_tp_size > 1` 不是当前 fresh matrix 的验证路径。
3. `attn_size > 1` 不是当前 fresh matrix 的验证路径。
4. NVSHMEM 只适用于 CUDA 路径；NPU 使用 HCCL P2P / collective。
5. 当前 `torch_npu 2.6.0` 栈不支持 FP8 compute；FP8 tensor 可以存储，但 NPU
   上的 `mm`、cast 等操作不可用。

## 6. NPU 预热

Ascend/HCCL 在首次 prefill 时可能出现按 shape 编译 / JIT 开销。`src/main.py`
支持 `--prefill-warmup-rounds`；NPU 默认会进行一次未计时 prefill warmup，CUDA/CPU
默认不做，除非显式指定。

这样可以避免把编译开销混入 layer timing。

## 7. 最新 NPU 结果

| 模式 | OK | OOM |
|---|---:|---:|
| `serial` | 45 | 5 |
| `prefill-dbo` | 25 | 8 |
| `decode-dbo` | 45 | 1 |
| `decode-dbo-crosslayer` | 45 | 1 |

NPU baseline audit：115 / 115 条有效 DBO 行均为 `ok`。

Pipeline 图是在结果拷回本地后生成的，因为 NPU 容器中没有安装 `matplotlib`。

## 8. 故障处理

| 现象 | 处理 |
|---|---|
| 一侧 rank OOM 后另一侧挂住 | 先检查两个 rank 日志；确认 OOM 后只 kill 对应 stuck peer PID。 |
| SSH 监控连接中断 | 减少并发 SSH 连接，稍后重试。 |
| 容器内画图失败 | 在本地 venv 使用 `matplotlib` 生成图。 |
| `Float8_e4m3fn has not been supported` | 当前 NPU 栈不支持 FP8 compute。 |
| HCCL timeout | 检查可见设备、旧进程和 `MASTER_PORT` 冲突。 |
