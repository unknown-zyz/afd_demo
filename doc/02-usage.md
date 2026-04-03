# AFD Demo 使用指南

## 1. 快速开始

### 1.1 环境准备

```bash
# 克隆项目
cd /path/to/afd_demo

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

**版本要求**:
- Python 3.10+
- CUDA 12.6+ runtime
- PyTorch 2.7.0 (cu126)
- Transformers 5.4.0

### 1.2 单元测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_pipeline.py -v
pytest tests/test_pipeline.py::TestMicroBatch::test_micro_batch_creation -v
```

### 1.3 快速验证

```bash
# 单机测试（需要 2+ GPU）
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m src.main \
  --local-test --model-name Qwen/Qwen2-1.5B \
  --prompt "Hello" --max-new-tokens 10
```

---

## 2. 运行脚本

### 2.1 单机测试

```bash
./scripts/test_local.sh [max_tokens] [batch_size]
```

**参数**:
- `max_tokens` - 生成 token 数量（默认: 5）
- `batch_size` - 批大小（默认: 1）

**示例**:
```bash
./scripts/test_local.sh           # 默认: 5 tokens, batch=1
./scripts/test_local.sh 10 2      # 10 tokens, batch=2
```

**GPU 分配** (需要 4 GPU):
- Attention 节点: GPU 0, 1
- FFN 节点: GPU 2, 3

### 2.2 多机测试

```bash
./scripts/test_multinode.sh [max_tokens] [batch_size]
```

**节点分配**:
- 本地机器 (Attention): 10.244.64.179, GPU 0,1
- 远程机器 (FFN): 192.168.5.32, GPU 0,1

**远程机器访问**:
```bash
ssh zyz@192.168.5.32 -p 31310 -i ~/.ssh/id_rsa_second
```

**示例**:
```bash
./scripts/test_multinode.sh       # 默认: 3 tokens, batch=1
./scripts/test_multinode.sh 10 1  # 10 tokens, batch=1
```

**注意**: 多机测试需要在两台机器上分别配置虚拟环境。

### 2.3 手动启动节点

适用于自定义配置或调试：

```bash
# 步骤 1: 在远程机器启动 FFN 节点
ssh zyz@192.168.5.32 -p 31310 -i ~/.ssh/id_rsa_second
cd /path/to/afd_demo
./scripts/run_ffn_node.sh <master_addr> <master_port>

# 步骤 2: 在本地机器启动 Attention 节点
./scripts/run_attn_node.sh <master_addr> <master_port> [options]
```

**示例**:
```bash
# 远程机器 (FFN 节点)
./scripts/run_ffn_node.sh 10.244.64.179 29500

# 本地机器 (Attention 节点)
./scripts/run_attn_node.sh 10.244.64.179 29500 \
  --prompt "Hello, how are you?" \
  --max-new-tokens 20

# 禁用 DBO（仅 AF 分离）
./scripts/run_attn_node.sh 10.244.64.179 29500 \
  --no-dbo --prompt "Hello"
```

---

## 3. 命令行参数

### 3.1 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-name` | 模型路径或 HuggingFace ID | `/data/Qwen/Qwen3-30B-A3B/` |
| `--max-seq-len` | 最大序列长度 | 4096 |

### 3.2 生成参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--prompt` | 输入提示词 | "Hello" |
| `--max-new-tokens` | 最大生成 token 数 | 5 |
| `--temperature` | 采样温度 | 0.7 |
| `--top-k` | Top-k 采样 | 50 |
| `--top-p` | Nucleus 采样 | 0.9 |
| `--greedy` | 使用贪婪解码 | False |

### 3.3 DBO 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--batch-size` | 总批大小 | 1 |
| `--num-micro-batches` | Micro-batch 数量 | 2 |
| `--no-dbo` | 禁用 DBO（仅 AF 分离） | False |

**DBO 模式说明**:
- **默认模式（DBO 开启）**: Prefill 和 Decode 都使用 Dual Batch Overlap
- **AF 分离模式（`--no-dbo`）**: 仅使用 Attention-FFN 分离，无异步重叠

### 3.4 其他参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--local-test` | 单机测试模式 | False |
| `--verbose` | 详细输出 | False |
| `--timing` | 启用计时统计 | False |

---

## 4. 基准测试

### 4.1 Prefill DBO 基准测试

```bash
./scripts/benchmark_dbo.sh [max_tokens] [batch_size] [dbo_mode]
```

**参数**:
- `max_tokens` - 生成 token 数（默认: 30）
- `batch_size` - 批大小（默认: 1）
- `dbo_mode` - `on` 或 `off`（默认: on）

**示例**:
```bash
./scripts/benchmark_dbo.sh                 # 默认配置
./scripts/benchmark_dbo.sh 50 2 on         # DBO ON
./scripts/benchmark_dbo.sh 50 2 off        # DBO OFF
```

**输出**:
- 日志: `results/prefill_dbo/benchmark_*.log`
- JSON: `results/prefill_dbo/timing_*.json`

---

## 5. 可视化

### 5.1 DBO Pipeline 时间线图

```bash
# 生成 Prefill DBO Pipeline 时间线
python scripts/visualize_dbo_pipeline.py \
  --attn-timing results/prefill_dbo/timing_attention_local_b4_s128_t5.json \
  --ffn-timing results/prefill_dbo/timing_ffn_local_b4_s128_t5.json \
  --output results/prefill_dbo/dbo_pipeline_local_b4_s128_t5.png \
  --start-layer 1 --num-layers 5
```

**输入**: `timing_attention_*.json`, `timing_ffn_*.json`  
**输出**: PNG Pipeline 时间线图

---

## 6. 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MODEL_NAME` | 模型路径 | `/data/Qwen/Qwen3-30B-A3B/` |
| `MASTER_ADDR` | Master 节点地址 | `127.0.0.1` |
| `MASTER_PORT` | Master 节点端口 | `29600` |
| `CUDA_VISIBLE_DEVICES` | 可用 GPU | `0,1` |
| `HF_ENDPOINT` | HuggingFace 镜像 | `https://hf-mirror.com` |
| `NCCL_SOCKET_TIMEOUT` | NCCL 超时（毫秒） | 600000 |

**示例**:
```bash
# 指定不同的模型
export MODEL_NAME=Qwen/Qwen2-1.5B

# 使用特定 GPU
export CUDA_VISIBLE_DEVICES=2,3

# 增加 NCCL 超时（多机环境）
export NCCL_SOCKET_TIMEOUT=1800000
```

---

## 7. 故障排查

### 7.1 NCCL 超时

**问题**: `NCCL error: connection timeout`

**解决**:
```bash
export NCCL_SOCKET_TIMEOUT=600000
export NCCL_DEBUG=INFO  # 启用详细日志
```

### 7.2 内存不足 (OOM)

**问题**: `torch.cuda.OutOfMemoryError`

**解决**:
- 确保每个节点有足够 GPU 内存（Qwen3-30B 需 2×32GB）
- 减小 `--batch-size`
- 减小 `--max-seq-len`
- 减小 `--max-new-tokens`

### 7.3 远程连接失败

**问题**: SSH 或 NCCL 连接远程机器失败

**排查**:
```bash
# 测试 SSH 连接
ssh -o ConnectTimeout=5 -p 31310 -i ~/.ssh/id_rsa_second \
  zyz@192.168.5.32 "echo OK"

# 测试网络连通性
ping 10.244.64.179

# 测试端口可达性
nc -zv 10.244.64.179 29500
```

### 7.4 进程退出警告

**现象**:
```
WARNING: destroy_process_group() was not called before program exit
```

**说明**: 这是 PyTorch 2.7 + NCCL 2.26 的已知问题，不影响运行结果，可以忽略。

### 7.5 KV Cache Shape 不匹配

**问题**: `RuntimeError: attention mask shape mismatch`

**原因**: Decode DBO 在 batch > 1 时存在 KV Cache 切片 bug

**状态**: 已修复（commit c3aba29），但存在性能问题

**建议**: 当前版本建议使用 `--no-dbo` 禁用 Decode DBO

---

## 8. 性能优化建议

### 8.1 Prefill 阶段

✅ **推荐使用 DBO**:
- 单机效率: 54.8% (Attention), 71.2% (FFN)
- 多机预期效率更高

**最佳配置**:
```bash
# 默认即可（DBO 已开启）
./scripts/test_local.sh 30 4  # batch=4 效果较好
```

### 8.2 Decode 阶段

❌ **当前不推荐使用 DBO**:
- 性能倒退 -44% ~ -46% (batch >= 4)
- 根因: KV Cache 对象创建开销

**推荐配置**:
```bash
# 禁用 DBO
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m src.main \
  --local-test --no-dbo --max-new-tokens 50
```

### 8.3 资源监控

```bash
# 监控 GPU 使用情况
watch -n 1 nvidia-smi

# 监控内存使用
free -h

# 监控进程
htop
```

---

## 9. 常见任务

### 9.1 对比 DBO 效果

```bash
# DBO ON
./scripts/benchmark_dbo.sh 50 4 on

# DBO OFF  
./scripts/benchmark_dbo.sh 50 4 off

# 对比结果
diff results/prefill_dbo/*_on.log results/prefill_dbo/*_off.log
```

### 9.2 测试新模型

```bash
# 下载模型（使用 HF 镜像）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2-7B --local-dir /path/to/model

# 运行测试
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m src.main \
  --local-test \
  --model-name /path/to/model \
  --prompt "Hello" \
  --max-new-tokens 20
```

### 9.3 调试通信问题

```bash
# 启用 NCCL 详细日志
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 运行测试
./scripts/test_local.sh

# 查看日志
tail -f nccl_debug.log
```

---

## 10. 相关文档

- [架构设计](01-architecture.md) - 系统架构和 DBO 实现
- [API 参考](03-api-reference.md) - 代码接口文档
- [部署指南](04-deployment.md) - 环境配置和资源要求
- [实验结果](../results/README.md) - 测试数据和分析
