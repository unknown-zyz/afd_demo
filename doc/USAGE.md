# AFD Demo 使用指南

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 单元测试

```bash
pytest tests/ -v
```

---

## 运行脚本

### 单机测试 (需要 4 GPU)

```bash
./scripts/test_local.sh [max_tokens] [batch_size]

# 示例
./scripts/test_local.sh           # 默认: 5 tokens, batch_size=1
./scripts/test_local.sh 10 2      # 10 tokens, batch_size=2
```

**GPU 分配**:
- Attention 节点: GPU 0, 1
- FFN 节点: GPU 2, 3

### 多机测试

```bash
./scripts/test_multinode.sh [max_tokens] [batch_size]

# 示例
./scripts/test_multinode.sh       # 默认: 3 tokens, batch_size=1
./scripts/test_multinode.sh 10 1  # 10 tokens, batch_size=1
```

**节点分配**:
- 本地 (Attention): 10.244.64.179, GPU 0,1
- 远程 (FFN): 192.168.5.32, GPU 0,1

**远程机器 SSH**:
```bash
ssh zyz@192.168.5.32 -p 31310 -i ~/.ssh/id_rsa_second
```

### 手动启动节点

适用于自定义配置或调试:

```bash
# 在远程机器启动 FFN 节点
./scripts/run_ffn_node.sh <master_addr> <master_port> [options]

# 在本地机器启动 Attention 节点
./scripts/run_attn_node.sh <master_addr> <master_port> [options]
```

**示例**:
```bash
# 远程 (FFN)
./scripts/run_ffn_node.sh 10.244.64.179 29500 --prompt "Hello"

# 本地 (Attention)
./scripts/run_attn_node.sh 10.244.64.179 29500 --prompt "Hello" --max-new-tokens 10
```

---

## 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-name` | 模型路径 | `/data/Qwen/Qwen3-30B-A3B/` |
| `--prompt` | 输入提示词 | "Hello" |
| `--max-new-tokens` | 最大生成 token 数 | 5 |
| `--batch-size` | 批大小 | 1 |
| `--greedy` | 使用贪婪解码 | False |
| `--no-decode-dbo` | 禁用 Decode DBO | False |
| `--verbose` | 详细输出 | False |
| `--timing` | 启用计时统计 | False |

---

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MODEL_NAME` | 模型路径 | `/data/Qwen/Qwen3-30B-A3B/` |
| `MASTER_ADDR` | Master 节点地址 | `127.0.0.1` |
| `MASTER_PORT` | Master 节点端口 | `29600` |
| `CUDA_DEVICES` | 可用 GPU | `0,1` |
| `HF_ENDPOINT` | HuggingFace 镜像 | `https://hf-mirror.com` |

---

## 故障排查

### 1. NCCL 超时

```bash
# 增加 NCCL 超时时间
export NCCL_SOCKET_TIMEOUT=600000
```

### 2. 内存不足 (OOM)

- 确保每个节点有 2 × 32GB GPU
- 减小 `--batch-size`
- 减小 `--max-seq-len`

### 3. 远程连接失败

```bash
# 测试 SSH 连接
ssh -o ConnectTimeout=5 -p 31310 -i ~/.ssh/id_rsa_second zyz@192.168.5.32 "echo OK"

# 测试网络连通性
ping 10.244.64.179
```

### 4. 进程退出警告

PyTorch 2.7 + NCCL 2.26 在进程退出时可能显示警告,这是已知问题,不影响运行结果:
```
WARNING: destroy_process_group() was not called before program exit
```

---

## 文件结构

```
scripts/
├── test_local.sh        # 单机测试脚本
├── test_multinode.sh    # 多机测试脚本
├── run_attn_node.sh     # 手动启动 Attention 节点
├── run_ffn_node.sh      # 手动启动 FFN 节点
└── visualize_dbo.py     # DBO 时间线可视化
```

---

## 相关文档

- [设计文档](design.md) - 架构设计和实现细节
- [测试报告](test_report.md) - 测试结果和性能数据
