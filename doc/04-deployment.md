# AFD Demo 部署指南

## 1. 硬件要求

### 1.1 最小配置

**单机测试** (2 GPU):
- GPU: 2 × NVIDIA GPU (推荐 V100 32GB 或更高)
- 显存: 每个 GPU ≥ 16GB
- 内存: ≥ 32GB
- CUDA: 12.6+

**单机生产** (4 GPU):
- GPU: 4 × NVIDIA GPU (推荐 V100 32GB 或 A100 40GB)
- 显存: 每个 GPU ≥ 32GB (Qwen3-30B)
- 内存: ≥ 64GB
- CUDA: 12.6+
- NVLink: 推荐（降低通信延迟）

**多机部署** (2 节点):
- 每节点: 2 × GPU (V100 32GB 或更高)
- 显存: 每个 GPU ≥ 32GB
- 内存: 每节点 ≥ 64GB
- 网络: 千兆以太网（最低）, InfiniBand（推荐）

### 1.2 测试环境

| 节点 | GPU | 内存 | IP |
|------|-----|------|-----|
| 本地机器 | V100-32GB × 4 | 300GB | 10.244.64.179 |
| 远程机器 | V100-32GB × 4 | 300GB | 192.168.5.32 |

---

## 2. 软件环境

### 2.1 系统要求

- **操作系统**: Linux (Ubuntu 20.04+ 推荐)
- **Python**: 3.10+
- **CUDA Driver**: 支持 CUDA 12.6 runtime

### 2.2 Python 依赖

**核心依赖**:
- PyTorch 2.7.0 (cu126)
- Transformers 5.4.0
- Accelerate 1.13.0

**完整依赖**: 见 `requirements.txt`

### 2.3 版本兼容性

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | 已测试 3.10, 3.12 |
| PyTorch | 2.7.0 | CUDA 12.6 build |
| CUDA | 12.6+ | Runtime 版本 |
| NCCL | 2.26+ | PyTorch 自带 |
| Transformers | 5.4.0 | HuggingFace 库 |

---

## 3. 单机部署

### 3.1 安装步骤

**步骤 1: 克隆项目**
```bash
git clone <repository_url> /path/to/afd_demo
cd /path/to/afd_demo
```

**步骤 2: 创建虚拟环境**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

**步骤 3: 安装依赖**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**步骤 4: 验证安装**
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# 应输出: 2.7.0+cu126
#        True

pytest tests/ -v
```

### 3.2 GPU 分配

**4 GPU 配置** (推荐):
- Attention 节点: GPU 0, 1
- FFN 节点: GPU 2, 3

**2 GPU 配置** (最小):
- Attention 节点: GPU 0
- FFN 节点: GPU 1

### 3.3 运行测试

```bash
# 使用默认配置
./scripts/test_local.sh

# 自定义配置
./scripts/test_local.sh 10 2  # 10 tokens, batch=2
```

### 3.4 资源监控

```bash
# 监控 GPU 使用
watch -n 1 nvidia-smi

# 监控内存
watch -n 1 free -h

# 监控进程
htop
```

---

## 4. 多机部署

### 4.1 网络配置

**要求**:
- 节点间网络连通
- 防火墙开放必要端口（默认 29500-29600）
- SSH 免密登录（推荐）

**测试连通性**:
```bash
# 从本地 ping 远程
ping 192.168.5.32

# 测试端口
nc -zv 192.168.5.32 29500
```

### 4.2 SSH 配置

**生成密钥** (如果没有):
```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa_second
```

**复制公钥到远程**:
```bash
ssh-copy-id -i ~/.ssh/id_rsa_second.pub -p 31310 zyz@192.168.5.32
```

**测试连接**:
```bash
ssh -p 31310 -i ~/.ssh/id_rsa_second zyz@192.168.5.32 "echo OK"
```

### 4.3 远程机器配置

**步骤 1: SSH 登录远程机器**
```bash
ssh zyz@192.168.5.32 -p 31310 -i ~/.ssh/id_rsa_second
```

**步骤 2: 配置远程环境** (与单机相同)
```bash
cd /path/to/afd_demo
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**步骤 3: 验证 GPU**
```bash
nvidia-smi
```

### 4.4 启动多机测试

**方法 1: 使用测试脚本** (自动)
```bash
# 在本地机器运行
./scripts/test_multinode.sh [max_tokens] [batch_size]
```

**方法 2: 手动启动** (用于调试)
```bash
# 步骤 1: 在远程机器启动 FFN 节点
ssh zyz@192.168.5.32 -p 31310 -i ~/.ssh/id_rsa_second
cd /path/to/afd_demo
source venv/bin/activate
./scripts/run_node.sh ffn 10.244.64.179 29500

# 步骤 2: 在本地机器启动 Attention 节点
./scripts/run_node.sh attention 10.244.64.179 29500 \
  --prompt "Hello" --max-new-tokens 20
```

### 4.5 多机故障排查

**NCCL 超时**:
```bash
export NCCL_SOCKET_TIMEOUT=1800000  # 30 分钟
export NCCL_IB_TIMEOUT=50          # InfiniBand 超时
```

**防火墙问题**:
```bash
# 检查端口
sudo netstat -tuln | grep 29500

# 临时开放端口（Ubuntu）
sudo ufw allow 29500:29600/tcp
```

**网络延迟测试**:
```bash
python scripts/measure_comm_latency.py
```

---

## 5. 模型准备

### 5.1 模型下载

**使用 HuggingFace 镜像** (推荐):
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2-1.5B --local-dir /data/Qwen/Qwen2-1.5B
```

**直接下载** (需要访问 HuggingFace):
```bash
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir /data/Qwen/Qwen3-30B-A3B
```

### 5.2 模型存储位置

**推荐目录结构**:
```
/data/
└── Qwen/
    ├── Qwen2-1.5B/          # 小模型，用于测试
    │   ├── config.json
    │   ├── model.safetensors
    │   └── ...
    └── Qwen3-30B-A3B/       # MoE 模型
        ├── config.json
        ├── model-*.safetensors
        └── ...
```

**磁盘空间要求**:
- Qwen2-1.5B: ~3GB
- Qwen3-30B-A3B: ~56GB

### 5.3 模型格式

支持的格式：
- ✅ SafeTensors（推荐）
- ✅ PyTorch bin

---

## 6. 性能调优

### 6.1 环境变量

**CUDA 优化**:
```bash
export CUDA_LAUNCH_BLOCKING=0        # 异步执行
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"  # 内存分配
```

**NCCL 优化**:
```bash
export NCCL_DEBUG=WARN               # 日志级别
export NCCL_SOCKET_TIMEOUT=600000    # 超时（毫秒）
export NCCL_IB_DISABLE=0             # 启用 InfiniBand
export NCCL_P2P_DISABLE=0            # 启用 P2P
```

**OMP 优化**:
```bash
export OMP_NUM_THREADS=8             # OpenMP 线程数
export MKL_NUM_THREADS=8             # MKL 线程数
```

### 6.2 批大小调整

**Prefill 阶段**:
- 单机: batch=2-4 (推荐)
- 多机: batch=4-8 (可更大)

**Decode 阶段**:
- 当前建议: batch=1-2 (DBO 存在性能问题)
- 优化后: batch=4-8

### 6.3 显存优化

**减少显存占用**:
```bash
# 减小 batch size
--batch-size 1

# 减小序列长度
--max-seq-len 2048

# 使用 bfloat16
--dtype bfloat16
```

**检查显存使用**:
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

---

## 7. 资源要求汇总

### 7.1 Qwen2-1.5B

| 配置 | Attention 节点 | FFN 节点 | 总计 |
|------|---------------|---------|------|
| 模型显存 | ~1.5GB | ~1.5GB | ~3GB |
| 运行显存 (batch=1) | ~2GB | ~2GB | ~4GB |
| 运行显存 (batch=4) | ~4GB | ~4GB | ~8GB |
| 推荐 GPU | 1×16GB | 1×16GB | 2×16GB |

### 7.2 Qwen3-30B-A3B (MoE)

| 配置 | Attention 节点 | FFN 节点 | 总计 |
|------|---------------|---------|------|
| 模型显存 | ~3GB | ~39GB | ~42GB |
| 运行显存 (batch=1) | ~4GB | ~40GB | ~44GB |
| 运行显存 (batch=4) | ~6GB | ~42GB | ~48GB |
| 推荐 GPU | 2×16GB | 2×32GB | 2×16GB + 2×32GB |

---

## 8. 生产环境建议

### 8.1 稳定性

**进程管理**:
- 使用 systemd 或 supervisor 管理进程
- 配置自动重启策略
- 监控进程健康状态

**日志管理**:
- 启用详细日志 (`--verbose`)
- 配置日志轮转
- 监控错误日志

### 8.2 高可用

**负载均衡**:
- 部署多个 Attention-FFN 节点对
- 使用负载均衡器分发请求

**故障恢复**:
- 监控节点健康
- 自动故障切换
- 保存 checkpoint

### 8.3 监控指标

**关键指标**:
- GPU 利用率
- 显存使用率
- 吞吐量 (tokens/s)
- 延迟 (ms/token)
- 网络带宽

**监控工具**:
- nvidia-smi
- Prometheus + Grafana
- 自定义监控脚本

---

## 9. 安全建议

### 9.1 网络安全

- 使用防火墙限制访问
- 启用 SSH 密钥认证
- 定期更新密钥
- 使用 VPN 或私有网络

### 9.2 访问控制

- 限制模型文件访问权限
- 使用用户权限隔离
- 审计访问日志

---

## 10. 已知限制

### 10.1 当前限制

1. **节点数量**: 固定 2 个节点 (Attention + FFN)
2. **Decode DBO 性能**: batch >= 4 时性能倒退 -44%
3. **模型架构**: 仅支持标准 Transformer 和 MoE
4. **并发请求**: 当前不支持多请求并发

### 10.2 计划改进

- 优化 Decode DBO KV Cache 切片
- 支持动态批处理
- 支持更多节点拓扑
- 添加请求队列管理

---

## 11. 故障排查清单

### 11.1 安装问题

- [ ] Python 版本正确 (3.10+)
- [ ] CUDA 版本匹配 (12.6+)
- [ ] PyTorch 正确安装 (2.7.0 cu126)
- [ ] GPU 可用 (`torch.cuda.is_available()`)

### 11.2 运行问题

- [ ] GPU 显存充足
- [ ] 系统内存充足
- [ ] NCCL 通信正常
- [ ] 防火墙端口开放
- [ ] SSH 连接正常（多机）

### 11.3 性能问题

- [ ] 使用推荐的 batch size
- [ ] 检查 GPU 利用率
- [ ] 检查网络延迟
- [ ] 启用 timing 分析瓶颈

---

## 12. 相关文档

- [架构设计](01-architecture.md) - 系统架构和实现
- [使用指南](02-usage.md) - 命令行参数和示例
- [API 参考](03-api-reference.md) - 代码接口文档
- [实验结果](../results/) - 性能测试数据
