# AFD Demo: Attention-FFN 分离 + DBO 流水线

**Attention-FFN Disaggregation (AFD)** 架构 + **Dual Batch Overlap (DBO)** 流水线优化的概念验证系统。已适配 PyTorch 2.7.0 + Transformers 5.4.0。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7.0](https://img.shields.io/badge/pytorch-2.7.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 功能特性

- ✅ **Attention-FFN 分离**: 分布式节点架构，Attention 和 FFN 分别部署
- ✅ **DBO 流水线**: 2-micro-batch 异步重叠优化
- ✅ **单机/多机部署**: 支持本地和跨节点部署
- ✅ **KV Cache**: 支持自回归文本生成
- ✅ **MoE 支持**: 适配 Mixture-of-Experts 模型（Qwen3-30B-A3B）

## 🚀 快速开始

### 安装

```bash
# 克隆项目
git clone <repository_url>
cd afd_demo

# 创建虚拟环境并安装依赖
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 运行测试

```bash
# 单元测试
pytest tests/ -v

# 单机测试（需要 2+ GPU）
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m src.main \
  --local-test --prompt "Hello" --max-new-tokens 10

# 使用测试脚本（需要 4 GPU）
./scripts/test_local.sh 10 2  # 10 tokens, batch=2
```

### 系统要求

- **Python**: 3.10+
- **CUDA**: 12.6+ runtime
- **PyTorch**: 2.7.0 (cu126)
- **Transformers**: 5.4.0
- **GPU**: 2+ NVIDIA GPUs (推荐 V100 32GB 或更高)

## 📖 文档

详见 [`doc/`](doc/) 目录：

- **[架构设计](doc/01-architecture.md)** - 系统架构、DBO 实现原理
- **[使用指南](doc/02-usage.md)** - 命令行参数、运行示例、故障排查
- **[API 参考](doc/03-api-reference.md)** - 代码接口、扩展指南
- **[部署指南](doc/04-deployment.md)** - 环境配置、资源要求、生产部署

## 🏗️ 架构

```
┌──────────────────────┐   NCCL P2P   ┌──────────────────────┐
│   Attention Node     │◄────────────►│      FFN Node        │
│  - Embedding         │              │  - LayerNorm         │
│  - Self-Attention    │              │  - MLP (FFN)         │
│  - LM Head           │              │  - MoE Router        │
│  - ★ KV Cache ★      │              │  - Experts           │
└──────────────────────┘              └──────────────────────┘
```

**DBO 流水线重叠**:
```
无 DBO (串行):
[Attn_MB0] → [Send] → [FFN_MB0] → [Send] → [Attn_MB1] ...

有 DBO (重叠):
[Attn_MB0][Attn_MB1]...  ← 计算与通信重叠
    [isend0]   [isend1]...
         [FFN_MB0]   [FFN_MB1]...
```

## 📊 性能特征

### Prefill DBO（推荐使用✅）

| 环境 | Attention 效率 | FFN 效率 | 推荐 |
|------|---------------|---------|------|
| 单机 | 54.8% | 71.2% | ✅ |
| 多机 | >70%（预期） | >80%（预期） | ✅ |

### Decode DBO（当前不推荐❌）

| Batch | 性能影响 | 状态 |
|-------|---------|------|
| 2 | -4% ~ -12% | ⚠️ 轻度倒退 |
| 4+ | **-44% ~ -46%** | ❌ 严重倒退 |

**原因**: KV Cache 对象创建开销过大  
**建议**: Prefill 启用 DBO，Decode 禁用 DBO（使用 `--no-dbo`）

详细报告见 [results/reports/](results/reports/)。

## 🛠️ 运行脚本

```bash
# 单机测试
./scripts/test_local.sh [max_tokens] [batch_size]

# 多机测试
./scripts/test_multinode.sh [max_tokens] [batch_size]

# Prefill DBO 基准测试
./scripts/benchmark_dbo.sh [tokens] [batch] [on|off]

# Decode DBO 基准测试
./scripts/benchmark_decode_dbo.sh

# P2P 传输时间测量
python scripts/measure_transfer_time.py [master_addr] [master_port]
```

详见 [scripts/README.md](scripts/README.md)。

## 📂 项目结构

```
afd_demo/
├── src/                          # 源代码
│   ├── main.py                   # 程序入口
│   ├── distributed/              # 分布式通信
│   │   └── communicator.py       # NCCL P2P 异步通信
│   ├── model/                    # Worker 和模型分割
│   │   ├── attention_worker.py   # Attention 节点 Worker
│   │   ├── ffn_worker.py         # FFN 节点 Worker
│   │   ├── disaggregated.py      # 分离模型主类
│   │   └── kv_cache.py           # KV Cache 管理
│   ├── pipeline/                 # DBO 调度器
│   │   ├── async_scheduler.py    # Prefill DBO 实现
│   │   ├── decode_scheduler.py   # Decode DBO 实现
│   │   ├── micro_batch.py        # Micro-batch 状态管理
│   │   └── scheduler.py          # 同步基准调度器
│   └── utils/                    # 工具函数
│       ├── profiler.py           # GPU 内存和性能分析
│       ├── sampling.py           # Token 采样策略
│       ├── timing.py             # DBO 时序记录
│       └── validation.py         # 输入验证
├── tests/                        # 单元测试
│   ├── test_communicator.py      # 通信模块测试
│   ├── test_correctness.py       # 正确性验证测试
│   └── test_pipeline.py          # 调度器测试
├── scripts/                      # 运行和测试脚本
│   ├── run_attn_node.sh          # 启动 Attention 节点
│   ├── run_ffn_node.sh           # 启动 FFN 节点
│   ├── test_local.sh             # 单机测试脚本
│   ├── test_multinode.sh         # 多机测试脚本
│   ├── benchmark_dbo.sh          # Prefill DBO 基准测试
│   ├── benchmark_decode_dbo.sh   # Decode DBO 基准测试
│   ├── profile_dbo_pipeline.sh   # DBO 时序 profiling
│   ├── visualize_dbo.py          # Prefill DBO 可视化
│   ├── visualize_dbo_pipeline.py # 4-lane 流水线图
│   ├── plot_dbo_summary.py       # DBO 对比汇总图
│   └── measure_transfer_time.py  # P2P 传输时间测量
├── doc/                          # 完整文档
│   ├── 01-architecture.md        # 架构设计
│   ├── 02-usage.md               # 使用指南
│   ├── 03-api-reference.md       # API 参考
│   └── 04-deployment.md          # 部署指南
├── results/                      # 实验结果和报告
│   ├── prefill_dbo/              # Prefill DBO 测试结果
│   ├── decode_dbo/               # Decode DBO 测试结果
│   ├── network_latency/          # 网络延迟测试结果
│   └── reports/                  # 综合分析报告
├── config/                       # 模型配置文件
└── requirements.txt              # Python 依赖
```

## 🧪 测试和基准测试

```bash
# 单元测试
pytest tests/ -v

# 对比 DBO 效果
./scripts/benchmark_dbo.sh 50 4 on   # DBO ON
./scripts/benchmark_dbo.sh 50 4 off  # DBO OFF

# Decode DBO 全面测试（12 个用例）
./scripts/benchmark_decode_dbo.sh

# 生成可视化
python scripts/visualize_dbo.py results/prefill_dbo/
python scripts/plot_dbo_summary.py
```

## 🔧 常用命令

### 禁用 DBO（仅 AF 分离）

```bash
# 单机
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m src.main \
  --local-test --no-dbo --prompt "Hello"

# 多机
./scripts/run_attn_node.sh 10.244.64.179 29500 --no-dbo
```

### 调试模式

```bash
# 详细输出
--verbose

# 启用计时统计
--timing

# NCCL 调试
export NCCL_DEBUG=INFO
```

### 环境变量

```bash
export MODEL_NAME=/path/to/model              # 模型路径
export MASTER_PORT=29600                       # Master 端口
export CUDA_VISIBLE_DEVICES=0,1                # GPU 选择
export NCCL_SOCKET_TIMEOUT=600000              # NCCL 超时
```

## 🌐 多机部署

### SSH 连接远程节点

```bash
# 远程机器连接命令
ssh zyz@192.168.5.32 -p 31310 -i ~/.ssh/id_rsa_second
```

### 启动多机测试

```bash
# 方法 1: 使用自动脚本
./scripts/test_multinode.sh 10 1

# 方法 2: 手动启动
# 步骤 1: 在远程机器启动 FFN 节点
ssh zyz@192.168.5.32 -p 31310 -i ~/.ssh/id_rsa_second
cd /path/to/afd_demo && source venv/bin/activate
./scripts/run_ffn_node.sh 10.244.64.179 29500

# 步骤 2: 在本地启动 Attention 节点
./scripts/run_attn_node.sh 10.244.64.179 29500 --prompt "Hello"
```

详见 [部署指南](doc/04-deployment.md#4-多机部署)。

## ⚠️ 已知问题

1. **Decode DBO 性能**: batch >= 4 时性能倒退 -44%，需优化后使用
2. **NCCL 退出警告**: PyTorch 2.7 + NCCL 2.26 在退出时可能出现清理警告，不影响结果
3. **torch_dtype 弃用**: 建议使用 `dtype` 参数替代（代码已更新）

## 📝 License

MIT License

## 🙏 参考

- [vLLM AFD #22799](https://github.com/vllm-project/vllm/issues/22799)
- [vLLM DBO #23693](https://github.com/vllm-project/vllm/pull/23693)
