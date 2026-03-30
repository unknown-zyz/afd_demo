# AFD Demo: Attention-FFN 分离流水线

实现 Attention-FFN 分离架构 + DBO (Dual Batch Overlap) 流水线，已适配 `PyTorch 2.7.0 + Transformers 5.4.0`。

## 功能特性

- ✅ Attention-FFN 分离到不同节点
- ✅ 2-micro-batch 流水线
- ✅ DBO 异步重叠（单机/多机可运行）
- ✅ 单机/多机部署
- ✅ KV Cache + 自回归文本生成

## 快速开始

```bash
# 安装
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 单机 DBO 测试（2 GPU）
./scripts/run_local.sh --scheduler compare

# 文本生成测试
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m src.main \
  --local-test --generate --max-new-tokens 30 --prompt "Hello"
```

### 版本要求（已验证）

- Python 3.10
- CUDA 驱动支持 CUDA 12.6 runtime
- PyTorch 2.7.0 (`cu126`)
- Transformers 5.4.0
- Accelerate 1.13.0

## 架构

```
┌──────────────────────┐    NCCL P2P    ┌──────────────────────┐
│   Attention Node     │◄──────────────►│      FFN Node        │
│  - Embedding         │                │  - LayerNorm         │
│  - Self-Attention    │                │  - MLP (FFN)         │
│  - LM Head           │                │                      │
└──────────────────────┘                └──────────────────────┘
```

## DBO 流水线

```
同步串行（无重叠）:
[Attn_MB0] → [Send] → [FFN_MB0] → [Send] → [Attn_MB1] ...
                ↑ 等待 ↑

异步重叠（DBO）:
[Attn_MB0][Attn_MB1]...
    [Send0]    [Send1]...  ← 计算与通信重叠
        [FFN_MB0]    [FFN_MB1]...

说明：升级后计时受环境/调度影响较大，建议优先关注端到端耗时与功能正确性。
```

## 命令参数

### DBO 流水线模式
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--scheduler` | async | sync / async (DBO) / compare |
| `--batch-size` | 2 | 总批次大小 |
| `--num-micro-batches` | 2 | 微批次数量 |
| `--timing` | - | 启用详细计时 |

### 生成模式
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--generate` | - | 启用文本生成模式 |
| `--prompt` | "Hello" | 输入提示 |
| `--max-new-tokens` | 50 | 最大生成长度 |
| `--temperature` | 0.7 | 采样温度 |
| `--top-k` | 50 | Top-k 采样 |
| `--top-p` | 0.9 | Nucleus 采样 |
| `--greedy` | - | 贪婪解码 |

## 脚本使用

```bash
# 单机 DBO 对比测试
./scripts/run_local.sh --scheduler compare

# 单机 DBO + 详细计时
./scripts/run_local.sh --scheduler async --timing

# 单机生成测试
./scripts/run_local.sh --generate --max-new-tokens 30 --prompt "Hello"

# 多机测试
# 本机 (Attention)
./scripts/run_attn_node.sh 10.244.64.173 29500 --scheduler async

# 远程 (FFN)
ssh -p 30549 -i ~/.ssh/id_rsa_v100 zyz@192.168.5.32
./scripts/run_ffn_node.sh 10.244.64.173 29500 --scheduler async

# 可视化
python scripts/visualize_dbo.py results/
```

> 注：当前环境下 `PyTorch 2.7.0 + NCCL 2.26` 在进程退出阶段可能出现 NCCL 清理崩溃。代码已加入规避，运行可完成；退出时会看到 `destroy_process_group() was not called` 警告，可忽略。

## 项目结构

```
afd_demo/
├── src/
│   ├── distributed/    # 分布式通信
│   ├── model/          # 模型组件 (attention_worker, ffn_worker, kv_cache)
│   ├── pipeline/       # 调度器 (sync, async)
│   ├── utils/          # 采样工具
│   └── main.py         # 入口
├── scripts/            # 启动脚本
└── results/            # 时间线数据和可视化
```

## 工作模式

### 1. DBO 流水线 (Prefill)
用于批量处理 prompt 的前向推理，micro-batch 流水线重叠计算和通信。

### 2. 文本生成
- **Prefill**: 处理完整 prompt，初始化 KV Cache
- **Decode**: 逐 token 自回归生成，使用 KV Cache

KV Cache 存储在 Attention 节点，不参与跨节点通信。

## 当前限制

- Decode 阶段暂无 DBO（每次只处理单 token）
- 不支持 batch 内不同长度序列

## 后续计划

- [ ] Phase 3: MoE 支持 (Qwen3-30B-A3B)
- [ ] Decode 阶段 DBO
- [ ] Continuous batching

## 参考

- [vLLM AFD #22799](https://github.com/vllm-project/vllm/issues/22799)
- [vLLM DBO #23693](https://github.com/vllm-project/vllm/pull/23693)
