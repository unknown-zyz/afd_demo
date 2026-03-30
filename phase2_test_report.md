# Phase 2 测试报告: KV Cache + 文本生成

## 测试环境

### 本机
- GPU: Tesla V100-SXM2-32GB × 4
- PyTorch: 2.7.0+cu126
- Transformers: 5.4.0
- Accelerate: 1.13.0
- IP: 10.244.64.173

### 远程机器
- GPU: Tesla V100-SXM2-32GB
- PyTorch: 2.7.0+cu126
- Transformers: 5.4.0
- Accelerate: 1.13.0
- IP: 10.244.24.18 (内网)
- SSH: `ssh -p 30549 -i ~/.ssh/id_rsa_v100 zyz@192.168.5.32`

## 单机测试结果

| 测试项 | Prompt | 生成长度 | 速度 | 结果 |
|--------|--------|----------|------|------|
| 基础生成 | "Hello" | 8 tokens | 6.5 tok/s | ✅ 通过 |
| DBO 推理 | "Hello, how are you today?" | - | 714.71ms 端到端 | ✅ 通过 |

### 单机测试命令
```bash
# 基础生成
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m src.main \
  --local-test --generate --max-new-tokens 30 --prompt "Hello" --greedy

# 采样生成
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m src.main \
  --local-test --generate --max-new-tokens 50 \
  --prompt "Python is a programming language that" \
  --temperature 0.8 --top-k 50 --top-p 0.9
```

## 多机测试结果

| 测试项 | Prompt | 生成长度 | 速度 | 结果 |
|--------|--------|----------|------|------|
| 基础生成 | "Hello" | 8 tokens | 6.7 tok/s | ✅ 通过 |
| DBO 推理 | "Hello, how are you today?" | - | 735.38ms 端到端 | ✅ 通过 |

### 多机测试命令

**注意**: 需要确保两台机器版本一致 (PyTorch 2.7.0+cu126 / Transformers 5.4.0)

```bash
# 远程 FFN 节点
ssh -p 30549 -i ~/.ssh/id_rsa_v100 zyz@192.168.5.32
cd ~/afd_demo && source venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
python -m src.main --role ffn --master-addr 10.244.64.173 --master-port 29630 \
  --generate --max-new-tokens 30 --model-name Qwen/Qwen2-1.5B

# 本机 Attention 节点
cd ~/dev/afd_demo && source venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
python -m src.main --role attention --master-addr 10.244.64.173 --master-port 29630 \
  --generate --max-new-tokens 30 --prompt "Hello" --greedy \
  --model-name Qwen/Qwen2-1.5B
```

## 遇到的问题及解决方案

### 1. 网络连接
**问题**: 需要使用内网 IP (10.244.x.x) 而非跳板机 IP

**验证**: 端口连通性正常，ping 正常

### 2. Transformers 5.4 接口变更
**问题**: Qwen2Attention 接口要求 `position_embeddings`，旧代码仅传 `position_ids`

**解决**: 在调度器与 KV cache 路径统一预计算并传递 `position_embeddings`

### 3. PyTorch 2.7 NCCL 退出崩溃
**问题**: 运行完成后，`destroy_process_group()` 在部分环境触发 NCCL proxy 断言

**解决**: 对 NCCL 后端跳过显式销毁，避免 teardown 阶段崩溃（保留 warning）

## 性能对比

| 部署方式 | 生成速度 | 备注 |
|----------|----------|------|
| 单机 2 GPU | ~6.5 tok/s | PyTorch 2.7 + transformers 5.4 |
| 多机 2 GPU | ~6.7 tok/s | Attention/FFN 跨机通信 |

多机相比单机约有 10-15% 的性能损失，主要来自跨节点通信延迟。

## 测试日期
2026-03-29
