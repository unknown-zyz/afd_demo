# Batch Scaling Experiments - 实施指南

本文档说明如何执行 batch/seq scaling 实验，并给出预期结果。

## 快速开始

### 准备工作
```bash
# 确保虚拟环境已激活
cd /home/zyz/afd_demo
source venv/bin/activate

# 创建必要目录
mkdir -p results/prefill_dbo/logs
mkdir -p results/prefill_dbo/batch_scaling/{b4,b8,b16,b32,b64}
mkdir -p results/prefill_dbo/seq_scaling/{s32,s64,s128,s256,s512}
```

### 运行实验

#### 方式 1: 使用自动化脚本 (推荐)
```bash
# 运行所有实验（自动化，但可能需要几个小时）
./scripts/batch_scaling_experiments.sh all

# 分阶段运行
./scripts/batch_scaling_experiments.sh batch      # Phase 1: batch scaling
./scripts/batch_scaling_experiments.sh seq        # Phase 2: seq scaling
./scripts/batch_scaling_experiments.sh combined   # Phase 3: high load
```

#### 方式 2: 手动运行 (逐个测试，便于调试)

**单机测试示例 (batch=8, seq=128)**:
```bash
# 1. 启动 FFN 节点 (后台)
CUDA_VISIBLE_DEVICES=2,3 python -m src.main \
  --model-name /data/Qwen/Qwen3-30B-A3B/ \
  --role ffn \
  --master-addr 127.0.0.1 \
  --master-port 29650 \
  --world-size 2 \
  --rank 1 \
  --local-rank 0 \
  --attn-node-rank 0 \
  --ffn-node-rank 1 \
  --batch-size 8 \
  --prefill-seq-len 128 \
  --max-new-tokens 5 \
  --timing \
  --timing-suffix "local_b8_s128_t5" \
  --no-generate \
  --verbose \
  2>&1 | tee results/prefill_dbo/logs/ffn_local_b8_s128_t5.log &

FFN_PID=$!
sleep 5

# 2. 启动 Attention 节点 (前台)
CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
  --model-name /data/Qwen/Qwen3-30B-A3B/ \
  --role attention \
  --master-addr 127.0.0.1 \
  --master-port 29650 \
  --world-size 2 \
  --rank 0 \
  --local-rank 0 \
  --attn-node-rank 0 \
  --ffn-node-rank 1 \
  --batch-size 8 \
  --prefill-seq-len 128 \
  --max-new-tokens 5 \
  --prompt "Hello world, this is a test prompt for batch scaling experiments." \
  --timing \
  --timing-suffix "local_b8_s128_t5" \
  --verbose \
  2>&1 | tee results/prefill_dbo/logs/attn_local_b8_s128_t5.log

# 3. 等待 FFN 完成
wait $FFN_PID

# 4. 生成可视化
python scripts/visualize_dbo_pipeline.py \
  --attn-timing results/prefill_dbo/timing_attention_local_b8_s128_t5.json \
  --ffn-timing results/prefill_dbo/timing_ffn_local_b8_s128_t5.json \
  --output results/prefill_dbo/dbo_pipeline_local_b8_s128_t5.png \
  --start-layer 1 --num-layers 5
```

**多机测试示例**:
```bash
# 1. 在远程机器上启动 FFN 节点
ssh zyz@192.168.5.32 -p 31310 -i ~/.ssh/id_rsa_second
cd /home/zyz/afd_demo
source venv/bin/activate
CUDA_VISIBLE_DEVICES=2,3 python -m src.main \
  --model-name /data/Qwen/Qwen3-30B-A3B/ \
  --role ffn \
  --master-addr 10.244.64.179 \
  --master-port 29650 \
  --world-size 2 \
  --rank 1 \
  --local-rank 0 \
  --attn-node-rank 0 \
  --ffn-node-rank 1 \
  --batch-size 8 \
  --prefill-seq-len 128 \
  --max-new-tokens 5 \
  --timing \
  --timing-suffix "multinode_b8_s128_t5" \
  --no-generate \
  --verbose \
  2>&1 | tee results/prefill_dbo/logs/ffn_multinode_b8_s128_t5.log

# 2. 在本地机器上启动 Attention 节点 (10 秒后)
# (在另一个终端)
CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
  --model-name /data/Qwen/Qwen3-30B-A3B/ \
  --role attention \
  --master-addr 10.244.64.179 \
  --master-port 29650 \
  --world-size 2 \
  --rank 0 \
  --local-rank 0 \
  --attn-node-rank 0 \
  --ffn-node-rank 1 \
  --batch-size 8 \
  --prefill-seq-len 128 \
  --max-new-tokens 5 \
  --prompt "Hello world, this is a test prompt for batch scaling experiments." \
  --timing \
  --timing-suffix "multinode_b8_s128_t5" \
  --verbose

# 3. 从远程拉取 FFN timing
scp -P 31310 -i ~/.ssh/id_rsa_second \
  zyz@192.168.5.32:/home/zyz/afd_demo/results/prefill_dbo/timing_ffn_multinode_b8_s128_t5.json \
  results/prefill_dbo/

# 4. 生成可视化
python scripts/visualize_dbo_pipeline.py \
  --attn-timing results/prefill_dbo/timing_attention_multinode_b8_s128_t5.json \
  --ffn-timing results/prefill_dbo/timing_ffn_multinode_b8_s128_t5.json \
  --output results/prefill_dbo/dbo_pipeline_multinode_b8_s128_t5.png \
  --start-layer 1 --num-layers 5
```

## 实验矩阵

### Phase 1: Batch Size Scaling (seq=128, tokens=5)
| Batch | 单机 | 多机 | 预期结果 |
|-------|------|------|---------|
| 4     | ✓    | ✓    | baseline |
| 8     | ✓    | ✓    | ~2x 数据量 |
| 16    | ✓    | ✓    | ~4x 数据量 |
| 32    | ✓    | ✓    | ~8x 数据量 |
| 64    | ?    | ?    | 可能 OOM |

### Phase 2: Sequence Length Scaling (batch=8, tokens=5)
| Seq   | 单机 | 多机 | 预期结果 |
|-------|------|------|---------|
| 32    | ✓    | ✓    | 小 seq |
| 64    | ✓    | ✓    | 中等 |
| 128   | ✓    | ✓    | baseline |
| 256   | ✓    | ✓    | 大 seq |
| 512   | ?    | ?    | 可能 OOM |

### Phase 3: High Load Combinations
| Batch | Seq  | 单机 | 多机 | 预期结果 |
|-------|------|------|------|---------|
| 8     | 256  | ✓    | ✓    | 高激活 |
| 16    | 128  | ✓    | ✓    | 高激活 |
| 32    | 64   | ✓    | ✓    | 接近上限 |

## 预期产出

### 文件结构
```
results/prefill_dbo/
├── batch_scaling/
│   ├── b4/
│   │   ├── timing_attention_local_b4_s128_t5.json
│   │   ├── timing_ffn_local_b4_s128_t5.json
│   │   ├── dbo_pipeline_local_b4_s128_t5.png
│   │   ├── timing_attention_multinode_b4_s128_t5.json
│   │   ├── timing_ffn_multinode_b4_s128_t5.json
│   │   └── dbo_pipeline_multinode_b4_s128_t5.png
│   ├── b8/ (同上)
│   └── ...
├── seq_scaling/
│   ├── s32/ (类似结构)
│   └── ...
└── logs/
    ├── attn_local_b4_s128_t5.log
    ├── ffn_local_b4_s128_t5.log
    └── ...
```

### 统计数据
- **总文件数**: ~78 个文件
  - Phase 1: 5 configs × 2 deployments × 3 files = 30
  - Phase 2: 5 configs × 2 deployments × 3 files = 30
  - Phase 3: 3 configs × 2 deployments × 3 files = 18
- **总日志数**: ~78 个日志 (每个配置 attn + ffn)

## 分析工具

### 查看单个配置的结果
```bash
# 检查 timing 数据
python << 'EOF'
import json
with open('results/prefill_dbo/timing_attention_local_b8_s128_t5.json') as f:
    data = json.load(f)
    print(f"Total time: {data['total_time_ms']:.1f}ms")
    print(f"Compute: {data['total_compute_ms']:.1f}ms ({data['compute_ratio']*100:.1f}%)")
EOF
```

### 批量对比分析
```bash
# 对比不同 batch 的性能
python scripts/analyze_batch_scaling.py  # (待创建)
```

## 常见问题

### OOM 处理
如果遇到 OOM:
1. 检查日志中的 "OutOfMemoryError"
2. 降低 batch size 或 seq length
3. 记录失败的配置（有助于确定内存边界）

### 多机测试失败
1. 检查网络连通性: `ping 10.244.64.179`
2. 确认远程机器的 Python 环境正常
3. 检查 master_port 是否被占用

### 可视化失败
1. 确认 timing JSON 文件存在且完整
2. 检查是否有 matplotlib 错误
3. 尝试减少 --num-layers

## 下一步

实验完成后:
1. 将结果文件移动到对应子目录
2. 生成对比分析报告 (`results/reports/batch_scaling_summary.md`)
3. 绘制性能曲线图
4. 更新项目 README
