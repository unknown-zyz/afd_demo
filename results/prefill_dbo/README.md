# Prefill DBO Results

本目录包含 AFD (Attention-FFN Disaggregation) + DBO (Dual Batch Overlap) 的 Prefill 阶段实验结果。

## 目录结构

```
prefill_dbo/
├── README.md                          # 本文件
├── logs/                              # 实验日志
├── archive/                           # 早期实验归档
│   ├── dbo_pipeline_4lanes.png
│   ├── dbo_prefill_timeline.png
│   ├── dbo_summary.png
│   └── timing_*.json
├── batch_scaling/                     # Batch size scaling 实验（待完成）
│   ├── b4/, b8/, b16/, ...
├── seq_scaling/                       # Sequence length scaling 实验（待完成）
│   ├── s32/, s64/, s128/, ...
├── timing_attention_*.json            # 当前测试数据
├── timing_ffn_*.json
└── dbo_pipeline_*.png                 # 当前测试可视化
```

## 文件命名规范

### Timing JSON 文件
格式: `timing_{node}_{deployment}_b{batch}_s{seq}_t{tokens}.json`

- `node`: `attention` 或 `ffn`
- `deployment`: `local` (单机) 或 `multinode` (多机)
- `batch`: batch size
- `seq`: sequence length (默认 1 时省略)
- `tokens`: 生成的 token 数

示例:
- `timing_attention_local_b4_t5.json` - 单机, batch=4, seq=1, tokens=5
- `timing_attention_local_b4_s128_t5.json` - 单机, batch=4, seq=128, tokens=5
- `timing_attention_multinode_b8_s256_t5.json` - 多机, batch=8, seq=256, tokens=5

### 可视化图片
格式: `dbo_pipeline_{deployment}_b{batch}_s{seq}_t{tokens}.png`

## 当前实验结果

### 基准测试 (已完成)
- **batch=4, seq=1, tokens=3/5**: 单机 + 多机
  - 验证 SEND_TRANSFER 时间测量
  - 修复 recv_wait 时间记录 bug
  - Layer 0 初始化开销: ~91ms (其他层 ~1.7ms)

### 关键发现
1. **通信延迟分析** (详见 `doc/communication_analysis.md`)
   - A2F: ~5.3ms (FFN 延迟就绪导致)
   - F2A: ~0.5ms (Attn 提前就绪)
   - 小数据 NCCL 开销占主导

2. **Layer 0 初始化**
   - 首次计算开销 50x 高于后续层
   - 可视化默认跳过 Layer 0 (--start-layer 1)

3. **Pipeline 时序**
   - ✓ 每层正确等待上一层完成
   - recv_wait 测量 wait() 阻塞时间 (~0.01-0.2ms)

## 待完成实验

### Phase 1: Batch Size Scaling (seq=128)
测试配置: batch=4, 8, 16, 32, 64 (单机 + 多机)

### Phase 2: Sequence Length Scaling (batch=8)
测试配置: seq=32, 64, 128, 256, 512 (单机 + 多机)

### Phase 3: High Load Combinations
测试配置: (batch=8, seq=256), (16, 128), (32, 64) (单机 + 多机)

## 运行实验

### 使用自动化脚本
```bash
# 运行所有实验
./scripts/run_qwen3_experiments.sh
```

### 手动运行单个配置
```bash
# 使用 profile_dbo_pipeline.sh (旧接口，seq=1)
./scripts/profile_dbo_pipeline.sh local 8 5

# 使用 src.main 直接运行 (支持 --prefill-seq-len)
# FFN 节点 (后台)
CUDA_VISIBLE_DEVICES=2,3 python -m src.main \
  --model-name /data/Qwen/Qwen3-30B-A3B/ \
  --role ffn --batch-size 8 --prefill-seq-len 128 \
  --max-new-tokens 5 --timing --timing-suffix "local_b8_s128_t5" &

# Attention 节点 (前台)
CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
  --model-name /data/Qwen/Qwen3-30B-A3B/ \
  --role attention --batch-size 8 --prefill-seq-len 128 \
  --max-new-tokens 5 --timing --timing-suffix "local_b8_s128_t5"
```

### 生成可视化
```bash
# 默认：Layer 1-5 (跳过 Layer 0 初始化开销)
python scripts/visualize_dbo_pipeline.py \
  --attn-timing timing_attention_local_b8_s128_t5.json \
  --ffn-timing timing_ffn_local_b8_s128_t5.json \
  --output dbo_pipeline_local_b8_s128_t5.png \
  --start-layer 1 --num-layers 5

# 包含 Layer 0
python scripts/visualize_dbo_pipeline.py \
  --start-layer 0 --num-layers 6
```

## 分析和报告

实验完成后：
- 结果保存在 `results/experiments_qwen3/` 中
- 包含实验总结、性能分析和对比图表

## 相关文档

- `doc/communication_analysis.md` - 通信延迟深度分析
- `scripts/README.md` - 脚本使用说明
- `README.md` - 项目总体文档
