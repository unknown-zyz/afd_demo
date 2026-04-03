# 脚本目录

本目录包含用于运行、测试和分析 AFD Demo 的所有脚本。

## 📋 脚本索引

### 🧪 测试和运行脚本

| 脚本 | 功能 | 用法 |
|------|------|------|
| `test_local.sh` | 单机测试（支持 --timing 基准测试） | `./scripts/test_local.sh [tokens] [batch] [--timing] [--no-dbo]` |
| `test_multinode.sh` | 多机测试 | `./scripts/test_multinode.sh [tokens] [batch]` |
| `run_node.sh` | 手动启动单个节点 | `./scripts/run_node.sh <attention\|ffn> [master_addr] [port] [opts]` |
| `run_single.sh` | 单次实验运行器（支持 --visualize） | `./scripts/run_single.sh <local\|multinode> <batch> <seq> [opts]` |
| `run_experiments.sh` | 全量实验套件 | `./scripts/run_experiments.sh [suite]` |

### 📊 分析脚本

| 脚本 | 功能 | 用法 |
|------|------|------|
| `measure_transfer_time.py` | P2P 传输时间测量 | `python scripts/measure_transfer_time.py [opts]` |

### 📈 可视化脚本

| 脚本 | 功能 | 用法 |
|------|------|------|
| `visualize_dbo_pipeline.py` | DBO Pipeline 4泳道图 | `python scripts/visualize_dbo_pipeline.py [opts]` |
| `plot_experiment_results.py` | 实验结果绘图 | `python scripts/plot_experiment_results.py [opts]` |
| `plot_scaling_comparison.py` | Scaling 对比图 | `python scripts/plot_scaling_comparison.py [opts]` |

---

## 详细说明

### test_local.sh

**功能**: 单机 4 GPU 测试（Attention 节点: GPU 0,1 / FFN 节点: GPU 2,3）。支持 `--timing` 启用计时统计（替代原 `benchmark_dbo.sh`）。

**语法**:
```bash
./scripts/test_local.sh [max_tokens] [batch_size] [--timing] [--no-dbo]
```

**参数**:
- `max_tokens` - 生成 token 数量（默认: 5）
- `batch_size` - 批大小（默认: 1）
- `--timing` - 启用计时统计（基准测试模式）
- `--no-dbo` - 禁用 DBO（仅 AF 分离）

**示例**:
```bash
./scripts/test_local.sh                        # 默认: 5 tokens, batch=1
./scripts/test_local.sh 10 2                   # 10 tokens, batch=2
./scripts/test_local.sh 50 4 --timing          # DBO ON + 计时
./scripts/test_local.sh 50 4 --timing --no-dbo # DBO OFF + 计时
```

**输出**: 终端输出 + `results/test_local_*.log`

---

### test_multinode.sh

**功能**: 多机测试（自动在远程机器启动 FFN 节点）

**语法**:
```bash
./scripts/test_multinode.sh [max_tokens] [batch_size]
```

**参数**:
- `max_tokens` - 生成 token 数量（默认: 3）
- `batch_size` - 批大小（默认: 1）

**前置条件**:
- 远程机器已配置虚拟环境
- SSH 密钥认证已设置
- 远程机器地址: `192.168.5.32:31310`

**示例**:
```bash
./scripts/test_multinode.sh       # 默认: 3 tokens, batch=1
./scripts/test_multinode.sh 10 1  # 10 tokens, batch=1
```

**输出**: 终端输出 + 远程日志

---

### run_node.sh

**功能**: 手动启动单个节点（Attention 或 FFN），用于自定义配置或调试。合并了原 `run_attn_node.sh` 和 `run_ffn_node.sh`。

**语法**:
```bash
./scripts/run_node.sh <attention|ffn> [master_addr] [master_port] [options]
```

**参数**:
- `attention|ffn` - 节点角色（必选）
- `master_addr` - Master 节点地址
- `master_port` - Master 节点端口
- `options` - 其他 src/main.py 参数

**示例**:
```bash
# 启动 Attention 节点
./scripts/run_node.sh attention 10.244.64.179 29500

# 自定义提示词和生成长度
./scripts/run_node.sh attention 10.244.64.179 29500 \
  --prompt "Hello, how are you?" \
  --max-new-tokens 20

# 禁用 DBO
./scripts/run_node.sh attention 10.244.64.179 29500 --no-dbo

# 在远程机器启动 FFN 节点
ssh zyz@192.168.5.32 -p 31310 -i ~/.ssh/id_rsa_second
cd /path/to/afd_demo && source venv/bin/activate
./scripts/run_node.sh ffn 10.244.64.179 29500
```

**注意**: FFN 节点需要先启动，Attention 节点后启动

---

### run_single.sh

**功能**: 单次实验运行器，支持 DBO/Serial、本地/多机配置。支持 `--visualize` 生成 DBO Pipeline 时序图（替代原 `run_experiment.sh` 和 `profile_dbo_pipeline.sh`）。

**语法**:
```bash
./scripts/run_single.sh <local|multinode> <batch> <seq> [--tokens N] [--no-dbo] [--visualize]
```

**参数**:
- `local|multinode` - 部署模式（必选）
- `batch` - 批大小（必选）
- `seq` - 序列长度（必选）
- `--tokens N` - 生成 token 数量（默认: 5）
- `--no-dbo` - 禁用 DBO（Serial baseline）
- `--visualize` - 生成 DBO Pipeline 时序可视化

**示例**:
```bash
# Local DBO, batch=8, seq=128
./scripts/run_single.sh local 8 128 --tokens 5

# Multinode DBO
./scripts/run_single.sh multinode 16 128 --tokens 5

# Serial baseline
./scripts/run_single.sh local 8 128 --no-dbo

# 带可视化 profiling
./scripts/run_single.sh local 4 128 --visualize
```

---

### run_experiments.sh

**功能**: 全量实验套件，支持选择不同实验组合（替代原 `run_qwen3_experiments.sh` 和 `run_qwen3_extended.sh`）。

**语法**:
```bash
./scripts/run_experiments.sh [decode|decode-large|prefill-batch|prefill-seq|prefill-multimb|all]
```

**参数**:
- `decode` - Decode 实验
- `decode-large` - 大规模 Decode 实验
- `prefill-batch` - Prefill batch scaling 实验
- `prefill-seq` - Prefill seq scaling 实验
- `prefill-multimb` - Prefill multi-micro-batch 实验
- `all` - 运行所有实验

**示例**:
```bash
./scripts/run_experiments.sh all             # 所有实验
./scripts/run_experiments.sh decode          # 仅 Decode 实验
./scripts/run_experiments.sh prefill-batch   # 仅 Prefill batch scaling
```

---

### measure_transfer_time.py

**功能**: 测量 NCCL P2P 单向传输时间（同步 send 基准测试）

**语法**:
```bash
# 单机测试 (默认)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/measure_transfer_time.py

# 多机测试
# 机器 1 (rank 1 - receiver):
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
  --master_addr=<master_addr> --master_port=29500 scripts/measure_transfer_time.py

# 机器 2 (rank 0 - sender):  
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
  --master_addr=<master_addr> --master_port=29500 scripts/measure_transfer_time.py
```

**测试内容**:
- 不同 tensor 大小（batch × hidden_dim）
- 单向同步传输延迟
- 带宽计算

**示例输出**:
```
Transfer size: 2.0 MB, Time: 0.11 ms, Bandwidth: 144.8 Gb/s
```

**用途**: 用于对比 DBO 的异步通信时间是否符合预期

---

### visualize_dbo_pipeline.py

**功能**: 生成 4 泳道 DBO Pipeline 可视化图，清晰展示 Attention、通信、FFN 模块的重叠关系

**语法**:
```bash
python scripts/visualize_dbo_pipeline.py [options]
```

**参数**:
- `--attn-timing` - Attention 节点 timing JSON 文件路径（默认: `results/prefill_dbo/timing_attention.json`）
- `--ffn-timing` - FFN 节点 timing JSON 文件路径（默认: `results/prefill_dbo/timing_ffn.json`）
- `--output` - 输出 PNG 文件路径（默认: `results/prefill_dbo/dbo_pipeline_4lanes.png`）
- `--num-layers` - 显示层数（默认: 2，推荐 2-4 层）

**示例**:
```bash
# 默认用法 - 显示前 2 层
python scripts/visualize_dbo_pipeline.py

# 显示前 4 层
python scripts/visualize_dbo_pipeline.py --num-layers 4

# 指定输入和输出
python scripts/visualize_dbo_pipeline.py \
  --attn-timing results/prefill_dbo/timing_attention.json \
  --ffn-timing results/prefill_dbo/timing_ffn.json \
  --output my_pipeline.png
```

**输出格式**:
- 4 个泳道（从上到下）:
  - **A (Attention)** - Attention 节点计算
  - **A→F (Comm)** - Attention 发送到 FFN 的真实传输时间
  - **F (FFN)** - FFN 节点计算
  - **F→A (Comm)** - FFN 发送回 Attention 的真实传输时间
- 使用 `send_transfer` 事件测量真实传输时间（后台轮询检测完成）
- 不同颜色区分不同的 micro-batch
- 在时间块上标注层号和持续时间
- 右上角显示性能统计信息

**优势**: 
- 不显示空闲等待时间，更直观
- 清晰展示 DBO 的计算-通信重叠效果
- 适合理解 pipeline 工作原理

---

## 🔧 常见任务

### 对比 DBO 效果

```bash
# Prefill 阶段
./scripts/test_local.sh 50 4 --timing          # DBO ON
./scripts/test_local.sh 50 4 --timing --no-dbo # DBO OFF

# 查看结果
diff results/prefill_dbo/*_on.log results/prefill_dbo/*_off.log
```

### 测量 P2P 传输时间

```bash
# 单机测试
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/measure_transfer_time.py
```

### 生成完整分析报告

```bash
# 1. 运行基准测试
./scripts/test_local.sh 50 4 --timing          # DBO ON
./scripts/test_local.sh 50 4 --timing --no-dbo # DBO OFF

# 2. 生成可视化
python scripts/visualize_dbo_pipeline.py

# 3. 查看结果
ls -lh results/*.png
```

---

## 💡 技巧

### 后台运行长时间测试

```bash
# 使用 nohup
nohup ./scripts/test_local.sh 50 4 --timing > test.log 2>&1 &

# 查看进度
tail -f test.log
```

### 监控资源使用

```bash
# GPU 监控
watch -n 1 nvidia-smi

# 在测试脚本中加入监控
./scripts/test_local.sh 50 4 --timing &
watch -n 1 'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv'
```

### 批量测试

```bash
# 测试不同 batch size
for batch in 1 2 4 8; do
  ./scripts/test_local.sh 50 $batch --timing
  ./scripts/test_local.sh 50 $batch --timing --no-dbo
done
```

---

## 🐛 故障排查

### 脚本权限问题

```bash
# 添加执行权限
chmod +x scripts/*.sh
```

### 虚拟环境未激活

```bash
# 脚本会自动激活，但手动运行时需要:
source venv/bin/activate
```

### CUDA_VISIBLE_DEVICES 冲突

```bash
# 脚本内部已设置，无需手动指定
# 如需覆盖，可在脚本前设置:
CUDA_VISIBLE_DEVICES=4,5,6,7 ./scripts/test_local.sh
```

---

## 相关文档

- [使用指南](../doc/02-usage.md) - 详细参数说明
- [部署指南](../doc/04-deployment.md) - 环境配置
- [实验结果](../results/README.md) - 测试数据和分析
