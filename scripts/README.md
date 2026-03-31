# 脚本目录

本目录包含用于运行、测试和分析 AFD Demo 的所有脚本。

## 📋 脚本索引

### 🧪 测试脚本

| 脚本 | 功能 | 用法 |
|------|------|------|
| `test_local.sh` | 单机测试 | `./scripts/test_local.sh [tokens] [batch]` |
| `test_multinode.sh` | 多机测试 | `./scripts/test_multinode.sh [tokens] [batch]` |
| `run_attn_node.sh` | 手动启动 Attention 节点 | `./scripts/run_attn_node.sh <master_addr> <port> [opts]` |
| `run_ffn_node.sh` | 手动启动 FFN 节点 | `./scripts/run_ffn_node.sh <master_addr> <port> [opts]` |

### 📊 基准测试脚本

| 脚本 | 功能 | 用法 |
|------|------|------|
| `benchmark_dbo.sh` | Prefill DBO 基准测试 | `./scripts/benchmark_dbo.sh [tokens] [batch] [on\|off]` |
| `benchmark_decode_dbo.sh` | Decode DBO 全面测试 | `./scripts/benchmark_decode_dbo.sh` |
| `measure_comm_latency.py` | 网络延迟测量 | `python scripts/measure_comm_latency.py [opts]` |

### 📈 可视化脚本

| 脚本 | 功能 | 用法 |
|------|------|------|
| `visualize_dbo.py` | DBO 时间线图 | `python scripts/visualize_dbo.py <results_dir> [opts]` |
| `visualize_dbo_pipeline.py` | DBO Pipeline 4泳道图 | `python scripts/visualize_dbo_pipeline.py [opts]` |
| `plot_dbo_summary.py` | 综合总结图 | `python scripts/plot_dbo_summary.py` |

---

## 详细说明

### test_local.sh

**功能**: 单机 4 GPU 测试（Attention 节点: GPU 0,1 / FFN 节点: GPU 2,3）

**语法**:
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

### run_attn_node.sh

**功能**: 手动启动 Attention 节点（用于自定义配置或调试）

**语法**:
```bash
./scripts/run_attn_node.sh <master_addr> <master_port> [options]
```

**参数**:
- `master_addr` - Master 节点地址
- `master_port` - Master 节点端口
- `options` - 其他 src/main.py 参数

**示例**:
```bash
# 基本用法
./scripts/run_attn_node.sh 10.244.64.179 29500

# 自定义提示词和生成长度
./scripts/run_attn_node.sh 10.244.64.179 29500 \
  --prompt "Hello, how are you?" \
  --max-new-tokens 20

# 禁用 DBO
./scripts/run_attn_node.sh 10.244.64.179 29500 --no-dbo
```

**依赖**: 需要先启动 FFN 节点

---

### run_ffn_node.sh

**功能**: 手动启动 FFN 节点

**语法**:
```bash
./scripts/run_ffn_node.sh <master_addr> <master_port> [options]
```

**参数**:
- `master_addr` - Master 节点地址
- `master_port` - Master 节点端口
- `options` - 其他 src/main.py 参数

**示例**:
```bash
# 在远程机器运行
ssh zyz@192.168.5.32 -p 31310 -i ~/.ssh/id_rsa_second
cd /path/to/afd_demo && source venv/bin/activate
./scripts/run_ffn_node.sh 10.244.64.179 29500
```

**注意**: FFN 节点需要先启动，Attention 节点后启动

---

### benchmark_dbo.sh

**功能**: Prefill DBO 基准测试（测量 Attention、通信、FFN 各阶段延迟）

**语法**:
```bash
./scripts/benchmark_dbo.sh [max_tokens] [batch_size] [dbo_mode]
```

**参数**:
- `max_tokens` - 生成 token 数量（默认: 30）
- `batch_size` - 批大小（默认: 1）
- `dbo_mode` - `on` 或 `off`（默认: on）

**示例**:
```bash
./scripts/benchmark_dbo.sh                 # 默认: 30 tokens, batch=1, DBO ON
./scripts/benchmark_dbo.sh 50 2 on         # DBO ON
./scripts/benchmark_dbo.sh 50 2 off        # DBO OFF
```

**输出**:
- 日志: `results/prefill_dbo/benchmark_*.log`
- JSON: `results/prefill_dbo/timing_*.json`

---

### benchmark_decode_dbo.sh

**功能**: Decode DBO 全面基准测试（12 个测试用例）

**语法**:
```bash
./scripts/benchmark_decode_dbo.sh
```

**测试矩阵**:
- Batch sizes: 2, 4, 8
- Sequence lengths: 128, 512
- DBO: ON vs OFF

**输出**:
- 日志: `results/decode_dbo/Qwen2-1.5B_batch*_seq*_dbo_*.log`
- 汇总: `results/decode_dbo/summary.csv`

**注意**: 测试需要 10-15 分钟，请耐心等待

---

### measure_comm_latency.py

**功能**: 测量单机或多机通信延迟（ping-pong 测试）

**语法**:
```bash
python scripts/measure_comm_latency.py [--size SIZE] [--iterations N]
```

**参数**:
- `--size` - 传输数据大小（默认: 4096 floats）
- `--iterations` - 测试次数（默认: 200）

**示例**:
```bash
# 默认配置
python scripts/measure_comm_latency.py

# 自定义大小和次数
python scripts/measure_comm_latency.py --size 8192 --iterations 500
```

**输出**:
- 终端输出：mean, P95, P99 延迟
- 日志: `results/network_latency/latency_*.log`

---

### visualize_dbo.py

**功能**: 生成 Prefill DBO 时间线可视化

**语法**:
```bash
python scripts/visualize_dbo.py <results_dir> [options]
```

**参数**:
- `results_dir` - 包含 timing JSON 的目录
- `--output` - 输出 PNG 文件路径
- `--max-layers` - 显示最多层数（默认: 所有层）

**示例**:
```bash
# 基本用法
python scripts/visualize_dbo.py results/prefill_dbo/

# 指定输出路径和层数
python scripts/visualize_dbo.py results/prefill_dbo/ \
  --output timeline.png \
  --max-layers 8
```

**输入文件**:
- `timing_attention.json` - Attention 节点计时数据
- `timing_ffn.json` - FFN 节点计时数据

**输出**: PNG 时间线图

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
- `--max-layers` - 显示层数（默认: 2，推荐 2-4 层）

**示例**:
```bash
# 默认用法 - 显示前 2 层
python scripts/visualize_dbo_pipeline.py

# 显示前 4 层
python scripts/visualize_dbo_pipeline.py --max-layers 4

# 指定输入和输出
python scripts/visualize_dbo_pipeline.py \
  --attn-timing results/prefill_dbo/timing_attention.json \
  --ffn-timing results/prefill_dbo/timing_ffn.json \
  --output my_pipeline.png
```

**输出格式**:
- 4 个泳道（从上到下）:
  - **A (Attention)** - Attention 节点计算
  - **A→F (Comm)** - Attention 发送到 FFN 的通信
  - **F (FFN)** - FFN 节点计算
  - **F→A (Comm)** - FFN 发送回 Attention 的通信
- 不同颜色区分不同的 micro-batch
- 在时间块上标注层号和持续时间
- 右上角显示性能统计信息

**优势**: 
- 相比 `visualize_dbo.py`，不显示空闲等待时间，更直观
- 清晰展示 DBO 的计算-通信重叠效果
- 适合理解 pipeline 工作原理

---

### plot_dbo_summary.py

**功能**: 生成综合总结图（6 个子图：计时、效率、对比等）

**语法**:
```bash
python scripts/plot_dbo_summary.py
```

**输入**: 从 `results/` 目录读取所有相关数据

**输出**: `results/dbo_summary.png`

**包含内容**:
- Prefill 阶段时间分解
- 计算效率分析
- Decode 吞吐量对比
- Attention/FFN 时间线
- 关键发现总结

---

## 🔧 常见任务

### 对比 DBO 效果

```bash
# Prefill 阶段
./scripts/benchmark_dbo.sh 50 4 on
./scripts/benchmark_dbo.sh 50 4 off

# 查看结果
diff results/prefill_dbo/*_on.log results/prefill_dbo/*_off.log
```

### 测试网络延迟

```bash
# 单机
python scripts/measure_comm_latency.py

# 多机（需要两台机器同时运行）
# 机器 1:
CUDA_VISIBLE_DEVICES=0 python scripts/measure_comm_latency.py --rank 0

# 机器 2:
CUDA_VISIBLE_DEVICES=0 python scripts/measure_comm_latency.py --rank 1
```

### 生成完整分析报告

```bash
# 1. 运行基准测试
./scripts/benchmark_dbo.sh 50 4 on
./scripts/benchmark_dbo.sh 50 4 off
./scripts/benchmark_decode_dbo.sh

# 2. 生成可视化
python scripts/visualize_dbo.py results/prefill_dbo/
python scripts/plot_dbo_summary.py

# 3. 查看结果
ls -lh results/*.png
cat results/decode_dbo/summary.csv
```

---

## 💡 技巧

### 后台运行长时间测试

```bash
# 使用 nohup
nohup ./scripts/benchmark_decode_dbo.sh > test.log 2>&1 &

# 查看进度
tail -f test.log
```

### 监控资源使用

```bash
# GPU 监控
watch -n 1 nvidia-smi

# 在测试脚本中加入监控
./scripts/benchmark_dbo.sh 50 4 on &
watch -n 1 'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv'
```

### 批量测试

```bash
# 测试不同 batch size
for batch in 1 2 4 8; do
  ./scripts/benchmark_dbo.sh 50 $batch on
  ./scripts/benchmark_dbo.sh 50 $batch off
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
