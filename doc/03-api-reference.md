# AFD Demo API 参考

## 1. 核心模块

### 1.1 分布式上下文

**文件**: `src/distributed/context.py`

#### `DistributedContext`

分布式环境的单例管理器。

**属性**:
```python
rank: int                    # 当前进程rank (0=Attention, 1=FFN)
world_size: int              # 总进程数 (固定为2)
device: torch.device         # 当前设备 (cuda:0 或 cuda:1)
role: str                    # 节点角色 ("attention" 或 "ffn")
is_attention_node: bool      # 是否为 Attention 节点
is_ffn_node: bool            # 是否为 FFN 节点
```

**使用**:
```python
from src.distributed import get_distributed_context

ctx = get_distributed_context()
if ctx.is_attention_node:
    # Attention 节点逻辑
    pass
elif ctx.is_ffn_node:
    # FFN 节点逻辑
    pass
```

---

### 1.2 通信模块

**文件**: `src/distributed/communicator.py`

#### `AFDCommunicator`

处理 Attention-FFN 节点间的异步通信。

**初始化**:
```python
communicator = AFDCommunicator(
    ctx: DistributedContext,
    num_micro_batches: int = 2
)
```

**主要方法**:

##### `async_send(tensor, layer_idx, mb_idx, to_ffn=True)`

异步发送 tensor。

**参数**:
- `tensor` (torch.Tensor): 要发送的数据
- `layer_idx` (int): 当前层索引
- `mb_idx` (int): Micro-batch 索引
- `to_ffn` (bool): True=发送到FFN, False=发送到Attention

**返回**: `torch.distributed.Work` 对象

##### `async_recv(shape, dtype, layer_idx, mb_idx, from_ffn=True)`

异步接收 tensor。

**参数**:
- `shape` (tuple): 接收数据的形状
- `dtype` (torch.dtype): 数据类型
- `layer_idx` (int): 当前层索引
- `mb_idx` (int): Micro-batch 索引
- `from_ffn` (bool): True=从FFN接收, False=从Attention接收

**返回**: `(torch.Tensor, torch.distributed.Work)` - 接收缓冲区和Work对象

---

### 1.3 Worker 模块

#### `AttentionWorker`

**文件**: `src/model/attention_worker.py`

处理 Embedding、Self-Attention 和 LM Head。

**初始化**:
```python
worker = AttentionWorker(
    model: PreTrainedModel,
    ctx: DistributedContext
)
```

**主要方法**:

##### `forward_embedding(input_ids)`

Embedding 层前向传播。

**参数**:
- `input_ids` (torch.Tensor): [batch, seq_len]

**返回**: `torch.Tensor` - [batch, seq_len, hidden_size]

##### `forward_attention_layer(hidden_states, layer_idx, ...)`

单层 Self-Attention 前向传播。

**参数**:
- `hidden_states` (torch.Tensor): [batch, seq, hidden]
- `layer_idx` (int): 层索引
- `kv_cache` (Optional[DynamicCache]): KV Cache
- `use_cache` (bool): 是否使用 cache

**返回**: 
- 如果 `use_cache=False`: `(attn_output, residual)`
- 如果 `use_cache=True`: `(attn_output, residual, updated_cache)`

##### `forward_lm_head(hidden_states)`

LM Head 前向传播，生成 logits。

**参数**:
- `hidden_states` (torch.Tensor): [batch, seq, hidden]

**返回**: `torch.Tensor` - [batch, seq, vocab_size]

#### `FFNWorker`

**文件**: `src/model/ffn_worker.py`

处理 LayerNorm 和 MLP/MoE。

**初始化**:
```python
worker = FFNWorker(
    model: PreTrainedModel,
    ctx: DistributedContext
)
```

**主要方法**:

##### `forward_ffn_layer(packed_input, layer_idx)`

单层 FFN 前向传播。

**参数**:
- `packed_input` (torch.Tensor): [batch, seq, hidden*2] (包含 attn_output 和 residual)
- `layer_idx` (int): 层索引

**返回**: `torch.Tensor` - [batch, seq, hidden]

---

### 1.4 Pipeline 调度器

#### `SimplePipelineScheduler`

**文件**: `src/pipeline/scheduler.py`

同步串行调度器（无 DBO）。

**初始化**:
```python
scheduler = SimplePipelineScheduler(
    worker: Union[AttentionWorker, FFNWorker],
    communicator: AFDCommunicator,
    ctx: DistributedContext
)
```

**主要方法**:

##### `run_prefill(input_ids)`

Prefill 阶段前向传播。

**参数**:
- `input_ids` (torch.Tensor): [batch, seq_len]

**返回**: `(logits, kv_cache)` - Attention 节点返回，FFN 节点返回 None

#### `AsyncPipelineScheduler`

**文件**: `src/pipeline/async_scheduler.py`

Prefill 阶段 DBO 调度器。

**初始化**:
```python
scheduler = AsyncPipelineScheduler(
    worker: Union[AttentionWorker, FFNWorker],
    communicator: AFDCommunicator,
    ctx: DistributedContext,
    num_micro_batches: int = 2,
    enable_timing: bool = False
)
```

**主要方法**:

##### `run_prefill(input_ids)`

Prefill 阶段前向传播（DBO 优化）。

**参数**:
- `input_ids` (torch.Tensor): [batch, seq_len]

**返回**: `(logits, kv_cache)` - Attention 节点返回，FFN 节点返回 None

#### `DecodeDBOScheduler`

**文件**: `src/pipeline/decode_scheduler.py`

Decode 阶段 DBO 调度器。

**初始化**:
```python
scheduler = DecodeDBOScheduler(
    worker: Union[AttentionWorker, FFNWorker],
    communicator: AFDCommunicator,
    ctx: DistributedContext,
    num_micro_batches: int = 2
)
```

**主要方法**:

##### `run_decode(input_ids, kv_cache)`

Decode 阶段单步前向传播。

**参数**:
- `input_ids` (torch.Tensor): [batch, 1] (当前token)
- `kv_cache` (DynamicCache): KV Cache

**返回**: `(logits, updated_cache)` - Attention 节点返回，FFN 节点返回 None

**注意**: 当前实现存在性能问题，batch >= 4 时性能倒退 -44%。

---

### 1.5 KV Cache 管理

**文件**: `src/model/kv_cache.py`

#### `KVCacheManager`

管理 KV Cache 的创建和更新。

**主要方法**:

##### `initialize_cache(batch_size, max_seq_len, num_layers, ...)`

初始化 KV Cache。

**参数**:
- `batch_size` (int): 批大小
- `max_seq_len` (int): 最大序列长度
- `num_layers` (int): 层数
- `num_heads` (int): 注意力头数
- `head_dim` (int): 每个头的维度
- `device` (torch.device): 设备
- `dtype` (torch.dtype): 数据类型

**返回**: `DynamicCache` 对象

##### `get_cache_shape(config, batch_size, seq_len)`

获取 KV Cache 的形状。

**返回**: `(batch, num_heads, seq_len, head_dim)`

---

## 2. 主程序入口

**文件**: `src/main.py`

### 命令行参数

参见 [使用指南](02-usage.md#3-命令行参数)。

### 主要函数

#### `main(args)`

主程序入口。

**流程**:
1. 初始化分布式环境
2. 加载模型
3. 创建 Worker
4. 创建 Communicator
5. 创建 Scheduler
6. 运行 Prefill
7. 运行 Decode 循环
8. 清理资源

---

## 3. 数据结构

### 3.1 MicroBatch

**文件**: `src/pipeline/micro_batch.py`

```python
@dataclass
class MicroBatch:
    input_ids: torch.Tensor          # 输入 token IDs
    batch_start: int                 # 在总 batch 中的起始索引
    batch_end: int                   # 在总 batch 中的结束索引
    state: MicroBatchState           # 当前状态
```

### 3.2 MicroBatchState

```python
class MicroBatchState(Enum):
    WAITING = 0              # 等待处理
    IN_ATTENTION = 1         # Attention 节点计算中
    SENDING_TO_FFN = 2       # 发送到 FFN
    IN_FFN = 3               # FFN 节点计算中
    SENDING_TO_ATTN = 4      # 发送回 Attention
    COMPLETE = 5             # 完成
```

---

## 4. 工具函数

### 4.1 模型分割

**文件**: `src/model/disaggregated.py`

#### `split_model(model, ctx)`

将模型分割为 Attention 和 FFN 部分。

**参数**:
- `model` (PreTrainedModel): HuggingFace 模型
- `ctx` (DistributedContext): 分布式上下文

**返回**: Worker 实例 (AttentionWorker 或 FFNWorker)

### 4.2 文本生成

**文件**: `src/generation/generator.py`

#### `generate(scheduler, tokenizer, prompt, max_new_tokens, ...)`

自回归文本生成。

**参数**:
- `scheduler`: Pipeline 调度器
- `tokenizer`: HuggingFace tokenizer
- `prompt` (str): 输入提示
- `max_new_tokens` (int): 最大生成 token 数
- `temperature` (float): 采样温度
- `top_k` (int): Top-k 采样
- `top_p` (float): Nucleus 采样
- `greedy` (bool): 是否使用贪婪解码

**返回**: `str` - 生成的文本

---

## 5. 性能分析

### 5.1 Timing 输出

启用 `--timing` 后，调度器会记录每个 micro-batch 的时间。

**输出格式** (JSON):
```json
{
  "layers": [
    {
      "layer_idx": 0,
      "micro_batches": [
        {
          "mb_idx": 0,
          "compute_start": 0.0,
          "compute_end": 0.05,
          "send_start": 0.05,
          "send_end": 0.051,
          "compute_time_ms": 50.0,
          "comm_time_ms": 1.0
        },
        ...
      ]
    },
    ...
  ]
}
```

### 5.2 可视化工具

#### `visualize_dbo_pipeline.py`

生成 Prefill DBO Pipeline 时间线图。

**用法**:
```bash
python scripts/visualize_dbo_pipeline.py \
  --attn-timing results/prefill_dbo/timing_attention_local_b4_s128_t5.json \
  --ffn-timing results/prefill_dbo/timing_ffn_local_b4_s128_t5.json \
  --output results/prefill_dbo/dbo_pipeline_local_b4_s128_t5.png \
  --start-layer 1 --num-layers 5
```

---

## 6. 测试

### 6.1 单元测试

**目录**: `tests/`

主要测试文件：
- `test_context.py` - 分布式上下文测试
- `test_communicator.py` - 通信模块测试
- `test_pipeline.py` - Pipeline 调度器测试
- `test_micro_batch.py` - Micro-batch 测试

### 6.2 运行测试

```bash
# 所有测试
pytest tests/ -v

# 特定文件
pytest tests/test_pipeline.py -v

# 特定测试
pytest tests/test_pipeline.py::TestMicroBatch -v
```

---

## 7. 扩展指南

### 7.1 添加新的调度器

1. 继承 `BasePipelineScheduler`
2. 实现 `run_prefill()` 和 `run_decode()` 方法
3. 在 `src/main.py` 中注册

### 7.2 支持新模型架构

1. 检查模型是否有 `self_attn`, `mlp`, `post_attention_layernorm`
2. 如果结构不同，修改 `AttentionWorker` 和 `FFNWorker` 的提取逻辑
3. 测试 Prefill 和 Decode 阶段

### 7.3 添加新的性能指标

1. 在调度器中添加计时点
2. 记录到 timing JSON
3. 更新可视化脚本

---

## 8. 相关文档

- [架构设计](01-architecture.md) - 系统架构和实现原理
- [使用指南](02-usage.md) - 命令行参数和运行示例
- [部署指南](04-deployment.md) - 环境配置和资源要求
