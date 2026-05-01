# 01. 架构设计

本文说明当前 AFD Demo 的执行结构、scheduler、KV cache、backend abstraction
和 timing 口径。

## 1. 总体目标

AFD Demo 将 Transformer 推理中的 Attention 子图和 FFN / MoE 子图拆成不同
角色：

```text
输入 token
   │
   ▼
Attention role：embedding、attention、KV cache、采样
   │  hidden states
   ▼
FFN role：MLP / MoE FFN
   │  hidden states
   ▼
Attention role：下一层 attention 或 lm_head
```

这种拆分允许 Attention 与 FFN 在不同设备、不同 rank 或不同节点上运行。DBO
在此基础上把 batch 拆成 micro-batch，使 Attention 与 FFN 在层间流水重叠。

## 2. 主要代码结构

| 模块 | 职责 |
|---|---|
| `src/main.py` | CLI、分布式初始化、模型加载、scheduler 选择、timing 输出。 |
| `src/model/disaggregated.py` | Qwen3 A/F 拆分模型封装、自回归生成、KV cache 维护。 |
| `src/model/attention_worker.py` | embedding、attention layer、norm、lm_head、采样。 |
| `src/model/ffn_worker.py` | FFN / MoE layer 计算。 |
| `src/pipeline/scheduler.py` | `SimplePipelineScheduler`，serial AF baseline。 |
| `src/pipeline/async_scheduler.py` | `AsyncPipelineScheduler`，prefill DBO。 |
| `src/pipeline/decode_scheduler.py` | `DecodeDBOScheduler`，decode DBO 与 crosslayer。 |
| `src/distributed/__init__.py` | rank、role、process group、P2P group 管理。 |
| `src/utils/device.py` | CUDA / NPU / CPU 设备与分布式 backend 抽象。 |
| `src/utils/timing.py` | timing event、layer timing、JSON 输出结构。 |

## 3. 执行模式

| 模式 | 触发方式 | Scheduler | 主要指标 |
|---|---|---|---|
| Serial baseline | `--no-dbo --generate` | `SimplePipelineScheduler` + generation path | `prefill_ms`、`decode_tpot_ms` |
| Prefill DBO | 默认单次 DBO；不加 `--generate` | `AsyncPipelineScheduler` | 模型侧 TTFT-path |
| Decode DBO | `--generate` | `DecodeDBOScheduler` | 准确 TPOT |
| Decode crosslayer | `--generate --crosslayer` | `DecodeDBOScheduler(use_crosslayer=True)` | 准确 TPOT |

Serial baseline 不是“未拆分模型”，而是关闭 DBO 的 A/F 分离串行路径。它是所有
speedup 的 mode-matched baseline。

## 4. Attention 角色

Attention role 负责：

1. token embedding；
2. 每层 self-attention；
3. 持有和更新 KV cache；
4. 与 FFN role 交换 hidden states；
5. 最后一层之后执行 norm、lm_head 和 sampling。

KV cache 位于 Attention role，不传给 FFN role。Decode 时每一步只把当前 token
对应的 hidden states 送到 FFN role。

## 5. FFN 角色

FFN role 负责每层 FFN / MoE FFN 计算。当前 Qwen3-30B-A3B 的 FFN 权重加载和
MoE 计算是 decode 阶段的重要开销来源，也是 GPU/NPU decode DBO 中位数没有达到
稳定正收益的主要背景之一。

## 6. 调度器（Scheduler）

### 6.1 SimplePipelineScheduler

`SimplePipelineScheduler` 按层串行执行：

```text
layer i attention -> send -> layer i ffn -> recv -> layer i+1 attention
```

它用于 serial AF baseline，提供 `prefill_ms` 和 `decode_tpot_ms`。

### 6.2 AsyncPipelineScheduler

`AsyncPipelineScheduler` 用于 prefill DBO。它把 batch 切成 micro-batch，在
Attention 和 FFN 之间交错发送 hidden states：

```text
ATT: MB0 L0 ─ send ─ MB1 L0 ─ send ─ MB0 L1 ...
FFN:      recv ─ MB0 L0 ─ recv ─ MB1 L0 ...
```

Prefill activation 与 `batch * seq` 强相关，因此 prefill DBO 的 OOM 边界会比
decode 更紧。

### 6.3 DecodeDBOScheduler

`DecodeDBOScheduler` 用于自回归 decode。每个 decode step 内部也会尝试
micro-batch overlap。`--crosslayer` 会启用跨层方向性通信组，减少部分层间气泡。

Crosslayer 是实验性路径，最终效果必须用准确 `decode_tpot_ms` 判断，不能只用
decode step 1 timing 判断。

## 7. KV cache 与自回归生成

当前实现使用 HuggingFace `DynamicCache`。生成流程是：

1. prefill 处理完整 prompt，初始化 KV cache；
2. 采样得到第一个 token；
3. decode loop 每次只处理最新 token；
4. Attention role 更新 KV cache；
5. FFN role 只处理 hidden states，不持有 cache。

因此项目已经实现 KV cache 和自回归生成。若单次输出只有一个 token，通常是因为
未开启 `--generate`、`--tokens` 设置过小，或运行的是 prefill-only DBO。

## 8. 后端抽象

`src/utils/device.py` 统一处理设备与 backend：

| 后端 | 设备 API | 分布式 backend |
|---|---|---|
| CUDA | `torch.cuda` | NCCL |
| Ascend NPU | `torch.npu` / `torch_npu` | HCCL |
| CPU | `torch.device("cpu")` | Gloo |

NPU 分支使用 `torch_npu.contrib.transfer_to_npu` 兼容部分 CUDA API 表面，但
NPU/HCCL 仍有独立脚本和环境变量。

## 9. 计时与加速比口径

Timing JSON 中常见字段：

| 字段 | 含义 |
|---|---|
| `total_time_ms` | 当前 scheduler 记录的总时间；prefill DBO speedup 使用它。 |
| `prefill_ms` | Serial cache 中的 prefill-only 时间。 |
| `decode_loop_ms` | 自回归 decode loop 总时间，不含 prefill 首 token 路径。 |
| `decode_steps` | decode loop 的 token step 数，通常是 `max_new_tokens - 1`。 |
| `decode_tpot_ms` | 准确 TPOT：`decode_loop_ms / decode_steps`。 |
| `prefill_seq_len` | 实验请求的 prefill 长度。 |
| `actual_prompt_len` | tokenizer 后实际输入长度；用于审计 `s<seq>` 标签是否真实生效。 |
| `events` | 用于 Gantt 图的 decode step 1 layer / micro-batch 事件。 |

Speedup 统一为：

```text
speedup = serial / DBO
```

| 模式 | 指标 | 公式 |
|---|---|---|
| Prefill DBO | 模型侧 TTFT-path | `serial_prefill_ms / dbo_total_time_ms` |
| Decode DBO | 准确 TPOT | `serial_decode_tpot_ms / dbo_decode_tpot_ms` |
| Crosslayer decode | 准确 TPOT | `serial_decode_tpot_ms / dbo_decode_tpot_ms` |

`events` 和 decode step 1 timing 可以解释 overlap，但不能作为最终 speedup 分母。

## 10. 结果解读边界

- 启动耗时很大一部分来自 Qwen3-30B-A3B 权重加载、进程启动和 warmup，不等于
  scheduler timing。
- OOM 行是容量边界，不是缺失数据。
- 旧 Qwen2 时代或旧 fallback 口径的性能结论不能作为当前结论。
- 当前 GPU/NPU 实验结论见 [08-gpu-npu-experiment-summary.md](08-gpu-npu-experiment-summary.md)。
