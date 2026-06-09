# 代码目录说明

`src/` 保存真实推理路径、流水调度、分布式拓扑、Coordinator 控制面和工具抽象。根入口是 `src/main.py`。

## 主调用链

```text
src/main.py
  -> init_backend() / init_distributed()
  -> DisaggregatedQwenModel.from_pretrained()
  -> scheduler: SimplePipelineScheduler / AsyncPipelineScheduler / DecodeDBOScheduler
  -> AttentionWorker.forward_attention_layer()
  -> FFNWorker.forward_ffn_layer() 或 EPFFNLayer work item
  -> TimingTracker JSON / pipeline report
```

Serial baseline 仍是 A/F 分离路径，只是关闭 DBO；decode speedup 使用 `serial_decode_tpot_ms / dbo_decode_tpot_ms`。

## 目录与文件职责

| 路径 | 职责 |
|---|---|
| `main.py` | CLI 入口、设备后端初始化、分布式拓扑、模型加载、scheduler 选择、生成模式、timing 输出和可选 coordinator routing。 |
| `model/disaggregated.py` | 静态 A/F 分离 Qwen 模型封装，连接 AttentionWorker、FFNWorker、KV cache、生成循环和 routing backend。 |
| `model/attention_worker.py` | Attention role：embedding、self-attention、KV cache、final norm、lm_head、sampling；包含 NPU official Attention、RMSNorm/RoPE fusion、layer input precopy 等实验路径。 |
| `model/ffn_worker.py` | FFN role：post-attention norm、MoE gate、experts、combine；根据 EP 配置构造普通 FFNLayer 或 EPFFNLayer。 |
| `model/ep_moe.py` | Expert parallel MoE 核心：expert ownership、ShardedExperts、broadcast/reduce overlap、all-to-all backend、sparse P2P 实验 backend。 |
| `pipeline/scheduler.py` | Serial A/F baseline scheduler。 |
| `pipeline/async_scheduler.py` | Prefill DBO scheduler，按 batch 维切分 microbatch。 |
| `pipeline/decode_scheduler.py` | Decode DBO、cross-layer、EP overlap、early recv 和 A/F P2P 调度。 |
| `pipeline/micro_batch.py` | Prefill microbatch 切分与合并工具。 |
| `distributed/__init__.py` | rank role、process group、P2P group、FFN EP group、rank/device 映射和 distributed context。 |
| `coordinator_arch/` | Coordinator 控制面、routing table、worker skeleton、MoE communicator 抽象、fallback all-to-all 和 DeepEP 接口。 |
| `utils/device.py` | CUDA/NPU/CPU 后端选择、stream/event 抽象、device helper。 |
| `utils/timing.py` | timing event、layer stage timing、JSON 输出和 profiling 元数据。 |

## E/A 分离职责边界

Attention role 负责：

1. token embedding；
2. self-attention 和 KV cache 更新；
3. A2F hidden states 发送；
4. F2A hidden states 接收；
5. final norm、lm_head、sampling。

FFN/Expert role 负责：

1. post-attention residual merge；
2. post-attention RMSNorm；
3. MoE router/gate；
4. expert dispatch 与 local expert compute；
5. combine/reduce 与 residual output；
6. F2A hidden states 返回。

KV cache 所有权保留在 Attention 侧，FFN ranks 只处理 hidden states，不拥有 KV cache。

## 主要通信后端

| Backend | 入口 | 状态 |
|---|---|---|
| `broadcast_reduce_sync` | `EPFFNLayer.forward()` | 正确性/对照路径。 |
| `broadcast_reduce_overlap` | `DecodeDBOScheduler._run_ffn_ep_overlap_decode()` | 当前真实 EPFFN decode 默认推荐基线。 |
| `all_to_all_single` | `EPFFNLayer` + `FallbackMoECommunicator` | 功能可用但当前 HCCL all-to-allv 固定开销较高，保留为实验 backend。 |
| `sparse_p2p_overlap` | `EPFFNLayer` experimental backend | CPU/Gloo reference 通过，NPU HCCL P2P 多 payload 场景 blocked。 |
| DeepEP / official MoE v2 | `coordinator_arch/comm/` 或 probe 脚本 | 中长期候选，需真实 Qwen nonzero expert path 继续验证。 |

## 新增或修改代码时的注意事项

1. 所有 FFN EP ranks 必须以完全相同的 layer-major、microbatch-major 顺序进入 HCCL collectives。
2. 异步 send/reduce/all-to-all 必须保留 tensor 引用直到 handle `wait()` 完成。
3. 不要把 decode step 1 pipeline 图当作最终 speedup 分母；最终 decode 对比使用 `decode_tpot_ms`。
4. NPU 功能和性能结论必须在 Ascend 910C 上验证；本地仅做静态检查、compileall 或无需硬件的单元测试。
