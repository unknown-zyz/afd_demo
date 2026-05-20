# Coordinator 通信方式与切换说明

本文说明 Coordinator 架构当前支持的 Attention ↔ FFN 通信方式、默认策略，以及如何切换。

---

## 1. 当前默认通信方式

当前默认通信方式是：

| 默认 | 实现 | 后端 | 状态 |
|---|---|---|---|
| ✅ | `FallbackMoECommunicator` | NPU 上为 `torch.distributed.all_to_all_single` over **HCCL** | 当前生产/验证路径 |

也就是说：

- 启动 worker 时**不传任何通信开关**，默认使用 Fallback
- 只有显式传 `--use-deepep` 时，才会启用 DeepEP
- 旧的 `--use-fallback` 已保留兼容，但现在是 no-op，因为 Fallback 已经是默认值

---

## 2. 支持的通信方式

### 2.1 Fallback（默认）

实现文件：

- `src/coordinator_arch/comm/fallback_a2a.py`

核心方法：

- `torch.distributed.all_to_all_single`

NPU 双机实验中，process group 由：

```python
dist.init_process_group(backend="hccl", init_method="env://", ...)
```

创建，因此 Fallback 在 NPU 上实际走 **HCCL**。CPU / 本地测试时才可能使用 `gloo`。

适用场景：

- 当前所有 Coordinator 架构代表性验证
- DeepEP 运行时未打通前的生产/实验路径
- 控制面、routing、micro-batch、dispatch/combine correctness 验证

优点：

- 已跨机验证可用
- 不依赖 DeepEP 自定义 kernel
- 与 PyTorch distributed 语义一致，便于调试

限制：

- 不是 kernel 级 MoE dispatch/combine 优化
- round-trip 时延高于理想 DeepEP low_latency 目标

### 2.2 DeepEP normal（实验）

实现文件：

- `src/coordinator_arch/comm/moe_communicator.py`

核心方法：

- `deep_ep.Buffer(..., low_latency_mode=False)`
- `get_dispatch_layout -> dispatch -> combine`

适用场景：

- prefill / 大 batch 风格通信
- DeepEP 修复后的性能对照

当前状态：

- 自定义 OPP vendor 安装问题已解决
- 运行时仍卡在跨机 `aclnnNotifyDispatch` timeout
- 因此标记为 **experimental**

### 2.3 DeepEP low_latency（实验）

实现文件：

- `src/coordinator_arch/comm/moe_communicator.py`

核心方法：

- `deep_ep.Buffer(..., low_latency_mode=True)`
- `low_latency_dispatch -> low_latency_combine`

适用场景：

- decode / 小 batch / 低延迟场景

当前状态：

- buffer 可创建
- warmup 首轮父进程 SIGKILL
- `rdma_size_hint=128B` 明显异常
- 因此标记为 **experimental**

---

## 3. 如何切换通信方式

### 3.1 使用默认 Fallback

不需要传额外参数：

```bash
bash scripts/launch_attn_worker.sh \
  --coord-addr 127.0.0.1:50051 \
  --rank 0 --world 1 --ffn-world 1 \
  --master-addr 127.0.0.1 --master-port 29500 \
  --device-id 0
```

```bash
bash scripts/launch_ffn_worker.sh \
  --coord-addr 127.0.0.1:50051 \
  --rank 1 --world 1 --attn-world 1 \
  --master-addr 127.0.0.1 --master-port 29500 \
  --device-id 0
```

日志中应看到：

```text
Built communicator: FallbackMoECommunicator (torch.distributed)
AttentionWorker communicator=FallbackMoECommunicator
FFNWorker communicator=FallbackMoECommunicator
```

### 3.2 显式启用 DeepEP normal

传 `--use-deepep --mode normal`：

```bash
bash scripts/launch_attn_worker.sh \
  --coord-addr 127.0.0.1:50051 \
  --rank 0 --world 1 --ffn-world 1 \
  --master-addr 127.0.0.1 --master-port 29500 \
  --mode normal \
  --use-deepep
```

### 3.3 显式启用 DeepEP low_latency

传 `--use-deepep --mode low_latency`：

```bash
bash scripts/launch_attn_worker.sh \
  --coord-addr 127.0.0.1:50051 \
  --rank 0 --world 1 --ffn-world 1 \
  --master-addr 127.0.0.1 --master-port 29500 \
  --mode low_latency \
  --use-deepep
```

注意：DeepEP 当前仍是 experimental。启用前必须先完成 DeepEP OPP vendor 安装并 source 环境，详见 `13-deepep-install-test-error-guide.md`。

---

## 4. 实验决策规则

当前推荐执行顺序：

1. 先用默认 Fallback 做代表性验证：
   - Host1 单机 smoke
   - 双机 1A1F smoke
   - cross-host fallback RT bench
   - representative E2E
2. 如果代表性实验性能符合预期，再扩到 full decode-dbo 全配置矩阵
3. 如果代表性实验性能不符合预期，先定位瓶颈：
   - HCCL `all_to_all_single` 通信耗时
   - micro-batch split/merge 开销
   - routing table / combine 开销
   - 不必要同步或 coordinator RPC 干扰
4. DeepEP 修复与 full decode-dbo 不互相阻塞；DeepEP 只有在 normal 或 low_latency 最小复现稳定后再切回主路径

---

## 5. 快速自检

检查当前默认是否为 Fallback：

```bash
python3 - <<'PY'
from unittest.mock import MagicMock, patch
from src.coordinator_arch.comm.factory import build_communicator

with patch("torch.distributed.get_world_size", return_value=1), \
     patch("torch.distributed.get_rank", return_value=0):
    comm = build_communicator(
        ep_group=MagicMock(),
        hidden_size=4096,
        num_experts=128,
        max_tokens_per_rank=4096,
        device=MagicMock(),
    )
    print(type(comm).__name__)
PY
```

期望输出：

```text
FallbackMoECommunicator
```
