# P1 — Cross-host FallbackMoECommunicator smoke

## 目的
P1：验证 `FallbackMoECommunicator.dispatch/combine`（即真实 worker 用的通信类）
在 Host1↔Host2 跨机环境下能正确执行，而不仅仅是裸 `all_to_all_single`
（后者已由 `cross_host_fallback_rt_bench.py` 验证）。

## 拓扑
- Host1 (192.168.0.125) `afd-npu-test`: rank0, NPU0
- Host2 (192.168.0.192) `afd-npu-test-h2`: rank1, NPU0
- WORLD=2, backend=HCCL, ep_group = `dist.group.WORLD`

## 启动参数
```
MASTER_PORT=29782  HCCL_IF_BASE_PORT=29792
HCCL_CONNECT_TIMEOUT=600  HCCL_EXEC_TIMEOUT=600
num_tokens=64  hidden=4096  num_experts=16  topk=4
iters=30  warmup=5
```

## 结果（2026-05-21）

| Rank | mean (µs) | p50 | p99 | min | max |
|---|---|---|---|---|---|
| rank0 (H1) | 1407.5 | 1305.6 | 2295.9 | 1275.1 | 3294.6 |
| rank1 (H2) | 1409.7 | 1306.8 | 2271.6 | 1278.7 | 3368.0 |

**结论：PASS**。output shape 验证通过；30 iters 0 errors。

## 解读
- 单次 dispatch+combine 含 5 次 collective `all_to_all_single`
  （counts ×2、hidden ×2、weights ×1）
- 平均 1.3 ms ≈ 5 × 260µs/coll，与之前裸 RT bench（~400µs/coll @ 512 KiB）量级一致
- 端到端业务延迟远高于 300µs RT 设计目标，需在 P3/P4 评估对真实 decode 的影响

## 修复过程
1. 初次失败 `EJ0003 bind IP/port`：HCCL 用 `dist.new_group()` 创建 ep_group 会额外占一个 bind 端口，与 `HCCL_IF_BASE_PORT` 冲突
2. 修复（`ea756bb`）：直接复用 `dist.group.WORLD` 作为 ep_group
3. 换 fresh 端口 29782/29792 后重跑 → PASS

## 文件
- `h1_rank0.log`, `h2_rank1.log`：完整运行日志
- 脚本：`scripts/cross_host_fallback_comm_smoke.py`
