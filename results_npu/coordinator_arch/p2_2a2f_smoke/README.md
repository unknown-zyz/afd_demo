# P2 — 2A2F cross-host FallbackMoECommunicator smoke

## 目的
P2：将 P1 的 2-rank 拓扑扩展到 4-rank（每机 2 NPU），验证
`FallbackMoECommunicator` 在多 rank 跨机环境下正确工作，为 P3 多 rank
真实 decode 路径奠定基础。

## 拓扑
- Host1 (192.168.0.125) `afd-npu-test`: rank0 NPU0, rank1 NPU1
- Host2 (192.168.0.192) `afd-npu-test-h2`: rank2 NPU0, rank3 NPU1
- WORLD=4, backend=HCCL, ep_group=`dist.group.WORLD`

## 启动参数
```
MASTER_PORT=29783  HCCL_IF_BASE_PORT=29793
HCCL_CONNECT_TIMEOUT=600  HCCL_EXEC_TIMEOUT=600
num_tokens=64  hidden=4096  num_experts=16  topk=4
iters=30  warmup=5
```

## 结果（2026-05-21）

| Rank | Host | mean (µs) | p50 | p99 | min | max |
|---|---|---|---|---|---|---|
| 0 | H1 | 2004.7 | 1850.4 | 3972.2 | 1787.4 | 4393.1 |
| 1 | H1 | 2005.7 | 1843.2 | 4048.5 | 1794.6 | 4410.4 |
| 2 | H2 | 2007.4 | 1842.4 | 4008.9 | 1781.9 | 4419.9 |
| 3 | H2 | 2007.0 | 1844.3 | 4020.8 | 1787.8 | 4411.4 |

**结论：PASS**。4 rank 时延高度一致（±3µs），同步性良好。

## 解读
- 相比 P1（2-rank, mean 1407µs），4-rank mean 2005µs（+40%），与
  collective all_to_all_single 在 4-rank 时双向数据量增加一致
- 均匀路由表：每个 rank 收到 ~1/4 token，counts/hidden/weights tensors 4-rank exchange
- 同步性表明 HCCL collective 在 2x2 跨机拓扑下未出现 rank 间漂移

## 文件
- `h1_rank{0,1}.log`, `h2_rank{2,3}.log`
- 脚本：`scripts/cross_host_fallback_comm_smoke.py`（与 P1 同）
