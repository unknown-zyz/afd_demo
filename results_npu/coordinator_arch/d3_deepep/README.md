# D3 — Cross-host DeepEP Buffer construction smoke

Date: 2026-05-20
Branch: `feat/coordinator-arch-skeleton` @ 2ccdc80
Script: `scripts/cross_host_deepep_smoke.py`

## Topology

| Rank | Host  | Container        | NPU | RoCE IP        |
|------|-------|------------------|-----|----------------|
| 0    | Host1 | afd-npu-test     | 0   | 192.168.0.125  |
| 1    | Host2 | afd-npu-test-h2  | 0   | 192.168.0.192  |

Env: `MASTER_ADDR=192.168.0.125 MASTER_PORT=29585 HCCL_IF_BASE_PORT=24600`

## Result: PASS

Both ranks:
1. `init_process_group(backend='hccl')` ok
2. `all_reduce` cross-host: sum=24 (expected 24), dt≈841 ms (first-op HCCL init overhead)
3. `deep_ep.Buffer(low_latency_mode=True)` constructed in <1 ms
4. `deep_ep.Buffer(low_latency_mode=False)` constructed in <1 ms

Buffer exposes `dispatch`, `combine`, `low_latency_combine`, `internode_dispatch`,
`internode_combine`, `fused_deep_moe` — confirms internode RoCE path is enabled.

## Note (rank 1)

`indexFromRank 1 is not equal indexFromCurDevice 0` warning is benign on
asymmetric HCCS topology (rank 1's local device id is 0 since `ASCEND_VISIBLE_DEVICES`
limits visibility); collective still succeeded.

## RT Bench: BLOCKED (API mismatch)

`scripts/cross_host_deepep_rt_bench.py` calls
`buf.get_dispatch_layout(topk_idx, num_experts)` → fails on both ranks:

```
RuntimeError: aclnnDispatchLayout or aclnnDispatchLayoutGetWorkspaceSize
not in libopapi.so, or libopapi.so not found.
```

Root cause: this CANN ships the *new* MoE distribute API
(`aclnnMoeDistributeDispatchV4` + `aclnnMoeDistributeCombineV2` + AddRmsNorm
variants, confirmed via `strings libopapi.so`), but the deep_ep
`1.0.0+0ff3be00.cann.8.5.0.b232` wheel was built against the *older*
`aclnnDispatchLayout` op which has been removed/renamed.

`Buffer.dispatch()` (normal mode) and `Buffer.get_dispatch_layout()` both rely on
the missing op. `low_latency_combine` is exposed but no matching
`low_latency_dispatch` is present in this wheel's symbol table either.

**Action**: D3 dispatch/combine RT benchmark cannot proceed with this DeepEP wheel +
CANN combination. Options to unblock (out of scope for this session):

1. Rebuild deep_ep against current CANN (uses `aclnnMoeDistributeDispatchV4`)
2. Install older CANN bundle matching the wheel
3. Bypass deep_ep and call `aclnnMoeDistributeDispatchV4` directly via
   `torch_npu.npu_*` (would replace `MoECommunicator` implementation)

For now we proceed with **D4 (fallback)** via `torch.distributed.all_to_all_single`
over HCCL, which is the system's documented fallback path when DeepEP is
unusable.

## Logs

- `rank0.log` / `rank1.log` — Buffer construction PASS
- `rt_rank0.log` / `rt_rank1.log` — RT bench FAIL (`aclnnDispatchLayout` missing)

