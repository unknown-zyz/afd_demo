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

## Next

- D3 next: actual dispatch+combine round-trip latency (<300 µs target) with real
  hidden_states + topk_indices.
- Then wire `MoECommunicator` into attn/ffn workers end-to-end (replace
  `--no-init-dist` fallback path).
