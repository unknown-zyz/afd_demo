# Host1 Coordinator-Arch Smoke

Local multi-process smoke for the Coordinator-based MoE architecture, executed
inside `afd-npu-test` on Host1 (1.95.114.229) on branch
`feat/coordinator-arch-skeleton`.

**Date**: 2026-05-20  
**Setup**: 1 coordinator + 2 ATTN workers + 2 FFN workers, all `localhost`,
`--use-fallback --no-init-dist`, CPU device (`--device-id -1`), 16 experts,
20s duration.

**Command**:

```bash
bash scripts/launch_coordinator_arch_smoke.sh \
  --attn-world 2 --ffn-world 2 --num-experts 16 --duration 20
```

## Result: PASS

- `coordinator.log`: bound 0.0.0.0:50061; 4/4 worker RegisterWorker
  acknowledged (2× ffn, 2× attn).
- `attn_rank{0,1}.log`: `CoordinatorClient connected` → `Fetched routing
  table version 1` → `AttentionWorker initialized` → `Successfully
  registered`.
- `ffn_rank{2,3}.log`: `CoordinatorClient connected` → `Fetched routing
  table version 1` → `FFNWorker initialized` → `Successfully registered`
  → entered heartbeat loop.
- No stale-sweep eviction during the 20s window.

See `20260520_030702/` for the captured per-process logs.

## What was verified

1. gRPC control plane round-trip (RegisterWorker / GetRoutingTable /
   UpdateMetrics) on localhost.
2. Coordinator routing table v1 reaches all 4 workers.
3. Workers stay alive (heartbeat metrics push every 2s) — no stale eviction.
4. CPU/fallback path runs end-to-end without DeepEP / NPU dependencies.

## Known limitations

- `--no-init-dist` mode: no torch.distributed comm exchange (dispatch/combine
  not exercised). Will be exercised in Phase 2 with proper init.
- FFN `RegisterWorker` payload sends empty role/host (logged as `role= host=`);
  cosmetic, does not affect routing.
