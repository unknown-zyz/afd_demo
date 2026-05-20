# Fallback default communicator validation

Date: 2026-05-20

Branch: `feat/fallback-default-comm`

## Summary

| Check | Topology | Status | Result |
|---|---|---|---|
| Factory default | Host2, 2 local NPU ranks | PASS | `build_communicator()` returns `FallbackMoECommunicator` without `--use-deepep` |
| Coordinator smoke | Host2, CPU/no-init-dist skeleton | PASS | 1 coordinator + 1 attention worker + 1 FFN worker registered and stayed alive |
| Fallback RT | Host2, 2 local NPU ranks over HCCL | PASS | 64 KiB payload: rank0 mean 314.2 us / p50 310.3 us / p99 338.6 us; rank1 mean 309.7 us / p50 302.6 us / p99 335.1 us |
| Fallback RT | Host1 NPU0 <-> Host2 NPU0 | BLOCKED | HCCL communicator creation fails with Host1 `EJ0003` bind IP/port and Host2 `EI0006` socket timeout |

## Interpretation

The fallback communicator is now the correct default and works on an actual HCCL process group in the single-host representative case. The measured Host2 single-host round trip is consistent with the prior D4 fallback baseline (roughly 300-330 us for 64 KiB).

The cross-host representative run did **not** reach a performance comparison point. It is blocked below the communicator implementation by HCCL/RoCE setup:

- Rank0: `Communication_Error_Bind_IP_Port(EJ0003): Failed to bind the IP port`
- Rank1: `Communication_Error_Get_Socket(EI0006): srcRank[192.168.0.192/0] connect destRank[192.168.0.125/0] fail`

Changing `MASTER_PORT` and `HCCL_IF_BASE_PORT` did not clear the failure. This matches the existing cross-host HCCL/DeepEP port-binding instability on Host1.

## Decision

Do **not** start full decode-dbo matrix yet. The representative fallback path passes on Host2 single-host, but the requested cross-host representative validation is blocked by HCCL port/link state. Per the performance gate, the next step is to localize and fix the cross-host HCCL bottleneck before expanding to full decode-dbo.

## Files

- `host2_intra_fallback_rt.log`: Host2 2-rank HCCL fallback RT PASS
- `crosshost_retry2_rank0.log`: Host1 rank0 cross-host HCCL failure
- `crosshost_retry2_rank1.log`: Host2 rank1 cross-host HCCL failure
- `host2_smoke/`: partial pulled smoke logs; terminal run reported PASS
