# GPU/NPU DBO experiment summary

This document summarizes the current GPU and Ascend 910C result sets after the
NPU branch was rebased onto the cleaned code baseline and the baseline resolver
was updated to report TTFT/TPOT-oriented metrics.

## Code baseline

| Target | Branch / commit | Notes |
|---|---|---|
| GPU | `feat/npu-910c-rebased` @ `bef3c92` | Uses existing `results/` plus refreshed baseline audit. |
| NPU 910C | `feat/npu-910c-rebased` @ `bef3c92` | Ported backend/HCCL/NPU scripts and prefill JIT warmup fix; executed in the long-lived `afd-npu-test` container. |

The NPU branch was not rebased by replaying the full old branch. It was rebuilt
on top of the cleaned baseline and only kept the NPU backend/device abstraction,
HCCL launch path, NPU matrix scripts, NPU docs/skill, and prefill JIT warmup
logic. Old cleanup-era code such as the removed custom `kv_cache.py`, historical
validation surface, and `P2PKeepalive` path was not restored.

## Metrics

| Metric | Definition | Baseline requirement |
|---|---|---|
| TTFT-path speedup | `serial_prefill_ms / dbo_prefill_ms` | Serial cache must contain `prefill_ms`. |
| TPOT speedup | `serial_decode_tpot_ms / dbo_decode_tpot_ms` | Both serial and DBO timing must contain `decode_tpot_ms`. |
| Representative ITL | One fixed DBO decode step used for pipeline Gantt detail | Not used for TPOT speedup. |

Speedup greater than `1.0x` means DBO is faster than the serial baseline.

## Baseline completeness

| Root | Mode | Native OK | Fallback | Missing | Notes |
|---|---|---:|---:|---:|---|
| `results/` GPU | prefill TTFT-path | 15 | 0 | 0 | All prefill entries have mode-matched `prefill_ms`. |
| `results/` GPU | decode TPOT | 0 | 0 | 48 | Older decode results must be regenerated to include exact `decode_tpot_ms`. |
| `results_npu/` NPU | prefill TTFT-path | 5 | 0 | 20 | Representative serial prefill split was added for 5 configs. |
| `results_npu/` NPU | decode TPOT | 0 | 0 | 90 | Older decode results must be regenerated to include exact DBO `decode_tpot_ms`. |

The refreshed audit CSVs are:

- `results/baseline_audit.csv`
- `results_npu/baseline_audit.csv`

## GPU result summary

| Mode | Native samples | Median speedup | Min | Max | Interpretation |
|---|---:|---:|---:|---:|---|
| Prefill TTFT-path | 15 | 1.16x | 1.05x | 1.44x | GPU prefill DBO remains positive on the native-baseline subset. |
| Decode TPOT | 15 | 0.64x | 0.34x | 0.73x | GPU decode DBO is slower than serial on the native-baseline subset. |
| Decode crosslayer TPOT | 15 | 0.58x | 0.34x | 0.68x | Crosslayer does not overcome GPU decode overhead in the current implementation. |

The remaining GPU fallback configs are:

```text
b32_s512, b64_s256, b64_s512,
b96_s128, b96_s256, b96_s512,
b128_s128, b128_s256, b128_s512
```

Attempted split补采 showed the large prefill-only path is not safe on the
current V100 environment:

- `b128_s128_t20`: CUDA OOM.
- `b32_s512_t20`: CUDA OOM.
- `b64_s256_t20`: exceeded the no-output threshold and was stopped.

Because smaller/easier configs already have native split and these larger
configs fail or hang during prefill-only capture, the remaining GPU fallback
entries should be treated as capacity-limited historical results, not as missing
script coverage.

## NPU 910C result summary

| Mode | Native samples | Median speedup | Min | Max | Interpretation |
|---|---:|---:|---:|---:|---|
| Prefill TTFT-path | 5 | 0.31x | 0.28x | 0.56x | NPU prefill DBO is slower than serial in the representative native-baseline subset. |
| Decode TPOT | 5 | 1.40x | 1.38x | 2.07x | NPU decode DBO is faster than serial on the native-baseline subset. |
| Decode crosslayer TPOT | 5 | 1.32x | 1.21x | 2.28x | Crosslayer remains positive on NPU, but the sample count is small. |

Native NPU split was added for:

| Config | Prefill TTFT-path | Decode TPOT | Decode crosslayer TPOT |
|---|---:|---:|---:|
| `b2_s128_t20` | 0.29x | 1.41x | 1.53x |
| `b2_s512_t20` | 0.31x | 1.39x | 1.21x |
| `b16_s128_t20` | 0.28x | 1.38x | 1.29x |
| `b16_s512_t20` | 0.39x | 1.40x | 1.32x |
| `b128_s128_t20` | 0.56x | 2.07x | 2.28x |

`b128_s512_t20` NPU prefill split was attempted and failed with HBM OOM in the
FFN rank during the warmup pass (`Memory_Allocation_Failure`, `aclnnSilu`).

## Current conclusion

The GPU/NPU difference is still directionally credible after switching to
native TPOT where available:

- GPU: prefill DBO is beneficial, decode DBO is not.
- NPU: decode DBO is beneficial on the representative native-baseline subset,
  while prefill DBO is slower.

The earlier NPU decode “5x” headline should not be used as a final claim unless
the corresponding config has native `decode_tpot_ms`. Earlier native split
results used a representative-step DBO denominator; new runs use exact TPOT.

Remaining older NPU entries should be regenerated before making TPOT speedup
claims because they do not contain exact `decode_tpot_ms`. The current code now
requires exact TPOT fields instead of falling back to `total_time_ms / tokens`.
