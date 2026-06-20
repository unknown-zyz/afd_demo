# Cross-host static EP timing summary

| EP | Backend | Attn | Fusion | Mode | MB | B | S | T | TPOT ms | Serial TPOT | Speedup | A avg/layer | F avg/layer | F/A | recv-wait | dispatch | local experts | reduce | overlap proxy |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | all_to_all_single | npu-official | rms+rope | decode-dbo | 2 | 32 | 256 | 20 | 636.2598664740002 |  |  | 0.986 | 1.160 | 1.18 | 3.732 | 4.368 | 1.042 | 1.043 | 0.197 |

说明：均值默认跳过 L0，以避免 pipeline/JIT warmup 干扰。`Speedup` 使用同 EP/backend/B/S/T 的 serial TPOT 作为 denominator。`F/A` 越接近 1，FFN 与 Attention 单层耗时越对齐。
