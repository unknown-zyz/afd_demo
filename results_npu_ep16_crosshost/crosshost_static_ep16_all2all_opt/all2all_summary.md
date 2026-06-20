# Cross-host static EP timing summary

| EP | Backend | Attn | Fusion | Mode | MB | B | S | T | TPOT ms | Serial TPOT | Speedup | A avg/layer | F avg/layer | F/A | recv-wait | dispatch | local experts | reduce | overlap proxy |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | all_to_all_single | npu-official | rms+rope | decode-dbo | 2 | 256 | 1024 | 20 | 1325.6596761118424 |  |  | 1.599 | 1.465 | 0.92 | 2.934 | 4.015 | 1.330 | 1.190 | 0.267 |
| 16 | all_to_all_single | npu-official | rms+rope | decode-dbo | 2 | 32 | 256 | 20 | 611.7748275742327 |  |  | 0.933 | 1.595 | 1.71 | 4.246 | 4.753 | 1.418 | 1.304 | 0.235 |

说明：均值默认跳过 L0，以避免 pipeline/JIT warmup 干扰。`Speedup` 使用同 EP/backend/B/S/T 的 serial TPOT 作为 denominator。`F/A` 越接近 1，FFN 与 Attention 单层耗时越对齐。
