# Cross-host static EP timing summary

| EP | Backend | Attn | Fusion | Mode | MB | B | S | T | TPOT ms | A avg/layer | F avg/layer | F/A | recv-wait | dispatch | local experts | reduce | overlap proxy |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | broadcast_reduce_overlap | npu-official | rms+rope | decode-dbo | 2 | 32 | 256 | 20 | 373.1301263783519 | 0.880 | 1.301 | 1.48 | 1.687 | 1.393 | 1.179 | 0.890 | 0.336 |
| 16 | broadcast_reduce_overlap | npu-official | rms+rope | decode-dbo-crosslayer | 2 | 32 | 256 | 20 | 383.88311604381 | 0.963 | 1.397 | 1.45 | 1.809 | 1.639 | 1.260 | 0.897 | 0.335 |
| 12 | broadcast_reduce_overlap | npu-official | rms+rope | decode-dbo | 2 | 32 | 256 | 20 | 381.80594363151806 | 0.982 | 1.453 | 1.48 | 1.771 | 1.548 | 1.327 | 0.984 | 0.345 |
| 8 | broadcast_reduce_overlap | npu-official | rms+rope | decode-dbo | 2 | 32 | 256 | 20 | 415.4334830512342 | 0.879 | 1.848 | 2.10 | 2.241 | 1.787 | 1.729 | 1.134 | 0.372 |
| 16 | broadcast_reduce_overlap | hf | - | decode-dbo | 2 | 2 | 64 | 3 | 277.74286549538374 | 0.990 | 0.735 | 0.74 | 1.287 | 1.255 | 0.601 | 0.648 | 0.329 |
| 16 | broadcast_reduce_sync | hf | - | decode-dbo | 2 | 2 | 64 | 2 | 264.0429721213877 | 0.000 | 0.000 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 12 | broadcast_reduce_overlap | hf | - | decode-dbo | 2 | 2 | 64 | 2 | 254.56662592478096 | 0.000 | 0.000 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 12 | broadcast_reduce_sync | hf | - | decode-dbo | 2 | 2 | 64 | 2 | 243.83210996165872 | 0.000 | 0.000 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 8 | broadcast_reduce_overlap | hf | - | decode-dbo | 2 | 2 | 64 | 2 | 233.518609078601 | 0.000 | 0.000 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 8 | broadcast_reduce_overlap | npu-official | rms+rope | decode-dbo | 2 | 2 | 64 | 3 | 266.6969584533945 | 0.859 | 0.844 | 0.98 | 1.228 | 1.169 | 0.726 | 0.634 | 0.293 |
| 8 | broadcast_reduce_sync | hf | - | decode-dbo | 2 | 2 | 64 | 2 | 273.8042778801173 | 0.000 | 0.000 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

说明：均值默认跳过 L0，以避免 pipeline/JIT warmup 干扰。`F/A` 越接近 1，FFN 与 Attention 单层耗时越对齐。
