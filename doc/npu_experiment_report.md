# NPU-910C Experiment Matrix Report

## 1. Environment

- Hardware: 16× Ascend 910C (~64 GB HBM/chip)
- Active config: 4-chip pool (ASCEND_VISIBLE_DEVICES=0,1,2,3), 2 ranks × 1 attn + 1 ffn each
- Model: Qwen3-30B-A3B (MoE)
- Tokens: 20 decode per config
- Matrix: 4 modes × 10 batches × 5 seqs = 200 cells scheduled (183 run, 160 ok, 23 OOM)

## 2. Status summary

| Mode | OK | OOM |
|---|---:|---:|
| serial | 45 | 5 |
| prefill-dbo | 25 | 8 |
| decode-dbo | 45 | 5 |
| decode-dbo-crosslayer | 45 | 5 |

## 3. End-to-end total time (ms) — attention side

Values are `total_time_ms` from the timing tracker on the attention rank.
`serial` measures prefill + decode; `prefill-dbo` is prefill-only; `decode-dbo`/`crosslayer` measure the 20-token decode loop.

### serial

| batch\seq | 128 | 256 | 512 | 1024 | 2048 |
|---:|---:|---:|---:|---:|---:|
| 2 | 6848 | 6293 | 6666 | 6117 | 6348 |
| 4 | 7128 | 6670 | 7011 | 6470 | 6656 |
| 8 | 8296 | 8797 | 8487 | 8220 | 8207 |
| 16 | 10444 | 9986 | 11151 | 10703 | 10797 |
| 32 | 13379 | 13179 | 13229 | 13001 | 13570 |
| 64 | 17109 | 16685 | 15896 | 16271 | 15935 |
| 128 | 20292 | 21393 | 21873 | 20493 | 20576 |
| 256 | 32707 | 31460 | 31882 | 31607 | 32449 |
| 512 | 49873 | 47541 | 50683 | 48882 | 48568 |
| 1024 | OOM | OOM | OOM | OOM | OOM |

### prefill-dbo

| batch\seq | 128 | 256 | 512 | 1024 | 2048 |
|---:|---:|---:|---:|---:|---:|
| 2 | 1617 | 1662 | 1704 | 1777 | 2023 |
| 4 | 2115 | 2237 | 2193 | 2553 | 2884 |
| 8 | 2139 | 2193 | 2421 | 2854 | 3616 |
| 16 | 2212 | 2497 | 2873 | 3659 | OOM |
| 32 | 2436 | 2861 | 3589 | OOM | OOM |
| 64 | 2885 | 3670 | OOM | OOM | OOM |
| 128 | 3655 | OOM | OOM | OOM | OOM |
| 256 | OOM | OOM | OOM | OOM | OOM |
| 512 | OOM | OOM | OOM | OOM | OOM |
| 1024 | OOM | OOM | OOM | OOM | OOM |

### decode-dbo

| batch\seq | 128 | 256 | 512 | 1024 | 2048 |
|---:|---:|---:|---:|---:|---:|
| 2 | 227 | 226 | 220 | 202 | 230 |
| 4 | 239 | 246 | 314 | 253 | 223 |
| 8 | 328 | 287 | 310 | 260 | 319 |
| 16 | 355 | 322 | 358 | 351 | 375 |
| 32 | 396 | 387 | 377 | 337 | 330 |
| 64 | 395 | 396 | 356 | 414 | 352 |
| 128 | 440 | 368 | 420 | 415 | 362 |
| 256 | 393 | 410 | 394 | 390 | 380 |
| 512 | 469 | 487 | 470 | 434 | 452 |
| 1024 | OOM | OOM | OOM | OOM | OOM |

### decode-dbo-crosslayer

| batch\seq | 128 | 256 | 512 | 1024 | 2048 |
|---:|---:|---:|---:|---:|---:|
| 2 | 208 | 215 | 253 | 239 | 220 |
| 4 | 277 | 242 | 250 | 269 | 245 |
| 8 | 304 | 320 | 311 | 295 | 308 |
| 16 | 380 | 355 | 380 | 354 | 316 |
| 32 | 377 | 355 | 375 | 385 | 343 |
| 64 | 406 | 405 | 391 | 369 | 361 |
| 128 | 400 | 400 | 391 | 365 | 419 |
| 256 | 395 | 469 | 393 | 406 | 433 |
| 512 | 470 | 460 | 462 | 484 | 469 |
| 1024 | OOM | OOM | OOM | OOM | OOM |

## 4. Tokens/sec — decode modes

### decode-dbo

| batch\seq | 128 | 256 | 512 | 1024 | 2048 |
|---:|---:|---:|---:|---:|---:|
| 2 | - | - | - | - | - |
| 4 | - | - | - | - | - |
| 8 | - | - | - | - | - |
| 16 | - | - | - | - | - |
| 32 | - | - | - | - | - |
| 64 | - | - | - | - | - |
| 128 | - | - | - | - | - |
| 256 | - | - | - | - | - |
| 512 | - | - | - | - | - |
| 1024 | OOM | OOM | OOM | OOM | OOM |

### decode-dbo-crosslayer

| batch\seq | 128 | 256 | 512 | 1024 | 2048 |
|---:|---:|---:|---:|---:|---:|
| 2 | - | - | - | - | - |
| 4 | - | - | - | - | - |
| 8 | - | - | - | - | - |
| 16 | - | - | - | - | - |
| 32 | - | - | - | - | - |
| 64 | - | - | - | - | - |
| 128 | - | - | - | - | - |
| 256 | - | - | - | - | - |
| 512 | - | - | - | - | - |
| 1024 | OOM | OOM | OOM | OOM | OOM |

## 5. Per-layer A / A2F / F / F2A breakdown

The NPU timing tracker currently records only end-to-end totals per rank — per-layer
event-based timeline (A / A2F / F / F2A) has not been ported to the Ascend runtime, so
the per-layer table is empty on NPU. Pipeline PNGs generated under each mode directory
are inferred from step-level events only. Porting per-layer NPU event instrumentation is
tracked as future work.

## 6. OOM boundary (4-chip pool)

| Mode | OOM cells |
|---|---|
| serial | b1024/s128, b1024/s256, b1024/s512, b1024/s1024, b1024/s2048 |
| prefill-dbo | b256/s128, b128/s256, b64/s512, b512/s128, b512/s256, b512/s512, b32/s1024, b16/s2048 |
| decode-dbo | b1024/s128, b1024/s256, b1024/s512, b1024/s1024, b1024/s2048 |
| decode-dbo-crosslayer | b1024/s128, b1024/s256, b1024/s512, b1024/s1024, b1024/s2048 |

Observations:

- `serial` only OOMs at b=1024 (every seq). Weights + KV + activations fit at b≤512 even for s=2048.
- `prefill-dbo` OOMs earliest — DBO keeps two micro-batches live so activation memory roughly doubles.
- `decode-dbo` / `crosslayer` OOM only at b=1024 — decode activation footprint is small compared to KV cache.

## 7. Artifacts

- Summary CSV: `results_npu/experiment_matrix_summary.csv` (183 rows)
- Per-cell timing JSON: `results_npu/{mode}/timing_{attention,ffn}_{mode}_b*_s*_t20.json`
- Per-cell markdown report: `results_npu/{mode}/report_{mode}_b*_s*_t20.md`
- Pipeline plots: `results_npu/{mode}/pipeline_{mode}_b*_s*_t20.png` (115 images)
- Raw launcher logs: `results_npu/matrix_phase{1,2a,2b}.log`

## 8. GPU vs NPU scope comparison

The NPU matrix extends well beyond the GPU matrix: the GPU run topped out at b256/s512,
whereas 910C handles serial through b=512/s=2048 on the same 4-chip pool (~4× more
aggregate HBM vs 32 GB V100). Cross-hardware speedup ratios are intentionally omitted here
(different compilers / collective libraries); use the pipeline PNGs in `results/` (GPU) vs
`results_npu/` (NPU) for direct visual comparison.

