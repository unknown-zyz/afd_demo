# Cross-layer F→A Pre-posting — Notes

This directory contains decode-DBO runs with `--crosslayer` enabled.
Below is what the flag actually does, and why the speedup numbers in the
accompanying `pipeline_*.png` plots do **not** show a consistent win over
the plain `decode-dbo/` runs on Qwen3-30B-A3B / V100-32GB.

## What `--crosslayer` changes

`src/pipeline/decode_scheduler.py:417-443`.

During one decode step, for every transformer layer the scheduler
- issues A→F `isend`s for each micro-batch,
- waits for FFN's F→A tensors with `irecv`s.

The micro-batch (MB) loop runs inside each layer, so "when do we post the
F→A `irecv`s for layer *L*?" is a policy choice:

| Flag                        | Posting policy                                                                                             |
| --------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `--crosslayer` **OFF** (default) | Drain all current-layer A→F sends (`prev_send_handle.wait()`), **then** post this layer's F→A irecvs as a batch. |
| `--crosslayer` **ON**       | Inside the per-MB loop: immediately after each MB's `isend`, post **that MB's** F→A `irecv` on `f2a_group`. |

The intent is to shrink the inter-layer bubble: with OFF the receiver
sits idle while sends drain; with ON the receiver overlaps with the
remaining MB sends of the same layer.

## Why the measured gain is inconsistent

Qwen3-30B-A3B decode numbers (b=8, s=128, V100×4, single step):

| Segment                  | Time     |
| ------------------------ | -------- |
| Attention compute / MB / layer | ≈ 2 ms   |
| FFN compute / MB / layer        | ≈ 5 ms   |
| A→F transfer / MB / layer       | ≈ 1–2 ms |
| F→A transfer / MB / layer       | ≈ 1–2 ms |

Potential per-layer saving from pre-posting F→A: **≈ 1–2 ms**.
Across 48 layers that's up to ≈ **60 ms**. Sounds useful — but:

- Single-step wall-clock is ~500 ms.
- Run-to-run NCCL scheduling variance (cold streams, P2P buffer
  scheduling, PCIe arbitration) easily reaches ±30 ms on a single step.
- Timing is captured on **step 1** (`DecodeTimer` after 1 warm-up step in
  `decode_scheduler.py:162`). No repeats, no statistical averaging.

Net result: signal (≤ 60 ms) ≈ noise (±30 ms × √2), so the
crosslayer/baseline ratio walks randomly. In the 15 configs where we
have mode-matched baselines, crosslayer vs. non-crosslayer speedups
are within ±0.04× of each other.

| Config       | decode-dbo speedup | crosslayer speedup | Δ       |
| ------------ | ------------------ | ------------------ | ------- |
| b8_s128      | 0.73×              | 0.68×              | −0.05   |
| b16_s128     | 0.67×              | 0.64×              | −0.03   |
| b32_s128     | 0.73×              | 0.68×              | −0.05   |
| b4_s256      | 0.66×              | 0.65×              | −0.01   |
| b16_s256     | 0.61×              | 0.61×              |  0.00   |
| b8_s512      | 0.46×              | 0.50×              | +0.04   |

The Δ column is indistinguishable from noise. Conclusion: on this
hardware and workload, crosslayer is neither a clear win nor loss for
decode.

## Where it would actually help

- **Prefill** (tensors 10–100× larger; comm much heavier). Crosslayer is
  currently wired only into `decode_scheduler.py`; porting to the
  prefill `AsyncPipelineScheduler` is a sensible follow-up.
- **Hardware with faster compute / slower comm** (e.g. NPU-910C + HCCL,
  where per-layer compute shrinks but inter-GPU bandwidth is fixed
  lower). There the comm bubble is a larger fraction of the step.

## If you need a real decide

To distinguish signal from noise, run each config ≥ 5 times per flag
and compare means ± std. A helper script is sketched at
`scripts/bench_crosslayer_variance.sh` (not required for correctness).
