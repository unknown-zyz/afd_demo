# AttentionWorker NPU optimization progress

## Summary

Branch: `feat/attn-worker-npu-optimizations`.

Validated on Ascend 910C in Host1 `afd-npu-test`; GPU results are not used as acceptance evidence.

Current conclusion:

- Official NPU decode core (`torch_npu.npu_incre_flash_attention`) reaches tens of microseconds for the tested Qwen3 layer shapes and is correct against HF/SDPA.
- Official prefill full-layer path is now numerically aligned after adding an explicit causal mask, but remains slower than the HF path for the tested seq=128 shapes.
- RMSNorm/RoPE fusion reduces the official prefill overhead versus official-only, but still does not beat HF prefill in the tested shapes.
- Community `MinghuasLab/flash-attention-npu` v2/v3 builds and runs, but its `flash_attn_with_kvcache` decode path is about two orders of magnitude slower than official IFA on the tested Qwen-like shapes and does not support RoPE fusion.
- A Host1 2-rank end-to-end decode-DBO check did not show TPOT improvement for official or official+fusion, so multi-node large-batch decode-DBO is not promoted yet.

## Implemented changes

| Area | Status | Notes |
|---|---|---|
| Optimization config and timing metadata | Implemented | CLI/env controls and timing JSON metadata for attention kernel, precopy, fusion, TP placeholder, stream overlap, active/reserved NPUs. |
| Layer input precopy | Implemented | `AttentionLayerInputCache` prepositions mask, position ids, and RoPE tensors per layer device. |
| Official NPU attention adapter | Implemented | Prefill uses `torch_npu.npu_prompt_flash_attention` when `npu_flash_attention` is unavailable; decode uses `torch_npu.npu_incre_flash_attention`. |
| Causal mask for official prefill | Implemented | Explicit upper-triangular bool mask is created when no mask is supplied and `seq_len > 1`. |
| RMSNorm/RoPE fusion | Implemented | Uses `torch_npu.npu_rms_norm` or `npu_fused_rms_norm` if present, and `torch_npu.npu_rotary_mul`. |
| msprof workflow | Implemented | `run_npu.sh`/matrix support msprof collection and analysis flags. |
| Reserved NPU pool | Implemented | `--reserved-npus` removes devices from active launch pool and records metadata. |
| NPU stream overlap | Implemented for static A/F paths | `--attn-stream-overlap` now controls accelerator compute streams in prefill and decode DBO schedulers; MB3/MB4 NPU smokes passed. |
| Async MoE communicator protocol | Implemented at communicator API level | Fallback all-to-all payloads use `async_op=True` handles with tensor lifetime retention; coordinator skeleton now calls async dispatch/combine. |
| Attention TP | Not implemented | Requires multi-attention-rank execution semantics and Q/K/V/O weight partitioning; blocked until a safe TP topology is designed. |

## Single-layer benchmark

Benchmark script: `scripts/bench_attention_layer_npu.py`.

Artifacts:

- `results_npu/attention_layer_bench/median_b1_s128_c32_128_512.json`
- `results_npu/attention_layer_bench/median_b4_s128_c128_512.json`

Environment:

- Host1 `afd-npu-test`
- Model: `/models/Qwen3-30B-A3B`
- Layer: 0
- dtype: `float16`
- warmup: 5
- iterations per repeat: 20
- repeats: 3
- reported latency: median of repeat averages

Command example:

```bash
ASCEND_VISIBLE_DEVICES=0 python scripts/bench_attention_layer_npu.py \
  --model-name /models/Qwen3-30B-A3B \
  --device npu:0 \
  --dtype float16 \
  --batch 1 \
  --seq 128 \
  --decode-cache-lens 32,128,512 \
  --warmup 5 \
  --iters 20 \
  --repeats 3 \
  --include-community \
  --community-root /tmp/flash-attention-npu \
  --output results_npu/attention_layer_bench/median_b1_s128_c32_128_512.json
```

### Prefill full-layer latency

| Batch | Seq | Case | Median latency | Max abs diff vs HF | Result |
|---:|---:|---|---:|---:|---|
| 1 | 128 | HF | 0.775 ms | N/A | Baseline |
| 1 | 128 | official | 1.115 ms | 1.22e-4 | Slower |
| 1 | 128 | official + fused RMSNorm | 1.007 ms | 2.44e-4 | Slower |
| 1 | 128 | official + fused RoPE | 0.999 ms | 1.22e-4 | Slower |
| 1 | 128 | official + both fusions | 0.936 ms | 2.44e-4 | Slower |
| 4 | 128 | HF | 0.691 ms | N/A | Baseline |
| 4 | 128 | official | 1.104 ms | 1.22e-4 | Slower |
| 4 | 128 | official + fused RMSNorm | 0.901 ms | 2.44e-4 | Slower |
| 4 | 128 | official + fused RoPE | 0.885 ms | 1.22e-4 | Slower |
| 4 | 128 | official + both fusions | 0.831 ms | 2.44e-4 | Slower |

### Decode core latency

| Batch | Cache len | Case | Median latency | Diff vs HF/SDPA | Result |
|---:|---:|---|---:|---:|---|
| 1 | 32 | HF/SDPA | 0.0519 ms | N/A | Baseline |
| 1 | 32 | official IFA | 0.0364 ms | 0 | Faster |
| 1 | 32 | community v2 | 4.7734 ms | 0 | Much slower |
| 1 | 32 | community v3 | 4.8970 ms | 0 | Much slower |
| 1 | 128 | HF/SDPA | 0.0397 ms | N/A | Baseline |
| 1 | 128 | official IFA | 0.0406 ms | 0 | Similar/slightly slower |
| 1 | 128 | community v2 | 4.5507 ms | 4.77e-7 | Much slower |
| 1 | 128 | community v3 | 5.1499 ms | 4.77e-7 | Much slower |
| 1 | 512 | HF/SDPA | 0.0417 ms | N/A | Baseline |
| 1 | 512 | official IFA | 0.0421 ms | 0 | Similar/slightly slower |
| 1 | 512 | community v2 | 4.4366 ms | 1.91e-6 | Much slower |
| 1 | 512 | community v3 | 4.8621 ms | 1.91e-6 | Much slower |
| 4 | 128 | HF/SDPA | 0.0403 ms | N/A | Baseline |
| 4 | 128 | official IFA | 0.0284 ms | 0 | Faster |
| 4 | 512 | HF/SDPA | 0.0514 ms | N/A | Baseline |
| 4 | 512 | official IFA | 0.0289 ms | 0 | Faster |

## End-to-end Host1 decode-DBO check

Topology: Host1 single machine, 2 ranks, `--attn-size 1 --ffn-size 1 --ffn-tp-size 1`, batch 8, seq 128, tokens 20.

Artifacts in Host1 worktree:

- `results/prefill_dbo/timing_attention_host1_hf_b8_s128_t20.json`
- `results/prefill_dbo/timing_attention_host1_official_b8_s128_t20.json`
- `results/prefill_dbo/timing_attention_host1_official_fused_b8_s128_t20.json`

| Case | Prefill | Decode TPOT | Total | Interpretation |
|---|---:|---:|---:|---|
| HF | 1248.0 ms | 319.3 ms | 325.6 ms | Baseline |
| official | 1350.5 ms | 374.6 ms | 312.7 ms | TPOT slower despite lower total timing field |
| official + fused RMSNorm/RoPE | 1673.7 ms | 348.6 ms | 310.8 ms | TPOT still slower than HF |

The single-layer decode-core improvement does not yet transfer cleanly to the real decode-DBO TPOT metric. Multi-node large-batch decode-DBO should wait until the full AttentionLayer path or scheduler integration shows TPOT-positive behavior on Host1.

## Stream-overlap and MB3/MB4 validation

`--attn-stream-overlap` is now wired into both prefill `AsyncPipelineScheduler` and decode `DecodeDBOScheduler`. When enabled on CUDA/NPU backends, scheduler compute sections run on an accelerator compute stream while communication remains asynchronous through the existing distributed send/recv handles.

Host1 NPU smoke commands used the 2-rank topology with `--attn-size 1 --ffn-size 1 --ffn-tp-size 1`:

```bash
ASCEND_VISIBLE_DEVICES=0,1 MASTER_PORT=29841 HCCL_IF_BASE_PORT=29901 \
  bash scripts/run_npu.sh --batch 3 --seq 32 --tokens 4 \
    --no-generate --attn-stream-overlap --num-micro-batches 3 \
    --model-name /models/Qwen3-30B-A3B

ASCEND_VISIBLE_DEVICES=2,3 MASTER_PORT=29842 HCCL_IF_BASE_PORT=29922 \
  bash scripts/run_npu.sh --batch 3 --seq 32 --tokens 4 \
    --attn-stream-overlap --num-micro-batches 3 \
    --model-name /models/Qwen3-30B-A3B

ASCEND_VISIBLE_DEVICES=4,5 MASTER_PORT=29843 HCCL_IF_BASE_PORT=29943 \
  bash scripts/run_npu.sh --batch 4 --seq 32 --tokens 4 \
    --no-generate --attn-stream-overlap --num-micro-batches 4 \
    --model-name /models/Qwen3-30B-A3B

ASCEND_VISIBLE_DEVICES=6,7 MASTER_PORT=29844 HCCL_IF_BASE_PORT=29964 \
  bash scripts/run_npu.sh --batch 4 --seq 32 --tokens 4 \
    --attn-stream-overlap --num-micro-batches 4 \
    --model-name /models/Qwen3-30B-A3B
```

All four runs exited 0. Timing metadata confirmed `attn_stream_overlap: true` and `num_micro_batches` of 3 or 4:

| Case | num_micro_batches | total_time_ms | decode_tpot_ms |
|---|---:|---:|---:|
| prefill b3/s32/t4 | 3 | 660.0 | N/A |
| decode b3/s32/t4 | 3 | 278.0 | 304.0 |
| prefill b4/s32/t4 | 4 | 858.3 | N/A |
| decode b4/s32/t4 | 4 | 387.4 | 405.6 |

## Async MoE communicator protocol

The coordinator communicator interface now includes:

- `dispatch_async(...)`
- `wait_dispatch(handle)`
- `combine_async(...)`
- `wait_combine(handle)`

For `FallbackMoECommunicator`, dispatch still exchanges route counts synchronously because payload sizes depend on those counts. The hidden-state and weight payload all-to-alls are then enqueued with `async_op=True`, and the returned handle keeps send buffers, count tensors, receive buffers, and `dist.Work` objects alive until `wait_dispatch()`.

`combine_async()` similarly enqueues the reverse all-to-all with `async_op=True`; `wait_combine()` waits and then performs inverse permutation plus top-k weighting. The DeepEP wrapper exposes the same methods as compatibility handles around its current synchronous wrapper calls. The coordinator skeleton worker now uses the async methods, so future real FFN serving can overlap dispatch of later microbatches with combine of earlier ones without changing the public communicator API.

Validation:

- Host1 container compile check for `src/coordinator_arch/comm/*`, coordinator attention worker, and communicator tests passed.
- Host1 mocked-collective smoke verified fallback `dispatch_async -> wait_dispatch -> combine_async -> wait_combine` returns the expected tensor and waits the payload work handles.

## Community flash-attention-npu assessment

Source: `MinghuasLab/flash-attention-npu`.

Preparation:

- Cloned locally because Host1 GitHub clone hit TLS timeouts.
- Pulled `csrc/catlass` submodule.
- Transferred tarball to Host1 and built in place under `/tmp/flash-attention-npu`.
- Built artifacts:
  - `/tmp/flash-attention-npu/flash_attn_npu_2.cpython-311-aarch64-linux-gnu.so`
  - `/tmp/flash-attention-npu/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`

Findings:

- `flash_attn_with_kvcache` imports and runs for decode.
- RoPE fusion is not available: passing `rotary_cos`/`rotary_sin` reports `NPU FlashAttention does not support rotary embedding`.
- v2/v3 decode latency is roughly 4.4-5.2 ms in tested shapes, versus official IFA at about 0.03-0.04 ms.
- Current data does not support using the community implementation as the production default for this project.

## Next plan

1. Do not default-enable official prefill or fusions yet; prefill full-layer is still slower than HF in the tested shapes.
2. Investigate why official decode core does not improve end-to-end decode TPOT:
   - include output projection and KV-cache update in the decode microbenchmark,
   - add per-stage timing inside `AttentionLayer._forward_npu_official_attention`,
   - compare layout transposes, QKV projection cost, and scheduler timing.
3. Extend the stream-overlap path to EP async MoE communicator handles only after the static A/F stream path has a TPOT-positive configuration.
4. Run msprof op profiling on HF vs official+fusion for the b8/s128/t20 decode-DBO case.
5. Only after Host1 end-to-end TPOT is positive, run cross-host large-batch decode-DBO with fresh `MASTER_PORT`/`HCCL_IF_BASE_PORT`.
6. Keep community flash-attention-npu as a benchmark-only comparator unless a future shape matrix shows a clear advantage.
