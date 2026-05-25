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
| Attention TP | Not implemented | Deferred until kernel/fusion path shows end-to-end benefit. |
| NPU stream overlap / async MoE communicator | Not implemented | Deferred behind TPOT-positive attention path. |

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
3. Run msprof op profiling on HF vs official+fusion for the b8/s128/t20 decode-DBO case.
4. Only after Host1 end-to-end TPOT is positive, run cross-host large-batch decode-DBO with fresh `MASTER_PORT`/`HCCL_IF_BASE_PORT`.
5. Keep community flash-attention-npu as a benchmark-only comparator unless a future shape matrix shows a clear advantage.
