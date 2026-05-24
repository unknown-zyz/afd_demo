# P3 Cross-host real-decode 1A7F — debug log (2026-05-21)

## Goal
Real Qwen3-30B-A3B decode across 1 Attention (Host1 NPU0) + 7 FFN (Host2 NPU0..6), fallback comm.

## Outcome so far
**HCCL bootstrap PASS**; **first prefill warmup blocked on TBE JIT compile > 60 min wall clock**.

## What works
- After both containers restarted + fresh ports (MASTER=29795, H1_BASE=30300, H2_BASE=30400):
  - All 8 ranks reach `Distributed initialized: world_size=8`
  - FFN EP sub-groups created on H2 ranks 1-7
  - Weights load, AFDCommunicator init, "Communicator set up"
  - First prefill warmup begins with "Split batch into 2 micro-batches"

## What fails
- Processes stay in 'S' (sleep) state with all CPU time burned in first ~20 min, then idle waiting on TBE compile dispatcher (`tbe.common.repository_manager.utils.multiprocess_util`).
- `timeout 3600` script-level wrapper kills python after 60 min wall clock — first prefill warmup never completes.
- No usable timing data produced.

## Fixes applied
- **Commit `6c42639` (feat/fallback-default-comm)**: bump torch dist collective timeout 30min → 2h in `src/distributed/__init__.py` for `init_process_group` + 5 `new_group` calls. Env override: `AFD_DIST_TIMEOUT_SEC`.
  - Verified working: in the second v6 attempt (with this fix), all 22 ranks survived past the previous 30-min death point. Previous run's `ERR02005 DIST internal error` at min 30 is fully eliminated.

## Recommended next steps (not executed; needs longer wall budget)
1. **Single-host TBE warmup per host**, e.g.:
   - H1: `bash scripts/run_npu.sh --preset npu-attention --batch 2 --seq 128 --tokens 5` (~30-60 min cold)
   - H2: `bash scripts/run_npu.sh --preset npu-ep7 --batch 2 --seq 128 --tokens 5` (~30-60 min cold)
   - Persists `kernel_meta/` cache so the subsequent cross-host run skips cold JIT.
2. After both `kernel_meta/` populated, bump script `timeout` to 1800s and re-run v6 cross-host. Should complete in 5-15 min total.
3. If even single-host TBE warmup exceeds 1h, consider:
   - Setting `ASCEND_GLOBAL_CACHE_ENABLE=1`
   - Running with `--prefill-warmup-rounds 0` (skip warmup) and accepting L0 timing pollution
   - Using a smaller model (Qwen3-8B) for the e2e plumbing validation
4. After P3 PASS, proceed to P4 representative 3x3 decode matrix vs `results_npu_ep7` baseline.

## Files
- `h1_rank0_v6_post-timeout-fix.log` — H1 log from second v6 attempt (with timeout fix). Shows process running, no errors, just no progress past warmup start.

## Repro
- Branch: `feat/fallback-default-comm` @ `6c42639`
- Launch scripts: `/tmp/p3_h1_attn_v6.sh` (H1), `/tmp/p3_h2_ffn_v6.sh` (H2) staged inside containers
- See checkpoint `016-p3-cross-host-v6-ej0003-restar.md` for prior debug history
