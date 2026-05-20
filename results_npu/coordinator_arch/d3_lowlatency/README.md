# D3 Phase E1 — DeepEP normal & low_latency cross-host RT bench results

## Summary

| Phase | Mode | Cross-host | Status | Notes |
|---|---|---|---|---|
| E1.0 OPP install | n/a | both hosts | ✅ DELIVERED | Custom OPP vendor installed on Host1 (via `.run`) and Host2 (manual cp) |
| E1.1 RT redo | normal | Host1 NPU0 ↔ Host2 NPU0 | ❌ FAIL — aicore timeout 507014 | Symbol resolved; kernel hangs on cross-host RDMA notify |
| E1.2 RT new | low_latency | Host1 NPU0 ↔ Host2 NPU0 | ❌ FAIL — TBE subprocess / parent SIGKILL | rdma_size_hint=128 B suspicious; main process crashes inside warmup iter 0 |
| (ref) D4 fallback | all_to_all_single (HCCL) | same | ✅ 330 µs mean | See `../d4_fallback/` |

## E1.0 — DeepEP custom OPP vendor install (CRITICAL prerequisite)

The previous session blocked D3 on `aclnnDispatchLayout` symbol missing in `libopapi.so`.
**Root cause is NOT a wheel/CANN version mismatch** — it is a missed install step.

DeepEP-Ascend ships its custom MoE ops (`aclnnDispatchLayout`,
`aclnnDispatchNormalA2`, `aclnnDispatchFFNCombine`, `aclnnMoeDistributeCombineA2`) as
a separate OPP vendor installer at
`csrc/deepep/ops/build_out/custom_opp_ubuntu_aarch64.run`. Without running this
installer, `/usr/local/Ascend/cann-X.Y/opp/vendors/` is empty and CANN cannot resolve
the symbols.

### Install commands

**Host1** (has SGLang-Kernel-NPU source):

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash /workspace/sglang-kernel-npu/csrc/deepep/ops/build_out/custom_opp_ubuntu_aarch64.run \
     --quiet --install-path=/usr/local/Ascend/cann-8.5.0/opp
```

**Host2** (no source — copy vendor tree from site-packages):

```bash
DEEPEP_DIR=/usr/local/python3.11.14/lib/python3.11/site-packages/deep_ep/vendors/hwcomputing
VENDORS=/usr/local/Ascend/cann-8.5.0/opp/vendors
mkdir -p "$VENDORS"
cp -a "$DEEPEP_DIR" "$VENDORS/"
# Rewrite set_env.bash with the real install path (default points at build dir):
cat > "$VENDORS/hwcomputing/bin/set_env.bash" <<'EOF'
#!/bin/bash
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-8.5.0/opp/vendors/hwcomputing:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.0/opp/vendors/hwcomputing/op_api/lib/:${LD_LIBRARY_PATH}
EOF
```

### Activation (every shell)

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-8.5.0/opp/vendors/hwcomputing/bin/set_env.bash
```

### Verify

```bash
nm -D /usr/local/Ascend/cann-8.5.0/opp/vendors/hwcomputing/op_api/lib/libcust_opapi.so \
  | grep aclnnDispatchLayout
# expect:  T aclnnDispatchLayout / T aclnnDispatchLayoutGetWorkspaceSize
```

## E1.1 redo — normal mode (RDMA notify timeout)

After installing the OPP vendor, the symbol error is gone. The bench reaches
`buf.get_dispatch_layout` and `buf.dispatch` successfully but then hangs on
`aclnnNotifyDispatch` → AICore timeout 507014. See `rt_normal_rank0.log`:

```
[rank0] hccl pg ok
[rank0] buffer (normal) created
[rank0] warmup...
EZ9999: Kernel task happen error, retCode=0x25, [aicore timeout]
[rank0] WARMUP FAIL @ iter 0: aclnnNotifyDispatch failed,
        EI0006 Communication_Error_Get_Socket: Getting socket times out.
        srcRank[192.168.0.125/0] connect destRank[192.168.0.192/0] fail.
```

**Diagnosis**: `aclnnNotifyDispatch` opens its own socket pair (separate from the HCCL
allreduce pg socket which already works), bound to ports based on `HCCL_IF_BASE_PORT`.
The connect times out → kernel waiting on RDMA notification never lands → aicore
timeout. Likely needs proper RoCE NIC binding (`HCCL_NET_DEVICE`, per-rank NIC mapping
via `hccn_tool -i 0 -ip -g`) which is not configured in our generic Ubuntu container.
This is **not** a wheel/CANN version issue.

## E1.2 — low_latency mode (parent process SIGKILL)

Script: `scripts/cross_host_deepep_lowlatency_rt_bench.py` (commit 8971c358).

API used:
```python
buf = deep_ep.Buffer(
    group=dist.group.WORLD,
    num_nvl_bytes=0,
    num_rdma_bytes=deep_ep.Buffer.get_low_latency_rdma_size_hint(
        num_max, hidden, world, num_experts),
    low_latency_mode=True,
    num_qps_per_rank=num_experts // world,
)
ret = buf.low_latency_dispatch(x, topk_idx, num_max, num_experts, use_fp8=False)
combined, _evt, _hook = buf.low_latency_combine(recv_x, topk_idx, topk_w, handle=ret[2])
```

Result (both ranks fail identically inside warmup iter 0), see `rt_ll_rank0.log`:

```
[rank0] rdma_size_hint=128 bytes      # suspiciously small
[rank0] buffer (low_latency) created
[rank0] warmup...
[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!
resource_tracker: There appear to be 30 leaked semaphore objects to clean up at shutdown
```

The python main process silently SIGKILLs (TBE subprocesses only print after parent
disappearance). Most likely the kernel triggers a hard segfault during HCCL/RDMA QP
setup, but the parent crash kills before any traceback is printed.

`rdma_size_hint=128 B` for hidden=4096, num_max=128, world=2, experts=128 is suspicious —
either the formula expects parameters differently or this DeepEP build is targeting a
future CANN with different memory layout. Likely correlates with the SIGKILL.

## Conclusion

The DeepEP-Ascend custom op symbols are now resolvable on both hosts (E1.0 unlock
delivered). However both **normal-mode** dispatch (E1.1) and **low_latency-mode**
dispatch (E1.2) fail at op execution time during cross-host RDMA setup. The fallback
`torch.distributed.all_to_all_single` path (D4) remains the only viable
attention↔FFN communication mechanism cross-host on this CANN 8.5.0.b232 / driver
25.5.1 / torch_npu 2.6.0 stack.

**Recommendation**: proceed E2/E3 with `FallbackMoECommunicator` as the production
arm; mark `DeepEPCommunicator` arm as experimental, gated behind a feature flag, and
re-enable when RDMA hardware path is characterised by a CANN/HCCL specialist.

## Files

- `rt_normal_rank0.log` / `rt_normal_rank1.log` — E1.1 normal-mode dispatch failure
- `rt_ll_rank0.log` / `rt_ll_rank1.log` — E1.2 low_latency-mode dispatch failure

## Future investigation (when RDMA specialist available)

1. Capture full ASCEND log around `aclnnNotifyDispatch` failure
   (`ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1`).
2. Inspect HCCL/RoCE — `hccn_tool -i 0 -ip -g`, `hccn_tool -i 0 -tls -g`, NIC pinning.
3. Try DeepEP unit tests bundled in `csrc/deepep/tests/` cross-host to confirm
   whether the library itself works on this stack.
4. Open issue against `unkown/SGLang-Kernel-NPU` with CANN 8.5.0.b232 + torch_npu 2.6.0
   to identify a known-working combination.
