# Controller baseline — centralized CPU relay for A/F communication

Branch: `exp/controller-baseline`

This directory summarizes the first MVP of a deliberately centralized A/F
communication baseline. The goal is to compare the current decentralized,
device-direct HCCL path with a center-controller design where data plane and
control plane are not separated.

## Current direct A/F communication

Current decode DBO sends hidden states directly between the Attention rank and
the FFN coordinator rank:

```text
Attention NPU tensor --HCCL isend/irecv--> FFN coordinator NPU tensor
FFN coordinator NPU tensor --HCCL isend/irecv--> Attention NPU tensor
```

CPU/Python participates in the control path: it creates `dist.isend` /
`dist.irecv` work handles and calls `wait()`. The payload itself stays as a
device tensor and is moved by HCCL/NCCL; it is not copied through CPU memory.

Key files:

- `src/distributed/__init__.py`: lazy directional `a2f_group` / `f2a_group`
- `src/pipeline/decode_scheduler.py`: A2F/F2A `dist.isend` / `dist.irecv`
- `scripts/run_npu.sh`: rank topology and `FFN_COORDINATOR_RANK`

## Controller CPU baseline

The new mode is:

```bash
--af-comm-mode controller-cpu
```

It uses a CPU socket controller:

```text
Attention NPU
  -> D2H copy
  -> TCP payload to controller
  -> controller store-and-forward
  -> TCP payload to FFN coordinator
  -> H2D copy
  -> FFN compute

FFN coordinator NPU
  -> D2H copy
  -> TCP payload to controller
  -> controller store-and-forward
  -> TCP payload to Attention
  -> H2D copy
```

This baseline is intentionally pessimistic but not artificial: it does not add
sleep, but it forces real costs that a centralized data/control-plane design
would suffer from:

- extra hop through a center controller
- D2H and H2D copies on every A2F/F2A tensor
- CPU serialization into bytes
- blocking store-and-forward
- single FIFO relay lock
- no async `Work` handle overlap for A/F
- `--crosslayer` disabled in controller mode

## Implementation

Files:

- `src/distributed/controller_relay.py`
  - `ControllerRelayServer`: blocking two-client TCP relay
  - `ControllerRelayClient`: Attention / FFN coordinator client
  - header + payload protocol; bf16/float16/float32/int tensors are sent as raw bytes
- `scripts/run_controller.py`
  - standalone controller server entrypoint
- `scripts/run_npu.sh`
  - starts the controller when `--af-comm-mode controller-cpu`
  - passes `--af-comm-mode`, `--controller-host`, `--controller-port` to ranks
- `scripts/run_experiment_matrix_npu.sh`
  - forwards controller transport flags into `run_npu.sh`
- `src/main.py`
  - adds CLI flags
- `src/model/disaggregated.py`
  - passes A/F transport config into decode DBO scheduler
- `src/pipeline/decode_scheduler.py`
  - direct path remains unchanged
  - `controller-cpu` path uses blocking CPU relay for decode DBO A2F/F2A

MVP scope: decode DBO A/F traffic only. FFN EP collectives still run normally
inside the FFN side. Prefill/async scheduler controller relay is left for a
future extension.

## Results

Remote environment: persistent `afd-npu-test` container, preset `npu-ep7`,
backend `broadcast_reduce_overlap`.

| config | mode | Decode TPOT | A2F avg/layer | F2A send avg/layer | F2A recv-wait avg/layer | slowdown |
|---|---|---:|---:|---:|---:|---:|
| b2/s128/t8 | direct-hccl | 244.527 ms | 0.063 ms | 0.041 ms | 1.311 ms | 1.00x |
| b2/s128/t8 | controller-cpu | 13818.372 ms | 94.231 ms | 94.359 ms | 47.978 ms | 56.51x |
| b16/s256/t20 | direct-hccl | 344.979 ms | 0.068 ms | 0.044 ms | 1.936 ms | 1.00x |
| b16/s256/t20 | controller-cpu | 13828.117 ms | 94.247 ms | 93.988 ms | 48.072 ms | 40.08x |

The controller baseline is therefore dramatically worse while preserving the
same model-side compute. This makes it a useful “centralized control+data plane”
baseline against the current direct HCCL decentralized path.

## Relay details

For b16/s256/t20 controller-cpu:

- A2F messages: 1824, mean relay total 143.925 ms
- F2A messages: 1824, mean relay total 144.004 ms
- Total relayed bytes: 119,537,664 bytes

The per-layer report shows A2F/F2A enqueue bars around 94 ms/layer, while direct
HCCL is around 0.04-0.07 ms/layer in enqueue mode. That gap is expected: the
controller path measures blocking CPU relay completion, not just `isend()`
enqueue overhead.

## Reproduction

Controller CPU smoke:

```bash
bash scripts/run_experiment_matrix_npu.sh \
  --modes decode-dbo \
  --batches 2 --seqs 128 --tokens 8 \
  --preset npu-ep7 \
  --ffn-ep-backend broadcast_reduce_overlap \
  --af-comm-mode controller-cpu \
  --output-root results_npu/controller_baseline_smoke \
  --no-cache
```

Direct HCCL comparison:

```bash
bash scripts/run_experiment_matrix_npu.sh \
  --modes decode-dbo \
  --batches 2 --seqs 128 --tokens 8 \
  --preset npu-ep7 \
  --ffn-ep-backend broadcast_reduce_overlap \
  --af-comm-mode direct-hccl \
  --output-root results_npu/controller_baseline_direct \
  --no-cache
```

Representative controller run:

```bash
bash scripts/run_experiment_matrix_npu.sh \
  --modes decode-dbo \
  --batches 16 --seqs 256 --tokens 20 \
  --preset npu-ep7 \
  --ffn-ep-backend broadcast_reduce_overlap \
  --af-comm-mode controller-cpu \
  --output-root results_npu/controller_baseline_b16 \
  --no-cache
```

