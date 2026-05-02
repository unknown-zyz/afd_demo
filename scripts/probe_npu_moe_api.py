#!/usr/bin/env python3
"""Probe torch_npu MoE/grouped-matmul APIs on Qwen3 MoE tensors."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM


def parse_shapes(value: str) -> list[tuple[str, int, int]]:
    shapes: list[tuple[str, int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        label, dims = item.split(":", 1) if ":" in item else ("custom", item)
        batch_s, seq_s = dims.lower().split("x", 1)
        shapes.append((label, int(batch_s), int(seq_s)))
    return shapes


def dtype_from_name(name: str) -> torch.dtype:
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def setup_device(device_name: str) -> torch.device:
    if device_name.startswith("npu"):
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu"):
            torch.npu.set_device(device_name)
    return torch.device(device_name)


def sync_device(device: torch.device) -> None:
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, fn: Callable[[], Any]) -> tuple[Any, float]:
    sync_device(device)
    start = time.perf_counter()
    out = fn()
    sync_device(device)
    return out, time.perf_counter() - start


def median_ms(values_s: list[float]) -> float:
    return statistics.median(values_s) * 1000.0


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float().cpu()
    got = candidate.float().cpu()
    diff = (ref - got).abs()
    denom = ref.abs().clamp_min(1e-6)
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "max_rel_error": float((diff / denom).max().item()),
        "mean_rel_error": float((diff / denom).mean().item()),
    }


def failure_record(
    *,
    probe: str,
    layer_idx: int,
    shape_label: str,
    batch: int,
    seq: int,
    dtype: torch.dtype,
    device: torch.device,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "probe": probe,
        "status": "error",
        "layer": layer_idx,
        "shape_label": shape_label,
        "batch": batch,
        "seq": seq,
        "tokens": batch * seq,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(limit=8),
    }


def build_expert_inputs(
    hidden_2d: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts)
    expert_mask = expert_mask.permute(2, 1, 0)
    active = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
    xs: list[torch.Tensor] = []
    token_indices: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for expert_idx in active:
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        xs.append(hidden_2d[token_idx].contiguous())
        token_indices.append(token_idx)
        weights.append(routing_weights[token_idx, top_k_pos, None].contiguous())
    return active, xs, token_indices, weights


def probe_router(
    *,
    torch_npu,
    layer,
    logits: torch.Tensor,
    hf_weights: torch.Tensor,
    hf_indices: torch.Tensor,
    layer_idx: int,
    shape_label: str,
    batch: int,
    seq: int,
    dtype: torch.dtype,
    device: torch.device,
    repeat: int,
) -> dict[str, Any]:
    k = int(hf_indices.shape[-1])
    times: list[float] = []
    out = None
    for _ in range(repeat):
        out, elapsed = timed(device, lambda: torch_npu.npu_moe_gating_top_k_softmax(logits, None, k))
        times.append(elapsed)
    npu_weights, npu_indices, _row_idx = out
    if getattr(layer.mlp.gate, "norm_topk_prob", False):
        npu_weights = npu_weights / npu_weights.sum(dim=-1, keepdim=True)
    mismatch = (hf_indices.to(torch.int32) != npu_indices).float().mean().item()
    errors = tensor_error(hf_weights, npu_weights.to(hf_weights.dtype))
    return {
        "probe": "router_topk",
        "status": "ok",
        "layer": layer_idx,
        "shape_label": shape_label,
        "batch": batch,
        "seq": seq,
        "tokens": batch * seq,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "latency_median_ms": median_ms(times),
        "topk_mismatch_rate": float(mismatch),
        **errors,
    }


def probe_token_permute(
    *,
    torch_npu,
    hidden_2d: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    layer_idx: int,
    shape_label: str,
    batch: int,
    seq: int,
    dtype: torch.dtype,
    device: torch.device,
    repeat: int,
) -> dict[str, Any]:
    times: list[float] = []
    permuted = sorted_indices = restored = None
    for _ in range(repeat):
        (permuted, sorted_indices), permute_s = timed(
            device,
            lambda: torch_npu.npu_moe_token_permute(
                hidden_2d,
                selected_experts.to(torch.int64),
                num_out_tokens=selected_experts.numel(),
            ),
        )
        restored, unpermute_s = timed(
            device,
            lambda: torch_npu.npu_moe_token_unpermute(
                permuted,
                sorted_indices,
                probs=routing_weights.to(dtype),
            ),
        )
        times.append(permute_s + unpermute_s)
    expected = hidden_2d * routing_weights.sum(dim=-1, keepdim=True).to(dtype)
    errors = tensor_error(expected, restored)
    return {
        "probe": "token_permute_roundtrip",
        "status": "ok",
        "layer": layer_idx,
        "shape_label": shape_label,
        "batch": batch,
        "seq": seq,
        "tokens": batch * seq,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "latency_median_ms": median_ms(times),
        "permuted_tokens": int(permuted.shape[0]),
        **errors,
    }


def grouped_experts_forward(
    *,
    torch_npu,
    layer,
    hidden_2d: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    experts = layer.mlp.experts
    active, xs, token_indices, token_weights = build_expert_inputs(
        hidden_2d,
        selected_experts,
        routing_weights,
        experts.num_experts,
    )
    gate_up_weights = [
        experts.gate_up_proj[expert_idx].transpose(0, 1).contiguous()
        for expert_idx in active
    ]
    gate_up_outs = torch_npu.npu_grouped_matmul(xs, gate_up_weights, group_type=-1)
    interms: list[torch.Tensor] = []
    for gate_up in gate_up_outs:
        gate, up = gate_up.chunk(2, dim=-1)
        interms.append((experts.act_fn(gate) * up).contiguous())
    down_weights = [
        experts.down_proj[expert_idx].transpose(0, 1).contiguous()
        for expert_idx in active
    ]
    down_outs = torch_npu.npu_grouped_matmul(interms, down_weights, group_type=-1)
    final = torch.zeros_like(hidden_2d)
    for token_idx, weight, current in zip(token_indices, token_weights, down_outs):
        final.index_add_(0, token_idx, current * weight.to(current.dtype))
    return final


def probe_grouped_experts(
    *,
    torch_npu,
    layer,
    hidden_2d: torch.Tensor,
    hf_output: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    layer_idx: int,
    shape_label: str,
    batch: int,
    seq: int,
    dtype: torch.dtype,
    device: torch.device,
    repeat: int,
) -> dict[str, Any]:
    times: list[float] = []
    grouped_out = None
    for _ in range(repeat):
        grouped_out, elapsed = timed(
            device,
            lambda: grouped_experts_forward(
                torch_npu=torch_npu,
                layer=layer,
                hidden_2d=hidden_2d,
                selected_experts=selected_experts,
                routing_weights=routing_weights,
            ),
        )
        times.append(elapsed)
    errors = tensor_error(hf_output, grouped_out)
    active, _, _, _ = build_expert_inputs(
        hidden_2d,
        selected_experts,
        routing_weights,
        layer.mlp.experts.num_experts,
    )
    return {
        "probe": "grouped_experts",
        "status": "ok",
        "layer": layer_idx,
        "shape_label": shape_label,
        "batch": batch,
        "seq": seq,
        "tokens": batch * seq,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "latency_median_ms": median_ms(times),
        "active_experts": len(active),
        **errors,
    }


def run_shape(
    *,
    torch_npu,
    layer,
    layer_idx: int,
    shape_label: str,
    batch: int,
    seq: int,
    dtype: torch.dtype,
    device: torch.device,
    repeat: int,
) -> list[dict[str, Any]]:
    hidden_size = layer.mlp.gate.hidden_dim
    hidden_states = torch.randn(batch, seq, hidden_size, device=device, dtype=dtype)
    hidden_2d = layer.post_attention_layernorm(hidden_states).reshape(-1, hidden_size).contiguous()
    sync_device(device)
    logits = torch.nn.functional.linear(hidden_2d, layer.mlp.gate.weight)
    router_probs = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1)
    hf_weights, hf_indices = torch.topk(router_probs, layer.mlp.gate.top_k, dim=-1)
    if layer.mlp.gate.norm_topk_prob:
        hf_weights = hf_weights / hf_weights.sum(dim=-1, keepdim=True)
    hf_weights = hf_weights.to(logits.dtype)
    sync_device(device)
    timed(device, lambda: layer.mlp.experts(hidden_2d, hf_indices, hf_weights))
    hf_output = None
    hf_expert_times: list[float] = []
    for _ in range(repeat):
        hf_output, elapsed = timed(
            device,
            lambda: layer.mlp.experts(hidden_2d, hf_indices, hf_weights),
        )
        hf_expert_times.append(elapsed)

    records: list[dict[str, Any]] = [
        {
            "probe": "hf_experts",
            "status": "ok",
            "layer": layer_idx,
            "shape_label": shape_label,
            "batch": batch,
            "seq": seq,
            "tokens": batch * seq,
            "dtype": str(dtype).replace("torch.", ""),
            "device": str(device),
            "latency_median_ms": median_ms(hf_expert_times),
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "max_rel_error": 0.0,
            "mean_rel_error": 0.0,
        }
    ]
    probes = [
        ("router_topk", lambda: probe_router(
            torch_npu=torch_npu,
            layer=layer,
            logits=logits,
            hf_weights=hf_weights,
            hf_indices=hf_indices,
            layer_idx=layer_idx,
            shape_label=shape_label,
            batch=batch,
            seq=seq,
            dtype=dtype,
            device=device,
            repeat=repeat,
        )),
        ("token_permute_roundtrip", lambda: probe_token_permute(
            torch_npu=torch_npu,
            hidden_2d=hidden_2d,
            routing_weights=hf_weights,
            selected_experts=hf_indices,
            layer_idx=layer_idx,
            shape_label=shape_label,
            batch=batch,
            seq=seq,
            dtype=dtype,
            device=device,
            repeat=repeat,
        )),
        ("grouped_experts", lambda: probe_grouped_experts(
            torch_npu=torch_npu,
            layer=layer,
            hidden_2d=hidden_2d,
            hf_output=hf_output,
            routing_weights=hf_weights,
            selected_experts=hf_indices,
            layer_idx=layer_idx,
            shape_label=shape_label,
            batch=batch,
            seq=seq,
            dtype=dtype,
            device=device,
            repeat=repeat,
        )),
    ]
    for probe_name, probe_fn in probes:
        try:
            records.append(probe_fn())
        except Exception as exc:  # probe result must record explicit API failure
            records.append(
                failure_record(
                    probe=probe_name,
                    layer_idx=layer_idx,
                    shape_label=shape_label,
                    batch=batch,
                    seq=seq,
                    dtype=dtype,
                    device=device,
                    exc=exc,
                )
            )
    return records


def write_outputs(records: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "torch_npu_moe_api_probe.json"
    csv_path = out_dir / "correctness_summary.csv"
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    fieldnames = sorted({key for record in records for key in record if key != "traceback"})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: value for key, value in record.items() if key in fieldnames}
            writer.writerow(row)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "/models/Qwen3-30B-A3B"))
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument(
        "--shapes",
        default="decode:32x1,decode:64x1,decode:128x1,prefill:16x512,prefill:32x512",
    )
    parser.add_argument("--device", default=os.environ.get("AFD_BENCH_DEVICE", "npu:0"))
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output-dir", default="results_npu/moe_backend_probe/api_smoke")
    args = parser.parse_args()

    import torch_npu

    device = setup_device(args.device)
    dtype = torch.bfloat16 if args.dtype.lower() in ("bf16", "bfloat16") else dtype_from_name(args.dtype)
    shapes = parse_shapes(args.shapes)

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    layer = model.model.layers[args.layer]
    layer.post_attention_layernorm.to(device=device, dtype=dtype)
    layer.mlp.to(device=device, dtype=dtype)
    layer.eval()

    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for shape_label, batch, seq in shapes:
            print(f"\nshape={shape_label}:{batch}x{seq}")
            shape_records = run_shape(
                torch_npu=torch_npu,
                layer=layer,
                layer_idx=args.layer,
                shape_label=shape_label,
                batch=batch,
                seq=seq,
                dtype=dtype,
                device=device,
                repeat=args.repeat,
            )
            for record in shape_records:
                status = record["status"]
                latency = record.get("latency_median_ms", "-")
                error = record.get("mean_abs_error", record.get("error", ""))
                print(f"  {record['probe']}: {status}, latency={latency}, error={error}")
            records.extend(shape_records)

    write_outputs(records, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
