#!/usr/bin/env python3
"""Benchmark one or more Qwen3 MoE FFN layers on NPU.

The script intentionally benchmarks a single FFN/MoE layer outside the full
AFD pipeline so we can identify whether router, expert loop, projections, or
combine dominates FFN latency.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM


def parse_layers(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_shapes(value: str) -> list[tuple[str, int, int]]:
    shapes: list[tuple[str, int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            label, dims = item.split(":", 1)
        else:
            label, dims = "custom", item
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


def timed_call(device: torch.device, fn) -> tuple[Any, float]:
    sync_device(device)
    start = time.perf_counter()
    out = fn()
    sync_device(device)
    return out, time.perf_counter() - start


def summarize(values: list[float]) -> dict[str, float]:
    values_ms = [value * 1000.0 for value in values]
    return {
        "median_ms": statistics.median(values_ms),
        "mean_ms": statistics.mean(values_ms),
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
    }


def benchmark_layer_shape(
    layer,
    *,
    layer_idx: int,
    shape_label: str,
    batch: int,
    seq: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    hidden_size = layer.mlp.gate.hidden_dim if hasattr(layer.mlp, "gate") else layer.self_attn.hidden_size
    hidden_states = torch.randn(batch, seq, hidden_size, device=device, dtype=dtype)

    def run_once() -> tuple[dict[str, float], int, int]:
        stage: dict[str, float] = {}
        residual = hidden_states
        normed, stage["norm_s"] = timed_call(device, lambda: layer.post_attention_layernorm(hidden_states))
        hidden_2d = normed.reshape(-1, hidden_size)
        gate_out, stage["router_s"] = timed_call(device, lambda: layer.mlp.gate(hidden_2d))
        _router_logits, routing_weights, selected_experts = gate_out
        sync_device(device)
        selected_unique = int(torch.unique(selected_experts).numel())
        top_k = int(selected_experts.shape[-1])
        experts_out, stage["experts_s"] = timed_call(
            device,
            lambda: layer.mlp.experts(hidden_2d, selected_experts, routing_weights),
        )
        experts_out = experts_out.reshape(batch, seq, hidden_size)
        combined, stage["residual_add_s"] = timed_call(device, lambda: residual + experts_out)
        stage["total_s"] = sum(stage.values())
        if combined.numel() == 0:
            raise RuntimeError("unexpected empty output")
        return stage, selected_unique, top_k

    for _ in range(warmup):
        run_once()

    samples: dict[str, list[float]] = {
        "norm_s": [],
        "router_s": [],
        "experts_s": [],
        "residual_add_s": [],
        "total_s": [],
    }
    selected_unique = 0
    top_k = 0
    for _ in range(repeat):
        stage, selected_unique, top_k = run_once()
        for key, value in stage.items():
            samples[key].append(value)

    record: dict[str, Any] = {
        "layer": layer_idx,
        "shape_label": shape_label,
        "batch": batch,
        "seq": seq,
        "tokens": batch * seq,
        "hidden_size": hidden_size,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "warmup": warmup,
        "repeat": repeat,
        "selected_experts": selected_unique,
        "top_k": top_k,
    }
    for stage_name, values in samples.items():
        stats = summarize(values)
        prefix = stage_name.replace("_s", "")
        for stat_name, stat_value in stats.items():
            record[f"{prefix}_{stat_name}"] = stat_value
    return record


def write_outputs(records: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "moe_layer_baseline.json"
    csv_path = out_dir / "moe_layer_baseline.csv"

    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    fieldnames = sorted({key for record in records for key in record})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "/models/Qwen3-30B-A3B"))
    parser.add_argument("--layers", default="24")
    parser.add_argument(
        "--shapes",
        default="decode:32x1,decode:64x1,decode:128x1,prefill:16x512,prefill:32x512",
        help="Comma list of label:batchxseq entries.",
    )
    parser.add_argument("--device", default=os.environ.get("AFD_BENCH_DEVICE", "npu:0"))
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output-dir", default="results_npu/moe_backend_probe/microbench")
    args = parser.parse_args()

    device = setup_device(args.device)
    dtype = dtype_from_name(args.dtype)
    layers = parse_layers(args.layers)
    shapes = parse_shapes(args.shapes)

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    records: list[dict[str, Any]] = []
    for layer_idx in layers:
        print(f"\nLayer {layer_idx}: moving FFN to {device}")
        layer = model.model.layers[layer_idx]
        layer.post_attention_layernorm.to(device=device, dtype=dtype)
        layer.mlp.to(device=device, dtype=dtype)
        layer.eval()
        for shape_label, batch, seq in shapes:
            print(f"  shape={shape_label}:{batch}x{seq}")
            with torch.inference_mode():
                record = benchmark_layer_shape(
                    layer,
                    layer_idx=layer_idx,
                    shape_label=shape_label,
                    batch=batch,
                    seq=seq,
                    device=device,
                    dtype=dtype,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
            print(
                "    total={total:.3f}ms router={router:.3f}ms experts={experts:.3f}ms".format(
                    total=record["total_median_ms"],
                    router=record["router_median_ms"],
                    experts=record["experts_median_ms"],
                )
            )
            records.append(record)
        layer.post_attention_layernorm.to(device="cpu")
        layer.mlp.to(device="cpu")
        gc.collect()
        sync_device(device)

    write_outputs(records, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
