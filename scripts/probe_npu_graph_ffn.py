#!/usr/bin/env python3
"""Probe NPUGraph replay for fixed-shape Qwen3 FFN/MoE decode workloads."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import time
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

        torch.npu.set_device(device_name)
    return torch.device(device_name)


def sync_device(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, fn: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, float]:
    sync_device(device)
    start = time.perf_counter()
    out = fn()
    sync_device(device)
    return out, time.perf_counter() - start


def median_ms(values_s: list[float]) -> float:
    return statistics.median(values_s) * 1000.0


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-8)
    rel = diff / denom
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "max_rel_error": float(rel.max().item()),
        "mean_rel_error": float(rel.mean().item()),
    }


def graph_capture(
    device: torch.device,
    fn: Callable[[], torch.Tensor],
    capture_error_mode: str,
) -> tuple[Any, torch.Tensor]:
    graph = torch.npu.NPUGraph()
    sync_device(device)
    with torch.npu.graph(graph, capture_error_mode=capture_error_mode):
        static_out = fn()
    sync_device(device)
    return graph, static_out


def run_graph_probe(
    device: torch.device,
    name: str,
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeat: int,
    capture_error_mode: str,
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    eager_ref, _ = timed(device, fn)

    eager_times: list[float] = []
    for _ in range(repeat):
        eager_out, elapsed = timed(device, fn)
        eager_times.append(elapsed)

    record: dict[str, Any] = {
        "probe": name,
        "status": "ok",
        "eager_median_ms": median_ms(eager_times),
    }

    try:
        graph, static_out = graph_capture(device, fn, capture_error_mode)
        graph.replay()
        sync_device(device)
        graph_times: list[float] = []
        for _ in range(repeat):
            sync_device(device)
            start = time.perf_counter()
            graph.replay()
            sync_device(device)
            graph_times.append(time.perf_counter() - start)
        record["graph_median_ms"] = median_ms(graph_times)
        record["speedup"] = record["eager_median_ms"] / record["graph_median_ms"]
        record.update(tensor_error(static_out, eager_ref))
    except Exception as exc:  # noqa: BLE001 - failures are data for this probe.
        record.update(
            {
                "status": "error",
                "graph_median_ms": None,
                "speedup": None,
                "capture_error_mode": capture_error_mode,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return record


def prepare_inputs(layer, batch: int, seq: int, hidden_size: int, device: torch.device, dtype: torch.dtype):
    hidden_states = torch.randn(batch, seq, hidden_size, device=device, dtype=dtype)
    with torch.inference_mode():
        normed = layer.post_attention_layernorm(hidden_states)
        hidden_2d = normed.reshape(-1, hidden_size)
        _router_logits, routing_weights, selected_experts = layer.mlp.gate(hidden_2d)
    return hidden_states, hidden_2d, selected_experts, routing_weights


def probe_shape(
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
    capture_error_mode: str,
) -> list[dict[str, Any]]:
    hidden_size = layer.mlp.gate.hidden_dim
    hidden_states, hidden_2d, selected_experts, routing_weights = prepare_inputs(
        layer, batch, seq, hidden_size, device, dtype
    )

    def experts_only() -> torch.Tensor:
        return layer.mlp.experts(hidden_2d, selected_experts, routing_weights)

    def full_ffn() -> torch.Tensor:
        residual = hidden_states
        normed = layer.post_attention_layernorm(hidden_states)
        flat = normed.reshape(-1, hidden_size)
        _router_logits, weights, experts = layer.mlp.gate(flat)
        experts_out = layer.mlp.experts(flat, experts, weights).reshape(batch, seq, hidden_size)
        return residual + experts_out

    rows: list[dict[str, Any]] = []
    for probe_name, fn in (("experts_only", experts_only), ("full_ffn", full_ffn)):
        with torch.inference_mode():
            record = run_graph_probe(
                device,
                probe_name,
                fn,
                warmup=warmup,
                repeat=repeat,
                capture_error_mode=capture_error_mode,
            )
        record.update(
            {
                "layer": layer_idx,
                "shape_label": shape_label,
                "batch": batch,
                "seq": seq,
                "tokens": batch * seq,
                "dtype": str(dtype).replace("torch.", ""),
                "device": str(device),
            }
        )
        rows.append(record)
    return rows


def write_outputs(records: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "decode_graph_probe.json"
    csv_path = out_dir / "decode_graph_probe.csv"
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
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--shapes", default="decode:32x1,decode:64x1,decode:128x1")
    parser.add_argument("--device", default=os.environ.get("AFD_BENCH_DEVICE", "npu:0"))
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--capture-error-mode", default="relaxed", choices=["global", "thread_local", "relaxed"])
    parser.add_argument("--output-dir", default="results_npu/moe_backend_probe/graph_probe")
    args = parser.parse_args()

    device = setup_device(args.device)
    dtype = dtype_from_name(args.dtype)
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
    for shape_label, batch, seq in shapes:
        print(f"\nshape={shape_label}:{batch}x{seq}")
        shape_records = probe_shape(
            layer,
            layer_idx=args.layer,
            shape_label=shape_label,
            batch=batch,
            seq=seq,
            device=device,
            dtype=dtype,
            warmup=args.warmup,
            repeat=args.repeat,
            capture_error_mode=args.capture_error_mode,
        )
        for record in shape_records:
            print(
                "  {probe}: {status} eager={eager:.3f}ms graph={graph} speedup={speedup}".format(
                    probe=record["probe"],
                    status=record["status"],
                    eager=record["eager_median_ms"],
                    graph=(
                        f"{record['graph_median_ms']:.3f}ms"
                        if isinstance(record.get("graph_median_ms"), float)
                        else record.get("graph_median_ms")
                    ),
                    speedup=(
                        f"{record['speedup']:.3f}x"
                        if isinstance(record.get("speedup"), float)
                        else record.get("speedup")
                    ),
                )
            )
        records.extend(shape_records)

    write_outputs(records, Path(args.output_dir))
    layer.post_attention_layernorm.to(device="cpu")
    layer.mlp.to(device="cpu")
    gc.collect()
    sync_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
