#!/usr/bin/env python3
"""Estimate decode-side MFU from timing JSONs and model config.

This is an inference-oriented, model-config-derived estimate rather than a
hardware-counter MFU. It is useful for comparing cross-host 1A7F runs so they
do not only report TPOT / throughput / EP bandwidth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
CONFIG_RE = re.compile(r"_b(?P<b>\d+)_s(?P<s>\d+)_t(?P<t>\d+)")
EP_RE = re.compile(r"_ep(?P<ep>\d+)(?:_|$)")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def infer_config_from_name(path: Path) -> tuple[int | None, int | None, int | None]:
    match = CONFIG_RE.search(path.name)
    if match is None:
        return None, None, None
    return int(match["b"]), int(match["s"]), int(match["t"])


def infer_ep_size(path: Path) -> int | None:
    match = EP_RE.search(path.name)
    if match is not None:
        return int(match["ep"])
    match = EP_RE.search(str(path.parent))
    if match is not None:
        return int(match["ep"])
    return None


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


@dataclass(frozen=True)
class ModelShape:
    model_type: str
    num_layers: int
    hidden_size: int
    kv_hidden_size: int
    vocab_size: int
    num_experts: int
    num_experts_per_tok: int
    intermediate_size: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    num_moe_layers: int
    num_dense_layers: int


def _to_int(obj: dict[str, Any], key: str, default: int = 0) -> int:
    value = obj.get(key, default)
    if value in (None, ""):
        return default
    return int(value)


def load_model_shape(model_config: Path) -> ModelShape:
    data = load_json(model_config)
    hidden = _to_int(data, "hidden_size")
    num_layers = _to_int(data, "num_hidden_layers")
    num_heads = _to_int(data, "num_attention_heads")
    num_kv_heads = _to_int(data, "num_key_value_heads", num_heads if num_heads else 0)
    kv_hidden = hidden if not num_heads else hidden * num_kv_heads // num_heads
    num_experts = _to_int(data, "num_experts")
    top_k = _to_int(data, "num_experts_per_tok")
    sparse_step = _to_int(data, "decoder_sparse_step", 0)
    moe_intermediate = _to_int(data, "moe_intermediate_size")
    dense_intermediate = _to_int(data, "intermediate_size")
    shared_intermediate = _to_int(data, "shared_expert_intermediate_size")
    if num_experts > 0 and top_k > 0 and moe_intermediate > 0:
        step = sparse_step if sparse_step > 0 else 1
        num_moe_layers = math.ceil(num_layers / step)
    else:
        num_moe_layers = 0
    num_dense_layers = max(0, num_layers - num_moe_layers)
    return ModelShape(
        model_type=str(data.get("model_type", "")),
        num_layers=num_layers,
        hidden_size=hidden,
        kv_hidden_size=kv_hidden,
        vocab_size=_to_int(data, "vocab_size"),
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        intermediate_size=dense_intermediate,
        moe_intermediate_size=moe_intermediate,
        shared_expert_intermediate_size=shared_intermediate,
        num_moe_layers=num_moe_layers,
        num_dense_layers=num_dense_layers,
    )


def resolve_model_config(args: argparse.Namespace) -> Path:
    if args.model_config:
        path = Path(args.model_config)
    elif args.model_name:
        model_name = Path(args.model_name)
        path = model_name / "config.json" if model_name.is_dir() or not model_name.suffix else model_name
    else:
        raise SystemExit("Either --model-config or --model-name is required")
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.is_file():
        raise SystemExit(f"model config not found: {path}")
    return path


def decode_tpot_ms(data: dict[str, Any]) -> float | None:
    value = data.get("decode_tpot_ms")
    if value not in (None, ""):
        return float(value)
    loop_ms = data.get("decode_loop_ms")
    steps = data.get("decode_steps")
    if loop_ms in (None, "") or steps in (None, "", 0):
        return None
    return float(loop_ms) / float(steps)


def average_decode_kv_len(data: dict[str, Any], seq_from_name: int | None, tokens_from_name: int | None) -> float:
    prompt_len = data.get("actual_prompt_len")
    if prompt_len in (None, ""):
        prompt_len = data.get("prefill_seq_len")
    if prompt_len in (None, ""):
        prompt_len = seq_from_name
    if prompt_len in (None, ""):
        raise ValueError("Cannot infer prompt length from timing JSON or file name")
    decode_steps = data.get("decode_steps")
    if decode_steps in (None, ""):
        decode_steps = max(1, (tokens_from_name or 0) - 1)
    decode_steps = max(1, int(decode_steps))
    return float(prompt_len) + (decode_steps - 1) / 2.0


def estimate_flops_per_token(model: ModelShape, avg_kv_len: float) -> dict[str, float]:
    hidden = float(model.hidden_size)
    kv_hidden = float(model.kv_hidden_size)
    attention_proj = 4.0 * hidden * hidden + 4.0 * hidden * kv_hidden
    attention_kv = 4.0 * hidden * avg_kv_len
    attention_per_layer = attention_proj + attention_kv
    router = 2.0 * hidden * float(model.num_experts) if model.num_experts > 0 else 0.0
    moe_ffn = 6.0 * hidden * float(model.moe_intermediate_size) * float(model.num_experts_per_tok)
    shared_ffn = 6.0 * hidden * float(model.shared_expert_intermediate_size)
    dense_ffn = 6.0 * hidden * float(model.intermediate_size)
    attn_total = attention_per_layer * float(model.num_layers) + 2.0 * hidden * float(model.vocab_size)
    ffn_total = (
        float(model.num_moe_layers) * (router + moe_ffn + shared_ffn)
        + float(model.num_dense_layers) * dense_ffn
    )
    return {
        "attn_flops_per_token": attn_total,
        "ffn_flops_per_token": ffn_total,
        "total_flops_per_token": attn_total + ffn_total,
    }


def tflops(flops_per_token: float, throughput_tok_s: float) -> float:
    return flops_per_token * throughput_tok_s / 1e12


def build_row(
    path: Path,
    model: ModelShape,
    default_attn_devices: int,
    default_ffn_devices: int,
    peak_tflops_per_device: float | None,
) -> dict[str, Any]:
    data = load_json(path)
    batch, seq, tokens = infer_config_from_name(path)
    tpot_ms = decode_tpot_ms(data)
    if tpot_ms is None or tpot_ms <= 0:
        raise ValueError(f"decode_tpot_ms missing in {path}")
    ep_size = infer_ep_size(path)
    attn_devices = default_attn_devices
    ffn_devices = ep_size if ep_size is not None and default_ffn_devices == 1 else default_ffn_devices
    total_devices = attn_devices + ffn_devices
    avg_kv_len = average_decode_kv_len(data, seq, tokens)
    flops = estimate_flops_per_token(model, avg_kv_len)
    throughput_tok_s = float(batch) * 1000.0 / tpot_ms if batch else None
    decode_steps = data.get("decode_steps")
    if decode_steps in (None, ""):
        decode_steps = max(1, (tokens or 0) - 1)
    decode_steps = int(decode_steps)
    attn_tflops = tflops(flops["attn_flops_per_token"], throughput_tok_s) if throughput_tok_s else None
    ffn_tflops = tflops(flops["ffn_flops_per_token"], throughput_tok_s) if throughput_tok_s else None
    system_tflops = tflops(flops["total_flops_per_token"], throughput_tok_s) if throughput_tok_s else None
    attn_peak = peak_tflops_per_device * attn_devices if peak_tflops_per_device is not None else None
    ffn_peak = peak_tflops_per_device * ffn_devices if peak_tflops_per_device is not None else None
    system_peak = peak_tflops_per_device * total_devices if peak_tflops_per_device is not None else None
    return {
        "file": relpath(path),
        "batch": batch,
        "seq": seq,
        "tokens": tokens,
        "model_type": model.model_type,
        "num_layers": model.num_layers,
        "hidden_size": model.hidden_size,
        "kv_hidden_size": model.kv_hidden_size,
        "num_experts": model.num_experts,
        "num_experts_per_tok": model.num_experts_per_tok,
        "num_moe_layers": model.num_moe_layers,
        "num_dense_layers": model.num_dense_layers,
        "decode_steps": decode_steps,
        "avg_decode_kv_len": avg_kv_len,
        "attn_devices": attn_devices,
        "ffn_devices": ffn_devices,
        "total_devices": total_devices,
        "routing_backend": data.get("routing_backend"),
        "routing_update_mode": data.get("routing_update_mode"),
        "routing_table_version": data.get("routing_table_version"),
        "routing_poll_count": data.get("routing_poll_count"),
        "routing_poll_ms": data.get("routing_poll_ms"),
        "decode_tpot_ms": tpot_ms,
        "throughput_tok_s": throughput_tok_s,
        "attn_flops_per_token": flops["attn_flops_per_token"],
        "ffn_flops_per_token": flops["ffn_flops_per_token"],
        "total_flops_per_token": flops["total_flops_per_token"],
        "attn_achieved_tflops": attn_tflops,
        "ffn_achieved_tflops": ffn_tflops,
        "system_achieved_tflops": system_tflops,
        "peak_tflops_per_device": peak_tflops_per_device,
        "attn_peak_tflops": attn_peak,
        "ffn_peak_tflops": ffn_peak,
        "system_peak_tflops": system_peak,
        "attn_mfu": (attn_tflops / attn_peak) if attn_tflops is not None and attn_peak else None,
        "ffn_mfu": (ffn_tflops / ffn_peak) if ffn_tflops is not None and ffn_peak else None,
        "system_mfu": (system_tflops / system_peak) if system_tflops is not None and system_peak else None,
    }


def write_csv(rows: list[dict[str, Any]], out: Path) -> None:
    fields = [
        "file",
        "batch",
        "seq",
        "tokens",
        "model_type",
        "num_layers",
        "hidden_size",
        "kv_hidden_size",
        "num_experts",
        "num_experts_per_tok",
        "num_moe_layers",
        "num_dense_layers",
        "decode_steps",
        "avg_decode_kv_len",
        "attn_devices",
        "ffn_devices",
        "total_devices",
        "routing_backend",
        "routing_update_mode",
        "routing_table_version",
        "routing_poll_count",
        "routing_poll_ms",
        "decode_tpot_ms",
        "throughput_tok_s",
        "attn_flops_per_token",
        "ffn_flops_per_token",
        "total_flops_per_token",
        "attn_achieved_tflops",
        "ffn_achieved_tflops",
        "system_achieved_tflops",
        "peak_tflops_per_device",
        "attn_peak_tflops",
        "ffn_peak_tflops",
        "system_peak_tflops",
        "attn_mfu",
        "ffn_mfu",
        "system_mfu",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k)) for k in fields})
    print(f"Wrote {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="Result root containing decode-dbo/ and timing_attention*.json, or a decode-dbo directory itself.",
    )
    parser.add_argument(
        "--glob",
        default="timing_attention*.json",
        help="Glob under decode-dbo/ (or --root if it already points there). Default: timing_attention*.json",
    )
    parser.add_argument("--model-name", default="", help="Model directory or config path (for example /models/Qwen3-30B-A3B).")
    parser.add_argument("--model-config", default="", help="Explicit config.json path. Overrides --model-name.")
    parser.add_argument("--attn-devices", type=int, default=1, help="Attention-role device count. Default: 1.")
    parser.add_argument("--ffn-devices", type=int, default=1, help="FFN-role device count when topology cannot be inferred from file names. Default: 1.")
    parser.add_argument(
        "--peak-tflops-per-device",
        type=float,
        default=None,
        help="Optional BF16/FP16 peak TFLOPS per device. When omitted, MFU columns stay empty and only achieved TFLOPS are reported.",
    )
    parser.add_argument("--out", default="", help="Optional CSV path. Defaults to <root>/decode_mfu_summary.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = (ROOT / root).resolve()
    decode_dir = root if root.name == "decode-dbo" else root / "decode-dbo"
    if not decode_dir.is_dir():
        raise SystemExit(f"decode-dbo directory not found: {decode_dir}")
    model_shape = load_model_shape(resolve_model_config(args))
    rows = [
        build_row(
            path,
            model_shape,
            default_attn_devices=args.attn_devices,
            default_ffn_devices=args.ffn_devices,
            peak_tflops_per_device=args.peak_tflops_per_device,
        )
        for path in sorted(decode_dir.glob(args.glob))
    ]
    if not rows:
        raise SystemExit(f"No timing files matched {args.glob} under {decode_dir}")
    out = Path(args.out) if args.out else decode_dir.parent / "decode_mfu_summary.csv"
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    write_csv(rows, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
