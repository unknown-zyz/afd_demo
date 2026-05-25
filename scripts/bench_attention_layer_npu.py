#!/usr/bin/env python3
"""Single-layer Attention benchmark for Ascend NPU optimization gates."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from src.model.attention_worker import AttentionLayer, AttentionOptimizationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one Qwen attention layer on NPU.")
    parser.add_argument("--model-name", default="/models/Qwen3-30B-A3B")
    parser.add_argument("--output", default="results_npu/attention_layer_bench/latest.json")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--decode-cache-lens", default="32,128,512")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--include-community",
        action="store_true",
        help="Benchmark flash-attention-npu if already built/importable.",
    )
    parser.add_argument(
        "--community-root",
        default="/tmp/flash-attention-npu",
        help="Directory containing in-place flash-attention-npu build artifacts.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def memory_stats(device: torch.device) -> dict[str, Optional[float]]:
    module = getattr(torch, device.type, None)
    if module is None or not hasattr(module, "memory_allocated"):
        return {"allocated_mib": None, "reserved_mib": None, "peak_allocated_mib": None}
    return {
        "allocated_mib": float(module.memory_allocated(device) / 1024**2),
        "reserved_mib": float(module.memory_reserved(device) / 1024**2)
        if hasattr(module, "memory_reserved")
        else None,
        "peak_allocated_mib": float(module.max_memory_allocated(device) / 1024**2)
        if hasattr(module, "max_memory_allocated")
        else None,
    }


def reset_peak_memory(device: torch.device) -> None:
    module = getattr(torch, device.type, None)
    if module is not None and hasattr(module, "reset_peak_memory_stats"):
        module.reset_peak_memory_stats(device)


def bench(
    fn: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
    repeats: int,
    device: torch.device,
) -> tuple[float, list[float], torch.Tensor]:
    with torch.no_grad():
        for _ in range(warmup):
            out = fn()
        sync(device)
        reset_peak_memory(device)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(iters):
                out = fn()
            sync(device)
            samples.append((time.perf_counter() - start) * 1000.0 / iters)
    return float(statistics.median(samples)), samples, out


def tensor_diff(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    diff = (candidate.float() - reference.float()).abs()
    ref = reference.float().abs().clamp_min(1e-6)
    return {
        "max_abs_diff": float(diff.max().cpu()),
        "mean_abs_diff": float(diff.mean().cpu()),
        "max_rel_diff": float((diff / ref).max().cpu()),
        "mean_rel_diff": float((diff / ref).mean().cpu()),
    }


def load_apply_rotary_pos_emb() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb
    except ImportError:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    return apply_rotary_pos_emb


def make_attention_layer(
    model: Any,
    layer_idx: int,
    device: torch.device,
    dtype: torch.dtype,
    config: AttentionOptimizationConfig,
) -> AttentionLayer:
    layer = model.model.layers[layer_idx]
    return AttentionLayer(
        input_layernorm=copy.deepcopy(layer.input_layernorm).to(device=device, dtype=dtype),
        self_attn=copy.deepcopy(layer.self_attn).to(device=device, dtype=dtype),
        hidden_size=model.config.hidden_size,
        layer_idx=layer_idx,
        layer_device=device,
        output_device=device,
        optimization_config=config,
    ).eval()


def position_embeddings(rotary_emb: Any, hidden: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return rotary_emb(hidden, position_ids)
    except TypeError:
        return rotary_emb(hidden, seq_len=hidden.shape[1])


def full_layer_prefill_bench(
    model: Any,
    rotary_emb: Any,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    hidden = torch.randn(args.batch, args.seq, model.config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(args.seq, device=device).unsqueeze(0).expand(args.batch, -1)
    pos = position_embeddings(rotary_emb, hidden, position_ids)

    configs = [
        ("hf", AttentionOptimizationConfig(attn_kernel="hf")),
        ("official", AttentionOptimizationConfig(attn_kernel="npu-official")),
        (
            "official_fused_rmsnorm",
            AttentionOptimizationConfig(attn_kernel="npu-official", fused_rmsnorm=True),
        ),
        (
            "official_fused_rope",
            AttentionOptimizationConfig(attn_kernel="npu-official", fused_rope=True),
        ),
        (
            "official_fused_both",
            AttentionOptimizationConfig(
                attn_kernel="npu-official",
                fused_rmsnorm=True,
                fused_rope=True,
            ),
        ),
    ]
    results: list[dict[str, Any]] = []
    reference: Optional[torch.Tensor] = None
    for name, config in configs:
        layer = make_attention_layer(model, args.layer_idx, device, dtype, config)

        def run() -> torch.Tensor:
            return layer(
                hidden_states=hidden,
                position_ids=position_ids,
                position_embeddings=pos,
            )[0]

        try:
            ms, samples, out = bench(run, args.warmup, args.iters, args.repeats, device)
            if reference is None:
                reference = out.detach()
            row: dict[str, Any] = {
                "case": name,
                "phase": "prefill_full_layer",
                "latency_ms": ms,
                "latency_samples_ms": samples,
                "ok": True,
                "memory": memory_stats(device),
                "optimization": config.to_dict(),
            }
            if reference is not None and name != "hf":
                row.update(tensor_diff(out, reference))
        except Exception as exc:
            row = {
                "case": name,
                "phase": "prefill_full_layer",
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "optimization": config.to_dict(),
            }
        results.append(row)
        del layer
        sync(device)
    return results


def project_decode_states(
    layer: AttentionLayer,
    hidden: torch.Tensor,
    rotary_emb: Any,
    use_fused_rope: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq = hidden.shape[1]
    position_ids = torch.arange(seq, device=hidden.device).unsqueeze(0).expand(hidden.shape[0], -1)
    pos = position_embeddings(rotary_emb, hidden, position_ids)
    normed = layer._apply_input_layernorm(hidden)
    head_dim = int(getattr(layer.self_attn, "head_dim"))
    hidden_shape = (hidden.shape[0], hidden.shape[1], -1, head_dim)
    q = layer.self_attn.q_proj(normed).view(hidden_shape)
    k = layer.self_attn.k_proj(normed).view(hidden_shape)
    v = layer.self_attn.v_proj(normed).view(hidden_shape)
    if hasattr(layer.self_attn, "q_norm"):
        q = layer.self_attn.q_norm(q)
    if hasattr(layer.self_attn, "k_norm"):
        k = layer.self_attn.k_norm(k)
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    cos, sin = pos
    if use_fused_rope:
        q = layer._apply_npu_rotary_mul(q, cos, sin)
        k = layer._apply_npu_rotary_mul(k, cos, sin)
    else:
        q, k = load_apply_rotary_pos_emb()(q, k, cos, sin)
    return q.contiguous(), k.contiguous(), v.contiguous()


def import_community(root: str, version: str) -> Optional[Callable[..., torch.Tensor]]:
    if not Path(root).exists():
        return None
    import importlib
    import sys

    if root not in sys.path:
        sys.path.insert(0, root)
    module_name = "flash_attn_npu_v3" if version == "v3" else "flash_attn_npu"
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, "flash_attn_with_kvcache", None)


def decode_core_bench(
    model: Any,
    rotary_emb: Any,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    official_layer = make_attention_layer(
        model,
        args.layer_idx,
        device,
        dtype,
        AttentionOptimizationConfig(attn_kernel="npu-official", fused_rmsnorm=True, fused_rope=True),
    )
    flash_prefill, flash_decode = official_layer._load_npu_attention_ops()
    del flash_prefill
    head_dim = int(getattr(official_layer.self_attn, "head_dim"))
    num_heads = official_layer._num_attention_heads(head_dim)
    num_kv_heads = official_layer._num_key_value_heads(head_dim)
    scale = float(getattr(official_layer.self_attn, "scaling", 1.0 / math.sqrt(head_dim)))
    community_v2 = import_community(args.community_root, "v2") if args.include_community else None
    community_v3 = import_community(args.community_root, "v3") if args.include_community else None
    results: list[dict[str, Any]] = []

    for cache_len in [int(part) for part in args.decode_cache_lens.split(",") if part.strip()]:
        cache_hidden = torch.randn(args.batch, cache_len, model.config.hidden_size, device=device, dtype=dtype)
        token_hidden = torch.randn(args.batch, 1, model.config.hidden_size, device=device, dtype=dtype)
        q, _, _ = project_decode_states(official_layer, token_hidden, rotary_emb, use_fused_rope=True)
        _, k_cache, v_cache = project_decode_states(official_layer, cache_hidden, rotary_emb, use_fused_rope=True)

        repeat_factor = num_heads // num_kv_heads
        k_repeated = k_cache.repeat_interleave(repeat_factor, dim=1)
        v_repeated = v_cache.repeat_interleave(repeat_factor, dim=1)

        def hf_sdpa_run() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                q,
                k_repeated,
                v_repeated,
                dropout_p=0.0,
                scale=scale,
            )

        hf_ms, hf_samples, hf_out = bench(hf_sdpa_run, args.warmup, args.iters, args.repeats, device)
        results.append(
            {
                "case": "hf_sdpa",
                "phase": "decode_core",
                "cache_len": cache_len,
                "latency_ms": hf_ms,
                "latency_samples_ms": hf_samples,
                "ok": True,
                "memory": memory_stats(device),
            }
        )

        def official_run() -> torch.Tensor:
            return flash_decode(
                q,
                k_cache,
                v_cache,
                num_heads=num_heads,
                input_layout="BNSD",
                scale_value=scale,
                num_key_value_heads=num_kv_heads,
            )

        official_ms, official_samples, official_out = bench(
            official_run, args.warmup, args.iters, args.repeats, device
        )
        official_row: dict[str, Any] = {
            "case": "official_ifa",
            "phase": "decode_core",
            "cache_len": cache_len,
            "latency_ms": official_ms,
            "latency_samples_ms": official_samples,
            "ok": True,
            "memory": memory_stats(device),
        }
        official_row.update(tensor_diff(official_out, hf_out))
        results.append(official_row)

        q_bsnd = q.transpose(1, 2).contiguous()
        k_bsnd = k_cache.transpose(1, 2).contiguous()
        v_bsnd = v_cache.transpose(1, 2).contiguous()
        hf_bsnd = hf_out.transpose(1, 2).contiguous()
        for name, fn in [("community_v2", community_v2), ("community_v3", community_v3)]:
            if fn is None:
                results.append(
                    {
                        "case": name,
                        "phase": "decode_core",
                        "cache_len": cache_len,
                        "ok": False,
                        "error": "flash-attention-npu is not importable",
                    }
                )
                continue

            def community_run(fn: Callable[..., torch.Tensor] = fn) -> torch.Tensor:
                return fn(
                    q_bsnd,
                    k_bsnd,
                    v_bsnd,
                    cache_seqlens=cache_len,
                    softmax_scale=scale,
                    causal=True,
                    rotary_interleaved=False,
                )

            try:
                ms, samples, out = bench(community_run, args.warmup, args.iters, args.repeats, device)
                row: dict[str, Any] = {
                    "case": name,
                    "phase": "decode_core",
                    "cache_len": cache_len,
                    "latency_ms": ms,
                    "latency_samples_ms": samples,
                    "ok": True,
                    "memory": memory_stats(device),
                }
                row.update(tensor_diff(out, hf_bsnd))
            except Exception as exc:
                row = {
                    "case": name,
                    "phase": "decode_core",
                    "cache_len": cache_len,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            results.append(row)
    del official_layer
    sync(device)
    return results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "npu":
        import torch_npu  # type: ignore[import-not-found]  # noqa: F401
    dtype = dtype_from_name(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()
    rotary_emb = model.model.rotary_emb.to(device=device)
    results = {
        "model_name": args.model_name,
        "device": str(device),
        "dtype": str(dtype),
        "layer_idx": args.layer_idx,
        "batch": args.batch,
        "seq": args.seq,
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "results": [],
    }
    results["results"].extend(full_layer_prefill_bench(model, rotary_emb, args, device, dtype))
    results["results"].extend(decode_core_bench(model, rotary_emb, args, device, dtype))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
