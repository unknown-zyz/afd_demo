#!/usr/bin/env python3
"""Single-layer Attention benchmark for Ascend NPU optimization gates."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from src.model.attention_worker import AttentionLayer, AttentionOptimizationConfig


REFERENCE_CASES = {
    "prefill_full_layer": "hf",
    "prefill_core": "hf_sdpa",
    "decode_full_layer": "hf",
    "decode_core": "hf_sdpa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen attention layers on NPU.")
    parser.add_argument("--model-name", default="/models/Qwen3-30B-A3B")
    parser.add_argument("--output", default="results_npu/attention_layer_bench/latest.json")
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--summary-md", default=None)
    parser.add_argument("--device", default="npu:0", help="Device that owns the benchmarked layer.")
    parser.add_argument(
        "--input-device",
        default=None,
        help="Device for layer-invariant inputs before optional precopy. Defaults to --device.",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--layer-idx", type=int, default=0, help="Backward-compatible single layer index.")
    parser.add_argument(
        "--layer-idxs",
        default=None,
        help="Layer indices to benchmark: e.g. 0, 0,23,47, or all. Defaults to --layer-idx.",
    )
    parser.add_argument("--batch", type=int, default=1, help="Backward-compatible single batch size.")
    parser.add_argument(
        "--batches",
        default=None,
        help="Comma-separated batch sizes. Example: 1,2,4,8,16,32,64,128.",
    )
    parser.add_argument("--seq", type=int, default=128, help="Backward-compatible single prefill seq length.")
    parser.add_argument(
        "--seqs",
        default=None,
        help="Comma-separated prefill seq lengths. Example: 1,16,32,64,128,256,512,1024.",
    )
    parser.add_argument("--decode-cache-lens", default="32,128,512")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Write partial JSON/summary after this many benchmark groups. 0 disables checkpoints.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress logs on stderr.")
    parser.add_argument("--no-print-json", action="store_true", help="Do not print final JSON to stdout.")
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument(
        "--include-prefill-core",
        action="store_true",
        help="Also benchmark prefill attention kernel core: SDPA vs NPU prompt flash attention.",
    )
    parser.add_argument(
        "--include-decode-full-layer",
        action="store_true",
        help="Also benchmark full decode AttentionLayer latency with fusion ablations.",
    )
    parser.add_argument(
        "--include-precopy",
        action="store_true",
        help="Add precopy_on cases for HF and official+fusion prefill full-layer benchmark.",
    )
    parser.add_argument(
        "--include-community",
        action="store_true",
        help="Benchmark flash-attention-npu if already built/importable.",
    )
    parser.add_argument(
        "--community-layer-idxs",
        default="0",
        help="Layer indices for community benchmark: e.g. 0, 0,23,47, or all.",
    )
    parser.add_argument(
        "--community-root",
        default="/tmp/flash-attention-npu",
        help="Directory containing in-place flash-attention-npu build artifacts.",
    )
    return parser.parse_args()


def parse_int_list(value: Optional[str], fallback: Iterable[int], name: str) -> list[int]:
    if value is None or value == "":
        return list(fallback)
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            item = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid integer in {name}: {part!r}") from exc
        if item < 1:
            raise ValueError(f"{name} values must be positive, got {item}")
        result.append(item)
    if not result:
        raise ValueError(f"{name} did not contain any values")
    return result


def parse_layer_indices(spec: Optional[str], fallback: int, num_layers: int, name: str) -> list[int]:
    if spec is None or spec == "":
        values = [fallback]
    elif spec.strip().lower() == "all":
        values = list(range(num_layers))
    else:
        values = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                values.append(int(part))
            except ValueError as exc:
                raise ValueError(f"Invalid layer index in {name}: {part!r}") from exc
    if not values:
        raise ValueError(f"{name} did not contain any layers")
    deduped = list(dict.fromkeys(values))
    for layer_idx in deduped:
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"{name} layer {layer_idx} is outside [0, {num_layers})")
    return deduped


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


def clear_memory(device: torch.device) -> None:
    gc.collect()
    module = getattr(torch, device.type, None)
    if module is not None and hasattr(module, "empty_cache"):
        module.empty_cache()
    sync(device)


def bench(
    fn: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
    repeats: int,
    device: torch.device,
) -> tuple[float, list[float], torch.Tensor]:
    with torch.no_grad():
        out = fn()
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


def position_embeddings(
    rotary_emb: Any,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return rotary_emb(hidden, position_ids)
    except TypeError:
        return rotary_emb(hidden, seq_len=hidden.shape[1])


def prefill_configs(include_precopy: bool) -> list[tuple[str, AttentionOptimizationConfig]]:
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
    if include_precopy:
        configs.extend(
            [
                ("hf_precopy", AttentionOptimizationConfig(attn_kernel="hf", precopy_layer_inputs=True)),
                (
                    "official_fused_both_precopy",
                    AttentionOptimizationConfig(
                        attn_kernel="npu-official",
                        precopy_layer_inputs=True,
                        fused_rmsnorm=True,
                        fused_rope=True,
                    ),
                ),
            ]
        )
    return configs


def full_layer_prefill_bench(
    model: Any,
    rotary_emb: Any,
    layer_idx: int,
    batch: int,
    seq: int,
    args: argparse.Namespace,
    layer_device: torch.device,
    input_device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    hidden = torch.randn(batch, seq, model.config.hidden_size, device=input_device, dtype=dtype)
    position_ids = torch.arange(seq, device=input_device).unsqueeze(0).expand(batch, -1)
    pos = position_embeddings(rotary_emb, hidden, position_ids)
    cached_position_ids = position_ids.to(layer_device, non_blocking=True)
    cached_pos = (
        pos[0].to(layer_device, non_blocking=True),
        pos[1].to(layer_device, non_blocking=True),
    )
    results: list[dict[str, Any]] = []
    reference: Optional[torch.Tensor] = None
    for name, config in prefill_configs(args.include_precopy):
        layer = make_attention_layer(model, layer_idx, layer_device, dtype, config)
        inputs_prepositioned = config.precopy_layer_inputs

        def run() -> torch.Tensor:
            return layer(
                hidden_states=hidden,
                position_ids=cached_position_ids if inputs_prepositioned else position_ids,
                position_embeddings=cached_pos if inputs_prepositioned else pos,
                inputs_prepositioned=inputs_prepositioned,
            )[0]

        row: dict[str, Any] = {
            "case": name,
            "phase": "prefill_full_layer",
            "layer_idx": layer_idx,
            "batch": batch,
            "seq": seq,
            "precopy_inputs": inputs_prepositioned,
            "optimization": config.to_dict(),
        }
        try:
            ms, samples, out = bench(run, args.warmup, args.iters, args.repeats, layer_device)
            if reference is None:
                reference = out.detach()
            row.update(
                {
                    "latency_ms": ms,
                    "latency_samples_ms": samples,
                    "ok": True,
                    "memory": memory_stats(layer_device),
                }
            )
            if reference is not None and name != REFERENCE_CASES["prefill_full_layer"]:
                row.update(tensor_diff(out, reference))
        except Exception as exc:
            row.update(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        results.append(row)
        del layer
        clear_memory(layer_device)
    return results


def prefill_core_bench(
    model: Any,
    rotary_emb: Any,
    layer_idx: int,
    batch: int,
    seq: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    layer = make_attention_layer(
        model,
        layer_idx,
        device,
        dtype,
        AttentionOptimizationConfig(attn_kernel="npu-official", fused_rope=True),
    )
    results: list[dict[str, Any]] = []
    try:
        flash_prefill, _ = layer._load_npu_attention_ops()
    except Exception as exc:
        results.append(
            {
                "case": "official_pfa",
                "phase": "prefill_core",
                "layer_idx": layer_idx,
                "batch": batch,
                "seq": seq,
                "cache_len": None,
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        del layer
        clear_memory(device)
        return results

    hidden = torch.randn(batch, seq, model.config.hidden_size, device=device, dtype=dtype)
    q, k, v = project_attention_states(layer, hidden, rotary_emb, use_fused_rope=True)
    head_dim = int(getattr(layer.self_attn, "head_dim"))
    num_heads = layer._num_attention_heads(head_dim)
    num_kv_heads = layer._num_key_value_heads(head_dim)
    scale = float(getattr(layer.self_attn, "scaling", 1.0 / math.sqrt(head_dim)))
    repeat_factor = num_heads // num_kv_heads
    k_repeated = k.repeat_interleave(repeat_factor, dim=1)
    v_repeated = v.repeat_interleave(repeat_factor, dim=1)
    causal_mask = None
    if seq > 1:
        causal_mask = torch.triu(
            torch.ones(seq, seq, dtype=torch.bool, device=device),
            diagonal=1,
        ).contiguous()

    def hf_sdpa_run() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q,
            k_repeated,
            v_repeated,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
        )

    hf_out = None
    try:
        hf_ms, hf_samples, hf_out = bench(hf_sdpa_run, args.warmup, args.iters, args.repeats, device)
        results.append(
            {
                "case": "hf_sdpa",
                "phase": "prefill_core",
                "layer_idx": layer_idx,
                "batch": batch,
                "seq": seq,
                "cache_len": None,
                "latency_ms": hf_ms,
                "latency_samples_ms": hf_samples,
                "ok": True,
                "memory": memory_stats(device),
            }
        )
    except Exception as exc:
        results.append(
            {
                "case": "hf_sdpa",
                "phase": "prefill_core",
                "layer_idx": layer_idx,
                "batch": batch,
                "seq": seq,
                "cache_len": None,
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )

    def official_pfa_run() -> torch.Tensor:
        return flash_prefill(
            q,
            k,
            v,
            atten_mask=causal_mask,
            num_heads=num_heads,
            input_layout="BNSD",
            scale_value=scale,
            num_key_value_heads=num_kv_heads,
        )

    official_row: dict[str, Any] = {
        "case": "official_pfa",
        "phase": "prefill_core",
        "layer_idx": layer_idx,
        "batch": batch,
        "seq": seq,
        "cache_len": None,
    }
    try:
        official_ms, official_samples, official_out = bench(
            official_pfa_run,
            args.warmup,
            args.iters,
            args.repeats,
            device,
        )
        official_row.update(
            {
                "latency_ms": official_ms,
                "latency_samples_ms": official_samples,
                "ok": True,
                "memory": memory_stats(device),
            }
        )
        if hf_out is not None:
            official_row.update(tensor_diff(official_out, hf_out))
    except Exception as exc:
        official_row.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    results.append(official_row)
    del layer
    clear_memory(device)
    return results


class FrozenLayerCache:
    """Read-only cache shim for stable decode full-layer benchmarking."""

    def __init__(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.layer_idx = layer_idx
        self.key_states = key_states
        self.value_states = value_states

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx != self.layer_idx:
            raise RuntimeError(f"Frozen cache for layer {self.layer_idx} cannot serve layer {layer_idx}")
        return (
            torch.cat((self.key_states, key_states), dim=2).contiguous(),
            torch.cat((self.value_states, value_states), dim=2).contiguous(),
        )


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


def project_attention_states(
    layer: AttentionLayer,
    hidden: torch.Tensor,
    rotary_emb: Any,
    use_fused_rope: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return project_decode_states(layer, hidden, rotary_emb, use_fused_rope)


def make_frozen_cache(
    layer: AttentionLayer,
    cache_hidden: torch.Tensor,
    rotary_emb: Any,
    use_fused_rope: bool,
) -> FrozenLayerCache:
    _, key_cache, value_cache = project_attention_states(layer, cache_hidden, rotary_emb, use_fused_rope)
    return FrozenLayerCache(layer.layer_idx, key_cache, value_cache)


def decode_full_layer_configs() -> list[tuple[str, AttentionOptimizationConfig]]:
    return [
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


def full_layer_decode_bench(
    model: Any,
    rotary_emb: Any,
    layer_idx: int,
    batch: int,
    cache_lens: list[int],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for cache_len in cache_lens:
        token_hidden = torch.randn(batch, 1, model.config.hidden_size, device=device, dtype=dtype)
        cache_hidden = torch.randn(batch, cache_len, model.config.hidden_size, device=device, dtype=dtype)
        position_ids = torch.full((batch, 1), cache_len, device=device, dtype=torch.long)
        reference: Optional[torch.Tensor] = None
        for name, config in decode_full_layer_configs():
            layer = make_attention_layer(model, layer_idx, device, dtype, config)
            pos = position_embeddings(rotary_emb, token_hidden, position_ids)
            cache = make_frozen_cache(layer, cache_hidden, rotary_emb, config.fused_rope)

            def run() -> torch.Tensor:
                return layer(
                    hidden_states=token_hidden,
                    position_ids=position_ids,
                    position_embeddings=pos,
                    past_key_value=cache,
                    use_cache=True,
                )[0]

            row: dict[str, Any] = {
                "case": name,
                "phase": "decode_full_layer",
                "layer_idx": layer_idx,
                "batch": batch,
                "seq": 1,
                "cache_len": cache_len,
                "optimization": config.to_dict(),
            }
            try:
                ms, samples, out = bench(run, args.warmup, args.iters, args.repeats, device)
                if reference is None:
                    reference = out.detach()
                row.update(
                    {
                        "latency_ms": ms,
                        "latency_samples_ms": samples,
                        "ok": True,
                        "memory": memory_stats(device),
                    }
                )
                if reference is not None and name != REFERENCE_CASES["decode_full_layer"]:
                    row.update(tensor_diff(out, reference))
            except Exception as exc:
                row.update(
                    {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            results.append(row)
            del layer, cache
            clear_memory(device)
    return results


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
    layer_idx: int,
    batch: int,
    cache_lens: list[int],
    include_community: bool,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    official_layer = make_attention_layer(
        model,
        layer_idx,
        device,
        dtype,
        AttentionOptimizationConfig(attn_kernel="npu-official", fused_rmsnorm=True, fused_rope=True),
    )
    results: list[dict[str, Any]] = []
    try:
        flash_prefill, flash_decode = official_layer._load_npu_attention_ops()
        del flash_prefill
    except Exception as exc:
        for cache_len in cache_lens:
            results.append(
                {
                    "case": "official_ifa",
                    "phase": "decode_core",
                    "layer_idx": layer_idx,
                    "batch": batch,
                    "seq": 1,
                    "cache_len": cache_len,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        del official_layer
        clear_memory(device)
        return results

    head_dim = int(getattr(official_layer.self_attn, "head_dim"))
    num_heads = official_layer._num_attention_heads(head_dim)
    num_kv_heads = official_layer._num_key_value_heads(head_dim)
    scale = float(getattr(official_layer.self_attn, "scaling", 1.0 / math.sqrt(head_dim)))
    community_v2 = import_community(args.community_root, "v2") if include_community else None
    community_v3 = import_community(args.community_root, "v3") if include_community else None

    for cache_len in cache_lens:
        cache_hidden = torch.randn(batch, cache_len, model.config.hidden_size, device=device, dtype=dtype)
        token_hidden = torch.randn(batch, 1, model.config.hidden_size, device=device, dtype=dtype)
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

        try:
            hf_ms, hf_samples, hf_out = bench(hf_sdpa_run, args.warmup, args.iters, args.repeats, device)
            results.append(
                {
                    "case": "hf_sdpa",
                    "phase": "decode_core",
                    "layer_idx": layer_idx,
                    "batch": batch,
                    "seq": 1,
                    "cache_len": cache_len,
                    "latency_ms": hf_ms,
                    "latency_samples_ms": hf_samples,
                    "ok": True,
                    "memory": memory_stats(device),
                }
            )
        except Exception as exc:
            hf_out = None
            results.append(
                {
                    "case": "hf_sdpa",
                    "phase": "decode_core",
                    "layer_idx": layer_idx,
                    "batch": batch,
                    "seq": 1,
                    "cache_len": cache_len,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
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

        official_row: dict[str, Any] = {
            "case": "official_ifa",
            "phase": "decode_core",
            "layer_idx": layer_idx,
            "batch": batch,
            "seq": 1,
            "cache_len": cache_len,
        }
        try:
            official_ms, official_samples, official_out = bench(
                official_run, args.warmup, args.iters, args.repeats, device
            )
            official_row.update(
                {
                    "latency_ms": official_ms,
                    "latency_samples_ms": official_samples,
                    "ok": True,
                    "memory": memory_stats(device),
                }
            )
            if hf_out is not None:
                official_row.update(tensor_diff(official_out, hf_out))
        except Exception as exc:
            official_row.update(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        results.append(official_row)

        q_bsnd = q.transpose(1, 2).contiguous()
        k_bsnd = k_cache.transpose(1, 2).contiguous()
        v_bsnd = v_cache.transpose(1, 2).contiguous()
        hf_bsnd = hf_out.transpose(1, 2).contiguous() if hf_out is not None else None
        if not include_community:
            clear_memory(device)
            continue

        for name, fn in [("community_v2", community_v2), ("community_v3", community_v3)]:
            if fn is None:
                results.append(
                    {
                        "case": name,
                        "phase": "decode_core",
                        "layer_idx": layer_idx,
                        "batch": batch,
                        "seq": 1,
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

            row: dict[str, Any] = {
                "case": name,
                "phase": "decode_core",
                "layer_idx": layer_idx,
                "batch": batch,
                "seq": 1,
                "cache_len": cache_len,
            }
            try:
                ms, samples, out = bench(community_run, args.warmup, args.iters, args.repeats, device)
                row.update(
                    {
                        "latency_ms": ms,
                        "latency_samples_ms": samples,
                        "ok": True,
                        "memory": memory_stats(device),
                    }
                )
                if hf_bsnd is not None:
                    row.update(tensor_diff(out, hf_bsnd))
            except Exception as exc:
                row.update(
                    {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            results.append(row)
        clear_memory(device)
    del official_layer
    clear_memory(device)
    return results


def reference_group_key(row: dict[str, Any], include_layer: bool) -> tuple[Any, ...]:
    key = [
        row.get("phase"),
        row.get("batch"),
        row.get("seq"),
        row.get("cache_len"),
    ]
    if include_layer:
        key.append(row.get("layer_idx"))
    return tuple(key)


def annotate_reference_deltas(rows: list[dict[str, Any]]) -> None:
    references: dict[tuple[Any, ...], float] = {}
    for row in rows:
        ref_case = REFERENCE_CASES.get(str(row.get("phase")))
        if row.get("case") == ref_case and row.get("ok") and "latency_ms" in row:
            references[reference_group_key(row, include_layer=True)] = float(row["latency_ms"])
    for row in rows:
        ref = references.get(reference_group_key(row, include_layer=True))
        if ref is None or not row.get("ok") or "latency_ms" not in row:
            continue
        latency = float(row["latency_ms"])
        row["delta_ms_vs_hf"] = latency - ref
        row["speedup_vs_hf"] = ref / latency if latency > 0 else None


def aggregate_rows(rows: list[dict[str, Any]], layer_indices: list[int]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("phase"),
            row.get("case"),
            row.get("batch"),
            row.get("seq"),
            row.get("cache_len"),
        )
        grouped[key].append(row)

    aggregates: list[dict[str, Any]] = []
    for (phase, case, batch, seq, cache_len), group in sorted(grouped.items(), key=lambda item: str(item[0])):
        ok_rows = [row for row in group if row.get("ok") and "latency_ms" in row]
        latencies = [float(row["latency_ms"]) for row in ok_rows]
        agg: dict[str, Any] = {
            "phase": phase,
            "case": case,
            "batch": batch,
            "seq": seq,
            "cache_len": cache_len,
            "layer_count_requested": len(layer_indices),
            "ok_count": len(ok_rows),
            "fail_count": len(group) - len(ok_rows),
            "layers": [row.get("layer_idx") for row in group],
        }
        if latencies:
            agg.update(
                {
                    "latency_median_ms": float(statistics.median(latencies)),
                    "latency_mean_ms": float(statistics.mean(latencies)),
                    "latency_min_ms": float(min(latencies)),
                    "latency_max_ms": float(max(latencies)),
                    "latency_std_ms": float(statistics.pstdev(latencies)) if len(latencies) > 1 else 0.0,
                }
            )
        for field in ("max_abs_diff", "mean_abs_diff", "delta_ms_vs_hf", "speedup_vs_hf"):
            values = [float(row[field]) for row in ok_rows if row.get(field) is not None]
            if values:
                agg[f"{field}_median"] = float(statistics.median(values))
                agg[f"{field}_max"] = float(max(values))
        aggregates.append(agg)

    aggregate_refs: dict[tuple[Any, ...], float] = {}
    for row in aggregates:
        ref_case = REFERENCE_CASES.get(str(row.get("phase")))
        if row.get("case") == ref_case and "latency_median_ms" in row:
            aggregate_refs[
                (row.get("phase"), row.get("batch"), row.get("seq"), row.get("cache_len"))
            ] = float(row["latency_median_ms"])
    for row in aggregates:
        ref = aggregate_refs.get((row.get("phase"), row.get("batch"), row.get("seq"), row.get("cache_len")))
        if ref is None or "latency_median_ms" not in row:
            continue
        latency = float(row["latency_median_ms"])
        row["aggregate_delta_ms_vs_hf"] = latency - ref
        row["aggregate_speedup_vs_hf"] = ref / latency if latency > 0 else None
    return aggregates


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "phase",
        "case",
        "batch",
        "seq",
        "cache_len",
        "ok_count",
        "fail_count",
        "latency_median_ms",
        "latency_mean_ms",
        "latency_min_ms",
        "latency_max_ms",
        "latency_std_ms",
        "aggregate_delta_ms_vs_hf",
        "aggregate_speedup_vs_hf",
        "max_abs_diff_median",
        "max_abs_diff_max",
        "mean_abs_diff_median",
        "mean_abs_diff_max",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def format_optional(value: Any, precision: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


def write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Attention layer benchmark summary",
        "",
        "| phase | case | batch | seq | cache_len | ok/fail | median ms | speedup vs HF | delta ms vs HF | max abs diff |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {phase} | {case} | {batch} | {seq} | {cache_len} | {ok}/{fail} | {lat} | {speedup} | {delta} | {diff} |".format(
                phase=row.get("phase"),
                case=row.get("case"),
                batch=row.get("batch"),
                seq=row.get("seq"),
                cache_len=format_optional(row.get("cache_len")),
                ok=row.get("ok_count"),
                fail=row.get("fail_count"),
                lat=format_optional(row.get("latency_median_ms")),
                speedup=format_optional(row.get("aggregate_speedup_vs_hf")),
                delta=format_optional(row.get("aggregate_delta_ms_vs_hf")),
                diff=format_optional(row.get("max_abs_diff_median")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def persist_results(
    args: argparse.Namespace,
    payload: dict[str, Any],
    aggregates: list[dict[str, Any]],
) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.summary_csv:
        write_summary_csv(Path(args.summary_csv), aggregates)
    if args.summary_md:
        write_summary_md(Path(args.summary_md), aggregates)


def log_progress(args: argparse.Namespace, message: str) -> None:
    if args.no_progress:
        return
    print(f"[bench] {message}", flush=True)


def main() -> None:
    args = parse_args()
    layer_device = torch.device(args.device)
    input_device = torch.device(args.input_device or args.device)
    if layer_device.type == "npu" or input_device.type == "npu":
        import torch_npu  # type: ignore[import-not-found]  # noqa: F401
    dtype = dtype_from_name(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()
    rotary_emb_input = copy.deepcopy(model.model.rotary_emb).to(device=input_device)
    rotary_emb_layer = copy.deepcopy(model.model.rotary_emb).to(device=layer_device)
    num_layers = len(model.model.layers)
    layer_indices = parse_layer_indices(args.layer_idxs, args.layer_idx, num_layers, "--layer-idxs")
    community_layer_indices = parse_layer_indices(
        args.community_layer_idxs,
        args.layer_idx,
        num_layers,
        "--community-layer-idxs",
    )
    batch_values = parse_int_list(args.batches, [args.batch], "--batches")
    seq_values = parse_int_list(args.seqs, [args.seq], "--seqs")
    cache_lens = parse_int_list(args.decode_cache_lens, [], "--decode-cache-lens")

    rows: list[dict[str, Any]] = []
    groups_done = 0

    def checkpoint() -> None:
        annotate_reference_deltas(rows)
        aggregates = aggregate_rows(rows, layer_indices)
        payload = {
            "model_name": args.model_name,
            "device": str(layer_device),
            "input_device": str(input_device),
            "dtype": str(dtype),
            "num_layers": num_layers,
            "layer_indices": layer_indices,
            "community_layer_indices": community_layer_indices if args.include_community else [],
            "batches": batch_values,
            "prefill_seqs": seq_values,
            "decode_cache_lens": cache_lens,
            "include_prefill_core": args.include_prefill_core,
            "include_decode_full_layer": args.include_decode_full_layer,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "complete": False,
            "results": rows,
            "aggregate_results": aggregates,
        }
        persist_results(args, payload, aggregates)

    for layer_idx in layer_indices:
        for batch in batch_values:
            if not args.skip_prefill:
                for seq in seq_values:
                    log_progress(args, f"prefill layer={layer_idx} batch={batch} seq={seq}")
                    rows.extend(
                        full_layer_prefill_bench(
                            model,
                            rotary_emb_input,
                            layer_idx,
                            batch,
                            seq,
                            args,
                            layer_device,
                            input_device,
                            dtype,
                        )
                    )
                    groups_done += 1
                    if args.checkpoint_every > 0 and groups_done % args.checkpoint_every == 0:
                        checkpoint()
                    if args.include_prefill_core:
                        log_progress(args, f"prefill-core layer={layer_idx} batch={batch} seq={seq}")
                        rows.extend(
                            prefill_core_bench(
                                model,
                                rotary_emb_layer,
                                layer_idx,
                                batch,
                                seq,
                                args,
                                layer_device,
                                dtype,
                            )
                        )
                        groups_done += 1
                        if args.checkpoint_every > 0 and groups_done % args.checkpoint_every == 0:
                            checkpoint()
            if not args.skip_decode:
                if args.include_decode_full_layer:
                    log_progress(
                        args,
                        f"decode-full-layer layer={layer_idx} batch={batch} cache_lens={','.join(map(str, cache_lens))}",
                    )
                    rows.extend(
                        full_layer_decode_bench(
                            model,
                            rotary_emb_layer,
                            layer_idx,
                            batch,
                            cache_lens,
                            args,
                            layer_device,
                            dtype,
                        )
                    )
                    groups_done += 1
                    if args.checkpoint_every > 0 and groups_done % args.checkpoint_every == 0:
                        checkpoint()
                log_progress(
                    args,
                    f"decode layer={layer_idx} batch={batch} cache_lens={','.join(map(str, cache_lens))}",
                )
                rows.extend(
                    decode_core_bench(
                        model,
                        rotary_emb_layer,
                        layer_idx,
                        batch,
                        cache_lens,
                        args.include_community and layer_idx in community_layer_indices,
                        args,
                        layer_device,
                        dtype,
                    )
                )
                groups_done += 1
                if args.checkpoint_every > 0 and groups_done % args.checkpoint_every == 0:
                    checkpoint()

    annotate_reference_deltas(rows)
    aggregates = aggregate_rows(rows, layer_indices)
    results = {
        "model_name": args.model_name,
        "device": str(layer_device),
        "input_device": str(input_device),
        "dtype": str(dtype),
        "num_layers": num_layers,
        "layer_indices": layer_indices,
        "community_layer_indices": community_layer_indices if args.include_community else [],
        "batches": batch_values,
        "prefill_seqs": seq_values,
        "decode_cache_lens": cache_lens,
        "include_prefill_core": args.include_prefill_core,
        "include_decode_full_layer": args.include_decode_full_layer,
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "complete": True,
        "results": rows,
        "aggregate_results": aggregates,
    }
    persist_results(args, results, aggregates)
    if not args.no_print_json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
