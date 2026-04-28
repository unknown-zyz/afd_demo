#!/usr/bin/env python3
"""Capture serial prefill timing and merge TTFT/TPOT split fields into cache JSONs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONFIG_RE = re.compile(r"^b(?P<batch>\d+)_s(?P<seq>\d+)_t(?P<tokens>\d+)\.json$")


@dataclass(frozen=True)
class Config:
    batch: int
    seq: int
    tokens: int
    cache_path: Path


def parse_list(raw: str) -> set[int] | None:
    if not raw:
        return None
    return {int(item) for item in raw.split(",") if item}


def iter_configs(cache_dir: Path, batches: set[int] | None, seqs: set[int] | None, tokens: set[int] | None) -> list[Config]:
    configs: list[Config] = []
    for path in sorted(cache_dir.glob("b*_s*_t*.json")):
        match = CONFIG_RE.match(path.name)
        if not match:
            continue
        config = Config(
            batch=int(match.group("batch")),
            seq=int(match.group("seq")),
            tokens=int(match.group("tokens")),
            cache_path=path,
        )
        if batches is not None and config.batch not in batches:
            continue
        if seqs is not None and config.seq not in seqs:
            continue
        if tokens is not None and config.tokens not in tokens:
            continue
        configs.append(config)
    return configs


def has_split(path: Path) -> bool:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return "prefill_ms" in data and "decode_step_ms" in data


def build_command(args: argparse.Namespace, config: Config) -> tuple[list[str], Path]:
    if args.backend == "npu":
        suffix = f"serial-prefill_npu_b{config.batch}_s{config.seq}_t{config.tokens}"
        raw = ROOT / "results" / "prefill_dbo" / f"timing_attention_{suffix}.json"
        cmd = [
            "bash",
            "scripts/run_npu.sh",
            "--attn-size",
            str(args.attn_size),
            "--ffn-size",
            str(args.ffn_size),
            "--ffn-tp-size",
            str(args.ffn_tp_size),
            "--batch",
            str(config.batch),
            "--seq",
            str(config.seq),
            "--tokens",
            str(config.tokens),
            "--model-name",
            args.model_name,
            "--no-dbo",
            "--no-generate",
        ]
        return cmd, raw

    suffix = f"warmup_serial_{args.deployment}_b{config.batch}_s{config.seq}_t{config.tokens}"
    raw = ROOT / "results" / "prefill_dbo" / f"timing_attention_{suffix}.json"
    cmd = [
        "bash",
        "scripts/run_single.sh",
        args.deployment,
        str(config.batch),
        str(config.seq),
        "--tokens",
        str(config.tokens),
        "--no-dbo",
        "--no-generate",
        "--warmup-p2p",
        "--warmup-rounds",
        str(args.warmup_rounds),
    ]
    return cmd, raw


def merge_split(cache_path: Path, raw_path: Path) -> tuple[float, float]:
    cache = json.loads(cache_path.read_text())
    raw = json.loads(raw_path.read_text())
    prefill_ms = float(raw["total_time_ms"])
    total_ms = float(cache["total_time_ms"])
    tokens = int(cache.get("max_new_tokens") or 0)
    if tokens <= 0:
        raise ValueError(f"{cache_path} has non-positive max_new_tokens")
    if total_ms < prefill_ms:
        raise ValueError(
            f"{cache_path} total_time_ms ({total_ms:.3f}) is smaller than captured "
            f"prefill_ms ({prefill_ms:.3f}); serial cache is not a full generation run"
        )
    decode_step_ms = (total_ms - prefill_ms) / tokens
    cache["prefill_ms"] = prefill_ms
    cache["decode_step_ms"] = decode_step_ms
    cache_path.write_text(json.dumps(cache, indent=2) + "\n")
    return prefill_ms, decode_step_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results", help="Results root containing serial/cache")
    parser.add_argument("--backend", choices=("cuda", "npu"), default="cuda")
    parser.add_argument("--deployment", default="local", help="run_single deployment for CUDA/GPU capture")
    parser.add_argument("--batches", default="", help="Comma-separated batch filter")
    parser.add_argument("--seqs", default="", help="Comma-separated sequence filter")
    parser.add_argument("--tokens", default="", help="Comma-separated token-count filter")
    parser.add_argument("--skip-batches", default="", help="Comma-separated batches to skip")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "/models/Qwen3-30B-A3B"))
    parser.add_argument("--attn-size", type=int, default=1)
    parser.add_argument("--ffn-size", type=int, default=1)
    parser.add_argument("--ffn-tp-size", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root
    cache_dir = root / "serial" / "cache"
    if not cache_dir.is_dir():
        print(f"Missing serial cache directory: {cache_dir}", file=sys.stderr)
        return 2

    skip_batches = parse_list(args.skip_batches) or set()
    configs = iter_configs(cache_dir, parse_list(args.batches), parse_list(args.seqs), parse_list(args.tokens))
    if not configs:
        print("No matching serial cache files.")
        return 0

    failures = 0
    for config in configs:
        name = config.cache_path.stem
        if config.batch in skip_batches:
            print(f"[skip-batch] {name}")
            continue
        if args.skip_existing and has_split(config.cache_path):
            print(f"[skip-existing] {name}")
            continue

        cmd, raw_path = build_command(args, config)
        print(f"\n=== capture serial split: {name} ===")
        print(" ".join(cmd))
        if args.dry_run:
            continue

        raw_path.unlink(missing_ok=True)
        env = os.environ.copy()
        if args.backend == "npu":
            env.setdefault("ASCEND_VISIBLE_DEVICES", env.get("VISIBLE_DEVS", "0,1,2,3"))
        env.setdefault("MASTER_PORT", str(29500 + (os.getpid() % 2000)))
        result = subprocess.run(cmd, cwd=ROOT, env=env)
        if result.returncode != 0:
            print(f"[FAIL rc={result.returncode}] {name}", file=sys.stderr)
            failures += 1
            continue
        if not raw_path.exists():
            print(f"[FAIL] missing raw prefill timing: {raw_path}", file=sys.stderr)
            failures += 1
            continue
        try:
            prefill_ms, decode_step_ms = merge_split(config.cache_path, raw_path)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            print(f"[FAIL] merge failed for {name}: {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"[merged] prefill_ms={prefill_ms:.3f} decode_step_ms={decode_step_ms:.3f}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
