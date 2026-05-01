#!/usr/bin/env python3
"""Two-rank communication microbenchmark for DBO send timing calibration.

Measures three related spans for the same payload:
- sender isend enqueue: time for dist.isend() to return
- sender Work completion: time from isend() start to Work.wait() return
- receiver wait: time from dist.irecv() post to Work.wait() return

The completion span includes real data movement, but can also include backend
queueing, receiver readiness, communication-stream scheduling, and completion
notification. It is therefore comparable to DBO's completion-mode pipeline bars,
not a pure hardware wire-time measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as stats
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "uint8": torch.uint8,
}


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": stats.mean(values),
        "median": stats.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def resolve_backend(requested: str) -> tuple[str, str, Any | None]:
    """Return (device_type, dist_backend, device_module)."""
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        else:
            try:
                import torch_npu  # noqa: F401
            except Exception:
                requested = "cpu"
            else:
                requested = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("backend=cuda requested but CUDA is not available")
        return "cuda", "nccl", torch.cuda
    if requested == "npu":
        try:
            import torch_npu  # noqa: F401
        except Exception as exc:
            raise RuntimeError("backend=npu requested but torch_npu cannot be imported") from exc
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("backend=npu requested but torch.npu is not available")
        return "npu", "hccl", torch.npu
    if requested == "cpu":
        return "cpu", "gloo", None
    raise ValueError(f"Unsupported backend: {requested}")


def synchronize(device_module: Any | None) -> None:
    if device_module is not None:
        device_module.synchronize()


def make_tensor(num_bytes: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    element_size = torch.empty((), dtype=dtype).element_size()
    numel = max(1, math.ceil(num_bytes / element_size))
    return torch.empty(numel, dtype=dtype, device=device)


def run_warmup(tensor: torch.Tensor, rank: int, peer: int, warmup: int, sender_delay_s: float) -> None:
    for i in range(warmup):
        tag = 10_000 + i
        dist.barrier()
        if rank == 0:
            if sender_delay_s > 0:
                time.sleep(sender_delay_s)
            handle = dist.isend(tensor, dst=peer, tag=tag)
            handle.wait()
        else:
            recv = torch.empty_like(tensor)
            handle = dist.irecv(recv, src=peer, tag=tag)
            handle.wait()
    dist.barrier()


def run_isend_case(
    tensor: torch.Tensor,
    rank: int,
    peer: int,
    iters: int,
    sender_delay_s: float,
    device_module: Any | None,
    tag_base: int,
) -> tuple[list[float], list[float], list[float]]:
    enqueue_ms: list[float] = []
    completion_ms: list[float] = []
    recv_wait_ms: list[float] = []
    for i in range(iters):
        tag = tag_base + i
        dist.barrier()
        synchronize(device_module)
        if rank == 0:
            if sender_delay_s > 0:
                time.sleep(sender_delay_s)
            start = time.perf_counter()
            handle = dist.isend(tensor, dst=peer, tag=tag)
            enqueued = time.perf_counter()
            handle.wait()
            done = time.perf_counter()
            enqueue_ms.append((enqueued - start) * 1000)
            completion_ms.append((done - start) * 1000)
        else:
            recv = torch.empty_like(tensor)
            start = time.perf_counter()
            handle = dist.irecv(recv, src=peer, tag=tag)
            handle.wait()
            done = time.perf_counter()
            recv_wait_ms.append((done - start) * 1000)
    dist.barrier()
    return enqueue_ms, completion_ms, recv_wait_ms


def run_blocking_case(
    tensor: torch.Tensor,
    rank: int,
    peer: int,
    iters: int,
    sender_delay_s: float,
    device_module: Any | None,
    tag_base: int,
) -> tuple[list[float], list[float]]:
    send_ms: list[float] = []
    recv_ms: list[float] = []
    for i in range(iters):
        tag = tag_base + i
        dist.barrier()
        synchronize(device_module)
        if rank == 0:
            if sender_delay_s > 0:
                time.sleep(sender_delay_s)
            start = time.perf_counter()
            dist.send(tensor, dst=peer, tag=tag)
            done = time.perf_counter()
            send_ms.append((done - start) * 1000)
        else:
            recv = torch.empty_like(tensor)
            start = time.perf_counter()
            dist.recv(recv, src=peer, tag=tag)
            done = time.perf_counter()
            recv_ms.append((done - start) * 1000)
    dist.barrier()
    return send_ms, recv_ms


def gather_rank_results(local: dict[str, Any], rank: int) -> list[dict[str, Any]]:
    gathered: list[dict[str, Any] | None] = [None, None] if rank == 0 else []
    dist.gather_object(local, gathered if rank == 0 else None, dst=0)
    return [item for item in gathered if item is not None] if rank == 0 else []


def write_csv(path: Path, data: dict[str, Any]) -> None:
    rows = []
    for item in data["results"]:
        rows.append({
            "size_mib_requested": item["size_mib_requested"],
            "tensor_mib": item["tensor_mib"],
            "tensor_bytes": item["tensor_bytes"],
            "sender_enqueue_mean_ms": item["sender_enqueue_ms"]["mean"],
            "sender_completion_mean_ms": item["sender_completion_ms"]["mean"],
            "receiver_wait_mean_ms": item["receiver_wait_ms"]["mean"],
            "blocking_send_mean_ms": item.get("blocking_send_ms", {}).get("mean", ""),
            "blocking_recv_mean_ms": item.get("blocking_recv_ms", {}).get("mean", ""),
            "completion_bandwidth_mib_s": item["completion_bandwidth_mib_s"],
        })
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["auto", "cuda", "npu", "cpu"], default="auto")
    parser.add_argument("--sizes-mib", default="0.004,0.031,1,16,32",
                        help="Comma-separated payload sizes in MiB.")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--sender-delay-ms", type=float, default=1.0,
                        help="Delay sender after barrier so receiver can post irecv/recv first.")
    parser.add_argument("--blocking", action="store_true",
                        help="Also measure blocking dist.send/dist.recv.")
    parser.add_argument("--output", default="results/comm_bench/comm_transfer.json")
    parser.add_argument("--csv-output", default="",
                        help="Optional CSV output path. Defaults to JSON path with .csv suffix.")
    args = parser.parse_args()

    device_type, dist_backend, device_module = resolve_backend(args.backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        raise RuntimeError("bench_comm_transfer.py requires exactly 2 ranks; launch with torchrun --nproc_per_node=2")

    if device_module is not None:
        device_module.set_device(local_rank)
    device = torch.device(f"{device_type}:{local_rank}" if device_type != "cpu" else "cpu")
    dist.init_process_group(backend=dist_backend, init_method="env://")

    peer = 1 - rank
    dtype = DTYPES[args.dtype]
    size_mib_values = parse_csv_floats(args.sizes_mib)
    sender_delay_s = args.sender_delay_ms / 1000.0
    local_results: list[dict[str, Any]] = []

    for size_idx, size_mib in enumerate(size_mib_values):
        requested_bytes = max(1, int(size_mib * 1024 * 1024))
        tensor = make_tensor(requested_bytes, dtype, device)
        tensor_bytes = tensor.numel() * tensor.element_size()
        tensor_mib = tensor_bytes / (1024 * 1024)

        run_warmup(tensor, rank, peer, args.warmup, sender_delay_s)
        enqueue_ms, completion_ms, recv_wait_ms = run_isend_case(
            tensor, rank, peer, args.iters, sender_delay_s, device_module, 20_000 + size_idx * 10_000
        )

        blocking_send_ms: list[float] = []
        blocking_recv_ms: list[float] = []
        if args.blocking:
            blocking_send_ms, blocking_recv_ms = run_blocking_case(
                tensor, rank, peer, args.iters, sender_delay_s, device_module, 200_000 + size_idx * 10_000
            )

        local_results.append({
            "size_mib_requested": size_mib,
            "tensor_bytes": tensor_bytes,
            "tensor_mib": tensor_mib,
            "sender_enqueue_samples_ms": enqueue_ms,
            "sender_completion_samples_ms": completion_ms,
            "receiver_wait_samples_ms": recv_wait_ms,
            "blocking_send_samples_ms": blocking_send_ms,
            "blocking_recv_samples_ms": blocking_recv_ms,
        })

    gathered = gather_rank_results({"rank": rank, "results": local_results}, rank)
    if rank == 0:
        by_rank = {entry["rank"]: entry["results"] for entry in gathered}
        sender_results = by_rank[0]
        receiver_results = by_rank[1]
        merged = []
        for sender, receiver in zip(sender_results, receiver_results):
            completion_summary = summarize(sender["sender_completion_samples_ms"])
            completion_mean = completion_summary["mean"]
            tensor_mib = sender["tensor_mib"]
            merged_item = {
                "size_mib_requested": sender["size_mib_requested"],
                "tensor_bytes": sender["tensor_bytes"],
                "tensor_mib": tensor_mib,
                "sender_enqueue_ms": summarize(sender["sender_enqueue_samples_ms"]),
                "sender_completion_ms": completion_summary,
                "receiver_wait_ms": summarize(receiver["receiver_wait_samples_ms"]),
                "completion_bandwidth_mib_s": (tensor_mib * 1000 / completion_mean) if completion_mean > 0 else 0.0,
            }
            if args.blocking:
                merged_item["blocking_send_ms"] = summarize(sender["blocking_send_samples_ms"])
                merged_item["blocking_recv_ms"] = summarize(receiver["blocking_recv_samples_ms"])
            merged.append(merged_item)

        output = {
            "meta": {
                "backend": device_type,
                "dist_backend": dist_backend,
                "dtype": args.dtype,
                "warmup": args.warmup,
                "iters": args.iters,
                "sender_delay_ms": args.sender_delay_ms,
                "world_size": world_size,
                "note": (
                    "sender_completion_ms is isend start -> Work.wait completion. "
                    "It includes real data movement plus queueing/receiver-readiness/scheduling overhead."
                ),
            },
            "results": merged,
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        csv_path = Path(args.csv_output) if args.csv_output else output_path.with_suffix(".csv")
        write_csv(csv_path, output)
        print(f"✅ Communication benchmark written: {output_path}")
        print(f"✅ CSV summary written: {csv_path}")
        for item in merged:
            print(
                f"{item['tensor_mib']:.4f} MiB: "
                f"enqueue={item['sender_enqueue_ms']['mean']:.4f}ms, "
                f"completion={item['sender_completion_ms']['mean']:.4f}ms, "
                f"recv_wait={item['receiver_wait_ms']['mean']:.4f}ms, "
                f"bw={item['completion_bandwidth_mib_s']:.2f} MiB/s"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
