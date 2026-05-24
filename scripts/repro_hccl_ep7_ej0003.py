#!/usr/bin/env python3
"""Minimal HCCL EP7 group/barrier repro for Host2 EJ0003 triage.

This script intentionally avoids Qwen model loading and TBE/JIT execution. It
only initializes torch.distributed/HCCL, creates the same FFN EP groups as the
real EP7 path, and runs barriers to isolate HCCL runtime / group bootstrap.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import timedelta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal local EP7 HCCL new_group/barrier repro."
    )
    parser.add_argument("--spawn-local", action="store_true", help="Spawn all ranks locally.")
    parser.add_argument("--world-size", type=int, default=8, help="Total ranks.")
    parser.add_argument("--rank", type=int, default=None, help="Current rank when not spawning.")
    parser.add_argument("--local-rank", type=int, default=None, help="Current local NPU index.")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", default="29901")
    parser.add_argument("--backend", default="hccl", choices=("hccl", "gloo"))
    parser.add_argument("--attn-size", type=int, default=1, help="Attention ranks start at 0.")
    parser.add_argument("--ffn-size", type=int, default=7, help="FFN ranks after attention ranks.")
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--groups",
        default="all",
        choices=("ep", "dispatch", "reduce", "all"),
        help="Which FFN subgroup(s) to create and barrier.",
    )
    parser.add_argument(
        "--skip-ep-groups",
        action="store_true",
        help="Only init the default world group and run world barriers.",
    )
    parser.add_argument(
        "--visible-devices",
        default=None,
        help="ASCEND_VISIBLE_DEVICES value for local spawn; default is 0..world_size-1.",
    )
    return parser.parse_args()


def _log(rank: int, message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} [rank{rank}] {message}", flush=True)


def _stage(rank: int, name: str, fn) -> None:
    _log(rank, f"BEGIN {name}")
    start = time.perf_counter()
    fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _log(rank, f"END   {name}: {elapsed_ms:.3f} ms")


def _selected_groups(name: str) -> list[str]:
    if name == "all":
        return ["ep", "dispatch", "reduce"]
    return [name]


def _spawn_local(args: argparse.Namespace) -> int:
    visible = args.visible_devices or os.environ.get("ASCEND_VISIBLE_DEVICES")
    if not visible:
        visible = ",".join(str(i) for i in range(args.world_size))

    base_cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--world-size",
        str(args.world_size),
        "--master-addr",
        args.master_addr,
        "--master-port",
        str(args.master_port),
        "--backend",
        args.backend,
        "--attn-size",
        str(args.attn_size),
        "--ffn-size",
        str(args.ffn_size),
        "--timeout-sec",
        str(args.timeout_sec),
        "--repeats",
        str(args.repeats),
        "--groups",
        args.groups,
    ]
    if args.skip_ep_groups:
        base_cmd.append("--skip-ep-groups")

    children: list[subprocess.Popen] = []

    def _terminate_children(signum: int | None = None, _frame=None) -> None:
        for child in children:
            if child.poll() is None:
                child.terminate()
        deadline = time.time() + 5
        for child in children:
            while child.poll() is None and time.time() < deadline:
                time.sleep(0.1)
            if child.poll() is None:
                child.kill()
        if signum is not None:
            raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _terminate_children)
    signal.signal(signal.SIGINT, _terminate_children)

    for rank in range(args.world_size):
        env = os.environ.copy()
        env.setdefault("HCCL_CONNECT_TIMEOUT", str(max(120, min(args.timeout_sec, 7200))))
        env.setdefault("HCCL_EXEC_TIMEOUT", str(max(120, min(args.timeout_sec, 7200))))
        env.setdefault("AFD_DIST_TIMEOUT_SEC", str(args.timeout_sec))
        env["MASTER_ADDR"] = args.master_addr
        env["MASTER_PORT"] = str(args.master_port)
        env["RANK"] = str(rank)
        env["WORLD_SIZE"] = str(args.world_size)
        env["LOCAL_RANK"] = str(rank)
        env["ASCEND_VISIBLE_DEVICES"] = visible
        env["ASCEND_RT_VISIBLE_DEVICES"] = visible
        cmd = base_cmd + ["--rank", str(rank), "--local-rank", str(rank)]
        children.append(subprocess.Popen(cmd, env=env))

    rc = 0
    remaining = set(children)
    while remaining:
        for child in list(remaining):
            child_rc = child.poll()
            if child_rc is None:
                continue
            remaining.remove(child)
            if child_rc != 0:
                rc = child_rc
                _terminate_children()
                return rc
        time.sleep(0.2)
    return rc


def _rank_main(args: argparse.Namespace) -> int:
    rank = args.rank if args.rank is not None else int(os.environ["RANK"])
    world_size = args.world_size
    local_rank = args.local_rank if args.local_rank is not None else int(os.environ["LOCAL_RANK"])
    ffn_ranks = list(range(args.attn_size, args.attn_size + args.ffn_size))
    timeout = timedelta(seconds=args.timeout_sec)

    if args.attn_size + args.ffn_size > world_size:
        raise ValueError(
            f"attn_size + ffn_size exceeds world_size: "
            f"{args.attn_size} + {args.ffn_size} > {world_size}"
        )

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    import torch
    import torch.distributed as dist

    if args.backend == "hccl":
        import torch_npu  # noqa: F401

        torch.npu.set_device(local_rank)
        device_desc = f"npu:{local_rank}"
    else:
        device_desc = "cpu"

    _log(
        rank,
        "env "
        f"local_rank={local_rank} device={device_desc} world={world_size} "
        f"master={args.master_addr}:{args.master_port} backend={args.backend} "
        f"HCCL_IF_BASE_PORT={os.environ.get('HCCL_IF_BASE_PORT')} "
        f"HCCL_IF_IP={os.environ.get('HCCL_IF_IP')} "
        f"ASCEND_VISIBLE_DEVICES={os.environ.get('ASCEND_VISIBLE_DEVICES')}",
    )

    _stage(
        rank,
        "init_process_group",
        lambda: dist.init_process_group(
            backend=args.backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        ),
    )
    _stage(rank, "world_barrier_after_init", dist.barrier)

    groups = {}
    if not args.skip_ep_groups:
        for name in _selected_groups(args.groups):
            _stage(
                rank,
                f"new_group_{name}_ranks_{ffn_ranks}",
                lambda ranks=ffn_ranks: groups.setdefault(
                    name, dist.new_group(ranks=ranks, timeout=timeout)
                ),
            )
        _stage(rank, "world_barrier_after_group_creation", dist.barrier)
    else:
        _log(rank, "skip EP groups by request")

    for repeat in range(args.repeats):
        _stage(rank, f"world_barrier_repeat_{repeat}", dist.barrier)
        for name, group in groups.items():
            if rank in ffn_ranks:
                _stage(rank, f"{name}_group_barrier_repeat_{repeat}", lambda g=group: dist.barrier(group=g))
            else:
                _log(rank, f"skip {name}_group_barrier_repeat_{repeat}: non-FFN rank")

    _stage(rank, "final_world_barrier", dist.barrier)
    dist.destroy_process_group()
    _log(rank, "completed successfully")
    return 0


def main() -> int:
    args = _parse_args()
    if args.spawn_local:
        return _spawn_local(args)
    try:
        return _rank_main(args)
    except Exception:
        rank = args.rank if args.rank is not None else int(os.environ.get("RANK", "-1"))
        _log(rank, "FAILED")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
