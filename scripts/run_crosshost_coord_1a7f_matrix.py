#!/usr/bin/env python3
"""Run cross-host 1A7F coordinator decode-DBO matrix.

This is a local control-plane orchestrator. It launches short detached scripts
inside the Host1/Host2 containers and then polls their logs, so long-running
model execution is not tied to a fragile nested SSH foreground session.
"""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Remote:
    name: str
    ssh: str
    container: str
    workdir: str
    inner_ssh: str = ""


@dataclass(frozen=True)
class Config:
    batch: int
    seq: int
    tokens: int
    master_port: int
    h1_hccl_port: int
    h2_hccl_port: int
    coord_port: int

    @property
    def tag(self) -> str:
        return f"b{self.batch}_s{self.seq}_t{self.tokens}"


def parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def run_cmd(
    cmd: list[str],
    *,
    input_text: str | None = None,
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if check and proc.returncode != 0:
        rendered = " ".join(shlex.quote(part) for part in cmd)
        raise RuntimeError(f"command failed rc={proc.returncode}: {rendered}\n{proc.stdout}")
    return proc


def remote_command(remote: Remote, command: str) -> list[str]:
    if remote.inner_ssh:
        return shlex.split(remote.ssh) + [f"{remote.inner_ssh} {quote(command)}"]
    return shlex.split(remote.ssh) + [command]


def remote_bash(
    remote: Remote,
    command: str,
    *,
    input_text: str | None = None,
    timeout: int = 120,
    check: bool = True,
) -> str:
    docker_args = ["docker", "exec"]
    if input_text is not None:
        docker_args.append("-i")
    docker_args += [remote.container, "bash", "-lc", command]
    rendered = " ".join(quote(part) for part in docker_args)
    proc = run_cmd(remote_command(remote, rendered), input_text=input_text, timeout=timeout, check=check)
    return proc.stdout


def remote_detached(remote: Remote, script_path: str, *, timeout: int = 120) -> str:
    rendered = " ".join(quote(part) for part in ["docker", "exec", "-d", remote.container, "bash", script_path])
    return run_cmd(remote_command(remote, rendered), timeout=timeout).stdout


def quote(value: str | int) -> str:
    return shlex.quote(str(value))


def write_remote_script(remote: Remote, script_path: str, content: str) -> None:
    remote_bash(
        remote,
        f"cat > {quote(script_path)} && chmod +x {quote(script_path)}",
        input_text=content,
        timeout=120,
    )


def host2_free_gb(host2: Remote) -> float:
    out = remote_bash(
        host2,
        "df -BG --output=avail /workspace | tail -n 1",
        timeout=60,
    ).strip()
    for line in reversed(out.splitlines()):
        match = re.match(r"^\s*(\d+)G?\s*$", line)
        if match:
            return float(match.group(1))
    return 0.0


def stale_processes(remote: Remote, pattern: str) -> str:
    return remote_bash(
        remote,
        f"ps -ef | grep -E {quote(pattern)} | grep -v grep || true",
        timeout=60,
        check=False,
    ).strip()


def read_remote_text(remote: Remote, path: str, *, tail_lines: int | None = None) -> str:
    if tail_lines is None:
        cmd = f"cat {quote(path)} 2>/dev/null || true"
    else:
        cmd = f"tail -n {int(tail_lines)} {quote(path)} 2>/dev/null || true"
    return remote_bash(remote, cmd, timeout=60, check=False)


def grep_remote(remote: Remote, path: str, pattern: str) -> str:
    return remote_bash(
        remote,
        f"grep -E {quote(pattern)} {quote(path)} 2>/dev/null | tail -n 1 || true",
        timeout=60,
        check=False,
    ).strip()


def build_side_script(
    *,
    side: str,
    workdir: str,
    out_dir: str,
    cfg: Config,
    timeout_sec: int,
    model_name: str,
) -> str:
    if side == "host1":
        side_args = [
            "--side host1",
            "--master-addr 192.168.0.125",
            f"--master-port {cfg.master_port}",
            f"--hccl-if-base-port {cfg.h1_hccl_port}",
            "--hccl-if-ip 192.168.0.125",
            f"--coord-bind 0.0.0.0:{cfg.coord_port}",
            f"--coord-addr 127.0.0.1:{cfg.coord_port}",
        ]
        side_out = "host1_side.out"
    elif side == "host2":
        side_args = [
            "--side host2",
            "--master-addr 192.168.0.125",
            f"--master-port {cfg.master_port}",
            f"--hccl-if-base-port {cfg.h2_hccl_port}",
            "--hccl-if-ip 192.168.0.192",
            f"--coord-addr 192.168.0.125:{cfg.coord_port}",
        ]
        side_out = "host2_side.out"
    else:
        raise ValueError(f"unknown side: {side}")

    args = [
        *side_args,
        f"--batch {cfg.batch}",
        f"--seq {cfg.seq}",
        f"--tokens {cfg.tokens}",
        f"--timeout-sec {timeout_sec}",
        f"--out-dir {quote(out_dir)}",
        f"--timing-suffix xhost_coord_matrix_{cfg.tag}",
        f"--model-name {quote(model_name)}",
    ]
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {quote(workdir)}
mkdir -p {quote(out_dir)}
: > {quote(out_dir)}/{side_out}
echo "matrix_side_start side={side} tag={cfg.tag} ts=$(date -Is)" >> {quote(out_dir)}/{side_out}
exec bash scripts/run_crosshost_coord_1a7f_smoke.sh \\
  {' '.join(args)} \\
  >> {quote(out_dir)}/{side_out} 2>&1
"""


def parse_rc(line: str, key: str) -> int | None:
    match = re.search(rf"{re.escape(key)}=(\d+)", line)
    return int(match.group(1)) if match else None


def parse_tpot(line: str) -> str:
    match = re.search(r"decode_tpot=([0-9.]+)ms", line)
    return match.group(1) if match else ""


def run_one(
    *,
    host1: Remote,
    host2: Remote,
    cfg: Config,
    out_root: str,
    timeout_sec: int,
    poll_sec: int,
    startup_check_sec: int,
    model_name: str,
    min_host2_free_gb: float,
    dry_run: bool,
) -> dict[str, str]:
    out_dir = f"{out_root}/{cfg.tag}"
    h1_side = f"{host1.workdir}/{out_dir}/host1_side.out"
    h2_side = f"{host2.workdir}/{out_dir}/host2_side.out"
    h1_rank = f"{host1.workdir}/{out_dir}/h1_rank0.log"
    h2_rank1 = f"{host2.workdir}/{out_dir}/h2_rank1.log"
    h2_rank7 = f"{host2.workdir}/{out_dir}/h2_rank7.log"
    status = "OK"
    detail = ""
    host2_free_before = host2_free_gb(host2)
    print(f"=== {cfg.tag} free_gb_before={host2_free_before:.1f} ===", flush=True)

    if host2_free_before < min_host2_free_gb:
        return {
            "batch": str(cfg.batch),
            "seq": str(cfg.seq),
            "tokens": str(cfg.tokens),
            "status": "SKIP_LOW_SPACE",
            "detail": f"host2_free_gb={host2_free_before:.1f}",
            "decode_tpot_ms": "",
            "host2_free_gb_before": f"{host2_free_before:.1f}",
            "host2_free_gb_after": f"{host2_free_before:.1f}",
            "h1_rank0_log": h1_rank,
            "h2_rank1_log": h2_rank1,
            "h2_rank7_log": h2_rank7,
        }

    h1_stale = stale_processes(host1, "src.main|coordinator_server|run_crosshost_coord")
    h2_stale = stale_processes(host2, "src.main")
    if h1_stale or h2_stale:
        return {
            "batch": str(cfg.batch),
            "seq": str(cfg.seq),
            "tokens": str(cfg.tokens),
            "status": "SKIP_STALE_PROCESS",
            "detail": f"h1={bool(h1_stale)} h2={bool(h2_stale)}",
            "decode_tpot_ms": "",
            "host2_free_gb_before": f"{host2_free_before:.1f}",
            "host2_free_gb_after": f"{host2_free_before:.1f}",
            "h1_rank0_log": h1_rank,
            "h2_rank1_log": h2_rank1,
            "h2_rank7_log": h2_rank7,
        }

    h1_script = f"/tmp/afd_xhost_matrix_h1_{cfg.tag}.sh"
    h2_script = f"/tmp/afd_xhost_matrix_h2_{cfg.tag}.sh"
    h1_content = build_side_script(
        side="host1",
        workdir=host1.workdir,
        out_dir=out_dir,
        cfg=cfg,
        timeout_sec=timeout_sec,
        model_name=model_name,
    )
    h2_content = build_side_script(
        side="host2",
        workdir=host2.workdir,
        out_dir=out_dir,
        cfg=cfg,
        timeout_sec=timeout_sec,
        model_name=model_name,
    )
    if dry_run:
        print(h1_content)
        print(h2_content)
        return {
            "batch": str(cfg.batch),
            "seq": str(cfg.seq),
            "tokens": str(cfg.tokens),
            "status": "DRY_RUN",
            "detail": "",
            "decode_tpot_ms": "",
            "host2_free_gb_before": f"{host2_free_before:.1f}",
            "host2_free_gb_after": f"{host2_free_before:.1f}",
            "h1_rank0_log": h1_rank,
            "h2_rank1_log": h2_rank1,
            "h2_rank7_log": h2_rank7,
        }

    write_remote_script(host1, h1_script, h1_content)
    write_remote_script(host2, h2_script, h2_content)
    remote_detached(host1, h1_script)
    time.sleep(8)
    remote_detached(host2, h2_script)

    startup_deadline = time.time() + startup_check_sec
    startup_ok = False
    while time.time() < startup_deadline:
        h1_started = grep_remote(host1, h1_side, r"coordinator_pid=|Initializing distributed")
        h2_started = grep_remote(host2, h2_side, r"rank=1 pid=|Initializing distributed")
        if h1_started and h2_started:
            startup_ok = True
            break
        time.sleep(5)
    if not startup_ok:
        status = "ORCHESTRATION_FAIL"
        detail = "rank logs/PIDs not visible within startup window"
    else:
        deadline = time.time() + timeout_sec + 300
        while time.time() < deadline:
            h1_done = grep_remote(host1, h1_side, r"host1_attention_rc=|Traceback|RuntimeError|ERROR|TIMEOUT")
            h2_done = grep_remote(host2, h2_side, r"host2_ffn_rc=|Traceback|RuntimeError|ERROR|TIMEOUT")
            h1_rc = parse_rc(h1_done, "host1_attention_rc")
            h2_rc = parse_rc(h2_done, "host2_ffn_rc")
            if h1_rc is not None and h2_rc is not None:
                if h1_rc == 0 and h2_rc == 0:
                    status = "OK"
                    detail = ""
                else:
                    status = "FAIL"
                    detail = f"h1_rc={h1_rc} h2_rc={h2_rc}"
                break
            if ("Traceback" in h1_done or "RuntimeError" in h1_done or "ERROR" in h1_done) or (
                "Traceback" in h2_done or "RuntimeError" in h2_done or "ERROR" in h2_done
            ):
                status = "FAIL"
                detail = f"h1={h1_done[-120:]} h2={h2_done[-120:]}"
                break
            time.sleep(poll_sec)
        else:
            status = "TIMEOUT_WAIT"
            detail = "side completion markers not observed"

    tpot = ""
    timing_line = grep_remote(host1, h1_rank, r"Generation timing:")
    if timing_line:
        tpot = parse_tpot(timing_line)
    if status == "OK" and not tpot:
        status = "FAIL_MISSING_TIMING"
        detail = "side rc markers observed but Host1 decode_tpot was not found"

    host2_free_after = host2_free_gb(host2)
    print(f"{cfg.tag}: {status} tpot={tpot} free_gb_after={host2_free_after:.1f} {detail}", flush=True)
    return {
        "batch": str(cfg.batch),
        "seq": str(cfg.seq),
        "tokens": str(cfg.tokens),
        "status": status,
        "detail": detail,
        "decode_tpot_ms": tpot,
        "host2_free_gb_before": f"{host2_free_before:.1f}",
        "host2_free_gb_after": f"{host2_free_after:.1f}",
        "h1_rank0_log": h1_rank,
        "h2_rank1_log": h2_rank1,
        "h2_rank7_log": h2_rank7,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host1-ssh", default="ssh 910C")
    parser.add_argument("--host2-gateway-ssh", default="ssh 910C")
    parser.add_argument("--host2-inner-ssh", default="sudo ssh -i /root/ssh_key/KeyPair-f1dd.pem root@192.168.0.192")
    parser.add_argument("--host1-container", default="afd-npu-test")
    parser.add_argument("--host2-container", default="afd-npu-test-h2")
    parser.add_argument("--host1-workdir", default="/workspace/afd_demo_exp_1a7f")
    parser.add_argument("--host2-workdir", default="/workspace/afd_demo_repo_exp_1a7f")
    parser.add_argument("--out-root", default="results_npu/coordinator_arch/crosshost_1a7f_decode_dbo")
    parser.add_argument("--model-name", default="/models/Qwen3-30B-A3B")
    parser.add_argument("--batches", default="2,4,8,16,32,64,128,256")
    parser.add_argument("--seqs", default="128,256,512")
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--poll-sec", type=int, default=30)
    parser.add_argument("--startup-check-sec", type=int, default=90)
    parser.add_argument("--min-host2-free-gb", type=float, default=5.0)
    parser.add_argument("--base-master-port", type=int, default=35500)
    parser.add_argument("--base-h1-hccl-port", type=int, default=40000)
    parser.add_argument("--base-h2-hccl-port", type=int, default=41000)
    parser.add_argument("--base-coord-port", type=int, default=50300)
    parser.add_argument("--summary-csv", default="results_npu/coordinator_arch/crosshost_1a7f_decode_dbo/matrix_summary.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-larger-batches-on-fail", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    host1 = Remote("host1", args.host1_ssh, args.host1_container, args.host1_workdir)
    host2 = Remote("host2", args.host2_gateway_ssh, args.host2_container, args.host2_workdir, args.host2_inner_ssh)
    batches = parse_csv_ints(args.batches)
    seqs = parse_csv_ints(args.seqs)
    summary_path = ROOT / args.summary_csv
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "batch",
        "seq",
        "tokens",
        "status",
        "detail",
        "decode_tpot_ms",
        "host2_free_gb_before",
        "host2_free_gb_after",
        "h1_rank0_log",
        "h2_rank1_log",
        "h2_rank7_log",
    ]

    rows: list[dict[str, str]] = []
    idx = 0
    for seq in seqs:
        stop_seq = False
        for batch in batches:
            if stop_seq:
                rows.append(
                    {
                        "batch": str(batch),
                        "seq": str(seq),
                        "tokens": str(args.tokens),
                        "status": "SKIP_AFTER_FAIL",
                        "detail": "larger batch skipped after failure in same seq",
                        "decode_tpot_ms": "",
                        "host2_free_gb_before": "",
                        "host2_free_gb_after": "",
                        "h1_rank0_log": "",
                        "h2_rank1_log": "",
                        "h2_rank7_log": "",
                    }
                )
                continue
            idx += 1
            cfg = Config(
                batch=batch,
                seq=seq,
                tokens=args.tokens,
                master_port=args.base_master_port + idx,
                h1_hccl_port=args.base_h1_hccl_port + idx * 20,
                h2_hccl_port=args.base_h2_hccl_port + idx * 20,
                coord_port=args.base_coord_port + idx,
            )
            row = run_one(
                host1=host1,
                host2=host2,
                cfg=cfg,
                out_root=args.out_root,
                timeout_sec=args.timeout_sec,
                poll_sec=args.poll_sec,
                startup_check_sec=args.startup_check_sec,
                model_name=args.model_name,
                min_host2_free_gb=args.min_host2_free_gb,
                dry_run=args.dry_run,
            )
            rows.append(row)
            with summary_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            if args.stop_larger_batches_on_fail and row["status"] not in {"OK", "DRY_RUN"}:
                stop_seq = True

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary_path}")
    failures = [row for row in rows if row["status"] not in {"OK", "DRY_RUN", "SKIP_AFTER_FAIL"}]
    return 1 if failures and not args.dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
