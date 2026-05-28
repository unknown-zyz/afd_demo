#!/usr/bin/env python3
"""Run cross-host static EP decode-DBO sweeps.

The launcher is intentionally parameterized: pass ``--ep-sizes`` and
``--host2-ffn-devices`` to manually choose how many FFN EP ranks run on Host2.
Host1 always runs only rank0 (Attention).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOST1_SSH = (
    "ssh -o BatchMode=yes -o ConnectTimeout=20 "
    "-o ServerAliveInterval=15 -o ServerAliveCountMax=3 "
    "-p 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229"
)
DEFAULT_HOST2_INNER = (
    "sudo ssh -o StrictHostKeyChecking=no "
    "-i /root/ssh_key/KeyPair-f1dd.pem root@192.168.0.192"
)


@dataclass(frozen=True)
class Remote:
    name: str
    ssh: str
    container: str
    workdir: str
    inner_ssh: str = ""


@dataclass(frozen=True)
class RunConfig:
    ep_size: int
    backend: str
    mode: str
    num_micro_batches: int
    batch: int
    seq: int
    tokens: int
    master_port: int
    h1_hccl_port: int
    h2_hccl_port: int

    @property
    def tag(self) -> str:
        mb_tag = f"_mb{self.num_micro_batches}" if self.num_micro_batches != 2 else ""
        return (
            f"{self.mode}_ep{self.ep_size}_{self.backend}{mb_tag}"
            f"_b{self.batch}_s{self.seq}_t{self.tokens}"
        )


def quote(value: str | int) -> str:
    return shlex.quote(str(value))


def parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_configs(value: str) -> list[tuple[int, int]]:
    configs: list[tuple[int, int]] = []
    for item in parse_csv_strings(value):
        match = re.fullmatch(r"b?(\d+)(?::|x|X|s|_s)(\d+)", item)
        if not match:
            raise ValueError(f"invalid config {item!r}; expected B:S, BxS, BsS, or bB_sS")
        configs.append((int(match.group(1)), int(match.group(2))))
    return configs


OOM_RE = re.compile(
    r"out of memory|OOM|ACL_ERROR_RT_MEMORY_ALLOCATION|memory allocation|MemoryError",
    re.IGNORECASE,
)
SSH_FAILURE_RE = re.compile(
    r"kex_exchange_identification|Connection (?:closed|reset)|ssh_exchange_identification",
    re.IGNORECASE,
)


def is_oom_text(text: str) -> bool:
    return bool(OOM_RE.search(text))


def classify_oom_side(h1_text: str, h2_text: str) -> str:
    h1_oom = is_oom_text(h1_text)
    h2_oom = is_oom_text(h2_text)
    if h1_oom and h2_oom:
        return "both_or_peer"
    if h1_oom:
        return "attention"
    if h2_oom:
        return "ffn"
    return ""


def is_ssh_failure_text(text: str) -> bool:
    return bool(SSH_FAILURE_RE.search(text))


def serial_baseline_tag(cfg: RunConfig) -> str:
    serial_cfg = RunConfig(
        ep_size=cfg.ep_size,
        backend=cfg.backend,
        mode="serial",
        num_micro_batches=2,
        batch=cfg.batch,
        seq=cfg.seq,
        tokens=cfg.tokens,
        master_port=cfg.master_port,
        h1_hccl_port=cfg.h1_hccl_port,
        h2_hccl_port=cfg.h2_hccl_port,
    )
    return serial_cfg.tag


def run_cmd(
    cmd: list[str],
    *,
    input_text: str | None = None,
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    attempts = 3
    proc: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, attempts + 1):
        proc = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 255 or attempt == attempts:
            break
        time.sleep(5 * attempt)
    assert proc is not None
    if check and proc.returncode != 0:
        rendered = " ".join(quote(part) for part in cmd)
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
    return strip_ssh_warnings(proc.stdout)


def strip_ssh_warnings(output: str) -> str:
    lines = []
    for line in output.splitlines():
        if line.startswith("** WARNING:"):
            continue
        if line.startswith('** This session may be vulnerable to "store now'):
            continue
        if line.startswith("** The server may need to be upgraded."):
            continue
        lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def remote_detached(remote: Remote, script_path: str, *, timeout: int = 120) -> str:
    rendered = " ".join(quote(part) for part in ["docker", "exec", "-d", remote.container, "bash", script_path])
    return run_cmd(remote_command(remote, rendered), timeout=timeout).stdout


def write_remote_script(remote: Remote, script_path: str, content: str) -> None:
    remote_bash(
        remote,
        f"cat > {quote(script_path)} && chmod +x {quote(script_path)}",
        input_text=content,
        timeout=120,
    )


def read_remote_text(remote: Remote, path: str, *, tail_lines: int | None = None) -> str:
    if tail_lines is None:
        cmd = f"cat {quote(path)} 2>/dev/null || true"
    else:
        cmd = f"tail -n {int(tail_lines)} {quote(path)} 2>/dev/null || true"
    return remote_bash(remote, cmd, timeout=90, check=False)


def grep_remote(remote: Remote, path: str, pattern: str) -> str:
    out = remote_bash(
        remote,
        f"grep -E {quote(pattern)} {quote(path)} 2>/dev/null | tail -n 1 || true",
        timeout=60,
        check=False,
    ).strip()
    return "" if is_ssh_failure_text(out) else out


def stale_processes(remote: Remote, pattern: str) -> str:
    out = remote_bash(
        remote,
        f"ps -ef | grep -E {quote(pattern)} | grep -v grep || true",
        timeout=60,
        check=False,
    ).strip()
    return "" if is_ssh_failure_text(out) else out


def host2_hbm_summary(host2: Remote) -> str:
    return remote_bash(
        host2,
        "npu-smi info | grep -E '/ 65536' || true",
        timeout=90,
        check=False,
    ).strip()


def build_side_script(
    *,
    side: str,
    workdir: str,
    out_dir: str,
    cfg: RunConfig,
    timeout_sec: int,
    model_name: str,
    host2_ffn_devices: str,
    debug_max_layers: int | None,
    attn_kernel: str,
    attn_precopy_layer_inputs: bool,
    attn_fused_rmsnorm: bool,
    attn_fused_rope: bool,
    attn_stream_overlap: bool,
    resource_monitor: bool,
    resource_monitor_interval: int,
) -> str:
    if side == "host1":
        side_args = [
            "--side host1",
            "--master-addr 192.168.0.125",
            f"--master-port {cfg.master_port}",
            f"--hccl-if-base-port {cfg.h1_hccl_port}",
            "--hccl-if-ip 192.168.0.125",
        ]
        side_out = "host1_side.out"
    elif side == "host2":
        side_args = [
            "--side host2",
            "--master-addr 192.168.0.125",
            f"--master-port {cfg.master_port}",
            f"--hccl-if-base-port {cfg.h2_hccl_port}",
            "--hccl-if-ip 192.168.0.192",
            f"--host2-ffn-devices {quote(host2_ffn_devices)}",
        ]
        side_out = "host2_side.out"
    else:
        raise ValueError(f"unknown side: {side}")

    args = [
        *side_args,
        f"--ffn-ep-size {cfg.ep_size}",
        f"--ffn-ep-backend {quote(cfg.backend)}",
        f"--mode {quote(cfg.mode)}",
        f"--batch {cfg.batch}",
        f"--seq {cfg.seq}",
        f"--tokens {cfg.tokens}",
        f"--num-micro-batches {cfg.num_micro_batches}",
        "--comm-timing-mode completion",
        f"--timeout-sec {timeout_sec}",
        f"--out-dir {quote(out_dir)}",
        f"--timing-suffix xhost_static_{cfg.tag}",
        f"--model-name {quote(model_name)}",
        f"--attn-kernel {quote(attn_kernel)}",
    ]
    if attn_precopy_layer_inputs:
        args.append("--attn-precopy-layer-inputs")
    if attn_fused_rmsnorm:
        args.append("--attn-fused-rmsnorm")
    if attn_fused_rope:
        args.append("--attn-fused-rope")
    if attn_stream_overlap:
        args.append("--attn-stream-overlap")
    if resource_monitor:
        args.append("--resource-monitor")
        args.append(f"--resource-monitor-interval {resource_monitor_interval}")
    if debug_max_layers is not None:
        args.append(f"--debug-max-layers {debug_max_layers}")
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {quote(workdir)}
mkdir -p {quote(out_dir)}
: > {quote(out_dir)}/{side_out}
echo "matrix_side_start side={side} tag={cfg.tag} ts=$(date -Is)" >> {quote(out_dir)}/{side_out}
exec bash scripts/run_crosshost_static_ep_smoke.sh \\
  {' '.join(args)} \\
  >> {quote(out_dir)}/{side_out} 2>&1
"""


def parse_rc(line: str, key: str) -> int | None:
    match = re.search(rf"{re.escape(key)}=(\d+)", line)
    return int(match.group(1)) if match else None


def parse_tpot(line: str) -> str:
    match = re.search(r"decode_tpot=([0-9.]+)ms", line)
    return match.group(1) if match else ""


def fetch_file(remote: Remote, remote_path: str, local_path: Path) -> bool:
    for attempt in range(1, 4):
        text = read_remote_text(remote, remote_path)
        if not text.strip() or is_ssh_failure_text(text):
            if attempt < 3:
                time.sleep(5 * attempt)
                continue
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(text)
        return True
    return False


def is_valid_json(path: Path) -> bool:
    try:
        with path.open() as f:
            json.load(f)
        return True
    except Exception:
        return False


def generate_local_artifacts(local_out: Path, cfg: RunConfig) -> tuple[str, str]:
    suffix = f"xhost_static_{cfg.tag}"
    attn = local_out / f"timing_attention_{suffix}.json"
    ffn = local_out / f"timing_ffn_coordinator_{suffix}.json"
    report = local_out / f"report_{suffix}.md"
    pipeline = local_out / f"pipeline_{suffix}.png"
    if not attn.exists() or not ffn.exists() or not is_valid_json(attn) or not is_valid_json(ffn):
        return "", ""
    report_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "gen_experiment_report.py"),
            "--attn-timing",
            str(attn),
            "--ffn-timing",
            str(ffn),
            "--output",
            str(report),
            "--mode",
            cfg.mode,
            "--batch",
            str(cfg.batch),
            "--seq",
            str(cfg.seq),
            "--tokens",
            str(cfg.tokens),
            "--comm-timing-mode",
            "completion",
    ]
    serial_attn = (
        local_out.parent
        / serial_baseline_tag(cfg)
        / f"timing_attention_xhost_static_{serial_baseline_tag(cfg)}.json"
    )
    if cfg.mode != "serial" and serial_attn.exists():
        report_cmd += ["--serial-baseline", str(serial_attn)]
    subprocess.run(
        report_cmd,
        cwd=ROOT,
        check=False,
    )
    if cfg.mode == "serial":
        return (str(report) if report.exists() else "", "")
    plot_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "visualize_dbo_pipeline.py"),
            "--attn-timing",
            str(attn),
            "--ffn-timing",
            str(ffn),
            "--output",
            str(pipeline),
            "--start-layer",
            "0",
            "--num-layers",
            "3",
            "--ffn-view",
            "fourlane",
    ]
    if serial_attn.exists():
        plot_cmd += ["--serial-timing", str(serial_attn)]
    subprocess.run(
        plot_cmd,
        cwd=ROOT,
        check=False,
    )
    return (str(report) if report.exists() else "", str(pipeline) if pipeline.exists() else "")


def run_one(
    *,
    host1: Remote,
    host2: Remote,
    cfg: RunConfig,
    out_root: str,
    timeout_sec: int,
    poll_sec: int,
    startup_check_sec: int,
    model_name: str,
    host2_ffn_devices: str,
    debug_max_layers: int | None,
    attn_kernel: str,
    attn_precopy_layer_inputs: bool,
    attn_fused_rmsnorm: bool,
    attn_fused_rope: bool,
    attn_stream_overlap: bool,
    resource_monitor: bool,
    resource_monitor_interval: int,
    dry_run: bool,
) -> dict[str, str]:
    out_dir = f"{out_root}/{cfg.tag}"
    suffix = f"xhost_static_{cfg.tag}"
    h1_side = f"{host1.workdir}/{out_dir}/host1_side.out"
    h2_side = f"{host2.workdir}/{out_dir}/host2_side.out"
    h1_rank = f"{host1.workdir}/{out_dir}/h1_rank0.log"
    h2_rank1 = f"{host2.workdir}/{out_dir}/h2_rank1.log"
    h2_rank_last = f"{host2.workdir}/{out_dir}/h2_rank{cfg.ep_size}.log"
    local_out = ROOT / out_dir
    status = "OK"
    detail = ""
    tpot = ""
    report = ""
    pipeline = ""
    oom_side = ""
    resource_h1 = ""
    resource_h2 = ""

    if dry_run:
        h1_content = build_side_script(
            side="host1",
            workdir=host1.workdir,
            out_dir=out_dir,
            cfg=cfg,
            timeout_sec=timeout_sec,
            model_name=model_name,
            host2_ffn_devices=host2_ffn_devices,
            debug_max_layers=debug_max_layers,
            attn_kernel=attn_kernel,
            attn_precopy_layer_inputs=attn_precopy_layer_inputs,
            attn_fused_rmsnorm=attn_fused_rmsnorm,
            attn_fused_rope=attn_fused_rope,
            attn_stream_overlap=attn_stream_overlap,
            resource_monitor=resource_monitor,
            resource_monitor_interval=resource_monitor_interval,
        )
        h2_content = build_side_script(
            side="host2",
            workdir=host2.workdir,
            out_dir=out_dir,
            cfg=cfg,
            timeout_sec=timeout_sec,
            model_name=model_name,
            host2_ffn_devices=host2_ffn_devices,
            debug_max_layers=debug_max_layers,
            attn_kernel=attn_kernel,
            attn_precopy_layer_inputs=attn_precopy_layer_inputs,
            attn_fused_rmsnorm=attn_fused_rmsnorm,
            attn_fused_rope=attn_fused_rope,
            attn_stream_overlap=attn_stream_overlap,
            resource_monitor=resource_monitor,
            resource_monitor_interval=resource_monitor_interval,
        )
        print(f"=== DRY RUN {cfg.tag} ===")
        print("--- Host2 script ---")
        print(h2_content)
        print("--- Host1 script ---")
        print(h1_content)
        status = "DRY_RUN"
    else:
        hbm_before = host2_hbm_summary(host2)
        print(f"=== {cfg.tag} ===", flush=True)
        print(hbm_before, flush=True)
        h1_stale = stale_processes(host1, "src[.]main|run_crosshost_static_[e]p")
        h2_stale = stale_processes(host2, "src[.]main|run_crosshost_static_[e]p")
        if h1_stale or h2_stale:
            status = "SKIP_STALE_PROCESS"
            detail = f"h1={bool(h1_stale)} h2={bool(h2_stale)}"
        else:
            h1_script = f"/tmp/afd_xhost_static_h1_{cfg.tag}.sh"
            h2_script = f"/tmp/afd_xhost_static_h2_{cfg.tag}.sh"
            write_remote_script(
                host1,
                h1_script,
                build_side_script(
                    side="host1",
                    workdir=host1.workdir,
                    out_dir=out_dir,
                    cfg=cfg,
                    timeout_sec=timeout_sec,
                    model_name=model_name,
                    host2_ffn_devices=host2_ffn_devices,
                    debug_max_layers=debug_max_layers,
                    attn_kernel=attn_kernel,
                    attn_precopy_layer_inputs=attn_precopy_layer_inputs,
                    attn_fused_rmsnorm=attn_fused_rmsnorm,
                    attn_fused_rope=attn_fused_rope,
                    attn_stream_overlap=attn_stream_overlap,
                    resource_monitor=resource_monitor,
                    resource_monitor_interval=resource_monitor_interval,
                ),
            )
            write_remote_script(
                host2,
                h2_script,
                build_side_script(
                    side="host2",
                    workdir=host2.workdir,
                    out_dir=out_dir,
                    cfg=cfg,
                    timeout_sec=timeout_sec,
                    model_name=model_name,
                    host2_ffn_devices=host2_ffn_devices,
                    debug_max_layers=debug_max_layers,
                    attn_kernel=attn_kernel,
                    attn_precopy_layer_inputs=attn_precopy_layer_inputs,
                    attn_fused_rmsnorm=attn_fused_rmsnorm,
                    attn_fused_rope=attn_fused_rope,
                    attn_stream_overlap=attn_stream_overlap,
                    resource_monitor=resource_monitor,
                    resource_monitor_interval=resource_monitor_interval,
                ),
            )
            remote_detached(host2, h2_script)
            time.sleep(8)
            remote_detached(host1, h1_script)

            startup_deadline = time.time() + startup_check_sec
            startup_ok = False
            while time.time() < startup_deadline:
                h1_started = grep_remote(host1, h1_side, r"Initializing distributed|cross-host static EP")
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
                    h1_done = grep_remote(host1, h1_side, r"host1_attention_rc=|Traceback|RuntimeError|ERROR|TIMEOUT|OOM")
                    h2_done = grep_remote(host2, h2_side, r"host2_ffn_rc=|Traceback|RuntimeError|ERROR|TIMEOUT|OOM")
                    h1_rc = parse_rc(h1_done, "host1_attention_rc")
                    h2_rc = parse_rc(h2_done, "host2_ffn_rc")
                    if h1_rc is not None and h2_rc is not None:
                        if h1_rc == 0 and h2_rc == 0:
                            status = "OK"
                        else:
                            h1_tail = read_remote_text(host1, h1_rank, tail_lines=120)
                            h2_tail = (
                                read_remote_text(host2, h2_rank1, tail_lines=80)
                                + "\n"
                                + read_remote_text(host2, h2_rank_last, tail_lines=80)
                            )
                            oom_side = classify_oom_side(h1_tail, h2_tail)
                            status = "OOM" if oom_side else "FAIL"
                            detail = f"h1_rc={h1_rc} h2_rc={h2_rc}"
                            if oom_side:
                                detail += f" oom_side={oom_side}"
                        break
                    if any(term in h1_done + h2_done for term in ("Traceback", "RuntimeError", "ERROR", "OOM")):
                        h1_tail = read_remote_text(host1, h1_rank, tail_lines=120)
                        h2_tail = (
                            read_remote_text(host2, h2_rank1, tail_lines=80)
                            + "\n"
                            + read_remote_text(host2, h2_rank_last, tail_lines=80)
                        )
                        oom_side = classify_oom_side(h1_tail, h2_tail)
                        status = "OOM" if oom_side else "FAIL"
                        detail = f"h1={h1_done[-120:]} h2={h2_done[-120:]}"
                        if oom_side:
                            detail += f" oom_side={oom_side}"
                        break
                    time.sleep(poll_sec)
                else:
                    status = "TIMEOUT_WAIT"
                    detail = "side completion markers not observed"

            timing_line = grep_remote(host1, h1_rank, r"Generation timing:")
            if timing_line:
                tpot = parse_tpot(timing_line)
            if status == "OK" and not tpot:
                status = "FAIL_MISSING_TIMING"
                detail = "side rc markers observed but Host1 decode_tpot was not found"
            if status == "OK":
                fetch_file(
                    host1,
                    f"{host1.workdir}/{out_dir}/timing_attention_{suffix}.json",
                    local_out / f"timing_attention_{suffix}.json",
                )
                fetch_file(
                    host2,
                    f"{host2.workdir}/{out_dir}/timing_ffn_coordinator_{suffix}.json",
                    local_out / f"timing_ffn_coordinator_{suffix}.json",
                )
                report, pipeline = generate_local_artifacts(local_out, cfg)
            if resource_monitor:
                h1_resource = f"{host1.workdir}/{out_dir}/npu_smi_host1.log"
                h2_resource = f"{host2.workdir}/{out_dir}/npu_smi_host2.log"
                if fetch_file(host1, h1_resource, local_out / "npu_smi_host1.log"):
                    resource_h1 = str(local_out / "npu_smi_host1.log")
                if fetch_file(host2, h2_resource, local_out / "npu_smi_host2.log"):
                    resource_h2 = str(local_out / "npu_smi_host2.log")
                fetch_file(
                    host1,
                    f"{host1.workdir}/{out_dir}/npu_smi_host1_start.log",
                    local_out / "npu_smi_host1_start.log",
                )
                fetch_file(
                    host2,
                    f"{host2.workdir}/{out_dir}/npu_smi_host2_start.log",
                    local_out / "npu_smi_host2_start.log",
                )
        print(f"{cfg.tag}: {status} tpot={tpot} {detail}", flush=True)

    return {
        "ep_size": str(cfg.ep_size),
        "backend": cfg.backend,
        "mode": cfg.mode,
        "num_micro_batches": str(cfg.num_micro_batches),
        "batch": str(cfg.batch),
        "seq": str(cfg.seq),
        "tokens": str(cfg.tokens),
        "status": status,
        "detail": detail,
        "decode_tpot_ms": tpot,
        "oom_side": oom_side,
        "report": report,
        "pipeline": pipeline,
        "resource_h1": resource_h1,
        "resource_h2": resource_h2,
        "h1_rank0_log": h1_rank,
        "h2_rank1_log": h2_rank1,
        "h2_rank_last_log": h2_rank_last,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host1-ssh", default=DEFAULT_HOST1_SSH)
    parser.add_argument("--host2-gateway-ssh", default=DEFAULT_HOST1_SSH)
    parser.add_argument("--host2-inner-ssh", default=DEFAULT_HOST2_INNER)
    parser.add_argument("--host1-container", default="afd-npu-test")
    parser.add_argument("--host2-container", default="afd-npu-test-h2")
    parser.add_argument("--host1-workdir", default="/workspace/afd_demo_crosshost_ep")
    parser.add_argument("--host2-workdir", default="/workspace/afd_demo_repo_crosshost_ep")
    parser.add_argument("--out-root", default="crosshost_static_ep16_sweep")
    parser.add_argument("--model-name", default="/models/Qwen3-30B-A3B")
    parser.add_argument("--ep-sizes", default="16")
    parser.add_argument("--host2-ffn-devices", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--backends", default="broadcast_reduce_overlap")
    parser.add_argument("--modes", default="serial,decode-dbo,decode-dbo-crosslayer")
    parser.add_argument("--num-micro-batches", default="2")
    parser.add_argument("--batches", default="2,4,8,16,32,64,128,256")
    parser.add_argument("--seqs", default="128,256,512,1024")
    parser.add_argument(
        "--configs",
        default="",
        help="Optional comma-separated B:S configs. If omitted, uses --batches × --seqs.",
    )
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--poll-sec", type=int, default=30)
    parser.add_argument("--startup-check-sec", type=int, default=120)
    parser.add_argument("--base-master-port", type=int, default=35600)
    parser.add_argument("--base-h1-hccl-port", type=int, default=42100)
    parser.add_argument("--base-h2-hccl-port", type=int, default=43100)
    parser.add_argument("--summary-csv", default="crosshost_static_ep16_sweep/matrix_summary.csv")
    parser.add_argument("--debug-max-layers", type=int, default=None)
    parser.add_argument("--attn-kernel", default="hf", choices=["hf", "npu-official"])
    parser.add_argument("--attn-precopy-layer-inputs", action="store_true")
    parser.add_argument("--attn-fused-rmsnorm", action="store_true")
    parser.add_argument("--attn-fused-rope", action="store_true")
    parser.add_argument("--attn-stream-overlap", action="store_true")
    parser.add_argument("--resource-monitor", action="store_true", default=True)
    parser.add_argument("--no-resource-monitor", action="store_false", dest="resource_monitor")
    parser.add_argument("--resource-monitor-interval", type=int, default=1)
    parser.add_argument("--adaptive-oom", action="store_true", default=True)
    parser.add_argument("--no-adaptive-oom", action="store_false", dest="adaptive_oom")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after-first-large-success", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    host1 = Remote("host1", args.host1_ssh, args.host1_container, args.host1_workdir)
    host2 = Remote("host2", args.host2_gateway_ssh, args.host2_container, args.host2_workdir, args.host2_inner_ssh)
    ep_sizes = parse_csv_ints(args.ep_sizes)
    backends = parse_csv_strings(args.backends)
    modes = parse_csv_strings(args.modes)
    mbs = parse_csv_ints(args.num_micro_batches)
    if args.configs:
        configs = parse_configs(args.configs)
    else:
        batches = parse_csv_ints(args.batches)
        seqs = parse_csv_ints(args.seqs)
        configs = [(batch, seq) for seq in seqs for batch in batches]
    if args.adaptive_oom:
        configs = sorted(configs, key=lambda item: (item[1], item[0]))
    host2_device_count = len(parse_csv_ints(args.host2_ffn_devices))
    for ep_size in ep_sizes:
        if ep_size > host2_device_count:
            raise ValueError(f"ep_size={ep_size} exceeds host2 device list length {host2_device_count}")

    summary_path = ROOT / args.summary_csv
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "ep_size",
        "backend",
        "mode",
        "num_micro_batches",
        "batch",
        "seq",
        "tokens",
        "status",
        "detail",
        "decode_tpot_ms",
        "oom_side",
        "report",
        "pipeline",
        "resource_h1",
        "resource_h2",
        "h1_rank0_log",
        "h2_rank1_log",
        "h2_rank_last_log",
    ]
    rows: list[dict[str, str]] = []
    idx = 0
    oom_stops: set[tuple[int, str, str, int, int]] = set()
    for batch, seq in configs:
        large_success = False
        for ep_size in ep_sizes:
            for backend in backends:
                for mode in modes:
                    for mb in mbs:
                        skip_key = (ep_size, backend, mode, mb, seq)
                        if args.adaptive_oom and skip_key in oom_stops:
                            cfg = RunConfig(
                                ep_size=ep_size,
                                backend=backend,
                                mode=mode,
                                num_micro_batches=mb,
                                batch=batch,
                                seq=seq,
                                tokens=args.tokens,
                                master_port=args.base_master_port + idx + 1,
                                h1_hccl_port=args.base_h1_hccl_port + (idx + 1) * 20,
                                h2_hccl_port=args.base_h2_hccl_port + (idx + 1) * 20,
                            )
                            row = {
                                "ep_size": str(cfg.ep_size),
                                "backend": cfg.backend,
                                "mode": cfg.mode,
                                "num_micro_batches": str(cfg.num_micro_batches),
                                "batch": str(cfg.batch),
                                "seq": str(cfg.seq),
                                "tokens": str(cfg.tokens),
                                "status": "SKIP_AFTER_OOM",
                                "detail": "larger batch skipped after earlier OOM for same ep/backend/mode/mb/seq",
                                "decode_tpot_ms": "",
                                "oom_side": "",
                                "report": "",
                                "pipeline": "",
                                "resource_h1": "",
                                "resource_h2": "",
                                "h1_rank0_log": "",
                                "h2_rank1_log": "",
                                "h2_rank_last_log": "",
                            }
                            rows.append(row)
                            with summary_path.open("w", newline="") as f:
                                writer = csv.DictWriter(f, fieldnames=fields)
                                writer.writeheader()
                                writer.writerows(rows)
                            continue
                        idx += 1
                        cfg = RunConfig(
                            ep_size=ep_size,
                            backend=backend,
                            mode=mode,
                            num_micro_batches=mb,
                            batch=batch,
                            seq=seq,
                            tokens=args.tokens,
                            master_port=args.base_master_port + idx,
                            h1_hccl_port=args.base_h1_hccl_port + idx * 20,
                            h2_hccl_port=args.base_h2_hccl_port + idx * 20,
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
                            host2_ffn_devices=args.host2_ffn_devices,
                            debug_max_layers=args.debug_max_layers,
                            attn_kernel=args.attn_kernel,
                            attn_precopy_layer_inputs=args.attn_precopy_layer_inputs,
                            attn_fused_rmsnorm=args.attn_fused_rmsnorm,
                            attn_fused_rope=args.attn_fused_rope,
                            attn_stream_overlap=args.attn_stream_overlap,
                            resource_monitor=args.resource_monitor,
                            resource_monitor_interval=args.resource_monitor_interval,
                            dry_run=args.dry_run,
                        )
                        rows.append(row)
                        if args.adaptive_oom and row["status"] == "OOM":
                            oom_stops.add(skip_key)
                        with summary_path.open("w", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=fields)
                            writer.writeheader()
                            writer.writerows(rows)
                        if row["status"] == "OK" and batch >= 16 and seq >= 256:
                            large_success = True
        if args.stop_after_first_large_success and large_success:
            break

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary_path}")
    failures = [row for row in rows if row["status"] not in {"OK", "DRY_RUN"}]
    return 1 if failures and not args.dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
