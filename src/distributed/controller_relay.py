"""CPU controller relay for intentionally centralized A/F communication.

This module implements a pessimistic baseline transport:

    Attention/FFN NPU tensor -> CPU bytes -> controller -> CPU bytes -> peer NPU

It is deliberately blocking and store-and-forward. The goal is not performance,
but a clean baseline for comparing against the current direct HCCL device-tensor
transport.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_LEN = struct.Struct("!Q")


def _recvall(sock: socket.socket, nbytes: int) -> bytes:
    chunks = []
    remaining = nbytes
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed while reading")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_frame(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    header = dict(header)
    header["payload_nbytes"] = len(payload)
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    sock.sendall(_LEN.pack(len(raw_header)))
    sock.sendall(raw_header)
    if payload:
        sock.sendall(payload)


def _recv_frame(sock: socket.socket) -> Tuple[dict, bytes]:
    (header_len,) = _LEN.unpack(_recvall(sock, _LEN.size))
    header = json.loads(_recvall(sock, header_len).decode("utf-8"))
    payload_len = int(header.get("payload_nbytes", 0))
    payload = _recvall(sock, payload_len) if payload_len else b""
    return header, payload


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float32:
        return "float32"
    if dtype is torch.int64:
        return "int64"
    if dtype is torch.int32:
        return "int32"
    raise TypeError(f"unsupported relay dtype: {dtype}")


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "int64": torch.int64,
        "int32": torch.int32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise TypeError(f"unsupported relay dtype: {name}") from exc


def _tensor_to_bytes(tensor: torch.Tensor) -> Tuple[dict, bytes]:
    cpu = tensor.detach().to("cpu", non_blocking=False).contiguous()
    byte_view = cpu.view(torch.uint8)
    payload = byte_view.numpy().tobytes()
    header = {
        "shape": list(cpu.shape),
        "dtype": _dtype_name(cpu.dtype),
    }
    return header, payload


def _bytes_to_tensor(payload: bytes, shape: Iterable[int], dtype_name: str) -> torch.Tensor:
    dtype = _dtype_from_name(dtype_name)
    # bytearray makes the buffer writable and avoids torch.frombuffer warnings.
    byte_tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
    return byte_tensor.view(dtype).reshape(tuple(int(x) for x in shape)).contiguous()


@dataclass
class RelayRecord:
    direction: str
    layer: int
    mb: int
    payload_nbytes: int
    recv_ms: float
    forward_ms: float
    total_ms: float

    def to_csv_row(self) -> str:
        return (
            f"{self.direction},{self.layer},{self.mb},{self.payload_nbytes},"
            f"{self.recv_ms:.6f},{self.forward_ms:.6f},{self.total_ms:.6f}\n"
        )


class ControllerRelayServer:
    """Blocking two-client CPU relay server."""

    def __init__(self, host: str, port: int, output: Optional[str] = None):
        self.host = host
        self.port = port
        self.output = output
        self._clients: Dict[str, socket.socket] = {}
        self._relay_lock = threading.Lock()
        self._records: list[RelayRecord] = []
        self._stop = threading.Event()

    def serve(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(2)
            logger.info("controller relay listening on %s:%s", self.host, self.port)
            while len(self._clients) < 2:
                conn, addr = srv.accept()
                header, _ = _recv_frame(conn)
                role = str(header.get("role", ""))
                if role not in {"attention", "ffn"}:
                    conn.close()
                    raise RuntimeError(f"unexpected controller client role {role!r} from {addr}")
                self._clients[role] = conn
                logger.info("controller client connected: role=%s addr=%s", role, addr)

            for sock in self._clients.values():
                _send_frame(sock, {"type": "ready"})

            threads = [
                threading.Thread(target=self._serve_source, args=("attention",), daemon=True),
                threading.Thread(target=self._serve_source, args=("ffn",), daemon=True),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self._write_records()

    def _serve_source(self, source: str) -> None:
        src_sock = self._clients[source]
        try:
            while not self._stop.is_set():
                msg_start = time.perf_counter()
                header, payload = _recv_frame(src_sock)
                recv_done = time.perf_counter()
                if header.get("type") == "close":
                    self._stop.set()
                    break
                direction = str(header.get("direction"))
                target = "ffn" if direction == "a2f" else "attention"
                dst_sock = self._clients[target]
                with self._relay_lock:
                    fwd_start = time.perf_counter()
                    _send_frame(dst_sock, header, payload)
                    fwd_done = time.perf_counter()
                    _send_frame(src_sock, {"type": "ack", "direction": direction,
                                           "layer": header.get("layer"), "mb": header.get("mb")})
                self._records.append(
                    RelayRecord(
                        direction=direction,
                        layer=int(header.get("layer", -1)),
                        mb=int(header.get("mb", -1)),
                        payload_nbytes=len(payload),
                        recv_ms=(recv_done - msg_start) * 1000,
                        forward_ms=(fwd_done - fwd_start) * 1000,
                        total_ms=(fwd_done - msg_start) * 1000,
                    )
                )
        except EOFError:
            self._stop.set()
        finally:
            try:
                src_sock.close()
            except OSError:
                pass

    def _write_records(self) -> None:
        if not self.output:
            return
        path = Path(self.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.write("direction,layer,mb,payload_nbytes,recv_ms,forward_ms,total_ms\n")
            for rec in self._records:
                f.write(rec.to_csv_row())


class ControllerRelayClient:
    """Blocking client for Attention or FFN coordinator."""

    def __init__(self, role: str, host: str, port: int, timeout_s: float = 300.0):
        if role not in {"attention", "ffn"}:
            raise ValueError(f"unsupported controller relay role: {role}")
        self.role = role
        self.host = host
        self.port = int(port)
        self.sock = socket.create_connection((host, self.port), timeout=timeout_s)
        self.sock.settimeout(None)
        _send_frame(self.sock, {"type": "hello", "role": role})
        header, _ = _recv_frame(self.sock)
        if header.get("type") != "ready":
            raise RuntimeError(f"controller did not send ready: {header}")

    def send_tensor(self, direction: str, layer: int, mb: int, tensor: torch.Tensor) -> None:
        tensor_header, payload = _tensor_to_bytes(tensor)
        header = {
            "type": "tensor",
            "role": self.role,
            "direction": direction,
            "layer": int(layer),
            "mb": int(mb),
            **tensor_header,
        }
        _send_frame(self.sock, header, payload)
        ack, _ = _recv_frame(self.sock)
        if ack.get("type") != "ack":
            raise RuntimeError(f"controller relay expected ack, got {ack}")

    def recv_tensor(
        self,
        direction: str,
        layer: int,
        mb: int,
        *,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        header, payload = _recv_frame(self.sock)
        if header.get("type") != "tensor":
            raise RuntimeError(f"controller relay expected tensor, got {header}")
        if header.get("direction") != direction or int(header.get("layer")) != layer or int(header.get("mb")) != mb:
            raise RuntimeError(
                "controller relay message order mismatch: "
                f"expected {(direction, layer, mb)}, got "
                f"{(header.get('direction'), header.get('layer'), header.get('mb'))}"
            )
        tensor = _bytes_to_tensor(payload, header["shape"], header["dtype"])
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        return tensor.to(device, non_blocking=False)

    def close(self) -> None:
        try:
            _send_frame(self.sock, {"type": "close", "role": self.role})
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU controller relay for A/F baseline")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    ControllerRelayServer(args.host, args.port, args.output or None).serve()


if __name__ == "__main__":
    main()
