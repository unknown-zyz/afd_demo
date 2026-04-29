"""Shared baseline resolution helpers for experiment reports and plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class BaselineResolution:
    value_ms: float | None
    unit: str | None
    source: str
    warning: str | None = None

    @property
    def available(self) -> bool:
        return self.value_ms is not None and self.value_ms > 0


def normalize_mode(mode: str | None) -> str | None:
    """Normalize report/dir mode names to the comparison class."""
    if not mode:
        return None
    mode_l = mode.lower().replace("_", "-")
    if "prefill" in mode_l:
        return "prefill"
    if "decode" in mode_l:
        return "decode"
    return None


def infer_mode_from_path(path: str | Path) -> str | None:
    return normalize_mode(str(path))


def resolve_serial_baseline(cache: Mapping[str, Any], mode: str | None) -> BaselineResolution:
    """Return the mode-matched serial baseline.

    Prefill/TTFT comparisons must use ``prefill_ms``. Decode/TPOT comparisons
    must use ``decode_tpot_ms`` so both serial and DBO use the same full
    decode-loop TPOT definition.
    """
    mode = normalize_mode(mode)

    if mode == "prefill":
        value = cache.get("prefill_ms")
        if value is not None:
            return BaselineResolution(float(value), "TTFT", "prefill_ms")
        return BaselineResolution(
            None,
            None,
            "missing",
            "TTFT baseline missing; serial total_time_ms/max_new_tokens is a TPOT-style metric and is not comparable to TTFT",
        )

    if mode == "decode":
        value = cache.get("decode_tpot_ms")
        if value is not None:
            return BaselineResolution(float(value), "TPOT", "decode_tpot_ms")

        legacy = cache.get("decode_step_ms")
        if legacy is not None:
            return BaselineResolution(
                None,
                None,
                "legacy-decode-step-ms",
                "decode_tpot_ms missing; legacy decode_step_ms is not used for exact TPOT speedup",
            )

        return BaselineResolution(
            None,
            None,
            "missing",
            "exact TPOT baseline missing; decode_tpot_ms unavailable",
        )

    return BaselineResolution(None, None, "unknown-mode", "could not infer prefill/decode comparison mode")
