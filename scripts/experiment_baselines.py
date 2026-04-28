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

    Prefill DBO is a full prefill pass and must be compared only with
    ``prefill_ms``. Decode DBO records one representative decode step, so a
    serial full-generation timing can be normalized by token count when
    ``decode_step_ms`` is absent.
    """
    mode = normalize_mode(mode)

    if mode == "prefill":
        value = cache.get("prefill_ms")
        if value is not None:
            return BaselineResolution(float(value), "prefill", "prefill_ms")
        return BaselineResolution(
            None,
            None,
            "missing",
            "prefill baseline missing; serial total_time_ms/max_new_tokens is a decode metric and is not comparable",
        )

    if mode == "decode":
        value = cache.get("decode_step_ms")
        if value is not None:
            return BaselineResolution(float(value), "step", "decode_step_ms")

        total = cache.get("total_time_ms")
        tokens = cache.get("max_new_tokens") or cache.get("tokens")
        if total is not None and tokens:
            return BaselineResolution(
                float(total) / int(tokens),
                "step",
                "total_time_ms/max_new_tokens fallback",
                "decode_step_ms missing; using serial total_time_ms / max_new_tokens",
            )

        return BaselineResolution(
            None,
            None,
            "missing",
            "decode baseline missing and total_time_ms/max_new_tokens fallback unavailable",
        )

    return BaselineResolution(None, None, "unknown-mode", "could not infer prefill/decode comparison mode")
