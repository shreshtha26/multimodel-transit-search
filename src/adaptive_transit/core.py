"""Core data containers for the unified adaptive-transit pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np


def _array(values, *, dtype=float) -> np.ndarray:
    return np.asarray(values, dtype=dtype).reshape(-1)


@dataclass(frozen=True)
class LightCurve:
    """Cadence-grid light curve passed between treatments and detectors."""

    time: np.ndarray
    flux: np.ndarray
    flux_error: np.ndarray | None = None
    cadenceno: np.ndarray | None = None
    segment_id: np.ndarray | None = None
    quality: np.ndarray | None = None
    row_present: np.ndarray | None = None
    usable_mask: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = _array(self.time)
        flux = _array(self.flux)
        if time.shape != flux.shape:
            raise ValueError("time and flux must have the same shape.")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "flux", flux)
        for name, dtype in (
            ("flux_error", float),
            ("cadenceno", float),
            ("segment_id", int),
            ("quality", float),
            ("row_present", bool),
            ("usable_mask", bool),
        ):
            value = getattr(self, name)
            if value is None:
                continue
            array = _array(value, dtype=dtype)
            if array.shape != time.shape:
                raise ValueError(f"{name} must have the same shape as time.")
            object.__setattr__(self, name, array)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_regularized_frame(
        cls,
        regular,
        *,
        flux_column: str = "normalized_flux",
        metadata: Mapping[str, Any] | None = None,
    ) -> "LightCurve":
        required = {"time", flux_column}
        missing = sorted(required.difference(regular.columns))
        if missing:
            raise ValueError(f"Regularized light curve is missing columns: {missing}")
        return cls(
            time=regular["time"].to_numpy(dtype=float),
            flux=regular[flux_column].to_numpy(dtype=float),
            flux_error=regular["flux_error"].to_numpy(dtype=float) if "flux_error" in regular.columns else None,
            cadenceno=regular["cadenceno"].to_numpy(dtype=float) if "cadenceno" in regular.columns else None,
            segment_id=regular["segment_id"].to_numpy(dtype=int) if "segment_id" in regular.columns else None,
            quality=regular["quality"].to_numpy(dtype=float) if "quality" in regular.columns else None,
            row_present=regular["row_present"].to_numpy(dtype=bool) if "row_present" in regular.columns else None,
            usable_mask=regular["usable"].to_numpy(dtype=bool) if "usable" in regular.columns else None,
            metadata=metadata or {},
        )

    def with_flux(self, flux, *, metadata: Mapping[str, Any] | None = None) -> "LightCurve":
        merged = dict(self.metadata)
        if metadata:
            merged.update(metadata)
        return replace(self, flux=_array(flux), metadata=merged)

    @property
    def finite_mask(self) -> np.ndarray:
        return np.isfinite(self.time) & np.isfinite(self.flux)


def array_digest(values) -> str:
    """Stable short digest for deterministic cache keys and test assertions."""

    import hashlib

    array = np.asarray(values, dtype=float).reshape(-1)
    payload = np.nan_to_num(array, nan=np.inf, posinf=np.inf, neginf=-np.inf)
    return hashlib.sha256(payload.tobytes()).hexdigest()[:16]


def stable_seed(*parts: object) -> int:
    """Derive a reproducible 32-bit seed from persistent identifiers."""

    import hashlib

    payload = "\x1f".join(str(part) for part in parts).encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % np.iinfo(np.int32).max
