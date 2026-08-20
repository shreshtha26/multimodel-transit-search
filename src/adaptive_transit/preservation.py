"""Signal-preservation metrics shared by all treatment/detector combinations."""

from __future__ import annotations

import numpy as np


def robust_scale(values) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    med = float(np.median(x))
    scale = float(1.4826 * np.median(np.abs(x - med)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(x, ddof=1))
    return scale


def periodic_depth_and_snr(values, in_transit) -> dict[str, float | int]:
    series = np.asarray(values, dtype=float).reshape(-1)
    mask = np.asarray(in_transit, dtype=bool).reshape(-1)
    if series.shape != mask.shape:
        raise ValueError("values and in_transit must have the same shape.")
    finite_in = mask & np.isfinite(series)
    finite_out = ~mask & np.isfinite(series)
    if finite_in.sum() == 0 or finite_out.sum() < 3:
        return {"depth": float("nan"), "snr": float("nan"), "in_transit_count": int(finite_in.sum())}
    depth = float(np.median(series[finite_out]) - np.median(series[finite_in]))
    noise = robust_scale(series[finite_out])
    snr = float(depth / noise * np.sqrt(finite_in.sum())) if np.isfinite(noise) and noise > 0 else float("nan")
    return {"depth": depth, "snr": snr, "in_transit_count": int(finite_in.sum())}


def preservation_row(
    *,
    run_id: str,
    config_hash: str,
    star_id: str,
    injection_id: str,
    treatment: str,
    injected_flux,
    treated_flux,
    in_transit,
) -> dict:
    before = periodic_depth_and_snr(injected_flux, in_transit)
    after = periodic_depth_and_snr(treated_flux, in_transit)
    before_depth = float(before["depth"])
    after_depth = float(after["depth"])
    before_snr = float(before["snr"])
    after_snr = float(after["snr"])
    return {
        "run_id": run_id,
        "config_hash": config_hash,
        "star_id": str(star_id),
        "injection_id": str(injection_id),
        "treatment": str(treatment),
        "depth_before": before_depth,
        "depth_after": after_depth,
        "depth_retention_fraction": float(after_depth / before_depth) if before_depth != 0 else float("nan"),
        "snr_before": before_snr,
        "snr_after": after_snr,
        "snr_retention_fraction": float(after_snr / before_snr) if before_snr != 0 else float("nan"),
        "in_transit_observation_count": int(before["in_transit_count"]),
        "success": bool(np.isfinite(after_depth) or np.isfinite(after_snr)),
    }
