"""Rolling robust scaling for ARIMA innovations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def robust_mad_scale(values: np.ndarray) -> float:
    """Return a robust normal-equivalent scale estimate."""

    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan")
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    if mad > 0:
        return 1.4826 * mad
    if clean.size > 1:
        return float(np.std(clean, ddof=1))
    return float("nan")


def trailing_robust_scale(
    values: np.ndarray,
    *,
    window: int = 96,
    min_periods: int | None = None,
    exclude_current: bool = True,
) -> np.ndarray:
    """Estimate local innovation scale using only current/past samples.

    By default the current point is excluded so a transit-like dip does not
    directly inflate its own denominator.
    """

    if window < 4:
        raise ValueError("window must be at least 4.")

    series = pd.Series(np.asarray(values, dtype=float).reshape(-1))
    source = series.shift(1) if exclude_current else series
    periods = min_periods if min_periods is not None else max(4, window // 4)

    local_scale = source.rolling(window=window, min_periods=periods).apply(
        robust_mad_scale,
        raw=True,
    )

    fallback = robust_mad_scale(series.to_numpy())
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    scale = local_scale.to_numpy(dtype=float)
    scale[~np.isfinite(scale) | (scale <= 0)] = fallback
    return scale


def standardize_innovations(
    innovations: np.ndarray,
    scale: np.ndarray,
    usable_mask: np.ndarray,
) -> np.ndarray:
    """Divide innovations by local scale while preserving unusable rows as NaN."""

    values = np.asarray(innovations, dtype=float).reshape(-1)
    local_scale = np.asarray(scale, dtype=float).reshape(-1)
    usable = np.asarray(usable_mask, dtype=bool).reshape(-1)
    if values.shape != local_scale.shape or values.shape != usable.shape:
        raise ValueError("innovations, scale, and usable_mask must have the same shape.")

    standardized = np.full(values.shape, np.nan, dtype=float)
    valid = usable & np.isfinite(values) & np.isfinite(local_scale) & (local_scale > 0)
    standardized[valid] = values[valid] / local_scale[valid]
    return standardized
