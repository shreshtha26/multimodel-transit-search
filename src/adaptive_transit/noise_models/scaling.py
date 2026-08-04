"""Rolling robust scaling for ARIMA innovations."""
import numpy as np
import pandas as pd
def robust_mad_scale(values):
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

def trailing_robust_scale(values, window=96, min_periods=None, exclude_current=True):
    """Estimate local scale from current or past samples."""
    if window < 4:
        raise ValueError("window must be at least 4.")
    series = pd.Series(np.asarray(values, dtype=float).reshape(-1))
    source = series.shift(1) if exclude_current else series
    periods = min_periods if min_periods is not None else max(4, window // 4)
    local_scale = source.rolling(window=window, min_periods=periods).apply(robust_mad_scale, raw=True)
    fallback = robust_mad_scale(series.to_numpy())
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0
    scale = local_scale.to_numpy(dtype=float)
    scale[~np.isfinite(scale) | (scale <= 0)] = fallback
    return scale

def standardize_innovations(innovations, scale, usable_mask):
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
