"""Thin wrapper around Transit Least Squares (TLS).

Project background-model outputs are zero-centered residual-like series. TLS
expects a flux convention near unity, so this wrapper recenters each finite
series and adds one before calling the external implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from transitleastsquares import transitleastsquares


def clean_tls_inputs(time, flux):
    t = np.asarray(time, dtype=float).reshape(-1)
    y = np.asarray(flux, dtype=float).reshape(-1)
    if t.shape != y.shape:
        raise ValueError("time and flux must have the same shape.")
    finite = np.isfinite(t) & np.isfinite(y)
    if finite.sum() < 100:
        raise ValueError("TLS needs at least 100 finite observations.")
    t = t[finite]
    y = y[finite]
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    y = 1.0 + (y - float(np.median(y)))
    return t, y


def run_tls(
    time,
    flux,
    *,
    period_min: float = 1.0,
    period_max: float = 15.0,
    use_threads: int = 1,
    oversampling_factor: int = 2,
):
    """Run TLS and return a compact best-result summary and periodogram."""
    t, y = clean_tls_inputs(time, flux)
    model = transitleastsquares(t, y)
    result = model.power(
        period_min=float(period_min),
        period_max=float(period_max),
        use_threads=int(use_threads),
        oversampling_factor=int(oversampling_factor),
        show_progress_bar=False,
        verbose=False,
    )
    summary = {
        "period_days": float(result.period),
        "duration_days": float(result.duration),
        "epoch_days": float(result.T0),
        "sde": float(result.SDE),
        "snr": float(result.snr),
        "depth_raw": float(result.depth),
        "n_observations": int(t.size),
    }
    periods = np.asarray(result.periods, dtype=float)
    power = np.asarray(result.power, dtype=float)
    periodogram = pd.DataFrame({"period_days": periods, "power": power})
    return {"summary": summary, "periodogram": periodogram, "raw_result": result}
