"""Box Least Squares baseline detector."""

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares

def clean_bls_inputs(time, flux, flux_error=None):
    """Return finite arrays suitable for Astropy BLS."""
    t = np.asarray(time, dtype=float).reshape(-1)
    y = np.asarray(flux, dtype=float).reshape(-1)
    if t.shape != y.shape:
        raise ValueError("time and flux must have the same shape.")
    valid = np.isfinite(t) & np.isfinite(y)
    dy = None
    if flux_error is not None:
        err = np.asarray(flux_error, dtype=float).reshape(-1)
        if err.shape != t.shape:
            raise ValueError("flux_error must have the same shape as time.")
        valid &= np.isfinite(err) & (err > 0)
        dy = err[valid]
    t = t[valid]
    y = y[valid]
    if t.size < 20:
        raise ValueError("BLS needs at least 20 finite observations.")
    order = np.argsort(t)
    return t[order], y[order], None if dy is None else dy[order], valid

def default_period_grid(time, min_period_days=0.5, max_period_days=None, n_periods=1000):
    """Build the first simple linear BLS period grid."""
    observed = np.asarray(time, dtype=float)
    finite = observed[np.isfinite(observed)]
    if finite.size < 2:
        raise ValueError("time must contain at least two finite values.")
    baseline = float(np.max(finite) - np.min(finite))
    max_period = 0.5 * baseline if max_period_days is None else float(max_period_days)
    if min_period_days <= 0 or max_period <= min_period_days:
        raise ValueError("period bounds must satisfy 0 < min_period_days < max_period_days.")
    return np.linspace(float(min_period_days), max_period, int(n_periods))

def default_duration_grid(min_duration_hours=1.5, max_duration_hours=12.0, n_durations=8):
    """Build the first simple BLS duration grid in days."""
    if min_duration_hours <= 0 or max_duration_hours <= min_duration_hours:
        raise ValueError("duration bounds must satisfy 0 < min_duration_hours < max_duration_hours.")
    return np.linspace(float(min_duration_hours) / 24.0, float(max_duration_hours) / 24.0, int(n_durations))

def bls_result_table(results):
    """Convert Astropy BLS results to a compact DataFrame."""
    columns = ["period", "power", "duration", "transit_time", "depth", "depth_err", "depth_snr", "log_likelihood"]
    return pd.DataFrame({column: np.asarray(results[column], dtype=float) for column in columns})

def run_bls(time, flux, flux_error=None, period_grid=None, duration_grid=None, objective="snr", top_k=5):
    """Run BLS and return the periodogram, top peaks, and best row."""
    t, y, dy, input_mask = clean_bls_inputs(time, flux, flux_error)
    periods = default_period_grid(t) if period_grid is None else np.asarray(period_grid, dtype=float)
    durations = default_duration_grid() if duration_grid is None else np.asarray(duration_grid, dtype=float)
    if periods.ndim != 1 or durations.ndim != 1 or periods.size == 0 or durations.size == 0:
        raise ValueError("period_grid and duration_grid must be non-empty one-dimensional arrays.")
    model = BoxLeastSquares(t, y, dy=dy)
    results = model.power(periods, durations, objective=objective)
    periodogram = bls_result_table(results)
    finite_power = periodogram["power"].replace([np.inf, -np.inf], np.nan)
    if finite_power.notna().sum() == 0:
        raise ValueError("BLS produced no finite power values.")
    best_index = int(finite_power.idxmax())
    top = periodogram.sort_values("power", ascending=False).head(int(top_k)).reset_index(drop=True)
    summary = periodogram.loc[best_index].to_dict()
    summary.update({"objective": objective, "n_observations": int(t.size), "n_periods": int(periods.size), "n_durations": int(durations.size)})
    return {"periodogram": periodogram, "top_peaks": top, "summary": summary, "input_mask": input_mask}

def period_match_fraction(recovered_period, injected_period, harmonic_factors=(0.5, 1.0, 2.0)):
    """Return the best fractional period error across accepted harmonics."""
    recovered = float(recovered_period)
    injected = float(injected_period)
    if recovered <= 0 or injected <= 0:
        return float("nan")
    errors = [abs(recovered - injected * factor) / (injected * factor) for factor in harmonic_factors]
    return float(min(errors))
