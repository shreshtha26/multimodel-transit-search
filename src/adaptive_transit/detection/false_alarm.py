"""Empirical false-alarm calibration for BLS."""

import numpy as np
import pandas as pd
from tqdm.auto import trange
from adaptive_transit.detection.bls import run_bls


def moving_block_surrogate(flux,block_size,rng):
    """
    Create a transit-free surrogate using moving-block resampling.
    Missing-value positions are preserved. Consecutive values inside each
    sampled block remain together, retaining short-range correlation.
    """
    series = np.asarray(flux, dtype=float).reshape(-1)
    finite_mask = np.isfinite(series)
    finite_values = series[finite_mask]

    if block_size < 2:
        raise ValueError("block_size must be at least 2.")
    if finite_values.size < 2 * block_size:
        raise ValueError("Not enough finite observations for block resampling.")

    center = float(np.median(finite_values))
    residuals = finite_values - center
    n_values = residuals.size
    n_blocks = int(np.ceil(n_values / block_size))
    starts = rng.integers(0, n_values, size=n_blocks)
    sampled_blocks = [np.take(residuals, np.arange(start, start + block_size), mode="wrap") for start in starts]
    sampled = np.concatenate(sampled_blocks)[:n_values]
    surrogate = np.full(series.shape, np.nan, dtype=float)
    surrogate[finite_mask] = center + sampled
    return surrogate


def empirical_fap(observed_power, null_max_powers):
    """
    Estimate FAP using a conservative finite-sample correction.
    """
    powers = np.asarray(null_max_powers, dtype=float)
    powers = powers[np.isfinite(powers)]
    if powers.size == 0:
        raise ValueError("No finite null powers were supplied.")
    exceedances = int(np.sum(powers >= float(observed_power)))
    return float((exceedances + 1) / (powers.size + 1))


def calibrate_bls_fap(time,flux,flux_error,period_grid,duration_grid, *, n_trials=200,
    block_size=24, fap_levels = (0.01, 0.001), objective= "snr",random_seed= 123,
    show_progress: bool = True):
    """
    Run BLS on many null surrogates and estimate FAP thresholds.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    for level in fap_levels:
        if not 0 < level < 1:
            raise ValueError("Every FAP level must be between 0 and 1.")
    rng = np.random.default_rng(random_seed)
    rows = []
    for trial in trange(n_trials, desc="BLS null trials", disable=not show_progress):
        null_flux = moving_block_surrogate(flux, block_size=block_size, rng=rng,)
        result = run_bls(time, null_flux, flux_error, period_grid, duration_grid, objective=objective, top_k=1)
        best = result["summary"]
        rows.append({
                "trial": trial,
                "max_power": float(best["power"]),
                "best_period": float(best["period"]),
                "best_duration": float(best["duration"]),
                "best_epoch": float(best["transit_time"]),
                "best_depth": float(best["depth"])})
    trials = pd.DataFrame(rows)
    max_powers = trials["max_power"].to_numpy(dtype=float)
    threshold_rows = []
    for level in fap_levels:
        threshold = float(np.quantile(max_powers, 1.0 - level, method="higher"))
        threshold_rows.append({"fap_level": float(level),
                "power_threshold": threshold,
                "null_trials": int(n_trials),
                "observed_exceedance_fraction": float(
                    np.mean(max_powers >= threshold))})
    thresholds = pd.DataFrame(threshold_rows)
    return trials, thresholds