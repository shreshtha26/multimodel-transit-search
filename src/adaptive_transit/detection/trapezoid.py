"""BLS-seeded periodic trapezoid morphology refiner.

This is intentionally labelled a refiner rather than an independent detector:
candidate periods come from BLS, then finite ingress/egress trapezoid templates
are fitted and re-ranked.  It provides a morphology bridge between a box search
and TLS without pretending to add an independent period-search algorithm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def periodic_trapezoid_shape(
    time,
    *,
    period_days: float,
    epoch_days: float,
    duration_days: float,
    ingress_fraction: float,
):
    t = np.asarray(time, dtype=float).reshape(-1)
    period = float(period_days)
    duration = float(duration_days)
    ingress_fraction = float(ingress_fraction)
    if period <= 0 or duration <= 0 or duration >= period:
        raise ValueError("Require 0 < duration_days < period_days.")
    if not 0 < ingress_fraction < 0.5:
        raise ValueError("ingress_fraction must be in (0, 0.5).")
    half = 0.5 * duration
    ingress = ingress_fraction * duration
    flat_half = half - ingress
    phase = np.mod(t - float(epoch_days) + 0.5 * period, period) - 0.5 * period
    distance = np.abs(phase)
    shape = np.zeros(t.shape, dtype=float)
    flat = np.isfinite(distance) & (distance <= flat_half)
    edge = np.isfinite(distance) & (distance > flat_half) & (distance < half)
    shape[flat] = 1.0
    shape[edge] = (half - distance[edge]) / ingress
    return shape


def robust_scale(values) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    median = float(np.median(x))
    scale = float(1.4826 * np.median(np.abs(x - median)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(x, ddof=1))
    return scale


def fit_trapezoid_template(time, flux, shape):
    t = np.asarray(time, dtype=float).reshape(-1)
    y = np.asarray(flux, dtype=float).reshape(-1)
    s = np.asarray(shape, dtype=float).reshape(-1)
    finite = np.isfinite(t) & np.isfinite(y) & np.isfinite(s)
    if finite.sum() < 20:
        raise ValueError("At least 20 finite observations are required.")
    # y = intercept - depth * shape
    design = np.column_stack([np.ones(int(finite.sum())), -s[finite]])
    coefficients, *_ = np.linalg.lstsq(design, y[finite], rcond=None)
    intercept, depth = (float(coefficients[0]), float(coefficients[1]))
    observed = y[finite]
    fitted_values = design @ coefficients
    residual = observed - fitted_values
    null_residual = observed - float(np.median(observed))
    sse_model = float(np.dot(residual, residual))
    sse_null = float(np.dot(null_residual, null_residual))
    improvement = max(sse_null - sse_model, 0.0)
    noise = robust_scale(observed[s[finite] < 1.0e-12])
    # Ranking is based on least-squares improvement, so a morphology mismatch is
    # penalized even when the out-of-transit series is nearly noiseless.
    score = float(improvement)
    return {
        "intercept": intercept,
        "depth": depth,
        "score": score,
        "noise": noise,
        "sse_model": sse_model,
        "sse_null": sse_null,
        "finite_count": int(finite.sum()),
    }


def run_bls_seeded_trapezoid(
    time,
    flux,
    bls_result,
    *,
    duration_grid,
    ingress_fractions=(0.10, 0.20, 0.30),
    top_k_periods: int = 5,
    phase_offset_fractions=(-0.25, -0.125, 0.0, 0.125, 0.25),
):
    """Refine and re-rank BLS candidates using periodic trapezoid templates."""
    peaks = bls_result["top_peaks"].head(int(top_k_periods))
    if peaks.empty:
        raise ValueError("BLS supplied no seed peaks.")
    duration_grid = np.asarray(duration_grid, dtype=float).reshape(-1)
    rows = []
    for seed_rank, seed in enumerate(peaks.to_dict(orient="records"), start=1):
        period = float(seed["period"])
        seed_epoch = float(seed["transit_time"])
        seed_duration = float(seed["duration"])
        durations = np.unique(np.r_[duration_grid, seed_duration])
        for duration in durations:
            if duration <= 0 or duration >= period:
                continue
            for ingress_fraction in ingress_fractions:
                for phase_fraction in phase_offset_fractions:
                    epoch = seed_epoch + float(phase_fraction) * float(duration)
                    shape = periodic_trapezoid_shape(
                        time,
                        period_days=period,
                        epoch_days=epoch,
                        duration_days=float(duration),
                        ingress_fraction=float(ingress_fraction),
                    )
                    fit = fit_trapezoid_template(time, flux, shape)
                    rows.append(
                        {
                            "seed_rank": int(seed_rank),
                            "period_days": period,
                            "epoch_days": epoch,
                            "duration_days": float(duration),
                            "ingress_fraction": float(ingress_fraction),
                            **fit,
                        }
                    )
    evaluated = pd.DataFrame(rows)
    if evaluated.empty or not np.isfinite(evaluated["score"]).any():
        raise ValueError("Trapezoid refinement produced no finite scores.")
    evaluated = evaluated.sort_values("score", ascending=False).reset_index(drop=True)
    summary = evaluated.iloc[0].to_dict()
    return {"summary": summary, "evaluated": evaluated}
