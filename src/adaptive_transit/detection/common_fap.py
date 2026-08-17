"""Small empirical-FAP helpers shared by TPS-like validation scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def finite_scores(values) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def empirical_threshold(values, fap_level: float = 0.01) -> float:
    """Conservative empirical upper-tail threshold using NumPy's 'higher' rule."""

    if not 0.0 < float(fap_level) < 1.0:
        raise ValueError("fap_level must be between 0 and 1.")
    scores = finite_scores(values)
    if scores.size == 0:
        raise ValueError("No finite null scores were supplied.")
    return float(np.quantile(scores, 1.0 - float(fap_level), method="higher"))


def empirical_p_value(score: float, null_values) -> float:
    """Finite-sample upper-tail p-value with the standard +1 correction."""

    score = float(score)
    null_scores = finite_scores(null_values)
    if not np.isfinite(score):
        return float("nan")
    if null_scores.size == 0:
        raise ValueError("No finite null scores were supplied.")
    exceedances = int(np.sum(null_scores >= score))
    return float((exceedances + 1) / (null_scores.size + 1))


def calibration_row(
    null_values,
    *,
    method: str,
    score_name: str,
    fap_level: float,
    requested_trials: int | None = None,
) -> dict:
    scores = finite_scores(null_values)
    threshold = empirical_threshold(scores, fap_level=fap_level)
    requested = int(requested_trials) if requested_trials is not None else int(scores.size)
    return {
        "method": str(method),
        "score_name": str(score_name),
        "fap_level": float(fap_level),
        "score_threshold": threshold,
        "successful_null_trials": int(scores.size),
        "requested_null_trials": requested,
        "success_fraction": float(scores.size / requested) if requested > 0 else np.nan,
        "observed_null_exceedance_fraction": float(np.mean(scores >= threshold)),
        "null_score_median": float(np.median(scores)),
        "null_score_max": float(np.max(scores)),
    }


def attach_empirical_p_values(
    frame: pd.DataFrame,
    *,
    score_column: str,
    null_scores,
    output_column: str,
) -> pd.DataFrame:
    out = frame.copy()
    out[output_column] = [
        empirical_p_value(value, null_scores)
        if np.isfinite(float(value))
        else np.nan
        for value in pd.to_numeric(out[score_column], errors="coerce")
    ]
    return out
