"""Provisional ARIMA model-selection logic for transit-search whitening."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rank_lower_is_better(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(np.inf, index=series.index)
    fill_value = float(numeric.max(skipna=True)) + 1.0
    return numeric.fillna(fill_value).rank(method="average", ascending=True)


def _rank_higher_is_better(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(np.inf, index=series.index)
    fill_value = float(numeric.min(skipna=True)) - 1.0
    return numeric.fillna(fill_value).rank(method="average", ascending=False)


def score_arima_candidates(results: pd.DataFrame) -> pd.DataFrame:
    """Add a provisional adequacy score; lower is better.

    The score balances forecasting, residual whiteness, stability, complexity,
    and transit preservation when preservation metrics are present.
    """

    scored = results.copy()
    if scored.empty:
        raise ValueError("Cannot score an empty ARIMA result table.")

    complexity = scored[["p", "d", "q"]].sum(axis=1)
    abs_residual_mean = pd.to_numeric(scored["residual_mean"], errors="coerce").abs()

    forecast_rank = (_rank_lower_is_better(scored["test_RMSE"]) + _rank_lower_is_better(scored["test_MAE"]) + _rank_lower_is_better(scored["mean_negative_log_score"])) / 3.0

    whiteness_rank = (_rank_lower_is_better(scored["max_abs_residual_acf"]) + _rank_higher_is_better(scored["minimum_ljung_box_p"])) / 2.0

    stability_rank = (
        _rank_lower_is_better(abs_residual_mean) + _rank_lower_is_better(scored["rolling_var_iqr"]) + _rank_lower_is_better(scored["outlier_fraction"]) + _rank_higher_is_better(scored["arch_pvalue"])
    ) / 4.0

    complexity_rank = _rank_lower_is_better(complexity)
    failure_penalty = np.where(scored["failure_reason"].astype(str) == "", 0.0, 1000.0)
    convergence_penalty = np.where(scored["converged"].astype(bool), 0.0, 100.0)
    beats_rmse = scored.get("beats_best_baseline_RMSE", pd.Series(False, index=scored.index))
    beats_mae = scored.get("beats_best_baseline_MAE", pd.Series(False, index=scored.index))
    baseline_penalty = np.where(beats_rmse.astype(bool) & beats_mae.astype(bool), 0.0, 25.0)

    preservation_rank = pd.Series(0.0, index=scored.index)
    preservation_penalty = np.zeros(len(scored), dtype=float)
    has_preservation = {
        "depth_retention_fraction",
        "snr_retention_fraction",
        "standardized_snr_after_arima",
        "event_center_shift_cadences",
    }.issubset(scored.columns)

    if has_preservation:
        depth_retention_error = (pd.to_numeric(scored["depth_retention_fraction"], errors="coerce") - 1.0).abs()
        center_shift = pd.to_numeric(scored["event_center_shift_cadences"], errors="coerce").abs()
        preservation_rank = (
            _rank_lower_is_better(depth_retention_error)
            + _rank_higher_is_better(scored["snr_retention_fraction"])
            + _rank_higher_is_better(scored["standardized_snr_after_arima"])
            + _rank_lower_is_better(center_shift)
        ) / 4.0

        failed = scored.get(
            "transit_preservation_failure",
            pd.Series(False, index=scored.index),
        )
        preservation_penalty = np.where(failed.astype(bool), 75.0, 0.0)

    scored["model_complexity"] = complexity
    scored["forecast_rank"] = forecast_rank
    scored["whiteness_rank"] = whiteness_rank
    scored["stability_rank"] = stability_rank
    scored["complexity_rank"] = complexity_rank
    scored["transit_preservation_rank"] = preservation_rank

    if has_preservation:
        scored["adequacy_score"] = (
            0.25 * forecast_rank
            + 0.25 * whiteness_rank
            + 0.10 * stability_rank
            + 0.05 * complexity_rank
            + 0.35 * preservation_rank
            + failure_penalty
            + convergence_penalty
            + baseline_penalty
            + preservation_penalty
        )
    else:
        scored["adequacy_score"] = 0.35 * forecast_rank + 0.40 * whiteness_rank + 0.15 * stability_rank + 0.10 * complexity_rank + failure_penalty + convergence_penalty + baseline_penalty
    return scored.sort_values("adequacy_score", ascending=True).reset_index(drop=True)


def select_noise_model(scored_results: pd.DataFrame) -> pd.Series:
    """Return the best usable row from a scored ARIMA result table."""

    if "adequacy_score" not in scored_results.columns:
        scored_results = score_arima_candidates(scored_results)

    usable = scored_results[(scored_results["failure_reason"].astype(str) == "") & scored_results["adequacy_score"].notna()]
    if usable.empty:
        raise ValueError("No usable ARIMA candidate was available for selection.")
    return usable.sort_values("adequacy_score", ascending=True).iloc[0]


def order_from_row(row: pd.Series) -> tuple[int, int, int]:
    """Extract `(p, d, q)` from a selected result row."""

    return int(row["p"]), int(row["d"]), int(row["q"])
