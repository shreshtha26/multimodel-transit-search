"""Hierarchical ARIMA selection for transit-search whitening."""

from __future__ import annotations

import numpy as np
import pandas as pd

WHITENESS_ACF_THRESHOLD = 0.10
LJUNG_BOX_ALPHA = 0.05
ARCH_ALPHA = 0.05
ROLLING_VAR_RATIO_THRESHOLD = 4.0
BOUNDARY_DISTANCE_MIN = 0.02
TRANSIT_DEPTH_RETENTION_MIN = 0.50
TRANSIT_SNR_RETENTION_MIN = 0.50
QUALITY_POLICY_PRIORITY = {"default": 0.0, "strict": 1.0, "permissive": 2.0}


def _numeric(scored: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    values = scored[column] if column in scored.columns else pd.Series(default, index=scored.index)
    return pd.to_numeric(values, errors="coerce")


def _boolean(scored: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    values = scored[column] if column in scored.columns else pd.Series(default, index=scored.index)
    return values.fillna(default).astype(bool)


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


def _constraint_violation(value: pd.Series, threshold: float, *, lower_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(value, errors="coerce")
    if lower_is_better:
        violation = numeric - threshold
    else:
        violation = threshold - numeric
    return violation.where(violation > 0.0, 0.0).fillna(np.inf)


def _selection_status(row: pd.Series, *, has_preservation: bool) -> str:
    if bool(row["statistical_validity_failed"]):
        return "invalid_or_unstable_fit"
    if bool(row["whitening_constraint_failed"]):
        return "valid_but_residual_autocorrelation_remains"
    if bool(row["variance_constraint_failed"]):
        return "valid_whitened_but_variance_unstable"
    if has_preservation and bool(row["transit_preservation_constraint_failed"]):
        return "valid_whitened_but_transit_distorted"
    if bool(row["differencing_requires_review"]):
        return "valid_whitened_transit_preserved_but_differencing_requires_review"
    if has_preservation:
        return "valid_whitened_transit_preserved"
    return "valid_whitened_no_transit_test"


def score_arima_candidates(results: pd.DataFrame) -> pd.DataFrame:
    """Add hierarchical selection diagnostics; lower adequacy score is better."""

    scored = results.copy()
    if scored.empty:
        raise ValueError("Cannot score an empty ARIMA result table.")

    complexity = _numeric(scored, "p", 0.0) + _numeric(scored, "d", 0.0) + _numeric(scored, "q", 0.0)
    failure_reason = scored["failure_reason"].astype(str) if "failure_reason" in scored.columns else pd.Series("", index=scored.index)
    converged = _boolean(scored, "converged", default=False)
    non_converged = _boolean(scored, "non_converged", default=False)
    boundary_count = _numeric(scored, "boundary_coefficient_count", default=0.0).fillna(0.0)
    boundary_distance = _numeric(scored, "min_boundary_distance", default=np.nan)
    boundary_too_close = boundary_distance.notna() & (boundary_distance < BOUNDARY_DISTANCE_MIN)

    max_acf = _numeric(scored, "max_abs_residual_acf")
    min_ljung = _numeric(scored, "minimum_ljung_box_p")
    arch_p = _numeric(scored, "arch_pvalue")
    rolling_ratio = _numeric(scored, "rolling_var_max_to_median")

    statistical_failed = failure_reason.ne("") | ~converged | non_converged | (boundary_count > 0) | boundary_too_close
    differenced_model = _numeric(scored, "d", 0.0).fillna(0.0) > 0
    differencing_justified = _boolean(scored, "differencing_justified", default=False)
    differencing_requires_review = differenced_model & ~differencing_justified
    acf_violation = _constraint_violation(max_acf, WHITENESS_ACF_THRESHOLD, lower_is_better=True)
    ljung_violation = _constraint_violation(min_ljung, LJUNG_BOX_ALPHA, lower_is_better=False)
    whitening_failed = _boolean(scored, "residual_autocorrelation_remaining", default=False) | (acf_violation > 0.0) | (ljung_violation > 0.0)

    arch_violation = _constraint_violation(arch_p, ARCH_ALPHA, lower_is_better=False).replace(np.inf, 0.0)
    rolling_violation = _constraint_violation(rolling_ratio, ROLLING_VAR_RATIO_THRESHOLD, lower_is_better=True).replace(np.inf, 0.0)
    variance_failed = _boolean(scored, "variance_instability", default=False) | (arch_violation > 0.0) | (rolling_violation > 0.0)

    has_preservation = {
        "depth_retention_fraction",
        "snr_retention_fraction",
        "standardized_snr_after_arima",
        "event_center_shift_cadences",
    }.issubset(scored.columns)

    if has_preservation:
        depth_retention = _numeric(scored, "depth_retention_fraction")
        snr_retention = _numeric(scored, "snr_retention_fraction")
        center_shift = _numeric(scored, "event_center_shift_cadences").abs()
        depth_error = (depth_retention - 1.0).abs()
        duration = _numeric(scored, "duration_cadences", default=2.0).fillna(2.0)
        center_limit = (duration / 2.0).clip(lower=1.0)
        preservation_failed = (
            _boolean(scored, "transit_preservation_failure", default=False)
            | depth_retention.lt(TRANSIT_DEPTH_RETENTION_MIN)
            | snr_retention.lt(TRANSIT_SNR_RETENTION_MIN)
            | center_shift.gt(center_limit)
            | depth_retention.isna()
            | snr_retention.isna()
        )
        transit_violation = (
            _constraint_violation(depth_retention, TRANSIT_DEPTH_RETENTION_MIN, lower_is_better=False)
            + _constraint_violation(snr_retention, TRANSIT_SNR_RETENTION_MIN, lower_is_better=False)
            + _constraint_violation(center_shift, center_limit, lower_is_better=True)
        )
        preservation_rank = (
            _rank_lower_is_better(depth_error)
            + _rank_higher_is_better(snr_retention)
            + _rank_higher_is_better(scored["standardized_snr_after_arima"])
            + _rank_lower_is_better(center_shift)
        ) / 4.0
    else:
        preservation_failed = pd.Series(False, index=scored.index)
        transit_violation = pd.Series(0.0, index=scored.index)
        preservation_rank = pd.Series(0.0, index=scored.index)

    beats_rmse = _boolean(scored, "beats_best_baseline_RMSE", default=True)
    beats_mae = _boolean(scored, "beats_best_baseline_MAE", default=True)
    baseline_failed = ~(beats_rmse & beats_mae)
    forecast_rank = (
        _rank_lower_is_better(_numeric(scored, "test_RMSE"))
        + _rank_lower_is_better(_numeric(scored, "test_MAE"))
        + _rank_lower_is_better(_numeric(scored, "mean_negative_log_score"))
    ) / 3.0
    information_rank = (_rank_lower_is_better(_numeric(scored, "BIC")) + _rank_lower_is_better(_numeric(scored, "AIC"))) / 2.0
    whiteness_violation = acf_violation + ljung_violation
    variance_violation = arch_violation + rolling_violation
    quality_policy_rank = scored["quality_policy"].map(QUALITY_POLICY_PRIORITY).fillna(99.0) if "quality_policy" in scored.columns else pd.Series(0.0, index=scored.index)

    scored["model_complexity"] = complexity
    scored["statistical_validity_failed"] = statistical_failed.astype(bool)
    scored["fit_metrics_trustworthy"] = ~statistical_failed.astype(bool)
    scored["differenced_model"] = differenced_model.astype(bool)
    scored["differencing_requires_review"] = differencing_requires_review.astype(bool)
    scored["whitening_constraint_failed"] = whitening_failed.astype(bool)
    scored["variance_constraint_failed"] = variance_failed.astype(bool)
    scored["transit_preservation_constraint_failed"] = preservation_failed.astype(bool)
    scored["baseline_forecast_failed"] = baseline_failed.astype(bool)
    scored["whiteness_violation"] = whiteness_violation
    scored["variance_violation"] = variance_violation
    scored["transit_distortion_violation"] = transit_violation
    scored["forecast_rank"] = forecast_rank
    scored["whiteness_rank"] = _rank_lower_is_better(whiteness_violation)
    scored["stability_rank"] = _rank_lower_is_better(variance_violation)
    scored["complexity_rank"] = _rank_lower_is_better(complexity)
    scored["information_rank"] = information_rank
    scored["quality_policy_rank"] = quality_policy_rank
    scored["transit_preservation_rank"] = preservation_rank
    scored["selection_status"] = scored.apply(_selection_status, axis=1, has_preservation=has_preservation)

    sort_columns = [
        "statistical_validity_failed",
        "whitening_constraint_failed",
        "quality_policy_rank",
        "whiteness_violation",
        "variance_constraint_failed",
        "variance_violation",
        "transit_preservation_constraint_failed",
        "transit_distortion_violation",
        "transit_preservation_rank",
        "baseline_forecast_failed",
        "model_complexity",
        "forecast_rank",
        "information_rank",
        "p",
        "d",
        "q",
    ]
    available_sort_columns = [column for column in sort_columns if column in scored.columns]
    scored = scored.sort_values(available_sort_columns, ascending=True, kind="mergesort").reset_index(drop=True)
    scored["selection_rank"] = np.arange(1, len(scored) + 1, dtype=float)
    if "mode" in scored.columns:
        scored["mode_selection_rank"] = scored.groupby("mode").cumcount().astype(float) + 1.0
    else:
        scored["mode_selection_rank"] = scored["selection_rank"]
    scored["adequacy_score"] = scored["selection_rank"]
    return scored


def select_noise_model(scored_results: pd.DataFrame) -> pd.Series:
    """Return the best row after hierarchical constraint sorting."""

    if "adequacy_score" not in scored_results.columns:
        scored_results = score_arima_candidates(scored_results)

    usable = scored_results[(scored_results["failure_reason"].astype(str) == "") & scored_results["adequacy_score"].notna()]
    if usable.empty:
        raise ValueError("No usable ARIMA candidate was available for selection.")
    return usable.sort_values("adequacy_score", ascending=True).iloc[0]


def order_from_row(row: pd.Series) -> tuple[int, int, int]:
    """Extract `(p, d, q)` from a selected result row."""

    return int(row["p"]), int(row["d"]), int(row["q"])
