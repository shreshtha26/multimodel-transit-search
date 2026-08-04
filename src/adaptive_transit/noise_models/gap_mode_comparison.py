"""Gap-mode comparison helpers for Phase 1 ARIMA diagnostics."""

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from adaptive_transit.noise_models.selection import score_arima_candidates, select_noise_model
from adaptive_transit.noise_models.stationarity import StationarityAssessment


REQUIRED_GAP_COMPARISON_COLUMNS = (
    "quality_policy",
    "gap_mode",
    "observations",
    "missing_fraction",
    "interpolated_fraction",
    "stationarity_conclusion",
    "recommended_d",
    "selected_order",
    "selected_candidate_status",
    "selected_differencing_alignment",
    "fit_metrics_trustworthy",
    "residual_acf_lag_1",
    "max_abs_residual_acf_1_24",
    "mean_abs_residual_acf_1_24",
    "max_abs_residual_acf_transit_lags",
    "ljung_box_rejected_lags",
    "variance_stability",
    "scientifically_acceptable",
    "failure_reasons",
)


def candidate_family_table(orders: tuple[tuple[int, int, int], ...], gap_modes: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for gap_mode in gap_modes:
        for rank, order in enumerate(orders, start=1):
            rows.append({"gap_mode": gap_mode, "candidate_family_rank": rank, "p": order[0], "d": order[1], "q": order[2], "order": f"ARIMA{order}"})
    return pd.DataFrame(rows)


def same_candidate_family_across_modes(candidate_family: pd.DataFrame) -> bool:
    grouped = candidate_family.groupby("gap_mode")[["p", "d", "q"]].apply(lambda frame: tuple(map(tuple, frame.to_numpy(dtype=int))))
    return bool(grouped.nunique() == 1)


def stationarity_candidate_fields(assessment: StationarityAssessment, candidate_d: int) -> dict[str, Any]:
    recommended_d = assessment.recommended_d
    if recommended_d == 0:
        family_role = "stationarity-supported primary family" if candidate_d == 0 else "differenced challenger family"
    elif recommended_d == 1:
        family_role = "stationarity-supported primary family" if candidate_d == 1 else "undifferenced challenger family"
    else:
        family_role = "unresolved family"
    if candidate_d == 0:
        alignment = "aligned_with_stationarity_evidence" if recommended_d == 0 else "unresolved"
        statistically_supported = recommended_d == 0
        requires_review = False
    elif recommended_d == 1:
        alignment = "aligned_with_stationarity_evidence"
        statistically_supported = True
        requires_review = False
    elif recommended_d == 0:
        alignment = "conflicts_with_stationarity_evidence"
        statistically_supported = False
        requires_review = True
    else:
        alignment = "unresolved"
        statistically_supported = False
        requires_review = True
    reason_codes = list(assessment.reason_codes)
    if candidate_d == 1 and recommended_d == 0:
        reason_codes.append("SELECTED_DIFFERENCED_MODEL_CONFLICTS_WITH_D0_EVIDENCE")
    elif candidate_d == 1 and recommended_d is None:
        reason_codes.append("SELECTED_DIFFERENCED_MODEL_HAS_UNRESOLVED_STATIONARITY_SUPPORT")
    return {
        "candidate_d": candidate_d,
        "candidate_family_role": family_role,
        "candidate_differencing_alignment": alignment,
        "differencing_justified": bool(candidate_d > 0 and statistically_supported),
        "differencing_statistically_supported": statistically_supported,
        "differencing_requires_review": requires_review,
        "selected_model_differencing_alignment": alignment,
        "selected_model_differencing_requires_review": requires_review,
        "stationarity_alpha": assessment.original_adf.alpha,
        "stationarity_diagnostics_available": True,
        "original_series_stationarity_conclusion": assessment.original_series_conclusion,
        "recommended_d": recommended_d,
        "recommendation_strength": assessment.recommendation_strength,
        "stationarity_reason_codes": tuple(dict.fromkeys(reason_codes)),
    }


def score_candidates_for_gap_mode(candidate_table: pd.DataFrame) -> pd.DataFrame:
    scored = score_arima_candidates(candidate_table.copy())
    scored["mode_selection_rank"] = scored["selection_rank"]
    return scored


def select_gap_mode_candidate(scored_candidates: pd.DataFrame) -> pd.Series:
    return select_noise_model(scored_candidates)


def ljung_box_rejected_lags(row: pd.Series | dict[str, Any], alpha: float = 0.05) -> tuple[int, ...]:
    rejected: list[int] = []
    for key, value in dict(row).items():
        if not key.startswith("ljung_box_p_lag_"):
            continue
        try:
            lag = int(key.rsplit("_", 1)[1])
            pvalue = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(pvalue) and pvalue < alpha:
            rejected.append(lag)
    return tuple(sorted(rejected))


def scientific_admissibility(row: pd.Series | dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    data = dict(row)
    reasons: list[str] = []
    if not bool(data.get("fit_metrics_trustworthy", False)):
        reasons.append("fit_metrics_not_trustworthy")
    if bool(data.get("statistical_validity_failed", False)):
        reasons.append("fit_invalid_or_unstable")
    if bool(data.get("differencing_requires_review", False)) or bool(data.get("selected_model_differencing_requires_review", False)):
        reasons.append("differencing_requires_review")
    if bool(data.get("whitening_constraint_failed", False)):
        reasons.append("residual_autocorrelation_remains")
    if bool(data.get("variance_constraint_failed", False)):
        reasons.append("variance_instability_remains")
    if data.get("transit_preservation_diagnostics_available") is False:
        reasons.append("transit_preservation_not_evaluated_for_gap_mode")
    if bool(data.get("transit_preservation_constraint_failed", False)):
        reasons.append("transit_preservation_failed")
    return len(reasons) == 0, tuple(dict.fromkeys(reasons))


def gap_mode_summary_row(
    *,
    quality_policy: str,
    selected: pd.Series,
    assessment: StationarityAssessment,
    metadata: dict[str, Any],
    alpha: float = 0.05,
) -> dict[str, Any]:
    order = (int(selected["p"]), int(selected["d"]), int(selected["q"]))
    scientifically_acceptable, failure_reasons = scientific_admissibility(selected)
    stationarity = asdict(assessment)
    return {
        "quality_policy": quality_policy,
        "gap_mode": str(metadata["gap_mode"]),
        "observations": int(metadata.get("observations", metadata.get("observed_cadences", 0))),
        "missing_fraction": float(metadata.get("missing_fraction", 0.0)),
        "interpolated_fraction": float(metadata.get("interpolated_fraction", 0.0)),
        "stationarity_conclusion": assessment.original_series_conclusion,
        "recommended_d": assessment.recommended_d,
        "selected_order": f"ARIMA{order}",
        "selected_p": order[0],
        "selected_d": order[1],
        "selected_q": order[2],
        "selected_candidate_status": str(selected.get("selection_status", "")),
        "selected_differencing_alignment": str(selected.get("selected_model_differencing_alignment", selected.get("candidate_differencing_alignment", "unknown"))),
        "fit_metrics_trustworthy": bool(selected.get("fit_metrics_trustworthy", False)),
        "residual_acf_lag_1": float(selected.get("residual_acf_lag_1", np.nan)),
        "max_abs_residual_acf_1_24": float(selected.get("max_abs_residual_acf_1_24", np.nan)),
        "mean_abs_residual_acf_1_24": float(selected.get("mean_abs_residual_acf_1_24", np.nan)),
        "max_abs_residual_acf_transit_lags": float(selected.get("max_abs_residual_acf_transit_lags", np.nan)),
        "ljung_box_rejected_lags": ",".join(str(lag) for lag in ljung_box_rejected_lags(selected, alpha=alpha)),
        "variance_stability": "unstable" if bool(selected.get("variance_constraint_failed", False)) else "stable",
        "scientifically_acceptable": bool(scientifically_acceptable),
        "failure_reasons": ";".join(failure_reasons),
        "differencing_requires_review": bool(selected.get("selected_model_differencing_requires_review", selected.get("differencing_requires_review", False))),
        "differencing_statistically_supported": bool(selected.get("differencing_statistically_supported", False)),
        "whitening_constraint_failed": bool(selected.get("whitening_constraint_failed", False)),
        "variance_constraint_failed": bool(selected.get("variance_constraint_failed", False)),
        "transit_preservation_diagnostics_available": bool(selected.get("transit_preservation_diagnostics_available", False)),
        "selection_rank": int(selected.get("mode_selection_rank", selected.get("selection_rank", 0))),
        "rmse": float(selected.get("rmse", np.nan)),
        "mae": float(selected.get("mae", np.nan)),
        "aic": float(selected.get("aic", np.nan)),
        "bic": float(selected.get("bic", np.nan)),
        "AIC": float(selected.get("AIC", selected.get("aic", np.nan))),
        "BIC": float(selected.get("BIC", selected.get("bic", np.nan))),
        "test_RMSE": float(selected.get("test_RMSE", selected.get("rmse", np.nan))),
        "test_MAE": float(selected.get("test_MAE", selected.get("mae", np.nan))),
        "minimum_ljung_box_p": float(selected.get("minimum_ljung_box_p", np.nan)),
        "residual_autocorrelation_remaining": bool(selected.get("residual_autocorrelation_remaining", selected.get("whitening_constraint_failed", False))),
        "variance_instability": bool(selected.get("variance_instability", selected.get("variance_constraint_failed", False))),
        "gap_count": int(metadata.get("gap_count", 0)),
        "max_gap_length": int(metadata.get("max_gap_length", 0)),
        "total_grid_length": int(metadata.get("total_grid_length", metadata.get("observations", 0))),
        "observed_cadences": int(metadata.get("observed_cadences", metadata.get("observations", 0))),
        "missing_cadences": int(metadata.get("missing_cadences", 0)),
        "fraction_of_quarter_retained": float(metadata.get("fraction_of_quarter_retained", 1.0)),
        "interpolation_method": metadata.get("interpolation_method"),
        "maximum_allowed_interpolated_gap": metadata.get("maximum_allowed_interpolated_gap"),
        "unfilled_long_gaps": metadata.get("unfilled_long_gaps"),
        "edge_extrapolation_policy": metadata.get("edge_extrapolation_policy"),
        "ordinary_lags_meaningful": bool(metadata.get("ordinary_lags_meaningful", False)),
        "stationarity": stationarity,
    }


def validate_gap_comparison_schema(summary_table: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_GAP_COMPARISON_COLUMNS if column not in summary_table.columns]
    if missing:
        raise ValueError(f"Gap-mode comparison table is missing required columns: {missing}")
