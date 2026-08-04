import numpy as np
import pandas as pd

from adaptive_transit.noise_models.selection import score_arima_candidates, select_noise_model
from adaptive_transit.noise_models.stationarity import (
    StationarityAssessment,
    UnitRootTestResult,
    assess_stationarity,
    interpret_stationarity_tests,
    stationarity_candidate_fields)
from scripts.run_single_target_arima import phase1_completion_report


def _test_result(test_name: str, reject_null: bool) -> UnitRootTestResult:
    return UnitRootTestResult(
        test_name=test_name,
        statistic=0.0,
        pvalue=0.01 if reject_null else 0.40,
        critical_values={},
        null_hypothesis="unit root / nonstationary" if test_name == "ADF" else "level stationary",
        reject_null=reject_null,
        alpha=0.05,
        n_observations=200,
        regression="c",
        lag_selection="AIC" if test_name == "ADF" else "auto",
        lags_used=1,
        status="ok",
        warning_messages=(),
    )


def _assessment_from_rejections(adf_rejects: bool, kpss_rejects: bool) -> StationarityAssessment:
    adf_result = _test_result("ADF", adf_rejects)
    kpss_result = _test_result("KPSS", kpss_rejects)
    conclusion, recommended_d, strength, agree, reason_codes = interpret_stationarity_tests(adf_result, kpss_result, original_series=True)
    return StationarityAssessment(
        original_adf=adf_result,
        original_kpss=kpss_result,
        differenced_adf=None,
        differenced_kpss=None,
        original_series_conclusion=conclusion,
        differenced_series_conclusion=None,
        recommended_d=recommended_d,
        recommendation_strength=strength,
        diagnostics_agree=agree,
        differencing_statistically_supported=conclusion == "nonstationary_supported",
        differencing_requires_review=conclusion != "nonstationary_supported",
        reason_codes=reason_codes,
        modelling_mode="full_gap",
        observations_used=200,
        missing_observations=0,
        preprocessing_summary={},
    )


def _valid_candidate(order: str, p: int, d: int, q: int, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "order": order,
        "p": p,
        "d": d,
        "q": q,
        "quality_policy": "default",
        "mode": "full_gap",
        "n_nan_gaps": 1,
        "converged": True,
        "test_RMSE": 0.03,
        "test_MAE": 0.02,
        "mean_negative_log_score": 0.1,
        "max_abs_residual_acf": 0.04,
        "minimum_ljung_box_p": 0.20,
        "residual_autocorrelation_remaining": False,
        "variance_instability": False,
        "arch_pvalue": 0.30,
        "rolling_var_max_to_median": 2.0,
        "failure_reason": "",
    }
    row.update(extra)
    return row


def _stationary_ar1(seed: int = 123, size: int = 800, phi: float = 0.45) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=1.0, size=size)
    values = np.empty(size)
    values[0] = noise[0]
    for index in range(1, size):
        values[index] = phi * values[index - 1] + noise[index]
    return values


def test_stationary_synthetic_process_supports_d0() -> None:
    assessment = assess_stationarity(_stationary_ar1(), modelling_mode="full_gap", preprocessing_summary={}, min_observations=50)

    assert assessment.original_series_conclusion == "stationary_supported"
    assert assessment.recommended_d == 0
    assert not assessment.differencing_statistically_supported


def test_random_walk_supports_d1() -> None:
    rng = np.random.default_rng(456)
    values = np.cumsum(rng.normal(size=800))
    assessment = assess_stationarity(values, modelling_mode="full_gap", preprocessing_summary={}, min_observations=50)

    assert assessment.original_series_conclusion == "nonstationary_supported"
    assert assessment.recommended_d == 1
    assert assessment.differencing_statistically_supported


def test_conflicting_stationarity_tests_do_not_recommend_d() -> None:
    conclusion, recommended_d, strength, agree, reasons = interpret_stationarity_tests(
        _test_result("ADF", True),
        _test_result("KPSS", True),
        original_series=True,
    )

    assert conclusion == "conflicting_rejections"
    assert recommended_d is None
    assert strength == "unresolved"
    assert not agree
    assert "CONFLICTING_STATIONARITY_TESTS" in reasons


def test_inconclusive_stationarity_tests_do_not_claim_d0_or_d1() -> None:
    conclusion, recommended_d, strength, agree, reasons = interpret_stationarity_tests(
        _test_result("ADF", False),
        _test_result("KPSS", False),
        original_series=True,
    )

    assert conclusion == "inconclusive_low_power"
    assert recommended_d is None
    assert strength == "unresolved"
    assert not agree
    assert "INCONCLUSIVE_STATIONARITY_TESTS" in reasons


def test_selected_d1_conflicts_with_d0_evidence() -> None:
    assessment = _assessment_from_rejections(adf_rejects=True, kpss_rejects=False)
    row = _valid_candidate("(1, 1, 0)", 1, 1, 0, **stationarity_candidate_fields(assessment, 1))

    scored = score_arima_candidates(pd.DataFrame([row]))
    selected = select_noise_model(scored)

    assert selected["candidate_differencing_alignment"] == "conflicts_with_stationarity_evidence"
    assert bool(selected["differencing_requires_review"])


def test_selected_d1_aligns_with_nonstationarity_evidence_but_readiness_still_needs_whitening() -> None:
    assessment = _assessment_from_rejections(adf_rejects=False, kpss_rejects=True)
    row = _valid_candidate(
        "(1, 1, 0)",
        1,
        1,
        0,
        **stationarity_candidate_fields(assessment, 1),
        residual_autocorrelation_remaining=True,
        max_abs_residual_acf=0.20,
    )
    scored = score_arima_candidates(pd.DataFrame([row]))
    selected = select_noise_model(scored)
    report = phase1_completion_report(
        selected=selected,
        preprocessing_summary={"quality_policy": "default", "normalization_fit_fraction": 0.8, "n_cadence_grid": 20, "n_raw": 20},
        stationarity_assessment=assessment,
        stability_summaries={"chronological_prefix": {"n_successful_runs": 1}, "segment": {"n_successful_runs": 1}},
        preservation_metrics={"depth_retention_fraction": 1.0, "snr_retention_fraction": 1.0, "transit_preservation_failure": False},
        transformed_template_metrics={"innovation_transformed_template_statistic": 2.0, "innovation_unchanged_box_statistic": 1.0, "transformed_template_improves_unchanged_box": True},
        template_scan_summary={"n_trial_centers": 2, "best_injected_neighborhood_rank": 1, "injected_center_recovered_as_best": True},
        multi_injection_summary={"n_injections": 1, "rank1_recovery_rate": 1.0, "rank3_recovery_rate": 1.0, "transit_preservation_failure_rate": 0.0},
        scored=scored,
        regular=pd.DataFrame({"row_present": [True] * 20, "gap_reason": ["usable"] * 20}),
        innovations=pd.DataFrame({"innovation": [0.0], "innovation_scale": [1.0], "standardized_innovation": [0.0]}),
    )

    assert selected["candidate_differencing_alignment"] == "aligned_with_stationarity_evidence"
    assert not bool(selected["differencing_requires_review"])
    assert bool(selected["whitening_constraint_failed"])
    assert not report["phase1_scientific_ready_for_phase2"]


def test_differenced_series_stationarity_does_not_imply_differencing_was_required() -> None:
    assessment = assess_stationarity(_stationary_ar1(seed=789), modelling_mode="full_gap", preprocessing_summary={}, min_observations=50)

    assert assessment.differenced_series_conclusion is not None
    assert assessment.recommended_d == 0
    assert not assessment.differencing_statistically_supported


def test_missing_and_interpolated_metadata_are_recorded() -> None:
    values = _stationary_ar1(size=160)
    values[10:14] = np.nan
    assessment = assess_stationarity(
        values,
        modelling_mode="interpolated",
        preprocessing_summary={"quality_policy": "default"},
        gaps_compressed=True,
        interpolated=True,
        contiguous_segment_used=False,
        series_representation="finite_values_from_interpolated_series",
        min_observations=50,
    )

    metadata = assessment.preprocessing_summary
    assert assessment.missing_observations == 4
    assert metadata["stationarity_original_observations"] == 160
    assert metadata["stationarity_removed_nonfinite"] == 4
    assert metadata["stationarity_gaps_compressed"]
    assert metadata["stationarity_interpolated"]
    assert "INTERPOLATED_SERIES_DIAGNOSTIC" in assessment.reason_codes
