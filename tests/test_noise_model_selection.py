from __future__ import annotations

import pandas as pd

from adaptive_transit.noise_models.selection import score_arima_candidates, select_noise_model


def test_selection_prefers_balanced_usable_model() -> None:
    results = pd.DataFrame(
        [
            {
                "order": "(1, 0, 0)",
                "p": 1,
                "d": 0,
                "q": 0,
                "converged": True,
                "test_RMSE": 0.20,
                "test_MAE": 0.10,
                "mean_negative_log_score": 1.0,
                "max_abs_residual_acf": 0.05,
                "minimum_ljung_box_p": 0.40,
                "residual_mean": 0.0,
                "rolling_var_iqr": 0.01,
                "outlier_fraction": 0.01,
                "arch_pvalue": 0.30,
                "failure_reason": "",
            },
            {
                "order": "(3, 1, 3)",
                "p": 3,
                "d": 1,
                "q": 3,
                "converged": False,
                "test_RMSE": 0.01,
                "test_MAE": 0.01,
                "mean_negative_log_score": -5.0,
                "max_abs_residual_acf": 0.01,
                "minimum_ljung_box_p": 0.90,
                "residual_mean": 0.0,
                "rolling_var_iqr": 0.01,
                "outlier_fraction": 0.01,
                "arch_pvalue": 0.30,
                "failure_reason": "",
            },
        ]
    )

    scored = score_arima_candidates(results)
    selected = select_noise_model(scored)

    assert selected["order"] == "(1, 0, 0)"
    invalid = scored.loc[scored["order"] == "(3, 1, 3)"].iloc[0]
    assert bool(invalid["statistical_validity_failed"])
    assert not bool(invalid["fit_metrics_trustworthy"])


def test_selection_penalizes_bad_transit_preservation() -> None:
    results = pd.DataFrame(
        [
            {
                "order": "(1, 0, 0)",
                "p": 1,
                "d": 0,
                "q": 0,
                "converged": True,
                "test_RMSE": 0.20,
                "test_MAE": 0.10,
                "mean_negative_log_score": 1.0,
                "max_abs_residual_acf": 0.10,
                "minimum_ljung_box_p": 0.20,
                "residual_mean": 0.0,
                "rolling_var_iqr": 0.02,
                "outlier_fraction": 0.01,
                "arch_pvalue": 0.30,
                "beats_best_baseline_RMSE": True,
                "beats_best_baseline_MAE": True,
                "depth_retention_fraction": 0.95,
                "snr_retention_fraction": 0.90,
                "standardized_snr_after_arima": 12.0,
                "event_center_shift_cadences": 0,
                "transit_preservation_failure": False,
                "failure_reason": "",
            },
            {
                "order": "(1, 0, 1)",
                "p": 1,
                "d": 0,
                "q": 1,
                "converged": True,
                "test_RMSE": 0.01,
                "test_MAE": 0.01,
                "mean_negative_log_score": -5.0,
                "max_abs_residual_acf": 0.01,
                "minimum_ljung_box_p": 0.90,
                "residual_mean": 0.0,
                "rolling_var_iqr": 0.01,
                "outlier_fraction": 0.01,
                "arch_pvalue": 0.30,
                "beats_best_baseline_RMSE": True,
                "beats_best_baseline_MAE": True,
                "depth_retention_fraction": 0.20,
                "snr_retention_fraction": 0.25,
                "standardized_snr_after_arima": 3.0,
                "event_center_shift_cadences": 4,
                "transit_preservation_failure": True,
                "failure_reason": "",
            },
        ]
    )

    scored = score_arima_candidates(results)
    selected = select_noise_model(scored)

    assert selected["order"] == "(1, 0, 0)"


def test_selection_treats_whitening_as_constraint_before_rmse() -> None:
    results = pd.DataFrame(
        [
            {
                "order": "(1, 0, 0)",
                "p": 1,
                "d": 0,
                "q": 0,
                "converged": True,
                "test_RMSE": 0.20,
                "test_MAE": 0.10,
                "mean_negative_log_score": 1.0,
                "max_abs_residual_acf": 0.05,
                "minimum_ljung_box_p": 0.20,
                "residual_autocorrelation_remaining": False,
                "variance_instability": False,
                "arch_pvalue": 0.30,
                "rolling_var_max_to_median": 2.0,
                "failure_reason": "",
            },
            {
                "order": "(0, 0, 1)",
                "p": 0,
                "d": 0,
                "q": 1,
                "converged": True,
                "test_RMSE": 0.01,
                "test_MAE": 0.01,
                "mean_negative_log_score": -5.0,
                "max_abs_residual_acf": 0.15,
                "minimum_ljung_box_p": 0.002,
                "residual_autocorrelation_remaining": True,
                "variance_instability": False,
                "arch_pvalue": 0.30,
                "rolling_var_max_to_median": 2.0,
                "failure_reason": "",
            },
        ]
    )

    scored = score_arima_candidates(results)
    selected = select_noise_model(scored)

    assert selected["order"] == "(1, 0, 0)"
    assert bool(scored.loc[scored["order"] == "(0, 0, 1)", "whitening_constraint_failed"].iloc[0])


def test_selection_ranks_least_bad_whitening_before_transit_tie_breakers() -> None:
    results = pd.DataFrame(
        [
            {
                "order": "(1, 1, 0)",
                "p": 1,
                "d": 1,
                "q": 0,
                "converged": True,
                "test_RMSE": 0.20,
                "test_MAE": 0.10,
                "mean_negative_log_score": 1.0,
                "max_abs_residual_acf": 0.16,
                "minimum_ljung_box_p": 0.01,
                "residual_autocorrelation_remaining": True,
                "variance_instability": True,
                "arch_pvalue": 0.01,
                "rolling_var_max_to_median": 6.0,
                "depth_retention_fraction": 0.20,
                "snr_retention_fraction": 0.20,
                "standardized_snr_after_arima": 2.0,
                "event_center_shift_cadences": 4,
                "transit_preservation_failure": True,
                "failure_reason": "",
            },
            {
                "order": "(3, 0, 1)",
                "p": 3,
                "d": 0,
                "q": 1,
                "converged": True,
                "test_RMSE": 0.01,
                "test_MAE": 0.01,
                "mean_negative_log_score": -5.0,
                "max_abs_residual_acf": 0.80,
                "minimum_ljung_box_p": 0.0,
                "residual_autocorrelation_remaining": True,
                "variance_instability": True,
                "arch_pvalue": 0.01,
                "rolling_var_max_to_median": 6.0,
                "depth_retention_fraction": 1.00,
                "snr_retention_fraction": 1.00,
                "standardized_snr_after_arima": 10.0,
                "event_center_shift_cadences": 0,
                "transit_preservation_failure": False,
                "failure_reason": "",
            },
        ]
    )

    scored = score_arima_candidates(results)
    selected = select_noise_model(scored)

    assert selected["order"] == "(1, 1, 0)"
    assert bool(selected["whitening_constraint_failed"])


def test_selection_does_not_reward_extra_ljung_box_pvalue_after_pass() -> None:
    results = pd.DataFrame(
        [
            {
                "order": "(1, 0, 0)",
                "p": 1,
                "d": 0,
                "q": 0,
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
            },
            {
                "order": "(1, 0, 1)",
                "p": 1,
                "d": 0,
                "q": 1,
                "converged": True,
                "test_RMSE": 0.04,
                "test_MAE": 0.03,
                "mean_negative_log_score": 0.2,
                "max_abs_residual_acf": 0.04,
                "minimum_ljung_box_p": 0.90,
                "residual_autocorrelation_remaining": False,
                "variance_instability": False,
                "arch_pvalue": 0.30,
                "rolling_var_max_to_median": 2.0,
                "failure_reason": "",
            },
        ]
    )

    scored = score_arima_candidates(results)
    selected = select_noise_model(scored)

    assert selected["order"] == "(1, 0, 0)"
    assert scored["whiteness_violation"].eq(0.0).all()


def test_selection_uses_deterministic_quality_policy_tie_break() -> None:
    base = {
        "order": "(1, 1, 0)",
        "p": 1,
        "d": 1,
        "q": 0,
        "mode": "full_gap",
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
    results = pd.DataFrame(
        [
            {**base, "quality_policy": "permissive", "runtime_seconds": 0.01},
            {**base, "quality_policy": "default", "runtime_seconds": 10.0},
        ]
    )

    scored = score_arima_candidates(results)
    selected = select_noise_model(scored)

    assert selected["quality_policy"] == "default"


def test_differenced_model_is_flagged_for_scientific_review() -> None:
    results = pd.DataFrame(
        [
            {
                "order": "(1, 1, 0)",
                "p": 1,
                "d": 1,
                "q": 0,
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
                "depth_retention_fraction": 0.95,
                "snr_retention_fraction": 0.90,
                "standardized_snr_after_arima": 12.0,
                "event_center_shift_cadences": 0,
                "transit_preservation_failure": False,
                "failure_reason": "",
            }
        ]
    )

    scored = score_arima_candidates(results)
    selected = select_noise_model(scored)

    assert bool(selected["fit_metrics_trustworthy"])
    assert bool(selected["differenced_model"])
    assert bool(selected["differencing_requires_review"])
    assert selected["selection_status"] == "valid_whitened_transit_preserved_but_differencing_requires_review"
