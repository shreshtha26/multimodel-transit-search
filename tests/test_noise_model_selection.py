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
