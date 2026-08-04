from __future__ import annotations

import numpy as np

from adaptive_transit.noise_models.diagnostics import max_abs_acf, residual_diagnostics


def test_max_abs_acf_detects_correlated_series() -> None:
    rng = np.random.default_rng(42)
    white = rng.normal(size=300)
    correlated = np.empty(300)
    correlated[0] = white[0]
    for index in range(1, len(correlated)):
        correlated[index] = 0.8 * correlated[index - 1] + white[index]

    assert max_abs_acf(correlated, nlags=20) > max_abs_acf(white, nlags=20)


def test_residual_diagnostics_has_stage_one_fields() -> None:
    rng = np.random.default_rng(7)
    residuals = rng.normal(size=250)

    diagnostics = residual_diagnostics(residuals, acf_lags=20, ljung_box_lags=(5, 10))

    assert "residual_mean" in diagnostics
    assert "residual_std" in diagnostics
    assert "max_abs_residual_acf" in diagnostics
    assert "residual_acf_lag_1" in diagnostics
    assert "max_abs_residual_acf_1_24" in diagnostics
    assert "mean_abs_residual_acf_1_24" in diagnostics
    assert "max_abs_residual_acf_transit_lags" in diagnostics
    assert "minimum_ljung_box_p" in diagnostics
    assert "outlier_fraction" in diagnostics
