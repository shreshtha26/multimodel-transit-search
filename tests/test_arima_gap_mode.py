from __future__ import annotations

import numpy as np

from adaptive_transit.noise_models.arima import evaluate_arima_candidate, fit_arima_model


def test_full_gap_arima_accepts_nan_gaps() -> None:
    rng = np.random.default_rng(123)
    values = rng.normal(scale=0.01, size=80)
    values[20:24] = np.nan
    values[50] = np.nan

    row = evaluate_arima_candidate(
        values,
        (1, 0, 0),
        mode="full_gap",
        allow_missing=True,
        test_fraction=0.20,
        acf_lags=10,
        ljung_box_lags=(5,),
    )

    assert row["mode"] == "full_gap"
    assert row["n_nan_gaps"] == 5
    assert row["failure_reason"] == ""


def test_fit_arima_marks_nan_gap_innovations_unusable() -> None:
    rng = np.random.default_rng(456)
    values = rng.normal(scale=0.01, size=80)
    values[10:13] = np.nan

    fit = fit_arima_model(values, (1, 0, 0), allow_missing=True, mode="full_gap")

    assert np.isnan(fit.innovations[10:13]).all()
    assert not fit.usable_mask[10:13].any()
