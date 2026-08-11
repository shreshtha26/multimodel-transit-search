import numpy as np
import pandas as pd

from adaptive_transit.noise_models.characterization import (
    characterize_regularized_light_curve,
    structural_diagnostic_comparison,
)


def _synthetic_regularized_frame(n_points=600, period_days=2.0):
    rng = np.random.default_rng(123)
    cadenceno = np.arange(n_points)
    cadence_days = 0.020433
    time = cadenceno * cadence_days
    flux = 0.002 * np.sin(2.0 * np.pi * time / period_days) + rng.normal(scale=0.0001, size=n_points)
    quality = np.zeros(n_points, dtype=int)
    row_present = np.ones(n_points, dtype=bool)
    usable = np.ones(n_points, dtype=bool)

    row_present[20:22] = False
    quality[100] = 1
    usable[20:22] = False
    usable[100] = False
    flux[20:22] = np.nan
    time[20:22] = np.nan

    return pd.DataFrame(
        {
            "time": time,
            "cadenceno": cadenceno,
            "normalized_flux": flux,
            "quality": quality,
            "row_present": row_present,
            "usable": usable,
        }
    )


def test_light_curve_characterization_records_phase_one_fields() -> None:
    frame = _synthetic_regularized_frame()

    diagnostics = characterize_regularized_light_curve(
        frame,
        target_id="11904151",
        quarter=5,
        preprocessing_summary={"quality_policy": "default"},
        acf_lags=40,
        ljung_box_lags=(5, 10),
        stationarity_min_observations=50,
        spectral_frequencies=800,
    )

    assert diagnostics["diagnostic_record_type"] == "light_curve_characterization_v1"
    assert diagnostics["target_id"] == "11904151"
    assert diagnostics["quarter"] == 5
    assert diagnostics["n_cadence_grid"] == len(frame)
    assert diagnostics["n_usable_observations"] == len(frame) - 3
    assert np.isclose(diagnostics["gap_fraction"], 3 / len(frame))
    assert diagnostics["quality_flag_fraction_observed"] > 0
    assert "original_adf_pvalue" in diagnostics
    assert "original_kpss_pvalue" in diagnostics
    assert "minimum_ljung_box_p" in diagnostics
    assert "acf_decay_e_days" in diagnostics
    assert "rolling_mean_range_over_robust_scale" in diagnostics
    assert "flux_excess_kurtosis" in diagnostics
    assert "spectral_entropy" in diagnostics
    assert abs(float(diagnostics["dominant_period_days"]) - 2.0) < 0.25


def test_structural_diagnostic_comparison_has_before_after_rows() -> None:
    rng = np.random.default_rng(321)
    white = rng.normal(size=300)
    correlated = np.empty(300)
    correlated[0] = white[0]
    for index in range(1, len(correlated)):
        correlated[index] = 0.8 * correlated[index - 1] + white[index]

    comparison = structural_diagnostic_comparison(
        {"raw": correlated, "whitened": white},
        cadence_days=0.020433,
        acf_lags=20,
        ljung_box_lags=(5, 10),
    )

    assert comparison["series"].tolist() == ["raw", "whitened"]
    assert comparison.loc[0, "max_abs_acf_1_n"] > comparison.loc[1, "max_abs_acf_1_n"]
    assert "rolling_variance_max_to_median" in comparison.columns
