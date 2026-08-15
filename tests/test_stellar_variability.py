import numpy as np
import pandas as pd

from adaptive_transit.noise_models.stellar_variability import (
    MODEL_SELECTION_FEATURE_COLUMNS,
    apply_population_variability_boundaries,
    model_selection_feature_frame,
    pairwise_regular_grid_acf,
    stellar_variability_summary,
)


def test_gap_preserving_acf_uses_true_cadence_lags() -> None:
    rng = np.random.default_rng(11)
    values = np.empty(800)
    values[0] = rng.normal()
    for i in range(1, len(values)):
        values[i] = 0.85 * values[i - 1] + rng.normal(scale=0.5)
    values[100:120] = np.nan
    values[400:410] = np.nan

    summary = pairwise_regular_grid_acf(values, cadence_days=0.020433, max_lag=80)
    assert summary["v2_acf_lag_1"] > 0.6
    assert summary["v2_acf_integrated_positive_days"] > 0


def test_periodic_signal_is_detected_as_candidate() -> None:
    rng = np.random.default_rng(12)
    cadence = 0.020433
    time = np.arange(2500) * cadence
    period = 2.0
    flux = 0.0015 * np.sin(2.0 * np.pi * time / period) + rng.normal(scale=0.00015, size=len(time))
    flux[500:515] = np.nan
    time_with_gap = time.copy()
    time_with_gap[500:515] = np.nan

    summary = stellar_variability_summary(
        time_with_gap,
        flux,
        cadence_days=cadence,
        acf_lags=500,
        spectral_frequencies=3000,
    )
    assert abs(summary["v2_ls_dominant_period_days"] - period) < 0.1
    assert summary["v2_periodicity_screen_pass"]
    assert summary["v2_coherent_periodic_candidate"]


def test_low_scatter_periodic_star_is_not_forced_to_quiet() -> None:
    # Population labels intentionally need more than amplitude.  The first row
    # has the smallest amplitude but is periodic, so it must be labelled as
    # low-scatter structured rather than quiet.
    frame = pd.DataFrame(
        {
            "target_id": ["A", "B", "C", "D", "E"],
            # With five rows, Q25 is exactly row B.  A and B are therefore the
            # low-amplitude population, but A is structured/periodic while B
            # is deliberately unstructured.  This tests the key distinction.
            "flux_robust_scale": [1e-4, 1.2e-4, 5e-4, 8e-4, 9e-4],
            "v2_acf_max_abs": [0.60, 0.05, 0.20, 0.80, 0.90],
            "v2_acf_lag_1": [0.60, 0.03, 0.15, 0.75, 0.85],
            "v2_segment_scale_relative_mad": [0.05, 0.02, 0.10, 0.40, 0.50],
            "v2_segment_median_range_over_global_scale": [0.05, 0.03, 0.10, 0.50, 0.60],
            "v2_coherent_periodic_candidate": [True, False, False, False, False],
            "v2_periodicity_screen_pass": [True, False, False, False, False],
            "v2_periodicity_supported_by_acf": [True, False, False, False, False],
            "v2_half_period_consistency_fraction": [1.0, np.nan, np.nan, np.nan, np.nan],
            "v2_spectral_harmonic_power_ratio": [0.3, 0.0, 0.0, 0.0, 0.0],
            "v2_pulsation_review_flag": [False, False, False, False, False],
        }
    )

    profiles, thresholds = apply_population_variability_boundaries(frame)
    assert thresholds["amplitude_q25"] > 0
    assert bool(profiles.loc[0, "v2_low_scatter_structured_candidate"])
    assert not bool(profiles.loc[0, "v2_quiet_candidate"])
    assert bool(profiles.loc[1, "v2_quiet_candidate"])


def test_model_selection_features_exclude_gap_metrics_and_labels() -> None:
    frame = pd.DataFrame({"target_id": ["1"], "gap_fraction": [0.2], "flux_robust_scale": [0.001]})
    features = model_selection_feature_frame(frame)
    assert tuple(features.columns) == MODEL_SELECTION_FEATURE_COLUMNS
    assert "gap_fraction" not in features.columns
    assert "v2_quiet_candidate" not in features.columns
