import numpy as np
import pandas as pd

from adaptive_transit.noise_models.stellar_variability import (
    CANONICAL_CHARACTERIZATION_COLUMNS,
    CANONICAL_CHARACTERIZATION_SCHEMA,
    apply_population_variability_boundaries,
    assign_dominant_statistical_behaviour,
)


def _population_frame(harmonic_ratio: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flux_robust_scale": [1.0, 2.0, 3.0, 4.0],
            "flux_skewness": [0.0, 0.1, 0.2, 0.3],
            "flux_outlier_fraction": [0.0, 0.0, 0.0, 0.01],
            "v2_acf_lag_1": [0.05, 0.10, 0.15, 0.30],
            "v2_acf_max_abs": [0.10, 0.20, 0.30, 0.40],
            "v2_acf_decay_e_days": [0.02, 0.02, 0.04, 0.20],
            "v2_ls_dominant_period_days": [2.0, 3.0, 4.0, 10.0],
            "v2_ls_acf_period_relative_error": [np.nan, np.nan, np.nan, 0.05],
            "v2_spectral_concentration": [0.01, 0.02, 0.03, 0.20],
            "v2_spectral_harmonic_power_ratio": [0.01, 0.02, 0.03, harmonic_ratio],
            "v2_segment_scale_relative_mad": [0.01, 0.02, 0.03, 0.90],
            "v2_segment_median_range_over_global_scale": [0.10, 0.10, 0.10, 0.90],
            "v2_coherent_periodic_candidate": [False, False, False, True],
            "v2_periodicity_screen_pass": [False, False, False, True],
            "v2_periodicity_supported_by_acf": [False, False, False, True],
            "v2_half_period_consistency_fraction": [1.0, 1.0, 1.0, 0.5],
            "spectral_power_fraction_period_lt_0_5d": [0.0, 0.0, 0.0, 0.0],
            "spectral_power_fraction_period_0_5_to_2d": [0.0, 0.0, 0.0, 0.0],
            "original_series_stationarity_conclusion": [
                "stationary_supported",
                "stationary_supported",
                "stationary_supported",
                "stationary_supported",
            ],
        }
    )


def test_frozen_canonical_schema_has_seven_domains_and_eleven_variables():
    assert len(CANONICAL_CHARACTERIZATION_COLUMNS) == 11
    assert len({item[0] for item in CANONICAL_CHARACTERIZATION_SCHEMA}) == 7


def test_quasi_periodic_flag_is_persisted_and_rotation_requires_harmonic_support():
    strong_harmonic, _ = apply_population_variability_boundaries(
        _population_frame(harmonic_ratio=0.20)
    )
    row = strong_harmonic.iloc[-1]
    assert bool(row["v2_quasi_periodic_candidate"])
    assert bool(row["v2_rotation_spot_review_flag"])

    weak_harmonic, _ = apply_population_variability_boundaries(
        _population_frame(harmonic_ratio=0.05)
    )
    row = weak_harmonic.iloc[-1]
    assert bool(row["v2_quasi_periodic_candidate"])
    assert not bool(row["v2_rotation_spot_review_flag"])


def test_dominant_behaviour_prefers_quasi_periodic_when_flags_overlap():
    profiled, _ = apply_population_variability_boundaries(
        _population_frame(harmonic_ratio=0.20)
    )
    labelled = assign_dominant_statistical_behaviour(profiled)
    assert labelled.iloc[-1]["v2_dominant_statistical_behaviour"] == "Quasi-periodic / structured"
