from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve


def test_preprocess_keeps_nan_flux_error_by_default() -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "flux": [100.0, 101.0, np.nan, 99.0],
            "flux_error": [0.1, np.nan, 0.2, 0.1],
            "quality": [0, 0, 0, 1],
            "cadenceno": [10, 11, 12, 13],
        }
    )

    regular, summary = preprocess_pdcsap_light_curve(frame)

    assert len(regular) == 4
    assert regular["usable"].tolist() == [True, True, False, False]
    assert summary.n_raw == 4
    assert summary.n_usable == 2
    assert regular["finite_flux_error"].tolist() == [True, False, True, True]


def test_preprocess_normalizes_flux_around_zero() -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0],
            "flux": [98.0, 100.0, 102.0],
            "flux_error": [0.1, 0.1, 0.1],
            "quality": [0, 0, 0],
            "cadenceno": [20, 21, 22],
        }
    )

    regular, _ = preprocess_pdcsap_light_curve(frame)

    np.testing.assert_allclose(regular["normalized_flux"], [-0.02, 0.0, 0.02])


def test_preprocess_keeps_absent_cadence_explicit() -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 3.0],
            "flux": [100.0, 101.0],
            "flux_error": [0.1, 0.1],
            "quality": [0, 0],
            "cadenceno": [30, 32],
        }
    )

    regular, summary = preprocess_pdcsap_light_curve(frame)

    assert regular["cadenceno"].tolist() == [30, 31, 32]
    assert regular["row_present"].tolist() == [True, False, True]
    assert regular["gap_reason"].tolist() == ["usable", "cadence_absent_from_file", "usable"]
    assert np.isnan(regular.loc[1, "normalized_flux"])
    assert summary.n_row_absent == 1


def test_normalization_uses_only_fit_fraction() -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0],
            "flux": [98.0, 100.0, 1000.0],
            "flux_error": [0.1, 0.1, 0.1],
            "quality": [0, 0, 0],
            "cadenceno": [40, 41, 42],
        }
    )

    regular, summary = preprocess_pdcsap_light_curve(frame, normalization_fit_fraction=2 / 3)

    assert summary.median_flux == 99.0
    assert summary.normalization_fit_count == 2
    np.testing.assert_allclose(regular.loc[:1, "normalized_flux"], [-1 / 99, 1 / 99])


def test_quality_policy_default_keeps_some_nonzero_flags() -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0],
            "flux": [100.0, 101.0],
            "flux_error": [0.1, 0.1],
            "quality": [0, 16],
            "cadenceno": [50, 51],
        }
    )

    strict, strict_summary = preprocess_pdcsap_light_curve(frame, quality_policy="strict")
    default, default_summary = preprocess_pdcsap_light_curve(frame, quality_policy="default")

    assert strict["usable"].tolist() == [True, False]
    assert default["usable"].tolist() == [True, True]
    assert strict_summary.quality_policy == "strict"
    assert default_summary.quality_policy == "default"
