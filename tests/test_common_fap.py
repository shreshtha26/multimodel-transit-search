import numpy as np

from adaptive_transit.detection.common_fap import (
    calibration_row,
    empirical_p_value,
    empirical_threshold,
)


def test_one_percent_higher_quantile_is_conservative():
    scores = np.arange(1.0, 101.0)
    threshold = empirical_threshold(scores, fap_level=0.01)
    assert threshold == 100.0
    assert np.mean(scores >= threshold) == 0.01


def test_empirical_p_value_uses_plus_one_correction():
    scores = np.arange(1.0, 101.0)
    p = empirical_p_value(101.0, scores)
    assert np.isclose(p, 1.0 / 101.0)


def test_calibration_row_records_requested_and_achieved_trials():
    scores = np.arange(1.0, 101.0)
    row = calibration_row(
        scores,
        method="demo",
        score_name="score",
        fap_level=0.01,
        requested_trials=110,
    )
    assert row["successful_null_trials"] == 100
    assert row["requested_null_trials"] == 110
    assert np.isclose(row["success_fraction"], 100 / 110)
    assert row["score_threshold"] == 100.0
