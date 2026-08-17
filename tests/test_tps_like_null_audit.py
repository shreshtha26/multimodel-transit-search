import numpy as np
import pandas as pd
import pytest

from adaptive_transit.tps_null_audit import (
    build_zero_injection_table,
    compare_zero_to_injected,
    dominant_period_cluster,
)


def _raw_zero(realized_depth=0.0, zero_flag=True):
    return pd.DataFrame(
        [
            {
                "target_id": "11245408",
                "quarter": 5,
                "sample_stratum": "long_memory",
                "success": True,
                "zero_injection_control": zero_flag,
                "requested_depth": 0.0,
                "recovered_period_days": 2.9425,
                "recovered_epoch_days": 1.0,
                "recovered_duration_hours": 4.0,
                "mes": 410.0,
                "max_ses": 389.0,
                "observed_event_count": 7,
                "expected_event_count": 32,
                "observability_fraction": 7 / 32,
                "runtime_seconds": 0.4,
                "realized_max_depth_on_observed_cadences": realized_depth,
            }
        ]
    )


def test_dominant_period_cluster_finds_persistent_winner():
    cluster = dominant_period_cluster(
        [2.9425, 2.9425, 2.9426, 8.0, 9.0, 10.0], tolerance_fraction=0.02
    )
    assert cluster.count == 3
    assert cluster.fraction == pytest.approx(0.5)
    assert cluster.period_days == pytest.approx(2.9425, rel=1e-4)


def test_zero_control_requires_explicit_zero_injection_flag():
    with pytest.raises(ValueError, match="--zero-injection"):
        build_zero_injection_table(_raw_zero(zero_flag=False))


def test_zero_control_rejects_nonzero_realized_depth():
    with pytest.raises(ValueError, match="not sufficiently null"):
        build_zero_injection_table(_raw_zero(realized_depth=1e-5), max_realized_depth=1e-12)


def test_zero_control_is_labeled_as_true_original_flux_baseline():
    zero = build_zero_injection_table(_raw_zero())
    assert zero.loc[0, "baseline_kind"] == "true_zero_injection_original_flux"
    assert zero.loc[0, "zero_injection_requested_depth"] == pytest.approx(0.0)
    assert zero.loc[0, "zero_injection_realized_depth"] == pytest.approx(0.0)


def test_compare_flags_zero_matching_persistent_period():
    zero = build_zero_injection_table(_raw_zero(realized_depth=0.0))
    injected = pd.DataFrame(
        [
            {
                "target_id": "11245408",
                "quarter": 5,
                "sample_stratum": "long_memory",
                "success": True,
                "injected_period_days": truth,
                "recovered_period_days": recovered,
                "mes": 413.0,
                "max_ses": 389.0,
                "harmonic_period_recovered": False,
            }
            for truth, recovered in [
                (2.0, 2.9425),
                (2.0, 2.9425),
                (5.0, 2.9425),
                (5.0, 2.9426),
                (10.0, 2.9425),
                (10.0, 2.9425),
            ]
        ]
    )
    cases, stars = compare_zero_to_injected(zero, injected, tolerance_fraction=0.02)
    assert cases["winner_matches_null_period"].all()
    assert bool(stars.loc[0, "persistent_injected_period_flag"])
    assert bool(stars.loc[0, "null_matches_persistent_period"])
    assert stars.loc[0, "persistent_cluster_truth_count"] == 3
