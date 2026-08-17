import numpy as np
import pandas as pd

from adaptive_transit.injections.gap_mode_experiment import (
    empirical_false_alarm_thresholds,
    finite_scan_mask,
    spurious_peak_metrics,
    summarize_gap_mode_injections,
    template_amplitude_estimate,
    transformed_template_shape_metrics,
)
from adaptive_transit.injections.synthetic import TransitInjection
from adaptive_transit.transit_models.box import box_transit_template


def test_template_amplitude_estimate_recovers_known_scale() -> None:
    template = np.array([0.0, -1.0, -1.0, 0.0, 0.0])
    observed = 2.5 * template + 0.1
    scale = np.ones(template.shape)
    usable = np.ones(template.shape, dtype=bool)

    amplitude = template_amplitude_estimate(observed, template, scale=scale, usable_mask=usable)

    assert np.isclose(amplitude, 2.5)


def test_transformed_template_shape_metrics_flags_edge_dominated_signal() -> None:
    cadence = np.arange(20)
    injection = TransitInjection(center_cadenceno=10, duration_cadences=6, depth=0.001)
    template, _ = box_transit_template(cadence, center_cadenceno=10, duration_cadences=6, depth=0.001)
    transformed = np.diff(np.r_[0.0, template])
    usable = np.ones(cadence.shape, dtype=bool)

    metrics = transformed_template_shape_metrics(
        cadence,
        transformed,
        injection,
        local_half_width_cadences=8,
        usable_mask=usable,
    )

    assert metrics["template_edge_abs_fraction"] > 0.9
    assert metrics["ingress_egress_distortion_fraction"] > 0.9


def test_empirical_false_alarm_thresholds_use_upper_quantiles() -> None:
    null_scan = pd.DataFrame({"innovation_transformed_template_statistic": np.arange(100, dtype=float)})

    thresholds = empirical_false_alarm_thresholds(null_scan, (0.10, 0.01))

    assert np.isclose(thresholds["threshold_far_0.1"], 89.1)
    assert np.isclose(thresholds["threshold_far_0.01"], 98.01)


def test_spurious_peak_metrics_compare_outside_to_injected_neighborhood() -> None:
    scan = pd.DataFrame(
        {
            "innovation_transformed_template_statistic": [8.0, 3.0, 9.0],
            "innovation_transformed_template_rank": [2, 3, 1],
            "is_injected_center_neighborhood": [True, False, False],
        }
    )

    metrics = spurious_peak_metrics(scan)

    assert metrics["best_injected_neighborhood_statistic"] == 8.0
    assert metrics["best_spurious_statistic"] == 9.0
    assert metrics["spurious_peak_exceeds_injected"]
    assert metrics["n_spurious_peaks_positive"] == 2


def test_summary_reports_recovery_rates_by_false_alarm_threshold() -> None:
    results = pd.DataFrame(
        {
            "gap_mode": ["full_grid_missing", "full_grid_missing", "longest_contiguous"],
            "depth_retention_fraction": [0.8, 0.6, 0.5],
            "snr_retention_fraction": [0.7, 0.5, 0.4],
            "ingress_egress_distortion_fraction": [0.2, 0.3, 0.9],
            "best_spurious_statistic": [2.0, 3.0, 4.0],
            "spurious_peak_exceeds_injected": [False, True, True],
            "detected_at_far_0.1": [True, False, True],
            "top_recovered_at_far_0.1": [True, False, False],
            "n_spurious_peaks_above_far_0.1": [0, 1, 2],
        }
    )

    summary = summarize_gap_mode_injections(results, (0.10,))

    full = summary.loc[summary["gap_mode"] == "full_grid_missing"].iloc[0]
    assert full["n_injections"] == 2
    assert full["recovery_rate_at_far_0.1"] == 0.5
    assert full["top_recovery_rate_at_far_0.1"] == 0.5


def test_finite_scan_mask_uses_interpolated_or_observed_finite_values() -> None:
    values = np.array([1.0, np.nan, 2.0])

    assert finite_scan_mask(values).tolist() == [True, False, True]
