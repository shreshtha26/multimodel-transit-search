from __future__ import annotations

import numpy as np

from adaptive_transit.detection.matched_filter import (
    arima_transformed_template,
    matched_filter_statistic,
    scan_arima_transformed_template,
    select_trial_centers,
)
from adaptive_transit.injections.synthetic import inject_box_transit
from adaptive_transit.noise_models.arima import fit_arima_model


def test_matched_filter_statistic_is_positive_for_matching_dip() -> None:
    template = np.zeros(40)
    template[18:22] = -0.01
    observed = template.copy()
    scale = np.full(template.shape, 0.001)

    score = matched_filter_statistic(observed, template, scale=scale)
    anti_score = matched_filter_statistic(-observed, template, scale=scale)

    assert score.statistic > 0
    assert anti_score.statistic < 0
    assert score.n_points == 40


def test_arima_transformed_template_uses_fixed_model_coefficients() -> None:
    rng = np.random.default_rng(42)
    values = np.zeros(90)
    noise = rng.normal(scale=0.001, size=values.size)
    for index in range(1, values.size):
        values[index] = 0.6 * values[index - 1] + noise[index]

    fitted = fit_arima_model(values, (1, 0, 0), allow_missing=False)
    template = np.zeros(values.shape)
    template[43:47] = -0.002

    transform = arima_transformed_template(
        values,
        template,
        fitted,
        allow_missing=False,
    )
    score = matched_filter_statistic(
        transform.injected_innovations,
        transform.transformed_template,
        usable_mask=transform.usable_mask,
    )

    assert transform.transformed_template.shape == values.shape
    assert np.nanmax(np.abs(transform.transformed_template)) > 0
    assert transform.usable_mask.sum() > 80
    assert score.statistic > 0


def test_select_trial_centers_keeps_required_center() -> None:
    cadenceno = np.arange(20)
    usable = np.ones(20, dtype=bool)

    centers = select_trial_centers(
        cadenceno,
        usable,
        stride=4,
        max_centers=3,
        required_centers=(11,),
    )

    assert 11 in centers
    assert len(centers) <= 4


def test_template_scan_ranks_injected_center_highest() -> None:
    rng = np.random.default_rng(7)
    cadenceno = np.arange(120)
    values = rng.normal(scale=0.0002, size=cadenceno.size)
    fitted = fit_arima_model(values, (1, 0, 0), allow_missing=False)

    observed, _, _ = inject_box_transit(
        values,
        cadenceno,
        center_cadenceno=60,
        duration_cadences=6,
        depth=0.004,
    )
    scan = scan_arima_transformed_template(
        cadenceno,
        values,
        observed,
        fitted,
        np.asarray([35, 50, 60, 75, 95]),
        duration_cadences=6,
        depth=0.004,
        local_half_width_cadences=14,
        scale_window=16,
        allow_missing=False,
        injected_center_cadenceno=60,
    )

    best = scan.sort_values(
        "innovation_transformed_template_statistic",
        ascending=False,
    ).iloc[0]
    assert int(best["trial_center_cadenceno"]) == 60
    assert int(best["rank"]) == 1
