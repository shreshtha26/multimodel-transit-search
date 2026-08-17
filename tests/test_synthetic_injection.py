import numpy as np

from adaptive_transit.injections.synthetic import (
    TransitInjection,
    choose_injection_centers,
    inject_box_transit,
    inject_periodic_box_transit,
    transit_preservation_metrics,
)


def test_box_transit_injection_adds_negative_dip() -> None:
    values = np.zeros(20)
    cadenceno = np.arange(20)

    injected, template, in_transit = inject_box_transit(
        values,
        cadenceno,
        center_cadenceno=10,
        duration_cadences=4,
        depth=0.01,
    )

    assert in_transit.sum() == 4
    assert np.all(template[in_transit] == -0.01)
    assert np.all(injected[in_transit] == -0.01)
    assert np.all(injected[~in_transit] == 0.0)


def test_transit_preservation_metrics_reports_retention() -> None:
    cadenceno = np.arange(60)
    rng = np.random.default_rng(99)
    injected_flux = rng.normal(scale=0.0005, size=60)
    injected_flux[28:33] -= 0.01
    innovations = injected_flux.copy()
    standardized = injected_flux / 0.001
    injection = TransitInjection(center_cadenceno=30, duration_cadences=6, depth=0.01)

    metrics = transit_preservation_metrics(
        cadenceno,
        injected_flux,
        innovations,
        standardized,
        injection,
        local_half_width_cadences=12,
    )

    assert metrics["depth_retention_fraction"] > 0.8
    assert metrics["local_snr_after_arima"] > 0
    assert metrics["standardized_snr_after_arima"] > 0


def test_choose_injection_centers_uses_long_clean_segments() -> None:
    import pandas as pd

    regular = pd.DataFrame(
        {
            "cadenceno": range(20),
            "segment_id": [0] * 12 + [-1] + [1] * 7,
        }
    )

    centers = choose_injection_centers(
        regular,
        duration_cadences=3,
        centers_per_segment=2,
        max_segments=1,
    )

    assert len(centers) == 2
    assert all(0 <= center <= 11 for center in centers)


def test_periodic_box_injection_preserves_nan_gaps():
    time = np.linspace(0.0, 10.0, 200)
    values = np.zeros(time.size)
    values[20] = np.nan
    injected, template, in_transit = inject_periodic_box_transit(time, values, period_days=3.0, epoch_days=1.0, duration_days=0.2, depth=0.01)
    assert np.isnan(injected[20])
    assert np.nanmin(injected) < 0.0
    assert template.shape == values.shape
    assert in_transit.shape == values.shape
