import numpy as np

from adaptive_transit.injections.batman import inject_batman_transit


def test_batman_injection_is_physical_and_additive():
    cadence = 29.4244 / 60.0 / 24.0
    time = np.arange(0.0, 20.0, cadence)
    flux = np.zeros_like(time)
    injected, template, in_transit, truth = inject_batman_transit(
        time,
        flux,
        period_days=5.0,
        epoch_days=2.0,
        duration_days=4.0 / 24.0,
        depth=5.0e-4,
        impact_parameter=0.3,
        supersample_factor=7,
        exposure_time_days=cadence,
    )
    assert np.allclose(injected, template)
    assert in_transit.sum() > 0
    assert np.nanmin(template) < 0
    assert truth.radius_ratio > 0
    assert truth.scaled_semimajor_axis > 1
    assert 80 < truth.inclination_degrees <= 90
    assert abs(truth.realized_max_depth_on_observed_cadences - 5.0e-4) < 1.5e-4
    active_values = template[in_transit]
    # Limb darkening + ingress/egress means the signal is not a two-level box.
    assert np.unique(np.round(active_values, 10)).size > 4
