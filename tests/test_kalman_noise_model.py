import numpy as np

from adaptive_transit.injections.synthetic import local_depth_and_snr
from adaptive_transit.noise_models.kalman import fit_kalman_local_level

def test_kalman_local_level_returns_positive_parameters_and_residuals():
    rng = np.random.default_rng(11)
    background = np.cumsum(rng.normal(scale=0.001, size=160))
    values = background + rng.normal(scale=0.002, size=160)
    fitted = fit_kalman_local_level(values, maxiter=80)

    assert fitted.parameters["process_variance"] > 0
    assert fitted.parameters["measurement_variance"] > 0
    assert fitted.predicted_background.shape == values.shape
    assert fitted.residuals.shape == values.shape
    assert np.isfinite(fitted.log_likelihood)
    assert np.isfinite(fitted.residuals[fitted.usable_mask]).sum() > 100

def test_kalman_local_level_skips_missing_updates_without_interpolation():
    values = np.linspace(0.0, 0.01, 80)
    values[20:25] = np.nan
    fitted = fit_kalman_local_level(values, process_variance=1.0e-7, measurement_variance=1.0e-6, estimate_parameters=False)

    assert np.isnan(fitted.residuals[20:25]).all()
    assert np.isfinite(fitted.predicted_background[20:25]).all()
    assert np.isfinite(fitted.filtered_background[20:25]).all()
    assert not fitted.usable_mask[20:25].any()

def test_kalman_residuals_preserve_simple_box_dip():
    rng = np.random.default_rng(21)
    cadenceno = np.arange(220)
    background = 0.002 * np.sin(np.linspace(0.0, 3.0 * np.pi, cadenceno.size))
    values = background + rng.normal(scale=0.0002, size=cadenceno.size)
    in_transit = (cadenceno >= 98) & (cadenceno <= 104)
    injected = values.copy()
    injected[in_transit] -= 0.002
    fitted = fit_kalman_local_level(injected, maxiter=80)
    before = local_depth_and_snr(cadenceno, injected, center_cadenceno=101, duration_cadences=7, local_half_width_cadences=24)
    after = local_depth_and_snr(cadenceno, fitted.residuals, center_cadenceno=101, duration_cadences=7, local_half_width_cadences=24)

    assert before["depth"] > 0
    assert after["depth"] > 0
