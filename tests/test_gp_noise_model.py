import numpy as np

from adaptive_transit.injections.synthetic import local_depth_and_snr
from adaptive_transit.noise_models.gp import fit_smooth_gp_background

def test_gp_background_returns_residuals_and_anchor_mask():
    rng = np.random.default_rng(31)
    time = np.linspace(0.0, 20.0, 140)
    background = 0.003 * np.sin(2.0 * np.pi * time / 15.0)
    values = background + rng.normal(scale=0.0002, size=time.size)
    fitted = fit_smooth_gp_background(time, values, max_train_points=45, length_scale_days=3.0, min_length_scale_days=1.0, max_length_scale_days=20.0, measurement_noise_fraction=0.20, n_restarts_optimizer=0)

    assert fitted.background_mean.shape == values.shape
    assert fitted.residuals.shape == values.shape
    assert fitted.training_mask.sum() <= 45
    assert fitted.parameters["length_scale_days"] >= 1.0
    assert np.isfinite(fitted.log_marginal_likelihood)
    assert np.isfinite(fitted.residuals[fitted.usable_mask]).sum() > 100

def test_gp_background_keeps_missing_residuals_nan():
    time = np.linspace(0.0, 12.0, 90)
    values = 0.002 * np.sin(time)
    values[20:25] = np.nan
    time[60:63] = np.nan
    fitted = fit_smooth_gp_background(time, values, max_train_points=35, length_scale_days=2.5, min_length_scale_days=1.0, max_length_scale_days=12.0, measurement_noise_fraction=0.30, n_restarts_optimizer=0)

    assert np.isnan(fitted.residuals[20:25]).all()
    assert not fitted.usable_mask[20:25].any()
    assert np.isnan(fitted.background_mean[60:63]).all()
    assert np.isnan(fitted.residuals[60:63]).all()

def test_gp_residuals_preserve_simple_box_dip_with_smooth_kernel():
    rng = np.random.default_rng(41)
    cadenceno = np.arange(220)
    time = cadenceno * 0.0204
    background = 0.0015 * np.sin(2.0 * np.pi * time / 8.0)
    values = background + rng.normal(scale=0.00015, size=cadenceno.size)
    in_transit = (cadenceno >= 98) & (cadenceno <= 104)
    injected = values.copy()
    injected[in_transit] -= 0.0015
    fitted = fit_smooth_gp_background(time, injected, max_train_points=70, length_scale_days=2.0, min_length_scale_days=1.0, max_length_scale_days=12.0, measurement_noise_fraction=0.50, n_restarts_optimizer=0)
    before = local_depth_and_snr(cadenceno, injected, center_cadenceno=101, duration_cadences=7, local_half_width_cadences=24)
    after = local_depth_and_snr(cadenceno, fitted.residuals, center_cadenceno=101, duration_cadences=7, local_half_width_cadences=24)

    assert before["depth"] > 0
    assert after["depth"] > 0
    assert after["depth"] / before["depth"] > 0.25

def test_gp_fixed_kernel_keeps_requested_length_scale():
    rng = np.random.default_rng(43)
    time = np.linspace(0.0, 16.0, 120)
    values = 0.002 * np.sin(2.0 * np.pi * time / 10.0) + rng.normal(scale=0.00015, size=time.size)
    fitted = fit_smooth_gp_background(time, values, max_train_points=50, length_scale_days=0.75, min_length_scale_days=0.05, max_length_scale_days=12.0, measurement_noise_fraction=0.40, n_restarts_optimizer=0, optimize_kernel=False)

    assert fitted.parameters["optimize_kernel"] is False
    assert fitted.parameters["estimated_parameter_count"] == 0
    assert fitted.parameters["length_scale_days"] == 0.75
