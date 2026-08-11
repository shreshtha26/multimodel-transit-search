import numpy as np
from adaptive_transit.noise_models.gp import apply_prepared_smooth_gp_filter, fit_smooth_gp_background, prepare_smooth_gp_filter
from adaptive_transit.noise_models.kalman import apply_fitted_kalman_filter, fit_kalman_local_level

def test_prepared_gp_filter_reproduces_base_fixed_kernel_fit():
    rng = np.random.default_rng(12)
    time = np.linspace(0.0, 20.0, 240)
    values = 0.001 * np.sin(time / 3.0) + rng.normal(0.0, 0.0002, size=time.size)
    values[50:55] = np.nan
    fitted = fit_smooth_gp_background(time, values, max_train_points=64, length_scale_days=3.0, min_length_scale_days=1.0, max_length_scale_days=10.0, measurement_noise_fraction=0.20, optimize_kernel=False)
    prepared = prepare_smooth_gp_filter(time, fitted)
    reapplied = apply_prepared_smooth_gp_filter(values, prepared)
    np.testing.assert_allclose(reapplied.background_mean, fitted.background_mean, rtol=1.0e-10, atol=1.0e-12, equal_nan=True)
    np.testing.assert_allclose(reapplied.residuals, fitted.residuals, rtol=1.0e-10, atol=1.0e-12, equal_nan=True)
    assert np.isclose(reapplied.log_marginal_likelihood, fitted.log_marginal_likelihood, rtol=1.0e-10, atol=1.0e-10)

def test_prepared_gp_filter_reuses_same_operator_for_injected_flux():
    rng = np.random.default_rng(13)
    time = np.linspace(0.0, 15.0, 180)
    values = 0.001 * np.sin(time / 2.5) + rng.normal(0.0, 0.00015, size=time.size)
    fitted = fit_smooth_gp_background(time, values, max_train_points=48, length_scale_days=2.5, min_length_scale_days=1.0, max_length_scale_days=8.0, measurement_noise_fraction=0.20, optimize_kernel=False)
    prepared = prepare_smooth_gp_filter(time, fitted)
    injected = values.copy()
    injected[(time > 5.0) & (time < 5.2)] -= 0.0008
    result = apply_prepared_smooth_gp_filter(injected, prepared)
    assert result.parameters["injection_mode"] == "fixed_base_hyperparameters_and_operator"
    assert np.isfinite(result.residuals).sum() == np.isfinite(injected).sum()
    assert result.parameters["length_scale_days"] == fitted.parameters["length_scale_days"]
    assert result.parameters["measurement_noise_variance"] == fitted.parameters["measurement_noise_variance"]

def test_kalman_apply_reuses_base_variance_parameters():
    rng = np.random.default_rng(14)
    values = np.cumsum(rng.normal(0.0, 0.00002, size=160)) + rng.normal(0.0, 0.0001, size=160)
    base = fit_kalman_local_level(values, maxiter=30, burn_in=1)
    injected = values.copy()
    injected[60:63] -= 0.0007
    result = apply_fitted_kalman_filter(injected, base, burn_in=1)
    assert result.parameters["process_variance"] == base.parameters["process_variance"]
    assert result.parameters["measurement_variance"] == base.parameters["measurement_variance"]
    assert result.message == "filtered with supplied parameters"
