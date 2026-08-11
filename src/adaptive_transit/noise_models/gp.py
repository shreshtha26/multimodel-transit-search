"""Smooth Gaussian Process background model for normalized Kepler light curves.
The baseline estimates long-timescale covariance structure and exposes residuals for transit search."""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

class FittedGaussianProcessModel:
    """Container for one fitted smooth GP background model."""
    def __init__(self, model_name, parameters, background_mean, background_std, residuals, standardized_residuals, usable_mask, training_mask, log_marginal_likelihood, converged, status, message):
        self.model_name = model_name
        self.parameters = parameters
        self.background_mean = background_mean
        self.background_std = background_std
        self.residuals = residuals
        self.standardized_residuals = standardized_residuals
        self.usable_mask = usable_mask
        self.training_mask = training_mask
        self.log_marginal_likelihood = log_marginal_likelihood
        self.converged = converged
        self.status = status
        self.message = message

    def summary(self):
        finite_count = int(np.isfinite(self.residuals).sum())
        parameter_count = int(self.parameters.get("estimated_parameter_count", 0))
        aic = float(2 * parameter_count - 2 * self.log_marginal_likelihood) if np.isfinite(self.log_marginal_likelihood) else float("nan")
        bic = float(parameter_count * np.log(finite_count) - 2 * self.log_marginal_likelihood) if finite_count > 0 and np.isfinite(self.log_marginal_likelihood) else float("nan")
        return {"model_name": self.model_name, "kernel_family": "constant_times_rbf", "observation_equation": "normalized_flux_t = smooth_gp_background(time_t) + measurement_noise", "missing_cadence_policy": "train_and_residuals_only_on_finite_observations_no_interpolation", "background_policy": "two_sided_gp_posterior_mean", "anchor_point_approximation": True, "converged": bool(self.converged), "status": str(self.status), "message": str(self.message), "log_marginal_likelihood": float(self.log_marginal_likelihood), "aic": aic, "bic": bic, "finite_residual_count": finite_count, **{key: json_float(value) for key, value in self.parameters.items()}}

def json_float(value):
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)

def finite_values(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.isfinite(values)]

def robust_scale(values):
    clean = finite_values(values)
    if clean.size < 2:
        raise ValueError("At least two finite values are required.")
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    scale = 1.4826 * mad if mad > 0 else float(np.std(clean, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(clean, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Cannot estimate a positive GP scale from the input series.")
    return float(scale)

def validate_time_and_values(time, values):
    time = np.asarray(time, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    if time.shape != values.shape:
        raise ValueError("time and values must have the same shape.")
    if np.isinf(time).any() or np.isinf(values).any():
        raise ValueError("GP input may contain NaN gaps, but not infinite values.")
    finite = np.isfinite(time) & np.isfinite(values)
    if finite.sum() < 20:
        raise ValueError("GP fitting needs at least 20 finite observations.")
    return time, values, finite

def select_anchor_indices(time, values, max_train_points=512):
    time, values, finite = validate_time_and_values(time, values)
    finite_indices = np.flatnonzero(finite)
    ordered_indices = finite_indices[np.argsort(time[finite_indices])]
    if ordered_indices.size <= int(max_train_points):
        return ordered_indices
    positions = np.unique(np.round(np.linspace(0, ordered_indices.size - 1, int(max_train_points))).astype(int))
    return ordered_indices[positions]

def kernel_parameter_summary(kernel):
    params = kernel.get_params()
    signal_variance = params.get("k1__constant_value", float("nan"))
    length_scale = params.get("k2__length_scale", float("nan"))
    return float(signal_variance), float(np.asarray(length_scale).reshape(-1)[0])

def fit_smooth_gp_background(time, values, max_train_points=512, length_scale_days=3.0, min_length_scale_days=1.0, max_length_scale_days=30.0, measurement_noise_fraction=0.20, n_restarts_optimizer=0, random_seed=123, optimize_kernel=True):
    """Fit a smooth anchor-point GP and return background residuals.
    The lower length-scale bound is intentionally longer than the transit durations in the baseline injection grid."""
    time, values, finite = validate_time_and_values(time, values)
    if max_train_points < 20:
        raise ValueError("max_train_points must be at least 20.")
    if min_length_scale_days <= 0 or max_length_scale_days <= min_length_scale_days:
        raise ValueError("length-scale bounds must satisfy 0 < min < max.")
    if length_scale_days < min_length_scale_days or length_scale_days > max_length_scale_days:
        raise ValueError("length_scale_days must lie inside the supplied bounds.")
    if measurement_noise_fraction <= 0:
        raise ValueError("measurement_noise_fraction must be positive.")
    anchor_indices = select_anchor_indices(time, values, max_train_points=max_train_points)
    training_mask = np.zeros(time.shape, dtype=bool)
    training_mask[anchor_indices] = True
    t0 = float(np.median(time[finite]))
    y_offset = float(np.median(values[finite]))
    y_train = values[anchor_indices] - y_offset
    scale = robust_scale(y_train)
    signal_variance_start = float(scale * scale)
    measurement_variance = float(max(signal_variance_start * float(measurement_noise_fraction), np.finfo(float).eps))
    signal_lower = max(signal_variance_start * 1.0e-4, np.finfo(float).eps)
    signal_upper = max(signal_variance_start * 1.0e4, signal_lower * 10.0)
    kernel = ConstantKernel(signal_variance_start, constant_value_bounds=(signal_lower, signal_upper)) * RBF(float(length_scale_days), length_scale_bounds=(float(min_length_scale_days), float(max_length_scale_days)))
    optimizer = "fmin_l_bfgs_b" if optimize_kernel else None
    gp = GaussianProcessRegressor(kernel=kernel, alpha=measurement_variance, normalize_y=False, optimizer=optimizer, n_restarts_optimizer=int(n_restarts_optimizer), random_state=int(random_seed))
    x_train = (time[anchor_indices] - t0).reshape(-1, 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        gp.fit(x_train, y_train)
    warning_messages = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    prediction_mask = np.isfinite(time)
    background_mean = np.full(time.shape, np.nan, dtype=float)
    background_std = np.full(time.shape, np.nan, dtype=float)
    if prediction_mask.any():
        x_predict = (time[prediction_mask] - t0).reshape(-1, 1)
        mean, std = gp.predict(x_predict, return_std=True)
        background_mean[prediction_mask] = mean + y_offset
        background_std[prediction_mask] = std
    residuals = np.full(values.shape, np.nan, dtype=float)
    residuals[finite] = values[finite] - background_mean[finite]
    standardized = np.full(values.shape, np.nan, dtype=float)
    denominator = np.sqrt(background_std * background_std + measurement_variance)
    usable_standardized = finite & np.isfinite(denominator) & (denominator > 0)
    standardized[usable_standardized] = residuals[usable_standardized] / denominator[usable_standardized]
    signal_variance, fitted_length_scale = kernel_parameter_summary(gp.kernel_)
    estimated_parameter_count = 2 if optimize_kernel else 0
    parameters = {"max_train_points": int(max_train_points), "training_point_count": int(anchor_indices.size), "time_origin_days": t0, "flux_offset": y_offset, "initial_signal_variance": signal_variance_start, "signal_variance": signal_variance, "initial_length_scale_days": float(length_scale_days), "length_scale_days": fitted_length_scale, "min_length_scale_days": float(min_length_scale_days), "max_length_scale_days": float(max_length_scale_days), "measurement_noise_variance": measurement_variance, "measurement_noise_fraction": float(measurement_noise_fraction), "optimize_kernel": bool(optimize_kernel), "estimated_parameter_count": estimated_parameter_count, "n_restarts_optimizer": int(n_restarts_optimizer)}
    converged = len(warning_messages) == 0
    status = 0 if converged else 1
    message = "converged" if converged else "; ".join(warning_messages)
    return FittedGaussianProcessModel("smooth_anchor_gp", parameters, background_mean, background_std, residuals, standardized, finite.copy(), training_mask, float(gp.log_marginal_likelihood_value_), converged, status, message)
