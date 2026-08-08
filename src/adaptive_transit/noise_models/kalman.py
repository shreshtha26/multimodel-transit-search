"""Simple local-level Kalman background model for normalized Kepler light curves.
The model estimates a drifting background level and exposes one-step residuals for transit search."""

import numpy as np
from scipy.optimize import minimize

class FittedKalmanModel:
    """Container for one fitted local-level state-space model."""
    def __init__(self, model_name, parameters, predicted_background, filtered_background, residuals, standardized_residuals, residual_variance, filtered_variance, usable_mask, log_likelihood, converged, status, message):
        self.model_name = model_name
        self.parameters = parameters
        self.predicted_background = predicted_background
        self.filtered_background = filtered_background
        self.residuals = residuals
        self.standardized_residuals = standardized_residuals
        self.residual_variance = residual_variance
        self.filtered_variance = filtered_variance
        self.usable_mask = usable_mask
        self.log_likelihood = log_likelihood
        self.converged = converged
        self.status = status
        self.message = message

    def summary(self):
        finite_count = int(np.isfinite(self.residuals).sum())
        parameter_count = int(len(self.parameters))
        aic = float(2 * parameter_count - 2 * self.log_likelihood) if np.isfinite(self.log_likelihood) else float("nan")
        bic = float(parameter_count * np.log(finite_count) - 2 * self.log_likelihood) if finite_count > 0 and np.isfinite(self.log_likelihood) else float("nan")
        return {"model_name": self.model_name, "state_equation": "background_t = background_t_minus_1 + process_noise", "observation_equation": "normalized_flux_t = background_t + measurement_noise", "missing_cadence_policy": "prediction_only_no_update", "converged": bool(self.converged), "status": str(self.status), "message": str(self.message), "log_likelihood": float(self.log_likelihood), "aic": aic, "bic": bic, "finite_residual_count": finite_count, **{key: float(value) for key, value in self.parameters.items()}}

def finite_values(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.isfinite(values)]

def robust_variance(values):
    clean = finite_values(values)
    if clean.size < 2:
        raise ValueError("At least two finite values are required.")
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    scale = 1.4826 * mad if mad > 0 else float(np.std(clean, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(clean, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Cannot estimate a positive variance from the input series.")
    return float(scale * scale)

def initial_state_from_values(values):
    clean = finite_values(values)
    if clean.size < 2:
        raise ValueError("At least two finite values are required.")
    return float(np.median(clean)), float(max(np.var(clean, ddof=1), robust_variance(clean)))

def run_local_level_filter(values, process_variance, measurement_variance, initial_state=None, initial_variance=None, burn_in=1):
    series = np.asarray(values, dtype=float).reshape(-1)
    if np.isinf(series).any():
        raise ValueError("Kalman input may contain NaN gaps, but not infinite values.")
    if np.isfinite(series).sum() < 10:
        raise ValueError("Kalman fitting needs at least 10 finite observations.")
    if process_variance <= 0 or measurement_variance <= 0:
        raise ValueError("process_variance and measurement_variance must be positive.")
    default_state, default_variance = initial_state_from_values(series)
    state_mean = default_state if initial_state is None else float(initial_state)
    state_variance = max(default_variance * 10.0, process_variance + measurement_variance) if initial_variance is None else float(initial_variance)
    predicted = np.full(series.shape, np.nan, dtype=float)
    filtered = np.full(series.shape, np.nan, dtype=float)
    residuals = np.full(series.shape, np.nan, dtype=float)
    standardized = np.full(series.shape, np.nan, dtype=float)
    residual_variance = np.full(series.shape, np.nan, dtype=float)
    filtered_variance = np.full(series.shape, np.nan, dtype=float)
    log_likelihood = 0.0
    for index, observed in enumerate(series):
        prior_mean = state_mean
        prior_variance = state_variance + float(process_variance)
        predicted[index] = prior_mean
        residual_variance[index] = prior_variance + float(measurement_variance)
        if np.isfinite(observed):
            innovation = float(observed) - prior_mean
            innovation_variance = residual_variance[index]
            gain = prior_variance / innovation_variance
            state_mean = prior_mean + gain * innovation
            state_variance = max((1.0 - gain) * prior_variance, np.finfo(float).eps)
            residuals[index] = innovation
            standardized[index] = innovation / np.sqrt(innovation_variance)
            log_likelihood += -0.5 * (np.log(2.0 * np.pi * innovation_variance) + innovation * innovation / innovation_variance)
        else:
            state_mean = prior_mean
            state_variance = prior_variance
        filtered[index] = state_mean
        filtered_variance[index] = state_variance
    usable_mask = np.isfinite(residuals)
    finite_positions = np.flatnonzero(usable_mask)
    if burn_in > 0 and finite_positions.size:
        usable_mask[finite_positions[:int(burn_in)]] = False
    return {"predicted_background": predicted, "filtered_background": filtered, "residuals": residuals, "standardized_residuals": standardized, "residual_variance": residual_variance, "filtered_variance": filtered_variance, "usable_mask": usable_mask, "log_likelihood": float(log_likelihood)}

def negative_log_likelihood(theta, values, initial_state, initial_variance):
    process_variance = float(np.exp(theta[0]))
    measurement_variance = float(np.exp(theta[1]))
    try:
        result = run_local_level_filter(values, process_variance, measurement_variance, initial_state=initial_state, initial_variance=initial_variance, burn_in=0)
    except ValueError:
        return np.inf
    return float(-result["log_likelihood"])

def estimate_local_level_parameters(values, maxiter=100):
    series = np.asarray(values, dtype=float).reshape(-1)
    initial_state, series_variance = initial_state_from_values(series)
    measurement_start = max(series_variance * 0.5, np.finfo(float).tiny)
    process_start = max(series_variance * 0.02, np.finfo(float).tiny)
    lower = np.log(max(series_variance * 1.0e-8, np.finfo(float).tiny))
    upper = np.log(max(series_variance * 1.0e2, np.finfo(float).tiny))
    optimum = minimize(negative_log_likelihood, np.log([process_start, measurement_start]), args=(series, initial_state, series_variance * 10.0), method="L-BFGS-B", bounds=[(lower, upper), (lower, upper)], options={"maxiter": int(maxiter)})
    process_variance = float(np.exp(optimum.x[0]))
    measurement_variance = float(np.exp(optimum.x[1]))
    parameters = {"process_variance": process_variance, "measurement_variance": measurement_variance, "initial_state": initial_state, "initial_variance": float(series_variance * 10.0)}
    return parameters, bool(optimum.success), int(optimum.status), str(optimum.message)

def fit_kalman_local_level(values, process_variance=None, measurement_variance=None, estimate_parameters=True, maxiter=100, burn_in=1):
    """Fit or apply a local-level state-space model and return one-step residuals."""
    if estimate_parameters or process_variance is None or measurement_variance is None:
        parameters, converged, status, message = estimate_local_level_parameters(values, maxiter=maxiter)
    else:
        initial_state, series_variance = initial_state_from_values(values)
        parameters = {"process_variance": float(process_variance), "measurement_variance": float(measurement_variance), "initial_state": initial_state, "initial_variance": float(series_variance * 10.0)}
        converged = True
        status = 0
        message = "parameters supplied"
    result = run_local_level_filter(values, parameters["process_variance"], parameters["measurement_variance"], initial_state=parameters["initial_state"], initial_variance=parameters["initial_variance"], burn_in=burn_in)
    return FittedKalmanModel("local_level", parameters, result["predicted_background"], result["filtered_background"], result["residuals"], result["standardized_residuals"], result["residual_variance"], result["filtered_variance"], result["usable_mask"], result["log_likelihood"], converged, status, message)

def apply_fitted_kalman_filter(values, fitted_model, burn_in=1):
    """Apply fitted local-level variances to another equal-scale series."""
    parameters = dict(fitted_model.parameters)
    result = run_local_level_filter(values, parameters["process_variance"], parameters["measurement_variance"], initial_state=parameters["initial_state"], initial_variance=parameters["initial_variance"], burn_in=burn_in)
    return FittedKalmanModel(fitted_model.model_name, parameters, result["predicted_background"], result["filtered_background"], result["residuals"], result["standardized_residuals"], result["residual_variance"], result["filtered_variance"], result["usable_mask"], result["log_likelihood"], True, 0, "filtered with supplied parameters")
