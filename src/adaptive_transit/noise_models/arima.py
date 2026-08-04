"""Simple ARIMA fitting for the first multi-model-transit-search noise-model milestone."""

import json
import time
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from adaptive_transit.noise_models.baselines import baseline_forecast_scores, score_forecast
from adaptive_transit.noise_models.diagnostics import residual_diagnostics, residual_failure_flags


class FittedArimaModel:
    def __init__(self, order, result, one_step_prediction, innovations, usable_mask, mode="contiguous"):
        self.order = order
        self.result = result
        self.one_step_prediction = one_step_prediction
        self.innovations = innovations
        self.usable_mask = usable_mask
        self.mode = mode


def generate_arima_orders(max_p, max_d, max_q, max_total_order=None, include_zero_order=False):
    """Generate a bounded ARIMA grid."""
    if max_p < 0 or max_d < 0 or max_q < 0:
        raise ValueError("max_p, max_d, and max_q must be non-negative.")
    if max_total_order is not None and max_total_order < 0:
        raise ValueError("max_total_order must be non-negative when provided.")

    orders = []
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                order = (p, d, q)
                if order == (0, 0, 0) and not include_zero_order:
                    continue
                if max_total_order is not None and sum(order) > max_total_order:
                    continue
                orders.append(order)
    return tuple(orders)


def _innovation_burn_in(result, order):
    return max(
        int(getattr(result, "loglikelihood_burn", 0)),
        order[1] + max(order[0], order[2], 1),
    )


def validate_series(values, allow_missing=False):
    series = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(series)
    if finite.sum() < 10:
        raise ValueError("ARIMA fitting needs at least 10 finite observations.")
    if allow_missing:
        if np.isinf(series).any():
            raise ValueError("ARIMA input may contain NaN gaps, but not infinite values.")
    elif not np.all(finite):
        raise ValueError("Contiguous ARIMA input contains non-finite values.")
    return series


def _fit_statsmodels_arima(model, fit_maxiter=None):
    if fit_maxiter is None:
        return model.fit()
    return model.fit(method_kwargs={"maxiter": int(fit_maxiter)})


def chronological_train_test_split(values, test_fraction=0.20):
    series = validate_series(values, allow_missing=False)
    if not 0.0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5.")

    split_index = int(round(series.size * (1.0 - test_fraction)))
    split_index = min(max(split_index, 5), series.size - 2)
    return series[:split_index], series[split_index:]


def chronological_observed_split_masks(values, test_fraction=0.20, allow_missing=False):
    series = validate_series(values, allow_missing=allow_missing)
    if not 0.0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5.")

    observed_positions = np.flatnonzero(np.isfinite(series))
    split_count = int(round(observed_positions.size * (1.0 - test_fraction)))
    split_count = min(max(split_count, 5), observed_positions.size - 2)

    train_mask = np.zeros(series.shape, dtype=bool)
    test_mask = np.zeros(series.shape, dtype=bool)
    train_mask[observed_positions[:split_count]] = True
    test_mask[observed_positions[split_count:]] = True
    return train_mask, test_mask


def fit_arima_model(values, order, allow_missing=False, mode="contiguous", fit_maxiter=None):
    series = validate_series(values, allow_missing=allow_missing)
    model = ARIMA(
        series,
        order=order,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    result = _fit_statsmodels_arima(model, fit_maxiter=fit_maxiter)

    prediction = result.get_prediction(start=0, end=series.size - 1, dynamic=False)
    one_step_prediction = np.asarray(prediction.predicted_mean, dtype=float)
    innovations = series - one_step_prediction

    # Drop startup residuals that mostly reflect Kalman initialization.
    burn_in = _innovation_burn_in(result, order)
    usable_mask = np.isfinite(innovations)
    usable_mask[:burn_in] = False

    return FittedArimaModel(
        order=order,
        result=result,
        one_step_prediction=one_step_prediction,
        innovations=innovations,
        usable_mask=usable_mask,
        mode=mode,
    )


def apply_fitted_arima_filter(values, fitted_model, allow_missing=False):
    series = validate_series(values, allow_missing=allow_missing)
    if series.size != fitted_model.one_step_prediction.size:
        raise ValueError("New series must have the same length as the fitted ARIMA series.")

    model = ARIMA(
        series,
        order=fitted_model.order,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    result = model.filter(fitted_model.result.params)
    prediction = result.get_prediction(start=0, end=series.size - 1, dynamic=False)
    one_step_prediction = np.asarray(prediction.predicted_mean, dtype=float)
    innovations = series - one_step_prediction

    usable_mask = np.isfinite(innovations)
    usable_mask[: _innovation_burn_in(result, fitted_model.order)] = False

    return FittedArimaModel(
        order=fitted_model.order,
        result=result,
        one_step_prediction=one_step_prediction,
        innovations=innovations,
        usable_mask=usable_mask,
        mode=fitted_model.mode,
    )


def forecast_metrics(values, train_mask, test_mask, order, allow_missing=False, fit_maxiter=None):
    series = validate_series(values, allow_missing=allow_missing)
    training_series = series.copy()
    training_series[test_mask] = np.nan

    model = ARIMA(
        training_series,
        order=order,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    result = _fit_statsmodels_arima(model, fit_maxiter=fit_maxiter)
    prediction = result.get_prediction(start=0, end=series.size - 1, dynamic=False)
    predicted_mean = np.asarray(prediction.predicted_mean, dtype=float)
    predicted_variance = np.asarray(prediction.var_pred_mean, dtype=float)
    score = score_forecast(
        series[test_mask],
        predicted_mean[test_mask],
        predictive_variance=predicted_variance[test_mask],
    )

    return {
        "test_RMSE": score.rmse,
        "test_MAE": score.mae,
        "mean_negative_log_score": score.mean_negative_log_score,
        "train_converged": bool(getattr(result, "mle_retvals", {}).get("converged", True)),
    }


def coefficient_diagnostics(result):
    names = list(getattr(result, "param_names", []))
    params = np.asarray(getattr(result, "params", []), dtype=float)
    bse = np.asarray(getattr(result, "bse", np.full(params.shape, np.nan)), dtype=float)

    try:
        conf_int = np.asarray(result.conf_int(), dtype=float)
    except (AttributeError, ValueError, np.linalg.LinAlgError):
        conf_int = np.full((len(params), 2), np.nan)

    rows = []
    boundary_distances = []
    boundary_names = []

    for index, value in enumerate(params):
        name = names[index] if index < len(names) else f"param_{index}"
        stderr = float(bse[index]) if index < len(bse) else float("nan")
        ci_low = float(conf_int[index, 0]) if index < len(conf_int) else float("nan")
        ci_high = float(conf_int[index, 1]) if index < len(conf_int) else float("nan")
        is_ar_or_ma = name.startswith(("ar.", "ma."))
        boundary_distance = float(1.0 - abs(value)) if is_ar_or_ma else float("nan")
        near_boundary = bool(is_ar_or_ma and np.isfinite(value) and abs(value) >= 0.98)
        if is_ar_or_ma and np.isfinite(boundary_distance):
            boundary_distances.append(boundary_distance)
        if near_boundary:
            boundary_names.append(name)

        rows.append(
            {
                "name": name,
                "estimate": float(value),
                "std_error": stderr,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "boundary_distance": boundary_distance,
                "near_boundary": near_boundary,
            }
        )

    min_boundary_distance = min(boundary_distances) if boundary_distances else float("nan")
    return {
        "coefficient_json": json.dumps(rows),
        "min_boundary_distance": float(min_boundary_distance),
        "boundary_coefficient_count": int(len(boundary_names)),
        "boundary_coefficients_json": json.dumps(boundary_names),
    }


def evaluate_arima_candidate(values, order, **options):
    mode = options.get("mode", "contiguous")
    allow_missing = options.get("allow_missing", False)
    test_fraction = options.get("test_fraction", 0.20)
    acf_lags = options.get("acf_lags", 80)
    ljung_box_lags = options.get("ljung_box_lags", (10, 20, 40))
    short_acf_lags = options.get("short_acf_lags", 24)
    transit_lag_range = options.get("transit_lag_range", (3, 24))
    fit_maxiter = options.get("fit_maxiter")
    started_at = time.perf_counter()
    warning_messages = []

    try:
        series = validate_series(values, allow_missing=allow_missing)
        train_mask, test_mask = chronological_observed_split_masks(
            series,
            test_fraction=test_fraction,
            allow_missing=allow_missing,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            forecast = forecast_metrics(
                series,
                train_mask,
                test_mask,
                order,
                allow_missing=allow_missing,
                fit_maxiter=fit_maxiter,
            )
            fitted = fit_arima_model(
                series,
                order,
                allow_missing=allow_missing,
                mode=mode,
                fit_maxiter=fit_maxiter,
            )

        warning_messages = [str(item.message) for item in caught]
        usable_innovations = fitted.innovations[fitted.usable_mask]
        diagnostics = residual_diagnostics(
            usable_innovations,
            acf_lags=acf_lags,
            ljung_box_lags=ljung_box_lags,
            short_acf_lags=short_acf_lags,
            transit_lag_range=transit_lag_range,
        )
        result = fitted.result
        converged = bool(getattr(result, "mle_retvals", {}).get("converged", True))
        coefficient_summary = coefficient_diagnostics(result)
        flags = residual_failure_flags(
            diagnostics,
            converged=converged,
            boundary_coefficient_count=int(coefficient_summary["boundary_coefficient_count"]),
        )
        baselines = baseline_forecast_scores(series, train_mask, test_mask)
        runtime_seconds = time.perf_counter() - started_at

        return {
            "mode": mode,
            "order": str(order),
            "p": order[0],
            "d": order[1],
            "q": order[2],
            "n_total": int(series.size),
            "n_observed": int(np.isfinite(series).sum()),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_nan_gaps": int(np.isnan(series).sum()),
            "converged": converged,
            "AIC": float(result.aic),
            "BIC": float(result.bic),
            "HQIC": float(result.hqic),
            **forecast,
            **baselines,
            "beats_best_baseline_RMSE": bool(forecast["test_RMSE"] < baselines["best_baseline_RMSE"]),
            "beats_best_baseline_MAE": bool(forecast["test_MAE"] < baselines["best_baseline_MAE"]),
            **diagnostics,
            **coefficient_summary,
            **flags,
            "runtime_seconds": float(runtime_seconds),
            "warning_count": len(warning_messages),
            "failure_reason": "",
        }
    except Exception as exc:
        runtime_seconds = time.perf_counter() - started_at
        acf_fields = {f"residual_acf_lag_{lag}": np.nan for lag in range(1, short_acf_lags + 1)}
        return {
            "mode": mode,
            "order": str(order),
            "p": order[0],
            "d": order[1],
            "q": order[2],
            "n_total": np.nan,
            "n_observed": np.nan,
            "n_train": np.nan,
            "n_test": np.nan,
            "n_nan_gaps": np.nan,
            "converged": False,
            "AIC": np.nan,
            "BIC": np.nan,
            "HQIC": np.nan,
            "test_RMSE": np.nan,
            "test_MAE": np.nan,
            "mean_negative_log_score": np.nan,
            "mean_baseline_RMSE": np.nan,
            "mean_baseline_MAE": np.nan,
            "median_baseline_RMSE": np.nan,
            "median_baseline_MAE": np.nan,
            "persistence_baseline_RMSE": np.nan,
            "persistence_baseline_MAE": np.nan,
            "best_baseline_RMSE": np.nan,
            "best_baseline_MAE": np.nan,
            "beats_best_baseline_RMSE": False,
            "beats_best_baseline_MAE": False,
            "residual_mean": np.nan,
            "residual_std": np.nan,
            "max_abs_residual_acf": np.nan,
            **acf_fields,
            "max_abs_residual_acf_1_24": np.nan,
            "mean_abs_residual_acf_1_24": np.nan,
            "max_abs_residual_acf_transit_lags": np.nan,
            "transit_relevant_lag_min": int(transit_lag_range[0]),
            "transit_relevant_lag_max": int(transit_lag_range[1]),
            "minimum_ljung_box_p": np.nan,
            "outlier_fraction": np.nan,
            "innovation_skew": np.nan,
            "innovation_kurtosis": np.nan,
            "arch_pvalue": np.nan,
            "rolling_var_median": np.nan,
            "rolling_var_iqr": np.nan,
            "rolling_var_max_to_median": np.nan,
            "coefficient_json": "[]",
            "min_boundary_distance": np.nan,
            "boundary_coefficient_count": np.nan,
            "boundary_coefficients_json": "[]",
            "residual_autocorrelation_remaining": True,
            "variance_instability": False,
            "outlier_heavy": False,
            "non_converged": True,
            "boundary_coefficients": False,
            "runtime_seconds": float(runtime_seconds),
            "warning_count": len(warning_messages),
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }


def evaluate_arima_candidates(values, orders, **options):
    rows = [
        evaluate_arima_candidate(
            values,
            order,
            **options,
        )
        for order in orders
    ]
    return pd.DataFrame(rows)
