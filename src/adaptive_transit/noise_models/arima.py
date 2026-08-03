"""Simple ARIMA fitting for the first multi-model-transit-search noise-model milestone."""

from __future__ import annotations
import json
import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from adaptive_transit.noise_models.baselines import baseline_forecast_scores, score_forecast
from adaptive_transit.noise_models.diagnostics import residual_diagnostics, residual_failure_flags
ArimaOrder = tuple[int, int, int]


@dataclass
class FittedArimaModel:
    """Selected ARIMA fit plus one-step-ahead innovations."""

    order: ArimaOrder
    result: Any
    one_step_prediction: np.ndarray
    innovations: np.ndarray
    usable_mask: np.ndarray
    mode: str = "contiguous"


def generate_arima_orders(
    *,
    max_p: int,
    max_d: int,
    max_q: int,
    max_total_order: int | None = None,
    include_zero_order: bool = False,
) -> tuple[ArimaOrder, ...]:
    """Generate a bounded ARIMA hyperparameter grid.

    The zero order `(0, 0, 0)` is excluded by default because it is just a
    white-noise mean model; multi-model-transit-search compares simpler mean/median/persistence
    baselines separately.
    """

    if max_p < 0 or max_d < 0 or max_q < 0:
        raise ValueError("max_p, max_d, and max_q must be non-negative.")
    if max_total_order is not None and max_total_order < 0:
        raise ValueError("max_total_order must be non-negative when provided.")

    orders: list[ArimaOrder] = []
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


def _innovation_burn_in(result: Any, order: ArimaOrder) -> int:
    """Return the initial residual count excluded from downstream diagnostics."""

    return max(
        int(getattr(result, "loglikelihood_burn", 0)),
        order[1] + max(order[0], order[2], 1),
    )


def validate_series(values: np.ndarray, *, allow_missing: bool = False) -> np.ndarray:
    """Convert input to a one-dimensional series for statsmodels."""

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


def chronological_train_test_split(
    values: np.ndarray,
    *,
    test_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a time series without shuffling future observations backward."""

    series = validate_series(values, allow_missing=False)
    if not 0.0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5.")

    split_index = int(round(series.size * (1.0 - test_fraction)))
    split_index = min(max(split_index, 5), series.size - 2)
    return series[:split_index], series[split_index:]


def chronological_observed_split_masks(
    values: np.ndarray,
    *,
    test_fraction: float = 0.20,
    allow_missing: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build train/test masks using only finite observations in time order."""

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


def fit_arima_model(
    values: np.ndarray,
    order: ArimaOrder,
    *,
    allow_missing: bool = False,
    mode: str = "contiguous",
) -> FittedArimaModel:
    """Fit ARIMA and return one-step-ahead predictions and innovations."""

    series = validate_series(values, allow_missing=allow_missing)
    model = ARIMA(
        series,
        order=order,
        # Stage 1 prefers stable, scalable noise models over unconstrained
        # maximum-likelihood fits that may look good but behave badly.
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    result = model.fit()

    prediction = result.get_prediction(start=0, end=series.size - 1, dynamic=False)
    one_step_prediction = np.asarray(prediction.predicted_mean, dtype=float)
    innovations = series - one_step_prediction

    # The first few Kalman-filter residuals depend on initialization. Excluding
    # them makes diagnostics and matched-filter inputs less dominated by startup
    # artifacts, especially for differenced models.
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


def apply_fitted_arima_filter(
    values: np.ndarray,
    fitted_model: FittedArimaModel,
    *,
    allow_missing: bool = False,
) -> FittedArimaModel:
    """Apply an already fitted ARIMA model to a new series without refitting.

    This is the key operation for transformed-template matched filtering: the
    light curve and the synthetic transit template must pass through the same
    fixed ARIMA prediction operator. Re-estimating coefficients after injection
    would let the noise model partially learn the transit itself.
    """

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


def forecast_metrics(
    values: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    order: ArimaOrder,
    *,
    allow_missing: bool = False,
) -> dict[str, float | bool]:
    """Fit on the past and score forecasts on the held-out future segment."""

    series = validate_series(values, allow_missing=allow_missing)
    training_series = series.copy()
    training_series[test_mask] = np.nan

    model = ARIMA(
        training_series,
        order=order,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    result = model.fit()
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


def coefficient_diagnostics(result: Any) -> dict[str, Any]:
    """Summarize coefficient estimates and simple boundary proximity checks."""

    names = list(getattr(result, "param_names", []))
    params = np.asarray(getattr(result, "params", []), dtype=float)
    bse = np.asarray(getattr(result, "bse", np.full(params.shape, np.nan)), dtype=float)

    try:
        conf_int = np.asarray(result.conf_int(), dtype=float)
    except (AttributeError, ValueError, np.linalg.LinAlgError):
        conf_int = np.full((len(params), 2), np.nan)

    rows: list[dict[str, float | str | bool]] = []
    boundary_distances: list[float] = []
    boundary_names: list[str] = []

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


def evaluate_arima_candidate(
    values: np.ndarray,
    order: ArimaOrder,
    *,
    mode: str = "contiguous",
    allow_missing: bool = False,
    test_fraction: float = 0.20,
    acf_lags: int = 80,
    ljung_box_lags: tuple[int, ...] = (10, 20, 40),
) -> dict[str, Any]:
    """Fit and diagnose one ARIMA order."""

    started_at = time.perf_counter()
    warning_messages: list[str] = []

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
            )
            fitted = fit_arima_model(
                series,
                order,
                allow_missing=allow_missing,
                mode=mode,
            )

        warning_messages = [str(item.message) for item in caught]
        usable_innovations = fitted.innovations[fitted.usable_mask]
        diagnostics = residual_diagnostics(
            usable_innovations,
            acf_lags=acf_lags,
            ljung_box_lags=ljung_box_lags,
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
    except Exception as exc:  # noqa: BLE001 - candidate failures are recorded, not hidden.
        runtime_seconds = time.perf_counter() - started_at
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


def evaluate_arima_candidates(
    values: np.ndarray,
    orders: Iterable[ArimaOrder],
    *,
    mode: str = "contiguous",
    allow_missing: bool = False,
    test_fraction: float = 0.20,
    acf_lags: int = 80,
    ljung_box_lags: tuple[int, ...] = (10, 20, 40),
) -> pd.DataFrame:
    """Evaluate a small, explicit list of ARIMA orders."""

    rows = [
        evaluate_arima_candidate(
            values,
            order,
            mode=mode,
            allow_missing=allow_missing,
            test_fraction=test_fraction,
            acf_lags=acf_lags,
            ljung_box_lags=ljung_box_lags,
        )
        for order in orders
    ]
    return pd.DataFrame(rows)
