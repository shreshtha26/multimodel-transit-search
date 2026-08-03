"""Leakage-free forecasting baselines for ARIMA comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForecastScore:
    """Common forecast metrics used for ARIMA and simple baselines."""

    rmse: float
    mae: float
    mean_negative_log_score: float


def finite_train_test(
    values: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract finite train/test observations from a possibly gapped series."""

    series = np.asarray(values, dtype=float).reshape(-1)
    train = series[np.asarray(train_mask, dtype=bool)]
    test = series[np.asarray(test_mask, dtype=bool)]
    train = train[np.isfinite(train)]
    test = test[np.isfinite(test)]
    if train.size == 0 or test.size == 0:
        raise ValueError("Forecast scoring requires finite train and test observations.")
    return train, test


def score_forecast(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    predictive_variance: np.ndarray | None = None,
) -> ForecastScore:
    """Score forecast mean and optional predictive variance against observations."""

    y_true = np.asarray(observed, dtype=float).reshape(-1)
    y_pred = np.asarray(predicted, dtype=float).reshape(-1)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() == 0:
        raise ValueError("No finite forecast points are available for scoring.")

    errors = y_true[finite] - y_pred[finite]
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))

    variance = np.full(errors.shape, np.var(errors, ddof=1) if errors.size > 1 else 1.0) if predictive_variance is None else np.asarray(predictive_variance, dtype=float).reshape(-1)[finite]

    variance_floor = np.finfo(float).eps
    safe_variance = np.maximum(variance, variance_floor)
    negative_log_score = 0.5 * (np.log(2.0 * np.pi * safe_variance) + (errors**2 / safe_variance))
    return ForecastScore(
        rmse=rmse,
        mae=mae,
        mean_negative_log_score=float(np.mean(negative_log_score)),
    )


def baseline_forecast_scores(
    values: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, float]:
    """Score mean, median, and persistence baselines without test leakage."""

    train, test = finite_train_test(values, train_mask, test_mask)
    last_observed = float(train[-1])

    baselines = {
        "mean": float(np.mean(train)),
        "median": float(np.median(train)),
        "persistence": last_observed,
    }

    scores: dict[str, float] = {}
    for name, prediction_value in baselines.items():
        predicted = np.full(test.shape, prediction_value, dtype=float)
        variance = np.full(test.shape, np.var(train, ddof=1) if train.size > 1 else 1.0)
        score = score_forecast(test, predicted, predictive_variance=variance)
        scores[f"{name}_baseline_RMSE"] = score.rmse
        scores[f"{name}_baseline_MAE"] = score.mae
        scores[f"{name}_baseline_mean_negative_log_score"] = score.mean_negative_log_score

    scores["best_baseline_RMSE"] = min(
        scores["mean_baseline_RMSE"],
        scores["median_baseline_RMSE"],
        scores["persistence_baseline_RMSE"],
    )
    scores["best_baseline_MAE"] = min(
        scores["mean_baseline_MAE"],
        scores["median_baseline_MAE"],
        scores["persistence_baseline_MAE"],
    )
    return scores
