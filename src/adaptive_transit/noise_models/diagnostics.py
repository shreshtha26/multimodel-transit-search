"""Residual diagnostics for ARIMA innovations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf


def finite_values(values: np.ndarray) -> np.ndarray:
    """Return a one-dimensional finite array for diagnostics."""

    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def robust_scale(values: np.ndarray) -> float:
    """MAD-based scale estimate used for outlier counting."""

    clean = finite_values(values)
    if clean.size == 0:
        return float("nan")
    median = np.median(clean)
    mad = np.median(np.abs(clean - median))
    if mad == 0:
        return float(np.std(clean, ddof=1)) if clean.size > 1 else float("nan")
    return float(1.4826 * mad)


def max_abs_acf(values: np.ndarray, *, nlags: int = 80) -> float:
    """Maximum absolute autocorrelation after lag 0."""

    clean = finite_values(values)
    if clean.size < 3:
        return float("nan")
    usable_lags = min(nlags, clean.size - 2)
    acf_values = acf(clean, nlags=usable_lags, fft=True, missing="none")
    if len(acf_values) <= 1:
        return float("nan")
    return float(np.max(np.abs(acf_values[1:])))


def ljung_box_summary(
    values: np.ndarray,
    *,
    lags: tuple[int, ...] = (10, 20, 40),
) -> dict[str, float]:
    """Run Ljung-Box tests at feasible lags and report the worst p-value."""

    clean = finite_values(values)
    feasible_lags = [lag for lag in lags if 1 <= lag < clean.size]
    if not feasible_lags:
        return {"minimum_ljung_box_p": float("nan")}

    table = acorr_ljungbox(clean, lags=feasible_lags, return_df=True)
    summary: dict[str, float] = {f"ljung_box_p_lag_{int(lag)}": float(table.loc[lag, "lb_pvalue"]) for lag in feasible_lags}
    summary["minimum_ljung_box_p"] = float(table["lb_pvalue"].min())
    return summary


def rolling_variance_summary(values: np.ndarray, *, window: int = 96) -> dict[str, float]:
    """Summarize time-varying residual variance without changing the series."""

    clean = finite_values(values)
    if clean.size < 4:
        return {
            "rolling_var_median": float("nan"),
            "rolling_var_iqr": float("nan"),
            "rolling_var_max_to_median": float("nan"),
        }

    usable_window = min(max(4, window), clean.size)
    rolling_var = pd.Series(clean).rolling(window=usable_window, min_periods=max(3, usable_window // 2)).var().dropna().to_numpy()
    if rolling_var.size == 0:
        return {
            "rolling_var_median": float("nan"),
            "rolling_var_iqr": float("nan"),
            "rolling_var_max_to_median": float("nan"),
        }

    median = float(np.median(rolling_var))
    iqr = float(np.percentile(rolling_var, 75) - np.percentile(rolling_var, 25))
    max_to_median = float(np.max(rolling_var) / median) if median > 0 else float("nan")
    return {
        "rolling_var_median": median,
        "rolling_var_iqr": iqr,
        "rolling_var_max_to_median": max_to_median,
    }


def arch_pvalue(values: np.ndarray, *, nlags: int = 12) -> float:
    """ARCH test p-value; low values suggest time-varying variance."""

    clean = finite_values(values)
    if clean.size < max(20, nlags + 2):
        return float("nan")
    try:
        _, pvalue, _, _ = het_arch(clean, nlags=min(nlags, clean.size // 5))
    except (FloatingPointError, ValueError):
        return float("nan")
    return float(pvalue)


def residual_diagnostics(
    residuals: np.ndarray,
    *,
    acf_lags: int = 80,
    ljung_box_lags: tuple[int, ...] = (10, 20, 40),
    rolling_window: int = 96,
    outlier_sigma: float = 5.0,
) -> dict[str, float]:
    """Compute the Stage 1 residual adequacy diagnostics."""

    clean = finite_values(residuals)
    if clean.size == 0:
        raise ValueError("Residual diagnostics require at least one finite value.")

    scale = robust_scale(clean)
    median = float(np.median(clean))
    outlier_fraction = float(np.mean(np.abs(clean - median) / scale > outlier_sigma)) if np.isfinite(scale) and scale > 0 else float("nan")

    diagnostics = {
        "residual_mean": float(np.mean(clean)),
        "residual_std": float(np.std(clean, ddof=1)) if clean.size > 1 else float("nan"),
        "max_abs_residual_acf": max_abs_acf(clean, nlags=acf_lags),
        "outlier_fraction": outlier_fraction,
        "innovation_skew": float(skew(clean, bias=False)) if clean.size > 2 else float("nan"),
        "innovation_kurtosis": float(kurtosis(clean, fisher=True, bias=False)) if clean.size > 3 else float("nan"),
        "arch_pvalue": arch_pvalue(clean),
    }
    diagnostics.update(ljung_box_summary(clean, lags=ljung_box_lags))
    diagnostics.update(rolling_variance_summary(clean, window=rolling_window))
    return diagnostics


def residual_failure_flags(
    diagnostics: dict[str, float],
    *,
    converged: bool,
    boundary_coefficient_count: int,
    acf_threshold: float = 0.10,
    ljung_box_alpha: float = 0.05,
    arch_alpha: float = 0.05,
    rolling_var_ratio_threshold: float = 4.0,
    outlier_fraction_threshold: float = 0.01,
) -> dict[str, bool]:
    """Convert diagnostics into named residual failure modes."""

    max_acf = diagnostics.get("max_abs_residual_acf", float("nan"))
    min_ljung = diagnostics.get("minimum_ljung_box_p", float("nan"))
    arch_p = diagnostics.get("arch_pvalue", float("nan"))
    rolling_ratio = diagnostics.get("rolling_var_max_to_median", float("nan"))
    outlier_fraction = diagnostics.get("outlier_fraction", float("nan"))

    autocorrelation_remaining = (np.isfinite(max_acf) and max_acf > acf_threshold) or (np.isfinite(min_ljung) and min_ljung < ljung_box_alpha)
    variance_instability = (np.isfinite(arch_p) and arch_p < arch_alpha) or (np.isfinite(rolling_ratio) and rolling_ratio > rolling_var_ratio_threshold)
    outlier_heavy = np.isfinite(outlier_fraction) and outlier_fraction > outlier_fraction_threshold

    return {
        "residual_autocorrelation_remaining": bool(autocorrelation_remaining),
        "variance_instability": bool(variance_instability),
        "outlier_heavy": bool(outlier_heavy),
        "non_converged": bool(not converged),
        "boundary_coefficients": bool(boundary_coefficient_count > 0),
    }
