"""Pre-model light-curve characterization diagnostics.

These features describe the observed light curve before choosing a background
model or transit detector. They are intentionally diagnostic, not a selector.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import acf

from adaptive_transit.noise_models.diagnostics import (
    finite_values,
    ljung_box_summary,
    robust_scale,
)
from adaptive_transit.noise_models.stationarity import (
    assess_stationarity,
    stationarity_report_fields,
)


DEFAULT_ACF_LAGS = 80
DEFAULT_LJUNG_BOX_LAGS = (10, 20, 40)
DEFAULT_ROLLING_WINDOW = 96
DEFAULT_OUTLIER_SIGMA = 5.0
DEFAULT_SPECTRAL_FREQUENCIES = 2000


def json_ready(value: Any) -> Any:
    """Convert NumPy/pandas values and non-finite floats into JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if pd.isna(value) and not isinstance(value, (list, tuple, dict)):
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _as_bool_array(values: Any, *, size: int, default: bool) -> np.ndarray:
    if values is None:
        return np.full(size, bool(default), dtype=bool)
    array = np.asarray(values, dtype=bool).reshape(-1)
    if array.size != size:
        raise ValueError("Mask inputs must have the same length as the light curve.")
    return array


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _count_true_runs(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    starts = mask & np.r_[True, ~mask[:-1]]
    return int(starts.sum())


def _max_true_run(mask: np.ndarray) -> int:
    max_run = 0
    current = 0
    for item in np.asarray(mask, dtype=bool):
        if item:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return int(max_run)


def _cadence_deltas_days(
    time: np.ndarray,
    cadenceno: np.ndarray | None,
    finite_mask: np.ndarray,
) -> np.ndarray:
    if finite_mask.sum() < 2:
        return np.array([], dtype=float)
    finite_time = time[finite_mask]
    if cadenceno is None:
        order = np.argsort(finite_time)
        return np.diff(finite_time[order])

    finite_cadence = np.asarray(cadenceno, dtype=float).reshape(-1)[finite_mask]
    order = np.argsort(finite_cadence)
    cadence_steps = np.diff(finite_cadence[order])
    time_steps = np.diff(finite_time[order])
    valid = np.isfinite(cadence_steps) & np.isfinite(time_steps) & (cadence_steps > 0) & (time_steps > 0)
    return time_steps[valid] / cadence_steps[valid]


def _cadence_irregularity_label(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value <= 0.02:
        return "low"
    if value <= 0.15:
        return "moderate"
    return "high"


def sampling_summary(
    time: np.ndarray,
    values: np.ndarray,
    *,
    cadenceno: np.ndarray | None = None,
    row_present: np.ndarray | None = None,
    usable_mask: np.ndarray | None = None,
    quality: np.ndarray | None = None,
) -> dict[str, object]:
    """Summarize cadence regularity, gaps, and quality flags."""

    time = _as_float_array(time)
    values = _as_float_array(values)
    if time.size != values.size:
        raise ValueError("time and values must have the same length.")

    present = _as_bool_array(row_present, size=time.size, default=True)
    usable = _as_bool_array(usable_mask, size=time.size, default=True)
    finite_sample = present & np.isfinite(time) & np.isfinite(values)
    observed = finite_sample & usable
    gap_mask = ~observed
    deltas = _cadence_deltas_days(time, cadenceno, observed)
    valid_deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    median_cadence = float(np.median(valid_deltas)) if valid_deltas.size else float("nan")
    cadence_mad = float(np.median(np.abs(valid_deltas - median_cadence))) if valid_deltas.size and np.isfinite(median_cadence) else float("nan")
    irregularity = float(cadence_mad / median_cadence) if np.isfinite(cadence_mad) and np.isfinite(median_cadence) and median_cadence > 0 else float("nan")

    quality_flag_fraction = float("nan")
    if quality is not None:
        quality_values = _as_float_array(quality)
        if quality_values.size != time.size:
            raise ValueError("quality must have the same length as the light curve.")
        present_quality = present & np.isfinite(quality_values)
        quality_flag_fraction = _safe_fraction(int((present_quality & (quality_values != 0)).sum()), int(present_quality.sum()))

    return {
        "n_cadence_grid": int(time.size),
        "n_present_rows": int(present.sum()),
        "n_finite_observations": int(finite_sample.sum()),
        "n_usable_observations": int(observed.sum()),
        "finite_observation_fraction": _safe_fraction(int(finite_sample.sum()), int(time.size)),
        "gap_fraction": _safe_fraction(int(gap_mask.sum()), int(time.size)),
        "gap_count": _count_true_runs(gap_mask),
        "max_gap_cadences": _max_true_run(gap_mask),
        "median_cadence_days": median_cadence,
        "cadence_mad_days": cadence_mad,
        "cadence_irregularity": irregularity,
        "cadence_irregularity_label": _cadence_irregularity_label(irregularity),
        "quality_flag_fraction_observed": quality_flag_fraction,
    }


def acf_characterization(
    values: np.ndarray,
    *,
    cadence_days: float,
    nlags: int = DEFAULT_ACF_LAGS,
    long_lag_start: int = 24,
) -> dict[str, float]:
    """Summarize short- and long-lag autocorrelation structure."""

    clean = finite_values(values)
    fields: dict[str, float] = {
        "acf_lag_1": float("nan"),
        "acf_lag_2": float("nan"),
        "acf_lag_5": float("nan"),
        "acf_lag_10": float("nan"),
        "acf_lag_24": float("nan"),
        "max_abs_acf_1_n": float("nan"),
        "mean_abs_acf_1_n": float("nan"),
        "long_lag_max_abs_acf": float("nan"),
        "acf_decay_e_cadences": float("nan"),
        "acf_decay_e_days": float("nan"),
        "acf_decay_half_cadences": float("nan"),
        "acf_decay_half_days": float("nan"),
        "acf_first_zero_cadences": float("nan"),
        "acf_first_zero_days": float("nan"),
        "integrated_positive_acf_cadences": float("nan"),
        "integrated_positive_acf_days": float("nan"),
    }
    if clean.size < 4 or np.nanstd(clean) == 0:
        return fields

    usable_lag = min(int(nlags), clean.size - 2)
    if usable_lag < 1:
        return fields
    acf_values = acf(clean, nlags=usable_lag, fft=True, missing="none")

    for lag in (1, 2, 5, 10, 24):
        if lag < len(acf_values):
            fields[f"acf_lag_{lag}"] = float(acf_values[lag])
    positive_lags = np.asarray(acf_values[1:], dtype=float)
    finite_lags = positive_lags[np.isfinite(positive_lags)]
    if finite_lags.size:
        fields["max_abs_acf_1_n"] = float(np.max(np.abs(finite_lags)))
        fields["mean_abs_acf_1_n"] = float(np.mean(np.abs(finite_lags)))

    start = max(1, int(long_lag_start))
    if start < len(acf_values):
        long_values = np.asarray(acf_values[start:], dtype=float)
        long_values = long_values[np.isfinite(long_values)]
        if long_values.size:
            fields["long_lag_max_abs_acf"] = float(np.max(np.abs(long_values)))

    cadence = float(cadence_days)
    for threshold, prefix in ((np.exp(-1.0), "acf_decay_e"), (0.5, "acf_decay_half"), (0.0, "acf_first_zero")):
        for lag in range(1, len(acf_values)):
            value = float(acf_values[lag])
            if np.isfinite(value) and value <= threshold:
                fields[f"{prefix}_cadences"] = float(lag)
                fields[f"{prefix}_days"] = float(lag * cadence) if np.isfinite(cadence) else float("nan")
                break

    positive_run = []
    for value in acf_values[1:]:
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            break
        positive_run.append(value)
    if positive_run:
        integrated = float(1.0 + 2.0 * np.sum(positive_run))
        fields["integrated_positive_acf_cadences"] = integrated
        fields["integrated_positive_acf_days"] = float(integrated * cadence) if np.isfinite(cadence) else float("nan")
    return fields


def rolling_structure_summary(values: np.ndarray, *, window: int = DEFAULT_ROLLING_WINDOW) -> dict[str, float]:
    """Summarize slow drift in local mean and variance."""

    clean = finite_values(values)
    fields = {
        "rolling_window_cadences": int(window),
        "rolling_mean_min": float("nan"),
        "rolling_mean_max": float("nan"),
        "rolling_mean_range": float("nan"),
        "rolling_mean_range_over_robust_scale": float("nan"),
        "rolling_variance_median": float("nan"),
        "rolling_variance_iqr": float("nan"),
        "rolling_variance_max_to_median": float("nan"),
    }
    if clean.size < 4:
        return fields

    usable_window = min(max(4, int(window)), clean.size)
    min_periods = max(3, usable_window // 2)
    series = pd.Series(clean)
    rolling_mean = series.rolling(window=usable_window, min_periods=min_periods).mean().dropna().to_numpy(dtype=float)
    rolling_var = series.rolling(window=usable_window, min_periods=min_periods).var().dropna().to_numpy(dtype=float)
    scale = robust_scale(clean)

    if rolling_mean.size:
        mean_min = float(np.min(rolling_mean))
        mean_max = float(np.max(rolling_mean))
        mean_range = float(mean_max - mean_min)
        fields.update(
            {
                "rolling_mean_min": mean_min,
                "rolling_mean_max": mean_max,
                "rolling_mean_range": mean_range,
                "rolling_mean_range_over_robust_scale": float(mean_range / scale) if np.isfinite(scale) and scale > 0 else float("nan"),
            }
        )
    if rolling_var.size:
        median = float(np.median(rolling_var))
        fields.update(
            {
                "rolling_variance_median": median,
                "rolling_variance_iqr": float(np.percentile(rolling_var, 75) - np.percentile(rolling_var, 25)),
                "rolling_variance_max_to_median": float(np.max(rolling_var) / median) if median > 0 else float("nan"),
            }
        )
    return fields


def distribution_summary(values: np.ndarray, *, outlier_sigma: float = DEFAULT_OUTLIER_SIGMA) -> dict[str, float]:
    """Summarize marginal flux distribution shape."""

    clean = finite_values(values)
    scale = robust_scale(clean)
    median = float(np.median(clean)) if clean.size else float("nan")
    if clean.size and np.isfinite(scale) and scale > 0:
        outlier_fraction = float(np.mean(np.abs(clean - median) / scale > outlier_sigma))
    else:
        outlier_fraction = float("nan")
    return {
        "flux_mean": float(np.mean(clean)) if clean.size else float("nan"),
        "flux_median": median,
        "flux_std": float(np.std(clean, ddof=1)) if clean.size > 1 else float("nan"),
        "flux_robust_scale": scale,
        "flux_skewness": float(skew(clean, bias=False)) if clean.size > 2 else float("nan"),
        "flux_excess_kurtosis": float(kurtosis(clean, fisher=True, bias=False)) if clean.size > 3 else float("nan"),
        "flux_outlier_fraction": outlier_fraction,
    }


def lomb_scargle_summary(
    time: np.ndarray,
    values: np.ndarray,
    *,
    cadence_days: float,
    n_frequencies: int = DEFAULT_SPECTRAL_FREQUENCIES,
) -> dict[str, float]:
    """Summarize broad spectral concentration with a Lomb-Scargle periodogram."""

    fields = {
        "dominant_frequency_cycles_per_day": float("nan"),
        "dominant_period_days": float("nan"),
        "dominant_lomb_scargle_power": float("nan"),
        "spectral_concentration": float("nan"),
        "spectral_entropy": float("nan"),
        "spectral_power_fraction_period_lt_0_5d": float("nan"),
        "spectral_power_fraction_period_0_5_to_2d": float("nan"),
        "spectral_power_fraction_period_gt_2d": float("nan"),
        "spectral_frequency_count": 0.0,
    }
    time = _as_float_array(time)
    values = _as_float_array(values)
    finite = np.isfinite(time) & np.isfinite(values)
    time = time[finite]
    clean = values[finite]
    if clean.size < 8 or np.nanstd(clean) == 0:
        return fields

    span = float(np.max(time) - np.min(time))
    cadence = float(cadence_days)
    if not np.isfinite(span) or span <= 0 or not np.isfinite(cadence) or cadence <= 0:
        return fields

    min_frequency = 1.0 / span
    max_frequency = 0.5 / cadence
    if not np.isfinite(max_frequency) or max_frequency <= min_frequency:
        return fields

    frequency_count = max(32, int(n_frequencies))
    frequencies = np.linspace(min_frequency, max_frequency, frequency_count)
    centered = clean - np.nanmedian(clean)
    try:
        power = LombScargle(time, centered).power(frequencies)
    except Exception:
        return fields
    finite_power = np.isfinite(power) & (power >= 0)
    if not finite_power.any():
        return fields

    frequencies = frequencies[finite_power]
    power = np.asarray(power[finite_power], dtype=float)
    total_power = float(np.sum(power))
    if not np.isfinite(total_power) or total_power <= 0:
        return fields

    peak_index = int(np.argmax(power))
    probabilities = power / total_power
    nonzero = probabilities[probabilities > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(len(probabilities))) if len(probabilities) > 1 else 0.0
    periods = 1.0 / frequencies

    def band_fraction(mask: np.ndarray) -> float:
        return float(np.sum(power[mask]) / total_power) if mask.any() else 0.0

    fields.update(
        {
            "dominant_frequency_cycles_per_day": float(frequencies[peak_index]),
            "dominant_period_days": float(periods[peak_index]),
            "dominant_lomb_scargle_power": float(power[peak_index]),
            "spectral_concentration": float(power[peak_index] / total_power),
            "spectral_entropy": entropy,
            "spectral_power_fraction_period_lt_0_5d": band_fraction(periods < 0.5),
            "spectral_power_fraction_period_0_5_to_2d": band_fraction((periods >= 0.5) & (periods < 2.0)),
            "spectral_power_fraction_period_gt_2d": band_fraction(periods >= 2.0),
            "spectral_frequency_count": float(len(frequencies)),
        }
    )
    return fields


def characterize_light_curve(
    time: np.ndarray,
    values: np.ndarray,
    *,
    cadenceno: np.ndarray | None = None,
    quality: np.ndarray | None = None,
    row_present: np.ndarray | None = None,
    usable_mask: np.ndarray | None = None,
    target_id: str | None = None,
    quarter: int | None = None,
    flux_column: str = "normalized_flux",
    preprocessing_summary: dict[str, object] | None = None,
    acf_lags: int = DEFAULT_ACF_LAGS,
    ljung_box_lags: tuple[int, ...] = DEFAULT_LJUNG_BOX_LAGS,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    outlier_sigma: float = DEFAULT_OUTLIER_SIGMA,
    stationarity_alpha: float = 0.05,
    stationarity_min_observations: int = 24,
    spectral_frequencies: int = DEFAULT_SPECTRAL_FREQUENCIES,
) -> dict[str, object]:
    """Compute the Phase 1 statistical fingerprint for one light curve."""

    time = _as_float_array(time)
    values = _as_float_array(values)
    if time.size != values.size:
        raise ValueError("time and values must have the same length.")
    usable = _as_bool_array(usable_mask, size=time.size, default=True)
    finite = usable & np.isfinite(time) & np.isfinite(values)
    clean_values = values[finite]
    clean_time = time[finite]
    if clean_values.size == 0:
        raise ValueError("Light-curve characterization requires at least one usable finite value.")

    sampling = sampling_summary(
        time,
        values,
        cadenceno=cadenceno,
        row_present=row_present,
        usable_mask=usable,
        quality=quality,
    )
    cadence_days = float(sampling["median_cadence_days"])
    stationarity = assess_stationarity(
        clean_values,
        modelling_mode="light_curve_characterization",
        preprocessing_summary=preprocessing_summary or {},
        alpha=stationarity_alpha,
        min_observations=stationarity_min_observations,
        gaps_compressed=True,
        interpolated=False,
        contiguous_segment_used=False,
        series_representation="finite_usable_normalized_flux_sequence",
    )
    record: dict[str, object] = {
        "target_id": target_id,
        "quarter": int(quarter) if quarter is not None else None,
        "diagnostic_record_type": "light_curve_characterization_v1",
        "flux_column": str(flux_column),
        **sampling,
        **stationarity_report_fields(stationarity),
        **acf_characterization(clean_values, cadence_days=cadence_days, nlags=acf_lags),
        **ljung_box_summary(clean_values, lags=ljung_box_lags),
        **rolling_structure_summary(clean_values, window=rolling_window),
        **distribution_summary(clean_values, outlier_sigma=outlier_sigma),
        **lomb_scargle_summary(
            clean_time,
            clean_values,
            cadence_days=cadence_days,
            n_frequencies=spectral_frequencies,
        ),
    }
    record["light_curve_characterization_feature_count"] = int(len(record))
    return record


def characterize_regularized_light_curve(
    regular: pd.DataFrame,
    *,
    target_id: str | None = None,
    quarter: int | None = None,
    preprocessing_summary: dict[str, object] | None = None,
    flux_column: str = "normalized_flux",
    **kwargs: Any,
) -> dict[str, object]:
    """Compute characterization features from a preprocessed cadence grid."""

    required = {"time", flux_column}
    missing = sorted(required.difference(regular.columns))
    if missing:
        raise ValueError(f"Regularized light curve is missing required columns: {missing}")

    return characterize_light_curve(
        regular["time"].to_numpy(dtype=float),
        regular[flux_column].to_numpy(dtype=float),
        cadenceno=regular["cadenceno"].to_numpy(dtype=float) if "cadenceno" in regular.columns else None,
        quality=regular["quality"].to_numpy(dtype=float) if "quality" in regular.columns else None,
        row_present=regular["row_present"].to_numpy(dtype=bool) if "row_present" in regular.columns else None,
        usable_mask=regular["usable"].to_numpy(dtype=bool) if "usable" in regular.columns else None,
        target_id=target_id,
        quarter=quarter,
        flux_column=flux_column,
        preprocessing_summary=preprocessing_summary,
        **kwargs,
    )


def structural_diagnostic_comparison(
    series_by_name: dict[str, np.ndarray],
    *,
    time: np.ndarray | None = None,
    cadence_days: float | None = None,
    acf_lags: int = DEFAULT_ACF_LAGS,
    ljung_box_lags: tuple[int, ...] = DEFAULT_LJUNG_BOX_LAGS,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    outlier_sigma: float = DEFAULT_OUTLIER_SIGMA,
) -> pd.DataFrame:
    """Return before-vs-after structural diagnostics for comparable series."""

    rows: list[dict[str, object]] = []
    if time is not None:
        time_values = _as_float_array(time)
        if time_values.size == 0:
            time_values = None
    else:
        time_values = None

    for name, values in series_by_name.items():
        array = _as_float_array(values)
        finite = np.isfinite(array)
        if time_values is not None:
            if time_values.size != array.size:
                raise ValueError("time and each comparison series must have the same length.")
            finite &= np.isfinite(time_values)
        clean = array[finite]
        if cadence_days is None and time_values is not None and finite.sum() >= 2:
            cadence = float(np.nanmedian(np.diff(np.sort(time_values[finite]))))
        else:
            cadence = float(cadence_days) if cadence_days is not None else float("nan")
        row = {
            "series": str(name),
            "n_observations": int(clean.size),
            **acf_characterization(clean, cadence_days=cadence, nlags=acf_lags),
            **ljung_box_summary(clean, lags=ljung_box_lags),
            **rolling_structure_summary(clean, window=rolling_window),
            **distribution_summary(clean, outlier_sigma=outlier_sigma),
        }
        rows.append(row)
    return pd.DataFrame(rows)
