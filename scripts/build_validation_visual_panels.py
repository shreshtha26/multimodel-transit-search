"""Generate standardized visual-validation panels for the 10-star holdout.

This script is the final *visual sanity-check* layer for characterization v2.
It is deliberately read-only with respect to characterization definitions:
it does NOT fit, tune, or modify any thresholds.

For each frozen independent-validation target it creates one PNG containing:

    A. Raw Kepler PDCSAP light curve
    B. Characterization-input light curve
    C. Flux distribution with robust tail markers
    D. Autocorrelation function (ACF)
    E. Lomb-Scargle periodogram
    F. Phase-folded light curve at the v2 dominant period
    G. Compact text summary of the existing v2 diagnostics

It also writes a manual-review template with the intentionally simple labels:

    CONSISTENT  -- statistical description agrees with visual morphology
    REVIEW      -- morphology/interpretation is ambiguous
    MISMATCH    -- statistical description clearly contradicts the light curve

IMPORTANT:
Do not use the ten holdout stars to retune thresholds merely because they look
different from the deliberately stratified 40-star development sample. A v2
change should be considered only if visual inspection exposes a repeated,
systematic characterization failure.

Inputs
------
Frozen validation manifest:
    outputs/target_selection/kepler_characterization_validation10.csv

Validation diagnostics:
    outputs/experiments/characterization_validation10/metrics/
        kic_<KIC>_q5_light_curve_diagnostics.csv

Saved characterization input:
    outputs/experiments/characterization_validation10/processed/
        kic_<KIC>_q5_characterization_input.parquet

Optional review queue:
    outputs/experiments/characterization_validation50/metrics/
        validation_review_queue.csv

Outputs
-------
    outputs/experiments/characterization_validation50/visual_review/
        kic_<KIC>_q5_validation_panel.png
        visual_review_manifest.csv
        visual_review_instructions.txt

Usage
-----
    python scripts/build_validation_visual_panels.py

The script uses the same project Kepler loader as the characterization runner
for the raw PDCSAP panel.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.timeseries import LombScargle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adaptive_transit.data.kepler_io import load_kepler_pdcsap  # noqa: E402


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "target_selection"
    / "kepler_characterization_validation10.csv"
)

DEFAULT_DIAGNOSTICS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "characterization_validation10"
    / "metrics"
)

DEFAULT_PROCESSED_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "characterization_validation10"
    / "processed"
)

DEFAULT_REVIEW_QUEUE = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "characterization_validation50"
    / "metrics"
    / "validation_review_queue.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "characterization_validation50"
    / "visual_review"
)

TARGET_ID_ALIASES = (
    "target_id",
    "kepid",
    "kic",
    "KIC",
    "kepler_id",
)

QUARTER_ALIASES = (
    "quarter",
    "Quarter",
    "q",
)

TIME_ALIASES = (
    "time",
    "time_days",
    "bkjd",
    "cadence_time",
)

FLUX_ALIASES = (
    "flux",
    "normalized_flux",
    "relative_flux",
    "pdcsap_flux",
    "characterization_flux",
)

METRIC_ALIASES = {
    "scatter": (
        "v2_robust_scatter",
        "robust_scatter",
        "flux_robust_scatter",
        "flux_mad_sigma",
        "normalized_flux_mad_sigma",
        "flux_std",
    ),
    "amplitude_label": (
        "v2_amplitude_population_label",
        "v2_amplitude_population",
        "v2_amplitude_label",
        "amplitude_population_label",
        "amplitude_population",
        "amplitude_label",
        "scatter_label",
    ),
    "memory_label": (
        "v2_memory_population_label",
        "v2_memory_population",
        "v2_memory_label",
        "memory_population_label",
        "memory_population",
        "memory_label",
    ),
    "stationarity": (
        # Prefer categorical interpretation/state columns.  Some diagnostics
        # also contain a Boolean field named simply "stationarity"; that field
        # is not the human-readable v2 classification we want on this panel.
        "v2_stationarity_label",
        "v2_stationarity_classification",
        "v2_stationarity_state",
        "stationarity_label",
        "stationarity_classification",
        "stationarity_state",
        "stationarity_interpretation",
        "stationarity_result",
        "v2_stationarity",
        "stationarity",
    ),
    "skewness": (
        "flux_skewness",
        "skewness",
    ),
    "excess_kurtosis": (
        "flux_excess_kurtosis",
        "excess_kurtosis",
        "kurtosis_excess",
    ),
    "outlier_fraction": (
        "flux_outlier_fraction",
        "outlier_fraction",
    ),
    "acf_lag_1": (
        "v2_acf_lag_1",
        "acf_lag_1",
    ),
    "acf_timescale": (
        "v2_acf_timescale_days",
        "acf_timescale_days",
        "acf_decay_timescale_days",
        "acf_characteristic_timescale_days",
    ),
    "dominant_period_v1": (
        "dominant_period_v1_days",
        "v1_dominant_period_days",
        "dominant_period_days",
        "dominant_period_v1",
    ),
    "dominant_period_v2": (
        # This is the actual v2 period used by the current characterization.
        # Keep it ahead of generic/legacy dominant-period fields.
        "v2_ls_dominant_period_days",
        "v2_dominant_period_days",
        "dominant_period_v2_days",
        "v2_ls_period_days",
        "dominant_period_v2",
    ),
    "ls_candidate_period": (
        "v2_ls_dominant_period_days",
        "v2_ls_candidate_period_days",
        "ls_candidate_period_v2_days",
        "v2_ls_period_days",
        "ls_period_v2_days",
        "ls_candidate_period_days",
        "lomb_scargle_candidate_period_days",
        "ls_best_period_days",
    ),
    "acf_candidate_period": (
        # Current v2 diagnostics use this word order.
        "v2_acf_period_candidate_days",
        "v2_acf_candidate_period_days",
        "acf_candidate_period_v2_days",
        "v2_acf_period_days",
        "acf_period_v2_days",
        "acf_candidate_period_days",
        "acf_peak_period_days",
        "acf_best_period_days",
    ),
    "period_selection_source": (
        "v2_period_selection_source",
        "period_selection_source",
        "v2_period_source",
        "period_source",
        "dominant_period_source",
        "selected_period_source",
    ),
    "period_selection_reason": (
        "v2_period_selection_reason",
        "period_selection_reason",
        "v2_period_reason",
        "period_reason",
        "dominant_period_reason",
        "selected_period_reason",
    ),
    "ls_fap": (
        "v2_ls_screening_fap",
        "ls_screening_fap_v2",
        "v2_ls_fap",
        "ls_fap_v2",
        "ls_screening_fap",
        "ls_fap",
        "v2_lomb_scargle_fap",
        "lomb_scargle_fap",
        "v2_ls_false_alarm_probability",
        "ls_false_alarm_probability",
    ),
    "ls_acf_relative_error": (
        "v2_ls_acf_period_relative_error",
        "ls_acf_period_relative_error",
    ),
    "spectral_concentration": (
        "v2_spectral_concentration",
        "spectral_concentration",
        "spectral_peak_concentration",
    ),
    "harmonic_ratio": (
        "v2_spectral_harmonic_power_ratio",
        "spectral_harmonic_power_ratio",
    ),
    "coherent_periodic": (
        "v2_coherent_periodic_candidate",
        "coherent_periodic_candidate_v2",
        "coherent_periodic_candidate",
    ),
    "rotation_review": (
        "v2_rotation_spot_review_flag",
        "rotation_spot_review_flag",
        "rotation_review_flag",
    ),
    "pulsation_review": (
        "v2_pulsation_review_flag",
        "pulsation_review_flag",
    ),
    "quiet_candidate": (
        "v2_quiet_candidate",
        "v2_quiet_star_candidate",
        "quiet_candidate",
        "quiet_star_candidate",
    ),
    "low_scatter_candidate": (
        "v2_low_scatter_structured_candidate",
        "v2_low_scatter_candidate",
        "low_scatter_structured_candidate",
        "low_scatter_candidate",
    ),
}


def _clean_target_id(value) -> str:
    text = str(value).strip()
    if text.upper().startswith("KIC"):
        text = text[3:].strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit():
        raise ValueError(f"Invalid Kepler target id: {value!r}")
    return text


def _resolve_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    for column in aliases:
        if column in frame.columns:
            return column
    return None


def _to_numpy(values) -> np.ndarray:
    """Convert Astropy/Lightkurve/pandas/numpy values to a float ndarray."""
    if hasattr(values, "value"):
        values = values.value
    return np.asarray(values, dtype=float)


def _extract_raw_time_flux(light_curve) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(light_curve, "time") and hasattr(light_curve, "flux"):
        time = _to_numpy(light_curve.time)
        flux = _to_numpy(light_curve.flux)
        return time, flux

    if isinstance(light_curve, pd.DataFrame):
        time_col = _resolve_column(light_curve, TIME_ALIASES)
        flux_col = _resolve_column(light_curve, FLUX_ALIASES)
        if time_col is None or flux_col is None:
            raise ValueError(
                "Could not identify raw time/flux columns in DataFrame; "
                f"columns={list(light_curve.columns)}"
            )
        return (
            pd.to_numeric(light_curve[time_col], errors="coerce").to_numpy(float),
            pd.to_numeric(light_curve[flux_col], errors="coerce").to_numpy(float),
        )

    raise TypeError(
        "Unsupported object returned by load_kepler_pdcsap: "
        f"{type(light_curve)!r}"
    )


def _load_characterization_input(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(path)

    time_col = _resolve_column(frame, TIME_ALIASES)
    flux_col = _resolve_column(frame, FLUX_ALIASES)

    if time_col is None:
        numeric_cols = [
            column
            for column in frame.columns
            if pd.to_numeric(frame[column], errors="coerce").notna().mean() > 0.95
        ]
        plausible_time = [
            column
            for column in numeric_cols
            if "time" in column.lower() or "bkjd" in column.lower()
        ]
        time_col = plausible_time[0] if plausible_time else None

    if flux_col is None:
        numeric_cols = [
            column
            for column in frame.columns
            if pd.to_numeric(frame[column], errors="coerce").notna().mean() > 0.95
        ]
        plausible_flux = [
            column
            for column in numeric_cols
            if "flux" in column.lower()
        ]
        flux_col = plausible_flux[0] if plausible_flux else None

    if time_col is None or flux_col is None:
        raise ValueError(
            f"Could not identify time/flux columns in {path}; "
            f"columns={list(frame.columns)}"
        )

    time = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(float)
    flux = pd.to_numeric(frame[flux_col], errors="coerce").to_numpy(float)
    return time, flux


def _finite_xy(time: np.ndarray, flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(time) & np.isfinite(flux)
    return time[mask], flux[mask]


def _normalize_flux_for_display(flux: np.ndarray) -> np.ndarray:
    finite = flux[np.isfinite(flux)]
    if finite.size == 0:
        return flux.copy()

    median = float(np.nanmedian(finite))

    # Kepler PDCSAP can be in electrons/s or already normalized. Displaying
    # relative flux makes raw and characterization panels directly comparable.
    if abs(median) > 0.1:
        return flux / median - 1.0
    return flux - median


def _load_diagnostics(path: Path) -> dict:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Diagnostics CSV is empty: {path}")

    if len(frame) == 1:
        return frame.iloc[0].to_dict()

    if {"metric", "value"}.issubset(frame.columns):
        return dict(zip(frame["metric"], frame["value"]))

    if {"name", "value"}.issubset(frame.columns):
        return dict(zip(frame["name"], frame["value"]))

    # Current pipeline diagnostics are expected to be one row. Retaining the
    # first row is safer than silently aggregating a future unfamiliar schema.
    return frame.iloc[0].to_dict()


def _normalize_metric_name(name: str) -> str:
    """Normalize historical diagnostic-column naming differences."""
    return "_".join(
        token
        for token in re.split(r"[^a-z0-9]+", str(name).lower())
        if token
    )


def _is_bool_like(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"true", "false"}


def _metric(record: dict, logical_name: str):
    """Resolve a logical v2 metric across historical diagnostic schemas.

    Exact aliases remain highest priority.  The fallback is intentionally
    semantic rather than positional, because the characterization CSV has
    evolved while the underlying quantities stayed the same.
    """
    aliases = METRIC_ALIASES[logical_name]

    # 1. Exact key match, in priority order.
    for alias in aliases:
        if alias not in record or pd.isna(record[alias]):
            continue
        value = record[alias]

        # For stationarity, skip the generic Boolean field if a categorical
        # interpretation/state exists elsewhere in the record.
        if logical_name == "stationarity" and _is_bool_like(value):
            continue
        return value

    # 2. Normalized exact match.
    normalized_record = {
        _normalize_metric_name(key): (key, value)
        for key, value in record.items()
        if pd.notna(value)
    }
    for alias in aliases:
        normalized_alias = _normalize_metric_name(alias)
        if normalized_alias not in normalized_record:
            continue
        _, value = normalized_record[normalized_alias]
        if logical_name == "stationarity" and _is_bool_like(value):
            continue
        return value

    # 3. Logical-name-specific semantic fallback.
    candidates = []
    for key, value in record.items():
        if pd.isna(value):
            continue

        name = _normalize_metric_name(key)
        score = 0

        if logical_name == "amplitude_label":
            if "amplitude" in name:
                score += 4
            if "population" in name or "label" in name:
                score += 3

        elif logical_name == "memory_label":
            if "memory" in name:
                score += 4
            if "population" in name or "label" in name:
                score += 3

        elif logical_name == "stationarity":
            if "stationar" in name:
                score += 5
            if any(
                token in name
                for token in ("label", "class", "state", "interpret", "result", "support")
            ):
                score += 4
            if _is_bool_like(value):
                score -= 10

        elif logical_name == "dominant_period_v1":
            if "period" in name:
                score += 3
            if "dominant" in name:
                score += 3
            if "v1" in name:
                score += 5
            # The historical unversioned dominant_period_days field is v1.
            if name == "dominant_period_days":
                score += 8
            if "v2" in name or "ls_dominant" in name:
                score -= 8

        elif logical_name == "dominant_period_v2":
            if "period" in name:
                score += 3
            if "dominant" in name:
                score += 3
            if "v2" in name:
                score += 7
            if "ls" in name or "lomb_scargle" in name:
                score += 4
            # Never allow the legacy unversioned v1 period to win as v2.
            if name == "dominant_period_days":
                score -= 20

        elif logical_name == "ls_fap":
            if "fap" in name:
                score += 6
            if "false_alarm" in name:
                score += 6
            if "ls" in name or "lomb_scargle" in name:
                score += 3
            if "screen" in name:
                score += 1

        elif logical_name == "ls_candidate_period":
            if "period" in name:
                score += 4
            if "ls" in name or "lomb_scargle" in name:
                score += 5
            if "candidate" in name or "best" in name or "peak" in name:
                score += 2
            if "relative_error" in name or "fap" in name:
                score -= 6

        elif logical_name == "acf_candidate_period":
            if "period" in name:
                score += 4
            if "acf" in name:
                score += 5
            if "candidate" in name or "best" in name or "peak" in name:
                score += 2
            if "relative_error" in name or "lag_" in name:
                score -= 6

        elif logical_name == "period_selection_source":
            if "period" in name:
                score += 3
            if "source" in name:
                score += 6
            if "select" in name or "dominant" in name:
                score += 2

        elif logical_name == "period_selection_reason":
            if "period" in name:
                score += 3
            if "reason" in name:
                score += 6
            if "select" in name or "dominant" in name:
                score += 2

        elif logical_name == "quiet_candidate":
            if "quiet" in name:
                score += 5
            if "candidate" in name:
                score += 2

        elif logical_name == "low_scatter_candidate":
            if "low_scatter" in name:
                score += 6
            if "candidate" in name or "structured" in name:
                score += 2

        elif logical_name == "rotation_review":
            if "rotation" in name or "spot" in name:
                score += 5
            if "review" in name or "flag" in name:
                score += 2

        elif logical_name == "pulsation_review":
            if "pulsation" in name or "pulsat" in name:
                score += 5
            if "review" in name or "flag" in name:
                score += 2

        else:
            tokens = [
                token
                for token in re.split(r"[_\W]+", aliases[0].lower())
                if token not in {"v1", "v2", "days"}
            ]
            score += sum(1 for token in tokens if token in name)

        if score > 0:
            candidates.append((score, str(key), value))

    if candidates:
        candidates.sort(key=lambda item: (-item[0], item[1]))
        best_score, _, best_value = candidates[0]

        minimum_score = {
            "amplitude_label": 6,
            "memory_label": 6,
            "stationarity": 5,
            "dominant_period_v1": 6,
            "dominant_period_v2": 10,
            "ls_fap": 6,
            "ls_candidate_period": 7,
            "acf_candidate_period": 7,
            "period_selection_source": 7,
            "period_selection_reason": 7,
            "quiet_candidate": 5,
            "low_scatter_candidate": 6,
            "rotation_review": 5,
            "pulsation_review": 5,
        }.get(logical_name, 2)

        if best_score >= minimum_score:
            return best_value

    return np.nan


def _as_float(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _as_bool(value):
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "NA"

    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"

    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value)
        return text if text and text.lower() != "nan" else "NA"

    if not np.isfinite(number):
        return "NA"

    absolute = abs(number)
    if absolute != 0 and (absolute < 1e-3 or absolute >= 1e4):
        return f"{number:.3e}"
    return f"{number:.{digits}g}"


def _robust_center_scale(flux: np.ndarray) -> tuple[float, float]:
    finite = flux[np.isfinite(flux)]
    if finite.size == 0:
        return np.nan, np.nan

    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    sigma = 1.4826 * mad

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(finite))
    return median, sigma


def _acf(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    max_lag_days: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Finite-sample ACF extended far enough to test long candidate periods."""
    time, flux = _finite_xy(time, flux)

    if len(flux) < 10:
        return np.array([]), np.array([]), np.nan

    cadence_days = float(np.nanmedian(np.diff(np.sort(time))))
    if not np.isfinite(cadence_days) or cadence_days <= 0:
        return np.array([]), np.array([]), np.nan

    max_lag = int(np.ceil(float(max_lag_days) / cadence_days))
    max_lag = min(max_lag, len(flux) - 2)

    y = flux - np.mean(flux)
    variance = float(np.dot(y, y))
    if not np.isfinite(variance) or variance <= 0:
        return np.array([]), np.array([]), np.nan

    lags = np.arange(max_lag + 1, dtype=int)
    values = np.empty(max_lag + 1, dtype=float)
    values[0] = 1.0

    for lag in range(1, max_lag + 1):
        values[lag] = float(np.dot(y[:-lag], y[lag:]) / variance)

    lag_days = lags * cadence_days
    return lag_days, values, cadence_days


def _lomb_scargle(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    min_period_days: float = 0.25,
    max_period_days: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    time, flux = _finite_xy(time, flux)
    if len(time) < 20:
        return np.array([]), np.array([])

    # Shift times for numerical stability.
    t = time - np.min(time)
    baseline = float(np.max(t) - np.min(t))
    if baseline <= 0:
        return np.array([]), np.array([])

    if max_period_days is None:
        max_period_days = min(50.0, max(2.0, 0.8 * baseline))

    max_period_days = min(float(max_period_days), 0.95 * baseline)
    if max_period_days <= min_period_days:
        return np.array([]), np.array([])

    y = flux - np.nanmedian(flux)

    min_frequency = 1.0 / max_period_days
    max_frequency = 1.0 / min_period_days

    frequency, power = LombScargle(t, y).autopower(
        minimum_frequency=min_frequency,
        maximum_frequency=max_frequency,
        samples_per_peak=8,
    )

    period = 1.0 / frequency
    order = np.argsort(period)
    return period[order], power[order]


def _phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    period_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    time, flux = _finite_xy(time, flux)
    if not np.isfinite(period_days) or period_days <= 0 or len(time) == 0:
        return np.array([]), np.array([])

    phase = ((time - np.min(time)) / period_days) % 1.0
    order = np.argsort(phase)
    return phase[order], flux[order]


def _phase_bin(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    if len(phase) == 0:
        return np.array([]), np.array([])

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    medians = np.full(n_bins, np.nan)

    which = np.digitize(phase, edges) - 1
    for index in range(n_bins):
        values = flux[which == index]
        if len(values):
            medians[index] = np.nanmedian(values)

    mask = np.isfinite(medians)
    return centers[mask], medians[mask]


def _robust_plot_limits(
    values: np.ndarray,
    *,
    sigma_multiplier: float = 6.0,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> tuple[float, float] | None:
    """Return display limits that preserve morphology despite rare excursions."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 10:
        return None

    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_sigma = 1.4826 * mad

    p_low, p_high = np.percentile(
        finite,
        [lower_percentile, upper_percentile],
    )

    if np.isfinite(robust_sigma) and robust_sigma > 0:
        robust_low = median - sigma_multiplier * robust_sigma
        robust_high = median + sigma_multiplier * robust_sigma
        low = max(float(p_low), robust_low)
        high = min(float(p_high), robust_high)
    else:
        low, high = float(p_low), float(p_high)

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None

    padding = 0.08 * (high - low)
    return low - padding, high + padding


def _review_reason_for_target(
    review_queue: pd.DataFrame | None,
    target_id: str,
    quarter: int,
) -> str:
    if review_queue is None or review_queue.empty:
        return ""

    subset = review_queue.loc[
        (review_queue["target_id"].astype(str).map(_clean_target_id) == target_id)
        & (pd.to_numeric(review_queue["quarter"], errors="coerce") == quarter)
    ]

    if subset.empty:
        return ""

    value = subset.iloc[0].get("review_reasons", "")
    return "" if pd.isna(value) else str(value)


def _compress_review_reason(reason: str) -> str:
    """Collapse the long robust-z queue into readable phenomenology."""
    if not reason:
        return "No automatic review flag."

    lower = reason.lower()
    summaries = []

    if "acf_lag" in lower or "acf_max" in lower or "max_abs_acf" in lower:
        summaries.append("weak/unusual short-lag ACF vs development sample")

    if "kurtosis" in lower or "skewness" in lower or "outlier_fraction" in lower:
        summaries.append("heavy-tail / asymmetric-flux diagnostics")

    if "ls_acf_period_relative_error" in lower:
        summaries.append("LS–ACF timescale disagreement")

    if "kpss" in lower or "stationarity" in lower:
        summaries.append("strong/ambiguous stationarity diagnostic")

    if "spectral_power_fraction" in lower or "harmonic" in lower:
        summaries.append("unusual spectral-power structure")

    if "dominant_period" in lower:
        summaries.append("dominant period outside development range")

    if not summaries:
        summaries.append("development-range feature outlier")

    return "; ".join(dict.fromkeys(summaries))


def _period_provenance_lines(record: dict) -> list[str]:
    """Surface stored period-selection fields without inventing provenance.

    This is intentionally diagnostic: it prints only fields already present in
    the characterization CSV whose names look relevant to LS/ACF/final-period
    selection.  It does not infer a selection rule when the source file does
    not explicitly contain one.
    """
    interesting = []

    for key, value in record.items():
        if pd.isna(value):
            continue

        name = _normalize_metric_name(key)
        if "period" not in name:
            continue

        relevant = (
            "ls" in name
            or "lomb_scargle" in name
            or "acf" in name
            or "dominant" in name
            or "select" in name
            or "harmonic" in name
            or "source" in name
            or "reason" in name
        )
        if not relevant:
            continue

        # Exclude obvious non-provenance arrays/grid bounds if they ever appear.
        if any(token in name for token in ("grid", "minimum", "maximum_frequency")):
            continue

        display_key = str(key)
        display_value = _fmt(value)
        interesting.append((display_key, display_value))

    # Keep the panel readable. The canonical fields are printed separately;
    # this block is supplementary provenance.
    lines = []
    seen = set()
    for key, value in sorted(interesting):
        normalized = _normalize_metric_name(key)
        if normalized in seen:
            continue
        seen.add(normalized)
        lines.append(f"{key}: {value}")
        if len(lines) >= 8:
            break

    return lines


def _text_summary(
    record: dict,
    automatic_reason: str,
    panel_ls_peak_period: float = np.nan,
) -> str:
    amplitude_label = _metric(record, "amplitude_label")
    memory_label = _metric(record, "memory_label")
    stationarity = _metric(record, "stationarity")

    skewness = _metric(record, "skewness")
    kurtosis = _metric(record, "excess_kurtosis")
    outlier_fraction = _metric(record, "outlier_fraction")

    acf_lag_1 = _metric(record, "acf_lag_1")
    acf_timescale = _metric(record, "acf_timescale")
    p_v1 = _metric(record, "dominant_period_v1")
    p_v2 = _metric(record, "dominant_period_v2")
    ls_candidate_period = _metric(record, "ls_candidate_period")
    acf_candidate_period = _metric(record, "acf_candidate_period")
    period_selection_source = _metric(record, "period_selection_source")
    period_selection_reason = _metric(record, "period_selection_reason")
    ls_fap = _metric(record, "ls_fap")
    ls_acf_error = _metric(record, "ls_acf_relative_error")
    concentration = _metric(record, "spectral_concentration")
    harmonic_ratio = _metric(record, "harmonic_ratio")

    coherent = _as_bool(_metric(record, "coherent_periodic"))
    rotation = _as_bool(_metric(record, "rotation_review"))
    pulsation = _as_bool(_metric(record, "pulsation_review"))
    quiet = _as_bool(_metric(record, "quiet_candidate"))
    low_scatter = _as_bool(_metric(record, "low_scatter_candidate"))

    panel_period_error = np.nan
    p_v2_float = _as_float(p_v2)
    if (
        np.isfinite(panel_ls_peak_period)
        and np.isfinite(p_v2_float)
        and p_v2_float > 0
    ):
        panel_period_error = abs(panel_ls_peak_period - p_v2_float) / p_v2_float

    lines = [
        "V2 CHARACTERIZATION",
        "",
        f"Amplitude population: {_fmt(amplitude_label)}",
        f"Memory population:    {_fmt(memory_label)}",
        f"Stationarity:         {_fmt(stationarity)}",
        "",
        f"Skewness:             {_fmt(skewness)}",
        f"Excess kurtosis:      {_fmt(kurtosis)}",
        f"Outlier fraction:     {_fmt(outlier_fraction)}",
        "",
        f"ACF lag-1:            {_fmt(acf_lag_1)}",
        f"ACF timescale [d]:    {_fmt(acf_timescale)}",
        "",
        f"Dominant period v1:   {_fmt(p_v1)} d",
        f"V2 LS dominant period:{_fmt(p_v2)} d",
        f"Stored LS candidate:  {_fmt(ls_candidate_period)} d",
        f"Stored ACF candidate: {_fmt(acf_candidate_period)} d",
        f"Period source:        {_fmt(period_selection_source)}",
        f"Period reason:        {_fmt(period_selection_reason)}",
        f"Panel LS max period:  {_fmt(panel_ls_peak_period)} d",
        f"Panel-v2 rel. error:  {_fmt(panel_period_error)}",
        f"LS screening FAP:     {_fmt(ls_fap)}",
        f"LS-ACF rel. error:    {_fmt(ls_acf_error)}",
        f"Spectral concentration:{_fmt(concentration)}",
        f"Harmonic power ratio: {_fmt(harmonic_ratio)}",
        "",
        f"Coherent periodic:    {_fmt(coherent)}",
        f"Quiet candidate:      {_fmt(quiet)}",
        f"Low-scatter structured:{_fmt(low_scatter)}",
        f"Rotation/spot review: {_fmt(rotation)}",
        f"Pulsation review:     {_fmt(pulsation)}",
        "",
        "AUTOMATIC REVIEW SUMMARY",
        _compress_review_reason(automatic_reason),
    ]

    provenance = _period_provenance_lines(record)
    if provenance:
        lines.extend(
            [
                "",
                "STORED PERIOD-PROVENANCE FIELDS",
                *provenance,
            ]
        )

    return "\n".join(lines)


def _add_title(ax, label: str, title: str) -> None:
    ax.set_title(f"{label}. {title}", loc="left", fontsize=10, fontweight="bold")


def build_panel(
    target_id: str,
    quarter: int,
    diagnostics: dict,
    raw_time: np.ndarray,
    raw_flux: np.ndarray,
    char_time: np.ndarray,
    char_flux: np.ndarray,
    automatic_reason: str,
    output_path: Path,
) -> None:
    raw_time, raw_flux = _finite_xy(raw_time, raw_flux)
    char_time, char_flux = _finite_xy(char_time, char_flux)

    raw_display = _normalize_flux_for_display(raw_flux)
    char_display = _normalize_flux_for_display(char_flux)

    median, robust_sigma = _robust_center_scale(char_display)

    v2_period = _as_float(_metric(diagnostics, "dominant_period_v2"))
    stored_ls_candidate = _as_float(_metric(diagnostics, "ls_candidate_period"))
    stored_acf_candidate = _as_float(_metric(diagnostics, "acf_candidate_period"))

    candidate_periods = [
        value
        for value in (v2_period, stored_ls_candidate, stored_acf_candidate)
        if np.isfinite(value) and value > 0
    ]
    acf_window_days = max(
        30.0,
        1.5 * max(candidate_periods) if candidate_periods else 30.0,
    )

    # Do not ask the ACF to span beyond ~80% of the observed baseline; very
    # long-lag estimates would otherwise be based on too few overlapping points.
    char_baseline_days = (
        float(np.nanmax(char_time) - np.nanmin(char_time))
        if len(char_time)
        else np.nan
    )
    if np.isfinite(char_baseline_days) and char_baseline_days > 0:
        acf_window_days = min(acf_window_days, 0.8 * char_baseline_days)

    lag_days, acf_values, _ = _acf(
        char_time,
        char_display,
        max_lag_days=acf_window_days,
    )

    ls_period, ls_power = _lomb_scargle(char_time, char_display)

    panel_ls_peak_period = np.nan
    if len(ls_period) and len(ls_power) and np.isfinite(ls_power).any():
        peak_index = int(np.nanargmax(ls_power))
        panel_ls_peak_period = float(ls_period[peak_index])

    phase, phase_flux = _phase_fold(char_time, char_display, v2_period)
    phase_bin, flux_bin = _phase_bin(phase, phase_flux)

    # 2 rows x 4 columns: six plots plus a text column spanning both rows.
    fig = plt.figure(figsize=(20, 10.5))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.2, 1.2, 1.2, 1.25],
        wspace=0.28,
        hspace=0.34,
    )

    ax_raw = fig.add_subplot(grid[0, 0])
    ax_char = fig.add_subplot(grid[0, 1])
    ax_hist = fig.add_subplot(grid[0, 2])
    ax_acf = fig.add_subplot(grid[1, 0])
    ax_ls = fig.add_subplot(grid[1, 1])
    ax_phase = fig.add_subplot(grid[1, 2])
    ax_text = fig.add_subplot(grid[:, 3])

    # A. Raw PDCSAP
    ax_raw.plot(raw_time, raw_display, ".", markersize=1.4, alpha=0.75)
    _add_title(ax_raw, "A", "Raw Kepler PDCSAP")
    ax_raw.set_xlabel("Time [BKJD]")
    ax_raw.set_ylabel("Relative flux")

    # B. Characterization input
    ax_char.plot(char_time, char_display, ".", markersize=1.4, alpha=0.75)
    _add_title(ax_char, "B", "Characterization input")
    ax_char.set_xlabel("Time [BKJD]")
    ax_char.set_ylabel("Relative flux")

    # C. Flux distribution
    ax_hist.hist(char_display, bins=60, alpha=0.75)
    if np.isfinite(median):
        ax_hist.axvline(median, linewidth=1.2, label="median")
    if np.isfinite(robust_sigma) and robust_sigma > 0:
        ax_hist.axvline(
            median - 5 * robust_sigma,
            linestyle="--",
            linewidth=1.0,
            label="±5 robust σ",
        )
        ax_hist.axvline(
            median + 5 * robust_sigma,
            linestyle="--",
            linewidth=1.0,
        )
    _add_title(ax_hist, "C", "Flux distribution")
    ax_hist.set_xlabel("Relative flux")
    ax_hist.set_ylabel("Cadence count")
    ax_hist.legend(fontsize=8)

    # D. ACF
    if len(lag_days):
        ax_acf.plot(lag_days, acf_values, linewidth=1.1)
        ax_acf.axhline(0.0, linewidth=0.8)

        if np.isfinite(v2_period) and 0 < v2_period <= np.nanmax(lag_days):
            ax_acf.axvline(
                v2_period,
                linestyle="--",
                linewidth=1.2,
                label=f"v2 LS period = {v2_period:.3g} d",
            )

        if (
            np.isfinite(stored_acf_candidate)
            and 0 < stored_acf_candidate <= np.nanmax(lag_days)
        ):
            ax_acf.axvline(
                stored_acf_candidate,
                linestyle=":",
                linewidth=1.4,
                label=f"stored ACF candidate = {stored_acf_candidate:.3g} d",
            )

        if (
            (np.isfinite(v2_period) and 0 < v2_period <= np.nanmax(lag_days))
            or (
                np.isfinite(stored_acf_candidate)
                and 0 < stored_acf_candidate <= np.nanmax(lag_days)
            )
        ):
            ax_acf.legend(fontsize=8)

    _add_title(
        ax_acf,
        "D",
        f"Autocorrelation (shown to {acf_window_days:.1f} d)",
    )
    ax_acf.set_xlabel("Lag [days]")
    ax_acf.set_ylabel("ACF")
    ax_acf.set_ylim(-1.0, 1.0)

    # E. Lomb-Scargle
    if len(ls_period):
        ax_ls.plot(ls_period, ls_power, linewidth=1.0)

        if np.isfinite(v2_period) and v2_period > 0:
            ax_ls.axvline(
                v2_period,
                linestyle="--",
                linewidth=1.2,
                label=f"v2 LS period = {v2_period:.3g} d",
            )

        if np.isfinite(panel_ls_peak_period) and panel_ls_peak_period > 0:
            ax_ls.axvline(
                panel_ls_peak_period,
                linestyle=":",
                linewidth=1.4,
                label=f"panel LS max = {panel_ls_peak_period:.3g} d",
            )

        if (
            np.isfinite(stored_ls_candidate)
            and stored_ls_candidate > 0
            and (
                not np.isfinite(v2_period)
                or abs(stored_ls_candidate - v2_period)
                / max(abs(v2_period), 1e-12)
                > 1e-3
            )
        ):
            ax_ls.axvline(
                stored_ls_candidate,
                linestyle="-.",
                linewidth=1.2,
                label=f"stored LS candidate = {stored_ls_candidate:.3g} d",
            )

        ax_ls.legend(fontsize=8)
    _add_title(ax_ls, "E", "Lomb–Scargle periodogram")
    ax_ls.set_xlabel("Period [days]")
    ax_ls.set_ylabel("Power")
    if len(ls_period) and np.nanmax(ls_period) / np.nanmin(ls_period) > 20:
        ax_ls.set_xscale("log")

    # F. Phase fold
    if len(phase):
        ax_phase.plot(
            phase,
            phase_flux,
            ".",
            markersize=1.4,
            alpha=0.28,
            label="cadences",
        )
        if len(phase_bin):
            ax_phase.plot(
                phase_bin,
                flux_bin,
                "o-",
                markersize=3,
                linewidth=1.1,
                label="phase-bin median",
            )

        robust_limits = _robust_plot_limits(phase_flux)
        if robust_limits is not None:
            y_low, y_high = robust_limits
            ax_phase.set_ylim(y_low, y_high)
            n_clipped = int(
                np.sum((phase_flux < y_low) | (phase_flux > y_high))
            )
            if n_clipped:
                ax_phase.text(
                    0.02,
                    0.96,
                    f"{n_clipped} extreme cadence(s) outside display range",
                    transform=ax_phase.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                )

        ax_phase.legend(fontsize=8)
    _add_title(
        ax_phase,
        "F",
        (
            f"Phase fold at v2 LS period = {v2_period:.4g} d"
            if np.isfinite(v2_period) and v2_period > 0
            else "Phase fold: no valid v2 LS period"
        ),
    )
    ax_phase.set_xlabel("Phase")
    ax_phase.set_ylabel("Relative flux")

    # G. Existing diagnostics + compressed automatic review reason.
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        _text_summary(
            diagnostics,
            automatic_reason,
            panel_ls_peak_period=panel_ls_peak_period,
        ),
        transform=ax_text.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    fig.suptitle(
        f"KIC {target_id} — Quarter {quarter} — Independent validation",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.01,
        0.012,
        (
            "Manual decision rule: CONSISTENT if the v2 description matches "
            "the visible morphology; REVIEW if ambiguous; MISMATCH only if "
            "the statistical description clearly contradicts the light curve."
        ),
        fontsize=9,
    )

    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    target_col = _resolve_column(frame, TARGET_ID_ALIASES)
    if target_col is None:
        raise ValueError(
            f"Validation manifest has no target id column: {list(frame.columns)}"
        )

    quarter_col = _resolve_column(frame, QUARTER_ALIASES)

    result = frame.copy()
    result["target_id"] = result[target_col].map(_clean_target_id)
    if quarter_col is None:
        result["quarter"] = 5
    else:
        result["quarter"] = pd.to_numeric(
            result[quarter_col],
            errors="raise",
        ).astype(int)

    return result


def _existing_manual_values(
    review_manifest_path: Path,
) -> dict[tuple[str, int], dict]:
    if not review_manifest_path.exists():
        return {}

    try:
        frame = pd.read_csv(review_manifest_path)
    except Exception:
        return {}

    values = {}
    for _, row in frame.iterrows():
        try:
            key = (
                _clean_target_id(row["target_id"]),
                int(row["quarter"]),
            )
        except Exception:
            continue

        values[key] = {
            "manual_assessment": row.get("manual_assessment", ""),
            "manual_morphology_note": row.get("manual_morphology_note", ""),
            "manual_failure_mode": row.get("manual_failure_mode", ""),
            "reviewer_note": row.get("reviewer_note", ""),
        }
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate standardized visual-validation panels for the frozen "
            "10-star independent characterization holdout."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=DEFAULT_DIAGNOSTICS_DIR,
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
    )
    parser.add_argument(
        "--review-queue",
        type=Path,
        default=DEFAULT_REVIEW_QUEUE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue generating later panels if one target fails.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    manifest = _load_manifest(args.manifest)
    if len(manifest) != 10:
        print(
            f"Warning: expected 10 validation stars in manifest; "
            f"found {len(manifest)}."
        )

    review_queue = None
    if args.review_queue.exists():
        review_queue = pd.read_csv(args.review_queue)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_manifest_path = args.output_dir / "visual_review_manifest.csv"
    preserved_manual = _existing_manual_values(review_manifest_path)

    rows = []
    failures = []

    print(f"Validation stars to visualize: {len(manifest)}")
    print(f"Output directory: {args.output_dir}")
    print(
        "Threshold policy: frozen -- this script performs visualization only."
    )

    for index, row in manifest.reset_index(drop=True).iterrows():
        target_id = _clean_target_id(row["target_id"])
        quarter = int(row["quarter"])

        diagnostics_path = (
            args.diagnostics_dir
            / f"kic_{target_id}_q{quarter}_light_curve_diagnostics.csv"
        )
        processed_path = (
            args.processed_dir
            / f"kic_{target_id}_q{quarter}_characterization_input.parquet"
        )
        panel_path = (
            args.output_dir
            / f"kic_{target_id}_q{quarter}_validation_panel.png"
        )

        print()
        print(f"[{index + 1}/{len(manifest)}] KIC {target_id} Q{quarter}")

        try:
            if not diagnostics_path.exists():
                raise FileNotFoundError(
                    f"Missing diagnostics: {diagnostics_path}"
                )
            if not processed_path.exists():
                raise FileNotFoundError(
                    f"Missing characterization input: {processed_path}"
                )

            diagnostics = _load_diagnostics(diagnostics_path)

            raw_light_curve = load_kepler_pdcsap(target_id, quarter)
            raw_time, raw_flux = _extract_raw_time_flux(raw_light_curve)
            char_time, char_flux = _load_characterization_input(processed_path)

            automatic_reason = _review_reason_for_target(
                review_queue,
                target_id,
                quarter,
            )

            build_panel(
                target_id=target_id,
                quarter=quarter,
                diagnostics=diagnostics,
                raw_time=raw_time,
                raw_flux=raw_flux,
                char_time=char_time,
                char_flux=char_flux,
                automatic_reason=automatic_reason,
                output_path=panel_path,
            )

            old = preserved_manual.get((target_id, quarter), {})
            rows.append(
                {
                    "target_id": target_id,
                    "quarter": quarter,
                    "panel_path": str(panel_path.relative_to(PROJECT_ROOT)),
                    "automatic_review_summary": _compress_review_reason(
                        automatic_reason
                    ),
                    "manual_assessment": old.get("manual_assessment", ""),
                    "manual_morphology_note": old.get(
                        "manual_morphology_note", ""
                    ),
                    "manual_failure_mode": old.get(
                        "manual_failure_mode", ""
                    ),
                    "reviewer_note": old.get("reviewer_note", ""),
                }
            )
            print(f"Panel: {panel_path}")

        except Exception as exc:
            failures.append(
                {
                    "target_id": target_id,
                    "quarter": quarter,
                    "error": repr(exc),
                }
            )
            print(f"ERROR: {exc}")
            if not args.continue_on_error:
                raise

    review_frame = pd.DataFrame(rows)
    review_frame.to_csv(review_manifest_path, index=False)

    instructions = """VISUAL VALIDATION INSTRUCTIONS
==============================

For each PNG, compare the visible morphology with the existing v2 statistics.

Enter exactly one value in manual_assessment:
    CONSISTENT
    REVIEW
    MISMATCH

CONSISTENT:
    The statistical description is a reasonable description of the observed
    morphology. An unusual star is not a failure merely because it lies outside
    the deliberately stratified 40-star development distribution.

REVIEW:
    The light curve is ambiguous enough that the statistical interpretation
    should be discussed or cross-checked, but there is no clear contradiction.

MISMATCH:
    The v2 statistical description clearly contradicts the visual morphology.

Only a repeated systematic MISMATCH pattern is a reason to consider changing
a characterization definition before freezing v2.

Suggested notes:
    manual_morphology_note:
        e.g. weak-memory / impulsive tails / quasi-periodic / smooth drift /
        narrow-band oscillation / isolated outliers / visually quiet

    manual_failure_mode:
        leave blank for CONSISTENT;
        otherwise describe the specific repeated mismatch.

Do NOT replace the independent validation stars after viewing these panels.
"""
    (args.output_dir / "visual_review_instructions.txt").write_text(
        instructions
    )

    if failures:
        pd.DataFrame(failures).to_csv(
            args.output_dir / "visual_panel_failures.csv",
            index=False,
        )

    print()
    print(
        f"Visual validation panels complete: "
        f"{len(rows)}/{len(manifest)} successful"
    )
    print(f"Review manifest: {review_manifest_path}")
    print(
        "Instructions:",
        args.output_dir / "visual_review_instructions.txt",
    )
    if failures:
        print(f"Failures: {len(failures)}")
        print(
            "Failure log:",
            args.output_dir / "visual_panel_failures.csv",
        )

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
