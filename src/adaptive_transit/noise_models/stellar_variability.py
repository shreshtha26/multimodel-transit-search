"""Scientifically conservative stellar-variability characterization.

This module adds a second-generation characterization layer for Kepler light
curves.  It is intentionally *not* an astrophysical classifier.  The primary
output is a continuous feature vector that can be used for model-selection
experiments (ARIMA / Kalman / GP / wavelet / etc.).  Human-readable labels are
only screening / interpretation aids.

Design rules
------------
1. Do not call a star "quiet" because its scatter is small.  Quietness is a
   multivariate condition: low amplitude *and* weak temporal / periodic /
   evolving structure relative to the population being studied.
2. Preserve the regular cadence index when measuring autocorrelation.  Missing
   cadences are ignored pairwise; they are not squeezed out of the lag axis.
3. Treat Lomb-Scargle false-alarm probabilities as a periodicity *screen*.
   Astropy's analytic FAP approximations assume a non-varying signal plus
   Gaussian noise and therefore are not a red-noise-calibrated astrophysical
   significance test.
4. ADF/KPSS are supporting diagnostics only.  They test specific statistical
   null hypotheses and are not a physical label for a star.
5. "Pulsation-like" and "rotation/spot-like" outputs below are review flags,
   not classifications.  Confirming physical variability classes requires
   stellar parameters, longer baselines and/or dedicated astrophysical checks.
6. Gap metrics are deliberately absent from MODEL_SELECTION_FEATURE_COLUMNS.
   They remain useful engineering/QC diagnostics elsewhere in the repository,
   but are not a scientific stellar-variability axis in this project.

Operational thresholds
----------------------
Several cutoffs below (1% LS FAP screen, 20% period agreement, |ACF1|=0.2/0.5,
10% harmonic tolerance) are *declared operational choices*, not universal
astrophysical boundaries.  Population-relative amplitude/evolution boundaries
are preferred where possible and are computed with robust quantiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks, peak_widths

from adaptive_transit.noise_models.diagnostics import robust_scale


# ---------------------------------------------------------------------------
# Public constants used by downstream ML / router code.
# IMPORTANT: keep this list continuous-valued.  Do not feed the human-readable
# regime labels to a model unless you are explicitly testing a label-based
# baseline; the scientific objective is to learn from measured variability.
# ---------------------------------------------------------------------------
MODEL_SELECTION_FEATURE_COLUMNS: tuple[str, ...] = (
    "flux_robust_scale",
    "flux_skewness",
    "flux_excess_kurtosis",
    "original_adf_pvalue",
    "original_kpss_pvalue",
    "rolling_mean_range_over_robust_scale",
    "rolling_variance_max_to_median",
    "v2_acf_lag_1",
    "v2_acf_max_abs",
    "v2_acf_integrated_positive_days",
    "v2_acf_decay_e_days",
    "v2_ls_dominant_period_days",
    "v2_ls_log10_fap",
    "v2_spectral_concentration",
    "v2_spectral_entropy",
    "v2_spectral_prominent_peak_count",
    "v2_spectral_nonharmonic_peak_count",
    "v2_spectral_harmonic_power_ratio",
    "v2_half_period_consistency_fraction",
    "v2_segment_scale_relative_mad",
    "v2_segment_median_range_over_global_scale",
)


# Frozen v2 scientific representation used by the 100-star characterization
# experiment.  The seven domain names are scientific groupings; the eleven
# source columns below are the actual canonical variables.
#
# IMPORTANT: keep this separate from MODEL_SELECTION_FEATURE_COLUMNS above.
# That larger list is retained as an expanded-feature ablation set so existing
# reranker code is not silently changed.  The compact canonical set is the
# primary representation for the next star-level model-selection experiment.
CANONICAL_CHARACTERIZATION_SCHEMA: tuple[tuple[str, str, str, str], ...] = (
    ("scatter_amplitude", "Robust scatter", "flux_robust_scale", "continuous"),
    ("distribution_shape", "Skewness", "flux_skewness", "continuous"),
    ("distribution_shape", "Outlier fraction", "flux_outlier_fraction", "continuous"),
    ("autocorrelation_memory", "ACF lag 1", "v2_acf_lag_1", "continuous"),
    ("autocorrelation_memory", "ACF e-fold timescale", "v2_acf_decay_e_days", "continuous"),
    ("stationarity", "Stationarity state", "original_series_stationarity_conclusion", "categorical"),
    ("spectral_structure", "Spectral concentration", "v2_spectral_concentration", "continuous"),
    ("spectral_structure", "Harmonic power ratio", "v2_spectral_harmonic_power_ratio", "continuous"),
    ("periodicity_coherence", "Dominant period", "v2_ls_dominant_period_days", "continuous"),
    ("periodicity_coherence", "LS-ACF period agreement error", "v2_ls_acf_period_relative_error", "continuous"),
    ("variance_evolution", "Segment-scale variability", "v2_segment_scale_relative_mad", "continuous"),
)

CANONICAL_CHARACTERIZATION_COLUMNS: tuple[str, ...] = tuple(
    item[2] for item in CANONICAL_CHARACTERIZATION_SCHEMA
)

CANONICAL_CONTINUOUS_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    item[2] for item in CANONICAL_CHARACTERIZATION_SCHEMA if item[3] == "continuous"
)

DOMINANT_STATISTICAL_BEHAVIOUR_ORDER: tuple[str, ...] = (
    "Quiet / low variability",
    "Low-scatter structured",
    "Short-memory stochastic",
    "Long-memory / correlated",
    "Coherent periodic",
    "Quasi-periodic / structured",
    "Evolving variability",
    "High-amplitude / high-scatter",
    "Tail-heavy / asymmetric",
    "Mixed / complex",
)

V2_FREEZE_ID = "stellar_variability_v2_100star_freeze"


@dataclass(frozen=True)
class VariabilityBoundaryConfig:
    """Operational boundaries for reproducible screening labels.

    These numbers are intentionally centralized so that a paper / experiment
    can report exactly which definitions were used.  None should be described
    as a universal stellar taxonomy.
    """

    # Periodicity screen.  This is an analytic Lomb-Scargle maximum-peak FAP,
    # not a red-noise-aware significance calibration.
    ls_fap_threshold: float = 0.01

    # LS period and ACF / segment periods are allowed to agree at the true
    # period or a simple harmonic.  20% is deliberately permissive because
    # spot evolution can broaden / shift photometric rotation peaks.
    period_agreement_fraction: float = 0.20

    # Convenience descriptors only.  Population-relative quantiles are still
    # saved and should be preferred for modelling.
    weak_abs_acf1: float = 0.20
    strong_abs_acf1: float = 0.50

    # A harmonic peak needs at least this fraction of the dominant LS power to
    # be used as supporting evidence for non-sinusoidal periodic modulation.
    harmonic_power_ratio_threshold: float = 0.10

    # Population-relative boundaries.  These avoid an unjustified universal
    # ppm threshold for "quiet" or "active" Kepler stars.
    low_amplitude_quantile: float = 0.25
    high_amplitude_quantile: float = 0.75
    low_memory_quantile: float = 0.25
    high_memory_quantile: float = 0.75
    high_evolution_quantile: float = 0.75


DEFAULT_BOUNDARIES = VariabilityBoundaryConfig()


def _as_float_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _robust_relative_mad(values: np.ndarray) -> float:
    """Return 1.4826*MAD / median for positive scale-like values."""
    values = _as_float_array(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    median = float(np.median(values))
    if median <= 0:
        return float("nan")
    mad = float(np.median(np.abs(values - median)))
    return float(1.4826 * mad / median)


def _period_harmonic_relative_error(
    period_a: float,
    period_b: float,
    harmonics: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> float:
    """Smallest relative period disagreement allowing simple harmonics.

    Example: if LS finds P while the ACF finds 2P because two spot groups are
    present, the two measurements can still support the same periodic process.
    This function does *not* prove a rotation period; it only prevents a simple
    harmonic convention from being treated as disagreement.
    """
    if not np.isfinite(period_a) or not np.isfinite(period_b) or period_a <= 0 or period_b <= 0:
        return float("nan")
    errors = [abs(period_a - factor * period_b) / period_a for factor in harmonics]
    return float(min(errors))


def pairwise_regular_grid_acf(
    values: np.ndarray,
    *,
    cadence_days: float,
    max_lag: int = 240,
    min_pairs: int = 20,
    min_period_days: float = 0.20,
) -> dict[str, float]:
    """Gap-preserving ACF-like pairwise correlations on the cadence grid.

    The existing v1 characterization removes non-finite points before calling
    statsmodels ACF.  That is acceptable as an engineering summary but it can
    change what "lag 10" means if missing cadences are squeezed out.  Here lag
    k always means k Kepler cadences: only pairs for which both endpoints are
    observed are used.

    This is a descriptive correlation sequence rather than an estimator with a
    universal sampling distribution.  It is used as a feature, not as a formal
    hypothesis test.
    """
    series = _as_float_array(values)
    fields = {
        "v2_acf_lag_1": float("nan"),
        "v2_acf_lag_2": float("nan"),
        "v2_acf_lag_5": float("nan"),
        "v2_acf_lag_10": float("nan"),
        "v2_acf_lag_24": float("nan"),
        "v2_acf_max_abs": float("nan"),
        "v2_acf_mean_abs": float("nan"),
        "v2_acf_decay_e_days": float("nan"),
        "v2_acf_decay_half_days": float("nan"),
        "v2_acf_first_zero_days": float("nan"),
        "v2_acf_integrated_positive_days": float("nan"),
        "v2_acf_period_candidate_days": float("nan"),
        "v2_acf_period_peak_height": float("nan"),
    }
    if series.size < max(4, min_pairs + 1) or not np.isfinite(cadence_days) or cadence_days <= 0:
        return fields

    usable_lag = min(int(max_lag), series.size - 2)
    correlations = np.full(usable_lag + 1, np.nan, dtype=float)
    correlations[0] = 1.0

    for lag in range(1, usable_lag + 1):
        left = series[:-lag]
        right = series[lag:]
        valid = np.isfinite(left) & np.isfinite(right)
        if int(valid.sum()) < int(min_pairs):
            continue
        x = left[valid]
        y = right[valid]
        if np.std(x) <= 0 or np.std(y) <= 0:
            continue
        correlations[lag] = float(np.corrcoef(x, y)[0, 1])

    for lag in (1, 2, 5, 10, 24):
        if lag < len(correlations):
            fields[f"v2_acf_lag_{lag}"] = float(correlations[lag])

    finite_corr = correlations[1:][np.isfinite(correlations[1:])]
    if finite_corr.size:
        fields["v2_acf_max_abs"] = float(np.max(np.abs(finite_corr)))
        fields["v2_acf_mean_abs"] = float(np.mean(np.abs(finite_corr)))

    # Decay times are measured only from the initial positive correlation run.
    # Once the sequence crosses a threshold we stop; later oscillatory peaks are
    # captured separately as periodicity evidence.
    for threshold, name in (
        (np.exp(-1.0), "v2_acf_decay_e_days"),
        (0.5, "v2_acf_decay_half_days"),
        (0.0, "v2_acf_first_zero_days"),
    ):
        for lag in range(1, len(correlations)):
            value = correlations[lag]
            if np.isfinite(value) and value <= threshold:
                fields[name] = float(lag * cadence_days)
                break

    positive_run: list[float] = []
    for value in correlations[1:]:
        if not np.isfinite(value) or value <= 0:
            break
        positive_run.append(float(value))
    if positive_run:
        # 1 + 2*sum(rho_k) is an integrated-correlation-time style measure in
        # cadence units.  It is a descriptive timescale here, not a claim of a
        # true long-memory stochastic process.
        tau_cadences = float(1.0 + 2.0 * np.sum(positive_run))
        fields["v2_acf_integrated_positive_days"] = float(tau_cadences * cadence_days)

    # ACF peak used only as independent support for an LS periodicity candidate.
    # We intentionally avoid calling this a rotation period by itself.
    min_lag = max(2, int(np.ceil(float(min_period_days) / cadence_days)))
    if min_lag < len(correlations) - 1:
        y = correlations[min_lag:].copy()
        y[~np.isfinite(y)] = -np.inf
        finite_y = y[np.isfinite(y)]
        if finite_y.size:
            # 0.10 prominence is an operational screen on a correlation scale,
            # not a significance threshold.  The final periodicity flag also
            # requires independent LS evidence.
            peaks, properties = find_peaks(y, prominence=0.10)
            if peaks.size:
                heights = y[peaks]
                best = int(np.argmax(heights))
                lag = int(min_lag + peaks[best])
                fields["v2_acf_period_candidate_days"] = float(lag * cadence_days)
                fields["v2_acf_period_peak_height"] = float(heights[best])

    return fields


def _frequency_grid(
    span_days: float,
    cadence_days: float,
    *,
    minimum_count: int,
    samples_per_peak: float,
    minimum_cycles: float,
    max_frequency_count: int,
) -> np.ndarray:
    """Build an LS grid with enough frequency resolution for long periods.

    v1 uses a fixed 2,000-point linear grid.  For a ~90 d Kepler quarter that
    can undersample long-period peaks.  The frequency spacing here is tied to
    the observational baseline: approximately `samples_per_peak / span`.
    """
    min_frequency = float(minimum_cycles / span_days)
    max_frequency = float(0.5 / cadence_days)  # regular-grid Nyquist frequency
    if max_frequency <= min_frequency:
        return np.array([], dtype=float)
    resolved_count = int(np.ceil((max_frequency - min_frequency) * span_days * samples_per_peak)) + 1
    count = max(int(minimum_count), resolved_count, 64)
    count = min(count, int(max_frequency_count))
    return np.linspace(min_frequency, max_frequency, count)


def _is_simple_harmonic_ratio(ratio: float, tolerance: float = 0.05) -> bool:
    """Return True if a frequency ratio is close to n or 1/n for n=1..5."""
    if not np.isfinite(ratio) or ratio <= 0:
        return False
    for n in range(1, 6):
        if abs(ratio - float(n)) / float(n) <= tolerance:
            return True
        reciprocal = 1.0 / float(n)
        if abs(ratio - reciprocal) / reciprocal <= tolerance:
            return True
    return False


def lomb_scargle_variability_summary(
    time: np.ndarray,
    values: np.ndarray,
    *,
    cadence_days: float,
    minimum_frequency_count: int = 4000,
    samples_per_peak: float = 5.0,
    minimum_cycles: float = 2.0,
    max_frequency_count: int = 50000,
) -> dict[str, float]:
    """Characterize periodic / multi-periodic spectral structure.

    Assumptions and interpretation
    ------------------------------
    * We require >=2 cycles across the analysed baseline for the default global
      period search.  A one-quarter light curve therefore cannot robustly label
      very long rotation periods; such stars should remain "unresolved/long".
    * The maximum frequency is the regular-grid Nyquist frequency.  Kepler long
      cadence can alias variability above Nyquist, so a high-frequency peak is
      not automatically a physical pulsation frequency.
    * `v2_ls_fap` is Astropy's Baluev approximation for the highest peak under a
      non-varying + Gaussian-noise null.  Correlated stellar noise violates that
      idealized null, hence this is only a periodicity screen.
    """
    fields = {
        "v2_ls_dominant_frequency_per_day": float("nan"),
        "v2_ls_dominant_period_days": float("nan"),
        "v2_ls_dominant_power": float("nan"),
        "v2_ls_fap": float("nan"),
        "v2_ls_log10_fap": float("nan"),
        "v2_spectral_concentration": float("nan"),
        "v2_spectral_entropy": float("nan"),
        "v2_spectral_prominent_peak_count": float("nan"),
        "v2_spectral_nonharmonic_peak_count": float("nan"),
        "v2_spectral_second_peak_power_ratio": float("nan"),
        "v2_spectral_third_peak_power_ratio": float("nan"),
        "v2_spectral_harmonic_power_ratio": float("nan"),
        "v2_ls_peak_width_fraction": float("nan"),
        "v2_ls_frequency_count": 0.0,
        "v2_ls_minimum_cycles": float(minimum_cycles),
    }

    time = _as_float_array(time)
    values = _as_float_array(values)
    finite = np.isfinite(time) & np.isfinite(values)
    t = time[finite]
    y = values[finite]
    if y.size < 24 or np.std(y) <= 0:
        return fields

    order = np.argsort(t)
    t = t[order]
    y = y[order]
    span = float(t[-1] - t[0])
    if not np.isfinite(span) or span <= 0 or not np.isfinite(cadence_days) or cadence_days <= 0:
        return fields

    frequencies = _frequency_grid(
        span,
        float(cadence_days),
        minimum_count=minimum_frequency_count,
        samples_per_peak=samples_per_peak,
        minimum_cycles=minimum_cycles,
        max_frequency_count=max_frequency_count,
    )
    if frequencies.size < 8:
        return fields

    centered = y - np.median(y)
    ls = LombScargle(t, centered, normalization="standard")
    try:
        power = np.asarray(ls.power(frequencies), dtype=float)
    except Exception:
        return fields
    valid = np.isfinite(power) & (power >= 0)
    frequencies = frequencies[valid]
    power = power[valid]
    if power.size < 8 or float(np.sum(power)) <= 0:
        return fields

    peak_index = int(np.argmax(power))
    peak_frequency = float(frequencies[peak_index])
    peak_power = float(power[peak_index])
    dominant_period = float(1.0 / peak_frequency)
    total_power = float(np.sum(power))

    probabilities = power / total_power
    nonzero = probabilities[probabilities > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(len(probabilities))) if len(probabilities) > 1 else 0.0

    # Analytic FAP is stored as a screening feature, never as final red-noise
    # significance.  Clamp only for the log representation so exact zero from
    # floating-point underflow remains distinguishable in the raw FAP field.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fap = float(
                ls.false_alarm_probability(
                    peak_power,
                    method="baluev",
                    minimum_frequency=float(frequencies[0]),
                    maximum_frequency=float(frequencies[-1]),
                )
            )
    except Exception:
        fap = float("nan")

    # Prominent peaks: prominence is measured relative to the robust spread of
    # the sampled periodogram.  Five robust-sigma is an operational feature
    # extraction rule, not a probability statement.
    median_power = float(np.median(power))
    power_mad = float(np.median(np.abs(power - median_power)))
    robust_power_scale = float(1.4826 * power_mad)
    prominence_floor = 5.0 * robust_power_scale
    if not np.isfinite(prominence_floor) or prominence_floor <= 0:
        prominence_floor = max(1e-12, 0.05 * peak_power)
    prominent_indices, _ = find_peaks(power, prominence=prominence_floor)
    if peak_index not in set(prominent_indices.tolist()):
        prominent_indices = np.unique(np.r_[prominent_indices, peak_index]).astype(int)

    ordered_peak_indices = prominent_indices[np.argsort(power[prominent_indices])[::-1]]
    ordered_powers = power[ordered_peak_indices]
    second_ratio = float(ordered_powers[1] / peak_power) if ordered_powers.size >= 2 and peak_power > 0 else float("nan")
    third_ratio = float(ordered_powers[2] / peak_power) if ordered_powers.size >= 3 and peak_power > 0 else float("nan")

    # Count prominent peaks that are not simple harmonics of the dominant peak.
    # Multiple non-harmonic coherent peaks are useful as a *pulsation review*
    # feature, but can also arise from aliases or multiple variability sources.
    nonharmonic_count = 0
    for idx in ordered_peak_indices[1:]:
        ratio = float(frequencies[idx] / peak_frequency)
        if not _is_simple_harmonic_ratio(ratio):
            nonharmonic_count += 1

    # Harmonic support is the strongest sampled power close to f/2 or 2f,
    # divided by the dominant peak.  It helps identify non-sinusoidal periodic
    # morphology but is not uniquely diagnostic of starspots.
    harmonic_powers: list[float] = []
    for target_frequency in (0.5 * peak_frequency, 2.0 * peak_frequency):
        if frequencies[0] <= target_frequency <= frequencies[-1]:
            idx = int(np.argmin(np.abs(frequencies - target_frequency)))
            harmonic_powers.append(float(power[idx]))
    harmonic_ratio = float(max(harmonic_powers) / peak_power) if harmonic_powers and peak_power > 0 else float("nan")

    # Fractional FWHM-like width: narrower peaks indicate greater frequency
    # coherence. Width is defined only for an interior peak with positive
    # prominence. Edge maxima, flat/degenerate peaks, and zero-prominence peaks
    # are left as NaN rather than being assigned a misleading zero-width value.
    #
    # This metric remains frequency-resolution dependent, so comparisons should
    # only be made when the periodogram construction is comparable.
    width_fraction = float("nan")

    is_interior_peak = (
            0 < peak_index < len(power) - 1
            and np.isfinite(power[peak_index])
            and power[peak_index] > power[peak_index - 1]
            and power[peak_index] > power[peak_index + 1]
    )

    if is_interior_peak and len(frequencies) > 1:
        try:
            from scipy.signal import peak_prominences

            prominence_data = peak_prominences(power, [peak_index])
            prominence = float(prominence_data[0][0])

            if np.isfinite(prominence) and prominence > 0:
                widths = peak_widths(
                    power,
                    [peak_index],
                    rel_height=0.5,
                    prominence_data=prominence_data,
                )[0]

                if widths.size and np.isfinite(widths[0]) and widths[0] > 0:
                    delta_f = float(np.median(np.diff(frequencies)))

                    if np.isfinite(delta_f) and delta_f > 0 and peak_frequency > 0:
                        width_fraction = float(
                            widths[0] * delta_f / peak_frequency
                        )

        except (ValueError, FloatingPointError):
            pass

    fields.update(
        {
            "v2_ls_dominant_frequency_per_day": peak_frequency,
            "v2_ls_dominant_period_days": dominant_period,
            "v2_ls_dominant_power": peak_power,
            "v2_ls_fap": fap,
            "v2_ls_log10_fap": float(np.log10(max(fap, np.finfo(float).tiny))) if np.isfinite(fap) and fap >= 0 else float("nan"),
            "v2_spectral_concentration": float(peak_power / total_power),
            "v2_spectral_entropy": entropy,
            "v2_spectral_prominent_peak_count": float(len(ordered_peak_indices)),
            "v2_spectral_nonharmonic_peak_count": float(nonharmonic_count),
            "v2_spectral_second_peak_power_ratio": second_ratio,
            "v2_spectral_third_peak_power_ratio": third_ratio,
            "v2_spectral_harmonic_power_ratio": harmonic_ratio,
            "v2_ls_peak_width_fraction": width_fraction,
            "v2_ls_frequency_count": float(len(frequencies)),
        }
    )
    return fields


def segmented_evolution_summary(
    time: np.ndarray,
    values: np.ndarray,
    *,
    cadence_days: float,
    global_period_days: float,
    amplitude_segments: int = 4,
    coherence_segments: int = 2,
    spectral_minimum_count: int = 1000,
    period_agreement_fraction: float = DEFAULT_BOUNDARIES.period_agreement_fraction,
) -> dict[str, float]:
    """Measure amplitude evolution and cross-segment periodic consistency.

    Equal-*time* segments are used rather than equal numbers of observations.
    This avoids giving densely sampled portions disproportionate leverage.
    Segment statistics are descriptive; they are not a change-point model.
    """
    fields = {
        "v2_segment_scale_relative_mad": float("nan"),
        "v2_segment_scale_max_to_min": float("nan"),
        "v2_segment_median_range_over_global_scale": float("nan"),
        "v2_half_period_1_days": float("nan"),
        "v2_half_period_2_days": float("nan"),
        "v2_half_period_consistency_fraction": float("nan"),
    }
    time = _as_float_array(time)
    values = _as_float_array(values)
    finite = np.isfinite(time) & np.isfinite(values)
    t = time[finite]
    y = values[finite]
    if y.size < 40:
        return fields

    start = float(np.min(t))
    stop = float(np.max(t))
    if stop <= start:
        return fields

    # Amplitude / location evolution.
    edges = np.linspace(start, stop, max(2, int(amplitude_segments)) + 1)
    scales: list[float] = []
    medians: list[float] = []
    for index in range(len(edges) - 1):
        right_closed = index == len(edges) - 2
        mask = (t >= edges[index]) & ((t <= edges[index + 1]) if right_closed else (t < edges[index + 1]))
        if int(mask.sum()) < 20:
            continue
        segment = y[mask]
        scale = robust_scale(segment)
        if np.isfinite(scale) and scale > 0:
            scales.append(float(scale))
        medians.append(float(np.median(segment)))

    if scales:
        scale_values = np.asarray(scales, dtype=float)
        fields["v2_segment_scale_relative_mad"] = _robust_relative_mad(scale_values)
        if np.min(scale_values) > 0:
            fields["v2_segment_scale_max_to_min"] = float(np.max(scale_values) / np.min(scale_values))
    global_scale = robust_scale(y)
    if medians and np.isfinite(global_scale) and global_scale > 0:
        fields["v2_segment_median_range_over_global_scale"] = float((max(medians) - min(medians)) / global_scale)

    # Period consistency in two broad time segments.  Two halves preserve a
    # longer baseline than four quarters, which matters for ~10-30 d rotation.
    # A period too long to complete >=2 cycles within a half will remain NaN;
    # that is preferable to over-interpreting an unresolved long timescale.
    coherence_edges = np.linspace(start, stop, max(2, int(coherence_segments)) + 1)
    segment_periods: list[float] = []
    for index in range(len(coherence_edges) - 1):
        right_closed = index == len(coherence_edges) - 2
        mask = (t >= coherence_edges[index]) & ((t <= coherence_edges[index + 1]) if right_closed else (t < coherence_edges[index + 1]))
        if int(mask.sum()) < 40:
            segment_periods.append(float("nan"))
            continue
        summary = lomb_scargle_variability_summary(
            t[mask],
            y[mask],
            cadence_days=cadence_days,
            minimum_frequency_count=spectral_minimum_count,
            samples_per_peak=5.0,
            minimum_cycles=2.0,
            max_frequency_count=15000,
        )
        segment_periods.append(float(summary["v2_ls_dominant_period_days"]))

    if len(segment_periods) >= 1:
        fields["v2_half_period_1_days"] = float(segment_periods[0])
    if len(segment_periods) >= 2:
        fields["v2_half_period_2_days"] = float(segment_periods[1])

    if np.isfinite(global_period_days) and global_period_days > 0:
        agreements: list[bool] = []
        for period in segment_periods:
            error = _period_harmonic_relative_error(global_period_days, period)
            if np.isfinite(error):
                agreements.append(bool(error <= float(period_agreement_fraction)))
        if agreements:
            fields["v2_half_period_consistency_fraction"] = float(np.mean(agreements))

    return fields


def stellar_variability_summary(
    time: np.ndarray,
    values: np.ndarray,
    *,
    cadence_days: float,
    acf_lags: int = 240,
    spectral_frequencies: int = 4000,
    boundaries: VariabilityBoundaryConfig = DEFAULT_BOUNDARIES,
) -> dict[str, object]:
    """Return continuous variability features plus conservative review flags."""
    acf_fields = pairwise_regular_grid_acf(
        values,
        cadence_days=cadence_days,
        max_lag=max(24, int(acf_lags)),
    )
    spectral_fields = lomb_scargle_variability_summary(
        time,
        values,
        cadence_days=cadence_days,
        minimum_frequency_count=max(512, int(spectral_frequencies)),
    )
    segment_fields = segmented_evolution_summary(
        time,
        values,
        cadence_days=cadence_days,
        global_period_days=float(spectral_fields["v2_ls_dominant_period_days"]),
        period_agreement_fraction=boundaries.period_agreement_fraction,
    )

    fap = float(spectral_fields["v2_ls_fap"])
    ls_period = float(spectral_fields["v2_ls_dominant_period_days"])
    acf_period = float(acf_fields["v2_acf_period_candidate_days"])
    half_consistency = float(segment_fields["v2_half_period_consistency_fraction"])
    period_error = _period_harmonic_relative_error(ls_period, acf_period)

    periodic_screen = bool(np.isfinite(fap) and fap <= boundaries.ls_fap_threshold)
    acf_support = bool(np.isfinite(period_error) and period_error <= boundaries.period_agreement_fraction)
    half_support = bool(np.isfinite(half_consistency) and half_consistency >= 0.5)
    coherent_periodic = bool(
        periodic_screen
        and acf_support
        and half_support
    )
    # Pulsation morphology is intentionally not classified at the single-star
    # stage. A significant LS peak or several local periodogram maxima are not
    # sufficient evidence for stellar pulsation: aliases, harmonics and evolving
    # low-frequency variability can generate the same pattern.
    #
    # The population/review stage combines coherence with the distribution of
    # spectral power across physically useful timescale bands.
    pulsation_review = False

    return {
        "scientific_characterization_version": "stellar_variability_v2",
        "v2_periodicity_screen_fap_threshold": float(boundaries.ls_fap_threshold),
        "v2_period_agreement_fraction_threshold": float(boundaries.period_agreement_fraction),
        "v2_periodicity_screen_pass": periodic_screen,
        "v2_periodicity_supported_by_acf": acf_support,
        "v2_periodicity_supported_by_segments": half_support,
        "v2_coherent_periodic_candidate": coherent_periodic,
        "v2_ls_acf_period_relative_error": period_error,
        "v2_pulsation_review_flag": pulsation_review,
        "v2_interpretation_note": (
            "Continuous variability features are primary. Candidate/review flags are operational screens, "
            "not astrophysical classifications; quietness is assigned only after population-relative boundaries."
        ),
        **acf_fields,
        **spectral_fields,
        **segment_fields,
    }


def _numeric_quantile(frame: pd.DataFrame, column: str, quantile: float) -> float:
    if column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.quantile(float(quantile))) if len(values) else float("nan")


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric Series aligned to ``frame`` even when column is absent.

    Population profiling is also used on older CSVs.  Returning an all-NaN
    Series instead of a scalar ``None`` keeps the boundary logic conservative:
    missing evidence cannot accidentally create a positive morphology label.
    """
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def apply_population_variability_boundaries(
    records: pd.DataFrame,
    *,
    boundaries: VariabilityBoundaryConfig = DEFAULT_BOUNDARIES,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Add population-relative descriptive labels and review flags.

    Why population-relative boundaries?
    -----------------------------------
    There is no defensible universal statement such as "scatter < X ppm means a
    quiet star" across all Kepler magnitudes, stellar types, quarters and data
    products.  We therefore define low/high amplitude and low/high evolution
    relative to the *actual comparison population* used in the experiment.

    The returned labels are for interpretation / stratified diagnostics.  Feed
    the continuous MODEL_SELECTION_FEATURE_COLUMNS into XGBoost / RF / neural
    models rather than the labels themselves.
    """
    frame = records.copy()
    if frame.empty:
        return frame, {}

    amplitude_col = "flux_robust_scale"
    memory_col = "v2_acf_max_abs"
    evolution_col = "v2_segment_scale_relative_mad"

    thresholds = {
        "amplitude_q25": _numeric_quantile(frame, amplitude_col, boundaries.low_amplitude_quantile),
        "amplitude_q75": _numeric_quantile(frame, amplitude_col, boundaries.high_amplitude_quantile),
        "memory_q25": _numeric_quantile(frame, memory_col, boundaries.low_memory_quantile),
        "memory_q75": _numeric_quantile(frame, memory_col, boundaries.high_memory_quantile),
        "evolution_q75": _numeric_quantile(frame, evolution_col, boundaries.high_evolution_quantile),
        "median_drift_q75": _numeric_quantile(frame, "v2_segment_median_range_over_global_scale", boundaries.high_evolution_quantile),
        "harmonic_power_ratio_threshold": float(boundaries.harmonic_power_ratio_threshold),
        "weak_abs_acf1": float(boundaries.weak_abs_acf1),
        "strong_abs_acf1": float(boundaries.strong_abs_acf1),
    }

    amplitude = _numeric_series(frame, amplitude_col)
    memory = _numeric_series(frame, memory_col)
    acf1 = _numeric_series(frame, "v2_acf_lag_1").abs()
    evolution = _numeric_series(frame, evolution_col)
    median_drift = _numeric_series(frame, "v2_segment_median_range_over_global_scale")
    harmonic_ratio = _numeric_series(frame, "v2_spectral_harmonic_power_ratio")
    periodic = frame.get("v2_coherent_periodic_candidate", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    periodic_screen = frame.get("v2_periodicity_screen_pass", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    acf_period_support = frame.get("v2_periodicity_supported_by_acf", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    segment_consistency = _numeric_series(frame, "v2_half_period_consistency_fraction")

    low_amp = np.isfinite(amplitude) & np.isfinite(thresholds["amplitude_q25"]) & (amplitude <= thresholds["amplitude_q25"])
    high_amp = np.isfinite(amplitude) & np.isfinite(thresholds["amplitude_q75"]) & (amplitude >= thresholds["amplitude_q75"])
    low_memory = np.isfinite(memory) & np.isfinite(thresholds["memory_q25"]) & (memory <= thresholds["memory_q25"])
    high_memory = np.isfinite(memory) & np.isfinite(thresholds["memory_q75"]) & (memory >= thresholds["memory_q75"])
    high_evolution = (
        (np.isfinite(evolution) & np.isfinite(thresholds["evolution_q75"]) & (evolution >= thresholds["evolution_q75"]))
        | (np.isfinite(median_drift) & np.isfinite(thresholds["median_drift_q75"]) & (median_drift >= thresholds["median_drift_q75"]))
    )

    frame["v2_amplitude_population_label"] = np.select(
        [low_amp, high_amp],
        ["low", "high"],
        default="typical",
    )
    frame["v2_memory_population_label"] = np.select(
        [low_memory, high_memory],
        ["low", "high"],
        default="typical",
    )

    # Convenience ACF descriptor.  The 0.2 / 0.5 cutoffs are an operational
    # communication aid only; model-selection uses the continuous ACF value.
    frame["v2_acf1_operational_label"] = np.select(
        [acf1 < boundaries.weak_abs_acf1, acf1 >= boundaries.strong_abs_acf1],
        ["weak", "strong"],
        default="moderate",
    )

    # QUIET CANDIDATE -- deliberately strict and multivariate.
    # A low-scatter light curve with coherent periodicity, strong memory or
    # substantial amplitude/level evolution is *not* called quiet.
    quiet = (
            low_amp
            & low_memory
            & (~periodic)
            & (~high_evolution)
    )
    frame["v2_quiet_candidate"] = quiet
    frame["v2_low_scatter_structured_candidate"] = low_amp & (~quiet)

    # Correlated stochastic morphology: strong relative memory without a robust
    # periodic candidate.  This is a statistical morphology, not a physical
    # statement about the origin of the variability.
    frame["v2_correlated_stochastic_candidate"] = high_memory & (~periodic)
    frame["v2_evolving_variability_candidate"] = high_evolution

    # Quasi-periodic morphology: independent LS + ACF evidence exists, but the
    # variability amplitude/location evolves strongly or the period is not
    # stable in both broad time segments.  This is deliberately allowed to
    # overlap the coherent-periodic flag when one half agrees and the other does
    # not; the single-valued "dominant behaviour" helper resolves that overlap
    # by preferring the more specific quasi-periodic description.
    inconsistent_segments = np.isfinite(segment_consistency) & (segment_consistency < 1.0)
    quasi_periodic = (
            periodic_screen
            & acf_period_support
            & (high_evolution | inconsistent_segments)
    )
    frame["v2_quasi_periodic_candidate"] = quasi_periodic

    # Rotation / star-spot review flag.  Require periodic evidence supported in
    # both the frequency and ACF domains *plus* a detectable harmonic component.
    # This matches the documented morphology and is intentionally conservative:
    # it is a review flag, never a physical rotation/star-spot classification.
    harmonic_support = (
        np.isfinite(harmonic_ratio)
        & (harmonic_ratio >= thresholds["harmonic_power_ratio_threshold"])
    )
    frame["v2_rotation_spot_review_flag"] = (
        (periodic | quasi_periodic)
        & acf_period_support
        & harmonic_support
    )

    # Pulsation review is based on coherent periodicity plus substantial
    # short-timescale spectral power. Raw local-peak counts are deliberately
    # avoided because aliases, harmonics and broad low-frequency structure can
    # create many periodogram maxima without representing independent modes.
    # This remains an operational review flag, not a physical classification.
    short_period_power = (
        _numeric_series(
            frame,
            "spectral_power_fraction_period_lt_0_5d",
        )
        + _numeric_series(
            frame,
            "spectral_power_fraction_period_0_5_to_2d",
        )
    )

    frame["v2_pulsation_review_flag"] = (
        periodic
        & np.isfinite(short_period_power)
        & (short_period_power >= 0.25)
    )

    return frame, thresholds


def _population_percentile_rank(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a [0, 1] population rank aligned to ``frame``."""
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.rank(method="average", pct=True)


def assign_dominant_statistical_behaviour(records: pd.DataFrame) -> pd.DataFrame:
    """Add one population-relative navigation label per characterized star.

    This is intentionally a descriptive summary of the seven scientific domains,
    not an astrophysical classifier and not an ML target.  Explicit v2 candidate
    flags take precedence; the continuous canonical variables provide a fallback
    so every star receives a useful label instead of a large "other" bucket.
    """
    frame = records.copy()
    if frame.empty:
        frame["v2_dominant_statistical_behaviour"] = pd.Series(dtype=object)
        return frame

    p_scatter = _population_percentile_rank(frame, "flux_robust_scale")
    p_acf1 = _population_percentile_rank(frame, "v2_acf_lag_1")
    p_tau = _population_percentile_rank(frame, "v2_acf_decay_e_days")
    p_spectral = _population_percentile_rank(frame, "v2_spectral_concentration")
    p_harmonic = _population_percentile_rank(frame, "v2_spectral_harmonic_power_ratio")
    p_evolution = _population_percentile_rank(frame, "v2_segment_scale_relative_mad")
    p_outlier = _population_percentile_rank(frame, "flux_outlier_fraction")

    if "flux_skewness" in frame.columns:
        abs_skew = pd.to_numeric(frame["flux_skewness"], errors="coerce").abs()
        p_abs_skew = abs_skew.rank(method="average", pct=True)
    else:
        p_abs_skew = pd.Series(np.nan, index=frame.index, dtype=float)

    agreement_error = _population_percentile_rank(frame, "v2_ls_acf_period_relative_error")
    agreement_strength = 1.0 - agreement_error

    memory_strength = pd.concat([p_acf1, p_tau], axis=1).max(axis=1, skipna=True)
    periodic_strength = pd.concat(
        [p_spectral, p_harmonic, agreement_strength],
        axis=1,
    ).mean(axis=1, skipna=True)
    distribution_strength = pd.concat(
        [p_abs_skew, p_outlier],
        axis=1,
    ).max(axis=1, skipna=True)

    labels = pd.Series("Mixed / complex", index=frame.index, dtype=object)

    explicit_priority = (
        ("v2_quasi_periodic_candidate", "Quasi-periodic / structured"),
        ("v2_coherent_periodic_candidate", "Coherent periodic"),
        ("v2_evolving_variability_candidate", "Evolving variability"),
        ("v2_correlated_stochastic_candidate", "Long-memory / correlated"),
        ("v2_low_scatter_structured_candidate", "Low-scatter structured"),
        ("v2_quiet_candidate", "Quiet / low variability"),
    )

    for idx, row in frame.iterrows():
        explicit_label = None
        for column, label in explicit_priority:
            if column not in row.index or pd.isna(row[column]):
                continue
            value = row[column]
            if isinstance(value, str):
                flag = value.strip().lower() in {"true", "1", "yes", "y"}
            else:
                flag = bool(value)
            if flag:
                explicit_label = label
                break
        if explicit_label is not None:
            labels.at[idx] = explicit_label
            continue

        def finite_or_middle(value: float) -> float:
            return 0.5 if not np.isfinite(value) else float(value)

        scatter_rank = finite_or_middle(p_scatter.at[idx])
        memory_rank = finite_or_middle(memory_strength.at[idx])
        tau_rank = finite_or_middle(p_tau.at[idx])
        spectral_rank = finite_or_middle(p_spectral.at[idx])
        periodic_rank = finite_or_middle(periodic_strength.at[idx])
        distribution_rank = finite_or_middle(distribution_strength.at[idx])
        evolution_rank = finite_or_middle(p_evolution.at[idx])

        if (
            scatter_rank <= 0.25
            and memory_rank <= 0.50
            and periodic_rank <= 0.50
            and evolution_rank <= 0.50
            and distribution_rank <= 0.65
        ):
            label = "Quiet / low variability"
        elif scatter_rank <= 0.35 and max(
            memory_rank,
            periodic_rank,
            evolution_rank,
            distribution_rank,
        ) >= 0.65:
            label = "Low-scatter structured"
        elif periodic_rank >= 0.75 and spectral_rank >= 0.60:
            label = "Coherent periodic"
        elif periodic_rank >= 0.65:
            label = "Quasi-periodic / structured"
        elif evolution_rank >= 0.80:
            label = "Evolving variability"
        elif memory_rank >= 0.75:
            label = "Long-memory / correlated"
        elif distribution_rank >= 0.85:
            label = "Tail-heavy / asymmetric"
        elif scatter_rank >= 0.75:
            label = "High-amplitude / high-scatter"
        elif tau_rank <= 0.40 and periodic_rank < 0.65 and evolution_rank < 0.80:
            label = "Short-memory stochastic"
        else:
            label = "Mixed / complex"

        labels.at[idx] = label

    frame["v2_dominant_statistical_behaviour"] = pd.Categorical(
        labels,
        categories=list(DOMINANT_STATISTICAL_BEHAVIOUR_ORDER),
        ordered=True,
    ).astype(str)
    return frame


def model_selection_feature_frame(records: pd.DataFrame) -> pd.DataFrame:
    """Return only continuous scientific features intended for router models.

    Missing columns are created as NaN so that older characterization files can
    be concatenated safely.  Gap/QC variables and derived regime labels are
    intentionally excluded.
    """
    output = pd.DataFrame(index=records.index)
    for column in MODEL_SELECTION_FEATURE_COLUMNS:
        if column in records.columns:
            output[column] = pd.to_numeric(records[column], errors="coerce")
        else:
            output[column] = np.nan
    return output
