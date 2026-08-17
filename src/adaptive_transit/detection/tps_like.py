"""TPS-like adaptive wavelet matched filter for Kepler cadence-grid light curves.

This module is intentionally labelled *TPS-like*.  It reproduces the core
scientific structure of Kepler TPS that matters for this project:

1. an overcomplete / undecimated wavelet representation;
2. time-varying, scale-dependent noise estimation;
3. the same whitening weights applied to the data and trial transit pulse;
4. a single-event matched-filter statistic (SES-like numerator/denominator);
5. coherent periodic combination into a multiple-event statistic (MES-like).

It is NOT a drop-in reproduction of the SOC 9.3 TPS implementation.  In
particular it does not implement the production Kepler vetoes, robust statistic,
bootstrap false-alarm machinery, harmonic-removal logic, quarter stitching, or
all gap-fill / edge-correction details.  Keep the method name ``tps_like`` in
plots, tables, and manuscript text until those differences are resolved.

The implementation is designed for the project's explicit cadence-grid
representation: contiguous usable segments are transformed independently, so
missing Kepler cadences are never interpolated across merely to satisfy a
wavelet transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, log2
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class PreparedWaveletSegment:
    """Noise model for one contiguous usable cadence segment."""

    indices: np.ndarray
    original_length: int
    padded_length: int
    pad_right: int
    level: int
    inverse_variance_bands: tuple[np.ndarray, ...]
    median_scale_bands: tuple[float, ...]


@dataclass(frozen=True)
class PreparedTPSLikeNoiseModel:
    """Fixed adaptive wavelet whitening operator fitted on the base star."""

    n_points: int
    wavelet: str
    max_level: int
    noise_window_cadences: int
    segments: tuple[PreparedWaveletSegment, ...]

    @property
    def segment_count(self) -> int:
        return len(self.segments)


@dataclass(frozen=True)
class SingleEventSeries:
    """SES-like matched-filter components for one trial pulse duration."""

    duration_cadences: int
    numerator: np.ndarray
    denominator_squared: np.ndarray
    statistic: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class PeriodicSearchCandidate:
    """Best MES-like periodic combination for one period/duration trial."""

    period_cadences: int
    epoch_phase_cadence: int
    duration_cadences: int
    mes: float
    observed_event_count: int
    expected_event_count: int
    observability_fraction: float


@dataclass(frozen=True)
class TPSLikeSearchResult:
    """Top candidate and compact diagnostics from the TPS-like search."""

    period_days: float
    epoch_days: float
    duration_hours: float
    mes: float
    max_ses: float
    observed_event_count: int
    expected_event_count: int
    observability_fraction: float
    period_cadences: int
    epoch_phase_cadence: int
    duration_cadences: int
    n_period_trials: int
    n_duration_trials: int
    cadence_days: float
    wavelet: str
    segment_count: int

    def to_summary_dict(self) -> dict[str, float | int | str]:
        return {
            "period_days": self.period_days,
            "epoch_days": self.epoch_days,
            "duration_hours": self.duration_hours,
            "mes": self.mes,
            "max_ses": self.max_ses,
            "observed_event_count": self.observed_event_count,
            "expected_event_count": self.expected_event_count,
            "observability_fraction": self.observability_fraction,
            "period_cadences": self.period_cadences,
            "epoch_phase_cadence": self.epoch_phase_cadence,
            "duration_cadences": self.duration_cadences,
            "n_period_trials": self.n_period_trials,
            "n_duration_trials": self.n_duration_trials,
            "cadence_days": self.cadence_days,
            "wavelet": self.wavelet,
            "segment_count": self.segment_count,
        }


def _import_pywt():
    import pywt

    return pywt


def robust_scale(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    med = float(np.median(x))
    scale = float(1.4826 * np.median(np.abs(x - med)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(x, ddof=1))
    return scale


def median_positive_cadence_days(time: np.ndarray) -> float:
    t = np.asarray(time, dtype=float).reshape(-1)
    finite = np.flatnonzero(np.isfinite(t))
    if finite.size < 2:
        raise ValueError("At least two finite time samples are required.")
    dt = np.diff(t[finite]) / np.diff(finite)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Could not estimate a positive cadence interval.")
    return float(np.median(dt))


def _rolling_robust_scale(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling MAD with a conservative global floor."""

    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        return x.copy()
    window = max(11, int(window))
    if window % 2 == 0:
        window += 1
    window = min(window, x.size if x.size % 2 == 1 else max(1, x.size - 1))
    if window < 3:
        global_sigma = robust_scale(x)
        global_sigma = global_sigma if np.isfinite(global_sigma) and global_sigma > 0 else 1.0
        return np.full(x.shape, global_sigma, dtype=float)

    series = pd.Series(x)
    min_periods = max(5, min(window, window // 4))
    center = series.rolling(window, center=True, min_periods=min_periods).median()
    deviation = (series - center).abs()
    local = 1.4826 * deviation.rolling(
        window, center=True, min_periods=min_periods
    ).median()

    global_sigma = robust_scale(x)
    if not np.isfinite(global_sigma) or global_sigma <= 0:
        global_sigma = float(np.nanstd(x))
    if not np.isfinite(global_sigma) or global_sigma <= 0:
        global_sigma = 1.0

    sigma = pd.to_numeric(local, errors="coerce").to_numpy(dtype=float)
    sigma[~np.isfinite(sigma)] = global_sigma
    floor_value = max(1.0e-12, 0.05 * global_sigma)
    sigma = np.maximum(sigma, floor_value)
    return sigma


def _choose_level(length: int, max_level: int) -> int:
    if length < 8:
        return 0
    natural = max(1, int(floor(log2(length))) - 2)
    return max(1, min(int(max_level), natural))


def _pad_to_level(values: np.ndarray, level: int) -> tuple[np.ndarray, int]:
    x = np.asarray(values, dtype=float).reshape(-1)
    if level < 1:
        return x.copy(), 0
    block = 2**int(level)
    pad_right = (-x.size) % block
    if pad_right == 0:
        return x.copy(), 0
    mode = "reflect" if x.size > 1 else "edge"
    return np.pad(x, (0, pad_right), mode=mode), int(pad_right)


def _swt_bands(values: np.ndarray, wavelet: str, level: int) -> tuple[np.ndarray, ...]:
    """Return all SWT detail bands plus the coarsest approximation band."""

    if level < 1:
        return (np.asarray(values, dtype=float).reshape(-1),)
    pywt = _import_pywt()
    coefficients = pywt.swt(
        np.asarray(values, dtype=float),
        wavelet,
        level=int(level),
        trim_approx=True,
        norm=True,
    )
    # With trim_approx=True PyWavelets returns [cA_n, cD_n, ..., cD_1].
    approximation = np.asarray(coefficients[0], dtype=float)
    details = [np.asarray(array, dtype=float) for array in coefficients[1:]]
    return tuple(details + [approximation])


def prepare_tps_like_noise_model(
    values: np.ndarray,
    segment_id: np.ndarray,
    *,
    wavelet: str = "db6",
    max_level: int = 6,
    noise_window_cadences: int = 193,
    min_segment_cadences: int = 32,
) -> PreparedTPSLikeNoiseModel:
    """Fit scale-dependent local noise estimates on the *uninjected* star.

    The noise model is then held fixed for all injections into the same star.
    This prevents the injected transit itself from redefining the whitening
    weights and mirrors the project's fixed-filter signal-transfer philosophy.
    """

    y = np.asarray(values, dtype=float).reshape(-1)
    seg = np.asarray(segment_id, dtype=int).reshape(-1)
    if y.shape != seg.shape:
        raise ValueError("values and segment_id must have the same shape.")
    if max_level < 1:
        raise ValueError("max_level must be positive.")

    prepared: list[PreparedWaveletSegment] = []
    for label in sorted(int(x) for x in np.unique(seg) if int(x) >= 0):
        indices = np.flatnonzero((seg == label) & np.isfinite(y))
        if indices.size < int(min_segment_cadences):
            continue
        # segment_id is expected to denote a contiguous usable cadence run.
        if indices.size > 1 and np.any(np.diff(indices) != 1):
            raise ValueError(f"segment_id={label} is not contiguous on the cadence grid.")

        segment_values = y[indices]
        level = _choose_level(segment_values.size, max_level)
        padded, pad_right = _pad_to_level(segment_values, level)
        bands = _swt_bands(padded, wavelet, level)
        inverse_variance = []
        median_scales = []
        for band in bands:
            sigma = _rolling_robust_scale(band, noise_window_cadences)
            inverse_variance.append(1.0 / np.square(sigma))
            median_scales.append(float(np.median(sigma)))

        prepared.append(
            PreparedWaveletSegment(
                indices=indices.astype(int),
                original_length=int(indices.size),
                padded_length=int(padded.size),
                pad_right=int(pad_right),
                level=int(level),
                inverse_variance_bands=tuple(inverse_variance),
                median_scale_bands=tuple(median_scales),
            )
        )

    if not prepared:
        raise ValueError("No contiguous segment is long enough for TPS-like whitening.")
    return PreparedTPSLikeNoiseModel(
        n_points=int(y.size),
        wavelet=str(wavelet),
        max_level=int(max_level),
        noise_window_cadences=int(noise_window_cadences),
        segments=tuple(prepared),
    )


def _centered_square_pulse(length: int, duration_cadences: int) -> np.ndarray:
    width = max(1, int(duration_cadences))
    width = min(width, int(length))
    pulse = np.zeros(int(length), dtype=float)
    center = int(length) // 2
    start = max(0, center - width // 2)
    stop = min(int(length), start + width)
    start = max(0, stop - width)
    pulse[start:stop] = -1.0
    return pulse


def compute_single_event_series(
    values: np.ndarray,
    prepared: PreparedTPSLikeNoiseModel,
    *,
    duration_cadences: int,
) -> SingleEventSeries:
    """Compute SES-like correlation and normalization series.

    Both data and unit-depth square pulse are transformed through the same SWT
    bank.  The scale/time-dependent inverse-variance arrays were fitted on the
    original star and are held fixed.
    """

    y = np.asarray(values, dtype=float).reshape(-1)
    if y.size != prepared.n_points:
        raise ValueError("values length does not match the prepared noise model.")
    width = max(1, int(duration_cadences))

    numerator = np.full(y.shape, np.nan, dtype=float)
    denominator_squared = np.full(y.shape, np.nan, dtype=float)

    for segment in prepared.segments:
        segment_values = y[segment.indices]
        if not np.isfinite(segment_values).all():
            raise ValueError("Prepared segment contains non-finite searched values.")
        padded, pad_right = _pad_to_level(segment_values, segment.level)
        if padded.size != segment.padded_length or pad_right != segment.pad_right:
            raise ValueError("Prepared wavelet geometry does not match searched values.")

        data_bands = _swt_bands(padded, prepared.wavelet, segment.level)
        pulse = _centered_square_pulse(segment.padded_length, width)
        template_bands = _swt_bands(pulse, prepared.wavelet, segment.level)
        if len(data_bands) != len(segment.inverse_variance_bands):
            raise ValueError("Wavelet band count changed after preparation.")

        seg_num = np.zeros(segment.padded_length, dtype=float)
        seg_den = np.zeros(segment.padded_length, dtype=float)
        for data_band, template_band, inv_var in zip(
            data_bands, template_bands, segment.inverse_variance_bands
        ):
            weighted_data = data_band * inv_var
            seg_num += fftconvolve(weighted_data, template_band[::-1], mode="same")
            seg_den += fftconvolve(inv_var, np.square(template_band[::-1]), mode="same")

        seg_num = seg_num[: segment.original_length]
        seg_den = seg_den[: segment.original_length]

        # Guard wavelet/pulse boundaries rather than manufacturing information
        # across segment edges.
        guard = min(
            segment.original_length // 4,
            max(width, 2 ** min(segment.level, 5)),
        )
        local_valid = np.isfinite(seg_num) & np.isfinite(seg_den) & (seg_den > 0)
        if guard > 0 and segment.original_length > 2 * guard:
            local_valid[:guard] = False
            local_valid[-guard:] = False

        local_num = np.full(segment.original_length, np.nan, dtype=float)
        local_den = np.full(segment.original_length, np.nan, dtype=float)
        local_num[local_valid] = seg_num[local_valid]
        local_den[local_valid] = seg_den[local_valid]
        numerator[segment.indices] = local_num
        denominator_squared[segment.indices] = local_den

    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator_squared)
        & (denominator_squared > 0)
    )
    statistic = np.full(y.shape, np.nan, dtype=float)
    statistic[valid] = numerator[valid] / np.sqrt(denominator_squared[valid])
    return SingleEventSeries(
        duration_cadences=width,
        numerator=numerator,
        denominator_squared=denominator_squared,
        statistic=statistic,
        valid_mask=valid,
    )


def _expected_event_count(n_points: int, period: int, phase: int) -> int:
    if phase >= n_points:
        return 0
    return int(1 + (n_points - 1 - phase) // period)


def combine_periodic_events(
    numerator: np.ndarray,
    denominator_squared: np.ndarray,
    period_cadences: int,
    *,
    duration_cadences: int,
    min_events: int = 3,
) -> PeriodicSearchCandidate | None:
    """Fold SES numerator/denominator terms into an MES-like statistic."""

    num = np.asarray(numerator, dtype=float).reshape(-1)
    den = np.asarray(denominator_squared, dtype=float).reshape(-1)
    if num.shape != den.shape:
        raise ValueError("numerator and denominator_squared must have the same shape.")
    period = int(period_cadences)
    if period < 2:
        raise ValueError("period_cadences must be at least 2.")
    valid_idx = np.flatnonzero(np.isfinite(num) & np.isfinite(den) & (den > 0))
    if valid_idx.size < int(min_events):
        return None

    phase = valid_idx % period
    sum_num = np.bincount(phase, weights=num[valid_idx], minlength=period)
    sum_den = np.bincount(phase, weights=den[valid_idx], minlength=period)
    counts = np.bincount(phase, minlength=period)
    mes = np.full(period, np.nan, dtype=float)
    eligible = (counts >= int(min_events)) & (sum_den > 0)
    mes[eligible] = sum_num[eligible] / np.sqrt(sum_den[eligible])
    if not np.isfinite(mes).any():
        return None

    best_phase = int(np.nanargmax(mes))
    expected = _expected_event_count(num.size, period, best_phase)
    observed = int(counts[best_phase])
    fraction = float(observed / expected) if expected > 0 else float("nan")
    return PeriodicSearchCandidate(
        period_cadences=period,
        epoch_phase_cadence=best_phase,
        duration_cadences=int(duration_cadences),
        mes=float(mes[best_phase]),
        observed_event_count=observed,
        expected_event_count=expected,
        observability_fraction=fraction,
    )


def duration_hours_to_cadences(duration_hours: float, cadence_days: float) -> int:
    if duration_hours <= 0 or cadence_days <= 0:
        raise ValueError("duration_hours and cadence_days must be positive.")
    return max(1, int(round((float(duration_hours) / 24.0) / float(cadence_days))))


def period_grid_to_unique_cadences(
    min_period_days: float,
    max_period_days: float,
    cadence_days: float,
) -> np.ndarray:
    if min_period_days <= 0 or max_period_days <= min_period_days or cadence_days <= 0:
        raise ValueError("Require 0 < min_period_days < max_period_days and cadence_days > 0.")
    start = max(2, int(ceil(float(min_period_days) / cadence_days)))
    stop = max(start, int(floor(float(max_period_days) / cadence_days)))
    return np.arange(start, stop + 1, dtype=int)


def cadence_phase_to_epoch_days(
    time: np.ndarray,
    cadence_days: float,
    phase_cadence: int,
) -> float:
    t = np.asarray(time, dtype=float).reshape(-1)
    idx = np.flatnonzero(np.isfinite(t))
    if idx.size == 0:
        return float("nan")
    origin = float(np.median(t[idx] - idx * float(cadence_days)))
    return origin + int(phase_cadence) * float(cadence_days)


def run_tps_like_search(
    time: np.ndarray,
    values: np.ndarray,
    segment_id: np.ndarray,
    *,
    prepared_noise_model: PreparedTPSLikeNoiseModel | None = None,
    min_period_days: float = 1.0,
    max_period_days: float = 15.0,
    duration_hours_grid: Iterable[float] = (1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0),
    wavelet: str = "db6",
    max_level: int = 6,
    noise_window_cadences: int = 193,
    min_segment_cadences: int = 32,
    min_events: int = 3,
) -> dict:
    """Run the compact TPS-like duration/period/epoch search.

    Periods are represented on the integer cadence grid for this POC.  The
    output includes observability so a high MES supported by only a small
    fraction of expected events is visible to downstream QC.
    """

    t = np.asarray(time, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    seg = np.asarray(segment_id, dtype=int).reshape(-1)
    if t.shape != y.shape or t.shape != seg.shape:
        raise ValueError("time, values, and segment_id must have the same shape.")
    cadence_days = median_positive_cadence_days(t)

    prepared = prepared_noise_model
    if prepared is None:
        prepared = prepare_tps_like_noise_model(
            y,
            seg,
            wavelet=wavelet,
            max_level=max_level,
            noise_window_cadences=noise_window_cadences,
            min_segment_cadences=min_segment_cadences,
        )

    periods = period_grid_to_unique_cadences(
        min_period_days, max_period_days, cadence_days
    )
    durations = sorted(
        set(duration_hours_to_cadences(float(x), cadence_days) for x in duration_hours_grid)
    )
    if not durations:
        raise ValueError("No duration trials were generated.")

    best: PeriodicSearchCandidate | None = None
    best_ses = float("nan")
    compact_rows: list[dict[str, float | int]] = []

    for duration in durations:
        ses = compute_single_event_series(y, prepared, duration_cadences=duration)
        local_max_ses = (
            float(np.nanmax(ses.statistic)) if np.isfinite(ses.statistic).any() else float("nan")
        )
        for period in periods:
            if period <= duration:
                continue
            candidate = combine_periodic_events(
                ses.numerator,
                ses.denominator_squared,
                int(period),
                duration_cadences=duration,
                min_events=min_events,
            )
            if candidate is None:
                continue
            compact_rows.append(
                {
                    "period_cadences": candidate.period_cadences,
                    "period_days": candidate.period_cadences * cadence_days,
                    "duration_cadences": duration,
                    "duration_hours": duration * cadence_days * 24.0,
                    "epoch_phase_cadence": candidate.epoch_phase_cadence,
                    "mes": candidate.mes,
                    "observed_event_count": candidate.observed_event_count,
                    "expected_event_count": candidate.expected_event_count,
                    "observability_fraction": candidate.observability_fraction,
                }
            )
            if best is None or candidate.mes > best.mes:
                best = candidate
                best_ses = local_max_ses

    if best is None:
        raise ValueError("No TPS-like periodic candidate satisfied the minimum-event rule.")

    result = TPSLikeSearchResult(
        period_days=float(best.period_cadences * cadence_days),
        epoch_days=float(
            cadence_phase_to_epoch_days(t, cadence_days, best.epoch_phase_cadence)
        ),
        duration_hours=float(best.duration_cadences * cadence_days * 24.0),
        mes=float(best.mes),
        max_ses=float(best_ses),
        observed_event_count=int(best.observed_event_count),
        expected_event_count=int(best.expected_event_count),
        observability_fraction=float(best.observability_fraction),
        period_cadences=int(best.period_cadences),
        epoch_phase_cadence=int(best.epoch_phase_cadence),
        duration_cadences=int(best.duration_cadences),
        n_period_trials=int(len(periods)),
        n_duration_trials=int(len(durations)),
        cadence_days=float(cadence_days),
        wavelet=str(prepared.wavelet),
        segment_count=int(prepared.segment_count),
    )
    periodogram = pd.DataFrame(compact_rows)
    if not periodogram.empty:
        periodogram = periodogram.sort_values("mes", ascending=False).reset_index(drop=True)
    return {
        "summary": result.to_summary_dict(),
        "periodogram": periodogram,
        "prepared_noise_model": prepared,
    }
