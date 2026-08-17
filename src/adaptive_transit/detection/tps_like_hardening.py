"""Event-consistency re-ranking for the TPS-like proof-of-concept search.

This module implements a transparent *TPS-like* proof-of-concept hardening
layer.  It is not a reproduction of the Kepler SOC robust statistic or its
chi-square discriminators.

Scientific design:
- start from independent high-MES candidates returned by the existing search;
- measure every predicted event directly in the searched light curve;
- retain the v2 observability/sign/dominance/leave-one-out diagnostics;
- add a weighted event-depth chi-square consistency test;
- add an odd/even depth-consistency test;
- add a Huber-weighted robust event statistic that down-weights outlier events;
- apply pre-specified event-support vetoes, use depth/odd-even as diagnostics,
  and only invoke the v2 support penalty when the raw MES winner fails;
- calibrate the resulting vetoed score empirically under the shared nulls.

The chi-square and robust quantities below are intentionally labelled
"TPS-like" / POC.  They must not be described as the exact Kepler SOC tests.
"""

from __future__ import annotations

import math

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import chi2


NORMAL_MEDIAN_STDERR_FACTOR = math.sqrt(math.pi / 2.0)
VETO_FLAG_COLUMNS = (
    "fails_min_events",
    "fails_observability",
    "fails_positive_events",
    "fails_single_event_dominance",
    "fails_leave_one_out",
    "fails_depth_chi2",
    "fails_odd_even",
    "fails_robust_sign",
)


@dataclass(frozen=True)
class EventConsistencyConfig:
    """Frozen POC settings for TPS-like event-consistency validation."""

    top_n_candidates: int = 64
    # Candidate-bank de-duplication.  Two rows are treated as the same local
    # peak only when both period and phase are close.  Harmonics such as P and
    # P/2 therefore remain separate candidates.
    duplicate_period_fraction: float = 0.01
    duplicate_phase_duration_factor: float = 0.75
    min_valid_events: int = 3
    in_transit_half_width_factor: float = 0.55
    local_guard_factor: float = 1.50
    local_outer_factor: float = 4.00
    min_in_transit_samples: int = 2
    min_out_of_transit_samples: int = 8

    # These are descriptive audit thresholds.  The headline score is calibrated
    # empirically and does not use this boolean as an uncalibrated hard veto.
    min_observability_fraction: float = 0.50
    min_positive_event_fraction: float = 2.0 / 3.0
    max_single_event_fraction: float = 0.70
    min_leave_one_out_ratio: float = 0.50

    # V3 transit-consistency diagnostics.  Depths are measured as local median
    # differences, so the error estimate uses the Normal-theory median standard
    # error factor sqrt(pi/2), not the mean standard error.
    depth_error_median_factor: float = NORMAL_MEDIAN_STDERR_FACTOR

    # V3 diagnostic policy.  A small chi-square or odd/even p-value alone is
    # not an operational hard veto: with many observed events, small stochastic
    # or windowing differences can become formally significant.  These flags
    # therefore require a material magnitude/effect-size failure and are carried
    # into the output for interpretation and empirical calibration.
    min_event_depth_chi2_pvalue: float = 0.01
    max_event_depth_reduced_chi2_for_veto: float = 3.0
    max_event_depth_relative_mad_for_veto: float = 1.0
    min_odd_even_group_events: int = 3
    min_odd_even_depth_pvalue: float = 0.01
    min_odd_even_depth_sigma_for_veto: float = 4.0
    min_odd_even_depth_fraction_for_veto: float = 0.50
    min_robust_event_snr: float = 0.0
    huber_tuning_constant: float = 1.345


DEFAULT_EVENT_CONSISTENCY_CONFIG = EventConsistencyConfig()


def config_dict(config: EventConsistencyConfig = DEFAULT_EVENT_CONSISTENCY_CONFIG) -> dict:
    return asdict(config)


def _finite_number(value, default=np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _first_numeric(record: Mapping, names, default=np.nan) -> float:
    for name in names:
        if name in record:
            value = _finite_number(record[name], default=np.nan)
            if np.isfinite(value):
                return value
    return float(default)


def _median_cadence_days(time: np.ndarray) -> float:
    finite = np.asarray(time, dtype=float)
    finite = np.unique(finite[np.isfinite(finite)])
    if finite.size < 2:
        return float("nan")
    diff = np.diff(finite)
    diff = diff[np.isfinite(diff) & (diff > 0)]
    return float(np.median(diff)) if diff.size else float("nan")


def _candidate_from_record(record: Mapping, time: np.ndarray) -> dict:
    cadence_days = _median_cadence_days(time)
    t_min = float(np.nanmin(time[np.isfinite(time)]))

    period_days = _first_numeric(record, ("period_days", "recovered_period_days", "period"))
    if not np.isfinite(period_days):
        period_cadences = _first_numeric(record, ("period_cadences",))
        if np.isfinite(period_cadences) and np.isfinite(cadence_days):
            period_days = period_cadences * cadence_days

    epoch_days = _first_numeric(
        record,
        ("epoch_days", "recovered_epoch_days", "epoch", "transit_time", "t0"),
    )
    if not np.isfinite(epoch_days):
        epoch_cadence = _first_numeric(
            record,
            (
                "epoch_phase_cadence",  # current TPS-like periodogram schema
                "epoch_cadence",
                "epoch_cadences",
                "phase_cadence",
                "phase_cadences",
            ),
        )
        if np.isfinite(epoch_cadence) and np.isfinite(cadence_days):
            epoch_days = t_min + epoch_cadence * cadence_days

    duration_hours = _first_numeric(
        record,
        ("duration_hours", "recovered_duration_hours", "requested_duration_hours"),
    )
    if not np.isfinite(duration_hours):
        duration_days = _first_numeric(record, ("duration_days",))
        if np.isfinite(duration_days):
            duration_hours = 24.0 * duration_days
    if not np.isfinite(duration_hours):
        duration_cadences = _first_numeric(record, ("duration_cadences",))
        if np.isfinite(duration_cadences) and np.isfinite(cadence_days):
            duration_hours = duration_cadences * cadence_days * 24.0
    if not np.isfinite(duration_hours):
        # Generic "duration" is commonly stored in days by transit-search APIs.
        generic_duration = _first_numeric(record, ("duration",))
        if np.isfinite(generic_duration):
            duration_hours = (
                generic_duration * 24.0 if generic_duration <= 1.0 else generic_duration
            )

    mes = _first_numeric(record, ("mes", "score", "statistic"))
    max_ses = _first_numeric(record, ("max_ses", "ses_max"))

    return {
        "period_days": period_days,
        "epoch_days": epoch_days,
        "duration_hours": duration_hours,
        "mes": mes,
        "max_ses": max_ses,
    }



def _circular_epoch_separation_days(
    period_days: float,
    epoch_a_days: float,
    epoch_b_days: float,
) -> float:
    """Smallest phase separation between two epochs on the same period."""
    period_days = float(period_days)
    if not np.isfinite(period_days) or period_days <= 0:
        return float("inf")
    delta = float(epoch_a_days) - float(epoch_b_days)
    wrapped = (delta + 0.5 * period_days) % period_days - 0.5 * period_days
    return float(abs(wrapped))


def _same_local_candidate_family(
    candidate: Mapping,
    selected: Mapping,
    *,
    duplicate_period_fraction: float,
    duplicate_phase_duration_factor: float,
) -> bool:
    """Return True only for nearby samples of the same period/phase peak.

    This intentionally does *not* collapse harmonics (e.g. 5 d vs 10 d).
    Duration is used only to set the phase-coincidence tolerance, so a nearby
    period with a materially different phase can still survive as an
    independent candidate.
    """
    p_a = _finite_number(candidate.get("period_days"))
    p_b = _finite_number(selected.get("period_days"))
    if not (np.isfinite(p_a) and np.isfinite(p_b) and p_a > 0 and p_b > 0):
        return False

    period_fraction = abs(p_a - p_b) / max(min(p_a, p_b), 1e-12)
    if period_fraction > float(duplicate_period_fraction):
        return False

    e_a = _finite_number(candidate.get("epoch_days"))
    e_b = _finite_number(selected.get("epoch_days"))
    if not (np.isfinite(e_a) and np.isfinite(e_b)):
        return True

    d_a = _finite_number(candidate.get("duration_hours"))
    d_b = _finite_number(selected.get("duration_hours"))
    duration_days = max(d_a, d_b) / 24.0
    phase_tolerance = max(
        float(duplicate_phase_duration_factor) * duration_days,
        1e-8,
    )
    reference_period = 0.5 * (p_a + p_b)
    phase_separation = _circular_epoch_separation_days(
        reference_period, e_a, e_b
    )
    return phase_separation <= phase_tolerance


def select_independent_candidates(
    candidates: pd.DataFrame,
    *,
    top_n: int,
    duplicate_period_fraction: float,
    duplicate_phase_duration_factor: float,
) -> pd.DataFrame:
    """Greedily retain high-MES candidates from distinct local peak families."""
    if candidates.empty:
        return candidates.copy()

    ordered = candidates.sort_values("mes", ascending=False).reset_index(drop=True)
    selected_rows = []

    for _, row in ordered.iterrows():
        record = row.to_dict()
        duplicate = any(
            _same_local_candidate_family(
                record,
                existing,
                duplicate_period_fraction=duplicate_period_fraction,
                duplicate_phase_duration_factor=duplicate_phase_duration_factor,
            )
            for existing in selected_rows
        )
        if duplicate:
            continue

        selected_rows.append(record)
        if len(selected_rows) >= max(int(top_n), 1):
            break

    out = pd.DataFrame(selected_rows)
    if out.empty:
        return out
    out = out.sort_values("mes", ascending=False).reset_index(drop=True)
    out["raw_rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def standardize_candidate_table(
    periodogram: pd.DataFrame | None,
    raw_summary: Mapping,
    time: np.ndarray,
    *,
    top_n: int = 64,
    duplicate_period_fraction: float = 0.01,
    duplicate_phase_duration_factor: float = 0.75,
) -> pd.DataFrame:
    """Return candidate geometry + raw MES for re-ranking.

    The existing TPS-like search may evolve its periodogram column names.  This
    adapter accepts the current summary names and a conservative set of aliases.
    If the periodogram does not expose enough geometry to evaluate events, the
    raw top-1 candidate is retained as a one-row fallback instead of inventing
    ephemerides.
    """

    rows = []
    if isinstance(periodogram, pd.DataFrame) and not periodogram.empty:
        for source_index, record in periodogram.iterrows():
            candidate = _candidate_from_record(record, time)
            candidate["source_index"] = int(source_index)
            rows.append(candidate)

    candidates = pd.DataFrame(rows)
    required = ["period_days", "epoch_days", "duration_hours", "mes"]
    if not candidates.empty:
        for column in required:
            candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
        candidates = candidates.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
        candidates = candidates[
            (candidates["period_days"] > 0)
            & (candidates["duration_hours"] > 0)
        ].copy()
        if not candidates.empty:
            candidates = select_independent_candidates(
                candidates,
                top_n=top_n,
                duplicate_period_fraction=duplicate_period_fraction,
                duplicate_phase_duration_factor=duplicate_phase_duration_factor,
            )
            if not candidates.empty:
                return candidates

    fallback = _candidate_from_record(raw_summary, time)
    if all(np.isfinite(fallback[name]) for name in required):
        fallback["source_index"] = -1
        fallback["raw_rank"] = 1
        return pd.DataFrame([fallback])

    raise ValueError(
        "TPS-like result does not expose enough candidate geometry for "
        "event-consistency validation (period, epoch, duration, MES)."
    )


def predicted_event_centers(
    time: np.ndarray,
    period_days: float,
    epoch_days: float,
) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    finite = time[np.isfinite(time)]
    if finite.size == 0 or period_days <= 0:
        return np.array([], dtype=float)

    first_k = int(np.ceil((np.min(finite) - epoch_days) / period_days))
    last_k = int(np.floor((np.max(finite) - epoch_days) / period_days))
    if last_k < first_k:
        return np.array([], dtype=float)
    k = np.arange(first_k, last_k + 1, dtype=int)
    return epoch_days + k * period_days


def _robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    med = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - med)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(values, ddof=1))
    return scale if np.isfinite(scale) and scale > 0 else float("nan")



def _weighted_mean_and_error(values: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    valid = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    if not np.any(valid):
        return float("nan"), float("nan")
    values = values[valid]
    errors = errors[valid]
    weights = 1.0 / np.square(errors)
    denom = float(np.sum(weights))
    if denom <= 0:
        return float("nan"), float("nan")
    mean = float(np.sum(weights * values) / denom)
    error = float(np.sqrt(1.0 / denom))
    return mean, error


def _event_depth_chi2(depths: np.ndarray, errors: np.ndarray) -> dict:
    """Weighted event-depth consistency around one common transit depth."""
    depths = np.asarray(depths, dtype=float)
    errors = np.asarray(errors, dtype=float)
    valid = np.isfinite(depths) & np.isfinite(errors) & (errors > 0)
    depths = depths[valid]
    errors = errors[valid]
    n = int(len(depths))
    if n < 2:
        return {
            "event_depth_weighted_mean": np.nan,
            "event_depth_weighted_mean_error": np.nan,
            "event_depth_chi2": np.nan,
            "event_depth_chi2_dof": 0,
            "event_depth_reduced_chi2": np.nan,
            "event_depth_chi2_pvalue": np.nan,
        }

    mean_depth, mean_error = _weighted_mean_and_error(depths, errors)
    statistic = float(np.sum(np.square((depths - mean_depth) / errors)))
    dof = n - 1
    pvalue = float(chi2.sf(statistic, dof))
    return {
        "event_depth_weighted_mean": mean_depth,
        "event_depth_weighted_mean_error": mean_error,
        "event_depth_chi2": statistic,
        "event_depth_chi2_dof": int(dof),
        "event_depth_reduced_chi2": float(statistic / dof),
        "event_depth_chi2_pvalue": pvalue,
    }


def _odd_even_depth_consistency(
    sequence_index: np.ndarray,
    depths: np.ndarray,
    errors: np.ndarray,
    *,
    min_group_events: int = 3,
) -> dict:
    """Compare inverse-variance weighted odd/even event depths."""
    sequence_index = np.asarray(sequence_index, dtype=int)
    depths = np.asarray(depths, dtype=float)
    errors = np.asarray(errors, dtype=float)
    valid = np.isfinite(depths) & np.isfinite(errors) & (errors > 0)
    sequence_index = sequence_index[valid]
    depths = depths[valid]
    errors = errors[valid]

    odd = (sequence_index % 2) == 1
    even = ~odd
    odd_count = int(np.sum(odd))
    even_count = int(np.sum(even))
    if odd_count < int(min_group_events) or even_count < int(min_group_events):
        return {
            "odd_even_tested": False,
            "odd_event_count": odd_count,
            "even_event_count": even_count,
            "odd_depth": np.nan,
            "even_depth": np.nan,
            "odd_even_depth_difference": np.nan,
            "odd_even_depth_difference_fraction": np.nan,
            "odd_even_depth_z": np.nan,
            "odd_even_depth_pvalue": 1.0,
        }

    odd_depth, odd_error = _weighted_mean_and_error(depths[odd], errors[odd])
    even_depth, even_error = _weighted_mean_and_error(depths[even], errors[even])
    denom = float(np.sqrt(odd_error**2 + even_error**2))
    difference = float(odd_depth - even_depth)
    reference_depth = float(0.5 * (abs(odd_depth) + abs(even_depth)))
    difference_fraction = (
        float(abs(difference) / reference_depth)
        if np.isfinite(reference_depth) and reference_depth > 0
        else float("nan")
    )
    if not np.isfinite(denom) or denom <= 0:
        return {
            "odd_even_tested": False,
            "odd_event_count": odd_count,
            "even_event_count": even_count,
            "odd_depth": odd_depth,
            "even_depth": even_depth,
            "odd_even_depth_difference": difference,
            "odd_even_depth_difference_fraction": difference_fraction,
            "odd_even_depth_z": np.nan,
            "odd_even_depth_pvalue": 1.0,
        }

    z = float(difference / denom)
    # Two-sided Normal tail without adding another dependency.
    pvalue = float(math.erfc(abs(z) / np.sqrt(2.0)))
    return {
        "odd_even_tested": True,
        "odd_event_count": odd_count,
        "even_event_count": even_count,
        "odd_depth": odd_depth,
        "even_depth": even_depth,
        "odd_even_depth_difference": difference,
        "odd_even_depth_difference_fraction": difference_fraction,
        "odd_even_depth_z": z,
        "odd_even_depth_pvalue": pvalue,
    }


def _huber_robust_event_snr(
    event_snr: np.ndarray,
    *,
    tuning_constant: float = 1.345,
    max_iterations: int = 25,
) -> dict:
    """Huber-weighted combined event SNR.

    The statistic keeps repeated moderate events while limiting the influence of
    one anomalously large event.  It is a POC robust statistic, not the Kepler
    SOC robust statistic.
    """
    z = np.asarray(event_snr, dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return {
            "robust_event_snr": np.nan,
            "robust_event_location": np.nan,
            "robust_event_weight_min": np.nan,
            "robust_event_effective_count": 0.0,
        }

    location = float(np.median(z))
    scale = float(1.4826 * np.median(np.abs(z - location)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(z, ddof=1)) if z.size > 1 else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0

    weights = np.ones_like(z)
    c = max(float(tuning_constant), 1e-6)
    for _ in range(max_iterations):
        residual = (z - location) / scale
        abs_residual = np.abs(residual)
        weights = np.ones_like(z)
        mask = abs_residual > c
        weights[mask] = c / abs_residual[mask]
        denom = float(np.sum(weights))
        if denom <= 0:
            break
        updated = float(np.sum(weights * z) / denom)
        if abs(updated - location) <= 1e-8 * max(1.0, abs(location)):
            location = updated
            break
        location = updated

    denom = float(np.sqrt(np.sum(np.square(weights))))
    robust_snr = float(np.sum(weights * z) / denom) if denom > 0 else np.nan
    effective_count = (
        float(np.square(np.sum(weights)) / np.sum(np.square(weights)))
        if np.sum(np.square(weights)) > 0
        else 0.0
    )
    return {
        "robust_event_snr": robust_snr,
        "robust_event_location": location,
        "robust_event_weight_min": float(np.min(weights)),
        "robust_event_effective_count": effective_count,
    }


def _consistency_veto_diagnostics(
    metrics: Mapping,
    *,
    config: EventConsistencyConfig,
) -> dict:
    """Return deterministic veto flags and a compact reason string."""

    valid = int(_finite_number(metrics.get("valid_event_count"), 0.0))
    observability = _finite_number(
        metrics.get("event_observability_fraction"), 0.0
    )
    positive_fraction = _finite_number(
        metrics.get("positive_event_fraction"), 0.0
    )
    single_fraction = _finite_number(metrics.get("single_event_fraction"), 1.0)
    leave_one_out = _finite_number(
        metrics.get("leave_one_out_ratio_normalized"), 0.0
    )
    robust_snr = _finite_number(metrics.get("robust_event_snr"), np.nan)
    chi2_pvalue = _finite_number(
        metrics.get("event_depth_chi2_pvalue"), np.nan
    )
    reduced_chi2 = _finite_number(
        metrics.get("event_depth_reduced_chi2"), np.nan
    )
    depth_relative_mad = _finite_number(
        metrics.get("event_depth_relative_mad"), np.nan
    )

    flags = {
        "fails_min_events": valid < int(config.min_valid_events),
        "fails_observability": (
            observability < float(config.min_observability_fraction)
        ),
        "fails_positive_events": (
            positive_fraction < float(config.min_positive_event_fraction)
        ),
        "fails_single_event_dominance": (
            single_fraction > float(config.max_single_event_fraction)
        ),
        "fails_leave_one_out": (
            leave_one_out < float(config.min_leave_one_out_ratio)
        ),
    }

    # Fail closed if the chi-square p-value is not defined for a candidate that
    # otherwise reaches this layer.  If it is defined, require tail
    # significance, materially large reduced chi-square, and robust depth
    # scatter comparable to the measured depth.  This keeps chi-square as a
    # consistency diagnostic without rejecting high-SNR candidates for small
    # fractional depth variations that have underestimated formal errors.
    flags["fails_depth_chi2"] = bool(
        not np.isfinite(chi2_pvalue)
        or (
            chi2_pvalue < float(config.min_event_depth_chi2_pvalue)
            and (
                not np.isfinite(reduced_chi2)
                or reduced_chi2
                >= float(config.max_event_depth_reduced_chi2_for_veto)
            )
            and (
                not np.isfinite(depth_relative_mad)
                or depth_relative_mad
                >= float(config.max_event_depth_relative_mad_for_veto)
            )
        )
    )

    odd_even_tested = bool(metrics.get("odd_even_tested", False))
    odd_even_pvalue = _finite_number(
        metrics.get("odd_even_depth_pvalue"), np.nan
    )
    odd_even_z = abs(_finite_number(metrics.get("odd_even_depth_z"), np.nan))
    odd_even_fraction = _finite_number(
        metrics.get("odd_even_depth_difference_fraction"), np.nan
    )
    if not odd_even_tested:
        flags["fails_odd_even"] = False
    else:
        flags["fails_odd_even"] = bool(
            not np.isfinite(odd_even_pvalue)
            or (
                odd_even_pvalue < float(config.min_odd_even_depth_pvalue)
                and odd_even_z
                >= float(config.min_odd_even_depth_sigma_for_veto)
                and odd_even_fraction
                >= float(config.min_odd_even_depth_fraction_for_veto)
            )
        )

    flags["fails_robust_sign"] = bool(
        not np.isfinite(robust_snr)
        or robust_snr <= float(config.min_robust_event_snr)
    )

    reasons = [name for name in VETO_FLAG_COLUMNS if flags[name]]
    operational_reasons = [
        name
        for name in (
            "fails_min_events",
            "fails_observability",
            "fails_positive_events",
            "fails_single_event_dominance",
            "fails_leave_one_out",
            "fails_robust_sign",
        )
        if flags[name]
    ]
    if not np.isfinite(chi2_pvalue):
        operational_reasons.append("fails_depth_chi2")
    if odd_even_tested and not np.isfinite(odd_even_pvalue):
        operational_reasons.append("fails_odd_even")
    return {
        **flags,
        "diagnostic_consistency_veto_pass": not any(flags.values()),
        "diagnostic_veto_reason": ";".join(reasons) if reasons else "pass",
        "veto_reason": (
            ";".join(operational_reasons) if operational_reasons else "pass"
        ),
        "event_consistent_flag": not any(
            flags[name]
            for name in (
                "fails_min_events",
                "fails_observability",
                "fails_positive_events",
                "fails_single_event_dominance",
                "fails_leave_one_out",
            )
        ),
        "transit_consistency_veto_pass": not operational_reasons,
    }


def candidate_event_metrics(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period_days: float,
    epoch_days: float,
    duration_hours: float,
    config: EventConsistencyConfig = DEFAULT_EVENT_CONSISTENCY_CONFIG,
) -> dict:
    """Measure repeated-event consistency directly in the searched flux."""

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)

    period_days = float(period_days)
    duration_days = float(duration_hours) / 24.0
    centers = predicted_event_centers(time[finite], period_days, float(epoch_days))
    expected = int(len(centers))

    in_half = config.in_transit_half_width_factor * duration_days
    guard_half = min(config.local_guard_factor * duration_days, 0.35 * period_days)
    outer_half = min(config.local_outer_factor * duration_days, 0.45 * period_days)
    if outer_half <= guard_half:
        outer_half = min(0.45 * period_days, guard_half + max(duration_days, 1e-8))

    event_rows = []
    for sequence_index, center in enumerate(centers):
        dt = np.abs(time - center)
        in_mask = finite & (dt <= in_half)
        oot_mask = finite & (dt >= guard_half) & (dt <= outer_half)

        n_in = int(np.sum(in_mask))
        n_oot = int(np.sum(oot_mask))
        if (
            n_in < config.min_in_transit_samples
            or n_oot < config.min_out_of_transit_samples
        ):
            continue

        in_flux = flux[in_mask]
        oot_flux = flux[oot_mask]
        local_scale = _robust_scale(oot_flux)
        if not np.isfinite(local_scale) or local_scale <= 0:
            continue

        depth = float(np.median(oot_flux) - np.median(in_flux))
        median_error_factor = _finite_number(
            config.depth_error_median_factor,
            default=NORMAL_MEDIAN_STDERR_FACTOR,
        )
        if not np.isfinite(median_error_factor) or median_error_factor <= 0:
            median_error_factor = NORMAL_MEDIAN_STDERR_FACTOR
        depth_error = (
            median_error_factor
            * local_scale
            * np.sqrt(1.0 / n_in + 1.0 / n_oot)
        )
        snr = depth / depth_error if depth_error > 0 else float("nan")
        if not np.isfinite(snr):
            continue

        event_rows.append(
            {
                "sequence_index": int(sequence_index),
                "center_days": float(center),
                "depth": depth,
                "depth_error": float(depth_error),
                "local_scale": local_scale,
                "snr": float(snr),
                "n_in": n_in,
                "n_oot": n_oot,
            }
        )

    events = pd.DataFrame(event_rows)
    valid = int(len(events))
    observability = float(valid / expected) if expected > 0 else 0.0

    if valid == 0:
        metrics = {
            "expected_event_count_event_check": expected,
            "valid_event_count": 0,
            "event_observability_fraction": observability,
            "positive_event_fraction": 0.0,
            "single_event_fraction": 1.0,
            "anti_dominance_score": 0.0,
            "combined_event_snr": np.nan,
            "leave_one_out_combined_snr_min": np.nan,
            "leave_one_out_ratio": 0.0,
            "leave_one_out_ratio_normalized": 0.0,
            "median_event_depth": np.nan,
            "event_depth_relative_mad": np.nan,
            "event_depth_weighted_mean": np.nan,
            "event_depth_weighted_mean_error": np.nan,
            "event_depth_chi2": np.nan,
            "event_depth_chi2_dof": 0,
            "event_depth_reduced_chi2": np.nan,
            "event_depth_chi2_pvalue": np.nan,
            "odd_even_tested": False,
            "odd_event_count": 0,
            "even_event_count": 0,
            "odd_depth": np.nan,
            "even_depth": np.nan,
            "odd_even_depth_difference": np.nan,
            "odd_even_depth_difference_fraction": np.nan,
            "odd_even_depth_z": np.nan,
            "odd_even_depth_pvalue": 1.0,
            "robust_event_snr": np.nan,
            "robust_event_location": np.nan,
            "robust_event_weight_min": np.nan,
            "robust_event_effective_count": 0.0,
        }
        metrics.update(_consistency_veto_diagnostics(metrics, config=config))
        metrics["event_table"] = events
        return metrics

    snr = events["snr"].to_numpy(dtype=float)
    depths = events["depth"].to_numpy(dtype=float)
    depth_errors = events["depth_error"].to_numpy(dtype=float)
    sequence_index = events["sequence_index"].to_numpy(dtype=int)
    positive = np.clip(snr, 0.0, None)
    positive_fraction = float(np.mean(snr > 0))

    positive_sum = float(np.sum(positive))
    single_fraction = (
        float(np.max(positive) / positive_sum) if positive_sum > 0 else 1.0
    )

    # Normalize dominance so equal-contribution events map to 1.0 for any N.
    if valid > 1:
        equal_share = 1.0 / valid
        anti_dominance = (1.0 - single_fraction) / (1.0 - equal_share)
        anti_dominance = float(np.clip(anti_dominance, 0.0, 1.0))
    else:
        anti_dominance = 0.0

    combined = float(np.sum(snr) / np.sqrt(valid))
    loo_values = []
    if valid > 1:
        for index in range(valid):
            kept = np.delete(snr, index)
            loo_values.append(float(np.sum(kept) / np.sqrt(len(kept))))
    loo_min = float(np.min(loo_values)) if loo_values else float("nan")

    if np.isfinite(combined) and combined > 0 and np.isfinite(loo_min):
        loo_ratio = float(np.clip(loo_min / combined, 0.0, 1.0))
    else:
        loo_ratio = 0.0

    # Equal-strength events naturally lose sqrt((N-1)/N) when one is removed.
    # Normalize by that reference so a perfectly even candidate maps to 1.0.
    if valid > 1:
        equal_loo_ratio = float(np.sqrt((valid - 1) / valid))
        loo_ratio_normalized = float(
            np.clip(loo_ratio / equal_loo_ratio, 0.0, 1.0)
        )
    else:
        loo_ratio_normalized = 0.0

    depth_median = float(np.median(depths))
    depth_mad = float(1.4826 * np.median(np.abs(depths - depth_median)))
    depth_relative_mad = (
        float(depth_mad / abs(depth_median))
        if np.isfinite(depth_median) and abs(depth_median) > 0
        else float("nan")
    )

    chi2_metrics = _event_depth_chi2(depths, depth_errors)
    odd_even_metrics = _odd_even_depth_consistency(
        sequence_index,
        depths,
        depth_errors,
        min_group_events=config.min_odd_even_group_events,
    )
    robust_metrics = _huber_robust_event_snr(
        snr, tuning_constant=config.huber_tuning_constant
    )

    metrics = {
        "expected_event_count_event_check": expected,
        "valid_event_count": valid,
        "event_observability_fraction": observability,
        "positive_event_fraction": positive_fraction,
        "single_event_fraction": single_fraction,
        "anti_dominance_score": anti_dominance,
        "combined_event_snr": combined,
        "leave_one_out_combined_snr_min": loo_min,
        "leave_one_out_ratio": loo_ratio,
        "leave_one_out_ratio_normalized": loo_ratio_normalized,
        "median_event_depth": depth_median,
        "event_depth_relative_mad": depth_relative_mad,
        **chi2_metrics,
        **odd_even_metrics,
        **robust_metrics,
    }
    metrics.update(_consistency_veto_diagnostics(metrics, config=config))
    metrics["event_table"] = events
    return metrics


def consistency_weight(metrics: Mapping) -> float:
    """Geometric-mean consistency weight in [0, 1]."""

    components = np.array(
        [
            _finite_number(metrics.get("event_observability_fraction"), 0.0),
            _finite_number(metrics.get("positive_event_fraction"), 0.0),
            _finite_number(metrics.get("anti_dominance_score"), 0.0),
            _finite_number(metrics.get("leave_one_out_ratio_normalized"), 0.0),
        ],
        dtype=float,
    )
    components = np.clip(components, 0.0, 1.0)
    if np.any(components <= 0):
        return 0.0
    return float(np.prod(components) ** (1.0 / len(components)))


def harden_tps_like_result(
    result: Mapping,
    time: np.ndarray,
    flux: np.ndarray,
    *,
    config: EventConsistencyConfig = DEFAULT_EVENT_CONSISTENCY_CONFIG,
) -> dict:
    """Re-rank the existing TPS-like candidate list by event consistency."""

    if "summary" not in result:
        raise KeyError("TPS-like result must contain a 'summary' mapping.")

    raw_summary = dict(result["summary"])
    candidates = standardize_candidate_table(
        result.get("periodogram"),
        raw_summary,
        np.asarray(time, dtype=float),
        top_n=config.top_n_candidates,
        duplicate_period_fraction=config.duplicate_period_fraction,
        duplicate_phase_duration_factor=config.duplicate_phase_duration_factor,
    )

    ranking_rows = []
    event_tables = {}
    for candidate_index, candidate in candidates.iterrows():
        metrics = candidate_event_metrics(
            time,
            flux,
            period_days=float(candidate["period_days"]),
            epoch_days=float(candidate["epoch_days"]),
            duration_hours=float(candidate["duration_hours"]),
            config=config,
        )
        event_table = metrics.pop("event_table")
        weight = consistency_weight(metrics)
        raw_mes = float(candidate["mes"])
        soft_score = max(raw_mes, 0.0) * weight
        veto_pass = bool(metrics["transit_consistency_veto_pass"])
        robust_veto_score = max(raw_mes, 0.0) if veto_pass else 0.0

        row = candidate.to_dict()
        row.update(metrics)
        row["_event_table_key"] = int(candidate_index)
        row["consistency_weight"] = weight
        row["event_consistency_score"] = float(soft_score)
        row["robust_veto_score"] = float(robust_veto_score)
        ranking_rows.append(row)
        event_tables[int(candidate_index)] = event_table

    ranking = pd.DataFrame(ranking_rows)
    if ranking.empty:
        raise ValueError("No TPS-like candidates survived event-consistency evaluation.")

    any_candidate_survives = bool(ranking["transit_consistency_veto_pass"].any())
    raw_top_mask = pd.to_numeric(ranking["raw_rank"], errors="coerce") == 1
    raw_top_survives = bool(
        raw_top_mask.any()
        and ranking.loc[raw_top_mask, "transit_consistency_veto_pass"]
        .astype(bool)
        .iloc[0]
    )

    if raw_top_survives:
        # Preserve the raw MES winner whenever it has repeated-event support.
        # The chi-square and odd/even fields remain visible diagnostics, but do
        # not promote a lower-MES alias over a supported MES maximum.
        ranking = ranking.assign(_selection_primary=raw_top_mask.astype(int))
        sort_columns = [
            "_selection_primary",
            "transit_consistency_veto_pass",
            "mes",
            "raw_rank",
        ]
        sort_ascending = [False, False, False, True]
        selection_status = "raw_top1_mes_preserved"
    elif any_candidate_survives:
        # Once the raw MES winner has failed explicit event-support checks, MES
        # alone has already proven unreliable for this case.  Use the v2
        # support-weighted penalty to choose the replacement rather than the new
        # chi-square/odd-even diagnostics or a tuned threshold.
        ranking = ranking.assign(_selection_primary=0)
        sort_columns = [
            "transit_consistency_veto_pass",
            "event_consistency_score",
            "mes",
            "raw_rank",
        ]
        sort_ascending = [False, False, False, True]
        selection_status = "raw_failed_event_penalty_rank"
    else:
        # Fall back explicitly and deterministically to the highest-MES
        # candidate with robust_veto_score = 0.0.  Empirical FAP calibration then
        # treats this as a non-detection on the hardened scale.
        ranking = ranking.assign(_selection_primary=0)
        sort_columns = ["mes", "raw_rank"]
        sort_ascending = [False, True]
        selection_status = "no_veto_survivor_raw_mes_fallback"

    ranking = ranking.sort_values(
        sort_columns,
        ascending=sort_ascending,
    ).reset_index(drop=True)
    ranking["hardened_rank"] = np.arange(1, len(ranking) + 1, dtype=int)

    winner = ranking.iloc[0].to_dict()
    winner_event_key = int(winner["_event_table_key"])
    ranking = ranking.drop(columns=["_event_table_key", "_selection_primary"])
    winner.pop("_event_table_key", None)
    raw = _candidate_from_record(raw_summary, np.asarray(time, dtype=float))

    raw_period = _finite_number(raw.get("period_days"))
    selected_period = _finite_number(winner.get("period_days"))
    period_changed = bool(
        np.isfinite(raw_period)
        and np.isfinite(selected_period)
        and not np.isclose(raw_period, selected_period, rtol=1e-10, atol=1e-12)
    )

    summary = {
        "period_days": float(winner["period_days"]),
        "epoch_days": float(winner["epoch_days"]),
        "duration_hours": float(winner["duration_hours"]),
        "selected_raw_mes": float(winner["mes"]),
        "selected_max_ses": _finite_number(winner.get("max_ses")),
        "event_consistency_score": float(winner["event_consistency_score"]),
        "robust_veto_score": float(winner["robust_veto_score"]),
        "consistency_weight": float(winner["consistency_weight"]),
        "valid_event_count": int(winner["valid_event_count"]),
        "expected_event_count_event_check": int(
            winner["expected_event_count_event_check"]
        ),
        "event_observability_fraction": float(
            winner["event_observability_fraction"]
        ),
        "positive_event_fraction": float(winner["positive_event_fraction"]),
        "single_event_fraction": float(winner["single_event_fraction"]),
        "anti_dominance_score": float(winner["anti_dominance_score"]),
        "combined_event_snr": _finite_number(winner.get("combined_event_snr")),
        "leave_one_out_combined_snr_min": _finite_number(
            winner.get("leave_one_out_combined_snr_min")
        ),
        "leave_one_out_ratio": float(winner["leave_one_out_ratio"]),
        "leave_one_out_ratio_normalized": float(
            winner["leave_one_out_ratio_normalized"]
        ),
        "median_event_depth": _finite_number(winner.get("median_event_depth")),
        "event_depth_relative_mad": _finite_number(
            winner.get("event_depth_relative_mad")
        ),
        "event_depth_weighted_mean": _finite_number(
            winner.get("event_depth_weighted_mean")
        ),
        "event_depth_weighted_mean_error": _finite_number(
            winner.get("event_depth_weighted_mean_error")
        ),
        "event_depth_chi2": _finite_number(winner.get("event_depth_chi2")),
        "event_depth_chi2_dof": int(winner.get("event_depth_chi2_dof", 0)),
        "event_depth_reduced_chi2": _finite_number(
            winner.get("event_depth_reduced_chi2")
        ),
        "event_depth_chi2_pvalue": _finite_number(
            winner.get("event_depth_chi2_pvalue")
        ),
        "odd_even_tested": bool(winner.get("odd_even_tested", False)),
        "odd_event_count": int(winner.get("odd_event_count", 0)),
        "even_event_count": int(winner.get("even_event_count", 0)),
        "odd_depth": _finite_number(winner.get("odd_depth")),
        "even_depth": _finite_number(winner.get("even_depth")),
        "odd_even_depth_difference": _finite_number(
            winner.get("odd_even_depth_difference")
        ),
        "odd_even_depth_difference_fraction": _finite_number(
            winner.get("odd_even_depth_difference_fraction")
        ),
        "odd_even_depth_z": _finite_number(winner.get("odd_even_depth_z")),
        "odd_even_depth_pvalue": _finite_number(
            winner.get("odd_even_depth_pvalue"), 1.0
        ),
        "robust_event_snr": _finite_number(winner.get("robust_event_snr")),
        "robust_event_location": _finite_number(
            winner.get("robust_event_location")
        ),
        "robust_event_weight_min": _finite_number(
            winner.get("robust_event_weight_min")
        ),
        "robust_event_effective_count": _finite_number(
            winner.get("robust_event_effective_count")
        ),
        "event_consistent_flag": bool(winner["event_consistent_flag"]),
        "transit_consistency_veto_pass": bool(
            winner["transit_consistency_veto_pass"]
        ),
        "diagnostic_consistency_veto_pass": bool(
            winner["diagnostic_consistency_veto_pass"]
        ),
        **{name: bool(winner.get(name, False)) for name in VETO_FLAG_COLUMNS},
        "veto_reason": str(winner.get("veto_reason", "")),
        "diagnostic_veto_reason": str(
            winner.get("diagnostic_veto_reason", "")
        ),
        "any_candidate_survives_veto": any_candidate_survives,
        "selection_status": selection_status,
        "raw_rank_of_selected_candidate": int(winner["raw_rank"]),
        "ranking_changed": period_changed or int(winner["raw_rank"]) != 1,
        "raw_top1_period_days": raw_period,
        "raw_top1_epoch_days": _finite_number(raw.get("epoch_days")),
        "raw_top1_duration_hours": _finite_number(raw.get("duration_hours")),
        "raw_top1_mes": _finite_number(raw.get("mes")),
        "raw_top1_max_ses": _finite_number(raw.get("max_ses")),
    }

    return {
        "summary": summary,
        "raw_summary": raw_summary,
        "ranking_table": ranking,
        "selected_event_table": event_tables.get(winner_event_key, pd.DataFrame()),
        "config": config_dict(config),
    }
