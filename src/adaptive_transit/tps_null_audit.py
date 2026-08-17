"""Utilities for the TPS-like true zero-injection control audit.

The control reuses the existing TPS-like preprocessing, adaptive wavelet noise
model, and period/duration search, but explicitly skips BATMAN and searches the
original preprocessed stellar flux.  This gives a native-background winner for
each star without relying on an artificial tiny-depth injection.

This is a diagnostic control, not a false-alarm calibration.  It must not be
used as a replacement for randomized/empirical null trials at a target FAP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PeriodCluster:
    period_days: float
    count: int
    fraction: float
    member_index: tuple[int, ...]


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def fractional_period_error(candidate: float, reference: float) -> float:
    candidate = float(candidate)
    reference = float(reference)
    if not np.isfinite(candidate) or not np.isfinite(reference) or reference <= 0:
        return np.nan
    return float(abs(candidate - reference) / reference)


def dominant_period_cluster(periods, tolerance_fraction: float = 0.02) -> PeriodCluster:
    """Return the densest simple fractional-period cluster.

    This is intentionally descriptive.  It is used only to summarize whether
    the injected runs on one star repeatedly select approximately the same
    winning period despite different injected truths.
    """
    values = pd.to_numeric(pd.Series(periods), errors="coerce")
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.empty:
        return PeriodCluster(np.nan, 0, 0.0, tuple())

    arr = finite.to_numpy(dtype=float)
    original_index = finite.index.to_numpy(dtype=int)
    best_members = np.array([0], dtype=int)
    best_spread = np.inf

    for i, seed in enumerate(arr):
        rel = np.abs(arr - seed) / seed
        members = np.flatnonzero(rel <= float(tolerance_fraction))
        representative = float(np.median(arr[members]))
        spread = float(np.median(np.abs(arr[members] - representative)))
        if len(members) > len(best_members) or (
            len(members) == len(best_members) and spread < best_spread
        ):
            best_members = members
            best_spread = spread

    representative = float(np.median(arr[best_members]))
    return PeriodCluster(
        period_days=representative,
        count=int(len(best_members)),
        fraction=float(len(best_members) / len(arr)),
        member_index=tuple(int(x) for x in original_index[best_members]),
    )


def build_zero_injection_table(
    raw: pd.DataFrame,
    max_realized_depth: float = 1e-12,
) -> pd.DataFrame:
    """Validate and normalize the one-row-per-star numerical-null run."""
    required = {
        "target_id",
        "quarter",
        "sample_stratum",
        "success",
        "zero_injection_control",
        "requested_depth",
        "recovered_period_days",
        "recovered_epoch_days",
        "recovered_duration_hours",
        "mes",
        "max_ses",
        "observed_event_count",
        "expected_event_count",
        "observability_fraction",
        "runtime_seconds",
        "realized_max_depth_on_observed_cadences",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Zero-injection raw results are missing columns: {missing}")

    frame = raw.copy()
    frame = frame.loc[_as_bool(frame["success"])].copy()
    if frame.empty:
        raise ValueError("Zero-injection runner produced no successful rows.")

    control_flag = _as_bool(frame["zero_injection_control"])
    if not control_flag.all():
        raise ValueError(
            "Zero-injection audit received successful rows that were not produced "
            "with --zero-injection."
        )

    duplicates = frame.duplicated(["target_id", "quarter"], keep=False)
    if duplicates.any():
        bad = frame.loc[duplicates, ["target_id", "quarter"]].drop_duplicates()
        raise ValueError(
            "Expected exactly one numerical-null case per target/quarter; duplicates: "
            + bad.to_dict(orient="records").__repr__()
        )

    realized = pd.to_numeric(
        frame["realized_max_depth_on_observed_cadences"], errors="coerce"
    ).abs()
    if realized.isna().any():
        raise ValueError(
            "Cannot verify the zero-injection control because realized depth is missing."
        )
    observed_max = float(realized.max())
    if observed_max > float(max_realized_depth):
        raise ValueError(
            "Zero-injection control is not sufficiently null: "
            f"max realized depth={observed_max:.3e} exceeds {max_realized_depth:.3e}."
        )

    columns = {
        "recovered_period_days": "null_recovered_period_days",
        "recovered_epoch_days": "null_recovered_epoch_days",
        "recovered_duration_hours": "null_recovered_duration_hours",
        "mes": "null_mes",
        "max_ses": "null_max_ses",
        "observed_event_count": "null_observed_event_count",
        "expected_event_count": "null_expected_event_count",
        "observability_fraction": "null_observability_fraction",
        "runtime_seconds": "null_runtime_seconds",
        "requested_depth": "zero_injection_requested_depth",
        "realized_max_depth_on_observed_cadences": "zero_injection_realized_depth",
    }
    keep = [
        "target_id",
        "quarter",
        "sample_stratum",
        *columns.keys(),
    ]
    for optional in (
        "wavelet",
        "segment_count",
        "n_period_trials",
        "n_duration_trials",
    ):
        if optional in frame.columns:
            keep.append(optional)

    out = frame[keep].rename(columns=columns).reset_index(drop=True)
    out.insert(3, "baseline_kind", "true_zero_injection_original_flux")
    out["null_max_ses_to_mes_ratio"] = np.where(
        pd.to_numeric(out["null_mes"], errors="coerce").abs() > 0,
        pd.to_numeric(out["null_max_ses"], errors="coerce").abs()
        / pd.to_numeric(out["null_mes"], errors="coerce").abs(),
        np.nan,
    )
    return out


def compare_zero_to_injected(
    zero: pd.DataFrame,
    injected: pd.DataFrame,
    tolerance_fraction: float = 0.02,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare true zero-injection winners with all injected-run winners."""
    required_injected = {
        "target_id",
        "quarter",
        "injected_period_days",
        "recovered_period_days",
        "mes",
        "max_ses",
        "success",
        "harmonic_period_recovered",
    }
    missing = sorted(required_injected - set(injected.columns))
    if missing:
        raise ValueError(f"Injected TPS-like results are missing columns: {missing}")

    inj = injected.loc[_as_bool(injected["success"])].copy()
    merged = inj.merge(
        zero,
        on=["target_id", "quarter"],
        how="inner",
        suffixes=("", "_null"),
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError("No target/quarter overlap between null and injected TPS-like runs.")

    merged["winner_vs_null_fractional_error"] = [
        fractional_period_error(c, r)
        for c, r in zip(
            merged["recovered_period_days"], merged["null_recovered_period_days"]
        )
    ]
    merged["winner_matches_null_period"] = (
        merged["winner_vs_null_fractional_error"] <= float(tolerance_fraction)
    )
    merged["injected_minus_null_mes"] = (
        pd.to_numeric(merged["mes"], errors="coerce")
        - pd.to_numeric(merged["null_mes"], errors="coerce")
    )
    merged["injected_minus_null_max_ses"] = (
        pd.to_numeric(merged["max_ses"], errors="coerce")
        - pd.to_numeric(merged["null_max_ses"], errors="coerce")
    )

    rows = []
    for (target_id, quarter), group in merged.groupby(["target_id", "quarter"]):
        group = group.copy()
        cluster = dominant_period_cluster(
            group["recovered_period_days"], tolerance_fraction=tolerance_fraction
        )
        member_mask = group.index.isin(cluster.member_index)
        member_truths = int(
            pd.to_numeric(
                group.loc[member_mask, "injected_period_days"], errors="coerce"
            ).nunique()
        )
        persistent = bool(cluster.fraction >= 0.5 and member_truths >= 2)

        null_period = float(group["null_recovered_period_days"].iloc[0])
        null_vs_persistent = fractional_period_error(cluster.period_days, null_period)
        harmonic = _as_bool(group["harmonic_period_recovered"])

        rows.append(
            {
                "target_id": target_id,
                "quarter": int(quarter),
                "sample_stratum": str(group["sample_stratum"].iloc[0]),
                "n_injected_cases": int(len(group)),
                "n_injected_period_truths": int(
                    pd.to_numeric(group["injected_period_days"], errors="coerce").nunique()
                ),
                "persistent_injected_period_days": cluster.period_days,
                "persistent_cluster_fraction": cluster.fraction,
                "persistent_cluster_truth_count": member_truths,
                "persistent_injected_period_flag": persistent,
                "null_recovered_period_days": null_period,
                "null_vs_persistent_fractional_error": null_vs_persistent,
                "null_matches_persistent_period": bool(
                    persistent
                    and np.isfinite(null_vs_persistent)
                    and null_vs_persistent <= float(tolerance_fraction)
                ),
                "fraction_injected_winners_matching_null_period": float(
                    group["winner_matches_null_period"].mean()
                ),
                "harmonic_recovery": float(harmonic.mean()),
                "null_mes": float(group["null_mes"].iloc[0]),
                "median_injected_mes": float(
                    pd.to_numeric(group["mes"], errors="coerce").median()
                ),
                "median_injected_minus_null_mes": float(
                    pd.to_numeric(
                        group["injected_minus_null_mes"], errors="coerce"
                    ).median()
                ),
                "null_max_ses": float(group["null_max_ses"].iloc[0]),
                "null_max_ses_to_mes_ratio": float(
                    group["null_max_ses_to_mes_ratio"].iloc[0]
                ),
                "null_observability_fraction": float(
                    group["null_observability_fraction"].iloc[0]
                ),
            }
        )

    star_summary = pd.DataFrame(rows).sort_values(
        ["persistent_injected_period_flag", "null_matches_persistent_period", "target_id"],
        ascending=[False, False, True],
    )
    return merged.reset_index(drop=True), star_summary.reset_index(drop=True)
