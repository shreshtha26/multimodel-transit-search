#!/usr/bin/env python3
"""Post-hoc reliability audit for the TPS-like BATMAN comparator.

This script does *not* change the detector, rerun injections, or perform false-alarm
calibration.  It diagnoses whether the current top-ranked TPS-like candidate is
being dominated by persistent star-specific periods and whether large MES-like
values are supported by repeated events or by one very strong SES-like excursion.

The audit is deliberately descriptive.  It should be used to decide what to fix
before common null/FAP calibration, not as a final detection-efficiency result.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_HARMONIC_RATIOS = {
    "P/3": 1.0 / 3.0,
    "P/2": 1.0 / 2.0,
    "2P/3": 2.0 / 3.0,
    "P": 1.0,
    "3P/2": 1.5,
    "2P": 2.0,
    "3P": 3.0,
}

REQUIRED_COLUMNS = {
    "target_id",
    "sample_stratum",
    "case_index",
    "injected_period_days",
    "recovered_period_days",
    "success",
    "exact_period_recovered",
    "harmonic_period_recovered",
    "mes",
    "max_ses",
    "observed_event_count",
    "expected_event_count",
    "observability_fraction",
}


def _to_bool(series: pd.Series) -> pd.Series:
    """Convert common CSV truth representations to bool without treating NaN as True."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def classify_period_ratio(
    ratio: float,
    tolerance: float = 0.03,
    harmonic_ratios: dict[str, float] | None = None,
) -> str:
    """Classify recovered/injected period ratio against simple harmonics.

    `tolerance` is fractional relative error around each named ratio.
    """
    if not np.isfinite(ratio) or ratio <= 0:
        return "invalid"

    ratios = harmonic_ratios or DEFAULT_HARMONIC_RATIOS
    best_name = "unrelated"
    best_error = np.inf
    for name, target in ratios.items():
        error = abs(float(ratio) - target) / target
        if error < best_error:
            best_error = error
            best_name = name
    return best_name if best_error <= tolerance else "unrelated"


def cluster_periods_relative(
    periods: Iterable[float],
    tolerance: float = 0.01,
) -> np.ndarray:
    """Assign simple deterministic relative-tolerance clusters to periods.

    This is intentionally lightweight: values are sorted and joined to the current
    cluster when they lie within `tolerance` of that cluster's running median.
    """
    arr = np.asarray(list(periods), dtype=float)
    labels = np.full(arr.shape, -1, dtype=int)
    finite_idx = np.flatnonzero(np.isfinite(arr) & (arr > 0))
    if finite_idx.size == 0:
        return labels

    sorted_idx = finite_idx[np.argsort(arr[finite_idx])]
    cluster_members: list[int] = []
    cluster_id = 0

    for idx in sorted_idx:
        value = arr[idx]
        if not cluster_members:
            cluster_members = [idx]
            labels[idx] = cluster_id
            continue

        center = float(np.median(arr[cluster_members]))
        rel_error = abs(value - center) / center
        if rel_error <= tolerance:
            cluster_members.append(idx)
            labels[idx] = cluster_id
        else:
            cluster_id += 1
            cluster_members = [idx]
            labels[idx] = cluster_id

    return labels


def add_case_diagnostics(
    results: pd.DataFrame,
    harmonic_tolerance: float = 0.03,
    period_cluster_tolerance: float = 0.01,
) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(results.columns)
    if missing:
        raise ValueError(f"Missing required TPS-like result columns: {sorted(missing)}")

    df = results.copy()
    df["success"] = _to_bool(df["success"])
    df["exact_period_recovered"] = _to_bool(df["exact_period_recovered"])
    df["harmonic_period_recovered"] = _to_bool(df["harmonic_period_recovered"])

    injected = pd.to_numeric(df["injected_period_days"], errors="coerce")
    recovered = pd.to_numeric(df["recovered_period_days"], errors="coerce")
    df["period_ratio"] = recovered / injected
    df["period_pattern"] = df["period_ratio"].apply(
        lambda x: classify_period_ratio(x, tolerance=harmonic_tolerance)
    )

    mes_abs = pd.to_numeric(df["mes"], errors="coerce").abs()
    ses_abs = pd.to_numeric(df["max_ses"], errors="coerce").abs()
    df["max_ses_to_mes_ratio"] = np.where(mes_abs > 0, ses_abs / mes_abs, np.nan)

    observed = pd.to_numeric(df["observed_event_count"], errors="coerce")
    expected = pd.to_numeric(df["expected_event_count"], errors="coerce")
    df["event_support_fraction"] = np.where(expected > 0, observed / expected, np.nan)

    # A heuristic only: ratios near one indicate the strongest individual event is
    # comparable in scale to the accumulated statistic.  Do not use as a detection
    # veto until validated against nulls and true injections.
    df["single_event_dominance_flag"] = df["max_ses_to_mes_ratio"] >= 0.80

    df["recovered_period_cluster"] = -1
    for target_id, idx in df.groupby("target_id", sort=False).groups.items():
        labels = cluster_periods_relative(
            df.loc[idx, "recovered_period_days"], tolerance=period_cluster_tolerance
        )
        df.loc[idx, "recovered_period_cluster"] = labels
    df["recovered_period_cluster"] = df["recovered_period_cluster"].astype(int)

    return df


def build_star_persistence_table(
    case_df: pd.DataFrame,
    min_cluster_fraction: float = 0.50,
    min_distinct_injected_periods: int = 2,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (target_id, stratum), group in case_df.groupby(
        ["target_id", "sample_stratum"], dropna=False, sort=True
    ):
        finite = group[
            np.isfinite(pd.to_numeric(group["recovered_period_days"], errors="coerce"))
            & (pd.to_numeric(group["recovered_period_days"], errors="coerce") > 0)
            & (group["recovered_period_cluster"] >= 0)
        ].copy()

        dominant_period = np.nan
        dominant_fraction = np.nan
        dominant_count = 0
        dominant_injected_periods = 0

        if not finite.empty:
            cluster_counts = finite["recovered_period_cluster"].value_counts()
            dominant_cluster = int(cluster_counts.index[0])
            dominant_rows = finite[finite["recovered_period_cluster"] == dominant_cluster]
            dominant_count = int(len(dominant_rows))
            dominant_fraction = float(dominant_count / len(finite))
            dominant_period = float(
                pd.to_numeric(dominant_rows["recovered_period_days"], errors="coerce").median()
            )
            dominant_injected_periods = int(
                pd.to_numeric(
                    dominant_rows["injected_period_days"], errors="coerce"
                ).dropna().nunique()
            )

        persistence_flag = bool(
            np.isfinite(dominant_fraction)
            and dominant_fraction >= min_cluster_fraction
            and dominant_injected_periods >= min_distinct_injected_periods
        )

        rows.append(
            {
                "target_id": target_id,
                "sample_stratum": stratum,
                "n_cases": int(len(group)),
                "exact_period_recovery": float(group["exact_period_recovered"].mean()),
                "harmonic_period_recovery": float(group["harmonic_period_recovered"].mean()),
                "median_mes": float(pd.to_numeric(group["mes"], errors="coerce").median()),
                "median_max_ses": float(
                    pd.to_numeric(group["max_ses"], errors="coerce").median()
                ),
                "median_max_ses_to_mes_ratio": float(
                    pd.to_numeric(group["max_ses_to_mes_ratio"], errors="coerce").median()
                ),
                "single_event_dominance_rate": float(
                    group["single_event_dominance_flag"].mean()
                ),
                "median_observability": float(
                    pd.to_numeric(group["observability_fraction"], errors="coerce").median()
                ),
                "dominant_recovered_period_days": dominant_period,
                "dominant_period_case_count": dominant_count,
                "dominant_period_fraction": dominant_fraction,
                "distinct_injected_periods_in_dominant_cluster": dominant_injected_periods,
                "persistent_star_period_flag": persistence_flag,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["persistent_star_period_flag", "dominant_period_fraction", "median_mes"],
        ascending=[False, False, False],
        kind="stable",
    )


def build_stratum_summary(case_df: pd.DataFrame) -> pd.DataFrame:
    return (
        case_df.groupby("sample_stratum", dropna=False)
        .agg(
            n_cases=("target_id", "size"),
            n_stars=("target_id", "nunique"),
            exact_period_recovery=("exact_period_recovered", "mean"),
            harmonic_period_recovery=("harmonic_period_recovered", "mean"),
            median_mes=("mes", "median"),
            median_max_ses=("max_ses", "median"),
            median_max_ses_to_mes_ratio=("max_ses_to_mes_ratio", "median"),
            single_event_dominance_rate=("single_event_dominance_flag", "mean"),
            median_observability=("observability_fraction", "median"),
        )
        .reset_index()
        .sort_values("harmonic_period_recovery", ascending=False, kind="stable")
    )


def attach_no_injection_baseline(
    star_table: pd.DataFrame,
    baseline_csv: Path | None,
    tolerance: float = 0.01,
) -> pd.DataFrame:
    """Optionally compare dominant injected-run periods to a future zero-injection run.

    Expected baseline columns: target_id, recovered_period_days.  Extra columns are
    preserved with a `baseline_` prefix where useful.
    """
    if baseline_csv is None:
        return star_table
    baseline = pd.read_csv(baseline_csv)
    required = {"target_id", "recovered_period_days"}
    missing = required - set(baseline.columns)
    if missing:
        raise ValueError(
            f"Baseline CSV is missing required columns: {sorted(missing)}"
        )

    keep = ["target_id", "recovered_period_days"]
    for optional in ["mes", "max_ses", "observability_fraction"]:
        if optional in baseline.columns:
            keep.append(optional)
    baseline = baseline[keep].copy()
    rename = {col: f"baseline_{col}" for col in keep if col != "target_id"}
    baseline = baseline.rename(columns=rename)
    baseline = baseline.drop_duplicates("target_id", keep="first")

    out = star_table.merge(baseline, on="target_id", how="left", validate="one_to_one")
    dominant = pd.to_numeric(out["dominant_recovered_period_days"], errors="coerce")
    base_period = pd.to_numeric(out["baseline_recovered_period_days"], errors="coerce")
    denom = np.where(np.isfinite(base_period) & (base_period > 0), base_period, np.nan)
    out["dominant_vs_no_injection_fractional_difference"] = abs(dominant - base_period) / denom
    out["matches_no_injection_period"] = (
        out["dominant_vs_no_injection_fractional_difference"] <= tolerance
    ).fillna(False)
    return out


def _write_summary_text(
    path: Path,
    case_df: pd.DataFrame,
    star_df: pd.DataFrame,
    stratum_df: pd.DataFrame,
    high_mes_df: pd.DataFrame,
    baseline_csv: Path | None,
) -> None:
    pattern_counts = case_df["period_pattern"].value_counts(dropna=False)
    persistent = star_df[star_df["persistent_star_period_flag"]]

    lines = [
        "TPS-LIKE RELIABILITY AUDIT",
        "=" * 72,
        "",
        "Scope:",
        "  Descriptive diagnostics only. No detector changes, no null calibration,",
        "  and no final detection-efficiency claims are made by this audit.",
        "",
        f"Cases: {len(case_df)}",
        f"Stars: {case_df['target_id'].nunique()}",
        f"Exact period recovery: {case_df['exact_period_recovered'].mean():.3f}",
        f"Harmonic-aware recovery: {case_df['harmonic_period_recovered'].mean():.3f}",
        f"Persistent star-period flags: {len(persistent)}/{case_df['target_id'].nunique()}",
        f"Single-event dominance heuristic rate: {case_df['single_event_dominance_flag'].mean():.3f}",
        "",
        "Period-pattern counts:",
    ]
    for name, count in pattern_counts.items():
        lines.append(f"  {name}: {int(count)}")

    lines.extend(["", "Persistent star-specific candidate periods:"])
    if persistent.empty:
        lines.append("  none under current descriptive thresholds")
    else:
        for row in persistent.itertuples(index=False):
            baseline_note = ""
            if hasattr(row, "matches_no_injection_period"):
                baseline_note = f", matches_no_injection={bool(row.matches_no_injection_period)}"
            lines.append(
                "  KIC {target}: period≈{period:.6f} d, cluster_fraction={frac:.3f}, "
                "injected_periods={ninj}, recovery={rec:.3f}{baseline}".format(
                    target=row.target_id,
                    period=row.dominant_recovered_period_days,
                    frac=row.dominant_period_fraction,
                    ninj=row.distinct_injected_periods_in_dominant_cluster,
                    rec=row.harmonic_period_recovery,
                    baseline=baseline_note,
                )
            )

    lines.extend(["", "Stratum summary:"])
    if stratum_df.empty:
        lines.append("  unavailable")
    else:
        for row in stratum_df.itertuples(index=False):
            lines.append(
                "  {stratum}: n={n}, recovery={rec:.3f}, median_MES={mes:.3f}, "
                "single_event_dominance={dom:.3f}".format(
                    stratum=row.sample_stratum,
                    n=row.n_cases,
                    rec=row.harmonic_period_recovery,
                    mes=row.median_mes,
                    dom=row.single_event_dominance_rate,
                )
            )

    lines.extend(
        [
            "",
            f"High-MES failed cases retained for inspection: {len(high_mes_df)}",
            "",
            "Interpretation guardrail:",
            "  max_ses_to_mes_ratio and persistent_star_period_flag are diagnostics,",
            "  not validated vetoes. Validate them against zero-injection/null trials",
            "  before using them to alter candidate ranking.",
        ]
    )
    if baseline_csv is None:
        lines.extend(
            [
                "",
                "No zero-injection baseline CSV was supplied. The star-period",
                "persistence result therefore shows insensitivity to injection truth,",
                "not yet direct identity with the uninjected star's strongest period.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(
    input_dir: Path,
    output_dir: Path | None = None,
    harmonic_tolerance: float = 0.03,
    period_cluster_tolerance: float = 0.01,
    min_cluster_fraction: float = 0.50,
    min_distinct_injected_periods: int = 2,
    high_mes_quantile: float = 0.75,
    baseline_csv: Path | None = None,
    baseline_period_tolerance: float = 0.01,
) -> dict[str, Path]:
    input_dir = Path(input_dir)
    results_path = input_dir / "tps_like_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"TPS-like results not found: {results_path}")

    results = pd.read_csv(results_path)
    case_df = add_case_diagnostics(
        results,
        harmonic_tolerance=harmonic_tolerance,
        period_cluster_tolerance=period_cluster_tolerance,
    )
    star_df = build_star_persistence_table(
        case_df,
        min_cluster_fraction=min_cluster_fraction,
        min_distinct_injected_periods=min_distinct_injected_periods,
    )
    star_df = attach_no_injection_baseline(
        star_df, baseline_csv, tolerance=baseline_period_tolerance
    )
    stratum_df = build_stratum_summary(case_df)

    failed = case_df[~case_df["harmonic_period_recovered"]].copy()
    finite_failed_mes = pd.to_numeric(failed["mes"], errors="coerce").dropna()
    if finite_failed_mes.empty:
        high_mes_df = failed.copy()
    else:
        threshold = float(finite_failed_mes.quantile(high_mes_quantile))
        high_mes_df = failed[
            pd.to_numeric(failed["mes"], errors="coerce") >= threshold
        ].copy()
        high_mes_df["high_mes_failure_threshold"] = threshold
    high_mes_df = high_mes_df.sort_values("mes", ascending=False, kind="stable")

    out_dir = Path(output_dir) if output_dir else input_dir / "reliability_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "case_diagnostics": out_dir / "tps_like_case_diagnostics.csv",
        "star_persistence": out_dir / "tps_like_star_persistence.csv",
        "stratum_summary": out_dir / "tps_like_stratum_summary.csv",
        "high_mes_failures": out_dir / "tps_like_high_mes_failures.csv",
        "summary": out_dir / "tps_like_reliability_summary.txt",
    }
    case_df.to_csv(paths["case_diagnostics"], index=False)
    star_df.to_csv(paths["star_persistence"], index=False)
    stratum_df.to_csv(paths["stratum_summary"], index=False)
    high_mes_df.to_csv(paths["high_mes_failures"], index=False)
    _write_summary_text(
        paths["summary"], case_df, star_df, stratum_df, high_mes_df, baseline_csv
    )

    print("\nTPS-like reliability audit complete.")
    print(f"Cases: {len(case_df)}")
    print(f"Stars: {case_df['target_id'].nunique()}")
    print(f"Exact period recovery: {case_df['exact_period_recovered'].mean():.3f}")
    print(
        "Harmonic-aware period recovery: "
        f"{case_df['harmonic_period_recovered'].mean():.3f}"
    )
    print(
        "Persistent star-period flags: "
        f"{int(star_df['persistent_star_period_flag'].sum())}/{len(star_df)}"
    )
    print(
        "Single-event dominance heuristic rate: "
        f"{case_df['single_event_dominance_flag'].mean():.3f}"
    )
    if baseline_csv is not None and "matches_no_injection_period" in star_df.columns:
        print(
            "Dominant injected-run period matches zero-injection period: "
            f"{int(star_df['matches_no_injection_period'].sum())}/{len(star_df)}"
        )
    print(f"Output: {out_dir}")
    print(
        "Interpretation: descriptive TPS-like diagnostics only; do not use these "
        "heuristic flags as detection vetoes until validated against nulls."
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "outputs/experiments/batman_physical_detection_poc/"
            "pilot10/tps_like_comparator"
        ),
        help="Directory containing tps_like_results.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--harmonic-tolerance", type=float, default=0.03)
    parser.add_argument("--period-cluster-tolerance", type=float, default=0.01)
    parser.add_argument("--min-cluster-fraction", type=float, default=0.50)
    parser.add_argument("--min-distinct-injected-periods", type=int, default=2)
    parser.add_argument("--high-mes-quantile", type=float, default=0.75)
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=None,
        help=(
            "Optional future zero-injection TPS-like baseline CSV with at least "
            "target_id,recovered_period_days."
        ),
    )
    parser.add_argument("--baseline-period-tolerance", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_audit(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        harmonic_tolerance=args.harmonic_tolerance,
        period_cluster_tolerance=args.period_cluster_tolerance,
        min_cluster_fraction=args.min_cluster_fraction,
        min_distinct_injected_periods=args.min_distinct_injected_periods,
        high_mes_quantile=args.high_mes_quantile,
        baseline_csv=args.baseline_csv,
        baseline_period_tolerance=args.baseline_period_tolerance,
    )


if __name__ == "__main__":
    main()
