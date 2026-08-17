"""QC and stratified analysis for the BATMAN physical-transit POC.

This script does not rerun injections or detectors. It reads the existing
physical_retention.csv, detector_results.csv, and base_models.csv outputs and
produces convergence-aware, per-star, per-injection, and tie-aware winner
summaries.

Winner conventions are intentionally transparent:
- morphology winner: highest median template correlation;
- whitening winner: lowest median absolute residual/background ACF(1);
- oracle-SNR winner: highest median oracle signal SNR;
- detector winner: highest exact/harmonic recovery rate independently.

No opaque composite score is created. Ties are retained with ``|`` separators.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("outputs/experiments/batman_physical_detection_poc/pilot10")
DEFAULT_OUTPUT_NAME = "qc_analysis"
BRANCH_ORDER = ("raw", "arima", "kalman", "gp")
DETECTOR_ORDER = ("bls", "trapezoid", "tls")


def normalize_target_id(value) -> str:
    return str(value).upper().replace("KIC", "").strip()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=None,
        help=(
            "Optional canonical stellar-feature CSV. If supplied, it is joined "
            "to the per-star winner table by target_id for the next routing step."
        ),
    )
    parser.add_argument(
        "--feature-id-column",
        default="target_id",
        help="Target-ID column in --features-csv (default: target_id).",
    )
    return parser.parse_args(argv)


def require_columns(frame: pd.DataFrame, required: Iterable[str], name: str):
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def boolish(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    values = series.astype(str).str.strip().str.lower()
    mapped = values.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
            "nan": pd.NA,
            "none": pd.NA,
            "": pd.NA,
        }
    )
    return mapped.astype("boolean")


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["target_id"] = frame["target_id"].map(normalize_target_id)
    frame["quarter"] = pd.to_numeric(frame["quarter"], errors="coerce").astype("Int64")
    frame["branch"] = frame["branch"].astype(str).str.lower().str.strip()
    return frame


def build_model_status(base: pd.DataFrame) -> pd.DataFrame:
    require_columns(base, ["target_id", "quarter", "branch", "converged"], "base_models.csv")
    base = normalize_keys(base)
    base = base[base["branch"].isin(BRANCH_ORDER)].copy()
    base["converged_bool"] = boolish(base["converged"])
    if "error" not in base:
        base["error"] = ""
    error = base["error"].fillna("").astype(str).str.strip()

    # GP boundary warnings are not ordinary fit failures.  In this project the
    # 1-day lower length-scale bound is a deliberate transit-protection
    # constraint.  A GP that reaches that bound but still produces a finite
    # model is therefore usable and should be labelled boundary_limited rather
    # than discarded as optimizer_flagged.
    if "length_scale_at_lower_bound" in base:
        explicit_boundary = boolish(base["length_scale_at_lower_bound"]).fillna(False)
    else:
        explicit_boundary = pd.Series(False, index=base.index, dtype=bool)

    # Backward compatibility for the already-completed BATMAN POC: the older
    # base_models.csv did not persist length_scale_at_lower_bound.  We verified
    # separately that every non-converged, error-free GP row in that run was
    # exactly pinned to the 1.0-day lower bound, so infer that specific legacy
    # condition here.  Future runs use the explicit flag written by the runner.
    length_scale = (
        pd.to_numeric(base["length_scale_days"], errors="coerce")
        if "length_scale_days" in base
        else pd.Series(np.nan, index=base.index)
    )
    legacy_boundary = (
        base["branch"].eq("gp")
        & ~base["converged_bool"].fillna(False)
        & error.eq("")
        & np.isclose(length_scale, 1.0, rtol=0.0, atol=1e-9)
    )
    base["boundary_limited_bool"] = (
        explicit_boundary.astype(bool) | legacy_boundary.astype(bool)
    )

    base["fit_status"] = np.where(
        error.ne(""),
        "failed",
        np.where(
            base["branch"].eq("gp") & base["boundary_limited_bool"],
            "boundary_limited",
            np.where(
                base["branch"].eq("gp") & base["converged_bool"].fillna(False),
                "interior_optimum",
                np.where(
                    base["converged_bool"].fillna(False),
                    "clean",
                    "optimizer_flagged",
                ),
            ),
        ),
    )
    base.loc[base["branch"].eq("raw"), "fit_status"] = "clean"

    # fit_clean is retained as a backward-compatible column name because the
    # bridge/winner scripts already consume it.  Semantically it now means
    # "usable for scientific comparison": clean, interior GP optimum, or
    # scientifically constrained boundary-limited GP.
    base["fit_clean"] = base["fit_status"].isin(
        ["clean", "interior_optimum", "boundary_limited"]
    )
    base["fit_interior"] = base["fit_status"].isin(["clean", "interior_optimum"])
    keep = [
        "target_id",
        "quarter",
        "sample_stratum",
        "branch",
        "converged_bool",
        "boundary_limited_bool",
        "fit_status",
        "fit_clean",
        "fit_interior",
        "error",
    ]
    for optional in [
        "runtime_seconds",
        "aic",
        "length_scale_days",
        "length_scale_at_lower_bound",
        "optimizer_warning_count",
        "optimizer_warning_message",
    ]:
        if optional in base:
            keep.append(optional)
    return base[keep].drop_duplicates(["target_id", "quarter", "branch"], keep="last")


def attach_status(frame: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    frame = normalize_keys(frame)
    merged = frame.merge(
        status[["target_id", "quarter", "branch", "fit_status", "fit_clean"]],
        on=["target_id", "quarter", "branch"],
        how="left",
        validate="many_to_one",
    )
    merged["fit_status"] = merged["fit_status"].fillna("unknown")
    merged["fit_clean"] = merged["fit_clean"].fillna(False).astype(bool)
    return merged


def median_abs(series: pd.Series) -> float:
    values = numeric(series).abs()
    return float(values.median())


def summarize_retention(retention: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    work = retention.copy()
    if "success" in work:
        work["success_bool"] = boolish(work["success"]).fillna(False).astype(bool)
    else:
        work["success_bool"] = True
    successful = work[work["success_bool"]].copy()
    successful["abs_background_acf1"] = numeric(successful["background_acf1"]).abs()
    successful["peak_depth_abs_error"] = (numeric(successful["peak_depth_ratio"]) - 1.0).abs()
    successful["amplitude_abs_error"] = (
        numeric(successful["template_amplitude_ratio"]) - 1.0
    ).abs()
    successful["depth_within_20pct"] = numeric(successful["peak_depth_ratio"]).between(0.8, 1.2)
    successful["corr_ge_0p8"] = numeric(successful["template_correlation"]).ge(0.8)

    agg_spec = dict(
        n_successful_cases=("case_index", "count"),
        median_template_amplitude_ratio=("template_amplitude_ratio", "median"),
        median_amplitude_abs_error=("amplitude_abs_error", "median"),
        median_peak_depth_ratio=("peak_depth_ratio", "median"),
        median_peak_depth_abs_error=("peak_depth_abs_error", "median"),
        fraction_peak_depth_within_20pct=("depth_within_20pct", "mean"),
        median_template_energy_ratio=("template_energy_ratio", "median"),
        median_template_correlation=("template_correlation", "median"),
        fraction_template_corr_ge_0p8=("corr_ge_0p8", "mean"),
        median_template_rmse_ppm=("template_rmse_ppm", "median"),
        median_oracle_signal_snr=("oracle_signal_snr", "median"),
        median_background_scale_ppm=("background_scale_ppm", "median"),
        median_abs_background_acf1=("abs_background_acf1", "median"),
    )
    if "target_id" in successful.columns and "target_id" not in group_cols:
        agg_spec["n_successful_stars"] = ("target_id", "nunique")
    agg = successful.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()

    attempted = work.groupby(group_cols, dropna=False).size().rename("n_attempted_cases").reset_index()
    result = attempted.merge(agg, on=group_cols, how="left")
    result["success_rate"] = result["n_successful_cases"] / result["n_attempted_cases"].clip(lower=1)
    return result


def summarize_detectors(detectors: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    work = detectors.copy()
    work["success_bool"] = boolish(work["success"]).fillna(False).astype(bool)
    work["exact_bool"] = boolish(work["exact_period_recovered"]).fillna(False).astype(bool)
    work["harmonic_bool"] = boolish(work["harmonic_period_recovered"]).fillna(False).astype(bool)

    attempted = work.groupby(group_cols, dropna=False).size().rename("n_attempted_runs").reset_index()
    successful = work[work["success_bool"]].copy()
    agg_spec = dict(
        n_successful_runs=("case_index", "count"),
        exact_period_recovery=("exact_bool", "mean"),
        harmonic_period_recovery=("harmonic_bool", "mean"),
        median_period_error=("period_exact_fractional_error", "median"),
        median_runtime_seconds=("runtime_seconds", "median"),
    )
    if "target_id" in successful.columns and "target_id" not in group_cols:
        agg_spec["n_successful_stars"] = ("target_id", "nunique")
    agg = successful.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()
    result = attempted.merge(agg, on=group_cols, how="left")
    result["run_success_rate"] = result["n_successful_runs"] / result["n_attempted_runs"].clip(lower=1)
    return result


def join_fit_status(summary: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    keys = ["target_id", "quarter", "branch"]
    return summary.merge(
        status[keys + ["fit_status", "fit_clean"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )


def tied_labels(
    frame: pd.DataFrame,
    metric: str,
    maximize: bool,
    eligible=None,
    exclude_raw: bool = False,
) -> str:
    data = frame.copy()
    if eligible is not None:
        data = data[data[eligible].fillna(False)]
    if exclude_raw:
        data = data[~data["branch"].eq("raw")]
    values = numeric(data[metric])
    data = data[values.notna()].copy()
    if data.empty:
        return ""
    data["_metric"] = numeric(data[metric])
    best = data["_metric"].max() if maximize else data["_metric"].min()
    mask = np.isclose(data["_metric"], best, rtol=1.0e-9, atol=1.0e-12)
    labels = data.loc[mask, "branch"].astype(str).tolist()
    ordered = [branch for branch in BRANCH_ORDER if branch in labels]
    ordered.extend(sorted(set(labels).difference(ordered)))
    return "|".join(ordered)


def build_star_winners(
    retention_star: pd.DataFrame,
    detector_star: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    star_keys = ["target_id", "quarter", "sample_stratum"]
    detector_names = sorted(set(detector_star["detector"].astype(str)))

    for keys, group in retention_star.groupby(star_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(star_keys, keys))
        row["n_clean_branches"] = int(group["fit_clean"].fillna(False).sum())
        row["morphology_winner_all"] = tied_labels(
            group, "median_template_correlation", maximize=True
        )
        row["morphology_winner_clean"] = tied_labels(
            group, "median_template_correlation", maximize=True, eligible="fit_clean"
        )
        row["whitening_winner_all"] = tied_labels(
            group, "median_abs_background_acf1", maximize=False
        )
        row["whitening_winner_clean"] = tied_labels(
            group, "median_abs_background_acf1", maximize=False, eligible="fit_clean"
        )
        row["oracle_snr_winner_all"] = tied_labels(
            group, "median_oracle_signal_snr", maximize=True
        )
        row["oracle_snr_winner_clean"] = tied_labels(
            group, "median_oracle_signal_snr", maximize=True, eligible="fit_clean"
        )
        # Raw is guaranteed to preserve the injected template by construction,
        # so report non-raw winners separately for morphology/background-model QC.
        row["morphology_winner_nonraw_clean"] = tied_labels(
            group,
            "median_template_correlation",
            maximize=True,
            eligible="fit_clean",
            exclude_raw=True,
        )
        row["whitening_winner_nonraw_clean"] = tied_labels(
            group,
            "median_abs_background_acf1",
            maximize=False,
            eligible="fit_clean",
            exclude_raw=True,
        )
        row["oracle_snr_winner_nonraw_clean"] = tied_labels(
            group,
            "median_oracle_signal_snr",
            maximize=True,
            eligible="fit_clean",
            exclude_raw=True,
        )

        target, quarter, _ = keys
        det_for_star = detector_star[
            detector_star["target_id"].eq(target)
            & detector_star["quarter"].eq(quarter)
        ]
        for detector in detector_names:
            dgroup = det_for_star[det_for_star["detector"].eq(detector)]
            row[f"{detector}_exact_winner_all"] = tied_labels(
                dgroup, "exact_period_recovery", maximize=True
            )
            row[f"{detector}_exact_winner_clean"] = tied_labels(
                dgroup, "exact_period_recovery", maximize=True, eligible="fit_clean"
            )
            row[f"{detector}_harmonic_winner_clean"] = tied_labels(
                dgroup, "harmonic_period_recovery", maximize=True, eligible="fit_clean"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_case_winners(retention_case_clean: pd.DataFrame, detector_case_clean: pd.DataFrame) -> pd.DataFrame:
    case_meta = [
        column
        for column in [
            "case_index",
            "injected_period_days",
            "requested_duration_hours",
            "requested_depth",
            "phase_fraction",
        ]
        if column in retention_case_clean.columns
    ]
    rows = []
    for keys, group in retention_case_clean.groupby(case_meta, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(case_meta, keys))
        row["morphology_winner_clean"] = tied_labels(
            group, "median_template_correlation", maximize=True
        )
        row["morphology_winner_nonraw_clean"] = tied_labels(
            group, "median_template_correlation", maximize=True, exclude_raw=True
        )
        row["whitening_winner_clean"] = tied_labels(
            group, "median_abs_background_acf1", maximize=False
        )
        row["whitening_winner_nonraw_clean"] = tied_labels(
            group, "median_abs_background_acf1", maximize=False, exclude_raw=True
        )
        row["oracle_snr_winner_clean"] = tied_labels(
            group, "median_oracle_signal_snr", maximize=True
        )
        row["oracle_snr_winner_nonraw_clean"] = tied_labels(
            group, "median_oracle_signal_snr", maximize=True, exclude_raw=True
        )

        case_index = row["case_index"]
        dcase = detector_case_clean[detector_case_clean["case_index"].eq(case_index)]
        for detector in sorted(set(dcase["detector"].astype(str))):
            dgroup = dcase[dcase["detector"].eq(detector)]
            row[f"{detector}_exact_winner_clean"] = tied_labels(
                dgroup, "exact_period_recovery", maximize=True
            )
            row[f"{detector}_harmonic_winner_clean"] = tied_labels(
                dgroup, "harmonic_period_recovery", maximize=True
            )
        rows.append(row)
    return pd.DataFrame(rows)


def recovered_branches(frame: pd.DataFrame, column: str, clean_only: bool) -> str:
    data = frame.copy()
    if clean_only:
        data = data[data["fit_clean"].fillna(False)]
    recovered = boolish(data[column]).fillna(False)
    labels = data.loc[recovered, "branch"].astype(str).tolist()
    ordered = [branch for branch in BRANCH_ORDER if branch in labels]
    ordered.extend(sorted(set(labels).difference(ordered)))
    return "|".join(ordered)


def build_star_case_detector_audit(detectors: pd.DataFrame) -> pd.DataFrame:
    successful = detectors[boolish(detectors["success"]).fillna(False)].copy()
    group_cols = [
        "target_id",
        "quarter",
        "sample_stratum",
        "case_index",
        "detector",
    ]
    rows = []
    for keys, group in successful.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["exact_recovered_branches_all"] = recovered_branches(
            group, "exact_period_recovered", clean_only=False
        )
        row["exact_recovered_branches_clean"] = recovered_branches(
            group, "exact_period_recovered", clean_only=True
        )
        row["harmonic_recovered_branches_clean"] = recovered_branches(
            group, "harmonic_period_recovered", clean_only=True
        )
        clean = group[group["fit_clean"].fillna(False)].copy()
        if clean.empty:
            row["lowest_period_error_branch_clean"] = ""
            row["lowest_period_error_clean"] = np.nan
        else:
            errors = numeric(clean["period_exact_fractional_error"])
            clean = clean[errors.notna()].copy()
            clean["_error"] = numeric(clean["period_exact_fractional_error"])
            if clean.empty:
                row["lowest_period_error_branch_clean"] = ""
                row["lowest_period_error_clean"] = np.nan
            else:
                best = float(clean["_error"].min())
                ties = clean[np.isclose(clean["_error"], best, rtol=1e-9, atol=1e-12)]
                labels = ties["branch"].astype(str).tolist()
                ordered = [b for b in BRANCH_ORDER if b in labels]
                row["lowest_period_error_branch_clean"] = "|".join(ordered)
                row["lowest_period_error_clean"] = best
        rows.append(row)
    return pd.DataFrame(rows)


def add_case_metadata(summary: pd.DataFrame, retention: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = [
        column
        for column in [
            "case_index",
            "injected_period_days",
            "requested_duration_hours",
            "requested_depth",
            "phase_fraction",
        ]
        if column in retention.columns
    ]
    meta = retention[metadata_cols].drop_duplicates("case_index")
    return summary.merge(meta, on="case_index", how="left", validate="many_to_one")


def write_optional_feature_join(
    winners: pd.DataFrame,
    features_csv: Path | None,
    feature_id_column: str,
    output_dir: Path,
):
    if features_csv is None:
        return
    features = pd.read_csv(features_csv, dtype={feature_id_column: str})
    if feature_id_column not in features:
        raise ValueError(f"Feature file does not contain {feature_id_column!r}.")
    features = features.copy()
    features["target_id"] = features[feature_id_column].map(normalize_target_id)
    if feature_id_column != "target_id":
        features = features.drop(columns=[feature_id_column])
    features = features.drop_duplicates("target_id", keep="last")
    merged = winners.merge(features, on="target_id", how="left", validate="many_to_one")
    merged.to_csv(output_dir / "per_star_winners_with_features.csv", index=False)


def main(argv=None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / DEFAULT_OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    retention = pd.read_csv(input_dir / "physical_retention.csv", dtype={"target_id": str})
    detectors = pd.read_csv(input_dir / "detector_results.csv", dtype={"target_id": str})
    base = pd.read_csv(input_dir / "base_models.csv", dtype={"target_id": str})

    require_columns(
        retention,
        [
            "target_id",
            "quarter",
            "sample_stratum",
            "case_index",
            "branch",
            "template_amplitude_ratio",
            "peak_depth_ratio",
            "template_energy_ratio",
            "template_correlation",
            "template_rmse_ppm",
            "oracle_signal_snr",
            "background_scale_ppm",
            "background_acf1",
        ],
        "physical_retention.csv",
    )
    require_columns(
        detectors,
        [
            "target_id",
            "quarter",
            "sample_stratum",
            "case_index",
            "branch",
            "detector",
            "success",
            "exact_period_recovered",
            "harmonic_period_recovered",
            "period_exact_fractional_error",
            "runtime_seconds",
        ],
        "detector_results.csv",
    )

    status = build_model_status(base)
    retention = attach_status(retention, status)
    detectors = attach_status(detectors, status)
    detectors["detector"] = detectors["detector"].astype(str).str.lower().str.strip()

    status.to_csv(output_dir / "convergence_audit.csv", index=False)

    retention_star = summarize_retention(
        retention, ["target_id", "quarter", "sample_stratum", "branch"]
    )
    retention_star = join_fit_status(retention_star, status)
    raw_snr = (
        retention_star[retention_star["branch"].eq("raw")][
            ["target_id", "quarter", "median_oracle_signal_snr"]
        ]
        .rename(columns={"median_oracle_signal_snr": "raw_median_oracle_signal_snr"})
    )
    retention_star = retention_star.merge(raw_snr, on=["target_id", "quarter"], how="left")
    retention_star["oracle_snr_gain_vs_raw"] = (
        retention_star["median_oracle_signal_snr"]
        / retention_star["raw_median_oracle_signal_snr"]
    )
    retention_star.to_csv(output_dir / "per_star_branch_retention.csv", index=False)

    detector_star = summarize_detectors(
        detectors,
        ["target_id", "quarter", "sample_stratum", "branch", "detector"],
    )
    detector_star = join_fit_status(detector_star, status)
    detector_star.to_csv(output_dir / "per_star_branch_detector.csv", index=False)

    # Explicit all-vs-clean aggregate tables answer whether optimizer warnings
    # materially change the headline result.
    retention_fit = summarize_retention(retention, ["branch", "fit_status"])
    retention_fit.to_csv(output_dir / "retention_by_branch_fit_status.csv", index=False)
    detector_fit = summarize_detectors(detectors, ["branch", "fit_status", "detector"])
    detector_fit.to_csv(output_dir / "detector_by_branch_fit_status.csv", index=False)

    retention_case_all = add_case_metadata(
        summarize_retention(retention, ["case_index", "branch"]), retention
    )
    retention_case_all.to_csv(
        output_dir / "per_injection_branch_retention_all.csv", index=False
    )
    retention_clean_rows = retention[retention["fit_clean"]].copy()
    retention_case_clean = add_case_metadata(
        summarize_retention(retention_clean_rows, ["case_index", "branch"]), retention
    )
    retention_case_clean.to_csv(
        output_dir / "per_injection_branch_retention_clean.csv", index=False
    )

    detector_case_all = add_case_metadata(
        summarize_detectors(detectors, ["case_index", "branch", "detector"]), retention
    )
    detector_case_all.to_csv(
        output_dir / "per_injection_branch_detector_all.csv", index=False
    )
    detector_clean_rows = detectors[detectors["fit_clean"]].copy()
    detector_case_clean = add_case_metadata(
        summarize_detectors(
            detector_clean_rows, ["case_index", "branch", "detector"]
        ),
        retention,
    )
    detector_case_clean.to_csv(
        output_dir / "per_injection_branch_detector_clean.csv", index=False
    )

    star_winners = build_star_winners(retention_star, detector_star)
    star_winners.to_csv(output_dir / "per_star_winners.csv", index=False)
    write_optional_feature_join(
        star_winners, args.features_csv, args.feature_id_column, output_dir
    )

    # Case winners are computed only from clean-fit rows, while preserving the
    # actual clean sample size for every branch/case in the companion tables.
    case_winners = build_case_winners(retention_case_clean, detector_case_clean)
    case_winners.to_csv(output_dir / "per_injection_winners.csv", index=False)

    star_case_audit = build_star_case_detector_audit(detectors)
    star_case_audit = add_case_metadata(star_case_audit, retention)
    star_case_audit.to_csv(output_dir / "star_case_detector_recovery_audit.csv", index=False)

    # Compact console view.
    print("\n=== CONVERGENCE AUDIT ===\n")
    convergence_view = (
        status.groupby(["branch", "fit_status"], dropna=False)
        .size()
        .rename("n_stars")
        .reset_index()
    )
    print(convergence_view.to_string(index=False))

    print("\n=== USABLE-FIT PER-STAR WINNERS (legacy *_clean column names) ===\n")
    winner_cols = [
        "target_id",
        "sample_stratum",
        "morphology_winner_clean",
        "whitening_winner_clean",
        "oracle_snr_winner_clean",
        "morphology_winner_nonraw_clean",
        "whitening_winner_nonraw_clean",
        "oracle_snr_winner_nonraw_clean",
    ]
    winner_cols.extend(
        column
        for column in ["bls_exact_winner_clean", "tls_exact_winner_clean"]
        if column in star_winners.columns
    )
    print(star_winners[winner_cols].to_string(index=False))

    print(f"\nWrote QC outputs to: {output_dir}")
    print("No injections, background models, or detector searches were rerun.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
