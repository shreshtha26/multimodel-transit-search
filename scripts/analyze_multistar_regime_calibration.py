"""Analyze regime-aware calibration for the multi-star BLS/ARIMA-TCF run."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_DIR = PROJECT_ROOT / "outputs/experiments/multistar_bls_tcf/optimized/metrics"


def default_settings():
    return SimpleNamespace(metrics_dir=DEFAULT_METRICS_DIR, fap_level=0.01, minimum_success_fraction=0.90)


def build_parser():
    defaults = default_settings()
    parser = argparse.ArgumentParser(description="Analyze multi-star BLS/ARIMA-TCF regime calibration.")
    parser.add_argument("--metrics-dir", type=Path, default=defaults.metrics_dir)
    parser.add_argument("--fap-level", type=float, default=defaults.fap_level)
    parser.add_argument("--minimum-success-fraction", type=float, default=defaults.minimum_success_fraction)
    return parser


def normalize_target_id(value):
    return str(value).upper().replace("KIC", "").strip()


def normalize_key_columns(frame):
    frame = frame.copy()
    if "target_id" in frame.columns:
        frame["target_id"] = frame["target_id"].map(normalize_target_id)
    if "quarter" in frame.columns:
        frame["quarter"] = pd.to_numeric(frame["quarter"], errors="raise").astype(int)
    return frame


def boolean_series(frame, column):
    return frame[column].astype(str).str.lower().isin(["true", "1"])


def calibrated_threshold(values, fap_level):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.quantile(values, 1.0 - float(fap_level), method="higher"))


def empirical_p_values(scores, null_scores):
    scores = np.asarray(scores, dtype=float)
    null_scores = np.asarray(null_scores, dtype=float)
    null_scores = null_scores[np.isfinite(null_scores)]
    if null_scores.size == 0:
        return np.full(scores.shape, np.nan, dtype=float)
    return np.asarray(
        [(np.sum(null_scores >= score) + 1.0) / (len(null_scores) + 1.0) if np.isfinite(score) else np.nan for score in scores],
        dtype=float,
    )


def metric_row(name, values):
    values = pd.Series(values).fillna(False).astype(bool)
    total = int(len(values))
    successes = int(values.sum())
    return {"metric": name, "successes": successes, "total": total, "rate": float(successes / total) if total else np.nan}


def load_inputs(metrics_dir):
    metrics_dir = Path(metrics_dir)
    injections = pd.read_csv(metrics_dir / "multistar_bls_tcf_injections.csv", dtype={"target_id": str})
    null_trials = pd.read_csv(metrics_dir / "multistar_null_trials.csv", dtype={"target_id": str})
    star_summaries = pd.read_csv(metrics_dir / "multistar_star_summary.csv", dtype={"target_id": str})
    return normalize_key_columns(injections), normalize_key_columns(null_trials), normalize_key_columns(star_summaries)


def attach_noise_quartile_to_nulls(null_trials, star_summaries):
    mapping = star_summaries[["target_id", "quarter", "noise_quartile", "robust_flux_scatter_ppm", "gap_fraction"]].drop_duplicates(
        ["target_id", "quarter"]
    )
    return null_trials.merge(mapping, on=["target_id", "quarter"], how="left", validate="many_to_one")


def thresholds_by_noise_quartile(null_trials, args):
    rows = []
    for noise_quartile, group in null_trials.groupby("noise_quartile", dropna=False):
        tcf_success = group[boolean_series(group, "tcf_success")]
        bls_success = group[boolean_series(group, "bls_success")]
        expected_tcf = int(len(group) * float(args.minimum_success_fraction))
        expected_bls = int(len(group) * float(args.minimum_success_fraction))
        tcf_scores = tcf_success["tcf_max_score"].to_numpy(dtype=float)
        bls_scores = bls_success["bls_max_sde"].to_numpy(dtype=float)
        tcf_threshold = calibrated_threshold(tcf_scores, args.fap_level)
        bls_threshold = calibrated_threshold(bls_scores, args.fap_level)
        rows.append(
            {
                "noise_quartile": noise_quartile,
                "star_count": int(group["target_id"].nunique()),
                "null_trial_count": int(len(group)),
                "tcf_successful_null_count": int(np.isfinite(tcf_scores).sum()),
                "bls_successful_null_count": int(np.isfinite(bls_scores).sum()),
                "expected_successful_null_count": expected_tcf,
                "tcf_success_fraction_ok": bool(np.isfinite(tcf_scores).sum() >= expected_tcf),
                "bls_success_fraction_ok": bool(np.isfinite(bls_scores).sum() >= expected_bls),
                "fap_level": float(args.fap_level),
                "tcf_score_threshold": tcf_threshold,
                "bls_sde_threshold": bls_threshold,
                "tcf_observed_exceedance_fraction": float(np.mean(tcf_scores >= tcf_threshold)) if np.isfinite(tcf_threshold) else np.nan,
                "bls_observed_exceedance_fraction": float(np.mean(bls_scores >= bls_threshold)) if np.isfinite(bls_threshold) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("noise_quartile").reset_index(drop=True)


def add_regime_calibration(injections, null_trials, thresholds):
    injections = injections.copy()
    thresholds = thresholds[["noise_quartile", "tcf_score_threshold", "bls_sde_threshold"]].copy()
    thresholds = thresholds.rename(columns={"tcf_score_threshold": "tcf_regime_fap_threshold", "bls_sde_threshold": "bls_regime_fap_threshold"})
    injections = injections.merge(thresholds, on="noise_quartile", how="left", validate="many_to_one")
    injections["tcf_regime_empirical_p_value"] = np.nan
    injections["bls_regime_empirical_p_value"] = np.nan
    for noise_quartile, group_index in injections.groupby("noise_quartile", dropna=False).groups.items():
        null_group = null_trials[null_trials["noise_quartile"].astype(str) == str(noise_quartile)]
        tcf_null = null_group[boolean_series(null_group, "tcf_success")]["tcf_max_score"].to_numpy(dtype=float)
        bls_null = null_group[boolean_series(null_group, "bls_success")]["bls_max_sde"].to_numpy(dtype=float)
        injections.loc[group_index, "tcf_regime_empirical_p_value"] = empirical_p_values(injections.loc[group_index, "tcf_score"], tcf_null)
        injections.loc[group_index, "bls_regime_empirical_p_value"] = empirical_p_values(injections.loc[group_index, "bls_sde"], bls_null)
    injections["tcf_passes_regime_fap"] = injections["tcf_score"] >= injections["tcf_regime_fap_threshold"]
    injections["bls_passes_regime_fap"] = injections["bls_sde"] >= injections["bls_regime_fap_threshold"]
    injections["tcf_harmonic_recovered_regime_fap"] = injections["tcf_period_matched"] & injections["tcf_passes_regime_fap"]
    injections["bls_harmonic_recovered_regime_fap"] = injections["bls_period_matched"] & injections["bls_passes_regime_fap"]
    injections["harmonic_union_regime_fap"] = injections["tcf_harmonic_recovered_regime_fap"] | injections["bls_harmonic_recovered_regime_fap"]
    injections["tcf_exact_recovered_regime_fap"] = injections["tcf_exact_period_matched"] & injections["tcf_passes_regime_fap"]
    injections["bls_exact_recovered_regime_fap"] = injections["bls_exact_period_matched"] & injections["bls_passes_regime_fap"]
    injections["exact_union_regime_fap"] = injections["tcf_exact_recovered_regime_fap"] | injections["bls_exact_recovered_regime_fap"]
    injections["exact_top10_union"] = injections["tcf_exact_period_present_top10"] | injections["bls_exact_period_present_top10"]
    injections["harmonic_rank1_union_before_fap"] = injections["tcf_period_matched"] | injections["bls_period_matched"]
    injections["exact_rank1_union_before_fap"] = injections["tcf_exact_period_matched"] | injections["bls_exact_period_matched"]
    return injections


def union_recall_summary(injections):
    rows = [
        metric_row("tcf_harmonic_rank1_before_fap", injections["tcf_period_matched"]),
        metric_row("bls_harmonic_rank1_before_fap", injections["bls_period_matched"]),
        metric_row("harmonic_union_rank1_before_fap", injections["harmonic_rank1_union_before_fap"]),
        metric_row("tcf_exact_rank1_before_fap", injections["tcf_exact_period_matched"]),
        metric_row("bls_exact_rank1_before_fap", injections["bls_exact_period_matched"]),
        metric_row("exact_union_rank1_before_fap", injections["exact_rank1_union_before_fap"]),
        metric_row("tcf_exact_recall_at_10", injections["tcf_exact_period_present_top10"]),
        metric_row("bls_exact_recall_at_10", injections["bls_exact_period_present_top10"]),
        metric_row("exact_union_recall_at_10", injections["exact_top10_union"]),
        metric_row("tcf_exact_recovery_global_fap", injections["tcf_exact_recovered"]),
        metric_row("bls_exact_recovery_global_fap", injections["bls_exact_recovered"]),
        metric_row("exact_union_recovery_global_fap", injections["exact_union"]),
        metric_row("tcf_exact_recovery_regime_fap", injections["tcf_exact_recovered_regime_fap"]),
        metric_row("bls_exact_recovery_regime_fap", injections["bls_exact_recovered_regime_fap"]),
        metric_row("exact_union_recovery_regime_fap", injections["exact_union_regime_fap"]),
        metric_row("tcf_harmonic_recovery_regime_fap", injections["tcf_harmonic_recovered_regime_fap"]),
        metric_row("bls_harmonic_recovery_regime_fap", injections["bls_harmonic_recovered_regime_fap"]),
        metric_row("harmonic_union_recovery_regime_fap", injections["harmonic_union_regime_fap"]),
    ]
    return pd.DataFrame(rows)


def recovery_by_noise_quartile(injections):
    return injections.groupby("noise_quartile", dropna=False, as_index=False).agg(
        injection_count=("target_id", "size"),
        star_count=("target_id", "nunique"),
        median_robust_flux_scatter_ppm=("robust_flux_scatter_ppm", "median"),
        median_gap_fraction=("gap_fraction", "median"),
        tcf_harmonic_rank1_before_fap=("tcf_period_matched", "mean"),
        bls_harmonic_rank1_before_fap=("bls_period_matched", "mean"),
        harmonic_union_rank1_before_fap=("harmonic_rank1_union_before_fap", "mean"),
        tcf_exact_rank1_before_fap=("tcf_exact_period_matched", "mean"),
        bls_exact_rank1_before_fap=("bls_exact_period_matched", "mean"),
        exact_union_rank1_before_fap=("exact_rank1_union_before_fap", "mean"),
        tcf_exact_recall_at_10=("tcf_exact_period_present_top10", "mean"),
        bls_exact_recall_at_10=("bls_exact_period_present_top10", "mean"),
        exact_union_recall_at_10=("exact_top10_union", "mean"),
        tcf_exact_recovery_global_fap=("tcf_exact_recovered", "mean"),
        bls_exact_recovery_global_fap=("bls_exact_recovered", "mean"),
        exact_union_recovery_global_fap=("exact_union", "mean"),
        tcf_exact_recovery_regime_fap=("tcf_exact_recovered_regime_fap", "mean"),
        bls_exact_recovery_regime_fap=("bls_exact_recovered_regime_fap", "mean"),
        exact_union_recovery_regime_fap=("exact_union_regime_fap", "mean"),
        tcf_only_exact_rank1_before_fap=("exact_tcf_only", "mean"),
        bls_only_exact_rank1_before_fap=("exact_bls_only", "mean"),
        neither_exact_rank1_before_fap=("exact_neither", "mean"),
    )


def original_candidates_by_noise_quartile(star_summaries, thresholds):
    star_summaries = star_summaries.copy()
    thresholds = thresholds[["noise_quartile", "tcf_score_threshold", "bls_sde_threshold"]].copy()
    star_summaries = star_summaries.merge(thresholds, on="noise_quartile", how="left", validate="many_to_one")
    star_summaries["original_tcf_passes_regime_fap"] = star_summaries["original_tcf_score"] >= star_summaries["tcf_score_threshold"]
    star_summaries["original_bls_passes_regime_fap"] = star_summaries["original_bls_sde"] >= star_summaries["bls_sde_threshold"]
    star_summaries["original_any_passes_global_fap"] = boolean_series(star_summaries, "original_tcf_passes_global_fap") | boolean_series(
        star_summaries, "original_bls_passes_global_fap"
    )
    star_summaries["original_any_passes_regime_fap"] = star_summaries["original_tcf_passes_regime_fap"] | star_summaries["original_bls_passes_regime_fap"]
    grouped = star_summaries.groupby("noise_quartile", dropna=False, as_index=False).agg(
        star_count=("target_id", "nunique"),
        median_robust_flux_scatter_ppm=("robust_flux_scatter_ppm", "median"),
        median_gap_fraction=("gap_fraction", "median"),
        original_tcf_candidate_rate_global_fap=("original_tcf_passes_global_fap", "mean"),
        original_bls_candidate_rate_global_fap=("original_bls_passes_global_fap", "mean"),
        original_any_candidate_rate_global_fap=("original_any_passes_global_fap", "mean"),
        original_tcf_candidate_rate_regime_fap=("original_tcf_passes_regime_fap", "mean"),
        original_bls_candidate_rate_regime_fap=("original_bls_passes_regime_fap", "mean"),
        original_any_candidate_rate_regime_fap=("original_any_passes_regime_fap", "mean"),
        original_tcf_candidate_count_regime_fap=("original_tcf_passes_regime_fap", "sum"),
        original_bls_candidate_count_regime_fap=("original_bls_passes_regime_fap", "sum"),
        original_any_candidate_count_regime_fap=("original_any_passes_regime_fap", "sum"),
    )
    return grouped, star_summaries


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def run_analysis(args):
    metrics_dir = Path(args.metrics_dir)
    injections, null_trials, star_summaries = load_inputs(metrics_dir)
    null_trials = attach_noise_quartile_to_nulls(null_trials, star_summaries)
    thresholds = thresholds_by_noise_quartile(null_trials, args)
    calibrated_injections = add_regime_calibration(injections, null_trials, thresholds)
    union_summary = union_recall_summary(calibrated_injections)
    recovery_by_noise = recovery_by_noise_quartile(calibrated_injections)
    original_by_noise, calibrated_stars = original_candidates_by_noise_quartile(star_summaries, thresholds)
    output_paths = {
        "union_recall_summary": metrics_dir / "multistar_union_recall_summary.csv",
        "recovery_by_noise_quartile": metrics_dir / "multistar_recovery_by_noise_quartile.csv",
        "thresholds_by_noise_quartile": metrics_dir / "multistar_thresholds_by_noise_quartile.csv",
        "original_candidates_by_noise_quartile": metrics_dir / "multistar_original_candidates_by_noise_quartile.csv",
        "regime_calibrated_injections": metrics_dir / "multistar_regime_calibrated_injections.csv",
        "regime_calibrated_star_summary": metrics_dir / "multistar_regime_calibrated_star_summary.csv",
        "summary": metrics_dir / "multistar_regime_calibration_summary.json",
    }
    union_summary.to_csv(output_paths["union_recall_summary"], index=False)
    recovery_by_noise.to_csv(output_paths["recovery_by_noise_quartile"], index=False)
    thresholds.to_csv(output_paths["thresholds_by_noise_quartile"], index=False)
    original_by_noise.to_csv(output_paths["original_candidates_by_noise_quartile"], index=False)
    calibrated_injections.to_csv(output_paths["regime_calibrated_injections"], index=False)
    calibrated_stars.to_csv(output_paths["regime_calibrated_star_summary"], index=False)
    summary = {
        "metrics_dir": metrics_dir,
        "fap_level": float(args.fap_level),
        "star_count": int(star_summaries["target_id"].nunique()),
        "injection_count": int(len(calibrated_injections)),
        "null_trial_count": int(len(null_trials)),
        "exact_union_recall_at_10": float(union_summary.loc[union_summary["metric"] == "exact_union_recall_at_10", "rate"].iloc[0]),
        "exact_union_rank1_before_fap": float(union_summary.loc[union_summary["metric"] == "exact_union_rank1_before_fap", "rate"].iloc[0]),
        "exact_union_recovery_global_fap": float(union_summary.loc[union_summary["metric"] == "exact_union_recovery_global_fap", "rate"].iloc[0]),
        "exact_union_recovery_regime_fap": float(union_summary.loc[union_summary["metric"] == "exact_union_recovery_regime_fap", "rate"].iloc[0]),
        "outputs": output_paths,
    }
    output_paths["summary"].write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    return output_paths, summary


def main(args=None):
    args = args or build_parser().parse_args()
    output_paths, summary = run_analysis(args)
    print(f"Metrics directory: {args.metrics_dir}")
    print(f"Stars: {summary['star_count']}")
    print(f"Injections: {summary['injection_count']}")
    print(f"Null trials: {summary['null_trial_count']}")
    print(f"Exact Recall@10 union: {summary['exact_union_recall_at_10']:.3f}")
    print(f"Exact rank-1 union before FAP: {summary['exact_union_rank1_before_fap']:.3f}")
    print(f"Exact union recovery at global FAP: {summary['exact_union_recovery_global_fap']:.3f}")
    print(f"Exact union recovery at regime FAP: {summary['exact_union_recovery_regime_fap']:.3f}")
    for path in output_paths.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
