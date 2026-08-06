"""Compare BLS and ARIMA-TCF recovery at independently calibrated 1 percent FAP."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLS_PATH = PROJECT_ROOT / "outputs/experiments/bls_injection_grid/metrics/kic_11904151_q5_bls_injection_grid.csv"
TCF_PATH = PROJECT_ROOT / "outputs/experiments/tcf_injection_grid/metrics/kic_11904151_q5_tcf_injection_grid.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/bls_tcf_comparison"
EXACT_PERIOD_TOLERANCE_FRACTION = 0.02

def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def boolean_series(values):
    return values.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

def grouped_comparison(comparison, column):
    return comparison.groupby(column, as_index=False).agg(injection_count=("bls_recovered", "size"), bls_recovery_rate=("bls_recovered", "mean"), tcf_recovery_rate=("tcf_recovered", "mean"), union_recovery_rate=("either_recovered", "mean"), bls_exact_recovery_rate=("bls_exact_recovered", "mean"), tcf_exact_recovery_rate=("tcf_exact_recovered", "mean"), exact_union_recovery_rate=("exact_either_recovered", "mean"), tcf_exact_period_present_top10_rate=("tcf_exact_period_present_top10", "mean"), tcf_exact_period_below_rank1_rate=("tcf_exact_period_below_rank1", "mean"), tcf_exact_period_absent_top10_rate=("tcf_exact_period_absent_top10", "mean"), tcf_top10_candidate_only_vs_bls_exact_rate=("tcf_top10_candidate_only_vs_bls_exact", "mean"), tcf_median_exact_period_rank_top10=("tcf_exact_period_rank_top10", "median"))

def run_comparison():
    if not BLS_PATH.exists():
        raise FileNotFoundError(f"BLS result file does not exist: {BLS_PATH}")
    if not TCF_PATH.exists():
        raise FileNotFoundError(f"TCF result file does not exist: {TCF_PATH}")
    bls = pd.read_csv(BLS_PATH)
    tcf = pd.read_csv(TCF_PATH)
    keys = ["injected_period_days", "injected_duration_hours", "injected_depth", "epoch_phase_fraction"]
    bls_columns = keys + ["recovered_period_days", "recovered_power", "period_matched", "passes_fap_1_percent", "recovered_at_fap_1_percent"]
    tcf_columns = keys + ["recovered_period_days", "recovered_score", "period_matched", "passes_fap_1_percent", "recovered_at_fap_1_percent", "exact_period_rank_top10", "exact_period_present_top10", "half_period_rank_top10", "double_period_rank_top10", "triple_period_rank_top10", "top_periods_json", "top_scores_json"]
    missing_bls_columns = set(bls_columns).difference(bls.columns)
    missing_tcf_columns = set(tcf_columns).difference(tcf.columns)
    if missing_bls_columns:
        raise ValueError(f"BLS result file is missing columns: {sorted(missing_bls_columns)}")
    if missing_tcf_columns:
        raise ValueError(f"TCF result file is missing columns: {sorted(missing_tcf_columns)}. Rerun scripts/run_tcf_injection_grid.py with the top-10 diagnostics.")
    bls = bls[bls_columns].rename(columns={"recovered_period_days": "bls_recovered_period_days", "recovered_power": "bls_recovered_score", "period_matched": "bls_period_matched", "passes_fap_1_percent": "bls_passes_fap_1_percent", "recovered_at_fap_1_percent": "bls_recovered"})
    tcf = tcf[tcf_columns].rename(columns={"recovered_period_days": "tcf_recovered_period_days", "recovered_score": "tcf_recovered_score", "period_matched": "tcf_period_matched", "passes_fap_1_percent": "tcf_passes_fap_1_percent", "recovered_at_fap_1_percent": "tcf_recovered", "exact_period_rank_top10": "tcf_exact_period_rank_top10", "exact_period_present_top10": "tcf_exact_period_present_top10", "half_period_rank_top10": "tcf_half_period_rank_top10", "double_period_rank_top10": "tcf_double_period_rank_top10", "triple_period_rank_top10": "tcf_triple_period_rank_top10", "top_periods_json": "tcf_top_periods_json", "top_scores_json": "tcf_top_scores_json"})
    comparison = bls.merge(tcf, on=keys, how="inner", validate="one_to_one")
    if len(comparison) != len(bls) or len(comparison) != len(tcf):
        raise ValueError("BLS and TCF injection grids do not contain the same cases.")
    boolean_columns = ["bls_period_matched", "bls_passes_fap_1_percent", "bls_recovered", "tcf_period_matched", "tcf_passes_fap_1_percent", "tcf_recovered", "tcf_exact_period_present_top10"]
    for column in boolean_columns:
        comparison[column] = boolean_series(comparison[column])
    rank_columns = ["tcf_exact_period_rank_top10", "tcf_half_period_rank_top10", "tcf_double_period_rank_top10", "tcf_triple_period_rank_top10"]
    for column in rank_columns:
        comparison[column] = pd.to_numeric(comparison[column], errors="coerce")
    comparison["both_recovered"] = comparison["bls_recovered"] & comparison["tcf_recovered"]
    comparison["bls_only"] = comparison["bls_recovered"] & ~comparison["tcf_recovered"]
    comparison["tcf_only"] = comparison["tcf_recovered"] & ~comparison["bls_recovered"]
    comparison["neither_recovered"] = ~comparison["bls_recovered"] & ~comparison["tcf_recovered"]
    comparison["either_recovered"] = comparison["bls_recovered"] | comparison["tcf_recovered"]
    comparison["bls_exact_period_error"] = np.abs(comparison["bls_recovered_period_days"] - comparison["injected_period_days"]) / comparison["injected_period_days"]
    comparison["tcf_exact_period_error"] = np.abs(comparison["tcf_recovered_period_days"] - comparison["injected_period_days"]) / comparison["injected_period_days"]
    comparison["bls_exact_period_matched"] = comparison["bls_exact_period_error"] <= EXACT_PERIOD_TOLERANCE_FRACTION
    comparison["tcf_exact_period_matched"] = comparison["tcf_exact_period_error"] <= EXACT_PERIOD_TOLERANCE_FRACTION
    comparison["bls_exact_recovered"] = comparison["bls_exact_period_matched"] & comparison["bls_passes_fap_1_percent"]
    comparison["tcf_exact_recovered"] = comparison["tcf_exact_period_matched"] & comparison["tcf_passes_fap_1_percent"]
    comparison["exact_both_recovered"] = comparison["bls_exact_recovered"] & comparison["tcf_exact_recovered"]
    comparison["exact_bls_only"] = comparison["bls_exact_recovered"] & ~comparison["tcf_exact_recovered"]
    comparison["exact_tcf_only"] = comparison["tcf_exact_recovered"] & ~comparison["bls_exact_recovered"]
    comparison["exact_neither_recovered"] = ~comparison["bls_exact_recovered"] & ~comparison["tcf_exact_recovered"]
    comparison["exact_either_recovered"] = comparison["bls_exact_recovered"] | comparison["tcf_exact_recovered"]
    comparison["tcf_exact_period_below_rank1"] = comparison["tcf_exact_period_present_top10"] & ~comparison["tcf_exact_period_matched"]
    comparison["tcf_exact_period_absent_top10"] = ~comparison["tcf_exact_period_present_top10"]
    comparison["tcf_top10_candidate_only_vs_bls_exact"] = comparison["tcf_exact_period_present_top10"] & ~comparison["bls_exact_recovered"]
    by_depth = grouped_comparison(comparison, "injected_depth")
    by_duration = grouped_comparison(comparison, "injected_duration_hours")
    by_period = grouped_comparison(comparison, "injected_period_days")
    weak_regime = comparison[np.isclose(comparison["injected_depth"], 0.0002)]
    exact_ranks = comparison["tcf_exact_period_rank_top10"].dropna()
    summary = {"injection_count": int(len(comparison)), "exact_period_tolerance_fraction": float(EXACT_PERIOD_TOLERANCE_FRACTION), "bls_recovery_rate": float(comparison["bls_recovered"].mean()), "tcf_recovery_rate": float(comparison["tcf_recovered"].mean()), "both_recovery_rate": float(comparison["both_recovered"].mean()), "bls_only_rate": float(comparison["bls_only"].mean()), "tcf_only_rate": float(comparison["tcf_only"].mean()), "neither_recovery_rate": float(comparison["neither_recovered"].mean()), "union_recovery_rate": float(comparison["either_recovered"].mean()), "tcf_additional_recoveries": int(comparison["tcf_only"].sum()), "bls_additional_recoveries": int(comparison["bls_only"].sum()), "bls_exact_period_recovery_rate": float(comparison["bls_exact_recovered"].mean()), "tcf_exact_period_recovery_rate": float(comparison["tcf_exact_recovered"].mean()), "exact_both_recovery_rate": float(comparison["exact_both_recovered"].mean()), "exact_bls_only_rate": float(comparison["exact_bls_only"].mean()), "exact_tcf_only_rate": float(comparison["exact_tcf_only"].mean()), "exact_neither_recovery_rate": float(comparison["exact_neither_recovered"].mean()), "exact_union_recovery_rate": float(comparison["exact_either_recovered"].mean()), "exact_tcf_additional_recoveries": int(comparison["exact_tcf_only"].sum()), "exact_bls_additional_recoveries": int(comparison["exact_bls_only"].sum()), "tcf_exact_period_present_top10_rate": float(comparison["tcf_exact_period_present_top10"].mean()), "tcf_exact_period_present_top10_count": int(comparison["tcf_exact_period_present_top10"].sum()), "tcf_exact_period_below_rank1_rate": float(comparison["tcf_exact_period_below_rank1"].mean()), "tcf_exact_period_below_rank1_count": int(comparison["tcf_exact_period_below_rank1"].sum()), "tcf_exact_period_absent_top10_rate": float(comparison["tcf_exact_period_absent_top10"].mean()), "tcf_exact_period_absent_top10_count": int(comparison["tcf_exact_period_absent_top10"].sum()), "tcf_top10_candidate_only_vs_bls_exact_count": int(comparison["tcf_top10_candidate_only_vs_bls_exact"].sum()), "median_tcf_exact_period_rank_top10": float(exact_ranks.median()) if not exact_ranks.empty else None, "weak_200ppm_bls_recovery_rate": float(weak_regime["bls_recovered"].mean()), "weak_200ppm_tcf_recovery_rate": float(weak_regime["tcf_recovered"].mean()), "weak_200ppm_union_recovery_rate": float(weak_regime["either_recovered"].mean()), "weak_200ppm_bls_exact_recovery_rate": float(weak_regime["bls_exact_recovered"].mean()), "weak_200ppm_tcf_exact_recovery_rate": float(weak_regime["tcf_exact_recovered"].mean()), "weak_200ppm_exact_union_recovery_rate": float(weak_regime["exact_either_recovered"].mean()), "weak_200ppm_tcf_exact_period_present_top10_rate": float(weak_regime["tcf_exact_period_present_top10"].mean())}
    cases = {"tcf_only": comparison[comparison["tcf_only"]], "bls_only": comparison[comparison["bls_only"]], "neither": comparison[comparison["neither_recovered"]], "exact_tcf_only": comparison[comparison["exact_tcf_only"]], "exact_bls_only": comparison[comparison["exact_bls_only"]], "exact_neither": comparison[comparison["exact_neither_recovered"]], "tcf_exact_below_rank1": comparison[comparison["tcf_exact_period_below_rank1"]], "tcf_exact_absent_top10": comparison[comparison["tcf_exact_period_absent_top10"]], "tcf_top10_candidate_only": comparison[comparison["tcf_top10_candidate_only_vs_bls_exact"]]}
    return comparison, by_depth, by_duration, by_period, summary, cases

def main():
    metrics_dir = OUTPUT_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    comparison, by_depth, by_duration, by_period, summary, cases = run_comparison()
    comparison.to_csv(metrics_dir / "kic_11904151_q5_bls_tcf_case_comparison.csv", index=False)
    by_depth.to_csv(metrics_dir / "kic_11904151_q5_bls_tcf_by_depth.csv", index=False)
    by_duration.to_csv(metrics_dir / "kic_11904151_q5_bls_tcf_by_duration.csv", index=False)
    by_period.to_csv(metrics_dir / "kic_11904151_q5_bls_tcf_by_period.csv", index=False)
    cases["tcf_only"].to_csv(metrics_dir / "kic_11904151_q5_tcf_only_recoveries.csv", index=False)
    cases["bls_only"].to_csv(metrics_dir / "kic_11904151_q5_bls_only_recoveries.csv", index=False)
    cases["neither"].to_csv(metrics_dir / "kic_11904151_q5_neither_recovered.csv", index=False)
    cases["exact_tcf_only"].to_csv(metrics_dir / "kic_11904151_q5_exact_tcf_only_recoveries.csv", index=False)
    cases["exact_bls_only"].to_csv(metrics_dir / "kic_11904151_q5_exact_bls_only_recoveries.csv", index=False)
    cases["exact_neither"].to_csv(metrics_dir / "kic_11904151_q5_exact_neither_recovered.csv", index=False)
    cases["tcf_exact_below_rank1"].to_csv(metrics_dir / "kic_11904151_q5_tcf_exact_period_below_rank1.csv", index=False)
    cases["tcf_exact_absent_top10"].to_csv(metrics_dir / "kic_11904151_q5_tcf_exact_period_absent_top10.csv", index=False)
    cases["tcf_top10_candidate_only"].to_csv(metrics_dir / "kic_11904151_q5_tcf_top10_candidate_only_vs_bls_exact.csv", index=False)
    summary_path = metrics_dir / "kic_11904151_q5_bls_tcf_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    print("\nHarmonic-aware rank-1 comparison:\n")
    print(f"BLS recovery rate: {summary['bls_recovery_rate']:.3f}")
    print(f"TCF recovery rate: {summary['tcf_recovery_rate']:.3f}")
    print(f"Combined recovery rate: {summary['union_recovery_rate']:.3f}")
    print(f"TCF-only recoveries: {summary['tcf_additional_recoveries']}")
    print(f"BLS-only recoveries: {summary['bls_additional_recoveries']}")
    print("\nExact-period rank-1 comparison:\n")
    print(f"BLS exact-period recovery rate: {summary['bls_exact_period_recovery_rate']:.3f}")
    print(f"TCF exact-period recovery rate: {summary['tcf_exact_period_recovery_rate']:.3f}")
    print(f"Exact-period combined recovery rate: {summary['exact_union_recovery_rate']:.3f}")
    print(f"Exact TCF-only recoveries: {summary['exact_tcf_additional_recoveries']}")
    print(f"Exact BLS-only recoveries: {summary['exact_bls_additional_recoveries']}")
    print("\nTCF top-10 diagnostic:\n")
    print(f"Exact injected period present in top 10: {summary['tcf_exact_period_present_top10_rate']:.3f} ({summary['tcf_exact_period_present_top10_count']}/{summary['injection_count']})")
    print(f"Exact period present below rank 1: {summary['tcf_exact_period_below_rank1_rate']:.3f} ({summary['tcf_exact_period_below_rank1_count']}/{summary['injection_count']})")
    print(f"Exact period absent from top 10: {summary['tcf_exact_period_absent_top10_rate']:.3f} ({summary['tcf_exact_period_absent_top10_count']}/{summary['injection_count']})")
    print(f"Median exact-period rank when present: {summary['median_tcf_exact_period_rank_top10']}")
    print(f"Top-10 TCF candidates where BLS missed exact period: {summary['tcf_top10_candidate_only_vs_bls_exact_count']}")
    print("\nExact-period rank distribution:\n")
    exact_ranks = comparison["tcf_exact_period_rank_top10"].dropna().astype(int)
    if exact_ranks.empty:
        print("No exact injected periods appeared in the top 10.")
    else:
        print(exact_ranks.value_counts().sort_index().to_string())
    display_columns = ["injected_depth", "injected_duration_hours", "injected_period_days", "epoch_phase_fraction", "bls_recovered_period_days", "bls_recovered_score", "tcf_recovered_period_days", "tcf_recovered_score", "tcf_exact_period_rank_top10", "tcf_half_period_rank_top10", "tcf_double_period_rank_top10", "tcf_triple_period_rank_top10"]
    print("\nCases where the exact TCF period exists below rank 1:\n")
    print(cases["tcf_exact_below_rank1"][display_columns].to_string(index=False))
    print("\nCases where the exact TCF period is absent from the top 10:\n")
    print(cases["tcf_exact_absent_top10"][display_columns].to_string(index=False))
    print("\nTCF top-10 candidates where BLS missed the exact period:\n")
    print(cases["tcf_top10_candidate_only"][display_columns].to_string(index=False))
    print("\nMost frequent rank-1 TCF periods:\n")
    print(comparison["tcf_recovered_period_days"].round(6).value_counts().head(10).to_string())
    print(f"\nComparison summary: {summary_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())