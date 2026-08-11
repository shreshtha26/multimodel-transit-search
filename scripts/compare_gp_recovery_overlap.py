"""Compare row-level recovery overlap after adding the GP branch."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLS_PATH = PROJECT_ROOT / "outputs/experiments/bls_injection_grid/metrics/kic_11904151_q5_bls_injection_grid.csv"
TCF_PATH = PROJECT_ROOT / "outputs/experiments/tcf_injection_grid/metrics/kic_11904151_q5_tcf_injection_grid.csv"
KALMAN_PATH = PROJECT_ROOT / "outputs/experiments/kalman_injection_grid/metrics/kic_11904151_q5_kalman_injection_grid.csv"
GP_PATH = PROJECT_ROOT / "outputs/experiments/gp_injection_grid/metrics/kic_11904151_q5_gp_injection_grid.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/gp_recovery_overlap"
INJECTION_KEYS = ["injected_period_days", "injected_duration_hours", "injected_depth", "epoch_phase_fraction"]
METHOD_COLUMNS = {"raw_bls": "raw_bls_recovered", "existing_tcf": "existing_tcf_recovered", "kalman_bls": "kalman_bls_recovered", "kalman_tcf": "kalman_tcf_recovered", "gp_bls": "gp_bls_recovered", "gp_tcf": "gp_tcf_recovered"}

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

def require_columns(frame, columns, name):
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} result file is missing columns: {sorted(missing)}")

def check_unique_injection_grid(frame, name):
    duplicated = frame.duplicated(INJECTION_KEYS)
    if duplicated.any():
        rows = frame.loc[duplicated, INJECTION_KEYS].head().to_dict(orient="records")
        raise ValueError(f"{name} result file has duplicate injection identities: {rows}")

def load_matched_results():
    for path in (BLS_PATH, TCF_PATH, KALMAN_PATH, GP_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required injection result file does not exist: {path}")
    bls = pd.read_csv(BLS_PATH)
    tcf = pd.read_csv(TCF_PATH)
    kalman = pd.read_csv(KALMAN_PATH)
    gp = pd.read_csv(GP_PATH)
    bls_columns = INJECTION_KEYS + ["injected_epoch_days", "observable_transit_count", "in_transit_observation_count", "recovered_period_days", "recovered_power", "period_matched", "passes_fap_1_percent", "recovered_at_fap_1_percent"]
    tcf_columns = INJECTION_KEYS + ["recovered_period_days", "recovered_score", "period_matched", "passes_fap_1_percent", "recovered_at_fap_1_percent", "exact_period_present_top10", "exact_period_rank_top10"]
    kalman_columns = INJECTION_KEYS + ["depth_retention_fraction", "snr_retention_fraction", "kalman_bls_recovered_period_days", "kalman_bls_recovered_power", "kalman_bls_recovered_at_fap_1_percent", "kalman_tcf_recovered_period_days", "kalman_tcf_recovered_score", "kalman_tcf_recovered_at_fap_1_percent"]
    gp_columns = INJECTION_KEYS + ["depth_retention_fraction", "snr_retention_fraction", "gp_length_scale_days", "gp_bls_recovered_period_days", "gp_bls_recovered_power", "gp_bls_recovered_at_fap_1_percent", "gp_tcf_recovered_period_days", "gp_tcf_recovered_score", "gp_tcf_recovered_at_fap_1_percent", "gp_tcf_exact_period_rank_top10"]
    require_columns(bls, bls_columns, "BLS")
    require_columns(tcf, tcf_columns, "TCF")
    require_columns(kalman, kalman_columns, "Kalman")
    require_columns(gp, gp_columns, "GP")
    check_unique_injection_grid(bls, "BLS")
    check_unique_injection_grid(tcf, "TCF")
    check_unique_injection_grid(kalman, "Kalman")
    check_unique_injection_grid(gp, "GP")
    bls = bls[bls_columns].rename(columns={"recovered_period_days": "raw_bls_recovered_period_days", "recovered_power": "raw_bls_recovered_score", "period_matched": "raw_bls_period_matched", "passes_fap_1_percent": "raw_bls_passes_fap_1_percent", "recovered_at_fap_1_percent": "raw_bls_recovered"})
    tcf = tcf[tcf_columns].rename(columns={"recovered_period_days": "existing_tcf_recovered_period_days", "recovered_score": "existing_tcf_recovered_score", "period_matched": "existing_tcf_period_matched", "passes_fap_1_percent": "existing_tcf_passes_fap_1_percent", "recovered_at_fap_1_percent": "existing_tcf_recovered", "exact_period_present_top10": "existing_tcf_exact_period_present_top10", "exact_period_rank_top10": "existing_tcf_exact_period_rank_top10"})
    kalman = kalman[kalman_columns].rename(columns={"depth_retention_fraction": "kalman_depth_retention_fraction", "snr_retention_fraction": "kalman_snr_retention_fraction", "kalman_bls_recovered_at_fap_1_percent": "kalman_bls_recovered", "kalman_tcf_recovered_at_fap_1_percent": "kalman_tcf_recovered"})
    gp = gp[gp_columns].rename(columns={"depth_retention_fraction": "gp_depth_retention_fraction", "snr_retention_fraction": "gp_snr_retention_fraction", "gp_bls_recovered_at_fap_1_percent": "gp_bls_recovered", "gp_tcf_recovered_at_fap_1_percent": "gp_tcf_recovered"})
    comparison = bls.merge(tcf, on=INJECTION_KEYS, how="inner", validate="one_to_one").merge(kalman, on=INJECTION_KEYS, how="inner", validate="one_to_one").merge(gp, on=INJECTION_KEYS, how="inner", validate="one_to_one")
    if len(comparison) != len(bls) or len(comparison) != len(tcf) or len(comparison) != len(kalman) or len(comparison) != len(gp):
        raise ValueError("The BLS, TCF, Kalman, and GP injection grids do not contain the same exact injection identities.")
    boolean_columns = ["raw_bls_period_matched", "raw_bls_passes_fap_1_percent", "raw_bls_recovered", "existing_tcf_period_matched", "existing_tcf_passes_fap_1_percent", "existing_tcf_recovered", "existing_tcf_exact_period_present_top10", "kalman_bls_recovered", "kalman_tcf_recovered", "gp_bls_recovered", "gp_tcf_recovered"]
    for column in boolean_columns:
        comparison[column] = boolean_series(comparison[column])
    for column in ["existing_tcf_exact_period_rank_top10", "kalman_depth_retention_fraction", "kalman_snr_retention_fraction", "gp_depth_retention_fraction", "gp_snr_retention_fraction", "gp_length_scale_days", "gp_tcf_exact_period_rank_top10"]:
        comparison[column] = pd.to_numeric(comparison[column], errors="coerce")
    add_overlap_columns(comparison)
    return comparison

def add_overlap_columns(comparison):
    comparison["existing_bls_tcf_union"] = comparison["raw_bls_recovered"] | comparison["existing_tcf_recovered"]
    comparison["existing_without_gp_union"] = comparison["raw_bls_recovered"] | comparison["existing_tcf_recovered"] | comparison["kalman_bls_recovered"] | comparison["kalman_tcf_recovered"]
    comparison["raw_bls_union_gp_tcf"] = comparison["raw_bls_recovered"] | comparison["gp_tcf_recovered"]
    comparison["existing_bls_tcf_union_gp_tcf"] = comparison["raw_bls_recovered"] | comparison["existing_tcf_recovered"] | comparison["gp_tcf_recovered"]
    comparison["existing_bls_tcf_kalman_tcf_union_gp_tcf"] = comparison["raw_bls_recovered"] | comparison["existing_tcf_recovered"] | comparison["kalman_tcf_recovered"] | comparison["gp_tcf_recovered"]
    comparison["all_six_union"] = comparison["raw_bls_recovered"] | comparison["existing_tcf_recovered"] | comparison["kalman_bls_recovered"] | comparison["kalman_tcf_recovered"] | comparison["gp_bls_recovered"] | comparison["gp_tcf_recovered"]
    comparison["gp_tcf_only_vs_raw_bls"] = comparison["gp_tcf_recovered"] & ~comparison["raw_bls_recovered"]
    comparison["raw_bls_only_vs_gp_tcf"] = comparison["raw_bls_recovered"] & ~comparison["gp_tcf_recovered"]
    comparison["gp_tcf_unique_vs_existing_without_gp"] = comparison["gp_tcf_recovered"] & ~comparison["existing_without_gp_union"]
    comparison["gp_bls_unique_vs_existing_without_gp"] = comparison["gp_bls_recovered"] & ~comparison["existing_without_gp_union"] & ~comparison["gp_tcf_recovered"]

def summarize_ordered_pairs(comparison):
    rows = []
    methods = list(METHOD_COLUMNS.items())
    total = len(comparison)
    for first_name, first_column in methods:
        for second_name, second_column in methods:
            if first_name == second_name:
                continue
            first = comparison[first_column]
            second = comparison[second_column]
            both = first & second
            first_only = first & ~second
            second_only = second & ~first
            neither = ~first & ~second
            rows.append({"first_method": first_name, "second_method": second_name, "injection_count": int(total), "first_recovered_count": int(first.sum()), "second_recovered_count": int(second.sum()), "both_recovered_count": int(both.sum()), "first_only_count": int(first_only.sum()), "second_only_count": int(second_only.sum()), "neither_count": int(neither.sum()), "first_recovery_rate": float(first.mean()), "second_recovery_rate": float(second.mean()), "intersection_recovery_rate": float(both.mean()), "union_recovery_rate": float((first | second).mean()), "incremental_recoveries_contributed_by_second": int(second_only.sum()), "incremental_recovery_rate_contributed_by_second": float(second_only.mean())})
    return pd.DataFrame(rows)

def summarize_combinations(comparison):
    combinations = {"raw_bls": ["raw_bls_recovered"], "existing_tcf": ["existing_tcf_recovered"], "kalman_tcf": ["kalman_tcf_recovered"], "gp_tcf": ["gp_tcf_recovered"], "existing_bls_tcf_union": ["raw_bls_recovered", "existing_tcf_recovered"], "raw_bls_union_gp_tcf": ["raw_bls_recovered", "gp_tcf_recovered"], "existing_bls_tcf_union_gp_tcf": ["raw_bls_recovered", "existing_tcf_recovered", "gp_tcf_recovered"], "existing_bls_tcf_kalman_tcf_union_gp_tcf": ["raw_bls_recovered", "existing_tcf_recovered", "kalman_tcf_recovered", "gp_tcf_recovered"], "all_six_union": ["raw_bls_recovered", "existing_tcf_recovered", "kalman_bls_recovered", "kalman_tcf_recovered", "gp_bls_recovered", "gp_tcf_recovered"]}
    rows = []
    total = len(comparison)
    existing_without_gp = comparison["existing_without_gp_union"]
    for name, columns in combinations.items():
        recovered = pd.Series(False, index=comparison.index)
        for column in columns:
            recovered = recovered | comparison[column]
        rows.append({"combination": name, "methods": "+".join(columns), "injection_count": int(total), "recovered_count": int(recovered.sum()), "recovery_rate": float(recovered.mean()), "new_recoveries_over_existing_without_gp_count": int((recovered & ~existing_without_gp).sum()), "new_recoveries_over_existing_without_gp_rate": float((recovered & ~existing_without_gp).mean())})
    return pd.DataFrame(rows)

def grouped_recovery(comparison, column):
    return comparison.groupby(column, as_index=False).agg(injection_count=("raw_bls_recovered", "size"), raw_bls_recovery_rate=("raw_bls_recovered", "mean"), existing_tcf_recovery_rate=("existing_tcf_recovered", "mean"), kalman_tcf_recovery_rate=("kalman_tcf_recovered", "mean"), gp_bls_recovery_rate=("gp_bls_recovered", "mean"), gp_tcf_recovery_rate=("gp_tcf_recovered", "mean"), existing_without_gp_union_rate=("existing_without_gp_union", "mean"), all_six_union_rate=("all_six_union", "mean"), gp_tcf_only_vs_raw_bls_count=("gp_tcf_only_vs_raw_bls", "sum"), raw_bls_only_vs_gp_tcf_count=("raw_bls_only_vs_gp_tcf", "sum"), gp_tcf_unique_vs_existing_without_gp_count=("gp_tcf_unique_vs_existing_without_gp", "sum"), median_gp_depth_retention_fraction=("gp_depth_retention_fraction", "median"), median_gp_snr_retention_fraction=("gp_snr_retention_fraction", "median"), median_gp_length_scale_days=("gp_length_scale_days", "median"))

def summarize_retention_correlations(comparison):
    rows = []
    for retention_column in ["gp_depth_retention_fraction", "gp_snr_retention_fraction"]:
        for method_name, recovered_column in METHOD_COLUMNS.items():
            clean = comparison[[retention_column, recovered_column]].dropna()
            if clean.empty or clean[recovered_column].nunique() < 2:
                pearson = float("nan")
                spearman = float("nan")
            else:
                pearson = float(clean[retention_column].corr(clean[recovered_column].astype(float), method="pearson"))
                spearman = float(clean[retention_column].corr(clean[recovered_column].astype(float), method="spearman"))
            recovered = comparison.loc[comparison[recovered_column], retention_column].dropna()
            missed = comparison.loc[~comparison[recovered_column], retention_column].dropna()
            rows.append({"retention_metric": retention_column, "method": method_name, "pearson_correlation_with_recovery": pearson, "spearman_correlation_with_recovery": spearman, "median_when_recovered": float(recovered.median()) if not recovered.empty else float("nan"), "median_when_missed": float(missed.median()) if not missed.empty else float("nan")})
    return pd.DataFrame(rows)

def make_summary(comparison, pairwise, combinations, by_depth, by_duration, by_period, retention_correlations):
    total = len(comparison)
    unique_gp_tcf = comparison["gp_tcf_unique_vs_existing_without_gp"]
    unique_gp_bls = comparison["gp_bls_unique_vs_existing_without_gp"]
    summary = {"target_id": "11904151", "quarter": 5, "injection_count": int(total), "join_keys": list(INJECTION_KEYS), "input_files": {"raw_bls": str(BLS_PATH), "existing_tcf": str(TCF_PATH), "kalman": str(KALMAN_PATH), "gp": str(GP_PATH)}, "method_recovery_rates": {name: float(comparison[column].mean()) for name, column in METHOD_COLUMNS.items()}, "required_union_rates": {"raw_bls_union_gp_tcf": float(comparison["raw_bls_union_gp_tcf"].mean()), "existing_bls_tcf_union_gp_tcf": float(comparison["existing_bls_tcf_union_gp_tcf"].mean()), "existing_bls_tcf_kalman_tcf_union_gp_tcf": float(comparison["existing_bls_tcf_kalman_tcf_union_gp_tcf"].mean()), "all_six_union": float(comparison["all_six_union"].mean())}, "unique_recoveries": {"gp_tcf_unique_vs_existing_without_gp_count": int(unique_gp_tcf.sum()), "gp_tcf_unique_vs_existing_without_gp_fraction": float(unique_gp_tcf.mean()), "gp_bls_unique_vs_existing_without_gp_count": int(unique_gp_bls.sum()), "gp_bls_unique_vs_existing_without_gp_fraction": float(unique_gp_bls.mean())}, "pairwise_overlap": pairwise.to_dict(orient="records"), "combination_recovery": combinations.to_dict(orient="records"), "retention_correlations": retention_correlations.to_dict(orient="records"), "regime_recovery_by_depth": by_depth.to_dict(orient="records"), "regime_recovery_by_duration": by_duration.to_dict(orient="records"), "regime_recovery_by_period": by_period.to_dict(orient="records")}
    gp_only = int(comparison["gp_tcf_only_vs_raw_bls"].sum())
    raw_only = int(comparison["raw_bls_only_vs_gp_tcf"].sum())
    summary["raw_bls_vs_gp_tcf"] = {"both_recovered_count": int((comparison["raw_bls_recovered"] & comparison["gp_tcf_recovered"]).sum()), "gp_tcf_only_count": gp_only, "raw_bls_only_count": raw_only, "neither_count": int((~comparison["raw_bls_recovered"] & ~comparison["gp_tcf_recovered"]).sum()), "gp_tcf_only_fraction": float(gp_only / total), "raw_bls_only_fraction": float(raw_only / total)}
    return summary

def main():
    metrics_dir = OUTPUT_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    comparison = load_matched_results()
    pairwise = summarize_ordered_pairs(comparison)
    combinations = summarize_combinations(comparison)
    by_depth = grouped_recovery(comparison, "injected_depth")
    by_duration = grouped_recovery(comparison, "injected_duration_hours")
    by_period = grouped_recovery(comparison, "injected_period_days")
    retention_correlations = summarize_retention_correlations(comparison)
    summary = make_summary(comparison, pairwise, combinations, by_depth, by_duration, by_period, retention_correlations)
    prefix = "kic_11904151_q5"
    comparison.to_csv(metrics_dir / f"{prefix}_gp_recovery_overlap.csv", index=False)
    pairwise.to_csv(metrics_dir / f"{prefix}_gp_pairwise_overlap.csv", index=False)
    combinations.to_csv(metrics_dir / f"{prefix}_gp_combination_overlap.csv", index=False)
    by_depth.to_csv(metrics_dir / f"{prefix}_gp_overlap_by_depth.csv", index=False)
    by_duration.to_csv(metrics_dir / f"{prefix}_gp_overlap_by_duration.csv", index=False)
    by_period.to_csv(metrics_dir / f"{prefix}_gp_overlap_by_period.csv", index=False)
    retention_correlations.to_csv(metrics_dir / f"{prefix}_gp_retention_recovery_correlations.csv", index=False)
    comparison[comparison["gp_tcf_only_vs_raw_bls"]].to_csv(metrics_dir / f"{prefix}_gp_tcf_only_vs_raw_bls.csv", index=False)
    comparison[comparison["raw_bls_only_vs_gp_tcf"]].to_csv(metrics_dir / f"{prefix}_raw_bls_only_vs_gp_tcf.csv", index=False)
    comparison[comparison["gp_tcf_unique_vs_existing_without_gp"]].to_csv(metrics_dir / f"{prefix}_gp_tcf_unique_vs_existing_without_gp.csv", index=False)
    summary_path = metrics_dir / f"{prefix}_gp_recovery_overlap_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    total = summary["injection_count"]
    gp_unique = summary["unique_recoveries"]["gp_tcf_unique_vs_existing_without_gp_count"]
    print(f"raw/BLS union GP-TCF recovery: {summary['required_union_rates']['raw_bls_union_gp_tcf']:.3f}")
    print(f"existing BLS union TCF union GP-TCF recovery: {summary['required_union_rates']['existing_bls_tcf_union_gp_tcf']:.3f}")
    print(f"existing BLS union TCF union Kalman-TCF union GP-TCF recovery: {summary['required_union_rates']['existing_bls_tcf_kalman_tcf_union_gp_tcf']:.3f}")
    print(f"all six methods union recovery: {summary['required_union_rates']['all_six_union']:.3f}")
    print(f"unique GP-TCF recoveries beyond existing non-GP methods: {gp_unique}/{total} ({summary['unique_recoveries']['gp_tcf_unique_vs_existing_without_gp_fraction']:.3f})")
    print(f"summary: {summary_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
