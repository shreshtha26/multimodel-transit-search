"""Characterize injection cases still missed by all saved methods after the GP branch."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_ID = "11904151"
QUARTER = 5
INPUT_PATH = PROJECT_ROOT / "outputs/experiments/gp_recovery_overlap/metrics/kic_11904151_q5_gp_recovery_overlap.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/gp_recovery_overlap"

def default_settings():
    return SimpleNamespace(target_id=TARGET_ID, quarter=QUARTER, input_path=INPUT_PATH, output_dir=OUTPUT_DIR)

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

def detector_period_ratios(frame):
    result = frame.copy()
    period = result["injected_period_days"].to_numpy(dtype=float)
    detector_columns = ["raw_bls_recovered_period_days", "existing_tcf_recovered_period_days", "kalman_bls_recovered_period_days", "kalman_tcf_recovered_period_days", "gp_bls_recovered_period_days", "gp_tcf_recovered_period_days"]
    for column in detector_columns:
        if column in result.columns:
            result[column.replace("_period_days", "_period_ratio")] = result[column].to_numpy(dtype=float) / period
    return result

def summarize_bins(frame, column):
    return frame.groupby(column, as_index=False).agg(miss_count=("all_six_union", "size"), median_gp_depth_retention_fraction=("gp_depth_retention_fraction", "median"), median_gp_snr_retention_fraction=("gp_snr_retention_fraction", "median"), median_kalman_depth_retention_fraction=("kalman_depth_retention_fraction", "median"), median_kalman_snr_retention_fraction=("kalman_snr_retention_fraction", "median"))

def run_analysis(args):
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Recovery-overlap table does not exist: {input_path}. Run scripts/compare_gp_recovery_overlap.py first.")
    frame = pd.read_csv(input_path)
    required = {"all_six_union", "injected_period_days", "injected_duration_hours", "injected_depth", "epoch_phase_fraction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Recovery-overlap table is missing columns: {sorted(missing)}")
    misses = frame.loc[~frame["all_six_union"].astype(bool)].copy()
    misses = detector_period_ratios(misses)
    total = int(len(frame))
    miss_count = int(len(misses))
    by_depth = summarize_bins(misses, "injected_depth") if miss_count else pd.DataFrame()
    by_duration = summarize_bins(misses, "injected_duration_hours") if miss_count else pd.DataFrame()
    by_period = summarize_bins(misses, "injected_period_days") if miss_count else pd.DataFrame()
    method_columns = [column for column in ("raw_bls_recovered", "existing_tcf_recovered", "kalman_bls_recovered", "kalman_tcf_recovered", "gp_bls_recovered", "gp_tcf_recovered") if column in frame.columns]
    summary = {"target_id": str(args.target_id), "quarter": int(args.quarter), "input_path": str(input_path), "injection_count": total, "all_method_miss_count": miss_count, "all_method_miss_fraction": float(miss_count / total) if total else float("nan"), "miss_period_values": sorted(float(value) for value in misses["injected_period_days"].unique()) if miss_count else [], "miss_duration_values_hours": sorted(float(value) for value in misses["injected_duration_hours"].unique()) if miss_count else [], "miss_depth_values": sorted(float(value) for value in misses["injected_depth"].unique()) if miss_count else [], "miss_epoch_phase_values": sorted(float(value) for value in misses["epoch_phase_fraction"].unique()) if miss_count else [], "method_columns": method_columns, "median_gp_depth_retention_fraction_for_misses": float(misses["gp_depth_retention_fraction"].median()) if miss_count and "gp_depth_retention_fraction" in misses.columns else float("nan"), "median_gp_snr_retention_fraction_for_misses": float(misses["gp_snr_retention_fraction"].median()) if miss_count and "gp_snr_retention_fraction" in misses.columns else float("nan"), "interpretation": "The remaining misses are cases where every saved method fails after harmonic-aware matching and calibrated detector thresholds. High GP retention in these rows indicates candidate-generation/scoring failure rather than GP transit erasure."}
    return misses, by_depth, by_duration, by_period, summary

def main(args=None):
    args = args or default_settings()
    metrics_dir = Path(args.output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"kic_{str(args.target_id).replace('KIC', '').strip()}_q{args.quarter}"
    misses, by_depth, by_duration, by_period, summary = run_analysis(args)
    misses_path = metrics_dir / f"{prefix}_gp_remaining_all_method_misses.csv"
    depth_path = metrics_dir / f"{prefix}_gp_remaining_misses_by_depth.csv"
    duration_path = metrics_dir / f"{prefix}_gp_remaining_misses_by_duration.csv"
    period_path = metrics_dir / f"{prefix}_gp_remaining_misses_by_period.csv"
    summary_path = metrics_dir / f"{prefix}_gp_remaining_all_method_misses_summary.json"
    misses.to_csv(misses_path, index=False)
    by_depth.to_csv(depth_path, index=False)
    by_duration.to_csv(duration_path, index=False)
    by_period.to_csv(period_path, index=False)
    summary_path.write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    print(f"Remaining all-method misses: {misses_path}")
    print(f"Remaining miss summary: {summary_path}")
    print(json.dumps(json_ready(summary), indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
