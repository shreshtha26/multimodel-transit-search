"""Run a periodic BLS injection-recovery grid at calibrated FAP thresholds."""
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.detection.bls import default_duration_grid, default_period_grid, period_match_fraction, run_bls
from adaptive_transit.injections.synthetic import inject_periodic_box_transit
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve
from adaptive_transit.transit_models.periodic import transit_center_times

TARGET_ID = "11904151"
QUARTER = 5
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/bls_injection_grid"
FAP_THRESHOLD_PATH = PROJECT_ROOT / "outputs/experiments/bls_baseline/metrics/kic_11904151_q5_bls_fap_thresholds.csv"


def default_settings():
    return SimpleNamespace(target_id=TARGET_ID, quarter=QUARTER, output_dir=OUTPUT_DIR, fap_threshold_path=FAP_THRESHOLD_PATH, quality_policy="default", require_finite_flux_error=False, test_fraction=0.20, injection_period_grid=(2.0, 5.0, 10.0), injection_duration_hours_grid=(2.0, 4.0, 8.0), injection_depth_grid=(0.0002, 0.0005, 0.001), epoch_phase_fraction_grid=(0.15, 0.45, 0.75), min_period_days=1.0, max_period_days=15.0, n_periods=1000, min_duration_hours=1.5, max_duration_hours=10.0, n_durations=8, objective="snr", top_k=5, period_match_tolerance_fraction=0.02)

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

def normalized_flux_error(regular, median_flux):
    error = regular["flux_error"].to_numpy() / float(median_flux)
    error[~regular["usable"].to_numpy()] = np.nan
    return error

def load_fap_thresholds(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FAP threshold file does not exist: {path}")
    thresholds = pd.read_csv(path)
    required_columns = {"fap_level", "power_threshold"}
    missing_columns = required_columns.difference(thresholds.columns)
    if missing_columns:
        raise ValueError(f"FAP threshold file is missing columns: {sorted(missing_columns)}")
    threshold_map = {float(row["fap_level"]): float(row["power_threshold"]) for _, row in thresholds.iterrows()}
    if 0.01 not in threshold_map:
        raise ValueError("The threshold file does not contain the 1% FAP threshold.")
    if 0.001 not in threshold_map:
        raise ValueError("The threshold file does not contain the 0.1% FAP threshold.")
    return thresholds, threshold_map

def run_one_injection(time, flux, flux_error, period_grid, duration_grid, injected_period, injected_duration_hours, injected_depth, epoch_phase_fraction, threshold_map, args):
    finite_time = time[np.isfinite(time) & np.isfinite(flux)]
    epoch = float(np.min(finite_time) + epoch_phase_fraction * injected_period)
    injected_duration_days = float(injected_duration_hours) / 24.0
    injected_flux, template, in_transit = inject_periodic_box_transit(time, flux, injected_period, epoch, injected_duration_days, injected_depth)
    result = run_bls(time, injected_flux, flux_error, period_grid, duration_grid, objective=args.objective, top_k=args.top_k)
    best = dict(result["summary"])
    recovered_period = float(best["period"])
    recovered_duration_hours = float(best["duration"]) * 24.0
    recovered_depth = float(best["depth"])
    recovered_power = float(best["power"])
    period_error = period_match_fraction(recovered_period, injected_period)
    period_matched = bool(np.isfinite(period_error) and period_error <= args.period_match_tolerance_fraction)
    passes_fap_1_percent = bool(recovered_power >= threshold_map[0.01])
    passes_fap_0_1_percent = bool(recovered_power >= threshold_map[0.001])
    centers = transit_center_times(time, injected_period, epoch, injected_duration_days)
    return {"injected_period_days": float(injected_period), "injected_epoch_days": epoch, "epoch_phase_fraction": float(epoch_phase_fraction), "injected_duration_hours": float(injected_duration_hours), "injected_depth": float(injected_depth), "observable_transit_count": int(len(centers)), "in_transit_observation_count": int(np.isfinite(flux[in_transit]).sum()), "recovered_period_days": recovered_period, "recovered_epoch_days": float(best["transit_time"]), "recovered_duration_hours": recovered_duration_hours, "recovered_depth": recovered_depth, "recovered_power": recovered_power, "period_error_fraction": float(period_error), "period_matched": period_matched, "depth_retention_fraction": float(recovered_depth / injected_depth), "duration_recovery_fraction": float(recovered_duration_hours / injected_duration_hours), "fap_1_percent_threshold": float(threshold_map[0.01]), "fap_0_1_percent_threshold": float(threshold_map[0.001]), "passes_fap_1_percent": passes_fap_1_percent, "passes_fap_0_1_percent": passes_fap_0_1_percent, "recovered_at_fap_1_percent": bool(period_matched and passes_fap_1_percent), "recovered_at_fap_0_1_percent": bool(period_matched and passes_fap_0_1_percent)}

def grouped_recovery(results, column):
    return results.groupby(column, as_index=False).agg(injection_count=("period_matched", "size"), period_match_rate=("period_matched", "mean"), detection_rate_fap_1_percent=("passes_fap_1_percent", "mean"), recovery_rate_fap_1_percent=("recovered_at_fap_1_percent", "mean"), detection_rate_fap_0_1_percent=("passes_fap_0_1_percent", "mean"), recovery_rate_fap_0_1_percent=("recovered_at_fap_0_1_percent", "mean"), median_period_error_fraction=("period_error_fraction", "median"), median_depth_retention_fraction=("depth_retention_fraction", "median"), median_duration_recovery_fraction=("duration_recovery_fraction", "median"))

def run_experiment(args):
    light_curve = load_kepler_pdcsap(args.target_id, args.quarter)
    regular, preprocessing = preprocess_pdcsap_light_curve(light_curve.to_dataframe(), quality_policy=args.quality_policy, require_finite_flux_error=args.require_finite_flux_error, normalization_fit_fraction=1.0 - args.test_fraction)
    time = regular["time"].to_numpy()
    flux = regular["normalized_flux"].to_numpy()
    flux_error = normalized_flux_error(regular, preprocessing.median_flux)
    finite_time = time[np.isfinite(time) & np.isfinite(flux)]
    if finite_time.size == 0:
        raise ValueError("No finite normalized flux samples are available for BLS.")
    period_grid = default_period_grid(time, min_period_days=args.min_period_days, max_period_days=args.max_period_days, n_periods=args.n_periods)
    duration_grid = default_duration_grid(args.min_duration_hours, args.max_duration_hours, args.n_durations)
    thresholds, threshold_map = load_fap_thresholds(args.fap_threshold_path)
    combinations = list(product(args.injection_period_grid, args.injection_duration_hours_grid, args.injection_depth_grid, args.epoch_phase_fraction_grid))
    rows = []
    for injected_period, injected_duration_hours, injected_depth, epoch_phase_fraction in tqdm(combinations, desc="BLS injection grid"):
        row = run_one_injection(time, flux, flux_error, period_grid, duration_grid, injected_period, injected_duration_hours, injected_depth, epoch_phase_fraction, threshold_map, args)
        rows.append(row)
    results = pd.DataFrame(rows)
    by_depth = grouped_recovery(results, "injected_depth")
    by_duration = grouped_recovery(results, "injected_duration_hours")
    by_period = grouped_recovery(results, "injected_period_days")
    summary = {"target_id": str(args.target_id), "quarter": int(args.quarter), "quality_policy": args.quality_policy, "injection_count": int(len(results)), "period_match_rate": float(results["period_matched"].mean()), "detection_rate_fap_1_percent": float(results["passes_fap_1_percent"].mean()), "recovery_rate_fap_1_percent": float(results["recovered_at_fap_1_percent"].mean()), "detection_rate_fap_0_1_percent": float(results["passes_fap_0_1_percent"].mean()), "recovery_rate_fap_0_1_percent": float(results["recovered_at_fap_0_1_percent"].mean()), "median_period_error_fraction": float(results["period_error_fraction"].median()), "median_depth_retention_fraction": float(results["depth_retention_fraction"].median()), "median_duration_recovery_fraction": float(results["duration_recovery_fraction"].median()), "fap_1_percent_threshold": float(threshold_map[0.01]), "fap_0_1_percent_threshold": float(threshold_map[0.001])}
    return regular, thresholds, results, by_depth, by_duration, by_period, summary

def main(args=None):
    args = args or default_settings()
    metrics_dir = Path(args.output_dir) / "metrics"
    processed_dir = Path(args.output_dir) / "processed"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    regular, thresholds, results, by_depth, by_duration, by_period, summary = run_experiment(args)
    prefix = f"kic_{str(args.target_id).replace('KIC', '').strip()}_q{args.quarter}"
    results.to_csv(metrics_dir / f"{prefix}_bls_injection_grid.csv", index=False)
    by_depth.to_csv(metrics_dir / f"{prefix}_bls_recovery_by_depth.csv", index=False)
    by_duration.to_csv(metrics_dir / f"{prefix}_bls_recovery_by_duration.csv", index=False)
    by_period.to_csv(metrics_dir / f"{prefix}_bls_recovery_by_period.csv", index=False)
    thresholds.to_csv(metrics_dir / f"{prefix}_bls_used_fap_thresholds.csv", index=False)
    regular.to_parquet(processed_dir / f"{prefix}_bls_injection_grid_input.parquet", index=False)
    (metrics_dir / f"{prefix}_bls_injection_grid_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    print(f"Injection results: {metrics_dir / f'{prefix}_bls_injection_grid.csv'}")
    print(f"Injection summary: {metrics_dir / f'{prefix}_bls_injection_grid_summary.json'}")
    print(f"Period match rate: {summary['period_match_rate']:.3f}")
    print(f"Recovery rate at 1% FAP: {summary['recovery_rate_fap_1_percent']:.3f}")
    print(f"Recovery rate at 0.1% FAP: {summary['recovery_rate_fap_0_1_percent']:.3f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())