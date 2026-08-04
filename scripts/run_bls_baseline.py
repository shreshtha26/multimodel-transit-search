"""Run the first PDCSAP plus BLS baseline."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.detection.bls import default_duration_grid, default_period_grid, period_match_fraction, run_bls
from adaptive_transit.injections.synthetic import inject_periodic_box_transit
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve
from adaptive_transit.transit_models.periodic import transit_center_times

TARGET_ID = "11904151"
QUARTER = 5
OUTPUT_DIR = Path("outputs/experiments/bls_baseline")

def default_settings():
    return SimpleNamespace(
        target_id=TARGET_ID,
        quarter=QUARTER,
        output_dir=OUTPUT_DIR,
        quality_policy="default",
        require_finite_flux_error=False,
        test_fraction=0.20,
        injection_period_days=5.0,
        injection_epoch_offset_days=1.0,
        injection_duration_hours=4.0,
        injection_depth=0.001,
        min_period_days=1.0,
        max_period_days=15.0,
        n_periods=1000,
        min_duration_hours=1.5,
        max_duration_hours=10.0,
        n_durations=8,
        objective="snr",
        top_k=10,
        period_match_tolerance_fraction=0.02,
    )

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
    error = regular["flux_error"].to_numpy(dtype=float) / float(median_flux)
    error[~regular["usable"].to_numpy(dtype=bool)] = np.nan
    return error

def run_experiment(args):
    light_curve = load_kepler_pdcsap(args.target_id, args.quarter)
    regular, preprocessing = preprocess_pdcsap_light_curve(
        light_curve.to_dataframe(),
        quality_policy=args.quality_policy,
        require_finite_flux_error=args.require_finite_flux_error,
        normalization_fit_fraction=1.0 - args.test_fraction,
    )
    time = regular["time"].to_numpy(dtype=float)
    flux = regular["normalized_flux"].to_numpy(dtype=float)
    flux_error = normalized_flux_error(regular, preprocessing.median_flux)
    finite_time = time[np.isfinite(time) & np.isfinite(flux)]
    if finite_time.size == 0:
        raise ValueError("No finite normalized flux samples are available for BLS.")
    epoch = float(np.min(finite_time) + args.injection_epoch_offset_days)
    duration_days = float(args.injection_duration_hours) / 24.0
    injected_flux, template, in_transit = inject_periodic_box_transit(time, flux, args.injection_period_days, epoch, duration_days, args.injection_depth)
    period_grid = default_period_grid(time, min_period_days=args.min_period_days, max_period_days=args.max_period_days, n_periods=args.n_periods)
    duration_grid = default_duration_grid(args.min_duration_hours, args.max_duration_hours, args.n_durations)
    injected_result = run_bls(time, injected_flux, flux_error, period_grid, duration_grid, objective=args.objective, top_k=args.top_k)
    null_result = run_bls(time, flux, flux_error, period_grid, duration_grid, objective=args.objective, top_k=args.top_k)
    injected_summary = dict(injected_result["summary"])
    null_summary = dict(null_result["summary"])
    period_error = period_match_fraction(injected_summary["period"], args.injection_period_days)
    centers = transit_center_times(time, args.injection_period_days, epoch, duration_days)
    summary = {
        "target_id": str(args.target_id),
        "quarter": int(args.quarter),
        "quality_policy": args.quality_policy,
        "injected_period_days": float(args.injection_period_days),
        "injected_epoch_days": epoch,
        "injected_duration_hours": float(args.injection_duration_hours),
        "injected_depth": float(args.injection_depth),
        "observable_transit_count": int(len(centers)),
        "in_transit_observation_count": int(np.isfinite(flux[in_transit]).sum()),
        "recovered_period_days": float(injected_summary["period"]),
        "recovered_epoch_days": float(injected_summary["transit_time"]),
        "recovered_duration_hours": float(injected_summary["duration"]) * 24.0,
        "recovered_depth": float(injected_summary["depth"]),
        "recovered_power": float(injected_summary["power"]),
        "period_error_fraction": float(period_error),
        "period_match_tolerance_fraction": float(args.period_match_tolerance_fraction),
        "period_matched": bool(np.isfinite(period_error) and period_error <= args.period_match_tolerance_fraction),
        "null_max_power": float(null_summary["power"]),
        "injected_beats_null": bool(float(injected_summary["power"]) > float(null_summary["power"])),
        "n_observations": int(injected_summary["n_observations"]),
        "n_periods": int(injected_summary["n_periods"]),
        "n_durations": int(injected_summary["n_durations"]),
    }
    return regular, template, injected_result, null_result, summary

def main(args=None):
    args = args or default_settings()
    metrics_dir = Path(args.output_dir) / "metrics"
    processed_dir = Path(args.output_dir) / "processed"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    regular, template, injected_result, null_result, summary = run_experiment(args)
    prefix = f"kic_{str(args.target_id).replace('KIC', '').strip()}_q{args.quarter}"
    injected_result["periodogram"].to_csv(metrics_dir / f"{prefix}_bls_periodogram.csv", index=False)
    injected_result["top_peaks"].to_csv(metrics_dir / f"{prefix}_bls_top_peaks.csv", index=False)
    null_result["top_peaks"].to_csv(metrics_dir / f"{prefix}_bls_null_top_peaks.csv", index=False)
    injected_result["periodogram"].to_parquet(processed_dir / f"{prefix}_bls_periodogram.parquet", index=False)
    regular.assign(injected_template=template).to_parquet(processed_dir / f"{prefix}_bls_input_light_curve.parquet", index=False)
    (metrics_dir / f"{prefix}_bls_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    print(f"BLS summary: {metrics_dir / f'{prefix}_bls_summary.json'}")
    print(f"BLS periodogram: {metrics_dir / f'{prefix}_bls_periodogram.csv'}")
    print(f"Recovered period: {summary['recovered_period_days']:.6g} days")
    print(f"Period matched: {summary['period_matched']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
