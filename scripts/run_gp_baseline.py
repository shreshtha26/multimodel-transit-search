"""Run one smooth Gaussian Process residual baseline with BLS and TCF detectors."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.detection.bls import default_duration_grid as bls_duration_grid
from adaptive_transit.detection.bls import default_period_grid as bls_period_grid
from adaptive_transit.detection.bls import period_match_fraction as bls_period_error
from adaptive_transit.detection.bls import run_bls
from adaptive_transit.detection.tcf import default_duration_grid as tcf_duration_grid
from adaptive_transit.detection.tcf import default_period_grid as tcf_period_grid
from adaptive_transit.detection.tcf import period_match_fraction as tcf_period_error
from adaptive_transit.detection.tcf import run_tcf
from adaptive_transit.injections.synthetic import inject_periodic_box_transit
from adaptive_transit.noise_models.diagnostics import residual_diagnostics
from adaptive_transit.noise_models.gp import fit_smooth_gp_background
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve
from adaptive_transit.transit_models.periodic import transit_center_times
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_ID = "11904151"
QUARTER = 5
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/gp_baseline"

def default_settings():
    return SimpleNamespace(target_id=TARGET_ID, quarter=QUARTER, output_dir=OUTPUT_DIR, quality_policy="default", require_finite_flux_error=False, test_fraction=0.20, gp_max_train_points=512, gp_length_scale_days=3.0, gp_min_length_scale_days=1.0, gp_max_length_scale_days=30.0, gp_measurement_noise_fraction=0.20, gp_n_restarts_optimizer=0, gp_random_seed=123, injection_period_days=5.0, injection_epoch_offset_days=1.0, injection_duration_hours=4.0, injection_depth=0.001, min_period_days=1.0, max_period_days=15.0, bls_n_periods=1000, tcf_n_periods=10000, min_duration_hours=1.5, max_duration_hours=10.0, n_durations=8, bls_objective="snr", top_k=10, edge_width_cadences=0, min_edge_observations=4, min_transit_events=3, min_event_consistency_fraction=0.60, period_match_tolerance_fraction=0.02)

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

def robust_scale(values):
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        return float("nan")
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    return float(1.4826 * mad if mad > 0 else np.std(clean, ddof=1))

def periodic_depth_and_snr(values, in_transit):
    series = np.asarray(values, dtype=float)
    mask = np.asarray(in_transit, dtype=bool)
    finite_in = mask & np.isfinite(series)
    finite_out = ~mask & np.isfinite(series)
    if finite_in.sum() == 0 or finite_out.sum() < 3:
        return {"depth": float("nan"), "snr": float("nan"), "in_transit_count": int(finite_in.sum())}
    depth = float(np.median(series[finite_out]) - np.median(series[finite_in]))
    noise = robust_scale(series[finite_out])
    snr = float(depth / noise * np.sqrt(finite_in.sum())) if np.isfinite(noise) and noise > 0 else float("nan")
    return {"depth": depth, "snr": snr, "in_transit_count": int(finite_in.sum())}

def retention_summary(injected_flux, residuals, in_transit):
    before = periodic_depth_and_snr(injected_flux, in_transit)
    after = periodic_depth_and_snr(residuals, in_transit)
    before_depth = float(before["depth"])
    before_snr = float(before["snr"])
    after_depth = float(after["depth"])
    after_snr = float(after["snr"])
    return {"observed_depth_before_gp": before_depth, "gp_residual_depth": after_depth, "depth_retention_fraction": float(after_depth / before_depth) if before_depth != 0 else float("nan"), "local_snr_before_gp": before_snr, "local_snr_after_gp": after_snr, "snr_retention_fraction": float(after_snr / before_snr) if before_snr != 0 else float("nan"), "in_transit_observation_count": int(before["in_transit_count"])}

def fit_and_diagnose(time, values, args):
    fitted = fit_smooth_gp_background(time, values, max_train_points=args.gp_max_train_points, length_scale_days=args.gp_length_scale_days, min_length_scale_days=args.gp_min_length_scale_days, max_length_scale_days=args.gp_max_length_scale_days, measurement_noise_fraction=args.gp_measurement_noise_fraction, n_restarts_optimizer=args.gp_n_restarts_optimizer, random_seed=args.gp_random_seed)
    diagnostics = residual_diagnostics(fitted.residuals[fitted.usable_mask])
    return fitted, diagnostics

def run_detector_pair(time, residuals, args):
    bls_periods = bls_period_grid(time, min_period_days=args.min_period_days, max_period_days=args.max_period_days, n_periods=args.bls_n_periods)
    bls_durations = bls_duration_grid(args.min_duration_hours, args.max_duration_hours, args.n_durations)
    tcf_periods = tcf_period_grid(time, min_period_days=args.min_period_days, max_period_days=args.max_period_days, n_periods=args.tcf_n_periods)
    tcf_durations = tcf_duration_grid(args.min_duration_hours, args.max_duration_hours, args.n_durations)
    bls_result = run_bls(time, residuals, None, bls_periods, bls_durations, objective=args.bls_objective, top_k=args.top_k)
    tcf_result = run_tcf(time, residuals, tcf_periods, tcf_durations, edge_width_cadences=args.edge_width_cadences, min_edge_observations=args.min_edge_observations, min_transit_events=args.min_transit_events, min_event_consistency_fraction=args.min_event_consistency_fraction, top_k=args.top_k)
    return bls_result, tcf_result

def run_experiment(args):
    light_curve = load_kepler_pdcsap(args.target_id, args.quarter)
    regular, preprocessing = preprocess_pdcsap_light_curve(light_curve.to_dataframe(), quality_policy=args.quality_policy, require_finite_flux_error=args.require_finite_flux_error, normalization_fit_fraction=1.0 - args.test_fraction)
    time = regular["time"].to_numpy(dtype=float)
    flux = regular["normalized_flux"].to_numpy(dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    if finite.sum() < 24:
        raise ValueError("Insufficient finite light-curve observations.")
    epoch = float(np.min(time[finite]) + args.injection_epoch_offset_days)
    duration_days = float(args.injection_duration_hours) / 24.0
    injected_flux, template, in_transit = inject_periodic_box_transit(time, flux, args.injection_period_days, epoch, duration_days, args.injection_depth)
    original_model, original_diagnostics = fit_and_diagnose(time, flux, args)
    injected_model, injected_diagnostics = fit_and_diagnose(time, injected_flux, args)
    original_bls, original_tcf = run_detector_pair(time, original_model.residuals, args)
    injected_bls, injected_tcf = run_detector_pair(time, injected_model.residuals, args)
    bls_error = bls_period_error(injected_bls["summary"]["period"], args.injection_period_days)
    tcf_error = tcf_period_error(injected_tcf["summary"]["period"], args.injection_period_days)
    centers = transit_center_times(time, args.injection_period_days, epoch, duration_days)
    retention = retention_summary(injected_flux, injected_model.residuals, in_transit)
    summary = {"target_id": str(args.target_id), "quarter": int(args.quarter), "quality_policy": args.quality_policy, "model_name": "smooth_anchor_gp", "injected_period_days": float(args.injection_period_days), "injected_epoch_days": epoch, "injected_duration_hours": float(args.injection_duration_hours), "injected_depth": float(args.injection_depth), "observable_transit_count": int(len(centers)), **retention, "original_model": original_model.summary(), "injected_model": injected_model.summary(), "original_residual_diagnostics": original_diagnostics, "injected_residual_diagnostics": injected_diagnostics, "gp_bls_recovered_period_days": float(injected_bls["summary"]["period"]), "gp_bls_recovered_power": float(injected_bls["summary"]["power"]), "gp_bls_period_error_fraction": float(bls_error), "gp_bls_period_matched": bool(np.isfinite(bls_error) and bls_error <= args.period_match_tolerance_fraction), "gp_tcf_recovered_period_days": float(injected_tcf["summary"]["period"]), "gp_tcf_recovered_score": float(injected_tcf["summary"]["score"]), "gp_tcf_period_error_fraction": float(tcf_error), "gp_tcf_period_matched": bool(np.isfinite(tcf_error) and tcf_error <= args.period_match_tolerance_fraction), "original_gp_bls_best_period_days": float(original_bls["summary"]["period"]), "original_gp_bls_max_power": float(original_bls["summary"]["power"]), "original_gp_tcf_best_period_days": float(original_tcf["summary"]["period"]), "original_gp_tcf_max_score": float(original_tcf["summary"]["score"]), "n_observations": int(finite.sum()), "bls_n_periods": int(args.bls_n_periods), "tcf_n_periods": int(args.tcf_n_periods), "n_durations": int(args.n_durations)}
    residual_table = pd.DataFrame({"time": time, "cadenceno": regular["cadenceno"].to_numpy(), "normalized_flux": flux, "injected_flux": injected_flux, "injected_template": template, "in_transit": in_transit, "original_gp_background": original_model.background_mean, "original_gp_background_std": original_model.background_std, "original_gp_residual": original_model.residuals, "original_gp_standardized_residual": original_model.standardized_residuals, "original_gp_training_anchor": original_model.training_mask, "injected_gp_background": injected_model.background_mean, "injected_gp_background_std": injected_model.background_std, "injected_gp_residual": injected_model.residuals, "injected_gp_standardized_residual": injected_model.standardized_residuals, "injected_gp_training_anchor": injected_model.training_mask})
    model_rows = pd.DataFrame([{**{"series": "original"}, **original_model.summary(), **original_diagnostics}, {**{"series": "injected"}, **injected_model.summary(), **injected_diagnostics}])
    return regular, residual_table, model_rows, original_bls, original_tcf, injected_bls, injected_tcf, summary

def main(args=None):
    args = args or default_settings()
    metrics_dir = Path(args.output_dir) / "metrics"
    processed_dir = Path(args.output_dir) / "processed"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    regular, residual_table, model_rows, original_bls, original_tcf, injected_bls, injected_tcf, summary = run_experiment(args)
    prefix = f"kic_{str(args.target_id).replace('KIC', '').strip()}_q{args.quarter}"
    injected_bls["periodogram"].to_csv(metrics_dir / f"{prefix}_gp_bls_injected_periodogram.csv", index=False)
    injected_bls["top_peaks"].to_csv(metrics_dir / f"{prefix}_gp_bls_injected_top_peaks.csv", index=False)
    original_bls["top_peaks"].to_csv(metrics_dir / f"{prefix}_gp_bls_original_top_peaks.csv", index=False)
    injected_tcf["periodogram"].to_csv(metrics_dir / f"{prefix}_gp_tcf_injected_periodogram.csv", index=False)
    injected_tcf["top_peaks"].to_csv(metrics_dir / f"{prefix}_gp_tcf_injected_top_peaks.csv", index=False)
    original_tcf["periodogram"].to_csv(metrics_dir / f"{prefix}_gp_tcf_original_periodogram.csv", index=False)
    original_tcf["top_peaks"].to_csv(metrics_dir / f"{prefix}_gp_tcf_original_top_peaks.csv", index=False)
    model_rows.to_csv(metrics_dir / f"{prefix}_gp_model_diagnostics.csv", index=False)
    regular.to_parquet(processed_dir / f"{prefix}_gp_baseline_input.parquet", index=False)
    residual_table.to_parquet(processed_dir / f"{prefix}_gp_residuals.parquet", index=False)
    summary_path = metrics_dir / f"{prefix}_gp_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    print(f"GP summary: {summary_path}")
    print(f"GP model diagnostics: {metrics_dir / f'{prefix}_gp_model_diagnostics.csv'}")
    print(f"GP residuals: {processed_dir / f'{prefix}_gp_residuals.parquet'}")
    print(f"BLS recovered period: {summary['gp_bls_recovered_period_days']:.6f} days")
    print(f"BLS period matched: {summary['gp_bls_period_matched']}")
    print(f"TCF recovered period: {summary['gp_tcf_recovered_period_days']:.6f} days")
    print(f"TCF period matched: {summary['gp_tcf_period_matched']}")
    print(f"Depth retention: {summary['depth_retention_fraction']:.3f}")
    print(f"SNR retention: {summary['snr_retention_fraction']:.3f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
