"""Run the matched periodic injection grid on local-level Kalman residuals."""
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.detection.bls import default_duration_grid as bls_duration_grid
from adaptive_transit.detection.bls import default_period_grid as bls_period_grid
from adaptive_transit.detection.bls import period_match_fraction as bls_period_error
from adaptive_transit.detection.bls import run_bls
from adaptive_transit.detection.tcf import default_duration_grid as tcf_duration_grid
from adaptive_transit.detection.tcf import default_period_grid as tcf_period_grid
from adaptive_transit.detection.tcf import harmonic_peak_rank
from adaptive_transit.detection.tcf import matching_peak_rank
from adaptive_transit.detection.tcf import period_match_fraction as tcf_period_error
from adaptive_transit.detection.tcf import run_tcf
from adaptive_transit.injections.synthetic import inject_periodic_box_transit
from adaptive_transit.noise_models.kalman import fit_kalman_local_level
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_ID = "11904151"
QUARTER = 5
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/kalman_injection_grid"
FAP_THRESHOLD_PATH = PROJECT_ROOT / "outputs/experiments/kalman_null_calibration/metrics/kic_11904151_q5_kalman_fap_thresholds.csv"

def default_settings():
    return SimpleNamespace(target_id=TARGET_ID, quarter=QUARTER, output_dir=OUTPUT_DIR, fap_threshold_path=FAP_THRESHOLD_PATH, quality_policy="default", require_finite_flux_error=False, test_fraction=0.20, kalman_maxiter=100, kalman_burn_in=1, injection_period_grid=(2.0, 5.0, 10.0), injection_duration_hours_grid=(2.0, 4.0, 8.0), injection_depth_grid=(0.0002, 0.0005, 0.001), epoch_phase_fraction_grid=(0.15, 0.45, 0.75), min_period_days=1.0, max_period_days=15.0, bls_n_periods=1000, tcf_n_periods=10000, min_duration_hours=1.5, max_duration_hours=10.0, n_durations=8, bls_objective="snr", bls_top_k=5, tcf_top_k=10, edge_width_cadences=0, min_edge_observations=4, min_transit_events=3, min_event_consistency_fraction=0.60, period_match_tolerance_fraction=0.02)

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
        return {"depth": float("nan"), "snr": float("nan")}
    depth = float(np.median(series[finite_out]) - np.median(series[finite_in]))
    noise = robust_scale(series[finite_out])
    snr = float(depth / noise * np.sqrt(finite_in.sum())) if np.isfinite(noise) and noise > 0 else float("nan")
    return {"depth": depth, "snr": snr}

def load_fap_thresholds(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Kalman FAP threshold file does not exist: {path}. Run scripts/run_kalman_null_calibration.py first.")
    thresholds = pd.read_csv(path)
    required = {"detector", "fap_level", "score_threshold"}
    missing = required.difference(thresholds.columns)
    if missing:
        raise ValueError(f"Kalman threshold file is missing columns: {sorted(missing)}")
    result = {}
    for detector in ("kalman_bls", "kalman_tcf"):
        match = thresholds[(thresholds["detector"] == detector) & np.isclose(thresholds["fap_level"], 0.01)]
        if match.empty:
            raise ValueError(f"Kalman threshold file does not contain a 1% FAP threshold for {detector}.")
        result[detector] = float(match.iloc[0]["score_threshold"])
    return thresholds, result

def top_peak_rank(peaks, target_period, tolerance_fraction):
    for rank, row in enumerate(peaks.to_dict(orient="records"), start=1):
        period = float(row.get("period_days", row.get("period", np.nan)))
        error = abs(period - float(target_period)) / float(target_period)
        if np.isfinite(error) and error <= float(tolerance_fraction):
            return rank
    return None

def run_one_injection(time, flux, bls_periods, bls_durations, tcf_periods, tcf_durations, injected_period, injected_duration_hours, injected_depth, epoch_phase_fraction, thresholds, args):
    finite = np.isfinite(time) & np.isfinite(flux)
    epoch = float(np.min(time[finite]) + float(epoch_phase_fraction) * float(injected_period))
    duration_days = float(injected_duration_hours) / 24.0
    injected_flux, template, in_transit = inject_periodic_box_transit(time, flux, injected_period, epoch, duration_days, injected_depth)
    before = periodic_depth_and_snr(injected_flux, in_transit)
    model = fit_kalman_local_level(injected_flux, maxiter=args.kalman_maxiter, burn_in=args.kalman_burn_in)
    after = periodic_depth_and_snr(model.residuals, in_transit)
    bls_result = run_bls(time, model.residuals, None, bls_periods, bls_durations, objective=args.bls_objective, top_k=args.bls_top_k)
    tcf_result = run_tcf(time, model.residuals, tcf_periods, tcf_durations, edge_width_cadences=args.edge_width_cadences, min_edge_observations=args.min_edge_observations, min_transit_events=args.min_transit_events, min_event_consistency_fraction=args.min_event_consistency_fraction, top_k=args.tcf_top_k)
    bls_best = bls_result["summary"]
    tcf_best = tcf_result["summary"]
    bls_period = float(bls_best["period"])
    tcf_period = float(tcf_best["period"])
    bls_harmonic_error = bls_period_error(bls_period, injected_period)
    tcf_harmonic_error = tcf_period_error(tcf_period, injected_period)
    bls_exact_error = abs(bls_period - float(injected_period)) / float(injected_period)
    tcf_exact_error = abs(tcf_period - float(injected_period)) / float(injected_period)
    bls_period_matched = bool(np.isfinite(bls_harmonic_error) and bls_harmonic_error <= args.period_match_tolerance_fraction)
    tcf_period_matched = bool(np.isfinite(tcf_harmonic_error) and tcf_harmonic_error <= args.period_match_tolerance_fraction)
    bls_exact_matched = bool(bls_exact_error <= args.period_match_tolerance_fraction)
    tcf_exact_matched = bool(tcf_exact_error <= args.period_match_tolerance_fraction)
    bls_passes = bool(float(bls_best["power"]) >= thresholds["kalman_bls"])
    tcf_passes = bool(float(tcf_best["score"]) >= thresholds["kalman_tcf"])
    tcf_top_peaks = tcf_result["top_peaks"]
    bls_top_peaks = bls_result["top_peaks"]
    top_periods = [float(value) for value in tcf_top_peaks["period_days"].to_numpy(dtype=float)]
    top_scores = [float(value) for value in tcf_top_peaks["score"].to_numpy(dtype=float)]
    return {"injected_period_days": float(injected_period), "injected_epoch_days": epoch, "epoch_phase_fraction": float(epoch_phase_fraction), "injected_duration_hours": float(injected_duration_hours), "injected_depth": float(injected_depth), "kalman_converged": bool(model.converged), "kalman_log_likelihood": float(model.log_likelihood), "kalman_process_variance": float(model.parameters["process_variance"]), "kalman_measurement_variance": float(model.parameters["measurement_variance"]), "observed_depth_before_kalman": float(before["depth"]), "kalman_residual_depth": float(after["depth"]), "depth_retention_fraction": float(after["depth"] / before["depth"]) if before["depth"] != 0 else float("nan"), "local_snr_before_kalman": float(before["snr"]), "local_snr_after_kalman": float(after["snr"]), "snr_retention_fraction": float(after["snr"] / before["snr"]) if before["snr"] != 0 else float("nan"), "kalman_bls_recovered_period_days": bls_period, "kalman_bls_recovered_power": float(bls_best["power"]), "kalman_bls_period_error_fraction": float(bls_harmonic_error), "kalman_bls_exact_period_error_fraction": float(bls_exact_error), "kalman_bls_period_matched": bls_period_matched, "kalman_bls_exact_period_matched": bls_exact_matched, "kalman_bls_exact_period_rank_top5": top_peak_rank(bls_top_peaks, injected_period, args.period_match_tolerance_fraction), "kalman_bls_fap_1_percent_threshold": float(thresholds["kalman_bls"]), "kalman_bls_passes_fap_1_percent": bls_passes, "kalman_bls_recovered_at_fap_1_percent": bool(bls_period_matched and bls_passes), "kalman_bls_exact_recovered_at_fap_1_percent": bool(bls_exact_matched and bls_passes), "kalman_tcf_recovered_period_days": tcf_period, "kalman_tcf_recovered_score": float(tcf_best["score"]), "kalman_tcf_recovered_raw_pooled_score": float(tcf_best["raw_pooled_score"]), "kalman_tcf_valid_transit_events": int(tcf_best["n_valid_transit_events"]), "kalman_tcf_positive_event_fraction": float(tcf_best["positive_event_fraction"]), "kalman_tcf_period_error_fraction": float(tcf_harmonic_error), "kalman_tcf_exact_period_error_fraction": float(tcf_exact_error), "kalman_tcf_period_matched": tcf_period_matched, "kalman_tcf_exact_period_matched": tcf_exact_matched, "kalman_tcf_exact_period_rank_top10": matching_peak_rank(tcf_top_peaks, injected_period, tolerance_fraction=args.period_match_tolerance_fraction), "kalman_tcf_half_period_rank_top10": harmonic_peak_rank(tcf_top_peaks, injected_period, 0.5, tolerance_fraction=args.period_match_tolerance_fraction), "kalman_tcf_double_period_rank_top10": harmonic_peak_rank(tcf_top_peaks, injected_period, 2.0, tolerance_fraction=args.period_match_tolerance_fraction), "kalman_tcf_top_periods_json": json.dumps(top_periods), "kalman_tcf_top_scores_json": json.dumps(top_scores), "kalman_tcf_fap_1_percent_threshold": float(thresholds["kalman_tcf"]), "kalman_tcf_passes_fap_1_percent": tcf_passes, "kalman_tcf_recovered_at_fap_1_percent": bool(tcf_period_matched and tcf_passes), "kalman_tcf_exact_recovered_at_fap_1_percent": bool(tcf_exact_matched and tcf_passes)}

def grouped_recovery(results, column):
    return results.groupby(column, as_index=False).agg(injection_count=("kalman_converged", "size"), kalman_convergence_rate=("kalman_converged", "mean"), kalman_bls_period_match_rate=("kalman_bls_period_matched", "mean"), kalman_bls_exact_period_match_rate=("kalman_bls_exact_period_matched", "mean"), kalman_bls_detection_rate_fap_1_percent=("kalman_bls_passes_fap_1_percent", "mean"), kalman_bls_recovery_rate_fap_1_percent=("kalman_bls_recovered_at_fap_1_percent", "mean"), kalman_bls_exact_recovery_rate_fap_1_percent=("kalman_bls_exact_recovered_at_fap_1_percent", "mean"), kalman_tcf_period_match_rate=("kalman_tcf_period_matched", "mean"), kalman_tcf_exact_period_match_rate=("kalman_tcf_exact_period_matched", "mean"), kalman_tcf_detection_rate_fap_1_percent=("kalman_tcf_passes_fap_1_percent", "mean"), kalman_tcf_recovery_rate_fap_1_percent=("kalman_tcf_recovered_at_fap_1_percent", "mean"), kalman_tcf_exact_recovery_rate_fap_1_percent=("kalman_tcf_exact_recovered_at_fap_1_percent", "mean"), median_depth_retention_fraction=("depth_retention_fraction", "median"), median_snr_retention_fraction=("snr_retention_fraction", "median"))

def run_experiment(args):
    light_curve = load_kepler_pdcsap(args.target_id, args.quarter)
    regular, preprocessing = preprocess_pdcsap_light_curve(light_curve.to_dataframe(), quality_policy=args.quality_policy, require_finite_flux_error=args.require_finite_flux_error, normalization_fit_fraction=1.0 - args.test_fraction)
    time = regular["time"].to_numpy(dtype=float)
    flux = regular["normalized_flux"].to_numpy(dtype=float)
    bls_periods = bls_period_grid(time, min_period_days=args.min_period_days, max_period_days=args.max_period_days, n_periods=args.bls_n_periods)
    bls_durations = bls_duration_grid(args.min_duration_hours, args.max_duration_hours, args.n_durations)
    tcf_periods = tcf_period_grid(time, min_period_days=args.min_period_days, max_period_days=args.max_period_days, n_periods=args.tcf_n_periods)
    tcf_durations = tcf_duration_grid(args.min_duration_hours, args.max_duration_hours, args.n_durations)
    threshold_table, thresholds = load_fap_thresholds(args.fap_threshold_path)
    combinations = list(product(args.injection_period_grid, args.injection_duration_hours_grid, args.injection_depth_grid, args.epoch_phase_fraction_grid))
    rows = []
    for injected_period, injected_duration_hours, injected_depth, epoch_phase_fraction in tqdm(combinations, desc="Kalman injection grid"):
        rows.append(run_one_injection(time, flux, bls_periods, bls_durations, tcf_periods, tcf_durations, injected_period, injected_duration_hours, injected_depth, epoch_phase_fraction, thresholds, args))
    results = pd.DataFrame(rows)
    by_depth = grouped_recovery(results, "injected_depth")
    by_duration = grouped_recovery(results, "injected_duration_hours")
    by_period = grouped_recovery(results, "injected_period_days")
    summary = {"target_id": str(args.target_id), "quarter": int(args.quarter), "quality_policy": args.quality_policy, "model_name": "local_level_kalman", "injection_count": int(len(results)), "kalman_convergence_rate": float(results["kalman_converged"].mean()), "kalman_bls_period_match_rate": float(results["kalman_bls_period_matched"].mean()), "kalman_bls_exact_period_match_rate": float(results["kalman_bls_exact_period_matched"].mean()), "kalman_bls_detection_rate_fap_1_percent": float(results["kalman_bls_passes_fap_1_percent"].mean()), "kalman_bls_recovery_rate_fap_1_percent": float(results["kalman_bls_recovered_at_fap_1_percent"].mean()), "kalman_bls_exact_recovery_rate_fap_1_percent": float(results["kalman_bls_exact_recovered_at_fap_1_percent"].mean()), "kalman_tcf_period_match_rate": float(results["kalman_tcf_period_matched"].mean()), "kalman_tcf_exact_period_match_rate": float(results["kalman_tcf_exact_period_matched"].mean()), "kalman_tcf_detection_rate_fap_1_percent": float(results["kalman_tcf_passes_fap_1_percent"].mean()), "kalman_tcf_recovery_rate_fap_1_percent": float(results["kalman_tcf_recovered_at_fap_1_percent"].mean()), "kalman_tcf_exact_recovery_rate_fap_1_percent": float(results["kalman_tcf_exact_recovered_at_fap_1_percent"].mean()), "median_depth_retention_fraction": float(results["depth_retention_fraction"].median()), "median_snr_retention_fraction": float(results["snr_retention_fraction"].median()), "kalman_bls_fap_1_percent_threshold": float(thresholds["kalman_bls"]), "kalman_tcf_fap_1_percent_threshold": float(thresholds["kalman_tcf"])}
    return regular, threshold_table, results, by_depth, by_duration, by_period, summary

def main(args=None):
    args = args or default_settings()
    metrics_dir = Path(args.output_dir) / "metrics"
    processed_dir = Path(args.output_dir) / "processed"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    regular, threshold_table, results, by_depth, by_duration, by_period, summary = run_experiment(args)
    prefix = f"kic_{str(args.target_id).replace('KIC', '').strip()}_q{args.quarter}"
    results_path = metrics_dir / f"{prefix}_kalman_injection_grid.csv"
    summary_path = metrics_dir / f"{prefix}_kalman_injection_grid_summary.json"
    results.to_csv(results_path, index=False)
    by_depth.to_csv(metrics_dir / f"{prefix}_kalman_recovery_by_depth.csv", index=False)
    by_duration.to_csv(metrics_dir / f"{prefix}_kalman_recovery_by_duration.csv", index=False)
    by_period.to_csv(metrics_dir / f"{prefix}_kalman_recovery_by_period.csv", index=False)
    threshold_table.to_csv(metrics_dir / f"{prefix}_kalman_used_fap_thresholds.csv", index=False)
    regular.to_parquet(processed_dir / f"{prefix}_kalman_injection_grid_input.parquet", index=False)
    summary_path.write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    print(f"Kalman injection results: {results_path}")
    print(f"Kalman injection summary: {summary_path}")
    print(f"Kalman-BLS recovery at 1% FAP: {summary['kalman_bls_recovery_rate_fap_1_percent']:.3f}")
    print(f"Kalman-TCF recovery at 1% FAP: {summary['kalman_tcf_recovery_rate_fap_1_percent']:.3f}")
    print(f"Median depth retention: {summary['median_depth_retention_fraction']:.3f}")
    print(f"Median SNR retention: {summary['median_snr_retention_fraction']:.3f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
