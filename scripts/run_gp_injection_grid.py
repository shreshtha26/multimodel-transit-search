"""Run the matched periodic injection grid on smooth GP residuals."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
import argparse
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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
from adaptive_transit.noise_models.gp import fit_smooth_gp_background
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_ID = "11904151"
QUARTER = 5
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/gp_injection_grid"
FAP_THRESHOLD_PATH = PROJECT_ROOT / "outputs/experiments/gp_null_calibration/metrics/kic_11904151_q5_gp_fap_thresholds.csv"

def default_n_jobs():
    cpu_count = os.cpu_count() or 1
    return max(1, min(6, cpu_count - 1))

def default_settings():
    return SimpleNamespace(target_id=TARGET_ID, quarter=QUARTER, output_dir=OUTPUT_DIR, fap_threshold_path=FAP_THRESHOLD_PATH, quality_policy="default", require_finite_flux_error=False, test_fraction=0.20, gp_max_train_points=512, gp_length_scale_days=3.0, gp_min_length_scale_days=1.0, gp_max_length_scale_days=30.0, gp_measurement_noise_fraction=0.20, gp_n_restarts_optimizer=0, gp_random_seed=123, injection_period_grid=(2.0, 5.0, 10.0), injection_duration_hours_grid=(2.0, 4.0, 8.0), injection_depth_grid=(0.0002, 0.0005, 0.001), epoch_phase_fraction_grid=(0.15, 0.45, 0.75), min_period_days=1.0, max_period_days=15.0, bls_n_periods=1000, tcf_n_periods=10000, min_duration_hours=1.5, max_duration_hours=10.0, n_durations=8, bls_objective="snr", bls_top_k=5, tcf_top_k=10, edge_width_cadences=0, min_edge_observations=4, min_transit_events=3, min_event_consistency_fraction=0.60, period_match_tolerance_fraction=0.02, n_jobs=default_n_jobs(), show_progress=True, resume=True, checkpoint_path=None)

def parse_float_tuple(value):
    return tuple(float(item.strip()) for item in str(value).split(",") if item.strip())

def parse_args():
    defaults = default_settings()
    parser = argparse.ArgumentParser(description="Run the GP residual injection grid.")
    parser.add_argument("--n-jobs", type=int, default=defaults.n_jobs)
    parser.add_argument("--gp-max-train-points", type=int, default=defaults.gp_max_train_points)
    parser.add_argument("--gp-length-scale-days", type=float, default=defaults.gp_length_scale_days)
    parser.add_argument("--gp-min-length-scale-days", type=float, default=defaults.gp_min_length_scale_days)
    parser.add_argument("--gp-measurement-noise-fraction", type=float, default=defaults.gp_measurement_noise_fraction)
    parser.add_argument("--bls-n-periods", type=int, default=defaults.bls_n_periods)
    parser.add_argument("--tcf-n-periods", type=int, default=defaults.tcf_n_periods)
    parser.add_argument("--period-grid", default=",".join(str(value) for value in defaults.injection_period_grid))
    parser.add_argument("--duration-grid", default=",".join(str(value) for value in defaults.injection_duration_hours_grid))
    parser.add_argument("--depth-grid", default=",".join(str(value) for value in defaults.injection_depth_grid))
    parser.add_argument("--epoch-phase-grid", default=",".join(str(value) for value in defaults.epoch_phase_fraction_grid))
    parser.add_argument("--fap-threshold-path", type=Path, default=defaults.fap_threshold_path)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parsed = parser.parse_args()
    defaults.n_jobs = int(parsed.n_jobs)
    defaults.gp_max_train_points = int(parsed.gp_max_train_points)
    defaults.gp_length_scale_days = float(parsed.gp_length_scale_days)
    defaults.gp_min_length_scale_days = float(parsed.gp_min_length_scale_days)
    defaults.gp_measurement_noise_fraction = float(parsed.gp_measurement_noise_fraction)
    defaults.bls_n_periods = int(parsed.bls_n_periods)
    defaults.tcf_n_periods = int(parsed.tcf_n_periods)
    defaults.injection_period_grid = parse_float_tuple(parsed.period_grid)
    defaults.injection_duration_hours_grid = parse_float_tuple(parsed.duration_grid)
    defaults.injection_depth_grid = parse_float_tuple(parsed.depth_grid)
    defaults.epoch_phase_fraction_grid = parse_float_tuple(parsed.epoch_phase_grid)
    defaults.fap_threshold_path = Path(parsed.fap_threshold_path)
    defaults.output_dir = Path(parsed.output_dir)
    defaults.resume = not parsed.no_resume
    defaults.show_progress = not parsed.no_progress
    return defaults

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
        raise FileNotFoundError(f"GP FAP threshold file does not exist: {path}. Run scripts/run_gp_null_calibration.py first.")
    thresholds = pd.read_csv(path)
    required = {"detector", "fap_level", "score_threshold"}
    missing = required.difference(thresholds.columns)
    if missing:
        raise ValueError(f"GP threshold file is missing columns: {sorted(missing)}")
    result = {}
    for detector in ("gp_bls", "gp_tcf"):
        match = thresholds[(thresholds["detector"] == detector) & np.isclose(thresholds["fap_level"], 0.01)]
        if match.empty:
            raise ValueError(f"GP threshold file does not contain a 1% FAP threshold for {detector}.")
        result[detector] = float(match.iloc[0]["score_threshold"])
    return thresholds, result

def fit_gp(time, values, args):
    return fit_smooth_gp_background(time, values, max_train_points=args.gp_max_train_points, length_scale_days=args.gp_length_scale_days, min_length_scale_days=args.gp_min_length_scale_days, max_length_scale_days=args.gp_max_length_scale_days, measurement_noise_fraction=args.gp_measurement_noise_fraction, n_restarts_optimizer=args.gp_n_restarts_optimizer, random_seed=args.gp_random_seed)

def top_peak_rank(peaks, target_period, tolerance_fraction):
    for rank, row in enumerate(peaks.to_dict(orient="records"), start=1):
        period = float(row.get("period_days", row.get("period", np.nan)))
        error = abs(period - float(target_period)) / float(target_period)
        if np.isfinite(error) and error <= float(tolerance_fraction):
            return rank
    return None

def run_one_injection(case_index, time, flux, bls_periods, bls_durations, tcf_periods, tcf_durations, injected_period, injected_duration_hours, injected_depth, epoch_phase_fraction, thresholds, args):
    finite = np.isfinite(time) & np.isfinite(flux)
    epoch = float(np.min(time[finite]) + float(epoch_phase_fraction) * float(injected_period))
    duration_days = float(injected_duration_hours) / 24.0
    injected_flux, template, in_transit = inject_periodic_box_transit(time, flux, injected_period, epoch, duration_days, injected_depth)
    before = periodic_depth_and_snr(injected_flux, in_transit)
    model = fit_gp(time, injected_flux, args)
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
    bls_passes = bool(float(bls_best["power"]) >= thresholds["gp_bls"])
    tcf_passes = bool(float(tcf_best["score"]) >= thresholds["gp_tcf"])
    tcf_top_peaks = tcf_result["top_peaks"]
    bls_top_peaks = bls_result["top_peaks"]
    top_periods = [float(value) for value in tcf_top_peaks["period_days"].to_numpy(dtype=float)]
    top_scores = [float(value) for value in tcf_top_peaks["score"].to_numpy(dtype=float)]
    return {"case_index": int(case_index), "injected_period_days": float(injected_period), "injected_epoch_days": epoch, "epoch_phase_fraction": float(epoch_phase_fraction), "injected_duration_hours": float(injected_duration_hours), "injected_depth": float(injected_depth), "gp_converged": bool(model.converged), "gp_log_marginal_likelihood": float(model.log_marginal_likelihood), "gp_training_point_count": int(model.parameters["training_point_count"]), "gp_signal_variance": float(model.parameters["signal_variance"]), "gp_length_scale_days": float(model.parameters["length_scale_days"]), "gp_measurement_noise_variance": float(model.parameters["measurement_noise_variance"]), "observed_depth_before_gp": float(before["depth"]), "gp_residual_depth": float(after["depth"]), "depth_retention_fraction": float(after["depth"] / before["depth"]) if before["depth"] != 0 else float("nan"), "local_snr_before_gp": float(before["snr"]), "local_snr_after_gp": float(after["snr"]), "snr_retention_fraction": float(after["snr"] / before["snr"]) if before["snr"] != 0 else float("nan"), "gp_bls_recovered_period_days": bls_period, "gp_bls_recovered_power": float(bls_best["power"]), "gp_bls_period_error_fraction": float(bls_harmonic_error), "gp_bls_exact_period_error_fraction": float(bls_exact_error), "gp_bls_period_matched": bls_period_matched, "gp_bls_exact_period_matched": bls_exact_matched, "gp_bls_exact_period_rank_top5": top_peak_rank(bls_top_peaks, injected_period, args.period_match_tolerance_fraction), "gp_bls_fap_1_percent_threshold": float(thresholds["gp_bls"]), "gp_bls_passes_fap_1_percent": bls_passes, "gp_bls_recovered_at_fap_1_percent": bool(bls_period_matched and bls_passes), "gp_bls_exact_recovered_at_fap_1_percent": bool(bls_exact_matched and bls_passes), "gp_tcf_recovered_period_days": tcf_period, "gp_tcf_recovered_score": float(tcf_best["score"]), "gp_tcf_recovered_raw_pooled_score": float(tcf_best["raw_pooled_score"]), "gp_tcf_valid_transit_events": int(tcf_best["n_valid_transit_events"]), "gp_tcf_positive_event_fraction": float(tcf_best["positive_event_fraction"]), "gp_tcf_period_error_fraction": float(tcf_harmonic_error), "gp_tcf_exact_period_error_fraction": float(tcf_exact_error), "gp_tcf_period_matched": tcf_period_matched, "gp_tcf_exact_period_matched": tcf_exact_matched, "gp_tcf_exact_period_rank_top10": matching_peak_rank(tcf_top_peaks, injected_period, tolerance_fraction=args.period_match_tolerance_fraction), "gp_tcf_half_period_rank_top10": harmonic_peak_rank(tcf_top_peaks, injected_period, 0.5, tolerance_fraction=args.period_match_tolerance_fraction), "gp_tcf_double_period_rank_top10": harmonic_peak_rank(tcf_top_peaks, injected_period, 2.0, tolerance_fraction=args.period_match_tolerance_fraction), "gp_tcf_top_periods_json": json.dumps(top_periods), "gp_tcf_top_scores_json": json.dumps(top_scores), "gp_tcf_fap_1_percent_threshold": float(thresholds["gp_tcf"]), "gp_tcf_passes_fap_1_percent": tcf_passes, "gp_tcf_recovered_at_fap_1_percent": bool(tcf_period_matched and tcf_passes), "gp_tcf_exact_recovered_at_fap_1_percent": bool(tcf_exact_matched and tcf_passes)}

def grouped_recovery(results, column):
    return results.groupby(column, as_index=False).agg(injection_count=("gp_converged", "size"), gp_convergence_rate=("gp_converged", "mean"), gp_bls_period_match_rate=("gp_bls_period_matched", "mean"), gp_bls_exact_period_match_rate=("gp_bls_exact_period_matched", "mean"), gp_bls_detection_rate_fap_1_percent=("gp_bls_passes_fap_1_percent", "mean"), gp_bls_recovery_rate_fap_1_percent=("gp_bls_recovered_at_fap_1_percent", "mean"), gp_bls_exact_recovery_rate_fap_1_percent=("gp_bls_exact_recovered_at_fap_1_percent", "mean"), gp_tcf_period_match_rate=("gp_tcf_period_matched", "mean"), gp_tcf_exact_period_match_rate=("gp_tcf_exact_period_matched", "mean"), gp_tcf_detection_rate_fap_1_percent=("gp_tcf_passes_fap_1_percent", "mean"), gp_tcf_recovery_rate_fap_1_percent=("gp_tcf_recovered_at_fap_1_percent", "mean"), gp_tcf_exact_recovery_rate_fap_1_percent=("gp_tcf_exact_recovered_at_fap_1_percent", "mean"), median_depth_retention_fraction=("depth_retention_fraction", "median"), median_snr_retention_fraction=("snr_retention_fraction", "median"), median_gp_length_scale_days=("gp_length_scale_days", "median"))

def load_completed_injection_rows(combinations, args):
    path = getattr(args, "checkpoint_path", None)
    if not getattr(args, "resume", False) or path is None or not Path(path).exists():
        return [], set()
    frame = pd.read_csv(path)
    if "case_index" not in frame.columns:
        return [], set()
    rows = []
    completed = set()
    for row in frame.to_dict(orient="records"):
        case_index = int(row["case_index"])
        if case_index < 0 or case_index >= len(combinations) or case_index in completed:
            continue
        expected = combinations[case_index]
        observed = (float(row["injected_period_days"]), float(row["injected_duration_hours"]), float(row["injected_depth"]), float(row["epoch_phase_fraction"]))
        if observed == tuple(float(value) for value in expected):
            rows.append(row)
            completed.add(case_index)
    return rows, completed

def write_injection_checkpoint(rows, args):
    path = getattr(args, "checkpoint_path", None)
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sorted(rows, key=lambda row: int(row["case_index"]))).to_csv(path, index=False)

def run_parallel_injections(combinations, time, flux, bls_periods, bls_durations, tcf_periods, tcf_durations, thresholds, args):
    rows, completed = load_completed_injection_rows(combinations, args)
    remaining = [(case_index, values) for case_index, values in enumerate(combinations) if case_index not in completed]
    if not remaining:
        return sorted(rows, key=lambda row: row["case_index"])
    tasks = (delayed(run_one_injection)(case_index, time, flux, bls_periods, bls_durations, tcf_periods, tcf_durations, injected_period, injected_duration_hours, injected_depth, epoch_phase_fraction, thresholds, args) for case_index, (injected_period, injected_duration_hours, injected_depth, epoch_phase_fraction) in remaining)
    worker_count = max(1, int(args.n_jobs))
    generator = Parallel(n_jobs=worker_count, prefer="processes", batch_size=1, return_as="generator_unordered")(tasks)
    for row in tqdm(generator, total=len(combinations), initial=len(completed), desc=f"GP injection grid ({worker_count} workers)", disable=not args.show_progress):
        rows.append(row)
        write_injection_checkpoint(rows, args)
    return sorted(rows, key=lambda row: row["case_index"])

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
    results = pd.DataFrame(run_parallel_injections(combinations, time, flux, bls_periods, bls_durations, tcf_periods, tcf_durations, thresholds, args))
    by_depth = grouped_recovery(results, "injected_depth")
    by_duration = grouped_recovery(results, "injected_duration_hours")
    by_period = grouped_recovery(results, "injected_period_days")
    summary = {"target_id": str(args.target_id), "quarter": int(args.quarter), "quality_policy": args.quality_policy, "model_name": "smooth_anchor_gp", "parallel_workers": int(args.n_jobs), "injection_count": int(len(results)), "gp_convergence_rate": float(results["gp_converged"].mean()), "gp_bls_period_match_rate": float(results["gp_bls_period_matched"].mean()), "gp_bls_exact_period_match_rate": float(results["gp_bls_exact_period_matched"].mean()), "gp_bls_detection_rate_fap_1_percent": float(results["gp_bls_passes_fap_1_percent"].mean()), "gp_bls_recovery_rate_fap_1_percent": float(results["gp_bls_recovered_at_fap_1_percent"].mean()), "gp_bls_exact_recovery_rate_fap_1_percent": float(results["gp_bls_exact_recovered_at_fap_1_percent"].mean()), "gp_tcf_period_match_rate": float(results["gp_tcf_period_matched"].mean()), "gp_tcf_exact_period_match_rate": float(results["gp_tcf_exact_period_matched"].mean()), "gp_tcf_detection_rate_fap_1_percent": float(results["gp_tcf_passes_fap_1_percent"].mean()), "gp_tcf_recovery_rate_fap_1_percent": float(results["gp_tcf_recovered_at_fap_1_percent"].mean()), "gp_tcf_exact_recovery_rate_fap_1_percent": float(results["gp_tcf_exact_recovered_at_fap_1_percent"].mean()), "median_depth_retention_fraction": float(results["depth_retention_fraction"].median()), "median_snr_retention_fraction": float(results["snr_retention_fraction"].median()), "median_gp_length_scale_days": float(results["gp_length_scale_days"].median()), "gp_bls_fap_1_percent_threshold": float(thresholds["gp_bls"]), "gp_tcf_fap_1_percent_threshold": float(thresholds["gp_tcf"])}
    return regular, threshold_table, results, by_depth, by_duration, by_period, summary

def main(args=None):
    args = args or default_settings()
    metrics_dir = Path(args.output_dir) / "metrics"
    processed_dir = Path(args.output_dir) / "processed"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"kic_{str(args.target_id).replace('KIC', '').strip()}_q{args.quarter}"
    results_path = metrics_dir / f"{prefix}_gp_injection_grid.csv"
    summary_path = metrics_dir / f"{prefix}_gp_injection_grid_summary.json"
    args.checkpoint_path = results_path
    regular, threshold_table, results, by_depth, by_duration, by_period, summary = run_experiment(args)
    results.to_csv(results_path, index=False)
    by_depth.to_csv(metrics_dir / f"{prefix}_gp_recovery_by_depth.csv", index=False)
    by_duration.to_csv(metrics_dir / f"{prefix}_gp_recovery_by_duration.csv", index=False)
    by_period.to_csv(metrics_dir / f"{prefix}_gp_recovery_by_period.csv", index=False)
    threshold_table.to_csv(metrics_dir / f"{prefix}_gp_used_fap_thresholds.csv", index=False)
    regular.to_parquet(processed_dir / f"{prefix}_gp_injection_grid_input.parquet", index=False)
    summary_path.write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    print(f"GP injection results: {results_path}")
    print(f"GP injection summary: {summary_path}")
    print(f"GP-BLS recovery at 1% FAP: {summary['gp_bls_recovery_rate_fap_1_percent']:.3f}")
    print(f"GP-TCF recovery at 1% FAP: {summary['gp_tcf_recovery_rate_fap_1_percent']:.3f}")
    print(f"Median depth retention: {summary['median_depth_retention_fraction']:.3f}")
    print(f"Median SNR retention: {summary['median_snr_retention_fraction']:.3f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
