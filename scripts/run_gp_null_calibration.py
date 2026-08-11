"""Calibrate BLS and TCF false-alarm thresholds on smooth GP residuals."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.detection.bls import default_duration_grid as bls_duration_grid
from adaptive_transit.detection.bls import default_period_grid as bls_period_grid
from adaptive_transit.detection.bls import run_bls
from adaptive_transit.detection.false_alarm import moving_block_surrogate
from adaptive_transit.detection.tcf import default_duration_grid as tcf_duration_grid
from adaptive_transit.detection.tcf import default_period_grid as tcf_period_grid
from adaptive_transit.detection.tcf import run_tcf
from adaptive_transit.noise_models.diagnostics import residual_diagnostics
from adaptive_transit.noise_models.gp import fit_smooth_gp_background
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_ID = "11904151"
QUARTER = 5
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/gp_null_calibration"

def default_n_jobs():
    cpu_count = os.cpu_count() or 1
    return max(1, min(6, cpu_count - 1))

def default_settings():
    return SimpleNamespace(target_id=TARGET_ID, quarter=QUARTER, output_dir=OUTPUT_DIR, quality_policy="default", require_finite_flux_error=False, test_fraction=0.20, gp_max_train_points=512, gp_length_scale_days=3.0, gp_min_length_scale_days=1.0, gp_max_length_scale_days=30.0, gp_measurement_noise_fraction=0.20, gp_n_restarts_optimizer=0, gp_random_seed=123, min_period_days=1.0, max_period_days=15.0, bls_n_periods=1000, tcf_n_periods=10000, min_duration_hours=1.5, max_duration_hours=10.0, n_durations=8, bls_objective="snr", edge_width_cadences=0, min_edge_observations=4, min_transit_events=3, min_event_consistency_fraction=0.60, top_k=1, search_mode="coarse_to_fine", n_coarse_periods=2000, n_refinement_regions=20, refinement_half_width_points=30, n_null_trials=1000, null_block_size_cadences=24, fap_levels=(0.01,), random_seed=123, minimum_success_fraction=0.95, n_jobs=default_n_jobs(), show_progress=True, resume=True, checkpoint_path=None)

def parse_args():
    defaults = default_settings()
    parser = argparse.ArgumentParser(description="Calibrate GP-BLS and GP-TCF null thresholds.")
    parser.add_argument("--n-null-trials", type=int, default=defaults.n_null_trials)
    parser.add_argument("--n-jobs", type=int, default=defaults.n_jobs)
    parser.add_argument("--gp-max-train-points", type=int, default=defaults.gp_max_train_points)
    parser.add_argument("--gp-length-scale-days", type=float, default=defaults.gp_length_scale_days)
    parser.add_argument("--gp-min-length-scale-days", type=float, default=defaults.gp_min_length_scale_days)
    parser.add_argument("--gp-measurement-noise-fraction", type=float, default=defaults.gp_measurement_noise_fraction)
    parser.add_argument("--bls-n-periods", type=int, default=defaults.bls_n_periods)
    parser.add_argument("--tcf-n-periods", type=int, default=defaults.tcf_n_periods)
    parser.add_argument("--n-coarse-periods", type=int, default=defaults.n_coarse_periods)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parsed = parser.parse_args()
    defaults.n_null_trials = int(parsed.n_null_trials)
    defaults.n_jobs = int(parsed.n_jobs)
    defaults.gp_max_train_points = int(parsed.gp_max_train_points)
    defaults.gp_length_scale_days = float(parsed.gp_length_scale_days)
    defaults.gp_min_length_scale_days = float(parsed.gp_min_length_scale_days)
    defaults.gp_measurement_noise_fraction = float(parsed.gp_measurement_noise_fraction)
    defaults.bls_n_periods = int(parsed.bls_n_periods)
    defaults.tcf_n_periods = int(parsed.tcf_n_periods)
    defaults.n_coarse_periods = int(parsed.n_coarse_periods)
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

def fit_gp(time, values, args):
    return fit_smooth_gp_background(time, values, max_train_points=args.gp_max_train_points, length_scale_days=args.gp_length_scale_days, min_length_scale_days=args.gp_min_length_scale_days, max_length_scale_days=args.gp_max_length_scale_days, measurement_noise_fraction=args.gp_measurement_noise_fraction, n_restarts_optimizer=args.gp_n_restarts_optimizer, random_seed=args.gp_random_seed)

def threshold_rows(successful, args):
    rows = []
    for detector, score_column in (("gp_bls", "bls_max_power"), ("gp_tcf", "tcf_max_score")):
        scores = successful[score_column].to_numpy(dtype=float)
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            continue
        for level in args.fap_levels:
            threshold = float(np.quantile(scores, 1.0 - float(level), method="higher"))
            rows.append({"detector": detector, "fap_level": float(level), "score_column": score_column, "score_threshold": threshold, "requested_null_trials": int(args.n_null_trials), "successful_null_trials": int(scores.size), "observed_exceedance_fraction": float(np.mean(scores >= threshold))})
    return pd.DataFrame(rows)

def run_one_trial(trial, seed, time, base_residuals, bls_periods, bls_durations, tcf_periods, tcf_durations, args):
    try:
        rng = np.random.default_rng(int(seed))
        surrogate = moving_block_surrogate(base_residuals, block_size=args.null_block_size_cadences, rng=rng)
        bls_result = run_bls(time, surrogate, None, bls_periods, bls_durations, objective=args.bls_objective, top_k=1)
        tcf_result = run_tcf(time, surrogate, tcf_periods, tcf_durations, edge_width_cadences=args.edge_width_cadences, min_edge_observations=args.min_edge_observations, min_transit_events=args.min_transit_events, min_event_consistency_fraction=args.min_event_consistency_fraction, top_k=args.top_k, search_mode=args.search_mode, n_coarse_periods=args.n_coarse_periods, n_refinement_regions=args.n_refinement_regions, refinement_half_width_points=args.refinement_half_width_points)
        bls_best = bls_result["summary"]
        tcf_best = tcf_result["summary"]
        return {"trial": int(trial), "trial_seed": int(seed), "success": True, "bls_max_power": float(bls_best["power"]), "bls_best_period": float(bls_best["period"]), "bls_best_duration": float(bls_best["duration"]), "bls_best_epoch": float(bls_best["transit_time"]), "tcf_max_score": float(tcf_best["score"]), "tcf_best_raw_pooled_score": float(tcf_best["raw_pooled_score"]), "tcf_best_period": float(tcf_best["period"]), "tcf_best_duration": float(tcf_best["duration"]), "tcf_best_epoch": float(tcf_best["epoch"]), "tcf_valid_transit_events": int(tcf_best["n_valid_transit_events"]), "tcf_positive_event_fraction": float(tcf_best["positive_event_fraction"]), "error": ""}
    except Exception as exc:
        return {"trial": int(trial), "trial_seed": int(seed), "success": False, "bls_max_power": np.nan, "bls_best_period": np.nan, "bls_best_duration": np.nan, "bls_best_epoch": np.nan, "tcf_max_score": np.nan, "tcf_best_raw_pooled_score": np.nan, "tcf_best_period": np.nan, "tcf_best_duration": np.nan, "tcf_best_epoch": np.nan, "tcf_valid_transit_events": np.nan, "tcf_positive_event_fraction": np.nan, "error": f"{type(exc).__name__}: {exc}"}

def create_trial_seeds(args):
    root = np.random.SeedSequence(int(args.random_seed))
    children = root.spawn(int(args.n_null_trials))
    return [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children]

def load_completed_trial_rows(seeds, args):
    path = getattr(args, "checkpoint_path", None)
    if not getattr(args, "resume", False) or path is None or not Path(path).exists():
        return [], set()
    frame = pd.read_csv(path)
    required = {"trial", "trial_seed", "success"}
    if not required.issubset(frame.columns):
        return [], set()
    rows = []
    completed = set()
    seed_map = {int(index): int(seed) for index, seed in enumerate(seeds)}
    for row in frame.to_dict(orient="records"):
        trial = int(row["trial"])
        if trial in seed_map and int(row["trial_seed"]) == seed_map[trial] and trial not in completed:
            rows.append(row)
            completed.add(trial)
    return rows, completed

def write_trial_checkpoint(rows, args):
    path = getattr(args, "checkpoint_path", None)
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sorted(rows, key=lambda row: int(row["trial"]))).to_csv(path, index=False)

def run_parallel_trials(seeds, time, base_residuals, bls_periods, bls_durations, tcf_periods, tcf_durations, args):
    rows, completed = load_completed_trial_rows(seeds, args)
    remaining = [(index, seed) for index, seed in enumerate(seeds) if index not in completed]
    if not remaining:
        return sorted(rows, key=lambda row: row["trial"])
    worker_count = max(1, int(args.n_jobs))
    tasks = (delayed(run_one_trial)(index, seed, time, base_residuals, bls_periods, bls_durations, tcf_periods, tcf_durations, args) for index, seed in remaining)
    generator = Parallel(n_jobs=worker_count, prefer="processes", batch_size=1, return_as="generator_unordered")(tasks)
    for row in tqdm(generator, total=len(seeds), initial=len(completed), desc=f"GP null trials ({worker_count} workers)", disable=not args.show_progress):
        rows.append(row)
        write_trial_checkpoint(rows, args)
    return sorted(rows, key=lambda row: row["trial"])

def run_calibration(args):
    light_curve = load_kepler_pdcsap(args.target_id, args.quarter)
    regular, preprocessing = preprocess_pdcsap_light_curve(light_curve.to_dataframe(), quality_policy=args.quality_policy, require_finite_flux_error=args.require_finite_flux_error, normalization_fit_fraction=1.0 - args.test_fraction)
    time = regular["time"].to_numpy(dtype=float)
    flux = regular["normalized_flux"].to_numpy(dtype=float)
    base_model = fit_gp(time, flux, args)
    base_residuals = base_model.residuals
    base_diagnostics = residual_diagnostics(base_residuals[base_model.usable_mask])
    bls_periods = bls_period_grid(time, min_period_days=args.min_period_days, max_period_days=args.max_period_days, n_periods=args.bls_n_periods)
    bls_durations = bls_duration_grid(args.min_duration_hours, args.max_duration_hours, args.n_durations)
    tcf_periods = tcf_period_grid(time, min_period_days=args.min_period_days, max_period_days=args.max_period_days, n_periods=args.tcf_n_periods)
    tcf_durations = tcf_duration_grid(args.min_duration_hours, args.max_duration_hours, args.n_durations)
    seeds = create_trial_seeds(args)
    trials = pd.DataFrame(run_parallel_trials(seeds, time, base_residuals, bls_periods, bls_durations, tcf_periods, tcf_durations, args))
    successful = trials[trials["success"]].copy()
    success_fraction = float(len(successful) / int(args.n_null_trials))
    if success_fraction < args.minimum_success_fraction:
        errors = trials.loc[~trials["success"], "error"].value_counts().head(10)
        raise RuntimeError(f"Only {success_fraction:.3f} of GP null trials succeeded.\n{errors.to_string()}")
    thresholds = threshold_rows(successful, args)
    summary = {"target_id": str(args.target_id), "quarter": int(args.quarter), "quality_policy": args.quality_policy, "model_name": "smooth_anchor_gp", "calibration_scope": "detector_conditional_on_fitted_gp_residuals", "base_model": base_model.summary(), "base_residual_diagnostics": base_diagnostics, "requested_null_trials": int(args.n_null_trials), "successful_null_trials": int(len(successful)), "success_fraction": success_fraction, "null_block_size_cadences": int(args.null_block_size_cadences), "random_seed": int(args.random_seed), "parallel_workers": int(args.n_jobs), "bls_median_max_power": float(successful["bls_max_power"].median()), "bls_maximum_null_power": float(successful["bls_max_power"].max()), "tcf_median_max_score": float(successful["tcf_max_score"].median()), "tcf_maximum_null_score": float(successful["tcf_max_score"].max()), "fap_levels": list(args.fap_levels), "thresholds": thresholds.to_dict(orient="records")}
    return regular, trials, thresholds, summary

def main(args=None):
    args = args or default_settings()
    metrics_dir = Path(args.output_dir) / "metrics"
    processed_dir = Path(args.output_dir) / "processed"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"kic_{str(args.target_id).replace('KIC', '').strip()}_q{args.quarter}"
    trials_path = metrics_dir / f"{prefix}_gp_null_trials.csv"
    thresholds_path = metrics_dir / f"{prefix}_gp_fap_thresholds.csv"
    summary_path = metrics_dir / f"{prefix}_gp_null_calibration_summary.json"
    args.checkpoint_path = trials_path
    regular, trials, thresholds, summary = run_calibration(args)
    trials.to_csv(trials_path, index=False)
    thresholds.to_csv(thresholds_path, index=False)
    regular.to_parquet(processed_dir / f"{prefix}_gp_null_calibration_input.parquet", index=False)
    summary_path.write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    print(f"GP null trials: {trials_path}")
    print(f"GP FAP thresholds: {thresholds_path}")
    print(f"GP null-calibration summary: {summary_path}")
    print(thresholds.to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
