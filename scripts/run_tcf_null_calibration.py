"""Calibrate the event-consistent TCF 1 percent false-alarm threshold in parallel."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm.auto import tqdm
from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.detection.false_alarm import moving_block_surrogate
from adaptive_transit.detection.tcf import default_duration_grid, default_period_grid, fit_arima_innovations, run_tcf
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve
warnings.filterwarnings("ignore", category=ConvergenceWarning)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_ID = "11904151"
QUARTER = 5
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/tcf_null_calibration"
_WORKER_TIME = None
_WORKER_BASE_SERIES = None
_WORKER_PERIOD_GRID = None
_WORKER_DURATION_GRID = None
_WORKER_CONFIG = None

def default_settings():
    return SimpleNamespace(target_id=TARGET_ID, quarter=QUARTER, output_dir=OUTPUT_DIR, quality_policy="default", require_finite_flux_error=False, test_fraction=0.20, arima_order=(1, 1, 0), fit_maxiter=100, surrogate_source="innovations", min_period_days=1.0, max_period_days=15.0, n_periods=10000, min_duration_hours=1.5, max_duration_hours=10.0, n_durations=8, edge_width_cadences=0, min_edge_observations=4, min_transit_events=3, min_event_consistency_fraction=0.60, top_k=1, search_mode="coarse_to_fine", n_coarse_periods=2000, n_refinement_regions=20, refinement_half_width_points=30, n_null_trials=1000, null_block_size_cadences=24, fap_levels=[0.01], random_seed=123, minimum_success_fraction=0.95, max_workers=None, worker_chunksize=1, show_progress=True)

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

def resolve_worker_count(args):
    available_cpus = os.cpu_count() or 1
    automatic_workers = max(1, min(6, available_cpus - 1 if available_cpus > 1 else 1))
    requested_workers = automatic_workers if args.max_workers is None else int(args.max_workers)
    return max(1, min(requested_workers, available_cpus, int(args.n_null_trials)))

def initialize_worker(time, base_series, period_grid, duration_grid, config):
    global _WORKER_TIME
    global _WORKER_BASE_SERIES
    global _WORKER_PERIOD_GRID
    global _WORKER_DURATION_GRID
    global _WORKER_CONFIG
    _WORKER_TIME = np.asarray(time, dtype=float)
    _WORKER_BASE_SERIES = np.asarray(base_series, dtype=float)
    _WORKER_PERIOD_GRID = np.asarray(period_grid, dtype=float)
    _WORKER_DURATION_GRID = np.asarray(duration_grid, dtype=float)
    _WORKER_CONFIG = dict(config)

def run_null_trial(task):
    trial, seed = task
    try:
        rng = np.random.default_rng(int(seed))
        surrogate = moving_block_surrogate(_WORKER_BASE_SERIES, block_size=_WORKER_CONFIG["null_block_size_cadences"], rng=rng)
        if _WORKER_CONFIG["surrogate_source"] == "flux":
            arima_result = fit_arima_innovations(surrogate, order=_WORKER_CONFIG["arima_order"], maxiter=_WORKER_CONFIG["fit_maxiter"])
            innovations = arima_result["innovations"]
            arima_converged = bool(arima_result["summary"]["converged"])
            arima_refitted = True
        else:
            innovations = np.asarray(surrogate, dtype=float)
            arima_converged = bool(_WORKER_CONFIG["base_arima_converged"])
            arima_refitted = False
        tcf_result = run_tcf(_WORKER_TIME, innovations, _WORKER_PERIOD_GRID, _WORKER_DURATION_GRID, edge_width_cadences=_WORKER_CONFIG["edge_width_cadences"], min_edge_observations=_WORKER_CONFIG["min_edge_observations"], min_transit_events=_WORKER_CONFIG["min_transit_events"], min_event_consistency_fraction=_WORKER_CONFIG["min_event_consistency_fraction"], top_k=_WORKER_CONFIG["top_k"], search_mode=_WORKER_CONFIG["search_mode"], n_coarse_periods=_WORKER_CONFIG["n_coarse_periods"], n_refinement_regions=_WORKER_CONFIG["n_refinement_regions"], refinement_half_width_points=_WORKER_CONFIG["refinement_half_width_points"])
        best = tcf_result["summary"]
        search_summary = tcf_result["search_summary"]
        return {"trial": int(trial), "trial_seed": int(seed), "success": True, "arima_refitted": arima_refitted, "arima_converged": arima_converged, "max_score": float(best["score"]), "best_raw_pooled_score": float(best["raw_pooled_score"]), "best_period": float(best["period"]), "best_duration": float(best["duration"]), "best_epoch": float(best["epoch"]), "best_valid_transit_events": int(best["n_valid_transit_events"]), "best_positive_transit_events": int(best["n_positive_transit_events"]), "best_positive_event_fraction": float(best["positive_event_fraction"]), "best_median_event_score": float(best["median_event_score"]), "requested_period_count": int(search_summary["requested_period_count"]), "evaluated_period_count": int(search_summary["evaluated_period_count"]), "coarse_period_count": int(search_summary["coarse_period_count"]), "refined_period_count": int(search_summary["refined_period_count"]), "error": ""}
    except Exception as exc:
        return {"trial": int(trial), "trial_seed": int(seed), "success": False, "arima_refitted": False, "arima_converged": False, "max_score": np.nan, "best_raw_pooled_score": np.nan, "best_period": np.nan, "best_duration": np.nan, "best_epoch": np.nan, "best_valid_transit_events": np.nan, "best_positive_transit_events": np.nan, "best_positive_event_fraction": np.nan, "best_median_event_score": np.nan, "requested_period_count": np.nan, "evaluated_period_count": np.nan, "coarse_period_count": np.nan, "refined_period_count": np.nan, "error": f"{type(exc).__name__}: {exc}"}

def create_trial_tasks(args):
    root_sequence = np.random.SeedSequence(int(args.random_seed))
    child_sequences = root_sequence.spawn(int(args.n_null_trials))
    seeds = [int(sequence.generate_state(1, dtype=np.uint64)[0]) for sequence in child_sequences]
    return [(trial, seeds[trial]) for trial in range(int(args.n_null_trials))]

def run_parallel_trials(time, base_series, period_grid, duration_grid, base_arima_converged, args):
    worker_count = resolve_worker_count(args)
    worker_config = {"surrogate_source": str(args.surrogate_source), "base_arima_converged": bool(base_arima_converged), "arima_order": tuple(args.arima_order), "fit_maxiter": int(args.fit_maxiter), "null_block_size_cadences": int(args.null_block_size_cadences), "edge_width_cadences": int(args.edge_width_cadences), "min_edge_observations": int(args.min_edge_observations), "min_transit_events": int(args.min_transit_events), "min_event_consistency_fraction": float(args.min_event_consistency_fraction), "top_k": int(args.top_k), "search_mode": str(args.search_mode), "n_coarse_periods": int(args.n_coarse_periods), "n_refinement_regions": int(args.n_refinement_regions), "refinement_half_width_points": int(args.refinement_half_width_points)}
    tasks = create_trial_tasks(args)
    multiprocessing_context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=multiprocessing_context, initializer=initialize_worker, initargs=(time, base_series, period_grid, duration_grid, worker_config)) as executor:
        mapped_rows = executor.map(run_null_trial, tasks, chunksize=int(args.worker_chunksize))
        rows = list(tqdm(mapped_rows, total=len(tasks), desc="TCF null trials", disable=not args.show_progress))
    return rows, worker_count

def prepare_surrogate_source(flux, args):
    if args.surrogate_source not in {"innovations", "flux"}:
        raise ValueError("surrogate_source must be innovations or flux.")
    if args.surrogate_source == "flux":
        return np.asarray(flux, dtype=float), True, None
    base_arima = fit_arima_innovations(flux, order=args.arima_order, maxiter=args.fit_maxiter)
    base_series = np.asarray(base_arima["innovations"], dtype=float)
    base_converged = bool(base_arima["summary"]["converged"])
    return base_series, base_converged, base_arima["summary"]

def calibrate_tcf(args):
    light_curve = load_kepler_pdcsap(args.target_id, args.quarter)
    regular, preprocessing = preprocess_pdcsap_light_curve(light_curve.to_dataframe(), quality_policy=args.quality_policy, require_finite_flux_error=args.require_finite_flux_error, normalization_fit_fraction=1.0 - args.test_fraction)
    time = regular["time"].to_numpy(dtype=float)
    flux = regular["normalized_flux"].to_numpy(dtype=float)
    period_grid = default_period_grid(time, min_period_days=args.min_period_days, max_period_days=args.max_period_days, n_periods=args.n_periods)
    duration_grid = default_duration_grid(args.min_duration_hours, args.max_duration_hours, args.n_durations)
    base_series, base_arima_converged, base_arima_summary = prepare_surrogate_source(flux, args)
    rows, worker_count = run_parallel_trials(time, base_series, period_grid, duration_grid, base_arima_converged, args)
    trials = pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)
    successful = trials[trials["success"] & np.isfinite(trials["max_score"])].copy()
    successful_scores = successful["max_score"].to_numpy(dtype=float)
    success_fraction = float(successful_scores.size / args.n_null_trials)
    if success_fraction < args.minimum_success_fraction:
        error_counts = trials.loc[~trials["success"], "error"].value_counts().head(10)
        raise RuntimeError(f"Only {success_fraction:.3f} of TCF null trials succeeded.\n{error_counts.to_string()}")
    threshold_rows = []
    for level in args.fap_levels:
        threshold = float(np.quantile(successful_scores, 1.0 - float(level), method="higher"))
        threshold_rows.append({"fap_level": float(level), "score_threshold": threshold, "requested_null_trials": int(args.n_null_trials), "successful_null_trials": int(successful_scores.size), "observed_exceedance_fraction": float(np.mean(successful_scores >= threshold))})
    thresholds = pd.DataFrame(threshold_rows)
    summary = {"target_id": str(args.target_id), "quarter": int(args.quarter), "quality_policy": args.quality_policy, "calibration_scope": "detector_conditional_on_fitted_arima" if args.surrogate_source == "innovations" else "full_pipeline_with_arima_refit", "surrogate_source": str(args.surrogate_source), "arima_order": tuple(args.arima_order), "base_arima_summary": base_arima_summary, "n_periods": int(args.n_periods), "n_durations": int(args.n_durations), "search_mode": str(args.search_mode), "n_coarse_periods": int(args.n_coarse_periods), "n_refinement_regions": int(args.n_refinement_regions), "refinement_half_width_points": int(args.refinement_half_width_points), "min_transit_events": int(args.min_transit_events), "min_event_consistency_fraction": float(args.min_event_consistency_fraction), "requested_null_trials": int(args.n_null_trials), "successful_null_trials": int(successful_scores.size), "success_fraction": success_fraction, "parallel_worker_count": int(worker_count), "worker_chunksize": int(args.worker_chunksize), "null_block_size_cadences": int(args.null_block_size_cadences), "random_seed": int(args.random_seed), "arima_convergence_rate": float(successful["arima_converged"].mean()), "median_max_score": float(successful["max_score"].median()), "maximum_null_score": float(successful["max_score"].max()), "median_raw_pooled_score": float(successful["best_raw_pooled_score"].median()), "median_best_period_days": float(successful["best_period"].median()), "median_best_duration_hours": float(successful["best_duration"].median() * 24.0), "median_valid_transit_events": float(successful["best_valid_transit_events"].median()), "median_positive_event_fraction": float(successful["best_positive_event_fraction"].median()), "median_event_score": float(successful["best_median_event_score"].median()), "median_evaluated_period_count": float(successful["evaluated_period_count"].median()), "fap_levels": list(args.fap_levels), "thresholds": threshold_rows}
    return regular, trials, thresholds, summary

def main(args=None):
    args = args or default_settings()
    metrics_dir = Path(args.output_dir) / "metrics"
    processed_dir = Path(args.output_dir) / "processed"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    print(f"Calibration scope: {args.surrogate_source}")
    print(f"Parallel workers: {resolve_worker_count(args)}")
    print(f"Null trials: {args.n_null_trials}")
    print(f"Requested periods: {args.n_periods}")
    print(f"Coarse periods: {args.n_coarse_periods}")
    print(f"Refinement regions: {args.n_refinement_regions}")
    regular, trials, thresholds, summary = calibrate_tcf(args)
    prefix = f"kic_{str(args.target_id).replace('KIC', '').strip()}_q{args.quarter}"
    trials_path = metrics_dir / f"{prefix}_tcf_null_trials.csv"
    thresholds_path = metrics_dir / f"{prefix}_tcf_fap_thresholds.csv"
    summary_path = metrics_dir / f"{prefix}_tcf_null_calibration_summary.json"
    trials.to_csv(trials_path, index=False)
    thresholds.to_csv(thresholds_path, index=False)
    regular.to_parquet(processed_dir / f"{prefix}_tcf_null_calibration_input.parquet", index=False)
    summary_path.write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    print(f"TCF null trials: {trials_path}")
    print(f"TCF FAP thresholds: {thresholds_path}")
    print(f"TCF null-calibration summary: {summary_path}")
    print("\nEvent-consistent TCF FAP thresholds:\n")
    print(thresholds.to_string(index=False))
    print("\nCalibration diagnostics:\n")
    print(f"Calibration scope: {summary['calibration_scope']}")
    print(f"Workers used: {summary['parallel_worker_count']}")
    print(f"Successful trials: {summary['successful_null_trials']}/{summary['requested_null_trials']}")
    print(f"Median periods evaluated per trial: {summary['median_evaluated_period_count']:.0f}")
    print(f"Median maximum score: {summary['median_max_score']:.6f}")
    print(f"Maximum null score: {summary['maximum_null_score']:.6f}")
    print(f"Median selected period: {summary['median_best_period_days']:.6f} days")
    print(f"Median valid events: {summary['median_valid_transit_events']:.1f}")
    print(f"Median positive-event fraction: {summary['median_positive_event_fraction']:.3f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())