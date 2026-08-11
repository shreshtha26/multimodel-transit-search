"""Star-level FAP calibration and master-table assembly for the multi-star challenger benchmark."""
import argparse
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import json
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import Manager, get_context
from pathlib import Path
from queue import Empty
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from adaptive_transit.detection.false_alarm import moving_block_surrogate
from adaptive_transit.detection.tcf import default_duration_grid, default_period_grid
from adaptive_transit.noise_models.kalman import fit_kalman_local_level
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve
from scripts.run_multistar_challenger_benchmark import DEFAULT_PIPELINES, PIPELINE_DEFINITIONS, TQDM_BAR_FORMAT, config_signature, default_settings, fit_gp, json_ready, lag_one_acf, load_light_curve_frame, load_manifest, load_or_fit_base_arima, normalize_target_id, robust_scale, run_bls_search, run_tcf_search, star_prefix
BACKGROUND_FEATURE_PATH = PROJECT_ROOT / "outputs/experiments/multistar_background_timescale/metrics/multistar_background_timescale_features.csv"

def parse_pipelines(value):
    pipelines = tuple(item.strip() for item in str(value).split(",") if item.strip())
    invalid = sorted(set(pipelines).difference(PIPELINE_DEFINITIONS))
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown pipeline(s): {invalid}. Valid pipelines are {sorted(PIPELINE_DEFINITIONS)}.")
    return pipelines

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate per-star FAP thresholds for the multi-star challenger benchmark.")
    parser.add_argument("--profile", choices=("pilot", "main", "smoke"), default="pilot")
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--background-feature-path", type=Path, default=BACKGROUND_FEATURE_PATH)
    parser.add_argument("--n-null-trials-per-star", type=int, default=100)
    parser.add_argument("--fap-level", type=float, default=0.01)
    parser.add_argument("--null-block-size-cadences", type=int, default=24)
    parser.add_argument("--minimum-success-fraction", type=float, default=0.90)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--pipelines", type=parse_pipelines)
    parser.add_argument("--no-download", dest="allow_download", action="store_false")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--rerun-failures", action="store_true")
    parsed = parser.parse_args(argv)
    args = default_settings(parsed.profile)
    args.benchmark_dir = Path(parsed.benchmark_dir) if parsed.benchmark_dir is not None else Path(args.output_dir)
    args.output_dir = args.benchmark_dir
    args.background_feature_path = Path(parsed.background_feature_path)
    args.n_null_trials_per_star = int(parsed.n_null_trials_per_star)
    args.fap_level = float(parsed.fap_level)
    args.null_block_size_cadences = int(parsed.null_block_size_cadences)
    args.minimum_success_fraction = float(parsed.minimum_success_fraction)
    args.max_workers = int(parsed.max_workers) if parsed.max_workers is not None else int(args.max_workers)
    args.pipelines = parsed.pipelines if parsed.pipelines is not None else tuple(DEFAULT_PIPELINES)
    args.allow_download = bool(parsed.allow_download)
    args.resume = bool(parsed.resume)
    args.rerun_failures = bool(parsed.rerun_failures)
    return args

def calibration_signature(args):
    signature = config_signature(args)
    signature.update({"n_null_trials_per_star": int(args.n_null_trials_per_star), "fap_level": float(args.fap_level), "null_block_size_cadences": int(args.null_block_size_cadences), "pipelines": tuple(args.pipelines)})
    return signature

def branch_names(pipelines):
    return sorted({PIPELINE_DEFINITIONS[pipeline][0] for pipeline in pipelines})

def branch_detector_groups(pipelines):
    groups = {}
    for pipeline in pipelines:
        branch, detector = PIPELINE_DEFINITIONS[pipeline]
        groups.setdefault(branch, []).append((pipeline, detector))
    return groups

def report_progress(progress_queue, target_id, quarter, stage, units=0, detail=""):
    if progress_queue is None:
        return
    try:
        progress_queue.put({"target_id": normalize_target_id(target_id), "quarter": int(quarter), "stage": str(stage), "units": int(units), "detail": str(detail)}, block=False)
    except Exception:
        pass

def star_calibration_dir(args, target_id, quarter):
    return Path(args["benchmark_dir"]) / "star_calibration" / star_prefix(target_id, quarter)

def star_calibration_config_matches(star_dir, args):
    path = Path(star_dir) / "calibration_config.json"
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("calibration_signature") == json_ready(calibration_signature(SimpleNamespace(**args) if isinstance(args, dict) else args))
    except Exception:
        return False

def create_trial_seeds(args, target_id, quarter):
    root = np.random.SeedSequence([int(args["random_seed"]), int(normalize_target_id(target_id)), int(quarter), 3917])
    children = root.spawn(int(args["n_null_trials_per_star"]))
    return [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children]

def fit_branch_series(time, flux, star_dir, target_id, quarter, args):
    series = {}
    model_fields = {}
    if "raw" in branch_names(args["pipelines"]):
        series["raw"] = flux
    if "arima" in branch_names(args["pipelines"]):
        base_arima, runtime, source = load_or_fit_base_arima(star_dir, target_id, quarter, flux, args)
        series["arima"] = base_arima["innovations"]
        model_fields.update({"base_arima_source": source, "base_arima_runtime_seconds": runtime, "base_arima_converged": bool(base_arima["summary"].get("converged", True))})
    if "kalman" in branch_names(args["pipelines"]):
        started = perf_counter()
        model = fit_kalman_local_level(flux, maxiter=args["kalman_maxiter"], burn_in=args["kalman_burn_in"])
        series["kalman"] = model.residuals
        model_fields.update({"base_kalman_runtime_seconds": float(perf_counter() - started), "base_kalman_converged": bool(model.converged), "base_kalman_process_variance": float(model.parameters["process_variance"]), "base_kalman_measurement_variance": float(model.parameters["measurement_variance"])})
    if "gp" in branch_names(args["pipelines"]):
        started = perf_counter()
        model = fit_gp(time, flux, args)
        series["gp"] = model.residuals
        model_fields.update({"base_gp_runtime_seconds": float(perf_counter() - started), "base_gp_converged": bool(model.converged), "base_gp_length_scale_days": float(model.parameters["length_scale_days"]), "base_gp_training_point_count": int(model.parameters["training_point_count"])})
    for branch, values in series.items():
        model_fields[f"{branch}_base_residual_std"] = float(np.nanstd(values, ddof=1))
        model_fields[f"{branch}_base_residual_acf1"] = lag_one_acf(values)
    return series, model_fields

def run_pipeline_on_surrogate(pipeline, detector, time, surrogate, period_grid, duration_grid, args):
    result = run_bls_search(time, surrogate, period_grid, duration_grid, args) if detector == "bls" else run_tcf_search(time, surrogate, period_grid, duration_grid, args)
    best = result["summary"]
    if detector == "bls":
        return {"score": float(best["sde"]), "power": float(best["power"]), "period_days": float(best["period_days"]), "duration_hours": float(best["duration_days"] * 24.0), "epoch_days": float(best["transit_time"])}
    return {"score": float(best["score"]), "raw_pooled_score": float(best["raw_pooled_score"]), "period_days": float(best["period"]), "duration_hours": float(best["duration"] * 24.0), "epoch_days": float(best["epoch"]), "valid_transit_events": int(best["n_valid_transit_events"]), "positive_event_fraction": float(best["positive_event_fraction"])}

def run_one_null_trial(trial, seed, target_id, quarter, time, branch_series, period_grid, duration_grid, args):
    rng = np.random.default_rng(int(seed))
    groups = branch_detector_groups(args["pipelines"])
    row = {"target_id": normalize_target_id(target_id), "quarter": int(quarter), "trial": int(trial), "trial_seed": int(seed)}
    for branch, members in groups.items():
        try:
            surrogate = moving_block_surrogate(branch_series[branch], block_size=args["null_block_size_cadences"], rng=rng)
            row[f"{branch}_null_std"] = float(np.nanstd(surrogate, ddof=1))
            for pipeline, detector in members:
                try:
                    result = run_pipeline_on_surrogate(pipeline, detector, time, surrogate, period_grid, duration_grid, args)
                    row[f"{pipeline}_success"] = True
                    row[f"{pipeline}_score"] = float(result["score"])
                    row[f"{pipeline}_best_period_days"] = float(result["period_days"])
                    row[f"{pipeline}_best_duration_hours"] = float(result["duration_hours"])
                    row[f"{pipeline}_best_epoch_days"] = float(result["epoch_days"])
                    row[f"{pipeline}_error"] = ""
                    if "power" in result:
                        row[f"{pipeline}_power"] = float(result["power"])
                    if "raw_pooled_score" in result:
                        row[f"{pipeline}_raw_pooled_score"] = float(result["raw_pooled_score"])
                        row[f"{pipeline}_valid_transit_events"] = int(result["valid_transit_events"])
                        row[f"{pipeline}_positive_event_fraction"] = float(result["positive_event_fraction"])
                except Exception as exc:
                    row[f"{pipeline}_success"] = False
                    row[f"{pipeline}_score"] = float("nan")
                    row[f"{pipeline}_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            for pipeline, detector in members:
                row[f"{pipeline}_success"] = False
                row[f"{pipeline}_score"] = float("nan")
                row[f"{pipeline}_error"] = f"{type(exc).__name__}: {exc}"
    return row

def threshold_rows(trials, args):
    rows = []
    for pipeline in args["pipelines"]:
        success_column = f"{pipeline}_success"
        score_column = f"{pipeline}_score"
        successful = trials[trials[success_column].fillna(False).astype(bool)] if success_column in trials.columns else pd.DataFrame()
        scores = pd.to_numeric(successful[score_column], errors="coerce").dropna().to_numpy(dtype=float) if score_column in successful.columns else np.asarray([], dtype=float)
        if scores.size:
            threshold = float(np.quantile(scores, 1.0 - float(args["fap_level"]), method="higher"))
            observed = float(np.mean(scores >= threshold))
        else:
            threshold = float("nan")
            observed = float("nan")
        rows.append({"pipeline": pipeline, "branch": PIPELINE_DEFINITIONS[pipeline][0], "detector": PIPELINE_DEFINITIONS[pipeline][1], "target_id": normalize_target_id(trials["target_id"].iloc[0]) if len(trials) else "", "quarter": int(trials["quarter"].iloc[0]) if len(trials) else -1, "fap_level": float(args["fap_level"]), "score_column": score_column, "score_threshold": threshold, "requested_null_trials": int(args["n_null_trials_per_star"]), "successful_null_trials": int(scores.size), "success_fraction": float(scores.size / int(args["n_null_trials_per_star"])) if int(args["n_null_trials_per_star"]) else float("nan"), "observed_exceedance_fraction": observed})
    return pd.DataFrame(rows)

def load_completed_null_rows(path, seeds, args):
    if not args.get("resume", True) or not Path(path).exists():
        return [], set()
    frame = pd.read_csv(path)
    if "trial" not in frame.columns or "trial_seed" not in frame.columns:
        return [], set()
    seed_map = {int(index): int(seed) for index, seed in enumerate(seeds)}
    rows = []
    completed = set()
    for row in frame.to_dict(orient="records"):
        trial = int(row["trial"])
        if trial in seed_map and int(row["trial_seed"]) == seed_map[trial] and trial not in completed:
            rows.append(row)
            completed.add(trial)
    return rows, completed

def save_null_rows(path, rows):
    frame = pd.DataFrame(sorted(rows, key=lambda row: int(row["trial"]))) if rows else pd.DataFrame()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame

def run_star_calibration(task):
    row, args, progress_queue = task
    target_id = normalize_target_id(row["target_id"])
    quarter = int(row["quarter"])
    selection_group = str(row.get("selection_group", "unspecified"))
    star_dir = Path(args["benchmark_dir"]) / "stars" / star_prefix(target_id, quarter)
    calibration_dir = star_calibration_dir(args, target_id, quarter)
    calibration_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    try:
        complete_path = calibration_dir / "COMPLETE"
        if complete_path.exists():
            complete_path.unlink()
        (calibration_dir / "calibration_config.json").write_text(json.dumps({"calibration_signature": json_ready(calibration_signature(SimpleNamespace(**args)))}, indent=2) + "\n")
        light_curve_frame, cache_hit = load_light_curve_frame(target_id, quarter, args, progress_queue=progress_queue)
        regular, preprocessing = preprocess_pdcsap_light_curve(light_curve_frame, quality_policy=args["quality_policy"], require_finite_flux_error=args["require_finite_flux_error"], normalization_fit_fraction=1.0 - args["test_fraction"])
        time = regular["time"].to_numpy(dtype=float)
        flux = regular["normalized_flux"].to_numpy(dtype=float)
        period_grid = default_period_grid(time, min_period_days=args["min_period_days"], max_period_days=args["max_period_days"], n_periods=args["n_periods"])
        duration_grid = default_duration_grid(args["min_duration_hours"], args["max_duration_hours"], args["n_durations"])
        branch_series, model_fields = fit_branch_series(time, flux, star_dir, target_id, quarter, args)
        seeds = create_trial_seeds(args, target_id, quarter)
        null_path = calibration_dir / "null_trials.csv"
        rows, completed = load_completed_null_rows(null_path, seeds, args)
        if completed:
            report_progress(progress_queue, target_id, quarter, "nulls resumed", units=len(completed), detail=f"{len(completed)}/{len(seeds)}")
        for trial, seed in enumerate(seeds):
            if trial in completed:
                continue
            rows.append(run_one_null_trial(trial, seed, target_id, quarter, time, branch_series, period_grid, duration_grid, args))
            completed.add(trial)
            save_null_rows(null_path, rows)
            report_progress(progress_queue, target_id, quarter, "null trial", units=1, detail=f"{len(completed)}/{len(seeds)}")
        trials = save_null_rows(null_path, rows)
        thresholds = threshold_rows(trials, args)
        thresholds.to_csv(calibration_dir / "fap_thresholds.csv", index=False)
        low_success = thresholds[thresholds["success_fraction"] < float(args["minimum_success_fraction"])]
        status = "failed" if not low_success.empty else "success"
        summary = {"target_id": target_id, "quarter": quarter, "selection_group": selection_group, "status": status, "runtime_seconds": float(perf_counter() - started), "light_curve_cache_hit": bool(cache_hit), "requested_null_trials": int(args["n_null_trials_per_star"]), "completed_null_trials": int(len(trials)), "minimum_success_fraction": float(args["minimum_success_fraction"]), **model_fields}
        (calibration_dir / "calibration_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
        if status == "failed":
            raise RuntimeError(f"Too few successful null trials for {len(low_success)} pipeline(s).")
        (calibration_dir / "COMPLETE").write_text("complete\n")
        return {"target_id": target_id, "quarter": quarter, "selection_group": selection_group, "status": "success", "calibration_dir": str(calibration_dir), "runtime_seconds": summary["runtime_seconds"], "error": ""}
    except Exception as exc:
        failure = {"target_id": target_id, "quarter": quarter, "selection_group": selection_group, "status": "failed", "calibration_dir": str(calibration_dir), "runtime_seconds": float(perf_counter() - started), "error": f"{type(exc).__name__}: {exc}"}
        (calibration_dir / "failure.json").write_text(json.dumps(json_ready(failure), indent=2) + "\n")
        return failure

def completed_result(args, row):
    target_id = normalize_target_id(row["target_id"])
    quarter = int(row["quarter"])
    calibration_dir = star_calibration_dir(vars(args), target_id, quarter)
    try:
        summary = json.loads((calibration_dir / "calibration_summary.json").read_text())
    except Exception:
        summary = {}
    return {"target_id": target_id, "quarter": quarter, "selection_group": str(row.get("selection_group", "unspecified")), "status": "success", "calibration_dir": str(calibration_dir), "runtime_seconds": float(summary.get("runtime_seconds", np.nan)), "error": ""}

def failure_result(args, row):
    target_id = normalize_target_id(row["target_id"])
    quarter = int(row["quarter"])
    calibration_dir = star_calibration_dir(vars(args), target_id, quarter)
    try:
        failure = json.loads((calibration_dir / "failure.json").read_text())
    except Exception:
        failure = {}
    return {"target_id": target_id, "quarter": quarter, "selection_group": str(row.get("selection_group", "unspecified")), "status": "failed", "calibration_dir": str(calibration_dir), "runtime_seconds": float(failure.get("runtime_seconds", np.nan)), "error": str(failure.get("error", "Existing failure.json"))}

def resolve_worker_count(args, target_count):
    available = os.cpu_count() or 1
    requested = int(args.max_workers) if args.max_workers is not None else max(1, available - 1)
    return max(1, min(requested, available, int(target_count)))

def settings_to_worker_dict(args):
    values = vars(args).copy()
    for key in ("manifest_path", "cache_dir", "output_dir", "benchmark_dir", "background_feature_path", "existing_arima_cache_root"):
        if key in values:
            values[key] = str(values[key])
    values["pipelines"] = tuple(values["pipelines"])
    values["arima_order"] = tuple(values["arima_order"])
    return values

def drain_progress_queue(progress_queue, progress):
    while True:
        try:
            event = progress_queue.get_nowait()
        except Empty:
            break
        except Exception:
            break
        units = max(0, int(event.get("units", 0)))
        if units:
            progress.update(units)
        progress.set_postfix_str(f"KIC {event.get('target_id')} Q{event.get('quarter')} {event.get('stage')} {event.get('detail', '')}".strip())

def run_pending_rows(pending_rows, worker_args, args):
    if not pending_rows:
        return []
    results = []
    context = get_context("spawn")
    worker_count = resolve_worker_count(args, len(pending_rows))
    total_trials = len(pending_rows) * int(args.n_null_trials_per_star)
    with Manager() as manager:
        progress_queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            future_map = {executor.submit(run_star_calibration, (row, worker_args, progress_queue)): row for row in pending_rows}
            pending = set(future_map)
            with tqdm(total=len(future_map), desc="Calibration stars", bar_format=TQDM_BAR_FORMAT, position=0) as star_progress, tqdm(total=total_trials, desc="Null trials", bar_format=TQDM_BAR_FORMAT, position=1) as trial_progress:
                while pending:
                    done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                    drain_progress_queue(progress_queue, trial_progress)
                    for future in done:
                        drain_progress_queue(progress_queue, trial_progress)
                        row = future_map[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = {"target_id": normalize_target_id(row["target_id"]), "quarter": int(row["quarter"]), "selection_group": str(row.get("selection_group", "unspecified")), "status": "failed", "calibration_dir": "", "runtime_seconds": float("nan"), "error": f"{type(exc).__name__}: {exc}"}
                        results.append(result)
                        star_progress.set_postfix_str(f"{result['status']} KIC {result['target_id']} Q{result['quarter']}")
                        star_progress.update(1)
                drain_progress_queue(progress_queue, trial_progress)
    return results

def load_calibration_outputs(task_results):
    trials = []
    thresholds = []
    summaries = []
    for result in task_results:
        if result["status"] != "success":
            continue
        directory = Path(result["calibration_dir"])
        trials.append(pd.read_csv(directory / "null_trials.csv", dtype={"target_id": str}))
        thresholds.append(pd.read_csv(directory / "fap_thresholds.csv", dtype={"target_id": str}))
        summaries.append(json.loads((directory / "calibration_summary.json").read_text()))
    return pd.concat(trials, ignore_index=True) if trials else pd.DataFrame(), pd.concat(thresholds, ignore_index=True) if thresholds else pd.DataFrame(), pd.DataFrame(summaries)

def load_benchmark_injections(args):
    path = Path(args.benchmark_dir) / "metrics/multistar_challenger_injections.csv"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark injection table is missing: {path}")
    frame = pd.read_csv(path, dtype={"target_id": str})
    frame["target_id"] = frame["target_id"].map(normalize_target_id)
    frame["quarter"] = pd.to_numeric(frame["quarter"], errors="raise").astype(int)
    return frame

def empirical_p_values(scores, null_scores):
    scores = pd.to_numeric(pd.Series(scores), errors="coerce").to_numpy(dtype=float)
    null_scores = pd.to_numeric(pd.Series(null_scores), errors="coerce").dropna().to_numpy(dtype=float)
    return np.asarray([(np.sum(null_scores >= score) + 1.0) / (len(null_scores) + 1.0) if np.isfinite(score) and len(null_scores) else np.nan for score in scores], dtype=float)

def apply_calibration(injections, null_trials, thresholds, pipelines):
    result = injections.copy()
    for pipeline in pipelines:
        threshold_map = thresholds[thresholds["pipeline"] == pipeline].set_index(["target_id", "quarter"])["score_threshold"].to_dict()
        result[f"{pipeline}_star_fap_threshold"] = [threshold_map.get((normalize_target_id(row["target_id"]), int(row["quarter"])), np.nan) for row in result.to_dict(orient="records")]
        p_values = []
        for (target_id, quarter), group in result.groupby(["target_id", "quarter"], sort=False):
            null_subset = null_trials[(null_trials["target_id"].map(normalize_target_id) == normalize_target_id(target_id)) & (pd.to_numeric(null_trials["quarter"], errors="coerce") == int(quarter))]
            p_values.extend(empirical_p_values(group[f"{pipeline}_score"], null_subset[f"{pipeline}_score"] if f"{pipeline}_score" in null_subset.columns else []))
        result[f"{pipeline}_star_empirical_p_value"] = p_values
        threshold = pd.to_numeric(result[f"{pipeline}_star_fap_threshold"], errors="coerce")
        score = pd.to_numeric(result[f"{pipeline}_score"], errors="coerce")
        result[f"{pipeline}_passes_star_fap"] = score >= threshold
        result[f"{pipeline}_score_over_star_fap_threshold"] = np.divide(score, threshold, out=np.full(len(result), np.nan, dtype=float), where=np.isfinite(threshold) & (threshold != 0))
        result[f"{pipeline}_harmonic_recovered_star_fap"] = result[f"{pipeline}_harmonic_rank1_matched"].fillna(False).astype(bool) & result[f"{pipeline}_passes_star_fap"].fillna(False).astype(bool)
        result[f"{pipeline}_exact_recovered_star_fap"] = result[f"{pipeline}_exact_rank1_matched"].fillna(False).astype(bool) & result[f"{pipeline}_passes_star_fap"].fillna(False).astype(bool)
    return result

def add_background_features(master, args):
    path = Path(args.background_feature_path)
    if not path.exists():
        return master
    features = pd.read_csv(path, dtype={"target_id": str})
    features["target_id"] = features["target_id"].map(normalize_target_id)
    features["quarter"] = pd.to_numeric(features["quarter"], errors="coerce").astype("Int64")
    join_keys = ["target_id", "quarter", "selection_group"] if "selection_group" in features.columns and "selection_group" in master.columns else ["target_id", "quarter"]
    result = master.merge(features, on=join_keys, how="left", validate="many_to_one", suffixes=("", "_background"))
    duration_days = result["injected_duration_hours"].astype(float) / 24.0
    for tau_column in ("background_tau_acf_e_days", "background_tau_acf_half_days", "background_tau_integrated_positive_acf_days"):
        if tau_column in result.columns:
            result[tau_column.replace("background_tau_", "background_to_transit_")] = result[tau_column].astype(float) / duration_days
    return result

def bool_union(frame, pipelines, suffix):
    values = pd.Series(False, index=frame.index)
    for pipeline in pipelines:
        values = values | frame[f"{pipeline}_{suffix}"].fillna(False).astype(bool)
    return values

def pipeline_summary(master, pipelines):
    rows = []
    for pipeline in pipelines:
        harmonic = master[f"{pipeline}_harmonic_recovered_star_fap"].fillna(False).astype(bool)
        exact = master[f"{pipeline}_exact_recovered_star_fap"].fillna(False).astype(bool)
        passes = master[f"{pipeline}_passes_star_fap"].fillna(False).astype(bool)
        rows.append({"pipeline": pipeline, "branch": PIPELINE_DEFINITIONS[pipeline][0], "detector": PIPELINE_DEFINITIONS[pipeline][1], "injection_count": int(len(master)), "detection_count_star_fap": int(passes.sum()), "detection_rate_star_fap": float(passes.mean()), "harmonic_recovery_count_star_fap": int(harmonic.sum()), "harmonic_recovery_rate_star_fap": float(harmonic.mean()), "exact_recovery_count_star_fap": int(exact.sum()), "exact_recovery_rate_star_fap": float(exact.mean()), "median_score_over_star_fap_threshold": float(pd.to_numeric(master[f"{pipeline}_score_over_star_fap_threshold"], errors="coerce").median())})
    return pd.DataFrame(rows)

def combination_summary(master, pipelines):
    named = [("raw_bls_union_gp_tcf", [pipeline for pipeline in ("raw_bls", "gp_tcf") if pipeline in pipelines]), ("existing_bls_tcf", [pipeline for pipeline in ("raw_bls", "arima_tcf") if pipeline in pipelines]), ("non_gp_union", [pipeline for pipeline in ("raw_bls", "arima_tcf", "kalman_bls", "kalman_tcf") if pipeline in pipelines]), ("gp_union", [pipeline for pipeline in ("gp_bls", "gp_tcf") if pipeline in pipelines]), ("all_pipelines", list(pipelines))]
    rows = []
    for name, members in named:
        if not members:
            continue
        harmonic = bool_union(master, members, "harmonic_recovered_star_fap")
        exact = bool_union(master, members, "exact_recovered_star_fap")
        rows.append({"combination": name, "pipelines": ",".join(members), "injection_count": int(len(master)), "harmonic_recovery_count_star_fap": int(harmonic.sum()), "harmonic_recovery_rate_star_fap": float(harmonic.mean()), "exact_recovery_count_star_fap": int(exact.sum()), "exact_recovery_rate_star_fap": float(exact.mean())})
    return pd.DataFrame(rows)

def grouped_summary(master, column, pipelines):
    aggregations = {"injection_count": ("target_id", "size"), "star_count": ("target_id", "nunique")}
    for pipeline in pipelines:
        aggregations[f"{pipeline}_harmonic_recovery_rate_star_fap"] = (f"{pipeline}_harmonic_recovered_star_fap", "mean")
        aggregations[f"{pipeline}_exact_recovery_rate_star_fap"] = (f"{pipeline}_exact_recovered_star_fap", "mean")
    return master.groupby(column, dropna=False, observed=False, as_index=False).agg(**aggregations)

def save_global_outputs(args, task_results, null_trials, thresholds, calibration_summaries):
    metrics_dir = Path(args.benchmark_dir) / "metrics"
    injections = load_benchmark_injections(args)
    master = apply_calibration(injections, null_trials, thresholds, args.pipelines)
    master = add_background_features(master, args)
    pipelines = tuple(args.pipelines)
    pipeline_table = pipeline_summary(master, pipelines)
    combination_table = combination_summary(master, pipelines)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(task_results).to_csv(metrics_dir / "multistar_challenger_calibration_status.csv", index=False)
    null_trials.to_csv(metrics_dir / "multistar_challenger_star_null_trials.csv", index=False)
    thresholds.to_csv(metrics_dir / "multistar_challenger_star_fap_thresholds.csv", index=False)
    calibration_summaries.to_csv(metrics_dir / "multistar_challenger_calibration_star_summary.csv", index=False)
    master.to_csv(metrics_dir / "multistar_challenger_master_results.csv", index=False)
    pipeline_table.to_csv(metrics_dir / "multistar_challenger_star_fap_pipeline_summary.csv", index=False)
    combination_table.to_csv(metrics_dir / "multistar_challenger_star_fap_combinations.csv", index=False)
    grouped_summary(master, "injected_depth", pipelines).to_csv(metrics_dir / "multistar_challenger_star_fap_by_depth.csv", index=False)
    grouped_summary(master, "injected_duration_hours", pipelines).to_csv(metrics_dir / "multistar_challenger_star_fap_by_duration.csv", index=False)
    grouped_summary(master, "injected_period_days", pipelines).to_csv(metrics_dir / "multistar_challenger_star_fap_by_period.csv", index=False)
    if "background_to_transit_acf_e_days" in master.columns:
        master["background_ratio_bin"] = pd.qcut(pd.to_numeric(master["background_to_transit_acf_e_days"], errors="coerce"), q=4, duplicates="drop")
        grouped_summary(master, "background_ratio_bin", pipelines).to_csv(metrics_dir / "multistar_challenger_star_fap_by_background_ratio.csv", index=False)
    all_harmonic = bool_union(master, pipelines, "harmonic_recovered_star_fap")
    summary = {"profile": str(args.profile), "target_count": int(len(calibration_summaries)), "successful_calibration_target_count": int((pd.DataFrame(task_results)["status"] == "success").sum()) if task_results else 0, "failed_calibration_target_count": int((pd.DataFrame(task_results)["status"] != "success").sum()) if task_results else 0, "injection_count": int(len(master)), "requested_null_trials_per_star": int(args.n_null_trials_per_star), "pooled_null_trial_rows": int(len(null_trials)), "fap_level": float(args.fap_level), "pipelines": list(pipelines), "calibration_scope": "per-star branch-conditional moving-block null scores; no KIC 11904151 numeric threshold reuse", "all_pipeline_harmonic_recovery_rate_star_fap": float(all_harmonic.mean()) if len(all_harmonic) else float("nan")}
    (metrics_dir / "multistar_challenger_star_fap_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    return metrics_dir, summary

def main(args=None):
    args = args or parse_args()
    manifest = pd.read_csv(Path(args.benchmark_dir) / "metrics/target_manifest_used.csv") if (Path(args.benchmark_dir) / "metrics/target_manifest_used.csv").exists() else load_manifest(args)
    manifest["target_id"] = manifest["target_id"].map(normalize_target_id)
    manifest["quarter"] = pd.to_numeric(manifest["quarter"], errors="raise").astype(int)
    task_results = []
    pending_rows = []
    for row in manifest.to_dict(orient="records"):
        directory = star_calibration_dir(vars(args), row["target_id"], row["quarter"])
        if args.resume and (directory / "COMPLETE").exists() and star_calibration_config_matches(directory, args):
            task_results.append(completed_result(args, row))
        elif args.resume and not args.rerun_failures and (directory / "failure.json").exists() and star_calibration_config_matches(directory, args):
            task_results.append(failure_result(args, row))
        else:
            pending_rows.append(row)
    print(f"Benchmark directory: {Path(args.benchmark_dir)}")
    print(f"Targets requested: {len(manifest)}")
    print(f"Targets resumed: {len(task_results)}")
    print(f"Targets to calibrate: {len(pending_rows)}")
    print(f"Null trials per star: {args.n_null_trials_per_star}")
    print(f"Parallel star workers: {resolve_worker_count(args, len(manifest))}")
    print(f"Pipelines: {', '.join(args.pipelines)}")
    worker_args = settings_to_worker_dict(args)
    task_results.extend(run_pending_rows(pending_rows, worker_args, args))
    null_trials, thresholds, calibration_summaries = load_calibration_outputs(task_results)
    metrics_dir, summary = save_global_outputs(args, task_results, null_trials, thresholds, calibration_summaries)
    print(f"Metrics directory: {metrics_dir}")
    print(f"All-pipeline harmonic recovery at star FAP: {summary['all_pipeline_harmonic_recovery_rate_star_fap']:.3f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
