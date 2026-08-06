"""Generate top-k null/original candidates and calibrate the frozen reranker."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import run_multistar_bls_tcf as bench
import train_candidate_rerankers as rerank


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "outputs/experiments/multistar_bls_tcf/optimized"
DEFAULT_METRICS_DIR = DEFAULT_EXPERIMENT_DIR / "metrics"
DEFAULT_CALIBRATION_DIR = DEFAULT_EXPERIMENT_DIR / "reranker_topk_calibration"
DEFAULT_RERANKER_CONFIG_PATH = PROJECT_ROOT / "configs/candidate_reranker_clean_v1.json"
DEFAULT_MODEL_PATH = DEFAULT_EXPERIMENT_DIR / "models/clean_reranker_v1_xgboost_classifier.joblib"
TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} ({percentage:3.0f}%) [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
CALIBRATION_VERSION = "topk_null_original_calibration_v1"
CALIBRATION_GROUP_COLUMNS = ["dataset_kind", "target_id", "quarter", "trial"]


def default_settings():
    return SimpleNamespace(
        experiment_dir=DEFAULT_EXPERIMENT_DIR,
        metrics_dir=DEFAULT_METRICS_DIR,
        calibration_dir=DEFAULT_CALIBRATION_DIR,
        reranker_config=DEFAULT_RERANKER_CONFIG_PATH,
        model_path=DEFAULT_MODEL_PATH,
        target_limit=None,
        target_ids=None,
        top_k=10,
        null_trials_per_star=None,
        max_workers=6,
        random_seed=None,
        fap_level=0.01,
        score_feature_mode="frozen_v1",
        resume=True,
    )


def parse_target_ids(value):
    values = tuple(bench.normalize_target_id(part) for part in str(value).split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one target id.")
    return values


def build_parser():
    defaults = default_settings()
    parser = argparse.ArgumentParser(description="Run top-k null/original candidate generation for frozen reranker calibration.")
    parser.add_argument("--experiment-dir", type=Path, default=defaults.experiment_dir)
    parser.add_argument("--metrics-dir", type=Path, default=defaults.metrics_dir)
    parser.add_argument("--calibration-dir", type=Path, default=defaults.calibration_dir)
    parser.add_argument("--reranker-config", type=Path, default=defaults.reranker_config)
    parser.add_argument("--model-path", type=Path, default=defaults.model_path)
    parser.add_argument("--target-limit", type=int, default=defaults.target_limit)
    parser.add_argument("--target-ids", type=parse_target_ids, default=defaults.target_ids)
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--null-trials-per-star", type=int, default=defaults.null_trials_per_star)
    parser.add_argument("--max-workers", type=int, default=defaults.max_workers)
    parser.add_argument("--random-seed", type=int, default=defaults.random_seed)
    parser.add_argument("--fap-level", type=float, default=defaults.fap_level)
    parser.add_argument("--score-feature-mode", choices=("frozen_v1", "full_diagnostics"), default=defaults.score_feature_mode)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def bool_series(series):
    if series.dtype == bool:
        return series
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})


def period_fractional_error(first, second):
    if pd.isna(first) or pd.isna(second):
        return np.nan
    first = float(first)
    second = float(second)
    if second <= 0 or first <= 0 or not np.isfinite(first):
        return np.nan
    return float(abs(first - second) / second)


def detector_harmonic_error(candidate_period, reference_period):
    if pd.isna(candidate_period) or pd.isna(reference_period):
        return np.nan, np.nan
    factors = (0.5, 1.0, 2.0, 3.0)
    errors = {factor: period_fractional_error(candidate_period, float(reference_period) * factor) for factor in factors}
    best_factor = min(errors, key=errors.get)
    return float(errors[best_factor]), float(best_factor)


def periods_match(first, second, tolerance_fraction):
    if pd.isna(first) or pd.isna(second):
        return False
    first = float(first)
    second = float(second)
    denominator = min(abs(first), abs(second))
    return denominator > 0 and abs(first - second) / denominator <= float(tolerance_fraction)


def parse_run_summary(metrics_dir):
    path = Path(metrics_dir) / "multistar_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def runtime_benchmark_args(args, run_summary):
    profile = str(run_summary.get("profile", "optimized"))
    bench_args = bench.default_settings(profile)
    bench_args.output_dir = Path(args.experiment_dir)
    bench_args.cache_dir = bench.CACHE_DIR
    bench_args.top_k = int(args.top_k)
    if args.random_seed is not None:
        bench_args.random_seed = int(args.random_seed)
    elif "random_seed" in run_summary:
        bench_args.random_seed = int(run_summary["random_seed"])
    if args.null_trials_per_star is not None:
        bench_args.null_trials_per_star = int(args.null_trials_per_star)
    elif run_summary.get("successful_target_count") and run_summary.get("pooled_bls_null_count"):
        bench_args.null_trials_per_star = int(round(float(run_summary["pooled_bls_null_count"]) / float(run_summary["successful_target_count"])))
    if "n_periods" in run_summary:
        bench_args.n_periods = int(run_summary["n_periods"])
    if "n_coarse_periods" in run_summary:
        bench_args.n_coarse_periods = int(run_summary["n_coarse_periods"])
    if "n_refinement_regions" in run_summary:
        bench_args.n_refinement_regions = int(run_summary["n_refinement_regions"])
    if "refinement_half_width_points" in run_summary:
        bench_args.refinement_half_width_points = int(run_summary["refinement_half_width_points"])
    if "bls_oversample" in run_summary:
        bench_args.bls_oversample = int(run_summary["bls_oversample"])
    if "search_mode" in run_summary:
        bench_args.search_mode = str(run_summary["search_mode"])
    return bench_args


def calibration_config(args, bench_args, reranker_spec, config_sha256):
    return {
        "version": CALIBRATION_VERSION,
        "reranker_version": reranker_spec["version"],
        "reranker_config_sha256": config_sha256,
        "top_k": int(args.top_k),
        "score_feature_mode": str(args.score_feature_mode),
        "null_trials_per_star": int(bench_args.null_trials_per_star),
        "random_seed": int(bench_args.random_seed),
        "n_periods": int(bench_args.n_periods),
        "n_durations": int(bench_args.n_durations),
        "search_mode": str(bench_args.search_mode),
        "n_coarse_periods": int(bench_args.n_coarse_periods),
        "n_refinement_regions": int(bench_args.n_refinement_regions),
        "refinement_half_width_points": int(bench_args.refinement_half_width_points),
        "bls_objective": str(bench_args.bls_objective),
        "bls_oversample": int(bench_args.bls_oversample),
        "null_block_size_cadences": int(bench_args.null_block_size_cadences),
        "fap_level": float(args.fap_level),
    }


def load_targets(metrics_dir, args):
    path = Path(metrics_dir) / "target_manifest_used.csv"
    if not path.exists():
        path = bench.MANIFEST_PATH
    manifest = pd.read_csv(path, dtype={"target_id": str})
    manifest["target_id"] = manifest["target_id"].map(bench.normalize_target_id)
    manifest["quarter"] = pd.to_numeric(manifest["quarter"], errors="raise").astype(int)
    if "selection_group" not in manifest.columns:
        manifest["selection_group"] = "unspecified"
    summary_path = Path(metrics_dir) / "multistar_star_summary.csv"
    if summary_path.exists():
        star_summary = pd.read_csv(summary_path, dtype={"target_id": str})
        star_summary["target_id"] = star_summary["target_id"].map(bench.normalize_target_id)
        star_summary["quarter"] = pd.to_numeric(star_summary["quarter"], errors="raise").astype(int)
        columns = [
            "target_id",
            "quarter",
            "selection_group",
            "noise_quartile",
            "robust_flux_scatter_ppm",
            "gap_fraction",
            "lag_one_flux_acf",
            "six_hour_scatter_proxy_ppm",
            "base_arima_converged",
        ]
        manifest = manifest.drop(columns=[column for column in ["selection_group"] if column in manifest.columns])
        manifest = manifest.merge(star_summary[columns], on=["target_id", "quarter"], how="left", validate="one_to_one")
    if args.target_ids:
        wanted = {bench.normalize_target_id(target_id) for target_id in args.target_ids}
        manifest = manifest[manifest["target_id"].isin(wanted)].copy()
    if args.target_limit is not None:
        manifest = manifest.head(int(args.target_limit)).copy()
    if manifest.empty:
        raise ValueError("No targets selected for calibration generation.")
    return manifest.reset_index(drop=True)


def peak_period(peak, detector):
    if detector == "tcf":
        return float(peak.get("period_days", peak.get("period")))
    return float(peak["period_days"])


def tcf_peak_value(peak, column, default=np.nan):
    return peak[column] if column in peak and pd.notna(peak[column]) else default


def detector_peak_rows(peaks, detector, meta, trial_seed, error=""):
    rows = []
    if peaks is None or len(peaks) == 0:
        return rows
    for _, peak in peaks.iterrows():
        rank = int(peak.get("rank", len(rows) + 1))
        period = peak_period(peak, detector)
        base = {
            **meta,
            "trial_seed": trial_seed,
            "detector": detector,
            "rank": rank,
            "period_days": period,
            "error": error,
        }
        if detector == "bls":
            base.update(
                {
                    "score": float(peak["sde"]),
                    "bls_sde": float(peak["sde"]),
                    "bls_power": float(peak["power"]),
                    "bls_duration_hours": float(peak["duration_days"]) * 24.0,
                    "bls_transit_time": float(peak["transit_time"]),
                    "bls_depth": float(peak["depth"]),
                }
            )
        else:
            base.update(
                {
                    "score": float(peak["score"]),
                    "tcf_score": float(peak["score"]),
                    "tcf_raw_pooled_score": float(tcf_peak_value(peak, "raw_pooled_score")),
                    "tcf_epoch_days": float(tcf_peak_value(peak, "epoch")),
                    "tcf_duration_hours": float(tcf_peak_value(peak, "duration")) * 24.0,
                    "tcf_valid_transit_events": int(tcf_peak_value(peak, "n_valid_transit_events", 0)),
                    "tcf_positive_transit_events": int(tcf_peak_value(peak, "n_positive_transit_events", 0)),
                    "tcf_positive_event_fraction": float(tcf_peak_value(peak, "positive_event_fraction")),
                    "tcf_median_event_score": float(tcf_peak_value(peak, "median_event_score")),
                }
            )
        rows.append(base)
    return rows


def base_candidate(meta, candidate_period, star_context):
    return {
        **meta,
        "selection_group": star_context.get("selection_group", "unspecified"),
        "candidate_period_days": float(candidate_period),
        "source_detector": "",
        "detector_agreement": False,
        "detector_count": 0,
        "bls_present": False,
        "tcf_present": False,
        "has_bls_candidate": False,
        "has_tcf_candidate": False,
        "has_tcf_event_diagnostics": False,
        "bls_candidate_period_days": np.nan,
        "tcf_candidate_period_days": np.nan,
        "bls_rank": np.nan,
        "tcf_rank": np.nan,
        "bls_sde": np.nan,
        "tcf_score": np.nan,
        "bls_score_relative_to_rank1": np.nan,
        "tcf_score_relative_to_rank1": np.nan,
        "detector_candidate_period_delta_fraction": np.nan,
        "detector_candidate_harmonic_error_fraction": np.nan,
        "detector_candidate_best_harmonic_factor": np.nan,
        "bls_period_delta_to_tcf_best_fraction": np.nan,
        "tcf_period_delta_to_bls_best_fraction": np.nan,
        "candidate_to_tcf_best_harmonic_error_fraction": np.nan,
        "candidate_to_tcf_best_harmonic_factor": np.nan,
        "candidate_to_bls_best_harmonic_error_fraction": np.nan,
        "candidate_to_bls_best_harmonic_factor": np.nan,
        "bls_tcf_rank1_harmonic_error_fraction": np.nan,
        "bls_tcf_rank1_harmonic_factor": np.nan,
        "tcf_valid_transit_events": np.nan,
        "tcf_positive_transit_events": np.nan,
        "tcf_positive_event_fraction": np.nan,
        "tcf_median_event_score": np.nan,
        "tcf_raw_pooled_score": np.nan,
        "candidate_duration_hours": np.nan,
        "candidate_depth": np.nan,
        "noise_quartile": star_context.get("noise_quartile", "unassigned"),
        "robust_flux_scatter_ppm": float(star_context.get("robust_flux_scatter_ppm", np.nan)),
        "gap_fraction": float(star_context.get("gap_fraction", np.nan)),
        "lag_one_flux_acf": float(star_context.get("lag_one_flux_acf", np.nan)),
        "six_hour_scatter_proxy_ppm": float(star_context.get("six_hour_scatter_proxy_ppm", np.nan)),
        "arima_converged": bool(str(star_context.get("base_arima_converged", False)).lower() in {"true", "1", "yes"}),
        "tcf_global_empirical_p_value": np.nan,
        "bls_global_empirical_p_value": np.nan,
        "tcf_regime_empirical_p_value": np.nan,
        "bls_regime_empirical_p_value": np.nan,
        "raw_candidate_rank": np.nan,
    }


def add_peak_candidate(candidates, detector, peak, meta, star_context, rank1_score, score_feature_mode, merge_tolerance_fraction):
    period = peak_period(peak, detector)
    match_index = None
    for index, candidate in enumerate(candidates):
        if periods_match(candidate["candidate_period_days"], period, merge_tolerance_fraction):
            match_index = index
            break
    if match_index is None:
        candidates.append(base_candidate(meta, period, star_context))
        match_index = len(candidates) - 1
    candidate = candidates[match_index]
    candidate["candidate_period_days"] = float(np.nanmean([candidate["candidate_period_days"], period]))
    candidate[f"{detector}_present"] = True
    candidate[f"has_{detector}_candidate"] = True
    candidate[f"{detector}_candidate_period_days"] = period
    candidate[f"{detector}_rank"] = int(peak.get("rank", 1))
    rank = int(candidate[f"{detector}_rank"])
    compatible_diagnostics = score_feature_mode == "full_diagnostics" or rank == 1
    if detector == "bls":
        score = float(peak["sde"])
        candidate["bls_sde"] = score
        if np.isfinite(rank1_score) and rank1_score != 0:
            candidate["bls_score_relative_to_rank1"] = float(score / rank1_score)
        if compatible_diagnostics:
            candidate["candidate_duration_hours"] = float(peak["duration_days"]) * 24.0
            candidate["candidate_depth"] = float(peak["depth"])
    else:
        score = float(peak["score"])
        candidate["tcf_score"] = score
        if np.isfinite(rank1_score) and rank1_score != 0:
            candidate["tcf_score_relative_to_rank1"] = float(score / rank1_score)
        if compatible_diagnostics:
            candidate["tcf_valid_transit_events"] = int(tcf_peak_value(peak, "n_valid_transit_events", 0))
            candidate["tcf_positive_transit_events"] = int(tcf_peak_value(peak, "n_positive_transit_events", 0))
            candidate["tcf_positive_event_fraction"] = float(tcf_peak_value(peak, "positive_event_fraction"))
            candidate["tcf_median_event_score"] = float(tcf_peak_value(peak, "median_event_score"))
            candidate["tcf_raw_pooled_score"] = float(tcf_peak_value(peak, "raw_pooled_score"))
            candidate["candidate_duration_hours"] = float(tcf_peak_value(peak, "duration")) * 24.0
            candidate["has_tcf_event_diagnostics"] = bool(np.isfinite(candidate["tcf_valid_transit_events"]))


def finalize_candidates(candidates, bls_best_period, tcf_best_period):
    finalized = []
    for candidate in candidates:
        candidate = dict(candidate)
        candidate["detector_count"] = int(candidate["bls_present"]) + int(candidate["tcf_present"])
        candidate["detector_agreement"] = bool(candidate["detector_count"] == 2)
        if candidate["detector_agreement"]:
            candidate["source_detector"] = "both"
        elif candidate["bls_present"]:
            candidate["source_detector"] = "bls"
        elif candidate["tcf_present"]:
            candidate["source_detector"] = "tcf"
        ranks = [candidate["bls_rank"], candidate["tcf_rank"]]
        finite_ranks = [float(rank) for rank in ranks if pd.notna(rank) and np.isfinite(float(rank))]
        candidate["raw_candidate_rank"] = float(min(finite_ranks)) if finite_ranks else np.nan
        candidate["bls_period_delta_to_tcf_best_fraction"] = period_fractional_error(candidate["candidate_period_days"], tcf_best_period)
        candidate["tcf_period_delta_to_bls_best_fraction"] = period_fractional_error(candidate["candidate_period_days"], bls_best_period)
        candidate["candidate_to_tcf_best_harmonic_error_fraction"], candidate["candidate_to_tcf_best_harmonic_factor"] = detector_harmonic_error(
            candidate["candidate_period_days"], tcf_best_period
        )
        candidate["candidate_to_bls_best_harmonic_error_fraction"], candidate["candidate_to_bls_best_harmonic_factor"] = detector_harmonic_error(
            candidate["candidate_period_days"], bls_best_period
        )
        candidate["bls_tcf_rank1_harmonic_error_fraction"], candidate["bls_tcf_rank1_harmonic_factor"] = detector_harmonic_error(
            bls_best_period, tcf_best_period
        )
        if candidate["detector_agreement"]:
            candidate["detector_candidate_period_delta_fraction"] = period_fractional_error(
                candidate["bls_candidate_period_days"], candidate["tcf_candidate_period_days"]
            )
            candidate["detector_candidate_harmonic_error_fraction"], candidate["detector_candidate_best_harmonic_factor"] = detector_harmonic_error(
                candidate["bls_candidate_period_days"], candidate["tcf_candidate_period_days"]
            )
        candidate["has_bls_candidate"] = bool(candidate["bls_present"])
        candidate["has_tcf_candidate"] = bool(candidate["tcf_present"])
        candidate["has_tcf_event_diagnostics"] = bool(np.isfinite(candidate["tcf_valid_transit_events"]))
        finalized.append(candidate)
    return finalized


def build_merged_candidates(bls_peaks, tcf_peaks, meta, star_context, args):
    candidates = []
    bls_best_period = np.nan
    tcf_best_period = np.nan
    bls_rank1_score = np.nan
    tcf_rank1_score = np.nan
    if bls_peaks is not None and not bls_peaks.empty:
        bls_best_period = peak_period(bls_peaks.iloc[0], "bls")
        bls_rank1_score = float(bls_peaks.iloc[0]["sde"])
        for _, peak in bls_peaks.iterrows():
            add_peak_candidate(
                candidates,
                "bls",
                peak,
                meta,
                star_context,
                bls_rank1_score,
                args["score_feature_mode"],
                args["merge_tolerance_fraction"],
            )
    if tcf_peaks is not None and not tcf_peaks.empty:
        tcf_best_period = peak_period(tcf_peaks.iloc[0], "tcf")
        tcf_rank1_score = float(tcf_peaks.iloc[0]["score"])
        for _, peak in tcf_peaks.iterrows():
            add_peak_candidate(
                candidates,
                "tcf",
                peak,
                meta,
                star_context,
                tcf_rank1_score,
                args["score_feature_mode"],
                args["merge_tolerance_fraction"],
            )
    return finalize_candidates(candidates, bls_best_period, tcf_best_period)


def completed_calibration_is_compatible(star_dir, config):
    star_dir = Path(star_dir)
    config_path = star_dir / "calibration_config.json"
    if not (star_dir / "COMPLETE").exists() or not config_path.exists():
        return False
    if not (star_dir / "detector_topk.csv").exists() or not (star_dir / "merged_candidates.csv").exists():
        return False
    try:
        return json.loads(config_path.read_text()) == json_ready(config)
    except Exception:
        return False


def star_context_from_row(row):
    return {
        key: row.get(key)
        for key in [
            "selection_group",
            "noise_quartile",
            "robust_flux_scatter_ppm",
            "gap_fraction",
            "lag_one_flux_acf",
            "six_hour_scatter_proxy_ppm",
            "base_arima_converged",
        ]
        if key in row
    }


def save_frame(path, rows):
    frame = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def run_star_calibration_task(task):
    row, bench_args, run_args, config = task
    target_id = bench.normalize_target_id(row["target_id"])
    quarter = int(row["quarter"])
    prefix = bench.star_prefix(target_id, quarter)
    source_star_dir = Path(bench_args["output_dir"]) / "stars" / prefix
    star_dir = Path(run_args["calibration_dir"]) / "stars" / prefix
    detector_path = star_dir / "detector_topk.csv"
    candidates_path = star_dir / "merged_candidates.csv"
    status_path = star_dir / "trial_status.csv"
    if run_args["resume"] and completed_calibration_is_compatible(star_dir, config):
        return {
            "target_id": target_id,
            "quarter": quarter,
            "status": "resumed",
            "star_dir": str(star_dir),
            "detector_path": str(detector_path),
            "candidates_path": str(candidates_path),
            "status_path": str(status_path),
            "error": "",
        }
    started = bench.perf_counter()
    detector_rows = []
    candidate_rows = []
    status_rows = []
    try:
        light_curve_frame, _ = bench.load_light_curve_frame(target_id, quarter, bench_args)
        regular, _ = bench.preprocess_pdcsap_light_curve(
            light_curve_frame,
            quality_policy=bench_args["quality_policy"],
            require_finite_flux_error=bench_args["require_finite_flux_error"],
            normalization_fit_fraction=1.0 - bench_args["test_fraction"],
        )
        time = regular["time"].to_numpy(dtype=float)
        flux = regular["normalized_flux"].to_numpy(dtype=float)
        period_grid = bench.default_period_grid(
            time,
            min_period_days=bench_args["min_period_days"],
            max_period_days=bench_args["max_period_days"],
            n_periods=bench_args["n_periods"],
        )
        duration_grid = bench.default_duration_grid(
            bench_args["min_duration_hours"],
            bench_args["max_duration_hours"],
            bench_args["n_durations"],
        )
        base_arima, _ = bench.load_or_fit_base_arima(source_star_dir, flux, bench_args)
        star_context = star_context_from_row(row)
        star_context["base_arima_converged"] = bool(base_arima["summary"]["converged"])

        meta = {"dataset_kind": "original", "target_id": target_id, "quarter": quarter, "trial": -1}
        tcf_error = ""
        bls_error = ""
        tcf_peaks = pd.DataFrame()
        bls_peaks = pd.DataFrame()
        try:
            tcf_result = bench.run_tcf_search(time, base_arima["innovations"], period_grid, duration_grid, bench_args)
            tcf_peaks = tcf_result["top_peaks"].head(int(run_args["top_k"])).copy()
            detector_rows.extend(detector_peak_rows(tcf_peaks, "tcf", meta, np.nan))
        except Exception as exc:
            tcf_error = f"{type(exc).__name__}: {exc}"
        try:
            bls_result = bench.run_bls(
                time,
                flux,
                period_grid,
                duration_grid,
                objective=bench_args["bls_objective"],
                oversample=bench_args["bls_oversample"],
                top_k=run_args["top_k"],
            )
            bls_peaks = bls_result["top_peaks"].head(int(run_args["top_k"])).copy()
            detector_rows.extend(detector_peak_rows(bls_peaks, "bls", meta, np.nan))
        except Exception as exc:
            bls_error = f"{type(exc).__name__}: {exc}"
        status_rows.append(
            {
                **meta,
                "trial_seed": np.nan,
                "tcf_success": bool(not tcf_peaks.empty),
                "bls_success": bool(not bls_peaks.empty),
                "tcf_error": tcf_error,
                "bls_error": bls_error,
            }
        )
        candidate_rows.extend(build_merged_candidates(bls_peaks, tcf_peaks, meta, star_context, run_args))

        root_sequence = np.random.SeedSequence([int(bench_args["random_seed"]), int(target_id), int(quarter)])
        child_sequences = root_sequence.spawn(int(bench_args["null_trials_per_star"]))
        for trial, sequence in enumerate(child_sequences):
            seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
            rng = np.random.default_rng(seed)
            meta = {"dataset_kind": "null_trial", "target_id": target_id, "quarter": quarter, "trial": int(trial)}
            tcf_error = ""
            bls_error = ""
            tcf_peaks = pd.DataFrame()
            bls_peaks = pd.DataFrame()
            try:
                surrogate_innovations = bench.moving_block_surrogate(
                    base_arima["innovations"],
                    block_size=bench_args["null_block_size_cadences"],
                    rng=rng,
                )
                tcf_result = bench.run_tcf_search(time, surrogate_innovations, period_grid, duration_grid, bench_args)
                tcf_peaks = tcf_result["top_peaks"].head(int(run_args["top_k"])).copy()
                detector_rows.extend(detector_peak_rows(tcf_peaks, "tcf", meta, seed))
            except Exception as exc:
                tcf_error = f"{type(exc).__name__}: {exc}"
            try:
                surrogate_flux = bench.moving_block_surrogate(
                    flux,
                    block_size=bench_args["null_block_size_cadences"],
                    rng=rng,
                )
                bls_result = bench.run_bls(
                    time,
                    surrogate_flux,
                    period_grid,
                    duration_grid,
                    objective=bench_args["bls_objective"],
                    oversample=bench_args["bls_oversample"],
                    top_k=run_args["top_k"],
                )
                bls_peaks = bls_result["top_peaks"].head(int(run_args["top_k"])).copy()
                detector_rows.extend(detector_peak_rows(bls_peaks, "bls", meta, seed))
            except Exception as exc:
                bls_error = f"{type(exc).__name__}: {exc}"
            status_rows.append(
                {
                    **meta,
                    "trial_seed": seed,
                    "tcf_success": bool(not tcf_peaks.empty),
                    "bls_success": bool(not bls_peaks.empty),
                    "tcf_error": tcf_error,
                    "bls_error": bls_error,
                }
            )
            candidate_rows.extend(build_merged_candidates(bls_peaks, tcf_peaks, meta, star_context, run_args))

        star_dir.mkdir(parents=True, exist_ok=True)
        detector = save_frame(detector_path, detector_rows)
        candidates = save_frame(candidates_path, candidate_rows)
        status = save_frame(status_path, status_rows)
        (star_dir / "calibration_config.json").write_text(json.dumps(json_ready(config), indent=2) + "\n")
        (star_dir / "COMPLETE").write_text("complete\n")
        return {
            "target_id": target_id,
            "quarter": quarter,
            "status": "success",
            "star_dir": str(star_dir),
            "detector_path": str(detector_path),
            "candidates_path": str(candidates_path),
            "status_path": str(status_path),
            "detector_rows": int(len(detector)),
            "candidate_rows": int(len(candidates)),
            "trial_rows": int(len(status)),
            "runtime_seconds": float(bench.perf_counter() - started),
            "error": "",
        }
    except Exception as exc:
        star_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "target_id": target_id,
            "quarter": quarter,
            "status": "failed",
            "star_dir": str(star_dir),
            "runtime_seconds": float(bench.perf_counter() - started),
            "error": f"{type(exc).__name__}: {exc}",
        }
        (star_dir / "failure.json").write_text(json.dumps(json_ready(failure), indent=2) + "\n")
        return failure


def combine_star_outputs(results):
    detector_tables = []
    candidate_tables = []
    status_tables = []
    for result in results:
        if result["status"] not in {"success", "resumed"}:
            continue
        detector_tables.append(pd.read_csv(result["detector_path"], dtype={"target_id": str}, keep_default_na=False))
        candidate_tables.append(pd.read_csv(result["candidates_path"], dtype={"target_id": str}, keep_default_na=False))
        status_tables.append(pd.read_csv(result["status_path"], dtype={"target_id": str}, keep_default_na=False))
    detector = pd.concat(detector_tables, ignore_index=True) if detector_tables else pd.DataFrame()
    candidates = pd.concat(candidate_tables, ignore_index=True) if candidate_tables else pd.DataFrame()
    status = pd.concat(status_tables, ignore_index=True) if status_tables else pd.DataFrame()
    if not candidates.empty:
        candidates["target_id"] = candidates["target_id"].map(bench.normalize_target_id)
        candidates["quarter"] = pd.to_numeric(candidates["quarter"], errors="raise").astype(int)
        candidates["candidate_id"] = np.arange(len(candidates), dtype=int)
    return detector, candidates, status


def empirical_p_values(scores, null_scores):
    scores = np.asarray(scores, dtype=float)
    null_scores = np.asarray(null_scores, dtype=float)
    null_scores = null_scores[np.isfinite(null_scores)]
    if null_scores.size == 0:
        return np.full(scores.shape, np.nan, dtype=float)
    return np.asarray(
        [(np.sum(null_scores >= score) + 1.0) / (len(null_scores) + 1.0) if np.isfinite(score) else np.nan for score in scores],
        dtype=float,
    )


def attach_empirical_p_values(candidates):
    candidates = candidates.copy()
    candidates["bls_global_empirical_p_value"] = np.nan
    candidates["tcf_global_empirical_p_value"] = np.nan
    candidates["bls_regime_empirical_p_value"] = np.nan
    candidates["tcf_regime_empirical_p_value"] = np.nan
    for detector, rank_column, score_column, global_column, regime_column in [
        ("bls", "bls_rank", "bls_sde", "bls_global_empirical_p_value", "bls_regime_empirical_p_value"),
        ("tcf", "tcf_rank", "tcf_score", "tcf_global_empirical_p_value", "tcf_regime_empirical_p_value"),
    ]:
        rank1_mask = pd.to_numeric(candidates[rank_column], errors="coerce") == 1
        null_rank1 = (candidates["dataset_kind"] == "null_trial") & rank1_mask
        null_scores = candidates.loc[null_rank1, score_column].to_numpy(dtype=float)
        candidates.loc[rank1_mask, global_column] = empirical_p_values(candidates.loc[rank1_mask, score_column], null_scores)
        for noise_quartile, group_index in candidates[rank1_mask].groupby("noise_quartile", dropna=False).groups.items():
            null_group = candidates.loc[null_rank1 & (candidates["noise_quartile"].astype(str) == str(noise_quartile)), score_column]
            candidates.loc[group_index, regime_column] = empirical_p_values(candidates.loc[group_index, score_column], null_group)
    return candidates


def rank_scored_candidates(candidates):
    candidates = candidates.copy()
    candidates["reranked_rank"] = 999_999
    for _, index in candidates.groupby(CALIBRATION_GROUP_COLUMNS, sort=False).groups.items():
        group = candidates.loc[index]
        scores = pd.to_numeric(group["predicted_score"], errors="coerce").fillna(-np.inf).to_numpy()
        raw_rank = pd.to_numeric(group["raw_candidate_rank"], errors="coerce").fillna(999_999).to_numpy()
        candidate_id = pd.to_numeric(group["candidate_id"], errors="coerce").fillna(999_999).to_numpy()
        order = np.lexsort((candidate_id, raw_rank, -scores))
        ranks = np.empty(len(group), dtype=int)
        ranks[order] = np.arange(1, len(group) + 1)
        candidates.loc[group.index, "reranked_rank"] = ranks
    return candidates


def probability_threshold(values, fap_level):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.quantile(values, 1.0 - float(fap_level), method="higher"))


def score_candidates(candidates, model_path, reranker_spec):
    candidates = attach_empirical_p_values(candidates)
    feature_columns = rerank.leakage_safe_feature_columns(candidates, reranker_spec)
    for column in rerank.BOOLEAN_COLUMNS:
        if column in candidates.columns:
            candidates[column] = bool_series(candidates[column]).astype(int)
    for column in feature_columns:
        if column not in candidates.columns:
            raise ValueError(f"Candidate table is missing frozen feature column: {column}")
        if column not in rerank.KNOWN_CATEGORICAL_COLUMNS:
            candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    model = joblib.load(model_path)
    candidates["predicted_score"] = model.predict_proba(candidates[feature_columns])[:, 1]
    return rank_scored_candidates(candidates), feature_columns


def load_injection_oof(metrics_dir):
    path = Path(metrics_dir) / "candidate_reranker_oof_predictions.csv"
    if not path.exists():
        return pd.DataFrame(), path
    predictions = pd.read_csv(path, dtype={"target_id": str})
    predictions["target_id"] = predictions["target_id"].map(bench.normalize_target_id)
    return predictions[predictions["model"] == "xgboost_classifier"].copy(), path


def top_rank_rows(scored):
    return scored[pd.to_numeric(scored["reranked_rank"], errors="coerce") == 1].copy()


def calibration_summary(scored, metrics_dir, fap_level):
    top = top_rank_rows(scored)
    null_top = top[top["dataset_kind"] == "null_trial"].copy()
    original_top = top[top["dataset_kind"] == "original"].copy()
    threshold = probability_threshold(null_top["predicted_score"], fap_level)
    null_top["passes_reranker_threshold"] = null_top["predicted_score"] >= threshold
    original_top["passes_reranker_threshold"] = original_top["predicted_score"] >= threshold

    injection_predictions, injection_path = load_injection_oof(metrics_dir)
    injection_summary = {}
    if not injection_predictions.empty:
        injection_top = injection_predictions[pd.to_numeric(injection_predictions["reranked_rank"], errors="coerce") == 1].copy()
        exact = bool_series(injection_top["exact_match"])
        scores = pd.to_numeric(injection_top["predicted_score"], errors="coerce")
        passes = scores >= threshold
        injection_summary = {
            "injection_oof_predictions": str(injection_path),
            "injection_groups": int(len(injection_top)),
            "exact_recall_at_1_without_threshold": float(exact.mean()),
            "exact_recovery_at_topk_calibrated_threshold": float((exact & passes).mean()),
            "candidate_fraction_at_topk_calibrated_threshold": float(passes.mean()),
            "top_candidate_brier_score": float(np.mean((scores.to_numpy(dtype=float) - exact.astype(float).to_numpy()) ** 2)),
        }

    regime_rows = []
    for noise_quartile, null_group in null_top.groupby("noise_quartile", dropna=False):
        local_threshold = probability_threshold(null_group["predicted_score"], fap_level)
        original_group = original_top[original_top["noise_quartile"].astype(str) == str(noise_quartile)]
        row = {
            "noise_quartile": noise_quartile,
            "null_light_curves": int(len(null_group)),
            "threshold": local_threshold,
            "observed_null_exceedance_rate": float((null_group["predicted_score"] >= local_threshold).mean())
            if len(null_group) and np.isfinite(local_threshold)
            else np.nan,
            "original_light_curves": int(len(original_group)),
            "original_candidate_fraction": float((original_group["predicted_score"] >= local_threshold).mean())
            if len(original_group) and np.isfinite(local_threshold)
            else np.nan,
        }
        if not injection_predictions.empty:
            injection_top = injection_predictions[pd.to_numeric(injection_predictions["reranked_rank"], errors="coerce") == 1].copy()
            injection_group = injection_top[injection_top["noise_quartile"].astype(str) == str(noise_quartile)]
            row["injection_groups"] = int(len(injection_group))
            row["exact_recovery_at_threshold"] = float(
                (bool_series(injection_group["exact_match"]) & (injection_group["predicted_score"] >= local_threshold)).mean()
            ) if len(injection_group) and np.isfinite(local_threshold) else np.nan
        regime_rows.append(row)

    summary = {
        "calibration_status": "topk_null_original_candidate_calibration",
        "fap_level": float(fap_level),
        "global_probability_threshold": threshold,
        "null_light_curves": int(len(null_top)),
        "null_candidate_rows": int((scored["dataset_kind"] == "null_trial").sum()),
        "observed_null_exceedance_rate": float(null_top["passes_reranker_threshold"].mean()) if len(null_top) else np.nan,
        "original_light_curves": int(len(original_top)),
        "original_candidate_rows": int((scored["dataset_kind"] == "original").sum()),
        "original_candidate_fraction_at_threshold": float(original_top["passes_reranker_threshold"].mean()) if len(original_top) else np.nan,
        **injection_summary,
    }
    return summary, pd.DataFrame(regime_rows), pd.concat([null_top, original_top], ignore_index=True)


def run_generation(args):
    args.metrics_dir = Path(args.metrics_dir)
    args.experiment_dir = Path(args.experiment_dir)
    args.calibration_dir = Path(args.calibration_dir)
    args.model_path = Path(args.model_path)
    if not args.model_path.exists():
        raise FileNotFoundError(f"Frozen reranker model not found: {args.model_path}")
    reranker_spec, reranker_config_path, reranker_config_sha256 = rerank.load_reranker_config(args.reranker_config)
    run_summary = parse_run_summary(args.metrics_dir)
    bench_args = runtime_benchmark_args(args, run_summary)
    config = calibration_config(args, bench_args, reranker_spec, reranker_config_sha256)
    targets = load_targets(args.metrics_dir, args)

    run_args = {
        "calibration_dir": str(args.calibration_dir),
        "resume": bool(args.resume),
        "top_k": int(args.top_k),
        "score_feature_mode": str(args.score_feature_mode),
        "merge_tolerance_fraction": 0.002,
    }
    bench_args_dict = vars(bench_args).copy()
    tasks = [(row.to_dict(), bench_args_dict, run_args, config) for _, row in targets.iterrows()]
    max_workers = max(1, min(int(args.max_workers), len(tasks)))
    results = []
    if max_workers == 1:
        iterator = (run_star_calibration_task(task) for task in tasks)
        for result in tqdm(iterator, total=len(tasks), desc="Top-k calibration", bar_format=TQDM_BAR_FORMAT):
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_star_calibration_task, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Top-k calibration", bar_format=TQDM_BAR_FORMAT):
                results.append(future.result())

    detector, candidates, status = combine_star_outputs(results)
    if candidates.empty:
        raise RuntimeError("No merged null/original candidates were generated.")
    scored, feature_columns = score_candidates(candidates, args.model_path, reranker_spec)
    summary, regime, top_scores = calibration_summary(scored, args.metrics_dir, args.fap_level)

    args.calibration_dir.mkdir(parents=True, exist_ok=True)
    detector_path = args.calibration_dir / "reranker_topk_detector_candidates.csv"
    candidates_path = args.calibration_dir / "reranker_topk_merged_candidates.csv"
    scored_path = args.calibration_dir / "reranker_topk_scored_candidates.csv"
    top_scores_path = args.calibration_dir / "reranker_topk_top_scores.csv"
    status_path = args.calibration_dir / "reranker_topk_trial_status.csv"
    regime_path = args.calibration_dir / "reranker_topk_probability_calibration_by_noise_quartile.csv"
    summary_path = args.calibration_dir / "reranker_topk_probability_calibration_summary.json"
    run_results_path = args.calibration_dir / "reranker_topk_generation_status.csv"
    detector.to_csv(detector_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    scored.to_csv(scored_path, index=False)
    top_scores.to_csv(top_scores_path, index=False)
    status.to_csv(status_path, index=False)
    regime.to_csv(regime_path, index=False)
    pd.DataFrame(results).to_csv(run_results_path, index=False)
    payload = {
        **summary,
        "calibration_version": CALIBRATION_VERSION,
        "reranker_version": reranker_spec["version"],
        "reranker_config": str(reranker_config_path),
        "reranker_config_sha256": reranker_config_sha256,
        "model_path": str(args.model_path),
        "model_sha256": rerank.file_sha256(args.model_path),
        "feature_columns": feature_columns,
        "config": config,
        "outputs": {
            "detector_candidates": str(detector_path),
            "merged_candidates": str(candidates_path),
            "scored_candidates": str(scored_path),
            "top_scores": str(top_scores_path),
            "trial_status": str(status_path),
            "regime_calibration": str(regime_path),
            "generation_status": str(run_results_path),
        },
        "generation": {
            "target_count": int(len(targets)),
            "successful_targets": int(sum(result["status"] in {"success", "resumed"} for result in results)),
            "failed_targets": int(sum(result["status"] not in {"success", "resumed"} for result in results)),
            "resumed_targets": int(sum(result["status"] == "resumed" for result in results)),
        },
    }
    summary_path.write_text(json.dumps(json_ready(payload), indent=2) + "\n")
    return payload, summary_path


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary, summary_path = run_generation(args)
    print(f"Summary: {summary_path}")
    print(f"Calibration status: {summary['calibration_status']}")
    print(f"Reranker version: {summary['reranker_version']}")
    print(f"Null light curves: {summary['null_light_curves']}")
    print(f"Null candidate rows: {summary['null_candidate_rows']}")
    print(f"Original light curves: {summary['original_light_curves']}")
    print(f"Original candidate rows: {summary['original_candidate_rows']}")
    print(f"Global probability threshold: {summary['global_probability_threshold']:.6f}")
    print(f"Observed null exceedance rate: {summary['observed_null_exceedance_rate']:.3f}")
    print(f"Original candidate fraction: {summary['original_candidate_fraction_at_threshold']:.3f}")
    if "exact_recovery_at_topk_calibrated_threshold" in summary:
        print(f"OOF exact recovery at threshold: {summary['exact_recovery_at_topk_calibrated_threshold']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
