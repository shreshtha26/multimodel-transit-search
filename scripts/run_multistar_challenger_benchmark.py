"""Run a checkpointed multi-star benchmark across raw, ARIMA, Kalman, and GP branches."""
import argparse
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import json
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from itertools import combinations as iter_combinations
from itertools import product
from multiprocessing import Manager, get_context
from pathlib import Path
from queue import Empty
from time import perf_counter, sleep
from types import SimpleNamespace
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm.auto import tqdm
from adaptive_transit.data.kepler_io import (
    DEFAULT_MAST_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_MAST_READ_TIMEOUT_SECONDS,
    KeplerFetchPolicy,
    load_kepler_pdcsap,
)
from adaptive_transit.detection.tcf import default_duration_grid, default_period_grid, fit_arima_innovations, harmonic_peak_rank, matching_peak_rank, period_match_fraction, run_tcf
from adaptive_transit.detection.tls import run_tls
from adaptive_transit.detection.tps_like import prepare_tps_like_noise_model, run_tps_like_search
from adaptive_transit.detection.trapezoid import run_bls_seeded_trapezoid
from adaptive_transit.injections.synthetic import inject_periodic_box_transit
from adaptive_transit.noise_models.gp import apply_prepared_smooth_gp_filter, fit_smooth_gp_background, prepare_smooth_gp_filter
from adaptive_transit.noise_models.kalman import apply_fitted_kalman_filter, fit_kalman_local_level
from adaptive_transit.noise_models.characterization import characterize_regularized_light_curve
from adaptive_transit.noise_models.stellar_variability import MODEL_SELECTION_FEATURE_COLUMNS
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Warning: the tpfmodel submodule is not available.*", category=UserWarning)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_MANIFEST_PATH = PROJECT_ROOT / "configs/kepler_50_star_manifest.csv"
CLEAN_MANIFEST_PATH = PROJECT_ROOT / "configs/kepler_clean_background_manifest.csv"
MANIFEST_PATH = CLEAN_MANIFEST_PATH
CACHE_DIR = PROJECT_ROOT / "outputs/cache/kepler_light_curves"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/experiments/multistar_challenger_benchmark"
BACKGROUND_FEATURE_PATH = PROJECT_ROOT / "outputs/target_selection/kepler_catalog_clean_candidate_features.csv"
EXISTING_ARIMA_CACHE_ROOT = PROJECT_ROOT / "outputs/experiments/multistar_bls_tcf/optimized/stars"
TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} ({percentage:3.0f}%) [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
BRANCHES = ("raw", "arima", "kalman", "gp")
CORE_DETECTORS = ("bls", "tcf", "tps_like")
CHALLENGER_DETECTORS = ("tls", "trapezoid")
DETECTORS = (*CORE_DETECTORS, *CHALLENGER_DETECTORS)
PIPELINE_DEFINITIONS = {f"{branch}_{detector}": (branch, detector) for branch in BRANCHES for detector in DETECTORS}
DEFAULT_PIPELINES = tuple(f"{branch}_{detector}" for branch in BRANCHES for detector in CORE_DETECTORS)
CLEAN_SELECTION_GROUP = "catalog_clean_background"
CATALOG_FLAG_COLUMNS = ("koi_flag", "tce_flag", "confirmed_planet_flag", "eb_flag")
BENCHMARK_SCHEMA_VERSION = 3
SEARCH_RESOLUTION_PRESETS = {"pilot": {"n_periods": 3000, "top_k": 5, "n_coarse_periods": 1000, "n_refinement_regions": 12, "refinement_half_width_points": 30, "bls_oversample": 5}, "medium": {"n_periods": 5000, "top_k": 5, "n_coarse_periods": 2000, "n_refinement_regions": 18, "refinement_half_width_points": 30, "bls_oversample": 5}, "high": {"n_periods": 10000, "top_k": 10, "n_coarse_periods": 4000, "n_refinement_regions": 30, "refinement_half_width_points": 40, "bls_oversample": 10}}

def default_settings(profile="smoke"):
    settings = {"profile": profile, "manifest_path": MANIFEST_PATH, "cache_dir": CACHE_DIR, "output_dir": OUTPUT_ROOT / profile, "background_feature_path": BACKGROUND_FEATURE_PATH, "existing_arima_cache_root": EXISTING_ARIMA_CACHE_ROOT, "target_limit": 10, "strict_target_count": True, "target_ids": None, "selection_group": CLEAN_SELECTION_GROUP, "require_catalog_clean": True, "stratified_pilot": True, "quality_policy": "default", "require_finite_flux_error": False, "test_fraction": 0.20, "pipelines": DEFAULT_PIPELINES, "arima_order": (1, 1, 0), "fit_maxiter": 200, "arima_injection_mode": "filter", "kalman_injection_mode": "filter", "kalman_maxiter": 100, "kalman_burn_in": 1, "gp_injection_mode": "filter", "gp_max_train_points": 512, "gp_length_scale_days": 3.0, "gp_min_length_scale_days": 1.0, "gp_max_length_scale_days": 30.0, "gp_measurement_noise_fraction": 0.20, "gp_n_restarts_optimizer": 0, "gp_random_seed": 123, "gp_optimize_kernel": True, "injection_period_grid": (2.0, 5.0), "injection_duration_hours_grid": (2.0, 4.0), "injection_depth_grid": (0.0005, 0.001), "epoch_phase_fraction_grid": (0.45,), "min_period_days": 1.0, "max_period_days": 15.0, "search_resolution": "pilot", "n_periods": 3000, "min_duration_hours": 1.5, "max_duration_hours": 10.0, "n_durations": 8, "edge_width_cadences": 0, "min_edge_observations": 4, "min_transit_events": 3, "min_event_consistency_fraction": 0.60, "top_k": 5, "search_mode": "coarse_to_fine", "n_coarse_periods": 1000, "n_refinement_regions": 12, "refinement_half_width_points": 30, "period_match_tolerance_fraction": 0.02, "bls_objective": "snr", "bls_oversample": 5, "tls_use_threads": 1, "tls_oversampling_factor": 2, "tps_wavelet": "db6", "tps_max_wavelet_level": 6, "tps_noise_window_cadences": 193, "tps_min_segment_cadences": 32, "max_workers": None, "reserve_cpu_cores": 2, "random_seed": 123, "allow_download": True, "download_max_attempts": 5, "download_initial_wait_seconds": 5.0, "download_backoff_factor": 2.0, "progress_interval": 1, "checkpoint_interval": 5, "prefetch_workers": 4, "resume": True, "rerun_failures": False, "save_regularized_inputs": False, "characterization_acf_lags": 2000, "characterization_spectral_frequencies": 4000, "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION}
    if profile == "main":
        settings.update({"output_dir": OUTPUT_ROOT / profile, "target_limit": 50, "stratified_pilot": False, "injection_period_grid": (2.0, 5.0, 10.0), "injection_duration_hours_grid": (2.0, 4.0, 8.0), "injection_depth_grid": (0.0002, 0.0005, 0.001), "epoch_phase_fraction_grid": (0.15, 0.45, 0.75), "search_resolution": "high", "n_periods": 10000, "top_k": 10, "n_coarse_periods": 4000, "n_refinement_regions": 30, "refinement_half_width_points": 40, "bls_oversample": 10}
        )
    elif profile == "smoke":
        settings.update({"output_dir": OUTPUT_ROOT / profile, "target_limit": 2, "strict_target_count": False, "stratified_pilot": False, "injection_period_grid": (5.0,), "injection_duration_hours_grid": (4.0,), "injection_depth_grid": (0.001,), "epoch_phase_fraction_grid": (0.45,), "search_resolution": "smoke", "n_periods": 400, "n_durations": 4, "top_k": 3, "n_coarse_periods": 150, "n_refinement_regions": 3, "refinement_half_width_points": 8, "bls_oversample": 3, "max_workers": 2, "gp_max_train_points": 192, "gp_optimize_kernel": False}
        )
    elif profile != "pilot":
        raise ValueError("profile must be pilot, main, or smoke.")
    return SimpleNamespace(**settings)

def apply_search_resolution(args, name):
    if name not in SEARCH_RESOLUTION_PRESETS:
        raise ValueError(f"Unknown search resolution: {name}")
    for key, value in SEARCH_RESOLUTION_PRESETS[name].items():
        setattr(args, key, value)
    args.search_resolution = str(name)
    return args

def parse_float_grid(value):
    values = tuple(float(part.strip()) for part in str(value).split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Grid values must be positive comma-separated floats.")
    return values

def parse_int_order(value):
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("ARIMA order must look like p,d,q.")
    order = tuple(int(part) for part in parts)
    if any(item < 0 for item in order):
        raise argparse.ArgumentTypeError("ARIMA order entries must be non-negative.")
    return order

def parse_pipelines(value):
    pipelines = tuple(item.strip() for item in str(value).split(",") if item.strip())
    invalid = sorted(set(pipelines).difference(PIPELINE_DEFINITIONS))
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown pipeline(s): {invalid}. Valid pipelines are {sorted(PIPELINE_DEFINITIONS)}.")
    return pipelines

def parse_target_ids(value):
    values = tuple(normalize_target_id(item) for item in str(value).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one target id.")
    return values

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Legacy wide-table challenger benchmark. Active scientific runs use scripts/run_adaptive_transit_benchmark.py with benchmark100 or benchmark1000.")
    parser.add_argument("--profile", choices=("pilot", "main", "smoke"), default="smoke")
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--background-feature-path", type=Path)
    parser.add_argument("--target-limit", type=int)
    parser.add_argument("--allow-partial-target-count", dest="strict_target_count", action="store_false")
    parser.add_argument("--target-ids", type=parse_target_ids)
    parser.add_argument("--selection-group", type=str)
    parser.add_argument("--allow-contaminated-cohort", dest="require_catalog_clean", action="store_false", default=None, help="Explicitly allow known-signal/legacy cohorts. Clean injection benchmarking is the default.")
    parser.add_argument("--no-stratified-pilot", dest="stratified_pilot", action="store_false")
    parser.add_argument("--pipelines", type=parse_pipelines)
    parser.add_argument("--arima-order", type=parse_int_order)
    parser.add_argument("--fit-maxiter", type=int)
    parser.add_argument("--arima-injection-mode", choices=("filter", "refit"))
    parser.add_argument("--kalman-injection-mode", choices=("filter", "refit"))
    parser.add_argument("--kalman-maxiter", type=int)
    parser.add_argument("--gp-max-train-points", type=int)
    parser.add_argument("--gp-length-scale-days", type=float)
    parser.add_argument("--gp-min-length-scale-days", type=float)
    parser.add_argument("--gp-max-length-scale-days", type=float)
    parser.add_argument("--gp-measurement-noise-fraction", type=float)
    parser.add_argument("--gp-n-restarts-optimizer", type=int)
    parser.add_argument("--gp-injection-mode", choices=("filter", "refit"))
    parser.add_argument("--gp-fixed-kernel", dest="gp_optimize_kernel", action="store_false")
    parser.add_argument("--injection-period-grid", type=parse_float_grid)
    parser.add_argument("--injection-duration-hours-grid", type=parse_float_grid)
    parser.add_argument("--injection-depth-grid", type=parse_float_grid)
    parser.add_argument("--epoch-phase-fraction-grid", type=parse_float_grid)
    parser.add_argument("--search-resolution", choices=("pilot", "medium", "high"))
    parser.add_argument("--n-periods", type=int)
    parser.add_argument("--n-durations", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--n-coarse-periods", type=int)
    parser.add_argument("--n-refinement-regions", type=int)
    parser.add_argument("--refinement-half-width-points", type=int)
    parser.add_argument("--bls-oversample", type=int)
    parser.add_argument("--tls-use-threads", type=int)
    parser.add_argument("--tls-oversampling-factor", type=int)
    parser.add_argument("--tps-wavelet", type=str)
    parser.add_argument("--tps-max-wavelet-level", type=int)
    parser.add_argument("--tps-noise-window-cadences", type=int)
    parser.add_argument("--tps-min-segment-cadences", type=int)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--reserve-cpu-cores", type=int)
    parser.add_argument("--no-download", dest="allow_download", action="store_false")
    parser.add_argument("--download-connect-timeout-seconds", type=float)
    parser.add_argument("--download-read-timeout-seconds", type=float)
    parser.add_argument("--download-max-attempts", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--progress-interval", type=int)
    parser.add_argument("--prefetch-workers", type=int)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--rerun-failures", action="store_true")
    parser.add_argument("--save-regularized-inputs", action="store_true")
    parser.add_argument("--characterization-acf-lags", type=int)
    parser.add_argument("--characterization-spectral-frequencies", type=int)
    parsed = parser.parse_args(argv)
    args = default_settings(parsed.profile)
    if parsed.search_resolution is not None:
        apply_search_resolution(args, parsed.search_resolution)
    for key, value in vars(parsed).items():
        if key in ("profile", "search_resolution"):
            continue
        if value is not None:
            setattr(args, key, value)
    return args

def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def normalize_target_id(value):
    return str(value).upper().replace("KIC", "").strip()

def star_prefix(target_id, quarter):
    return f"kic_{normalize_target_id(target_id)}_q{int(quarter)}"

def injection_cases(args):
    return list(product(args.injection_period_grid, args.injection_duration_hours_grid, args.injection_depth_grid, args.epoch_phase_fraction_grid))

def config_signature(args):
    keys = ("benchmark_schema_version", "profile", "selection_group", "require_catalog_clean", "pipelines", "quality_policy", "require_finite_flux_error", "test_fraction", "arima_order", "fit_maxiter", "arima_injection_mode", "kalman_injection_mode", "kalman_maxiter", "kalman_burn_in", "gp_injection_mode", "gp_max_train_points", "gp_length_scale_days", "gp_min_length_scale_days", "gp_max_length_scale_days", "gp_measurement_noise_fraction", "gp_n_restarts_optimizer", "gp_optimize_kernel", "injection_period_grid", "injection_duration_hours_grid", "injection_depth_grid", "epoch_phase_fraction_grid", "min_period_days", "max_period_days", "search_resolution", "n_periods", "min_duration_hours", "max_duration_hours", "n_durations", "top_k", "search_mode", "n_coarse_periods", "n_refinement_regions", "refinement_half_width_points", "bls_objective", "bls_oversample", "tls_use_threads", "tls_oversampling_factor", "tps_wavelet", "tps_max_wavelet_level", "tps_noise_window_cadences", "tps_min_segment_cadences", "period_match_tolerance_fraction", "characterization_acf_lags", "characterization_spectral_frequencies")
    return {key: json_ready(getattr(args, key)) for key in keys}

def write_benchmark_config(args):
    path = Path(args.output_dir) / "benchmark_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config_signature": json_ready(config_signature(args))}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path

def write_star_checkpoint(star_dir, target_id, quarter, status, stage, **extra):
    star_dir.mkdir(parents=True, exist_ok=True)
    payload = {"target_id": normalize_target_id(target_id), "quarter": int(quarter), "status": str(status), "stage": str(stage), **extra}
    (star_dir / "checkpoint.json").write_text(json.dumps(json_ready(payload), indent=2) + "\n")

def report_progress(progress_queue, target_id, quarter, stage, units=0, detail=""):
    if progress_queue is None:
        return
    try:
        progress_queue.put({"target_id": normalize_target_id(target_id), "quarter": int(quarter), "stage": str(stage), "units": int(units), "detail": str(detail)}, block=False)
    except Exception:
        pass

def light_curve_cache_path(args, target_id, quarter):
    return Path(args["cache_dir"]) / f"{star_prefix(target_id, quarter)}_pdcsap.parquet"

def is_transient_download_error(exc):
    message = f"{type(exc).__name__}: {exc}"
    if "No Kepler light curve found" in message:
        return False
    markers = ("ReadTimeout", "ConnectTimeout", "ConnectionError", "HTTPSConnectionPool", "Max retries exceeded", "Temporary failure", "temporarily unavailable", "Connection reset", "RemoteDisconnected", "mast.stsci.edu", "KeplerLightCurveFetchError", "Kepler MAST fetch failed")
    return any(marker in message for marker in markers)

def load_light_curve_frame(target_id, quarter, args, progress_queue=None):
    path = light_curve_cache_path(args, target_id, quarter)
    if path.exists():
        return pd.read_parquet(path), True
    if not args.get("allow_download", True):
        raise FileNotFoundError(f"Cached light curve is missing: {path}")
    max_attempts = max(1, int(args.get("download_max_attempts", 1)))
    connect_timeout = args.get("download_connect_timeout_seconds") or DEFAULT_MAST_CONNECT_TIMEOUT_SECONDS
    read_timeout = args.get("download_read_timeout_seconds") or DEFAULT_MAST_READ_TIMEOUT_SECONDS
    fetch_policy = KeplerFetchPolicy(
        connect_timeout_seconds=float(connect_timeout),
        read_timeout_seconds=float(read_timeout),
        max_attempts=1,
    )
    for attempt in range(1, max_attempts + 1):
        try:
            report_progress(progress_queue, target_id, quarter, "download attempt", detail=f"{attempt}/{max_attempts}")
            light_curve = load_kepler_pdcsap(target_id, quarter, fetch_policy=fetch_policy)
            frame = light_curve.to_dataframe()
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            return frame, False
        except Exception as exc:
            if attempt >= max_attempts or not is_transient_download_error(exc):
                raise
            wait_seconds = float(args.get("download_initial_wait_seconds", 5.0)) * float(args.get("download_backoff_factor", 2.0)) ** (attempt - 1)
            report_progress(progress_queue, target_id, quarter, "download retry", detail=f"{attempt}/{max_attempts}; waiting {wait_seconds:.0f}s")
            sleep(wait_seconds)
    raise RuntimeError("Light-curve download retry loop ended unexpectedly.")

def load_manifest(args):
    path = Path(args.manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Target manifest does not exist: {path}")
    manifest = pd.read_csv(path)
    if not {"target_id", "quarter"}.issubset(manifest.columns):
        raise ValueError("Manifest must contain target_id and quarter columns.")
    manifest = manifest.copy()
    manifest["target_id"] = manifest["target_id"].map(normalize_target_id)
    manifest["quarter"] = pd.to_numeric(manifest["quarter"], errors="raise").astype(int)
    if "selection_group" not in manifest.columns:
        manifest["selection_group"] = "unspecified"
    manifest["selection_group"] = manifest["selection_group"].fillna("unspecified").astype(str)
    if "sample_stratum" not in manifest.columns:
        manifest["sample_stratum"] = "unspecified"
    manifest["sample_stratum"] = manifest["sample_stratum"].fillna("unspecified").astype(str)
    manifest = manifest.drop_duplicates(["target_id", "quarter"], keep="first").reset_index(drop=True)
    if getattr(args, "selection_group", None):
        manifest = manifest[manifest["selection_group"] == str(args.selection_group)].reset_index(drop=True)
    if args.target_ids:
        wanted = {normalize_target_id(target_id) for target_id in args.target_ids}
        manifest = manifest[manifest["target_id"].isin(wanted)].reset_index(drop=True)
    if len(manifest) > int(args.target_limit) and "sample_stratum" in manifest.columns and manifest["sample_stratum"].nunique() > 1:
        manifest = balanced_stratum_manifest(manifest, int(args.target_limit))
    elif args.stratified_pilot and str(args.profile) == "pilot" and Path(args.background_feature_path).exists():
        manifest = stratified_pilot_manifest(manifest, args)
    else:
        manifest = manifest.head(int(args.target_limit)).copy()
    if args.strict_target_count and len(manifest) != int(args.target_limit):
        raise ValueError(f"Expected exactly {args.target_limit} target-quarter rows but found {len(manifest)}.")
    if manifest.empty:
        raise ValueError("Manifest contains no usable rows.")
    validate_manifest_cohort(manifest, args)
    return manifest.reset_index(drop=True)


def truthy_catalog_flag(series):
    normalized = series.fillna(False).astype(str).str.strip().str.lower()
    return normalized.isin({"1", "1.0", "true", "t", "yes", "y"})

def validate_manifest_cohort(manifest, args):
    if not bool(getattr(args, "require_catalog_clean", True)):
        return
    if "selection_group" not in manifest.columns:
        raise ValueError("Clean injection benchmark requires a selection_group column.")
    bad_group = manifest[manifest["selection_group"] != CLEAN_SELECTION_GROUP]
    if not bad_group.empty:
        examples = bad_group[["target_id", "quarter", "selection_group"]].head(10).to_dict(orient="records")
        raise ValueError(
            f"Clean injection benchmark contains non-clean selection groups. Expected only {CLEAN_SELECTION_GROUP!r}; examples: {examples}. "
            "Use --allow-contaminated-cohort only for an explicitly labelled stress-test/known-signal run."
        )
    missing = [column for column in CATALOG_FLAG_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(
            "Clean injection benchmark is missing catalog contamination flags: "
            f"{missing}. Build the manifest with scripts/build_clean_kepler_manifest.py before running injections."
        )
    contaminated = pd.Series(False, index=manifest.index)
    for column in CATALOG_FLAG_COLUMNS:
        contaminated = contaminated | truthy_catalog_flag(manifest[column])
    if contaminated.any():
        columns = ["target_id", "quarter", "selection_group", *CATALOG_FLAG_COLUMNS]
        examples = manifest.loc[contaminated, columns].head(10).to_dict(orient="records")
        raise ValueError(
            "Clean injection benchmark contains cataloged KOI/TCE/confirmed-planet/EB hosts; "
            f"examples: {examples}."
        )


def balanced_stratum_manifest(manifest, target_limit):
    """Deterministically round-robin across preassigned sampling strata."""
    groups = []
    for stratum in sorted(manifest["sample_stratum"].dropna().astype(str).unique()):
        group = manifest.loc[manifest["sample_stratum"].astype(str) == stratum].sort_values(["target_id", "quarter"]).reset_index(drop=True)
        if not group.empty:
            groups.append(group)
    selected = []
    depth = 0
    while len(selected) < int(target_limit) and groups:
        added = False
        for group in groups:
            if depth < len(group):
                selected.append(group.iloc[depth].to_dict())
                added = True
                if len(selected) >= int(target_limit):
                    break
        if not added:
            break
        depth += 1
    return pd.DataFrame(selected)

def add_target(selected, selected_keys, row, reason):
    key = (normalize_target_id(row["target_id"]), int(row["quarter"]))
    if key in selected_keys:
        return
    payload = dict(row)
    payload["challenger_selection_reason"] = reason
    selected.append(payload)
    selected_keys.add(key)

def stratified_pilot_manifest(manifest, args):
    features = pd.read_csv(args.background_feature_path)
    features["target_id"] = features["target_id"].map(normalize_target_id)
    features["quarter"] = pd.to_numeric(features["quarter"], errors="coerce").astype("Int64")
    joined = manifest.merge(features, on=["target_id", "quarter", "selection_group"], how="left")
    selected = []
    selected_keys = set()
    # Legacy pilot selection only.  Low scatter is *not* equivalent to a quiet
    # star, and gap structure is no longer treated as a scientific variability
    # regime.  Final population labels are assigned after full v2 characterization.
    criteria = [("robust_flux_scatter_ppm", True, 2, "low_scatter_screen"), ("background_tau_integrated_positive_acf_days", False, 2, "long_integrated_acf"), ("background_tau_acf_e_days", False, 2, "long_acf_e"), ("robust_flux_scatter_ppm", False, 2, "high_scatter")]
    for column, ascending, count, reason in criteria:
        if column not in joined.columns:
            continue
        candidates = joined.sort_values(column, ascending=ascending, na_position="last")
        added = 0
        for row in candidates.to_dict(orient="records"):
            if added >= count or len(selected) >= int(args.target_limit):
                break
            before = len(selected)
            add_target(selected, selected_keys, row, reason)
            added += int(len(selected) > before)
    for row in manifest.to_dict(orient="records"):
        if len(selected) >= int(args.target_limit):
            break
        add_target(selected, selected_keys, row, "manifest_fill")
    return pd.DataFrame(selected).head(int(args.target_limit))

def robust_scale(values):
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        return float("nan")
    median = float(np.median(clean))
    scale = float(1.4826 * np.median(np.abs(clean - median)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(clean, ddof=1))
    return scale

def robust_standardize(values):
    values = np.asarray(values, dtype=float)
    scale = robust_scale(values)
    location = float(np.nanmedian(values))
    if not np.isfinite(scale) or scale <= 0:
        return np.full(values.shape, np.nan, dtype=float)
    return (values - location) / scale

def local_maximum_indices(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return np.asarray([], dtype=int)
    if scores.size == 1:
        return np.asarray([0], dtype=int) if np.isfinite(scores[0]) else np.asarray([], dtype=int)
    local = np.zeros(scores.size, dtype=bool)
    local[0] = np.isfinite(scores[0]) and scores[0] >= scores[1]
    local[-1] = np.isfinite(scores[-1]) and scores[-1] >= scores[-2]
    local[1:-1] = np.isfinite(scores[1:-1]) & (scores[1:-1] >= scores[:-2]) & (scores[1:-1] >= scores[2:])
    return np.flatnonzero(local)

def select_bls_top_peaks(periodogram, top_k=10, separation_fraction=0.01):
    candidates = periodogram.iloc[local_maximum_indices(periodogram["sde"].to_numpy(dtype=float))].copy()
    candidates = candidates[np.isfinite(candidates["sde"])].sort_values("sde", ascending=False)
    selected = []
    for index, row in candidates.iterrows():
        period = float(row["period_days"])
        if any(abs(period - item["period_days"]) / item["period_days"] <= float(separation_fraction) for item in selected):
            continue
        selected.append({"periodogram_index": int(index), "period_days": period, "sde": float(row["sde"]), "power": float(row["power"]), "duration_days": float(row["duration_days"]), "transit_time": float(row["transit_time"]), "depth": float(row["depth"])})
        if len(selected) >= int(top_k):
            break
    peaks = pd.DataFrame(selected)
    if not peaks.empty:
        peaks.insert(0, "rank", np.arange(1, len(peaks) + 1, dtype=int))
    return peaks

def select_top_periodogram_peaks(periodogram, score_column, top_k=10, separation_fraction=0.01):
    if periodogram.empty or "period_days" not in periodogram.columns or score_column not in periodogram.columns:
        return pd.DataFrame()
    scores = pd.to_numeric(periodogram[score_column], errors="coerce").to_numpy(dtype=float)
    peak_indices = local_maximum_indices(scores)
    candidates = periodogram.iloc[peak_indices].copy()
    candidates = candidates[np.isfinite(pd.to_numeric(candidates[score_column], errors="coerce"))]
    if candidates.empty:
        candidates = periodogram[np.isfinite(pd.to_numeric(periodogram[score_column], errors="coerce"))].copy()
    candidates = candidates.sort_values(score_column, ascending=False)
    selected = []
    for index, row in candidates.iterrows():
        period = float(row["period_days"])
        if any(abs(period - item["period_days"]) / item["period_days"] <= float(separation_fraction) for item in selected):
            continue
        item = row.to_dict()
        item["periodogram_index"] = int(index)
        item["period_days"] = period
        item["score"] = float(row[score_column])
        selected.append(item)
        if len(selected) >= int(top_k):
            break
    peaks = pd.DataFrame(selected)
    if not peaks.empty:
        peaks.insert(0, "rank", np.arange(1, len(peaks) + 1, dtype=int))
    return peaks

def run_bls_search(time, flux, period_grid, duration_grid, args):
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    if finite.sum() < 24:
        raise ValueError("At least 24 finite observations are required for BLS.")
    model = BoxLeastSquares(time[finite], flux[finite])
    result = model.power(np.asarray(period_grid, dtype=float), np.asarray(duration_grid, dtype=float), objective=args["bls_objective"], method="fast", oversample=int(args["bls_oversample"]))
    periodogram = pd.DataFrame({"period_days": np.asarray(result.period, dtype=float), "power": np.asarray(result.power, dtype=float), "duration_days": np.asarray(result.duration, dtype=float), "transit_time": np.asarray(result.transit_time, dtype=float), "depth": np.asarray(result.depth, dtype=float)})
    periodogram["sde"] = robust_standardize(periodogram["power"].to_numpy(dtype=float))
    top_peaks = select_bls_top_peaks(periodogram, top_k=args["top_k"], separation_fraction=0.01)
    if top_peaks.empty:
        raise ValueError("BLS did not produce any finite local peaks.")
    return {"summary": top_peaks.iloc[0].to_dict(), "periodogram": periodogram, "top_peaks": top_peaks}

def run_tcf_search(time, residuals, period_grid, duration_grid, args):
    return run_tcf(time, residuals, period_grid, duration_grid, edge_width_cadences=args["edge_width_cadences"], min_edge_observations=args["min_edge_observations"], min_transit_events=args["min_transit_events"], min_event_consistency_fraction=args["min_event_consistency_fraction"], top_k=args["top_k"], search_mode=args["search_mode"], n_coarse_periods=args["n_coarse_periods"], n_refinement_regions=args["n_refinement_regions"], refinement_half_width_points=args["refinement_half_width_points"])

def run_tls_search(time, flux, args):
    result = run_tls(
        time,
        flux,
        period_min=args["min_period_days"],
        period_max=args["max_period_days"],
        use_threads=args["tls_use_threads"],
        oversampling_factor=args["tls_oversampling_factor"],
    )
    summary = dict(result["summary"])
    summary["score"] = float(summary["sde"])
    periodogram = result["periodogram"].copy()
    if "power" in periodogram.columns:
        periodogram["sde"] = robust_standardize(periodogram["power"].to_numpy(dtype=float))
        top_peaks = select_top_periodogram_peaks(periodogram, "sde", top_k=args["top_k"], separation_fraction=0.01)
    else:
        top_peaks = pd.DataFrame()
    best_peak = {
        "rank": 1,
        "period_days": float(summary["period_days"]),
        "duration_days": float(summary["duration_days"]),
        "epoch_days": float(summary["epoch_days"]),
        "score": float(summary["score"]),
        "sde": float(summary["sde"]),
        "snr": float(summary["snr"]),
        "depth_raw": float(summary["depth_raw"]),
    }
    if top_peaks.empty:
        top_peaks = pd.DataFrame([best_peak])
    else:
        top_peaks = top_peaks[abs(top_peaks["period_days"] - best_peak["period_days"]) / best_peak["period_days"] > 0.01]
        top_peaks = pd.concat([pd.DataFrame([best_peak]), top_peaks], ignore_index=True).head(int(args["top_k"]))
        top_peaks["rank"] = np.arange(1, len(top_peaks) + 1, dtype=int)
        top_peaks["score"] = pd.to_numeric(top_peaks.get("score", top_peaks.get("sde")), errors="coerce")
        top_peaks["sde"] = pd.to_numeric(top_peaks.get("sde", top_peaks.get("score")), errors="coerce")
    return {"summary": summary, "periodogram": periodogram, "top_peaks": top_peaks, "raw_result": result.get("raw_result")}

def trapezoid_seed_result(bls_result):
    seeded = dict(bls_result)
    peaks = bls_result["top_peaks"].copy()
    if "period" not in peaks.columns and "period_days" in peaks.columns:
        peaks["period"] = peaks["period_days"]
    if "duration" not in peaks.columns and "duration_days" in peaks.columns:
        peaks["duration"] = peaks["duration_days"]
    seeded["top_peaks"] = peaks
    return seeded

def run_trapezoid_search(time, flux, period_grid, duration_grid, args, cache=None):
    cache = {} if cache is None else cache
    bls_result = cache.get("bls")
    if bls_result is None:
        bls_result = run_bls_search(time, flux, period_grid, duration_grid, args)
        cache["bls"] = bls_result
    result = run_bls_seeded_trapezoid(
        time,
        flux,
        trapezoid_seed_result(bls_result),
        duration_grid=duration_grid,
        top_k_periods=args["top_k"],
    )
    evaluated = result["evaluated"].copy()
    top_peaks = evaluated.sort_values("score", ascending=False).head(int(args["top_k"])).reset_index(drop=True)
    top_peaks.insert(0, "rank", np.arange(1, len(top_peaks) + 1, dtype=int))
    return {"summary": result["summary"], "periodogram": evaluated, "top_peaks": top_peaks}

def run_tps_like_detector_search(time, values, segment_id, duration_grid, args, cache=None):
    if segment_id is None:
        raise ValueError("TPS-like detector requires cadence-grid segment_id.")
    cache = {} if cache is None else cache
    if "tps_like_noise_model_error" in cache:
        raise ValueError(str(cache["tps_like_noise_model_error"]))
    prepared = cache.get("tps_like_noise_model")
    if prepared is None:
        prepared = prepare_tps_like_noise_model(
            values,
            segment_id,
            wavelet=args["tps_wavelet"],
            max_level=args["tps_max_wavelet_level"],
            noise_window_cadences=args["tps_noise_window_cadences"],
            min_segment_cadences=args["tps_min_segment_cadences"],
        )
        cache["tps_like_noise_model"] = prepared
    duration_hours_grid = [float(duration) * 24.0 for duration in np.asarray(duration_grid, dtype=float)]
    result = run_tps_like_search(
        time,
        values,
        segment_id,
        prepared_noise_model=prepared,
        min_period_days=args["min_period_days"],
        max_period_days=args["max_period_days"],
        duration_hours_grid=duration_hours_grid,
        wavelet=args["tps_wavelet"],
        max_level=args["tps_max_wavelet_level"],
        noise_window_cadences=args["tps_noise_window_cadences"],
        min_segment_cadences=args["tps_min_segment_cadences"],
        min_events=args["min_transit_events"],
    )
    summary = dict(result["summary"])
    summary["score"] = float(summary["mes"])
    periodogram = result["periodogram"].copy()
    top_peaks = periodogram.sort_values("mes", ascending=False).head(int(args["top_k"])).reset_index(drop=True)
    if top_peaks.empty:
        raise ValueError("TPS-like detector produced no finite top candidates.")
    top_peaks.insert(0, "rank", np.arange(1, len(top_peaks) + 1, dtype=int))
    top_peaks["score"] = pd.to_numeric(top_peaks["mes"], errors="coerce")
    return {"summary": summary, "periodogram": periodogram, "top_peaks": top_peaks, "prepared_noise_model": prepared}

def run_detector_search(detector, time, values, period_grid, duration_grid, args, *, segment_id=None, cache=None):
    cache = {} if cache is None else cache
    if detector in cache:
        return cache[detector]
    if detector == "bls":
        result = run_bls_search(time, values, period_grid, duration_grid, args)
    elif detector == "tcf":
        result = run_tcf_search(time, values, period_grid, duration_grid, args)
    elif detector == "tls":
        result = run_tls_search(time, values, args)
    elif detector == "trapezoid":
        result = run_trapezoid_search(time, values, period_grid, duration_grid, args, cache=cache)
    elif detector == "tps_like":
        result = run_tps_like_detector_search(time, values, segment_id, duration_grid, args, cache=cache)
    else:
        raise ValueError(f"Unknown detector: {detector}")
    cache[detector] = result
    return result

def median_cadence(time):
    values = np.sort(np.unique(np.asarray(time, dtype=float)[np.isfinite(time)]))
    differences = np.diff(values)
    differences = differences[np.isfinite(differences) & (differences > 0)]
    return float(np.median(differences)) if differences.size else float("nan")

def lag_one_acf(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    valid = finite[:-1] & finite[1:]
    if valid.sum() < 3:
        return float("nan")
    first = values[:-1][valid]
    second = values[1:][valid]
    if np.std(first) <= 0 or np.std(second) <= 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])

def calculate_star_metrics(time, flux):
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    finite_time = time[finite]
    finite_flux = flux[finite]
    return {"n_grid_observations": int(len(time)), "n_finite_observations": int(finite.sum()), "finite_fraction": float(finite.mean()), "gap_fraction": float(1.0 - finite.mean()), "baseline_days": float(np.max(finite_time) - np.min(finite_time)), "median_cadence_days": median_cadence(finite_time), "robust_flux_scatter_ppm": float(robust_scale(finite_flux) * 1.0e6), "lag_one_flux_acf": lag_one_acf(flux)}

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

def base_arima_cache_paths(star_dir):
    star_dir = Path(star_dir)
    return {"innovations": star_dir / "base_arima_innovations.npy", "params": star_dir / "base_arima_params.npy", "summary": star_dir / "base_arima_summary.json"}

def base_arima_cache_is_compatible(summary, args, flux):
    return tuple(summary.get("order", ())) == tuple(args["arima_order"]) and int(summary.get("fit_maxiter", -1)) == int(args["fit_maxiter"]) and int(summary.get("flux_length", -1)) == len(flux)

def load_base_arima_cache_from_dir(star_dir, flux, args):
    paths = base_arima_cache_paths(star_dir)
    if not all(path.exists() for path in paths.values()):
        return None
    try:
        summary = json.loads(paths["summary"].read_text())
        if not base_arima_cache_is_compatible(summary, args, flux):
            return None
        return {"innovations": np.load(paths["innovations"]), "params": np.load(paths["params"]), "fit": None, "summary": summary, "from_cache": True}
    except Exception:
        return None

def save_base_arima_cache(star_dir, base_arima, flux, args):
    paths = base_arima_cache_paths(star_dir)
    Path(star_dir).mkdir(parents=True, exist_ok=True)
    summary = dict(base_arima["summary"])
    summary["fit_maxiter"] = int(args["fit_maxiter"])
    summary["flux_length"] = int(len(flux))
    params = base_arima.get("params")
    if params is None and base_arima.get("fit") is not None:
        params = getattr(base_arima["fit"], "params", None)
    if params is None:
        return
    np.save(paths["innovations"], np.asarray(base_arima["innovations"], dtype=float))
    np.save(paths["params"], np.asarray(params, dtype=float))
    paths["summary"].write_text(json.dumps(json_ready(summary), indent=2) + "\n")

def load_or_fit_base_arima(star_dir, target_id, quarter, flux, args):
    cached = load_base_arima_cache_from_dir(star_dir, flux, args)
    if cached is not None:
        return cached, 0.0, "local_cache"
    existing_dir = Path(args.get("existing_arima_cache_root", "")) / star_prefix(target_id, quarter)
    cached = load_base_arima_cache_from_dir(existing_dir, flux, args)
    if cached is not None:
        save_base_arima_cache(star_dir, cached, flux, args)
        return cached, 0.0, "existing_multistar_cache"
    started = perf_counter()
    result = fit_arima_innovations(flux, order=tuple(args["arima_order"]), maxiter=args["fit_maxiter"])
    runtime = float(perf_counter() - started)
    params = getattr(result["fit"], "params", None)
    result["params"] = np.asarray(params, dtype=float) if params is not None else None
    result["from_cache"] = False
    save_base_arima_cache(star_dir, result, flux, args)
    return result, runtime, "fit"

def filter_arima_innovations(flux, base_arima, args):
    params = base_arima.get("params")
    if params is None and base_arima.get("fit") is not None:
        params = getattr(base_arima["fit"], "params", None)
    if params is None:
        raise ValueError("ARIMA parameters are unavailable.")
    series = np.asarray(flux, dtype=float).reshape(-1)
    model = ARIMA(series, order=tuple(args["arima_order"]), trend="n", enforce_stationarity=False, enforce_invertibility=False)
    filtered = model.filter(np.asarray(params, dtype=float))
    innovations = np.asarray(filtered.resid, dtype=float).reshape(-1)
    innovations[~np.isfinite(series)] = np.nan
    burn = int(getattr(filtered, "loglikelihood_burn", 0))
    if burn > 0:
        innovations[:burn] = np.nan
    base_summary = dict(base_arima.get("summary", {}))
    return {"innovations": innovations, "fit": filtered, "params": np.asarray(params, dtype=float), "summary": {"order": tuple(args["arima_order"]), "converged": bool(base_summary.get("converged", True)), "base_fit_converged": bool(base_summary.get("converged", True)), "fit_performed_for_injection": False, "filter_succeeded": True, "aic": float(filtered.aic), "bic": float(filtered.bic), "finite_innovation_count": int(np.isfinite(innovations).sum()), "mode": "filter_fixed_base_params", "base_optimizer_warnflag": base_summary.get("optimizer_warnflag"), "base_optimizer_iterations": base_summary.get("optimizer_iterations"), "base_optimizer_function_calls": base_summary.get("optimizer_function_calls")}}

def fit_gp(time, values, args):
    return fit_smooth_gp_background(time, values, max_train_points=args["gp_max_train_points"], length_scale_days=args["gp_length_scale_days"], min_length_scale_days=args["gp_min_length_scale_days"], max_length_scale_days=args["gp_max_length_scale_days"], measurement_noise_fraction=args["gp_measurement_noise_fraction"], n_restarts_optimizer=args["gp_n_restarts_optimizer"], random_seed=args["gp_random_seed"], optimize_kernel=args["gp_optimize_kernel"])

def fit_base_kalman(flux, args):
    started = perf_counter()
    model = fit_kalman_local_level(flux, maxiter=args["kalman_maxiter"], burn_in=args["kalman_burn_in"])
    return model, float(perf_counter() - started)

def fit_base_gp(time, flux, args):
    started = perf_counter()
    model = fit_gp(time, flux, args)
    prepared = prepare_smooth_gp_filter(time, model)
    return model, prepared, float(perf_counter() - started)

def top_peak_rank(peaks, target_period, tolerance_fraction):
    for fallback_rank, row in enumerate(peaks.to_dict(orient="records"), start=1):
        period = float(row.get("period_days", row.get("period", np.nan)))
        rank = int(row.get("rank", fallback_rank))
        error = abs(period - float(target_period)) / float(target_period)
        if np.isfinite(error) and error <= float(tolerance_fraction):
            return rank
    return None

def harmonic_rank(peaks, target_period, factor, tolerance_fraction):
    return top_peak_rank(peaks, float(target_period) * float(factor), tolerance_fraction)

def match_fields(period, injected_period, tolerance):
    harmonic_error = float(period_match_fraction(period, injected_period))
    exact_error = float(abs(float(period) - float(injected_period)) / float(injected_period))
    return harmonic_error, exact_error, bool(harmonic_error <= float(tolerance)), bool(exact_error <= float(tolerance))

def detector_summary_period_days(summary):
    if "period_days" in summary:
        return float(summary["period_days"])
    return float(summary["period"])

def detector_summary_duration_hours(summary):
    if "duration_hours" in summary:
        return float(summary["duration_hours"])
    if "duration_days" in summary:
        return float(summary["duration_days"]) * 24.0
    return float(summary["duration"]) * 24.0

def detector_summary_epoch_days(summary):
    if "epoch_days" in summary:
        return float(summary["epoch_days"])
    if "transit_time" in summary:
        return float(summary["transit_time"])
    return float(summary.get("epoch", np.nan))

def detector_summary_score(summary, detector):
    if detector == "bls":
        return float(summary["sde"])
    if detector == "tls":
        return float(summary["sde"])
    if detector == "tps_like":
        return float(summary["mes"])
    return float(summary["score"])

def peak_score_series(peaks, detector):
    for column in ("score", "sde", "mes", "power"):
        if column in peaks.columns:
            return pd.to_numeric(peaks[column], errors="coerce").to_numpy(dtype=float)
    return np.full(len(peaks), np.nan, dtype=float)

def peak_duration_hours(peak):
    if "duration_hours" in peak and np.isfinite(float(peak["duration_hours"])):
        return float(peak["duration_hours"])
    if "duration_days" in peak and np.isfinite(float(peak["duration_days"])):
        return float(peak["duration_days"]) * 24.0
    if "duration" in peak and np.isfinite(float(peak["duration"])):
        return float(peak["duration"]) * 24.0
    return float("nan")

def detector_result_fields(pipeline, result, detector, injected_period, args, runtime, base_rank1_period=None):
    summary = result["summary"]
    peaks = result["top_peaks"]
    period = detector_summary_period_days(summary)
    harmonic_error, exact_error, harmonic_matched, exact_matched = match_fields(period, injected_period, args["period_match_tolerance_fraction"])
    exact_rank = top_peak_rank(peaks, injected_period, args["period_match_tolerance_fraction"])
    half_rank = harmonic_rank(peaks, injected_period, 0.5, args["period_match_tolerance_fraction"])
    double_rank = harmonic_rank(peaks, injected_period, 2.0, args["period_match_tolerance_fraction"])
    harmonic_topk = any(rank is not None for rank in (exact_rank, half_rank, double_rank))
    base_rank1_period = float(base_rank1_period) if base_rank1_period is not None and np.isfinite(base_rank1_period) else float("nan")
    base_rank1_error = abs(period - base_rank1_period) / base_rank1_period if np.isfinite(base_rank1_period) and base_rank1_period > 0 else float("nan")
    matches_base_rank1 = bool(np.isfinite(base_rank1_error) and base_rank1_error <= float(args["period_match_tolerance_fraction"]))
    if harmonic_matched:
        failure_mode = "rank1_recovery"
    elif harmonic_topk:
        failure_mode = "ranking_failure"
    elif matches_base_rank1:
        failure_mode = "base_rank1_competition"
    else:
        failure_mode = "candidate_generation_failure"
    fields = {f"{pipeline}_success": True, f"{pipeline}_runtime_seconds": float(runtime), f"{pipeline}_recovered_period_days": period, f"{pipeline}_period_error_fraction": harmonic_error, f"{pipeline}_exact_period_error_fraction": exact_error, f"{pipeline}_harmonic_rank1_matched": harmonic_matched, f"{pipeline}_exact_rank1_matched": exact_matched, f"{pipeline}_harmonic_topk_matched": harmonic_topk, f"{pipeline}_exact_rank_topk": exact_rank, f"{pipeline}_half_period_rank_topk": half_rank, f"{pipeline}_double_period_rank_topk": double_rank, f"{pipeline}_base_rank1_period_days": base_rank1_period, f"{pipeline}_base_rank1_error_fraction": base_rank1_error, f"{pipeline}_matches_base_rank1": matches_base_rank1, f"{pipeline}_failure_mode": failure_mode, f"{pipeline}_top_periods_json": json.dumps([float(row.get("period_days", row.get("period", np.nan))) for row in peaks.to_dict(orient="records")])}
    fields.update({
        f"{pipeline}_score": detector_summary_score(summary, detector),
        f"{pipeline}_duration_hours": detector_summary_duration_hours(summary),
        f"{pipeline}_epoch_days": detector_summary_epoch_days(summary),
        f"{pipeline}_top_scores_json": json.dumps([float(value) for value in peak_score_series(peaks, detector)]),
    })
    if detector == "bls":
        fields.update({f"{pipeline}_power": float(summary["power"]), f"{pipeline}_transit_time": float(summary["transit_time"]), f"{pipeline}_depth": float(summary["depth"])})
    elif detector == "tcf":
        fields.update({f"{pipeline}_raw_pooled_score": float(summary["raw_pooled_score"]), f"{pipeline}_valid_transit_events": int(summary["n_valid_transit_events"]), f"{pipeline}_positive_event_fraction": float(summary["positive_event_fraction"])})
    elif detector == "tls":
        fields.update({f"{pipeline}_tls_snr": float(summary["snr"]), f"{pipeline}_depth_raw": float(summary["depth_raw"]), f"{pipeline}_n_observations": int(summary["n_observations"])})
    elif detector == "trapezoid":
        fields.update({f"{pipeline}_depth": float(summary["depth"]), f"{pipeline}_ingress_fraction": float(summary["ingress_fraction"]), f"{pipeline}_bls_seed_rank": int(summary["seed_rank"]), f"{pipeline}_seed_source": "bls_top_peaks"})
    elif detector == "tps_like":
        fields.update({f"{pipeline}_max_ses": float(summary["max_ses"]), f"{pipeline}_observed_event_count": int(summary["observed_event_count"]), f"{pipeline}_expected_event_count": int(summary["expected_event_count"]), f"{pipeline}_observability_fraction": float(summary["observability_fraction"]), f"{pipeline}_period_cadences": int(summary["period_cadences"]), f"{pipeline}_duration_cadences": int(summary["duration_cadences"]), f"{pipeline}_segment_count": int(summary["segment_count"]), f"{pipeline}_wavelet": str(summary["wavelet"])})
    return fields

def empty_pipeline_fields(pipeline, error):
    return {f"{pipeline}_success": False, f"{pipeline}_error": str(error), f"{pipeline}_runtime_seconds": float("nan"), f"{pipeline}_recovered_period_days": float("nan"), f"{pipeline}_score": float("nan"), f"{pipeline}_period_error_fraction": float("nan"), f"{pipeline}_exact_period_error_fraction": float("nan"), f"{pipeline}_harmonic_rank1_matched": False, f"{pipeline}_exact_rank1_matched": False, f"{pipeline}_harmonic_topk_matched": False, f"{pipeline}_exact_rank_topk": None, f"{pipeline}_half_period_rank_topk": None, f"{pipeline}_double_period_rank_topk": None, f"{pipeline}_base_rank1_period_days": float("nan"), f"{pipeline}_base_rank1_error_fraction": float("nan"), f"{pipeline}_matches_base_rank1": False, f"{pipeline}_failure_mode": "pipeline_error"}

def add_branch_diagnostics(row, branch, series, before, in_transit):
    after = periodic_depth_and_snr(series, in_transit)
    row[f"{branch}_residual_depth"] = float(after["depth"])
    row[f"{branch}_local_snr"] = float(after["snr"])
    row[f"{branch}_depth_retention_fraction"] = float(after["depth"] / before["depth"]) if before["depth"] != 0 else float("nan")
    row[f"{branch}_snr_retention_fraction"] = float(after["snr"] / before["snr"]) if before["snr"] != 0 else float("nan")
    row[f"{branch}_residual_std"] = float(np.nanstd(series, ddof=1))
    row[f"{branch}_residual_acf1"] = lag_one_acf(series)

def required_branches(pipelines):
    return sorted({PIPELINE_DEFINITIONS[pipeline][0] for pipeline in pipelines})

def required_detectors(pipelines):
    return sorted({PIPELINE_DEFINITIONS[pipeline][1] for pipeline in pipelines})

def prepare_branch_detector_caches(branch_series, segment_id, pipelines, args):
    caches = {branch: {} for branch in branch_series}
    if "tps_like" not in required_detectors(pipelines):
        return caches
    for branch, values in branch_series.items():
        try:
            caches[branch]["tps_like_noise_model"] = prepare_tps_like_noise_model(
                values,
                segment_id,
                wavelet=args["tps_wavelet"],
                max_level=args["tps_max_wavelet_level"],
                noise_window_cadences=args["tps_noise_window_cadences"],
                min_segment_cadences=args["tps_min_segment_cadences"],
            )
        except Exception as exc:
            caches[branch]["tps_like_noise_model_error"] = f"{type(exc).__name__}: {exc}"
    return caches

def base_candidate_rows(time, branch_series, segment_id, period_grid, duration_grid, target_id, quarter, selection_group, sample_stratum, args, detector_caches=None):
    rows = []
    rank1 = {}
    detector_caches = detector_caches or {branch: {} for branch in branch_series}
    for pipeline in args["pipelines"]:
        branch, detector = PIPELINE_DEFINITIONS[pipeline]
        if branch not in branch_series:
            rows.append({"target_id": normalize_target_id(target_id), "quarter": int(quarter), "selection_group": str(selection_group), "sample_stratum": str(sample_stratum), "pipeline": pipeline, "branch": branch, "detector": detector, "success": False, "rank": 0, "period_days": np.nan, "score": np.nan, "error": "branch unavailable"})
            rank1[pipeline] = np.nan
            continue
        try:
            cache = detector_caches.setdefault(branch, {})
            result = run_detector_search(detector, time, branch_series[branch], period_grid, duration_grid, args, segment_id=segment_id, cache=cache)
            peaks = result["top_peaks"]
            rank1[pipeline] = detector_summary_period_days(result["summary"])
            for fallback_rank, peak in enumerate(peaks.to_dict(orient="records"), start=1):
                period = float(peak.get("period_days", peak.get("period", np.nan)))
                score = float(peak.get("score", peak.get("sde", peak.get("mes", peak.get("power", np.nan)))))
                rows.append({"target_id": normalize_target_id(target_id), "quarter": int(quarter), "selection_group": str(selection_group), "sample_stratum": str(sample_stratum), "pipeline": pipeline, "branch": branch, "detector": detector, "success": True, "rank": int(peak.get("rank", fallback_rank)), "period_days": period, "score": score, "duration_hours": peak_duration_hours(peak), "error": ""})
        except Exception as exc:
            rank1[pipeline] = np.nan
            rows.append({"target_id": normalize_target_id(target_id), "quarter": int(quarter), "selection_group": str(selection_group), "sample_stratum": str(sample_stratum), "pipeline": pipeline, "branch": branch, "detector": detector, "success": False, "rank": 0, "period_days": np.nan, "score": np.nan, "error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows), rank1

def base_branch_series(flux, base_arima, base_kalman, base_gp, pipelines):
    branches = required_branches(pipelines)
    values = {}
    if "raw" in branches:
        values["raw"] = np.asarray(flux, dtype=float)
    if "arima" in branches and base_arima is not None:
        values["arima"] = np.asarray(base_arima["innovations"], dtype=float)
    if "kalman" in branches and base_kalman is not None:
        values["kalman"] = np.asarray(base_kalman.residuals, dtype=float)
    if "gp" in branches and base_gp is not None:
        values["gp"] = np.asarray(base_gp.residuals, dtype=float)
    return values

def run_one_case(case_index, case, time, flux, segment_id, period_grid, duration_grid, base_arima, base_kalman, prepared_gp, target_id, quarter, selection_group, sample_stratum, base_rank1_periods, base_detector_caches, args):
    injected_period, injected_duration_hours, injected_depth, epoch_phase_fraction = case
    finite = np.isfinite(time) & np.isfinite(flux)
    epoch = float(np.min(time[finite]) + float(epoch_phase_fraction) * float(injected_period))
    duration_days = float(injected_duration_hours) / 24.0
    injected_flux, template, in_transit = inject_periodic_box_transit(time, flux, injected_period, epoch, duration_days, injected_depth)
    before = periodic_depth_and_snr(injected_flux, in_transit)
    row = {"target_id": normalize_target_id(target_id), "quarter": int(quarter), "selection_group": str(selection_group), "sample_stratum": str(sample_stratum), "case_index": int(case_index), "injected_period_days": float(injected_period), "injected_epoch_days": epoch, "epoch_phase_fraction": float(epoch_phase_fraction), "injected_duration_hours": float(injected_duration_hours), "injected_depth": float(injected_depth), "in_transit_observation_count": int(np.isfinite(flux[in_transit]).sum()), "observed_depth_before_model": float(before["depth"]), "local_snr_before_model": float(before["snr"])}
    branch_series = {}
    branch_errors = {}
    if "raw" in required_branches(args["pipelines"]):
        branch_series["raw"] = injected_flux
        add_branch_diagnostics(row, "raw", injected_flux, before, in_transit)
    if "arima" in required_branches(args["pipelines"]):
        try:
            started = perf_counter()
            arima = filter_arima_innovations(injected_flux, base_arima, args) if args["arima_injection_mode"] == "filter" else fit_arima_innovations(injected_flux, order=tuple(args["arima_order"]), maxiter=args["fit_maxiter"])
            branch_series["arima"] = arima["innovations"]
            row["arima_model_runtime_seconds"] = float(perf_counter() - started)
            row["arima_injection_mode"] = str(args["arima_injection_mode"])
            row["arima_converged"] = bool(arima["summary"].get("converged", True))
            row["arima_fit_performed_for_injection"] = bool(args["arima_injection_mode"] != "filter")
            row["arima_filter_succeeded"] = bool(args["arima_injection_mode"] == "filter")
            row["arima_base_fit_converged"] = bool(base_arima["summary"].get("converged", True)) if base_arima is not None else np.nan
            row["arima_base_optimizer_warnflag"] = base_arima["summary"].get("optimizer_warnflag") if base_arima is not None else None
            row["arima_base_optimizer_iterations"] = base_arima["summary"].get("optimizer_iterations") if base_arima is not None else None
            row["arima_base_optimizer_function_calls"] = base_arima["summary"].get("optimizer_function_calls") if base_arima is not None else None
            add_branch_diagnostics(row, "arima", arima["innovations"], before, in_transit)
        except Exception as exc:
            branch_errors["arima"] = f"{type(exc).__name__}: {exc}"
            row["arima_error"] = branch_errors["arima"]
    if "kalman" in required_branches(args["pipelines"]):
        try:
            started = perf_counter()
            if args["kalman_injection_mode"] == "filter":
                if base_kalman is None:
                    raise ValueError("Base Kalman fit is unavailable for fixed-parameter filtering.")
                model = apply_fitted_kalman_filter(injected_flux, base_kalman, burn_in=args["kalman_burn_in"])
            else:
                model = fit_kalman_local_level(injected_flux, maxiter=args["kalman_maxiter"], burn_in=args["kalman_burn_in"])
            branch_series["kalman"] = model.residuals
            row["kalman_model_runtime_seconds"] = float(perf_counter() - started)
            row["kalman_injection_mode"] = str(args["kalman_injection_mode"])
            row["kalman_converged"] = bool(model.converged)
            row["kalman_log_likelihood"] = float(model.log_likelihood)
            row["kalman_process_variance"] = float(model.parameters["process_variance"])
            row["kalman_measurement_variance"] = float(model.parameters["measurement_variance"])
            add_branch_diagnostics(row, "kalman", model.residuals, before, in_transit)
        except Exception as exc:
            branch_errors["kalman"] = f"{type(exc).__name__}: {exc}"
            row["kalman_error"] = branch_errors["kalman"]
    if "gp" in required_branches(args["pipelines"]):
        try:
            started = perf_counter()
            if args["gp_injection_mode"] == "filter":
                if prepared_gp is None:
                    raise ValueError("Prepared GP filter is unavailable for fixed-hyperparameter filtering.")
                model = apply_prepared_smooth_gp_filter(injected_flux, prepared_gp)
            else:
                model = fit_gp(time, injected_flux, args)
            branch_series["gp"] = model.residuals
            row["gp_model_runtime_seconds"] = float(perf_counter() - started)
            row["gp_injection_mode"] = str(args["gp_injection_mode"])
            row["gp_converged"] = bool(model.converged)
            row["gp_log_marginal_likelihood"] = float(model.log_marginal_likelihood)
            row["gp_training_point_count"] = int(model.parameters["training_point_count"])
            row["gp_length_scale_days"] = float(model.parameters["length_scale_days"])
            row["gp_measurement_noise_variance"] = float(model.parameters["measurement_noise_variance"])
            row["gp_fit_performed_for_injection"] = bool(args["gp_injection_mode"] != "filter")
            row["gp_filter_succeeded"] = bool(args["gp_injection_mode"] == "filter")
            row["gp_base_fit_converged"] = bool(prepared_gp.converged) if prepared_gp is not None else np.nan
            row["gp_base_fit_status"] = int(prepared_gp.status) if prepared_gp is not None else np.nan
            row["gp_base_fit_message"] = str(prepared_gp.message) if prepared_gp is not None else ""
            row["gp_length_scale_at_lower_bound"] = bool(model.parameters.get("length_scale_at_lower_bound", False))
            row["gp_length_scale_at_upper_bound"] = bool(model.parameters.get("length_scale_at_upper_bound", False))
            row["gp_signal_variance_at_lower_bound"] = bool(model.parameters.get("signal_variance_at_lower_bound", False))
            row["gp_signal_variance_at_upper_bound"] = bool(model.parameters.get("signal_variance_at_upper_bound", False))
            row["gp_optimizer_warning_count"] = int(model.parameters.get("optimizer_warning_count", 0))
            row["gp_optimizer_warning_message"] = str(model.parameters.get("optimizer_warning_message", ""))
            add_branch_diagnostics(row, "gp", model.residuals, before, in_transit)
        except Exception as exc:
            branch_errors["gp"] = f"{type(exc).__name__}: {exc}"
            row["gp_error"] = branch_errors["gp"]
    case_detector_caches = {branch: dict(base_detector_caches.get(branch, {})) for branch in branch_series}
    for pipeline in args["pipelines"]:
        branch, detector = PIPELINE_DEFINITIONS[pipeline]
        if branch not in branch_series:
            row.update(empty_pipeline_fields(pipeline, branch_errors.get(branch, "branch unavailable")))
            continue
        try:
            started = perf_counter()
            detector_cache = case_detector_caches.setdefault(branch, dict(base_detector_caches.get(branch, {})))
            result = run_detector_search(detector, time, branch_series[branch], period_grid, duration_grid, args, segment_id=segment_id, cache=detector_cache)
            row.update(detector_result_fields(pipeline, result, detector, injected_period, args, perf_counter() - started, base_rank1_period=base_rank1_periods.get(pipeline)))
            row[f"{pipeline}_error"] = ""
        except Exception as exc:
            row.update(empty_pipeline_fields(pipeline, f"{type(exc).__name__}: {exc}"))
    return row

def case_key(case):
    return tuple(round(float(value), 12) for value in case)

def row_case_key(row):
    return case_key((row["injected_period_days"], row["injected_duration_hours"], row["injected_depth"], row["epoch_phase_fraction"]))

def star_config_matches(star_dir, args):
    path = Path(star_dir) / "run_config.json"
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("config_signature") == json_ready(config_signature(SimpleNamespace(**args) if isinstance(args, dict) else args))
    except Exception:
        return False

def load_existing_injection_rows(star_dir, cases, args, resume_compatible=None):
    path = Path(star_dir) / "injections.csv"
    compatible = star_config_matches(star_dir, args) if resume_compatible is None else bool(resume_compatible)
    if not args.get("resume", True) or not path.exists() or not compatible:
        return [], set()
    frame = pd.read_csv(path)
    if "case_index" not in frame.columns:
        return [], set()
    rows = []
    completed = set()
    for row in frame.to_dict(orient="records"):
        case_index = int(row["case_index"])
        if case_index < 0 or case_index >= len(cases):
            continue
        if row_case_key(row) != case_key(cases[case_index]):
            continue
        rows.append(row)
        completed.add(case_index)
    return rows, completed

def save_rows(path, rows):
    frame = pd.DataFrame(sorted(rows, key=lambda row: int(row["case_index"]))) if rows else pd.DataFrame()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame

def run_star_injections(star_dir, cases, time, flux, segment_id, period_grid, duration_grid, base_arima, base_kalman, prepared_gp, target_id, quarter, selection_group, sample_stratum, base_rank1_periods, base_detector_caches, args, progress_queue, resume_compatible=None):
    rows, completed = load_existing_injection_rows(star_dir, cases, args, resume_compatible=resume_compatible)
    if completed:
        report_progress(progress_queue, target_id, quarter, "injections resumed", units=len(completed), detail=f"{len(completed)}/{len(cases)}")
    unreported = 0
    progress_interval = max(1, int(args.get("progress_interval", 1)))
    checkpoint_interval = max(1, int(args.get("checkpoint_interval", 5)))
    for case_index, case in enumerate(cases):
        if case_index in completed:
            continue
        rows.append(run_one_case(case_index, case, time, flux, segment_id, period_grid, duration_grid, base_arima, base_kalman, prepared_gp, target_id, quarter, selection_group, sample_stratum, base_rank1_periods, base_detector_caches, args))
        completed.add(case_index)
        unreported += 1
        if len(completed) % checkpoint_interval == 0 or len(completed) == len(cases):
            save_rows(Path(star_dir) / "injections.csv", rows)
            write_star_checkpoint(star_dir, target_id, quarter, "running", "injections", completed_injections=len(completed), requested_injections=len(cases))
        if unreported >= progress_interval or len(completed) == len(cases):
            report_progress(progress_queue, target_id, quarter, "injection", units=unreported, detail=f"{len(completed)}/{len(cases)}")
            unreported = 0
    return save_rows(Path(star_dir) / "injections.csv", rows)

def prepare_star_run(star_dir, args):
    star_dir = Path(star_dir)
    star_dir.mkdir(parents=True, exist_ok=True)
    compatible = star_config_matches(star_dir, args)
    if not compatible:
        for name in ("COMPLETE", "injections.csv", "base_light_curve_candidates.csv", "star_summary.json", "stellar_characterization.json", "failure.json"):
            path = star_dir / name
            if path.exists():
                path.unlink()
    (star_dir / "run_config.json").write_text(json.dumps({"config_signature": json_ready(config_signature(SimpleNamespace(**args) if isinstance(args, dict) else args))}, indent=2) + "\n")
    return compatible

def run_star_task(task):
    row, args, progress_queue = task
    target_id = normalize_target_id(row["target_id"])
    quarter = int(row["quarter"])
    selection_group = str(row.get("selection_group", "unspecified"))
    sample_stratum = str(row.get("sample_stratum", "unspecified"))
    star_dir = Path(args["output_dir"]) / "stars" / star_prefix(target_id, quarter)
    started = perf_counter()
    try:
        resume_compatible = prepare_star_run(star_dir, args)
        write_star_checkpoint(star_dir, target_id, quarter, "running", "started")
        light_curve_frame, cache_hit = load_light_curve_frame(target_id, quarter, args, progress_queue=progress_queue)
        regular, preprocessing = preprocess_pdcsap_light_curve(light_curve_frame, quality_policy=args["quality_policy"], require_finite_flux_error=args["require_finite_flux_error"], normalization_fit_fraction=1.0 - args["test_fraction"])
        time = regular["time"].to_numpy(dtype=float)
        flux = regular["normalized_flux"].to_numpy(dtype=float)
        segment_id = regular["segment_id"].to_numpy(dtype=int)
        if np.isfinite(time).sum() < 24 or np.isfinite(flux).sum() < 24:
            raise ValueError("Insufficient finite observations.")

        # Characterize the observed stellar background ONCE before any synthetic
        # transit is injected.  These are the X_star features for later model
        # selection / routing.  They must never depend on injected transit truth.
        report_progress(progress_queue, target_id, quarter, "stellar characterization", detail="v2 features")
        characterization = characterize_regularized_light_curve(
            regular,
            target_id=target_id,
            quarter=quarter,
            preprocessing_summary=preprocessing.to_dict(),
            variability_acf_lags=int(args["characterization_acf_lags"]),
            variability_spectral_frequencies=int(args["characterization_spectral_frequencies"]),
        )
        (star_dir / "stellar_characterization.json").write_text(json.dumps(json_ready(characterization), indent=2) + "\n")

        period_grid = default_period_grid(time, min_period_days=args["min_period_days"], max_period_days=args["max_period_days"], n_periods=args["n_periods"])
        duration_grid = default_duration_grid(args["min_duration_hours"], args["max_duration_hours"], args["n_durations"])
        branches = required_branches(args["pipelines"])
        base_arima = None
        base_arima_runtime = 0.0
        base_arima_source = "not_required"
        if "arima" in branches:
            base_arima, base_arima_runtime, base_arima_source = load_or_fit_base_arima(star_dir, target_id, quarter, flux, args)
        base_kalman = None
        base_kalman_runtime = 0.0
        if "kalman" in branches:
            report_progress(progress_queue, target_id, quarter, "base Kalman fit", detail="fit once for base-candidate diagnostics")
            base_kalman, base_kalman_runtime = fit_base_kalman(flux, args)
        base_gp = None
        prepared_gp = None
        base_gp_runtime = 0.0
        if "gp" in branches:
            report_progress(progress_queue, target_id, quarter, "base GP fit", detail="fit once for base-candidate diagnostics")
            base_gp, prepared_base_gp, base_gp_runtime = fit_base_gp(time, flux, args)
            if args["gp_injection_mode"] == "filter":
                prepared_gp = prepared_base_gp
        base_series = base_branch_series(flux, base_arima, base_kalman, base_gp, args["pipelines"])
        base_detector_caches = prepare_branch_detector_caches(base_series, segment_id, args["pipelines"], args)
        base_candidate_caches = {branch: dict(cache) for branch, cache in base_detector_caches.items()}
        base_candidates, base_rank1_periods = base_candidate_rows(time, base_series, segment_id, period_grid, duration_grid, target_id, quarter, selection_group, sample_stratum, args, detector_caches=base_candidate_caches)
        base_candidates.to_csv(star_dir / "base_light_curve_candidates.csv", index=False)
        cases = [tuple(case) for case in args["cases"]]
        injections = run_star_injections(star_dir, cases, time, flux, segment_id, period_grid, duration_grid, base_arima, base_kalman, prepared_gp, target_id, quarter, selection_group, sample_stratum, base_rank1_periods, base_detector_caches, args, progress_queue, resume_compatible=resume_compatible)
        successful = injections.copy()
        star_metrics = calculate_star_metrics(time, flux)
        # Only continuous, pre-injection stellar-background measurements enter the
        # future model-selection feature matrix.  Human-readable morphology flags
        # are stored separately for interpretation and visual review.
        characterization_features = {key: characterization.get(key) for key in MODEL_SELECTION_FEATURE_COLUMNS}
        characterization_flags = {
            key: characterization.get(key)
            for key in (
                "scientific_characterization_version",
                "v2_periodicity_screen_pass",
                "v2_coherent_periodic_candidate",
                "v2_quasi_periodic_candidate",
                "v2_rotation_spot_review_flag",
                "v2_pulsation_review_flag",
                "v2_low_scatter_structured_candidate",
            )
        }
        summary = {"target_id": target_id, "quarter": quarter, "selection_group": selection_group, "sample_stratum": sample_stratum, "status": "success", "profile": str(args["profile"]), "search_resolution": str(args["search_resolution"]), "pipelines": list(args["pipelines"]), "runtime_seconds": float(perf_counter() - started), "light_curve_cache_hit": bool(cache_hit), "base_arima_source": str(base_arima_source), "base_arima_runtime_seconds": float(base_arima_runtime), "base_arima_converged": bool(base_arima["summary"].get("converged", True)) if base_arima is not None else None, "kalman_injection_mode": str(args["kalman_injection_mode"]), "base_kalman_runtime_seconds": float(base_kalman_runtime), "base_kalman_converged": bool(base_kalman.converged) if base_kalman is not None else None, "gp_injection_mode": str(args["gp_injection_mode"]), "base_gp_runtime_seconds": float(base_gp_runtime), "base_gp_converged": bool(base_gp.converged) if base_gp is not None else None, "base_gp_length_scale_days": float(base_gp.parameters["length_scale_days"]) if base_gp is not None else None, "injection_count_requested": int(len(cases)), "injection_count_completed": int(len(successful)), **star_metrics, **characterization_features, **characterization_flags}
        summary.update({"arima_injection_mode": str(args["arima_injection_mode"]), "base_arima_optimizer_warnflag": base_arima["summary"].get("optimizer_warnflag") if base_arima is not None else None, "base_arima_optimizer_iterations": base_arima["summary"].get("optimizer_iterations") if base_arima is not None else None, "base_arima_optimizer_function_calls": base_arima["summary"].get("optimizer_function_calls") if base_arima is not None else None, "base_gp_status": int(base_gp.status) if base_gp is not None else None, "base_gp_message": str(base_gp.message) if base_gp is not None else None, "base_gp_length_scale_at_lower_bound": bool(base_gp.parameters.get("length_scale_at_lower_bound", False)) if base_gp is not None else None, "base_gp_length_scale_at_upper_bound": bool(base_gp.parameters.get("length_scale_at_upper_bound", False)) if base_gp is not None else None, "base_gp_signal_variance_at_lower_bound": bool(base_gp.parameters.get("signal_variance_at_lower_bound", False)) if base_gp is not None else None, "base_gp_signal_variance_at_upper_bound": bool(base_gp.parameters.get("signal_variance_at_upper_bound", False)) if base_gp is not None else None, "base_gp_optimizer_warning_count": int(base_gp.parameters.get("optimizer_warning_count", 0)) if base_gp is not None else None, "base_gp_optimizer_warning_message": str(base_gp.parameters.get("optimizer_warning_message", "")) if base_gp is not None else None})
        for pipeline in args["pipelines"]:
            column = f"{pipeline}_harmonic_rank1_matched"
            exact_column = f"{pipeline}_exact_rank1_matched"
            summary[f"{pipeline}_success_count"] = int(successful[f"{pipeline}_success"].astype(bool).sum()) if f"{pipeline}_success" in successful.columns else 0
            topk_column = f"{pipeline}_harmonic_topk_matched"
            summary[f"{pipeline}_harmonic_rank1_rate"] = float(successful[column].astype(bool).mean()) if column in successful.columns and len(successful) else float("nan")
            summary[f"{pipeline}_harmonic_topk_rate"] = float(successful[topk_column].astype(bool).mean()) if topk_column in successful.columns and len(successful) else float("nan")
            summary[f"{pipeline}_ranking_gap_rate"] = float((successful[topk_column].astype(bool) & ~successful[column].astype(bool)).mean()) if topk_column in successful.columns and column in successful.columns and len(successful) else float("nan")
            summary[f"{pipeline}_exact_rank1_rate"] = float(successful[exact_column].astype(bool).mean()) if exact_column in successful.columns and len(successful) else float("nan")
        (star_dir / "star_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
        if args["save_regularized_inputs"]:
            regular.to_parquet(star_dir / "regularized_light_curve.parquet", index=False)
        (star_dir / "COMPLETE").write_text("complete\n")
        write_star_checkpoint(star_dir, target_id, quarter, "success", "complete", runtime_seconds=summary["runtime_seconds"])
        return {"target_id": target_id, "quarter": quarter, "selection_group": selection_group, "sample_stratum": sample_stratum, "status": "success", "star_dir": str(star_dir), "runtime_seconds": summary["runtime_seconds"], "error": ""}
    except Exception as exc:
        failure = {"target_id": target_id, "quarter": quarter, "selection_group": selection_group, "sample_stratum": sample_stratum, "status": "failed", "star_dir": str(star_dir), "runtime_seconds": float(perf_counter() - started), "error": f"{type(exc).__name__}: {exc}"}
        star_dir.mkdir(parents=True, exist_ok=True)
        (star_dir / "failure.json").write_text(json.dumps(json_ready(failure), indent=2) + "\n")
        write_star_checkpoint(star_dir, target_id, quarter, "failed", "failed", error=failure["error"], runtime_seconds=failure["runtime_seconds"])
        return failure

def completed_star_is_compatible(star_dir, args):
    return (Path(star_dir) / "COMPLETE").exists() and star_config_matches(star_dir, args)

def completed_result(output_dir, row):
    target_id = normalize_target_id(row["target_id"])
    quarter = int(row["quarter"])
    star_dir = Path(output_dir) / "stars" / star_prefix(target_id, quarter)
    summary = json.loads((star_dir / "star_summary.json").read_text())
    return {"target_id": target_id, "quarter": quarter, "selection_group": str(row.get("selection_group", "unspecified")), "sample_stratum": str(row.get("sample_stratum", "unspecified")), "status": "success", "star_dir": str(star_dir), "runtime_seconds": float(summary.get("runtime_seconds", np.nan)), "error": ""}

def failure_result(output_dir, row):
    target_id = normalize_target_id(row["target_id"])
    quarter = int(row["quarter"])
    star_dir = Path(output_dir) / "stars" / star_prefix(target_id, quarter)
    try:
        failure = json.loads((star_dir / "failure.json").read_text())
    except Exception:
        failure = {}
    return {"target_id": target_id, "quarter": quarter, "selection_group": str(row.get("selection_group", "unspecified")), "sample_stratum": str(row.get("sample_stratum", "unspecified")), "status": "failed", "star_dir": str(star_dir), "runtime_seconds": float(failure.get("runtime_seconds", np.nan)), "error": str(failure.get("error", "Existing failure.json"))}

def resolve_worker_count(args, target_count):
    available = os.cpu_count() or 1
    reserve = max(0, int(getattr(args, "reserve_cpu_cores", 2)))
    requested = int(args.max_workers) if args.max_workers is not None else max(1, available - reserve)
    return max(1, min(requested, available, int(target_count)))

def settings_to_worker_dict(args):
    values = vars(args).copy()
    values["manifest_path"] = str(values["manifest_path"])
    values["cache_dir"] = str(values["cache_dir"])
    values["output_dir"] = str(values["output_dir"])
    values["background_feature_path"] = str(values["background_feature_path"])
    values["existing_arima_cache_root"] = str(values["existing_arima_cache_root"])
    values["cases"] = [tuple(case) for case in injection_cases(args)]
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

def prefetch_light_curve_task(task):
    row, args = task
    target_id = normalize_target_id(row["target_id"])
    quarter = int(row["quarter"])
    _, cache_hit = load_light_curve_frame(target_id, quarter, args)
    return {"target_id": target_id, "quarter": quarter, "cache_hit": bool(cache_hit)}

def prefetch_manifest_light_curves(rows, args):
    if not args.allow_download or int(args.prefetch_workers) <= 0:
        return []
    missing = [row for row in rows if not light_curve_cache_path({"cache_dir": str(args.cache_dir)}, row["target_id"], row["quarter"]).exists()]
    if not missing:
        return []
    worker_count = max(1, min(int(args.prefetch_workers), len(missing)))
    prefetch_args = {
        "cache_dir": str(args.cache_dir),
        "allow_download": True,
        "download_connect_timeout_seconds": float(
            getattr(args, "download_connect_timeout_seconds", None) or DEFAULT_MAST_CONNECT_TIMEOUT_SECONDS
        ),
        "download_read_timeout_seconds": float(
            getattr(args, "download_read_timeout_seconds", None) or DEFAULT_MAST_READ_TIMEOUT_SECONDS
        ),
        "download_max_attempts": int(args.download_max_attempts),
        "download_initial_wait_seconds": float(args.download_initial_wait_seconds),
        "download_backoff_factor": float(args.download_backoff_factor),
    }
    results = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {executor.submit(prefetch_light_curve_task, (row, prefetch_args)): row for row in missing}
        with tqdm(total=len(future_map), desc="Prefetch light curves", unit="star", dynamic_ncols=True, mininterval=0.25) as progress:
            for future in as_completed(future_map):
                row = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"target_id": normalize_target_id(row["target_id"]), "quarter": int(row["quarter"]), "error": f"{type(exc).__name__}: {exc}"}
                    tqdm.write(f"Prefetch failed for KIC {result['target_id']} Q{result['quarter']}: {result['error']}")
                results.append(result)
                progress.set_postfix_str(f"KIC {result['target_id']} Q{result['quarter']}")
                progress.update(1)
    return results

def run_pending_rows(pending_rows, worker_args, args):
    if not pending_rows:
        return []
    worker_count = resolve_worker_count(args, len(pending_rows))
    results = []
    context = get_context("spawn")
    total_cases = len(pending_rows) * len(injection_cases(args))
    with Manager() as manager:
        progress_queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            future_map = {executor.submit(run_star_task, (row, worker_args, progress_queue)): row for row in pending_rows}
            pending = set(future_map)
            with tqdm(total=len(future_map), desc="Stars", unit="star", bar_format=TQDM_BAR_FORMAT, position=0, dynamic_ncols=True, mininterval=0.25) as star_progress, tqdm(total=total_cases, desc="Injection cases", unit="case", bar_format=TQDM_BAR_FORMAT, position=1, dynamic_ncols=True, mininterval=0.25) as case_progress:
                while pending:
                    done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                    drain_progress_queue(progress_queue, case_progress)
                    for future in done:
                        drain_progress_queue(progress_queue, case_progress)
                        row = future_map[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = {"target_id": normalize_target_id(row["target_id"]), "quarter": int(row["quarter"]), "selection_group": str(row.get("selection_group", "unspecified")), "sample_stratum": str(row.get("sample_stratum", "unspecified")), "status": "failed", "star_dir": "", "runtime_seconds": float("nan"), "error": f"{type(exc).__name__}: {exc}"}
                        results.append(result)
                        if result["status"] != "success":
                            tqdm.write(f"FAILED KIC {result['target_id']} Q{result['quarter']}: {result['error']}")
                        star_progress.set_postfix_str(f"{result['status']} KIC {result['target_id']} Q{result['quarter']}")
                        star_progress.update(1)
                drain_progress_queue(progress_queue, case_progress)
    return results

def load_completed_outputs(task_results):
    injections = []
    summaries = []
    base_candidates = []
    for result in task_results:
        if result["status"] != "success":
            continue
        star_dir = Path(result["star_dir"])
        injections.append(pd.read_csv(star_dir / "injections.csv", dtype={"target_id": str}))
        summaries.append(json.loads((star_dir / "star_summary.json").read_text()))
        base_path = star_dir / "base_light_curve_candidates.csv"
        if base_path.exists():
            base_candidates.append(pd.read_csv(base_path, dtype={"target_id": str}))
    return (
        pd.concat(injections, ignore_index=True) if injections else pd.DataFrame(),
        pd.DataFrame(summaries),
        pd.concat(base_candidates, ignore_index=True) if base_candidates else pd.DataFrame(),
    )

def metric_row(name, values):
    values = pd.Series(values).fillna(False).astype(bool)
    total = int(len(values))
    successes = int(values.sum())
    return {"metric": str(name), "successes": successes, "total": total, "rate": float(successes / total) if total else float("nan")}

def pipeline_summary(injections, pipelines):
    rows = []
    for pipeline in pipelines:
        success = injections[f"{pipeline}_success"].fillna(False).astype(bool)
        harmonic = injections[f"{pipeline}_harmonic_rank1_matched"].fillna(False).astype(bool)
        exact = injections[f"{pipeline}_exact_rank1_matched"].fillna(False).astype(bool)
        harmonic_topk = injections[f"{pipeline}_harmonic_topk_matched"].fillna(False).astype(bool)
        rows.append({"pipeline": pipeline, "branch": PIPELINE_DEFINITIONS[pipeline][0], "detector": PIPELINE_DEFINITIONS[pipeline][1], "injection_count": int(len(injections)), "success_count": int(success.sum()), "success_rate": float(success.mean()), "harmonic_rank1_count": int(harmonic.sum()), "harmonic_rank1_rate": float(harmonic.mean()), "harmonic_topk_count": int(harmonic_topk.sum()), "harmonic_topk_rate": float(harmonic_topk.mean()), "ranking_gap_count": int((harmonic_topk & ~harmonic).sum()), "ranking_gap_rate": float((harmonic_topk & ~harmonic).mean()), "exact_rank1_count": int(exact.sum()), "exact_rank1_rate": float(exact.mean()), "base_rank1_competition_count": int((injections[f"{pipeline}_failure_mode"] == "base_rank1_competition").sum()), "candidate_generation_failure_count": int((injections[f"{pipeline}_failure_mode"] == "candidate_generation_failure").sum()), "median_score": float(pd.to_numeric(injections[f"{pipeline}_score"], errors="coerce").median()), "median_runtime_seconds": float(pd.to_numeric(injections[f"{pipeline}_runtime_seconds"], errors="coerce").median())})
    return pd.DataFrame(rows)

def bool_union(injections, pipelines, suffix):
    if not pipelines:
        return pd.Series(False, index=injections.index)
    result = pd.Series(False, index=injections.index)
    for pipeline in pipelines:
        result = result | injections[f"{pipeline}_{suffix}"].fillna(False).astype(bool)
    return result

def combination_summary(injections, pipelines):
    named = [("raw_bls_union_gp_tcf", [pipeline for pipeline in ("raw_bls", "gp_tcf") if pipeline in pipelines]), ("existing_bls_tcf", [pipeline for pipeline in ("raw_bls", "arima_tcf") if pipeline in pipelines]), ("non_gp_union", [pipeline for pipeline in ("raw_bls", "arima_tcf", "kalman_bls", "kalman_tcf") if pipeline in pipelines]), ("gp_union", [pipeline for pipeline in ("gp_bls", "gp_tcf") if pipeline in pipelines]), ("all_pipelines", list(pipelines))]
    rows = []
    for name, members in named:
        if not members:
            continue
        harmonic = bool_union(injections, members, "harmonic_rank1_matched")
        harmonic_topk = bool_union(injections, members, "harmonic_topk_matched")
        exact = bool_union(injections, members, "exact_rank1_matched")
        rows.append({"combination": name, "pipelines": ",".join(members), "injection_count": int(len(injections)), "harmonic_rank1_count": int(harmonic.sum()), "harmonic_rank1_rate": float(harmonic.mean()), "harmonic_topk_count": int(harmonic_topk.sum()), "harmonic_topk_rate": float(harmonic_topk.mean()), "ranking_gap_count": int((harmonic_topk & ~harmonic).sum()), "ranking_gap_rate": float((harmonic_topk & ~harmonic).mean()), "exact_rank1_count": int(exact.sum()), "exact_rank1_rate": float(exact.mean())})
    return pd.DataFrame(rows)

def pairwise_overlap(injections, pipelines, suffix):
    rows = []
    for first, second in iter_combinations(pipelines, 2):
        first_values = injections[f"{first}_{suffix}"].fillna(False).astype(bool)
        second_values = injections[f"{second}_{suffix}"].fillna(False).astype(bool)
        rows.append({"metric": suffix, "first_pipeline": first, "second_pipeline": second, "injection_count": int(len(injections)), "both_recovered": int((first_values & second_values).sum()), "first_only": int((first_values & ~second_values).sum()), "second_only": int((second_values & ~first_values).sum()), "neither": int((~first_values & ~second_values).sum()), "intersection_rate": float((first_values & second_values).mean()), "union_rate": float((first_values | second_values).mean()), "incremental_recoveries_from_second": int((second_values & ~first_values).sum())})
    return pd.DataFrame(rows)

def grouped_summary(injections, column, pipelines):
    aggregations = {"injection_count": ("target_id", "size"), "star_count": ("target_id", "nunique")}
    for pipeline in pipelines:
        aggregations[f"{pipeline}_harmonic_rank1_rate"] = (f"{pipeline}_harmonic_rank1_matched", "mean")
        aggregations[f"{pipeline}_harmonic_topk_rate"] = (f"{pipeline}_harmonic_topk_matched", "mean")
        aggregations[f"{pipeline}_exact_rank1_rate"] = (f"{pipeline}_exact_rank1_matched", "mean")
    return injections.groupby(column, dropna=False, as_index=False).agg(**aggregations)

def failure_mode_summary(injections, pipelines):
    rows = []
    for pipeline in pipelines:
        counts = injections[f"{pipeline}_failure_mode"].fillna("unknown").value_counts(dropna=False)
        for mode, count in counts.items():
            rows.append({"pipeline": pipeline, "branch": PIPELINE_DEFINITIONS[pipeline][0], "detector": PIPELINE_DEFINITIONS[pipeline][1], "failure_mode": str(mode), "count": int(count), "rate": float(count / len(injections)) if len(injections) else float("nan")})
    return pd.DataFrame(rows)

def save_global_outputs(args, manifest, task_results, injections, star_summaries, base_candidates):
    metrics_dir = Path(args.output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pipelines = tuple(args.pipelines)
    pipe_summary = pipeline_summary(injections, pipelines) if not injections.empty else pd.DataFrame()
    combo_summary = combination_summary(injections, pipelines) if not injections.empty else pd.DataFrame()
    pairwise = pd.concat([pairwise_overlap(injections, pipelines, "harmonic_rank1_matched"), pairwise_overlap(injections, pipelines, "harmonic_topk_matched"), pairwise_overlap(injections, pipelines, "exact_rank1_matched")], ignore_index=True) if not injections.empty else pd.DataFrame()
    failures = failure_mode_summary(injections, pipelines) if not injections.empty else pd.DataFrame()
    manifest.to_csv(metrics_dir / "target_manifest_used.csv", index=False)
    pd.DataFrame(task_results).to_csv(metrics_dir / "target_execution_status.csv", index=False)
    injections.to_csv(metrics_dir / "multistar_challenger_injections.csv", index=False)
    star_summaries.to_csv(metrics_dir / "multistar_challenger_star_summary.csv", index=False)
    base_candidates.to_csv(metrics_dir / "multistar_challenger_base_candidates.csv", index=False)
    pipe_summary.to_csv(metrics_dir / "multistar_challenger_pipeline_summary.csv", index=False)
    combo_summary.to_csv(metrics_dir / "multistar_challenger_combinations.csv", index=False)
    pairwise.to_csv(metrics_dir / "multistar_challenger_pairwise_overlap.csv", index=False)
    failures.to_csv(metrics_dir / "multistar_challenger_failure_modes.csv", index=False)
    if not injections.empty:
        grouped_summary(injections, "injected_depth", pipelines).to_csv(metrics_dir / "multistar_challenger_by_depth.csv", index=False)
        grouped_summary(injections, "injected_duration_hours", pipelines).to_csv(metrics_dir / "multistar_challenger_by_duration.csv", index=False)
        grouped_summary(injections, "injected_period_days", pipelines).to_csv(metrics_dir / "multistar_challenger_by_period.csv", index=False)
        grouped_summary(injections, "target_id", pipelines).to_csv(metrics_dir / "multistar_challenger_by_star.csv", index=False)
        if "sample_stratum" in injections.columns:
            grouped_summary(injections, "sample_stratum", pipelines).to_csv(metrics_dir / "multistar_challenger_by_stratum.csv", index=False)
    all_harmonic = bool_union(injections, pipelines, "harmonic_rank1_matched") if not injections.empty else pd.Series(dtype=bool)
    all_harmonic_topk = bool_union(injections, pipelines, "harmonic_topk_matched") if not injections.empty else pd.Series(dtype=bool)
    cohort_counts = manifest["selection_group"].value_counts().to_dict() if "selection_group" in manifest.columns else {}
    summary = {
        "benchmark_schema_version": int(BENCHMARK_SCHEMA_VERSION),
        "profile": str(args.profile),
        "scientific_cohort": str(args.selection_group) if getattr(args, "selection_group", None) else "mixed_or_unspecified",
        "require_catalog_clean": bool(args.require_catalog_clean),
        "cohort_counts": {str(key): int(value) for key, value in cohort_counts.items()},
        "search_resolution": str(args.search_resolution),
        "kalman_injection_mode": str(args.kalman_injection_mode),
        "gp_injection_mode": str(args.gp_injection_mode),
        "target_count": int(len(manifest)),
        "successful_target_count": int((pd.DataFrame(task_results)["status"] == "success").sum()) if task_results else 0,
        "failed_target_count": int((pd.DataFrame(task_results)["status"] != "success").sum()) if task_results else 0,
        "pipeline_count": int(len(pipelines)),
        "pipelines": list(pipelines),
        "injection_count": int(len(injections)),
        "injections_per_target": int(len(injection_cases(args))),
        "parallel_star_workers": int(resolve_worker_count(args, len(manifest))),
        "calibration_status": "injection benchmark only; FAP calibration must be run separately for final false-alarm-controlled claims",
        "all_pipeline_harmonic_rank1_rate": float(all_harmonic.mean()) if len(all_harmonic) else float("nan"),
        "all_pipeline_harmonic_topk_rate": float(all_harmonic_topk.mean()) if len(all_harmonic_topk) else float("nan"),
        "all_pipeline_ranking_gap_rate": float((all_harmonic_topk & ~all_harmonic).mean()) if len(all_harmonic) else float("nan"),
    }
    (metrics_dir / "multistar_challenger_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    return metrics_dir, summary

def main(args=None):
    args = args or parse_args()
    manifest = load_manifest(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    write_benchmark_config(args)
    task_results = []
    pending_rows = []
    for row in manifest.to_dict(orient="records"):
        star_dir = Path(args.output_dir) / "stars" / star_prefix(row["target_id"], row["quarter"])
        if args.resume and completed_star_is_compatible(star_dir, args):
            task_results.append(completed_result(args.output_dir, row))
        elif args.resume and not args.rerun_failures and (star_dir / "failure.json").exists() and star_config_matches(star_dir, args):
            task_results.append(failure_result(args.output_dir, row))
        else:
            pending_rows.append(row)
    print(f"Profile: {args.profile}")
    print(f"Manifest: {args.manifest_path}")
    print(f"Selection group: {args.selection_group or 'mixed_or_unspecified'}")
    print(f"Catalog-clean guard: {'ON' if args.require_catalog_clean else 'OFF (explicit contaminated/stress-test mode)'}")
    print(f"Targets requested: {len(manifest)}")
    print(f"Targets resumed: {len(task_results)}")
    print(f"Targets to run: {len(pending_rows)}")
    print(f"Parallel star workers: {resolve_worker_count(args, len(manifest))}")
    print(f"Injections per star: {len(injection_cases(args))}")
    print(f"Pipelines: {', '.join(args.pipelines)}")
    print(f"Search resolution: {args.search_resolution} (BLS periods={args.n_periods}, TCF coarse periods={args.n_coarse_periods}, refinements={args.n_refinement_regions}, BLS oversample={args.bls_oversample})")
    print(f"Kalman injection mode: {args.kalman_injection_mode}")
    print(f"GP injection mode: {args.gp_injection_mode}")
    prefetch_manifest_light_curves(pending_rows, args)
    worker_args = settings_to_worker_dict(args)
    task_results.extend(run_pending_rows(pending_rows, worker_args, args))
    injections, star_summaries, base_candidates = load_completed_outputs(task_results)
    metrics_dir, summary = save_global_outputs(args, manifest, task_results, injections, star_summaries, base_candidates)
    print(f"Metrics directory: {metrics_dir}")
    print(f"All-pipeline harmonic rank-1 union: {summary['all_pipeline_harmonic_rank1_rate']:.3f}")
    print(f"All-pipeline harmonic top-k union: {summary['all_pipeline_harmonic_topk_rate']:.3f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
