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
from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.detection.tcf import default_duration_grid, default_period_grid, fit_arima_innovations, harmonic_peak_rank, matching_peak_rank, period_match_fraction, run_tcf
from adaptive_transit.injections.synthetic import inject_periodic_box_transit
from adaptive_transit.noise_models.gp import fit_smooth_gp_background
from adaptive_transit.noise_models.kalman import fit_kalman_local_level
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Warning: the tpfmodel submodule is not available.*", category=UserWarning)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "configs/kepler_50_star_manifest.csv"
CACHE_DIR = PROJECT_ROOT / "outputs/cache/kepler_light_curves"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/experiments/multistar_challenger_benchmark"
BACKGROUND_FEATURE_PATH = PROJECT_ROOT / "outputs/experiments/multistar_background_timescale/metrics/multistar_background_timescale_features.csv"
EXISTING_ARIMA_CACHE_ROOT = PROJECT_ROOT / "outputs/experiments/multistar_bls_tcf/optimized/stars"
TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} ({percentage:3.0f}%) [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
DEFAULT_PIPELINES = ("raw_bls", "raw_tcf", "arima_bls", "arima_tcf", "kalman_bls", "kalman_tcf", "gp_bls", "gp_tcf")
PIPELINE_DEFINITIONS = {"raw_bls": ("raw", "bls"), "raw_tcf": ("raw", "tcf"), "arima_bls": ("arima", "bls"), "arima_tcf": ("arima", "tcf"), "kalman_bls": ("kalman", "bls"), "kalman_tcf": ("kalman", "tcf"), "gp_bls": ("gp", "bls"), "gp_tcf": ("gp", "tcf")}

def default_settings(profile="pilot"):
    settings = {"profile": profile, "manifest_path": MANIFEST_PATH, "cache_dir": CACHE_DIR, "output_dir": OUTPUT_ROOT / profile, "background_feature_path": BACKGROUND_FEATURE_PATH, "existing_arima_cache_root": EXISTING_ARIMA_CACHE_ROOT, "target_limit": 10, "strict_target_count": True, "target_ids": None, "stratified_pilot": True, "quality_policy": "default", "require_finite_flux_error": False, "test_fraction": 0.20, "pipelines": DEFAULT_PIPELINES, "arima_order": (1, 1, 0), "fit_maxiter": 200, "arima_injection_mode": "filter", "kalman_maxiter": 100, "kalman_burn_in": 1, "gp_max_train_points": 512, "gp_length_scale_days": 3.0, "gp_min_length_scale_days": 1.0, "gp_max_length_scale_days": 30.0, "gp_measurement_noise_fraction": 0.20, "gp_n_restarts_optimizer": 0, "gp_random_seed": 123, "gp_optimize_kernel": True, "injection_period_grid": (2.0, 5.0), "injection_duration_hours_grid": (2.0, 4.0), "injection_depth_grid": (0.0005, 0.001), "epoch_phase_fraction_grid": (0.45,), "min_period_days": 1.0, "max_period_days": 15.0, "n_periods": 3000, "min_duration_hours": 1.5, "max_duration_hours": 10.0, "n_durations": 8, "edge_width_cadences": 0, "min_edge_observations": 4, "min_transit_events": 3, "min_event_consistency_fraction": 0.60, "top_k": 5, "search_mode": "coarse_to_fine", "n_coarse_periods": 1000, "n_refinement_regions": 12, "refinement_half_width_points": 30, "period_match_tolerance_fraction": 0.02, "bls_objective": "snr", "bls_oversample": 5, "max_workers": None, "reserve_cpu_cores": 2, "random_seed": 123, "allow_download": True, "download_max_attempts": 5, "download_initial_wait_seconds": 5.0, "download_backoff_factor": 2.0, "progress_interval": 1, "checkpoint_interval": 5, "prefetch_workers": 4, "resume": True, "rerun_failures": False, "save_regularized_inputs": False}
    if profile == "main":
        settings.update({"output_dir": OUTPUT_ROOT / profile, "target_limit": 50, "stratified_pilot": False, "injection_period_grid": (2.0, 5.0, 10.0), "injection_duration_hours_grid": (2.0, 4.0, 8.0), "injection_depth_grid": (0.0002, 0.0005, 0.001), "epoch_phase_fraction_grid": (0.15, 0.45, 0.75), "n_periods": 10000, "top_k": 10, "n_coarse_periods": 4000, "n_refinement_regions": 30, "refinement_half_width_points": 40, "bls_oversample": 10}
        )
    elif profile == "smoke":
        settings.update({"output_dir": OUTPUT_ROOT / profile, "target_limit": 2, "strict_target_count": False, "stratified_pilot": False, "injection_period_grid": (5.0,), "injection_duration_hours_grid": (4.0,), "injection_depth_grid": (0.001,), "epoch_phase_fraction_grid": (0.45,), "n_periods": 400, "n_durations": 4, "top_k": 3, "n_coarse_periods": 150, "n_refinement_regions": 3, "refinement_half_width_points": 8, "bls_oversample": 3, "max_workers": 2, "gp_max_train_points": 192, "gp_optimize_kernel": False}
        )
    elif profile != "pilot":
        raise ValueError("profile must be pilot, main, or smoke.")
    return SimpleNamespace(**settings)

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
    parser = argparse.ArgumentParser(description="Run a heavily parallel multi-star benchmark across raw, ARIMA, Kalman, and GP residual branches.")
    parser.add_argument("--profile", choices=("pilot", "main", "smoke"), default="pilot")
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--background-feature-path", type=Path)
    parser.add_argument("--target-limit", type=int)
    parser.add_argument("--allow-partial-target-count", dest="strict_target_count", action="store_false")
    parser.add_argument("--target-ids", type=parse_target_ids)
    parser.add_argument("--no-stratified-pilot", dest="stratified_pilot", action="store_false")
    parser.add_argument("--pipelines", type=parse_pipelines)
    parser.add_argument("--arima-order", type=parse_int_order)
    parser.add_argument("--fit-maxiter", type=int)
    parser.add_argument("--arima-injection-mode", choices=("filter", "refit"))
    parser.add_argument("--kalman-maxiter", type=int)
    parser.add_argument("--gp-max-train-points", type=int)
    parser.add_argument("--gp-length-scale-days", type=float)
    parser.add_argument("--gp-min-length-scale-days", type=float)
    parser.add_argument("--gp-max-length-scale-days", type=float)
    parser.add_argument("--gp-measurement-noise-fraction", type=float)
    parser.add_argument("--gp-n-restarts-optimizer", type=int)
    parser.add_argument("--gp-fixed-kernel", dest="gp_optimize_kernel", action="store_false")
    parser.add_argument("--injection-period-grid", type=parse_float_grid)
    parser.add_argument("--injection-duration-hours-grid", type=parse_float_grid)
    parser.add_argument("--injection-depth-grid", type=parse_float_grid)
    parser.add_argument("--epoch-phase-fraction-grid", type=parse_float_grid)
    parser.add_argument("--n-periods", type=int)
    parser.add_argument("--n-durations", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--n-coarse-periods", type=int)
    parser.add_argument("--n-refinement-regions", type=int)
    parser.add_argument("--refinement-half-width-points", type=int)
    parser.add_argument("--bls-oversample", type=int)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--reserve-cpu-cores", type=int)
    parser.add_argument("--no-download", dest="allow_download", action="store_false")
    parser.add_argument("--download-max-attempts", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--progress-interval", type=int)
    parser.add_argument("--prefetch-workers", type=int)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--rerun-failures", action="store_true")
    parser.add_argument("--save-regularized-inputs", action="store_true")
    parsed = parser.parse_args(argv)
    args = default_settings(parsed.profile)
    for key, value in vars(parsed).items():
        if key == "profile":
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
    keys = ("profile", "pipelines", "quality_policy", "require_finite_flux_error", "test_fraction", "arima_order", "fit_maxiter", "arima_injection_mode", "kalman_maxiter", "kalman_burn_in", "gp_max_train_points", "gp_length_scale_days", "gp_min_length_scale_days", "gp_max_length_scale_days", "gp_measurement_noise_fraction", "gp_n_restarts_optimizer", "gp_optimize_kernel", "injection_period_grid", "injection_duration_hours_grid", "injection_depth_grid", "epoch_phase_fraction_grid", "min_period_days", "max_period_days", "n_periods", "min_duration_hours", "max_duration_hours", "n_durations", "top_k", "search_mode", "n_coarse_periods", "n_refinement_regions", "refinement_half_width_points", "bls_objective", "bls_oversample", "period_match_tolerance_fraction")
    return {key: json_ready(getattr(args, key)) for key in keys}

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
    markers = ("ReadTimeout", "ConnectTimeout", "ConnectionError", "HTTPSConnectionPool", "Max retries exceeded", "Temporary failure", "temporarily unavailable", "Connection reset", "RemoteDisconnected", "mast.stsci.edu")
    return any(marker in message for marker in markers)

def load_light_curve_frame(target_id, quarter, args, progress_queue=None):
    path = light_curve_cache_path(args, target_id, quarter)
    if path.exists():
        return pd.read_parquet(path), True
    if not args.get("allow_download", True):
        raise FileNotFoundError(f"Cached light curve is missing: {path}")
    max_attempts = max(1, int(args.get("download_max_attempts", 1)))
    for attempt in range(1, max_attempts + 1):
        try:
            report_progress(progress_queue, target_id, quarter, "download attempt", detail=f"{attempt}/{max_attempts}")
            light_curve = load_kepler_pdcsap(target_id, quarter)
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
    manifest = manifest.drop_duplicates(["target_id", "quarter"], keep="first").reset_index(drop=True)
    if args.target_ids:
        wanted = {normalize_target_id(target_id) for target_id in args.target_ids}
        manifest = manifest[manifest["target_id"].isin(wanted)].reset_index(drop=True)
    if args.stratified_pilot and str(args.profile) == "pilot" and Path(args.background_feature_path).exists():
        manifest = stratified_pilot_manifest(manifest, args)
    else:
        manifest = manifest.head(int(args.target_limit)).copy()
    if args.strict_target_count and len(manifest) != int(args.target_limit):
        raise ValueError(f"Expected exactly {args.target_limit} target-quarter rows but found {len(manifest)}.")
    if manifest.empty:
        raise ValueError("Manifest contains no usable rows.")
    return manifest.reset_index(drop=True)

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
    criteria = [("robust_flux_scatter_ppm", True, 2, "quiet_low_scatter"), ("background_tau_integrated_positive_acf_days", False, 2, "long_integrated_acf"), ("background_tau_acf_e_days", False, 2, "long_acf_e"), ("robust_flux_scatter_ppm", False, 2, "high_scatter"), ("gap_fraction", False, 2, "gap_heavy")]
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
    return {"innovations": innovations, "fit": filtered, "params": np.asarray(params, dtype=float), "summary": {"order": tuple(args["arima_order"]), "converged": True, "aic": float(filtered.aic), "bic": float(filtered.bic), "finite_innovation_count": int(np.isfinite(innovations).sum()), "mode": "filter_fixed_base_params"}}

def fit_gp(time, values, args):
    return fit_smooth_gp_background(time, values, max_train_points=args["gp_max_train_points"], length_scale_days=args["gp_length_scale_days"], min_length_scale_days=args["gp_min_length_scale_days"], max_length_scale_days=args["gp_max_length_scale_days"], measurement_noise_fraction=args["gp_measurement_noise_fraction"], n_restarts_optimizer=args["gp_n_restarts_optimizer"], random_seed=args["gp_random_seed"], optimize_kernel=args["gp_optimize_kernel"])

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

def detector_result_fields(pipeline, result, detector, injected_period, args, runtime):
    summary = result["summary"]
    peaks = result["top_peaks"]
    period = float(summary["period_days"] if detector == "bls" else summary["period"])
    harmonic_error, exact_error, harmonic_matched, exact_matched = match_fields(period, injected_period, args["period_match_tolerance_fraction"])
    fields = {f"{pipeline}_success": True, f"{pipeline}_runtime_seconds": float(runtime), f"{pipeline}_recovered_period_days": period, f"{pipeline}_period_error_fraction": harmonic_error, f"{pipeline}_exact_period_error_fraction": exact_error, f"{pipeline}_harmonic_rank1_matched": harmonic_matched, f"{pipeline}_exact_rank1_matched": exact_matched, f"{pipeline}_exact_rank_topk": top_peak_rank(peaks, injected_period, args["period_match_tolerance_fraction"]), f"{pipeline}_half_period_rank_topk": harmonic_rank(peaks, injected_period, 0.5, args["period_match_tolerance_fraction"]), f"{pipeline}_double_period_rank_topk": harmonic_rank(peaks, injected_period, 2.0, args["period_match_tolerance_fraction"]), f"{pipeline}_top_periods_json": json.dumps([float(row.get("period_days", row.get("period", np.nan))) for row in peaks.to_dict(orient="records")])}
    if detector == "bls":
        fields.update({f"{pipeline}_score": float(summary["sde"]), f"{pipeline}_power": float(summary["power"]), f"{pipeline}_duration_hours": float(summary["duration_days"] * 24.0), f"{pipeline}_transit_time": float(summary["transit_time"]), f"{pipeline}_depth": float(summary["depth"]), f"{pipeline}_top_scores_json": json.dumps([float(value) for value in peaks["sde"].to_numpy(dtype=float)])})
    else:
        fields.update({f"{pipeline}_score": float(summary["score"]), f"{pipeline}_raw_pooled_score": float(summary["raw_pooled_score"]), f"{pipeline}_duration_hours": float(summary["duration"] * 24.0), f"{pipeline}_epoch_days": float(summary["epoch"]), f"{pipeline}_valid_transit_events": int(summary["n_valid_transit_events"]), f"{pipeline}_positive_event_fraction": float(summary["positive_event_fraction"]), f"{pipeline}_top_scores_json": json.dumps([float(value) for value in peaks["score"].to_numpy(dtype=float)])})
    return fields

def empty_pipeline_fields(pipeline, error):
    return {f"{pipeline}_success": False, f"{pipeline}_error": str(error), f"{pipeline}_runtime_seconds": float("nan"), f"{pipeline}_recovered_period_days": float("nan"), f"{pipeline}_score": float("nan"), f"{pipeline}_period_error_fraction": float("nan"), f"{pipeline}_exact_period_error_fraction": float("nan"), f"{pipeline}_harmonic_rank1_matched": False, f"{pipeline}_exact_rank1_matched": False, f"{pipeline}_exact_rank_topk": None, f"{pipeline}_half_period_rank_topk": None, f"{pipeline}_double_period_rank_topk": None}

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

def run_one_case(case_index, case, time, flux, bls_periods, tcf_periods, duration_grid, base_arima, target_id, quarter, selection_group, args):
    injected_period, injected_duration_hours, injected_depth, epoch_phase_fraction = case
    finite = np.isfinite(time) & np.isfinite(flux)
    epoch = float(np.min(time[finite]) + float(epoch_phase_fraction) * float(injected_period))
    duration_days = float(injected_duration_hours) / 24.0
    injected_flux, template, in_transit = inject_periodic_box_transit(time, flux, injected_period, epoch, duration_days, injected_depth)
    before = periodic_depth_and_snr(injected_flux, in_transit)
    row = {"target_id": normalize_target_id(target_id), "quarter": int(quarter), "selection_group": str(selection_group), "case_index": int(case_index), "injected_period_days": float(injected_period), "injected_epoch_days": epoch, "epoch_phase_fraction": float(epoch_phase_fraction), "injected_duration_hours": float(injected_duration_hours), "injected_depth": float(injected_depth), "in_transit_observation_count": int(np.isfinite(flux[in_transit]).sum()), "observed_depth_before_model": float(before["depth"]), "local_snr_before_model": float(before["snr"])}
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
            row["arima_converged"] = bool(arima["summary"].get("converged", True))
            add_branch_diagnostics(row, "arima", arima["innovations"], before, in_transit)
        except Exception as exc:
            branch_errors["arima"] = f"{type(exc).__name__}: {exc}"
            row["arima_error"] = branch_errors["arima"]
    if "kalman" in required_branches(args["pipelines"]):
        try:
            started = perf_counter()
            model = fit_kalman_local_level(injected_flux, maxiter=args["kalman_maxiter"], burn_in=args["kalman_burn_in"])
            branch_series["kalman"] = model.residuals
            row["kalman_model_runtime_seconds"] = float(perf_counter() - started)
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
            model = fit_gp(time, injected_flux, args)
            branch_series["gp"] = model.residuals
            row["gp_model_runtime_seconds"] = float(perf_counter() - started)
            row["gp_converged"] = bool(model.converged)
            row["gp_log_marginal_likelihood"] = float(model.log_marginal_likelihood)
            row["gp_training_point_count"] = int(model.parameters["training_point_count"])
            row["gp_length_scale_days"] = float(model.parameters["length_scale_days"])
            row["gp_measurement_noise_variance"] = float(model.parameters["measurement_noise_variance"])
            add_branch_diagnostics(row, "gp", model.residuals, before, in_transit)
        except Exception as exc:
            branch_errors["gp"] = f"{type(exc).__name__}: {exc}"
            row["gp_error"] = branch_errors["gp"]
    for pipeline in args["pipelines"]:
        branch, detector = PIPELINE_DEFINITIONS[pipeline]
        if branch not in branch_series:
            row.update(empty_pipeline_fields(pipeline, branch_errors.get(branch, "branch unavailable")))
            continue
        try:
            started = perf_counter()
            result = run_bls_search(time, branch_series[branch], bls_periods, duration_grid, args) if detector == "bls" else run_tcf_search(time, branch_series[branch], tcf_periods, duration_grid, args)
            row.update(detector_result_fields(pipeline, result, detector, injected_period, args, perf_counter() - started))
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

def run_star_injections(star_dir, cases, time, flux, bls_periods, tcf_periods, duration_grid, base_arima, target_id, quarter, selection_group, args, progress_queue, resume_compatible=None):
    rows, completed = load_existing_injection_rows(star_dir, cases, args, resume_compatible=resume_compatible)
    if completed:
        report_progress(progress_queue, target_id, quarter, "injections resumed", units=len(completed), detail=f"{len(completed)}/{len(cases)}")
    unreported = 0
    progress_interval = max(1, int(args.get("progress_interval", 1)))
    checkpoint_interval = max(1, int(args.get("checkpoint_interval", 5)))
    for case_index, case in enumerate(cases):
        if case_index in completed:
            continue
        rows.append(run_one_case(case_index, case, time, flux, bls_periods, tcf_periods, duration_grid, base_arima, target_id, quarter, selection_group, args))
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
        for name in ("COMPLETE", "injections.csv", "star_summary.json", "failure.json"):
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
    star_dir = Path(args["output_dir"]) / "stars" / star_prefix(target_id, quarter)
    started = perf_counter()
    try:
        resume_compatible = prepare_star_run(star_dir, args)
        write_star_checkpoint(star_dir, target_id, quarter, "running", "started")
        light_curve_frame, cache_hit = load_light_curve_frame(target_id, quarter, args, progress_queue=progress_queue)
        regular, preprocessing = preprocess_pdcsap_light_curve(light_curve_frame, quality_policy=args["quality_policy"], require_finite_flux_error=args["require_finite_flux_error"], normalization_fit_fraction=1.0 - args["test_fraction"])
        time = regular["time"].to_numpy(dtype=float)
        flux = regular["normalized_flux"].to_numpy(dtype=float)
        if np.isfinite(time).sum() < 24 or np.isfinite(flux).sum() < 24:
            raise ValueError("Insufficient finite observations.")
        period_grid = default_period_grid(time, min_period_days=args["min_period_days"], max_period_days=args["max_period_days"], n_periods=args["n_periods"])
        duration_grid = default_duration_grid(args["min_duration_hours"], args["max_duration_hours"], args["n_durations"])
        base_arima, base_arima_runtime, base_arima_source = load_or_fit_base_arima(star_dir, target_id, quarter, flux, args)
        cases = [tuple(case) for case in args["cases"]]
        injections = run_star_injections(star_dir, cases, time, flux, period_grid, period_grid, duration_grid, base_arima, target_id, quarter, selection_group, args, progress_queue, resume_compatible=resume_compatible)
        successful = injections.copy()
        star_metrics = calculate_star_metrics(time, flux)
        summary = {"target_id": target_id, "quarter": quarter, "selection_group": selection_group, "status": "success", "profile": str(args["profile"]), "pipelines": list(args["pipelines"]), "runtime_seconds": float(perf_counter() - started), "light_curve_cache_hit": bool(cache_hit), "base_arima_source": str(base_arima_source), "base_arima_runtime_seconds": float(base_arima_runtime), "base_arima_converged": bool(base_arima["summary"].get("converged", True)), "injection_count_requested": int(len(cases)), "injection_count_completed": int(len(successful)), **star_metrics}
        for pipeline in args["pipelines"]:
            column = f"{pipeline}_harmonic_rank1_matched"
            exact_column = f"{pipeline}_exact_rank1_matched"
            summary[f"{pipeline}_success_count"] = int(successful[f"{pipeline}_success"].astype(bool).sum()) if f"{pipeline}_success" in successful.columns else 0
            summary[f"{pipeline}_harmonic_rank1_rate"] = float(successful[column].astype(bool).mean()) if column in successful.columns and len(successful) else float("nan")
            summary[f"{pipeline}_exact_rank1_rate"] = float(successful[exact_column].astype(bool).mean()) if exact_column in successful.columns and len(successful) else float("nan")
        (star_dir / "star_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
        if args["save_regularized_inputs"]:
            regular.to_parquet(star_dir / "regularized_light_curve.parquet", index=False)
        (star_dir / "COMPLETE").write_text("complete\n")
        write_star_checkpoint(star_dir, target_id, quarter, "success", "complete", runtime_seconds=summary["runtime_seconds"])
        return {"target_id": target_id, "quarter": quarter, "selection_group": selection_group, "status": "success", "star_dir": str(star_dir), "runtime_seconds": summary["runtime_seconds"], "error": ""}
    except Exception as exc:
        failure = {"target_id": target_id, "quarter": quarter, "selection_group": selection_group, "status": "failed", "star_dir": str(star_dir), "runtime_seconds": float(perf_counter() - started), "error": f"{type(exc).__name__}: {exc}"}
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
    return {"target_id": target_id, "quarter": quarter, "selection_group": str(row.get("selection_group", "unspecified")), "status": "success", "star_dir": str(star_dir), "runtime_seconds": float(summary.get("runtime_seconds", np.nan)), "error": ""}

def failure_result(output_dir, row):
    target_id = normalize_target_id(row["target_id"])
    quarter = int(row["quarter"])
    star_dir = Path(output_dir) / "stars" / star_prefix(target_id, quarter)
    try:
        failure = json.loads((star_dir / "failure.json").read_text())
    except Exception:
        failure = {}
    return {"target_id": target_id, "quarter": quarter, "selection_group": str(row.get("selection_group", "unspecified")), "status": "failed", "star_dir": str(star_dir), "runtime_seconds": float(failure.get("runtime_seconds", np.nan)), "error": str(failure.get("error", "Existing failure.json"))}

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
    prefetch_args = {"cache_dir": str(args.cache_dir), "allow_download": True, "download_max_attempts": int(args.download_max_attempts), "download_initial_wait_seconds": float(args.download_initial_wait_seconds), "download_backoff_factor": float(args.download_backoff_factor)}
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
                            result = {"target_id": normalize_target_id(row["target_id"]), "quarter": int(row["quarter"]), "selection_group": str(row.get("selection_group", "unspecified")), "status": "failed", "star_dir": "", "runtime_seconds": float("nan"), "error": f"{type(exc).__name__}: {exc}"}
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
    for result in task_results:
        if result["status"] != "success":
            continue
        star_dir = Path(result["star_dir"])
        injections.append(pd.read_csv(star_dir / "injections.csv", dtype={"target_id": str}))
        summaries.append(json.loads((star_dir / "star_summary.json").read_text()))
    return pd.concat(injections, ignore_index=True) if injections else pd.DataFrame(), pd.DataFrame(summaries)

def metric_row(name, values):
    values = pd.Series(values).fillna(False).astype(bool)
    total = int(len(values))
    successes = int(values.sum())
    return {"metric": str(name), "successes": successes, "total": total, "rate": float(successes / total) if total else float("nan")}

def pipeline_summary(injections, pipelines):
    rows = []
    for pipeline in pipelines:
        success = injections[f"{pipeline}_success"].astype(bool)
        harmonic = injections[f"{pipeline}_harmonic_rank1_matched"].astype(bool)
        exact = injections[f"{pipeline}_exact_rank1_matched"].astype(bool)
        rows.append({"pipeline": pipeline, "branch": PIPELINE_DEFINITIONS[pipeline][0], "detector": PIPELINE_DEFINITIONS[pipeline][1], "injection_count": int(len(injections)), "success_count": int(success.sum()), "success_rate": float(success.mean()), "harmonic_rank1_count": int(harmonic.sum()), "harmonic_rank1_rate": float(harmonic.mean()), "exact_rank1_count": int(exact.sum()), "exact_rank1_rate": float(exact.mean()), "median_score": float(pd.to_numeric(injections[f"{pipeline}_score"], errors="coerce").median()), "median_runtime_seconds": float(pd.to_numeric(injections[f"{pipeline}_runtime_seconds"], errors="coerce").median())})
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
        exact = bool_union(injections, members, "exact_rank1_matched")
        rows.append({"combination": name, "pipelines": ",".join(members), "injection_count": int(len(injections)), "harmonic_rank1_count": int(harmonic.sum()), "harmonic_rank1_rate": float(harmonic.mean()), "exact_rank1_count": int(exact.sum()), "exact_rank1_rate": float(exact.mean())})
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
        aggregations[f"{pipeline}_exact_rank1_rate"] = (f"{pipeline}_exact_rank1_matched", "mean")
    return injections.groupby(column, dropna=False, as_index=False).agg(**aggregations)

def save_global_outputs(args, manifest, task_results, injections, star_summaries):
    metrics_dir = Path(args.output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pipelines = tuple(args.pipelines)
    pipe_summary = pipeline_summary(injections, pipelines) if not injections.empty else pd.DataFrame()
    combo_summary = combination_summary(injections, pipelines) if not injections.empty else pd.DataFrame()
    pairwise = pd.concat([pairwise_overlap(injections, pipelines, "harmonic_rank1_matched"), pairwise_overlap(injections, pipelines, "exact_rank1_matched")], ignore_index=True) if not injections.empty else pd.DataFrame()
    manifest.to_csv(metrics_dir / "target_manifest_used.csv", index=False)
    pd.DataFrame(task_results).to_csv(metrics_dir / "target_execution_status.csv", index=False)
    injections.to_csv(metrics_dir / "multistar_challenger_injections.csv", index=False)
    star_summaries.to_csv(metrics_dir / "multistar_challenger_star_summary.csv", index=False)
    pipe_summary.to_csv(metrics_dir / "multistar_challenger_pipeline_summary.csv", index=False)
    combo_summary.to_csv(metrics_dir / "multistar_challenger_combinations.csv", index=False)
    pairwise.to_csv(metrics_dir / "multistar_challenger_pairwise_overlap.csv", index=False)
    if not injections.empty:
        grouped_summary(injections, "injected_depth", pipelines).to_csv(metrics_dir / "multistar_challenger_by_depth.csv", index=False)
        grouped_summary(injections, "injected_duration_hours", pipelines).to_csv(metrics_dir / "multistar_challenger_by_duration.csv", index=False)
        grouped_summary(injections, "injected_period_days", pipelines).to_csv(metrics_dir / "multistar_challenger_by_period.csv", index=False)
        grouped_summary(injections, "target_id", pipelines).to_csv(metrics_dir / "multistar_challenger_by_star.csv", index=False)
    all_harmonic = bool_union(injections, pipelines, "harmonic_rank1_matched") if not injections.empty else pd.Series(dtype=bool)
    summary = {"profile": str(args.profile), "target_count": int(len(manifest)), "successful_target_count": int((pd.DataFrame(task_results)["status"] == "success").sum()) if task_results else 0, "failed_target_count": int((pd.DataFrame(task_results)["status"] != "success").sum()) if task_results else 0, "pipeline_count": int(len(pipelines)), "pipelines": list(pipelines), "injection_count": int(len(injections)), "injections_per_target": int(len(injection_cases(args))), "parallel_star_workers": int(resolve_worker_count(args, len(manifest))), "calibration_status": "rank-1 injection benchmark only; FAP calibration must be run separately for final false-alarm-controlled claims", "all_pipeline_harmonic_rank1_rate": float(all_harmonic.mean()) if len(all_harmonic) else float("nan")}
    (metrics_dir / "multistar_challenger_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    return metrics_dir, summary

def main(args=None):
    args = args or parse_args()
    manifest = load_manifest(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
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
    print(f"Targets requested: {len(manifest)}")
    print(f"Targets resumed: {len(task_results)}")
    print(f"Targets to run: {len(pending_rows)}")
    print(f"Parallel star workers: {resolve_worker_count(args, len(manifest))}")
    print(f"Injections per star: {len(injection_cases(args))}")
    print(f"Pipelines: {', '.join(args.pipelines)}")
    prefetch_manifest_light_curves(pending_rows, args)
    worker_args = settings_to_worker_dict(args)
    task_results.extend(run_pending_rows(pending_rows, worker_args, args))
    injections, star_summaries = load_completed_outputs(task_results)
    metrics_dir, summary = save_global_outputs(args, manifest, task_results, injections, star_summaries)
    print(f"Metrics directory: {metrics_dir}")
    print(f"All-pipeline harmonic rank-1 union: {summary['all_pipeline_harmonic_rank1_rate']:.3f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
