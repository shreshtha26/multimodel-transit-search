"""Run a parallel 50-star Kepler BLS and event-consistent ARIMA-TCF benchmark."""
import argparse
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import json
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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
from adaptive_transit.detection.false_alarm import moving_block_surrogate
from adaptive_transit.detection.tcf import default_duration_grid, default_period_grid, fit_arima_innovations, harmonic_peak_rank, matching_peak_rank, period_match_fraction, run_tcf
from adaptive_transit.injections.synthetic import inject_periodic_box_transit
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Warning: the tpfmodel submodule is not available.*", category=UserWarning)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "configs/kepler_50_star_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/multistar_bls_tcf"
CACHE_DIR = PROJECT_ROOT / "outputs/cache/kepler_light_curves"
TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} ({percentage:3.0f}%) [{elapsed}<{remaining}, {rate_fmt}] {postfix}"

def default_settings(profile="optimized"):
    settings = {
        "profile": profile,
        "manifest_path": MANIFEST_PATH,
        "output_dir": OUTPUT_DIR / profile,
        "cache_dir": CACHE_DIR,
        "target_limit": 50,
        "strict_target_count": True,
        "target_ids": None,
        "batch_index": 1,
        "batch_count": 1,
        "quality_policy": "default",
        "require_finite_flux_error": False,
        "test_fraction": 0.20,
        "arima_order": (1, 1, 0),
        "fit_maxiter": 200,
        "arima_injection_mode": "filter",
        "injection_period_grid": (2.0, 5.0, 10.0),
        "injection_duration_hours_grid": (2.0, 4.0, 8.0),
        "injection_depth_grid": (0.0005, 0.001),
        "epoch_phase_fraction_grid": (0.45,),
        "min_period_days": 1.0,
        "max_period_days": 15.0,
        "n_periods": 3000,
        "min_duration_hours": 1.5,
        "max_duration_hours": 10.0,
        "n_durations": 8,
        "edge_width_cadences": 0,
        "min_edge_observations": 4,
        "min_transit_events": 3,
        "min_event_consistency_fraction": 0.60,
        "top_k": 5,
        "search_mode": "coarse_to_fine",
        "n_coarse_periods": 1000,
        "n_refinement_regions": 12,
        "refinement_half_width_points": 30,
        "period_match_tolerance_fraction": 0.02,
        "bls_objective": "snr",
        "bls_oversample": 5,
        "null_trials_per_star": 8,
        "null_block_size_cadences": 24,
        "fap_level": 0.01,
        "random_seed": 123,
        "minimum_success_fraction": 0.90,
        "max_workers": 6,
        "star_timeout_seconds": None,
        "download_max_attempts": 5,
        "download_initial_wait_seconds": 5.0,
        "download_backoff_factor": 2.0,
        "progress_interval": 1,
        "checkpoint_interval": 5,
        "resume": True,
        "rerun_failures": False,
        "cache_light_curves": True,
        "cache_base_arima": True,
        "save_regularized_inputs": False,
    }
    if profile == "full":
        settings.update(
            arima_injection_mode="refit",
            injection_depth_grid=(0.0002, 0.0005, 0.001),
            epoch_phase_fraction_grid=(0.15, 0.45, 0.75),
            n_periods=10000,
            top_k=10,
            n_coarse_periods=4000,
            n_refinement_regions=30,
            refinement_half_width_points=40,
            bls_oversample=10,
            null_trials_per_star=20,
        )
    elif profile == "smoke":
        settings.update(
            target_limit=3,
            strict_target_count=False,
            n_periods=500,
            n_durations=4,
            top_k=3,
            n_coarse_periods=200,
            n_refinement_regions=4,
            refinement_half_width_points=10,
            bls_oversample=3,
            null_trials_per_star=2,
            injection_period_grid=(5.0,),
            injection_duration_hours_grid=(4.0,),
            injection_depth_grid=(0.001,),
            epoch_phase_fraction_grid=(0.45,),
            max_workers=2,
        )
    elif profile != "optimized":
        raise ValueError("profile must be optimized, full, or smoke.")
    return SimpleNamespace(**settings)

def parse_order(value):
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("ARIMA order must look like p,d,q.")
    try:
        order = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ARIMA order entries must be integers.") from exc
    if any(part < 0 for part in order):
        raise argparse.ArgumentTypeError("ARIMA order entries must be non-negative.")
    return order

def parse_float_grid(value):
    try:
        values = tuple(float(part.strip()) for part in str(value).split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated floats.") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Grid values must be positive.")
    return values

def parse_target_ids(value):
    values = tuple(normalize_target_id(part) for part in str(value).split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one target id.")
    return values

def build_parser():
    defaults = default_settings()
    parser = argparse.ArgumentParser(description="Run a multi-star Kepler BLS and ARIMA-TCF benchmark.")
    parser.add_argument("--profile", choices=("optimized", "full", "smoke"), default=defaults.profile)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--target-limit", type=int)
    parser.add_argument("--strict-target-count", dest="strict_target_count", action="store_true")
    parser.add_argument("--allow-partial-target-count", dest="strict_target_count", action="store_false")
    parser.set_defaults(strict_target_count=None)
    parser.add_argument("--target-ids", type=parse_target_ids)
    parser.add_argument("--batch-index", type=int)
    parser.add_argument("--batch-count", type=int)
    parser.add_argument("--arima-order", type=parse_order)
    parser.add_argument("--fit-maxiter", type=int)
    parser.add_argument("--arima-injection-mode", choices=("filter", "refit"))
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
    parser.add_argument("--null-trials-per-star", type=int)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--star-timeout-seconds", type=float)
    parser.add_argument("--download-max-attempts", type=int)
    parser.add_argument("--download-initial-wait-seconds", type=float)
    parser.add_argument("--download-backoff-factor", type=float)
    parser.add_argument("--progress-interval", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--rerun-failures", action="store_true")
    parser.add_argument("--no-light-curve-cache", dest="cache_light_curves", action="store_false")
    parser.add_argument("--no-base-arima-cache", dest="cache_base_arima", action="store_false")
    parser.add_argument("--save-regularized-inputs", action="store_true")
    return parser

def parse_args(argv=None):
    parser = build_parser()
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

def report_progress(progress_queue, target_id, quarter, stage, units=0, detail=""):
    if progress_queue is None:
        return
    try:
        progress_queue.put({"target_id": normalize_target_id(target_id), "quarter": int(quarter), "stage": str(stage), "units": int(units), "detail": str(detail)}, block=False)
    except Exception:
        pass

def write_star_checkpoint(star_dir, target_id, quarter, status, stage, **extra):
    star_dir.mkdir(parents=True, exist_ok=True)
    payload = {"target_id": normalize_target_id(target_id), "quarter": int(quarter), "status": str(status), "stage": str(stage), **extra}
    (star_dir / "checkpoint.json").write_text(json.dumps(json_ready(payload), indent=2) + "\n")

def check_star_timeout(start, args, target_id, quarter):
    timeout = args.get("star_timeout_seconds")
    if timeout is None:
        return
    elapsed = perf_counter() - float(start)
    if elapsed > float(timeout):
        raise TimeoutError(f"KIC {normalize_target_id(target_id)} Q{int(quarter)} exceeded star timeout of {float(timeout):.1f} seconds.")

def should_checkpoint(count, args):
    interval = max(1, int(args.get("checkpoint_interval", 1)))
    return int(count) % interval == 0

def progress_interval(args):
    return max(1, int(args.get("progress_interval", 1)))

def light_curve_cache_path(args, target_id, quarter):
    return Path(args["cache_dir"]) / f"{star_prefix(target_id, quarter)}_pdcsap.parquet"

def is_nonretryable_download_error(exc):
    message = str(exc)
    return "No Kepler light curve found" in message

def is_transient_download_error_message(message):
    message = str(message)
    transient_markers = (
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectionError",
        "HTTPSConnectionPool",
        "Max retries exceeded",
        "Temporary failure",
        "temporarily unavailable",
        "Connection reset",
        "RemoteDisconnected",
        "mast.stsci.edu",
    )
    return any(marker in message for marker in transient_markers)

def is_transient_download_error(exc):
    if is_nonretryable_download_error(exc):
        return False
    return is_transient_download_error_message(f"{type(exc).__name__}: {exc}")

def download_retry_wait_seconds(args, attempt):
    return float(args.get("download_initial_wait_seconds", 5.0)) * float(args.get("download_backoff_factor", 2.0)) ** max(0, int(attempt) - 1)

def load_light_curve_frame(target_id, quarter, args, progress_queue=None):
    cache_path = light_curve_cache_path(args, target_id, quarter)
    if args.get("cache_light_curves", True) and cache_path.exists():
        return pd.read_parquet(cache_path), True
    max_attempts = max(1, int(args.get("download_max_attempts", 1)))
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            report_progress(progress_queue, target_id, quarter, "download attempt", detail=f"{attempt}/{max_attempts}")
            light_curve = load_kepler_pdcsap(target_id, quarter)
            break
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts or not is_transient_download_error(exc):
                raise
            wait_seconds = download_retry_wait_seconds(args, attempt)
            report_progress(progress_queue, target_id, quarter, "download retry", detail=f"{attempt}/{max_attempts} failed; waiting {wait_seconds:.0f}s")
            sleep(wait_seconds)
    else:
        raise last_exc
    frame = light_curve.to_dataframe()
    if args.get("cache_light_curves", True):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path, index=False)
    return frame, False

def base_arima_cache_paths(star_dir):
    star_dir = Path(star_dir)
    return {
        "innovations": star_dir / "base_arima_innovations.npy",
        "params": star_dir / "base_arima_params.npy",
        "summary": star_dir / "base_arima_summary.json",
    }

def base_arima_cache_is_compatible(summary, args, flux):
    return (
        tuple(summary.get("order", ())) == tuple(args["arima_order"])
        and int(summary.get("fit_maxiter", -1)) == int(args["fit_maxiter"])
        and int(summary.get("flux_length", -1)) == len(flux)
    )

def load_base_arima_cache(star_dir, flux, args):
    paths = base_arima_cache_paths(star_dir)
    if not args.get("resume", True) or not args.get("cache_base_arima", True):
        return None
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
    if not args.get("cache_base_arima", True):
        return
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

def load_or_fit_base_arima(star_dir, flux, args):
    cached = load_base_arima_cache(star_dir, flux, args)
    if cached is not None:
        return cached, 0.0
    started = perf_counter()
    result = fit_arima_innovations(flux, order=tuple(args["arima_order"]), maxiter=args["fit_maxiter"])
    runtime = float(perf_counter() - started)
    params = getattr(result["fit"], "params", None)
    result["params"] = np.asarray(params, dtype=float) if params is not None else None
    result["from_cache"] = False
    save_base_arima_cache(star_dir, result, flux, args)
    return result, runtime

def filter_arima_innovations(flux, base_arima, args):
    params = base_arima.get("params")
    if params is None and base_arima.get("fit") is not None:
        params = getattr(base_arima["fit"], "params", None)
    if params is None:
        raise ValueError("Cached ARIMA parameters are unavailable; cannot filter injected flux.")
    series = np.asarray(flux, dtype=float).reshape(-1)
    if np.isfinite(series).sum() < 24:
        raise ValueError("At least 24 finite flux values are required.")
    model = ARIMA(series, order=tuple(args["arima_order"]), trend="n", enforce_stationarity=False, enforce_invertibility=False)
    filtered = model.filter(np.asarray(params, dtype=float))
    innovations = np.asarray(filtered.resid, dtype=float).reshape(-1)
    innovations[~np.isfinite(series)] = np.nan
    burn = int(getattr(filtered, "loglikelihood_burn", 0))
    if burn > 0:
        innovations[:burn] = np.nan
    summary = {"order": tuple(args["arima_order"]), "converged": True, "aic": float(filtered.aic), "bic": float(filtered.bic), "finite_innovation_count": int(np.isfinite(innovations).sum()), "mode": "filter_fixed_base_params"}
    return {"innovations": innovations, "fit": filtered, "params": np.asarray(params, dtype=float), "summary": summary}

def original_searches_cache_path(star_dir):
    return Path(star_dir) / "original_searches.json"

def original_searches_cache_config(args):
    return {
        "arima_order": tuple(args["arima_order"]),
        "n_periods": int(args["n_periods"]),
        "n_durations": int(args["n_durations"]),
        "search_mode": str(args["search_mode"]),
        "n_coarse_periods": int(args["n_coarse_periods"]),
        "n_refinement_regions": int(args["n_refinement_regions"]),
        "refinement_half_width_points": int(args["refinement_half_width_points"]),
        "top_k": int(args["top_k"]),
        "bls_objective": str(args["bls_objective"]),
        "bls_oversample": int(args["bls_oversample"]),
    }

def load_original_searches_cache(star_dir, args):
    path = original_searches_cache_path(star_dir)
    if not args.get("resume", True) or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        if payload.get("config") != json_ready(original_searches_cache_config(args)):
            return None
        return payload
    except Exception:
        return None

def save_original_searches_cache(star_dir, payload):
    path = original_searches_cache_path(star_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2) + "\n")

def select_manifest_batch(manifest, args):
    batch_count = int(args.batch_count)
    batch_index = int(args.batch_index)
    if batch_count < 1:
        raise ValueError("batch_count must be at least 1.")
    if batch_index < 1 or batch_index > batch_count:
        raise ValueError("batch_index must be between 1 and batch_count.")
    if batch_count == 1:
        return manifest
    batch_size = int(np.ceil(len(manifest) / batch_count))
    start = (batch_index - 1) * batch_size
    stop = min(len(manifest), start + batch_size)
    selected = manifest.iloc[start:stop].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"Batch {batch_index}/{batch_count} selected no targets.")
    return selected

def load_manifest(args):
    path = Path(args.manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Target manifest does not exist: {path}")
    manifest = pd.read_csv(path)
    required = {"target_id", "quarter"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Target manifest is missing columns: {sorted(missing)}")
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
    manifest = manifest.head(int(args.target_limit)).copy()
    if args.strict_target_count and len(manifest) != int(args.target_limit):
        raise ValueError(f"Expected exactly {args.target_limit} unique target-quarter rows but found {len(manifest)}.")
    if manifest.empty:
        raise ValueError("Target manifest contains no usable rows.")
    return select_manifest_batch(manifest, args)

def resolve_worker_count(args, target_count):
    available_cpus = os.cpu_count() or 1
    requested = int(args.max_workers) if args.max_workers is not None else max(1, available_cpus - 1)
    return max(1, min(requested, available_cpus, int(target_count)))

def robust_scale(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    median = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - median)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(values, ddof=1))
    return scale

def robust_standardize(values):
    values = np.asarray(values, dtype=float)
    location = float(np.nanmedian(values))
    scale = robust_scale(values)
    if not np.isfinite(scale) or scale <= 0:
        return np.full(values.shape, np.nan, dtype=float)
    return (values - location) / scale

def median_cadence(time):
    values = np.sort(np.unique(np.asarray(time, dtype=float)[np.isfinite(time)]))
    differences = np.diff(values)
    differences = differences[np.isfinite(differences) & (differences > 0)]
    return float(np.median(differences)) if differences.size else np.nan

def lag_one_acf(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    first = values[:-1]
    second = values[1:]
    valid = finite[:-1] & finite[1:]
    if valid.sum() < 3:
        return np.nan
    first = first[valid]
    second = second[valid]
    if np.std(first) <= 0 or np.std(second) <= 0:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])

def six_hour_scatter_proxy(time, flux):
    cadence = median_cadence(time)
    if not np.isfinite(cadence) or cadence <= 0:
        return np.nan
    window = max(3, int(round((6.0 / 24.0) / cadence)))
    series = pd.Series(np.asarray(flux, dtype=float))
    rolling = series.rolling(window=window, center=True, min_periods=max(2, window // 2)).mean()
    residual = series - rolling
    return float(robust_scale(residual.to_numpy(dtype=float)) * 1.0e6)

def calculate_star_metrics(time, flux):
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    finite_time = time[finite]
    finite_flux = flux[finite]
    return {"n_grid_observations": int(len(time)), "n_finite_observations": int(finite.sum()), "finite_fraction": float(finite.mean()), "gap_fraction": float(1.0 - finite.mean()), "baseline_days": float(np.max(finite_time) - np.min(finite_time)), "median_cadence_days": median_cadence(finite_time), "median_flux": float(np.median(finite_flux)), "robust_flux_scatter_ppm": float(robust_scale(finite_flux) * 1.0e6), "flux_standard_deviation_ppm": float(np.std(finite_flux, ddof=1) * 1.0e6), "lag_one_flux_acf": lag_one_acf(flux), "six_hour_scatter_proxy_ppm": six_hour_scatter_proxy(time, flux)}

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

def run_bls(time, flux, period_grid, duration_grid, objective="snr", oversample=10, top_k=10):
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    if finite.sum() < 24:
        raise ValueError("At least 24 finite observations are required for BLS.")
    model = BoxLeastSquares(time[finite], flux[finite])
    result = model.power(np.asarray(period_grid, dtype=float), np.asarray(duration_grid, dtype=float), objective=objective, method="fast", oversample=int(oversample))
    periodogram = pd.DataFrame({"period_days": np.asarray(result.period, dtype=float), "power": np.asarray(result.power, dtype=float), "duration_days": np.asarray(result.duration, dtype=float), "transit_time": np.asarray(result.transit_time, dtype=float), "depth": np.asarray(result.depth, dtype=float)})
    periodogram["sde"] = robust_standardize(periodogram["power"].to_numpy(dtype=float))
    top_peaks = select_bls_top_peaks(periodogram, top_k=top_k, separation_fraction=0.01)
    if top_peaks.empty:
        raise ValueError("BLS did not produce any finite local peaks.")
    summary = top_peaks.iloc[0].to_dict()
    summary["objective"] = str(objective)
    return {"summary": summary, "periodogram": periodogram, "top_peaks": top_peaks}

def matching_rank_from_periods(periods, target_period, tolerance_fraction):
    periods = np.asarray(periods, dtype=float)
    errors = np.abs(periods - float(target_period)) / float(target_period)
    matches = np.flatnonzero(errors <= float(tolerance_fraction))
    return int(matches[0] + 1) if matches.size else None

def harmonic_rank_from_periods(periods, injected_period, factor, tolerance_fraction):
    return matching_rank_from_periods(periods, float(injected_period) * float(factor), tolerance_fraction)

def run_tcf_search(time, innovations, period_grid, duration_grid, args):
    return run_tcf(time, innovations, period_grid, duration_grid, edge_width_cadences=args["edge_width_cadences"], min_edge_observations=args["min_edge_observations"], min_transit_events=args["min_transit_events"], min_event_consistency_fraction=args["min_event_consistency_fraction"], top_k=args["top_k"], search_mode=args["search_mode"], n_coarse_periods=args["n_coarse_periods"], n_refinement_regions=args["n_refinement_regions"], refinement_half_width_points=args["refinement_half_width_points"])

def run_original_searches(time, flux, period_grid, duration_grid, base_arima, args):
    tcf_start = perf_counter()
    tcf_result = run_tcf_search(time, base_arima["innovations"], period_grid, duration_grid, args)
    tcf_runtime = float(perf_counter() - tcf_start)
    bls_start = perf_counter()
    bls_result = run_bls(time, flux, period_grid, duration_grid, objective=args["bls_objective"], oversample=args["bls_oversample"], top_k=args["top_k"])
    bls_runtime = float(perf_counter() - bls_start)
    return tcf_result, bls_result, tcf_runtime, bls_runtime

def load_or_run_original_searches(star_dir, time, flux, period_grid, duration_grid, base_arima, args):
    cached = load_original_searches_cache(star_dir, args)
    if cached is not None:
        return {"summary": cached["original_tcf_summary"]}, {"summary": cached["original_bls_summary"]}, float(cached["original_tcf_runtime_seconds"]), float(cached["original_bls_runtime_seconds"]), True
    tcf_result, bls_result, tcf_runtime, bls_runtime = run_original_searches(time, flux, period_grid, duration_grid, base_arima, args)
    save_original_searches_cache(
        star_dir,
        {
            "config": original_searches_cache_config(args),
            "original_tcf_runtime_seconds": tcf_runtime,
            "original_bls_runtime_seconds": bls_runtime,
            "original_tcf_summary": tcf_result["summary"],
            "original_bls_summary": bls_result["summary"],
        },
    )
    return tcf_result, bls_result, tcf_runtime, bls_runtime, False

def save_rows(path, rows, sort_columns):
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(sort_columns).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame

def run_null_trials(time, flux, base_innovations, period_grid, duration_grid, target_id, quarter, args, star_dir=None, progress_queue=None, start=None):
    root_sequence = np.random.SeedSequence([int(args["random_seed"]), int(normalize_target_id(target_id)), int(quarter)])
    child_sequences = root_sequence.spawn(int(args["null_trials_per_star"]))
    total_trials = int(args["null_trials_per_star"])
    last_reported_count = 0
    null_path = Path(star_dir) / "null_trials.csv" if star_dir is not None else None
    rows = []
    completed_trials = set()
    if null_path is not None and args.get("resume", True) and null_path.exists():
        existing = pd.read_csv(null_path)
        if "trial" in existing.columns:
            rows = existing.to_dict("records")
            completed_trials = {int(value) for value in existing["trial"].dropna().to_numpy(dtype=int)}
            if completed_trials:
                last_reported_count = len(completed_trials)
                report_progress(progress_queue, target_id, quarter, "null trials resumed", units=last_reported_count, detail=f"{last_reported_count}/{total_trials}")
    for trial, sequence in enumerate(child_sequences):
        if trial in completed_trials:
            continue
        if start is not None:
            check_star_timeout(start, args, target_id, quarter)
        seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
        rng = np.random.default_rng(seed)
        row = {"target_id": normalize_target_id(target_id), "quarter": int(quarter), "trial": int(trial), "trial_seed": seed, "tcf_success": False, "bls_success": False, "tcf_max_score": np.nan, "tcf_best_period_days": np.nan, "tcf_valid_transit_events": np.nan, "tcf_positive_event_fraction": np.nan, "bls_max_sde": np.nan, "bls_max_power": np.nan, "bls_best_period_days": np.nan, "tcf_error": "", "bls_error": ""}
        try:
            surrogate_innovations = moving_block_surrogate(base_innovations, block_size=args["null_block_size_cadences"], rng=rng)
            tcf_result = run_tcf_search(time, surrogate_innovations, period_grid, duration_grid, args)
            best = tcf_result["summary"]
            row.update({"tcf_success": True, "tcf_max_score": float(best["score"]), "tcf_best_period_days": float(best["period"]), "tcf_valid_transit_events": int(best["n_valid_transit_events"]), "tcf_positive_event_fraction": float(best["positive_event_fraction"])})
        except Exception as exc:
            row["tcf_error"] = f"{type(exc).__name__}: {exc}"
        try:
            surrogate_flux = moving_block_surrogate(flux, block_size=args["null_block_size_cadences"], rng=rng)
            bls_result = run_bls(time, surrogate_flux, period_grid, duration_grid, objective=args["bls_objective"], oversample=args["bls_oversample"], top_k=1)
            best = bls_result["summary"]
            row.update({"bls_success": True, "bls_max_sde": float(best["sde"]), "bls_max_power": float(best["power"]), "bls_best_period_days": float(best["period_days"])})
        except Exception as exc:
            row["bls_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        completed_count = len({int(item["trial"]) for item in rows})
        if null_path is not None and (should_checkpoint(completed_count, args) or completed_count == total_trials):
            save_rows(null_path, rows, ["trial"])
            write_star_checkpoint(star_dir, target_id, quarter, "running", "null_trials", completed_null_trials=completed_count)
        if completed_count - last_reported_count >= progress_interval(args) or completed_count == total_trials:
            report_progress(progress_queue, target_id, quarter, "null trial", units=completed_count - last_reported_count, detail=f"{completed_count}/{total_trials}")
            last_reported_count = completed_count
    return save_rows(null_path, rows, ["trial"]) if null_path is not None else pd.DataFrame(rows)

def run_one_injection(time, flux, period_grid, duration_grid, case, target_id, quarter, selection_group, args, base_arima=None):
    injected_period, injected_duration_hours, injected_depth, epoch_phase_fraction = case
    finite = np.isfinite(time) & np.isfinite(flux)
    epoch = float(np.min(time[finite]) + float(epoch_phase_fraction) * float(injected_period))
    duration_days = float(injected_duration_hours) / 24.0
    injected_flux, template, in_transit = inject_periodic_box_transit(time, flux, injected_period, epoch, duration_days, injected_depth)
    arima_mode = str(args.get("arima_injection_mode", "refit"))
    row = {"target_id": normalize_target_id(target_id), "quarter": int(quarter), "selection_group": str(selection_group), "injected_period_days": float(injected_period), "injected_epoch_days": epoch, "epoch_phase_fraction": float(epoch_phase_fraction), "injected_duration_hours": float(injected_duration_hours), "injected_depth": float(injected_depth), "in_transit_observation_count": int(np.isfinite(flux[in_transit]).sum()), "arima_injection_mode": arima_mode, "success": False, "error": ""}
    total_start = perf_counter()
    try:
        arima_start = perf_counter()
        if arima_mode == "filter":
            arima_result = filter_arima_innovations(injected_flux, base_arima, args)
        else:
            arima_result = fit_arima_innovations(injected_flux, order=tuple(args["arima_order"]), maxiter=args["fit_maxiter"])
        arima_runtime = float(perf_counter() - arima_start)
        tcf_start = perf_counter()
        tcf_result = run_tcf_search(time, arima_result["innovations"], period_grid, duration_grid, args)
        tcf_runtime = float(perf_counter() - tcf_start)
        bls_start = perf_counter()
        bls_result = run_bls(time, injected_flux, period_grid, duration_grid, objective=args["bls_objective"], oversample=args["bls_oversample"], top_k=args["top_k"])
        bls_runtime = float(perf_counter() - bls_start)
        tcf_best = tcf_result["summary"]
        bls_best = bls_result["summary"]
        tcf_peaks = tcf_result["top_peaks"]
        bls_peaks = bls_result["top_peaks"]
        tcf_periods = tcf_peaks["period_days"].to_numpy(dtype=float)
        bls_periods = bls_peaks["period_days"].to_numpy(dtype=float)
        tcf_recovered_period = float(tcf_best["period"])
        bls_recovered_period = float(bls_best["period_days"])
        tcf_period_error = float(period_match_fraction(tcf_recovered_period, injected_period))
        bls_period_error = float(period_match_fraction(bls_recovered_period, injected_period))
        tcf_exact_error = float(abs(tcf_recovered_period - injected_period) / injected_period)
        bls_exact_error = float(abs(bls_recovered_period - injected_period) / injected_period)
        tolerance = float(args["period_match_tolerance_fraction"])
        tcf_exact_rank = matching_peak_rank(tcf_peaks, injected_period, tolerance_fraction=tolerance)
        bls_exact_rank = matching_rank_from_periods(bls_periods, injected_period, tolerance)
        row.update({"success": True, "arima_converged": bool(arima_result["summary"]["converged"]), "arima_runtime_seconds": arima_runtime, "tcf_runtime_seconds": tcf_runtime, "bls_runtime_seconds": bls_runtime, "total_runtime_seconds": float(perf_counter() - total_start), "tcf_recovered_period_days": tcf_recovered_period, "tcf_recovered_epoch_days": float(tcf_best["epoch"]), "tcf_recovered_duration_hours": float(tcf_best["duration"] * 24.0), "tcf_score": float(tcf_best["score"]), "tcf_raw_pooled_score": float(tcf_best["raw_pooled_score"]), "tcf_valid_transit_events": int(tcf_best["n_valid_transit_events"]), "tcf_positive_transit_events": int(tcf_best["n_positive_transit_events"]), "tcf_positive_event_fraction": float(tcf_best["positive_event_fraction"]), "tcf_median_event_score": float(tcf_best["median_event_score"]), "tcf_period_error_fraction": tcf_period_error, "tcf_exact_period_error_fraction": tcf_exact_error, "tcf_period_matched": bool(tcf_period_error <= tolerance), "tcf_exact_period_matched": bool(tcf_exact_error <= tolerance), "tcf_exact_period_rank_top10": tcf_exact_rank, "tcf_exact_period_present_top10": bool(tcf_exact_rank is not None), "tcf_half_period_rank_top10": harmonic_peak_rank(tcf_peaks, injected_period, 0.5, tolerance_fraction=tolerance), "tcf_double_period_rank_top10": harmonic_peak_rank(tcf_peaks, injected_period, 2.0, tolerance_fraction=tolerance), "tcf_triple_period_rank_top10": harmonic_peak_rank(tcf_peaks, injected_period, 3.0, tolerance_fraction=tolerance), "tcf_top_periods_json": json.dumps([float(value) for value in tcf_periods]), "tcf_top_scores_json": json.dumps([float(value) for value in tcf_peaks["score"].to_numpy(dtype=float)]), "bls_recovered_period_days": bls_recovered_period, "bls_recovered_transit_time": float(bls_best["transit_time"]), "bls_recovered_duration_hours": float(bls_best["duration_days"] * 24.0), "bls_recovered_depth": float(bls_best["depth"]), "bls_power": float(bls_best["power"]), "bls_sde": float(bls_best["sde"]), "bls_period_error_fraction": bls_period_error, "bls_exact_period_error_fraction": bls_exact_error, "bls_period_matched": bool(bls_period_error <= tolerance), "bls_exact_period_matched": bool(bls_exact_error <= tolerance), "bls_exact_period_rank_top10": bls_exact_rank, "bls_exact_period_present_top10": bool(bls_exact_rank is not None), "bls_half_period_rank_top10": harmonic_rank_from_periods(bls_periods, injected_period, 0.5, tolerance), "bls_double_period_rank_top10": harmonic_rank_from_periods(bls_periods, injected_period, 2.0, tolerance), "bls_triple_period_rank_top10": harmonic_rank_from_periods(bls_periods, injected_period, 3.0, tolerance), "bls_top_periods_json": json.dumps([float(value) for value in bls_periods]), "bls_top_sde_json": json.dumps([float(value) for value in bls_peaks["sde"].to_numpy(dtype=float)])})
    except Exception as exc:
        row.update({"total_runtime_seconds": float(perf_counter() - total_start), "error": f"{type(exc).__name__}: {exc}"})
    return row

def injection_case_key(case):
    return tuple(round(float(value), 12) for value in case)

def injection_row_key(row):
    return injection_case_key((row["injected_period_days"], row["injected_duration_hours"], row["injected_depth"], row["epoch_phase_fraction"]))

def run_injection_grid(time, flux, period_grid, duration_grid, cases, target_id, quarter, selection_group, args, base_arima, star_dir=None, progress_queue=None, start=None):
    injection_path = Path(star_dir) / "injections.csv" if star_dir is not None else None
    rows = []
    completed_keys = set()
    last_reported_count = 0
    if injection_path is not None and args.get("resume", True) and injection_path.exists():
        existing = pd.read_csv(injection_path)
        required = {"injected_period_days", "injected_duration_hours", "injected_depth", "epoch_phase_fraction"}
        if required.issubset(existing.columns):
            rows = existing.to_dict("records")
            completed_keys = {injection_row_key(row) for row in rows}
            if completed_keys:
                last_reported_count = len(completed_keys)
                report_progress(progress_queue, target_id, quarter, "injections resumed", units=last_reported_count, detail=f"{last_reported_count}/{len(cases)}")
    for index, case in enumerate(cases, start=1):
        key = injection_case_key(case)
        if key in completed_keys:
            continue
        if start is not None:
            check_star_timeout(start, args, target_id, quarter)
        rows.append(run_one_injection(time, flux, period_grid, duration_grid, case, target_id, quarter, selection_group, args, base_arima=base_arima))
        completed_count = len({injection_row_key(row) for row in rows})
        if injection_path is not None and (should_checkpoint(completed_count, args) or completed_count == len(cases)):
            save_rows(injection_path, rows, ["injected_period_days", "injected_duration_hours", "injected_depth", "epoch_phase_fraction"])
            write_star_checkpoint(star_dir, target_id, quarter, "running", "injections", completed_injections=completed_count, requested_injections=len(cases))
        if completed_count - last_reported_count >= progress_interval(args) or completed_count == len(cases):
            report_progress(progress_queue, target_id, quarter, "injection", units=completed_count - last_reported_count, detail=f"{completed_count}/{len(cases)}")
            last_reported_count = completed_count
    return save_rows(injection_path, rows, ["injected_period_days", "injected_duration_hours", "injected_depth", "epoch_phase_fraction"]) if injection_path is not None else pd.DataFrame(rows)

def save_star_outputs(star_dir, injections, null_trials, summary, regular, args):
    star_dir.mkdir(parents=True, exist_ok=True)
    injections.to_csv(star_dir / "injections.csv", index=False)
    null_trials.to_csv(star_dir / "null_trials.csv", index=False)
    (star_dir / "star_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    if args["save_regularized_inputs"]:
        regular.to_parquet(star_dir / "regularized_light_curve.parquet", index=False)
    (star_dir / "COMPLETE").write_text("complete\n")

def run_star_task(task):
    if len(task) == 3:
        row, args, progress_queue = task
    else:
        row, args = task
        progress_queue = None
    target_id = normalize_target_id(row["target_id"])
    quarter = int(row["quarter"])
    selection_group = str(row.get("selection_group", "unspecified"))
    prefix = star_prefix(target_id, quarter)
    star_dir = Path(args["output_dir"]) / "stars" / prefix
    start = perf_counter()
    try:
        write_star_checkpoint(star_dir, target_id, quarter, "running", "started")
        report_progress(progress_queue, target_id, quarter, "started", units=0)
        light_curve_frame, light_curve_cache_hit = load_light_curve_frame(target_id, quarter, args, progress_queue=progress_queue)
        report_progress(progress_queue, target_id, quarter, "light curve cache" if light_curve_cache_hit else "light curve download", units=1)
        check_star_timeout(start, args, target_id, quarter)
        regular, preprocessing = preprocess_pdcsap_light_curve(light_curve_frame, quality_policy=args["quality_policy"], require_finite_flux_error=args["require_finite_flux_error"], normalization_fit_fraction=1.0 - args["test_fraction"])
        report_progress(progress_queue, target_id, quarter, "preprocess", units=1)
        time = regular["time"].to_numpy(dtype=float)
        flux = regular["normalized_flux"].to_numpy(dtype=float)
        finite = np.isfinite(time) & np.isfinite(flux)
        if finite.sum() < 24:
            raise ValueError("Insufficient finite light-curve observations.")
        period_grid = default_period_grid(time, min_period_days=args["min_period_days"], max_period_days=args["max_period_days"], n_periods=args["n_periods"])
        duration_grid = default_duration_grid(args["min_duration_hours"], args["max_duration_hours"], args["n_durations"])
        star_metrics = calculate_star_metrics(time, flux)
        write_star_checkpoint(star_dir, target_id, quarter, "running", "preprocessed", finite_observations=int(finite.sum()))
        check_star_timeout(start, args, target_id, quarter)
        base_arima, base_arima_runtime = load_or_fit_base_arima(star_dir, flux, args)
        report_progress(progress_queue, target_id, quarter, "base ARIMA cache" if base_arima.get("from_cache") else "base ARIMA fit", units=1)
        write_star_checkpoint(star_dir, target_id, quarter, "running", "base_arima", base_arima_from_cache=bool(base_arima.get("from_cache")), base_arima_runtime_seconds=base_arima_runtime)
        check_star_timeout(start, args, target_id, quarter)
        original_tcf, original_bls, original_tcf_runtime, original_bls_runtime, original_searches_from_cache = load_or_run_original_searches(star_dir, time, flux, period_grid, duration_grid, base_arima, args)
        report_progress(progress_queue, target_id, quarter, "original searches cache" if original_searches_from_cache else "original searches", units=1)
        write_star_checkpoint(star_dir, target_id, quarter, "running", "original_searches", original_searches_from_cache=bool(original_searches_from_cache), original_tcf_runtime_seconds=original_tcf_runtime, original_bls_runtime_seconds=original_bls_runtime)
        check_star_timeout(start, args, target_id, quarter)
        null_trials = run_null_trials(time, flux, base_arima["innovations"], period_grid, duration_grid, target_id, quarter, args, star_dir=star_dir, progress_queue=progress_queue, start=start)
        cases = list(product(args["injection_period_grid"], args["injection_duration_hours_grid"], args["injection_depth_grid"], args["epoch_phase_fraction_grid"]))
        check_star_timeout(start, args, target_id, quarter)
        injections = run_injection_grid(time, flux, period_grid, duration_grid, cases, target_id, quarter, selection_group, args, base_arima, star_dir=star_dir, progress_queue=progress_queue, start=start)
        successful = injections[injections["success"]].copy()
        tcf_best = original_tcf["summary"]
        bls_best = original_bls["summary"]
        exact_ranks = successful["tcf_exact_period_rank_top10"].dropna() if not successful.empty else pd.Series(dtype=float)
        summary = {"target_id": target_id, "quarter": quarter, "selection_group": selection_group, "status": "success", "profile": str(args.get("profile", "unknown")), "runtime_seconds": float(perf_counter() - start), "light_curve_cache_hit": bool(light_curve_cache_hit), "base_arima_from_cache": bool(base_arima.get("from_cache")), "original_searches_from_cache": bool(original_searches_from_cache), "arima_injection_mode": str(args.get("arima_injection_mode", "refit")), "injection_count_requested": int(len(cases)), "injection_count_successful": int(len(successful)), "injection_success_fraction": float(len(successful) / len(cases)), "null_trials_requested": int(args["null_trials_per_star"]), "tcf_null_trials_successful": int(null_trials["tcf_success"].sum()), "bls_null_trials_successful": int(null_trials["bls_success"].sum()), "base_arima_converged": bool(base_arima["summary"]["converged"]), "base_arima_runtime_seconds": base_arima_runtime, "original_tcf_runtime_seconds": original_tcf_runtime, "original_bls_runtime_seconds": original_bls_runtime, "original_tcf_period_days": float(tcf_best["period"]), "original_tcf_score": float(tcf_best["score"]), "original_tcf_raw_pooled_score": float(tcf_best["raw_pooled_score"]), "original_tcf_valid_transit_events": int(tcf_best["n_valid_transit_events"]), "original_tcf_positive_event_fraction": float(tcf_best["positive_event_fraction"]), "original_bls_period_days": float(bls_best["period_days"]), "original_bls_sde": float(bls_best["sde"]), "original_bls_power": float(bls_best["power"]), "tcf_harmonic_rank1_rate_before_fap": float(successful["tcf_period_matched"].mean()), "tcf_exact_rank1_rate_before_fap": float(successful["tcf_exact_period_matched"].mean()), "tcf_exact_top10_rate": float(successful["tcf_exact_period_present_top10"].mean()), "tcf_median_exact_rank_when_present": float(exact_ranks.median()) if not exact_ranks.empty else None, "bls_harmonic_rank1_rate_before_fap": float(successful["bls_period_matched"].mean()), "bls_exact_rank1_rate_before_fap": float(successful["bls_exact_period_matched"].mean()), "bls_exact_top10_rate": float(successful["bls_exact_period_present_top10"].mean()), **star_metrics}
        save_star_outputs(star_dir, injections, null_trials, summary, regular, args)
        write_star_checkpoint(star_dir, target_id, quarter, "success", "complete", runtime_seconds=summary["runtime_seconds"])
        report_progress(progress_queue, target_id, quarter, "saved outputs", units=1)
        return {"target_id": target_id, "quarter": quarter, "selection_group": selection_group, "status": "success", "star_dir": str(star_dir), "runtime_seconds": summary["runtime_seconds"], "error": ""}
    except Exception as exc:
        star_dir.mkdir(parents=True, exist_ok=True)
        failure = {"target_id": target_id, "quarter": quarter, "selection_group": selection_group, "status": "failed", "star_dir": str(star_dir), "runtime_seconds": float(perf_counter() - start), "error": f"{type(exc).__name__}: {exc}"}
        (star_dir / "failure.json").write_text(json.dumps(json_ready(failure), indent=2) + "\n")
        write_star_checkpoint(star_dir, target_id, quarter, "failed", "failed", error=failure["error"], runtime_seconds=failure["runtime_seconds"])
        return failure

def completed_summary_is_compatible(summary, args):
    expected_injections = injection_case_count(args)
    expected_nulls = int(args.null_trials_per_star)
    if int(summary.get("injection_count_requested", -1)) != expected_injections:
        return False
    if int(summary.get("null_trials_requested", -1)) != expected_nulls:
        return False
    profile = summary.get("profile")
    if profile is not None and str(profile) != str(args.profile):
        return False
    arima_mode = summary.get("arima_injection_mode")
    if arima_mode is not None and str(arima_mode) != str(args.arima_injection_mode):
        return False
    return True

def star_is_complete(output_dir, target_id, quarter, args=None):
    star_dir = Path(output_dir) / "stars" / star_prefix(target_id, quarter)
    if not (star_dir / "COMPLETE").exists():
        return False
    if args is None:
        return True
    summary_path = star_dir / "star_summary.json"
    if not summary_path.exists():
        return False
    try:
        return completed_summary_is_compatible(json.loads(summary_path.read_text()), args)
    except Exception:
        return False

def star_has_failure(output_dir, target_id, quarter):
    return (Path(output_dir) / "stars" / star_prefix(target_id, quarter) / "failure.json").exists()

def failed_result(output_dir, row):
    target_id = normalize_target_id(row["target_id"])
    quarter = int(row["quarter"])
    star_dir = Path(output_dir) / "stars" / star_prefix(target_id, quarter)
    try:
        failure = json.loads((star_dir / "failure.json").read_text())
    except Exception:
        failure = {}
    return {"target_id": target_id, "quarter": quarter, "selection_group": str(row.get("selection_group", failure.get("selection_group", "unspecified"))), "status": "failed", "star_dir": str(star_dir), "runtime_seconds": float(failure.get("runtime_seconds", np.nan)), "error": str(failure.get("error", "Existing failure.json"))}

def existing_failure_is_transient(output_dir, target_id, quarter):
    star_dir = Path(output_dir) / "stars" / star_prefix(target_id, quarter)
    try:
        failure = json.loads((star_dir / "failure.json").read_text())
    except Exception:
        return False
    return is_transient_download_error_message(failure.get("error", ""))

def completed_result(output_dir, row):
    target_id = normalize_target_id(row["target_id"])
    quarter = int(row["quarter"])
    star_dir = Path(output_dir) / "stars" / star_prefix(target_id, quarter)
    summary = json.loads((star_dir / "star_summary.json").read_text())
    return {"target_id": target_id, "quarter": quarter, "selection_group": str(row.get("selection_group", "unspecified")), "status": "success", "star_dir": str(star_dir), "runtime_seconds": float(summary.get("runtime_seconds", np.nan)), "error": ""}

def normalize_key_columns(frame):
    frame = frame.copy()
    if "target_id" in frame.columns:
        frame["target_id"] = frame["target_id"].map(normalize_target_id)
    if "quarter" in frame.columns:
        frame["quarter"] = pd.to_numeric(frame["quarter"], errors="raise").astype(int)
    return frame

def load_completed_outputs(results):
    injection_tables = []
    null_tables = []
    summaries = []
    for result in results:
        if result["status"] != "success":
            continue
        star_dir = Path(result["star_dir"])
        injection_tables.append(pd.read_csv(star_dir / "injections.csv", dtype={"target_id": str}))
        null_tables.append(pd.read_csv(star_dir / "null_trials.csv", dtype={"target_id": str}))
        summaries.append(json.loads((star_dir / "star_summary.json").read_text()))
    injections = pd.concat(injection_tables, ignore_index=True) if injection_tables else pd.DataFrame()
    null_trials = pd.concat(null_tables, ignore_index=True) if null_tables else pd.DataFrame()
    star_summaries = pd.DataFrame(summaries)
    injections = normalize_key_columns(injections)
    null_trials = normalize_key_columns(null_trials)
    star_summaries = normalize_key_columns(star_summaries)
    return injections, null_trials, star_summaries

def calibrated_threshold(values, fap_level):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite null scores are available for calibration.")
    return float(np.quantile(values, 1.0 - float(fap_level), method="higher"))

def empirical_p_values(scores, null_scores):
    scores = np.asarray(scores, dtype=float)
    null_scores = np.asarray(null_scores, dtype=float)
    null_scores = null_scores[np.isfinite(null_scores)]
    return np.asarray([(np.sum(null_scores >= score) + 1.0) / (len(null_scores) + 1.0) if np.isfinite(score) else np.nan for score in scores], dtype=float)

def apply_global_calibration(injections, null_trials, star_summaries, args):
    successful_injections = injections[injections["success"].astype(str).str.lower().isin(["true", "1"])].copy()
    tcf_null = null_trials[null_trials["tcf_success"].astype(str).str.lower().isin(["true", "1"])]["tcf_max_score"].to_numpy(dtype=float)
    bls_null = null_trials[null_trials["bls_success"].astype(str).str.lower().isin(["true", "1"])]["bls_max_sde"].to_numpy(dtype=float)
    expected_tcf = int(len(null_trials) * float(args.minimum_success_fraction))
    expected_bls = int(len(null_trials) * float(args.minimum_success_fraction))
    if np.isfinite(tcf_null).sum() < expected_tcf:
        raise RuntimeError("Too few successful TCF null trials for pooled calibration.")
    if np.isfinite(bls_null).sum() < expected_bls:
        raise RuntimeError("Too few successful BLS null trials for pooled calibration.")
    tcf_threshold = calibrated_threshold(tcf_null, args.fap_level)
    bls_threshold = calibrated_threshold(bls_null, args.fap_level)
    successful_injections["tcf_fap_threshold"] = tcf_threshold
    successful_injections["bls_fap_threshold"] = bls_threshold
    successful_injections["tcf_global_empirical_p_value"] = empirical_p_values(successful_injections["tcf_score"], tcf_null)
    successful_injections["bls_global_empirical_p_value"] = empirical_p_values(successful_injections["bls_sde"], bls_null)
    successful_injections["tcf_passes_fap"] = successful_injections["tcf_score"] >= tcf_threshold
    successful_injections["bls_passes_fap"] = successful_injections["bls_sde"] >= bls_threshold
    successful_injections["tcf_harmonic_recovered"] = successful_injections["tcf_period_matched"] & successful_injections["tcf_passes_fap"]
    successful_injections["bls_harmonic_recovered"] = successful_injections["bls_period_matched"] & successful_injections["bls_passes_fap"]
    successful_injections["tcf_exact_recovered"] = successful_injections["tcf_exact_period_matched"] & successful_injections["tcf_passes_fap"]
    successful_injections["bls_exact_recovered"] = successful_injections["bls_exact_period_matched"] & successful_injections["bls_passes_fap"]
    successful_injections["harmonic_both"] = successful_injections["tcf_harmonic_recovered"] & successful_injections["bls_harmonic_recovered"]
    successful_injections["harmonic_tcf_only"] = successful_injections["tcf_harmonic_recovered"] & ~successful_injections["bls_harmonic_recovered"]
    successful_injections["harmonic_bls_only"] = successful_injections["bls_harmonic_recovered"] & ~successful_injections["tcf_harmonic_recovered"]
    successful_injections["harmonic_neither"] = ~successful_injections["tcf_harmonic_recovered"] & ~successful_injections["bls_harmonic_recovered"]
    successful_injections["harmonic_union"] = successful_injections["tcf_harmonic_recovered"] | successful_injections["bls_harmonic_recovered"]
    successful_injections["exact_both"] = successful_injections["tcf_exact_recovered"] & successful_injections["bls_exact_recovered"]
    successful_injections["exact_tcf_only"] = successful_injections["tcf_exact_recovered"] & ~successful_injections["bls_exact_recovered"]
    successful_injections["exact_bls_only"] = successful_injections["bls_exact_recovered"] & ~successful_injections["tcf_exact_recovered"]
    successful_injections["exact_neither"] = ~successful_injections["tcf_exact_recovered"] & ~successful_injections["bls_exact_recovered"]
    successful_injections["exact_union"] = successful_injections["tcf_exact_recovered"] | successful_injections["bls_exact_recovered"]
    star_summaries = star_summaries.copy()
    star_summaries["tcf_global_fap_threshold"] = tcf_threshold
    star_summaries["bls_global_fap_threshold"] = bls_threshold
    star_summaries["original_tcf_passes_global_fap"] = star_summaries["original_tcf_score"] >= tcf_threshold
    star_summaries["original_bls_passes_global_fap"] = star_summaries["original_bls_sde"] >= bls_threshold
    star_summaries["original_tcf_global_empirical_p_value"] = empirical_p_values(star_summaries["original_tcf_score"], tcf_null)
    star_summaries["original_bls_global_empirical_p_value"] = empirical_p_values(star_summaries["original_bls_sde"], bls_null)
    return successful_injections, star_summaries, tcf_threshold, bls_threshold, tcf_null, bls_null

def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = z * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))

def metric_row(name, values):
    values = pd.Series(values).fillna(False).astype(bool)
    total = int(len(values))
    successes = int(values.sum())
    lower, upper = wilson_interval(successes, total)
    return {"metric": str(name), "successes": successes, "total": total, "rate": float(successes / total) if total else np.nan, "ci95_lower": lower, "ci95_upper": upper}

def detector_metrics(injections):
    rows = [metric_row("tcf_harmonic_rank1_before_fap", injections["tcf_period_matched"]), metric_row("bls_harmonic_rank1_before_fap", injections["bls_period_matched"]), metric_row("harmonic_union_rank1_before_fap", injections["tcf_period_matched"] | injections["bls_period_matched"]), metric_row("tcf_exact_rank1_before_fap", injections["tcf_exact_period_matched"]), metric_row("bls_exact_rank1_before_fap", injections["bls_exact_period_matched"]), metric_row("exact_union_rank1_before_fap", injections["tcf_exact_period_matched"] | injections["bls_exact_period_matched"]), metric_row("tcf_exact_period_top10", injections["tcf_exact_period_present_top10"]), metric_row("bls_exact_period_top10", injections["bls_exact_period_present_top10"]), metric_row("tcf_detection_at_global_fap", injections["tcf_passes_fap"]), metric_row("bls_detection_at_global_fap", injections["bls_passes_fap"]), metric_row("tcf_harmonic_recovery_at_global_fap", injections["tcf_harmonic_recovered"]), metric_row("bls_harmonic_recovery_at_global_fap", injections["bls_harmonic_recovered"]), metric_row("harmonic_union_recovery_at_global_fap", injections["harmonic_union"]), metric_row("tcf_exact_recovery_at_global_fap", injections["tcf_exact_recovered"]), metric_row("bls_exact_recovery_at_global_fap", injections["bls_exact_recovered"]), metric_row("exact_union_recovery_at_global_fap", injections["exact_union"])]
    return pd.DataFrame(rows)

def grouped_comparison(injections, column):
    return injections.groupby(column, dropna=False, as_index=False).agg(injection_count=("target_id", "size"), star_count=("target_id", "nunique"), tcf_harmonic_rank1_rate=("tcf_period_matched", "mean"), bls_harmonic_rank1_rate=("bls_period_matched", "mean"), tcf_exact_rank1_rate=("tcf_exact_period_matched", "mean"), bls_exact_rank1_rate=("bls_exact_period_matched", "mean"), tcf_exact_top10_rate=("tcf_exact_period_present_top10", "mean"), bls_exact_top10_rate=("bls_exact_period_present_top10", "mean"), tcf_detection_rate_fap=("tcf_passes_fap", "mean"), bls_detection_rate_fap=("bls_passes_fap", "mean"), tcf_harmonic_recovery_rate_fap=("tcf_harmonic_recovered", "mean"), bls_harmonic_recovery_rate_fap=("bls_harmonic_recovered", "mean"), harmonic_union_recovery_rate_fap=("harmonic_union", "mean"), tcf_exact_recovery_rate_fap=("tcf_exact_recovered", "mean"), bls_exact_recovery_rate_fap=("bls_exact_recovered", "mean"), exact_union_recovery_rate_fap=("exact_union", "mean"), exact_tcf_only_rate=("exact_tcf_only", "mean"), exact_bls_only_rate=("exact_bls_only", "mean"), exact_neither_rate=("exact_neither", "mean"), median_tcf_score=("tcf_score", "median"), median_bls_sde=("bls_sde", "median"), median_tcf_exact_rank_top10=("tcf_exact_period_rank_top10", "median"), median_bls_exact_rank_top10=("bls_exact_period_rank_top10", "median"), arima_convergence_rate=("arima_converged", "mean"), median_total_runtime_seconds=("total_runtime_seconds", "median"))

def add_noise_regimes(injections, star_summaries):
    injections = normalize_key_columns(injections)
    star_summaries = star_summaries.copy()
    star_summaries = normalize_key_columns(star_summaries)
    labels = ["lowest_scatter", "low_scatter", "high_scatter", "highest_scatter"]
    try:
        star_summaries["noise_quartile"] = pd.qcut(star_summaries["robust_flux_scatter_ppm"], q=4, labels=labels, duplicates="drop").astype(str)
    except Exception:
        star_summaries["noise_quartile"] = "unassigned"
    mapping = star_summaries[["target_id", "quarter", "noise_quartile", "robust_flux_scatter_ppm", "gap_fraction", "lag_one_flux_acf", "six_hour_scatter_proxy_ppm"]]
    injections = injections.merge(mapping, on=["target_id", "quarter"], how="left", validate="many_to_one")
    return injections, star_summaries

def per_star_comparison(injections):
    return grouped_comparison(injections, "target_id").merge(injections[["target_id", "quarter", "selection_group", "noise_quartile"]].drop_duplicates("target_id"), on="target_id", how="left")

def rank_distribution(injections, detector):
    column = f"{detector}_exact_period_rank_top10"
    distribution = injections[column].value_counts(dropna=False).sort_index().rename_axis("rank").reset_index(name="count")
    distribution["detector"] = detector
    distribution["fraction"] = distribution["count"] / len(injections)
    return distribution[["detector", "rank", "count", "fraction"]]

def build_global_summary(manifest, task_results, injections, null_trials, star_summaries, detector_summary, tcf_threshold, bls_threshold, args):
    metric_map = detector_summary.set_index("metric")["rate"].to_dict()
    return {"requested_target_count": int(len(manifest)), "successful_target_count": int((pd.DataFrame(task_results)["status"] == "success").sum()), "failed_target_count": int((pd.DataFrame(task_results)["status"] != "success").sum()), "successful_injection_count": int(len(injections)), "expected_injection_count_per_star": int(len(args.injection_period_grid) * len(args.injection_duration_hours_grid) * len(args.injection_depth_grid) * len(args.epoch_phase_fraction_grid)), "pooled_tcf_null_count": int(null_trials["tcf_success"].astype(str).str.lower().isin(["true", "1"]).sum()), "pooled_bls_null_count": int(null_trials["bls_success"].astype(str).str.lower().isin(["true", "1"]).sum()), "profile": str(args.profile), "arima_injection_mode": str(args.arima_injection_mode), "bls_oversample": int(args.bls_oversample), "fap_level": float(args.fap_level), "tcf_global_score_threshold": float(tcf_threshold), "bls_global_sde_threshold": float(bls_threshold), "calibration_scope": "pooled detector-level null maxima across the selected stars", "tcf_surrogate_source": "ARIMA innovations", "bls_surrogate_source": "normalized flux", "search_mode": str(args.search_mode), "n_periods": int(args.n_periods), "n_coarse_periods": int(args.n_coarse_periods), "n_refinement_regions": int(args.n_refinement_regions), "refinement_half_width_points": int(args.refinement_half_width_points), "tcf_harmonic_rank1_rate_before_fap": float(metric_map["tcf_harmonic_rank1_before_fap"]), "bls_harmonic_rank1_rate_before_fap": float(metric_map["bls_harmonic_rank1_before_fap"]), "harmonic_union_rank1_rate_before_fap": float(metric_map["harmonic_union_rank1_before_fap"]), "tcf_exact_rank1_rate_before_fap": float(metric_map["tcf_exact_rank1_before_fap"]), "bls_exact_rank1_rate_before_fap": float(metric_map["bls_exact_rank1_before_fap"]), "exact_union_rank1_rate_before_fap": float(metric_map["exact_union_rank1_before_fap"]), "tcf_exact_top10_rate": float(metric_map["tcf_exact_period_top10"]), "bls_exact_top10_rate": float(metric_map["bls_exact_period_top10"]), "tcf_harmonic_recovery_rate_fap": float(metric_map["tcf_harmonic_recovery_at_global_fap"]), "bls_harmonic_recovery_rate_fap": float(metric_map["bls_harmonic_recovery_at_global_fap"]), "harmonic_union_recovery_rate_fap": float(metric_map["harmonic_union_recovery_at_global_fap"]), "tcf_exact_recovery_rate_fap": float(metric_map["tcf_exact_recovery_at_global_fap"]), "bls_exact_recovery_rate_fap": float(metric_map["bls_exact_recovery_at_global_fap"]), "exact_union_recovery_rate_fap": float(metric_map["exact_union_recovery_at_global_fap"]), "original_tcf_global_false_positive_fraction": float(star_summaries["original_tcf_passes_global_fap"].mean()), "original_bls_global_false_positive_fraction": float(star_summaries["original_bls_passes_global_fap"].mean()), "median_star_runtime_seconds": float(pd.DataFrame(task_results).loc[pd.DataFrame(task_results)["status"] == "success", "runtime_seconds"].median()), "random_seed": int(args.random_seed)}

def save_global_outputs(output_dir, manifest, task_results, injections, null_trials, star_summaries, detector_summary, grouped_tables, rank_table, summary):
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(metrics_dir / "target_manifest_used.csv", index=False)
    pd.DataFrame(task_results).to_csv(metrics_dir / "target_execution_status.csv", index=False)
    injections.to_csv(metrics_dir / "multistar_bls_tcf_injections.csv", index=False)
    null_trials.to_csv(metrics_dir / "multistar_null_trials.csv", index=False)
    star_summaries.to_csv(metrics_dir / "multistar_star_summary.csv", index=False)
    detector_summary.to_csv(metrics_dir / "multistar_detector_summary.csv", index=False)
    rank_table.to_csv(metrics_dir / "multistar_exact_rank_distribution.csv", index=False)
    for name, table in grouped_tables.items():
        table.to_csv(metrics_dir / f"multistar_comparison_by_{name}.csv", index=False)
    injections[injections["exact_tcf_only"]].to_csv(metrics_dir / "multistar_exact_tcf_only.csv", index=False)
    injections[injections["exact_bls_only"]].to_csv(metrics_dir / "multistar_exact_bls_only.csv", index=False)
    injections[injections["exact_neither"]].to_csv(metrics_dir / "multistar_exact_neither.csv", index=False)
    injections[~injections["tcf_exact_period_present_top10"]].to_csv(metrics_dir / "multistar_tcf_exact_absent_top10.csv", index=False)
    star_summaries[star_summaries["original_tcf_passes_global_fap"]].to_csv(metrics_dir / "multistar_original_tcf_false_positive_candidates.csv", index=False)
    star_summaries[star_summaries["original_bls_passes_global_fap"]].to_csv(metrics_dir / "multistar_original_bls_false_positive_candidates.csv", index=False)
    (metrics_dir / "multistar_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    return metrics_dir

def settings_to_worker_dict(args):
    values = vars(args).copy()
    values["manifest_path"] = str(values["manifest_path"])
    values["output_dir"] = str(values["output_dir"])
    values["cache_dir"] = str(values["cache_dir"])
    values["arima_order"] = tuple(values["arima_order"])
    values["injection_period_grid"] = tuple(values["injection_period_grid"])
    values["injection_duration_hours_grid"] = tuple(values["injection_duration_hours_grid"])
    values["injection_depth_grid"] = tuple(values["injection_depth_grid"])
    values["epoch_phase_fraction_grid"] = tuple(values["epoch_phase_fraction_grid"])
    if values.get("target_ids") is not None:
        values["target_ids"] = tuple(values["target_ids"])
    return values

def injection_case_count(args):
    return int(len(args.injection_period_grid) * len(args.injection_duration_hours_grid) * len(args.injection_depth_grid) * len(args.epoch_phase_fraction_grid))

def star_progress_units(args):
    return 5 + int(args.null_trials_per_star) + injection_case_count(args)

def drain_progress_queue(progress_queue, progress, unit_counts=None, max_units_per_star=None):
    while True:
        try:
            event = progress_queue.get_nowait()
        except Empty:
            break
        except Exception:
            break
        key = (normalize_target_id(event.get("target_id")), int(event.get("quarter")))
        units = max(0, int(event.get("units", 0)))
        if unit_counts is not None:
            already_seen = int(unit_counts.get(key, 0))
            if max_units_per_star is not None:
                units = min(units, max(0, int(max_units_per_star) - already_seen))
            unit_counts[key] = already_seen + units
        if units:
            progress.update(units)
        progress.set_postfix_str(f"KIC {event.get('target_id')} Q{event.get('quarter')} {event.get('stage')} {event.get('detail', '')}".strip())

def run_pending_rows(pending_rows, worker_args, worker_count, args):
    if not pending_rows:
        return []
    multiprocessing_context = get_context("spawn")
    results = []
    with Manager() as manager:
        progress_queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=multiprocessing_context) as executor:
            future_map = {executor.submit(run_star_task, (row, worker_args, progress_queue)): row for row in pending_rows}
            pending = set(future_map)
            unit_counts = {}
            units_per_star = star_progress_units(args)
            total_work_units = len(pending_rows) * star_progress_units(args)
            with tqdm(total=len(future_map), desc="Stars", bar_format=TQDM_BAR_FORMAT, position=0) as star_progress, tqdm(total=total_work_units, desc="Star work", bar_format=TQDM_BAR_FORMAT, position=1) as work_progress:
                while pending:
                    done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                    drain_progress_queue(progress_queue, work_progress, unit_counts=unit_counts, max_units_per_star=units_per_star)
                    for future in done:
                        drain_progress_queue(progress_queue, work_progress, unit_counts=unit_counts, max_units_per_star=units_per_star)
                        row = future_map[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            target_id = normalize_target_id(row["target_id"])
                            quarter = int(row["quarter"])
                            star_dir = Path(args.output_dir) / "stars" / star_prefix(target_id, quarter)
                            result = {"target_id": target_id, "quarter": quarter, "selection_group": str(row.get("selection_group", "unspecified")), "status": "failed", "star_dir": str(star_dir), "runtime_seconds": np.nan, "error": f"{type(exc).__name__}: {exc}"}
                        results.append(result)
                        key = (normalize_target_id(result["target_id"]), int(result["quarter"]))
                        remaining_units = max(0, units_per_star - int(unit_counts.get(key, 0)))
                        if remaining_units:
                            unit_counts[key] = int(unit_counts.get(key, 0)) + remaining_units
                            work_progress.update(remaining_units)
                        star_progress.set_postfix_str(f"{result['status']} KIC {result['target_id']} Q{result['quarter']}")
                        star_progress.update(1)
                drain_progress_queue(progress_queue, work_progress, unit_counts=unit_counts, max_units_per_star=units_per_star)
    return results

def main(args=None):
    args = args or default_settings()
    manifest = load_manifest(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    worker_count = resolve_worker_count(args, len(manifest))
    worker_args = settings_to_worker_dict(args)
    task_results = []
    pending_rows = []
    for row in manifest.to_dict("records"):
        if args.resume and star_is_complete(args.output_dir, row["target_id"], row["quarter"], args):
            task_results.append(completed_result(args.output_dir, row))
        elif args.resume and not args.rerun_failures and star_has_failure(args.output_dir, row["target_id"], row["quarter"]) and not existing_failure_is_transient(args.output_dir, row["target_id"], row["quarter"]):
            task_results.append(failed_result(args.output_dir, row))
        else:
            pending_rows.append(row)
    print(f"Targets requested: {len(manifest)}")
    print(f"Targets resumed: {len(task_results)}")
    print(f"Targets to run: {len(pending_rows)}")
    print(f"Rerun existing failures: {args.rerun_failures}")
    print(f"Parallel star workers: {worker_count}")
    print(f"Profile: {args.profile}")
    print(f"ARIMA injection mode: {args.arima_injection_mode}")
    print(f"Injections per star: {injection_case_count(args)}")
    print(f"Null trials per star: {args.null_trials_per_star}")
    print(f"Period grid: {args.n_periods} requested, {args.n_coarse_periods} coarse")
    print(f"BLS oversample: {args.bls_oversample}")
    print(f"Light-curve cache: {'on' if args.cache_light_curves else 'off'} ({args.cache_dir})")
    print(f"Batch: {args.batch_index}/{args.batch_count}")
    if pending_rows:
        task_results.extend(run_pending_rows(pending_rows, worker_args, worker_count, args))
    injections, null_trials, star_summaries = load_completed_outputs(task_results)
    if injections.empty or null_trials.empty or star_summaries.empty:
        raise RuntimeError("No complete successful star outputs were produced.")
    injections, star_summaries, tcf_threshold, bls_threshold, tcf_null, bls_null = apply_global_calibration(injections, null_trials, star_summaries, args)
    injections, star_summaries = add_noise_regimes(injections, star_summaries)
    detector_summary = detector_metrics(injections)
    grouped_tables = {"depth": grouped_comparison(injections, "injected_depth"), "duration": grouped_comparison(injections, "injected_duration_hours"), "period": grouped_comparison(injections, "injected_period_days"), "epoch": grouped_comparison(injections, "epoch_phase_fraction"), "selection_group": grouped_comparison(injections, "selection_group"), "noise_quartile": grouped_comparison(injections, "noise_quartile"), "star": per_star_comparison(injections)}
    rank_table = pd.concat([rank_distribution(injections, "tcf"), rank_distribution(injections, "bls")], ignore_index=True)
    summary = build_global_summary(manifest, task_results, injections, null_trials, star_summaries, detector_summary, tcf_threshold, bls_threshold, args)
    metrics_dir = save_global_outputs(args.output_dir, manifest, task_results, injections, null_trials, star_summaries, detector_summary, grouped_tables, rank_table, summary)
    print(f"\nMetrics directory: {metrics_dir}")
    print(f"Successful stars: {summary['successful_target_count']}/{summary['requested_target_count']}")
    print(f"Successful injections: {summary['successful_injection_count']}")
    print(f"Pooled TCF null maxima: {summary['pooled_tcf_null_count']}")
    print(f"Pooled BLS null maxima: {summary['pooled_bls_null_count']}")
    print(f"TCF global 1% FAP threshold: {summary['tcf_global_score_threshold']:.6f}")
    print(f"BLS global 1% FAP threshold: {summary['bls_global_sde_threshold']:.6f}")
    print("\nRank-1 recovery before FAP:\n")
    print(f"TCF harmonic-aware: {summary['tcf_harmonic_rank1_rate_before_fap']:.3f}")
    print(f"BLS harmonic-aware: {summary['bls_harmonic_rank1_rate_before_fap']:.3f}")
    print(f"Combined harmonic-aware: {summary['harmonic_union_rank1_rate_before_fap']:.3f}")
    print(f"TCF exact: {summary['tcf_exact_rank1_rate_before_fap']:.3f}")
    print(f"BLS exact: {summary['bls_exact_rank1_rate_before_fap']:.3f}")
    print(f"Combined exact: {summary['exact_union_rank1_rate_before_fap']:.3f}")
    print(f"TCF exact period in top 10: {summary['tcf_exact_top10_rate']:.3f}")
    print(f"BLS exact period in top 10: {summary['bls_exact_top10_rate']:.3f}")
    print("\nRecovery at pooled 1% FAP:\n")
    print(f"TCF harmonic-aware: {summary['tcf_harmonic_recovery_rate_fap']:.3f}")
    print(f"BLS harmonic-aware: {summary['bls_harmonic_recovery_rate_fap']:.3f}")
    print(f"Combined harmonic-aware: {summary['harmonic_union_recovery_rate_fap']:.3f}")
    print(f"TCF exact: {summary['tcf_exact_recovery_rate_fap']:.3f}")
    print(f"BLS exact: {summary['bls_exact_recovery_rate_fap']:.3f}")
    print(f"Combined exact: {summary['exact_union_recovery_rate_fap']:.3f}")
    print("\nOriginal-light-curve false-positive fractions:\n")
    print(f"TCF: {summary['original_tcf_global_false_positive_fraction']:.3f}")
    print(f"BLS: {summary['original_bls_global_false_positive_fraction']:.3f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
