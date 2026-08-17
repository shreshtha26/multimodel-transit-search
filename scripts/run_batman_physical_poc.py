"""Run the physical-transit proof of concept.

New scientific axes relative to the existing challenger benchmark:
1. BATMAN limb-darkened, exposure-integrated transit truth instead of boxes.
2. Explicit signal-transfer measurement by differencing each background branch
   with and without the same injected signal.
3. TLS plus a BLS-seeded trapezoid morphology refiner alongside BLS.

This runner deliberately does NOT claim to implement Kepler TPS. A proper
wavelet-whitened TPS-like comparator should be a separate validated milestone.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from tqdm.auto import tqdm

from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.detection.bls import default_duration_grid, run_bls
from adaptive_transit.detection.tcf import fit_arima_innovations
from adaptive_transit.detection.tls import run_tls
from adaptive_transit.detection.trapezoid import run_bls_seeded_trapezoid
from adaptive_transit.injections.batman import inject_batman_transit
from adaptive_transit.noise_models.gp import (
    apply_prepared_smooth_gp_filter,
    fit_smooth_gp_background,
    prepare_smooth_gp_filter,
)
from adaptive_transit.noise_models.kalman import (
    apply_fitted_kalman_filter,
    fit_kalman_local_level,
)
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs/kepler_clean_background_manifest_10star.csv"
DEFAULT_CASE_FILE = PROJECT_ROOT / "configs/batman_physical_poc_cases.csv"
DEFAULT_CACHE = PROJECT_ROOT / "outputs/cache/kepler_light_curves"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/experiments/batman_physical_detection_poc"


def normalize_target_id(value) -> str:
    return str(value).upper().replace("KIC", "").strip()


def star_prefix(target_id, quarter) -> str:
    return f"kic_{normalize_target_id(target_id)}_q{int(quarter)}"


def parse_csv_names(value: str, allowed: set[str]) -> tuple[str, ...]:
    items = tuple(item.strip().lower() for item in str(value).split(",") if item.strip())
    unknown = set(items).difference(allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown values: {sorted(unknown)}; allowed={sorted(allowed)}")
    if not items:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return items


def parse_arima_order(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part.strip()) for part in str(value).split(","))
    if len(parts) != 3 or any(part < 0 for part in parts):
        raise argparse.ArgumentTypeError("ARIMA order must be non-negative p,d,q.")
    return parts


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--target-limit", type=int, default=10)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument(
        "--branches",
        type=lambda x: parse_csv_names(x, {"raw", "arima", "kalman", "gp"}),
        default=("raw", "arima", "kalman", "gp"),
    )
    parser.add_argument(
        "--detectors",
        type=lambda x: parse_csv_names(x, {"bls", "trapezoid", "tls"}),
        default=("bls", "trapezoid", "tls"),
    )
    parser.add_argument("--quality-policy", default="default")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-resume", action="store_true")

    parser.add_argument("--arima-order", type=parse_arima_order, default=(1, 1, 0))
    parser.add_argument("--arima-maxiter", type=int, default=200)
    parser.add_argument("--kalman-maxiter", type=int, default=100)
    parser.add_argument("--kalman-burn-in", type=int, default=1)
    parser.add_argument("--gp-max-train-points", type=int, default=512)
    parser.add_argument("--gp-length-scale-days", type=float, default=3.0)
    parser.add_argument("--gp-min-length-scale-days", type=float, default=1.0)
    parser.add_argument("--gp-max-length-scale-days", type=float, default=30.0)
    parser.add_argument("--gp-measurement-noise-fraction", type=float, default=0.20)
    parser.add_argument("--gp-fixed-kernel", action="store_true")

    parser.add_argument("--impact-parameter", type=float, default=0.30)
    parser.add_argument("--limb-u1", type=float, default=0.30)
    parser.add_argument("--limb-u2", type=float, default=0.20)
    parser.add_argument("--supersample-factor", type=int, default=7)

    parser.add_argument("--min-period-days", type=float, default=1.0)
    parser.add_argument("--max-period-days", type=float, default=15.0)
    parser.add_argument("--n-periods", type=int, default=1200)
    parser.add_argument("--min-duration-hours", type=float, default=1.5)
    parser.add_argument("--max-duration-hours", type=float, default=10.0)
    parser.add_argument("--n-durations", type=int, default=8)
    parser.add_argument("--bls-top-k", type=int, default=5)
    parser.add_argument("--tls-use-threads", type=int, default=1)
    parser.add_argument("--tls-oversampling-factor", type=int, default=2)
    parser.add_argument("--period-match-tolerance-fraction", type=float, default=0.02)
    return parser.parse_args(argv)


def light_curve_cache_path(cache_dir: Path, target_id, quarter) -> Path:
    return Path(cache_dir) / f"{star_prefix(target_id, quarter)}_pdcsap.parquet"


def load_frame(target_id, quarter, args):
    path = light_curve_cache_path(args.cache_dir, target_id, quarter)
    if path.exists():
        return pd.read_parquet(path), True
    if args.no_download:
        raise FileNotFoundError(f"Missing cache: {path}")
    frame = load_kepler_pdcsap(target_id, quarter).to_dataframe()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame, False


def load_manifest(path: Path, target_limit: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"target_id", "quarter"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest must contain {sorted(required)}.")
    frame = frame.copy()
    frame["target_id"] = frame["target_id"].map(normalize_target_id)
    frame["quarter"] = pd.to_numeric(frame["quarter"], errors="raise").astype(int)
    if "sample_stratum" not in frame:
        frame["sample_stratum"] = "unspecified"
    if "selection_group" not in frame:
        frame["selection_group"] = "unspecified"
    return frame.drop_duplicates(["target_id", "quarter"]).head(int(target_limit)).reset_index(drop=True)


def load_cases(path: Path, case_limit: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"period_days", "duration_hours", "depth", "phase_fraction"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Case file must contain {sorted(required)}.")
    frame = frame.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (frame["period_days"] <= 0).any() or (frame["duration_hours"] <= 0).any():
        raise ValueError("Periods and durations must be positive.")
    if ((frame["depth"] <= 0) | (frame["depth"] >= 0.25)).any():
        raise ValueError("Depths must lie in (0, 0.25).")
    if case_limit is not None:
        frame = frame.head(int(case_limit))
    frame = frame.reset_index(drop=True)
    frame["case_index"] = np.arange(len(frame), dtype=int)
    return frame


def robust_scale(values) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    center = float(np.median(x))
    scale = float(1.4826 * np.median(np.abs(x - center)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(x, ddof=1))
    return scale


def acf1(values) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    pair = np.isfinite(x[:-1]) & np.isfinite(x[1:])
    if pair.sum() < 3:
        return float("nan")
    first = x[:-1][pair]
    second = x[1:][pair]
    if np.std(first) <= 0 or np.std(second) <= 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def filter_fixed_arima(values, base_arima, order):
    params = np.asarray(base_arima["params"], dtype=float)
    series = np.asarray(values, dtype=float).reshape(-1)
    model = ARIMA(
        series,
        order=tuple(order),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.filter(params)
    residuals = np.asarray(fitted.resid, dtype=float).reshape(-1)
    residuals[~np.isfinite(series)] = np.nan
    burn = int(getattr(fitted, "loglikelihood_burn", 0))
    if burn > 0:
        residuals[:burn] = np.nan
    return residuals


def fit_base_branches(time, flux, args):
    base = {}
    metadata = {}

    if "raw" in args.branches:
        base["raw"] = np.asarray(flux, dtype=float)
        metadata["raw"] = {"converged": True}

    if "arima" in args.branches:
        started = perf_counter()
        try:
            fitted = fit_arima_innovations(
                flux, order=args.arima_order, maxiter=args.arima_maxiter
            )
            params = np.asarray(fitted["fit"].params, dtype=float)
            fitted["params"] = params
            base["arima"] = np.asarray(fitted["innovations"], dtype=float)
            metadata["arima"] = {
                "converged": bool(fitted["summary"].get("converged", True)),
                "aic": float(fitted["summary"].get("aic", np.nan)),
                "runtime_seconds": float(perf_counter() - started),
                "params_json": json.dumps(params.tolist()),
            }
            metadata["_arima_model"] = fitted
        except Exception as exc:
            metadata["arima"] = {"converged": False, "error": f"{type(exc).__name__}: {exc}"}

    if "kalman" in args.branches:
        started = perf_counter()
        try:
            fitted = fit_kalman_local_level(
                flux, maxiter=args.kalman_maxiter, burn_in=args.kalman_burn_in
            )
            base["kalman"] = np.asarray(fitted.residuals, dtype=float)
            metadata["kalman"] = {
                "converged": bool(fitted.converged),
                "runtime_seconds": float(perf_counter() - started),
                "process_variance": float(fitted.parameters["process_variance"]),
                "measurement_variance": float(fitted.parameters["measurement_variance"]),
            }
            metadata["_kalman_model"] = fitted
        except Exception as exc:
            metadata["kalman"] = {"converged": False, "error": f"{type(exc).__name__}: {exc}"}

    if "gp" in args.branches:
        started = perf_counter()
        try:
            fitted = fit_smooth_gp_background(
                time,
                flux,
                max_train_points=args.gp_max_train_points,
                length_scale_days=args.gp_length_scale_days,
                min_length_scale_days=args.gp_min_length_scale_days,
                max_length_scale_days=args.gp_max_length_scale_days,
                measurement_noise_fraction=args.gp_measurement_noise_fraction,
                n_restarts_optimizer=0,
                random_seed=123,
                optimize_kernel=not args.gp_fixed_kernel,
            )
            prepared = prepare_smooth_gp_filter(time, fitted)
            base["gp"] = np.asarray(fitted.residuals, dtype=float)
            gp_params = fitted.parameters
            gp_boundary_limited = bool(
                gp_params.get("length_scale_at_lower_bound", False)
            )
            metadata["gp"] = {
                # Keep the library-level optimizer convergence flag verbatim,
                # but also persist the scientifically meaningful constrained
                # status used by downstream QC.
                "converged": bool(fitted.converged),
                "fit_status": (
                    "boundary_limited"
                    if gp_boundary_limited
                    else "interior_optimum"
                    if fitted.converged
                    else "optimizer_flagged"
                ),
                "fit_usable": bool(fitted.converged or gp_boundary_limited),
                "runtime_seconds": float(perf_counter() - started),
                "length_scale_days": float(gp_params["length_scale_days"]),
                "length_scale_at_lower_bound": gp_boundary_limited,
                "length_scale_at_upper_bound": bool(
                    gp_params.get("length_scale_at_upper_bound", False)
                ),
                "gp_min_length_scale_days": float(args.gp_min_length_scale_days),
                "optimizer_warning_count": int(
                    gp_params.get("optimizer_warning_count", 0)
                ),
                "optimizer_warning_message": str(
                    gp_params.get("optimizer_warning_message", "")
                ),
                "training_point_count": int(gp_params["training_point_count"]),
            }
            metadata["_gp_model"] = fitted
            metadata["_gp_prepared"] = prepared
        except Exception as exc:
            metadata["gp"] = {"converged": False, "error": f"{type(exc).__name__}: {exc}"}

    return base, metadata


def apply_branch(branch, injected_flux, base_metadata, args):
    if branch == "raw":
        return np.asarray(injected_flux, dtype=float)
    if branch == "arima":
        model = base_metadata.get("_arima_model")
        if model is None:
            raise ValueError("Base ARIMA fit unavailable.")
        return filter_fixed_arima(injected_flux, model, args.arima_order)
    if branch == "kalman":
        model = base_metadata.get("_kalman_model")
        if model is None:
            raise ValueError("Base Kalman fit unavailable.")
        return np.asarray(
            apply_fitted_kalman_filter(
                injected_flux, model, burn_in=args.kalman_burn_in
            ).residuals,
            dtype=float,
        )
    if branch == "gp":
        prepared = base_metadata.get("_gp_prepared")
        if prepared is None:
            raise ValueError("Prepared GP filter unavailable.")
        return np.asarray(
            apply_prepared_smooth_gp_filter(injected_flux, prepared).residuals,
            dtype=float,
        )
    raise ValueError(f"Unknown branch: {branch}")


def signal_transfer_metrics(original_template, transmitted_template, base_series):
    truth = np.asarray(original_template, dtype=float).reshape(-1)
    transmitted = np.asarray(transmitted_template, dtype=float).reshape(-1)
    background = np.asarray(base_series, dtype=float).reshape(-1)
    finite = np.isfinite(truth) & np.isfinite(transmitted)
    active = finite & (truth < -max(1.0e-12, 1.0e-6 * np.nanmax(np.abs(truth[finite]))))
    if finite.sum() < 20 or active.sum() < 2:
        return {
            "template_amplitude_ratio": np.nan,
            "peak_depth_ratio": np.nan,
            "template_energy_ratio": np.nan,
            "template_correlation": np.nan,
            "template_rmse_ppm": np.nan,
            "oracle_signal_snr": np.nan,
            "background_scale_ppm": np.nan,
            "background_acf1": np.nan,
        }

    denominator = float(np.dot(truth[finite], truth[finite]))
    amplitude = (
        float(np.dot(transmitted[finite], truth[finite]) / denominator)
        if denominator > 0
        else np.nan
    )
    truth_peak = float(np.max(-truth[active]))
    transmitted_peak = float(np.max(-transmitted[active]))
    peak_ratio = transmitted_peak / truth_peak if truth_peak > 0 else np.nan
    truth_energy = float(np.linalg.norm(truth[finite]))
    transmitted_energy = float(np.linalg.norm(transmitted[finite]))
    energy_ratio = transmitted_energy / truth_energy if truth_energy > 0 else np.nan

    if np.std(truth[active]) > 0 and np.std(transmitted[active]) > 0:
        correlation = float(np.corrcoef(truth[active], transmitted[active])[0, 1])
    else:
        correlation = np.nan

    rmse_ppm = float(
        1.0e6 * np.sqrt(np.mean((transmitted[finite] - truth[finite]) ** 2))
    )
    background_scale = robust_scale(background)
    oracle_snr = (
        float(transmitted_energy / background_scale)
        if np.isfinite(background_scale) and background_scale > 0
        else np.nan
    )
    return {
        "template_amplitude_ratio": amplitude,
        "peak_depth_ratio": peak_ratio,
        "template_energy_ratio": energy_ratio,
        "template_correlation": correlation,
        "template_rmse_ppm": rmse_ppm,
        "oracle_signal_snr": oracle_snr,
        "background_scale_ppm": float(background_scale * 1.0e6),
        "background_acf1": acf1(background),
    }


def period_match(recovered, injected, tolerance):
    recovered = float(recovered)
    injected = float(injected)
    if not np.isfinite(recovered) or injected <= 0:
        return np.nan, False, False
    exact_error = abs(recovered - injected) / injected
    harmonic_errors = [
        abs(recovered - factor * injected) / (factor * injected)
        for factor in (0.5, 1.0, 2.0)
    ]
    harmonic_error = float(min(harmonic_errors))
    return exact_error, bool(exact_error <= tolerance), bool(harmonic_error <= tolerance)


def detector_row(
    *,
    detector,
    branch,
    target_id,
    quarter,
    sample_stratum,
    case,
    injected_period,
    result,
    tolerance,
    runtime,
):
    if detector == "bls":
        summary = result["summary"]
        period = float(summary["period"])
        duration = float(summary["duration"])
        epoch = float(summary["transit_time"])
        score = float(summary["power"])
        depth = float(summary["depth"])
        extra = {"seed_source": ""}
    elif detector == "trapezoid":
        summary = result["summary"]
        period = float(summary["period_days"])
        duration = float(summary["duration_days"])
        epoch = float(summary["epoch_days"])
        score = float(summary["score"])
        depth = float(summary["depth"])
        extra = {
            "seed_source": "bls_top_peaks",
            "ingress_fraction": float(summary["ingress_fraction"]),
            "bls_seed_rank": int(summary["seed_rank"]),
        }
    elif detector == "tls":
        summary = result["summary"]
        period = float(summary["period_days"])
        duration = float(summary["duration_days"])
        epoch = float(summary["epoch_days"])
        score = float(summary["sde"])
        depth = float(summary["depth_raw"])
        extra = {"seed_source": "", "tls_snr": float(summary["snr"])}
    else:
        raise ValueError(detector)
    exact_error, exact_match, harmonic_match = period_match(period, injected_period, tolerance)
    return {
        "target_id": normalize_target_id(target_id),
        "quarter": int(quarter),
        "sample_stratum": str(sample_stratum),
        "case_index": int(case["case_index"]),
        "branch": branch,
        "detector": detector,
        "pipeline": f"{branch}_{detector}",
        "success": True,
        "recovered_period_days": period,
        "recovered_duration_hours": duration * 24.0,
        "recovered_epoch_days": epoch,
        "recovered_depth_raw": depth,
        "score": score,
        "period_exact_fractional_error": exact_error,
        "exact_period_recovered": exact_match,
        "harmonic_period_recovered": harmonic_match,
        "runtime_seconds": float(runtime),
        "error": "",
        **extra,
    }


def run_detectors(time, branch_series, args, injected_period, common, case):
    period_grid = np.linspace(
        float(args.min_period_days), float(args.max_period_days), int(args.n_periods)
    )
    duration_grid = default_duration_grid(
        args.min_duration_hours, args.max_duration_hours, args.n_durations
    )
    rows = []
    bls_result = None

    if "bls" in args.detectors or "trapezoid" in args.detectors:
        try:
            started = perf_counter()
            bls_result = run_bls(
                time,
                branch_series,
                period_grid=period_grid,
                duration_grid=duration_grid,
                objective="snr",
                top_k=args.bls_top_k,
            )
            if "bls" in args.detectors:
                rows.append(
                    detector_row(
                        detector="bls",
                        branch=common["branch"],
                        target_id=common["target_id"],
                        quarter=common["quarter"],
                        sample_stratum=common["sample_stratum"],
                        case=case,
                        injected_period=injected_period,
                        result=bls_result,
                        tolerance=args.period_match_tolerance_fraction,
                        runtime=perf_counter() - started,
                    )
                )
        except Exception as exc:
            if "bls" in args.detectors:
                rows.append(
                    {
                        **common,
                        "case_index": int(case["case_index"]),
                        "detector": "bls",
                        "pipeline": f"{common['branch']}_bls",
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    if "trapezoid" in args.detectors:
        try:
            if bls_result is None:
                raise ValueError("BLS seeds unavailable.")
            started = perf_counter()
            result = run_bls_seeded_trapezoid(
                time,
                branch_series,
                bls_result,
                duration_grid=duration_grid,
                top_k_periods=args.bls_top_k,
            )
            rows.append(
                detector_row(
                    detector="trapezoid",
                    branch=common["branch"],
                    target_id=common["target_id"],
                    quarter=common["quarter"],
                    sample_stratum=common["sample_stratum"],
                    case=case,
                    injected_period=injected_period,
                    result=result,
                    tolerance=args.period_match_tolerance_fraction,
                    runtime=perf_counter() - started,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    **common,
                    "case_index": int(case["case_index"]),
                    "detector": "trapezoid",
                    "pipeline": f"{common['branch']}_trapezoid",
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "seed_source": "bls_top_peaks",
                }
            )

    if "tls" in args.detectors:
        try:
            started = perf_counter()
            result = run_tls(
                time,
                branch_series,
                period_min=args.min_period_days,
                period_max=args.max_period_days,
                use_threads=args.tls_use_threads,
                oversampling_factor=args.tls_oversampling_factor,
            )
            rows.append(
                detector_row(
                    detector="tls",
                    branch=common["branch"],
                    target_id=common["target_id"],
                    quarter=common["quarter"],
                    sample_stratum=common["sample_stratum"],
                    case=case,
                    injected_period=injected_period,
                    result=result,
                    tolerance=args.period_match_tolerance_fraction,
                    runtime=perf_counter() - started,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    **common,
                    "case_index": int(case["case_index"]),
                    "detector": "tls",
                    "pipeline": f"{common['branch']}_tls",
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def jsonable_args(args):
    payload = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = list(value)
        else:
            payload[key] = value
    return payload


def write_summary(retention: pd.DataFrame, detectors: pd.DataFrame, output_dir: Path):
    if not retention.empty:
        summary = (
            retention.groupby("branch", dropna=False)
            .agg(
                n_cases=("case_index", "size"),
                median_template_amplitude_ratio=("template_amplitude_ratio", "median"),
                median_peak_depth_ratio=("peak_depth_ratio", "median"),
                median_template_energy_ratio=("template_energy_ratio", "median"),
                median_template_correlation=("template_correlation", "median"),
                median_template_rmse_ppm=("template_rmse_ppm", "median"),
                median_oracle_signal_snr=("oracle_signal_snr", "median"),
                median_background_scale_ppm=("background_scale_ppm", "median"),
                median_abs_background_acf1=("background_acf1", lambda x: np.nanmedian(np.abs(x))),
            )
            .reset_index()
        )
        raw_snr = summary.loc[summary["branch"] == "raw", "median_oracle_signal_snr"]
        reference = float(raw_snr.iloc[0]) if len(raw_snr) else np.nan
        summary["oracle_snr_gain_vs_raw"] = summary["median_oracle_signal_snr"] / reference
        summary.to_csv(output_dir / "summary_by_branch.csv", index=False)

    if not detectors.empty:
        valid = detectors.loc[detectors["success"].fillna(False)].copy()
        if not valid.empty:
            summary = (
                valid.groupby(["branch", "detector"], dropna=False)
                .agg(
                    n_successful_runs=("case_index", "size"),
                    exact_period_recovery=("exact_period_recovered", "mean"),
                    harmonic_period_recovery=("harmonic_period_recovered", "mean"),
                    median_period_error=("period_exact_fractional_error", "median"),
                    median_runtime_seconds=("runtime_seconds", "median"),
                )
                .reset_index()
            )
            summary.to_csv(output_dir / "summary_branch_detector.csv", index=False)


def process_star(manifest_row, cases, args):
    target_id = normalize_target_id(manifest_row["target_id"])
    quarter = int(manifest_row["quarter"])
    stratum = str(manifest_row.get("sample_stratum", "unspecified"))
    star_dir = Path(args.output_dir) / "stars" / star_prefix(target_id, quarter)
    retention_path = star_dir / "retention.csv"
    detector_path = star_dir / "detectors.csv"
    base_path = star_dir / "base_models.json"

    if (
        not args.no_resume
        and retention_path.exists()
        and detector_path.exists()
        and base_path.exists()
    ):
        return pd.read_csv(retention_path), pd.read_csv(detector_path), json.loads(base_path.read_text())

    star_dir.mkdir(parents=True, exist_ok=True)
    frame, from_cache = load_frame(target_id, quarter, args)
    regular, prep_summary = preprocess_pdcsap_light_curve(
        frame,
        quality_policy=args.quality_policy,
        require_finite_flux_error=False,
        normalization_fit_fraction=1.0,
    )
    time = regular["time"].to_numpy(dtype=float)
    flux = regular["normalized_flux"].to_numpy(dtype=float)

    base_series, base_metadata = fit_base_branches(time, flux, args)
    serializable_base = {
        key: value
        for key, value in base_metadata.items()
        if not key.startswith("_")
    }
    base_payload = {
        "target_id": target_id,
        "quarter": quarter,
        "sample_stratum": stratum,
        "from_light_curve_cache": bool(from_cache),
        "preprocessing": prep_summary.to_dict(),
        "models": serializable_base,
    }
    base_path.write_text(json.dumps(base_payload, indent=2, default=str) + "\n")

    retention_rows = []
    detector_rows = []
    finite = np.isfinite(time) & np.isfinite(flux)
    t_min = float(np.min(time[finite]))

    for _, case in cases.iterrows():
        period = float(case["period_days"])
        duration_days = float(case["duration_hours"]) / 24.0
        depth = float(case["depth"])
        phase = float(case["phase_fraction"])
        epoch = t_min + phase * period

        injected, template, in_transit, truth = inject_batman_transit(
            time,
            flux,
            period_days=period,
            epoch_days=epoch,
            duration_days=duration_days,
            depth=depth,
            impact_parameter=args.impact_parameter,
            limb_darkening_coefficients=(args.limb_u1, args.limb_u2),
            supersample_factor=args.supersample_factor,
        )

        for branch in args.branches:
            common = {
                "target_id": target_id,
                "quarter": quarter,
                "sample_stratum": stratum,
                "branch": branch,
            }
            if branch not in base_series:
                error = serializable_base.get(branch, {}).get("error", "base branch unavailable")
                retention_rows.append(
                    {
                        **common,
                        "case_index": int(case["case_index"]),
                        "success": False,
                        "error": error,
                    }
                )
                continue
            try:
                processed_injected = apply_branch(branch, injected, base_metadata, args)
                processed_base = np.asarray(base_series[branch], dtype=float)
                transmitted = processed_injected - processed_base
                metrics = signal_transfer_metrics(template, transmitted, processed_base)
                retention_rows.append(
                    {
                        **common,
                        "case_index": int(case["case_index"]),
                        "success": True,
                        "error": "",
                        "injected_period_days": period,
                        "injected_epoch_days": epoch,
                        "requested_duration_hours": float(case["duration_hours"]),
                        "requested_depth": depth,
                        "phase_fraction": phase,
                        "in_transit_cadence_count": int(np.sum(in_transit & np.isfinite(flux))),
                        "radius_ratio": truth.radius_ratio,
                        "scaled_semimajor_axis": truth.scaled_semimajor_axis,
                        "inclination_degrees": truth.inclination_degrees,
                        "impact_parameter": truth.impact_parameter,
                        "limb_darkening_u1": truth.limb_darkening_u1,
                        "limb_darkening_u2": truth.limb_darkening_u2,
                        "exposure_time_minutes": truth.exposure_time_days * 24.0 * 60.0,
                        "supersample_factor": truth.supersample_factor,
                        "realized_max_depth_on_observed_cadences": truth.realized_max_depth_on_observed_cadences,
                        **metrics,
                    }
                )
                detector_rows.extend(
                    run_detectors(
                        time,
                        processed_injected,
                        args,
                        period,
                        common,
                        case,
                    )
                )
            except Exception as exc:
                retention_rows.append(
                    {
                        **common,
                        "case_index": int(case["case_index"]),
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    retention = pd.DataFrame(retention_rows)
    detectors = pd.DataFrame(detector_rows)
    retention.to_csv(retention_path, index=False)
    detectors.to_csv(detector_path, index=False)
    return retention, detectors, base_payload


def main(argv=None):
    args = parse_args(argv)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(jsonable_args(args), indent=2) + "\n"
    )

    manifest = load_manifest(args.manifest_path, args.target_limit)
    cases = load_cases(args.case_file, args.case_limit)
    manifest.to_csv(args.output_dir / "manifest_used.csv", index=False)
    cases.to_csv(args.output_dir / "cases_used.csv", index=False)

    retention_frames = []
    detector_frames = []
    base_rows = []
    for row in tqdm(manifest.to_dict(orient="records"), desc="BATMAN physical POC"):
        try:
            retention, detectors, base_payload = process_star(row, cases, args)
            retention_frames.append(retention)
            detector_frames.append(detectors)
            for branch, metadata in base_payload.get("models", {}).items():
                base_rows.append(
                    {
                        "target_id": base_payload["target_id"],
                        "quarter": base_payload["quarter"],
                        "sample_stratum": base_payload["sample_stratum"],
                        "branch": branch,
                        **metadata,
                    }
                )
        except Exception as exc:
            base_rows.append(
                {
                    "target_id": normalize_target_id(row["target_id"]),
                    "quarter": int(row["quarter"]),
                    "sample_stratum": str(row.get("sample_stratum", "unspecified")),
                    "branch": "star",
                    "converged": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        if retention_frames:
            pd.concat(retention_frames, ignore_index=True).to_csv(
                args.output_dir / "physical_retention.csv", index=False
            )
        if detector_frames:
            pd.concat(detector_frames, ignore_index=True).to_csv(
                args.output_dir / "detector_results.csv", index=False
            )
        pd.DataFrame(base_rows).to_csv(args.output_dir / "base_models.csv", index=False)

    retention = (
        pd.concat(retention_frames, ignore_index=True)
        if retention_frames
        else pd.DataFrame()
    )
    detectors = (
        pd.concat(detector_frames, ignore_index=True)
        if detector_frames
        else pd.DataFrame()
    )
    write_summary(retention, detectors, args.output_dir)

    print("\nPhysical POC complete.")
    print(f"Retention rows: {len(retention)}")
    print(f"Detector rows: {len(detectors)}")
    print(f"Output: {args.output_dir}")
    print(
        "\nInterpret retention first (signal transfer + background suppression), "
        "then detector recovery. Trapezoid is BLS-seeded and is not an independent "
        "period-search method. No TPS claim is made by this runner."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
