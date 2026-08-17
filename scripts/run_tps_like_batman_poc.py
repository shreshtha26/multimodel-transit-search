#!/usr/bin/env python3
"""Run the TPS-like integrated comparator on the existing BATMAN POC cases.

This runner adds one new pipeline only:

    BATMAN-injected PDCSAP -> adaptive SWT whitening -> matched pulse bank
    -> SES-like single-event statistics -> MES-like periodic combination.

It deliberately does NOT place Raw/ARIMA/Kalman/GP in front of the TPS-like
search because the comparator already contains its own adaptive whitening.
Existing BLS/TLS results are read from the completed physical POC for a common
period-recovery summary; no BLS/TLS rerun is required.

The output is proof-of-concept recovery only.  It is not FAP calibrated and it
must be labelled ``TPS-like`` rather than ``TPS``.
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
from tqdm.auto import tqdm

from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.detection.tps_like import (
    prepare_tps_like_noise_model,
    run_tps_like_search,
)
from adaptive_transit.injections.batman import inject_batman_transit
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs/kepler_clean_background_manifest_10star.csv"
DEFAULT_CASE_FILE = PROJECT_ROOT / "configs/batman_physical_poc_cases.csv"
DEFAULT_CACHE = PROJECT_ROOT / "outputs/cache/kepler_light_curves"
DEFAULT_PHYSICAL_POC = (
    PROJECT_ROOT / "outputs/experiments/batman_physical_detection_poc/pilot10"
)
DEFAULT_OUTPUT = DEFAULT_PHYSICAL_POC / "tps_like_comparator"


def normalize_target_id(value) -> str:
    return str(value).upper().replace("KIC", "").strip()


def star_prefix(target_id, quarter) -> str:
    return f"kic_{normalize_target_id(target_id)}_q{int(quarter)}"


def parse_float_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in str(text).split(",") if x.strip())
    if not values or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError("Duration grid must contain positive numbers.")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--physical-poc-dir", type=Path, default=DEFAULT_PHYSICAL_POC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-limit", type=int, default=10)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--quality-policy", default="default")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--save-periodograms", action="store_true")
    parser.add_argument(
        "--zero-injection",
        action="store_true",
        help=(
            "Run one true zero-injection control per star: skip BATMAN entirely "
            "and search the original preprocessed PDCSAP flux with the same "
            "prepared TPS-like noise model and search settings."
        ),
    )

    parser.add_argument("--wavelet", default="db6")
    parser.add_argument("--max-wavelet-level", type=int, default=6)
    parser.add_argument("--noise-window-cadences", type=int, default=193)
    parser.add_argument("--min-segment-cadences", type=int, default=32)
    parser.add_argument("--min-events", type=int, default=3)
    parser.add_argument("--min-period-days", type=float, default=1.0)
    parser.add_argument("--max-period-days", type=float, default=15.0)
    parser.add_argument(
        "--duration-hours-grid",
        type=parse_float_grid,
        default=(1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0),
    )
    parser.add_argument("--period-match-tolerance-fraction", type=float, default=0.02)

    parser.add_argument("--impact-parameter", type=float, default=0.30)
    parser.add_argument("--limb-u1", type=float, default=0.30)
    parser.add_argument("--limb-u2", type=float, default=0.20)
    parser.add_argument("--supersample-factor", type=int, default=7)
    return parser.parse_args(argv)


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
    return (
        frame.drop_duplicates(["target_id", "quarter"])
        .head(int(target_limit))
        .reset_index(drop=True)
    )


def load_cases(path: Path, case_limit: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"period_days", "duration_hours", "depth", "phase_fraction"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Case file must contain {sorted(required)}.")
    frame = frame.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if case_limit is not None:
        frame = frame.head(int(case_limit))
    frame = frame.reset_index(drop=True)
    frame["case_index"] = np.arange(len(frame), dtype=int)
    return frame


def zero_injection_cases() -> pd.DataFrame:
    """Return the single control row used by true zero-injection mode.

    The row exists only so the existing per-star result machinery remains
    one-row-per-case.  No physical injection parameters are consumed when
    ``--zero-injection`` is active.
    """
    return pd.DataFrame(
        [
            {
                "case_index": 0,
                "period_days": np.nan,
                "duration_hours": np.nan,
                "depth": 0.0,
                "phase_fraction": np.nan,
                "case_label": "true_zero_injection_control",
            }
        ]
    )


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


def period_match(recovered: float, injected: float, tolerance: float):
    recovered = float(recovered)
    injected = float(injected)
    if not np.isfinite(recovered) or not np.isfinite(injected) or injected <= 0:
        return np.nan, False, False
    exact_error = abs(recovered - injected) / injected
    harmonic_errors = [
        abs(recovered - factor * injected) / (factor * injected)
        for factor in (0.5, 1.0, 2.0)
    ]
    return (
        float(exact_error),
        bool(exact_error <= float(tolerance)),
        bool(min(harmonic_errors) <= float(tolerance)),
    )


def _noise_model_row(target_id, quarter, stratum, prepared, from_cache):
    levels = [segment.level for segment in prepared.segments]
    lengths = [segment.original_length for segment in prepared.segments]
    scales = [
        scale
        for segment in prepared.segments
        for scale in segment.median_scale_bands
        if np.isfinite(scale)
    ]
    return {
        "target_id": target_id,
        "quarter": int(quarter),
        "sample_stratum": str(stratum),
        "wavelet": prepared.wavelet,
        "segment_count": prepared.segment_count,
        "median_segment_length_cadences": float(np.median(lengths)) if lengths else np.nan,
        "median_wavelet_level": float(np.median(levels)) if levels else np.nan,
        "median_band_noise_scale": float(np.median(scales)) if scales else np.nan,
        "noise_window_cadences": prepared.noise_window_cadences,
        "from_light_curve_cache": bool(from_cache),
    }


def _existing_comparison(existing_path: Path, tps_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if existing_path.exists():
        old = pd.read_csv(existing_path)
        if "success" in old:
            success = old["success"]
            if success.dtype != bool:
                success = success.astype(str).str.lower().isin({"true", "1"})
            old = old.loc[success].copy()
        if not old.empty:
            summary = (
                old.groupby(["branch", "detector"], dropna=False)
                .agg(
                    n_cases=("case_index", "size"),
                    exact_period_recovery=("exact_period_recovered", "mean"),
                    harmonic_period_recovery=("harmonic_period_recovered", "mean"),
                    median_period_error=("period_exact_fractional_error", "median"),
                    median_runtime_seconds=("runtime_seconds", "median"),
                )
                .reset_index()
            )
            summary["pipeline"] = summary["branch"].astype(str) + "_" + summary["detector"].astype(str)
            rows.append(summary[[
                "pipeline", "branch", "detector", "n_cases", "exact_period_recovery",
                "harmonic_period_recovery", "median_period_error", "median_runtime_seconds"
            ]])

    valid = tps_rows.loc[tps_rows["success"].fillna(False)].copy()
    if not valid.empty:
        tps_summary = pd.DataFrame(
            [{
                "pipeline": "tps_like_wavelet_matched_filter",
                "branch": "integrated_wavelet_whitening",
                "detector": "tps_like",
                "n_cases": int(len(valid)),
                "exact_period_recovery": float(valid["exact_period_recovered"].mean()),
                "harmonic_period_recovery": float(valid["harmonic_period_recovered"].mean()),
                "median_period_error": float(valid["period_exact_fractional_error"].median()),
                "median_runtime_seconds": float(valid["runtime_seconds"].median()),
            }]
        )
        rows.append(tps_summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_summaries(results: pd.DataFrame, args):
    valid = results.loc[results["success"].fillna(False)].copy()
    if valid.empty:
        return

    overall = pd.DataFrame(
        [{
            "pipeline": "tps_like_wavelet_matched_filter",
            "n_cases": int(len(valid)),
            "exact_period_recovery": float(valid["exact_period_recovered"].mean()),
            "harmonic_period_recovery": float(valid["harmonic_period_recovered"].mean()),
            "median_period_error": float(valid["period_exact_fractional_error"].median()),
            "median_mes": float(valid["mes"].median()),
            "median_candidate_observability": float(valid["observability_fraction"].median()),
            "median_runtime_seconds": float(valid["runtime_seconds"].median()),
        }]
    )
    overall.to_csv(args.output_dir / "summary_tps_like.csv", index=False)

    by_case = (
        valid.groupby(["injected_period_days", "requested_duration_hours", "requested_depth"], dropna=False)
        .agg(
            n_cases=("case_index", "size"),
            exact_period_recovery=("exact_period_recovered", "mean"),
            harmonic_period_recovery=("harmonic_period_recovered", "mean"),
            median_mes=("mes", "median"),
            median_observability=("observability_fraction", "median"),
        )
        .reset_index()
    )
    by_case.to_csv(args.output_dir / "summary_by_injection_regime.csv", index=False)

    comparison = _existing_comparison(
        Path(args.physical_poc_dir) / "detector_results.csv", results
    )
    if not comparison.empty:
        comparison.to_csv(args.output_dir / "comparison_with_existing_detectors.csv", index=False)


def process_star(row, cases, args):
    target_id = normalize_target_id(row["target_id"])
    quarter = int(row["quarter"])
    stratum = str(row.get("sample_stratum", "unspecified"))
    star_dir = args.output_dir / "stars" / star_prefix(target_id, quarter)
    result_path = star_dir / "tps_like_results.csv"
    noise_path = star_dir / "noise_model.json"

    if not args.no_resume and result_path.exists() and noise_path.exists():
        return pd.read_csv(result_path), json.loads(noise_path.read_text())

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
    segment_id = regular["segment_id"].to_numpy(dtype=int)

    # Important: the prepared noise model is built from the original stellar
    # light curve in both injected and zero-injection modes.  This keeps the
    # whitening/search machinery identical while the control simply bypasses
    # BATMAN and searches ``flux`` itself.
    prepared = prepare_tps_like_noise_model(
        flux,
        segment_id,
        wavelet=args.wavelet,
        max_level=args.max_wavelet_level,
        noise_window_cadences=args.noise_window_cadences,
        min_segment_cadences=args.min_segment_cadences,
    )
    noise_record = _noise_model_row(
        target_id, quarter, stratum, prepared, from_cache
    )
    noise_record["preprocessing"] = prep_summary.to_dict()
    noise_record["zero_injection_control"] = bool(args.zero_injection)
    noise_path.write_text(json.dumps(noise_record, indent=2, default=str) + "\n")

    finite = np.isfinite(time) & np.isfinite(flux)
    t_min = float(np.min(time[finite]))
    rows = []
    for _, case in cases.iterrows():
        if args.zero_injection:
            period = np.nan
            duration_days = np.nan
            depth = 0.0
            phase = np.nan
            epoch = np.nan
        else:
            period = float(case["period_days"])
            duration_days = float(case["duration_hours"]) / 24.0
            depth = float(case["depth"])
            phase = float(case["phase_fraction"])
            epoch = t_min + phase * period

        common = {
            "target_id": target_id,
            "quarter": quarter,
            "sample_stratum": stratum,
            "case_index": int(case["case_index"]),
            "pipeline": "tps_like_wavelet_matched_filter",
            "branch": "integrated_wavelet_whitening",
            "detector": "tps_like",
            "zero_injection_control": bool(args.zero_injection),
            "injected_period_days": period,
            "injected_epoch_days": epoch,
            "requested_duration_hours": (
                np.nan if args.zero_injection else float(case["duration_hours"])
            ),
            "requested_depth": depth,
            "phase_fraction": phase,
        }
        try:
            if args.zero_injection:
                search_flux = flux.copy()
                in_transit = np.zeros_like(finite, dtype=bool)
                truth = None
            else:
                search_flux, _, in_transit, truth = inject_batman_transit(
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

            started = perf_counter()
            result = run_tps_like_search(
                time,
                search_flux,
                segment_id,
                prepared_noise_model=prepared,
                min_period_days=args.min_period_days,
                max_period_days=args.max_period_days,
                duration_hours_grid=args.duration_hours_grid,
                wavelet=args.wavelet,
                max_level=args.max_wavelet_level,
                noise_window_cadences=args.noise_window_cadences,
                min_segment_cadences=args.min_segment_cadences,
                min_events=args.min_events,
            )
            runtime = perf_counter() - started
            summary = result["summary"]

            if args.zero_injection:
                period_error, exact, harmonic = np.nan, False, False
                radius_ratio = np.nan
                realized_depth = 0.0
                in_transit_count = 0
            else:
                period_error, exact, harmonic = period_match(
                    summary["period_days"], period, args.period_match_tolerance_fraction
                )
                radius_ratio = float(truth.radius_ratio)
                realized_depth = float(
                    truth.realized_max_depth_on_observed_cadences
                )
                in_transit_count = int(np.sum(in_transit & finite))

            rows.append(
                {
                    **common,
                    "success": True,
                    "error": "",
                    "recovered_period_days": float(summary["period_days"]),
                    "recovered_epoch_days": float(summary["epoch_days"]),
                    "recovered_duration_hours": float(summary["duration_hours"]),
                    "mes": float(summary["mes"]),
                    "max_ses": float(summary["max_ses"]),
                    "observed_event_count": int(summary["observed_event_count"]),
                    "expected_event_count": int(summary["expected_event_count"]),
                    "observability_fraction": float(summary["observability_fraction"]),
                    "period_cadences": int(summary["period_cadences"]),
                    "duration_cadences": int(summary["duration_cadences"]),
                    "period_exact_fractional_error": period_error,
                    "exact_period_recovered": exact,
                    "harmonic_period_recovered": harmonic,
                    "runtime_seconds": float(runtime),
                    "wavelet": str(summary["wavelet"]),
                    "segment_count": int(summary["segment_count"]),
                    "n_period_trials": int(summary["n_period_trials"]),
                    "n_duration_trials": int(summary["n_duration_trials"]),
                    "radius_ratio": radius_ratio,
                    "realized_max_depth_on_observed_cadences": realized_depth,
                    "in_transit_cadence_count": in_transit_count,
                }
            )
            if args.save_periodograms:
                label = "zero_injection" if args.zero_injection else f"case_{int(case['case_index']):02d}"
                result["periodogram"].to_csv(
                    star_dir / f"{label}_periodogram.csv",
                    index=False,
                )
        except Exception as exc:
            rows.append(
                {
                    **common,
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv(result_path, index=False)
    return results, noise_record


def main(argv=None):
    args = parse_args(argv)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest_path, args.target_limit)
    cases = zero_injection_cases() if args.zero_injection else load_cases(
        args.case_file, args.case_limit
    )
    manifest.to_csv(args.output_dir / "manifest_used.csv", index=False)
    cases.to_csv(args.output_dir / "cases_used.csv", index=False)

    config = vars(args).copy()
    config = {
        key: (str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value)
        for key, value in config.items()
    }
    config["method_label"] = "tps_like_wavelet_matched_filter"
    config["zero_injection_control"] = bool(args.zero_injection)
    config["scientific_status"] = "proof_of_concept_not_full_kepler_tps_not_fap_calibrated"
    config["differences_from_full_kepler_tps"] = [
        "simplified local wavelet variance estimator",
        "integer-cadence period grid",
        "no SOC harmonic-removal stage",
        "no robust-statistic or chi-square vetoes",
        "no bootstrap false-alarm calibration",
        "no SOC-specific gap fill, quarter stitching, or edge corrections",
        "single-quarter POC only",
    ]
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )

    frames = []
    noise_rows = []
    progress_label = (
        "TPS-like zero-injection control"
        if args.zero_injection
        else "TPS-like BATMAN POC"
    )
    for row in tqdm(manifest.to_dict(orient="records"), desc=progress_label):
        try:
            result, noise = process_star(row, cases, args)
            frames.append(result)
            compact_noise = {key: value for key, value in noise.items() if key != "preprocessing"}
            noise_rows.append(compact_noise)
        except Exception as exc:
            noise_rows.append(
                {
                    "target_id": normalize_target_id(row["target_id"]),
                    "quarter": int(row["quarter"]),
                    "sample_stratum": str(row.get("sample_stratum", "unspecified")),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                args.output_dir / "tps_like_results.csv", index=False
            )
        pd.DataFrame(noise_rows).to_csv(
            args.output_dir / "tps_like_star_noise_models.csv", index=False
        )

    results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not args.zero_injection:
        write_summaries(results, args)

    run_label = "zero-injection control" if args.zero_injection else "BATMAN comparator"
    print(f"\nTPS-like {run_label} complete.")
    print(f"Result rows: {len(results)}")
    if not results.empty:
        valid = results.loc[results["success"].fillna(False)]
        print(f"Successful rows: {len(valid)}")
        if not valid.empty:
            if not args.zero_injection:
                print(f"Exact period recovery: {valid['exact_period_recovered'].mean():.3f}")
                print(f"Harmonic-aware recovery: {valid['harmonic_period_recovered'].mean():.3f}")
            print(f"Median MES-like statistic: {valid['mes'].median():.3f}")
            print(f"Median candidate observability: {valid['observability_fraction'].median():.3f}")
    print(f"Output: {args.output_dir}")
    if args.zero_injection:
        print(
            "\nThis is a true zero-injection/native-background diagnostic: BATMAN "
            "was skipped and the original preprocessed stellar flux was searched. "
            "It is not a randomized false-alarm calibration."
        )
    else:
        print(
            "\nInterpret this as a TPS-like integrated wavelet matched-filter POC. "
            "Do not call it Kepler TPS and do not compare detection efficiency as final "
            "until common false-alarm calibration is added."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
