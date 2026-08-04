"""Run transit-injection preservation and recovery checks across gap modes."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.injections.gap_mode_experiment import (
    GapModeInjectionConfig,
    empirical_false_alarm_thresholds,
    run_null_scan_for_thresholds,
    run_single_gap_mode_injection,
    summarize_gap_mode_injections,
)
from adaptive_transit.injections.synthetic import TransitInjection, choose_injection_centers
from adaptive_transit.noise_models.arima import evaluate_arima_candidates, fit_arima_model
from adaptive_transit.noise_models.gap_mode_comparison import (
    gap_mode_summary_row,
    score_candidates_for_gap_mode,
    select_gap_mode_candidate,
    stationarity_candidate_fields,
    validate_gap_comparison_schema,
)
from adaptive_transit.noise_models.stationarity import assess_stationarity
from adaptive_transit.preprocessing.gap_modes import GapModeRepresentation, build_gap_mode_representations
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve

DEFAULT_ORDERS = (
    (1, 0, 0),
    (2, 0, 0),
    (3, 0, 0),
    (1, 0, 1),
    (2, 0, 1),
    (3, 0, 1),
    (1, 1, 0),
    (2, 1, 0),
    (1, 1, 1),
)
DEFAULT_GAP_MODES = ("longest_contiguous", "full_grid_missing", "interpolated_full_grid")


def parse_order(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Orders must look like 'p,d,q'.")
    try:
        order = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ARIMA order entries must be integers.") from exc
    if any(part < 0 for part in order):
        raise argparse.ArgumentTypeError("ARIMA order entries must be non-negative.")
    return order


def parse_float_grid(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated floats.") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("All grid values must be positive.")
    return values


def parse_int_grid(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("All grid values must be positive.")
    return values


def parse_adf_autolag(value: str) -> str | None:
    normalized = value.strip()
    if normalized.lower() == "none":
        return None
    if normalized not in {"AIC", "BIC", "t-stat"}:
        raise argparse.ArgumentTypeError("ADF autolag must be AIC, BIC, t-stat, or none.")
    return normalized


def parse_kpss_nlags(value: str) -> str | int:
    normalized = value.strip()
    if normalized in {"auto", "legacy"}:
        return normalized
    try:
        nlags = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("KPSS nlags must be auto, legacy, or a non-negative integer.") from exc
    if nlags < 0:
        raise argparse.ArgumentTypeError("KPSS nlags must be non-negative.")
    return nlags


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run gap-mode ARIMA transit-injection diagnostics.")
    parser.add_argument("--target-id", default="11904151")
    parser.add_argument("--quarter", type=int, default=5)
    parser.add_argument("--quality-policy", default="default")
    parser.add_argument("--gap-mode", dest="gap_modes", action="append", choices=DEFAULT_GAP_MODES, default=None)
    parser.add_argument("--order", dest="orders", action="append", type=parse_order)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--acf-lags", type=int, default=80)
    parser.add_argument("--short-acf-lags", type=int, default=24)
    parser.add_argument("--transit-lag-min", type=int, default=3)
    parser.add_argument("--transit-lag-max", type=int, default=24)
    parser.add_argument("--stationarity-alpha", type=float, default=0.05)
    parser.add_argument("--stationarity-min-observations", type=int, default=24)
    parser.add_argument("--adf-regression", choices=("c", "ct", "ctt", "n"), default="c")
    parser.add_argument("--adf-autolag", type=parse_adf_autolag, default="AIC")
    parser.add_argument("--kpss-regression", choices=("c", "ct"), default="c")
    parser.add_argument("--kpss-nlags", type=parse_kpss_nlags, default="auto")
    parser.add_argument("--interpolation-method", choices=("linear",), default="linear")
    parser.add_argument("--max-interpolated-gap-cadences", type=int, default=12)
    parser.add_argument("--edge-extrapolation", action="store_true")
    parser.add_argument("--injection-depth-grid", type=parse_float_grid, default=(0.0005, 0.001, 0.002))
    parser.add_argument("--injection-duration-grid", type=parse_int_grid, default=(6,))
    parser.add_argument("--centers-per-duration", type=int, default=3)
    parser.add_argument("--local-half-width-cadences", type=int, default=24)
    parser.add_argument("--scan-stride", type=int, default=20)
    parser.add_argument("--scan-max-centers", type=int, default=120)
    parser.add_argument("--scale-window", type=int, default=96)
    parser.add_argument("--false-alarm-rates", type=parse_float_grid, default=(0.10, 0.05, 0.01))
    parser.add_argument("--fit-maxiter", type=int, default=200)
    parser.add_argument("--require-finite-flux-error", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/injections"))
    return parser


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None
    return value


def table_ready(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    for column in clean.columns:
        if clean[column].dtype != "object":
            continue
        clean[column] = clean[column].map(
            lambda value: json.dumps(json_ready(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
        )
    return clean


def stationarity_for_representation(representation: GapModeRepresentation, preprocessing_summary: dict[str, object], args: argparse.Namespace):
    metadata = {**preprocessing_summary, **representation.metadata}
    if representation.gap_mode == "longest_contiguous":
        series_representation = "finite_values_from_longest_contiguous_segment"
    elif representation.gap_mode == "full_grid_missing":
        series_representation = "finite_observed_values_from_full_grid_missing_series"
    elif bool(representation.metadata.get("remaining_missing_values", 0)):
        series_representation = "finite_observed_and_interpolated_values_from_interpolated_full_grid"
    else:
        series_representation = "interpolated_full_regular_grid"
    return assess_stationarity(
        representation.values,
        modelling_mode=representation.gap_mode,
        preprocessing_summary=metadata,
        alpha=args.stationarity_alpha,
        adf_regression=args.adf_regression,
        adf_autolag=args.adf_autolag,
        kpss_regression=args.kpss_regression,
        kpss_nlags=args.kpss_nlags,
        min_observations=args.stationarity_min_observations,
        gaps_compressed=bool(representation.metadata["missing_observations_compressed_for_stationarity"]),
        interpolated=bool(representation.metadata["interpolation_used"]),
        contiguous_segment_used=representation.gap_mode == "longest_contiguous",
        series_representation=series_representation,
    )


def evaluate_and_select(
    representation: GapModeRepresentation,
    *,
    quality_policy: str,
    orders: tuple[tuple[int, int, int], ...],
    assessment,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    results = evaluate_arima_candidates(
        representation.values,
        orders,
        mode=representation.gap_mode,
        allow_missing=representation.allow_missing,
        test_fraction=args.test_fraction,
        acf_lags=args.acf_lags,
        short_acf_lags=args.short_acf_lags,
        transit_lag_range=(args.transit_lag_min, args.transit_lag_max),
        fit_maxiter=args.fit_maxiter,
    )
    results["quality_policy"] = quality_policy
    results["gap_mode"] = representation.gap_mode
    results["transit_preservation_diagnostics_available"] = False
    for key, value in representation.metadata.items():
        if key != "gap_lengths":
            results[key] = value
    stationarity_rows = [stationarity_candidate_fields(assessment, int(row["d"])) for _, row in results.iterrows()]
    results = pd.concat([results.reset_index(drop=True), pd.DataFrame(stationarity_rows)], axis=1)
    scored = score_candidates_for_gap_mode(results)
    selected = select_gap_mode_candidate(scored)
    summary = gap_mode_summary_row(quality_policy=quality_policy, selected=selected, assessment=assessment, metadata=representation.metadata, alpha=args.stationarity_alpha)
    validate_gap_comparison_schema(pd.DataFrame([summary]))
    return scored, selected, summary


def build_shared_injections(reference: GapModeRepresentation, config: GapModeInjectionConfig) -> list[TransitInjection]:
    injections: list[TransitInjection] = []
    for duration in config.durations_cadences:
        centers = choose_injection_centers(
            reference.frame,
            duration_cadences=duration,
            centers_per_segment=config.centers_per_duration,
            max_segments=1,
        )
        for center in centers:
            for depth in config.depths:
                injections.append(TransitInjection(center_cadenceno=int(center), duration_cadences=int(duration), depth=float(depth)))
    return injections


def human_report(summary: pd.DataFrame, model_summary: pd.DataFrame, report: dict[str, Any]) -> str:
    model_columns = [
        "gap_mode",
        "selected_order",
        "stationarity_conclusion",
        "recommended_d",
        "selected_differencing_alignment",
        "scientifically_acceptable",
    ]
    lines = [
        "# Gap-Mode Transit-Injection Experiment",
        "",
        "This experiment injects known box transits into the same Kepler light curve representation used by each selected ARIMA gap-mode model.",
        "It does not run BLS, TLS, TCF, GP, ML, or false-alarm calibration beyond empirical single-light-curve null thresholds.",
        "",
        "## Selected Gap-Mode Models",
        "",
        "```text",
        model_summary[model_columns].to_string(index=False),
        "```",
        "",
        "## Injection Summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        f"- Any mode scientifically acceptable before injection? {report['any_mode_scientifically_acceptable_before_injection']}",
        f"- Any mode has perfect top recovery at FAR 0.01? {report['any_mode_top_recovers_all_at_far_0.01']}",
        f"- Best median SNR retention mode: {report['best_median_snr_retention_mode']}",
        f"- Highest spurious-peak-exceeds-injected rate mode: {report['highest_spurious_peak_rate_mode']}",
    ]
    return "\n".join(lines) + "\n"


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    orders = tuple(args.orders) if args.orders else DEFAULT_ORDERS
    gap_modes = tuple(args.gap_modes) if args.gap_modes else DEFAULT_GAP_MODES
    config = GapModeInjectionConfig(
        depths=tuple(args.injection_depth_grid),
        durations_cadences=tuple(args.injection_duration_grid),
        centers_per_duration=args.centers_per_duration,
        local_half_width_cadences=args.local_half_width_cadences,
        scan_stride=args.scan_stride,
        scan_max_centers=args.scan_max_centers,
        scale_window=args.scale_window,
        false_alarm_rates=tuple(args.false_alarm_rates),
    )
    light_curve = load_kepler_pdcsap(args.target_id, args.quarter)
    raw = light_curve.to_dataframe()
    regular, preprocessing_summary = preprocess_pdcsap_light_curve(
        raw,
        quality_policy=args.quality_policy,
        require_finite_flux_error=args.require_finite_flux_error,
        normalization_fit_fraction=1.0 - args.test_fraction,
    )
    representations = build_gap_mode_representations(
        regular,
        interpolation_method=args.interpolation_method,
        max_interpolated_gap_cadences=args.max_interpolated_gap_cadences,
        edge_extrapolation=args.edge_extrapolation,
    )
    shared_injections = build_shared_injections(representations["longest_contiguous"], config)
    candidate_tables: list[pd.DataFrame] = []
    model_summary_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    injection_rows: list[dict[str, Any]] = []

    for gap_mode in gap_modes:
        representation = representations[gap_mode]
        assessment = stationarity_for_representation(representation, preprocessing_summary.to_dict(), args)
        candidates, selected, model_summary = evaluate_and_select(
            representation,
            quality_policy=args.quality_policy,
            orders=orders,
            assessment=assessment,
            args=args,
        )
        candidate_tables.append(candidates)
        model_summary_rows.append(model_summary)
        selected_order = (int(selected["p"]), int(selected["d"]), int(selected["q"]))
        fitted = fit_arima_model(
            representation.values,
            selected_order,
            allow_missing=representation.allow_missing,
            mode=gap_mode,
            fit_maxiter=args.fit_maxiter,
        )
        threshold_by_duration_depth: dict[tuple[int, float], dict[str, float]] = {}
        for duration in config.durations_cadences:
            for depth in config.depths:
                null_scan = run_null_scan_for_thresholds(
                    frame=representation.frame,
                    values=representation.values,
                    fitted_model=fitted,
                    duration_cadences=duration,
                    depth=depth,
                    allow_missing=representation.allow_missing,
                    config=config,
                )
                thresholds = empirical_false_alarm_thresholds(null_scan, config.false_alarm_rates)
                threshold_by_duration_depth[(duration, depth)] = thresholds
                threshold_rows.append(
                    {
                        "gap_mode": gap_mode,
                        "duration_cadences": duration,
                        "depth": depth,
                        "n_null_trials": int(len(null_scan)),
                        **thresholds,
                    }
                )
        for injection_id, injection in enumerate(shared_injections, start=1):
            row = run_single_gap_mode_injection(
                gap_mode=gap_mode,
                frame=representation.frame,
                values=representation.values,
                fitted_model=fitted,
                injection=injection,
                allow_missing=representation.allow_missing,
                config=config,
                thresholds=threshold_by_duration_depth[(injection.duration_cadences, injection.depth)],
            )
            row.update(
                {
                    "injection_id": injection_id,
                    "quality_policy": args.quality_policy,
                    "selected_order": f"ARIMA{selected_order}",
                    "selected_candidate_status": str(selected.get("selection_status", "")),
                    "selected_differencing_alignment": str(selected.get("candidate_differencing_alignment", "unresolved")),
                    "gap_mode_scientifically_acceptable_before_injection": bool(model_summary["scientifically_acceptable"]),
                }
            )
            injection_rows.append(row)

    injection_results = pd.DataFrame(injection_rows)
    injection_summary = summarize_gap_mode_injections(injection_results, config.false_alarm_rates)
    model_summary_table = pd.DataFrame(model_summary_rows)
    threshold_table = pd.DataFrame(threshold_rows)
    candidate_table = pd.concat(candidate_tables, ignore_index=True)
    best_snr_mode = injection_summary.sort_values("median_snr_retention_fraction", ascending=False, na_position="last").iloc[0]["gap_mode"]
    highest_spurious_mode = injection_summary.sort_values("spurious_peak_exceeds_injected_rate", ascending=False, na_position="last").iloc[0]["gap_mode"]
    report = {
        "target_id": str(args.target_id),
        "quarter": int(args.quarter),
        "quality_policy": args.quality_policy,
        "config": config.to_dict(),
        "orders": [str(order) for order in orders],
        "injection_count": int(len(injection_results)),
        "model_summary": json_ready(model_summary_table.to_dict(orient="records")),
        "summary": json_ready(injection_summary.to_dict(orient="records")),
        "any_mode_scientifically_acceptable_before_injection": bool(model_summary_table["scientifically_acceptable"].astype(bool).any()),
        "any_mode_top_recovers_all_at_far_0.01": bool((injection_summary.get("top_recovery_rate_at_far_0.01", pd.Series(dtype=float)) >= 1.0).any()),
        "best_median_snr_retention_mode": str(best_snr_mode),
        "highest_spurious_peak_rate_mode": str(highest_spurious_mode),
    }
    return injection_results, injection_summary, threshold_table, model_summary_table, candidate_table, report


def main(args=None):
    args = args or build_parser().parse_args()
    if args.centers_per_duration < 1:
        raise ValueError("--centers-per-duration must be positive.")
    if args.local_half_width_cadences < 1:
        raise ValueError("--local-half-width-cadences must be positive.")
    if args.scan_stride < 1:
        raise ValueError("--scan-stride must be positive.")
    if args.scan_max_centers < 1:
        raise ValueError("--scan-max-centers must be positive.")
    if args.fit_maxiter <= 0:
        raise ValueError("--fit-maxiter must be positive.")
    if any(rate <= 0 or rate >= 1 for rate in args.false_alarm_rates):
        raise ValueError("--false-alarm-rates must be in (0, 1).")

    metrics_dir = args.output_dir / "metrics"
    processed_dir = args.output_dir / "processed"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    injection_results, injection_summary, threshold_table, model_summary, candidate_table, report = run_experiment(args)
    prefix = f"kic_{str(args.target_id).replace('KIC', '').strip()}_q{args.quarter}"
    results_csv = metrics_dir / f"{prefix}_gap_mode_injection_results.csv"
    summary_csv = metrics_dir / f"{prefix}_gap_mode_injection_summary.csv"
    thresholds_csv = metrics_dir / f"{prefix}_gap_mode_injection_thresholds.csv"
    model_summary_csv = metrics_dir / f"{prefix}_gap_mode_injection_model_summary.csv"
    candidates_csv = metrics_dir / f"{prefix}_gap_mode_injection_candidates.csv"
    report_json = metrics_dir / f"{prefix}_gap_mode_injection_report.json"
    report_md = metrics_dir / f"{prefix}_gap_mode_injection_report.md"
    results_parquet = processed_dir / f"{prefix}_gap_mode_injection_results.parquet"
    summary_parquet = processed_dir / f"{prefix}_gap_mode_injection_summary.parquet"

    table_ready(injection_results).to_csv(results_csv, index=False)
    table_ready(injection_summary).to_csv(summary_csv, index=False)
    table_ready(threshold_table).to_csv(thresholds_csv, index=False)
    table_ready(model_summary).to_csv(model_summary_csv, index=False)
    table_ready(candidate_table).to_csv(candidates_csv, index=False)
    table_ready(injection_results).to_parquet(results_parquet, index=False)
    table_ready(injection_summary).to_parquet(summary_parquet, index=False)
    report_json.write_text(json.dumps(json_ready(report), indent=2) + "\n")
    report_md.write_text(human_report(injection_summary, model_summary, report))

    print(f"Injection results: {results_csv}")
    print(f"Injection summary: {summary_csv}")
    print(f"Null thresholds: {thresholds_csv}")
    print(f"Model summary: {model_summary_csv}")
    print(f"Report: {report_md}")
    print()
    print(injection_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
