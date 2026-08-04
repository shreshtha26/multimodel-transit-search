"""Compare explicit gap-handling modes for the ARIMA noise-model branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.noise_models.arima import evaluate_arima_candidates, fit_arima_model
from adaptive_transit.noise_models.correlograms import CorrelogramPlotRecord, save_correlogram_plot, skipped_correlogram_record
from adaptive_transit.noise_models.gap_mode_comparison import (
    candidate_family_table,
    gap_mode_summary_row,
    scientific_admissibility,
    score_candidates_for_gap_mode,
    select_gap_mode_candidate,
    same_candidate_family_across_modes,
    stationarity_candidate_fields,
    validate_gap_comparison_schema,
)
from adaptive_transit.noise_models.stationarity import assess_stationarity, stationarity_report_fields
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
    """Parse `p,d,q` CLI values."""

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


def parse_adf_autolag(value: str) -> str | None:
    """Parse ADF autolag configuration."""

    normalized = value.strip()
    if normalized.lower() == "none":
        return None
    if normalized not in {"AIC", "BIC", "t-stat"}:
        raise argparse.ArgumentTypeError("ADF autolag must be AIC, BIC, t-stat, or none.")
    return normalized


def parse_kpss_nlags(value: str) -> str | int:
    """Parse KPSS lag-selection configuration."""

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
    """Build the gap-mode comparison CLI."""

    parser = argparse.ArgumentParser(description="Compare ARIMA gap-handling representations.")
    parser.add_argument("--target-id", default="11904151")
    parser.add_argument("--quarter", type=int, default=5)
    parser.add_argument("--quality-policy", dest="quality_policies", action="append", default=None)
    parser.add_argument("--gap-mode", dest="gap_modes", action="append", choices=DEFAULT_GAP_MODES, default=None)
    parser.add_argument("--order", dest="orders", action="append", type=parse_order)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--acf-lags", type=int, default=80)
    parser.add_argument("--correlogram-lags", type=int, default=24)
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
    parser.add_argument("--fit-maxiter", type=int, default=200)
    parser.add_argument("--require-finite-flux-error", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gap_modes"))
    return parser


def json_ready(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-compatible structures."""

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
    """Convert nested object cells into stable strings before CSV/Parquet writes."""

    clean = frame.copy()
    for column in clean.columns:
        if clean[column].dtype != "object":
            continue
        clean[column] = clean[column].map(
            lambda value: json.dumps(json_ready(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
        )
    return clean


def attach_stationarity(results: pd.DataFrame, assessment) -> pd.DataFrame:
    """Attach stationarity context to every candidate row for one gap mode."""

    rows = [stationarity_candidate_fields(assessment, int(row["d"])) for _, row in results.iterrows()]
    return pd.concat([results.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def stationarity_assessment_for_representation(
    representation: GapModeRepresentation,
    *,
    preprocessing_summary: dict[str, object],
    args: argparse.Namespace,
):
    """Run ADF/KPSS on the exact finite representation documented for this mode."""

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


def plot_annotation(representation: GapModeRepresentation, *, compressed: bool) -> str:
    """Build concise plot metadata text."""

    return (
        f"compressed={compressed}; interpolation={representation.metadata['interpolation_used']}; "
        f"repr={representation.metadata['series_plot_representation']}"
    )


def save_series_plots(
    representation: GapModeRepresentation,
    *,
    prefix: str,
    quality_policy: str,
    figures_dir: Path,
    args: argparse.Namespace,
    cadence_days: float,
    transit_lag_range: tuple[int, int],
) -> list[CorrelogramPlotRecord]:
    """Save series-level ACF/PACF diagnostics for a gap representation."""

    records: list[CorrelogramPlotRecord] = []
    base = f"{prefix}_{quality_policy}_{representation.gap_mode}"
    interpolation_used = bool(representation.metadata["interpolation_used"])
    if representation.gap_mode == "full_grid_missing":
        values = representation.values
        diff_values = np.diff(values)
        for series_kind, series_values in (("modelling_series", values), ("first_differenced_series", diff_values)):
            path = figures_dir / f"{base}_{series_kind}_acf.png"
            records.append(
                save_correlogram_plot(
                    series_values,
                    path,
                    gap_mode=representation.gap_mode,
                    series_kind=series_kind,
                    plot_kind="acf",
                    max_lag=args.correlogram_lags,
                    cadence_days=cadence_days,
                    missing_observations_compressed=False,
                    interpolation_used=False,
                    missing_strategy="conservative",
                    transit_lag_range=transit_lag_range,
                    annotation="missing-aware conservative ACF; PACF omitted",
                )
            )
            pacf_path = figures_dir / f"{base}_{series_kind}_pacf.png"
            records.append(
                skipped_correlogram_record(
                    gap_mode=representation.gap_mode,
                    series_kind=series_kind,
                    plot_kind="pacf",
                    path=pacf_path,
                    reason="PACF omitted because compressing missing full-grid cadences would change the cadence-lag meaning.",
                    n_observations=int(np.isfinite(series_values).sum()),
                    max_lag=args.correlogram_lags,
                    missing_observations_compressed=False,
                    interpolation_used=False,
                    missing_strategy="not_applicable",
                )
            )
        return records

    values = representation.series_plot_values
    diff_values = np.diff(values)
    compressed = representation.gap_mode == "interpolated_full_grid" and bool(representation.metadata["remaining_missing_values"])
    annotation = plot_annotation(representation, compressed=compressed)
    for series_kind, series_values in (("modelling_series", values), ("first_differenced_series", diff_values)):
        for plot_kind in ("acf", "pacf"):
            records.append(
                save_correlogram_plot(
                    series_values,
                    figures_dir / f"{base}_{series_kind}_{plot_kind}.png",
                    gap_mode=representation.gap_mode,
                    series_kind=series_kind,
                    plot_kind=plot_kind,
                    max_lag=args.correlogram_lags,
                    cadence_days=cadence_days,
                    missing_observations_compressed=compressed,
                    interpolation_used=interpolation_used,
                    missing_strategy="none",
                    transit_lag_range=transit_lag_range,
                    annotation=annotation,
                )
            )
    return records


def save_residual_plots(
    representation: GapModeRepresentation,
    selected: pd.Series,
    *,
    prefix: str,
    quality_policy: str,
    figures_dir: Path,
    args: argparse.Namespace,
    cadence_days: float,
    transit_lag_range: tuple[int, int],
) -> list[CorrelogramPlotRecord]:
    """Save residual ACF/PACF plots for a trustworthy selected fit."""

    records: list[CorrelogramPlotRecord] = []
    base = f"{prefix}_{quality_policy}_{representation.gap_mode}_selected_residual"
    order = (int(selected["p"]), int(selected["d"]), int(selected["q"]))
    if str(selected.get("failure_reason", "")) or not bool(selected.get("fit_metrics_trustworthy", False)):
        for plot_kind in ("acf", "pacf"):
            records.append(
                skipped_correlogram_record(
                    gap_mode=representation.gap_mode,
                    series_kind="selected_residual",
                    plot_kind=plot_kind,
                    path=figures_dir / f"{base}_{plot_kind}.png",
                    reason="Selected fit is not trustworthy enough for residual correlogram plotting.",
                    n_observations=0,
                    max_lag=args.correlogram_lags,
                    missing_observations_compressed=representation.allow_missing,
                    interpolation_used=bool(representation.metadata["interpolation_used"]),
                    missing_strategy="not_applicable",
                )
            )
        return records

    fitted = fit_arima_model(
        representation.values,
        order,
        allow_missing=representation.allow_missing,
        mode=representation.gap_mode,
        fit_maxiter=args.fit_maxiter,
    )
    residuals = fitted.innovations.copy()
    residuals[~fitted.usable_mask] = np.nan
    finite_residuals = residuals[np.isfinite(residuals)]
    compressed = bool(np.isnan(residuals).any())
    annotation = f"finite selected innovations; compressed={compressed}; order={order}"
    for plot_kind in ("acf", "pacf"):
        records.append(
            save_correlogram_plot(
                finite_residuals,
                figures_dir / f"{base}_{plot_kind}.png",
                gap_mode=representation.gap_mode,
                series_kind="selected_residual",
                plot_kind=plot_kind,
                max_lag=args.correlogram_lags,
                cadence_days=cadence_days,
                missing_observations_compressed=compressed,
                interpolation_used=bool(representation.metadata["interpolation_used"]),
                missing_strategy="none",
                transit_lag_range=transit_lag_range,
                annotation=annotation,
            )
        )
    return records


def comparison_questions(summary: pd.DataFrame) -> dict[str, Any]:
    """Answer the fixed scientific questions from the selected-row summary."""

    selected_orders = summary["selected_order"].astype(str)
    selected_d = selected_orders.str.extract(r"\((\d+),\s*(\d+),\s*(\d+)\)")[1].dropna().astype(int)
    full_order = summary.loc[summary["gap_mode"] == "full_grid_missing", "selected_order"]
    interp = summary.loc[summary["gap_mode"] == "interpolated_full_grid"]
    full = summary.loc[summary["gap_mode"] == "full_grid_missing"]
    interpolation_fit_note = "not_available"
    if not interp.empty and not full.empty:
        interp_rmse = float(interp["test_RMSE"].iloc[0])
        full_rmse = float(full["test_RMSE"].iloc[0])
        interpolation_fit_note = "lower_rmse_than_full_grid_missing" if interp_rmse < full_rmse else "not_lower_rmse_than_full_grid_missing"
    return {
        "selected_d_changes_by_gap_mode": bool(selected_d.nunique() > 1),
        "selected_order_changes_by_gap_mode": bool(selected_orders.nunique() > 1),
        "stationarity_conclusion_changes_by_gap_mode": bool(summary["stationarity_conclusion"].astype(str).nunique() > 1),
        "residual_whitening_improves_in_any_mode": bool(summary["residual_autocorrelation_remaining"].eq(False).any()),
        "variance_stability_improves_in_any_mode": bool(summary["variance_instability"].eq(False).any()),
        "interpolation_fit_metric_note": interpolation_fit_note,
        "longest_contiguous_favours_d0": bool(
            summary.loc[summary["gap_mode"] == "longest_contiguous", "selected_order"].astype(str).str.contains(r", 0,|,0,", regex=True).any()
        ),
        "any_mode_scientifically_acceptable": bool(summary["scientifically_acceptable"].astype(bool).any()),
        "full_grid_missing_order": str(full_order.iloc[0]) if not full_order.empty else "",
    }


def best_available_row(summary: pd.DataFrame) -> dict[str, Any]:
    """Select the best diagnostic row without treating forecast metrics as primary evidence."""

    ranked = summary.copy()
    alignment_priority = {
        "aligned_with_stationarity_evidence": 0,
        "unresolved": 1,
        "conflicts_with_stationarity_evidence": 2,
    }
    ranked["alignment_priority"] = ranked["selected_differencing_alignment"].map(alignment_priority).fillna(9)
    ranked["model_complexity"] = ranked["selected_p"].astype(float) + ranked["selected_d"].astype(float) + ranked["selected_q"].astype(float)
    ranked["not_default_full_grid"] = (ranked["gap_mode"] != "full_grid_missing").astype(int)
    sort_columns = [
        "scientifically_acceptable",
        "fit_metrics_trustworthy",
        "alignment_priority",
        "whitening_constraint_failed",
        "max_abs_residual_acf_transit_lags",
        "variance_constraint_failed",
        "interpolated_fraction",
        "model_complexity",
        "test_RMSE",
        "BIC",
        "not_default_full_grid",
    ]
    ranked = ranked.sort_values(
        sort_columns,
        ascending=[False, False, True, True, True, True, True, True, True, True, True],
        kind="mergesort",
    )
    return ranked.iloc[0].drop(labels=["alignment_priority", "model_complexity", "not_default_full_grid"]).to_dict()


def human_report(summary: pd.DataFrame, questions: dict[str, Any], best_available: dict[str, Any]) -> str:
    """Build a concise Markdown report for manual review."""

    lines = [
        "# Gap-Mode ARIMA Comparison",
        "",
        "This report compares explicit gap representations only. It does not add BLS, TCF, GP, ML, or false-alarm calibration.",
        "",
        "## Selected Models",
        "",
        summary[
            [
                "quality_policy",
                "gap_mode",
                "selected_order",
                "stationarity_conclusion",
                "recommended_d",
                "selected_differencing_alignment",
                "residual_autocorrelation_remaining",
                "variance_instability",
                "scientifically_acceptable",
            ]
        ].to_markdown(index=False),
        "",
        "## Questions",
        "",
        f"- Does selected d change by gap mode? {questions['selected_d_changes_by_gap_mode']}",
        f"- Does selected ARIMA order change? {questions['selected_order_changes_by_gap_mode']}",
        f"- Does the stationarity conclusion change? {questions['stationarity_conclusion_changes_by_gap_mode']}",
        f"- Does residual whitening improve? {questions['residual_whitening_improves_in_any_mode']}",
        f"- Does variance stability improve? {questions['variance_stability_improves_in_any_mode']}",
        f"- Does interpolation artificially improve fit metrics? {questions['interpolation_fit_metric_note']}",
        f"- Does the longest contiguous segment favour d=0? {questions['longest_contiguous_favours_d0']}",
        f"- Is any mode scientifically acceptable? {questions['any_mode_scientifically_acceptable']}",
        "",
        "## Best Available",
        "",
        f"Best available mode/model: {best_available.get('gap_mode', '')} {best_available.get('selected_order', '')}",
        "",
        "If all modes remain scientifically inadequate, the best available row is diagnostic only.",
    ]
    if not bool(questions["any_mode_scientifically_acceptable"]):
        lines.append("")
        lines.append("No scientifically acceptable gap representation/model combination was found.")
    return "\n".join(lines) + "\n"


def run_gap_mode_comparison(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the package-level gap-mode comparison."""

    orders = tuple(args.orders) if args.orders else DEFAULT_ORDERS
    quality_policies = tuple(args.quality_policies) if args.quality_policies else ("default",)
    gap_modes = tuple(args.gap_modes) if args.gap_modes else DEFAULT_GAP_MODES
    transit_lag_range = (int(args.transit_lag_min), int(args.transit_lag_max))
    light_curve = load_kepler_pdcsap(args.target_id, args.quarter)
    raw = light_curve.to_dataframe()

    candidate_frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    plot_records: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    stationarity_rows: list[dict[str, Any]] = []
    candidate_family = candidate_family_table(orders, gap_modes)
    prefix = f"kic_{str(args.target_id).replace('KIC', '').strip()}_q{args.quarter}"
    figures_dir = args.output_dir / "figures"

    for quality_policy in quality_policies:
        regular, preprocessing_summary = preprocess_pdcsap_light_curve(
            raw,
            quality_policy=quality_policy,
            require_finite_flux_error=args.require_finite_flux_error,
            normalization_fit_fraction=1.0 - args.test_fraction,
        )
        summary_dict = preprocessing_summary.to_dict()
        cadence_days = float(summary_dict["median_cadence_days"])
        representations = build_gap_mode_representations(
            regular,
            interpolation_method=args.interpolation_method,
            max_interpolated_gap_cadences=args.max_interpolated_gap_cadences,
            edge_extrapolation=args.edge_extrapolation,
        )

        for gap_mode in gap_modes:
            representation = representations[gap_mode]
            metadata = representation.metadata
            metadata_rows.append({"quality_policy": quality_policy, **metadata, "gap_lengths": "|".join(map(str, metadata["gap_lengths"]))})
            assessment = stationarity_assessment_for_representation(
                representation,
                preprocessing_summary=summary_dict,
                args=args,
            )
            stationarity_rows.append(
                {
                    "quality_policy": quality_policy,
                    "gap_mode": gap_mode,
                    "ordinary_lags_meaningful": bool(representation.ordinary_lags_meaningful),
                    **stationarity_report_fields(assessment),
                    "stationarity_series_representation": assessment.preprocessing_summary.get("stationarity_series_representation", ""),
                    "stationarity_gaps_compressed": assessment.preprocessing_summary.get("stationarity_gaps_compressed", False),
                    "stationarity_interpolated": assessment.preprocessing_summary.get("stationarity_interpolated", False),
                    "stationarity_contiguous_segment_used": assessment.preprocessing_summary.get("stationarity_contiguous_segment_used", False),
                    "stationarity_removed_nonfinite": assessment.preprocessing_summary.get("stationarity_removed_nonfinite", 0),
                }
            )
            results = evaluate_arima_candidates(
                representation.values,
                orders,
                mode=gap_mode,
                allow_missing=representation.allow_missing,
                test_fraction=args.test_fraction,
                acf_lags=args.acf_lags,
                short_acf_lags=args.correlogram_lags,
                transit_lag_range=transit_lag_range,
                fit_maxiter=args.fit_maxiter,
            )
            results["quality_policy"] = quality_policy
            results["gap_mode"] = gap_mode
            results["transit_preservation_diagnostics_available"] = False
            results["transit_preservation_failure_reason"] = "not_run_in_gap_mode_comparison"
            for key, value in metadata.items():
                if key != "gap_lengths":
                    results[key] = value
            results = attach_stationarity(results, assessment)
            scored = score_candidates_for_gap_mode(results)
            selected = select_gap_mode_candidate(scored)
            acceptable, reasons = scientific_admissibility(selected)
            scored["scientifically_acceptable"] = False
            scored["scientific_admissibility_reasons"] = ""
            selected_index = selected.name
            scored.loc[selected_index, "scientifically_acceptable"] = acceptable
            scored.loc[selected_index, "scientific_admissibility_reasons"] = ";".join(reasons)
            candidate_frames.append(scored)
            selected_rows.append(
                gap_mode_summary_row(
                    quality_policy=quality_policy,
                    selected=selected,
                    assessment=assessment,
                    metadata=metadata,
                    alpha=args.stationarity_alpha,
                )
            )
            plot_records.extend(
                record.to_dict()
                for record in save_series_plots(
                    representation,
                    prefix=prefix,
                    quality_policy=quality_policy,
                    figures_dir=figures_dir,
                    args=args,
                    cadence_days=cadence_days,
                    transit_lag_range=transit_lag_range,
                )
            )
            plot_records.extend(
                record.to_dict()
                for record in save_residual_plots(
                    representation,
                    selected,
                    prefix=prefix,
                    quality_policy=quality_policy,
                    figures_dir=figures_dir,
                    args=args,
                    cadence_days=cadence_days,
                    transit_lag_range=transit_lag_range,
                )
            )

    detailed = pd.concat(candidate_frames, ignore_index=True)
    summary = pd.DataFrame(selected_rows)
    plots = pd.DataFrame(plot_records)
    metadata_table = pd.DataFrame(metadata_rows)
    stationarity_table = pd.DataFrame(stationarity_rows)
    validate_gap_comparison_schema(summary)
    questions = comparison_questions(summary)
    best = best_available_row(summary)
    report = {
        "target_id": str(args.target_id),
        "quarter": int(args.quarter),
        "orders": [str(order) for order in orders],
        "same_candidate_family_across_modes": same_candidate_family_across_modes(candidate_family),
        "fit_maxiter": int(args.fit_maxiter) if args.fit_maxiter is not None else None,
        "comparison_questions": questions,
        "best_available": json_ready(best),
        "no_scientifically_acceptable_combination": not bool(questions["any_mode_scientifically_acceptable"]),
    }
    full_report = {
        **report,
        "candidate_family": json_ready(candidate_family.to_dict(orient="records")),
        "metadata": json_ready(metadata_table.to_dict(orient="records")),
    }
    return detailed, summary, plots, metadata_table, stationarity_table, candidate_family, full_report


def main() -> int:
    args = build_parser().parse_args()
    if args.max_interpolated_gap_cadences < 0:
        raise ValueError("--max-interpolated-gap-cadences must be non-negative.")
    if args.fit_maxiter is not None and args.fit_maxiter <= 0:
        raise ValueError("--fit-maxiter must be positive when provided.")
    if args.transit_lag_min < 1 or args.transit_lag_max < args.transit_lag_min:
        raise ValueError("--transit-lag-min must be >= 1 and --transit-lag-max must be >= --transit-lag-min.")

    metrics_dir = args.output_dir / "metrics"
    processed_dir = args.output_dir / "processed"
    figures_dir = args.output_dir / "figures"
    for directory in (metrics_dir, processed_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    detailed, summary, plots, metadata_table, stationarity_table, candidate_family, report = run_gap_mode_comparison(args)
    prefix = f"kic_{str(args.target_id).replace('KIC', '').strip()}_q{args.quarter}"
    candidates_parquet = processed_dir / f"{prefix}_gap_mode_candidates.parquet"
    candidates_csv = metrics_dir / f"{prefix}_gap_mode_candidates.csv"
    summary_parquet = processed_dir / f"{prefix}_gap_mode_comparison.parquet"
    summary_csv = metrics_dir / f"{prefix}_gap_mode_comparison.csv"
    metadata_csv = metrics_dir / f"{prefix}_gap_mode_metadata.csv"
    stationarity_csv = metrics_dir / f"{prefix}_gap_mode_stationarity.csv"
    candidate_family_csv = metrics_dir / f"{prefix}_gap_mode_candidate_family.csv"
    plots_csv = metrics_dir / f"{prefix}_gap_mode_plot_manifest.csv"
    report_json = metrics_dir / f"{prefix}_gap_mode_report.json"
    report_md = metrics_dir / f"{prefix}_gap_mode_report.md"

    table_ready(detailed).to_parquet(candidates_parquet, index=False)
    table_ready(detailed).to_csv(candidates_csv, index=False)
    table_ready(summary).to_parquet(summary_parquet, index=False)
    table_ready(summary).to_csv(summary_csv, index=False)
    table_ready(metadata_table).to_csv(metadata_csv, index=False)
    table_ready(stationarity_table).to_csv(stationarity_csv, index=False)
    table_ready(candidate_family).to_csv(candidate_family_csv, index=False)
    table_ready(plots).to_csv(plots_csv, index=False)
    report_json.write_text(json.dumps(json_ready(report), indent=2) + "\n")
    report_md.write_text(human_report(summary, report["comparison_questions"], report["best_available"]))

    print(f"Gap-mode summary: {summary_csv}")
    print(f"Detailed candidates: {candidates_parquet}")
    print(f"Candidate family: {candidate_family_csv}")
    print(f"Stationarity table: {stationarity_csv}")
    print(f"Plot manifest: {plots_csv}")
    print(f"Human report: {report_md}")
    print()
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
