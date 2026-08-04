"""Run multi-model-transit-search Stage 1 for one Kepler target and quarter.
PDCSAP flux -> cadence grid -> explicit masks/gaps -> leakage-free
normalization -> ARIMA diagnostics -> selected model -> one-step-ahead innovations.
"""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.detection.matched_filter import (
    ArimaTemplateTransform,
    arima_transformed_template,
    local_window_mask,
    matched_filter_statistic,
    scan_arima_transformed_template,
    select_trial_centers)
from adaptive_transit.injections.synthetic import (
    TransitInjection,
    choose_injection_center,
    choose_injection_centers,
    inject_box_transit,
    transit_preservation_metrics)
from adaptive_transit.noise_models.arima import (
    apply_fitted_arima_filter,
    evaluate_arima_candidates,
    fit_arima_model,
    generate_arima_orders)
from adaptive_transit.noise_models.scaling import (
    standardize_innovations,
    trailing_robust_scale)
from adaptive_transit.noise_models.selection import (
    order_from_row,
    score_arima_candidates,
    select_noise_model)
from adaptive_transit.noise_models.stationarity import (
    StationarityAssessment,
    assess_stationarity,
    stationarity_candidate_fields,
    stationarity_report_fields)
from adaptive_transit.noise_models.stability import (
    chronological_prefix_stability,
    segment_stability,
    summarize_stability)
from adaptive_transit.preprocessing.normalization import (
    longest_contiguous_segment,
    preprocess_pdcsap_light_curve,
    segment_lengths)

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
DEFAULT_QUALITY_POLICIES = ("strict", "default", "permissive")


def parse_order(value: str) -> tuple[int, int, int]:
    """Parse command-line values like `1,0,0` into an ARIMA order."""

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Orders must look like 'p,d,q', for example '1,0,0'.")
    try:
        order = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ARIMA order entries must be integers.") from exc
    if any(part < 0 for part in order):
        raise argparse.ArgumentTypeError("ARIMA order entries must be non-negative.")
    return order


def parse_float_grid(value: str) -> tuple[float, ...]:
    """Parse comma-separated positive floats."""

    try:
        values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated floats.") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one float is required.")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Grid values must be positive.")
    return values


def parse_int_grid(value: str) -> tuple[int, ...]:
    """Parse comma-separated positive integers."""

    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Grid values must be positive.")
    return values


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
    parser = argparse.ArgumentParser(description="Run single-target PDCSAP ARIMA diagnostics.")
    parser.add_argument(
        "--target-id",
        default="11904151",
        help="Kepler target, for example 11904151.",
    )
    parser.add_argument("--quarter", type=int, default=5, help="Kepler quarter to download.")
    parser.add_argument(
        "--order",
        dest="orders",
        action="append",
        type=parse_order,
        help="Candidate ARIMA order as p,d,q. Repeat this flag for multiple orders.",
    )
    parser.add_argument(
        "--expanded-arima-grid",
        action="store_true",
        help="Search a generated ARIMA grid instead of the conservative default list.",
    )
    parser.add_argument("--max-p", type=int, default=5)
    parser.add_argument("--max-d", type=int, default=1)
    parser.add_argument("--max-q", type=int, default=5)
    parser.add_argument(
        "--max-total-order",
        type=int,
        default=None,
        help="Optional cap on p+d+q for generated ARIMA grids.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--acf-lags", type=int, default=80)
    parser.add_argument("--fit-maxiter", type=int, default=200)
    parser.add_argument("--stationarity-alpha", type=float, default=0.05)
    parser.add_argument("--stationarity-min-observations", type=int, default=24)
    parser.add_argument("--adf-regression", choices=("c", "ct", "ctt", "n"), default="c")
    parser.add_argument("--adf-autolag", type=parse_adf_autolag, default="AIC")
    parser.add_argument("--kpss-regression", choices=("c", "ct"), default="c")
    parser.add_argument("--kpss-nlags", type=parse_kpss_nlags, default="auto")
    parser.add_argument("--transit-lag-min", type=int, default=3)
    parser.add_argument("--transit-lag-max", type=int, default=24)
    parser.add_argument(
        "--quality-policy",
        dest="quality_policies",
        action="append",
        help="Quality mask policy to compare. Repeat to compare several.",
    )
    parser.add_argument("--stability-folds", type=int, default=3)
    parser.add_argument("--stability-segments", type=int, default=3)
    parser.add_argument("--scale-window", type=int, default=96)
    parser.add_argument("--injection-depth", type=float, default=0.001)
    parser.add_argument("--injection-duration-cadences", type=int, default=6)
    parser.add_argument(
        "--injection-depth-grid",
        type=parse_float_grid,
        default=None,
        help="Comma-separated injection depths for multi-injection recovery.",
    )
    parser.add_argument(
        "--injection-duration-grid",
        type=parse_int_grid,
        default=None,
        help="Comma-separated durations for multi-injection recovery.",
    )
    parser.add_argument(
        "--injection-centers-per-duration",
        type=int,
        default=3,
        help="Number of clean centers per duration for selected-model recovery tests.",
    )
    parser.add_argument(
        "--injection-max-segments",
        type=int,
        default=3,
        help="Maximum contiguous usable segments used for selected-model injections.",
    )
    parser.add_argument("--injection-local-half-width-cadences", type=int, default=24)
    parser.add_argument(
        "--scan-stride",
        type=int,
        default=10,
        help="Use every Nth usable cadence as a blind scan trial center.",
    )
    parser.add_argument(
        "--scan-max-centers",
        type=int,
        default=250,
        help="Cap the blind scan to this many trial centers, plus the injected center.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--require-finite-flux-error",
        action="store_true",
        help="Also drop rows where PDCSAP_FLUX_ERR is non-finite.",
    )
    return parser


def save_flux_plot(regular: pd.DataFrame, path: Path) -> None:
    """Save the normalized PDCSAP light curve for visual inspection."""

    usable = regular["usable"].to_numpy(dtype=bool)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        regular.loc[usable, "time"],
        regular.loc[usable, "normalized_flux"],
        ".",
        markersize=2,
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Time [BKJD]")
    ax.set_ylabel("Normalized PDCSAP flux")
    ax.set_title("Single Kepler Quarter: Normalized PDCSAP Flux")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_gap_plot(regular: pd.DataFrame, path: Path) -> None:
    """Save a cadence-index plot that makes missing rows and flags visible."""

    fig, ax = plt.subplots(figsize=(12, 3))
    y = np.zeros(len(regular), dtype=float)
    usable = regular["usable"].to_numpy(dtype=bool)
    absent = ~regular["row_present"].to_numpy(dtype=bool)
    flagged = regular["row_present"].to_numpy(dtype=bool) & ~usable

    ax.plot(regular["cadenceno"], y, color="0.85", linewidth=1)
    ax.plot(regular.loc[usable, "cadenceno"], y[usable], ".", markersize=2, label="usable")
    ax.plot(
        regular.loc[flagged, "cadenceno"],
        y[flagged] + 0.05,
        ".",
        markersize=2,
        label="present but unusable",
    )
    ax.plot(
        regular.loc[absent, "cadenceno"],
        y[absent] - 0.05,
        ".",
        markersize=2,
        label="absent cadence",
    )
    ax.set_yticks([])
    ax.set_xlabel("Kepler cadence number")
    ax.set_title("Explicit Cadence Gaps And Quality-Masked Rows")
    ax.legend(loc="upper right", markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_innovation_plot(
    innovations: pd.DataFrame,
    selected_order: tuple[int, int, int],
    path: Path,
) -> None:
    """Save observed flux and selected-model innovations on the same time base."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(innovations["time"], innovations["normalized_flux"], ".", markersize=2)
    axes[0].set_ylabel("Flux")
    axes[0].set_title("Normalized Flux")

    usable = innovations["innovation_usable"].to_numpy(dtype=bool)
    axes[1].plot(
        innovations.loc[usable, "time"],
        innovations.loc[usable, "innovation"],
        ".",
        markersize=2,
    )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("Innovation")
    axes[1].set_xlabel("Time [BKJD]")
    axes[1].set_title(f"One-Step-Ahead Innovations: ARIMA{selected_order}")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_standardized_innovation_plot(
    innovations: pd.DataFrame,
    selected_order: tuple[int, int, int],
    path: Path,
) -> None:
    """Save raw and standardized innovations for variance-behavior inspection."""

    usable = innovations["innovation_usable"].to_numpy(dtype=bool)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(
        innovations.loc[usable, "time"],
        innovations.loc[usable, "innovation"],
        ".",
        markersize=2,
    )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_ylabel("Innovation")
    axes[0].set_title(f"Raw Innovations: ARIMA{selected_order}")

    axes[1].plot(
        innovations.loc[usable, "time"],
        innovations.loc[usable, "standardized_innovation"],
        ".",
        markersize=2,
    )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("Standardized")
    axes[1].set_xlabel("Time [BKJD]")
    axes[1].set_title("Rolling-Scale Standardized Innovations")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_acf_plot(
    innovations: pd.DataFrame,
    selected_order: tuple[int, int, int],
    acf_lags: int,
    path: Path,
) -> None:
    """Save original flux ACF beside selected-model innovation ACF."""

    usable = innovations["innovation_usable"].to_numpy(dtype=bool)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    observed_flux = innovations.loc[innovations["observed_mask"], "normalized_flux"]
    plot_acf(observed_flux, lags=acf_lags, ax=axes[0])
    axes[0].set_title("Original Normalized Flux ACF")

    plot_acf(innovations.loc[usable, "innovation"], lags=acf_lags, ax=axes[1])
    axes[1].set_title(f"Innovation ACF: ARIMA{selected_order}")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def selected_by_mode(scored: pd.DataFrame, mode: str) -> pd.Series | None:
    mode_rows = scored[scored["mode"] == mode]
    if mode_rows.empty:
        return None
    try:
        return select_noise_model(mode_rows)
    except ValueError:
        return None


def coefficient_table(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, candidate in scored.iterrows():
        for coefficient in json.loads(str(candidate.get("coefficient_json", "[]"))):
            rows.append(
                {
                    "quality_policy": candidate["quality_policy"],
                    "mode": candidate["mode"],
                    "order": candidate["order"],
                    **coefficient,
                }
            )
    return pd.DataFrame(rows)


def top_segment_values(regular: pd.DataFrame, max_segments: int) -> list[tuple[int, np.ndarray]]:
    lengths = segment_lengths(regular)
    segments: list[tuple[int, np.ndarray]] = []
    for segment_id in lengths.index[:max_segments]:
        values = regular.loc[regular["segment_id"] == segment_id, "normalized_flux"].to_numpy(dtype=float)
        segments.append((int(segment_id), values))
    return segments


def quality_comparison_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Return the best candidate row for each quality policy and mode."""

    rows = []
    for (quality_policy, mode), group in scored.groupby(["quality_policy", "mode"]):
        usable = group[group["failure_reason"].astype(str) == ""]
        if usable.empty:
            continue
        best = usable.sort_values("adequacy_score", ascending=True).iloc[0].copy()
        best["quality_policy"] = quality_policy
        best["mode"] = mode
        rows.append(best)
    return pd.DataFrame(rows)


def attach_stationarity_context(
    results: pd.DataFrame,
    assessments: dict[tuple[str, str], StationarityAssessment],
) -> pd.DataFrame:
    """Attach stationarity evidence to each candidate without changing rank directly."""

    rows: list[dict[str, object]] = []
    for _, candidate in results.iterrows():
        key = (str(candidate["quality_policy"]), str(candidate["mode"]))
        assessment = assessments[key]
        rows.append(stationarity_candidate_fields(assessment, int(candidate["d"])))
    return pd.concat([results.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def stationarity_assessment_table(
    assessments: dict[tuple[str, str], StationarityAssessment],
) -> pd.DataFrame:
    """Flatten all per-policy/per-mode stationarity assessments for CSV output."""

    rows: list[dict[str, object]] = []
    for (quality_policy, mode), assessment in assessments.items():
        metadata = assessment.preprocessing_summary
        rows.append(
            {
                "quality_policy": quality_policy,
                "mode": mode,
                **stationarity_report_fields(assessment),
                "stationarity_series_representation": metadata.get("stationarity_series_representation", ""),
                "stationarity_gaps_compressed": metadata.get("stationarity_gaps_compressed", False),
                "stationarity_interpolated": metadata.get("stationarity_interpolated", False),
                "stationarity_contiguous_segment_used": metadata.get("stationarity_contiguous_segment_used", False),
                "stationarity_removed_nonfinite": metadata.get("stationarity_removed_nonfinite", 0),
            }
        )
    return pd.DataFrame(rows)


def format_optional_float(value: object, digits: int = 6) -> str:
    """Format nullable numeric values for concise terminal output."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    if not np.isfinite(numeric):
        return "unavailable"
    return f"{numeric:.{digits}g}"


def arima_orders_from_args(args: argparse.Namespace) -> tuple[tuple[int, int, int], ...]:
    """Resolve manual/default/generated ARIMA order choices from CLI args."""

    if args.orders:
        return tuple(args.orders)
    if args.expanded_arima_grid:
        return generate_arima_orders(
            max_p=args.max_p,
            max_d=args.max_d,
            max_q=args.max_q,
            max_total_order=args.max_total_order,
        )
    return DEFAULT_ORDERS


def build_injection_grid(
    frame: pd.DataFrame,
    *,
    depth_grid: tuple[float, ...],
    duration_grid: tuple[int, ...],
    centers_per_duration: int,
    max_segments: int,
) -> list[TransitInjection]:
    """Build a deterministic grid of synthetic transit injections."""

    injections: list[TransitInjection] = []
    for duration in duration_grid:
        centers = choose_injection_centers(
            frame,
            duration_cadences=duration,
            centers_per_segment=centers_per_duration,
            max_segments=max_segments,
        )
        for center in centers:
            for depth in depth_grid:
                injections.append(
                    TransitInjection(
                        center_cadenceno=center,
                        duration_cadences=duration,
                        depth=depth,
                    )
                )
    return injections


def evaluate_top_segment_fits(
    regular: pd.DataFrame,
    orders: tuple[tuple[int, int, int], ...],
    *,
    quality_policy: str,
    max_segments: int,
    test_fraction: float,
    acf_lags: int,
    transit_lag_range: tuple[int, int],
    fit_maxiter: int | None = None,
) -> pd.DataFrame:
    """Run the full candidate grid on the longest usable segments."""

    rows: list[pd.DataFrame] = []
    for segment_rank, (segment_id, values) in enumerate(
        top_segment_values(regular, max_segments),
        start=1,
    ):
        segment_results = evaluate_arima_candidates(
            values,
            orders,
            mode="segment_fit",
            allow_missing=False,
            test_fraction=test_fraction,
            acf_lags=acf_lags,
            transit_lag_range=transit_lag_range,
            fit_maxiter=fit_maxiter,
        )
        segment_results["quality_policy"] = quality_policy
        segment_results["segment_rank"] = segment_rank
        segment_results["segment_id"] = segment_id
        rows.append(segment_results)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def save_transit_preservation_plot(
    frame: pd.DataFrame,
    injected_flux: np.ndarray,
    injected_innovations: np.ndarray,
    injected_standardized: np.ndarray,
    injection: TransitInjection,
    *,
    local_half_width_cadences: int,
    path: Path,
) -> None:
    """Plot the injected event before and after ARIMA."""

    cadence = frame["cadenceno"].to_numpy(dtype=int)
    local = np.abs(cadence - injection.center_cadenceno) <= local_half_width_cadences
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(frame.loc[local, "cadenceno"], frame.loc[local, "normalized_flux"], ".-", ms=3)
    axes[0].plot(frame.loc[local, "cadenceno"], injected_flux[local], ".-", ms=3)
    axes[0].set_ylabel("Flux")
    axes[0].set_title("Original vs Injected Normalized Flux")

    axes[1].plot(frame.loc[local, "cadenceno"], injected_innovations[local], ".-", ms=3)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("Innovation")
    axes[1].set_title("Injected-Series ARIMA Innovations")

    axes[2].plot(frame.loc[local, "cadenceno"], injected_standardized[local], ".-", ms=3)
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_ylabel("Standardized")
    axes[2].set_xlabel("Cadence number")
    axes[2].set_title("Rolling-Scale Standardized Innovations")

    for ax in axes:
        ax.axvspan(
            injection.center_cadenceno - injection.duration_cadences / 2,
            injection.center_cadenceno + injection.duration_cadences / 2,
            color="0.85",
            zorder=-1,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_transformed_template_plot(
    frame: pd.DataFrame,
    injected_flux: np.ndarray,
    box_template: np.ndarray,
    transform: ArimaTemplateTransform,
    injection: TransitInjection,
    *,
    local_half_width_cadences: int,
    path: Path,
) -> None:
    """Plot the unchanged and ARIMA-transformed transit templates."""

    cadence = frame["cadenceno"].to_numpy(dtype=int)
    local = local_window_mask(
        cadence,
        center_cadenceno=injection.center_cadenceno,
        half_width_cadences=local_half_width_cadences,
    )

    def unit_shape(values: np.ndarray) -> np.ndarray:
        shaped = np.asarray(values, dtype=float)
        local_values = shaped[local]
        finite_values = local_values[np.isfinite(local_values)]
        if finite_values.size == 0:
            return np.full(shaped.shape, np.nan)
        scale = np.nanmax(np.abs(finite_values))
        if not np.isfinite(scale) or scale <= 0:
            return np.full(shaped.shape, np.nan)
        return shaped / scale

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(frame.loc[local, "cadenceno"], injected_flux[local], ".-", ms=3)
    axes[0].set_ylabel("Flux")
    axes[0].set_title("Injected Normalized Flux")

    axes[1].plot(
        frame.loc[local, "cadenceno"],
        unit_shape(box_template)[local],
        ".-",
        ms=3,
        label="unchanged box",
    )
    axes[1].plot(
        frame.loc[local, "cadenceno"],
        unit_shape(transform.transformed_template)[local],
        ".-",
        ms=3,
        label="ARIMA-transformed template",
    )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("Unit shape")
    axes[1].set_title("Template Shape In Innovation Space")
    axes[1].legend(loc="best")

    axes[2].plot(
        frame.loc[local, "cadenceno"],
        transform.injected_innovations[local],
        ".-",
        ms=3,
    )
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_ylabel("Innovation")
    axes[2].set_xlabel("Cadence number")
    axes[2].set_title("Fixed-Parameter ARIMA Innovations After Injection")

    for ax in axes:
        ax.axvspan(
            injection.center_cadenceno - injection.duration_cadences / 2,
            injection.center_cadenceno + injection.duration_cadences / 2,
            color="0.85",
            zorder=-1,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _positive_ratio(numerator: float, denominator: float) -> float:
    """Return a ratio only when both matched-filter scores are positive."""

    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def run_transformed_template_match(
    frame: pd.DataFrame,
    values: np.ndarray,
    fitted_model,
    injection: TransitInjection,
    *,
    allow_missing: bool,
    local_half_width_cadences: int,
    scale_window: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, ArimaTemplateTransform]:
    """Compare unchanged-box and ARIMA-transformed-template matched filters."""

    cadence = frame["cadenceno"].to_numpy(dtype=int)
    injected_values, box_template, _ = inject_box_transit(
        values,
        cadence,
        center_cadenceno=injection.center_cadenceno,
        duration_cadences=injection.duration_cadences,
        depth=injection.depth,
    )
    local = local_window_mask(
        cadence,
        center_cadenceno=injection.center_cadenceno,
        half_width_cadences=local_half_width_cadences,
    )

    transform = arima_transformed_template(
        values,
        box_template,
        fitted_model,
        allow_missing=allow_missing,
    )

    # The raw-flux score is the baseline: search the injected light curve with
    # the original, unchanged box template before ARIMA whitening.
    raw_flux_scale = trailing_robust_scale(values, window=scale_window)
    raw_flux_score = matched_filter_statistic(
        injected_values,
        box_template,
        scale=raw_flux_scale,
        usable_mask=local & frame["usable"].to_numpy(dtype=bool),
    )

    # This intentionally incorrect comparison shows what happens if we whiten
    # the light curve but forget to transform the transit template.
    innovation_scale = trailing_robust_scale(transform.base_innovations, window=scale_window)
    unchanged_box_score = matched_filter_statistic(
        transform.injected_innovations,
        box_template,
        scale=innovation_scale,
        usable_mask=local & transform.usable_mask,
    )

    # This is the compatible multi-model-transit-search statistic: both data and template have gone
    # through the same fixed ARIMA innovation filter.
    transformed_template_score = matched_filter_statistic(
        transform.injected_innovations,
        transform.transformed_template,
        scale=innovation_scale,
        usable_mask=local & transform.usable_mask,
    )

    raw_stat = raw_flux_score.statistic
    unchanged_stat = unchanged_box_score.statistic
    transformed_stat = transformed_template_score.statistic

    metrics: dict[str, object] = {
        **injection.to_dict(),
        **raw_flux_score.to_dict("raw_flux_box_"),
        **unchanged_box_score.to_dict("innovation_unchanged_box_"),
        **transformed_template_score.to_dict("innovation_transformed_template_"),
        "transformed_template_minus_raw_flux": float(transformed_stat - raw_stat) if np.isfinite(transformed_stat) and np.isfinite(raw_stat) else float("nan"),
        "transformed_template_minus_unchanged_box": float(transformed_stat - unchanged_stat) if np.isfinite(transformed_stat) and np.isfinite(unchanged_stat) else float("nan"),
        "transformed_template_to_raw_flux_ratio": _positive_ratio(
            transformed_stat,
            raw_stat,
        ),
        "transformed_template_to_unchanged_box_ratio": _positive_ratio(
            transformed_stat,
            unchanged_stat,
        ),
        "transformed_template_positive": bool(np.isfinite(transformed_stat) and transformed_stat > 0),
        "transformed_template_improves_unchanged_box": bool(np.isfinite(transformed_stat) and np.isfinite(unchanged_stat) and transformed_stat > unchanged_stat),
    }
    return metrics, injected_values, box_template, transform


def run_fixed_arima_transit_preservation(
    frame: pd.DataFrame,
    values: np.ndarray,
    fitted_model,
    injection: TransitInjection,
    *,
    allow_missing: bool,
    local_half_width_cadences: int,
    scale_window: int,
) -> tuple[dict[str, object], np.ndarray]:
    """Measure injected-event preservation using fixed selected ARIMA coefficients."""

    cadenceno = frame["cadenceno"].to_numpy(dtype=int)
    injected_values, _, _ = inject_box_transit(
        values,
        cadenceno,
        center_cadenceno=injection.center_cadenceno,
        duration_cadences=injection.duration_cadences,
        depth=injection.depth,
    )
    injected_fit = apply_fitted_arima_filter(
        injected_values,
        fitted_model,
        allow_missing=allow_missing,
    )
    injected_scale = trailing_robust_scale(
        injected_fit.innovations,
        window=scale_window,
    )
    injected_standardized = standardize_innovations(
        injected_fit.innovations,
        injected_scale,
        injected_fit.usable_mask,
    )
    metrics = transit_preservation_metrics(
        cadenceno,
        injected_values,
        injected_fit.innovations,
        injected_standardized,
        injection,
        local_half_width_cadences=local_half_width_cadences,
    )

    depth_retention = float(metrics["depth_retention_fraction"])
    snr_retention = float(metrics["snr_retention_fraction"])
    center_shift = abs(int(metrics["event_center_shift_cadences"]))
    metrics["transit_preservation_failure"] = bool(
        not np.isfinite(depth_retention) or not np.isfinite(snr_retention) or depth_retention < 0.50 or snr_retention < 0.50 or center_shift > max(1, injection.duration_cadences // 2)
    )
    return metrics, injected_values


def summarize_template_scan(scan: pd.DataFrame) -> dict[str, object]:
    """Summarize whether the blind scan recovered the injected center."""

    if scan.empty:
        return {
            "n_trial_centers": 0,
            "best_transformed_template_center_cadenceno": -1,
            "best_transformed_template_statistic": float("nan"),
            "best_transformed_template_center_error_cadences": float("nan"),
            "best_injected_neighborhood_rank": pd.NA,
            "best_injected_neighborhood_statistic": float("nan"),
            "injected_center_recovered_as_best": False,
        }

    score_column = "innovation_transformed_template_statistic"
    ranked = scan.sort_values(score_column, ascending=False, na_position="last")
    best = ranked.iloc[0]

    injected_rows = scan.loc[scan["is_injected_center_neighborhood"].astype(bool)] if "is_injected_center_neighborhood" in scan.columns else pd.DataFrame()
    if injected_rows.empty:
        best_injected_rank = pd.NA
        best_injected_statistic = float("nan")
    else:
        best_injected = injected_rows.sort_values(
            score_column,
            ascending=False,
            na_position="last",
        ).iloc[0]
        best_injected_rank = best_injected["innovation_transformed_template_rank"]
        best_injected_statistic = float(best_injected[score_column])

    center_error = int(best["center_offset_cadences"]) if "center_offset_cadences" in scan.columns else float("nan")
    recovered_as_best = bool("is_injected_center_neighborhood" in scan.columns and bool(best["is_injected_center_neighborhood"]))

    return {
        "n_trial_centers": int(len(scan)),
        "best_transformed_template_center_cadenceno": int(best["trial_center_cadenceno"]),
        "best_transformed_template_statistic": float(best[score_column]),
        "best_transformed_template_center_error_cadences": center_error,
        "best_injected_neighborhood_rank": best_injected_rank,
        "best_injected_neighborhood_statistic": best_injected_statistic,
        "injected_center_recovered_as_best": recovered_as_best,
    }


def run_multi_injection_recovery(
    frame: pd.DataFrame,
    values: np.ndarray,
    fitted_model,
    injections: list[TransitInjection],
    *,
    allow_missing: bool,
    local_half_width_cadences: int,
    scale_window: int,
    scan_stride: int,
    scan_max_centers: int | None,
) -> pd.DataFrame:
    """Run selected-model preservation and blind recovery over many injections."""

    rows: list[dict[str, object]] = []
    cadence = frame["cadenceno"].to_numpy(dtype=int)
    usable = frame["usable"].to_numpy(dtype=bool)
    for injection_id, injection in enumerate(injections, start=1):
        preservation, injected_values = run_fixed_arima_transit_preservation(
            frame,
            values,
            fitted_model,
            injection,
            allow_missing=allow_missing,
            local_half_width_cadences=local_half_width_cadences,
            scale_window=scale_window,
        )
        transformed_match, _, _, _ = run_transformed_template_match(
            frame,
            values,
            fitted_model,
            injection,
            allow_missing=allow_missing,
            local_half_width_cadences=local_half_width_cadences,
            scale_window=scale_window,
        )
        trial_centers = select_trial_centers(
            cadence,
            usable,
            stride=scan_stride,
            max_centers=scan_max_centers,
            required_centers=(injection.center_cadenceno,),
        )
        scan = scan_arima_transformed_template(
            cadence,
            values,
            injected_values,
            fitted_model,
            trial_centers,
            duration_cadences=injection.duration_cadences,
            depth=injection.depth,
            local_half_width_cadences=local_half_width_cadences,
            scale_window=scale_window,
            allow_missing=allow_missing,
            usable_mask=usable,
            injected_center_cadenceno=injection.center_cadenceno,
            injected_neighborhood_cadences=max(1, injection.duration_cadences // 2),
        )
        scan_summary = summarize_template_scan(scan)
        rows.append(
            {
                "injection_id": injection_id,
                **injection.to_dict(),
                **{key: value for key, value in preservation.items() if key not in {"center_cadenceno", "duration_cadences", "depth"}},
                "raw_flux_box_statistic": transformed_match["raw_flux_box_statistic"],
                "innovation_unchanged_box_statistic": transformed_match["innovation_unchanged_box_statistic"],
                "innovation_transformed_template_statistic": transformed_match["innovation_transformed_template_statistic"],
                "transformed_template_improves_unchanged_box": transformed_match["transformed_template_improves_unchanged_box"],
                **scan_summary,
            }
        )
    return pd.DataFrame(rows)


def summarize_multi_injection_recovery(recovery: pd.DataFrame) -> dict[str, object]:
    """Summarize selected-model recovery over a synthetic injection grid."""

    if recovery.empty:
        return {
            "n_injections": 0,
            "rank1_recovery_rate": float("nan"),
            "rank3_recovery_rate": float("nan"),
            "median_best_injected_neighborhood_rank": float("nan"),
            "median_transformed_template_statistic": float("nan"),
        }

    ranks = pd.to_numeric(recovery["best_injected_neighborhood_rank"], errors="coerce")
    transformed_stats = pd.to_numeric(
        recovery["innovation_transformed_template_statistic"],
        errors="coerce",
    )
    recovered_as_best = recovery["injected_center_recovered_as_best"].astype(bool)

    return {
        "n_injections": int(len(recovery)),
        "unique_depths": sorted(float(value) for value in recovery["depth"].unique()),
        "unique_durations_cadences": sorted(int(value) for value in recovery["duration_cadences"].unique()),
        "rank1_recovery_rate": float(recovered_as_best.mean()),
        "rank3_recovery_rate": float((ranks <= 3).mean()),
        "median_best_injected_neighborhood_rank": float(ranks.median()),
        "median_transformed_template_statistic": float(transformed_stats.median()),
        "minimum_transformed_template_statistic": float(transformed_stats.min()),
        "transformed_template_improvement_rate": float(recovery["transformed_template_improves_unchanged_box"].astype(bool).mean()),
        "transit_preservation_failure_rate": float(recovery["transit_preservation_failure"].astype(bool).mean()),
    }


def save_template_scan_plot(
    scan: pd.DataFrame,
    injection: TransitInjection,
    *,
    path: Path,
) -> None:
    """Plot blind matched-filter statistics over scanned trial centers."""

    fig, ax = plt.subplots(figsize=(12, 4))
    x = scan["trial_center_cadenceno"]
    ax.plot(
        x,
        scan["raw_flux_box_statistic"],
        ".-",
        markersize=3,
        linewidth=1,
        label="raw flux + box",
    )
    ax.plot(
        x,
        scan["innovation_unchanged_box_statistic"],
        ".-",
        markersize=3,
        linewidth=1,
        label="ARIMA innovations + unchanged box",
    )
    ax.plot(
        x,
        scan["innovation_transformed_template_statistic"],
        ".-",
        markersize=3,
        linewidth=1,
        label="ARIMA innovations + transformed template",
    )
    ax.axvline(
        injection.center_cadenceno,
        color="black",
        linestyle="--",
        linewidth=1,
        label="injected center",
    )
    ax.axvspan(
        injection.center_cadenceno - max(1, injection.duration_cadences // 2),
        injection.center_cadenceno + max(1, injection.duration_cadences // 2),
        color="0.90",
        zorder=-1,
    )
    ax.set_xlabel("Trial center cadence")
    ax.set_ylabel("Matched-filter statistic")
    ax.set_title("Blind Single-Transit Template Scan")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def phase1_completion_report(
    *,
    selected: pd.Series,
    preprocessing_summary: dict[str, object],
    stationarity_assessment: StationarityAssessment,
    stability_summaries: dict[str, dict[str, object]],
    preservation_metrics: dict[str, object],
    transformed_template_metrics: dict[str, object],
    template_scan_summary: dict[str, object],
    multi_injection_summary: dict[str, object],
    scored: pd.DataFrame,
    regular: pd.DataFrame,
    innovations: pd.DataFrame,
) -> dict[str, object]:
    """Build a reproducible Phase 1 completion audit.

    A criterion passing here means the pipeline has implemented and recorded the
    required check. Scientific warnings are kept separate because a residual
    failure is a result to document, not missing code.
    """

    selected_failure_reason = str(selected.get("failure_reason", ""))
    residual_autocorrelation_remaining = bool(selected.get("residual_autocorrelation_remaining", False))
    variance_instability = bool(selected.get("variance_instability", False))
    transit_preservation_failure = bool(preservation_metrics.get("transit_preservation_failure", False))
    differencing_requires_review = bool(selected.get("differencing_requires_review", False))
    stationarity_fields = stationarity_report_fields(stationarity_assessment)
    selected_acf_1 = float(selected.get("residual_acf_lag_1", np.nan))
    selected_max_acf = float(selected.get("max_abs_residual_acf", np.nan))
    selected_max_short_acf = float(selected.get("max_abs_residual_acf_1_24", np.nan))
    selected_transit_acf = float(selected.get("max_abs_residual_acf_transit_lags", np.nan))
    lag1_materially_negative = bool(np.isfinite(selected_acf_1) and selected_acf_1 < -0.10)
    short_lag_concentrated = bool(np.isfinite(selected_max_short_acf) and np.isfinite(selected_max_acf) and selected_max_short_acf >= 0.90 * selected_max_acf)
    transit_lag_overlap = bool(np.isfinite(selected_transit_acf) and selected_transit_acf > 0.10)
    selected_alignment = str(selected.get("candidate_differencing_alignment", "unresolved"))
    selected_reason_codes = list(stationarity_assessment.reason_codes)
    if bool(selected.get("differenced_model", False)) and selected_alignment == "conflicts_with_stationarity_evidence" and stationarity_assessment.recommended_d == 0:
        selected_reason_codes.append("SELECTED_DIFFERENCED_MODEL_CONFLICTS_WITH_D0_EVIDENCE")
    elif bool(selected.get("differenced_model", False)) and selected_alignment == "unresolved":
        selected_reason_codes.append("SELECTED_DIFFERENCED_MODEL_HAS_UNRESOLVED_STATIONARITY_EVIDENCE")
    transformed_improves = bool(
        transformed_template_metrics.get(
            "transformed_template_improves_unchanged_box",
            False,
        )
    )
    scan_recovers = bool(template_scan_summary.get("injected_center_recovered_as_best", False))
    rank1_recovery_rate = float(multi_injection_summary.get("rank1_recovery_rate", 0.0))
    selected_constraints_pass = not any(
        [
            bool(selected_failure_reason),
            bool(selected.get("statistical_validity_failed", False)),
            bool(selected.get("whitening_constraint_failed", False)),
            bool(selected.get("variance_constraint_failed", False)),
            bool(selected.get("transit_preservation_constraint_failed", False)),
            differencing_requires_review,
        ]
    )

    required_criteria = {
        "preprocessing_is_explicit_and_reproducible": bool(preprocessing_summary.get("quality_policy") and preprocessing_summary.get("normalization_fit_fraction") is not None),
        "missing_cadences_remain_explicit": bool(
            "row_present" in regular.columns and "gap_reason" in regular.columns and int(preprocessing_summary.get("n_cadence_grid", 0)) >= int(preprocessing_summary.get("n_raw", 0))
        ),
        "normalization_is_leakage_free": bool(float(preprocessing_summary.get("normalization_fit_fraction", 1.0)) < 1.0),
        "variance_behavior_is_characterized": bool("arch_pvalue" in selected.index and "rolling_var_max_to_median" in selected.index and pd.notna(selected.get("rolling_var_max_to_median"))),
        "order_selection_runs_across_folds_and_segments": bool(
            stability_summaries.get("chronological_prefix", {}).get("n_successful_runs", 0) and stability_summaries.get("segment", {}).get("n_successful_runs", 0)
        ),
        "full_quarter_nan_gap_model_is_evaluated": bool(((scored["mode"] == "full_gap") & (scored["n_nan_gaps"] > 0)).any()),
        "simple_baselines_are_compared": bool({"best_baseline_RMSE", "beats_best_baseline_RMSE", "beats_best_baseline_MAE"}.issubset(scored.columns)),
        "coefficient_boundary_diagnostics_are_recorded": bool({"coefficient_json", "boundary_coefficient_count", "boundary_coefficients_json"}.issubset(scored.columns)),
        "residual_failure_modes_are_documented": bool(
            {
                "residual_autocorrelation_remaining",
                "variance_instability",
                "outlier_heavy",
                "non_converged",
                "boundary_coefficients",
            }.issubset(scored.columns)
        ),
        "standardized_innovations_are_saved": bool({"innovation", "innovation_scale", "standardized_innovation"}.issubset(innovations.columns)),
        "injected_transit_preservation_is_measured": bool(
            {
                "depth_retention_fraction",
                "snr_retention_fraction",
                "transit_preservation_failure",
            }.issubset(preservation_metrics)
        ),
        "arima_transformed_template_match_is_measured": bool(
            {
                "innovation_transformed_template_statistic",
                "innovation_unchanged_box_statistic",
            }.issubset(transformed_template_metrics)
        ),
        "blind_single_event_scan_is_measured": bool(int(template_scan_summary.get("n_trial_centers", 0)) > 1 and template_scan_summary.get("best_injected_neighborhood_rank") is not None),
        "multi_injection_recovery_is_measured": bool(int(multi_injection_summary.get("n_injections", 0)) > 0),
        "hierarchical_model_selection_is_explicit": bool(
            {
                "selection_rank",
                "mode_selection_rank",
                "selection_status",
                "statistical_validity_failed",
                "whitening_constraint_failed",
                "variance_constraint_failed",
                "transit_preservation_constraint_failed",
                "fit_metrics_trustworthy",
                "differenced_model",
                "differencing_requires_review",
            }.issubset(scored.columns)
        ),
        "stationarity_diagnostics_are_recorded": bool(stationarity_fields["stationarity_diagnostics_available"] and "candidate_differencing_alignment" in scored.columns),
    }

    scientific_findings = {
        "selected_model_has_candidate_failure_reason": bool(selected_failure_reason),
        "selected_model_selection_status": str(selected.get("selection_status", "")),
        "selected_model_statistical_validity_failed": bool(selected.get("statistical_validity_failed", False)),
        "selected_model_whitening_constraint_failed": bool(selected.get("whitening_constraint_failed", False)),
        "selected_model_variance_constraint_failed": bool(selected.get("variance_constraint_failed", False)),
        "selected_model_transit_preservation_constraint_failed": bool(selected.get("transit_preservation_constraint_failed", False)),
        "selected_model_constraints_passed": selected_constraints_pass,
        "selected_model_fit_metrics_trustworthy": bool(selected.get("fit_metrics_trustworthy", False)),
        "selected_model_uses_differencing": bool(selected.get("differenced_model", False)),
        "selected_model_differencing_requires_review": differencing_requires_review,
        "selected_model_differencing_alignment": selected_alignment,
        "selected_model_differencing_statistically_supported": bool(stationarity_assessment.differencing_statistically_supported and bool(selected.get("differenced_model", False))),
        "selected_model_stationarity_reason_codes": selected_reason_codes,
        "original_series_stationarity_conclusion": str(stationarity_assessment.original_series_conclusion),
        "recommended_d": stationarity_assessment.recommended_d,
        "lag1_residual_dependence_materially_negative": lag1_materially_negative,
        "remaining_correlation_concentrated_at_short_lags": short_lag_concentrated,
        "remaining_correlation_overlaps_transit_duration_lags": transit_lag_overlap,
        "residual_autocorrelation_remaining": residual_autocorrelation_remaining,
        "variance_instability": variance_instability,
        "transit_preservation_failure": transit_preservation_failure,
        "transformed_template_improves_unchanged_box": transformed_improves,
        "blind_scan_recovers_injected_center_as_best": scan_recovers,
        "multi_injection_rank1_recovery_rate": rank1_recovery_rate,
        "multi_injection_rank3_recovery_rate": float(multi_injection_summary.get("rank3_recovery_rate", 0.0)),
        "multi_injection_transit_preservation_failure_rate": float(multi_injection_summary.get("transit_preservation_failure_rate", 0.0)),
    }
    phase1_engineering_complete = all(required_criteria.values())
    phase1_scientific_ready_for_phase2 = bool(phase1_engineering_complete and selected_constraints_pass and transformed_improves and scan_recovers and rank1_recovery_rate >= 0.50)

    return {
        "phase": "multi-model-transit-search Phase 1: single-target ARIMA transformed-template prototype",
        "phase1_engineering_complete": phase1_engineering_complete,
        "phase1_scientific_ready_for_phase2": phase1_scientific_ready_for_phase2,
        "selected_quality_policy": str(selected.get("quality_policy", "")),
        "selected_mode": str(selected.get("mode", "")),
        "selected_order": str(selected.get("order", "")),
        "selected_global_rank": float(selected.get("selection_rank", np.nan)),
        "selected_mode_rank": float(selected.get("mode_selection_rank", np.nan)),
        "stationarity": {
            **stationarity_fields,
            "selected_model_differencing_alignment": selected_alignment,
            "selected_model_differencing_requires_review": differencing_requires_review,
            "selected_model_stationarity_reason_codes": selected_reason_codes,
            "assessment": stationarity_assessment.to_dict(),
        },
        "required_criteria": required_criteria,
        "scientific_findings": scientific_findings,
        "interpretation": [
            "Phase 1 is complete as a single-target prototype when all required criteria are true.",
            "ARIMA selection is hierarchical: validity, whitening, variance stability, transit preservation, then forecasting tie-breakers.",
            "Non-converged candidates may report fit metrics, but those metrics are not trustworthy for selection.",
            "ADF and KPSS stationarity diagnostics are advisory constraints for differencing, not proof that an ARIMA model is suitable for transit detection.",
            "Differenced models require review unless the exact modelled series has joint ADF/KPSS support for ordinary differencing.",
            "Residual autocorrelation, variance instability, or transit-preservation failure remain documented scientific limitations.",
            "Scientific readiness for scale-up requires the selected model to pass those constraints, not only recover one injected event.",
        ],
    }


def run_transit_preservation(
    frame: pd.DataFrame,
    values: np.ndarray,
    order: tuple[int, int, int],
    *,
    mode: str,
    allow_missing: bool,
    depth: float,
    duration_cadences: int,
    local_half_width_cadences: int,
    scale_window: int,
    fit_maxiter: int | None = None,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, TransitInjection]:
    """Inject one transit, run ARIMA, and measure morphology/detectability."""

    injection_center = choose_injection_center(
        frame,
        duration_cadences=duration_cadences,
    )
    injection = TransitInjection(
        center_cadenceno=injection_center,
        duration_cadences=duration_cadences,
        depth=depth,
    )
    cadenceno = frame["cadenceno"].to_numpy(dtype=int)
    injected_values, _, _ = inject_box_transit(
        values,
        cadenceno,
        center_cadenceno=injection.center_cadenceno,
        duration_cadences=injection.duration_cadences,
        depth=injection.depth,
    )
    injected_fit = fit_arima_model(
        injected_values,
        order,
        allow_missing=allow_missing,
        mode=mode,
        fit_maxiter=fit_maxiter,
    )
    injected_scale = trailing_robust_scale(
        injected_fit.innovations,
        window=scale_window,
    )
    injected_standardized = standardize_innovations(
        injected_fit.innovations,
        injected_scale,
        injected_fit.usable_mask,
    )
    metrics = transit_preservation_metrics(
        cadenceno,
        injected_values,
        injected_fit.innovations,
        injected_standardized,
        injection,
        local_half_width_cadences=local_half_width_cadences,
    )

    depth_retention = float(metrics["depth_retention_fraction"])
    snr_retention = float(metrics["snr_retention_fraction"])
    center_shift = abs(int(metrics["event_center_shift_cadences"]))
    metrics["transit_preservation_failure"] = bool(
        not np.isfinite(depth_retention) or not np.isfinite(snr_retention) or depth_retention < 0.50 or snr_retention < 0.50 or center_shift > max(1, duration_cadences // 2)
    )
    return metrics, injected_values, injected_fit.innovations, injected_standardized, injection


def transit_preservation_table(
    candidates: pd.DataFrame,
    regular_by_policy: dict[str, pd.DataFrame],
    *,
    depth: float,
    duration_cadences: int,
    local_half_width_cadences: int,
    scale_window: int,
    fit_maxiter: int | None = None,
) -> pd.DataFrame:
    """Evaluate transit preservation for each quality policy, mode, and order."""

    rows: list[dict[str, object]] = []
    for _, candidate in candidates.iterrows():
        quality_policy = str(candidate["quality_policy"])
        mode = str(candidate["mode"])
        order = (int(candidate["p"]), int(candidate["d"]), int(candidate["q"]))
        regular = regular_by_policy[quality_policy]

        if mode == "full_gap":
            frame = regular
            values = regular["normalized_flux"].to_numpy(dtype=float)
            allow_missing = True
        elif mode == "longest_segment":
            frame = longest_contiguous_segment(regular)
            values = frame["normalized_flux"].to_numpy(dtype=float)
            allow_missing = False
        else:
            continue

        row: dict[str, object] = {
            "quality_policy": quality_policy,
            "mode": mode,
            "order": str(candidate["order"]),
            "p": order[0],
            "d": order[1],
            "q": order[2],
        }

        if str(candidate.get("failure_reason", "")):
            row.update(
                {
                    "transit_preservation_failure": True,
                    "transit_preservation_failure_reason": "base_candidate_failed",
                }
            )
            rows.append(row)
            continue

        try:
            metrics, _, _, _, _ = run_transit_preservation(
                frame,
                values,
                order,
                mode=mode,
                allow_missing=allow_missing,
                depth=depth,
                duration_cadences=duration_cadences,
                local_half_width_cadences=local_half_width_cadences,
                scale_window=scale_window,
                fit_maxiter=fit_maxiter,
            )
            row.update(metrics)
            row["transit_preservation_failure_reason"] = ""
        except Exception as exc:  # noqa: BLE001 - preserve candidate-level failure context.
            row.update(
                {
                    "transit_preservation_failure": True,
                    "transit_preservation_failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


def main(args=None):
    args = args or build_parser().parse_args()
    if not 0.0 < args.stationarity_alpha < 1.0:
        raise ValueError("--stationarity-alpha must be between 0 and 1.")
    if args.stationarity_min_observations < 8:
        raise ValueError("--stationarity-min-observations must be at least 8.")
    if args.fit_maxiter is not None and args.fit_maxiter <= 0:
        raise ValueError("--fit-maxiter must be positive when provided.")
    if args.transit_lag_min < 1 or args.transit_lag_max < args.transit_lag_min:
        raise ValueError("--transit-lag-min must be >= 1 and --transit-lag-max must be >= --transit-lag-min.")

    orders = arima_orders_from_args(args)
    quality_policies = tuple(args.quality_policies) if args.quality_policies else DEFAULT_QUALITY_POLICIES
    injection_depth_grid = args.injection_depth_grid or (args.injection_depth,)
    injection_duration_grid = args.injection_duration_grid or (args.injection_duration_cadences,)
    transit_lag_range = (int(args.transit_lag_min), int(args.transit_lag_max))

    metrics_dir = args.output_dir / "metrics"
    figures_dir = args.output_dir / "figures"
    processed_dir = args.output_dir / "processed"
    for directory in (metrics_dir, figures_dir, processed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Loading Kepler PDCSAP target={args.target_id} quarter={args.quarter}", flush=True)
    light_curve = load_kepler_pdcsap(args.target_id, args.quarter)
    raw = light_curve.to_dataframe()
    print(f"Loaded {len(raw)} raw cadences", flush=True)

    regular_by_policy: dict[str, pd.DataFrame] = {}
    summary_by_policy: dict[str, dict[str, object]] = {}
    stationarity_assessments: dict[tuple[str, str], StationarityAssessment] = {}
    result_frames: list[pd.DataFrame] = []
    for quality_policy in quality_policies:
        print(f"Evaluating quality policy: {quality_policy}", flush=True)
        regular, preprocessing_summary = preprocess_pdcsap_light_curve(
            raw,
            quality_policy=quality_policy,
            require_finite_flux_error=args.require_finite_flux_error,
            normalization_fit_fraction=1.0 - args.test_fraction,
        )
        regular_by_policy[quality_policy] = regular
        summary_by_policy[quality_policy] = preprocessing_summary.to_dict()

        full_values = regular["normalized_flux"].to_numpy(dtype=float)
        stationarity_assessments[(quality_policy, "full_gap")] = assess_stationarity(
            full_values,
            modelling_mode="full_gap",
            preprocessing_summary=summary_by_policy[quality_policy],
            alpha=args.stationarity_alpha,
            adf_regression=args.adf_regression,
            adf_autolag=args.adf_autolag,
            kpss_regression=args.kpss_regression,
            kpss_nlags=args.kpss_nlags,
            min_observations=args.stationarity_min_observations,
            gaps_compressed=bool(np.isnan(full_values).any()),
            interpolated=False,
            contiguous_segment_used=False,
            series_representation="finite_observed_values_from_full_gap_series",
        )
        full_results = evaluate_arima_candidates(
            full_values,
            orders,
            mode="full_gap",
            allow_missing=True,
            test_fraction=args.test_fraction,
            acf_lags=args.acf_lags,
            transit_lag_range=transit_lag_range,
            fit_maxiter=args.fit_maxiter,
        )
        full_results["quality_policy"] = quality_policy
        result_frames.append(full_results)
        print(f"Finished full-gap candidates for {quality_policy}", flush=True)

        try:
            longest_segment = longest_contiguous_segment(regular)
            segment_values = longest_segment["normalized_flux"].to_numpy(dtype=float)
            stationarity_assessments[(quality_policy, "longest_segment")] = assess_stationarity(
                segment_values,
                modelling_mode="longest_segment",
                preprocessing_summary=summary_by_policy[quality_policy],
                alpha=args.stationarity_alpha,
                adf_regression=args.adf_regression,
                adf_autolag=args.adf_autolag,
                kpss_regression=args.kpss_regression,
                kpss_nlags=args.kpss_nlags,
                min_observations=args.stationarity_min_observations,
                gaps_compressed=False,
                interpolated=False,
                contiguous_segment_used=True,
                series_representation="finite_values_from_longest_contiguous_segment",
            )
            segment_results = evaluate_arima_candidates(
                segment_values,
                orders,
                mode="longest_segment",
                allow_missing=False,
                test_fraction=args.test_fraction,
                acf_lags=args.acf_lags,
                transit_lag_range=transit_lag_range,
                fit_maxiter=args.fit_maxiter,
            )
            segment_results["quality_policy"] = quality_policy
            result_frames.append(segment_results)
            print(f"Finished longest-segment candidates for {quality_policy}", flush=True)
        except ValueError:
            pass

    results = attach_stationarity_context(pd.concat(result_frames, ignore_index=True), stationarity_assessments)
    print("Evaluating transit preservation for candidate rows", flush=True)
    preservation_by_candidate = transit_preservation_table(
        results,
        regular_by_policy,
        depth=args.injection_depth,
        duration_cadences=args.injection_duration_cadences,
        local_half_width_cadences=args.injection_local_half_width_cadences,
        scale_window=args.scale_window,
        fit_maxiter=args.fit_maxiter,
    )
    results = results.merge(
        preservation_by_candidate,
        on=["quality_policy", "mode", "order", "p", "d", "q"],
        how="left",
    )
    scored = score_arima_candidates(results)

    selected_full = selected_by_mode(scored, "full_gap")
    selected_segment = selected_by_mode(scored, "longest_segment")
    selected = selected_full if selected_full is not None else select_noise_model(scored)
    selected_order = order_from_row(selected)
    selected_mode = str(selected["mode"])
    selected_quality_policy = str(selected["quality_policy"])
    print(f"Selected {selected_quality_policy} {selected_mode} {selected['order']}", flush=True)
    selected_stationarity_assessment = stationarity_assessments[(selected_quality_policy, selected_mode)]
    regular = regular_by_policy[selected_quality_policy]
    full_values = regular["normalized_flux"].to_numpy(dtype=float)
    longest_segment = longest_contiguous_segment(regular)
    segment_values = longest_segment["normalized_flux"].to_numpy(dtype=float)
    selected_values = full_values if selected_mode == "full_gap" else segment_values
    selected_frame = regular if selected_mode == "full_gap" else longest_segment
    selected_fit = fit_arima_model(
        selected_values,
        selected_order,
        allow_missing=selected_mode == "full_gap",
        mode=selected_mode,
        fit_maxiter=args.fit_maxiter,
    )

    gap_sensitive = (
        selected_full is not None
        and selected_segment is not None
        and (str(selected_full["order"]) != str(selected_segment["order"]) or str(selected_full["quality_policy"]) != str(selected_segment["quality_policy"]))
    )
    scored["gap_sensitive"] = (scored["mode"] == "full_gap") & gap_sensitive

    innovations = selected_frame[["cadenceno", "time", "normalized_flux", "observed_mask", "usable", "gap_reason"]].copy()
    innovations["one_step_prediction"] = selected_fit.one_step_prediction
    innovations["innovation"] = selected_fit.innovations
    innovations["innovation_usable"] = selected_fit.usable_mask
    innovations["innovation_scale"] = trailing_robust_scale(
        selected_fit.innovations,
        window=args.scale_window,
    )
    innovations["standardized_innovation"] = standardize_innovations(
        selected_fit.innovations,
        innovations["innovation_scale"].to_numpy(dtype=float),
        selected_fit.usable_mask,
    )

    prefix_fractions = np.linspace(0.55, 1.0, max(args.stability_folds, 1))
    print("Running stability checks", flush=True)
    fold_stability = chronological_prefix_stability(
        full_values,
        orders,
        mode="full_gap",
        allow_missing=True,
        test_fraction=args.test_fraction,
        acf_lags=args.acf_lags,
        prefix_fractions=tuple(float(value) for value in prefix_fractions),
        fit_maxiter=args.fit_maxiter,
    )
    segment_stability_table = segment_stability(
        top_segment_values(regular, args.stability_segments),
        orders,
        test_fraction=args.test_fraction,
        acf_lags=args.acf_lags,
        max_segments=args.stability_segments,
        fit_maxiter=args.fit_maxiter,
    )
    stability = pd.concat([fold_stability, segment_stability_table], ignore_index=True)
    stability_summaries = {
        "chronological_prefix": summarize_stability(fold_stability).to_dict(),
        "segment": summarize_stability(segment_stability_table).to_dict(),
    }
    segment_fit_results = evaluate_top_segment_fits(
        regular,
        orders,
        quality_policy=selected_quality_policy,
        max_segments=args.stability_segments,
        test_fraction=args.test_fraction,
        acf_lags=args.acf_lags,
        transit_lag_range=transit_lag_range,
        fit_maxiter=args.fit_maxiter,
    )
    print("Running transformed-template diagnostics", flush=True)

    (
        preservation_metrics,
        injected_values,
        injected_innovations,
        injected_standardized,
        injection,
    ) = run_transit_preservation(
        selected_frame,
        selected_values,
        selected_order,
        mode=selected_mode,
        allow_missing=selected_mode == "full_gap",
        depth=args.injection_depth,
        duration_cadences=args.injection_duration_cadences,
        local_half_width_cadences=args.injection_local_half_width_cadences,
        scale_window=args.scale_window,
    )
    preservation_metrics.update(
        {
            "quality_policy": selected_quality_policy,
            "mode": selected_mode,
            "order": str(selected_order),
        }
    )
    (
        transformed_template_metrics,
        template_injected_values,
        box_template,
        template_transform,
    ) = run_transformed_template_match(
        selected_frame,
        selected_values,
        selected_fit,
        injection,
        allow_missing=selected_mode == "full_gap",
        local_half_width_cadences=args.injection_local_half_width_cadences,
        scale_window=args.scale_window,
    )
    transformed_template_metrics.update(
        {
            "quality_policy": selected_quality_policy,
            "mode": selected_mode,
            "order": str(selected_order),
        }
    )
    preservation_metrics.update(
        {
            "innovation_transformed_template_statistic": transformed_template_metrics["innovation_transformed_template_statistic"],
            "innovation_unchanged_box_statistic": transformed_template_metrics["innovation_unchanged_box_statistic"],
            "raw_flux_box_statistic": transformed_template_metrics["raw_flux_box_statistic"],
            "transformed_template_improves_unchanged_box": transformed_template_metrics["transformed_template_improves_unchanged_box"],
        }
    )
    trial_centers = select_trial_centers(
        selected_frame["cadenceno"].to_numpy(dtype=int),
        selected_frame["usable"].to_numpy(dtype=bool),
        stride=args.scan_stride,
        max_centers=args.scan_max_centers,
        required_centers=(injection.center_cadenceno,),
    )
    template_scan = scan_arima_transformed_template(
        selected_frame["cadenceno"].to_numpy(dtype=int),
        selected_values,
        injected_values,
        selected_fit,
        trial_centers,
        duration_cadences=injection.duration_cadences,
        depth=injection.depth,
        local_half_width_cadences=args.injection_local_half_width_cadences,
        scale_window=args.scale_window,
        allow_missing=selected_mode == "full_gap",
        usable_mask=selected_frame["usable"].to_numpy(dtype=bool),
        injected_center_cadenceno=injection.center_cadenceno,
        injected_neighborhood_cadences=max(1, injection.duration_cadences // 2),
    )
    template_scan_summary = summarize_template_scan(template_scan)
    template_scan_summary.update(
        {
            "quality_policy": selected_quality_policy,
            "mode": selected_mode,
            "order": str(selected_order),
            "scan_stride": args.scan_stride,
            "scan_max_centers": args.scan_max_centers,
        }
    )
    injection_grid = build_injection_grid(
        selected_frame,
        depth_grid=injection_depth_grid,
        duration_grid=injection_duration_grid,
        centers_per_duration=args.injection_centers_per_duration,
        max_segments=args.injection_max_segments,
    )
    multi_injection_recovery = run_multi_injection_recovery(
        selected_frame,
        selected_values,
        selected_fit,
        injection_grid,
        allow_missing=selected_mode == "full_gap",
        local_half_width_cadences=args.injection_local_half_width_cadences,
        scale_window=args.scale_window,
        scan_stride=args.scan_stride,
        scan_max_centers=args.scan_max_centers,
    )
    multi_injection_summary = summarize_multi_injection_recovery(
        multi_injection_recovery,
    )
    multi_injection_summary.update(
        {
            "quality_policy": selected_quality_policy,
            "mode": selected_mode,
            "order": str(selected_order),
            "scan_stride": args.scan_stride,
            "scan_max_centers": args.scan_max_centers,
        }
    )
    phase1_report = phase1_completion_report(
        selected=selected,
        preprocessing_summary=summary_by_policy[selected_quality_policy],
        stationarity_assessment=selected_stationarity_assessment,
        stability_summaries=stability_summaries,
        preservation_metrics=preservation_metrics,
        transformed_template_metrics=transformed_template_metrics,
        template_scan_summary=template_scan_summary,
        multi_injection_summary=multi_injection_summary,
        scored=scored,
        regular=regular,
        innovations=innovations,
    )

    prefix = f"kic_{str(args.target_id).replace('KIC', '').strip()}_q{args.quarter}"
    results_path = metrics_dir / f"{prefix}_arima_candidates.csv"
    quality_comparison_path = metrics_dir / f"{prefix}_quality_comparison.csv"
    stationarity_path = metrics_dir / f"{prefix}_stationarity_diagnostics.csv"
    summary_path = metrics_dir / f"{prefix}_preprocessing_summary.json"
    stability_path = metrics_dir / f"{prefix}_order_stability.csv"
    stability_summary_path = metrics_dir / f"{prefix}_order_stability_summary.json"
    segment_fits_path = metrics_dir / f"{prefix}_segment_fits.csv"
    coefficient_path = metrics_dir / f"{prefix}_coefficient_diagnostics.csv"
    preservation_path = metrics_dir / f"{prefix}_transit_preservation.csv"
    transformed_template_path = metrics_dir / f"{prefix}_transformed_template_match.csv"
    template_scan_path = metrics_dir / f"{prefix}_template_scan.csv"
    template_scan_summary_path = metrics_dir / f"{prefix}_template_scan_summary.csv"
    multi_injection_path = metrics_dir / f"{prefix}_multi_injection_recovery.csv"
    multi_injection_summary_path = metrics_dir / f"{prefix}_multi_injection_recovery_summary.json"
    phase1_completion_path = metrics_dir / f"{prefix}_phase1_completion.json"
    preservation_by_candidate_path = metrics_dir / f"{prefix}_transit_preservation_by_candidate.csv"
    regular_path = processed_dir / f"{prefix}_regularized_light_curve.parquet"
    innovations_path = processed_dir / f"{prefix}_innovations.parquet"

    scored.to_csv(results_path, index=False)
    quality_comparison_table(scored).to_csv(quality_comparison_path, index=False)
    stationarity_assessment_table(stationarity_assessments).to_csv(stationarity_path, index=False)
    coefficient_table(scored).to_csv(coefficient_path, index=False)
    preservation_by_candidate.to_csv(preservation_by_candidate_path, index=False)
    stability.to_csv(stability_path, index=False)
    segment_fit_results.to_csv(segment_fits_path, index=False)
    regular.to_parquet(regular_path, index=False)
    innovations.to_parquet(innovations_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "selected_quality_policy": selected_quality_policy,
                "policies": summary_by_policy,
            },
            indent=2,
        )
        + "\n"
    )
    stability_summary_path.write_text(json.dumps(stability_summaries, indent=2) + "\n")
    pd.DataFrame([preservation_metrics]).to_csv(preservation_path, index=False)
    pd.DataFrame([transformed_template_metrics]).to_csv(
        transformed_template_path,
        index=False,
    )
    template_scan.to_csv(template_scan_path, index=False)
    pd.DataFrame([template_scan_summary]).to_csv(template_scan_summary_path, index=False)
    multi_injection_recovery.to_csv(multi_injection_path, index=False)
    multi_injection_summary_path.write_text(json.dumps(multi_injection_summary, indent=2) + "\n")
    phase1_completion_path.write_text(json.dumps(phase1_report, indent=2) + "\n")

    save_flux_plot(regular, figures_dir / f"{prefix}_normalized_flux.png")
    save_gap_plot(regular, figures_dir / f"{prefix}_cadence_gaps.png")
    save_innovation_plot(innovations, selected_order, figures_dir / f"{prefix}_innovations.png")
    save_standardized_innovation_plot(
        innovations,
        selected_order,
        figures_dir / f"{prefix}_standardized_innovations.png",
    )
    save_acf_plot(innovations, selected_order, args.acf_lags, figures_dir / f"{prefix}_acf.png")
    save_transit_preservation_plot(
        selected_frame,
        injected_values,
        injected_innovations,
        injected_standardized,
        injection,
        local_half_width_cadences=args.injection_local_half_width_cadences,
        path=figures_dir / f"{prefix}_transit_preservation.png",
    )
    save_transformed_template_plot(
        selected_frame,
        template_injected_values,
        box_template,
        template_transform,
        injection,
        local_half_width_cadences=args.injection_local_half_width_cadences,
        path=figures_dir / f"{prefix}_transformed_template_match.png",
    )
    save_template_scan_plot(
        template_scan,
        injection,
        path=figures_dir / f"{prefix}_template_scan.png",
    )

    print(f"Selected quality policy: {selected_quality_policy}")
    print(f"Selected ARIMA mode: {selected_mode}")
    print(f"Selected ARIMA order: {selected_order}")
    print(f"Selected model status: {selected.get('selection_status', '')}")
    print("Stationarity assessment:")
    print(f"  ADF p-value: {format_optional_float(selected_stationarity_assessment.original_adf.pvalue)}")
    print(f"  KPSS p-value: {format_optional_float(selected_stationarity_assessment.original_kpss.pvalue)}")
    print(f"  conclusion: {selected_stationarity_assessment.original_series_conclusion}")
    print(f"  recommended d: {selected_stationarity_assessment.recommended_d if selected_stationarity_assessment.recommended_d is not None else 'unresolved'}")
    print(f"  selected differencing alignment: {selected.get('candidate_differencing_alignment', 'unresolved')}")
    print(f"  selected differencing requires review: {selected.get('differencing_requires_review', False)}")
    print(f"Gap-sensitive selected order: {gap_sensitive}")
    print(f"Candidate table: {results_path}")
    print(f"Quality comparison: {quality_comparison_path}")
    print(f"Stationarity diagnostics: {stationarity_path}")
    print(f"Preprocessing summary: {summary_path}")
    print(f"Order stability: {stability_path}")
    print(f"Segment fits: {segment_fits_path}")
    print(f"Coefficient diagnostics: {coefficient_path}")
    print(f"Transit preservation: {preservation_path}")
    print(f"Transformed-template match: {transformed_template_path}")
    print(f"Template scan: {template_scan_path}")
    print(f"Template scan summary: {template_scan_summary_path}")
    print(f"Multi-injection recovery: {multi_injection_path}")
    print(f"Multi-injection summary: {multi_injection_summary_path}")
    print(f"Phase 1 completion report: {phase1_completion_path}")
    print(f"Transit preservation by candidate: {preservation_by_candidate_path}")
    print(f"Regularized light curve: {regular_path}")
    print(f"Innovations: {innovations_path}")
    print(f"Phase 1 engineering complete: {phase1_report['phase1_engineering_complete']}")
    print(f"Phase 1 scientifically ready for Phase 2: {phase1_report['phase1_scientific_ready_for_phase2']}")
    print(f"Multi-injection rank-1 recovery rate: {multi_injection_summary['rank1_recovery_rate']:.3f}")
    print()
    print(
        scored[
            [
                "order",
                "quality_policy",
                "mode",
                "selection_rank",
                "mode_selection_rank",
                "selection_status",
                "statistical_validity_failed",
                "whitening_constraint_failed",
                "variance_constraint_failed",
                "transit_preservation_constraint_failed",
                "fit_metrics_trustworthy",
                "differenced_model",
                "differencing_requires_review",
                "candidate_d",
                "candidate_family_role",
                "candidate_differencing_alignment",
                "recommended_d",
                "original_adf_pvalue",
                "original_kpss_pvalue",
                "converged",
                "AIC",
                "BIC",
                "test_RMSE",
                "test_MAE",
                "mean_negative_log_score",
                "best_baseline_RMSE",
                "beats_best_baseline_RMSE",
                "depth_retention_fraction",
                "snr_retention_fraction",
                "transit_preservation_rank",
                "transit_preservation_failure",
                "max_abs_residual_acf",
                "residual_acf_lag_1",
                "max_abs_residual_acf_1_24",
                "mean_abs_residual_acf_1_24",
                "max_abs_residual_acf_transit_lags",
                "minimum_ljung_box_p",
                "residual_autocorrelation_remaining",
                "variance_instability",
                "boundary_coefficient_count",
                "gap_sensitive",
                "runtime_seconds",
                "failure_reason",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
