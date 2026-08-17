"""Gap-mode transit-injection experiment helpers."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from adaptive_transit.detection.matched_filter import (
    arima_transformed_template,
    local_window_mask,
    matched_filter_statistic,
    scan_arima_transformed_template,
    select_trial_centers,
)
from adaptive_transit.injections.synthetic import TransitInjection, inject_box_transit, local_depth_and_snr
from adaptive_transit.noise_models.arima import FittedArimaModel
from adaptive_transit.noise_models.scaling import trailing_robust_scale
from adaptive_transit.transit_models.box import box_transit_template


@dataclass(frozen=True)
class GapModeInjectionConfig:
    """Configuration for one reproducible gap-mode injection experiment."""

    depths: tuple[float, ...] = (0.0005, 0.001, 0.002)
    durations_cadences: tuple[int, ...] = (4, 6, 8)
    centers_per_duration: int = 3
    local_half_width_cadences: int = 24
    scan_stride: int = 10
    scan_max_centers: int = 250
    scale_window: int = 96
    false_alarm_rates: tuple[float, ...] = (0.10, 0.05, 0.01)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def finite_scan_mask(values: np.ndarray) -> np.ndarray:
    """Use all finite represented cadences for scan trials."""

    return np.isfinite(np.asarray(values, dtype=float).reshape(-1))


def template_amplitude_estimate(
    observed: np.ndarray,
    template: np.ndarray,
    *,
    scale: np.ndarray,
    usable_mask: np.ndarray,
) -> float:
    """Estimate multiplicative template amplitude using the matched-filter weighting."""

    y = np.asarray(observed, dtype=float).reshape(-1)
    h = np.asarray(template, dtype=float).reshape(-1)
    sigma = np.asarray(scale, dtype=float).reshape(-1)
    usable = np.asarray(usable_mask, dtype=bool).reshape(-1)
    valid = usable & np.isfinite(y) & np.isfinite(h) & np.isfinite(sigma) & (sigma > 0)
    if valid.sum() < 3:
        return float("nan")
    y_valid = y[valid]
    h_valid = h[valid]
    weights = 1.0 / np.square(sigma[valid])
    h_centered = h_valid - np.average(h_valid, weights=weights)
    denominator = float(np.sum(weights * h_centered * h_centered))
    if denominator <= 0 or not np.isfinite(denominator):
        return float("nan")
    return float(np.sum(weights * h_centered * y_valid) / denominator)


def transformed_template_shape_metrics(
    cadenceno: np.ndarray,
    transformed_template: np.ndarray,
    injection: TransitInjection,
    *,
    local_half_width_cadences: int,
    usable_mask: np.ndarray,
) -> dict[str, float]:
    """Summarize whether the ARIMA transform turns a box into edge-dominated spikes."""

    cadence = np.asarray(cadenceno, dtype=int).reshape(-1)
    transformed = np.asarray(transformed_template, dtype=float).reshape(-1)
    usable = np.asarray(usable_mask, dtype=bool).reshape(-1)
    local = local_window_mask(cadence, center_cadenceno=injection.center_cadenceno, half_width_cadences=local_half_width_cadences)
    _, in_transit = box_transit_template(
        cadence,
        center_cadenceno=injection.center_cadenceno,
        duration_cadences=injection.duration_cadences,
        depth=injection.depth,
    )
    valid = local & usable & np.isfinite(transformed)
    in_valid = valid & in_transit
    if in_valid.sum() == 0 or valid.sum() < 3:
        return {
            "template_abs_area": float("nan"),
            "template_edge_abs_fraction": float("nan"),
            "template_interior_abs_fraction": float("nan"),
            "ingress_egress_asymmetry_fraction": float("nan"),
            "ingress_egress_distortion_fraction": float("nan"),
        }

    in_positions = np.flatnonzero(in_valid)
    ingress = int(in_positions[0])
    egress = int(in_positions[-1])
    edge_window = valid & (
        (np.abs(cadence - cadence[ingress]) <= 1)
        | (np.abs(cadence - cadence[egress]) <= 1)
        | (np.abs(cadence - (cadence[egress] + 1)) <= 1)
    )
    abs_values = np.abs(transformed[valid])
    total_abs = float(abs_values.sum())
    if total_abs <= 0 or not np.isfinite(total_abs):
        return {
            "template_abs_area": total_abs,
            "template_edge_abs_fraction": float("nan"),
            "template_interior_abs_fraction": float("nan"),
            "ingress_egress_asymmetry_fraction": float("nan"),
            "ingress_egress_distortion_fraction": float("nan"),
        }

    edge_abs = float(np.abs(transformed[edge_window]).sum())
    interior_abs = float(np.abs(transformed[in_valid & ~edge_window]).sum())
    ingress_abs = float(np.abs(transformed[valid & (np.abs(cadence - cadence[ingress]) <= 1)]).sum())
    egress_abs = float(np.abs(transformed[valid & (np.abs(cadence - (cadence[egress] + 1)) <= 1)]).sum())
    asymmetry_denominator = ingress_abs + egress_abs
    asymmetry = abs(ingress_abs - egress_abs) / asymmetry_denominator if asymmetry_denominator > 0 else float("nan")
    edge_fraction = edge_abs / total_abs
    return {
        "template_abs_area": total_abs,
        "template_edge_abs_fraction": float(edge_fraction),
        "template_interior_abs_fraction": float(interior_abs / total_abs),
        "ingress_egress_asymmetry_fraction": float(asymmetry),
        "ingress_egress_distortion_fraction": float(edge_fraction),
    }


def empirical_false_alarm_thresholds(null_scan: pd.DataFrame, false_alarm_rates: tuple[float, ...]) -> dict[str, float]:
    """Estimate single-light-curve trial-level thresholds from a no-injection scan."""

    scores = pd.to_numeric(null_scan.get("innovation_transformed_template_statistic", pd.Series(dtype=float)), errors="coerce").dropna()
    if scores.empty:
        return {f"threshold_far_{rate:g}": float("nan") for rate in false_alarm_rates}
    return {f"threshold_far_{rate:g}": float(scores.quantile(1.0 - rate)) for rate in false_alarm_rates}


def spurious_peak_metrics(scan: pd.DataFrame) -> dict[str, float | int | bool]:
    """Measure strongest non-injected residual peaks in an injected scan."""

    if scan.empty or "is_injected_center_neighborhood" not in scan.columns:
        return {
            "best_injected_neighborhood_statistic": float("nan"),
            "best_injected_neighborhood_rank": pd.NA,
            "best_spurious_statistic": float("nan"),
            "spurious_to_injected_statistic_ratio": float("nan"),
            "spurious_peak_exceeds_injected": False,
            "n_spurious_peaks_positive": 0,
        }
    score_column = "innovation_transformed_template_statistic"
    injected = scan.loc[scan["is_injected_center_neighborhood"].astype(bool)]
    outside = scan.loc[~scan["is_injected_center_neighborhood"].astype(bool)]
    best_injected_stat = float(pd.to_numeric(injected[score_column], errors="coerce").max()) if not injected.empty else float("nan")
    best_spurious_stat = float(pd.to_numeric(outside[score_column], errors="coerce").max()) if not outside.empty else float("nan")
    ranks = pd.to_numeric(injected.get("innovation_transformed_template_rank", pd.Series(dtype=float)), errors="coerce")
    ratio = best_spurious_stat / best_injected_stat if np.isfinite(best_spurious_stat) and np.isfinite(best_injected_stat) and best_injected_stat != 0 else float("nan")
    return {
        "best_injected_neighborhood_statistic": best_injected_stat,
        "best_injected_neighborhood_rank": int(ranks.min()) if ranks.notna().any() else pd.NA,
        "best_spurious_statistic": best_spurious_stat,
        "spurious_to_injected_statistic_ratio": float(ratio),
        "spurious_peak_exceeds_injected": bool(np.isfinite(best_spurious_stat) and np.isfinite(best_injected_stat) and best_spurious_stat >= best_injected_stat),
        "n_spurious_peaks_positive": int((pd.to_numeric(outside[score_column], errors="coerce") > 0).sum()) if not outside.empty else 0,
    }


def run_single_gap_mode_injection(
    *,
    gap_mode: str,
    frame: pd.DataFrame,
    values: np.ndarray,
    fitted_model: FittedArimaModel,
    injection: TransitInjection,
    allow_missing: bool,
    config: GapModeInjectionConfig,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Run one fixed-ARIMA injection measurement for one gap representation."""

    cadence = frame["cadenceno"].to_numpy(dtype=int)
    scan_mask = finite_scan_mask(values)
    injected_values, box_template, _ = inject_box_transit(
        values,
        cadence,
        center_cadenceno=injection.center_cadenceno,
        duration_cadences=injection.duration_cadences,
        depth=injection.depth,
    )
    transform = arima_transformed_template(values, box_template, fitted_model, allow_missing=allow_missing)
    local = local_window_mask(cadence, center_cadenceno=injection.center_cadenceno, half_width_cadences=config.local_half_width_cadences)
    raw_scale = trailing_robust_scale(values, window=config.scale_window)
    innovation_scale = trailing_robust_scale(fitted_model.innovations, window=config.scale_window)
    raw_score = matched_filter_statistic(injected_values, box_template, scale=raw_scale, usable_mask=local & scan_mask)
    unchanged_score = matched_filter_statistic(transform.injected_innovations, box_template, scale=innovation_scale, usable_mask=local & transform.usable_mask)
    transformed_score = matched_filter_statistic(
        transform.injected_innovations,
        transform.transformed_template,
        scale=innovation_scale,
        usable_mask=local & transform.usable_mask,
    )
    amplitude = template_amplitude_estimate(
        transform.injected_innovations,
        transform.transformed_template,
        scale=innovation_scale,
        usable_mask=local & transform.usable_mask,
    )
    recovered_depth = amplitude * injection.depth if np.isfinite(amplitude) else float("nan")
    before = local_depth_and_snr(
        cadence,
        injected_values,
        center_cadenceno=injection.center_cadenceno,
        duration_cadences=injection.duration_cadences,
        local_half_width_cadences=config.local_half_width_cadences,
    )
    after = local_depth_and_snr(
        cadence,
        transform.injected_innovations,
        center_cadenceno=injection.center_cadenceno,
        duration_cadences=injection.duration_cadences,
        local_half_width_cadences=config.local_half_width_cadences,
    )
    shape = transformed_template_shape_metrics(
        cadence,
        transform.transformed_template,
        injection,
        local_half_width_cadences=config.local_half_width_cadences,
        usable_mask=transform.usable_mask,
    )
    trial_centers = select_trial_centers(
        cadence,
        scan_mask,
        stride=config.scan_stride,
        max_centers=config.scan_max_centers,
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
        local_half_width_cadences=config.local_half_width_cadences,
        scale_window=config.scale_window,
        allow_missing=allow_missing,
        usable_mask=scan_mask,
        injected_center_cadenceno=injection.center_cadenceno,
        injected_neighborhood_cadences=max(1, injection.duration_cadences // 2),
    )
    spurious = spurious_peak_metrics(scan)
    input_snr = raw_score.statistic
    recovered_snr = transformed_score.statistic
    row: dict[str, Any] = {
        "gap_mode": gap_mode,
        **injection.to_dict(),
        "injected_transit_depth": float(injection.depth),
        "recovered_transformed_template_depth": float(recovered_depth),
        "transformed_template_amplitude": float(amplitude),
        "depth_retention_fraction": float(recovered_depth / injection.depth) if injection.depth > 0 and np.isfinite(recovered_depth) else float("nan"),
        "input_matched_filter_snr": float(input_snr),
        "recovered_transformed_template_snr": float(recovered_snr),
        "snr_retention_fraction": float(recovered_snr / input_snr) if np.isfinite(input_snr) and input_snr != 0 and np.isfinite(recovered_snr) else float("nan"),
        "input_local_depth": float(before["depth"]),
        "recovered_innovation_local_depth": float(after["depth"]),
        "input_local_snr": float(before["local_snr"]),
        "recovered_innovation_local_snr": float(after["local_snr"]),
        "local_snr_retention_fraction": float(after["local_snr"] / before["local_snr"]) if float(before["local_snr"]) != 0 else float("nan"),
        **raw_score.to_dict("raw_flux_box_"),
        **unchanged_score.to_dict("innovation_unchanged_box_"),
        **transformed_score.to_dict("innovation_transformed_template_"),
        **shape,
        **spurious,
        "n_trial_centers": int(len(scan)),
    }
    for key, threshold in thresholds.items():
        rate = key.replace("threshold_far_", "")
        detected = bool(np.isfinite(spurious["best_injected_neighborhood_statistic"]) and float(spurious["best_injected_neighborhood_statistic"]) >= threshold)
        best_rank = pd.to_numeric(pd.Series([spurious["best_injected_neighborhood_rank"]]), errors="coerce").iloc[0]
        outside_scan = scan.loc[~scan["is_injected_center_neighborhood"].astype(bool), "innovation_transformed_template_statistic"]
        spurious_above_threshold = pd.to_numeric(outside_scan, errors="coerce") >= threshold
        row[key] = float(threshold)
        row[f"detected_at_far_{rate}"] = detected
        row[f"top_recovered_at_far_{rate}"] = bool(detected and pd.notna(best_rank) and int(best_rank) == 1)
        row[f"n_spurious_peaks_above_far_{rate}"] = int(spurious_above_threshold.sum())
    return row


def run_null_scan_for_thresholds(
    *,
    frame: pd.DataFrame,
    values: np.ndarray,
    fitted_model: FittedArimaModel,
    duration_cadences: int,
    depth: float,
    allow_missing: bool,
    config: GapModeInjectionConfig,
) -> pd.DataFrame:
    """Scan the non-injected represented light curve to build empirical thresholds."""

    cadence = frame["cadenceno"].to_numpy(dtype=int)
    scan_mask = finite_scan_mask(values)
    trial_centers = select_trial_centers(
        cadence,
        scan_mask,
        stride=config.scan_stride,
        max_centers=config.scan_max_centers,
    )
    return scan_arima_transformed_template(
        cadence,
        values,
        values,
        fitted_model,
        trial_centers,
        duration_cadences=duration_cadences,
        depth=depth,
        local_half_width_cadences=config.local_half_width_cadences,
        scale_window=config.scale_window,
        allow_missing=allow_missing,
        usable_mask=scan_mask,
    )


def summarize_gap_mode_injections(results: pd.DataFrame, false_alarm_rates: tuple[float, ...]) -> pd.DataFrame:
    """Summarize injection preservation and controlled-threshold recovery by gap mode."""

    rows: list[dict[str, Any]] = []
    for gap_mode, group in results.groupby("gap_mode"):
        row: dict[str, Any] = {
            "gap_mode": gap_mode,
            "n_injections": int(len(group)),
            "median_depth_retention_fraction": float(pd.to_numeric(group["depth_retention_fraction"], errors="coerce").median()),
            "median_snr_retention_fraction": float(pd.to_numeric(group["snr_retention_fraction"], errors="coerce").median()),
            "median_ingress_egress_distortion_fraction": float(pd.to_numeric(group["ingress_egress_distortion_fraction"], errors="coerce").median()),
            "median_best_spurious_statistic": float(pd.to_numeric(group["best_spurious_statistic"], errors="coerce").median()),
            "spurious_peak_exceeds_injected_rate": float(group["spurious_peak_exceeds_injected"].astype(bool).mean()),
        }
        for rate in false_alarm_rates:
            label = f"{rate:g}"
            row[f"recovery_rate_at_far_{label}"] = float(group[f"detected_at_far_{label}"].astype(bool).mean())
            row[f"top_recovery_rate_at_far_{label}"] = float(group[f"top_recovered_at_far_{label}"].astype(bool).mean())
            row[f"median_spurious_peaks_above_far_{label}"] = float(pd.to_numeric(group[f"n_spurious_peaks_above_far_{label}"], errors="coerce").median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("gap_mode").reset_index(drop=True)
