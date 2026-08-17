"""Single-event synthetic transit injection and preservation metrics."""

from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from adaptive_transit.noise_models.scaling import robust_mad_scale
from adaptive_transit.transit_models.box import box_transit_mask, box_transit_template
from adaptive_transit.transit_models.periodic import periodic_box_transit_template


@dataclass(frozen=True)
class TransitInjection:
    """Definition of one box-shaped injected transit."""
    center_cadenceno: int
    duration_cadences: int
    depth: float
    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def choose_injection_center(
    regular: pd.DataFrame,
    *,
    duration_cadences: int,
    segment_id: int | None = None,
) -> int:
    """Choose the center cadence from a contiguous usable segment."""

    if duration_cadences < 1:
        raise ValueError("duration_cadences must be positive.")

    usable = regular.loc[regular["segment_id"] >= 0]
    if usable.empty:
        raise ValueError("No usable segment is available for injection.")

    if segment_id is None:
        segment_id = int(usable.groupby("segment_id").size().idxmax())

    segment = regular.loc[regular["segment_id"] == segment_id].copy()
    if len(segment) < duration_cadences + 4:
        raise ValueError(f"Segment {segment_id} is too short for duration {duration_cadences}.")

    middle_index = len(segment) // 2
    return int(segment.iloc[middle_index]["cadenceno"])


def choose_injection_centers(
    regular: pd.DataFrame,
    *,
    duration_cadences: int,
    centers_per_segment: int = 3,
    max_segments: int = 3,
) -> tuple[int, ...]:
    """Choose several injection centers from the longest usable segments."""

    if duration_cadences < 1:
        raise ValueError("duration_cadences must be positive.")
    if centers_per_segment < 1:
        raise ValueError("centers_per_segment must be positive.")
    if max_segments < 1:
        raise ValueError("max_segments must be positive.")

    usable = regular.loc[regular["segment_id"] >= 0]
    if usable.empty:
        raise ValueError("No usable segment is available for injection.")

    centers: list[int] = []
    segment_ids = usable.groupby("segment_id").size().sort_values(ascending=False).index
    margin = max(2, duration_cadences)
    for segment_id in segment_ids[:max_segments]:
        segment = regular.loc[regular["segment_id"] == segment_id].copy()
        if len(segment) < duration_cadences + 2 * margin + 1:
            continue

        lower = margin
        upper = len(segment) - margin - 1
        positions = np.linspace(lower, upper, centers_per_segment, dtype=int)
        for position in np.unique(positions):
            centers.append(int(segment.iloc[int(position)]["cadenceno"]))

    if not centers:
        centers.append(choose_injection_center(regular, duration_cadences=duration_cadences))
    return tuple(sorted(set(centers)))


def inject_box_transit(
    values: np.ndarray,
    cadenceno: np.ndarray,
    *,
    center_cadenceno: int,
    duration_cadences: int,
    depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add a box transit to finite normalized-flux samples."""

    template, in_transit = box_transit_template(
        cadenceno,
        center_cadenceno=center_cadenceno,
        duration_cadences=duration_cadences,
        depth=depth,
    )
    return _inject_additive_template(values, template, in_transit)


def inject_periodic_box_transit(time, values, period_days, epoch_days, duration_days, depth):
    """Add a repeated box transit to finite flux samples."""

    template, in_transit = periodic_box_transit_template(time, period_days, epoch_days, duration_days, depth)
    return _inject_additive_template(values, template, in_transit)


def _inject_additive_template(
    values: np.ndarray,
    template: np.ndarray,
    in_transit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add a precomputed additive transit template to finite samples."""

    series = np.asarray(values, dtype=float).copy()
    additive_template = np.asarray(template, dtype=float)
    transit_mask = np.asarray(in_transit, dtype=bool)
    if series.shape != additive_template.shape or series.shape != transit_mask.shape:
        raise ValueError("values, template, and in_transit must have the same shape.")

    finite = np.isfinite(series)
    injected = series.copy()
    injected[finite] = injected[finite] + additive_template[finite]
    return injected, additive_template, transit_mask


def local_depth_and_snr(
    cadenceno: np.ndarray,
    values: np.ndarray,
    *,
    center_cadenceno: int,
    duration_cadences: int,
    local_half_width_cadences: int,
) -> dict[str, float | int]:
    """Measure local box-event depth and SNR from a time series."""

    cadence = np.asarray(cadenceno, dtype=float)
    series = np.asarray(values, dtype=float)
    local = np.abs(cadence - center_cadenceno) <= local_half_width_cadences
    in_transit = box_transit_mask(
        cadence,
        center_cadenceno=center_cadenceno,
        duration_cadences=duration_cadences,
    )
    out_transit = local & ~in_transit

    finite_in = in_transit & np.isfinite(series)
    finite_out = out_transit & np.isfinite(series)
    if finite_in.sum() == 0 or finite_out.sum() < 3:
        return {
            "depth": float("nan"),
            "local_snr": float("nan"),
            "event_center_cadenceno": -1,
        }

    baseline = float(np.median(series[finite_out]))
    in_value = float(np.median(series[finite_in]))
    depth = baseline - in_value
    noise = robust_mad_scale(series[finite_out])
    local_snr = float(depth / noise * np.sqrt(finite_in.sum())) if np.isfinite(noise) and noise > 0 else float("nan")

    local_finite = local & np.isfinite(series)
    center = int(cadence[local_finite][np.argmin(series[local_finite])]) if local_finite.any() else -1

    return {
        "depth": float(depth),
        "local_snr": local_snr,
        "event_center_cadenceno": center,
    }


def transit_preservation_metrics(
    cadenceno: np.ndarray,
    injected_flux: np.ndarray,
    innovations: np.ndarray,
    standardized_innovations: np.ndarray,
    injection: TransitInjection,
    *,
    local_half_width_cadences: int = 24,
) -> dict[str, float | int]:
    """Measure whether an injected box transit remains visible after ARIMA."""

    before = local_depth_and_snr(
        cadenceno,
        injected_flux,
        center_cadenceno=injection.center_cadenceno,
        duration_cadences=injection.duration_cadences,
        local_half_width_cadences=local_half_width_cadences,
    )
    after = local_depth_and_snr(
        cadenceno,
        innovations,
        center_cadenceno=injection.center_cadenceno,
        duration_cadences=injection.duration_cadences,
        local_half_width_cadences=local_half_width_cadences,
    )
    standardized = local_depth_and_snr(
        cadenceno,
        standardized_innovations,
        center_cadenceno=injection.center_cadenceno,
        duration_cadences=injection.duration_cadences,
        local_half_width_cadences=local_half_width_cadences,
    )

    before_depth = float(before["depth"])
    after_depth = float(after["depth"])
    before_snr = float(before["local_snr"])
    after_snr = float(after["local_snr"])

    depth_retention = after_depth / before_depth if before_depth != 0 else float("nan")
    snr_retention = after_snr / before_snr if before_snr != 0 else float("nan")

    return {
        **injection.to_dict(),
        "observed_depth_before_arima": before_depth,
        "innovation_depth_after_arima": after_depth,
        "standardized_innovation_depth_after_arima": float(standardized["depth"]),
        "depth_retention_fraction": float(depth_retention),
        "local_snr_before_arima": before_snr,
        "local_snr_after_arima": after_snr,
        "standardized_snr_after_arima": float(standardized["local_snr"]),
        "snr_retention_fraction": float(snr_retention),
        "event_center_shift_cadences": int(int(after["event_center_cadenceno"]) - injection.center_cadenceno),
    }
