"""Explicit gap-handling representations for ARIMA comparison."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from adaptive_transit.preprocessing.normalization import longest_contiguous_segment, segment_lengths


@dataclass(frozen=True)
class GapModeRepresentation:
    """One model-ready light-curve representation plus its audit metadata."""

    gap_mode: str
    frame: pd.DataFrame
    values: np.ndarray
    allow_missing: bool
    ordinary_lags_meaningful: bool
    series_plot_values: np.ndarray
    series_plot_label: str
    metadata: dict[str, Any]


def gap_runs(missing_mask: np.ndarray) -> list[dict[str, int]]:
    """Return contiguous True runs as start/end/length dictionaries."""

    mask = np.asarray(missing_mask, dtype=bool).reshape(-1)
    runs: list[dict[str, int]] = []
    start: int | None = None
    for index, is_missing in enumerate(mask):
        if is_missing and start is None:
            start = index
        elif not is_missing and start is not None:
            runs.append({"start_index": start, "end_index": index - 1, "length": index - start})
            start = None
    if start is not None:
        runs.append({"start_index": start, "end_index": len(mask) - 1, "length": len(mask) - start})
    return runs


def cadence_consistency(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize cadence-number consistency for a represented series."""

    cadence = frame["cadenceno"].to_numpy(dtype=int)
    if cadence.size < 2:
        return {
            "cadence_step_min": 0,
            "cadence_step_max": 0,
            "cadence_step_median": 0.0,
            "cadence_consistent": True,
        }
    steps = np.diff(cadence)
    return {
        "cadence_step_min": int(np.min(steps)),
        "cadence_step_max": int(np.max(steps)),
        "cadence_step_median": float(np.median(steps)),
        "cadence_consistent": bool(np.all(steps == 1)),
    }


def finite_segment_runs(values: np.ndarray) -> list[dict[str, int]]:
    """Return contiguous finite-value runs."""

    finite = np.isfinite(np.asarray(values, dtype=float).reshape(-1))
    runs: list[dict[str, int]] = []
    start: int | None = None
    for index, is_finite in enumerate(finite):
        if is_finite and start is None:
            start = index
        elif not is_finite and start is not None:
            runs.append({"start_index": start, "end_index": index - 1, "length": index - start})
            start = None
    if start is not None:
        runs.append({"start_index": start, "end_index": len(finite) - 1, "length": len(finite) - start})
    return runs


def longest_finite_slice(values: np.ndarray) -> slice:
    """Return the longest finite run as a Python slice."""

    finite_runs = finite_segment_runs(values)
    if not finite_runs:
        return slice(0, 0)
    longest = max(finite_runs, key=lambda run: run["length"])
    return slice(int(longest["start_index"]), int(longest["end_index"]) + 1)


def interpolate_eligible_gaps(
    values: np.ndarray,
    *,
    max_gap_cadences: int,
    method: str = "linear",
    edge_extrapolation: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Interpolate only bracketed gaps whose length is within the configured limit."""

    if method != "linear":
        raise ValueError("Only linear interpolation is currently implemented.")
    if max_gap_cadences < 0:
        raise ValueError("max_gap_cadences must be non-negative.")

    original = np.asarray(values, dtype=float).reshape(-1)
    filled = original.copy()
    interpolated = np.zeros(filled.shape, dtype=bool)
    runs = gap_runs(~np.isfinite(filled))
    unfilled_long = 0
    unfilled_edge = 0

    for run in runs:
        start = int(run["start_index"])
        end = int(run["end_index"])
        length = int(run["length"])
        has_left = start > 0 and np.isfinite(filled[start - 1])
        has_right = end + 1 < filled.size and np.isfinite(filled[end + 1])
        if length > max_gap_cadences:
            unfilled_long += 1
            continue
        if not (has_left and has_right):
            if edge_extrapolation:
                fill_value = filled[start - 1] if has_left else filled[end + 1] if has_right else np.nan
                if np.isfinite(fill_value):
                    filled[start : end + 1] = fill_value
                    interpolated[start : end + 1] = True
                else:
                    unfilled_edge += 1
            else:
                unfilled_edge += 1
            continue
        x = np.array([start - 1, end + 1], dtype=float)
        y = np.array([filled[start - 1], filled[end + 1]], dtype=float)
        positions = np.arange(start, end + 1, dtype=float)
        filled[start : end + 1] = np.interp(positions, x, y)
        interpolated[start : end + 1] = True

    metadata = {
        "interpolation_method": method,
        "max_allowed_interpolated_gap": int(max_gap_cadences),
        "maximum_allowed_interpolated_gap": int(max_gap_cadences),
        "edge_extrapolation": bool(edge_extrapolation),
        "edge_extrapolation_policy": "nearest_edge_value" if edge_extrapolation else "none",
        "interpolated_values": int(interpolated.sum()),
        "interpolated_fraction": float(interpolated.sum() / filled.size) if filled.size else 0.0,
        "unfilled_long_gaps": int(unfilled_long),
        "unfilled_edge_gaps": int(unfilled_edge),
        "remaining_missing_values": int(np.isnan(filled).sum()),
    }
    return filled, interpolated, metadata


def _base_gap_metadata(regular: pd.DataFrame, values: np.ndarray, *, gap_mode: str) -> dict[str, Any]:
    missing = ~np.isfinite(values)
    runs = gap_runs(missing)
    lengths = [int(run["length"]) for run in runs]
    return {
        "gap_mode": gap_mode,
        "total_grid_length": int(len(regular)),
        "observations": int(np.isfinite(values).sum()),
        "observed_cadences": int(np.isfinite(values).sum()),
        "missing_cadences": int(missing.sum()),
        "missing_fraction": float(missing.mean()) if len(missing) else 0.0,
        "gap_count": int(len(runs)),
        "gap_lengths": tuple(lengths),
        "max_gap_length": int(max(lengths)) if lengths else 0,
        "fraction_of_quarter_retained": float(np.isfinite(values).sum() / len(regular)) if len(regular) else 0.0,
        "known_events_available": False,
        "known_events_retained": None,
        "ordinary_lags_meaningful": False,
    }


def build_gap_mode_representations(
    regular: pd.DataFrame,
    *,
    interpolation_method: str = "linear",
    max_interpolated_gap_cadences: int = 12,
    edge_extrapolation: bool = False,
) -> dict[str, GapModeRepresentation]:
    """Build the three explicit gap modes used by the comparison runner."""

    full_values = regular["normalized_flux"].to_numpy(dtype=float)
    full_metadata = _base_gap_metadata(regular, full_values, gap_mode="full_grid_missing")
    full_metadata.update(cadence_consistency(regular))
    full_metadata.update(
        {
            "missing_observations_compressed_for_stationarity": bool(np.isnan(full_values).any()),
            "interpolation_used": False,
            "ordinary_lags_meaningful": False,
            "series_plot_representation": "missing_valued_regular_grid",
        }
    )

    segment = longest_contiguous_segment(regular)
    segment_values = segment["normalized_flux"].to_numpy(dtype=float)
    segment_metadata = _base_gap_metadata(regular, segment_values, gap_mode="longest_contiguous")
    segment_metadata.update(cadence_consistency(segment))
    segment_metadata.update(
        {
            "segment_id": int(segment["segment_id"].iloc[0]),
            "segment_start_cadenceno": int(segment["cadenceno"].iloc[0]),
            "segment_end_cadenceno": int(segment["cadenceno"].iloc[-1]),
            "segment_start_time": float(segment["time"].iloc[0]),
            "segment_end_time": float(segment["time"].iloc[-1]),
            "segment_duration_cadences": int(segment["cadenceno"].iloc[-1] - segment["cadenceno"].iloc[0] + 1),
            "segment_duration_days": float(segment["time"].iloc[-1] - segment["time"].iloc[0]),
            "available_segment_count": int(len(segment_lengths(regular))),
            "missing_observations_compressed_for_stationarity": False,
            "interpolation_used": False,
            "ordinary_lags_meaningful": True,
            "series_plot_representation": "longest_contiguous_segment",
        }
    )

    interpolated_values, interpolated_mask, interpolation_metadata = interpolate_eligible_gaps(
        full_values,
        max_gap_cadences=max_interpolated_gap_cadences,
        method=interpolation_method,
        edge_extrapolation=edge_extrapolation,
    )
    interpolated_frame = regular.copy()
    interpolated_frame["interpolated_normalized_flux"] = interpolated_values
    interpolated_frame["interpolated_value"] = interpolated_mask
    plot_slice = longest_finite_slice(interpolated_values)
    plot_values = interpolated_values[plot_slice]
    interpolated_metadata = _base_gap_metadata(regular, interpolated_values, gap_mode="interpolated_full_grid")
    interpolated_metadata.update(cadence_consistency(regular))
    interpolated_metadata.update(interpolation_metadata)
    interpolated_metadata.update(
        {
            "missing_observations_compressed_for_stationarity": bool(np.isnan(interpolated_values).any()),
            "interpolation_used": True,
            "ordinary_lags_meaningful": bool(not np.isnan(interpolated_values).any()),
            "series_plot_representation": "interpolated_full_grid" if not np.isnan(interpolated_values).any() else "longest_finite_after_interpolation",
        }
    )

    return {
        "longest_contiguous": GapModeRepresentation(
            gap_mode="longest_contiguous",
            frame=segment,
            values=segment_values,
            allow_missing=False,
            ordinary_lags_meaningful=True,
            series_plot_values=segment_values,
            series_plot_label="longest contiguous segment",
            metadata=segment_metadata,
        ),
        "full_grid_missing": GapModeRepresentation(
            gap_mode="full_grid_missing",
            frame=regular,
            values=full_values,
            allow_missing=True,
            ordinary_lags_meaningful=False,
            series_plot_values=full_values,
            series_plot_label="full regular grid with missing values",
            metadata=full_metadata,
        ),
        "interpolated_full_grid": GapModeRepresentation(
            gap_mode="interpolated_full_grid",
            frame=interpolated_frame,
            values=interpolated_values,
            allow_missing=bool(np.isnan(interpolated_values).any()),
            ordinary_lags_meaningful=bool(np.isfinite(plot_values).all() and plot_values.size > 2),
            series_plot_values=plot_values,
            series_plot_label=str(interpolated_metadata["series_plot_representation"]),
            metadata=interpolated_metadata,
        ),
    }
