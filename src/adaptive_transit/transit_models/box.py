"""Box-shaped transit templates on integer cadence grids."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_positive_integer(value: float, name: str) -> int:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 1 or not numeric.is_integer():
        raise ValueError(f"{name} must be a positive finite integer.")
    return int(numeric)


def _as_finite_integer(value: float, name: str) -> int:
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be a finite integer cadence number.")
    return int(numeric)


def _as_positive_float(value: float, name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return numeric


def box_transit_mask(
    cadenceno: ArrayLike,
    *,
    center_cadenceno: float,
    duration_cadences: float,
) -> NDArray[np.bool_]:
    """Return the in-transit mask for a single box event.

    `duration_cadences` is interpreted as an integer cadence count. For even
    durations, the extra cadence is placed after the center cadence.
    """

    duration = _as_positive_integer(duration_cadences, "duration_cadences")
    center = _as_finite_integer(center_cadenceno, "center_cadenceno")
    cadence: NDArray[np.float64] = np.asarray(cadenceno, dtype=np.float64)

    start = center - (duration - 1) // 2
    stop = start + duration
    return np.isfinite(cadence) & (cadence >= start) & (cadence < stop)


def box_transit_template(
    cadenceno: ArrayLike,
    center_cadenceno: float,
    duration_cadences: float,
    depth: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return additive box-transit signal and in-transit mask."""

    transit_depth = _as_positive_float(depth, "depth")
    in_transit = box_transit_mask(
        cadenceno,
        center_cadenceno=center_cadenceno,
        duration_cadences=duration_cadences,
    )
    template = np.zeros(in_transit.shape, dtype=np.float64)
    template[in_transit] = -transit_depth
    return template, in_transit
