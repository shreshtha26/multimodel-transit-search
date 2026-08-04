"""Periodic box-transit templates."""
import numpy as np


def transit_center_times(time, period_days, epoch_days, duration_days=0.0):
    """Return transit centers whose windows overlap the observed time span."""
    if period_days <= 0:
        raise ValueError("period_days must be positive.")
    if duration_days < 0:
        raise ValueError("duration_days must be non-negative.")
    observed = np.asarray(time, dtype=float)
    finite = observed[np.isfinite(observed)]
    if finite.size == 0:
        return np.asarray([], dtype=float)
    start = float(np.min(finite) - duration_days)
    stop = float(np.max(finite) + duration_days)
    first = int(np.ceil((start - epoch_days) / period_days))
    last = int(np.floor((stop - epoch_days) / period_days))
    if last < first:
        return np.asarray([], dtype=float)
    return epoch_days + np.arange(first, last + 1, dtype=float) * period_days

def periodic_box_transit_template(time, period_days, epoch_days, duration_days, depth):
    """Return a repeated additive box-transit signal and in-transit mask."""
    if period_days <= 0:
        raise ValueError("period_days must be positive.")
    if duration_days <= 0:
        raise ValueError("duration_days must be positive.")
    if depth <= 0:
        raise ValueError("depth must be positive.")
    observed = np.asarray(time, dtype=float)
    phase = ((observed - epoch_days + 0.5 * period_days) % period_days) - 0.5 * period_days
    in_transit = np.isfinite(observed) & (np.abs(phase) < 0.5 * duration_days)
    template = np.zeros(observed.shape, dtype=float)
    template[in_transit] = -float(depth)
    return template, in_transit
