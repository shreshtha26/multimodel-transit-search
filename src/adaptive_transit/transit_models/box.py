"""Box-shaped transit templates."""
import numpy as np


def box_transit_template(cadenceno, center_cadenceno, duration_cadences, depth):
    """Return additive box-transit signal and in-transit mask."""
    if duration_cadences < 1:
        raise ValueError("duration_cadences must be positive.")
    if depth <= 0:
        raise ValueError("depth must be positive.")
    cadence = np.asarray(cadenceno, dtype=float)
    half_width = duration_cadences / 2.0
    in_transit = np.abs(cadence - center_cadenceno) < half_width
    template = np.zeros(cadence.shape, dtype=float)
    template[in_transit] = -float(depth)
    return template, in_transit