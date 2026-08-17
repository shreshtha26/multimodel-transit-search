"""Physical BATMAN transit injection for zero-centered normalized light curves.

The project preprocessing convention is normalized_flux = flux / median(flux) - 1,
so BATMAN's unity-normalized light curve is converted to an additive template by
subtracting one before injection.

The POC mode below deliberately preserves the project's familiar period / T14 /
depth / phase controls.  It is a morphology upgrade, not yet a population-level
stellar-parameter injection prior.  A later production injection layer should
derive a/R* and limb darkening from stellar parameters and let duration emerge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import acos, degrees, pi, sin

import batman
import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class BatmanInjectionTruth:
    period_days: float
    epoch_days: float
    requested_duration_days: float
    requested_depth: float
    radius_ratio: float
    scaled_semimajor_axis: float
    inclination_degrees: float
    impact_parameter: float
    eccentricity: float
    omega_degrees: float
    limb_darkening_law: str
    limb_darkening_u1: float
    limb_darkening_u2: float
    exposure_time_days: float
    supersample_factor: int
    realized_max_depth_on_observed_cadences: float

    def to_dict(self) -> dict:
        return asdict(self)


def median_positive_cadence_days(time) -> float:
    values = np.asarray(time, dtype=float).reshape(-1)
    values = np.sort(np.unique(values[np.isfinite(values)]))
    if values.size < 2:
        raise ValueError("At least two finite times are required to estimate cadence.")
    steps = np.diff(values)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if steps.size == 0:
        raise ValueError("A positive cadence could not be estimated.")
    return float(np.median(steps))


def circular_geometry_from_t14(
    period_days: float,
    duration_days: float,
    radius_ratio: float,
    impact_parameter: float,
) -> tuple[float, float]:
    """Return (a/R*, inclination_deg) for circular geometry matching T14.

    Uses the standard circular-orbit first-to-fourth-contact duration relation
    and solves it algebraically for a/R* at fixed radius ratio and impact
    parameter.
    """
    period = float(period_days)
    duration = float(duration_days)
    rp = float(radius_ratio)
    b = float(impact_parameter)
    if period <= 0 or duration <= 0 or duration >= 0.5 * period:
        raise ValueError("Require 0 < duration_days < period_days / 2.")
    if rp <= 0 or rp >= 1:
        raise ValueError("radius_ratio must be in (0, 1).")
    if b < 0 or b >= 1.0 + rp:
        raise ValueError("impact_parameter must satisfy 0 <= b < 1 + radius_ratio.")

    s = sin(pi * duration / period)
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Invalid period/duration geometry.")
    numerator = (1.0 + rp) ** 2 - b**2
    if numerator <= 0:
        raise ValueError("Requested impact parameter does not produce a transit.")
    a_squared = b**2 + numerator / (s**2)
    a_over_rstar = float(np.sqrt(a_squared))
    if a_over_rstar <= b:
        raise ValueError("Derived a/R* is incompatible with the impact parameter.")
    inclination = float(degrees(acos(np.clip(b / a_over_rstar, -1.0, 1.0))))
    return a_over_rstar, inclination


def make_batman_params(
    *,
    epoch_days: float,
    period_days: float,
    radius_ratio: float,
    scaled_semimajor_axis: float,
    inclination_degrees: float,
    limb_darkening_coefficients=(0.3, 0.2),
    eccentricity: float = 0.0,
    omega_degrees: float = 90.0,
):
    params = batman.TransitParams()
    params.t0 = float(epoch_days)
    params.per = float(period_days)
    params.rp = float(radius_ratio)
    params.a = float(scaled_semimajor_axis)
    params.inc = float(inclination_degrees)
    params.ecc = float(eccentricity)
    params.w = float(omega_degrees)
    params.limb_dark = "quadratic"
    params.u = [float(limb_darkening_coefficients[0]), float(limb_darkening_coefficients[1])]
    return params


def model_flux(
    time,
    params,
    *,
    supersample_factor: int = 7,
    exposure_time_days: float | None = None,
):
    t = np.asarray(time, dtype=float).reshape(-1)
    if not np.isfinite(t).all():
        raise ValueError("BATMAN model times must be finite.")
    kwargs = {}
    if exposure_time_days is not None and float(exposure_time_days) > 0:
        kwargs["supersample_factor"] = int(supersample_factor)
        kwargs["exp_time"] = float(exposure_time_days)
    model = batman.TransitModel(params, t, **kwargs)
    return np.asarray(model.light_curve(params), dtype=float)


def _local_realized_depth(
    radius_ratio: float,
    *,
    period_days: float,
    duration_days: float,
    impact_parameter: float,
    limb_darkening_coefficients,
    exposure_time_days: float,
    supersample_factor: int,
) -> float:
    a_over_rstar, inclination = circular_geometry_from_t14(
        period_days, duration_days, radius_ratio, impact_parameter
    )
    params = make_batman_params(
        epoch_days=0.0,
        period_days=period_days,
        radius_ratio=radius_ratio,
        scaled_semimajor_axis=a_over_rstar,
        inclination_degrees=inclination,
        limb_darkening_coefficients=limb_darkening_coefficients,
    )
    window = max(float(duration_days), float(exposure_time_days) * 2.0)
    local_time = np.linspace(-window, window, 401)
    flux = model_flux(
        local_time,
        params,
        supersample_factor=supersample_factor,
        exposure_time_days=exposure_time_days,
    )
    return float(1.0 - np.nanmin(flux))


def solve_radius_ratio_for_observed_depth(
    *,
    period_days: float,
    duration_days: float,
    requested_depth: float,
    impact_parameter: float = 0.3,
    limb_darkening_coefficients=(0.3, 0.2),
    exposure_time_days: float,
    supersample_factor: int = 7,
) -> float:
    """Solve Rp/R* so the exposure-integrated model reaches requested depth."""
    depth = float(requested_depth)
    if depth <= 0 or depth >= 0.25:
        raise ValueError("requested_depth must be in (0, 0.25).")

    def objective(rp):
        realized = _local_realized_depth(
            rp,
            period_days=period_days,
            duration_days=duration_days,
            impact_parameter=impact_parameter,
            limb_darkening_coefficients=limb_darkening_coefficients,
            exposure_time_days=exposure_time_days,
            supersample_factor=supersample_factor,
        )
        return realized - depth

    lower = 1.0e-4
    upper = max(0.05, 3.0 * np.sqrt(depth))
    upper = min(upper, 0.5)
    f_lower = objective(lower)
    f_upper = objective(upper)
    while f_upper < 0 and upper < 0.8:
        upper = min(0.8, upper * 1.5)
        f_upper = objective(upper)
    if f_lower > 0 or f_upper < 0:
        raise ValueError(
            f"Could not bracket Rp/R* for requested depth={depth:g}; "
            f"objective bounds=({f_lower:g}, {f_upper:g})."
        )
    return float(brentq(objective, lower, upper, xtol=1e-10, rtol=1e-10, maxiter=100))


def inject_batman_transit(
    time,
    values,
    *,
    period_days: float,
    epoch_days: float,
    duration_days: float,
    depth: float,
    impact_parameter: float = 0.3,
    limb_darkening_coefficients=(0.3, 0.2),
    supersample_factor: int = 7,
    exposure_time_days: float | None = None,
):
    """Inject a limb-darkened BATMAN transit into a zero-centered light curve.

    Returns
    -------
    injected_values, additive_template, in_transit_mask, truth
    """
    t = np.asarray(time, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    if t.shape != y.shape:
        raise ValueError("time and values must have the same shape.")
    finite_time = np.isfinite(t)
    if finite_time.sum() < 20:
        raise ValueError("At least 20 finite time values are required.")
    exp_time = (
        median_positive_cadence_days(t)
        if exposure_time_days is None
        else float(exposure_time_days)
    )

    rp = solve_radius_ratio_for_observed_depth(
        period_days=period_days,
        duration_days=duration_days,
        requested_depth=depth,
        impact_parameter=impact_parameter,
        limb_darkening_coefficients=limb_darkening_coefficients,
        exposure_time_days=exp_time,
        supersample_factor=supersample_factor,
    )
    a_over_rstar, inclination = circular_geometry_from_t14(
        period_days, duration_days, rp, impact_parameter
    )
    params = make_batman_params(
        epoch_days=epoch_days,
        period_days=period_days,
        radius_ratio=rp,
        scaled_semimajor_axis=a_over_rstar,
        inclination_degrees=inclination,
        limb_darkening_coefficients=limb_darkening_coefficients,
    )

    unity_flux = np.ones(t.shape, dtype=float)
    unity_flux[finite_time] = model_flux(
        t[finite_time],
        params,
        supersample_factor=supersample_factor,
        exposure_time_days=exp_time,
    )
    additive_template = unity_flux - 1.0

    injected = y.copy()
    usable = np.isfinite(y) & finite_time
    injected[usable] = y[usable] + additive_template[usable]
    # BATMAN can return tiny floating roundoff away from transit; use a threshold
    # relative to the requested depth rather than exact inequality to zero.
    in_transit = finite_time & (additive_template < -max(float(depth) * 1.0e-6, 1.0e-12))
    realized_depth = float(-np.nanmin(additive_template[finite_time]))

    truth = BatmanInjectionTruth(
        period_days=float(period_days),
        epoch_days=float(epoch_days),
        requested_duration_days=float(duration_days),
        requested_depth=float(depth),
        radius_ratio=rp,
        scaled_semimajor_axis=a_over_rstar,
        inclination_degrees=inclination,
        impact_parameter=float(impact_parameter),
        eccentricity=0.0,
        omega_degrees=90.0,
        limb_darkening_law="quadratic",
        limb_darkening_u1=float(limb_darkening_coefficients[0]),
        limb_darkening_u2=float(limb_darkening_coefficients[1]),
        exposure_time_days=exp_time,
        supersample_factor=int(supersample_factor),
        realized_max_depth_on_observed_cadences=realized_depth,
    )
    return injected, additive_template, in_transit, truth
