"""Matched-filter utilities for ARIMA-transformed transit templates."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from adaptive_transit.noise_models.arima import (
    FittedArimaModel,
    apply_fitted_arima_filter,
)
from adaptive_transit.noise_models.scaling import trailing_robust_scale
from adaptive_transit.transit_models.box import box_transit_template


@dataclass(frozen=True)
class MatchedFilterScore:
    """Small audit record for one matched-filter calculation."""

    statistic: float
    numerator: float
    denominator: float
    n_points: int

    def to_dict(self, prefix: str = "") -> dict[str, float | int]:
        return {f"{prefix}{key}": value for key, value in asdict(self).items()}


@dataclass(frozen=True)
class ArimaTemplateTransform:
    """Fixed-ARIMA transformation of a synthetic transit template."""

    base_innovations: np.ndarray
    injected_innovations: np.ndarray
    transformed_template: np.ndarray
    usable_mask: np.ndarray


def local_window_mask(
    cadenceno: np.ndarray,
    *,
    center_cadenceno: int,
    half_width_cadences: int,
) -> np.ndarray:
    """Return a local cadence window around one trial transit center."""

    if half_width_cadences < 1:
        raise ValueError("half_width_cadences must be positive.")
    cadence = np.asarray(cadenceno, dtype=float)
    return np.abs(cadence - center_cadenceno) <= half_width_cadences


def select_trial_centers(
    cadenceno: np.ndarray,
    usable_mask: np.ndarray,
    *,
    stride: int = 1,
    max_centers: int | None = None,
    required_centers: tuple[int, ...] = (),
) -> np.ndarray:
    """Choose trial centers for a blind single-duration template scan."""

    if stride < 1:
        raise ValueError("stride must be at least 1.")
    if max_centers is not None and max_centers < 1:
        raise ValueError("max_centers must be positive when provided.")

    cadence = np.asarray(cadenceno, dtype=float).reshape(-1)
    usable = np.asarray(usable_mask, dtype=bool).reshape(-1)
    if cadence.shape != usable.shape:
        raise ValueError("cadenceno and usable_mask must have the same shape.")

    eligible_indices = np.flatnonzero(usable & np.isfinite(cadence))
    eligible_indices = eligible_indices[::stride]

    if max_centers is not None and eligible_indices.size > max_centers:
        selected_positions = np.linspace(
            0,
            eligible_indices.size - 1,
            max_centers,
            dtype=int,
        )
        eligible_indices = eligible_indices[np.unique(selected_positions)]

    centers = {int(cadence[index]) for index in eligible_indices}
    centers.update(int(center) for center in required_centers if np.isfinite(center))
    return np.asarray(sorted(centers), dtype=int)


def matched_filter_statistic(
    observed: np.ndarray,
    template: np.ndarray,
    *,
    scale: np.ndarray | None = None,
    usable_mask: np.ndarray | None = None,
    demean_template: bool = True,
) -> MatchedFilterScore:
    """Compute a weighted matched-filter response.

    The template is demeaned by default so the statistic responds to a transit
    shape rather than to a local constant offset in the light curve.
    """

    y = np.asarray(observed, dtype=float).reshape(-1)
    h = np.asarray(template, dtype=float).reshape(-1)
    if y.shape != h.shape:
        raise ValueError("observed and template must have the same shape.")

    if usable_mask is None:
        usable = np.ones(y.shape, dtype=bool)
    else:
        usable = np.asarray(usable_mask, dtype=bool).reshape(-1).copy()
        if usable.shape != y.shape:
            raise ValueError("usable_mask must have the same shape as observed.")

    if scale is None:
        weights = np.ones(y.shape, dtype=float)
    else:
        sigma = np.asarray(scale, dtype=float).reshape(-1)
        if sigma.shape != y.shape:
            raise ValueError("scale must have the same shape as observed.")
        weights = np.zeros(y.shape, dtype=float)
        finite_scale = np.isfinite(sigma) & (sigma > 0)
        weights[finite_scale] = 1.0 / np.square(sigma[finite_scale])
        usable &= finite_scale

    valid = usable & np.isfinite(y) & np.isfinite(h) & np.isfinite(weights) & (weights > 0)
    if valid.sum() < 3:
        return MatchedFilterScore(
            statistic=float("nan"),
            numerator=float("nan"),
            denominator=float("nan"),
            n_points=int(valid.sum()),
        )

    y_valid = y[valid]
    h_valid = h[valid]
    w_valid = weights[valid]

    if demean_template:
        template_mean = np.average(h_valid, weights=w_valid)
        h_valid = h_valid - template_mean

    numerator = float(np.sum(w_valid * h_valid * y_valid))
    denominator = float(np.sqrt(np.sum(w_valid * h_valid * h_valid)))
    statistic = float("nan") if denominator <= 0 or not np.isfinite(denominator) else float(numerator / denominator)

    return MatchedFilterScore(
        statistic=statistic,
        numerator=numerator,
        denominator=denominator,
        n_points=int(valid.sum()),
    )


def arima_transformed_template(
    values: np.ndarray,
    template: np.ndarray,
    fitted_model: FittedArimaModel,
    *,
    allow_missing: bool,
) -> ArimaTemplateTransform:
    """Transform a template through the selected ARIMA prediction operator.

    We filter the original series and the original-plus-template series with
    fixed ARIMA coefficients, then subtract their innovations. That difference
    is the template shape that the matched filter should use in innovation
    space.
    """

    base = np.asarray(values, dtype=float).reshape(-1)
    additive_template = np.asarray(template, dtype=float).reshape(-1)
    if base.shape != additive_template.shape:
        raise ValueError("values and template must have the same shape.")

    injected = base.copy()
    finite = np.isfinite(injected) & np.isfinite(additive_template)
    injected[finite] = injected[finite] + additive_template[finite]

    if fitted_model.innovations.shape != base.shape or fitted_model.usable_mask.shape != base.shape:
        raise ValueError("fitted_model arrays must have the same length as values.")

    injected_fit = apply_fitted_arima_filter(
        injected,
        fitted_model,
        allow_missing=allow_missing,
    )

    transformed_template = injected_fit.innovations - fitted_model.innovations
    usable = fitted_model.usable_mask & injected_fit.usable_mask & np.isfinite(transformed_template) & np.isfinite(additive_template)

    return ArimaTemplateTransform(
        base_innovations=fitted_model.innovations,
        injected_innovations=injected_fit.innovations,
        transformed_template=transformed_template,
        usable_mask=usable,
    )


def _statistic_rank(series: pd.Series) -> pd.Series:
    """Rank a detection statistic with the largest finite value ranked first."""

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(pd.array([pd.NA] * len(numeric), dtype="Int64"), index=series.index)
    filled = numeric.fillna(float("-inf"))
    return filled.rank(method="min", ascending=False).astype("Int64")


def scan_arima_transformed_template(
    cadenceno: np.ndarray,
    values: np.ndarray,
    observed_values: np.ndarray,
    fitted_model: FittedArimaModel,
    trial_centers: np.ndarray,
    *,
    duration_cadences: int,
    depth: float,
    local_half_width_cadences: int,
    scale_window: int,
    allow_missing: bool,
    usable_mask: np.ndarray | None = None,
    injected_center_cadenceno: int | None = None,
    injected_neighborhood_cadences: int | None = None,
) -> pd.DataFrame:
    """Scan trial centers with raw and ARIMA-transformed matched filters.

    `observed_values` is the searched light curve. In the first Stage 1 blind
    test this is the original light curve plus one hidden injected transit.
    Trial templates are generated independently at many possible centers.
    """

    cadence = np.asarray(cadenceno, dtype=int).reshape(-1)
    base_values = np.asarray(values, dtype=float).reshape(-1)
    observed = np.asarray(observed_values, dtype=float).reshape(-1)
    centers = np.asarray(trial_centers, dtype=int).reshape(-1)
    if cadence.shape != base_values.shape or cadence.shape != observed.shape:
        raise ValueError("cadenceno, values, and observed_values must have the same shape.")

    if usable_mask is None:
        usable = np.isfinite(base_values) & np.isfinite(observed)
    else:
        usable = np.asarray(usable_mask, dtype=bool).reshape(-1)
        if usable.shape != cadence.shape:
            raise ValueError("usable_mask must have the same shape as values.")

    observed_fit = apply_fitted_arima_filter(
        observed,
        fitted_model,
        allow_missing=allow_missing,
    )
    raw_flux_scale = trailing_robust_scale(base_values, window=scale_window)
    innovation_scale = trailing_robust_scale(fitted_model.innovations, window=scale_window)
    transformed_observed_mask = observed_fit.usable_mask & fitted_model.usable_mask

    neighborhood = max(1, duration_cadences // 2) if injected_neighborhood_cadences is None else injected_neighborhood_cadences
    rows: list[dict[str, float | int | bool]] = []
    for center in centers:
        box_template, _ = box_transit_template(
            cadence,
            center_cadenceno=int(center),
            duration_cadences=duration_cadences,
            depth=depth,
        )
        local = local_window_mask(
            cadence,
            center_cadenceno=int(center),
            half_width_cadences=local_half_width_cadences,
        )
        transform = arima_transformed_template(
            base_values,
            box_template,
            fitted_model,
            allow_missing=allow_missing,
        )

        raw_score = matched_filter_statistic(
            observed,
            box_template,
            scale=raw_flux_scale,
            usable_mask=local & usable,
        )
        unchanged_score = matched_filter_statistic(
            observed_fit.innovations,
            box_template,
            scale=innovation_scale,
            usable_mask=local & transformed_observed_mask,
        )
        transformed_score = matched_filter_statistic(
            observed_fit.innovations,
            transform.transformed_template,
            scale=innovation_scale,
            usable_mask=local & transformed_observed_mask & transform.usable_mask,
        )

        row: dict[str, float | int | bool] = {
            "trial_center_cadenceno": int(center),
            "duration_cadences": int(duration_cadences),
            **raw_score.to_dict("raw_flux_box_"),
            **unchanged_score.to_dict("innovation_unchanged_box_"),
            **transformed_score.to_dict("innovation_transformed_template_"),
        }
        if injected_center_cadenceno is not None:
            offset = int(center) - int(injected_center_cadenceno)
            row["injected_center_cadenceno"] = int(injected_center_cadenceno)
            row["center_offset_cadences"] = offset
            row["is_injected_center_neighborhood"] = bool(abs(offset) <= neighborhood)
        rows.append(row)

    scan = pd.DataFrame(rows)
    if scan.empty:
        return scan

    for column in (
        "raw_flux_box_statistic",
        "innovation_unchanged_box_statistic",
        "innovation_transformed_template_statistic",
    ):
        scan[column.replace("_statistic", "_rank")] = _statistic_rank(scan[column])
    scan["rank"] = scan["innovation_transformed_template_rank"]
    return scan.sort_values("trial_center_cadenceno").reset_index(drop=True)
