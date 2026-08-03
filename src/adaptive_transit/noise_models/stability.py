"""Order-selection stability checks across folds and contiguous segments."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from adaptive_transit.noise_models.arima import ArimaOrder, evaluate_arima_candidates
from adaptive_transit.noise_models.selection import score_arima_candidates, select_noise_model


@dataclass(frozen=True)
class StabilitySummary:
    """Compact summary of how stable order selection was."""

    stability_kind: str
    n_runs: int
    n_successful_runs: int
    modal_order: str
    modal_order_fraction: float
    unique_selected_orders: int
    stable: bool

    def to_dict(self) -> dict[str, str | int | float | bool]:
        return asdict(self)


def _safe_select(
    values: np.ndarray,
    orders: Iterable[ArimaOrder],
    *,
    mode: str,
    allow_missing: bool,
    test_fraction: float,
    acf_lags: int,
) -> tuple[str, float, str]:
    results = evaluate_arima_candidates(
        values,
        orders,
        mode=mode,
        allow_missing=allow_missing,
        test_fraction=test_fraction,
        acf_lags=acf_lags,
    )
    scored = score_arima_candidates(results)
    selected = select_noise_model(scored)
    return (
        str(selected["order"]),
        float(selected["adequacy_score"]),
        str(selected["failure_reason"]),
    )


def chronological_prefix_stability(
    values: np.ndarray,
    orders: Sequence[ArimaOrder],
    *,
    mode: str,
    allow_missing: bool,
    test_fraction: float = 0.20,
    acf_lags: int = 80,
    prefix_fractions: Sequence[float] = (0.55, 0.70, 0.85, 1.0),
) -> pd.DataFrame:
    """Check whether selected order changes across growing chronological prefixes."""

    series = np.asarray(values, dtype=float).reshape(-1)
    observed_positions = np.flatnonzero(np.isfinite(series))
    rows: list[dict[str, str | int | float]] = []

    for fold_index, fraction in enumerate(prefix_fractions, start=1):
        observed_count = max(10, int(round(len(observed_positions) * fraction)))
        observed_count = min(observed_count, len(observed_positions))
        end_position = int(observed_positions[observed_count - 1]) + 1
        prefix = series[:end_position]

        try:
            selected_order, score, failure_reason = _safe_select(
                prefix,
                orders,
                mode=mode,
                allow_missing=allow_missing,
                test_fraction=test_fraction,
                acf_lags=acf_lags,
            )
        except Exception as exc:  # noqa: BLE001 - stability should report failures.
            selected_order = ""
            score = float("nan")
            failure_reason = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "stability_kind": "chronological_prefix",
                "mode": mode,
                "fold": fold_index,
                "observed_fraction": float(fraction),
                "n_total": int(prefix.size),
                "n_observed": int(np.isfinite(prefix).sum()),
                "selected_order": selected_order,
                "adequacy_score": score,
                "failure_reason": failure_reason,
            }
        )

    return pd.DataFrame(rows)


def segment_stability(
    segments: Sequence[tuple[int, np.ndarray]],
    orders: Sequence[ArimaOrder],
    *,
    test_fraction: float = 0.20,
    acf_lags: int = 80,
    max_segments: int = 3,
) -> pd.DataFrame:
    """Check selected orders on the longest contiguous usable segments."""

    rows: list[dict[str, str | int | float]] = []
    for run_index, (segment_id, values) in enumerate(segments[:max_segments], start=1):
        try:
            selected_order, score, failure_reason = _safe_select(
                values,
                orders,
                mode="segment",
                allow_missing=False,
                test_fraction=test_fraction,
                acf_lags=acf_lags,
            )
        except Exception as exc:  # noqa: BLE001 - stability should report failures.
            selected_order = ""
            score = float("nan")
            failure_reason = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "stability_kind": "segment",
                "mode": "segment",
                "fold": run_index,
                "segment_id": int(segment_id),
                "n_total": int(len(values)),
                "n_observed": int(np.isfinite(values).sum()),
                "selected_order": selected_order,
                "adequacy_score": score,
                "failure_reason": failure_reason,
            }
        )

    return pd.DataFrame(rows)


def summarize_stability(
    stability: pd.DataFrame,
    *,
    stable_fraction_threshold: float = 0.60,
) -> StabilitySummary:
    """Summarize whether the same order is selected often enough."""

    if stability.empty:
        return StabilitySummary(
            stability_kind="unknown",
            n_runs=0,
            n_successful_runs=0,
            modal_order="",
            modal_order_fraction=0.0,
            unique_selected_orders=0,
            stable=False,
        )

    successful = stability[stability["selected_order"].astype(str) != ""]
    if successful.empty:
        return StabilitySummary(
            stability_kind=str(stability["stability_kind"].iloc[0]),
            n_runs=int(len(stability)),
            n_successful_runs=0,
            modal_order="",
            modal_order_fraction=0.0,
            unique_selected_orders=0,
            stable=False,
        )

    counts = successful["selected_order"].value_counts()
    modal_order = str(counts.index[0])
    modal_fraction = float(counts.iloc[0] / len(successful))
    return StabilitySummary(
        stability_kind=str(stability["stability_kind"].iloc[0]),
        n_runs=int(len(stability)),
        n_successful_runs=int(len(successful)),
        modal_order=modal_order,
        modal_order_fraction=modal_fraction,
        unique_selected_orders=int(successful["selected_order"].nunique()),
        stable=bool(modal_fraction >= stable_fraction_threshold),
    )
