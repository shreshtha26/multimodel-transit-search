"""Order-selection stability checks across folds and contiguous segments."""

import numpy as np
import pandas as pd
from adaptive_transit.noise_models.arima import evaluate_arima_candidates
from adaptive_transit.noise_models.selection import score_arima_candidates, select_noise_model


class StabilitySummary:
    def __init__(self, stability_kind, n_runs, n_successful_runs, modal_order, modal_order_fraction, unique_selected_orders, stable):
        self.stability_kind = stability_kind
        self.n_runs = n_runs
        self.n_successful_runs = n_successful_runs
        self.modal_order = modal_order
        self.modal_order_fraction = modal_order_fraction
        self.unique_selected_orders = unique_selected_orders
        self.stable = stable

    def to_dict(self):
        return self.__dict__.copy()


def _safe_select(values, orders, mode, allow_missing, test_fraction, acf_lags, fit_maxiter=None):
    results = evaluate_arima_candidates(
        values,
        orders,
        mode=mode,
        allow_missing=allow_missing,
        test_fraction=test_fraction,
        acf_lags=acf_lags,
        fit_maxiter=fit_maxiter,
    )
    scored = score_arima_candidates(results)
    selected = select_noise_model(scored)
    return (
        str(selected["order"]),
        float(selected["adequacy_score"]),
        str(selected["failure_reason"]),
    )


def chronological_prefix_stability(values, orders, mode, allow_missing, test_fraction=0.20, acf_lags=80, prefix_fractions=(0.55, 0.70, 0.85, 1.0), fit_maxiter=None):
    series = np.asarray(values, dtype=float).reshape(-1)
    observed_positions = np.flatnonzero(np.isfinite(series))
    rows = []

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
                fit_maxiter=fit_maxiter,
            )
        except Exception as exc:
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


def segment_stability(segments, orders, test_fraction=0.20, acf_lags=80, max_segments=3, fit_maxiter=None):
    rows = []
    for run_index, (segment_id, values) in enumerate(segments[:max_segments], start=1):
        try:
            selected_order, score, failure_reason = _safe_select(
                values,
                orders,
                mode="segment",
                allow_missing=False,
                test_fraction=test_fraction,
                acf_lags=acf_lags,
                fit_maxiter=fit_maxiter,
            )
        except Exception as exc:
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


def summarize_stability(stability, stable_fraction_threshold=0.60):
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
