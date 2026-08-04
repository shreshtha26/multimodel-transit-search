"""ACF/PACF diagnostic plot helpers."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf, pacf


@dataclass(frozen=True)
class CorrelogramPlotRecord:
    """Machine-readable record for one requested correlogram plot."""

    gap_mode: str
    series_kind: str
    plot_kind: str
    generated: bool
    path: str
    reason: str
    n_observations: int
    max_lag: int
    missing_observations_compressed: bool
    interpolation_used: bool
    missing_strategy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def skipped_correlogram_record(
    *,
    gap_mode: str,
    series_kind: str,
    plot_kind: str,
    path: Path,
    reason: str,
    n_observations: int,
    max_lag: int,
    missing_observations_compressed: bool,
    interpolation_used: bool,
    missing_strategy: str,
) -> CorrelogramPlotRecord:
    """Return a plot record for an intentionally omitted diagnostic."""

    return CorrelogramPlotRecord(
        gap_mode=gap_mode,
        series_kind=series_kind,
        plot_kind=plot_kind,
        generated=False,
        path=str(path),
        reason=reason,
        n_observations=int(n_observations),
        max_lag=int(max_lag),
        missing_observations_compressed=bool(missing_observations_compressed),
        interpolation_used=bool(interpolation_used),
        missing_strategy=missing_strategy,
    )


def correlogram_values(
    values: np.ndarray,
    *,
    plot_kind: str,
    max_lag: int,
    missing_strategy: str = "none",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute ACF or PACF values without hiding missing-data assumptions."""

    series = np.asarray(values, dtype=float).reshape(-1)
    n_observed = int(np.isfinite(series).sum())
    if n_observed < 4:
        raise ValueError("Correlogram plots require at least four finite observations.")
    usable_lag = min(int(max_lag), n_observed - 2)
    if usable_lag < 1:
        raise ValueError("No positive lag is available for correlogram plotting.")

    if plot_kind == "acf":
        missing = "conservative" if missing_strategy == "conservative" else "none"
        values_out = acf(series, nlags=usable_lag, fft=True, missing=missing)
    elif plot_kind == "pacf":
        if not np.all(np.isfinite(series)):
            raise ValueError("PACF is not computed on missing-valued series without an explicit finite representation.")
        values_out = pacf(series, nlags=usable_lag, method="ywm")
    else:
        raise ValueError("plot_kind must be 'acf' or 'pacf'.")
    lags = np.arange(len(values_out), dtype=int)
    return lags, np.asarray(values_out, dtype=float), n_observed


def save_correlogram_plot(
    values: np.ndarray,
    path: Path,
    *,
    gap_mode: str,
    series_kind: str,
    plot_kind: str,
    max_lag: int,
    cadence_days: float,
    missing_observations_compressed: bool,
    interpolation_used: bool,
    missing_strategy: str,
    transit_lag_range: tuple[int, int],
    annotation: str,
) -> CorrelogramPlotRecord:
    """Save one ACF or PACF diagnostic plot with explicit assumptions."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lags, coefficients, n_observed = correlogram_values(
            values,
            plot_kind=plot_kind,
            max_lag=max_lag,
            missing_strategy=missing_strategy,
        )
    except ValueError as exc:
        return skipped_correlogram_record(
            gap_mode=gap_mode,
            series_kind=series_kind,
            plot_kind=plot_kind,
            path=path,
            reason=str(exc),
            n_observations=int(np.isfinite(np.asarray(values, dtype=float)).sum()),
            max_lag=max_lag,
            missing_observations_compressed=missing_observations_compressed,
            interpolation_used=interpolation_used,
            missing_strategy=missing_strategy,
        )

    confidence = 1.96 / np.sqrt(n_observed)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axhline(confidence, color="0.45", linestyle="--", linewidth=1, label="approx. 95% limits")
    ax.axhline(-confidence, color="0.45", linestyle="--", linewidth=1)
    ax.axvspan(
        transit_lag_range[0],
        transit_lag_range[1],
        color="0.90",
        zorder=-1,
        label="transit-relevant lags",
    )
    ax.vlines(lags, 0.0, coefficients, color="tab:blue", linewidth=1)
    ax.plot(lags, coefficients, "o", color="tab:blue", markersize=3)
    ax.set_xlim(-0.5, max_lag + 0.5)
    ax.set_xlabel("Lag [cadences]")
    ax.set_ylabel(plot_kind.upper())
    ax.set_title(f"{gap_mode}: {series_kind} {plot_kind.upper()}")
    ax.text(
        0.01,
        0.02,
        f"n={n_observed}, cadence={cadence_days:.6g} d\n{annotation}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return CorrelogramPlotRecord(
        gap_mode=gap_mode,
        series_kind=series_kind,
        plot_kind=plot_kind,
        generated=True,
        path=str(path),
        reason="",
        n_observations=n_observed,
        max_lag=int(max_lag),
        missing_observations_compressed=bool(missing_observations_compressed),
        interpolation_used=bool(interpolation_used),
        missing_strategy=missing_strategy,
    )
