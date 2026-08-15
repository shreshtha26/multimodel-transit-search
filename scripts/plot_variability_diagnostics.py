"""Visual validation panels for stellar-variability characterization.

This script is intentionally diagnostic rather than classificatory.

For each selected KIC it plots:
1. the Q5 characterization light curve;
2. the autocorrelation function;
3. the full-light-curve Lomb-Scargle periodogram;
4. Lomb-Scargle periodograms for the first and second halves.

The purpose is to distinguish:
- stable coherent periodicity,
- evolving / quasi-periodic morphology,
- harmonically structured variability,
- genuinely multi-frequency behaviour.

No astrophysical classification is made from these plots alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from statsmodels.tsa.stattools import acf


DEFAULT_KICS = [
    10677499,
    10718455,
    8038388,
    11450313,
]


def _find_column(
    frame: pd.DataFrame,
    preferred: list[str],
    contains: str,
) -> str:
    """Return the first suitable column name."""

    for column in preferred:
        if column in frame.columns:
            return column

    matches = [
        column
        for column in frame.columns
        if contains.lower() in column.lower()
    ]

    if not matches:
        raise KeyError(
            f"Could not identify a column containing {contains!r}. "
            f"Available columns: {frame.columns.tolist()}"
        )

    return matches[0]


def _load_light_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(path)

    time_column = _find_column(
        frame,
        preferred=["time", "time_days"],
        contains="time",
    )

    flux_column = _find_column(
        frame,
        preferred=["normalized_flux"],
        contains="flux",
    )

    time = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(float)
    flux = pd.to_numeric(frame[flux_column], errors="coerce").to_numpy(float)

    finite = np.isfinite(time) & np.isfinite(flux)
    time = time[finite]
    flux = flux[finite]

    order = np.argsort(time)
    return time[order], flux[order]


def _frequency_grid(
    time: np.ndarray,
    minimum_cycles: float = 2.0,
    samples: int = 12000,
) -> np.ndarray:
    baseline = float(np.nanmax(time) - np.nanmin(time))

    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError("Invalid observing baseline.")

    cadence = float(np.nanmedian(np.diff(time)))

    minimum_frequency = minimum_cycles / baseline

    # Conservative Nyquist-like upper limit for the sampled series.
    maximum_frequency = 0.5 / cadence

    return np.linspace(
        minimum_frequency,
        maximum_frequency,
        samples,
    )


def _periodogram(
    time: np.ndarray,
    flux: np.ndarray,
    frequency: np.ndarray,
) -> np.ndarray:
    centered_flux = flux - np.nanmedian(flux)

    model = LombScargle(
        time,
        centered_flux,
        normalization="standard",
    )

    return np.asarray(model.power(frequency), dtype=float)


def _acf_values(
    time: np.ndarray,
    flux: np.ndarray,
    nlags: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    centered = flux - np.nanmedian(flux)

    nlag = min(int(nlags), max(1, len(centered) - 1))

    values = acf(
        centered,
        nlags=nlag,
        fft=True,
        missing="drop",
    )

    cadence = float(np.nanmedian(np.diff(time)))
    lag_days = np.arange(len(values), dtype=float) * cadence

    return lag_days, np.asarray(values, dtype=float)


def _dominant_period(
    frequency: np.ndarray,
    power: np.ndarray,
) -> float:
    finite = np.isfinite(power)

    if not np.any(finite):
        return float("nan")

    idx = int(np.nanargmax(power))

    if frequency[idx] <= 0:
        return float("nan")

    return float(1.0 / frequency[idx])


def plot_star(
    kic: int,
    quarter: int,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    path = (
        input_dir
        / f"kic_{kic}_q{quarter}_characterization_input.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    time, flux = _load_light_curve(path)

    frequency = _frequency_grid(time)
    power = _periodogram(time, flux, frequency)

    period = 1.0 / frequency
    dominant_period = _dominant_period(frequency, power)

    lag_days, acf_power = _acf_values(time, flux)

    midpoint = float(np.nanmedian(time))

    first = time <= midpoint
    second = time > midpoint

    first_power = _periodogram(
        time[first],
        flux[first],
        frequency,
    )

    second_power = _periodogram(
        time[second],
        flux[second],
        frequency,
    )

    # One figure per star; panels are deliberately separated within the
    # scientific diagnostic figure rather than combining different stars.
    fig = plt.figure(figsize=(12, 14))

    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(time, flux, ".", markersize=1)
    ax1.set_title(
        f"KIC {kic} — Q{quarter} variability diagnostics"
    )
    ax1.set_xlabel("Time [days]")
    ax1.set_ylabel("Normalized flux")

    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(lag_days, acf_power)
    ax2.axhline(0.0, linewidth=0.8)
    ax2.set_xlabel("Lag [days]")
    ax2.set_ylabel("ACF")
    ax2.set_title("Autocorrelation")

    ax3 = fig.add_subplot(4, 1, 3)

    # Periods are easier to interpret astrophysically than frequencies.
    valid_period = np.isfinite(period) & (period > 0)

    order = np.argsort(period[valid_period])

    plot_period = period[valid_period][order]
    plot_power = power[valid_period][order]

    ax3.plot(plot_period, plot_power)
    ax3.set_xlabel("Period [days]")
    ax3.set_ylabel("LS power")
    ax3.set_xscale("log")
    ax3.set_title(
        f"Full-quarter Lomb–Scargle "
        f"(dominant ≈ {dominant_period:.3f} d)"
    )

    ax4 = fig.add_subplot(4, 1, 4)

    plot_first = first_power[valid_period][order]
    plot_second = second_power[valid_period][order]

    ax4.plot(
        plot_period,
        plot_first,
        label="First half",
    )
    ax4.plot(
        plot_period,
        plot_second,
        label="Second half",
    )

    ax4.set_xlabel("Period [days]")
    ax4.set_ylabel("LS power")
    ax4.set_xscale("log")
    ax4.set_title("Temporal persistence of spectral structure")
    ax4.legend()

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir
        / f"kic_{kic}_q{quarter}_variability_diagnostics.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot stellar-variability validation diagnostics."
    )

    parser.add_argument(
        "--target-id",
        type=int,
        nargs="*",
        default=DEFAULT_KICS,
        help=(
            "KIC IDs. Defaults to the four current review candidates."
        ),
    )

    parser.add_argument(
        "--quarter",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "outputs/experiments/"
            "characterization_validation/processed"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/experiments/"
            "characterization_validation/diagnostic_plots"
        ),
    )

    args = parser.parse_args()

    for kic in args.target_id:
        output = plot_star(
            kic=kic,
            quarter=args.quarter,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )

        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
