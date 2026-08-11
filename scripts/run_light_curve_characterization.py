"""Generate a pre-model statistical characterization record for one light curve."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.noise_models.characterization import (
    characterize_regularized_light_curve,
    json_ready,
)
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve


TARGET_ID = "11904151"
QUARTER = 5
OUTPUT_DIR = Path("outputs/experiments/characterization")


def default_settings():
    return SimpleNamespace(
        target_id=TARGET_ID,
        quarter=QUARTER,
        output_dir=OUTPUT_DIR,
        quality_policy="default",
        require_finite_flux_error=False,
        test_fraction=0.20,
        acf_lags=80,
        ljung_box_lags=(10, 20, 40),
        rolling_window=96,
        outlier_sigma=5.0,
        stationarity_alpha=0.05,
        stationarity_min_observations=24,
        spectral_frequencies=2000,
    )


def parse_lag_grid(value):
    values = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected positive comma-separated Ljung-Box lags.")
    return values


def build_parser():
    defaults = default_settings()
    parser = argparse.ArgumentParser(description="Generate pre-model light-curve diagnostics.")
    parser.add_argument("--target-id", default=defaults.target_id)
    parser.add_argument("--quarter", type=int, default=defaults.quarter)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--quality-policy", default=defaults.quality_policy)
    parser.add_argument("--require-finite-flux-error", action="store_true")
    parser.add_argument("--test-fraction", type=float, default=defaults.test_fraction)
    parser.add_argument("--acf-lags", type=int, default=defaults.acf_lags)
    parser.add_argument("--ljung-box-lags", type=parse_lag_grid, default=defaults.ljung_box_lags)
    parser.add_argument("--rolling-window", type=int, default=defaults.rolling_window)
    parser.add_argument("--outlier-sigma", type=float, default=defaults.outlier_sigma)
    parser.add_argument("--stationarity-alpha", type=float, default=defaults.stationarity_alpha)
    parser.add_argument("--stationarity-min-observations", type=int, default=defaults.stationarity_min_observations)
    parser.add_argument("--spectral-frequencies", type=int, default=defaults.spectral_frequencies)
    return parser


def target_prefix(target_id, quarter):
    clean_target_id = str(target_id).replace("KIC", "").strip()
    return f"kic_{clean_target_id}_q{quarter}"


def run_experiment(args):
    light_curve = load_kepler_pdcsap(args.target_id, args.quarter)
    regular, preprocessing = preprocess_pdcsap_light_curve(
        light_curve.to_dataframe(),
        quality_policy=args.quality_policy,
        require_finite_flux_error=args.require_finite_flux_error,
        normalization_fit_fraction=1.0 - args.test_fraction,
    )
    diagnostics = characterize_regularized_light_curve(
        regular,
        target_id=str(args.target_id).replace("KIC", "").strip(),
        quarter=int(args.quarter),
        preprocessing_summary=preprocessing.to_dict(),
        acf_lags=args.acf_lags,
        ljung_box_lags=tuple(args.ljung_box_lags),
        rolling_window=args.rolling_window,
        outlier_sigma=args.outlier_sigma,
        stationarity_alpha=args.stationarity_alpha,
        stationarity_min_observations=args.stationarity_min_observations,
        spectral_frequencies=args.spectral_frequencies,
    )
    return regular, diagnostics


def main(args=None):
    args = args or build_parser().parse_args()
    metrics_dir = Path(args.output_dir) / "metrics"
    processed_dir = Path(args.output_dir) / "processed"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    regular, diagnostics = run_experiment(args)
    prefix = target_prefix(args.target_id, args.quarter)
    json_path = metrics_dir / f"{prefix}_light_curve_diagnostics.json"
    csv_path = metrics_dir / f"{prefix}_light_curve_diagnostics.csv"
    regular_path = processed_dir / f"{prefix}_characterization_input.parquet"

    json_path.write_text(json.dumps(json_ready(diagnostics), indent=2) + "\n")
    pd.DataFrame([json_ready(diagnostics)]).to_csv(csv_path, index=False)
    regular.to_parquet(regular_path, index=False)

    print(f"Light-curve diagnostics: {json_path}")
    print(f"Light-curve diagnostics CSV: {csv_path}")
    print(f"Characterization input: {regular_path}")
    print(f"Stationarity: {diagnostics['original_series_stationarity_conclusion']}")
    print(f"Minimum Ljung-Box p: {diagnostics['minimum_ljung_box_p']}")
    print(f"Dominant period: {diagnostics['dominant_period_days']} days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
