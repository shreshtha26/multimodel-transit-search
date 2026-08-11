"""Repeat the GP time-scale sensitivity test across transit durations."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import run_gp_timescale_sensitivity as single_duration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/gp_multiduration_timescale_sensitivity"

def default_settings():
    args = single_duration.default_settings()
    args.output_dir = OUTPUT_DIR
    args.injection_duration_hours_grid = (2.0, 4.0, 8.0)
    args.optimized_reference = False
    return args

def parse_float_tuple(value):
    return tuple(float(item.strip()) for item in str(value).split(",") if item.strip())

def parse_args():
    defaults = default_settings()
    parser = argparse.ArgumentParser(description="Run GP length-scale/transit-duration sensitivity across multiple transit durations.")
    parser.add_argument("--duration-grid", default=",".join(str(value) for value in defaults.injection_duration_hours_grid))
    parser.add_argument("--length-scale-factors", default=",".join(str(value) for value in defaults.length_scale_factors))
    parser.add_argument("--gp-max-train-points", type=int, default=defaults.gp_max_train_points)
    parser.add_argument("--gp-measurement-noise-fraction", type=float, default=defaults.gp_measurement_noise_fraction)
    parser.add_argument("--injection-period-days", type=float, default=defaults.injection_period_days)
    parser.add_argument("--injection-depth", type=float, default=defaults.injection_depth)
    parser.add_argument("--epoch-phase-fraction", type=float, default=defaults.epoch_phase_fraction)
    parser.add_argument("--bls-n-periods", type=int, default=defaults.bls_n_periods)
    parser.add_argument("--tcf-n-periods", type=int, default=defaults.tcf_n_periods)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--include-optimized-reference", action="store_true")
    parsed = parser.parse_args()
    defaults.injection_duration_hours_grid = parse_float_tuple(parsed.duration_grid)
    defaults.length_scale_factors = parse_float_tuple(parsed.length_scale_factors)
    defaults.gp_max_train_points = int(parsed.gp_max_train_points)
    defaults.gp_measurement_noise_fraction = float(parsed.gp_measurement_noise_fraction)
    defaults.injection_period_days = float(parsed.injection_period_days)
    defaults.injection_depth = float(parsed.injection_depth)
    defaults.epoch_phase_fraction = float(parsed.epoch_phase_fraction)
    defaults.bls_n_periods = int(parsed.bls_n_periods)
    defaults.tcf_n_periods = int(parsed.tcf_n_periods)
    defaults.output_dir = Path(parsed.output_dir)
    defaults.optimized_reference = bool(parsed.include_optimized_reference)
    return defaults

def finite_corr(frame, x_column, y_column, method):
    subset = frame[[x_column, y_column]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < 3:
        return float("nan")
    return float(subset[x_column].corr(subset[y_column], method=method))

def grouped_by_ratio(frame):
    fixed = frame[~frame["optimize_kernel"]].copy()
    grouped = fixed.groupby("configured_timescale_ratio", as_index=False).agg(duration_count=("injected_duration_hours", "nunique"), configuration_count=("configuration", "size"), median_background_absorption_fraction=("background_absorption_fraction", "median"), min_background_absorption_fraction=("background_absorption_fraction", "min"), max_background_absorption_fraction=("background_absorption_fraction", "max"), median_depth_retention_fraction=("depth_retention_fraction", "median"), min_depth_retention_fraction=("depth_retention_fraction", "min"), max_depth_retention_fraction=("depth_retention_fraction", "max"), std_depth_retention_fraction=("depth_retention_fraction", "std"), median_snr_retention_fraction=("snr_retention_fraction", "median"), min_snr_retention_fraction=("snr_retention_fraction", "min"), max_snr_retention_fraction=("snr_retention_fraction", "max"), std_snr_retention_fraction=("snr_retention_fraction", "std"), median_residual_std=("residual_std", "median"), median_max_abs_residual_acf_1_24=("max_abs_residual_acf_1_24", "median"), bls_period_match_rate=("bls_period_matched", "mean"), tcf_period_match_rate=("tcf_period_matched", "mean"))
    return grouped.sort_values("configured_timescale_ratio").reset_index(drop=True)

def summarize(frame, by_ratio, args):
    fixed = frame[~frame["optimize_kernel"]].copy()
    return {"target_id": str(args.target_id), "quarter": int(args.quarter), "model": "smooth_anchor_gp", "hypothesis": "If time-scale separation controls GP transit preservation, rows with the same ell_GP / transit_duration should show similar absorption and retention across transit durations.", "not_fap_calibrated": True, "fap_note": "Detector scores are not formal FAP-calibrated recovery metrics because each fixed GP configuration would require its own null calibration.", "injected_period_days": float(args.injection_period_days), "injected_depth": float(args.injection_depth), "epoch_phase_fraction": float(args.epoch_phase_fraction), "duration_grid_hours": list(args.injection_duration_hours_grid), "length_scale_factors": list(args.length_scale_factors), "configuration_count": int(len(frame)), "fixed_configuration_count": int(len(fixed)), "spearman_ratio_depth_retention": finite_corr(fixed, "configured_timescale_ratio", "depth_retention_fraction", "spearman"), "spearman_ratio_snr_retention": finite_corr(fixed, "configured_timescale_ratio", "snr_retention_fraction", "spearman"), "spearman_ratio_background_absorption": finite_corr(fixed, "configured_timescale_ratio", "background_absorption_fraction", "spearman"), "spearman_ratio_residual_acf": finite_corr(fixed, "configured_timescale_ratio", "max_abs_residual_acf_1_24", "spearman"), "ratio_table_rows": int(len(by_ratio))}

def plot_ratio_collapse(frame, path):
    fixed = frame[~frame["optimize_kernel"]].copy()
    durations = sorted(fixed["injected_duration_hours"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    panels = [("depth_retention_fraction", "Depth retention"), ("snr_retention_fraction", "SNR retention"), ("background_absorption_fraction", "Background absorption"), ("max_abs_residual_acf_1_24", "Max residual ACF 1-24")]
    for axis, (column, label) in zip(axes.reshape(-1), panels):
        for duration in durations:
            subset = fixed[fixed["injected_duration_hours"] == duration].sort_values("configured_timescale_ratio")
            axis.plot(subset["configured_timescale_ratio"], subset[column], marker="o", label=f"{duration:g} h")
        axis.set_xscale("log")
        axis.set_xlabel("GP length scale / transit duration")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
    axes[0][0].legend(title="duration")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def run_experiment(args):
    rows = []
    ratio_rows = []
    last_regular = None
    for duration in args.injection_duration_hours_grid:
        duration_args = single_duration.default_settings()
        duration_args.output_dir = args.output_dir
        duration_args.optimized_reference = args.optimized_reference
        duration_args.length_scale_factors = args.length_scale_factors
        duration_args.gp_max_train_points = args.gp_max_train_points
        duration_args.gp_measurement_noise_fraction = args.gp_measurement_noise_fraction
        duration_args.injection_period_days = args.injection_period_days
        duration_args.injection_duration_hours = float(duration)
        duration_args.injection_depth = args.injection_depth
        duration_args.epoch_phase_fraction = args.epoch_phase_fraction
        duration_args.bls_n_periods = args.bls_n_periods
        duration_args.tcf_n_periods = args.tcf_n_periods
        regular, preservation_sorted, ratio_sorted, windows, metadata = single_duration.run_experiment(duration_args)
        last_regular = regular
        preservation_sorted = preservation_sorted.copy()
        ratio_sorted = ratio_sorted.copy()
        preservation_sorted.insert(0, "injected_duration_hours", float(duration))
        ratio_sorted.insert(0, "injected_duration_hours", float(duration))
        rows.append(preservation_sorted)
        ratio_rows.append(ratio_sorted)
    result = pd.concat(rows, ignore_index=True)
    result_by_duration_ratio = pd.concat(ratio_rows, ignore_index=True)
    result = result.sort_values(["snr_retention_fraction", "depth_retention_fraction"], ascending=[False, False]).reset_index(drop=True)
    result_by_duration_ratio = result_by_duration_ratio.sort_values(["injected_duration_hours", "configured_timescale_ratio", "optimize_kernel"]).reset_index(drop=True)
    by_ratio = grouped_by_ratio(result_by_duration_ratio)
    summary = summarize(result_by_duration_ratio, by_ratio, args)
    return last_regular, result, result_by_duration_ratio, by_ratio, summary

def main(args=None):
    args = args or default_settings()
    metrics_dir = Path(args.output_dir) / "metrics"
    processed_dir = Path(args.output_dir) / "processed"
    figures_dir = Path(args.output_dir) / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    regular, result, result_by_duration_ratio, by_ratio, summary = run_experiment(args)
    prefix = f"kic_{str(args.target_id).replace('KIC', '').strip()}_q{args.quarter}"
    result_path = metrics_dir / f"{prefix}_gp_multiduration_timescale_sensitivity.csv"
    duration_ratio_path = metrics_dir / f"{prefix}_gp_multiduration_timescale_by_duration_ratio.csv"
    ratio_path = metrics_dir / f"{prefix}_gp_multiduration_timescale_by_ratio.csv"
    summary_path = metrics_dir / f"{prefix}_gp_multiduration_timescale_summary.json"
    plot_path = figures_dir / f"{prefix}_gp_multiduration_timescale_ratio_collapse.png"
    result.to_csv(result_path, index=False)
    result_by_duration_ratio.to_csv(duration_ratio_path, index=False)
    by_ratio.to_csv(ratio_path, index=False)
    if regular is not None:
        regular.to_parquet(processed_dir / f"{prefix}_gp_multiduration_timescale_input.parquet", index=False)
    summary_path.write_text(json.dumps(single_duration.json_ready(summary), indent=2) + "\n")
    plot_ratio_collapse(result_by_duration_ratio, plot_path)
    display_columns = ["configured_timescale_ratio", "duration_count", "median_background_absorption_fraction", "median_depth_retention_fraction", "std_depth_retention_fraction", "median_snr_retention_fraction", "std_snr_retention_fraction", "median_max_abs_residual_acf_1_24", "bls_period_match_rate", "tcf_period_match_rate"]
    print(f"GP multi-duration time-scale rows: {result_path}")
    print(f"GP multi-duration by duration/ratio: {duration_ratio_path}")
    print(f"GP multi-duration by ratio: {ratio_path}")
    print(f"GP multi-duration summary: {summary_path}")
    print(f"GP ratio-collapse plot: {plot_path}")
    print(by_ratio[display_columns].to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
