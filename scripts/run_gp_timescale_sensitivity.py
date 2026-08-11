"""Test whether GP transit preservation follows background-to-transit time-scale separation."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.detection.bls import default_duration_grid as bls_duration_grid
from adaptive_transit.detection.bls import default_period_grid as bls_period_grid
from adaptive_transit.detection.bls import period_match_fraction as bls_period_error
from adaptive_transit.detection.bls import run_bls
from adaptive_transit.detection.tcf import default_duration_grid as tcf_duration_grid
from adaptive_transit.detection.tcf import default_period_grid as tcf_period_grid
from adaptive_transit.detection.tcf import period_match_fraction as tcf_period_error
from adaptive_transit.detection.tcf import run_tcf
from adaptive_transit.injections.synthetic import inject_periodic_box_transit
from adaptive_transit.noise_models.diagnostics import residual_diagnostics
from adaptive_transit.noise_models.gp import fit_smooth_gp_background
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve
from adaptive_transit.transit_models.periodic import transit_center_times
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_ID = "11904151"
QUARTER = 5
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/gp_timescale_sensitivity"

def default_settings():
    return SimpleNamespace(target_id=TARGET_ID, quarter=QUARTER, output_dir=OUTPUT_DIR, quality_policy="default", require_finite_flux_error=False, test_fraction=0.20, gp_max_train_points=512, gp_measurement_noise_fraction=0.20, gp_n_restarts_optimizer=0, gp_random_seed=123, optimized_reference=True, optimized_reference_length_scale_days=3.0, optimized_reference_min_length_scale_days=1.0, optimized_reference_max_length_scale_days=30.0, length_scale_factors=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0), injection_period_days=5.0, injection_duration_hours=4.0, injection_depth=0.001, epoch_phase_fraction=0.20, min_period_days=1.0, max_period_days=15.0, bls_n_periods=1000, tcf_n_periods=10000, min_duration_hours=1.5, max_duration_hours=10.0, n_durations=8, bls_objective="snr", top_k=10, edge_width_cadences=0, min_edge_observations=4, min_transit_events=3, min_event_consistency_fraction=0.60, period_match_tolerance_fraction=0.02, plot_half_width_days=0.35, n_plot_transits=3)

def parse_float_tuple(value):
    return tuple(float(item.strip()) for item in str(value).split(",") if item.strip())

def parse_args():
    defaults = default_settings()
    parser = argparse.ArgumentParser(description="Run a GP length-scale/transit-duration sensitivity experiment.")
    parser.add_argument("--length-scale-factors", default=",".join(str(value) for value in defaults.length_scale_factors))
    parser.add_argument("--gp-max-train-points", type=int, default=defaults.gp_max_train_points)
    parser.add_argument("--gp-measurement-noise-fraction", type=float, default=defaults.gp_measurement_noise_fraction)
    parser.add_argument("--injection-period-days", type=float, default=defaults.injection_period_days)
    parser.add_argument("--injection-duration-hours", type=float, default=defaults.injection_duration_hours)
    parser.add_argument("--injection-depth", type=float, default=defaults.injection_depth)
    parser.add_argument("--epoch-phase-fraction", type=float, default=defaults.epoch_phase_fraction)
    parser.add_argument("--bls-n-periods", type=int, default=defaults.bls_n_periods)
    parser.add_argument("--tcf-n-periods", type=int, default=defaults.tcf_n_periods)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--no-optimized-reference", action="store_true")
    parser.add_argument("--n-plot-transits", type=int, default=defaults.n_plot_transits)
    parser.add_argument("--plot-half-width-days", type=float, default=defaults.plot_half_width_days)
    parsed = parser.parse_args()
    defaults.length_scale_factors = parse_float_tuple(parsed.length_scale_factors)
    defaults.gp_max_train_points = int(parsed.gp_max_train_points)
    defaults.gp_measurement_noise_fraction = float(parsed.gp_measurement_noise_fraction)
    defaults.injection_period_days = float(parsed.injection_period_days)
    defaults.injection_duration_hours = float(parsed.injection_duration_hours)
    defaults.injection_depth = float(parsed.injection_depth)
    defaults.epoch_phase_fraction = float(parsed.epoch_phase_fraction)
    defaults.bls_n_periods = int(parsed.bls_n_periods)
    defaults.tcf_n_periods = int(parsed.tcf_n_periods)
    defaults.output_dir = Path(parsed.output_dir)
    defaults.optimized_reference = not parsed.no_optimized_reference
    defaults.n_plot_transits = int(parsed.n_plot_transits)
    defaults.plot_half_width_days = float(parsed.plot_half_width_days)
    return defaults

def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def robust_scale(values):
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        return float("nan")
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    return float(1.4826 * mad if mad > 0 else np.std(clean, ddof=1))

def periodic_depth_and_snr(values, in_transit):
    series = np.asarray(values, dtype=float)
    mask = np.asarray(in_transit, dtype=bool)
    finite_in = mask & np.isfinite(series)
    finite_out = ~mask & np.isfinite(series)
    if finite_in.sum() == 0 or finite_out.sum() < 3:
        return {"depth": float("nan"), "snr": float("nan"), "in_transit_count": int(finite_in.sum())}
    depth = float(np.median(series[finite_out]) - np.median(series[finite_in]))
    noise = robust_scale(series[finite_out])
    snr = float(depth / noise * np.sqrt(finite_in.sum())) if np.isfinite(noise) and noise > 0 else float("nan")
    return {"depth": depth, "snr": snr, "in_transit_count": int(finite_in.sum())}

def build_configurations(args):
    duration_days = float(args.injection_duration_hours) / 24.0
    rows = []
    if args.optimized_reference:
        rows.append({"configuration": "optimized_reference", "optimize_kernel": True, "configured_length_scale_days": float(args.optimized_reference_length_scale_days), "min_length_scale_days": float(args.optimized_reference_min_length_scale_days), "max_length_scale_days": float(args.optimized_reference_max_length_scale_days), "configured_timescale_ratio": float(args.optimized_reference_length_scale_days) / duration_days})
    for factor in args.length_scale_factors:
        length_scale_days = duration_days * float(factor)
        minimum = max(length_scale_days / 100.0, 1.0e-4)
        maximum = max(length_scale_days * 100.0, minimum * 10.0)
        rows.append({"configuration": f"fixed_l_over_duration_{float(factor):.4g}", "optimize_kernel": False, "configured_length_scale_days": length_scale_days, "min_length_scale_days": minimum, "max_length_scale_days": maximum, "configured_timescale_ratio": float(factor)})
    return rows

def run_detectors(time, residuals, bls_periods, bls_durations, tcf_periods, tcf_durations, args):
    bls_result = run_bls(time, residuals, None, bls_periods, bls_durations, objective=args.bls_objective, top_k=args.top_k)
    tcf_result = run_tcf(time, residuals, tcf_periods, tcf_durations, edge_width_cadences=args.edge_width_cadences, min_edge_observations=args.min_edge_observations, min_transit_events=args.min_transit_events, min_event_consistency_fraction=args.min_event_consistency_fraction, top_k=args.top_k)
    return bls_result, tcf_result

def evaluate_configuration(config, time, injected_flux, in_transit, observed, bls_periods, bls_durations, tcf_periods, tcf_durations, args):
    model = fit_smooth_gp_background(time, injected_flux, max_train_points=args.gp_max_train_points, length_scale_days=config["configured_length_scale_days"], min_length_scale_days=config["min_length_scale_days"], max_length_scale_days=config["max_length_scale_days"], measurement_noise_fraction=args.gp_measurement_noise_fraction, n_restarts_optimizer=args.gp_n_restarts_optimizer if config["optimize_kernel"] else 0, random_seed=args.gp_random_seed, optimize_kernel=config["optimize_kernel"])
    diagnostics = residual_diagnostics(model.residuals[model.usable_mask])
    residual_retention = periodic_depth_and_snr(model.residuals, in_transit)
    background_retention = periodic_depth_and_snr(model.background_mean, in_transit)
    bls_result, tcf_result = run_detectors(time, model.residuals, bls_periods, bls_durations, tcf_periods, tcf_durations, args)
    bls_best = bls_result["summary"]
    tcf_best = tcf_result["summary"]
    bls_error = bls_period_error(bls_best["period"], args.injection_period_days)
    tcf_error = tcf_period_error(tcf_best["period"], args.injection_period_days)
    exact_bls_error = abs(float(bls_best["period"]) - float(args.injection_period_days)) / float(args.injection_period_days)
    exact_tcf_error = abs(float(tcf_best["period"]) - float(args.injection_period_days)) / float(args.injection_period_days)
    duration_days = float(args.injection_duration_hours) / 24.0
    row = {"configuration": config["configuration"], "optimize_kernel": bool(config["optimize_kernel"]), "configured_length_scale_days": float(config["configured_length_scale_days"]), "fitted_length_scale_days": float(model.parameters["length_scale_days"]), "transit_duration_days": duration_days, "configured_timescale_ratio": float(config["configured_timescale_ratio"]), "fitted_timescale_ratio": float(model.parameters["length_scale_days"]) / duration_days, "measurement_noise_fraction": float(args.gp_measurement_noise_fraction), "measurement_noise_variance": float(model.parameters["measurement_noise_variance"]), "signal_variance": float(model.parameters["signal_variance"]), "training_point_count": int(model.parameters["training_point_count"]), "converged": bool(model.converged), "log_marginal_likelihood": float(model.log_marginal_likelihood), "residual_std": float(diagnostics["residual_std"]), "residual_mean": float(diagnostics["residual_mean"]), "max_abs_residual_acf_1_24": float(diagnostics["max_abs_residual_acf_1_24"]), "mean_abs_residual_acf_1_24": float(diagnostics["mean_abs_residual_acf_1_24"]), "minimum_ljung_box_p": float(diagnostics["minimum_ljung_box_p"]), "rolling_var_max_to_median": float(diagnostics["rolling_var_max_to_median"]), "arch_pvalue": float(diagnostics["arch_pvalue"]), "observed_depth_before_gp": float(observed["depth"]), "gp_residual_depth": float(residual_retention["depth"]), "depth_retention_fraction": float(residual_retention["depth"] / observed["depth"]) if observed["depth"] != 0 else float("nan"), "gp_background_depth": float(background_retention["depth"]), "background_absorption_fraction": float(background_retention["depth"] / observed["depth"]) if observed["depth"] != 0 else float("nan"), "local_snr_before_gp": float(observed["snr"]), "local_snr_after_gp": float(residual_retention["snr"]), "snr_retention_fraction": float(residual_retention["snr"] / observed["snr"]) if observed["snr"] != 0 else float("nan"), "bls_recovered_period_days": float(bls_best["period"]), "bls_score": float(bls_best["power"]), "bls_period_error_fraction": float(bls_error), "bls_exact_period_error_fraction": float(exact_bls_error), "bls_period_matched": bool(np.isfinite(bls_error) and bls_error <= args.period_match_tolerance_fraction), "bls_exact_period_matched": bool(np.isfinite(exact_bls_error) and exact_bls_error <= args.period_match_tolerance_fraction), "tcf_recovered_period_days": float(tcf_best["period"]), "tcf_score": float(tcf_best["score"]), "tcf_raw_pooled_score": float(tcf_best["raw_pooled_score"]), "tcf_period_error_fraction": float(tcf_error), "tcf_exact_period_error_fraction": float(exact_tcf_error), "tcf_period_matched": bool(np.isfinite(tcf_error) and tcf_error <= args.period_match_tolerance_fraction), "tcf_exact_period_matched": bool(np.isfinite(exact_tcf_error) and exact_tcf_error <= args.period_match_tolerance_fraction)}
    return model, row

def selected_transit_centers(time, period, epoch, duration, args):
    centers = transit_center_times(time, period, epoch, duration)
    if len(centers) <= int(args.n_plot_transits):
        return centers
    positions = np.linspace(0, len(centers) - 1, int(args.n_plot_transits), dtype=int)
    return [float(centers[int(position)]) for position in positions]

def window_rows(time, flux, injected_flux, in_transit, models, centers, args):
    rows = []
    for center_index, center in enumerate(centers):
        local = np.abs(time - float(center)) <= float(args.plot_half_width_days)
        for configuration, model in models.items():
            for index in np.flatnonzero(local):
                rows.append({"configuration": configuration, "transit_window": int(center_index), "center_time": float(center), "time": float(time[index]), "phase_days": float(time[index] - center), "normalized_flux": float(flux[index]) if np.isfinite(flux[index]) else np.nan, "injected_flux": float(injected_flux[index]) if np.isfinite(injected_flux[index]) else np.nan, "in_transit": bool(in_transit[index]), "gp_background": float(model.background_mean[index]) if np.isfinite(model.background_mean[index]) else np.nan, "gp_background_std": float(model.background_std[index]) if np.isfinite(model.background_std[index]) else np.nan, "gp_residual": float(model.residuals[index]) if np.isfinite(model.residuals[index]) else np.nan})
    return pd.DataFrame(rows)

def plot_windows(window_table, configurations, path):
    plotted_configs = [config for config in configurations if config in set(window_table["configuration"])]
    window_ids = sorted(window_table["transit_window"].unique())
    fig, axes = plt.subplots(len(window_ids), len(plotted_configs), figsize=(5 * len(plotted_configs), 3.5 * len(window_ids)), squeeze=False)
    for row_index, window_id in enumerate(window_ids):
        for column_index, configuration in enumerate(plotted_configs):
            axis = axes[row_index][column_index]
            subset = window_table[(window_table["transit_window"] == window_id) & (window_table["configuration"] == configuration)].sort_values("phase_days")
            axis.plot(subset["phase_days"], subset["injected_flux"], ".", ms=3, label="injected flux")
            axis.plot(subset["phase_days"], subset["gp_background"], "-", lw=1.2, label="GP background")
            axis.plot(subset["phase_days"], subset["gp_residual"], "-", lw=1.0, label="GP residual")
            axis.axvline(0.0, color="black", lw=0.8, alpha=0.5)
            axis.set_title(f"{configuration} window {window_id}")
            axis.set_xlabel("days from transit center")
            axis.set_ylabel("normalized flux")
            if row_index == 0 and column_index == 0:
                axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def run_experiment(args):
    light_curve = load_kepler_pdcsap(args.target_id, args.quarter)
    regular, preprocessing = preprocess_pdcsap_light_curve(light_curve.to_dataframe(), quality_policy=args.quality_policy, require_finite_flux_error=args.require_finite_flux_error, normalization_fit_fraction=1.0 - args.test_fraction)
    time = regular["time"].to_numpy(dtype=float)
    flux = regular["normalized_flux"].to_numpy(dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    if finite.sum() < 24:
        raise ValueError("Insufficient finite light-curve observations.")
    epoch = float(np.min(time[finite]) + float(args.epoch_phase_fraction) * float(args.injection_period_days))
    duration_days = float(args.injection_duration_hours) / 24.0
    injected_flux, template, in_transit = inject_periodic_box_transit(time, flux, args.injection_period_days, epoch, duration_days, args.injection_depth)
    observed = periodic_depth_and_snr(injected_flux, in_transit)
    bls_periods = bls_period_grid(time, min_period_days=args.min_period_days, max_period_days=args.max_period_days, n_periods=args.bls_n_periods)
    bls_durations = bls_duration_grid(args.min_duration_hours, args.max_duration_hours, args.n_durations)
    tcf_periods = tcf_period_grid(time, min_period_days=args.min_period_days, max_period_days=args.max_period_days, n_periods=args.tcf_n_periods)
    tcf_durations = tcf_duration_grid(args.min_duration_hours, args.max_duration_hours, args.n_durations)
    rows = []
    models = {}
    for config in build_configurations(args):
        model, row = evaluate_configuration(config, time, injected_flux, in_transit, observed, bls_periods, bls_durations, tcf_periods, tcf_durations, args)
        rows.append(row)
        models[row["configuration"]] = model
    preservation_sorted = pd.DataFrame(rows).sort_values(["snr_retention_fraction", "depth_retention_fraction"], ascending=[False, False]).reset_index(drop=True)
    ratio_sorted = pd.DataFrame(rows).sort_values(["configured_timescale_ratio", "optimize_kernel"], ascending=[True, True]).reset_index(drop=True)
    centers = selected_transit_centers(time[finite], args.injection_period_days, epoch, duration_days, args)
    plot_configs = []
    if "optimized_reference" in models:
        plot_configs.append("optimized_reference")
    for target in (min(args.length_scale_factors), 1.0, 5.0, 20.0):
        matches = [row["configuration"] for row in rows if not row["optimize_kernel"] and np.isclose(row["configured_timescale_ratio"], target)]
        plot_configs.extend(matches[:1])
    plot_configs.append(str(preservation_sorted.iloc[0]["configuration"]))
    plot_configs = list(dict.fromkeys(plot_configs))
    plot_models = {configuration: models[configuration] for configuration in plot_configs if configuration in models}
    windows = window_rows(time, flux, injected_flux, in_transit, plot_models, centers, args)
    metadata = {"target_id": str(args.target_id), "quarter": int(args.quarter), "quality_policy": args.quality_policy, "model": "smooth_anchor_gp", "hypothesis": "GP advantage should increase when the GP background timescale is well separated from the transit duration, until the GP becomes too rigid to remove relevant background variability.", "causal_test": "same light curve and same injection, fixed GP length scales varied relative to transit duration", "not_fap_calibrated": True, "fap_note": "Detector scores are reported without applying configuration-specific FAP thresholds; each GP length-scale setting would require its own null calibration for formal recovery at FAP.", "injected_period_days": float(args.injection_period_days), "injected_epoch_days": epoch, "epoch_phase_fraction": float(args.epoch_phase_fraction), "injected_duration_hours": float(args.injection_duration_hours), "injected_depth": float(args.injection_depth), "transit_duration_days": duration_days, "observed_depth_before_gp": float(observed["depth"]), "local_snr_before_gp": float(observed["snr"]), "configuration_count": int(len(preservation_sorted)), "fixed_length_scale_factors": list(args.length_scale_factors), "diagnostic_plot_configurations": plot_configs, "sorted_by": ["snr_retention_fraction", "depth_retention_fraction"]}
    return regular, preservation_sorted, ratio_sorted, windows, metadata

def main(args=None):
    args = args or default_settings()
    metrics_dir = Path(args.output_dir) / "metrics"
    processed_dir = Path(args.output_dir) / "processed"
    figures_dir = Path(args.output_dir) / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    regular, preservation_sorted, ratio_sorted, windows, metadata = run_experiment(args)
    prefix = f"kic_{str(args.target_id).replace('KIC', '').strip()}_q{args.quarter}"
    summary_path = metrics_dir / f"{prefix}_gp_timescale_sensitivity_summary.csv"
    ratio_path = metrics_dir / f"{prefix}_gp_timescale_sensitivity_by_ratio.csv"
    metadata_path = metrics_dir / f"{prefix}_gp_timescale_sensitivity_metadata.json"
    windows_path = processed_dir / f"{prefix}_gp_timescale_transit_window_samples.csv"
    plot_path = figures_dir / f"{prefix}_gp_timescale_transit_window_diagnostics.png"
    preservation_sorted.to_csv(summary_path, index=False)
    ratio_sorted.to_csv(ratio_path, index=False)
    windows.to_csv(windows_path, index=False)
    regular.to_parquet(processed_dir / f"{prefix}_gp_timescale_sensitivity_input.parquet", index=False)
    metadata_path.write_text(json.dumps(json_ready(metadata), indent=2) + "\n")
    plot_windows(windows, metadata["diagnostic_plot_configurations"], plot_path)
    display_columns = ["configuration", "configured_timescale_ratio", "fitted_timescale_ratio", "background_absorption_fraction", "depth_retention_fraction", "snr_retention_fraction", "residual_std", "max_abs_residual_acf_1_24", "minimum_ljung_box_p", "bls_recovered_period_days", "bls_score", "tcf_recovered_period_days", "tcf_score"]
    print(f"GP time-scale sensitivity summary: {summary_path}")
    print(f"GP time-scale sensitivity by ratio: {ratio_path}")
    print(f"Transit window samples: {windows_path}")
    print(f"Transit window diagnostic plot: {plot_path}")
    print(preservation_sorted[display_columns].to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
