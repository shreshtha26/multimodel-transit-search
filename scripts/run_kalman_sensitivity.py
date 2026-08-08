"""Evaluate Kalman process-noise sensitivity for transit preservation versus whitening."""
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
from adaptive_transit.noise_models.kalman import fit_kalman_local_level
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve
from adaptive_transit.transit_models.periodic import transit_center_times
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_ID = "11904151"
QUARTER = 5
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/kalman_sensitivity"

def default_settings():
    return SimpleNamespace(target_id=TARGET_ID, quarter=QUARTER, output_dir=OUTPUT_DIR, quality_policy="default", require_finite_flux_error=False, test_fraction=0.20, kalman_maxiter=100, kalman_burn_in=1, injection_period_days=5.0, injection_epoch_offset_days=1.0, injection_duration_hours=4.0, injection_depth=0.001, process_to_measurement_ratios=(0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.686306450454942), min_period_days=1.0, max_period_days=15.0, bls_n_periods=1000, tcf_n_periods=10000, min_duration_hours=1.5, max_duration_hours=10.0, n_durations=8, bls_objective="snr", top_k=10, edge_width_cadences=0, min_edge_observations=4, min_transit_events=3, min_event_consistency_fraction=0.60, period_match_tolerance_fraction=0.02, plot_half_width_days=0.35, n_plot_transits=3)

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

def kalman_gain_series(model):
    measurement_variance = float(model.parameters["measurement_variance"])
    innovation_variance = np.asarray(model.residual_variance, dtype=float)
    gain = np.full(innovation_variance.shape, np.nan, dtype=float)
    finite = np.isfinite(innovation_variance) & (innovation_variance > 0)
    gain[finite] = (innovation_variance[finite] - measurement_variance) / innovation_variance[finite]
    return gain

def build_configurations(injected_reference, original_reference, args):
    measurement_variance = float(injected_reference.parameters["measurement_variance"])
    rows = [{"configuration": "mle_injected_reference", "parameter_mode": "estimated_on_injected_flux", "process_variance": float(injected_reference.parameters["process_variance"]), "measurement_variance": measurement_variance}, {"configuration": "mle_original_applied_to_injected", "parameter_mode": "estimated_on_original_flux", "process_variance": float(original_reference.parameters["process_variance"]), "measurement_variance": float(original_reference.parameters["measurement_variance"])}]
    for ratio in args.process_to_measurement_ratios:
        rows.append({"configuration": f"fixed_q_over_r_{float(ratio):.4g}", "parameter_mode": "fixed_ratio_reference_measurement_noise", "process_variance": measurement_variance * float(ratio), "measurement_variance": measurement_variance})
    unique = []
    seen = set()
    for row in rows:
        key = row["configuration"]
        if key not in seen:
            unique.append(row)
            seen.add(key)
    return unique

def run_detectors(time, residuals, bls_periods, bls_durations, tcf_periods, tcf_durations, args):
    bls_result = run_bls(time, residuals, None, bls_periods, bls_durations, objective=args.bls_objective, top_k=args.top_k)
    tcf_result = run_tcf(time, residuals, tcf_periods, tcf_durations, edge_width_cadences=args.edge_width_cadences, min_edge_observations=args.min_edge_observations, min_transit_events=args.min_transit_events, min_event_consistency_fraction=args.min_event_consistency_fraction, top_k=args.top_k)
    return bls_result, tcf_result

def evaluate_configuration(config, time, injected_flux, in_transit, observed_depth, observed_snr, bls_periods, bls_durations, tcf_periods, tcf_durations, args):
    estimate = config["parameter_mode"] == "estimated_on_injected_flux"
    model = fit_kalman_local_level(injected_flux, process_variance=config["process_variance"], measurement_variance=config["measurement_variance"], estimate_parameters=estimate, maxiter=args.kalman_maxiter, burn_in=args.kalman_burn_in)
    diagnostics = residual_diagnostics(model.residuals[model.usable_mask])
    residual_retention = periodic_depth_and_snr(model.residuals, in_transit)
    predicted_background_retention = periodic_depth_and_snr(model.predicted_background, in_transit)
    filtered_background_retention = periodic_depth_and_snr(model.filtered_background, in_transit)
    gain = kalman_gain_series(model)
    bls_result, tcf_result = run_detectors(time, model.residuals, bls_periods, bls_durations, tcf_periods, tcf_durations, args)
    bls_best = bls_result["summary"]
    tcf_best = tcf_result["summary"]
    bls_error = bls_period_error(bls_best["period"], args.injection_period_days)
    tcf_error = tcf_period_error(tcf_best["period"], args.injection_period_days)
    row = {"configuration": config["configuration"], "parameter_mode": config["parameter_mode"], "process_variance": float(model.parameters["process_variance"]), "measurement_variance": float(model.parameters["measurement_variance"]), "process_to_measurement_ratio": float(model.parameters["process_variance"] / model.parameters["measurement_variance"]), "converged": bool(model.converged), "log_likelihood": float(model.log_likelihood), "residual_std": float(diagnostics["residual_std"]), "residual_mean": float(diagnostics["residual_mean"]), "max_abs_residual_acf_1_24": float(diagnostics["max_abs_residual_acf_1_24"]), "mean_abs_residual_acf_1_24": float(diagnostics["mean_abs_residual_acf_1_24"]), "minimum_ljung_box_p": float(diagnostics["minimum_ljung_box_p"]), "rolling_var_max_to_median": float(diagnostics["rolling_var_max_to_median"]), "arch_pvalue": float(diagnostics["arch_pvalue"]), "kalman_gain_median": float(np.nanmedian(gain)), "kalman_gain_in_transit_median": float(np.nanmedian(gain[in_transit])), "observed_depth_before_kalman": float(observed_depth), "kalman_residual_depth": float(residual_retention["depth"]), "depth_retention_fraction": float(residual_retention["depth"] / observed_depth) if observed_depth != 0 else float("nan"), "predicted_background_depth_fraction": float(predicted_background_retention["depth"] / observed_depth) if observed_depth != 0 else float("nan"), "filtered_background_depth_fraction": float(filtered_background_retention["depth"] / observed_depth) if observed_depth != 0 else float("nan"), "local_snr_before_kalman": float(observed_snr), "local_snr_after_kalman": float(residual_retention["snr"]), "snr_retention_fraction": float(residual_retention["snr"] / observed_snr) if observed_snr != 0 else float("nan"), "bls_recovered_period_days": float(bls_best["period"]), "bls_score": float(bls_best["power"]), "bls_period_error_fraction": float(bls_error), "bls_period_matched": bool(np.isfinite(bls_error) and bls_error <= args.period_match_tolerance_fraction), "tcf_recovered_period_days": float(tcf_best["period"]), "tcf_score": float(tcf_best["score"]), "tcf_raw_pooled_score": float(tcf_best["raw_pooled_score"]), "tcf_period_error_fraction": float(tcf_error), "tcf_period_matched": bool(np.isfinite(tcf_error) and tcf_error <= args.period_match_tolerance_fraction)}
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
                rows.append({"configuration": configuration, "transit_window": int(center_index), "center_time": float(center), "time": float(time[index]), "phase_days": float(time[index] - center), "normalized_flux": float(flux[index]) if np.isfinite(flux[index]) else np.nan, "injected_flux": float(injected_flux[index]) if np.isfinite(injected_flux[index]) else np.nan, "in_transit": bool(in_transit[index]), "predicted_background": float(model.predicted_background[index]) if np.isfinite(model.predicted_background[index]) else np.nan, "filtered_background": float(model.filtered_background[index]) if np.isfinite(model.filtered_background[index]) else np.nan, "kalman_residual": float(model.residuals[index]) if np.isfinite(model.residuals[index]) else np.nan, "kalman_gain": float(kalman_gain_series(model)[index]) if np.isfinite(kalman_gain_series(model)[index]) else np.nan})
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
            axis.plot(subset["phase_days"], subset["predicted_background"], "-", lw=1.2, label="predicted background")
            axis.plot(subset["phase_days"], subset["filtered_background"], "--", lw=1.0, label="filtered background")
            axis.plot(subset["phase_days"], subset["kalman_residual"], "-", lw=1.0, label="residual")
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
    epoch = float(np.min(time[finite]) + args.injection_epoch_offset_days)
    duration_days = float(args.injection_duration_hours) / 24.0
    injected_flux, template, in_transit = inject_periodic_box_transit(time, flux, args.injection_period_days, epoch, duration_days, args.injection_depth)
    observed = periodic_depth_and_snr(injected_flux, in_transit)
    original_reference = fit_kalman_local_level(flux, maxiter=args.kalman_maxiter, burn_in=args.kalman_burn_in)
    injected_reference = fit_kalman_local_level(injected_flux, maxiter=args.kalman_maxiter, burn_in=args.kalman_burn_in)
    bls_periods = bls_period_grid(time, min_period_days=args.min_period_days, max_period_days=args.max_period_days, n_periods=args.bls_n_periods)
    bls_durations = bls_duration_grid(args.min_duration_hours, args.max_duration_hours, args.n_durations)
    tcf_periods = tcf_period_grid(time, min_period_days=args.min_period_days, max_period_days=args.max_period_days, n_periods=args.tcf_n_periods)
    tcf_durations = tcf_duration_grid(args.min_duration_hours, args.max_duration_hours, args.n_durations)
    configurations = build_configurations(injected_reference, original_reference, args)
    rows = []
    models = {}
    for config in configurations:
        model, row = evaluate_configuration(config, time, injected_flux, in_transit, float(observed["depth"]), float(observed["snr"]), bls_periods, bls_durations, tcf_periods, tcf_durations, args)
        rows.append(row)
        models[row["configuration"]] = model
    summary = pd.DataFrame(rows).sort_values(["snr_retention_fraction", "depth_retention_fraction"], ascending=[False, False]).reset_index(drop=True)
    centers = selected_transit_centers(time[finite], args.injection_period_days, epoch, duration_days, args)
    plot_configs = list(dict.fromkeys(["mle_injected_reference", summary.iloc[0]["configuration"], "fixed_q_over_r_0.01"]))
    plot_models = {configuration: models[configuration] for configuration in plot_configs if configuration in models}
    windows = window_rows(time, flux, injected_flux, in_transit, plot_models, centers, args)
    metadata = {"target_id": str(args.target_id), "quarter": int(args.quarter), "quality_policy": args.quality_policy, "model": "local_level_kalman", "state_vector": ["background"], "state_transition": [[1.0]], "observation_matrix": [[1.0]], "current_baseline_uses_smoothing": False, "current_residual_definition": "normalized_flux_t - one_step_predicted_background_t", "injected_period_days": float(args.injection_period_days), "injected_epoch_days": epoch, "injected_duration_hours": float(args.injection_duration_hours), "injected_depth": float(args.injection_depth), "observed_depth_before_kalman": float(observed["depth"]), "local_snr_before_kalman": float(observed["snr"]), "configuration_count": int(len(summary)), "sorted_by": ["snr_retention_fraction", "depth_retention_fraction"], "reference_injected_parameters": injected_reference.summary(), "reference_original_parameters": original_reference.summary()}
    return regular, summary, windows, metadata, plot_configs

def main(args=None):
    args = args or default_settings()
    metrics_dir = Path(args.output_dir) / "metrics"
    processed_dir = Path(args.output_dir) / "processed"
    figures_dir = Path(args.output_dir) / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    regular, summary, windows, metadata, plot_configs = run_experiment(args)
    prefix = f"kic_{str(args.target_id).replace('KIC', '').strip()}_q{args.quarter}"
    summary_path = metrics_dir / f"{prefix}_kalman_sensitivity_summary.csv"
    metadata_path = metrics_dir / f"{prefix}_kalman_sensitivity_metadata.json"
    windows_path = processed_dir / f"{prefix}_kalman_transit_window_samples.csv"
    plot_path = figures_dir / f"{prefix}_kalman_transit_window_diagnostics.png"
    summary.to_csv(summary_path, index=False)
    windows.to_csv(windows_path, index=False)
    regular.to_parquet(processed_dir / f"{prefix}_kalman_sensitivity_input.parquet", index=False)
    metadata["diagnostic_plot_configurations"] = plot_configs
    metadata_path.write_text(json.dumps(json_ready(metadata), indent=2) + "\n")
    plot_windows(windows, plot_configs, plot_path)
    display_columns = ["configuration", "process_to_measurement_ratio", "kalman_gain_median", "depth_retention_fraction", "snr_retention_fraction", "residual_std", "max_abs_residual_acf_1_24", "minimum_ljung_box_p", "bls_recovered_period_days", "bls_score", "tcf_recovered_period_days", "tcf_score"]
    print(f"Kalman sensitivity summary: {summary_path}")
    print(f"Transit window samples: {windows_path}")
    print(f"Transit window diagnostic plot: {plot_path}")
    print(summary[display_columns].to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
