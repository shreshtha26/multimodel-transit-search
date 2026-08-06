# Input: accept time + ARIMA innovations
# create trial period, epoch and duration combinations
# place negative spikes at ingress
# place positive spikes at egress
# calculate event-consistent matching scores
# return the best period, epoch, duration and score
# Output: best_period
# best_epoch
# best_duration
# best_score
# periodogram
# top_peaks
"""Run one engineering TCF baseline with and without a periodic injection."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.detection.tcf import default_duration_grid, default_period_grid, fit_arima_innovations, period_match_fraction, run_tcf
from adaptive_transit.injections.synthetic import inject_periodic_box_transit
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_ID = "11904151"
QUARTER = 5
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/tcf_baseline"

def default_settings():
    return SimpleNamespace(target_id=TARGET_ID, quarter=QUARTER, output_dir=OUTPUT_DIR, quality_policy="default", require_finite_flux_error=False, test_fraction=0.20, arima_order=(1, 1, 0), fit_maxiter=200, injection_period_days=5.0, injection_epoch_offset_days=1.0, injection_duration_hours=4.0, injection_depth=0.001, min_period_days=0.5, max_period_days=15.0, n_periods=10000, min_duration_hours=1.5, max_duration_hours=10.0, n_durations=8, edge_width_cadences=0, min_edge_observations=4, min_transit_events=3, min_event_consistency_fraction=0.60, top_k=10, period_match_tolerance_fraction=0.02)

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

def observable_transit_count(time, period, epoch):
    finite_time = np.asarray(time, dtype=float)
    finite_time = finite_time[np.isfinite(finite_time)]
    first_index = int(np.ceil((np.min(finite_time) - float(epoch)) / float(period)))
    last_index = int(np.floor((np.max(finite_time) - float(epoch)) / float(period)))
    return max(last_index - first_index + 1, 0)

def run_experiment(args):
    light_curve = load_kepler_pdcsap(args.target_id, args.quarter)
    regular, preprocessing = preprocess_pdcsap_light_curve(light_curve.to_dataframe(), quality_policy=args.quality_policy, require_finite_flux_error=args.require_finite_flux_error, normalization_fit_fraction=1.0 - args.test_fraction)
    time = regular["time"].to_numpy(dtype=float)
    flux = regular["normalized_flux"].to_numpy(dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    if finite.sum() < 24:
        raise ValueError("Insufficient finite light-curve observations.")
    period_grid = default_period_grid(time, min_period_days=args.min_period_days, max_period_days=args.max_period_days, n_periods=args.n_periods)
    duration_grid = default_duration_grid(args.min_duration_hours, args.max_duration_hours, args.n_durations)
    epoch = float(np.min(time[finite]) + args.injection_epoch_offset_days)
    duration_days = float(args.injection_duration_hours) / 24.0
    injected_flux, template, in_transit = inject_periodic_box_transit(time, flux, args.injection_period_days, epoch, duration_days, args.injection_depth)
    injected_arima = fit_arima_innovations(injected_flux, order=args.arima_order, maxiter=args.fit_maxiter)
    null_arima = fit_arima_innovations(flux, order=args.arima_order, maxiter=args.fit_maxiter)
    injected_result = run_tcf(time, injected_arima["innovations"], period_grid, duration_grid, edge_width_cadences=args.edge_width_cadences, min_edge_observations=args.min_edge_observations, min_transit_events=args.min_transit_events, min_event_consistency_fraction=args.min_event_consistency_fraction, top_k=args.top_k)
    null_result = run_tcf(time, null_arima["innovations"], period_grid, duration_grid, edge_width_cadences=args.edge_width_cadences, min_edge_observations=args.min_edge_observations, min_transit_events=args.min_transit_events, min_event_consistency_fraction=args.min_event_consistency_fraction, top_k=args.top_k)
    injected_summary = injected_result["summary"]
    null_summary = null_result["summary"]
    period_error = period_match_fraction(injected_summary["period"], args.injection_period_days)
    summary = {"target_id": str(args.target_id), "quarter": int(args.quarter), "quality_policy": args.quality_policy, "arima_order": tuple(args.arima_order), "min_transit_events": int(args.min_transit_events), "min_event_consistency_fraction": float(args.min_event_consistency_fraction), "arima_converged_injected": bool(injected_arima["summary"]["converged"]), "arima_converged_null": bool(null_arima["summary"]["converged"]), "injected_period_days": float(args.injection_period_days), "injected_epoch_days": epoch, "injected_duration_hours": float(args.injection_duration_hours), "injected_depth": float(args.injection_depth), "observable_transit_count": observable_transit_count(time, args.injection_period_days, epoch), "in_transit_observation_count": int(np.isfinite(flux[in_transit]).sum()), "recovered_period_days": float(injected_summary["period"]), "recovered_epoch_days": float(injected_summary["epoch"]), "recovered_duration_hours": float(injected_summary["duration"] * 24.0), "recovered_score": float(injected_summary["score"]), "recovered_raw_pooled_score": float(injected_summary["raw_pooled_score"]), "recovered_valid_transit_events": int(injected_summary["n_valid_transit_events"]), "recovered_positive_transit_events": int(injected_summary["n_positive_transit_events"]), "recovered_positive_event_fraction": float(injected_summary["positive_event_fraction"]), "recovered_median_event_score": float(injected_summary["median_event_score"]), "period_error_fraction": float(period_error), "period_matched": bool(period_error <= args.period_match_tolerance_fraction), "null_best_period_days": float(null_summary["period"]), "null_best_epoch_days": float(null_summary["epoch"]), "null_best_duration_hours": float(null_summary["duration"] * 24.0), "null_max_score": float(null_summary["score"]), "null_raw_pooled_score": float(null_summary["raw_pooled_score"]), "null_valid_transit_events": int(null_summary["n_valid_transit_events"]), "null_positive_transit_events": int(null_summary["n_positive_transit_events"]), "null_positive_event_fraction": float(null_summary["positive_event_fraction"]), "null_median_event_score": float(null_summary["median_event_score"]), "injected_beats_null": bool(injected_summary["score"] > null_summary["score"]), "injected_score_increase_over_null": float(injected_summary["score"] - null_summary["score"]), "injected_score_ratio_to_null": float(injected_summary["score"] / null_summary["score"]) if null_summary["score"] != 0 else None, "n_observations": int(finite.sum()), "n_periods": int(len(period_grid)), "n_durations": int(len(duration_grid))}
    innovation_table = pd.DataFrame({"time": time, "null_innovations": null_arima["innovations"], "injected_innovations": injected_arima["innovations"]})
    return regular, innovation_table, injected_result, null_result, summary

def main(args=None):
    args = args or default_settings()
    metrics_dir = Path(args.output_dir) / "metrics"
    processed_dir = Path(args.output_dir) / "processed"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    regular, innovation_table, injected_result, null_result, summary = run_experiment(args)
    prefix = f"kic_{str(args.target_id).replace('KIC', '').strip()}_q{args.quarter}"
    injected_result["periodogram"].to_csv(metrics_dir / f"{prefix}_tcf_injected_periodogram.csv", index=False)
    injected_result["top_peaks"].to_csv(metrics_dir / f"{prefix}_tcf_injected_top_peaks.csv", index=False)
    null_result["periodogram"].to_csv(metrics_dir / f"{prefix}_tcf_null_periodogram.csv", index=False)
    null_result["top_peaks"].to_csv(metrics_dir / f"{prefix}_tcf_null_top_peaks.csv", index=False)
    innovation_table.to_parquet(processed_dir / f"{prefix}_tcf_baseline_innovations.parquet", index=False)
    regular.to_parquet(processed_dir / f"{prefix}_tcf_baseline_input.parquet", index=False)
    summary_path = metrics_dir / f"{prefix}_tcf_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    print(f"TCF summary: {summary_path}")
    print("\nInjected result:\n")
    print(f"Recovered period: {summary['recovered_period_days']:.6f}")
    print(f"Period matched: {summary['period_matched']}")
    print(f"Event-consistent score: {summary['recovered_score']:.6f}")
    print(f"Raw pooled score: {summary['recovered_raw_pooled_score']:.6f}")
    print(f"Valid transit events: {summary['recovered_valid_transit_events']}")
    print(f"Positive transit events: {summary['recovered_positive_transit_events']}")
    print(f"Positive-event fraction: {summary['recovered_positive_event_fraction']:.3f}")
    print(f"Median event score: {summary['recovered_median_event_score']:.6f}")
    print("\nOriginal no-injection result:\n")
    print(f"Best period: {summary['null_best_period_days']:.6f}")
    print(f"Event-consistent score: {summary['null_max_score']:.6f}")
    print(f"Raw pooled score: {summary['null_raw_pooled_score']:.6f}")
    print(f"Valid transit events: {summary['null_valid_transit_events']}")
    print(f"Positive transit events: {summary['null_positive_transit_events']}")
    print(f"Positive-event fraction: {summary['null_positive_event_fraction']:.3f}")
    print(f"Median event score: {summary['null_median_event_score']:.6f}")
    print(f"Injected score increase over original: {summary['injected_score_increase_over_null']:.6f}")
    top_peak_columns = ["rank", "period_days", "score", "raw_pooled_score", "duration", "epoch", "n_valid_transit_events", "n_positive_transit_events", "positive_event_fraction", "median_event_score"]
    print("\nOriginal no-injection top 10 peaks:\n")
    print(null_result["top_peaks"][top_peak_columns].to_string(index=False))
    print("\nInjected top 10 peaks:\n")
    print(injected_result["top_peaks"][top_peak_columns].to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())