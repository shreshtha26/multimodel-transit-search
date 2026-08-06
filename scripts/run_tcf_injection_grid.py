"""Run the periodic event-consistent ARIMA-TCF injection-recovery grid."""
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.detection.tcf import default_duration_grid, default_period_grid, fit_arima_innovations, harmonic_peak_rank, matching_peak_rank, period_match_fraction, run_tcf
from adaptive_transit.injections.synthetic import inject_periodic_box_transit
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_ID = "11904151"
QUARTER = 5
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/tcf_injection_grid"
FAP_THRESHOLD_PATH = PROJECT_ROOT / "outputs/experiments/tcf_null_calibration/metrics/kic_11904151_q5_tcf_fap_thresholds.csv"

def default_settings():
    return SimpleNamespace(target_id=TARGET_ID, quarter=QUARTER, output_dir=OUTPUT_DIR, fap_threshold_path=FAP_THRESHOLD_PATH, quality_policy="default", require_finite_flux_error=False, test_fraction=0.20, arima_order=(1, 1, 0), fit_maxiter=200, injection_period_grid=(2.0, 5.0, 10.0), injection_duration_hours_grid=(2.0, 4.0, 8.0), injection_depth_grid=(0.0002, 0.0005, 0.001), epoch_phase_fraction_grid=(0.15, 0.45, 0.75), min_period_days=1.0, max_period_days=15.0, n_periods=10000, min_duration_hours=1.5, max_duration_hours=10.0, n_durations=8, edge_width_cadences=0, min_edge_observations=4, min_transit_events=3, min_event_consistency_fraction=0.60, top_k=10, period_match_tolerance_fraction=0.02)

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

def load_fap_threshold(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TCF FAP threshold file does not exist: {path}")
    thresholds = pd.read_csv(path)
    required = {"fap_level", "score_threshold"}
    missing = required.difference(thresholds.columns)
    if missing:
        raise ValueError(f"TCF threshold file is missing columns: {sorted(missing)}")
    matches = thresholds[np.isclose(thresholds["fap_level"], 0.01)]
    if matches.empty:
        raise ValueError("TCF threshold file does not contain a 1% FAP threshold.")
    return thresholds, float(matches.iloc[0]["score_threshold"])

def run_one_injection(time, flux, period_grid, duration_grid, injected_period, injected_duration_hours, injected_depth, epoch_phase_fraction, threshold, args):
    finite = np.isfinite(time) & np.isfinite(flux)
    epoch = float(np.min(time[finite]) + float(epoch_phase_fraction) * float(injected_period))
    duration_days = float(injected_duration_hours) / 24.0
    injected_flux, template, in_transit = inject_periodic_box_transit(time, flux, injected_period, epoch, duration_days, injected_depth)
    arima_result = fit_arima_innovations(injected_flux, order=args.arima_order, maxiter=args.fit_maxiter)
    tcf_result = run_tcf(time, arima_result["innovations"], period_grid, duration_grid, edge_width_cadences=args.edge_width_cadences, min_edge_observations=args.min_edge_observations, min_transit_events=args.min_transit_events, min_event_consistency_fraction=args.min_event_consistency_fraction, top_k=args.top_k)
    best = tcf_result["summary"]
    top_peaks = tcf_result["top_peaks"]
    recovered_period = float(best["period"])
    recovered_score = float(best["score"])
    period_error = period_match_fraction(recovered_period, injected_period)
    exact_period_error = abs(recovered_period - float(injected_period)) / float(injected_period)
    period_matched = bool(period_error <= args.period_match_tolerance_fraction)
    exact_period_matched = bool(exact_period_error <= args.period_match_tolerance_fraction)
    passes_fap = bool(recovered_score >= threshold)
    exact_rank = matching_peak_rank(top_peaks, injected_period, tolerance_fraction=args.period_match_tolerance_fraction)
    half_rank = harmonic_peak_rank(top_peaks, injected_period, 0.5, tolerance_fraction=args.period_match_tolerance_fraction)
    double_rank = harmonic_peak_rank(top_peaks, injected_period, 2.0, tolerance_fraction=args.period_match_tolerance_fraction)
    triple_rank = harmonic_peak_rank(top_peaks, injected_period, 3.0, tolerance_fraction=args.period_match_tolerance_fraction)
    top_periods = [float(value) for value in top_peaks["period_days"].to_numpy(dtype=float)]
    top_scores = [float(value) for value in top_peaks["score"].to_numpy(dtype=float)]
    top_raw_pooled_scores = [float(value) for value in top_peaks["raw_pooled_score"].to_numpy(dtype=float)]
    top_valid_event_counts = [int(value) for value in top_peaks["n_valid_transit_events"].to_numpy(dtype=int)]
    top_positive_event_fractions = [float(value) for value in top_peaks["positive_event_fraction"].to_numpy(dtype=float)]
    return {"injected_period_days": float(injected_period), "injected_epoch_days": epoch, "epoch_phase_fraction": float(epoch_phase_fraction), "injected_duration_hours": float(injected_duration_hours), "injected_depth": float(injected_depth), "in_transit_observation_count": int(np.isfinite(flux[in_transit]).sum()), "arima_converged": bool(arima_result["summary"]["converged"]), "recovered_period_days": recovered_period, "recovered_epoch_days": float(best["epoch"]), "recovered_duration_hours": float(best["duration"] * 24.0), "recovered_score": recovered_score, "recovered_raw_pooled_score": float(best["raw_pooled_score"]), "recovered_valid_transit_events": int(best["n_valid_transit_events"]), "recovered_positive_transit_events": int(best["n_positive_transit_events"]), "recovered_positive_event_fraction": float(best["positive_event_fraction"]), "recovered_median_event_score": float(best["median_event_score"]), "period_error_fraction": float(period_error), "exact_period_error_fraction": float(exact_period_error), "period_matched": period_matched, "exact_period_matched": exact_period_matched, "fap_1_percent_threshold": float(threshold), "passes_fap_1_percent": passes_fap, "recovered_at_fap_1_percent": bool(period_matched and passes_fap), "exact_recovered_at_fap_1_percent": bool(exact_period_matched and passes_fap), "exact_period_rank_top10": exact_rank, "exact_period_present_top10": bool(exact_rank is not None), "half_period_rank_top10": half_rank, "double_period_rank_top10": double_rank, "triple_period_rank_top10": triple_rank, "top_periods_json": json.dumps(top_periods), "top_scores_json": json.dumps(top_scores), "top_raw_pooled_scores_json": json.dumps(top_raw_pooled_scores), "top_valid_event_counts_json": json.dumps(top_valid_event_counts), "top_positive_event_fractions_json": json.dumps(top_positive_event_fractions)}

def grouped_recovery(results, column):
    return results.groupby(column, as_index=False).agg(injection_count=("period_matched", "size"), arima_convergence_rate=("arima_converged", "mean"), period_match_rate=("period_matched", "mean"), exact_period_match_rate=("exact_period_matched", "mean"), exact_period_present_top10_rate=("exact_period_present_top10", "mean"), detection_rate_fap_1_percent=("passes_fap_1_percent", "mean"), recovery_rate_fap_1_percent=("recovered_at_fap_1_percent", "mean"), exact_recovery_rate_fap_1_percent=("exact_recovered_at_fap_1_percent", "mean"), median_period_error_fraction=("period_error_fraction", "median"), median_exact_period_error_fraction=("exact_period_error_fraction", "median"), median_score=("recovered_score", "median"), median_raw_pooled_score=("recovered_raw_pooled_score", "median"), median_valid_transit_events=("recovered_valid_transit_events", "median"), median_positive_event_fraction=("recovered_positive_event_fraction", "median"), median_event_score=("recovered_median_event_score", "median"))

def run_experiment(args):
    light_curve = load_kepler_pdcsap(args.target_id, args.quarter)
    regular, preprocessing = preprocess_pdcsap_light_curve(light_curve.to_dataframe(), quality_policy=args.quality_policy, require_finite_flux_error=args.require_finite_flux_error, normalization_fit_fraction=1.0 - args.test_fraction)
    time = regular["time"].to_numpy(dtype=float)
    flux = regular["normalized_flux"].to_numpy(dtype=float)
    period_grid = default_period_grid(time, min_period_days=args.min_period_days, max_period_days=args.max_period_days, n_periods=args.n_periods)
    duration_grid = default_duration_grid(args.min_duration_hours, args.max_duration_hours, args.n_durations)
    thresholds, threshold = load_fap_threshold(args.fap_threshold_path)
    combinations = list(product(args.injection_period_grid, args.injection_duration_hours_grid, args.injection_depth_grid, args.epoch_phase_fraction_grid))
    rows = []
    for injected_period, injected_duration_hours, injected_depth, epoch_phase_fraction in tqdm(combinations, desc="TCF injection grid"):
        rows.append(run_one_injection(time, flux, period_grid, duration_grid, injected_period, injected_duration_hours, injected_depth, epoch_phase_fraction, threshold, args))
    results = pd.DataFrame(rows)
    by_depth = grouped_recovery(results, "injected_depth")
    by_duration = grouped_recovery(results, "injected_duration_hours")
    by_period = grouped_recovery(results, "injected_period_days")
    exact_ranks = results["exact_period_rank_top10"].dropna()
    summary = {"target_id": str(args.target_id), "quarter": int(args.quarter), "quality_policy": args.quality_policy, "arima_order": tuple(args.arima_order), "min_transit_events": int(args.min_transit_events), "min_event_consistency_fraction": float(args.min_event_consistency_fraction), "injection_count": int(len(results)), "arima_convergence_rate": float(results["arima_converged"].mean()), "period_match_rate": float(results["period_matched"].mean()), "exact_period_match_rate": float(results["exact_period_matched"].mean()), "exact_period_present_top10_rate": float(results["exact_period_present_top10"].mean()), "median_exact_period_rank_top10": float(exact_ranks.median()) if not exact_ranks.empty else None, "detection_rate_fap_1_percent": float(results["passes_fap_1_percent"].mean()), "recovery_rate_fap_1_percent": float(results["recovered_at_fap_1_percent"].mean()), "exact_recovery_rate_fap_1_percent": float(results["exact_recovered_at_fap_1_percent"].mean()), "median_period_error_fraction": float(results["period_error_fraction"].median()), "median_exact_period_error_fraction": float(results["exact_period_error_fraction"].median()), "median_recovered_score": float(results["recovered_score"].median()), "median_raw_pooled_score": float(results["recovered_raw_pooled_score"].median()), "median_valid_transit_events": float(results["recovered_valid_transit_events"].median()), "median_positive_event_fraction": float(results["recovered_positive_event_fraction"].median()), "median_event_score": float(results["recovered_median_event_score"].median()), "fap_1_percent_threshold": float(threshold)}
    return regular, thresholds, results, by_depth, by_duration, by_period, summary

def main(args=None):
    args = args or default_settings()
    metrics_dir = Path(args.output_dir) / "metrics"
    processed_dir = Path(args.output_dir) / "processed"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    regular, thresholds, results, by_depth, by_duration, by_period, summary = run_experiment(args)
    prefix = f"kic_{str(args.target_id).replace('KIC', '').strip()}_q{args.quarter}"
    results_path = metrics_dir / f"{prefix}_tcf_injection_grid.csv"
    summary_path = metrics_dir / f"{prefix}_tcf_injection_grid_summary.json"
    results.to_csv(results_path, index=False)
    by_depth.to_csv(metrics_dir / f"{prefix}_tcf_recovery_by_depth.csv", index=False)
    by_duration.to_csv(metrics_dir / f"{prefix}_tcf_recovery_by_duration.csv", index=False)
    by_period.to_csv(metrics_dir / f"{prefix}_tcf_recovery_by_period.csv", index=False)
    thresholds.to_csv(metrics_dir / f"{prefix}_tcf_used_fap_thresholds.csv", index=False)
    regular.to_parquet(processed_dir / f"{prefix}_tcf_injection_grid_input.parquet", index=False)
    summary_path.write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    print(f"TCF injection results: {results_path}")
    print(f"TCF injection summary: {summary_path}")
    print(f"Event-consistent 1% FAP threshold: {summary['fap_1_percent_threshold']:.6f}")
    print(f"Harmonic-aware period match rate: {summary['period_match_rate']:.3f}")
    print(f"Exact rank-1 period match rate: {summary['exact_period_match_rate']:.3f}")
    print(f"Exact period present in top 10: {summary['exact_period_present_top10_rate']:.3f}")
    print(f"Detection rate at 1% FAP: {summary['detection_rate_fap_1_percent']:.3f}")
    print(f"Recovery rate at 1% FAP: {summary['recovery_rate_fap_1_percent']:.3f}")
    print(f"Exact recovery rate at 1% FAP: {summary['exact_recovery_rate_fap_1_percent']:.3f}")
    print(f"Median event-consistent score: {summary['median_recovered_score']:.6f}")
    print(f"Median raw pooled score: {summary['median_raw_pooled_score']:.6f}")
    print(f"Median valid transit events: {summary['median_valid_transit_events']:.1f}")
    print(f"Median positive-event fraction: {summary['median_positive_event_fraction']:.3f}")
    print(f"Median per-event score: {summary['median_event_score']:.6f}")
    print("\nExact-period rank when present:\n")
    exact_ranks = results["exact_period_rank_top10"].dropna().astype(int)
    if exact_ranks.empty:
        print("No exact injected periods appeared in the top 10.")
    else:
        print(exact_ranks.value_counts().sort_index().to_string())
    print("\nRank and event diagnostics:\n")
    display_columns = ["injected_period_days", "injected_duration_hours", "injected_depth", "epoch_phase_fraction", "recovered_period_days", "recovered_score", "recovered_raw_pooled_score", "recovered_valid_transit_events", "recovered_positive_event_fraction", "recovered_median_event_score", "exact_period_rank_top10", "half_period_rank_top10", "double_period_rank_top10", "triple_period_rank_top10"]
    print(results[display_columns].to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())