"""Validate coarse-to-fine TCF against exhaustive TCF on representative cases."""
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.detection.tcf import default_duration_grid, default_period_grid, fit_arima_innovations, matching_peak_rank, period_match_fraction, run_tcf
from adaptive_transit.injections.synthetic import inject_periodic_box_transit
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_ID = "11904151"
QUARTER = 5
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/tcf_search_mode_validation"
FAP_THRESHOLD_PATH = PROJECT_ROOT / "outputs/experiments/tcf_null_calibration/metrics/kic_11904151_q5_tcf_fap_thresholds.csv"

def default_settings():
    validation_cases = ({"case_name": "original_no_injection", "injected": False}, {"case_name": "weak_10d_short", "injected": True, "period_days": 10.0, "duration_hours": 2.0, "depth": 0.0002, "epoch_phase_fraction": 0.45}, {"case_name": "medium_5d", "injected": True, "period_days": 5.0, "duration_hours": 4.0, "depth": 0.0005, "epoch_phase_fraction": 0.45}, {"case_name": "strong_2d_harmonic_case", "injected": True, "period_days": 2.0, "duration_hours": 4.0, "depth": 0.001, "epoch_phase_fraction": 0.45}, {"case_name": "strong_5d_baseline", "injected": True, "period_days": 5.0, "duration_hours": 4.0, "depth": 0.001, "epoch_phase_fraction": 0.45})
    return SimpleNamespace(target_id=TARGET_ID, quarter=QUARTER, output_dir=OUTPUT_DIR, fap_threshold_path=FAP_THRESHOLD_PATH, quality_policy="default", require_finite_flux_error=False, test_fraction=0.20, arima_order=(1, 1, 0), fit_maxiter=200, min_period_days=1.0, max_period_days=15.0, n_periods=10000, min_duration_hours=1.5, max_duration_hours=10.0, n_durations=8, edge_width_cadences=0, min_edge_observations=4, min_transit_events=3, min_event_consistency_fraction=0.60, top_k=10, n_coarse_periods=4000, n_refinement_regions=30, refinement_half_width_points=40, period_match_tolerance_fraction=0.02, validation_cases=validation_cases)

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

def build_case_flux(time, flux, case):
    if not case["injected"]:
        return np.asarray(flux, dtype=float).copy(), np.nan, 0
    finite = np.isfinite(time) & np.isfinite(flux)
    period = float(case["period_days"])
    duration_days = float(case["duration_hours"]) / 24.0
    epoch = float(np.min(time[finite]) + float(case["epoch_phase_fraction"]) * period)
    injected_flux, template, in_transit = inject_periodic_box_transit(time, flux, period, epoch, duration_days, float(case["depth"]))
    in_transit_count = int(np.isfinite(flux[in_transit]).sum())
    return injected_flux, epoch, in_transit_count

def count_top_period_matches(first_periods, second_periods, tolerance_fraction):
    unmatched = [float(value) for value in second_periods]
    match_count = 0
    for first_period in first_periods:
        if not unmatched:
            break
        errors = [abs(float(first_period) - second_period) / second_period for second_period in unmatched]
        best_index = int(np.argmin(errors))
        if errors[best_index] <= float(tolerance_fraction):
            match_count += 1
            unmatched.pop(best_index)
    return int(match_count)

def run_search(time, innovations, period_grid, duration_grid, search_mode, args):
    start = perf_counter()
    result = run_tcf(time, innovations, period_grid, duration_grid, edge_width_cadences=args.edge_width_cadences, min_edge_observations=args.min_edge_observations, min_transit_events=args.min_transit_events, min_event_consistency_fraction=args.min_event_consistency_fraction, top_k=args.top_k, search_mode=search_mode, n_coarse_periods=args.n_coarse_periods, n_refinement_regions=args.n_refinement_regions, refinement_half_width_points=args.refinement_half_width_points)
    runtime_seconds = float(perf_counter() - start)
    return result, runtime_seconds

def prepare_top_peaks(case_name, search_mode, result):
    top_peaks = result["top_peaks"].copy()
    top_peaks.insert(0, "search_mode", search_mode)
    top_peaks.insert(0, "case_name", case_name)
    return top_peaks

def validate_case(time, flux, period_grid, duration_grid, threshold, case, args):
    case_flux, injected_epoch, in_transit_count = build_case_flux(time, flux, case)
    arima_result = fit_arima_innovations(case_flux, order=args.arima_order, maxiter=args.fit_maxiter)
    innovations = arima_result["innovations"]
    coarse_result, coarse_runtime = run_search(time, innovations, period_grid, duration_grid, "coarse_to_fine", args)
    exhaustive_result, exhaustive_runtime = run_search(time, innovations, period_grid, duration_grid, "exhaustive", args)
    coarse_best = coarse_result["summary"]
    exhaustive_best = exhaustive_result["summary"]
    coarse_peaks = coarse_result["top_peaks"]
    exhaustive_peaks = exhaustive_result["top_peaks"]
    coarse_periods = coarse_peaks["period_days"].to_numpy(dtype=float)
    exhaustive_periods = exhaustive_peaks["period_days"].to_numpy(dtype=float)
    overlap_count = count_top_period_matches(coarse_periods, exhaustive_periods, args.period_match_tolerance_fraction)
    overlap_fraction = float(overlap_count / max(1, min(len(coarse_periods), len(exhaustive_periods))))
    coarse_rank1_period = float(coarse_best["period"])
    exhaustive_rank1_period = float(exhaustive_best["period"])
    rank1_exact_error = float(abs(coarse_rank1_period - exhaustive_rank1_period) / exhaustive_rank1_period)
    rank1_harmonic_error = float(period_match_fraction(coarse_rank1_period, exhaustive_rank1_period))
    exhaustive_score = float(exhaustive_best["score"])
    coarse_score = float(coarse_best["score"])
    score_loss_fraction = float(max(0.0, exhaustive_score - coarse_score) / abs(exhaustive_score)) if exhaustive_score != 0 else np.nan
    is_injected = bool(case["injected"])
    injected_period = float(case["period_days"]) if is_injected else np.nan
    coarse_exact_rank = matching_peak_rank(coarse_peaks, injected_period, tolerance_fraction=args.period_match_tolerance_fraction) if is_injected else None
    exhaustive_exact_rank = matching_peak_rank(exhaustive_peaks, injected_period, tolerance_fraction=args.period_match_tolerance_fraction) if is_injected else None
    coarse_exact_present = bool(coarse_exact_rank is not None) if is_injected else False
    exhaustive_exact_present = bool(exhaustive_exact_rank is not None) if is_injected else False
    row = {"case_name": str(case["case_name"]), "is_injected": is_injected, "injected_period_days": injected_period, "injected_epoch_days": float(injected_epoch) if is_injected else np.nan, "injected_duration_hours": float(case["duration_hours"]) if is_injected else np.nan, "injected_depth": float(case["depth"]) if is_injected else np.nan, "epoch_phase_fraction": float(case["epoch_phase_fraction"]) if is_injected else np.nan, "in_transit_observation_count": int(in_transit_count), "arima_converged": bool(arima_result["summary"]["converged"]), "coarse_rank1_period_days": coarse_rank1_period, "exhaustive_rank1_period_days": exhaustive_rank1_period, "rank1_exact_error_fraction": rank1_exact_error, "rank1_exact_agreement": bool(rank1_exact_error <= args.period_match_tolerance_fraction), "rank1_harmonic_error_fraction": rank1_harmonic_error, "rank1_harmonic_agreement": bool(rank1_harmonic_error <= args.period_match_tolerance_fraction), "coarse_score": coarse_score, "exhaustive_score": exhaustive_score, "coarse_score_loss_fraction": score_loss_fraction, "coarse_exact_period_rank_top10": coarse_exact_rank, "exhaustive_exact_period_rank_top10": exhaustive_exact_rank, "coarse_exact_period_present_top10": coarse_exact_present, "exhaustive_exact_period_present_top10": exhaustive_exact_present, "exact_period_retained_by_coarse": bool(coarse_exact_present or not exhaustive_exact_present) if is_injected else True, "top10_overlap_count": overlap_count, "top10_overlap_fraction": overlap_fraction, "coarse_runtime_seconds": coarse_runtime, "exhaustive_runtime_seconds": exhaustive_runtime, "exhaustive_to_coarse_speedup": float(exhaustive_runtime / coarse_runtime) if coarse_runtime > 0 else np.nan, "coarse_evaluated_period_count": int(coarse_result["search_summary"]["evaluated_period_count"]), "exhaustive_evaluated_period_count": int(exhaustive_result["search_summary"]["evaluated_period_count"]), "current_coarse_fap_threshold": float(threshold), "coarse_passes_current_threshold": bool(coarse_score >= threshold), "exhaustive_passes_current_coarse_threshold": bool(exhaustive_score >= threshold), "current_threshold_decision_agreement": bool((coarse_score >= threshold) == (exhaustive_score >= threshold))}
    top_peak_tables = [prepare_top_peaks(case["case_name"], "coarse_to_fine", coarse_result), prepare_top_peaks(case["case_name"], "exhaustive", exhaustive_result)]
    return row, top_peak_tables

def run_validation(args):
    light_curve = load_kepler_pdcsap(args.target_id, args.quarter)
    regular, preprocessing = preprocess_pdcsap_light_curve(light_curve.to_dataframe(), quality_policy=args.quality_policy, require_finite_flux_error=args.require_finite_flux_error, normalization_fit_fraction=1.0 - args.test_fraction)
    time = regular["time"].to_numpy(dtype=float)
    flux = regular["normalized_flux"].to_numpy(dtype=float)
    period_grid = default_period_grid(time, min_period_days=args.min_period_days, max_period_days=args.max_period_days, n_periods=args.n_periods)
    duration_grid = default_duration_grid(args.min_duration_hours, args.max_duration_hours, args.n_durations)
    thresholds, threshold = load_fap_threshold(args.fap_threshold_path)
    rows = []
    top_peak_tables = []
    for case in tqdm(args.validation_cases, desc="TCF search-mode validation"):
        row, case_peak_tables = validate_case(time, flux, period_grid, duration_grid, threshold, case, args)
        rows.append(row)
        top_peak_tables.extend(case_peak_tables)
    results = pd.DataFrame(rows)
    top_peaks = pd.concat(top_peak_tables, ignore_index=True)
    injected_results = results[results["is_injected"]].copy()
    exhaustive_detected = injected_results[injected_results["exhaustive_exact_period_present_top10"]]
    candidate_retention_rate = float(exhaustive_detected["coarse_exact_period_present_top10"].mean()) if not exhaustive_detected.empty else np.nan
    summary = {"target_id": str(args.target_id), "quarter": int(args.quarter), "validation_case_count": int(len(results)), "injected_case_count": int(len(injected_results)), "rank1_exact_agreement_rate": float(results["rank1_exact_agreement"].mean()), "rank1_harmonic_agreement_rate": float(results["rank1_harmonic_agreement"].mean()), "median_top10_overlap_fraction": float(results["top10_overlap_fraction"].median()), "minimum_top10_overlap_fraction": float(results["top10_overlap_fraction"].min()), "coarse_exact_period_top10_rate": float(injected_results["coarse_exact_period_present_top10"].mean()), "exhaustive_exact_period_top10_rate": float(injected_results["exhaustive_exact_period_present_top10"].mean()), "coarse_candidate_retention_rate_when_exhaustive_detects": candidate_retention_rate, "median_coarse_score_loss_fraction": float(results["coarse_score_loss_fraction"].median()), "maximum_coarse_score_loss_fraction": float(results["coarse_score_loss_fraction"].max()), "median_speedup": float(results["exhaustive_to_coarse_speedup"].median()), "threshold_decision_agreement_rate": float(results["current_threshold_decision_agreement"].mean()), "current_threshold": float(threshold), "threshold_scope_note": "The current FAP threshold was calibrated for coarse-to-fine TCF. Exhaustive threshold comparisons are diagnostic only.", "coarse_search_acceptable_for_multistar": bool(candidate_retention_rate == 1.0 and results["coarse_score_loss_fraction"].max() <= 0.05 and results["top10_overlap_fraction"].median() >= 0.80)}
    return regular, thresholds, results, top_peaks, summary

def main(args=None):
    args = args or default_settings()
    metrics_dir = Path(args.output_dir) / "metrics"
    processed_dir = Path(args.output_dir) / "processed"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    regular, thresholds, results, top_peaks, summary = run_validation(args)
    prefix = f"kic_{str(args.target_id).replace('KIC', '').strip()}_q{args.quarter}"
    results_path = metrics_dir / f"{prefix}_tcf_search_mode_validation.csv"
    top_peaks_path = metrics_dir / f"{prefix}_tcf_search_mode_top_peaks.csv"
    summary_path = metrics_dir / f"{prefix}_tcf_search_mode_validation_summary.json"
    results.to_csv(results_path, index=False)
    top_peaks.to_csv(top_peaks_path, index=False)
    thresholds.to_csv(metrics_dir / f"{prefix}_tcf_search_mode_used_thresholds.csv", index=False)
    regular.to_parquet(processed_dir / f"{prefix}_tcf_search_mode_validation_input.parquet", index=False)
    summary_path.write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    print(f"Validation results: {results_path}")
    print(f"Validation top peaks: {top_peaks_path}")
    print(f"Validation summary: {summary_path}")
    print("\nSearch-mode comparison:\n")
    display_columns = ["case_name", "coarse_rank1_period_days", "exhaustive_rank1_period_days", "rank1_exact_agreement", "rank1_harmonic_agreement", "coarse_score", "exhaustive_score", "coarse_score_loss_fraction", "coarse_exact_period_rank_top10", "exhaustive_exact_period_rank_top10", "top10_overlap_fraction", "coarse_runtime_seconds", "exhaustive_runtime_seconds", "exhaustive_to_coarse_speedup"]
    print(results[display_columns].to_string(index=False))
    print("\nValidation summary:\n")
    print(f"Rank-1 exact agreement: {summary['rank1_exact_agreement_rate']:.3f}")
    print(f"Rank-1 harmonic agreement: {summary['rank1_harmonic_agreement_rate']:.3f}")
    print(f"Median top-10 overlap: {summary['median_top10_overlap_fraction']:.3f}")
    print(f"Coarse exact-period top-10 rate: {summary['coarse_exact_period_top10_rate']:.3f}")
    print(f"Exhaustive exact-period top-10 rate: {summary['exhaustive_exact_period_top10_rate']:.3f}")
    print(f"Candidate retention when exhaustive detects: {summary['coarse_candidate_retention_rate_when_exhaustive_detects']:.3f}")
    print(f"Median coarse score loss: {summary['median_coarse_score_loss_fraction']:.3%}")
    print(f"Maximum coarse score loss: {summary['maximum_coarse_score_loss_fraction']:.3%}")
    print(f"Median speedup: {summary['median_speedup']:.2f}x")
    print(f"Current-threshold decision agreement: {summary['threshold_decision_agreement_rate']:.3f}")
    print(f"Coarse search acceptable for multistar run: {summary['coarse_search_acceptable_for_multistar']}")
    print("\nImportant: the existing FAP threshold belongs to coarse-to-fine TCF. The exhaustive threshold decisions shown here are diagnostic, not formally calibrated.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())