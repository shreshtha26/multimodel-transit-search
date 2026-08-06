"""Build a candidate-level reranking dataset from multi-star BLS/ARIMA-TCF outputs."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_DIR = PROJECT_ROOT / "outputs/experiments/multistar_bls_tcf/optimized/metrics"


def default_settings():
    return SimpleNamespace(metrics_dir=DEFAULT_METRICS_DIR, period_tolerance_fraction=0.02, merge_tolerance_fraction=0.002)


def build_parser():
    defaults = default_settings()
    parser = argparse.ArgumentParser(description="Build multi-star BLS/ARIMA-TCF candidate reranking rows.")
    parser.add_argument("--metrics-dir", type=Path, default=defaults.metrics_dir)
    parser.add_argument("--period-tolerance-fraction", type=float, default=defaults.period_tolerance_fraction)
    parser.add_argument("--merge-tolerance-fraction", type=float, default=defaults.merge_tolerance_fraction)
    return parser


def normalize_target_id(value):
    return str(value).upper().replace("KIC", "").strip()


def normalize_key_columns(frame):
    frame = frame.copy()
    if "target_id" in frame.columns:
        frame["target_id"] = frame["target_id"].map(normalize_target_id)
    if "quarter" in frame.columns:
        frame["quarter"] = pd.to_numeric(frame["quarter"], errors="raise").astype(int)
    return frame


def parse_json_list(value):
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [float(item) for item in parsed if item is not None and np.isfinite(float(item))]


def period_fractional_error(candidate_period, target_period):
    candidate_period = float(candidate_period)
    target_period = float(target_period)
    if target_period <= 0 or not np.isfinite(candidate_period):
        return np.nan
    return float(abs(candidate_period - target_period) / target_period)


def harmonic_fractional_error(candidate_period, injected_period):
    candidate_period = float(candidate_period)
    injected_period = float(injected_period)
    factors = (0.5, 1.0, 2.0, 3.0)
    errors = {factor: period_fractional_error(candidate_period, injected_period * factor) for factor in factors}
    best_factor = min(errors, key=errors.get)
    return float(errors[best_factor]), float(best_factor)


def periods_match(first, second, tolerance_fraction):
    first = float(first)
    second = float(second)
    denominator = min(abs(first), abs(second))
    if denominator <= 0:
        return False
    return abs(first - second) / denominator <= float(tolerance_fraction)


def detector_harmonic_fractional_error(candidate_period, reference_period):
    if pd.isna(candidate_period) or pd.isna(reference_period):
        return np.nan, np.nan
    candidate_period = float(candidate_period)
    reference_period = float(reference_period)
    if candidate_period <= 0 or reference_period <= 0:
        return np.nan, np.nan
    factors = (0.5, 1.0, 2.0, 3.0)
    errors = {factor: period_fractional_error(candidate_period, reference_period * factor) for factor in factors}
    best_factor = min(errors, key=errors.get)
    return float(errors[best_factor]), float(best_factor)


def empty_candidate(row, candidate_period):
    exact_error = period_fractional_error(candidate_period, row["injected_period_days"])
    harmonic_error, harmonic_factor = harmonic_fractional_error(candidate_period, row["injected_period_days"])
    return {
        "target_id": normalize_target_id(row["target_id"]),
        "quarter": int(row["quarter"]),
        "injection_index": int(row["injection_index"]),
        "selection_group": row.get("selection_group", "unspecified"),
        "injected_period_days": float(row["injected_period_days"]),
        "injected_duration_hours": float(row["injected_duration_hours"]),
        "injected_depth": float(row["injected_depth"]),
        "epoch_phase_fraction": float(row["epoch_phase_fraction"]),
        "candidate_period_days": float(candidate_period),
        "candidate_period_error_fraction": exact_error,
        "candidate_harmonic_error_fraction": harmonic_error,
        "candidate_best_harmonic_factor": harmonic_factor,
        "exact_match": bool(exact_error <= 0.02),
        "harmonic_match": bool(harmonic_error <= 0.02),
        "source_detector": "",
        "detector_agreement": False,
        "detector_count": 0,
        "bls_present": False,
        "tcf_present": False,
        "has_bls_candidate": False,
        "has_tcf_candidate": False,
        "has_tcf_event_diagnostics": False,
        "bls_candidate_period_days": np.nan,
        "tcf_candidate_period_days": np.nan,
        "bls_rank": np.nan,
        "tcf_rank": np.nan,
        "bls_sde": np.nan,
        "tcf_score": np.nan,
        "bls_score_relative_to_rank1": np.nan,
        "tcf_score_relative_to_rank1": np.nan,
        "detector_candidate_period_delta_fraction": np.nan,
        "detector_candidate_harmonic_error_fraction": np.nan,
        "detector_candidate_best_harmonic_factor": np.nan,
        "bls_period_delta_to_tcf_best_fraction": np.nan,
        "tcf_period_delta_to_bls_best_fraction": np.nan,
        "candidate_to_tcf_best_harmonic_error_fraction": np.nan,
        "candidate_to_tcf_best_harmonic_factor": np.nan,
        "candidate_to_bls_best_harmonic_error_fraction": np.nan,
        "candidate_to_bls_best_harmonic_factor": np.nan,
        "bls_tcf_rank1_harmonic_error_fraction": np.nan,
        "bls_tcf_rank1_harmonic_factor": np.nan,
        "tcf_valid_transit_events": np.nan,
        "tcf_positive_transit_events": np.nan,
        "tcf_positive_event_fraction": np.nan,
        "tcf_median_event_score": np.nan,
        "tcf_raw_pooled_score": np.nan,
        "candidate_duration_hours": np.nan,
        "candidate_depth": np.nan,
        "noise_quartile": row.get("noise_quartile", "unassigned"),
        "robust_flux_scatter_ppm": float(row.get("robust_flux_scatter_ppm", np.nan)),
        "gap_fraction": float(row.get("gap_fraction", np.nan)),
        "lag_one_flux_acf": float(row.get("lag_one_flux_acf", np.nan)),
        "six_hour_scatter_proxy_ppm": float(row.get("six_hour_scatter_proxy_ppm", np.nan)),
        "in_transit_observation_count": int(row.get("in_transit_observation_count", 0)),
        "arima_converged": bool(str(row.get("arima_converged", False)).lower() in {"true", "1"}),
        "tcf_global_empirical_p_value": np.nan,
        "bls_global_empirical_p_value": np.nan,
        "tcf_regime_empirical_p_value": np.nan,
        "bls_regime_empirical_p_value": np.nan,
    }


def add_detector_candidate(candidates, row, detector, rank, period, score, merge_tolerance_fraction):
    match_index = None
    for index, candidate in enumerate(candidates):
        if periods_match(candidate["candidate_period_days"], period, merge_tolerance_fraction):
            match_index = index
            break
    if match_index is None:
        candidates.append(empty_candidate(row, period))
        match_index = len(candidates) - 1
    candidate = candidates[match_index]
    candidate["candidate_period_days"] = float(np.nanmean([candidate["candidate_period_days"], float(period)]))
    candidate[f"{detector}_present"] = True
    candidate[f"has_{detector}_candidate"] = True
    candidate[f"{detector}_candidate_period_days"] = float(period)
    candidate[f"{detector}_rank"] = int(rank)
    if detector == "bls":
        candidate["bls_sde"] = float(score)
        if int(rank) == 1:
            candidate["bls_global_empirical_p_value"] = float(row.get("bls_global_empirical_p_value", np.nan))
            candidate["bls_regime_empirical_p_value"] = float(row.get("bls_regime_empirical_p_value", np.nan))
        candidate["candidate_duration_hours"] = float(row["bls_recovered_duration_hours"]) if int(rank) == 1 else candidate["candidate_duration_hours"]
        candidate["candidate_depth"] = float(row["bls_recovered_depth"]) if int(rank) == 1 else candidate["candidate_depth"]
    else:
        candidate["tcf_score"] = float(score)
        if int(rank) == 1:
            candidate["tcf_valid_transit_events"] = float(row["tcf_valid_transit_events"])
            candidate["tcf_positive_transit_events"] = float(row["tcf_positive_transit_events"])
            candidate["tcf_positive_event_fraction"] = float(row["tcf_positive_event_fraction"])
            candidate["tcf_median_event_score"] = float(row["tcf_median_event_score"])
            candidate["tcf_raw_pooled_score"] = float(row["tcf_raw_pooled_score"])
            candidate["tcf_global_empirical_p_value"] = float(row.get("tcf_global_empirical_p_value", np.nan))
            candidate["tcf_regime_empirical_p_value"] = float(row.get("tcf_regime_empirical_p_value", np.nan))
            candidate["has_tcf_event_diagnostics"] = bool(np.isfinite(candidate["tcf_valid_transit_events"]))
            candidate["candidate_duration_hours"] = float(row["tcf_recovered_duration_hours"])


def finalize_candidate(candidate, row, period_tolerance_fraction):
    candidate = dict(candidate)
    candidate["detector_count"] = int(candidate["bls_present"]) + int(candidate["tcf_present"])
    candidate["detector_agreement"] = bool(candidate["detector_count"] == 2)
    if candidate["detector_agreement"]:
        candidate["source_detector"] = "both"
    elif candidate["bls_present"]:
        candidate["source_detector"] = "bls"
    elif candidate["tcf_present"]:
        candidate["source_detector"] = "tcf"
    candidate["candidate_period_error_fraction"] = period_fractional_error(candidate["candidate_period_days"], candidate["injected_period_days"])
    harmonic_error, harmonic_factor = harmonic_fractional_error(candidate["candidate_period_days"], candidate["injected_period_days"])
    candidate["candidate_harmonic_error_fraction"] = harmonic_error
    candidate["candidate_best_harmonic_factor"] = harmonic_factor
    candidate["exact_match"] = bool(candidate["candidate_period_error_fraction"] <= float(period_tolerance_fraction))
    candidate["harmonic_match"] = bool(candidate["candidate_harmonic_error_fraction"] <= float(period_tolerance_fraction))
    if np.isfinite(candidate["bls_sde"]) and np.isfinite(row.get("bls_sde", np.nan)) and float(row["bls_sde"]) != 0:
        candidate["bls_score_relative_to_rank1"] = float(candidate["bls_sde"] / row["bls_sde"])
    if np.isfinite(candidate["tcf_score"]) and np.isfinite(row.get("tcf_score", np.nan)) and float(row["tcf_score"]) != 0:
        candidate["tcf_score_relative_to_rank1"] = float(candidate["tcf_score"] / row["tcf_score"])
    candidate["bls_period_delta_to_tcf_best_fraction"] = period_fractional_error(candidate["candidate_period_days"], row["tcf_recovered_period_days"])
    candidate["tcf_period_delta_to_bls_best_fraction"] = period_fractional_error(candidate["candidate_period_days"], row["bls_recovered_period_days"])
    candidate["candidate_to_tcf_best_harmonic_error_fraction"], candidate["candidate_to_tcf_best_harmonic_factor"] = detector_harmonic_fractional_error(
        candidate["candidate_period_days"], row["tcf_recovered_period_days"]
    )
    candidate["candidate_to_bls_best_harmonic_error_fraction"], candidate["candidate_to_bls_best_harmonic_factor"] = detector_harmonic_fractional_error(
        candidate["candidate_period_days"], row["bls_recovered_period_days"]
    )
    candidate["bls_tcf_rank1_harmonic_error_fraction"], candidate["bls_tcf_rank1_harmonic_factor"] = detector_harmonic_fractional_error(
        row["bls_recovered_period_days"], row["tcf_recovered_period_days"]
    )
    if candidate["detector_agreement"]:
        candidate["detector_candidate_period_delta_fraction"] = period_fractional_error(
            candidate["bls_candidate_period_days"], candidate["tcf_candidate_period_days"]
        )
        candidate["detector_candidate_harmonic_error_fraction"], candidate["detector_candidate_best_harmonic_factor"] = detector_harmonic_fractional_error(
            candidate["bls_candidate_period_days"], candidate["tcf_candidate_period_days"]
        )
    candidate["has_bls_candidate"] = bool(candidate["bls_present"])
    candidate["has_tcf_candidate"] = bool(candidate["tcf_present"])
    candidate["has_tcf_event_diagnostics"] = bool(np.isfinite(candidate["tcf_valid_transit_events"]))
    return candidate


def build_candidates_for_injection(row, args):
    tcf_periods = parse_json_list(row["tcf_top_periods_json"])
    tcf_scores = parse_json_list(row["tcf_top_scores_json"])
    bls_periods = parse_json_list(row["bls_top_periods_json"])
    bls_scores = parse_json_list(row["bls_top_sde_json"])
    candidates = []
    for rank, (period, score) in enumerate(zip(bls_periods, bls_scores), start=1):
        add_detector_candidate(candidates, row, "bls", rank, period, score, args.merge_tolerance_fraction)
    for rank, (period, score) in enumerate(zip(tcf_periods, tcf_scores), start=1):
        add_detector_candidate(candidates, row, "tcf", rank, period, score, args.merge_tolerance_fraction)
    return [finalize_candidate(candidate, row, args.period_tolerance_fraction) for candidate in candidates]


def injection_case_columns():
    return ["target_id", "quarter", "injection_index"]


def load_injections(metrics_dir):
    metrics_dir = Path(metrics_dir)
    preferred = metrics_dir / "multistar_regime_calibrated_injections.csv"
    fallback = metrics_dir / "multistar_bls_tcf_injections.csv"
    path = preferred if preferred.exists() else fallback
    injections = pd.read_csv(path, dtype={"target_id": str})
    injections = normalize_key_columns(injections)
    if "injection_index" not in injections.columns:
        injections["injection_index"] = injections.groupby(["target_id", "quarter"], sort=False).cumcount().astype(int)
    return injections, path


def build_candidate_dataset(args):
    injections, source_path = load_injections(args.metrics_dir)
    required = {
        "tcf_top_periods_json",
        "tcf_top_scores_json",
        "bls_top_periods_json",
        "bls_top_sde_json",
        "injected_period_days",
        "injected_duration_hours",
        "injected_depth",
        "epoch_phase_fraction",
    }
    missing = required.difference(injections.columns)
    if missing:
        raise ValueError(f"Injection table is missing columns: {sorted(missing)}")
    rows = []
    for _, injection in injections.iterrows():
        rows.extend(build_candidates_for_injection(injection, args))
    candidates = pd.DataFrame(rows)
    candidates = normalize_key_columns(candidates)
    candidates["candidate_id"] = np.arange(len(candidates), dtype=int)
    ordering = ["candidate_id", *injection_case_columns(), "candidate_period_days", "source_detector", "exact_match", "harmonic_match"]
    remaining = [column for column in candidates.columns if column not in ordering]
    candidates = candidates[ordering + remaining]
    output_path = Path(args.metrics_dir) / "multistar_candidate_reranking_dataset.csv"
    summary_path = Path(args.metrics_dir) / "multistar_candidate_reranking_dataset_summary.json"
    candidates.to_csv(output_path, index=False)
    summary = {
        "source_injections": str(source_path),
        "output_path": str(output_path),
        "candidate_count": int(len(candidates)),
        "injection_count": int(len(injections)),
        "star_count": int(candidates["target_id"].nunique()),
        "period_tolerance_fraction": float(args.period_tolerance_fraction),
        "merge_tolerance_fraction": float(args.merge_tolerance_fraction),
        "candidate_count_per_injection_median": float(candidates.groupby(injection_case_columns()).size().median()),
        "exact_match_candidate_count": int(candidates["exact_match"].sum()),
        "harmonic_match_candidate_count": int(candidates["harmonic_match"].sum()),
        "exact_recall_at_candidate_set": float(candidates.groupby(injection_case_columns())["exact_match"].any().mean()),
        "harmonic_recall_at_candidate_set": float(candidates.groupby(injection_case_columns())["harmonic_match"].any().mean()),
        "source_detector_counts": candidates["source_detector"].value_counts(dropna=False).to_dict(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return candidates, output_path, summary_path, summary


def main(args=None):
    args = args or build_parser().parse_args()
    candidates, output_path, summary_path, summary = build_candidate_dataset(args)
    print(f"Candidate dataset: {output_path}")
    print(f"Summary: {summary_path}")
    print(f"Candidates: {len(candidates)}")
    print(f"Injections: {summary['injection_count']}")
    print(f"Stars: {summary['star_count']}")
    print(f"Median candidates per injection: {summary['candidate_count_per_injection_median']:.1f}")
    print(f"Exact candidate-set recall: {summary['exact_recall_at_candidate_set']:.3f}")
    print(f"Harmonic candidate-set recall: {summary['harmonic_recall_at_candidate_set']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
