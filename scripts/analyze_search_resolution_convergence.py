"""Compare transit-search resolution runs on identical target/injection cases."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
KEY_COLUMNS = ["target_id", "quarter", "case_index", "injected_period_days", "injected_duration_hours", "injected_depth", "epoch_phase_fraction"]

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare candidate detector-resolution runs against a higher-resolution reference run.")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--minimum-harmonic-agreement", type=float, default=0.98)
    parser.add_argument("--maximum-recovery-rate-difference", type=float, default=0.02)
    return parser.parse_args(argv)

def injection_path(directory):
    directory = Path(directory)
    direct = directory / "multistar_challenger_injections.csv"
    metrics = directory / "metrics" / "multistar_challenger_injections.csv"
    if metrics.exists():
        return metrics
    if direct.exists():
        return direct
    raise FileNotFoundError(f"Could not find multistar_challenger_injections.csv under {directory}")

def load_injections(directory):
    frame = pd.read_csv(injection_path(directory), dtype={"target_id": str})
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Injection table is missing join columns: {missing}")
    frame["target_id"] = frame["target_id"].astype(str)
    frame["quarter"] = pd.to_numeric(frame["quarter"], errors="raise").astype(int)
    frame["case_index"] = pd.to_numeric(frame["case_index"], errors="raise").astype(int)
    return frame

def available_pipelines(frame):
    suffix = "_harmonic_rank1_matched"
    return sorted(column[:-len(suffix)] for column in frame.columns if column.endswith(suffix))

def bool_values(series):
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin(("true", "1", "yes"))

def compare_candidate(reference, candidate, candidate_name, minimum_agreement, maximum_rate_difference):
    pipelines = sorted(set(available_pipelines(reference)).intersection(available_pipelines(candidate)))
    if not pipelines:
        raise ValueError(f"No common pipelines found for {candidate_name}")
    reference_columns = KEY_COLUMNS.copy()
    candidate_columns = KEY_COLUMNS.copy()
    for pipeline in pipelines:
        for suffix in ("harmonic_rank1_matched", "exact_rank1_matched", "recovered_period_days", "runtime_seconds"):
            column = f"{pipeline}_{suffix}"
            if column in reference.columns:
                reference_columns.append(column)
            if column in candidate.columns:
                candidate_columns.append(column)
    joined = candidate[candidate_columns].merge(reference[reference_columns], on=KEY_COLUMNS, how="inner", suffixes=("_candidate", "_reference"))
    rows = []
    for pipeline in pipelines:
        candidate_harmonic = bool_values(joined[f"{pipeline}_harmonic_rank1_matched_candidate"])
        reference_harmonic = bool_values(joined[f"{pipeline}_harmonic_rank1_matched_reference"])
        harmonic_agreement = float((candidate_harmonic == reference_harmonic).mean()) if len(joined) else float("nan")
        candidate_rate = float(candidate_harmonic.mean()) if len(joined) else float("nan")
        reference_rate = float(reference_harmonic.mean()) if len(joined) else float("nan")
        rate_difference = float(abs(candidate_rate - reference_rate)) if np.isfinite(candidate_rate) and np.isfinite(reference_rate) else float("nan")
        exact_candidate_column = f"{pipeline}_exact_rank1_matched_candidate"
        exact_reference_column = f"{pipeline}_exact_rank1_matched_reference"
        exact_agreement = float((bool_values(joined[exact_candidate_column]) == bool_values(joined[exact_reference_column])).mean()) if exact_candidate_column in joined.columns and exact_reference_column in joined.columns and len(joined) else float("nan")
        period_candidate_column = f"{pipeline}_recovered_period_days_candidate"
        period_reference_column = f"{pipeline}_recovered_period_days_reference"
        period_fraction_difference = float("nan")
        if period_candidate_column in joined.columns and period_reference_column in joined.columns:
            candidate_period = pd.to_numeric(joined[period_candidate_column], errors="coerce").to_numpy(dtype=float)
            reference_period = pd.to_numeric(joined[period_reference_column], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(candidate_period) & np.isfinite(reference_period) & (reference_period > 0)
            if valid.any():
                period_fraction_difference = float(np.median(np.abs(candidate_period[valid] - reference_period[valid]) / reference_period[valid]))
        runtime_candidate_column = f"{pipeline}_runtime_seconds_candidate"
        runtime_reference_column = f"{pipeline}_runtime_seconds_reference"
        candidate_runtime = float(pd.to_numeric(joined[runtime_candidate_column], errors="coerce").median()) if runtime_candidate_column in joined.columns else float("nan")
        reference_runtime = float(pd.to_numeric(joined[runtime_reference_column], errors="coerce").median()) if runtime_reference_column in joined.columns else float("nan")
        speedup = float(reference_runtime / candidate_runtime) if np.isfinite(reference_runtime) and np.isfinite(candidate_runtime) and candidate_runtime > 0 else float("nan")
        converged = bool(np.isfinite(harmonic_agreement) and harmonic_agreement >= float(minimum_agreement) and np.isfinite(rate_difference) and rate_difference <= float(maximum_rate_difference))
        rows.append({"candidate": candidate_name, "pipeline": pipeline, "matched_case_count": int(len(joined)), "harmonic_label_agreement": harmonic_agreement, "exact_label_agreement": exact_agreement, "candidate_harmonic_recovery_rate": candidate_rate, "reference_harmonic_recovery_rate": reference_rate, "absolute_harmonic_recovery_rate_difference": rate_difference, "median_recovered_period_fraction_difference": period_fraction_difference, "candidate_median_runtime_seconds": candidate_runtime, "reference_median_runtime_seconds": reference_runtime, "runtime_speedup_vs_reference": speedup, "converged": converged})
    return pd.DataFrame(rows)

def main(argv=None):
    args = parse_args(argv)
    reference = load_injections(args.reference_dir)
    tables = []
    for candidate_dir in args.candidate_dir:
        candidate = load_injections(candidate_dir)
        tables.append(compare_candidate(reference, candidate, Path(candidate_dir).name, args.minimum_harmonic_agreement, args.maximum_recovery_rate_difference))
    comparison = pd.concat(tables, ignore_index=True)
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path(args.reference_dir) / "resolution_convergence"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "search_resolution_convergence.csv", index=False)
    summary_rows = []
    for candidate, group in comparison.groupby("candidate", sort=False):
        summary_rows.append({"candidate": candidate, "pipeline_count": int(len(group)), "all_pipelines_converged": bool(group["converged"].all()), "minimum_harmonic_label_agreement": float(group["harmonic_label_agreement"].min()), "maximum_absolute_recovery_rate_difference": float(group["absolute_harmonic_recovery_rate_difference"].max()), "median_runtime_speedup_vs_reference": float(group["runtime_speedup_vs_reference"].median())})
    summary = {"reference_dir": str(args.reference_dir), "minimum_harmonic_agreement": float(args.minimum_harmonic_agreement), "maximum_recovery_rate_difference": float(args.maximum_recovery_rate_difference), "candidates": summary_rows}
    (output_dir / "search_resolution_convergence_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(comparison.to_string(index=False))
    print(f"Output directory: {output_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
