"""Inspect recurring periods in TCF null trials."""
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NULL_TRIALS_PATH = PROJECT_ROOT / "outputs/experiments/tcf_null_calibration/metrics/kic_11904151_q5_tcf_null_trials.csv"
INJECTION_RESULTS_PATH = PROJECT_ROOT / "outputs/experiments/tcf_injection_grid/metrics/kic_11904151_q5_tcf_injection_grid.csv"

def load_successful_null_trials(path):
    if not path.exists():
        raise FileNotFoundError(f"TCF null-trial file does not exist: {path}")
    trials = pd.read_csv(path)
    required_columns = {"success", "best_period", "max_score"}
    missing_columns = required_columns.difference(trials.columns)
    if missing_columns:
        raise ValueError(f"TCF null-trial file is missing columns: {sorted(missing_columns)}")
    success = trials["success"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    successful = trials[success & np.isfinite(trials["best_period"]) & np.isfinite(trials["max_score"])].copy()
    if successful.empty:
        raise ValueError("No successful finite TCF null trials were found.")
    return successful

def load_injection_results(path):
    if not path.exists():
        raise FileNotFoundError(f"TCF injection-result file does not exist: {path}")
    results = pd.read_csv(path)
    required_columns = {"recovered_period_days", "recovered_score", "injected_period_days"}
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        raise ValueError(f"TCF injection-result file is missing columns: {sorted(missing_columns)}")
    return results

def preferred_period_table(null_trials, injection_results):
    null_counts = null_trials["best_period"].round(6).value_counts().rename("null_count")
    injection_counts = injection_results["recovered_period_days"].round(6).value_counts().rename("injection_count")
    comparison = pd.concat([null_counts, injection_counts], axis=1).fillna(0).reset_index()
    comparison = comparison.rename(columns={"index": "period_days"})
    comparison["null_count"] = comparison["null_count"].astype(int)
    comparison["injection_count"] = comparison["injection_count"].astype(int)
    comparison["null_fraction"] = comparison["null_count"] / len(null_trials)
    comparison["injection_fraction"] = comparison["injection_count"] / len(injection_results)
    comparison = comparison.sort_values(["injection_count", "null_count"], ascending=False)
    return comparison

def main():
    null_trials = load_successful_null_trials(NULL_TRIALS_PATH)
    injection_results = load_injection_results(INJECTION_RESULTS_PATH)
    comparison = preferred_period_table(null_trials, injection_results)
    print("\nMost frequent null TCF periods:\n")
    print(null_trials["best_period"].round(6).value_counts().head(20).to_string())
    print("\nMost frequent injection-grid TCF periods:\n")
    print(injection_results["recovered_period_days"].round(6).value_counts().head(20).to_string())
    print("\nPeriods recurring in null and injection searches:\n")
    print(comparison.head(20).to_string(index=False))
    print("\nNull-period summary:\n")
    print(null_trials["best_period"].describe().to_string())
    print("\nBoundary selections:\n")
    print(f"At lower boundary near 1 day: {int((null_trials['best_period'] <= 1.001).sum())}")
    print(f"At upper boundary near 15 days: {int((null_trials['best_period'] >= 14.999).sum())}")
    preferred_periods = [1.0, 2.513514, 3.998999, 9.996997, 15.0]
    print("\nSelected preferred-period counts in null trials:\n")
    for period in preferred_periods:
        count = int(np.isclose(null_trials["best_period"], period, rtol=0, atol=0.001).sum())
        fraction = float(count / len(null_trials))
        print(f"{period:.6f} days: {count} null trials ({fraction:.3%})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())