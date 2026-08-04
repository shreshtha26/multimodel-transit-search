"""Inspect BLS injection-recovery performance by regime."""
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
path = PROJECT_ROOT / "outputs/experiments/bls_injection_grid/metrics/kic_11904151_q5_bls_injection_grid.csv"

results = pd.read_csv(path)
columns = ["injected_depth", "injected_duration_hours", "injected_period_days"]
regimes = results.groupby(columns, as_index=False).agg(injection_count=("recovered_at_fap_1_percent", "size"), period_match_rate=("period_matched", "mean"), recovery_rate_fap_1_percent=("recovered_at_fap_1_percent", "mean"), median_power=("recovered_power", "median"))
regimes = regimes.sort_values(["recovery_rate_fap_1_percent", "period_match_rate", "median_power"])
print("\nRecovery by combined regime:\n")
print(regimes.to_string(index=False))
failures = results[~results["recovered_at_fap_1_percent"]]
failure_columns = columns + ["epoch_phase_fraction", "recovered_period_days", "recovered_power", "period_matched", "passes_fap_1_percent"]
print("\nFailed injections:\n")
spurious = results[np.isclose(results["recovered_period_days"], 11.720721, rtol=0, atol=0.01)]
spurious_columns = columns + ["epoch_phase_fraction", "recovered_period_days", "recovered_power", "period_matched", "passes_fap_1_percent"]
print("\nInjections selecting the recurring 11.72-day peak:\n")
print(spurious[spurious_columns].to_string(index=False))
print(failures[failure_columns].to_string(index=False))