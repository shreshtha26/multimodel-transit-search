from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_tps_like_reliability.py"
spec = importlib.util.spec_from_file_location("analyze_tps_like_reliability", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_classify_period_ratio_distinguishes_harmonics_from_unrelated():
    assert mod.classify_period_ratio(1.002) == "P"
    assert mod.classify_period_ratio(0.501) == "P/2"
    assert mod.classify_period_ratio(2.01) == "2P"
    assert mod.classify_period_ratio(1.83) == "unrelated"
    assert mod.classify_period_ratio(np.nan) == "invalid"


def test_cluster_periods_relative_groups_persistent_candidate_periods():
    periods = [2.9425, 2.9430, 2.9418, 8.30, 8.31, np.nan]
    labels = mod.cluster_periods_relative(periods, tolerance=0.01)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]
    assert labels[5] == -1


def test_star_persistence_flag_requires_same_candidate_across_injection_periods():
    df = pd.DataFrame(
        {
            "target_id": [11245408] * 6,
            "sample_stratum": ["long_memory"] * 6,
            "case_index": list(range(6)),
            "injected_period_days": [2, 2, 5, 5, 10, 10],
            "recovered_period_days": [2.9425, 2.9426, 2.9424, 2.9425, 2.9427, 2.9425],
            "success": [True] * 6,
            "exact_period_recovered": [False] * 6,
            "harmonic_period_recovered": [False] * 6,
            "mes": [413, 416, 413, 413, 413, 413],
            "max_ses": [389] * 6,
            "observed_event_count": [7] * 6,
            "expected_event_count": [32] * 6,
            "observability_fraction": [7 / 32] * 6,
        }
    )

    cases = mod.add_case_diagnostics(df)
    stars = mod.build_star_persistence_table(cases)
    row = stars.iloc[0]

    assert bool(row["persistent_star_period_flag"])
    assert row["distinct_injected_periods_in_dominant_cluster"] == 3
    assert row["dominant_period_fraction"] == 1.0
    assert abs(row["dominant_recovered_period_days"] - 2.9425) < 1e-3
    assert row["single_event_dominance_rate"] == 1.0
