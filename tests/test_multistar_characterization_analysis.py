import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_script(name, path):
    script_dir = str(Path(path).parent.resolve())
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_multistar_characterization_builds_improvement_table() -> None:
    analysis = load_script("analyze_multistar_characterization_effects", "scripts/analyze_multistar_characterization_effects.py")
    features = pd.DataFrame(
        [
            {
                "target_id": "1",
                "quarter": 5,
                "selection_group": "quiet",
                "acf_decay_e_days": 0.1,
                "integrated_positive_acf_days": 0.2,
                "acf_lag_1": 0.2,
                "spectral_concentration": 0.05,
                "spectral_entropy": 0.9,
                "rolling_variance_max_to_median": 1.1,
                "gap_fraction": 0.01,
            },
            {
                "target_id": "2",
                "quarter": 5,
                "selection_group": "active",
                "acf_decay_e_days": 1.0,
                "integrated_positive_acf_days": 2.0,
                "acf_lag_1": 0.8,
                "spectral_concentration": 0.30,
                "spectral_entropy": 0.4,
                "rolling_variance_max_to_median": 3.0,
                "gap_fraction": 0.03,
            },
        ]
    )
    injections = pd.DataFrame(
        [
            {
                "target_id": "1",
                "quarter": 5,
                "raw_bls_harmonic_recovered_star_fap": True,
                "arima_tcf_harmonic_recovered_star_fap": False,
                "kalman_bls_harmonic_recovered_star_fap": True,
                "kalman_tcf_harmonic_recovered_star_fap": False,
                "gp_bls_harmonic_recovered_star_fap": True,
                "gp_tcf_harmonic_recovered_star_fap": True,
                "raw_residual_acf1": 0.2,
                "raw_local_snr": 10.0,
                "arima_residual_acf1": 0.1,
                "arima_snr_retention_fraction": 0.4,
                "arima_depth_retention_fraction": 0.5,
                "kalman_residual_acf1": 0.05,
                "kalman_snr_retention_fraction": 0.8,
                "kalman_depth_retention_fraction": 0.9,
                "gp_residual_acf1": 0.08,
                "gp_snr_retention_fraction": 0.9,
                "gp_depth_retention_fraction": 0.9,
            },
            {
                "target_id": "2",
                "quarter": 5,
                "raw_bls_harmonic_recovered_star_fap": False,
                "arima_tcf_harmonic_recovered_star_fap": True,
                "kalman_bls_harmonic_recovered_star_fap": True,
                "kalman_tcf_harmonic_recovered_star_fap": True,
                "gp_bls_harmonic_recovered_star_fap": False,
                "gp_tcf_harmonic_recovered_star_fap": True,
                "raw_residual_acf1": 0.8,
                "raw_local_snr": 4.0,
                "arima_residual_acf1": 0.1,
                "arima_snr_retention_fraction": 0.7,
                "arima_depth_retention_fraction": 0.8,
                "kalman_residual_acf1": 0.2,
                "kalman_snr_retention_fraction": 0.9,
                "kalman_depth_retention_fraction": 0.9,
                "gp_residual_acf1": 0.5,
                "gp_snr_retention_fraction": 0.6,
                "gp_depth_retention_fraction": 0.7,
            },
        ]
    )

    table = analysis.build_star_table(features, injections)

    first = table[table["target_id"] == "1"].iloc[0]
    second = table[table["target_id"] == "2"].iloc[0]
    assert first["arima_improvement"] == -1.0
    assert bool(first["raw_bls_preferable_or_tied"])
    assert second["arima_improvement"] == 1.0
    assert second["kalman_improvement"] == 1.0
    assert second["gp_improvement"] == 1.0
    assert np.isclose(second["arima_whitening_abs_acf1_reduction"], 0.7)


def test_multistar_worker_count_reserves_cpu_cores(monkeypatch) -> None:
    analysis = load_script("analyze_multistar_characterization_effects_workers", "scripts/analyze_multistar_characterization_effects.py")
    monkeypatch.setattr(analysis.os, "cpu_count", lambda: 10)

    assert analysis.resolve_worker_count(None, 2, 50) == 8
    assert analysis.resolve_worker_count(None, 2, 4) == 4
    assert analysis.resolve_worker_count(3, 2, 50) == 3


def test_challenger_runner_worker_count_reserves_cpu_cores(monkeypatch) -> None:
    runner = load_script("run_multistar_challenger_benchmark_workers", "scripts/run_multistar_challenger_benchmark.py")
    monkeypatch.setattr(runner.os, "cpu_count", lambda: 10)

    args = runner.default_settings("pilot")
    assert args.max_workers is None
    assert args.reserve_cpu_cores == 2
    assert runner.resolve_worker_count(args, 50) == 8

    args.max_workers = 3
    assert runner.resolve_worker_count(args, 50) == 3
