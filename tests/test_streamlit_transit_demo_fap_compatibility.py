import ast
from pathlib import Path
import sqlite3
import types

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "streamlit_transit_demo.py"


class _FakeCache:
    def __call__(self, *_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


class _FakeStreamlit:
    cache_data = _FakeCache()


def load_dashboard_helpers():
    tree = ast.parse(SCRIPT.read_text())
    wanted = {
        "normalize_target_id",
        "read_csv_safe",
        "read_live_table_safe",
        "_read_live_and_csv_table",
        "_target_quarter_from_star_id",
        "collect_calibration",
        "harmonized_thresholds",
        "harmonized_null_scores",
        "latest_compatible_thresholds",
        "calibration_coverage",
        "add_calibrated_columns",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    module = types.SimpleNamespace()
    namespace = {
        "Path": Path,
        "sqlite3": sqlite3,
        "pd": pd,
        "np": np,
        "re": __import__("re"),
        "st": _FakeStreamlit(),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    for name in wanted:
        setattr(module, name, namespace[name])
    return module


def test_thresholds_match_injection_run_with_different_fap_run_id():
    helpers = load_dashboard_helpers()
    injections = pd.DataFrame(
        [
            {
                "run_id": "injection_run_old",
                "config_hash": "injection_hash",
                "target_id": "12645975",
                "quarter": 5,
                "raw_bls_score_name": "bls_power",
                "raw_bls_score": 12.0,
                "raw_bls_harmonic_rank1_matched": True,
                "raw_bls_exact_rank1_matched": True,
            }
        ]
    )
    thresholds = pd.DataFrame(
        [
            {
                "run_id": "demo50_662c028d7106501c",
                "config_hash": "fap_hash",
                "star_id": "kic_12645975_q5",
                "treatment": "raw",
                "detector": "bls",
                "score_name": "bls_power",
                "score_definition": "bls_power",
                "fap_level": 0.01,
                "fap_threshold": 10.0,
                "null_trial_count": 1000,
                "_calibration_source_mtime": 100.0,
            }
        ]
    )

    compatible = helpers.latest_compatible_thresholds(thresholds, injections, ["raw_bls"])
    assert compatible["fap_calibration_run_id"].tolist() == ["demo50_662c028d7106501c"]
    assert compatible["score_threshold"].astype(float).tolist() == [10.0]

    calibrated = helpers.add_calibrated_columns(injections, thresholds, ["raw_bls"])
    row = calibrated.iloc[0]
    assert row["injection_run_id"] == "injection_run_old"
    assert row["raw_bls_fap01_calibration_run_id"] == "demo50_662c028d7106501c"
    assert int(row["raw_bls_fap01_null_trial_count"]) == 1000
    assert float(row["raw_bls_fap01_level"]) == 0.01
    assert bool(row["raw_bls_fap01_detected"]) is True
    assert bool(row["raw_bls_fap01_harmonic_recovered"]) is True


def test_collects_latest_sibling_adaptive_calibration_without_exact_run_match(tmp_path):
    helpers = load_dashboard_helpers()
    run_root = tmp_path / "adaptive_transit"
    injection_dir = run_root / "old_injection"
    calibration_dir = run_root / "demo50"
    injection_dir.mkdir(parents=True)
    calibration_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "run_id": "demo50_662c028d7106501c",
                "config_hash": "fap_hash",
                "star_id": "kic_12645975_q5",
                "treatment": "raw",
                "detector": "bls",
                "score_name": "bls_power",
                "score_definition": "bls_power",
                "fap_level": 0.01,
                "fap_threshold": 10.0,
                "null_trial_count": 1000,
            }
        ]
    ).to_csv(calibration_dir / "fap_thresholds.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_id": "demo50_662c028d7106501c",
                "config_hash": "fap_hash",
                "star_id": "kic_12645975_q5",
                "trial": 0,
                "treatment": "raw",
                "detector": "bls",
                "score_name": "bls_power",
                "score_definition": "bls_power",
                "score": 3.0,
                "success": True,
            }
        ]
    ).to_csv(calibration_dir / "null_score.csv", index=False)

    thresholds, nulls = helpers.collect_calibration(str(injection_dir))

    injections = pd.DataFrame(
        [
            {
                "run_id": "injection_run_old",
                "config_hash": "injection_hash",
                "target_id": "12645975",
                "quarter": 5,
                "raw_bls_score_name": "bls_power",
                "raw_bls_score": 12.0,
                "raw_bls_harmonic_rank1_matched": True,
            }
        ]
    )
    coverage = helpers.calibration_coverage(injections, thresholds, nulls, ["raw_bls"])
    assert coverage["threshold_pairs"] == 1
    assert coverage["null_stars"] == 1
    assert coverage["complete"] is True
    calibrated = helpers.add_calibrated_columns(injections, thresholds, ["raw_bls"])
    assert calibrated["raw_bls_fap01_threshold"].astype(float).tolist() == [10.0]
    assert calibrated["raw_bls_fap01_calibration_run_id"].tolist() == ["demo50_662c028d7106501c"]
