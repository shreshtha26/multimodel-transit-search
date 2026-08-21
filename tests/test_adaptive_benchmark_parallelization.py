import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_transit.data.kepler_io import light_curve_cache_path
from adaptive_transit.progress import LiveBenchmarkStore


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_adaptive_transit_benchmark.py"


def load_benchmark(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_adaptive_worker_count_defaults_to_serial_and_caps(monkeypatch):
    benchmark = load_benchmark("adaptive_benchmark_worker_count")
    monkeypatch.setattr(benchmark.os, "cpu_count", lambda: 4)

    assert benchmark.resolve_star_worker_count(None, 3) == 1
    assert benchmark.resolve_star_worker_count(2, 3) == 2
    assert benchmark.resolve_star_worker_count(99, 3) == 3
    assert benchmark.resolve_star_worker_count(2, 0) == 0
    with pytest.raises(ValueError, match="--max-workers"):
        benchmark.resolve_star_worker_count(0, 3)


def test_star_worker_merge_filters_to_completed_star(tmp_path):
    benchmark = load_benchmark("adaptive_benchmark_star_merge")
    run_id = "run"
    config_hash = "hash"
    star = "kic_1_q5"
    other_star = "kic_2_q5"
    worker_dir = tmp_path / "worker"
    worker_dir.mkdir()
    pd.DataFrame(
        [
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "star_id": star,
                "trial": 0,
                "trial_seed": 11,
                "treatment": "raw",
                "detector": "bls",
                "score_name": "bls_power",
                "score_definition": "bls_power",
                "score": 2.0,
                "success": True,
            },
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "star_id": other_star,
                "trial": 0,
                "trial_seed": 22,
                "treatment": "raw",
                "detector": "bls",
                "score_name": "bls_power",
                "score_definition": "bls_power",
                "score": 3.0,
                "success": True,
            },
        ]
    ).to_csv(worker_dir / "null_score.csv", index=False)

    with LiveBenchmarkStore(tmp_path / "main") as store:
        imported = benchmark.merge_star_output_into_main_store(
            store,
            worker_dir,
            run_id=run_id,
            config_hash=config_hash,
            star=star,
        )
        assert imported == {"null_score": 1}
        merged = store.read("null_score")

    assert merged["star_id"].tolist() == [star]
    assert merged["trial_seed"].astype(int).tolist() == [11]


def _write_manifest(path: Path) -> Path:
    manifest = path / "manifest.csv"
    pd.DataFrame(
        [
            {"target_id": "1001", "quarter": 5, "selection_group": "random_clean_q5_unstratified"},
            {"target_id": "1002", "quarter": 5, "selection_group": "random_clean_q5_unstratified"},
        ]
    ).to_csv(manifest, index=False)
    return manifest


def _write_cached_light_curves(cache_dir: Path) -> None:
    time = np.arange(240, dtype=float) * 0.05
    for index, target_id in enumerate(("1001", "1002"), start=1):
        flux = 1.0 + 0.0002 * np.sin(2.0 * np.pi * time / (2.5 + index * 0.1))
        frame = pd.DataFrame(
            {
                "time": time,
                "flux": flux,
                "flux_error": np.full(time.size, 0.0001),
                "quality": np.zeros(time.size, dtype=np.int64),
                "cadenceno": np.arange(index * 1000, index * 1000 + time.size, dtype=np.int64),
            }
        )
        path = light_curve_cache_path(cache_dir, target_id, 5)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)


def _run_tiny_calibration(output_dir: Path, cache_dir: Path, manifest: Path, workers: int) -> None:
    command = [
        sys.executable,
        str(SCRIPT),
        "--profile",
        "smoke",
        "--manifest-path",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--target-limit",
        "2",
        "--active-combinations",
        "raw_bls",
        "--calibrate-fap",
        "--n-null-trials-per-star",
        "2",
        "--cache-dir",
        str(cache_dir),
        "--no-download",
        "--max-workers",
        str(workers),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _scientific_null_scores(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "null_score.csv")
    columns = [
        "run_id",
        "config_hash",
        "star_id",
        "trial",
        "trial_seed",
        "treatment",
        "detector",
        "score_name",
        "score_definition",
        "score",
        "success",
        "best_period_days",
    ]
    return frame[columns].sort_values(
        ["star_id", "trial", "treatment", "detector", "score_definition"]
    ).reset_index(drop=True)


def _scientific_thresholds(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "fap_thresholds.csv")
    columns = [
        "run_id",
        "config_hash",
        "star_id",
        "treatment",
        "detector",
        "score_name",
        "score_definition",
        "fap_level",
        "fap_threshold",
        "null_trial_count",
    ]
    return frame[columns].sort_values(["star_id", "treatment", "detector", "score_definition"]).reset_index(drop=True)


def test_tiny_two_star_two_null_parallel_matches_serial_science(tmp_path):
    cache_dir = tmp_path / "cache"
    manifest = _write_manifest(tmp_path)
    _write_cached_light_curves(cache_dir)
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"

    _run_tiny_calibration(serial_dir, cache_dir, manifest, workers=1)
    _run_tiny_calibration(parallel_dir, cache_dir, manifest, workers=2)

    pd.testing.assert_frame_equal(
        _scientific_null_scores(serial_dir),
        _scientific_null_scores(parallel_dir),
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=1e-15,
    )
    pd.testing.assert_frame_equal(
        _scientific_thresholds(serial_dir),
        _scientific_thresholds(parallel_dir),
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=1e-15,
    )
    assert (parallel_dir / "_star_work").is_dir()
