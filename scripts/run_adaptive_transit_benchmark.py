#!/usr/bin/env python
"""Run the unified long-format adaptive-transit benchmark."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adaptive_transit.config import (  # noqa: E402
    ACTIVE_SCIENTIFIC_BENCHMARKS,
    benchmark_profile,
    parse_pipeline_specs,
)
from adaptive_transit.core import LightCurve  # noqa: E402
from adaptive_transit.data.kepler_io import load_kepler_pdcsap  # noqa: E402
from adaptive_transit.detectors import DETECTORS  # noqa: E402
from adaptive_transit.fap import threshold_table_from_null_scores  # noqa: E402
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve  # noqa: E402
from adaptive_transit.resume import LongTableStore, PRIMARY_KEYS  # noqa: E402
from adaptive_transit.runner import UnifiedPipelineRunner  # noqa: E402
from adaptive_transit.schemas import LONG_TABLE_SCHEMAS, json_ready  # noqa: E402

THRESHOLD_KEYS = ("run_id", "config_hash", "star_id", "treatment", "detector", "score_definition", "fap_level")
SHARD_WIDTH = 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run benchmark100 or benchmark1000 with the unified adaptive-transit runner."
    )
    parser.add_argument("--profile", choices=(*ACTIVE_SCIENTIFIC_BENCHMARKS, "smoke"), default="benchmark100")
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-limit", type=int)
    parser.add_argument("--active-combinations", type=str, help="Comma-separated treatment_detector entries.")
    parser.add_argument("--thresholds-path", type=Path, help="Optional common-FAP threshold CSV for calibrated detections.")
    parser.add_argument("--calibrate-fap", action="store_true", help="Run moving-block null calibration instead of injections.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, help="One-based shard id.")
    parser.add_argument("--verify-shards", action="store_true", help="Verify deterministic shard partitioning and exit.")
    parser.add_argument("--qc-only", action="store_true", help="Run shard/canonical output QC without executing science.")
    parser.add_argument("--merge-shards", action="store_true", help="Merge completed shard outputs into canonical profile artifacts.")
    parser.add_argument(
        "--n-null-trials-per-star",
        type=int,
        help="Moving-block null trials per star for common-FAP calibration. Scientific default is 1000.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    return parser.parse_args(argv)


def normalize_target_id(value) -> str:
    return str(value).upper().replace("KIC", "").strip()


def load_manifest(path: Path, target_limit: int, config=None) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"target_id", "quarter"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    manifest = manifest.copy()
    manifest["quarter"] = pd.to_numeric(manifest["quarter"], errors="raise").astype(int)
    manifest["_target_key"] = manifest["target_id"].map(normalize_target_id)
    duplicates = manifest[manifest.duplicated(["_target_key", "quarter"], keep=False)]
    if not duplicates.empty:
        examples = duplicates[["target_id", "quarter"]].head(10).to_dict(orient="records")
        raise ValueError(f"Manifest contains duplicate target/quarter rows: {examples}")
    if config is not None and config.expected_quarter is not None:
        bad_quarter = manifest[manifest["quarter"] != int(config.expected_quarter)]
        if not bad_quarter.empty:
            examples = bad_quarter[["target_id", "quarter"]].head(10).to_dict(orient="records")
            raise ValueError(f"Manifest contains rows outside expected Q{config.expected_quarter}: {examples}")
    if config is not None and config.allowed_selection_groups:
        if "selection_group" not in manifest.columns:
            raise ValueError("Manifest is missing selection_group required by this benchmark profile.")
        allowed = set(config.allowed_selection_groups)
        bad_group = manifest[~manifest["selection_group"].astype(str).isin(allowed)]
        if not bad_group.empty:
            examples = bad_group[["target_id", "quarter", "selection_group"]].head(10).to_dict(orient="records")
            raise ValueError(f"Manifest contains selection groups outside {sorted(allowed)}: {examples}")
    return manifest.head(int(target_limit)).drop(columns=["_target_key"]).copy().reset_index(drop=True)


def star_id(target_id, quarter) -> str:
    return f"kic_{normalize_target_id(target_id)}_q{int(quarter)}"


def shard_name(shard_id: int) -> str:
    return f"shard_{int(shard_id):0{SHARD_WIDTH}d}"


def validate_shard_request(num_shards: int, shard_id: int | None, *, allow_all_shards: bool = False) -> None:
    if int(num_shards) < 1:
        raise ValueError("--num-shards must be at least 1.")
    if shard_id is None:
        if int(num_shards) != 1 and not allow_all_shards:
            raise ValueError("--shard-id is required when --num-shards is greater than 1.")
        return
    if not 1 <= int(shard_id) <= int(num_shards):
        raise ValueError("--shard-id must be between 1 and --num-shards.")


def shard_slices(row_count: int, num_shards: int) -> list[tuple[int, int]]:
    row_count = int(row_count)
    num_shards = int(num_shards)
    if row_count % num_shards != 0:
        raise ValueError(f"{row_count} manifest rows cannot be split evenly into {num_shards} shards.")
    shard_size = row_count // num_shards
    return [(index * shard_size, (index + 1) * shard_size) for index in range(num_shards)]


def shard_manifest(manifest: pd.DataFrame, *, num_shards: int, shard_id: int) -> pd.DataFrame:
    start, end = shard_slices(len(manifest), num_shards)[int(shard_id) - 1]
    return manifest.iloc[start:end].copy().reset_index(drop=True)


def verify_shard_partition(manifest: pd.DataFrame, *, num_shards: int) -> dict:
    key_list = [
        (normalize_target_id(row["target_id"]), int(row["quarter"]))
        for _, row in manifest.iterrows()
    ]
    if len(set(key_list)) != len(manifest):
        raise ValueError("Manifest does not contain exactly unique target/quarter rows.")
    expected = set(key_list)
    shards = []
    seen = set()
    overlaps = set()
    frames = []
    for index, (start, end) in enumerate(shard_slices(len(manifest), num_shards), start=1):
        shard = manifest.iloc[start:end].copy()
        shard_keys = {
            (normalize_target_id(row["target_id"]), int(row["quarter"]))
            for _, row in shard.iterrows()
        }
        overlap = seen.intersection(shard_keys)
        if overlap:
            overlaps.update(overlap)
        seen.update(shard_keys)
        frames.append(shard)
        shards.append(
            {
                "shard_id": index,
                "row_start_1based": start + 1,
                "row_end_1based": end,
                "star_count": int(len(shard)),
            }
        )
    missing = expected.difference(seen)
    unexpected = seen.difference(expected)
    if overlaps:
        raise ValueError(f"Shard partition overlaps earlier shards: {sorted(overlaps)[:5]}")
    if missing:
        raise ValueError(f"Shard partition is missing stars: {sorted(missing)[:5]}")
    if unexpected:
        raise ValueError(f"Shard partition contains unexpected stars: {sorted(unexpected)[:5]}")
    reconstructed = pd.concat(frames, ignore_index=True)
    if not reconstructed.equals(manifest.reset_index(drop=True)):
        raise ValueError("Concatenating shard manifests does not reconstruct the frozen manifest exactly.")
    return {
        "unique_stars": int(len(seen)),
        "overlap_count": int(len(overlaps)),
        "missing_count": int(len(missing)),
        "unexpected_count": int(len(unexpected)),
        "reconstructs_manifest": True,
        "shards": shards,
    }


def detector_parameters(config, detector_name: str) -> dict:
    return {
        **config.detector_parameters.get(detector_name, {}),
        "min_period_days": config.min_period_days,
        "max_period_days": config.max_period_days,
        "top_k": config.top_k,
    }


def score_definition_rows(config) -> list[dict]:
    rows = []
    for spec in config.active_combinations:
        detector = DETECTORS[spec.detector]
        for score_definition in detector.active_score_definitions(detector_parameters(config, spec.detector)):
            rows.append(
                {
                    "treatment": spec.treatment,
                    "detector": spec.detector,
                    "score_definition": str(score_definition),
                }
            )
    return rows


def score_definition_count(config) -> int:
    return len(score_definition_rows(config))


def expected_detection_keys(config, run_id: str, star: str, injection_ids: tuple[str, ...]) -> set[tuple]:
    keys = PRIMARY_KEYS["detection"]
    rows = []
    for injection_id in injection_ids:
        for spec in config.active_combinations:
            detector = DETECTORS[spec.detector]
            for score_definition in detector.active_score_definitions(detector_parameters(config, spec.detector)):
                rows.append(
                    {
                        "run_id": run_id,
                        "config_hash": config.config_hash,
                        "star_id": star,
                        "injection_id": injection_id,
                        "treatment": spec.treatment,
                        "detector": spec.detector,
                        "score_definition": score_definition,
                    }
                )
    return {tuple(row[key] for key in keys) for row in rows}


def star_completed(store: LongTableStore, config, run_id: str, star: str, injection_ids: tuple[str, ...]) -> bool:
    frame = store.read("detection")
    if frame.empty:
        return False
    current = frame[
        frame["run_id"].astype(str).eq(str(run_id))
        & frame["config_hash"].astype(str).eq(str(config.config_hash))
        & frame["star_id"].astype(str).eq(str(star))
    ]
    if current.empty:
        return False
    expected = expected_detection_keys(config, run_id, star, injection_ids)
    observed = {
        (
            row["run_id"],
            row["config_hash"],
            row["star_id"],
            row["injection_id"],
            row["treatment"],
            row["detector"],
            row["score_definition"],
        )
        for _, row in current.iterrows()
    }
    return expected.issubset(observed)


def expected_null_score_keys(config, run_id: str, star: str, n_trials: int) -> set[tuple]:
    keys = PRIMARY_KEYS["null_score"]
    rows = []
    for trial in range(int(n_trials)):
        for spec in config.active_combinations:
            detector = DETECTORS[spec.detector]
            for score_definition in detector.active_score_definitions(detector_parameters(config, spec.detector)):
                rows.append(
                    {
                        "run_id": run_id,
                        "config_hash": config.config_hash,
                        "star_id": star,
                        "trial": trial,
                        "treatment": spec.treatment,
                        "detector": spec.detector,
                        "score_definition": score_definition,
                    }
                )
    return {tuple(row[key] for key in keys) for row in rows}


def null_calibration_completed(
    store: LongTableStore,
    config,
    run_id: str,
    star: str,
    n_trials: int,
) -> bool:
    observed = store.completed_keys("null_score", config_hash=config.config_hash)
    expected = expected_null_score_keys(config, run_id, star, n_trials)
    return expected.issubset(observed)


def write_run_metadata(config, run_id: str, output_dir: Path, *, mode: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_metadata.json"
    payload = {
        "run_id": run_id,
        "mode": mode,
        "config_hash": config.config_hash,
        "config": config.to_hash_payload(),
        "schemas": {name: list(columns) for name, columns in LONG_TABLE_SCHEMAS.items()},
    }
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    return path


def write_shard_metadata(
    config,
    run_id: str,
    output_dir: Path,
    *,
    mode: str,
    shard_id: int | None = None,
    num_shards: int = 1,
    shard_rows: tuple[int, int] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_metadata.json"
    payload = {
        "run_id": run_id,
        "mode": mode,
        "config_hash": config.config_hash,
        "shard_id": shard_id,
        "num_shards": int(num_shards),
        "shard_name": None if shard_id is None else shard_name(shard_id),
        "shard_row_start_1based": None if shard_rows is None else int(shard_rows[0]) + 1,
        "shard_row_end_1based": None if shard_rows is None else int(shard_rows[1]),
        "config": config.to_hash_payload(),
        "schemas": {name: list(columns) for name, columns in LONG_TABLE_SCHEMAS.items()},
    }
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    return path


def load_thresholds(path: Path, config) -> pd.DataFrame:
    thresholds = pd.read_csv(path)
    if "config_hash" in thresholds.columns:
        current = thresholds[thresholds["config_hash"].astype(str).eq(str(config.config_hash))].copy()
        if current.empty:
            raise ValueError(
                f"Threshold file {path} does not contain rows for config_hash={config.config_hash}."
            )
        return current
    return thresholds


def current_rows(frame: pd.DataFrame, *, run_id: str, config_hash: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "run_id" not in frame.columns or "config_hash" not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame[
        frame["run_id"].astype(str).eq(str(run_id))
        & frame["config_hash"].astype(str).eq(str(config_hash))
    ].copy()


def duplicate_primary_key_count(frame: pd.DataFrame, keys: Iterable[str]) -> int:
    key_list = [key for key in keys if key in frame.columns]
    if not key_list or frame.empty:
        return 0
    return int(frame.duplicated(key_list).sum())


def expected_star_ids(manifest: pd.DataFrame) -> set[str]:
    return {star_id(row["target_id"], row["quarter"]) for _, row in manifest.iterrows()}


def assert_expected_star_count(stars: set[str], expected_count: int, *, context: str) -> None:
    if len(stars) != int(expected_count):
        raise ValueError(f"{context} expected {expected_count} unique stars, found {len(stars)}.")


def assert_star_set(observed: set[str], expected: set[str], *, context: str) -> dict:
    missing = expected.difference(observed)
    unexpected = observed.difference(expected)
    if missing or unexpected:
        raise ValueError(
            f"{context} star set mismatch: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_examples={sorted(missing)[:5]} unexpected_examples={sorted(unexpected)[:5]}"
        )
    return {
        "star_count": int(len(observed)),
        "missing_stars": 0,
        "unexpected_stars": 0,
    }


def assert_config_hash_consistency(frame: pd.DataFrame, config_hash: str, *, context: str) -> None:
    if frame.empty:
        return
    if "config_hash" not in frame.columns:
        raise ValueError(f"{context} is missing config_hash.")
    hashes = set(frame["config_hash"].dropna().astype(str))
    if hashes != {str(config_hash)}:
        raise ValueError(f"{context} config hashes differ from {config_hash}: {sorted(hashes)}")


def load_metadata(root: Path) -> dict:
    metadata_path = Path(root) / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing run metadata: {metadata_path}")
    return json.loads(metadata_path.read_text())


def read_table(root: Path, table_name: str) -> pd.DataFrame:
    path = Path(root) / f"{table_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {table_name} table: {path}")
    return pd.read_csv(path)


def read_thresholds(root: Path) -> pd.DataFrame:
    path = Path(root) / "fap_thresholds.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing FAP threshold table: {path}")
    return pd.read_csv(path)


def expected_score_definitions(config) -> set[tuple[str, str, str]]:
    return {
        (row["treatment"], row["detector"], row["score_definition"])
        for row in score_definition_rows(config)
    }


def assert_threshold_completeness(thresholds: pd.DataFrame, config, stars: set[str]) -> None:
    expected_defs = expected_score_definitions(config)
    for current_star in sorted(stars):
        star_rows = thresholds[thresholds["star_id"].astype(str).eq(current_star)]
        observed_defs = {
            (str(row["treatment"]), str(row["detector"]), str(row["score_definition"]))
            for _, row in star_rows.iterrows()
        }
        missing = expected_defs.difference(observed_defs)
        unexpected = observed_defs.difference(expected_defs)
        if missing or unexpected:
            raise ValueError(
                f"FAP threshold definitions incomplete for {current_star}: "
                f"missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}"
            )


def qc_thresholds(thresholds: pd.DataFrame, config, run_id: str, stars: set[str]) -> dict:
    current = current_rows(thresholds, run_id=run_id, config_hash=config.config_hash)
    if current.empty:
        raise ValueError("No FAP thresholds match the current run_id/config_hash.")
    assert_config_hash_consistency(current, config.config_hash, context="FAP thresholds")
    expected = len(stars) * len(expected_score_definitions(config))
    duplicates = duplicate_primary_key_count(current, THRESHOLD_KEYS)
    if duplicates:
        raise ValueError(f"FAP thresholds contain {duplicates} duplicate primary keys.")
    if len(current) != expected:
        raise ValueError(f"Expected {expected} FAP threshold rows, found {len(current)}.")
    assert_star_set(set(current["star_id"].astype(str)), stars, context="FAP threshold")
    if not current["fap_level"].astype(float).eq(float(config.fap_level)).all():
        raise ValueError("FAP threshold table contains an unexpected FAP level.")
    bad_counts = current[current["null_trial_count"].astype(int) != int(config.n_null_trials_per_star)]
    if not bad_counts.empty:
        examples = bad_counts[["star_id", "treatment", "detector", "score_definition", "null_trial_count"]].head(10).to_dict(orient="records")
        raise ValueError(f"FAP thresholds were not derived from {config.n_null_trials_per_star} finite null scores: {examples}")
    assert_threshold_completeness(current, config, stars)
    return {"threshold_rows": int(len(current))}


def qc_calibration_output(root: Path, config, run_id: str, manifest: pd.DataFrame) -> dict:
    stars = expected_star_ids(manifest)
    assert_expected_star_count(stars, len(manifest), context="Calibration manifest")
    metadata = load_metadata(root)
    if metadata.get("config_hash") != config.config_hash:
        raise ValueError("run_metadata.json config_hash does not match active config.")
    if metadata.get("run_id") != run_id:
        raise ValueError("run_metadata.json run_id does not match active run.")
    null_scores = current_rows(read_table(root, "null_score"), run_id=run_id, config_hash=config.config_hash)
    assert_config_hash_consistency(null_scores, config.config_hash, context="Null-score table")
    expected_rows = len(stars) * int(config.n_null_trials_per_star) * score_definition_count(config)
    if len(null_scores) != expected_rows:
        raise ValueError(f"Expected {expected_rows} null-score rows, found {len(null_scores)}.")
    duplicates = duplicate_primary_key_count(null_scores, PRIMARY_KEYS["null_score"])
    if duplicates:
        raise ValueError(f"Null scores contain {duplicates} duplicate primary keys.")
    star_info = assert_star_set(set(null_scores["star_id"].astype(str)), stars, context="Null-score")
    grouped = null_scores.groupby(["treatment", "detector", "score_definition"], dropna=False)["success"].agg(["count", "sum"])
    systematic = grouped[grouped["sum"].astype(int) == 0]
    if not systematic.empty:
        raise ValueError(f"Systematic null-score failures detected:\n{systematic.to_string()}")
    thresholds_info = qc_thresholds(read_thresholds(root), config, run_id, stars)
    return {
        **star_info,
        "null_score_rows": int(len(null_scores)),
        **thresholds_info,
    }


def qc_benchmark_output(root: Path, config, run_id: str, manifest: pd.DataFrame) -> dict:
    stars = expected_star_ids(manifest)
    assert_expected_star_count(stars, len(manifest), context="Benchmark manifest")
    metadata = load_metadata(root)
    if metadata.get("config_hash") != config.config_hash:
        raise ValueError("run_metadata.json config_hash does not match active config.")
    if metadata.get("run_id") != run_id:
        raise ValueError("run_metadata.json run_id does not match active run.")
    cases = build_case_count(config)
    treatment_count = len({spec.treatment for spec in config.active_combinations})
    score_count = score_definition_count(config)
    expected_counts = {
        "characterization": len(stars),
        "injection": len(stars) * cases,
        "preservation": len(stars) * cases * treatment_count,
        "detection": len(stars) * cases * score_count,
    }
    observed_counts = {}
    tables = {}
    for table_name, expected in expected_counts.items():
        table = current_rows(read_table(root, table_name), run_id=run_id, config_hash=config.config_hash)
        assert_config_hash_consistency(table, config.config_hash, context=f"{table_name} table")
        tables[table_name] = table
        observed_counts[table_name] = int(len(table))
        if len(table) != expected:
            raise ValueError(f"Expected {expected} {table_name} rows, found {len(table)}.")
        duplicates = duplicate_primary_key_count(table, PRIMARY_KEYS[table_name])
        if duplicates:
            raise ValueError(f"{table_name} contains {duplicates} duplicate primary keys.")
        assert_star_set(set(table["star_id"].astype(str)), stars, context=table_name)
    detection = tables["detection"]
    injection = tables["injection"][["star_id", "injection_id", "batman_used"]]
    joined = detection.merge(injection, on=["star_id", "injection_id"], how="left")
    applicable = joined[joined["batman_used"].astype(str).str.lower().isin({"true", "1", "1.0"}) & joined["success"].astype(str).str.lower().isin({"true", "1", "1.0"})]
    if applicable[["exact_recovery", "harmonic_recovery", "exact_period_error", "harmonic_period_error"]].isna().any(axis=None):
        raise ValueError("Detection recovery fields are missing for successful BATMAN injection rows.")
    grouped = detection.groupby(["treatment", "detector", "score_definition"], dropna=False)["success"].agg(["count", "sum"])
    systematic = grouped[grouped["sum"].astype(int) == 0]
    if not systematic.empty:
        raise ValueError(f"Systematic detection failures detected:\n{systematic.to_string()}")
    thresholds_info = qc_thresholds(read_thresholds(root), config, run_id, stars)
    return {"star_count": int(len(stars)), **observed_counts, **thresholds_info}


def build_case_count(config) -> int:
    grid = (
        len(config.injection_period_grid)
        * len(config.injection_duration_hours_grid)
        * len(config.injection_depth_grid)
        * len(config.epoch_phase_fraction_grid)
    )
    return int(grid + (1 if config.include_native_zero_injection else 0))


def write_manifest_used(root: Path, manifest: pd.DataFrame) -> None:
    Path(root).mkdir(parents=True, exist_ok=True)
    manifest.to_csv(Path(root) / "target_manifest_used.csv", index=False)


def qc_output(root: Path, config, run_id: str, manifest: pd.DataFrame, *, mode: str) -> dict:
    if mode == "null_calibration":
        return qc_calibration_output(root, config, run_id, manifest)
    if mode == "benchmark":
        return qc_benchmark_output(root, config, run_id, manifest)
    if mode == "complete":
        return {
            "null_calibration": qc_calibration_output(root, config, run_id, manifest),
            "benchmark": qc_benchmark_output(root, config, run_id, manifest),
        }
    raise ValueError(f"Unknown QC mode: {mode}")


def shard_output_dir(base_output_dir: Path, shard_id: int | None) -> Path:
    if shard_id is None:
        return Path(base_output_dir)
    return Path(base_output_dir) / shard_name(shard_id)


def merge_table_frames(frames: list[pd.DataFrame], table_name: str) -> pd.DataFrame:
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    keys = PRIMARY_KEYS.get(table_name)
    if keys is None:
        keys = THRESHOLD_KEYS
    duplicates = duplicate_primary_key_count(merged, keys)
    if duplicates:
        raise ValueError(f"Merged {table_name} contains {duplicates} duplicate primary keys.")
    sort_keys = [key for key in keys if key in merged.columns]
    if sort_keys and not merged.empty:
        merged = merged.sort_values(sort_keys).reset_index(drop=True)
    return merged


def merge_shard_outputs(base_output_dir: Path, config, run_id: str, full_manifest: pd.DataFrame, *, num_shards: int) -> dict:
    partition = verify_shard_partition(full_manifest, num_shards=num_shards)
    canonical = Path(base_output_dir)
    shard_dirs = [canonical / shard_name(index) for index in range(1, int(num_shards) + 1)]
    all_stars = expected_star_ids(full_manifest)
    tables = {name: [] for name in ("characterization", "injection", "preservation", "detection", "null_score")}
    threshold_frames = []
    observed_stars: set[str] = set()
    overlaps: set[str] = set()
    shard_config_hashes: set[str] = set()
    for index, shard_dir in enumerate(shard_dirs, start=1):
        manifest = shard_manifest(full_manifest, num_shards=num_shards, shard_id=index)
        metadata = load_metadata(shard_dir)
        shard_config_hashes.add(str(metadata.get("config_hash")))
        if metadata.get("config_hash") != config.config_hash:
            raise ValueError(f"{shard_dir} config_hash does not match {config.config_hash}.")
        if metadata.get("run_id") != run_id:
            raise ValueError(f"{shard_dir} run_id does not match {run_id}.")
        qc_calibration_output(shard_dir, config, run_id, manifest)
        qc_benchmark_output(shard_dir, config, run_id, manifest)
        shard_stars = expected_star_ids(manifest)
        overlaps.update(observed_stars.intersection(shard_stars))
        observed_stars.update(shard_stars)
        for table_name in tables:
            tables[table_name].append(current_rows(read_table(shard_dir, table_name), run_id=run_id, config_hash=config.config_hash))
        threshold_frames.append(current_rows(read_thresholds(shard_dir), run_id=run_id, config_hash=config.config_hash))
    missing = all_stars.difference(observed_stars)
    unexpected = observed_stars.difference(all_stars)
    if overlaps or missing or unexpected:
        raise ValueError(
            "Merged shard star coverage failed: "
            f"overlap={len(overlaps)} missing={len(missing)} unexpected={len(unexpected)}"
        )
    if shard_config_hashes != {config.config_hash}:
        raise ValueError(f"Shard config hashes are not identical: {sorted(shard_config_hashes)}")

    canonical.mkdir(parents=True, exist_ok=True)
    write_manifest_used(canonical, full_manifest)
    written = {}
    for table_name, frames in tables.items():
        merged = merge_table_frames(frames, table_name)
        if set(merged["star_id"].astype(str)) != all_stars:
            raise ValueError(f"Merged {table_name} star set does not match frozen manifest.")
        path = canonical / f"{table_name}.csv"
        merged.to_csv(path, index=False)
        written[table_name] = {"path": str(path), "rows": int(len(merged))}
    thresholds = merge_table_frames(threshold_frames, "fap_thresholds")
    if set(thresholds["star_id"].astype(str)) != all_stars:
        raise ValueError("Merged FAP threshold star set does not match frozen manifest.")
    threshold_path = canonical / "fap_thresholds.csv"
    thresholds.to_csv(threshold_path, index=False)
    written["fap_thresholds"] = {"path": str(threshold_path), "rows": int(len(thresholds))}
    write_shard_metadata(config, run_id, canonical, mode="merged_benchmark", num_shards=num_shards)
    qc_calibration_output(canonical, config, run_id, full_manifest)
    qc_benchmark_output(canonical, config, run_id, full_manifest)
    summary_path = canonical / "merge_qc_summary.json"
    summary = {
        "run_id": run_id,
        "config_hash": config.config_hash,
        "num_shards": int(num_shards),
        "unique_stars": int(len(all_stars)),
        "overlap_count": int(len(overlaps)),
        "missing_count": int(len(missing)),
        "unexpected_count": int(len(unexpected)),
        "config_hashes_identical": True,
        "partition": partition,
        "written": written,
    }
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n")
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    config = benchmark_profile(args.profile)
    if args.manifest_path is not None:
        config = replace(config, manifest_path=args.manifest_path)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if args.target_limit is not None:
        config = replace(config, target_limit=args.target_limit)
    if args.active_combinations:
        config = replace(config, active_combinations=parse_pipeline_specs(args.active_combinations.split(",")))
    n_null_trials = (
        int(args.n_null_trials_per_star)
        if args.n_null_trials_per_star is not None
        else int(config.n_null_trials_per_star)
    )
    config = replace(config, n_null_trials_per_star=n_null_trials)

    validate_shard_request(
        args.num_shards,
        args.shard_id,
        allow_all_shards=bool(args.verify_shards or args.merge_shards),
    )
    run_id = f"{config.profile}_{config.config_hash}"
    mode = "null_calibration" if args.calibrate_fap else "benchmark"
    full_manifest = load_manifest(config.manifest_path, config.target_limit, config)
    if config.strict_target_count and len(full_manifest) != config.target_limit:
        raise ValueError(f"Expected {config.target_limit} manifest rows, found {len(full_manifest)}.")
    partition = verify_shard_partition(full_manifest, num_shards=args.num_shards)

    if args.verify_shards:
        print(json.dumps(json_ready(partition), indent=2, sort_keys=True))
        return 0
    if args.merge_shards:
        if int(args.num_shards) < 2:
            raise ValueError("--merge-shards requires --num-shards greater than 1.")
        summary = merge_shard_outputs(config.output_dir, config, run_id, full_manifest, num_shards=args.num_shards)
        print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
        return 0

    selected_manifest = full_manifest
    shard_rows = None
    base_output_dir = config.output_dir
    if args.shard_id is not None:
        shard_rows = shard_slices(len(full_manifest), args.num_shards)[int(args.shard_id) - 1]
        selected_manifest = shard_manifest(full_manifest, num_shards=args.num_shards, shard_id=args.shard_id)
        config = replace(config, output_dir=shard_output_dir(base_output_dir, args.shard_id))

    runner = UnifiedPipelineRunner(config)
    cases = runner.default_injection_cases()

    if args.dry_run:
        print(f"profile={config.profile}")
        print(f"mode={mode}")
        print(f"manifest={config.manifest_path}")
        print(f"base_output_dir={base_output_dir}")
        print(f"output_dir={config.output_dir}")
        print(f"config_hash={config.config_hash}")
        print(f"run_id={run_id}")
        print(f"target_limit={config.target_limit}")
        print(f"full_target_count={len(full_manifest)}")
        print(f"selected_target_count={len(selected_manifest)}")
        if args.shard_id is not None:
            print(f"num_shards={int(args.num_shards)}")
            print(f"shard_id={int(args.shard_id)}")
            print(f"shard_name={shard_name(args.shard_id)}")
            print(f"shard_rows_1based={shard_rows[0] + 1}-{shard_rows[1]}")
        print(f"n_null_trials_per_star={n_null_trials}")
        print("active_combinations=" + ",".join(spec.pipeline_id for spec in config.active_combinations))
        return 0

    if args.qc_only:
        qc_mode = "null_calibration" if args.calibrate_fap else "complete"
        summary = qc_output(config.output_dir, config, run_id, selected_manifest, mode=qc_mode)
        print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
        return 0

    store = LongTableStore(config.output_dir)
    write_shard_metadata(
        config,
        run_id,
        config.output_dir,
        mode=mode,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        shard_rows=shard_rows,
    )
    write_manifest_used(config.output_dir, selected_manifest)
    injection_ids = tuple(case.injection_id for case in cases)
    thresholds = load_thresholds(args.thresholds_path, config) if args.thresholds_path is not None else None
    for _, row in selected_manifest.iterrows():
        target_id = str(row["target_id"])
        quarter = int(row["quarter"])
        sid = star_id(target_id, quarter)
        if args.calibrate_fap and args.resume and null_calibration_completed(
            store,
            config,
            run_id,
            sid,
            n_null_trials,
        ):
            print(f"skip completed null calibration {sid}")
            continue
        if not args.calibrate_fap and args.resume and star_completed(store, config, run_id, sid, injection_ids):
            print(f"skip completed {sid}")
            continue
        raw = load_kepler_pdcsap(target_id, quarter).to_dataframe()
        regular, _summary = preprocess_pdcsap_light_curve(
            raw,
            quality_policy=config.quality_policy,
            require_finite_flux_error=config.require_finite_flux_error,
        )
        native = LightCurve.from_regularized_frame(
            regular,
            metadata={"target_id": target_id, "quarter": quarter},
        )
        if args.calibrate_fap:
            null_scores = runner.run_null_scores(
                run_id=run_id,
                star_id=sid,
                native=native,
                n_trials=n_null_trials,
            )
            store.append_rows("null_score", null_scores.to_dict(orient="records"))
            print(f"completed null calibration {sid}")
            continue
        result = runner.run_lightcurve(
            run_id=run_id,
            star_id=sid,
            target_id=target_id,
            quarter=quarter,
            native=native,
            injection_cases=cases,
            thresholds=thresholds,
        )
        store.append_rows("characterization", result.characterization.to_dict(orient="records"))
        store.append_rows("injection", result.injection.to_dict(orient="records"))
        store.append_rows("preservation", result.preservation.to_dict(orient="records"))
        store.append_rows("detection", result.detection.to_dict(orient="records"))
        print(f"completed {sid}")
    if args.calibrate_fap:
        null_scores = store.read("null_score")
        null_scores = null_scores[
            null_scores["run_id"].astype(str).eq(run_id)
            & null_scores["config_hash"].astype(str).eq(config.config_hash)
        ]
        thresholds = threshold_table_from_null_scores(null_scores, fap_level=config.fap_level)
        threshold_path = config.output_dir / "fap_thresholds.csv"
        thresholds.to_csv(threshold_path, index=False)
        print(f"wrote {threshold_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
