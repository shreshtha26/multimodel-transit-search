#!/usr/bin/env python
"""Run the unified long-format adaptive-transit benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
import json
from multiprocessing import get_context
import os
from pathlib import Path
from queue import Empty
import sys
from typing import Iterable

SCIENCE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _variable in SCIENCE_THREAD_ENV_VARS:
    os.environ.setdefault(_variable, "1")

import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adaptive_transit.config import (  # noqa: E402
    ACTIVE_SCIENTIFIC_BENCHMARKS,
    benchmark_profile,
    parse_pipeline_specs,
)
from adaptive_transit.core import LightCurve  # noqa: E402
from adaptive_transit.data.kepler_io import (  # noqa: E402
    DEFAULT_LIGHT_CURVE_CACHE_DIR,
    DEFAULT_MAST_BACKOFF_FACTOR,
    DEFAULT_MAST_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_MAST_INITIAL_WAIT_SECONDS,
    DEFAULT_MAST_MAX_ATTEMPTS,
    DEFAULT_MAST_READ_TIMEOUT_SECONDS,
    KeplerFetchPolicy,
    load_cached_kepler_pdcsap_frame,
)
from adaptive_transit.detectors import DETECTORS  # noqa: E402
from adaptive_transit.fap import threshold_table_from_null_scores  # noqa: E402
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve  # noqa: E402
from adaptive_transit.progress import LiveBenchmarkStore  # noqa: E402
from adaptive_transit.resume import PRIMARY_KEYS  # noqa: E402
from adaptive_transit.runner import UnifiedPipelineRunner  # noqa: E402
from adaptive_transit.schemas import LONG_TABLE_SCHEMAS, json_ready  # noqa: E402

THRESHOLD_KEYS = ("run_id", "config_hash", "star_id", "treatment", "detector", "score_definition", "fap_level")
SHARD_WIDTH = 2
OUTPUT_TABLES = (
    "characterization",
    "treatment",
    "injection",
    "preservation",
    "detection",
    "null_score",
    "fap_thresholds",
    "run_status",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run benchmark100 or benchmark1000 with the unified adaptive-transit runner."
    )
    parser.add_argument("--profile", choices=(*ACTIVE_SCIENTIFIC_BENCHMARKS, "demo50", "smoke"), default="benchmark100")
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-limit", type=int)
    parser.add_argument("--active-combinations", type=str, help="Comma-separated treatment_detector entries.")
    parser.add_argument("--injection-period-grid", type=parse_float_grid, help="Comma-separated BATMAN periods in days.")
    parser.add_argument("--injection-duration-hours-grid", type=parse_float_grid, help="Comma-separated BATMAN durations in hours.")
    parser.add_argument("--injection-depth-grid", type=parse_float_grid, help="Comma-separated BATMAN depths in relative flux, e.g. 0.0002,0.0005,0.001.")
    parser.add_argument("--epoch-phase-fraction-grid", type=parse_phase_grid, help="Comma-separated phase fractions in [0, 1).")
    parser.add_argument("--thresholds-path", type=Path, help="Optional common-FAP threshold CSV for calibrated detections.")
    parser.add_argument("--calibrate-fap", action="store_true", help="Run moving-block null calibration instead of injections.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_LIGHT_CURVE_CACHE_DIR)
    parser.add_argument("--no-download", dest="allow_download", action="store_false", default=True)
    parser.add_argument("--download-connect-timeout-seconds", type=float, default=DEFAULT_MAST_CONNECT_TIMEOUT_SECONDS)
    parser.add_argument("--download-read-timeout-seconds", type=float, default=DEFAULT_MAST_READ_TIMEOUT_SECONDS)
    parser.add_argument("--download-max-attempts", type=int, default=DEFAULT_MAST_MAX_ATTEMPTS)
    parser.add_argument("--download-initial-wait-seconds", type=float, default=DEFAULT_MAST_INITIAL_WAIT_SECONDS)
    parser.add_argument("--download-backoff-factor", type=float, default=DEFAULT_MAST_BACKOFF_FACTOR)
    parser.add_argument("--fail-fast-data-fetch", action="store_true")
    parser.add_argument("--retry-fetch-failures", action="store_true")
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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum independent-star worker processes. Use 1 for serial execution.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    return parser.parse_args(argv)


def parse_float_grid(value):
    values = tuple(float(part.strip()) for part in str(value).split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Grid values must be positive comma-separated floats.")
    return values


def parse_phase_grid(value):
    values = tuple(float(part.strip()) for part in str(value).split(",") if part.strip())
    if not values or any(item < 0.0 or item >= 1.0 for item in values):
        raise argparse.ArgumentTypeError("Phase fractions must be comma-separated floats in [0, 1).")
    return values


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


def star_completed(store, config, run_id: str, star: str, injection_ids: tuple[str, ...]) -> bool:
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
    store,
    config,
    run_id: str,
    star: str,
    n_trials: int,
) -> bool:
    observed = store.completed_keys("null_score", config_hash=config.config_hash)
    expected = expected_null_score_keys(config, run_id, star, n_trials)
    return expected.issubset(observed)


def preserve_incompatible_run_metadata(path: Path, config_hash: str) -> None:
    if not path.exists():
        return
    try:
        existing_text = path.read_text()
        existing = json.loads(existing_text)
    except Exception:
        return
    old_hash = str(existing.get("config_hash", ""))
    if not old_hash or old_hash == str(config_hash):
        return
    archive = path.with_name(f"run_metadata_{old_hash}.json")
    if not archive.exists():
        archive.write_text(existing_text)


def write_run_metadata(config, run_id: str, output_dir: Path, *, mode: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_metadata.json"
    preserve_incompatible_run_metadata(path, config.config_hash)
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
    preserve_incompatible_run_metadata(path, config.config_hash)
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


class BenchmarkProgressDisplay:
    """Persistent status lines underneath the outer star progress bar."""

    def __init__(self, *, mode: str) -> None:
        self.mode = str(mode)
        self.lines: list[tqdm] = []
        if self.mode == "null_calibration":
            self.star_line = self._line("Star: -/-", position=1)
            self.null_line = self._line("Null trial: -/-", position=2)
        else:
            self.star_line = self._line("Current star: -", position=1)
            self.injection_line = self._line("Injection: -/-", position=2)
            self.treatment_line = self._line("Treatment: -", position=3)
            self.detector_line = self._line("Detector: -", position=4)

    def _line(self, text: str, *, position: int):
        line = tqdm(total=0, bar_format="{desc}", position=position, leave=False)
        line.set_description_str(text, refresh=True)
        self.lines.append(line)
        return line

    def set_star(self, *, star_id: str, star_index: int, star_total: int) -> None:
        if self.mode == "null_calibration":
            self.star_line.set_description_str(f"Star: {int(star_index)}/{int(star_total)}", refresh=True)
            self.null_line.set_description_str("Null trial: -/-", refresh=True)
        else:
            self.star_line.set_description_str(f"Current star: {star_id}", refresh=True)
            self.injection_line.set_description_str("Injection: -/-", refresh=True)
            self.treatment_line.set_description_str("Treatment: -", refresh=True)
            self.detector_line.set_description_str("Detector: -", refresh=True)

    def update_fetch(self, event: dict) -> None:
        status = str(event.get("status", ""))
        attempt = event.get("attempt")
        max_attempts = event.get("max_attempts")
        if self.mode == "null_calibration":
            if attempt is None:
                self.null_line.set_description_str(f"Null trial: data_fetch {status}", refresh=True)
            else:
                self.null_line.set_description_str(
                    f"Null trial: data_fetch {attempt}/{max_attempts} {status}",
                    refresh=True,
                )
            return
        self.injection_line.set_description_str("Injection: data_fetch", refresh=True)
        self.treatment_line.set_description_str("Treatment: data_fetch", refresh=True)
        if attempt is None:
            self.detector_line.set_description_str(f"Detector: {status}", refresh=True)
        else:
            self.detector_line.set_description_str(f"Detector: MAST {attempt}/{max_attempts} {status}", refresh=True)

    def update_runner(self, event: dict) -> None:
        stage = str(event.get("stage", ""))
        if stage == "null_trial" and self.mode == "null_calibration":
            self.null_line.set_description_str(
                f"Null trial: {int(event['trial_index'])}/{int(event['trial_total'])}",
                refresh=True,
            )
        elif stage == "detection" and self.mode != "null_calibration":
            self.injection_line.set_description_str(
                f"Injection: {int(event['injection_index'])}/{int(event['injection_total'])}",
                refresh=True,
            )
            self.treatment_line.set_description_str(f"Treatment: {event['treatment']}", refresh=True)
            self.detector_line.set_description_str(f"Detector: {event['detector']}", refresh=True)

    def close(self) -> None:
        for line in reversed(self.lines):
            line.close()


def current_rows(frame: pd.DataFrame, *, run_id: str, config_hash: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "run_id" not in frame.columns or "config_hash" not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame[
        frame["run_id"].astype(str).eq(str(run_id))
        & frame["config_hash"].astype(str).eq(str(config_hash))
    ].copy()


def data_fetch_failed(store, *, run_id: str, config_hash: str, star: str) -> bool:
    frame = store.read("run_status")
    if frame.empty:
        return False
    required = {"run_id", "config_hash", "star_id", "stage", "status"}
    if not required.issubset(frame.columns):
        return False
    current = frame[
        frame["run_id"].astype(str).eq(str(run_id))
        & frame["config_hash"].astype(str).eq(str(config_hash))
        & frame["star_id"].astype(str).eq(str(star))
        & frame["stage"].astype(str).eq("data_fetch")
    ]
    if current.empty:
        return False
    return current["status"].astype(str).eq("failed").any()


def record_data_fetch_event(store, *, config, run_id: str, star: str, event: dict) -> None:
    error = str(event.get("error", ""))
    if event.get("wait_seconds") is not None and error:
        error = f"{error}; wait_seconds={float(event['wait_seconds']):.1f}"
    store.record_status(
        run_id=run_id,
        config_hash=config.config_hash,
        star_id=star,
        stage="data_fetch",
        status=str(event.get("status", "")),
        error=error,
    )


_THREADPOOL_LIMITER = None


def cap_science_threads_per_worker(max_threads: int = 1) -> None:
    """Bound BLAS/OpenMP-style pools inside each process worker."""

    global _THREADPOOL_LIMITER
    threads = str(max(1, int(max_threads)))
    for variable in SCIENCE_THREAD_ENV_VARS:
        os.environ[variable] = threads
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return
    _THREADPOOL_LIMITER = threadpool_limits(limits=int(threads))


def resolve_star_worker_count(max_workers: int | None, task_count: int) -> int:
    if int(task_count) <= 0:
        return 0
    requested = 1 if max_workers is None else int(max_workers)
    if requested < 1:
        raise ValueError("--max-workers must be at least 1.")
    available = os.cpu_count() or 1
    return max(1, min(requested, available, int(task_count)))


def fetch_policy_to_payload(policy: KeplerFetchPolicy) -> dict:
    return {
        "connect_timeout_seconds": float(policy.connect_timeout_seconds),
        "read_timeout_seconds": float(policy.read_timeout_seconds),
        "max_attempts": int(policy.max_attempts),
        "initial_wait_seconds": float(policy.initial_wait_seconds),
        "backoff_factor": float(policy.backoff_factor),
    }


def fetch_policy_from_payload(payload: dict) -> KeplerFetchPolicy:
    policy = KeplerFetchPolicy(**dict(payload))
    policy.validate()
    return policy


def star_work_output_dir(base_output_dir: Path, run_id: str, star: str) -> Path:
    return Path(base_output_dir) / "_star_work" / str(run_id) / str(star)


def filter_current_star_rows(
    frame: pd.DataFrame,
    *,
    run_id: str,
    config_hash: str,
    star: str,
) -> pd.DataFrame:
    current = current_rows(frame, run_id=run_id, config_hash=config_hash)
    if current.empty or "star_id" not in current.columns:
        return current.iloc[0:0].copy()
    return current[current["star_id"].astype(str).eq(str(star))].copy()


def import_star_rows_from_root(
    store,
    root: Path,
    *,
    run_id: str,
    config_hash: str,
    star: str,
    table_names: Iterable[str] = OUTPUT_TABLES,
) -> dict[str, int]:
    imported = {}
    for table_name in table_names:
        path = Path(root) / f"{table_name}.csv"
        if not path.exists():
            continue
        frame = filter_current_star_rows(
            pd.read_csv(path),
            run_id=run_id,
            config_hash=config_hash,
            star=star,
        )
        if frame.empty:
            continue
        store.upsert_rows(table_name, frame.to_dict(orient="records"))
        imported[table_name] = int(len(frame))
    return imported


def seed_star_work_output(
    *,
    main_output_dir: Path,
    work_output_dir: Path,
    run_id: str,
    config_hash: str,
    star: str,
) -> dict[str, int]:
    with LiveBenchmarkStore(work_output_dir) as worker_store:
        worker_store.import_existing_csvs(run_id=run_id, config_hash=config_hash, compatible_only=True)
        imported = import_star_rows_from_root(
            worker_store,
            main_output_dir,
            run_id=run_id,
            config_hash=config_hash,
            star=star,
        )
        worker_store.export_csvs(run_id=run_id, config_hash=config_hash)
        return imported


def merge_star_output_into_main_store(
    store,
    work_output_dir: Path,
    *,
    run_id: str,
    config_hash: str,
    star: str,
) -> dict[str, int]:
    return import_star_rows_from_root(
        store,
        work_output_dir,
        run_id=run_id,
        config_hash=config_hash,
        star=star,
    )


def write_calibration_thresholds_from_store(store, config, run_id: str) -> Path | None:
    null_scores = current_rows(store.read("null_score"), run_id=run_id, config_hash=config.config_hash)
    if null_scores.empty:
        return config.output_dir / "fap_thresholds.csv"
    thresholds = threshold_table_from_null_scores(null_scores, fap_level=config.fap_level)
    store.upsert_rows("fap_thresholds", thresholds.to_dict(orient="records"))
    written = store.export_csvs(run_id=run_id, config_hash=config.config_hash)
    return written.get("fap_thresholds", config.output_dir / "fap_thresholds.csv")


def emit_worker_progress(progress_queue, event: dict) -> None:
    if progress_queue is None:
        return
    try:
        progress_queue.put(json_ready(event))
    except Exception:
        return


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
        "treatment": len(stars) * treatment_count,
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
    tables = {name: [] for name in ("characterization", "treatment", "injection", "preservation", "detection", "null_score")}
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


def build_star_task_payload(
    *,
    row: dict,
    config,
    run_id: str,
    mode: str,
    n_null_trials: int,
    cache_dir: Path,
    allow_download: bool,
    fetch_policy: KeplerFetchPolicy,
    resume: bool,
    retry_fetch_failures: bool,
    fail_fast_data_fetch: bool,
    thresholds_path: Path | None,
    shard_id: int | None,
    num_shards: int,
    shard_rows: tuple[int, int] | None,
) -> dict:
    sid = star_id(row["target_id"], row["quarter"])
    work_dir = star_work_output_dir(config.output_dir, run_id, sid)
    return {
        "row": dict(row),
        "star_id": sid,
        "config": config,
        "run_id": run_id,
        "mode": mode,
        "n_null_trials": int(n_null_trials),
        "cache_dir": str(cache_dir),
        "allow_download": bool(allow_download),
        "fetch_policy": fetch_policy_to_payload(fetch_policy),
        "resume": bool(resume),
        "retry_fetch_failures": bool(retry_fetch_failures),
        "fail_fast_data_fetch": bool(fail_fast_data_fetch),
        "thresholds_path": None if thresholds_path is None else str(thresholds_path),
        "work_output_dir": str(work_dir),
        "shard_id": shard_id,
        "num_shards": int(num_shards),
        "shard_rows": None if shard_rows is None else tuple(shard_rows),
    }


def run_star_task(payload: dict) -> dict:
    cap_science_threads_per_worker(1)
    row = dict(payload["row"])
    target_id = str(row["target_id"])
    quarter = int(row["quarter"])
    sid = str(payload.get("star_id") or star_id(target_id, quarter))
    run_id = str(payload["run_id"])
    mode = str(payload["mode"])
    progress_queue = payload.get("progress_queue")
    work_output_dir = Path(payload["work_output_dir"])
    config = replace(payload["config"], output_dir=work_output_dir)
    fetch_policy = fetch_policy_from_payload(payload["fetch_policy"])

    with LiveBenchmarkStore(config.output_dir) as store:
        try:
            if payload.get("resume", True):
                store.import_existing_csvs(run_id=run_id, config_hash=config.config_hash, compatible_only=True)
            write_shard_metadata(
                config,
                run_id,
                config.output_dir,
                mode=mode,
                shard_id=payload.get("shard_id"),
                num_shards=int(payload.get("num_shards", 1)),
                shard_rows=payload.get("shard_rows"),
            )
            write_manifest_used(config.output_dir, pd.DataFrame([row]))

            runner = UnifiedPipelineRunner(config)
            cases = runner.default_injection_cases()
            injection_ids = tuple(case.injection_id for case in cases)
            n_null_trials = int(payload["n_null_trials"])
            if mode == "null_calibration" and payload.get("resume", True) and null_calibration_completed(
                store,
                config,
                run_id,
                sid,
                n_null_trials,
            ):
                write_calibration_thresholds_from_store(store, config, run_id)
                store.export_csvs(run_id=run_id, config_hash=config.config_hash)
                emit_worker_progress(progress_queue, {"kind": "star", "star_id": sid, "status": "skipped_existing"})
                return {
                    "status": "success",
                    "skipped": True,
                    "star_id": sid,
                    "output_dir": str(config.output_dir),
                    "message": "completed null calibration already present",
                }
            if mode != "null_calibration" and payload.get("resume", True) and star_completed(
                store,
                config,
                run_id,
                sid,
                injection_ids,
            ):
                store.export_csvs(run_id=run_id, config_hash=config.config_hash)
                emit_worker_progress(progress_queue, {"kind": "star", "star_id": sid, "status": "skipped_existing"})
                return {
                    "status": "success",
                    "skipped": True,
                    "star_id": sid,
                    "output_dir": str(config.output_dir),
                    "message": "completed benchmark already present",
                }
            if (
                payload.get("resume", True)
                and not payload.get("retry_fetch_failures", False)
                and data_fetch_failed(store, run_id=run_id, config_hash=config.config_hash, star=sid)
            ):
                store.export_csvs(run_id=run_id, config_hash=config.config_hash)
                emit_worker_progress(progress_queue, {"kind": "fetch", "star_id": sid, "status": "skipped_failed"})
                return {
                    "status": "skipped_failed_fetch",
                    "star_id": sid,
                    "output_dir": str(config.output_dir),
                    "message": "previous data fetch failure",
                }

            def fetch_progress(event):
                record_data_fetch_event(store, config=config, run_id=run_id, star=sid, event=event)
                emit_worker_progress(progress_queue, {"kind": "fetch", "star_id": sid, **dict(event)})

            try:
                raw, cache_hit = load_cached_kepler_pdcsap_frame(
                    target_id,
                    quarter,
                    cache_dir=Path(payload["cache_dir"]),
                    allow_download=bool(payload["allow_download"]),
                    fetch_policy=fetch_policy,
                    progress_callback=fetch_progress,
                )
                store.record_status(
                    run_id=run_id,
                    config_hash=config.config_hash,
                    star_id=sid,
                    stage="data_fetch",
                    status="complete_cached" if cache_hit else "complete_downloaded",
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                store.record_status(
                    run_id=run_id,
                    config_hash=config.config_hash,
                    star_id=sid,
                    stage="data_fetch",
                    status="failed",
                    error=error,
                )
                store.export_csvs(run_id=run_id, config_hash=config.config_hash)
                emit_worker_progress(progress_queue, {"kind": "fetch", "star_id": sid, "status": "failed", "error": error})
                if payload.get("fail_fast_data_fetch", False):
                    raise
                return {
                    "status": "failed",
                    "stage": "data_fetch",
                    "star_id": sid,
                    "output_dir": str(config.output_dir),
                    "error": error,
                }

            regular, _summary = preprocess_pdcsap_light_curve(
                raw,
                quality_policy=config.quality_policy,
                require_finite_flux_error=config.require_finite_flux_error,
            )
            native = LightCurve.from_regularized_frame(
                regular,
                metadata={"target_id": target_id, "quarter": quarter},
            )

            def runner_progress(event):
                stage = str(event.get("stage", ""))
                units = 0
                if mode == "null_calibration" and stage == "null_trial":
                    units = 1
                elif mode != "null_calibration" and stage == "detection":
                    units = 1
                emit_worker_progress(progress_queue, {"kind": "runner", "star_id": sid, "units": units, **dict(event)})

            if mode == "null_calibration":
                runner.run_null_scores(
                    run_id=run_id,
                    star_id=sid,
                    native=native,
                    n_trials=n_null_trials,
                    progress_store=store,
                    progress_callback=runner_progress,
                    show_progress=False,
                )
                write_calibration_thresholds_from_store(store, config, run_id)
                store.record_status(
                    run_id=run_id,
                    config_hash=config.config_hash,
                    star_id=sid,
                    stage="star",
                    status="complete",
                )
                store.export_csvs(run_id=run_id, config_hash=config.config_hash)
                return {
                    "status": "success",
                    "stage": "null_calibration",
                    "star_id": sid,
                    "output_dir": str(config.output_dir),
                }

            thresholds_path = payload.get("thresholds_path")
            thresholds = load_thresholds(Path(thresholds_path), config) if thresholds_path is not None else None
            runner.run_lightcurve(
                run_id=run_id,
                star_id=sid,
                target_id=target_id,
                quarter=quarter,
                native=native,
                injection_cases=cases,
                thresholds=thresholds,
                progress_store=store,
                progress_callback=runner_progress,
                show_progress=False,
            )
            store.record_status(
                run_id=run_id,
                config_hash=config.config_hash,
                star_id=sid,
                stage="star",
                status="complete",
            )
            store.export_csvs(run_id=run_id, config_hash=config.config_hash)
            return {
                "status": "success",
                "stage": "benchmark",
                "star_id": sid,
                "output_dir": str(config.output_dir),
            }
        except Exception as exc:
            store.record_status(
                run_id=run_id,
                config_hash=config.config_hash,
                star_id=sid,
                stage="star",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            store.export_csvs(run_id=run_id, config_hash=config.config_hash)
            raise


def worker_progress_description(event: dict) -> str:
    sid = str(event.get("star_id", ""))
    kind = str(event.get("kind", ""))
    if kind == "fetch":
        status = str(event.get("status", ""))
        attempt = event.get("attempt")
        max_attempts = event.get("max_attempts")
        if attempt is not None:
            return f"{sid} data_fetch {attempt}/{max_attempts} {status}".strip()
        return f"{sid} data_fetch {status}".strip()
    if str(event.get("stage", "")) == "null_trial":
        return f"{sid} null {int(event['trial_index'])}/{int(event['trial_total'])}"
    if str(event.get("stage", "")) == "detection":
        return f"{sid} {event.get('injection_id')} {event.get('treatment')} {event.get('detector')}"
    return f"{sid} {event.get('status', '')}".strip()


def drain_worker_progress_queue(progress_queue, progress) -> None:
    while True:
        try:
            event = progress_queue.get_nowait()
        except Empty:
            break
        except Exception:
            break
        units = max(0, int(event.get("units", 0)))
        if units and progress is not None:
            progress.update(units)
        if progress is not None:
            progress.set_postfix_str(worker_progress_description(event))


def parallel_work_unit_total(mode: str, payloads: list[dict], config, cases, n_null_trials: int) -> tuple[int, str, str]:
    if mode == "null_calibration":
        return len(payloads) * int(n_null_trials), "Null trials", "trial"
    return len(payloads) * len(cases) * len(config.active_combinations), "Pipeline searches", "search"


def run_stars_parallel(
    *,
    store,
    config,
    run_id: str,
    selected_manifest: pd.DataFrame,
    args,
    fetch_policy: KeplerFetchPolicy,
    mode: str,
    n_null_trials: int,
    cases,
    shard_rows: tuple[int, int] | None,
) -> list[dict]:
    injection_ids = tuple(case.injection_id for case in cases)
    payloads: list[dict] = []
    for _, row in selected_manifest.iterrows():
        row_dict = row.to_dict()
        sid = star_id(row_dict["target_id"], row_dict["quarter"])
        if args.calibrate_fap and args.resume and null_calibration_completed(
            store,
            config,
            run_id,
            sid,
            n_null_trials,
        ):
            tqdm.write(f"skip completed null calibration {sid}")
            continue
        if not args.calibrate_fap and args.resume and star_completed(store, config, run_id, sid, injection_ids):
            tqdm.write(f"skip completed {sid}")
            continue
        if (
            args.resume
            and not args.retry_fetch_failures
            and data_fetch_failed(store, run_id=run_id, config_hash=config.config_hash, star=sid)
        ):
            tqdm.write(f"skip failed data fetch {sid}; pass --retry-fetch-failures to retry it")
            continue
        work_dir = star_work_output_dir(config.output_dir, run_id, sid)
        if args.resume:
            seed_star_work_output(
                main_output_dir=config.output_dir,
                work_output_dir=work_dir,
                run_id=run_id,
                config_hash=config.config_hash,
                star=sid,
            )
        payloads.append(
            build_star_task_payload(
                row=row_dict,
                config=config,
                run_id=run_id,
                mode=mode,
                n_null_trials=n_null_trials,
                cache_dir=args.cache_dir,
                allow_download=bool(args.allow_download),
                fetch_policy=fetch_policy,
                resume=bool(args.resume),
                retry_fetch_failures=bool(args.retry_fetch_failures),
                fail_fast_data_fetch=bool(args.fail_fast_data_fetch),
                thresholds_path=args.thresholds_path,
                shard_id=args.shard_id,
                num_shards=args.num_shards,
                shard_rows=shard_rows,
            )
        )
    if not payloads:
        return []

    worker_count = resolve_star_worker_count(args.max_workers, len(payloads))
    tqdm.write(f"running {len(payloads)} pending stars with {worker_count} worker processes")
    context = get_context("spawn")
    results: list[dict] = []
    work_total, work_desc, work_unit = parallel_work_unit_total(mode, payloads, config, cases, n_null_trials)
    with context.Manager() as manager:
        progress_queue = manager.Queue()
        for payload in payloads:
            payload["progress_queue"] = progress_queue
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=cap_science_threads_per_worker,
            initargs=(1,),
        ) as executor:
            future_map = {executor.submit(run_star_task, payload): payload for payload in payloads}
            pending = set(future_map)
            with tqdm(
                total=len(future_map),
                desc="Stars",
                unit="star",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                position=0,
            ) as star_progress, tqdm(
                total=work_total,
                desc=work_desc,
                unit=work_unit,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                position=1,
                leave=False,
            ) as work_progress:
                while pending:
                    done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                    drain_worker_progress_queue(progress_queue, work_progress)
                    for future in done:
                        drain_worker_progress_queue(progress_queue, work_progress)
                        payload = future_map[future]
                        sid = str(payload["star_id"])
                        work_dir = Path(payload["work_output_dir"])
                        try:
                            result = future.result()
                        except Exception as exc:
                            merge_star_output_into_main_store(
                                store,
                                work_dir,
                                run_id=run_id,
                                config_hash=config.config_hash,
                                star=sid,
                            )
                            store.export_csvs(run_id=run_id, config_hash=config.config_hash)
                            for item in pending:
                                item.cancel()
                            raise RuntimeError(f"worker failed for {sid}: {type(exc).__name__}: {exc}") from exc
                        merge_star_output_into_main_store(
                            store,
                            work_dir,
                            run_id=run_id,
                            config_hash=config.config_hash,
                            star=sid,
                        )
                        store.export_csvs(run_id=run_id, config_hash=config.config_hash)
                        results.append(result)
                        status = str(result.get("status", ""))
                        if status == "success" and result.get("skipped"):
                            tqdm.write(f"skip completed {sid}")
                        elif status == "success" and mode == "null_calibration":
                            tqdm.write(f"completed null calibration {sid}")
                        elif status == "success":
                            tqdm.write(f"completed {sid}")
                        elif status == "skipped_failed_fetch":
                            tqdm.write(f"skip failed data fetch {sid}; pass --retry-fetch-failures to retry it")
                        elif status == "failed" and result.get("stage") == "data_fetch":
                            tqdm.write(f"failed data fetch {sid}: {result.get('error', '')}")
                            if args.fail_fast_data_fetch:
                                for item in pending:
                                    item.cancel()
                                raise RuntimeError(f"failed data fetch {sid}: {result.get('error', '')}")
                        else:
                            tqdm.write(f"failed {sid}: {result.get('error', '')}")
                        star_progress.set_postfix_str(f"{status} {sid}")
                        star_progress.update(1)
                drain_worker_progress_queue(progress_queue, work_progress)
    return results


def run_stars_serial(
    *,
    store,
    config,
    run_id: str,
    selected_manifest: pd.DataFrame,
    args,
    fetch_policy: KeplerFetchPolicy,
    mode: str,
    n_null_trials: int,
    cases,
) -> None:
    runner = UnifiedPipelineRunner(config)
    injection_ids = tuple(case.injection_id for case in cases)
    thresholds = load_thresholds(args.thresholds_path, config) if args.thresholds_path is not None else None
    progress_display = BenchmarkProgressDisplay(mode=mode)
    star_iterator = tqdm(
        list(selected_manifest.iterrows()),
        total=len(selected_manifest),
        desc="Stars",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        position=0,
    )
    try:
        for star_index, (_, row) in enumerate(star_iterator, start=1):
            target_id = str(row["target_id"])
            quarter = int(row["quarter"])
            sid = star_id(target_id, quarter)
            progress_display.set_star(star_id=sid, star_index=star_index, star_total=len(selected_manifest))
            if args.calibrate_fap and args.resume and null_calibration_completed(
                store,
                config,
                run_id,
                sid,
                n_null_trials,
            ):
                tqdm.write(f"skip completed null calibration {sid}")
                continue
            if not args.calibrate_fap and args.resume and star_completed(store, config, run_id, sid, injection_ids):
                tqdm.write(f"skip completed {sid}")
                continue
            if (
                args.resume
                and not args.retry_fetch_failures
                and data_fetch_failed(store, run_id=run_id, config_hash=config.config_hash, star=sid)
            ):
                tqdm.write(f"skip failed data fetch {sid}; pass --retry-fetch-failures to retry it")
                continue

            def fetch_progress(event):
                progress_display.update_fetch(event)
                record_data_fetch_event(store, config=config, run_id=run_id, star=sid, event=event)

            try:
                raw, cache_hit = load_cached_kepler_pdcsap_frame(
                    target_id,
                    quarter,
                    cache_dir=args.cache_dir,
                    allow_download=bool(args.allow_download),
                    fetch_policy=fetch_policy,
                    progress_callback=fetch_progress,
                )
                store.record_status(
                    run_id=run_id,
                    config_hash=config.config_hash,
                    star_id=sid,
                    stage="data_fetch",
                    status="complete_cached" if cache_hit else "complete_downloaded",
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                store.record_status(
                    run_id=run_id,
                    config_hash=config.config_hash,
                    star_id=sid,
                    stage="data_fetch",
                    status="failed",
                    error=error,
                )
                store.export_csvs(run_id=run_id, config_hash=config.config_hash)
                tqdm.write(f"failed data fetch {sid}: {error}")
                if args.fail_fast_data_fetch:
                    raise
                continue

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
                runner.run_null_scores(
                    run_id=run_id,
                    star_id=sid,
                    native=native,
                    n_trials=n_null_trials,
                    progress_store=store,
                    progress_callback=progress_display.update_runner,
                    show_progress=False,
                )
                store.export_csvs(run_id=run_id, config_hash=config.config_hash)
                tqdm.write(f"completed null calibration {sid}")
                continue
            runner.run_lightcurve(
                run_id=run_id,
                star_id=sid,
                target_id=target_id,
                quarter=quarter,
                native=native,
                injection_cases=cases,
                thresholds=thresholds,
                progress_store=store,
                progress_callback=progress_display.update_runner,
                show_progress=False,
            )
            store.export_csvs(run_id=run_id, config_hash=config.config_hash)
            tqdm.write(f"completed {sid}")
    finally:
        star_iterator.close()
        progress_display.close()


def main(argv=None) -> int:
    args = parse_args(argv)
    if int(args.max_workers) < 1:
        raise ValueError("--max-workers must be at least 1.")
    args.cache_dir = args.cache_dir.expanduser().resolve()
    fetch_policy = KeplerFetchPolicy(
        connect_timeout_seconds=float(args.download_connect_timeout_seconds),
        read_timeout_seconds=float(args.download_read_timeout_seconds),
        max_attempts=int(args.download_max_attempts),
        initial_wait_seconds=float(args.download_initial_wait_seconds),
        backoff_factor=float(args.download_backoff_factor),
    )
    fetch_policy.validate()
    config = benchmark_profile(args.profile)
    if args.manifest_path is not None:
        config = replace(config, manifest_path=args.manifest_path)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if args.target_limit is not None:
        config = replace(config, target_limit=args.target_limit)
    if args.active_combinations:
        config = replace(config, active_combinations=parse_pipeline_specs(args.active_combinations.split(",")))
    if args.injection_period_grid is not None:
        config = replace(config, injection_period_grid=args.injection_period_grid)
    if args.injection_duration_hours_grid is not None:
        config = replace(config, injection_duration_hours_grid=args.injection_duration_hours_grid)
    if args.injection_depth_grid is not None:
        config = replace(config, injection_depth_grid=args.injection_depth_grid)
    if args.epoch_phase_fraction_grid is not None:
        config = replace(config, epoch_phase_fraction_grid=args.epoch_phase_fraction_grid)
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
        print(f"cache_dir={args.cache_dir}")
        print(f"allow_download={bool(args.allow_download)}")
        print(f"download_connect_timeout_seconds={fetch_policy.connect_timeout_seconds}")
        print(f"download_read_timeout_seconds={fetch_policy.read_timeout_seconds}")
        print(f"download_max_attempts={fetch_policy.max_attempts}")
        print(f"download_initial_wait_seconds={fetch_policy.initial_wait_seconds}")
        print(f"download_backoff_factor={fetch_policy.backoff_factor}")
        print(f"target_limit={config.target_limit}")
        print(f"full_target_count={len(full_manifest)}")
        print(f"selected_target_count={len(selected_manifest)}")
        print(f"max_workers={int(args.max_workers)}")
        print(f"resolved_worker_count={resolve_star_worker_count(args.max_workers, len(selected_manifest))}")
        if args.shard_id is not None:
            print(f"num_shards={int(args.num_shards)}")
            print(f"shard_id={int(args.shard_id)}")
            print(f"shard_name={shard_name(args.shard_id)}")
            print(f"shard_rows_1based={shard_rows[0] + 1}-{shard_rows[1]}")
        print(f"n_null_trials_per_star={n_null_trials}")
        print("active_combinations=" + ",".join(spec.pipeline_id for spec in config.active_combinations))
        print("injection_period_grid=" + ",".join(str(value) for value in config.injection_period_grid))
        print("injection_duration_hours_grid=" + ",".join(str(value) for value in config.injection_duration_hours_grid))
        print("injection_depth_grid=" + ",".join(str(value) for value in config.injection_depth_grid))
        print("epoch_phase_fraction_grid=" + ",".join(str(value) for value in config.epoch_phase_fraction_grid))
        return 0

    if args.qc_only:
        qc_mode = "null_calibration" if args.calibrate_fap else "complete"
        summary = qc_output(config.output_dir, config, run_id, selected_manifest, mode=qc_mode)
        print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
        return 0

    store = LiveBenchmarkStore(config.output_dir)
    try:
        if args.resume:
            store.import_existing_csvs(run_id=run_id, config_hash=config.config_hash, compatible_only=True)
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
        worker_count = resolve_star_worker_count(args.max_workers, len(selected_manifest))
        if worker_count > 1:
            run_stars_parallel(
                store=store,
                config=config,
                run_id=run_id,
                selected_manifest=selected_manifest,
                args=args,
                fetch_policy=fetch_policy,
                mode=mode,
                n_null_trials=n_null_trials,
                cases=cases,
                shard_rows=shard_rows,
            )
        else:
            run_stars_serial(
                store=store,
                config=config,
                run_id=run_id,
                selected_manifest=selected_manifest,
                args=args,
                fetch_policy=fetch_policy,
                mode=mode,
                n_null_trials=n_null_trials,
                cases=cases,
            )
        if args.calibrate_fap:
            threshold_path = write_calibration_thresholds_from_store(store, config, run_id)
            tqdm.write(f"wrote {threshold_path}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
