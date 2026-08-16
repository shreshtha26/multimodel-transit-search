"""Build the final 1,000-star Kepler Q5 stellar-background characterization.

This script deliberately DOES NOT redefine stellar-characterization science.
The v2 scientific definitions were frozen during the 100-star development run.
This population run:

    reference 100 already characterized
        + 900 new independently selected clean Q5 stars
        = 1,000-star final characterization population

Scientific contract
-------------------
1. The seven scientific domains and eleven canonical variables come directly
   from ``adaptive_transit.noise_models.stellar_variability``. This script does
   not change their definitions.
2. The original 100-star development cohort is retained exactly and is joined
   to 900 newly characterized stars.
3. The 900 expansion targets are selected independently of all light-curve
   characterization variables/labels. Only target identity, upstream catalog
   cleanliness/vetoes, Quarter-5 availability and successful data retrieval may
   influence inclusion.
4. Population-relative amplitude/memory/evolution boundaries are recomputed
   ONCE on the final 1,000 successful stars. These 1,000-star values become the
   final v2 population reference boundaries.
5. Human-readable behaviour/review labels are descriptive outputs. They are not
   astrophysical classifications and are not intended to replace the canonical
   continuous variables in downstream machine-learning experiments.
6. The run is checkpointed. Interrupted expansion work can be resumed without
   changing target ordering.

Typical use
-----------
PYTHONPATH=src .venv/bin/python scripts/run_characterization_population1000.py \
  --candidate-pool outputs/target_selection/kepler_catalog_clean_pool.csv \
  --trust-clean-pool \
  --allow-download \
  --max-workers 4

The larger ``kepler_catalog_clean_pool.csv`` should be used here, NOT the old
250-candidate file, because the final cohort requires at least 900 additional
Q5-eligible clean targets beyond the frozen reference 100.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Reuse the exact data-loading/preprocessing/characterization path already
# validated by the 100-star runner. Executing this file from ``scripts/`` puts
# that directory on sys.path, so the import is stable without making scripts a
# package.
import run_characterization_100star as base

from adaptive_transit.noise_models.stellar_variability import (
    CANONICAL_CHARACTERIZATION_COLUMNS,
    CANONICAL_CHARACTERIZATION_SCHEMA,
    CANONICAL_CONTINUOUS_FEATURE_COLUMNS,
    DEFAULT_BOUNDARIES,
    DOMINANT_STATISTICAL_BEHAVIOUR_ORDER,
    V2_FREEZE_ID,
    apply_population_variability_boundaries,
    assign_dominant_statistical_behaviour,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_SAMPLE_SIZE = 1000
REFERENCE_SAMPLE_SIZE = 100
EXPANSION_SUCCESS_COUNT = FINAL_SAMPLE_SIZE - REFERENCE_SAMPLE_SIZE
QUARTER = 5

# This seed is intentionally distinct from the 100-star development selector.
# The ordered list is frozen before any expansion target is characterized.
EXPANSION_SELECTION_SEED = 2026081601

DEFAULT_CANDIDATE_POOL = (
    PROJECT_ROOT / "outputs/target_selection/kepler_catalog_clean_pool.csv"
)
DEFAULT_REFERENCE_DIR = (
    PROJECT_ROOT / "outputs/experiments/characterization_validation100/metrics"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/experiments/characterization_population1000"
)
DEFAULT_CACHE_DIR = PROJECT_ROOT / "outputs/cache/kepler_light_curves"
DEFAULT_MANIFEST_OUT = (
    PROJECT_ROOT / "configs/kepler_q5_clean_1000_star_manifest.csv"
)
DEFAULT_BOUNDARY_OUT = (
    PROJECT_ROOT / "configs/stellar_characterization_v2_population_boundaries_1000.json"
)

REFERENCE_MASTER_NAME = "characterization_master_100.csv"
REFERENCE_FREEZE_NAME = "characterization_v2_freeze.json"
REFERENCE_BOUNDARIES_NAME = "population_boundaries_100.json"
REFERENCE_BEHAVIOUR_COUNTS_NAME = "dominant_statistical_behaviour_counts.csv"

CHECKPOINT_SUCCESS_NAME = "expansion_checkpoint_successes.parquet"
CHECKPOINT_FAILURE_NAME = "expansion_checkpoint_failures.csv"
CHECKPOINT_STATE_NAME = "expansion_checkpoint_state.json"

# Structural missingness is expected here because this diagnostic exists only
# when both LS and ACF produce usable period candidates.
CONDITIONAL_CANONICAL_FEATURES = {
    "v2_ls_acf_period_relative_error",
}

REVIEW_FLAG_COLUMNS = base.REVIEW_FLAG_COLUMNS
CANDIDATE_FLAG_COLUMNS = base.CANDIDATE_FLAG_COLUMNS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def core_module_path() -> Path:
    return (
        PROJECT_ROOT
        / "src/adaptive_transit/noise_models/stellar_variability.py"
    )


def canonical_schema_hash() -> str:
    payload = json.dumps(
        [list(item) for item in CANONICAL_CHARACTERIZATION_SCHEMA],
        sort_keys=True,
    )
    return sha256_text(payload)


def normalize_target_series(values: pd.Series) -> pd.Series:
    return values.map(base.normalize_target_id).astype(str)


def load_reference_100(reference_dir: Path) -> tuple[pd.DataFrame, dict]:
    master_path = reference_dir / REFERENCE_MASTER_NAME
    freeze_path = reference_dir / REFERENCE_FREEZE_NAME

    if not master_path.exists():
        raise FileNotFoundError(
            f"Reference 100-star master file not found: {master_path}"
        )
    if not freeze_path.exists():
        raise FileNotFoundError(
            f"Reference 100-star freeze file not found: {freeze_path}"
        )

    reference = pd.read_csv(master_path, dtype={"target_id": str})
    reference["target_id"] = normalize_target_series(reference["target_id"])

    if "quarter" not in reference.columns:
        reference["quarter"] = QUARTER

    reference = (
        reference.sort_values(
            "selection_order"
            if "selection_order" in reference.columns
            else "target_id"
        )
        .drop_duplicates(["target_id", "quarter"], keep="first")
        .reset_index(drop=True)
    )

    if len(reference) != REFERENCE_SAMPLE_SIZE:
        raise ValueError(
            f"Expected exactly {REFERENCE_SAMPLE_SIZE} unique reference stars; "
            f"found {len(reference)}."
        )
    if reference["target_id"].nunique() != REFERENCE_SAMPLE_SIZE:
        raise ValueError("Reference cohort contains duplicate target IDs.")

    freeze = json.loads(freeze_path.read_text())
    freeze_id = freeze.get("freeze_id")
    if freeze_id != V2_FREEZE_ID:
        raise ValueError(
            f"Reference freeze_id={freeze_id!r} does not match current "
            f"V2_FREEZE_ID={V2_FREEZE_ID!r}. Do not mix characterization versions."
        )

    missing_core = [
        column
        for column in CANONICAL_CHARACTERIZATION_COLUMNS
        if column not in reference.columns
    ]
    if missing_core:
        raise ValueError(
            "Reference master does not contain the frozen canonical inputs: "
            f"{missing_core}"
        )

    # The original 100 population labels were derived using 100-star quantiles.
    # They are intentionally NOT treated as final here; all population-relative
    # labels are recomputed from scratch after the full 1,000 stars are fixed.
    reference = reference.copy()
    reference["selection_order"] = np.arange(
        1, REFERENCE_SAMPLE_SIZE + 1, dtype=int
    )
    reference["selection_group"] = "development_reference_100"
    reference["population_cohort"] = "development_reference_100"

    return reference, freeze


def prepare_expansion_pool(
    candidate_pool: Path,
    reference_target_ids: set[str],
    *,
    trust_clean_pool: bool,
) -> tuple[pd.DataFrame, dict]:
    # Reuse the same upstream-clean validation logic as the successful 100-star
    # runner. Statistical columns, if present, are ignored by that function.
    pool, provenance = base.prepare_clean_q5_pool(
        candidate_pool,
        trust_clean_pool=trust_clean_pool,
    )

    pool["target_id"] = normalize_target_series(pool["target_id"])
    pool = pool[~pool["target_id"].isin(reference_target_ids)].copy()
    pool = pool.drop_duplicates(["target_id", "quarter"]).reset_index(drop=True)

    if len(pool) < EXPANSION_SUCCESS_COUNT:
        raise ValueError(
            f"After excluding the reference 100, only {len(pool)} candidate "
            f"targets remain. Need at least {EXPANSION_SUCCESS_COUNT}. "
            "Use the larger outputs/target_selection/kepler_catalog_clean_pool.csv "
            "rather than the 250-target candidate file."
        )

    # Re-randomize the expansion list with its own fixed seed. No statistical
    # feature is evaluated before this ordering is frozen.
    pool = pool.sample(
        frac=1.0,
        random_state=EXPANSION_SELECTION_SEED,
    ).reset_index(drop=True)

    pool["selection_order"] = np.arange(
        REFERENCE_SAMPLE_SIZE + 1,
        REFERENCE_SAMPLE_SIZE + 1 + len(pool),
        dtype=int,
    )
    pool["selection_group"] = "random_clean_q5_population_expansion"
    pool["population_cohort"] = "population_expansion_900"

    provenance = dict(provenance)
    provenance.update(
        {
            "population_target": FINAL_SAMPLE_SIZE,
            "reference_target_count": REFERENCE_SAMPLE_SIZE,
            "required_expansion_successes": EXPANSION_SUCCESS_COUNT,
            "expansion_selection_seed": EXPANSION_SELECTION_SEED,
            "expansion_candidate_count_after_reference_exclusion": int(len(pool)),
            "statistical_selection_columns_used_for_expansion": [],
            "expansion_selection_policy": (
                "Reference 100 are retained. Remaining clean Q5 targets are put "
                "into one deterministic random order before characterization. "
                "Only data-availability/characterization success may cause a "
                "target to be skipped in favor of the next pre-ordered target."
            ),
        }
    )
    return pool, provenance


def checkpoint_paths(output_dir: Path) -> dict[str, Path]:
    metrics = output_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    return {
        "success": metrics / CHECKPOINT_SUCCESS_NAME,
        "failure": metrics / CHECKPOINT_FAILURE_NAME,
        "state": metrics / CHECKPOINT_STATE_NAME,
    }


def selection_signature(
    candidate_pool: Path,
    reference_target_ids: set[str],
) -> dict[str, object]:
    return {
        "candidate_pool_path": str(candidate_pool.resolve()),
        "candidate_pool_sha256": sha256_file(candidate_pool),
        "reference_target_hash": sha256_text(
            "\n".join(sorted(reference_target_ids))
        ),
        "reference_target_count": len(reference_target_ids),
        "expansion_selection_seed": EXPANSION_SELECTION_SEED,
        "final_sample_size": FINAL_SAMPLE_SIZE,
        "v2_freeze_id": V2_FREEZE_ID,
        "canonical_schema_sha256": canonical_schema_hash(),
        "core_module_sha256": sha256_file(core_module_path()),
    }


def load_checkpoint(
    output_dir: Path,
    expected_signature: dict[str, object],
    *,
    restart: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, set[int]]:
    paths = checkpoint_paths(output_dir)

    if restart:
        for path in paths.values():
            if path.exists():
                path.unlink()
        return pd.DataFrame(), pd.DataFrame(), set()

    if not paths["state"].exists():
        return pd.DataFrame(), pd.DataFrame(), set()

    saved = json.loads(paths["state"].read_text())
    saved_signature = saved.get("selection_signature", {})

    if saved_signature != expected_signature:
        raise RuntimeError(
            "An existing checkpoint was created with a different candidate pool, "
            "reference cohort, core characterization file or schema. Use "
            "--restart to discard it deliberately."
        )

    successes = (
        pd.read_parquet(paths["success"])
        if paths["success"].exists()
        else pd.DataFrame()
    )
    failures = (
        pd.read_csv(paths["failure"])
        if paths["failure"].exists()
        else pd.DataFrame()
    )

    attempted_orders: set[int] = set()
    if not successes.empty and "selection_order" in successes.columns:
        attempted_orders.update(
            pd.to_numeric(
                successes["selection_order"], errors="coerce"
            ).dropna().astype(int).tolist()
        )
    if not failures.empty and "selection_order" in failures.columns:
        attempted_orders.update(
            pd.to_numeric(
                failures["selection_order"], errors="coerce"
            ).dropna().astype(int).tolist()
        )

    return successes, failures, attempted_orders


def write_checkpoint(
    output_dir: Path,
    successes: pd.DataFrame,
    failures: pd.DataFrame,
    signature: dict[str, object],
    *,
    attempted_count: int,
) -> None:
    paths = checkpoint_paths(output_dir)

    if not successes.empty:
        successes.to_parquet(paths["success"], index=False)
    if not failures.empty:
        failures.to_csv(paths["failure"], index=False)

    state = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_signature": signature,
        "expansion_success_count": int(len(successes)),
        "attempted_count": int(attempted_count),
    }
    paths["state"].write_text(json.dumps(state, indent=2) + "\n")


def characterize_expansion(
    ordered_pool: pd.DataFrame,
    args: argparse.Namespace,
    signature: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    checkpoint_success, checkpoint_failure, attempted_orders = load_checkpoint(
        args.output_dir,
        signature,
        restart=bool(args.restart),
    )

    success_records: list[dict[str, object]] = []
    failure_records: list[dict[str, object]] = []

    if not checkpoint_success.empty:
        success_records.extend(
            checkpoint_success.to_dict(orient="records")
        )
    if not checkpoint_failure.empty:
        failure_records.extend(
            checkpoint_failure.to_dict(orient="records")
        )

    # Deduplicate checkpointed successes before deciding how much work remains.
    if success_records:
        checkpoint_success = pd.DataFrame(success_records)
        checkpoint_success["target_id"] = normalize_target_series(
            checkpoint_success["target_id"]
        )
        checkpoint_success = (
            checkpoint_success.sort_values("selection_order")
            .drop_duplicates(["target_id", "quarter"], keep="first")
            .reset_index(drop=True)
        )
        success_records = checkpoint_success.to_dict(orient="records")

    max_attempts = min(int(args.max_attempts), len(ordered_pool))
    worker_count = 0

    candidate_rows = ordered_pool.iloc[:max_attempts].copy()
    if attempted_orders:
        candidate_rows = candidate_rows[
            ~candidate_rows["selection_order"].isin(attempted_orders)
        ].copy()

    print(
        f"Expansion checkpoint: {len(success_records)}/{EXPANSION_SUCCESS_COUNT} "
        f"successful stars already available."
    )

    cursor = 0
    batch_size = max(1, int(args.batch_size))

    while len(success_records) < EXPANSION_SUCCESS_COUNT and cursor < len(candidate_rows):
        needed = EXPANSION_SUCCESS_COUNT - len(success_records)
        n = min(batch_size, len(candidate_rows) - cursor)

        batch = candidate_rows.iloc[cursor : cursor + n].copy()
        cursor += n

        batch_success_results, batch_failure_results, worker_count = (
            base.characterize_batch(batch, args)
        )

        for item in batch_success_results:
            diagnostics = dict(item["diagnostics"])
            diagnostics["population_cohort"] = "population_expansion_900"
            success_records.append(diagnostics)

        for item in batch_failure_results:
            failure_records.append(item)

        success_frame = pd.DataFrame(success_records)
        if not success_frame.empty:
            success_frame["target_id"] = normalize_target_series(
                success_frame["target_id"]
            )
            success_frame = (
                success_frame.sort_values("selection_order")
                .drop_duplicates(["target_id", "quarter"], keep="first")
                .head(EXPANSION_SUCCESS_COUNT)
                .reset_index(drop=True)
            )
            success_records = success_frame.to_dict(orient="records")

        failure_frame = pd.DataFrame(failure_records)

        attempted_total = len(attempted_orders) + cursor
        write_checkpoint(
            args.output_dir,
            success_frame,
            failure_frame,
            signature,
            attempted_count=attempted_total,
        )

        print(
            f"Population expansion progress: "
            f"{len(success_frame)}/{EXPANSION_SUCCESS_COUNT} successes; "
            f"{attempted_total} expansion targets attempted."
        )

    success_frame = pd.DataFrame(success_records)
    if not success_frame.empty:
        success_frame["target_id"] = normalize_target_series(
            success_frame["target_id"]
        )
        success_frame = (
            success_frame.sort_values("selection_order")
            .drop_duplicates(["target_id", "quarter"], keep="first")
            .head(EXPANSION_SUCCESS_COUNT)
            .reset_index(drop=True)
        )

    failure_frame = pd.DataFrame(failure_records)
    attempted_total = len(attempted_orders) + cursor

    if len(success_frame) != EXPANSION_SUCCESS_COUNT:
        raise RuntimeError(
            f"Only {len(success_frame)} of {EXPANSION_SUCCESS_COUNT} expansion "
            f"stars were successfully characterized after {attempted_total} attempts. "
            "Increase --max-attempts or inspect the checkpoint failure table. "
            "The next run will resume from the checkpoint automatically."
        )

    return success_frame, failure_frame, worker_count, attempted_total


def combine_reference_and_expansion(
    reference: pd.DataFrame,
    expansion: pd.DataFrame,
) -> pd.DataFrame:
    reference = reference.copy()
    expansion = expansion.copy()

    reference["population_cohort"] = "development_reference_100"
    expansion["population_cohort"] = "population_expansion_900"

    combined = pd.concat(
        [reference, expansion],
        ignore_index=True,
        sort=False,
    )
    combined["target_id"] = normalize_target_series(combined["target_id"])
    combined["quarter"] = pd.to_numeric(
        combined["quarter"], errors="coerce"
    ).fillna(QUARTER).astype(int)

    combined = (
        combined.sort_values("selection_order")
        .drop_duplicates(["target_id", "quarter"], keep="first")
        .reset_index(drop=True)
    )

    if len(combined) != FINAL_SAMPLE_SIZE:
        raise RuntimeError(
            f"Combined population contains {len(combined)} unique stars; "
            f"expected {FINAL_SAMPLE_SIZE}."
        )
    if combined["target_id"].nunique() != FINAL_SAMPLE_SIZE:
        raise RuntimeError("Final 1,000-star population contains duplicate IDs.")

    return combined


def build_canonical_table(profiled: pd.DataFrame) -> pd.DataFrame:
    interpretation_columns = [
        "v2_amplitude_population_label",
        "v2_memory_population_label",
        "v2_acf1_operational_label",
        *CANDIDATE_FLAG_COLUMNS,
        *REVIEW_FLAG_COLUMNS,
        "v2_dominant_statistical_behaviour",
    ]

    columns = [
        "target_id",
        "quarter",
        "selection_order",
        "population_cohort",
        *CANONICAL_CHARACTERIZATION_COLUMNS,
        *[
            column
            for column in interpretation_columns
            if column in profiled.columns
        ],
    ]
    columns = [column for column in columns if column in profiled.columns]
    return (
        profiled[columns]
        .sort_values("selection_order")
        .reset_index(drop=True)
    )


def canonical_schema_frame() -> pd.DataFrame:
    # Reuse the descriptions already frozen in the 100-star runner.
    return base.canonical_schema_frame()


def numeric_distribution_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return base.numeric_distribution_summary(frame)


def missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return base.missingness_summary(frame)


def redundancy_pairs(
    spearman: pd.DataFrame,
    threshold: float = 0.90,
) -> pd.DataFrame:
    return base.redundancy_pairs(spearman, threshold=threshold)


def categorical_counts(
    frame: pd.DataFrame,
    column: str,
    label_name: str,
) -> pd.DataFrame:
    return base.categorical_counts(frame, column, label_name)


def boolean_count_table(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label_name: str,
) -> pd.DataFrame:
    return base.boolean_count_table(frame, columns, label_name)


def conditional_feature_availability(canonical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in CANONICAL_CHARACTERIZATION_COLUMNS:
        if column not in canonical.columns:
            finite_count = 0
        elif column in CANONICAL_CONTINUOUS_FEATURE_COLUMNS:
            values = pd.to_numeric(canonical[column], errors="coerce")
            finite_count = int(np.isfinite(values).sum())
        else:
            values = canonical[column]
            finite_count = int(values.notna().sum())

        rows.append(
            {
                "feature": column,
                "role": (
                    "conditional diagnostic"
                    if column in CONDITIONAL_CANONICAL_FEATURES
                    else "core canonical variable"
                ),
                "n_total": int(len(canonical)),
                "n_available": finite_count,
                "availability_fraction": (
                    float(finite_count / len(canonical))
                    if len(canonical)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def load_reference_boundaries(reference_dir: Path) -> dict[str, float]:
    boundary_path = reference_dir / REFERENCE_BOUNDARIES_NAME
    if boundary_path.exists():
        raw = json.loads(boundary_path.read_text())
        return {
            str(k): float(v)
            for k, v in raw.items()
            if v is not None
        }

    freeze_path = reference_dir / REFERENCE_FREEZE_NAME
    freeze = json.loads(freeze_path.read_text())
    raw = freeze.get("realized_population_boundaries", {})
    return {
        str(k): float(v)
        for k, v in raw.items()
        if v is not None
    }


def boundary_comparison(
    reference: dict[str, float],
    final: dict[str, float],
) -> pd.DataFrame:
    rows = []
    keys = sorted(set(reference) | set(final))
    for key in keys:
        old = reference.get(key, np.nan)
        new = final.get(key, np.nan)
        delta = (
            float(new - old)
            if np.isfinite(old) and np.isfinite(new)
            else np.nan
        )
        rel = (
            float(delta / abs(old))
            if np.isfinite(delta) and np.isfinite(old) and old != 0
            else np.nan
        )
        rows.append(
            {
                "boundary": key,
                "reference_100": old,
                "population_1000": new,
                "absolute_change": delta,
                "relative_change": rel,
            }
        )
    return pd.DataFrame(rows)


def behaviour_fraction_table(
    frame: pd.DataFrame,
    sample_name: str,
) -> pd.DataFrame:
    counts = categorical_counts(
        frame,
        "v2_dominant_statistical_behaviour",
        "dominant_statistical_behaviour",
    )
    if counts.empty:
        return counts
    counts["sample"] = sample_name
    return counts


def behaviour_comparison(
    reference_profiled: pd.DataFrame,
    final_profiled: pd.DataFrame,
) -> pd.DataFrame:
    ref = behaviour_fraction_table(reference_profiled, "reference_100")
    final = behaviour_fraction_table(final_profiled, "population_1000")

    if ref.empty and final.empty:
        return pd.DataFrame()

    ref = ref.rename(
        columns={
            "count": "count_100",
            "fraction": "fraction_100",
        }
    ).drop(columns=["sample"], errors="ignore")
    final = final.rename(
        columns={
            "count": "count_1000",
            "fraction": "fraction_1000",
        }
    ).drop(columns=["sample"], errors="ignore")

    out = ref.merge(
        final,
        on="dominant_statistical_behaviour",
        how="outer",
    ).fillna(0)

    out["fraction_change"] = (
        out["fraction_1000"] - out["fraction_100"]
    )
    order = {
        label: i
        for i, label in enumerate(DOMINANT_STATISTICAL_BEHAVIOUR_ORDER)
    }
    out["_order"] = out["dominant_statistical_behaviour"].map(order).fillna(999)
    return (
        out.sort_values(["_order", "dominant_statistical_behaviour"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def final_output_paths(output_dir: Path) -> dict[str, Path]:
    metrics = output_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    return {
        "master": metrics / "characterization_master_1000.csv",
        "canonical_1000": metrics / "stellar_features_v2_1000.csv",
        "canonical_latest": metrics / "stellar_features_v2.csv",
        "continuous_1000": metrics / "stellar_features_v2_continuous_1000.csv",
        "continuous_latest": metrics / "stellar_features_v2_continuous.csv",
        "schema": metrics / "canonical_feature_schema.csv",
        "distribution": metrics / "canonical_feature_distribution_summary.csv",
        "missingness": metrics / "canonical_missingness_summary.csv",
        "availability": metrics / "canonical_feature_availability.csv",
        "spearman": metrics / "canonical_spearman_correlation.csv",
        "pearson": metrics / "canonical_pearson_correlation.csv",
        "redundancy": metrics / "canonical_redundancy_pairs.csv",
        "behaviour_counts": metrics / "dominant_statistical_behaviour_counts.csv",
        "candidate_flag_counts": metrics / "candidate_flag_counts.csv",
        "review_flag_counts": metrics / "review_flag_counts.csv",
        "stationarity_counts": metrics / "stationarity_state_counts.csv",
        "amplitude_counts": metrics / "amplitude_population_counts.csv",
        "memory_counts": metrics / "memory_population_counts.csv",
        "boundaries": metrics / "population_boundaries_1000.json",
        "boundary_comparison": metrics / "reference_100_vs_population_1000_boundaries.csv",
        "behaviour_comparison": metrics / "reference_100_vs_population_1000_behaviours.csv",
        "failures": metrics / "characterization_failures.csv",
        "manifest": metrics / "target_manifest_used.csv",
        "quality": metrics / "characterization_population_quality_checks.json",
        "freeze": metrics / "characterization_v2_population_freeze_1000.json",
        "summary": metrics / "characterization_population1000_summary.txt",
    }


def write_final_outputs(
    final_profiled: pd.DataFrame,
    reference_profiled: pd.DataFrame,
    failures: pd.DataFrame,
    thresholds: dict[str, float],
    reference_boundaries: dict[str, float],
    reference_freeze: dict,
    provenance: dict[str, object],
    signature: dict[str, object],
    args: argparse.Namespace,
    *,
    worker_count: int,
    attempted_expansion_count: int,
) -> dict[str, Path]:
    paths = final_output_paths(args.output_dir)

    canonical = build_canonical_table(final_profiled)
    continuous = canonical[
        [
            "target_id",
            "quarter",
            "population_cohort",
            *CANONICAL_CONTINUOUS_FEATURE_COLUMNS,
        ]
    ].copy()

    final_profiled.to_csv(paths["master"], index=False)
    canonical.to_csv(paths["canonical_1000"], index=False)
    canonical.to_csv(paths["canonical_latest"], index=False)
    continuous.to_csv(paths["continuous_1000"], index=False)
    continuous.to_csv(paths["continuous_latest"], index=False)

    canonical_schema_frame().to_csv(paths["schema"], index=False)
    numeric_distribution_summary(canonical).to_csv(
        paths["distribution"], index=False
    )
    missingness = missingness_summary(canonical)
    missingness.to_csv(paths["missingness"], index=False)
    availability = conditional_feature_availability(canonical)
    availability.to_csv(paths["availability"], index=False)

    numeric = continuous[
        list(CANONICAL_CONTINUOUS_FEATURE_COLUMNS)
    ].apply(pd.to_numeric, errors="coerce")
    spearman = numeric.corr(method="spearman")
    pearson = numeric.corr(method="pearson")
    spearman.to_csv(paths["spearman"])
    pearson.to_csv(paths["pearson"])

    redundant = redundancy_pairs(spearman, threshold=0.90)
    redundant.to_csv(paths["redundancy"], index=False)

    behaviour = categorical_counts(
        canonical,
        "v2_dominant_statistical_behaviour",
        "dominant_statistical_behaviour",
    )
    if not behaviour.empty:
        order = {
            label: i
            for i, label in enumerate(DOMINANT_STATISTICAL_BEHAVIOUR_ORDER)
        }
        behaviour["_order"] = behaviour[
            "dominant_statistical_behaviour"
        ].map(order).fillna(999)
        behaviour = (
            behaviour.sort_values(
                ["_order", "dominant_statistical_behaviour"]
            )
            .drop(columns="_order")
            .reset_index(drop=True)
        )
    behaviour.to_csv(paths["behaviour_counts"], index=False)

    boolean_count_table(
        canonical,
        CANDIDATE_FLAG_COLUMNS,
        "candidate_flag",
    ).to_csv(paths["candidate_flag_counts"], index=False)

    boolean_count_table(
        canonical,
        REVIEW_FLAG_COLUMNS,
        "review_flag",
    ).to_csv(paths["review_flag_counts"], index=False)

    categorical_counts(
        canonical,
        "original_series_stationarity_conclusion",
        "stationarity_state",
    ).to_csv(paths["stationarity_counts"], index=False)

    categorical_counts(
        canonical,
        "v2_amplitude_population_label",
        "amplitude_population",
    ).to_csv(paths["amplitude_counts"], index=False)

    categorical_counts(
        canonical,
        "v2_memory_population_label",
        "memory_population",
    ).to_csv(paths["memory_counts"], index=False)

    failures.to_csv(paths["failures"], index=False)

    manifest = canonical[
        ["target_id", "quarter", "selection_order", "population_cohort"]
    ].copy()
    manifest["selection_group"] = np.where(
        manifest["population_cohort"].eq("development_reference_100"),
        "development_reference_100",
        "random_clean_q5_population_expansion",
    )
    manifest.to_csv(paths["manifest"], index=False)

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest_out, index=False)

    realized_boundaries = {
        key: float(value) if np.isfinite(value) else None
        for key, value in thresholds.items()
    }
    paths["boundaries"].write_text(
        json.dumps(realized_boundaries, indent=2) + "\n"
    )
    args.boundary_out.parent.mkdir(parents=True, exist_ok=True)
    args.boundary_out.write_text(
        json.dumps(
            {
                "freeze_id": V2_FREEZE_ID,
                "population_reference": "clean_kepler_q5_1000",
                "sample_size": FINAL_SAMPLE_SIZE,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "boundaries": realized_boundaries,
                "boundary_config": asdict(DEFAULT_BOUNDARIES),
            },
            indent=2,
        )
        + "\n"
    )

    boundary_cmp = boundary_comparison(
        reference_boundaries,
        thresholds,
    )
    boundary_cmp.to_csv(paths["boundary_comparison"], index=False)

    behaviour_cmp = behaviour_comparison(
        reference_profiled,
        final_profiled,
    )
    behaviour_cmp.to_csv(paths["behaviour_comparison"], index=False)

    nonconditional_missing = missingness[
        ~missingness["feature"].isin(CONDITIONAL_CANONICAL_FEATURES)
    ]
    max_nonconditional_missing = (
        float(nonconditional_missing["missing_fraction"].max())
        if not nonconditional_missing.empty
        else np.nan
    )

    conditional_row = availability[
        availability["feature"].eq(
            "v2_ls_acf_period_relative_error"
        )
    ]
    ls_acf_availability = (
        float(conditional_row["availability_fraction"].iloc[0])
        if not conditional_row.empty
        else np.nan
    )

    quality = {
        "unique_star_count": int(canonical["target_id"].nunique()),
        "row_count": int(len(canonical)),
        "reference_star_count": int(
            (canonical["population_cohort"] == "development_reference_100").sum()
        ),
        "expansion_star_count": int(
            (canonical["population_cohort"] == "population_expansion_900").sum()
        ),
        "canonical_variable_count": int(
            len(CANONICAL_CHARACTERIZATION_COLUMNS)
        ),
        "canonical_continuous_variable_count": int(
            len(CANONICAL_CONTINUOUS_FEATURE_COLUMNS)
        ),
        "max_missing_fraction_excluding_conditional_features": (
            max_nonconditional_missing
        ),
        "ls_acf_period_agreement_availability_fraction": (
            ls_acf_availability
        ),
        "high_redundancy_pair_count_abs_spearman_ge_0_90": int(
            len(redundant)
        ),
        "expansion_attempt_count": int(attempted_expansion_count),
        "expansion_failure_count": int(len(failures)),
    }
    paths["quality"].write_text(json.dumps(quality, indent=2) + "\n")

    freeze = {
        "freeze_id": V2_FREEZE_ID,
        "freeze_stage": "final_population_characterization_1000",
        "scientific_characterization_version": "stellar_variability_v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "current_git_commit": git_commit(),
        "definition_development_reference_git_commit": reference_freeze.get(
            "git_commit"
        ),
        "sample_size": FINAL_SAMPLE_SIZE,
        "quarter": QUARTER,
        "reference_sample_size": REFERENCE_SAMPLE_SIZE,
        "expansion_sample_size": EXPANSION_SUCCESS_COUNT,
        "expansion_selection_seed": EXPANSION_SELECTION_SEED,
        "selection_provenance": provenance,
        "selection_signature": signature,
        "population_boundary_config": asdict(DEFAULT_BOUNDARIES),
        "realized_population_boundaries_1000": realized_boundaries,
        "canonical_domains": sorted(
            {item[0] for item in CANONICAL_CHARACTERIZATION_SCHEMA}
        ),
        "canonical_feature_count": int(
            len(CANONICAL_CHARACTERIZATION_COLUMNS)
        ),
        "canonical_columns": list(CANONICAL_CHARACTERIZATION_COLUMNS),
        "canonical_continuous_columns": list(
            CANONICAL_CONTINUOUS_FEATURE_COLUMNS
        ),
        "conditional_canonical_features": sorted(
            CONDITIONAL_CANONICAL_FEATURES
        ),
        "worker_count": int(worker_count),
        "characterization_parameters": {
            "quality_policy": args.quality_policy,
            "require_finite_flux_error": bool(
                args.require_finite_flux_error
            ),
            "test_fraction": float(args.test_fraction),
            "v1_acf_lags": int(args.v1_acf_lags),
            "v2_acf_lags": int(args.v2_acf_lags),
            "rolling_window": int(args.rolling_window),
            "v1_spectral_frequencies": int(
                args.v1_spectral_frequencies
            ),
            "v2_spectral_frequencies": int(
                args.v2_spectral_frequencies
            ),
            "stationarity_min_observations": int(
                args.stationarity_min_observations
            ),
        },
        "scientific_freeze_statement": (
            "The seven-domain/eleven-variable characterization definitions were "
            "frozen before this population expansion. The 1,000-star cohort is "
            "used to finalize population distributions, population-relative "
            "boundaries and prevalence estimates; it does not redefine the "
            "underlying feature calculations."
        ),
        "selection_contract": (
            "The original development 100 are retained. The 900 expansion stars "
            "are chosen from a deterministic pre-characterization random ordering "
            "of the upstream clean Q5 pool. No light-curve statistical feature or "
            "derived v2 label is used for expansion inclusion/ranking."
        ),
        "quality_checks": quality,
    }

    freeze["sha256"] = {
        "core_stellar_variability.py": sha256_file(core_module_path()),
        "target_manifest_used.csv": sha256_file(paths["manifest"]),
        "stellar_features_v2_1000.csv": sha256_file(
            paths["canonical_1000"]
        ),
        "characterization_master_1000.csv": sha256_file(
            paths["master"]
        ),
        "population_boundaries_1000.json": sha256_file(
            paths["boundaries"]
        ),
        "canonical_feature_schema.csv": sha256_file(paths["schema"]),
    }
    paths["freeze"].write_text(json.dumps(freeze, indent=2) + "\n")

    summary_lines = [
        "FINAL STELLAR BACKGROUND CHARACTERIZATION — v2",
        "=" * 48,
        "",
        f"Sample: {FINAL_SAMPLE_SIZE} unique clean Kepler Q5 stars",
        f"Reference development cohort retained: {REFERENCE_SAMPLE_SIZE}",
        f"New population-expansion stars: {EXPANSION_SUCCESS_COUNT}",
        f"Frozen scientific domains: {len(set(item[0] for item in CANONICAL_CHARACTERIZATION_SCHEMA))}",
        f"Frozen canonical variables: {len(CANONICAL_CHARACTERIZATION_COLUMNS)}",
        f"Continuous canonical variables: {len(CANONICAL_CONTINUOUS_FEATURE_COLUMNS)}",
        "",
        "Final 1,000-star population boundaries:",
    ]
    for key, value in thresholds.items():
        summary_lines.append(f"  {key}: {value}")

    summary_lines.extend(
        [
            "",
            "Quality checks:",
            f"  non-conditional max missing fraction: {max_nonconditional_missing}",
            f"  LS-ACF agreement availability fraction: {ls_acf_availability}",
            f"  |Spearman rho| >= 0.90 pairs: {len(redundant)}",
            "",
            "Dominant statistical behaviour counts:",
        ]
    )
    if behaviour.empty:
        summary_lines.append("  <none>")
    else:
        for _, row in behaviour.iterrows():
            summary_lines.append(
                f"  {row['dominant_statistical_behaviour']}: "
                f"{int(row['count'])} "
                f"({100.0 * float(row['fraction']):.1f}%)"
            )

    summary_lines.extend(
        [
            "",
            "Interpretation:",
            "  The 1,000-star population finalizes population distributions and",
            "  relative boundaries. Behaviour labels remain descriptive screens;",
            "  downstream ML should consume the underlying canonical variables.",
            "",
        ]
    )
    paths["summary"].write_text("\n".join(summary_lines))

    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Expand the frozen v2 100-star development characterization to a "
            "final 1,000-star clean-Q5 population."
        )
    )
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        default=DEFAULT_CANDIDATE_POOL,
        help=(
            "Large upstream catalog-clean target pool. Use "
            "outputs/target_selection/kepler_catalog_clean_pool.csv, not the "
            "250-target characterization candidate file."
        ),
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help="Metrics directory from the completed 100-star frozen v2 run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=DEFAULT_MANIFEST_OUT,
    )
    parser.add_argument(
        "--boundary-out",
        type=Path,
        default=DEFAULT_BOUNDARY_OUT,
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--trust-clean-pool",
        action="store_true",
        help=(
            "Use only after verifying that the candidate file is the upstream "
            "catalog-clean pool and was not built from light-curve statistics."
        ),
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard any existing 900-star expansion checkpoint.",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--reserve-cpu-cores", type=int, default=2)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Expansion targets submitted per checkpointed batch.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1800,
        help=(
            "Maximum expansion candidates attempted to obtain 900 successes. "
            "Increase if Q5 availability failures are common."
        ),
    )

    # These defaults MUST match the frozen 100-star characterization run unless
    # you deliberately intend to create a new characterization version.
    parser.add_argument("--quality-policy", default="default")
    parser.add_argument(
        "--require-finite-flux-error",
        action="store_true",
    )
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--v1-acf-lags", type=int, default=80)
    parser.add_argument("--v2-acf-lags", type=int, default=240)
    parser.add_argument("--rolling-window", type=int, default=96)
    parser.add_argument(
        "--v1-spectral-frequencies",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--v2-spectral-frequencies",
        type=int,
        default=4000,
    )
    parser.add_argument(
        "--stationarity-min-observations",
        type=int,
        default=24,
    )
    return parser


def validate_parameters_against_reference(
    args: argparse.Namespace,
    reference_freeze: dict,
) -> None:
    frozen = reference_freeze.get("characterization_parameters", {})
    requested = {
        "quality_policy": args.quality_policy,
        "require_finite_flux_error": bool(
            args.require_finite_flux_error
        ),
        "test_fraction": float(args.test_fraction),
        "v1_acf_lags": int(args.v1_acf_lags),
        "v2_acf_lags": int(args.v2_acf_lags),
        "rolling_window": int(args.rolling_window),
        "v1_spectral_frequencies": int(
            args.v1_spectral_frequencies
        ),
        "v2_spectral_frequencies": int(
            args.v2_spectral_frequencies
        ),
        "stationarity_min_observations": int(
            args.stationarity_min_observations
        ),
    }

    differences = {}
    for key, value in requested.items():
        if key in frozen and frozen[key] != value:
            differences[key] = {
                "reference_100": frozen[key],
                "requested": value,
            }

    if differences:
        raise ValueError(
            "Characterization parameters differ from the frozen 100-star run. "
            "Do not silently redefine v2 during population expansion. "
            f"Differences: {differences}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    args.candidate_pool = args.candidate_pool.expanduser().resolve()
    args.reference_dir = args.reference_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.manifest_out = args.manifest_out.expanduser().resolve()
    args.boundary_out = args.boundary_out.expanduser().resolve()

    if not args.candidate_pool.exists():
        raise FileNotFoundError(args.candidate_pool)

    reference, reference_freeze = load_reference_100(
        args.reference_dir
    )
    validate_parameters_against_reference(args, reference_freeze)

    reference_ids = set(reference["target_id"].astype(str))
    expansion_pool, provenance = prepare_expansion_pool(
        args.candidate_pool,
        reference_ids,
        trust_clean_pool=bool(args.trust_clean_pool),
    )

    if int(args.max_attempts) < EXPANSION_SUCCESS_COUNT:
        raise ValueError(
            f"--max-attempts must be at least {EXPANSION_SUCCESS_COUNT}."
        )

    signature = selection_signature(
        args.candidate_pool,
        reference_ids,
    )

    print()
    print("Frozen v2 final population characterization")
    print("-------------------------------------------")
    print(f"Reference cohort: {REFERENCE_SAMPLE_SIZE} stars")
    print(f"Expansion target: {EXPANSION_SUCCESS_COUNT} new stars")
    print(f"Final population target: {FINAL_SAMPLE_SIZE} stars")
    print(f"Candidate pool: {args.candidate_pool}")
    print(
        f"Clean expansion candidates available after excluding reference 100: "
        f"{len(expansion_pool)}"
    )
    print(f"Expansion selection seed: {EXPANSION_SELECTION_SEED}")
    print("Statistical selection variables used for expansion: NONE")
    print(f"Core v2 freeze ID: {V2_FREEZE_ID}")
    print()

    expansion, failures, workers, attempted = characterize_expansion(
        expansion_pool,
        args,
        signature,
    )

    combined_raw = combine_reference_and_expansion(
        reference,
        expansion,
    )

    # The central population-freeze operation:
    # final quantiles and all population-relative labels are recomputed from the
    # full 1,000-star cohort. No 100-star population threshold is carried forward.
    final_profiled, thresholds = apply_population_variability_boundaries(
        combined_raw,
        boundaries=DEFAULT_BOUNDARIES,
    )
    final_profiled = assign_dominant_statistical_behaviour(
        final_profiled
    )
    final_profiled = final_profiled.sort_values(
        "selection_order"
    ).reset_index(drop=True)

    # Re-profile the original 100 using ONLY the final 1,000-star thresholds is
    # not directly supported by apply_population_variability_boundaries (which
    # derives its own quantiles), so the 100-side behaviour comparison below
    # intentionally represents the original development characterization. The
    # final 1,000 table contains the authoritative population labels.
    reference_profiled, _ = apply_population_variability_boundaries(
        reference,
        boundaries=DEFAULT_BOUNDARIES,
    )
    reference_profiled = assign_dominant_statistical_behaviour(
        reference_profiled
    )

    reference_boundaries = load_reference_boundaries(
        args.reference_dir
    )

    paths = write_final_outputs(
        final_profiled,
        reference_profiled,
        failures,
        thresholds,
        reference_boundaries,
        reference_freeze,
        provenance,
        signature,
        args,
        worker_count=workers,
        attempted_expansion_count=attempted,
    )

    print()
    print("1,000-star stellar-background characterization complete.")
    print(f"Output directory: {args.output_dir / 'metrics'}")
    print(f"Frozen 1,000-star manifest: {args.manifest_out}")
    print(f"Frozen 1,000-star population boundaries: {args.boundary_out}")
    print()
    print("FINAL 1,000-star population boundaries:")
    for key, value in thresholds.items():
        print(f"  {key}: {value}")

    print()
    print("Dominant statistical behaviour counts:")
    behaviour = pd.read_csv(paths["behaviour_counts"])
    for _, row in behaviour.iterrows():
        print(
            f"  {row['dominant_statistical_behaviour']}: "
            f"{int(row['count'])} "
            f"({100.0 * float(row['fraction']):.1f}%)"
        )

    print()
    print("Key freeze outputs:")
    for key in (
        "canonical_1000",
        "distribution",
        "missingness",
        "availability",
        "spearman",
        "redundancy",
        "boundary_comparison",
        "behaviour_comparison",
        "quality",
        "freeze",
        "summary",
    ):
        print(f"  {key}: {paths[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
