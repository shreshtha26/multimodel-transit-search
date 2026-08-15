"""Select and characterize 10 independent Kepler Q5 validation stars.

Scientific role
---------------
The existing characterization-development set contains 40 stars deliberately
sampled from four named background strata:

    high_scatter
    long_memory
    quiet_low_scatter
    smooth_background_dominant

Those stars are useful for developing and vetting characterization definitions,
but using the same labelled set to validate the final definitions would be
circular.  This script therefore creates a small *independent validation set*.

The validation stars are selected with these rules:

1. Start from the already-vetted clean Q5 candidate pool produced during the
   clean-manifest workflow.
2. Exclude every target-quarter pair used by the previous clean 50-star
   benchmark.  This is stricter than excluding only the 40 development stars.
3. Do NOT use ``sample_stratum`` or any characterization-derived quantity in
   the validation selection.
4. Put the unseen candidates into a fixed-seed random order.
5. Apply only a Q5-data-availability screen: keep a target if Lightkurve can
   locate a Kepler-author, long-cadence light curve for that target in Q5.
   This screen does not inspect flux values or characterization diagnostics.
6. Keep the first 10 available targets in that random order and freeze the
   resulting manifest on disk.
7. Run the exact existing ``run_light_curve_characterization.py`` workflow.
   No stationarity, periodicity, scatter, memory, or coherence thresholds are
   estimated or changed here.

This is intentionally a validation workflow, not a fifth stellar regime.

Expected pre-existing inputs
----------------------------
Candidate pool:
    outputs/target_selection/kepler_catalog_clean_candidates_q5.csv

Previously used 50-star manifest:
    outputs/experiments/multistar_challenger_benchmark/
        clean_q5_50star/metrics/target_manifest_used.csv

Outputs
-------
Frozen validation manifest:
    outputs/target_selection/kepler_characterization_validation10.csv

Per-star v2 characterization:
    outputs/experiments/characterization_validation10/

Execution status:
    outputs/experiments/characterization_validation10/metrics/
        validation_execution_status.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from lightkurve import search_lightcurve


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CANDIDATE_POOL = (
    PROJECT_ROOT
    / "outputs"
    / "target_selection"
    / "kepler_catalog_clean_candidates_q5.csv"
)

DEFAULT_EXCLUSION_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "multistar_challenger_benchmark"
    / "clean_q5_50star"
    / "metrics"
    / "target_manifest_used.csv"
)

DEFAULT_VALIDATION_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "target_selection"
    / "kepler_characterization_validation10.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "characterization_validation10"
)

CHARACTERIZATION_SCRIPT = (
    PROJECT_ROOT / "scripts" / "run_light_curve_characterization.py"
)

DEFAULT_N_STARS = 10
DEFAULT_QUARTER = 5
DEFAULT_SEED = 20260815


TARGET_ID_ALIASES = (
    "target_id",
    "kepid",
    "kic",
    "KIC",
    "kepler_id",
)

QUARTER_ALIASES = (
    "quarter",
    "Quarter",
    "q",
)


def _resolve_column(frame: pd.DataFrame, aliases: tuple[str, ...], label: str) -> str:
    """Resolve a required logical field without assuming one historical schema."""
    for column in aliases:
        if column in frame.columns:
            return column
    raise ValueError(
        f"Could not find {label} column. Tried {aliases}; "
        f"available columns are {list(frame.columns)}"
    )


def _clean_target_id(value) -> str:
    """Return a canonical numeric KIC identifier as a string."""
    text = str(value).strip()
    if text.upper().startswith("KIC"):
        text = text[3:].strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text or not text.isdigit():
        raise ValueError(f"Invalid Kepler target id: {value!r}")
    return text


def _canonicalize(frame: pd.DataFrame, default_quarter: int) -> pd.DataFrame:
    """Add canonical target_id/quarter columns while retaining source metadata."""
    frame = frame.copy()
    target_column = _resolve_column(frame, TARGET_ID_ALIASES, "target id")
    frame["target_id"] = frame[target_column].map(_clean_target_id)

    quarter_column = next(
        (column for column in QUARTER_ALIASES if column in frame.columns),
        None,
    )
    if quarter_column is None:
        frame["quarter"] = int(default_quarter)
    else:
        frame["quarter"] = pd.to_numeric(
            frame[quarter_column], errors="raise"
        ).astype(int)

    return frame


def _target_key(frame: pd.DataFrame) -> pd.Series:
    """Target-quarter is the independence unit used by this validation sample."""
    return frame["target_id"].astype(str) + "_q" + frame["quarter"].astype(str)


def _has_kepler_long_cadence_data(target_id: str, quarter: int) -> bool:
    """Check only whether the required Kepler Q5 product exists.

    This is deliberately an *availability* screen.  It does not download or
    inspect the light-curve flux and therefore cannot select validation stars
    based on scatter, variability, periodicity, stationarity, or any other
    characterization result.
    """
    query = f"KIC {_clean_target_id(target_id)}"
    result = search_lightcurve(
        query,
        mission="Kepler",
        author="Kepler",
        cadence="long",
        quarter=int(quarter),
    )
    return len(result) > 0


def select_validation_manifest(
    candidate_pool_path: Path,
    exclusion_manifest_path: Path,
    output_manifest_path: Path,
    *,
    n_stars: int,
    quarter: int,
    seed: int,
    resample: bool,
) -> pd.DataFrame:
    """Create or load the frozen independent validation manifest."""
    if output_manifest_path.exists() and not resample:
        selected = _canonicalize(pd.read_csv(output_manifest_path), quarter)
        if len(selected) != n_stars:
            raise ValueError(
                f"Frozen validation manifest has {len(selected)} rows; "
                f"expected {n_stars}. Use --resample only if you intentionally "
                "want to replace the frozen sample."
            )
        print(f"Reusing frozen validation manifest: {output_manifest_path}")
        return selected

    if not candidate_pool_path.exists():
        raise FileNotFoundError(
            f"Clean candidate pool not found: {candidate_pool_path}\n"
            "Run the existing clean-Q5 target-selection workflow first."
        )
    if not exclusion_manifest_path.exists():
        raise FileNotFoundError(
            f"50-star exclusion manifest not found: {exclusion_manifest_path}\n"
            "This validation sample must exclude every previously used target."
        )

    candidates = _canonicalize(pd.read_csv(candidate_pool_path), quarter)
    excluded = _canonicalize(pd.read_csv(exclusion_manifest_path), quarter)

    # Q5 only.  The clean candidate file should already be Q5, but keeping this
    # explicit prevents a future mixed-quarter input from silently changing the
    # validation design.
    candidates = candidates.loc[candidates["quarter"] == int(quarter)].copy()

    # One target-quarter row per candidate before random selection.
    candidates["_target_key"] = _target_key(candidates)
    excluded["_target_key"] = _target_key(excluded)
    candidates = candidates.drop_duplicates("_target_key", keep="first")

    excluded_keys = set(excluded["_target_key"].astype(str))
    eligible = candidates.loc[
        ~candidates["_target_key"].isin(excluded_keys)
    ].copy()

    # IMPORTANT: do not filter, rank, stratify, or weight by sample_stratum,
    # scatter, ACF, periodogram, stationarity, coherence, or any v2 diagnostic.
    # The only extra eligibility screen below is whether the required Kepler
    # long-cadence product actually exists in Q5.
    if len(eligible) < n_stars:
        raise ValueError(
            f"Only {len(eligible)} unseen clean candidates remain; "
            f"cannot draw {n_stars} validation stars."
        )

    # Rejection sampling in a fixed-seed random order gives a reproducible
    # unstratified sample from the subset that actually has the required Q5
    # Kepler long-cadence product.  Availability is checked before any flux or
    # v2 characterization value is inspected.
    random_order = eligible.sample(
        frac=1.0,
        replace=False,
        random_state=int(seed),
    ).reset_index(drop=True)

    selected_rows = []
    availability_checks = 0

    print("Checking Q5 Kepler long-cadence availability...")
    for _, row in random_order.iterrows():
        target_id = _clean_target_id(row["target_id"])
        target_quarter = int(row["quarter"])
        availability_checks += 1

        available = _has_kepler_long_cadence_data(target_id, target_quarter)
        state = "available" if available else "unavailable"
        print(
            f"  [{availability_checks}] KIC {target_id} "
            f"Q{target_quarter}: {state}"
        )

        if not available:
            continue

        selected_row = row.copy()
        selected_row["q5_long_cadence_available"] = True
        selected_rows.append(selected_row)

        if len(selected_rows) == int(n_stars):
            break

    if len(selected_rows) < int(n_stars):
        raise RuntimeError(
            f"Found only {len(selected_rows)} available unseen Q5 targets "
            f"after checking {availability_checks} candidates; "
            f"needed {n_stars}."
        )

    selected = pd.DataFrame(selected_rows).copy()

    # Stable display order is useful for reproducible logs.  Sorting happens
    # only *after* availability-blind random ordering and acceptance.
    selected = selected.sort_values(
        ["target_id", "quarter"],
        key=lambda col: pd.to_numeric(col, errors="ignore"),
    ).reset_index(drop=True)

    selected["sample_role"] = "independent_validation"
    selected["validation_selection_method"] = (
        "fixed_seed_random_order_with_q5_long_cadence_availability_screen"
    )
    selected["availability_checks_required"] = int(availability_checks)
    selected["validation_selection_seed"] = int(seed)
    selected["excluded_previous_sample_size"] = int(
        excluded["_target_key"].nunique()
    )
    selected["source_candidate_pool"] = str(
        candidate_pool_path.relative_to(PROJECT_ROOT)
    )

    selected = selected.drop(columns=["_target_key"], errors="ignore")

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_manifest_path, index=False)

    # Hard leakage/overlap guard.
    selected_keys = set(_target_key(selected))
    overlap = selected_keys & excluded_keys
    if overlap:
        raise RuntimeError(
            "Validation sample overlaps the previous benchmark: "
            + ", ".join(sorted(overlap))
        )

    print(f"Candidate rows available: {len(candidates)}")
    print(f"Previously used target-quarter rows excluded: {len(excluded_keys)}")
    print(f"Eligible unseen clean candidates: {len(eligible)}")
    print(f"Availability checks required: {availability_checks}")
    print(f"Validation stars selected: {len(selected)}")
    print(f"Selection seed: {seed}")
    print(f"Frozen validation manifest: {output_manifest_path}")
    print()
    print(selected[["target_id", "quarter"]].to_string(index=False))
    return selected


def run_characterization(
    manifest: pd.DataFrame,
    output_dir: Path,
    *,
    stop_on_error: bool,
) -> pd.DataFrame:
    """Run the existing v2 characterization unchanged for each validation star."""
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        src_path
        if not existing_pythonpath
        else src_path + os.pathsep + existing_pythonpath
    )

    status_rows = []

    print()
    print(f"Independent validation stars to characterize: {len(manifest)}")
    print("Sample role: independent_validation")
    print(
        "Threshold policy: frozen -- this script does not estimate or modify "
        "characterization boundaries."
    )

    for index, row in manifest.reset_index(drop=True).iterrows():
        target_id = _clean_target_id(row["target_id"])
        quarter = int(row["quarter"])
        print()
        print(f"[{index + 1}/{len(manifest)}] KIC {target_id} Q{quarter}")

        command = [
            sys.executable,
            str(CHARACTERIZATION_SCRIPT),
            "--target-id",
            target_id,
            "--quarter",
            str(quarter),
            "--output-dir",
            str(output_dir),
        ]

        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            check=False,
        )

        status_rows.append(
            {
                "target_id": target_id,
                "quarter": quarter,
                "sample_role": "independent_validation",
                "return_code": int(completed.returncode),
                "success": bool(completed.returncode == 0),
            }
        )

        pd.DataFrame(status_rows).to_csv(
            metrics_dir / "validation_execution_status.csv",
            index=False,
        )

        if completed.returncode != 0 and stop_on_error:
            raise subprocess.CalledProcessError(
                completed.returncode,
                command,
            )

    status = pd.DataFrame(status_rows)
    success_count = int(status["success"].sum()) if not status.empty else 0
    print()
    print(
        f"Validation characterization complete: "
        f"{success_count}/{len(status)} successful"
    )
    print(f"Outputs: {output_dir}")
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select 10 unseen clean Q5 stars in fixed-seed random order, "
            "requiring only Kepler Q5 long-cadence availability, and run the "
            "existing v2 light-curve characterization."
        )
    )
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        default=DEFAULT_CANDIDATE_POOL,
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        default=DEFAULT_EXCLUSION_MANIFEST,
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=DEFAULT_VALIDATION_MANIFEST,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--n-stars", type=int, default=DEFAULT_N_STARS)
    parser.add_argument("--quarter", type=int, default=DEFAULT_QUARTER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="Freeze/print the validation manifest without running characterization.",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help=(
            "Replace an existing frozen validation manifest. Avoid this after "
            "looking at validation results."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later stars if one characterization call fails.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.n_stars <= 0:
        raise ValueError("--n-stars must be positive.")

    manifest = select_validation_manifest(
        args.candidate_pool,
        args.exclude_manifest,
        args.validation_manifest,
        n_stars=args.n_stars,
        quarter=args.quarter,
        seed=args.seed,
        resample=args.resample,
    )

    if args.select_only:
        return 0

    status = run_characterization(
        manifest,
        args.output_dir,
        stop_on_error=not args.continue_on_error,
    )
    return 0 if status["success"].all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
