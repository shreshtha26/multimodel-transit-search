"""Run the frozen stellar-variability v2 characterization on 100 clean Q5 stars.

Scientific contract
-------------------
1. Target selection is independent of the statistical characterization labels.
   The selector may use only: target identity, Quarter 5 membership, upstream
   clean/vetting flags, and data-availability success.  It never selects on ACF,
   scatter, stationarity, spectral structure, periodicity, variance evolution,
   or any v2 candidate label.
2. Every selected star is characterized with the same preprocessing and v1+v2
   feature extraction path.
3. Population-relative Q25/Q75 boundaries are computed only after exactly 100
   successful stars are available, using the full 100-star comparison sample.
4. The compact scientific output is seven domains represented by eleven
   canonical variables.  The larger diagnostic record is retained for audit.
5. Candidate/review labels and the single-valued dominant behaviour are
   interpretation aids.  They are not astrophysical classifications and should
   not replace the continuous features in a future router.

Typical run
-----------
python scripts/run_characterization_100star.py \
    --candidate-pool /path/to/upstream_clean_q5_pool.csv \
    --allow-download \
    --max-workers 4

If the upstream pool has already been externally vetted but does not expose the
KOI/TCE/confirmed/EB veto columns, add --trust-clean-pool.  Do that only when
you know the file is the pre-characterization clean pool, not the old
statistically stratified 50-star manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.noise_models.characterization import characterize_regularized_light_curve
from adaptive_transit.noise_models.stellar_variability import (
    CANONICAL_CHARACTERIZATION_COLUMNS,
    CANONICAL_CHARACTERIZATION_SCHEMA,
    CANONICAL_CONTINUOUS_FEATURE_COLUMNS,
    DEFAULT_BOUNDARIES,
    DOMINANT_STATISTICAL_BEHAVIOUR_ORDER,
    V2_FREEZE_ID,
    apply_population_variability_boundaries,
    assign_dominant_statistical_behaviour,
    stellar_variability_summary,
)
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/characterization_validation100"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "outputs/cache/kepler_light_curves"
DEFAULT_MANIFEST_OUT = PROJECT_ROOT / "configs/kepler_q5_clean_100_star_manifest.csv"

SAMPLE_SIZE = 100
QUARTER = 5
SELECTION_SEED = 20260816

TARGET_ID_ALIASES = (
    "target_id",
    "kepid",
    "kic",
    "kic_id",
    "KIC",
)

QUARTER_ALIASES = ("quarter", "q", "kepler_quarter")

# A positive upstream clean flag is enough to establish catalog-level cleaning.
CLEAN_PASS_ALIASES = (
    "clean_candidate",
    "clean_sample_eligible",
    "clean_vetting_pass",
    "visual_review_pass",
    "eligible",
    "is_clean",
    "keep",
    "accepted",
)

# Otherwise require all four families below unless --trust-clean-pool is used.
# These are catalog/vetting exclusions, not light-curve statistical labels.
VETO_FLAG_ALIASES = {
    "koi": (
        "koi_flag",
        "is_koi",
        "has_koi",
        "known_koi",
        "koi",
    ),
    "tce": (
        "tce_flag",
        "is_tce",
        "has_tce",
        "known_tce",
        "tce",
    ),
    "confirmed_planet": (
        "confirmed_planet_flag",
        "is_confirmed_planet",
        "has_confirmed_planet",
        "confirmed_planet",
        "known_planet_flag",
        "planet_host_flag",
    ),
    "eclipsing_binary": (
        "eb_flag",
        "is_eb",
        "has_eb",
        "known_eb",
        "eclipsing_binary_flag",
        "is_eclipsing_binary",
    ),
}

FORBIDDEN_SELECTION_NAME_TOKENS = (
    "acf",
    "scatter",
    "spectral",
    "stationarity",
    "variability",
    "memory",
    "periodicity",
    "coherent",
    "quiet",
    "regime",
    "v2_",
)

REVIEW_FLAG_COLUMNS = (
    "v2_rotation_spot_review_flag",
    "v2_pulsation_review_flag",
)

CANDIDATE_FLAG_COLUMNS = (
    "v2_quiet_candidate",
    "v2_low_scatter_structured_candidate",
    "v2_correlated_stochastic_candidate",
    "v2_evolving_variability_candidate",
    "v2_quasi_periodic_candidate",
    "v2_coherent_periodic_candidate",
)


def normalize_target_id(value: object) -> str:
    text = str(value).upper().replace("KIC", "").strip()
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def first_existing(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    available = set(columns)
    return next((name for name in aliases if name in available), None)


def bool_series(series: pd.Series) -> pd.Series:
    """Parse common CSV boolean encodings conservatively."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()

    out = pd.Series(False, index=series.index, dtype=bool)
    finite_numeric = numeric.notna()
    out.loc[finite_numeric] = numeric.loc[finite_numeric] != 0
    out.loc[~finite_numeric] = text.loc[~finite_numeric].isin(
        {"true", "t", "yes", "y", "pass", "keep", "accepted", "eligible"}
    )
    return out


def validate_independent_selection_source(
    pool: pd.DataFrame,
    *,
    trust_clean_pool: bool,
) -> dict[str, object]:
    """Identify the non-statistical columns allowed to influence selection."""
    target_col = first_existing(pool.columns, TARGET_ID_ALIASES)
    if target_col is None:
        raise ValueError(
            "Candidate pool needs a target identifier column such as target_id/kepid/kic."
        )

    quarter_col = first_existing(pool.columns, QUARTER_ALIASES)
    clean_col = first_existing(pool.columns, CLEAN_PASS_ALIASES)

    veto_columns: dict[str, str] = {}
    for family, aliases in VETO_FLAG_ALIASES.items():
        match = first_existing(pool.columns, aliases)
        if match is not None:
            veto_columns[family] = match

    # Guard against accidentally using the old statistically stratified sample.
    suspicious_selection_columns = [
        column
        for column in pool.columns
        if column.lower() in {"selection_group", "statistical_behaviour", "dominant_behaviour", "regime"}
    ]
    if suspicious_selection_columns and not trust_clean_pool:
        for column in suspicious_selection_columns:
            values = {
                str(value).strip().lower()
                for value in pool[column].dropna().unique().tolist()
            }
            benign = {
                "",
                "unspecified",
                "clean",
                "clean_q5",
                "random",
                "random_clean_q5",
                "random_clean_q5_unstratified",
            }
            if values - benign:
                raise ValueError(
                    f"{column!r} contains non-benign selection strata {sorted(values - benign)[:10]}. "
                    "Use the upstream clean Q5 pool that existed BEFORE statistical stratification. "
                    "Only pass --trust-clean-pool if you have independently verified that this file "
                    "was not selected using light-curve statistics."
                )

    if clean_col is None and len(veto_columns) < len(VETO_FLAG_ALIASES) and not trust_clean_pool:
        missing = sorted(set(VETO_FLAG_ALIASES) - set(veto_columns))
        raise ValueError(
            "The pool does not expose a recognized positive clean flag and is missing "
            f"catalog veto families {missing}. To preserve the scientific selection contract, "
            "use the upstream clean-vetted pool or pass --trust-clean-pool only if the file "
            "was already cleaned independently of the statistical characterization."
        )

    statistical_columns_present_but_ignored = [
        column
        for column in pool.columns
        if any(token in column.lower() for token in FORBIDDEN_SELECTION_NAME_TOKENS)
    ]

    return {
        "target_column": target_col,
        "quarter_column": quarter_col,
        "clean_pass_column": clean_col,
        "veto_columns": veto_columns,
        "trust_clean_pool": bool(trust_clean_pool),
        "statistical_columns_present_but_ignored": statistical_columns_present_but_ignored,
    }


def prepare_clean_q5_pool(
    candidate_pool_path: Path,
    *,
    trust_clean_pool: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    pool = pd.read_csv(candidate_pool_path)
    provenance = validate_independent_selection_source(
        pool,
        trust_clean_pool=trust_clean_pool,
    )

    target_col = str(provenance["target_column"])
    quarter_col = provenance["quarter_column"]
    clean_col = provenance["clean_pass_column"]
    veto_columns = dict(provenance["veto_columns"])

    working = pool.copy()
    working["target_id"] = working[target_col].map(normalize_target_id)
    working = working[working["target_id"].str.len() > 0].copy()

    if quarter_col is not None:
        quarter_text = (
            working[str(quarter_col)]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace("Q", "", regex=False)
        )
        quarter_values = pd.to_numeric(quarter_text, errors="coerce")
        working = working[quarter_values == QUARTER].copy()
    working["quarter"] = QUARTER

    if clean_col is not None:
        working = working[bool_series(working[str(clean_col)])].copy()

    for _, column in veto_columns.items():
        working = working[~bool_series(working[column])].copy()

    # Target identity is the only row-level variable carried into the random
    # selector.  Statistical columns can exist in the source file but have zero
    # influence on inclusion/ranking.
    working = working[["target_id", "quarter"]].drop_duplicates().reset_index(drop=True)

    if len(working) < SAMPLE_SIZE:
        raise ValueError(
            f"Only {len(working)} unique clean Q5 targets remain; need at least {SAMPLE_SIZE}."
        )

    # Deterministic random order.  Data-availability failures are replaced by
    # the next target in this pre-generated order; no characterization value can
    # change the ordering.
    working = working.sample(frac=1.0, random_state=SELECTION_SEED).reset_index(drop=True)
    working["selection_order"] = np.arange(1, len(working) + 1, dtype=int)
    working["selection_group"] = "random_clean_q5_unstratified"

    provenance.update(
        {
            "candidate_pool_path": str(candidate_pool_path),
            "input_row_count": int(len(pool)),
            "clean_q5_unique_target_count": int(len(working)),
            "selection_seed": int(SELECTION_SEED),
            "quarter": int(QUARTER),
            "statistical_selection_columns_used": [],
            "selection_policy": (
                "Deterministic random order after Q5 + upstream clean/catalog vetoes only; "
                "no light-curve statistical feature or v2 label is used."
            ),
        }
    )
    return working, provenance


def cache_path(cache_dir: Path, target_id: str, quarter: int) -> Path:
    return cache_dir / f"kic_{normalize_target_id(target_id)}_q{int(quarter)}_pdcsap.parquet"


def load_frame(
    cache_dir: Path,
    target_id: str,
    quarter: int,
    *,
    allow_download: bool,
) -> tuple[pd.DataFrame, bool]:
    path = cache_path(cache_dir, target_id, quarter)
    if path.exists():
        return pd.read_parquet(path), True
    if not allow_download:
        raise FileNotFoundError(
            f"Cached light curve missing: {path}. Re-run with --allow-download."
        )

    light_curve = load_kepler_pdcsap(target_id, quarter)
    frame = light_curve.to_dataframe()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame, False


def worker_settings(args: argparse.Namespace) -> dict[str, object]:
    return {
        "cache_dir": str(args.cache_dir),
        "allow_download": bool(args.allow_download),
        "quality_policy": str(args.quality_policy),
        "require_finite_flux_error": bool(args.require_finite_flux_error),
        "test_fraction": float(args.test_fraction),
        "v1_acf_lags": int(args.v1_acf_lags),
        "v2_acf_lags": int(args.v2_acf_lags),
        "rolling_window": int(args.rolling_window),
        "v1_spectral_frequencies": int(args.v1_spectral_frequencies),
        "v2_spectral_frequencies": int(args.v2_spectral_frequencies),
        "stationarity_min_observations": int(args.stationarity_min_observations),
    }


def characterize_task(task: tuple[dict[str, object], dict[str, object]]) -> dict[str, object]:
    row, settings = task
    target_id = normalize_target_id(row["target_id"])
    quarter = int(row["quarter"])
    try:
        frame, cache_hit = load_frame(
            Path(str(settings["cache_dir"])),
            target_id,
            quarter,
            allow_download=bool(settings["allow_download"]),
        )
        regular, preprocessing = preprocess_pdcsap_light_curve(
            frame,
            quality_policy=str(settings["quality_policy"]),
            require_finite_flux_error=bool(settings["require_finite_flux_error"]),
            normalization_fit_fraction=1.0 - float(settings["test_fraction"]),
        )

        diagnostics = characterize_regularized_light_curve(
            regular,
            target_id=target_id,
            quarter=quarter,
            preprocessing_summary=preprocessing.to_dict(),
            acf_lags=int(settings["v1_acf_lags"]),
            rolling_window=int(settings["rolling_window"]),
            spectral_frequencies=int(settings["v1_spectral_frequencies"]),
            stationarity_min_observations=int(settings["stationarity_min_observations"]),
        )

        cadence_days = float(diagnostics["median_cadence_days"])
        v2 = stellar_variability_summary(
            regular["time"].to_numpy(dtype=float),
            regular["normalized_flux"].to_numpy(dtype=float),
            cadence_days=cadence_days,
            acf_lags=int(settings["v2_acf_lags"]),
            spectral_frequencies=int(settings["v2_spectral_frequencies"]),
        )

        diagnostics.update(v2)
        diagnostics["selection_order"] = int(row["selection_order"])
        diagnostics["selection_group"] = str(row["selection_group"])
        diagnostics["light_curve_cache_hit"] = bool(cache_hit)
        diagnostics["characterization_success"] = True
        return {
            "ok": True,
            "target_id": target_id,
            "quarter": quarter,
            "selection_order": int(row["selection_order"]),
            "diagnostics": diagnostics,
        }
    except Exception as exc:
        return {
            "ok": False,
            "target_id": target_id,
            "quarter": quarter,
            "selection_order": int(row["selection_order"]),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def resolve_worker_count(requested: int | None, reserve_cores: int, task_count: int) -> int:
    available = os.cpu_count() or 1
    default = max(1, available - max(0, int(reserve_cores)))
    value = default if requested is None else max(1, int(requested))
    return max(1, min(value, available, max(1, int(task_count))))


def characterize_batch(
    rows: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    records = rows.to_dict(orient="records")
    if not records:
        return [], [], 0

    workers = resolve_worker_count(args.max_workers, args.reserve_cpu_cores, len(records))
    settings = worker_settings(args)
    successes: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {
            executor.submit(characterize_task, (record, settings)): record
            for record in records
        }
        with tqdm(
            total=len(futures),
            desc="Characterizing clean Q5 stars",
            unit="star",
            dynamic_ncols=True,
        ) as progress:
            for future in as_completed(futures):
                result = future.result()
                if result["ok"]:
                    successes.append(result)
                    progress.set_postfix_str(f"KIC {result['target_id']}")
                else:
                    failures.append(result)
                    tqdm.write(
                        f"FAILED KIC {result['target_id']} Q{result['quarter']}: "
                        f"{result['error_type']}: {result['error']}"
                    )
                progress.update(1)

    return successes, failures, workers


def characterize_until_100(
    ordered_pool: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    successes: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    next_index = 0
    max_attempts = min(int(args.max_attempts), len(ordered_pool))
    worker_count = 0

    while len(successes) < SAMPLE_SIZE and next_index < max_attempts:
        needed = SAMPLE_SIZE - len(successes)
        batch_size = min(max(needed, 10), max_attempts - next_index)
        batch = ordered_pool.iloc[next_index : next_index + batch_size]
        batch_success, batch_failure, worker_count = characterize_batch(batch, args)
        successes.extend(batch_success)
        failures.extend(batch_failure)
        next_index += len(batch)

    success_frame = pd.DataFrame(
        [item["diagnostics"] for item in successes]
    )
    if not success_frame.empty:
        success_frame = (
            success_frame.sort_values("selection_order")
            .drop_duplicates(["target_id", "quarter"], keep="first")
            .head(SAMPLE_SIZE)
            .reset_index(drop=True)
        )

    failure_frame = pd.DataFrame(failures)
    if len(success_frame) != SAMPLE_SIZE:
        metrics = Path(args.output_dir) / "metrics"
        metrics.mkdir(parents=True, exist_ok=True)
        failure_frame.to_csv(metrics / "characterization_failures.csv", index=False)
        success_frame.to_csv(metrics / "characterization_partial_successes.csv", index=False)
        raise RuntimeError(
            f"Only {len(success_frame)} stars characterized successfully after "
            f"{next_index} attempts. Increase --max-attempts, enable --allow-download, "
            f"or inspect {metrics / 'characterization_failures.csv'}."
        )

    return success_frame, failure_frame, worker_count


def canonical_schema_frame() -> pd.DataFrame:
    descriptions = {
        "flux_robust_scale": (
            "Robust amplitude of normalized-flux variability.",
            "Robust scale (MAD-derived) of the finite normalized-flux distribution.",
        ),
        "flux_skewness": (
            "Asymmetry of the normalized-flux distribution.",
            "Bias-corrected sample skewness of finite normalized flux.",
        ),
        "flux_outlier_fraction": (
            "Fraction of robust tail excursions.",
            "Fraction of finite points beyond the configured robust outlier-sigma threshold.",
        ),
        "v2_acf_lag_1": (
            "Immediate cadence-to-cadence dependence.",
            "Gap-preserving pairwise correlation at one regular Kepler cadence.",
        ),
        "v2_acf_decay_e_days": (
            "Persistence timescale of the initial positive autocorrelation.",
            "First regular-grid lag where the pairwise ACF falls to <= exp(-1), in days.",
        ),
        "original_series_stationarity_conclusion": (
            "Joint ADF/KPSS stationarity interpretation.",
            "Conservative categorical conclusion from the two formal stationarity tests.",
        ),
        "v2_spectral_concentration": (
            "Concentration of variability power in the dominant spectral peak.",
            "Dominant Lomb-Scargle peak power divided by total sampled spectral power.",
        ),
        "v2_spectral_harmonic_power_ratio": (
            "Strength of simple harmonic support.",
            "Strongest sampled power near f/2 or 2f divided by dominant-peak power.",
        ),
        "v2_ls_dominant_period_days": (
            "Dominant coherent timescale.",
            "Period of the strongest sampled v2 Lomb-Scargle peak.",
        ),
        "v2_ls_acf_period_relative_error": (
            "Cross-diagnostic period agreement.",
            "Minimum harmonic-aware relative error between LS and ACF period candidates.",
        ),
        "v2_segment_scale_relative_mad": (
            "Evolution of variability amplitude through the quarter.",
            "Robust relative MAD of equal-time-segment robust-scale estimates.",
        ),
    }
    rows = []
    for domain, label, source, kind in CANONICAL_CHARACTERIZATION_SCHEMA:
        scientific_use, calculation = descriptions[source]
        rows.append(
            {
                "scientific_domain": domain,
                "canonical_variable": label,
                "source_column": source,
                "kind": kind,
                "calculation": calculation,
                "scientific_use": scientific_use,
            }
        )
    return pd.DataFrame(rows)


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
        *CANONICAL_CHARACTERIZATION_COLUMNS,
        *[column for column in interpretation_columns if column in profiled.columns],
    ]
    columns = [column for column in columns if column in profiled.columns]
    return profiled[columns].sort_values("selection_order").reset_index(drop=True)


def numeric_distribution_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in CANONICAL_CONTINUOUS_FEATURE_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "feature": column,
                "n_total": int(len(values)),
                "n_finite": int(len(finite)),
                "missing_fraction": float(1.0 - len(finite) / len(values)) if len(values) else np.nan,
                "min": float(finite.min()) if len(finite) else np.nan,
                "q25": float(finite.quantile(0.25)) if len(finite) else np.nan,
                "median": float(finite.median()) if len(finite) else np.nan,
                "q75": float(finite.quantile(0.75)) if len(finite) else np.nan,
                "max": float(finite.max()) if len(finite) else np.nan,
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "std": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in CANONICAL_CHARACTERIZATION_COLUMNS:
        if column not in frame.columns:
            n_missing = len(frame)
        elif column in CANONICAL_CONTINUOUS_FEATURE_COLUMNS:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            n_missing = int((~np.isfinite(numeric)).sum())
        else:
            values = frame[column]
            normalized = values.fillna("").astype(str).str.strip().str.lower()
            missing_mask = values.isna() | normalized.isin({"", "unknown", "nan", "none"})
            n_missing = int(missing_mask.sum())
        rows.append(
            {
                "feature": column,
                "n_total": int(len(frame)),
                "n_missing": int(n_missing),
                "missing_fraction": float(n_missing / len(frame)) if len(frame) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def redundancy_pairs(spearman: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    rows = []
    columns = list(spearman.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            rho = spearman.loc[left, right]
            if np.isfinite(rho) and abs(float(rho)) >= float(threshold):
                rows.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "spearman_rho": float(rho),
                        "abs_spearman_rho": abs(float(rho)),
                    }
                )
    result = pd.DataFrame(
        rows,
        columns=["feature_a", "feature_b", "spearman_rho", "abs_spearman_rho"],
    )
    if not result.empty:
        result = result.sort_values("abs_spearman_rho", ascending=False).reset_index(drop=True)
    return result


def boolean_count_table(frame: pd.DataFrame, columns: Iterable[str], label_name: str) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in frame.columns:
            continue
        flag = frame[column].fillna(False).astype(bool)
        rows.append(
            {
                label_name: column,
                "count": int(flag.sum()),
                "fraction": float(flag.mean()) if len(flag) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def categorical_counts(frame: pd.DataFrame, column: str, label_name: str) -> pd.DataFrame:
    if column not in frame.columns:
        return pd.DataFrame(columns=[label_name, "count", "fraction"])
    counts = frame[column].fillna("missing").astype(str).value_counts(dropna=False)
    rows = [
        {
            label_name: label,
            "count": int(count),
            "fraction": float(count / len(frame)) if len(frame) else np.nan,
        }
        for label, count in counts.items()
    ]
    return pd.DataFrame(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def write_outputs(
    profiled: pd.DataFrame,
    failures: pd.DataFrame,
    thresholds: dict[str, float],
    provenance: dict[str, object],
    args: argparse.Namespace,
    worker_count: int,
) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    metrics = output_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)

    canonical = build_canonical_table(profiled)
    continuous = canonical[
        ["target_id", "quarter", *CANONICAL_CONTINUOUS_FEATURE_COLUMNS]
    ].copy()

    paths = {
        "master": metrics / "characterization_master_100.csv",
        "canonical_100": metrics / "stellar_features_v2_100.csv",
        "canonical_latest": metrics / "stellar_features_v2.csv",
        "continuous_100": metrics / "stellar_features_v2_continuous_100.csv",
        "continuous_latest": metrics / "stellar_features_v2_continuous.csv",
        "schema": metrics / "canonical_feature_schema.csv",
        "distribution": metrics / "canonical_feature_distribution_summary.csv",
        "missingness": metrics / "canonical_missingness_summary.csv",
        "spearman": metrics / "canonical_spearman_correlation.csv",
        "pearson": metrics / "canonical_pearson_correlation.csv",
        "redundancy": metrics / "canonical_redundancy_pairs.csv",
        "behaviour_counts": metrics / "dominant_statistical_behaviour_counts.csv",
        "candidate_flag_counts": metrics / "candidate_flag_counts.csv",
        "review_flag_counts": metrics / "review_flag_counts.csv",
        "stationarity_counts": metrics / "stationarity_state_counts.csv",
        "amplitude_counts": metrics / "amplitude_population_counts.csv",
        "memory_counts": metrics / "memory_population_counts.csv",
        "thresholds": metrics / "population_boundaries_100.json",
        "freeze": metrics / "characterization_v2_freeze.json",
        "failures": metrics / "characterization_failures.csv",
        "manifest_used": metrics / "target_manifest_used.csv",
    }

    profiled.to_csv(paths["master"], index=False)
    canonical.to_csv(paths["canonical_100"], index=False)
    canonical.to_csv(paths["canonical_latest"], index=False)
    continuous.to_csv(paths["continuous_100"], index=False)
    continuous.to_csv(paths["continuous_latest"], index=False)
    canonical_schema_frame().to_csv(paths["schema"], index=False)
    numeric_distribution_summary(canonical).to_csv(paths["distribution"], index=False)
    missingness_summary(canonical).to_csv(paths["missingness"], index=False)

    numeric = continuous[list(CANONICAL_CONTINUOUS_FEATURE_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    spearman = numeric.corr(method="spearman")
    pearson = numeric.corr(method="pearson")
    spearman.to_csv(paths["spearman"])
    pearson.to_csv(paths["pearson"])
    redundancy_pairs(spearman).to_csv(paths["redundancy"], index=False)

    behaviour = categorical_counts(
        canonical,
        "v2_dominant_statistical_behaviour",
        "dominant_statistical_behaviour",
    )
    if not behaviour.empty:
        behaviour["_order"] = behaviour["dominant_statistical_behaviour"].map(
            {label: i for i, label in enumerate(DOMINANT_STATISTICAL_BEHAVIOUR_ORDER)}
        ).fillna(999)
        behaviour = behaviour.sort_values(["_order", "dominant_statistical_behaviour"]).drop(columns="_order")
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

    manifest = canonical[["target_id", "quarter"]].copy()
    manifest["selection_group"] = "random_clean_q5_unstratified"
    manifest.to_csv(paths["manifest_used"], index=False)

    manifest_out = Path(args.manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_out, index=False)

    paths["thresholds"].write_text(
        json.dumps({key: float(value) if np.isfinite(value) else None for key, value in thresholds.items()}, indent=2)
        + "\n"
    )

    freeze = {
        "freeze_id": V2_FREEZE_ID,
        "scientific_characterization_version": "stellar_variability_v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "sample_size": int(len(canonical)),
        "quarter": int(QUARTER),
        "selection_seed": int(SELECTION_SEED),
        "selection_provenance": provenance,
        "population_boundary_config": asdict(DEFAULT_BOUNDARIES),
        "realized_population_boundaries": {
            key: float(value) if np.isfinite(value) else None
            for key, value in thresholds.items()
        },
        "canonical_domains": sorted({item[0] for item in CANONICAL_CHARACTERIZATION_SCHEMA}),
        "canonical_feature_count": int(len(CANONICAL_CHARACTERIZATION_COLUMNS)),
        "canonical_columns": list(CANONICAL_CHARACTERIZATION_COLUMNS),
        "canonical_continuous_columns": list(CANONICAL_CONTINUOUS_FEATURE_COLUMNS),
        "worker_count": int(worker_count),
        "characterization_parameters": {
            "quality_policy": args.quality_policy,
            "require_finite_flux_error": bool(args.require_finite_flux_error),
            "test_fraction": float(args.test_fraction),
            "v1_acf_lags": int(args.v1_acf_lags),
            "v2_acf_lags": int(args.v2_acf_lags),
            "rolling_window": int(args.rolling_window),
            "v1_spectral_frequencies": int(args.v1_spectral_frequencies),
            "v2_spectral_frequencies": int(args.v2_spectral_frequencies),
            "stationarity_min_observations": int(args.stationarity_min_observations),
        },
        "selection_contract": (
            "No statistical characterization feature/label is used to select the 100 targets. "
            "Population boundaries are computed only after the final 100 successful stars are fixed."
        ),
    }

    # Add hashes after the files exist so the frozen experiment can be audited.
    freeze["sha256"] = {
        "target_manifest_used.csv": sha256_file(paths["manifest_used"]),
        "stellar_features_v2_100.csv": sha256_file(paths["canonical_100"]),
        "characterization_master_100.csv": sha256_file(paths["master"]),
    }
    paths["freeze"].write_text(json.dumps(freeze, indent=2) + "\n")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select and characterize 100 clean Q5 stars with the frozen v2 workflow."
    )
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        required=True,
        help=(
            "Upstream clean Q5 candidate pool from BEFORE statistical stratification. "
            "Do not pass the old 50-star regime/behaviour manifest."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_OUT)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--trust-clean-pool", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--reserve-cpu-cores", type=int, default=2)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=150,
        help="Maximum targets attempted in deterministic random order to obtain 100 successes.",
    )

    # Freeze these defaults for the 100-star experiment. They can still be
    # changed explicitly, and the realized values are written to freeze JSON.
    parser.add_argument("--quality-policy", default="default")
    parser.add_argument("--require-finite-flux-error", action="store_true")
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--v1-acf-lags", type=int, default=80)
    parser.add_argument("--v2-acf-lags", type=int, default=240)
    parser.add_argument("--rolling-window", type=int, default=96)
    parser.add_argument("--v1-spectral-frequencies", type=int, default=2000)
    parser.add_argument("--v2-spectral-frequencies", type=int, default=4000)
    parser.add_argument("--stationarity-min-observations", type=int, default=24)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.candidate_pool = Path(args.candidate_pool).expanduser().resolve()
    args.output_dir = Path(args.output_dir)
    args.cache_dir = Path(args.cache_dir)
    args.manifest_out = Path(args.manifest_out)

    if not args.candidate_pool.exists():
        raise FileNotFoundError(args.candidate_pool)
    if int(args.max_attempts) < SAMPLE_SIZE:
        raise ValueError(f"--max-attempts must be at least {SAMPLE_SIZE}.")

    ordered_pool, provenance = prepare_clean_q5_pool(
        args.candidate_pool,
        trust_clean_pool=bool(args.trust_clean_pool),
    )

    print()
    print("Frozen v2 100-star characterization")
    print("------------------------------------")
    print(f"Candidate pool: {args.candidate_pool}")
    print(f"Clean Q5 unique targets available: {len(ordered_pool)}")
    print(f"Selection seed: {SELECTION_SEED}")
    print("Statistical selection variables used: NONE")
    print()

    # Create the metrics directory now so failure diagnostics can be written even
    # if the run cannot reach 100 successes.
    metrics = Path(args.output_dir) / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)

    try:
        raw_features, failures, workers = characterize_until_100(ordered_pool, args)
    except RuntimeError:
        # characterize_until_100 raises only after it has exhausted the allowed
        # attempts. Re-run individual failures can then be diagnosed from stdout.
        raise

    # Critical ordering: population boundaries are computed ONCE, on the final
    # complete 100-star sample. No 50-star quantile is carried forward.
    profiled, thresholds = apply_population_variability_boundaries(
        raw_features,
        boundaries=DEFAULT_BOUNDARIES,
    )
    profiled = assign_dominant_statistical_behaviour(profiled)
    profiled = profiled.sort_values("selection_order").reset_index(drop=True)

    paths = write_outputs(
        profiled,
        failures,
        thresholds,
        provenance,
        args,
        workers,
    )

    print()
    print("100-star v2 characterization complete.")
    print(f"Output directory: {Path(args.output_dir) / 'metrics'}")
    print(f"Frozen target manifest: {args.manifest_out}")
    print()
    print("Population boundaries from the FULL 100-star sample:")
    for key, value in thresholds.items():
        print(f"  {key}: {value}")
    print()
    print("Dominant statistical behaviour counts:")
    behaviour = pd.read_csv(paths["behaviour_counts"])
    if behaviour.empty:
        print("  <none>")
    else:
        for _, row in behaviour.iterrows():
            print(
                f"  {row['dominant_statistical_behaviour']}: "
                f"{int(row['count'])} ({100.0 * float(row['fraction']):.1f}%)"
            )
    print()
    print("Key files:")
    for key in (
        "canonical_100",
        "distribution",
        "missingness",
        "spearman",
        "redundancy",
        "behaviour_counts",
        "freeze",
    ):
        print(f"  {key}: {paths[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
