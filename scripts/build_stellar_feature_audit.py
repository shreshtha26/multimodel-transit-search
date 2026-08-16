#!/usr/bin/env python3
"""
Build the canonical stellar-characterization feature set from the full
diagnostic table.

Purpose
-------
The characterization pipeline intentionally keeps many low-level diagnostics
for validation and debugging.  Those diagnostics are NOT all independent
scientific features.  This script separates three layers:

1. Raw/audit diagnostics
2. Seven scientific domains
3. A compact canonical feature vector (~11 variables)

It does not run PCA.  PCA should only be considered after the canonical
scientific variables have been audited for redundancy.

Default input
-------------
outputs/experiments/characterization_validation50/metrics/
    characterization_master_50.csv

Default output
--------------
outputs/experiments/characterization_feature_audit/metrics/

The script is conservative:
- exact aliases are preferred;
- semantic fallback is scored and reported;
- ambiguous or unresolved canonical variables are left as NaN;
- no threshold is silently retuned from the 10-star validation set.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "outputs/experiments/characterization_validation50/metrics/"
    "characterization_master_50.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/experiments/characterization_feature_audit/metrics"
)


# ---------------------------------------------------------------------------
# Seven permanent scientific domains
# ---------------------------------------------------------------------------

DOMAIN_ORDER = (
    "scatter_amplitude",
    "distribution_shape",
    "autocorrelation_memory",
    "stationarity",
    "spectral_structure",
    "periodicity_coherence",
    "variance_evolution",
)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    domain: str
    kind: str
    unit: str
    rationale: str
    aliases: tuple[str, ...]
    positive_keywords: tuple[str, ...]
    negative_keywords: tuple[str, ...] = ()
    minimum_score: float = 6.0


CANONICAL_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="robust_scatter",
        domain="scatter_amplitude",
        kind="continuous",
        unit="relative_flux",
        rationale=(
            "Robust amplitude of stellar-background variability; preferred "
            "over ordinary standard deviation when isolated excursions exist."
        ),
        aliases=(
            # Existing characterization field: global robust flux scale.
            "flux_robust_scale",
            "v2_robust_scatter",
            "robust_scatter",
            "v2_robust_sigma",
            "robust_sigma",
            "flux_robust_sigma",
            "scaled_mad",
            "mad_sigma",
            "flux_mad_scaled",
            "flux_mad",
        ),
        positive_keywords=("robust", "scatter", "sigma", "mad"),
        negative_keywords=("label", "population", "candidate", "threshold"),
        minimum_score=7.0,
    ),
    FeatureSpec(
        name="skewness",
        domain="distribution_shape",
        kind="continuous",
        unit="dimensionless",
        rationale="Asymmetry of the flux distribution.",
        aliases=(
            "v2_flux_skewness",
            "flux_skewness",
            "v2_skewness",
            "skewness",
        ),
        positive_keywords=("skew",),
        negative_keywords=("population", "label"),
        minimum_score=6.0,
    ),
    FeatureSpec(
        name="outlier_fraction",
        domain="distribution_shape",
        kind="continuous",
        unit="fraction",
        rationale=(
            "Robust tail/impulsive-variability measure. Raw excess kurtosis is "
            "retained as an audit diagnostic rather than a canonical router feature."
        ),
        aliases=(
            "v2_outlier_fraction",
            "outlier_fraction",
            "v2_outlier_fraction_5sigma",
            "outlier_fraction_5sigma",
            "robust_outlier_fraction",
            "flux_outlier_fraction",
        ),
        positive_keywords=("outlier", "fraction"),
        negative_keywords=("count", "n_outlier", "threshold"),
        minimum_score=7.0,
    ),
    FeatureSpec(
        name="acf_lag1",
        domain="autocorrelation_memory",
        kind="continuous",
        unit="correlation",
        rationale="Short-lag correlation strength of the stellar background.",
        aliases=(
            "v2_acf_lag_1",
            "v2_acf_lag1",
            "acf_lag_1_v2",
            "acf_lag1_v2",
            "acf_lag_1",
            "acf_lag1",
        ),
        positive_keywords=("acf", "lag", "1"),
        negative_keywords=("candidate", "period", "relative", "max"),
        minimum_score=8.0,
    ),
    FeatureSpec(
        name="acf_timescale_days",
        domain="autocorrelation_memory",
        kind="continuous",
        unit="days",
        rationale=(
            "Persistence timescale of correlated variability; distinct from "
            "one-cadence ACF strength."
        ),
        aliases=(
            # Canonical memory timescale = e-folding ACF decay time.
            "v2_acf_decay_e_days",
            "v2_acf_timescale_days",
            "acf_timescale_days",
            "v2_acf_decay_timescale_days",
            "acf_decay_timescale_days",
            "v2_acf_e_folding_time_days",
            "acf_e_folding_time_days",
            "memory_timescale_days",
            "acf_correlation_time_days",
        ),
        positive_keywords=("acf", "timescale", "day", "decay", "memory"),
        negative_keywords=("period_candidate", "candidate_period"),
        minimum_score=7.0,
    ),
    FeatureSpec(
        name="stationarity_state",
        domain="stationarity",
        kind="categorical",
        unit="category",
        rationale=(
            "Joint interpretation of ADF/KPSS evidence; raw ADF/KPSS "
            "statistics and p-values remain supporting diagnostics."
        ),
        aliases=(
            # Final joint ADF/KPSS conclusion for the original
            # (undifferenced) series; "original" does not mean v1.
            "original_series_stationarity_conclusion",
            "v2_stationarity_state",
            "v2_stationarity_label",
            "v2_stationarity_classification",
            "stationarity_state",
            "stationarity_label",
            "stationarity_classification",
            "stationarity_interpretation",
            "stationarity",
        ),
        positive_keywords=("stationar", "state", "label", "classification", "interpret"),
        negative_keywords=("adf", "kpss", "statistic", "pvalue", "p_value", "flag"),
        minimum_score=6.0,
    ),
    FeatureSpec(
        name="spectral_concentration",
        domain="spectral_structure",
        kind="continuous",
        unit="fraction",
        rationale=(
            "Degree to which spectral power is concentrated rather than broadly "
            "distributed across frequencies."
        ),
        aliases=(
            "v2_spectral_concentration",
            "spectral_concentration",
            "v2_spectral_power_concentration",
            "spectral_power_concentration",
            "periodogram_concentration",
        ),
        positive_keywords=("spectral", "concentration", "periodogram"),
        negative_keywords=("band", "0_5", "2d", "10d"),
        minimum_score=7.0,
    ),
    FeatureSpec(
        name="harmonic_power_ratio",
        domain="spectral_structure",
        kind="continuous",
        unit="ratio",
        rationale="Strength of harmonically related spectral structure.",
        aliases=(
            "v2_spectral_harmonic_power_ratio",
            "v2_harmonic_power_ratio",
            "harmonic_power_ratio",
            "v2_harmonic_ratio",
            "harmonic_ratio",
        ),
        positive_keywords=("harmonic", "power", "ratio"),
        negative_keywords=("period_relative_error",),
        minimum_score=7.0,
    ),
    FeatureSpec(
        name="dominant_period_days",
        domain="periodicity_coherence",
        kind="continuous",
        unit="days",
        rationale=(
            "Final v2 Lomb-Scargle dominant timescale. Legacy "
            "dominant_period_days is intentionally lower priority."
        ),
        aliases=(
            "v2_ls_dominant_period_days",
            "v2_dominant_period_days",
            "dominant_period_v2_days",
            "v2_ls_period_days",
        ),
        positive_keywords=("v2", "ls", "dominant", "period", "day"),
        negative_keywords=("acf", "candidate", "relative_error", "fap"),
        minimum_score=9.0,
    ),
    FeatureSpec(
        name="period_agreement_error",
        domain="periodicity_coherence",
        kind="continuous",
        unit="relative_error",
        rationale=(
            "Harmonic-aware disagreement between LS and ACF timescales; lower "
            "values imply stronger independent support for the same periodic structure."
        ),
        aliases=(
            "v2_ls_acf_period_relative_error",
            "ls_acf_period_relative_error",
            "v2_period_agreement_error",
            "period_agreement_error",
        ),
        positive_keywords=("ls", "acf", "period", "relative", "error"),
        negative_keywords=("panel",),
        minimum_score=9.0,
    ),
    FeatureSpec(
        name="segment_scale_variability",
        domain="variance_evolution",
        kind="continuous",
        unit="dimensionless",
        rationale=(
            "Robust segment-to-segment variability of the local flux scale, "
            "capturing amplitude/variance nonstationarity across the quarter "
            "without implying a monotonic drift."
        ),
        aliases=(
            "v2_segment_scale_relative_mad",
            "segment_scale_relative_mad",
            "v2_variance_drift",
            "variance_drift",
            "v2_variance_drift_score",
            "variance_drift_score",
            "rolling_variance_drift",
            "segment_variance_drift",
            "rolling_variance_cv",
            "segment_variance_cv",
            "variance_cv",
        ),
        positive_keywords=("segment", "scale", "relative", "mad", "variance"),
        negative_keywords=("mean", "median", "mu", "window_count"),
        minimum_score=7.0,
    ),
)


IDENTIFIER_ALIASES = (
    "target_id",
    "kic",
    "kic_id",
    "kepid",
    "quarter",
    "sample_role",
    "sample_stratum",
    "cohort",
    "set",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_name(name: str) -> str:
    text = name.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _safe_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return pd.Series(np.nan, index=series.index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _is_bool_like(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    values = series.dropna()
    if values.empty:
        return False
    lowered = set(values.astype(str).str.strip().str.lower().unique())
    return lowered <= {"true", "false", "0", "1", "yes", "no"}


def _finite_count(series: pd.Series) -> int:
    values = _safe_numeric(series)
    return int(np.isfinite(values.to_numpy(dtype=float)).sum())


def _first_existing(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    by_norm = {_normalise_name(c): c for c in columns}
    for alias in aliases:
        col = by_norm.get(_normalise_name(alias))
        if col is not None:
            return col
    return None


def _keyword_score(column: str, spec: FeatureSpec) -> float:
    name = _normalise_name(column)
    tokens = set(name.split("_"))
    score = 0.0

    for keyword in spec.positive_keywords:
        key = _normalise_name(keyword)
        if not key:
            continue
        if key == "1":
            if "1" in tokens:
                score += 2.0
            continue
        if key in name:
            score += 2.0
        if key in tokens:
            score += 1.0

    for keyword in spec.negative_keywords:
        key = _normalise_name(keyword)
        if key and key in name:
            score -= 4.0

    # Final v2 definitions are preferred whenever the logical feature is v2-based.
    if name.startswith("v2_") or "_v2_" in name or name.endswith("_v2"):
        score += 1.5

    # Avoid generic legacy dominant_period_days from masquerading as v2.
    if spec.name == "dominant_period_days" and name == "dominant_period_days":
        score -= 20.0

    return score


def resolve_feature(
    df: pd.DataFrame,
    spec: FeatureSpec,
) -> tuple[str | None, str, float, list[tuple[str, float]]]:
    """
    Resolve one canonical feature.

    Returns
    -------
    selected_column, status, score, ranked_candidates

    status is one of:
      exact
      semantic
      ambiguous
      unresolved
    """
    columns = list(df.columns)
    by_norm = {_normalise_name(c): c for c in columns}

    for alias in spec.aliases:
        match = by_norm.get(_normalise_name(alias))
        if match is None:
            continue

        if spec.kind == "continuous":
            if _finite_count(df[match]) == 0:
                continue
        elif df[match].dropna().empty:
            continue

        return match, "exact", 100.0, [(match, 100.0)]

    ranked: list[tuple[str, float]] = []
    for col in columns:
        series = df[col]

        if spec.kind == "continuous":
            if _finite_count(series) == 0:
                continue
        elif series.dropna().empty:
            continue

        score = _keyword_score(col, spec)
        if score > 0:
            ranked.append((col, score))

    ranked.sort(key=lambda item: (-item[1], item[0]))

    if not ranked:
        return None, "unresolved", math.nan, []

    best_col, best_score = ranked[0]
    if best_score < spec.minimum_score:
        return None, "unresolved", best_score, ranked[:5]

    if len(ranked) > 1:
        second_score = ranked[1][1]
        if second_score >= spec.minimum_score and (best_score - second_score) < 2.0:
            return None, "ambiguous", best_score, ranked[:5]

    return best_col, "semantic", best_score, ranked[:5]


# ---------------------------------------------------------------------------
# Domain assignment for every diagnostic column
# ---------------------------------------------------------------------------

DOMAIN_PATTERNS: dict[str, tuple[str, ...]] = {
    "scatter_amplitude": (
        "scatter",
        "robust_sigma",
        "sigma",
        "std",
        "mad",
        "amplitude",
        "iqr",
        "range",
    ),
    "distribution_shape": (
        "skew",
        "kurt",
        "outlier",
        "tail",
        "quantile",
        "percentile",
    ),
    "autocorrelation_memory": (
        "acf",
        "autocorr",
        "autocorrelation",
        "memory",
        "correlation_time",
    ),
    "stationarity": (
        "stationar",
        "adf",
        "kpss",
    ),
    "spectral_structure": (
        "spectral",
        "power_fraction",
        "frequency",
        "harmonic_power",
        "harmonic_ratio",
        "periodogram_concentration",
    ),
    "periodicity_coherence": (
        "dominant_period",
        "ls_acf",
        "periodic",
        "coherent",
        "lomb_scargle",
        "ls_fap",
        "period_candidate",
        "period_relative_error",
    ),
    "variance_evolution": (
        "variance_drift",
        "rolling_variance",
        "segment_variance",
        "variance_cv",
        "local_variance",
        "variance_ratio",
    ),
}


def diagnostic_domains(column: str) -> list[str]:
    name = _normalise_name(column)
    matched: list[str] = []

    for domain in DOMAIN_ORDER:
        patterns = DOMAIN_PATTERNS[domain]
        if any(_normalise_name(pattern) in name for pattern in patterns):
            matched.append(domain)

    # Generic "period" is too broad; use it only when another spectral/ACF
    # domain did not already classify the field.
    if (
        "period" in name
        and "periodicity_coherence" not in matched
        and not any(
            token in name
            for token in (
                "spectral_power_fraction",
                "acf_period_candidate",
            )
        )
    ):
        matched.append("periodicity_coherence")

    return matched


def role_for_column(
    column: str,
    series: pd.Series,
    canonical_sources: set[str],
    all_columns: set[str],
) -> str:
    name = _normalise_name(column)

    if column in canonical_sources:
        return "canonical_source"

    if name in {_normalise_name(a) for a in IDENTIFIER_ALIASES}:
        return "bookkeeping_identifier"

    if any(
        token in name
        for token in (
            "target_id",
            "quarter",
            "sample_stratum",
            "sample_role",
            "cohort",
            "source",
            "reason",
            "review",
            "population_label",
        )
    ):
        return "bookkeeping_or_label"

    if _is_bool_like(series) or any(
        token in name
        for token in (
            "_flag",
            "_candidate",
            "_supported",
            "_classification",
            "_label",
        )
    ):
        return "derived_classification"

    # Common development pattern: old quantity and v2_<old quantity> coexist.
    if not name.startswith("v2_"):
        v2_name = f"v2_{name}"
        if any(_normalise_name(c) == v2_name for c in all_columns):
            return "legacy_v1"

    if (
        name == "dominant_period_days"
        and any(_normalise_name(c) == "v2_ls_dominant_period_days" for c in all_columns)
    ):
        return "legacy_v1"

    return "supporting_diagnostic"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_column_inventory(
    df: pd.DataFrame,
    resolutions: dict[str, dict],
) -> pd.DataFrame:
    canonical_sources = {
        entry["source_column"]
        for entry in resolutions.values()
        if entry["source_column"] is not None
    }
    all_columns = set(df.columns)

    rows = []
    for col in df.columns:
        series = df[col]
        domains = diagnostic_domains(col)
        numeric = _safe_numeric(series)
        finite = np.isfinite(numeric.to_numpy(dtype=float))

        rows.append(
            {
                "column": col,
                "normalized_name": _normalise_name(col),
                "dtype": str(series.dtype),
                "n_rows": len(series),
                "n_non_null": int(series.notna().sum()),
                "n_numeric_finite": int(finite.sum()),
                "unique_non_null": int(series.dropna().nunique()),
                "bool_like": _is_bool_like(series),
                "role": role_for_column(
                    col,
                    series,
                    canonical_sources,
                    all_columns,
                ),
                "scientific_domain": domains[0] if len(domains) == 1 else (
                    "ambiguous" if len(domains) > 1 else "unassigned"
                ),
                "matched_domains": "|".join(domains),
            }
        )

    return pd.DataFrame(rows)


def distribution_summary(features: pd.DataFrame, continuous: list[str]) -> pd.DataFrame:
    rows = []
    for col in continuous:
        values = pd.to_numeric(features[col], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            rows.append(
                {
                    "feature": col,
                    "n_finite": 0,
                    "missing_fraction": 1.0,
                }
            )
            continue

        q = finite.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        rows.append(
            {
                "feature": col,
                "n_finite": int(len(finite)),
                "missing_fraction": float(1.0 - len(finite) / len(features)),
                "mean": float(finite.mean()),
                "std": float(finite.std(ddof=1)) if len(finite) > 1 else math.nan,
                "min": float(finite.min()),
                "p05": float(q.loc[0.05]),
                "p25": float(q.loc[0.25]),
                "median": float(q.loc[0.5]),
                "p75": float(q.loc[0.75]),
                "p95": float(q.loc[0.95]),
                "max": float(finite.max()),
                "n_unique": int(finite.nunique()),
            }
        )

    return pd.DataFrame(rows)


def redundancy_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = corr.loc[left, right]
            if pd.isna(value):
                continue
            if abs(float(value)) >= threshold:
                rows.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "spearman_rho": float(value),
                        "abs_spearman_rho": abs(float(value)),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=(
                "feature_a",
                "feature_b",
                "spearman_rho",
                "abs_spearman_rho",
            )
        )

    return pd.DataFrame(rows).sort_values(
        "abs_spearman_rho",
        ascending=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collapse the full characterization diagnostics into seven "
            "scientific domains and a compact canonical feature vector."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Master characterization CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output metrics directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.90,
        help="Absolute Spearman correlation used to flag redundancy (default: 0.90)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if any canonical feature is unresolved/ambiguous. Without "
            "--strict, unresolved features are emitted as NaN and clearly reported."
        ),
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Characterization master table not found: {args.input}"
        )

    if not 0.0 < args.corr_threshold <= 1.0:
        raise ValueError("--corr-threshold must be in (0, 1].")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if df.empty:
        raise ValueError(f"Input table is empty: {args.input}")

    # ------------------------------------------------------------------
    # Resolve the canonical feature sources.
    # ------------------------------------------------------------------
    resolutions: dict[str, dict] = {}
    resolution_rows = []

    for spec in CANONICAL_FEATURES:
        source, status, score, ranked = resolve_feature(df, spec)
        resolutions[spec.name] = {
            "source_column": source,
            "status": status,
            "score": score,
        }

        resolution_rows.append(
            {
                "canonical_feature": spec.name,
                "domain": spec.domain,
                "kind": spec.kind,
                "unit": spec.unit,
                "source_column": source,
                "resolution_status": status,
                "resolution_score": score,
                "top_candidates": json.dumps(
                    [
                        {"column": col, "score": candidate_score}
                        for col, candidate_score in ranked
                    ]
                ),
                "rationale": spec.rationale,
            }
        )

    resolution_df = pd.DataFrame(resolution_rows)

    unresolved = resolution_df[
        resolution_df["resolution_status"].isin(["unresolved", "ambiguous"])
    ]

    if args.strict and not unresolved.empty:
        print(resolution_df.to_string(index=False))
        raise RuntimeError(
            "Strict mode: one or more canonical features are unresolved/ambiguous."
        )

    # ------------------------------------------------------------------
    # Build compact feature table.
    # ------------------------------------------------------------------
    id_columns = []
    for alias in IDENTIFIER_ALIASES:
        match = _first_existing(df.columns, (alias,))
        if match is not None and match not in id_columns:
            id_columns.append(match)

    # Always preserve target_id / KIC-like identity if one exists.
    if not id_columns:
        for col in df.columns:
            name = _normalise_name(col)
            if "target" in name or "kic" in name or "kepid" in name:
                id_columns.append(col)

    features = df[id_columns].copy() if id_columns else pd.DataFrame(index=df.index)

    for spec in CANONICAL_FEATURES:
        source = resolutions[spec.name]["source_column"]

        if source is None:
            if spec.kind == "continuous":
                features[spec.name] = np.nan
            else:
                features[spec.name] = pd.Series(
                    pd.NA,
                    index=df.index,
                    dtype="string",
                )
            continue

        if spec.kind == "continuous":
            features[spec.name] = pd.to_numeric(
                df[source],
                errors="coerce",
            )
        else:
            features[spec.name] = df[source].astype("string")

    # ------------------------------------------------------------------
    # Classify every raw diagnostic column.
    # ------------------------------------------------------------------
    inventory = build_column_inventory(df, resolutions)

    domain_summary = (
        inventory.groupby(["scientific_domain", "role"], dropna=False)
        .size()
        .rename("n_columns")
        .reset_index()
        .sort_values(["scientific_domain", "role"])
    )

    # ------------------------------------------------------------------
    # Redundancy diagnostics on continuous canonical variables only.
    # ------------------------------------------------------------------
    continuous = [
        spec.name
        for spec in CANONICAL_FEATURES
        if spec.kind == "continuous"
    ]

    continuous_matrix = features[continuous].apply(
        pd.to_numeric,
        errors="coerce",
    )

    spearman = continuous_matrix.corr(
        method="spearman",
        min_periods=5,
    )
    pearson = continuous_matrix.corr(
        method="pearson",
        min_periods=5,
    )
    redundant = redundancy_pairs(spearman, args.corr_threshold)
    dist_summary = distribution_summary(features, continuous)

    # ------------------------------------------------------------------
    # Write outputs.
    # ------------------------------------------------------------------
    inventory.to_csv(out_dir / "diagnostic_column_inventory.csv", index=False)
    domain_summary.to_csv(out_dir / "diagnostic_domain_summary.csv", index=False)
    resolution_df.to_csv(out_dir / "canonical_feature_schema.csv", index=False)
    features.to_csv(out_dir / "stellar_features_v2_50.csv", index=False)
    continuous_matrix.to_csv(
        out_dir / "stellar_features_v2_continuous_50.csv",
        index=False,
    )
    spearman.to_csv(out_dir / "canonical_spearman_correlation.csv")
    pearson.to_csv(out_dir / "canonical_pearson_correlation.csv")
    redundant.to_csv(out_dir / "canonical_redundancy_pairs.csv", index=False)
    dist_summary.to_csv(
        out_dir / "canonical_feature_distribution_summary.csv",
        index=False,
    )

    # A tiny machine-readable schema for downstream code.
    schema_json = {
        "schema_version": "stellar_characterization_v2",
        "scientific_domains": list(DOMAIN_ORDER),
        "canonical_features": [
            {
                "name": spec.name,
                "domain": spec.domain,
                "kind": spec.kind,
                "unit": spec.unit,
                "source_column": resolutions[spec.name]["source_column"],
                "resolution_status": resolutions[spec.name]["status"],
                "rationale": spec.rationale,
            }
            for spec in CANONICAL_FEATURES
        ],
    }
    (out_dir / "stellar_feature_schema_v2.json").write_text(
        json.dumps(schema_json, indent=2) + "\n"
    )

    n_raw = len(df.columns)
    n_resolved = int(
        resolution_df["resolution_status"].isin(["exact", "semantic"]).sum()
    )
    n_unresolved = len(resolution_df) - n_resolved
    n_redundant = len(redundant)

    lines = [
        "Stellar characterization v2 feature audit",
        "==========================================",
        "",
        f"Input: {args.input}",
        f"Stars/rows: {len(df)}",
        f"Raw diagnostic columns: {n_raw}",
        f"Scientific domains: {len(DOMAIN_ORDER)}",
        f"Canonical variables requested: {len(CANONICAL_FEATURES)}",
        f"Canonical variables resolved: {n_resolved}",
        f"Canonical variables unresolved/ambiguous: {n_unresolved}",
        (
            "Highly redundant canonical pairs "
            f"(|Spearman rho| >= {args.corr_threshold:.2f}): {n_redundant}"
        ),
        "",
        "Canonical feature resolutions",
        "-----------------------------",
    ]

    for _, row in resolution_df.iterrows():
        source = row["source_column"]
        if pd.isna(source) or source is None:
            source = "<UNRESOLVED>"
        lines.append(
            f"{row['domain']:26s}  "
            f"{row['canonical_feature']:24s} <- "
            f"{source} [{row['resolution_status']}]"
        )

    lines.extend(
        [
            "",
            "Interpretation",
            "--------------",
            "The raw diagnostic table remains the audit/debug layer.",
            "The seven domains are the scientific characterization taxonomy.",
            (
                "stellar_features_v2_50.csv is the compact canonical representation "
                "to inspect before any PCA or model-selection work."
            ),
            (
                "Do not fit PCA on repeated injection rows; dimensionality reduction "
                "must be fit at the unique-star level."
            ),
            "",
        ]
    )

    if not unresolved.empty:
        lines.extend(
            [
                "Manual action required",
                "----------------------",
                (
                    "At least one canonical variable could not be resolved "
                    "unambiguously. Inspect canonical_feature_schema.csv before "
                    "freezing the feature vector."
                ),
                "",
            ]
        )

    if n_redundant:
        lines.extend(
            [
                "Redundancy review",
                "-----------------",
                (
                    "canonical_redundancy_pairs.csv contains pairs above the "
                    "configured Spearman threshold. Correlation alone is not a "
                    "reason to delete a scientifically distinct variable, but these "
                    "pairs should be reviewed before PCA."
                ),
                "",
            ]
        )

    summary_path = out_dir / "feature_audit_summary.txt"
    summary_path.write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
