#!/usr/bin/env python3
"""Join frozen stellar-characterization variables to BATMAN POC outcomes.

This is a descriptive bridge only.  It deliberately does NOT train a router:
the current BATMAN POC has only ten stars, so ML fitting would be unjustified.

Inputs
------
- BATMAN POC QC outputs created by scripts/analyze_batman_physical_poc.py
- A stellar-characterization CSV containing the 11 frozen canonical variables,
  or their final v2 source columns.

Outputs
-------
- one-row-per-star router prototype table
- long star x branch performance table
- descriptive feature/performance Spearman associations
- stationarity-state branch summaries
- feature-source coverage audit
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs/experiments/batman_physical_detection_poc/pilot10"

CANONICAL_FEATURES = (
    "robust_scatter",
    "skewness",
    "outlier_fraction",
    "acf_lag1",
    "acf_timescale_days",
    "stationarity_state",
    "spectral_concentration",
    "harmonic_power_ratio",
    "dominant_period_days",
    "period_agreement_error",
    "segment_scale_variability",
)
CONTINUOUS_FEATURES = tuple(x for x in CANONICAL_FEATURES if x != "stationarity_state")

# Final characterization-v2 source columns.  The compact canonical CSV should
# already use the canonical names above, but accepting these source columns
# lets the bridge work directly from a characterization master table as well.
SOURCE_TO_CANONICAL = {
    "flux_robust_scale": "robust_scatter",
    "flux_skewness": "skewness",
    "flux_outlier_fraction": "outlier_fraction",
    "v2_acf_lag_1": "acf_lag1",
    "v2_acf_decay_e_days": "acf_timescale_days",
    "original_series_stationarity_conclusion": "stationarity_state",
    "v2_spectral_concentration": "spectral_concentration",
    "v2_spectral_harmonic_power_ratio": "harmonic_power_ratio",
    "v2_ls_dominant_period_days": "dominant_period_days",
    "v2_ls_acf_period_relative_error": "period_agreement_error",
    "v2_segment_scale_relative_mad": "segment_scale_variability",
}


def normalize_target_id(value) -> str:
    return str(value).upper().replace("KIC", "").strip()


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "target_id" not in out:
        raise ValueError("Expected a target_id column.")
    out["target_id"] = out["target_id"].map(normalize_target_id)
    if "quarter" in out:
        out["quarter"] = pd.to_numeric(out["quarter"], errors="coerce").astype("Int64")
    return out


def canonicalize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return canonical columns, copying final source columns when necessary."""
    out = normalize_keys(frame)
    for source, canonical in SOURCE_TO_CANONICAL.items():
        if canonical not in out.columns and source in out.columns:
            out[canonical] = out[source]
    missing = [name for name in CANONICAL_FEATURES if name not in out.columns]
    if missing:
        raise ValueError(f"Feature table is missing frozen canonical variables: {missing}")
    for column in CONTINUOUS_FEATURES:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["stationarity_state"] = out["stationarity_state"].fillna("missing").astype(str)
    return out


def candidate_feature_paths(root: Path) -> list[Path]:
    patterns = (
        "stellar_features_v2*.csv",
        "*canonical*feature*.csv",
        "characterization_master*.csv",
    )
    paths: set[Path] = set()
    for base in (root / "outputs", root / "configs"):
        if not base.exists():
            continue
        for pattern in patterns:
            paths.update(path for path in base.rglob(pattern) if path.is_file())
    return sorted(paths)


def score_feature_candidate(path: Path, wanted_targets: set[str]) -> tuple[int, int, pd.DataFrame] | None:
    try:
        frame = pd.read_csv(path)
        canon = canonicalize_feature_frame(frame)
    except Exception:
        return None
    matches = int(canon["target_id"].isin(wanted_targets).sum())
    unique_matches = int(canon.loc[canon["target_id"].isin(wanted_targets), "target_id"].nunique())
    return unique_matches, int(len(canon)), canon


def discover_feature_file(root: Path, wanted_targets: Iterable[str]) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    wanted = {normalize_target_id(x) for x in wanted_targets}
    audit_rows = []
    best = None
    for path in candidate_feature_paths(root):
        scored = score_feature_candidate(path, wanted)
        if scored is None:
            audit_rows.append({"path": str(path), "valid_11_feature_schema": False, "matched_targets": 0, "rows": np.nan})
            continue
        matches, rows, canon = scored
        audit_rows.append({"path": str(path), "valid_11_feature_schema": True, "matched_targets": matches, "rows": rows})
        key = (matches, rows)
        if best is None or key > best[0]:
            best = (key, path, canon)
    audit = pd.DataFrame(audit_rows)
    if best is None:
        raise FileNotFoundError(
            "Could not discover a CSV with all 11 frozen canonical variables. "
            "Pass --features-csv explicitly."
        )
    return best[1], best[2], audit


def load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return normalize_keys(pd.read_csv(path))


def pivot_retention(retention: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "fit_status",
        "fit_clean",
        "median_template_amplitude_ratio",
        "median_peak_depth_ratio",
        "median_template_energy_ratio",
        "median_template_correlation",
        "median_template_rmse_ppm",
        "median_oracle_signal_snr",
        "median_background_scale_ppm",
        "median_abs_background_acf1",
    ]
    metrics = [m for m in metrics if m in retention.columns]
    key_cols = [c for c in ("target_id", "quarter", "sample_stratum") if c in retention.columns]
    blocks = []
    for branch, group in retention.groupby("branch", dropna=False):
        keep = group[key_cols + metrics].copy()
        keep = keep.rename(columns={m: f"{branch}_{m}" for m in metrics})
        blocks.append(keep)
    if not blocks:
        return pd.DataFrame(columns=key_cols)
    wide = blocks[0]
    for block in blocks[1:]:
        merge_keys = [c for c in key_cols if c in block.columns and c in wide.columns]
        wide = wide.merge(block, on=merge_keys, how="outer", validate="one_to_one")
    return wide


def pivot_detectors(detectors: pd.DataFrame) -> pd.DataFrame:
    key_cols = [c for c in ("target_id", "quarter", "sample_stratum") if c in detectors.columns]
    metrics = [m for m in ("exact_period_recovery", "harmonic_period_recovery", "median_period_error") if m in detectors.columns]
    work = detectors[key_cols + ["branch", "detector"] + metrics].copy()
    pieces = []
    for (branch, detector), group in work.groupby(["branch", "detector"], dropna=False):
        keep = group[key_cols + metrics].copy()
        prefix = f"{branch}_{detector}"
        keep = keep.rename(columns={m: f"{prefix}_{m}" for m in metrics})
        pieces.append(keep)
    if not pieces:
        return pd.DataFrame(columns=key_cols)
    wide = pieces[0]
    for piece in pieces[1:]:
        merge_keys = [c for c in key_cols if c in piece.columns and c in wide.columns]
        wide = wide.merge(piece, on=merge_keys, how="outer", validate="one_to_one")
    return wide


def attach_raw_relative_metrics(long: pd.DataFrame) -> pd.DataFrame:
    out = long.copy()
    key_cols = [c for c in ("target_id", "quarter") if c in out.columns]
    raw = out.loc[out["branch"].astype(str) == "raw", key_cols + [
        "median_oracle_signal_snr",
        "median_abs_background_acf1",
    ]].copy()
    raw = raw.rename(columns={
        "median_oracle_signal_snr": "raw_oracle_signal_snr",
        "median_abs_background_acf1": "raw_abs_background_acf1",
    })
    raw = raw.drop_duplicates(key_cols)
    out = out.merge(raw, on=key_cols, how="left", validate="many_to_one")
    out["oracle_snr_gain_vs_raw"] = (
        pd.to_numeric(out["median_oracle_signal_snr"], errors="coerce") /
        pd.to_numeric(out["raw_oracle_signal_snr"], errors="coerce")
    )
    out["whitening_gain_vs_raw"] = (
        pd.to_numeric(out["raw_abs_background_acf1"], errors="coerce") -
        pd.to_numeric(out["median_abs_background_acf1"], errors="coerce")
    )
    return out


def add_detector_metrics_to_long(long: pd.DataFrame, detectors: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in ("exact_period_recovery", "harmonic_period_recovery", "median_period_error") if m in detectors.columns]
    if not metrics:
        return long
    keys = [c for c in ("target_id", "quarter", "branch") if c in long.columns and c in detectors.columns]
    pivots = []
    for detector, group in detectors.groupby("detector", dropna=False):
        keep = group[keys + metrics].copy()
        keep = keep.rename(columns={m: f"{detector}_{m}" for m in metrics})
        pivots.append(keep)
    out = long.copy()
    for piece in pivots:
        out = out.merge(piece, on=keys, how="left", validate="one_to_one")

    raw_tls = out.loc[out["branch"].astype(str) == "raw", keys[:2] + ["tls_exact_period_recovery"]].copy() if "tls_exact_period_recovery" in out else pd.DataFrame()
    if not raw_tls.empty:
        raw_tls = raw_tls.rename(columns={"tls_exact_period_recovery": "raw_tls_exact_period_recovery"}).drop_duplicates(keys[:2])
        out = out.merge(raw_tls, on=keys[:2], how="left", validate="many_to_one")
        out["tls_exact_gain_vs_raw"] = (
            pd.to_numeric(out["tls_exact_period_recovery"], errors="coerce") -
            pd.to_numeric(out["raw_tls_exact_period_recovery"], errors="coerce")
        )
    return out


def spearman_associations(long: pd.DataFrame) -> pd.DataFrame:
    outcomes = [
        "median_template_correlation",
        "oracle_snr_gain_vs_raw",
        "whitening_gain_vs_raw",
        "tls_exact_gain_vs_raw",
    ]
    rows = []
    for branch, group in long.groupby("branch", dropna=False):
        for feature in CONTINUOUS_FEATURES:
            if feature not in group:
                continue
            x = pd.to_numeric(group[feature], errors="coerce")
            for outcome in outcomes:
                if outcome not in group:
                    continue
                y = pd.to_numeric(group[outcome], errors="coerce")
                mask = x.notna() & y.notna()
                n = int(mask.sum())
                rho = np.nan
                if n >= 3 and x[mask].nunique() > 1 and y[mask].nunique() > 1:
                    rho = float(x[mask].corr(y[mask], method="spearman"))
                rows.append({
                    "branch": branch,
                    "feature": feature,
                    "outcome": outcome,
                    "n_stars": n,
                    "spearman_rho": rho,
                    "interpretation_scope": "descriptive_poc_only_no_significance_claim",
                })
    return pd.DataFrame(rows)


def stationarity_summary(long: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in (
        "median_template_correlation",
        "oracle_snr_gain_vs_raw",
        "whitening_gain_vs_raw",
        "tls_exact_period_recovery",
        "tls_exact_gain_vs_raw",
    ) if m in long]
    if not metrics:
        return pd.DataFrame()
    clean = long.copy()
    if "fit_clean" in clean:
        clean = clean[clean["fit_clean"].fillna(False).astype(bool)]
    agg = {m: (m, "median") for m in metrics}
    agg["n_stars"] = ("target_id", "nunique")
    return clean.groupby(["stationarity_state", "branch"], dropna=False).agg(**agg).reset_index()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Bridge 11 canonical stellar variables to BATMAN POC performance.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--features-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-partial-coverage", action="store_true")
    args = parser.parse_args(argv)

    input_dir = args.input_dir.resolve()
    qc_dir = input_dir / "qc_analysis"
    output_dir = (args.output_dir or (qc_dir / "canonical_bridge")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    winners = load_required_csv(qc_dir / "per_star_winners.csv")
    retention = load_required_csv(qc_dir / "per_star_branch_retention.csv")
    detectors = load_required_csv(qc_dir / "per_star_branch_detector.csv")
    wanted_targets = sorted(winners["target_id"].unique())

    if args.features_csv is not None:
        feature_path = args.features_csv.resolve()
        features = canonicalize_feature_frame(pd.read_csv(feature_path))
        discovery_audit = pd.DataFrame([{
            "path": str(feature_path),
            "valid_11_feature_schema": True,
            "matched_targets": int(features["target_id"].isin(wanted_targets).sum()),
            "rows": len(features),
        }])
    else:
        feature_path, features, discovery_audit = discover_feature_file(PROJECT_ROOT, wanted_targets)

    discovery_audit.to_csv(output_dir / "feature_source_discovery_audit.csv", index=False)

    # Prefer target+quarter match when the characterization file carries quarter.
    winner_keys = ["target_id"]
    if "quarter" in winners and "quarter" in features:
        winner_keys.append("quarter")
    feature_keep = winner_keys + list(CANONICAL_FEATURES)
    features = features[feature_keep].drop_duplicates(winner_keys, keep="last")

    coverage = winners[winner_keys].drop_duplicates().merge(
        features[winner_keys].drop_duplicates().assign(canonical_features_present=True),
        on=winner_keys,
        how="left",
    )
    coverage["canonical_features_present"] = coverage["canonical_features_present"].fillna(False).astype(bool)
    coverage.to_csv(output_dir / "canonical_feature_coverage.csv", index=False)
    missing_targets = coverage.loc[~coverage["canonical_features_present"], "target_id"].tolist()
    if missing_targets and not args.allow_partial_coverage:
        raise ValueError(
            f"Canonical-feature coverage is incomplete for {missing_targets}. "
            "Pass a characterization file covering these stars, or use --allow-partial-coverage for audit-only output."
        )

    router = winners.merge(features, on=winner_keys, how="left", validate="one_to_one")
    retention_wide = pivot_retention(retention)
    detector_wide = pivot_detectors(detectors)
    merge_keys = [c for c in ("target_id", "quarter", "sample_stratum") if c in router.columns and c in retention_wide.columns]
    router = router.merge(retention_wide, on=merge_keys, how="left", validate="one_to_one")
    merge_keys_det = [c for c in ("target_id", "quarter", "sample_stratum") if c in router.columns and c in detector_wide.columns]
    router = router.merge(detector_wide, on=merge_keys_det, how="left", validate="one_to_one")
    router.to_csv(output_dir / "canonical_pipeline_router_poc.csv", index=False)

    long_keys = ["target_id"]
    if "quarter" in retention and "quarter" in features:
        long_keys.append("quarter")
    long = retention.merge(features, on=long_keys, how="left", validate="many_to_one")
    long = attach_raw_relative_metrics(long)
    long = add_detector_metrics_to_long(long, detectors)
    long.to_csv(output_dir / "canonical_pipeline_performance_long.csv", index=False)

    associations = spearman_associations(long)
    associations.to_csv(output_dir / "canonical_feature_performance_spearman.csv", index=False)
    states = stationarity_summary(long)
    states.to_csv(output_dir / "stationarity_state_branch_summary.csv", index=False)

    print("\n=== CANONICAL → PIPELINE BRIDGE ===\n")
    print(f"Feature source: {feature_path}")
    print(f"POC stars: {len(winners)}")
    print(f"Stars with all 11 canonical variables: {int(coverage['canonical_features_present'].sum())}/{len(coverage)}")
    print("\nThis is a descriptive 10-star POC bridge, not an ML training set.")
    show = [c for c in (
        "target_id", "sample_stratum", "stationarity_state",
        "oracle_snr_winner_nonraw_clean", "tls_exact_winner_clean",
    ) if c in router.columns]
    if show:
        print("\n" + router[show].to_string(index=False))
    print(f"\nWrote: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
