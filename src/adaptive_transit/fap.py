"""Common-FAP threshold identities and empirical calibration helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from adaptive_transit.detection.common_fap import empirical_p_value, empirical_threshold


@dataclass(frozen=True)
class FAPThresholdKey:
    star_id: str
    treatment: str
    detector: str
    score_definition: str
    fap_level: float = 0.01

    def to_dict(self) -> dict:
        return asdict(self)


def threshold_key(
    *,
    star_id: str,
    treatment: str,
    detector: str,
    score_definition: str,
    fap_level: float = 0.01,
) -> FAPThresholdKey:
    return FAPThresholdKey(
        star_id=str(star_id),
        treatment=str(treatment),
        detector=str(detector),
        score_definition=str(score_definition),
        fap_level=float(fap_level),
    )


def threshold_table_from_null_scores(null_scores: pd.DataFrame, *, fap_level: float = 0.01) -> pd.DataFrame:
    required = {"star_id", "treatment", "detector", "score_definition", "score"}
    missing = sorted(required.difference(null_scores.columns))
    if missing:
        raise ValueError(f"Null score table is missing columns: {missing}")
    rows = []
    for keys, group in null_scores.groupby(["star_id", "treatment", "detector", "score_definition"], dropna=False):
        star_id, treatment, detector, score_definition = keys
        run_ids = sorted(str(value) for value in group["run_id"].dropna().unique()) if "run_id" in group.columns else []
        config_hashes = (
            sorted(str(value) for value in group["config_hash"].dropna().unique())
            if "config_hash" in group.columns
            else []
        )
        threshold = empirical_threshold(group["score"].to_numpy(dtype=float), fap_level=fap_level)
        rows.append(
            {
                "run_id": run_ids[0] if len(run_ids) == 1 else "",
                "config_hash": config_hashes[0] if len(config_hashes) == 1 else "",
                "star_id": str(star_id),
                "treatment": str(treatment),
                "detector": str(detector),
                "score_name": str(score_definition),
                "score_definition": str(score_definition),
                "fap_level": float(fap_level),
                "fap_threshold": threshold,
                "null_trial_count": int(group["score"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def lookup_threshold(
    thresholds: pd.DataFrame | None,
    key: FAPThresholdKey,
) -> float | None:
    if thresholds is None or thresholds.empty:
        return None
    mask = (
        thresholds["star_id"].astype(str).eq(key.star_id)
        & thresholds["treatment"].astype(str).eq(key.treatment)
        & thresholds["detector"].astype(str).eq(key.detector)
        & thresholds["score_definition"].astype(str).eq(key.score_definition)
        & thresholds["fap_level"].astype(float).eq(float(key.fap_level))
    )
    if not mask.any():
        return None
    return float(thresholds.loc[mask, "fap_threshold"].iloc[0])


def calibrated_detection_fields(score: float | None, threshold: float | None) -> dict:
    if score is None or threshold is None:
        return {"fap_threshold": threshold, "above_threshold": None}
    return {"fap_threshold": float(threshold), "above_threshold": bool(float(score) > float(threshold))}


__all__ = [
    "FAPThresholdKey",
    "calibrated_detection_fields",
    "empirical_p_value",
    "empirical_threshold",
    "lookup_threshold",
    "threshold_key",
    "threshold_table_from_null_scores",
]
