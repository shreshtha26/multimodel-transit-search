"""Long-format schemas and result containers for adaptive-transit runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_transit.core import LightCurve
from adaptive_transit.noise_models.stellar_variability import CANONICAL_CHARACTERIZATION_COLUMNS


@dataclass(frozen=True)
class TreatmentResult:
    treatment: str
    lightcurve: LightCurve
    success: bool = True
    runtime_seconds: float | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionResult:
    success: bool
    best_period_days: float | None
    best_epoch: float | None
    best_duration_days: float | None
    best_depth: float | None
    raw_score: float | None
    exact_recovery: bool | None = None
    harmonic_recovery: bool | None = None
    period_error: float | None = None
    observability: float | None = None
    runtime_seconds: float | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def with_recovery(
        self,
        *,
        exact_recovery: bool | None,
        harmonic_recovery: bool | None,
        period_error: float | None,
    ) -> "DetectionResult":
        return DetectionResult(
            success=self.success,
            best_period_days=self.best_period_days,
            best_epoch=self.best_epoch,
            best_duration_days=self.best_duration_days,
            best_depth=self.best_depth,
            raw_score=self.raw_score,
            exact_recovery=exact_recovery,
            harmonic_recovery=harmonic_recovery,
            period_error=period_error,
            observability=self.observability,
            runtime_seconds=self.runtime_seconds,
            diagnostics=dict(self.diagnostics),
        )


CHARACTERIZATION_COLUMNS = (
    "run_id",
    "config_hash",
    "star_id",
    "target_id",
    "quarter",
    "characterization_version",
    "success",
    *CANONICAL_CHARACTERIZATION_COLUMNS,
)

INJECTION_COLUMNS = (
    "run_id",
    "config_hash",
    "star_id",
    "injection_id",
    "injection_kind",
    "period_days",
    "epoch_days",
    "duration_days",
    "depth",
    "epoch_phase_fraction",
    "batman_used",
    "seed",
    "template_hash",
    "success",
)

TREATMENT_COLUMNS = (
    "run_id",
    "config_hash",
    "star_id",
    "treatment",
    "success",
    "runtime_seconds",
    "diagnostics",
)

PRESERVATION_COLUMNS = (
    "run_id",
    "config_hash",
    "star_id",
    "injection_id",
    "treatment",
    "depth_before",
    "depth_after",
    "depth_retention_fraction",
    "snr_before",
    "snr_after",
    "snr_retention_fraction",
    "in_transit_observation_count",
    "template_hash",
    "success",
)

DETECTION_COLUMNS = (
    "run_id",
    "config_hash",
    "star_id",
    "injection_id",
    "treatment",
    "detector",
    "score_name",
    "score_definition",
    "fap_level",
    "fap_threshold",
    "above_threshold",
    "success",
    "best_period_days",
    "best_epoch",
    "best_duration_days",
    "best_depth",
    "raw_score",
    "exact_recovery",
    "harmonic_recovery",
    "period_error",
    "exact_period_error",
    "harmonic_period_error",
    "observability",
    "runtime_seconds",
    "diagnostics",
)

NULL_SCORE_COLUMNS = (
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
    "runtime_seconds",
)

LONG_TABLE_SCHEMAS = {
    "characterization": CHARACTERIZATION_COLUMNS,
    "treatment": TREATMENT_COLUMNS,
    "injection": INJECTION_COLUMNS,
    "preservation": PRESERVATION_COLUMNS,
    "detection": DETECTION_COLUMNS,
    "null_score": NULL_SCORE_COLUMNS,
}


def assert_long_schema(frame: pd.DataFrame, table_name: str) -> None:
    required = set(LONG_TABLE_SCHEMAS[table_name])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{table_name} table is missing columns: {missing}")


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def diagnostics_json(value: Mapping[str, Any] | None) -> str:
    return json.dumps(json_ready(dict(value or {})), sort_keys=True, separators=(",", ":"))
