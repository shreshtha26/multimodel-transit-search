"""Unified long-format adaptive-transit experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from time import perf_counter
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from adaptive_transit.config import AdaptiveTransitConfig
from adaptive_transit.core import LightCurve, stable_seed
from adaptive_transit.detection.bls import default_duration_grid, default_period_grid
from adaptive_transit.detection.false_alarm import moving_block_surrogate
from adaptive_transit.detectors import DETECTORS, DetectorContext, TransitDetector
from adaptive_transit.fap import calibrated_detection_fields, lookup_threshold, threshold_key
from adaptive_transit.injection_plan import InjectionCase, build_injection_cases, realize_injection
from adaptive_transit.noise_models.characterization import characterize_light_curve
from adaptive_transit.noise_models.stellar_variability import CANONICAL_CHARACTERIZATION_COLUMNS, V2_FREEZE_ID
from adaptive_transit.preservation import preservation_row
from adaptive_transit.schemas import DetectionResult, assert_long_schema, diagnostics_json, finite_or_none
from adaptive_transit.treatments import BACKGROUND_MODELS, BackgroundTreatment, make_background_treatment


@dataclass(frozen=True)
class RunnerResult:
    characterization: pd.DataFrame
    injection: pd.DataFrame
    preservation: pd.DataFrame
    detection: pd.DataFrame


def _period_errors(
    recovered_period: float | None,
    truth_period: float | None,
    tolerance: float = 0.02,
) -> tuple[bool | None, bool | None, float | None, float | None]:
    if recovered_period is None or truth_period is None:
        return None, None, None, None
    recovered = float(recovered_period)
    truth = float(truth_period)
    if not np.isfinite(recovered) or not np.isfinite(truth) or truth <= 0:
        return None, None, None, None
    exact_error = abs(recovered - truth) / truth
    harmonic_error = min(abs(recovered - truth * factor) / (truth * factor) for factor in (0.5, 1.0, 2.0))
    return bool(exact_error <= tolerance), bool(harmonic_error <= tolerance), float(exact_error), float(harmonic_error)


class UnifiedPipelineRunner:
    """Runs treatment x detector grids without hard-coded pipeline branches."""

    def __init__(
        self,
        config: AdaptiveTransitConfig,
        *,
        treatment_registry: Mapping[str, BackgroundTreatment] | None = None,
        detector_registry: Mapping[str, TransitDetector] | None = None,
        characterizer: Callable[..., dict] = characterize_light_curve,
    ) -> None:
        self.config = config
        self.treatment_registry = treatment_registry or BACKGROUND_MODELS
        self.detector_registry = detector_registry or DETECTORS
        self.characterizer = characterizer

    @property
    def config_hash(self) -> str:
        return self.config.config_hash

    def default_injection_cases(self) -> tuple[InjectionCase, ...]:
        return build_injection_cases(
            periods=self.config.injection_period_grid,
            durations_hours=self.config.injection_duration_hours_grid,
            depths=self.config.injection_depth_grid,
            epoch_phase_fractions=self.config.epoch_phase_fraction_grid,
            include_native_zero=self.config.include_native_zero_injection,
            seed=self.config.random_seed,
        )

    def period_grid(self, lightcurve: LightCurve) -> np.ndarray:
        return default_period_grid(
            lightcurve.time,
            min_period_days=self.config.min_period_days,
            max_period_days=self.config.max_period_days,
            n_periods=self.config.n_periods,
        )

    def duration_grid(self) -> np.ndarray:
        return default_duration_grid(
            self.config.min_duration_hours,
            self.config.max_duration_hours,
            self.config.n_durations,
        )

    def detector_parameters(self, detector_name: str) -> dict:
        return {
            **self.config.detector_parameters.get(detector_name, {}),
            "min_period_days": self.config.min_period_days,
            "max_period_days": self.config.max_period_days,
            "top_k": self.config.top_k,
        }

    def characterize_native(self, *, run_id: str, star_id: str, target_id: str, quarter: int, native: LightCurve) -> dict:
        record = self.characterizer(
            native.time,
            native.flux,
            cadenceno=native.cadenceno,
            quality=native.quality,
            row_present=native.row_present,
            usable_mask=native.usable_mask,
            target_id=target_id,
            quarter=quarter,
        )
        row = {
            "run_id": run_id,
            "config_hash": self.config_hash,
            "star_id": star_id,
            "target_id": str(target_id),
            "quarter": int(quarter),
            "characterization_version": V2_FREEZE_ID,
            "success": True,
        }
        for column in CANONICAL_CHARACTERIZATION_COLUMNS:
            row[column] = record.get(column)
        return row

    def fit_native_treatments(self, native: LightCurve) -> dict[str, BackgroundTreatment]:
        fitted = {}
        for treatment_name in sorted({spec.treatment for spec in self.config.active_combinations}):
            parameters = self.config.treatment_parameters.get(treatment_name, {})
            if self.treatment_registry is BACKGROUND_MODELS:
                treatment = make_background_treatment(treatment_name, parameters)
            else:
                treatment = deepcopy(self.treatment_registry[treatment_name])
            treatment.fit(native)
            fitted[treatment_name] = treatment
        return fitted

    def native_treatment_results(
        self,
        native: LightCurve,
        treatment_models: Mapping[str, BackgroundTreatment],
    ) -> dict[str, object]:
        return {name: treatment.transform(native) for name, treatment in treatment_models.items()}

    def base_detector_caches(self, native_treated: Mapping[str, object]) -> dict[tuple[str, str], dict]:
        caches: dict[tuple[str, str], dict] = {}
        for spec in self.config.active_combinations:
            detector = self.detector_registry[spec.detector]
            params = self.detector_parameters(spec.detector)
            treatment_result = native_treated[spec.treatment]
            try:
                caches[(spec.treatment, spec.detector)] = detector.prepare_native(treatment_result.lightcurve, params)
            except Exception as exc:
                caches[(spec.treatment, spec.detector)] = {"prepare_native_error": f"{type(exc).__name__}: {exc}"}
        return caches

    def run_lightcurve(
        self,
        *,
        run_id: str,
        star_id: str,
        target_id: str,
        quarter: int,
        native: LightCurve,
        injection_cases: tuple[InjectionCase, ...] | None = None,
        thresholds: pd.DataFrame | None = None,
    ) -> RunnerResult:
        period_grid = self.period_grid(native)
        duration_grid = self.duration_grid()
        treatment_models = self.fit_native_treatments(native)
        native_treated = self.native_treatment_results(native, treatment_models)
        base_detector_caches = self.base_detector_caches(native_treated)
        characterization_rows = [
            self.characterize_native(
                run_id=run_id,
                star_id=star_id,
                target_id=target_id,
                quarter=quarter,
                native=native,
            )
        ]
        injection_rows: list[dict] = []
        preservation_rows: list[dict] = []
        detection_rows: list[dict] = []

        for case in injection_cases or self.default_injection_cases():
            realization = realize_injection(native, case)
            case = realization.case
            injection_rows.append(
                {
                    "run_id": run_id,
                    "config_hash": self.config_hash,
                    "star_id": star_id,
                    "injection_id": case.injection_id,
                    "injection_kind": case.kind,
                    "period_days": case.period_days,
                    "epoch_days": case.epoch_days,
                    "duration_days": case.duration_days,
                    "depth": case.depth,
                    "epoch_phase_fraction": case.epoch_phase_fraction,
                    "batman_used": bool(realization.batman_used),
                    "seed": case.seed,
                    "template_hash": realization.template_hash,
                    "success": True,
                }
            )

            treatment_results = {}
            for treatment_name, treatment in treatment_models.items():
                result = treatment.transform(realization.lightcurve)
                treatment_results[treatment_name] = result
                row = preservation_row(
                    run_id=run_id,
                    config_hash=self.config_hash,
                    star_id=star_id,
                    injection_id=case.injection_id,
                    treatment=treatment_name,
                    injected_flux=realization.lightcurve.flux,
                    treated_flux=result.lightcurve.flux,
                    in_transit=realization.in_transit,
                )
                row["template_hash"] = realization.template_hash
                preservation_rows.append(row)

            for spec in self.config.active_combinations:
                treatment_result = treatment_results[spec.treatment]
                detector = self.detector_registry[spec.detector]
                params = self.detector_parameters(spec.detector)
                preserve = next(
                    item
                    for item in preservation_rows
                    if item["injection_id"] == case.injection_id and item["treatment"] == spec.treatment
                )
                context = DetectorContext(
                    period_grid=period_grid,
                    duration_grid=duration_grid,
                    parameters=params,
                    cache={**base_detector_caches.get((spec.treatment, spec.detector), {})},
                    preservation_row=preserve,
                )
                started = perf_counter()
                try:
                    result = detector.search(treatment_result.lightcurve, context)
                    if result.runtime_seconds is None:
                        result = DetectionResult(
                            **{**result.__dict__, "runtime_seconds": float(perf_counter() - started)}
                        )
                except Exception as exc:
                    result = DetectionResult(
                        success=False,
                        best_period_days=None,
                        best_epoch=None,
                        best_duration_days=None,
                        best_depth=None,
                        raw_score=None,
                        runtime_seconds=float(perf_counter() - started),
                        diagnostics={"error": f"{type(exc).__name__}: {exc}"},
                    )
                if result.success:
                    score_records = detector.score_records(result)
                else:
                    score_records = tuple(
                        {
                            "score_name": score_definition,
                            "score_definition": score_definition,
                            "score": None,
                            "best_period_days": None,
                            "best_epoch": None,
                            "best_duration_days": None,
                            "best_depth": None,
                            "observability": None,
                        }
                        for score_definition in detector.active_score_definitions(params)
                    )
                for score_record in score_records:
                    score_definition = str(score_record["score_definition"])
                    score = score_record.get("score")
                    best_period = finite_or_none(score_record.get("best_period_days"))
                    exact, harmonic, exact_error, harmonic_error = _period_errors(
                        best_period,
                        case.period_days,
                        tolerance=self.config.period_match_tolerance_fraction,
                    )
                    key = threshold_key(
                        star_id=star_id,
                        treatment=spec.treatment,
                        detector=spec.detector,
                        score_definition=score_definition,
                        fap_level=self.config.fap_level,
                    )
                    threshold = lookup_threshold(thresholds, key)
                    fap_fields = calibrated_detection_fields(score, threshold)
                    detection_rows.append(
                        {
                            "run_id": run_id,
                            "config_hash": self.config_hash,
                            "star_id": star_id,
                            "injection_id": case.injection_id,
                            "treatment": spec.treatment,
                            "detector": spec.detector,
                            "score_name": str(score_record.get("score_name", score_definition)),
                            "score_definition": score_definition,
                            "fap_level": self.config.fap_level,
                            **fap_fields,
                            "success": result.success,
                            "best_period_days": best_period,
                            "best_epoch": finite_or_none(score_record.get("best_epoch")),
                            "best_duration_days": finite_or_none(score_record.get("best_duration_days")),
                            "best_depth": finite_or_none(score_record.get("best_depth")),
                            "raw_score": finite_or_none(score),
                            "exact_recovery": exact,
                            "harmonic_recovery": harmonic,
                            "period_error": harmonic_error,
                            "exact_period_error": exact_error,
                            "harmonic_period_error": harmonic_error,
                            "observability": finite_or_none(score_record.get("observability")),
                            "runtime_seconds": result.runtime_seconds,
                            "diagnostics": diagnostics_json(result.diagnostics),
                        }
                    )

        frames = RunnerResult(
            characterization=pd.DataFrame(characterization_rows),
            injection=pd.DataFrame(injection_rows),
            preservation=pd.DataFrame(preservation_rows),
            detection=pd.DataFrame(detection_rows),
        )
        for name, frame in (
            ("characterization", frames.characterization),
            ("injection", frames.injection),
            ("preservation", frames.preservation),
            ("detection", frames.detection),
        ):
            assert_long_schema(frame, name)
        return frames

    def run_null_scores(
        self,
        *,
        run_id: str,
        star_id: str,
        native: LightCurve,
        n_trials: int,
    ) -> pd.DataFrame:
        """Run moving-block native-background null trials through the same grid."""

        period_grid = self.period_grid(native)
        duration_grid = self.duration_grid()
        treatment_models = self.fit_native_treatments(native)
        native_treated = self.native_treatment_results(native, treatment_models)
        base_detector_caches = self.base_detector_caches(native_treated)
        rows: list[dict] = []

        for trial in range(int(n_trials)):
            seed = stable_seed(self.config.null_generation_seed, star_id, trial)
            trial_rng = np.random.default_rng(int(seed))
            for treatment_name, treatment_result in native_treated.items():
                surrogate_flux = moving_block_surrogate(
                    treatment_result.lightcurve.flux,
                    block_size=int(self.config.null_block_size_cadences),
                    rng=trial_rng,
                )
                surrogate = treatment_result.lightcurve.with_flux(
                    surrogate_flux,
                    metadata={"null_trial": int(trial), "treatment": treatment_name},
                )
                for spec in self.config.active_combinations:
                    if spec.treatment != treatment_name:
                        continue
                    detector = self.detector_registry[spec.detector]
                    params = self.detector_parameters(spec.detector)
                    context = DetectorContext(
                        period_grid=period_grid,
                        duration_grid=duration_grid,
                        parameters=params,
                        cache={**base_detector_caches.get((spec.treatment, spec.detector), {})},
                    )
                    try:
                        result = detector.search(surrogate, context)
                        score_records = detector.score_records(result)
                        for score_record in score_records:
                            score_definition = str(score_record["score_definition"])
                            rows.append(
                                {
                                    "run_id": run_id,
                                    "config_hash": self.config_hash,
                                    "star_id": star_id,
                                    "trial": int(trial),
                                    "trial_seed": int(seed),
                                    "treatment": spec.treatment,
                                    "detector": spec.detector,
                                    "score_name": str(score_record.get("score_name", score_definition)),
                                    "score_definition": score_definition,
                                    "score": score_record.get("score"),
                                    "success": True,
                                    "best_period_days": finite_or_none(score_record.get("best_period_days")),
                                    "runtime_seconds": result.runtime_seconds,
                                }
                        )
                    except Exception as exc:
                        for score_definition in detector.active_score_definitions(params):
                            rows.append(
                                {
                                    "run_id": run_id,
                                    "config_hash": self.config_hash,
                                    "star_id": star_id,
                                    "trial": int(trial),
                                    "trial_seed": int(seed),
                                    "treatment": spec.treatment,
                                    "detector": spec.detector,
                                    "score_name": str(score_definition),
                                    "score_definition": str(score_definition),
                                    "score": np.nan,
                                    "success": False,
                                    "best_period_days": np.nan,
                                    "runtime_seconds": np.nan,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
        frame = pd.DataFrame(rows)
        assert_long_schema(frame, "null_score")
        return frame
