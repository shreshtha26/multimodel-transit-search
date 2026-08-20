"""Transit-detector adapters with a common result schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd

from adaptive_transit.core import LightCurve
from adaptive_transit.detection.bls import run_bls
from adaptive_transit.detection.tls import run_tls
from adaptive_transit.detection.tps_like import prepare_tps_like_noise_model, run_tps_like_search
from adaptive_transit.detection.tps_like_hardening import harden_tps_like_result
from adaptive_transit.detection.trapezoid import run_bls_seeded_trapezoid
from adaptive_transit.schemas import DetectionResult


@dataclass
class DetectorContext:
    period_grid: np.ndarray
    duration_grid: np.ndarray
    parameters: Mapping[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)
    preservation_row: Mapping[str, Any] | None = None


class TransitDetector:
    name = "base"
    score_definition = "score"
    score_definitions = ("score",)

    def prepare_native(self, lightcurve: LightCurve, parameters: Mapping[str, Any]) -> dict[str, Any]:
        return {}

    def active_score_definitions(self, parameters: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(self.score_definitions)

    def score_values(self, result: DetectionResult) -> tuple[tuple[str, float | None], ...]:
        score_definition = str(result.diagnostics.get("score_definition", self.score_definition))
        return ((score_definition, result.raw_score),)

    def score_records(self, result: DetectionResult) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "score_name": score_definition,
                "score_definition": score_definition,
                "score": score,
                "best_period_days": result.best_period_days,
                "best_epoch": result.best_epoch,
                "best_duration_days": result.best_duration_days,
                "best_depth": result.best_depth,
                "observability": result.observability,
            }
            for score_definition, score in self.score_values(result)
        )

    def search(self, lightcurve: LightCurve, context: DetectorContext) -> DetectionResult:
        raise NotImplementedError


def _top_peaks_json(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    kept = [column for column in columns if column in frame.columns]
    return frame[kept].head(20).to_dict(orient="records")


def _ranked_periodogram_peaks(periodogram: pd.DataFrame, score_column: str, top_k: int) -> pd.DataFrame:
    if periodogram.empty or score_column not in periodogram.columns:
        return pd.DataFrame()
    top = periodogram.sort_values(score_column, ascending=False).head(int(top_k)).reset_index(drop=True)
    top.insert(0, "rank", np.arange(1, len(top) + 1, dtype=int))
    return top


class BLSDetector(TransitDetector):
    name = "bls"
    score_definition = "bls_power"
    score_definitions = ("bls_power",)

    def search(self, lightcurve: LightCurve, context: DetectorContext) -> DetectionResult:
        started = perf_counter()
        objective = context.parameters.get("bls_objective", "snr")
        top_k = int(context.parameters.get("top_k", 5))
        result = run_bls(
            lightcurve.time,
            lightcurve.flux,
            flux_error=lightcurve.flux_error,
            period_grid=context.period_grid,
            duration_grid=context.duration_grid,
            objective=objective,
            top_k=top_k,
        )
        context.cache["bls_raw_result"] = result
        summary = result["summary"]
        return DetectionResult(
            success=True,
            best_period_days=float(summary["period"]),
            best_epoch=float(summary["transit_time"]),
            best_duration_days=float(summary["duration"]),
            best_depth=float(summary["depth"]),
            raw_score=float(summary["power"]),
            runtime_seconds=float(perf_counter() - started),
            diagnostics={
                "score_definition": self.score_definition,
                "objective": str(objective),
                "n_observations": int(summary["n_observations"]),
                "top_peaks": _top_peaks_json(result["top_peaks"], ("period", "power", "duration", "transit_time")),
            },
        )


class TLSDetector(TransitDetector):
    name = "tls"
    score_definition = "tls_sde"
    score_definitions = ("tls_sde",)

    def search(self, lightcurve: LightCurve, context: DetectorContext) -> DetectionResult:
        started = perf_counter()
        params = context.parameters
        result = run_tls(
            lightcurve.time,
            lightcurve.flux,
            period_min=float(params.get("min_period_days", np.nanmin(context.period_grid))),
            period_max=float(params.get("max_period_days", np.nanmax(context.period_grid))),
            use_threads=int(params.get("tls_use_threads", 1)),
            oversampling_factor=int(params.get("tls_oversampling_factor", 2)),
        )
        summary = result["summary"]
        return DetectionResult(
            success=True,
            best_period_days=float(summary["period_days"]),
            best_epoch=float(summary["epoch_days"]),
            best_duration_days=float(summary["duration_days"]),
            best_depth=float(summary["depth_raw"]),
            raw_score=float(summary["sde"]),
            runtime_seconds=float(perf_counter() - started),
            diagnostics={
                "score_definition": self.score_definition,
                "snr": float(summary["snr"]),
                "n_observations": int(summary["n_observations"]),
                "top_peaks": _top_peaks_json(
                    _ranked_periodogram_peaks(result["periodogram"], "power", int(params.get("top_k", 5))),
                    ("rank", "period_days", "power"),
                ),
            },
        )


class TrapezoidDetector(TransitDetector):
    name = "trapezoid"
    score_definition = "trapezoid_sse_improvement"
    score_definitions = ("trapezoid_sse_improvement",)

    def search(self, lightcurve: LightCurve, context: DetectorContext) -> DetectionResult:
        started = perf_counter()
        bls_result = context.cache.get("bls_raw_result")
        if bls_result is None:
            bls_result = run_bls(
                lightcurve.time,
                lightcurve.flux,
                flux_error=lightcurve.flux_error,
                period_grid=context.period_grid,
                duration_grid=context.duration_grid,
                objective=context.parameters.get("bls_objective", "snr"),
                top_k=int(context.parameters.get("top_k", 5)),
            )
            context.cache["bls_raw_result"] = bls_result
        result = run_bls_seeded_trapezoid(
            lightcurve.time,
            lightcurve.flux,
            bls_result,
            duration_grid=context.duration_grid,
            top_k_periods=int(context.parameters.get("top_k", 5)),
        )
        summary = result["summary"]
        return DetectionResult(
            success=True,
            best_period_days=float(summary["period_days"]),
            best_epoch=float(summary["epoch_days"]),
            best_duration_days=float(summary["duration_days"]),
            best_depth=float(summary["depth"]),
            raw_score=float(summary["score"]),
            runtime_seconds=float(perf_counter() - started),
            diagnostics={
                "score_definition": self.score_definition,
                "ingress_fraction": float(summary["ingress_fraction"]),
                "seed_rank": int(summary["seed_rank"]),
                "top_peaks": _top_peaks_json(result["evaluated"], ("period_days", "epoch_days", "duration_days", "score")),
            },
        )


class TPSLikeDetector(TransitDetector):
    name = "tps_like"
    score_definition = "tps_like_mes"
    score_definitions = ("tps_like_mes", "tps_like_robust_veto_score", "tps_like_event_consistency_score")

    def active_score_definitions(self, parameters: Mapping[str, Any]) -> tuple[str, ...]:
        if bool(parameters.get("tps_harden", True)):
            return tuple(self.score_definitions)
        return (self.score_definition,)

    def score_values(self, result: DetectionResult) -> tuple[tuple[str, float | None], ...]:
        values = result.diagnostics.get("score_values")
        if isinstance(values, Mapping):
            return tuple((str(key), None if value is None else float(value)) for key, value in values.items())
        return super().score_values(result)

    def score_records(self, result: DetectionResult) -> tuple[dict[str, Any], ...]:
        values = dict(self.score_values(result))
        hardened_summary = result.diagnostics.get("hardened_summary")
        if not isinstance(hardened_summary, Mapping):
            return super().score_records(result)

        raw_record = {
            "score_name": "tps_like_mes",
            "score_definition": "tps_like_mes",
            "score": values.get("tps_like_mes"),
            "best_period_days": result.best_period_days,
            "best_epoch": result.best_epoch,
            "best_duration_days": result.best_duration_days,
            "best_depth": result.best_depth,
            "observability": result.observability,
        }
        hardened_record = {
            "best_period_days": hardened_summary.get("period_days"),
            "best_epoch": hardened_summary.get("epoch_days"),
            "best_duration_days": (
                None
                if hardened_summary.get("duration_hours") is None
                else float(hardened_summary["duration_hours"]) / 24.0
            ),
            "best_depth": result.best_depth,
            "observability": hardened_summary.get("event_observability_fraction", result.observability),
        }
        records = [raw_record]
        for score_definition in ("tps_like_robust_veto_score", "tps_like_event_consistency_score"):
            if score_definition in values:
                records.append(
                    {
                        "score_name": score_definition,
                        "score_definition": score_definition,
                        "score": values.get(score_definition),
                        **hardened_record,
                    }
                )
        return tuple(records)

    def prepare_native(self, lightcurve: LightCurve, parameters: Mapping[str, Any]) -> dict[str, Any]:
        if lightcurve.segment_id is None:
            return {}
        prepared = prepare_tps_like_noise_model(
            lightcurve.flux,
            lightcurve.segment_id,
            wavelet=str(parameters.get("tps_wavelet", "db6")),
            max_level=int(parameters.get("tps_max_wavelet_level", 6)),
            noise_window_cadences=int(parameters.get("tps_noise_window_cadences", 193)),
            min_segment_cadences=int(parameters.get("tps_min_segment_cadences", 32)),
        )
        return {"tps_like_noise_model": prepared}

    def search(self, lightcurve: LightCurve, context: DetectorContext) -> DetectionResult:
        started = perf_counter()
        if lightcurve.segment_id is None:
            raise ValueError("TPS-like detector requires segment_id.")
        params = context.parameters
        if "prepare_native_error" in context.cache:
            raise ValueError(str(context.cache["prepare_native_error"]))
        prepared = context.cache.get("tps_like_noise_model")
        if prepared is None:
            prepared = prepare_tps_like_noise_model(
                lightcurve.flux,
                lightcurve.segment_id,
                wavelet=str(params.get("tps_wavelet", "db6")),
                max_level=int(params.get("tps_max_wavelet_level", 6)),
                noise_window_cadences=int(params.get("tps_noise_window_cadences", 193)),
                min_segment_cadences=int(params.get("tps_min_segment_cadences", 32)),
            )
            context.cache["tps_like_noise_model"] = prepared
        duration_hours_grid = [float(value) * 24.0 for value in np.asarray(context.duration_grid, dtype=float)]
        result = run_tps_like_search(
            lightcurve.time,
            lightcurve.flux,
            lightcurve.segment_id,
            prepared_noise_model=prepared,
            min_period_days=float(params.get("min_period_days", np.nanmin(context.period_grid))),
            max_period_days=float(params.get("max_period_days", np.nanmax(context.period_grid))),
            duration_hours_grid=duration_hours_grid,
            wavelet=str(params.get("tps_wavelet", "db6")),
            max_level=int(params.get("tps_max_wavelet_level", 6)),
            noise_window_cadences=int(params.get("tps_noise_window_cadences", 193)),
            min_segment_cadences=int(params.get("tps_min_segment_cadences", 32)),
            min_events=int(params.get("min_transit_events", 3)),
        )
        summary = result["summary"]
        score_values = {"tps_like_mes": float(summary["mes"])}
        hardened_summary = None
        if bool(params.get("tps_harden", True)):
            hardened = harden_tps_like_result(result, lightcurve.time, lightcurve.flux)
            hardened_summary = hardened["summary"]
            score_values["tps_like_robust_veto_score"] = float(hardened_summary["robust_veto_score"])
            score_values["tps_like_event_consistency_score"] = float(hardened_summary["event_consistency_score"])
        return DetectionResult(
            success=True,
            best_period_days=float(summary["period_days"]),
            best_epoch=float(summary["epoch_days"]),
            best_duration_days=float(summary["duration_hours"]) / 24.0,
            best_depth=None,
            raw_score=float(summary["mes"]),
            observability=float(summary["observability_fraction"]),
            runtime_seconds=float(perf_counter() - started),
            diagnostics={
                "score_definition": self.score_definition,
                "score_values": score_values,
                "hardened_summary": hardened_summary,
                "max_ses": float(summary["max_ses"]),
                "observed_event_count": int(summary["observed_event_count"]),
                "expected_event_count": int(summary["expected_event_count"]),
                "top_peaks": _top_peaks_json(result["periodogram"], ("period_days", "duration_hours", "mes", "observability_fraction")),
            },
        )


DETECTORS: dict[str, TransitDetector] = {
    "bls": BLSDetector(),
    "tls": TLSDetector(),
    "trapezoid": TrapezoidDetector(),
    "tps_like": TPSLikeDetector(),
}


def make_detector(name: str) -> TransitDetector:
    if name not in DETECTORS:
        raise KeyError(f"Unknown detector: {name}")
    return DETECTORS[name]
