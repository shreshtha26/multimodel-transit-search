"""Background-treatment adapters around existing scientific implementations."""

from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from adaptive_transit.core import LightCurve
from adaptive_transit.noise_models.arima import apply_fitted_arima_filter, fit_arima_model
from adaptive_transit.noise_models.gp import (
    apply_prepared_smooth_gp_filter,
    fit_smooth_gp_background,
    prepare_smooth_gp_filter,
)
from adaptive_transit.noise_models.kalman import apply_fitted_kalman_filter, fit_kalman_local_level
from adaptive_transit.schemas import TreatmentResult


class BackgroundTreatment:
    name = "base"

    def fit(self, lightcurve: LightCurve) -> "BackgroundTreatment":
        raise NotImplementedError

    def transform(self, lightcurve: LightCurve) -> TreatmentResult:
        raise NotImplementedError

    def fit_transform(self, lightcurve: LightCurve) -> TreatmentResult:
        self.fit(lightcurve)
        return self.transform(lightcurve)

    def diagnostics(self) -> dict[str, Any]:
        return {}


class RawTreatment(BackgroundTreatment):
    name = "raw"

    def __init__(self) -> None:
        self._diagnostics = {"identity": True}

    def fit(self, lightcurve: LightCurve) -> "RawTreatment":
        self._diagnostics = {
            "identity": True,
            "finite_observation_count": int(np.isfinite(lightcurve.flux).sum()),
        }
        return self

    def transform(self, lightcurve: LightCurve) -> TreatmentResult:
        return TreatmentResult(
            treatment=self.name,
            lightcurve=lightcurve.with_flux(lightcurve.flux.copy(), metadata={"treatment": self.name}),
            diagnostics=self.diagnostics(),
        )

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._diagnostics)


class ARIMATreatment(BackgroundTreatment):
    name = "arima"

    def __init__(
        self,
        *,
        order: tuple[int, int, int] = (1, 1, 0),
        fit_maxiter: int = 200,
        allow_missing: bool = True,
    ) -> None:
        self.order = tuple(order)
        self.fit_maxiter = int(fit_maxiter)
        self.allow_missing = bool(allow_missing)
        self._model = None
        self._diagnostics: dict[str, Any] = {}

    def fit(self, lightcurve: LightCurve) -> "ARIMATreatment":
        started = perf_counter()
        self._model = fit_arima_model(
            lightcurve.flux,
            order=self.order,
            allow_missing=self.allow_missing,
            mode="cadence_grid",
            fit_maxiter=self.fit_maxiter,
        )
        self._diagnostics = {
            "order": self.order,
            "fit_maxiter": self.fit_maxiter,
            "allow_missing": self.allow_missing,
            "fit_runtime_seconds": float(perf_counter() - started),
            "finite_innovation_count": int(np.isfinite(self._model.innovations).sum()),
            "mode": self._model.mode,
        }
        return self

    def transform(self, lightcurve: LightCurve) -> TreatmentResult:
        if self._model is None:
            raise RuntimeError("ARIMA treatment has not been fitted.")
        started = perf_counter()
        filtered = apply_fitted_arima_filter(lightcurve.flux, self._model, allow_missing=self.allow_missing)
        return TreatmentResult(
            treatment=self.name,
            lightcurve=lightcurve.with_flux(filtered.innovations, metadata={"treatment": self.name}),
            runtime_seconds=float(perf_counter() - started),
            diagnostics={**self.diagnostics(), "transform_mode": "fixed_base_parameters"},
        )

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._diagnostics)


class KalmanTreatment(BackgroundTreatment):
    name = "kalman"

    def __init__(self, *, maxiter: int = 100, burn_in: int = 1) -> None:
        self.maxiter = int(maxiter)
        self.burn_in = int(burn_in)
        self._model = None

    def fit(self, lightcurve: LightCurve) -> "KalmanTreatment":
        self._model = fit_kalman_local_level(lightcurve.flux, maxiter=self.maxiter, burn_in=self.burn_in)
        return self

    def transform(self, lightcurve: LightCurve) -> TreatmentResult:
        if self._model is None:
            raise RuntimeError("Kalman treatment has not been fitted.")
        started = perf_counter()
        filtered = apply_fitted_kalman_filter(lightcurve.flux, self._model, burn_in=self.burn_in)
        return TreatmentResult(
            treatment=self.name,
            lightcurve=lightcurve.with_flux(filtered.residuals, metadata={"treatment": self.name}),
            runtime_seconds=float(perf_counter() - started),
            diagnostics=filtered.summary(),
        )

    def diagnostics(self) -> dict[str, Any]:
        return {} if self._model is None else self._model.summary()


class GPTreatment(BackgroundTreatment):
    name = "gp"

    def __init__(
        self,
        *,
        max_train_points: int = 512,
        length_scale_days: float = 3.0,
        min_length_scale_days: float = 1.0,
        max_length_scale_days: float = 30.0,
        measurement_noise_fraction: float = 0.20,
        n_restarts_optimizer: int = 0,
        random_seed: int = 123,
        optimize_kernel: bool = True,
    ) -> None:
        self.params = {
            "max_train_points": int(max_train_points),
            "length_scale_days": float(length_scale_days),
            "min_length_scale_days": float(min_length_scale_days),
            "max_length_scale_days": float(max_length_scale_days),
            "measurement_noise_fraction": float(measurement_noise_fraction),
            "n_restarts_optimizer": int(n_restarts_optimizer),
            "random_seed": int(random_seed),
            "optimize_kernel": bool(optimize_kernel),
        }
        self._model = None
        self._prepared = None

    def fit(self, lightcurve: LightCurve) -> "GPTreatment":
        self._model = fit_smooth_gp_background(lightcurve.time, lightcurve.flux, **self.params)
        self._prepared = prepare_smooth_gp_filter(lightcurve.time, self._model)
        return self

    def transform(self, lightcurve: LightCurve) -> TreatmentResult:
        if self._prepared is None:
            raise RuntimeError("GP treatment has not been fitted.")
        started = perf_counter()
        filtered = apply_prepared_smooth_gp_filter(lightcurve.flux, self._prepared)
        return TreatmentResult(
            treatment=self.name,
            lightcurve=lightcurve.with_flux(filtered.residuals, metadata={"treatment": self.name}),
            runtime_seconds=float(perf_counter() - started),
            diagnostics=filtered.summary(),
        )

    def diagnostics(self) -> dict[str, Any]:
        return {} if self._model is None else self._model.summary()


BACKGROUND_MODELS: dict[str, BackgroundTreatment] = {
    "raw": RawTreatment(),
    "arima": ARIMATreatment(),
    "kalman": KalmanTreatment(),
    "gp": GPTreatment(),
}


def make_background_treatment(name: str, parameters: Mapping[str, Any] | None = None) -> BackgroundTreatment:
    if name not in BACKGROUND_MODELS:
        raise KeyError(f"Unknown background treatment: {name}")
    treatment = deepcopy(BACKGROUND_MODELS[name])
    params = dict(parameters or {})
    if not params:
        return treatment
    if name == "arima":
        return ARIMATreatment(
            order=tuple(params.get("order", getattr(treatment, "order", (1, 1, 0)))),
            fit_maxiter=int(params.get("fit_maxiter", getattr(treatment, "fit_maxiter", 200))),
            allow_missing=bool(params.get("allow_missing", getattr(treatment, "allow_missing", True))),
        )
    if name == "kalman":
        return KalmanTreatment(
            maxiter=int(params.get("maxiter", getattr(treatment, "maxiter", 100))),
            burn_in=int(params.get("burn_in", getattr(treatment, "burn_in", 1))),
        )
    if name == "gp":
        return GPTreatment(**{**getattr(treatment, "params", {}), **params})
    return treatment
