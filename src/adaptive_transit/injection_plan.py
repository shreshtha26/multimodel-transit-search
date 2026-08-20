"""Injection definitions and single-realization BATMAN generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable

import numpy as np

from adaptive_transit.core import LightCurve, array_digest
from adaptive_transit.injections.batman import inject_batman_transit


@dataclass(frozen=True)
class InjectionCase:
    injection_id: str
    period_days: float | None
    duration_days: float | None
    depth: float
    epoch_phase_fraction: float | None = None
    epoch_days: float | None = None
    seed: int | None = None
    kind: str = "batman"

    @property
    def is_native_zero(self) -> bool:
        return self.kind == "native" or float(self.depth) == 0.0

    def to_row(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class InjectionRealization:
    case: InjectionCase
    lightcurve: LightCurve
    additive_template: np.ndarray
    in_transit: np.ndarray
    truth: dict
    batman_used: bool

    @property
    def template_hash(self) -> str:
        return array_digest(self.additive_template)


def native_zero_case() -> InjectionCase:
    return InjectionCase(
        injection_id="native_zero",
        period_days=None,
        duration_days=None,
        depth=0.0,
        epoch_phase_fraction=None,
        epoch_days=None,
        kind="native",
    )


def build_injection_cases(
    *,
    periods: Iterable[float],
    durations_hours: Iterable[float],
    depths: Iterable[float],
    epoch_phase_fractions: Iterable[float],
    include_native_zero: bool = True,
    seed: int = 123,
) -> tuple[InjectionCase, ...]:
    cases: list[InjectionCase] = []
    if include_native_zero:
        cases.append(native_zero_case())
    for index, (period, duration_h, depth, phase) in enumerate(
        product(periods, durations_hours, depths, epoch_phase_fractions)
    ):
        duration_days = float(duration_h) / 24.0
        cases.append(
            InjectionCase(
                injection_id=f"batman_{index:05d}",
                period_days=float(period),
                duration_days=duration_days,
                depth=float(depth),
                epoch_phase_fraction=float(phase),
                seed=int(seed) + int(index),
                kind="batman",
            )
        )
    return tuple(cases)


def realize_injection(lightcurve: LightCurve, case: InjectionCase) -> InjectionRealization:
    if case.is_native_zero:
        template = np.zeros(lightcurve.flux.shape, dtype=float)
        in_transit = np.zeros(lightcurve.flux.shape, dtype=bool)
        return InjectionRealization(
            case=case,
            lightcurve=lightcurve.with_flux(lightcurve.flux.copy(), metadata={"injection_id": case.injection_id}),
            additive_template=template,
            in_transit=in_transit,
            truth={},
            batman_used=False,
        )
    finite_time = lightcurve.time[np.isfinite(lightcurve.time)]
    if finite_time.size == 0:
        raise ValueError("Cannot place an injection without finite time values.")
    period = float(case.period_days)
    duration = float(case.duration_days)
    epoch = (
        float(case.epoch_days)
        if case.epoch_days is not None
        else float(np.min(finite_time) + float(case.epoch_phase_fraction) * period)
    )
    injected, template, in_transit, truth = inject_batman_transit(
        lightcurve.time,
        lightcurve.flux,
        period_days=period,
        epoch_days=epoch,
        duration_days=duration,
        depth=float(case.depth),
    )
    realized_case = InjectionCase(
        injection_id=case.injection_id,
        period_days=period,
        duration_days=duration,
        depth=float(case.depth),
        epoch_phase_fraction=case.epoch_phase_fraction,
        epoch_days=epoch,
        seed=case.seed,
        kind=case.kind,
    )
    return InjectionRealization(
        case=realized_case,
        lightcurve=lightcurve.with_flux(injected, metadata={"injection_id": case.injection_id}),
        additive_template=template,
        in_transit=in_transit,
        truth=truth.to_dict(),
        batman_used=True,
    )
