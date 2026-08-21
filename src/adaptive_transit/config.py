"""Benchmark profiles and deterministic configuration hashes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/experiments/adaptive_transit"

ACTIVE_TREATMENTS = ("raw", "arima", "kalman", "gp")
ACTIVE_DETECTORS = ("bls", "tcf", "tps_like")
CHALLENGER_DETECTORS = ("tls", "trapezoid")
ACTIVE_SCIENTIFIC_BENCHMARKS = ("benchmark100", "benchmark1000")
EXPLORATORY_PROFILES = ("demo50",)
LEGACY_NONSCIENTIFIC_PROFILES = ("main50", "pilot10", "pilot5", "pilot1", "main", "pilot")


@dataclass(frozen=True)
class PipelineSpec:
    treatment: str
    detector: str

    @property
    def pipeline_id(self) -> str:
        return f"{self.treatment}_{self.detector}"


DEFAULT_ACTIVE_COMBINATIONS = tuple(
    PipelineSpec(treatment, detector)
    for treatment in ACTIVE_TREATMENTS
    for detector in ACTIVE_DETECTORS
)


@dataclass(frozen=True)
class AdaptiveTransitConfig:
    profile: str
    manifest_path: Path
    output_dir: Path
    target_limit: int
    strict_target_count: bool = True
    active_combinations: tuple[PipelineSpec, ...] = DEFAULT_ACTIVE_COMBINATIONS
    injection_period_grid: tuple[float, ...] = (2.0, 5.0, 10.0)
    injection_duration_hours_grid: tuple[float, ...] = (2.0, 4.0, 8.0)
    injection_depth_grid: tuple[float, ...] = (0.0002, 0.0005, 0.001)
    epoch_phase_fraction_grid: tuple[float, ...] = (0.15, 0.45, 0.75)
    include_native_zero_injection: bool = True
    min_period_days: float = 1.0
    max_period_days: float = 15.0
    n_periods: int = 5000
    min_duration_hours: float = 1.5
    max_duration_hours: float = 10.0
    n_durations: int = 8
    top_k: int = 5
    fap_level: float = 0.01
    n_null_trials_per_star: int = 1000
    period_match_tolerance_fraction: float = 0.02
    null_block_size_cadences: int = 24
    random_seed: int = 123
    null_generation_seed: int = 456
    quality_policy: str = "default"
    require_finite_flux_error: bool = False
    expected_quarter: int | None = 5
    allowed_selection_groups: tuple[str, ...] = ()
    treatment_parameters: dict[str, Any] = field(default_factory=dict)
    detector_parameters: dict[str, Any] = field(default_factory=dict)
    runner_module: str = "adaptive_transit.runner.UnifiedPipelineRunner"
    schema_version: int = 1

    @property
    def scientific(self) -> bool:
        return self.profile in ACTIVE_SCIENTIFIC_BENCHMARKS

    def to_hash_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["manifest_path"] = str(self.manifest_path)
        payload.pop("output_dir", None)
        payload["active_combinations"] = [asdict(item) for item in self.active_combinations]
        return payload

    @property
    def config_hash(self) -> str:
        return config_hash(self.to_hash_payload())


def config_hash(payload: Any) -> str:
    encoded = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def benchmark_profile(name: str) -> AdaptiveTransitConfig:
    profile = str(name)
    if profile == "benchmark100":
        return AdaptiveTransitConfig(
            profile=profile,
            manifest_path=CONFIG_ROOT / "kepler_q5_clean_100_star_manifest.csv",
            output_dir=OUTPUT_ROOT / "benchmark100",
            target_limit=100,
            n_periods=5000,
            top_k=5,
            allowed_selection_groups=("random_clean_q5_unstratified",),
        )
    if profile == "benchmark1000":
        return AdaptiveTransitConfig(
            profile=profile,
            manifest_path=CONFIG_ROOT / "kepler_q5_clean_1000_star_manifest.csv",
            output_dir=OUTPUT_ROOT / "benchmark1000",
            target_limit=1000,
            n_periods=10000,
            top_k=10,
            allowed_selection_groups=(
                "development_reference_100",
                "random_clean_q5_population_expansion",
            ),
        )
    if profile == "demo50":
        return AdaptiveTransitConfig(
            profile=profile,
            manifest_path=CONFIG_ROOT / "kepler_clean_background_manifest.csv",
            output_dir=OUTPUT_ROOT / "demo50",
            target_limit=50,
            strict_target_count=True,
            allowed_selection_groups=("catalog_clean_background",),
            injection_period_grid=(5.0,),
            injection_duration_hours_grid=(4.0,),
            injection_depth_grid=(0.0002, 0.0005, 0.001),
            epoch_phase_fraction_grid=(0.45,),
            include_native_zero_injection=False,
            n_periods=3000,
            top_k=5,
        )
    if profile == "smoke":
        return AdaptiveTransitConfig(
            profile=profile,
            manifest_path=CONFIG_ROOT / "kepler_q5_clean_100_star_manifest.csv",
            output_dir=OUTPUT_ROOT / "smoke",
            target_limit=1,
            strict_target_count=False,
            active_combinations=(
                PipelineSpec("raw", "bls"),
                PipelineSpec("arima", "bls"),
            ),
            injection_period_grid=(5.0,),
            injection_duration_hours_grid=(4.0,),
            injection_depth_grid=(0.001,),
            epoch_phase_fraction_grid=(0.45,),
            n_periods=300,
            n_durations=4,
            top_k=3,
            allowed_selection_groups=("random_clean_q5_unstratified",),
        )
    raise ValueError(
        "profile must be benchmark100, benchmark1000, demo50, or smoke. "
        "Legacy main50/pilot profiles are not active scientific benchmarks."
    )


def active_scientific_profiles() -> tuple[str, ...]:
    return ACTIVE_SCIENTIFIC_BENCHMARKS


def parse_pipeline_specs(values: Iterable[str]) -> tuple[PipelineSpec, ...]:
    specs = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        try:
            treatment, detector = item.split("_", 1)
        except ValueError as exc:
            raise ValueError(f"Pipeline must look like treatment_detector: {item}") from exc
        if treatment not in ACTIVE_TREATMENTS:
            raise ValueError(f"Unknown treatment: {treatment}")
        if detector not in ACTIVE_DETECTORS:
            raise ValueError(f"Unknown detector: {detector}")
        specs.append(PipelineSpec(treatment, detector))
    if not specs:
        raise ValueError("At least one pipeline combination is required.")
    return tuple(specs)
