import numpy as np

from adaptive_transit.config import AdaptiveTransitConfig, PipelineSpec
from adaptive_transit.core import LightCurve
from adaptive_transit.detectors import DetectorContext, TransitDetector
from adaptive_transit.injection_plan import native_zero_case
from adaptive_transit.runner import UnifiedPipelineRunner
from adaptive_transit.schemas import DetectionResult
from adaptive_transit.treatments import RawTreatment


def small_lightcurve() -> LightCurve:
    time = np.linspace(0.0, 12.0, 160)
    flux = 0.0002 * np.sin(2 * np.pi * time / 3.0)
    return LightCurve(
        time=time,
        flux=flux,
        segment_id=np.zeros(time.size, dtype=int),
        usable_mask=np.ones(time.size, dtype=bool),
        row_present=np.ones(time.size, dtype=bool),
    )


def unit_config(tmp_path, combinations) -> AdaptiveTransitConfig:
    return AdaptiveTransitConfig(
        profile="unit",
        manifest_path=tmp_path / "manifest.csv",
        output_dir=tmp_path,
        target_limit=1,
        strict_target_count=False,
        active_combinations=tuple(combinations),
        include_native_zero_injection=False,
        min_period_days=1.0,
        max_period_days=4.0,
        n_periods=60,
        min_duration_hours=1.0,
        max_duration_hours=4.0,
        n_durations=3,
        detector_parameters={spec.detector: {} for spec in combinations},
        treatment_parameters={},
    )


def fake_characterizer(*_args, **_kwargs):
    return {}


class FixedDetector(TransitDetector):
    name = "bls"
    score_definition = "bls_score"
    score_definitions = ("bls_score",)

    def active_score_definitions(self, parameters):
        return self.score_definitions

    def search(self, lightcurve: LightCurve, context: DetectorContext) -> DetectionResult:
        return DetectionResult(
            success=True,
            best_period_days=2.0,
            best_epoch=1.0,
            best_duration_days=0.1,
            best_depth=0.001,
            raw_score=9.0,
            diagnostics={"score_definition": self.score_definition},
        )


def test_runner_reports_current_injection_treatment_detector(tmp_path):
    config = unit_config(tmp_path, (PipelineSpec("raw", "bls"),))
    events = []
    runner = UnifiedPipelineRunner(
        config,
        treatment_registry={"raw": RawTreatment()},
        detector_registry={"bls": FixedDetector()},
        characterizer=fake_characterizer,
    )

    runner.run_lightcurve(
        run_id="run",
        star_id="kic_1_q5",
        target_id="1",
        quarter=5,
        native=small_lightcurve(),
        injection_cases=(native_zero_case(),),
        progress_callback=events.append,
    )

    assert events == [
        {
            "stage": "detection",
            "star_id": "kic_1_q5",
            "injection_id": "native_zero",
            "injection_index": 1,
            "injection_total": 1,
            "treatment": "raw",
            "detector": "bls",
        }
    ]


def test_runner_reports_current_null_trial(tmp_path):
    config = unit_config(tmp_path, (PipelineSpec("raw", "bls"),))
    events = []
    runner = UnifiedPipelineRunner(
        config,
        treatment_registry={"raw": RawTreatment()},
        detector_registry={"bls": FixedDetector()},
        characterizer=fake_characterizer,
    )

    runner.run_null_scores(
        run_id="run",
        star_id="kic_1_q5",
        native=small_lightcurve(),
        n_trials=2,
        progress_callback=events.append,
    )

    assert events == [
        {"stage": "null_trial", "star_id": "kic_1_q5", "trial_index": 1, "trial_total": 2},
        {"stage": "null_trial", "star_id": "kic_1_q5", "trial_index": 2, "trial_total": 2},
    ]
