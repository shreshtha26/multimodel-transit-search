import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_transit.config import AdaptiveTransitConfig, PipelineSpec
from adaptive_transit.core import LightCurve, stable_seed
from adaptive_transit.detectors import DetectorContext, TCFDetector, TransitDetector
from adaptive_transit.injection_plan import InjectionCase, native_zero_case
from adaptive_transit.progress import LiveBenchmarkStore
from adaptive_transit.runner import UnifiedPipelineRunner
from adaptive_transit.schemas import DetectionResult
from adaptive_transit.treatments import RawTreatment


ROOT = Path(__file__).resolve().parents[1]
PANEL_SPEC = importlib.util.spec_from_file_location(
    "streamlit_characterization_v2_panel_for_tests",
    ROOT / "streamlit_characterization_v2_panel.py",
)
panel = importlib.util.module_from_spec(PANEL_SPEC)
assert PANEL_SPEC.loader is not None
PANEL_SPEC.loader.exec_module(panel)


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


def unit_config(tmp_path: Path, combinations) -> AdaptiveTransitConfig:
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
    return {
        "flux_robust_scale": 0.001,
        "flux_skewness": 0.1,
        "flux_outlier_fraction": 0.0,
        "v2_acf_lag_1": 0.2,
        "v2_acf_decay_e_days": 0.03,
        "original_series_stationarity_conclusion": "stationary_supported",
        "v2_spectral_concentration": 0.1,
        "v2_spectral_harmonic_power_ratio": 0.5,
        "v2_ls_dominant_period_days": 3.0,
        "v2_ls_acf_period_relative_error": 0.05,
        "v2_segment_scale_relative_mad": 0.02,
    }


class FixedDetector(TransitDetector):
    def __init__(self, name: str, calls: list | None = None, *, interrupt_on_trial=None, interrupt_always=False):
        self.name = name
        self.score_definition = f"{name}_score"
        self.score_definitions = (self.score_definition,)
        self.calls = calls if calls is not None else []
        self.interrupt_on_trial = interrupt_on_trial
        self.interrupt_always = interrupt_always

    def active_score_definitions(self, parameters):
        return self.score_definitions

    def search(self, lightcurve: LightCurve, context: DetectorContext) -> DetectionResult:
        trial = lightcurve.metadata.get("null_trial")
        self.calls.append((self.name, trial))
        if self.interrupt_always or (self.interrupt_on_trial is not None and trial == self.interrupt_on_trial):
            raise SimulatedInterruption("simulated interruption")
        return DetectionResult(
            success=True,
            best_period_days=2.0,
            best_epoch=1.0,
            best_duration_days=0.1,
            best_depth=0.001,
            raw_score=9.0,
            diagnostics={"score_definition": self.score_definition},
        )


class SimulatedInterruption(BaseException):
    pass


def test_tcf_adapter_documents_input_representation_and_preserves_series(monkeypatch):
    captured = {}

    def fake_run_tcf(time, flux, period_grid, duration_grid, **kwargs):
        captured["flux"] = np.asarray(flux).copy()
        return {
            "summary": {
                "period": 2.0,
                "epoch": 1.0,
                "duration": 0.1,
                "score": 5.0,
                "raw_pooled_score": 4.0,
                "edge_amplitude": 0.001,
                "n_edge_observations": 20,
                "n_valid_transit_events": 4,
                "positive_event_fraction": 1.0,
                "median_event_score": 2.5,
                "cadence_days": 0.02,
                "innovation_scale": 0.001,
                "search_mode": "coarse_to_fine",
                "requested_period_count": 3,
                "evaluated_period_count": 3,
            },
            "top_peaks": pd.DataFrame(
                {
                    "rank": [1],
                    "period_days": [2.0],
                    "score": [5.0],
                    "raw_pooled_score": [4.0],
                    "duration": [0.1],
                    "epoch": [1.0],
                    "n_valid_transit_events": [4],
                    "positive_event_fraction": [1.0],
                }
            ),
        }

    monkeypatch.setattr("adaptive_transit.detectors.run_tcf", fake_run_tcf)
    flux = np.array([0.0, 0.1, -0.1, 0.0])
    lc = LightCurve(time=np.arange(flux.size), flux=flux, metadata={"treatment": "gp"})
    result = TCFDetector().search(
        lc,
        DetectorContext(period_grid=np.array([1.0, 2.0, 3.0]), duration_grid=np.array([0.1])),
    )
    np.testing.assert_allclose(captured["flux"], flux)
    assert result.raw_score == 5.0
    assert "GP: GP residual representation" in result.diagnostics["tcf_input_representation"]
    assert "does not add a second" in result.diagnostics["tcf_internal_transform"]


def test_incremental_detection_persistence_survives_interruption(tmp_path):
    config = unit_config(tmp_path, (PipelineSpec("raw", "bls"), PipelineSpec("raw", "tcf")))
    store = LiveBenchmarkStore(tmp_path)
    runner = UnifiedPipelineRunner(
        config,
        treatment_registry={"raw": RawTreatment()},
        detector_registry={"bls": FixedDetector("bls"), "tcf": FixedDetector("tcf", interrupt_always=True)},
        characterizer=fake_characterizer,
    )
    with pytest.raises(SimulatedInterruption):
        runner.run_lightcurve(
            run_id="run",
            star_id="kic_1_q5",
            target_id="1",
            quarter=5,
            native=small_lightcurve(),
            injection_cases=(native_zero_case(),),
            progress_store=store,
        )

    detection = store.read_table("detection")
    assert len(detection) == 1
    assert detection.iloc[0]["detector"] == "bls"
    events = [json.loads(line) for line in (tmp_path / "run_events.jsonl").read_text().splitlines()]
    assert any(event.get("stage") == "detection" and event.get("detector") == "bls" for event in events)
    store.close()


def test_restart_skips_completed_detection_without_duplicates(tmp_path):
    config = unit_config(tmp_path, (PipelineSpec("raw", "bls"),))
    calls: list = []
    with LiveBenchmarkStore(tmp_path) as store:
        runner = UnifiedPipelineRunner(
            config,
            treatment_registry={"raw": RawTreatment()},
            detector_registry={"bls": FixedDetector("bls", calls)},
            characterizer=fake_characterizer,
        )
        kwargs = dict(
            run_id="run",
            star_id="kic_1_q5",
            target_id="1",
            quarter=5,
            native=small_lightcurve(),
            injection_cases=(native_zero_case(),),
            progress_store=store,
        )
        runner.run_lightcurve(**kwargs)
        runner.run_lightcurve(**kwargs)
        assert calls == [("bls", None)]
        assert len(store.read_table("detection")) == 1
        assert store.read_table("run_status")["status"].astype(str).eq("skipped_existing").any()


def test_null_trial_resume_skips_completed_trials(tmp_path):
    config = unit_config(tmp_path, (PipelineSpec("raw", "bls"),))
    calls: list = []
    with LiveBenchmarkStore(tmp_path) as store:
        runner = UnifiedPipelineRunner(
            config,
            treatment_registry={"raw": RawTreatment()},
            detector_registry={"bls": FixedDetector("bls", calls, interrupt_on_trial=2)},
            characterizer=fake_characterizer,
        )
        with pytest.raises(SimulatedInterruption):
            runner.run_null_scores(
                run_id="run",
                star_id="kic_1_q5",
                native=small_lightcurve(),
                n_trials=3,
                progress_store=store,
            )
        assert sorted(store.read_table("null_score")["trial"].astype(int).unique()) == [0, 1]

        restart_calls: list = []
        runner = UnifiedPipelineRunner(
            config,
            treatment_registry={"raw": RawTreatment()},
            detector_registry={"bls": FixedDetector("bls", restart_calls)},
            characterizer=fake_characterizer,
        )
        runner.run_null_scores(
            run_id="run",
            star_id="kic_1_q5",
            native=small_lightcurve(),
            n_trials=3,
            progress_store=store,
        )
        assert restart_calls == [("bls", 2)]
        assert sorted(store.read_table("null_score")["trial"].astype(int).unique()) == [0, 1, 2]


def test_live_store_import_export_filters_current_hash_and_preserves_history(tmp_path):
    old = pd.DataFrame(
        [
            {
                "run_id": "old",
                "config_hash": "oldhash",
                "star_id": "kic_old_q5",
                "injection_id": "i0",
                "treatment": "raw",
                "detector": "bls",
                "score_definition": "bls_power",
                "raw_score": 1.0,
            }
        ]
    )
    old.to_csv(tmp_path / "detection.csv", index=False)
    with LiveBenchmarkStore(tmp_path) as store:
        imported = store.import_existing_csvs(run_id="run", config_hash="newhash", compatible_only=True)
        assert "detection" not in imported
        store.upsert_row(
            "detection",
            {
                "run_id": "run",
                "config_hash": "newhash",
                "star_id": "kic_1_q5",
                "injection_id": "i0",
                "treatment": "raw",
                "detector": "bls",
                "score_name": "bls_power",
                "score_definition": "bls_power",
                "raw_score": 2.0,
            },
        )
        store.export_csvs(table_names=("detection",), run_id="run", config_hash="newhash")
    exported = pd.read_csv(tmp_path / "detection.csv")
    assert set(exported["config_hash"].astype(str)) == {"oldhash", "newhash"}
    assert exported.duplicated(["run_id", "config_hash", "star_id", "injection_id", "treatment", "detector", "score_definition"]).sum() == 0


def test_deterministic_null_seed_is_stable():
    assert stable_seed(456, "kic_1_q5", 643) == stable_seed(456, "kic_1_q5", 643)
    assert stable_seed(456, "kic_1_q5", 643) != stable_seed(456, "kic_1_q5", 644)


def test_streamlit_live_loader_reads_partial_star(tmp_path):
    with LiveBenchmarkStore(tmp_path) as store:
        store.upsert_row(
            "characterization",
            {
                "run_id": "run",
                "config_hash": "hash",
                "star_id": "kic_1_q5",
                "target_id": "1",
                "quarter": 5,
                "characterization_version": "unit",
                "success": True,
                "flux_robust_scale": 0.001,
            },
        )
        store.upsert_row(
            "detection",
            {
                "run_id": "run",
                "config_hash": "hash",
                "star_id": "kic_1_q5",
                "injection_id": "batman_00000",
                "treatment": "raw",
                "detector": "bls",
                "score_name": "bls_power",
                "score_definition": "bls_power",
                "success": True,
                "harmonic_recovery": True,
                "raw_score": 7.0,
            },
        )
    panel._read_live_table.clear()
    panel.load_live_benchmark_bundle.clear()
    bundle = panel.load_live_benchmark_bundle(str(tmp_path))
    assert bundle["characterization"]["target_id"].tolist() == ["1"]
    assert bundle["detection"]["target_id"].tolist() == ["1"]


def test_characterization_visual_missing_value_table_and_percentiles():
    features = pd.DataFrame(
        {
            "target_id": ["1", "2", "3"],
            "robust_scatter": [1.0, np.nan, 3.0],
            "skewness": [0.0, 1.0, 2.0],
        }
    )
    row = pd.Series({"target_id": "2", "robust_scatter": np.nan, "skewness": 1.0})
    table = panel._methods_table(row, features, "2")
    robust = table[table["Variable"].eq("Robust scatter")].iloc[0]
    skew = table[table["Variable"].eq("Skewness")].iloc[0]
    assert robust["Value"] == "--"
    assert robust["Population percentile"] == "Unavailable"
    assert skew["Population percentile"] == "50.0"


def test_acf_k_lag_table_marks_lags_after_one_as_diagnostic():
    time = np.arange(80, dtype=float)
    flux = np.sin(time / 4.0)
    table = panel._acf_lag_table(time, flux, max_lag=10)
    assert table["Lag"].tolist() == list(range(1, 11))
    assert table.iloc[0]["Role"] == "canonical"
    assert set(table.iloc[1:]["Role"]) == {"diagnostic"}


def test_variability_stability_display_renames_variance_domain():
    item = next(item for item in panel.CANONICAL_SCHEMA if item["feature"] == "segment_scale_variability")
    assert item["domain"] == "variance_evolution"
    assert item["domain_label"] == "Variability stability"


def test_pipeline_performance_marks_provisional_vs_calibrated():
    detection = pd.DataFrame(
        [
            {
                "target_id": "1",
                "star_id": "kic_1_q5",
                "injection_id": "batman_00000",
                "treatment": "raw",
                "detector": "bls",
                "success": True,
                "harmonic_recovery": True,
            }
        ]
    )
    injection = pd.DataFrame(
        [{"target_id": "1", "star_id": "kic_1_q5", "injection_id": "batman_00000", "batman_used": True}]
    )
    table, label = panel._star_pipeline_performance_table(detection, injection, "1")
    assert len(table) == 12
    assert label == "PROVISIONAL - NOT COMMON-FAP CALIBRATED"
    assert table.loc[table["pipeline"].eq("raw_bls"), "Recovery (%)"].iloc[0] == 100.0

    calibrated = detection.assign(above_threshold=True)
    _, label = panel._star_pipeline_performance_table(calibrated, injection, "1")
    assert label == "Common-FAP calibrated"
