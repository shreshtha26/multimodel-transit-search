from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from adaptive_transit.config import (
    ACTIVE_SCIENTIFIC_BENCHMARKS,
    LEGACY_NONSCIENTIFIC_PROFILES,
    AdaptiveTransitConfig,
    PipelineSpec,
    benchmark_profile,
)
from adaptive_transit.core import LightCurve, stable_seed
from adaptive_transit.detectors import DETECTORS, DetectorContext, TPSLikeDetector, TransitDetector
from adaptive_transit.fap import threshold_key
from adaptive_transit.injection_plan import InjectionCase, native_zero_case, realize_injection
from adaptive_transit.resume import LongTableStore
from adaptive_transit.runner import UnifiedPipelineRunner
from adaptive_transit.schemas import DetectionResult, assert_long_schema
from adaptive_transit.treatments import BACKGROUND_MODELS, BackgroundTreatment, RawTreatment


def small_lightcurve() -> LightCurve:
    time = np.linspace(0.0, 12.0, 240)
    flux = 0.0002 * np.sin(time)
    return LightCurve(
        time=time,
        flux=flux,
        cadenceno=np.arange(time.size),
        segment_id=np.zeros(time.size, dtype=int),
        usable_mask=np.ones(time.size, dtype=bool),
        row_present=np.ones(time.size, dtype=bool),
    )


def unit_config(*, combinations=(PipelineSpec("raw", "probe"),), output_dir=Path(".")) -> AdaptiveTransitConfig:
    return AdaptiveTransitConfig(
        profile="unit",
        manifest_path=Path("unit_manifest.csv"),
        output_dir=Path(output_dir),
        target_limit=1,
        strict_target_count=False,
        active_combinations=tuple(combinations),
        include_native_zero_injection=False,
        min_period_days=1.0,
        max_period_days=4.0,
        n_periods=80,
        min_duration_hours=1.0,
        max_duration_hours=4.0,
        n_durations=3,
        detector_parameters={"probe": {}},
        treatment_parameters={},
    )


def fake_characterizer(*_args, **_kwargs):
    return {}


class ProbeDetector(TransitDetector):
    name = "probe"
    score_definition = "probe_score"
    score_definitions = ("probe_score",)

    def __init__(self, events=None):
        self.events = events if events is not None else []

    def search(self, lightcurve: LightCurve, context: DetectorContext) -> DetectionResult:
        assert context.preservation_row is not None
        self.events.append("detector")
        return DetectionResult(
            success=True,
            best_period_days=2.0,
            best_epoch=1.0,
            best_duration_days=0.1,
            best_depth=0.001,
            raw_score=9.0,
            diagnostics={"score_definition": self.score_definition},
        )


class CopyTreatment(BackgroundTreatment):
    name = "copy"

    def fit(self, lightcurve: LightCurve):
        return self

    def transform(self, lightcurve: LightCurve):
        from adaptive_transit.schemas import TreatmentResult

        return TreatmentResult(treatment=self.name, lightcurve=lightcurve.with_flux(lightcurve.flux.copy()))


def test_registry_loading_exposes_active_treatments_and_detectors():
    assert set(BACKGROUND_MODELS) == {"raw", "arima", "kalman", "gp"}
    assert set(DETECTORS) == {"bls", "tls", "trapezoid", "tps_like"}


def test_raw_treatment_identity_behaviour():
    lc = small_lightcurve()
    result = RawTreatment().fit_transform(lc)
    np.testing.assert_allclose(result.lightcurve.flux, lc.flux)
    assert result.diagnostics["identity"] is True


def test_same_batman_injection_is_shared_across_treatments():
    lc = small_lightcurve()
    case = InjectionCase(
        injection_id="shared",
        period_days=2.0,
        duration_days=2.0 / 24.0,
        depth=0.001,
        epoch_phase_fraction=0.25,
    )
    config = unit_config(combinations=(PipelineSpec("raw", "probe"), PipelineSpec("copy", "probe")))
    runner = UnifiedPipelineRunner(
        config,
        treatment_registry={"raw": RawTreatment(), "copy": CopyTreatment()},
        detector_registry={"probe": ProbeDetector()},
        characterizer=fake_characterizer,
    )
    result = runner.run_lightcurve(
        run_id="run",
        star_id="star",
        target_id="1",
        quarter=5,
        native=lc,
        injection_cases=(case,),
    )
    hashes = result.preservation.groupby("injection_id")["template_hash"].nunique()
    assert hashes.loc["shared"] == 1
    assert set(result.preservation["treatment"]) == {"raw", "copy"}


def test_characterization_uses_native_flux_only():
    lc = small_lightcurve()
    captured = {}

    def characterizer(time, values, **_kwargs):
        captured["values"] = np.asarray(values).copy()
        return {}

    case = InjectionCase(
        injection_id="injected",
        period_days=2.0,
        duration_days=2.0 / 24.0,
        depth=0.001,
        epoch_phase_fraction=0.25,
    )
    runner = UnifiedPipelineRunner(
        unit_config(),
        detector_registry={"probe": ProbeDetector()},
        characterizer=characterizer,
    )
    runner.run_lightcurve(run_id="run", star_id="star", target_id="1", quarter=5, native=lc, injection_cases=(case,))
    np.testing.assert_allclose(captured["values"], lc.flux)


def test_preservation_is_calculated_before_detector_logic():
    events = []
    runner = UnifiedPipelineRunner(
        unit_config(),
        detector_registry={"probe": ProbeDetector(events)},
        characterizer=fake_characterizer,
    )
    runner.run_lightcurve(
        run_id="run",
        star_id="star",
        target_id="1",
        quarter=5,
        native=small_lightcurve(),
        injection_cases=(native_zero_case(),),
    )
    assert events == ["detector"]


def test_long_format_schema_validity():
    runner = UnifiedPipelineRunner(
        unit_config(),
        detector_registry={"probe": ProbeDetector()},
        characterizer=fake_characterizer,
    )
    result = runner.run_lightcurve(
        run_id="run",
        star_id="star",
        target_id="1",
        quarter=5,
        native=small_lightcurve(),
        injection_cases=(native_zero_case(),),
    )
    assert_long_schema(result.characterization, "characterization")
    assert_long_schema(result.injection, "injection")
    assert_long_schema(result.preservation, "preservation")
    assert_long_schema(result.detection, "detection")


def test_fap_threshold_identity_includes_treatment_detector_and_score():
    raw_bls = threshold_key(star_id="s", treatment="raw", detector="bls", score_definition="bls_power")
    gp_bls = threshold_key(star_id="s", treatment="gp", detector="bls", score_definition="bls_power")
    raw_tls = threshold_key(star_id="s", treatment="raw", detector="tls", score_definition="tls_sde")
    assert raw_bls != gp_bls
    assert raw_bls != raw_tls


def test_tps_like_detection_emits_one_row_per_active_score(monkeypatch):
    import adaptive_transit.detectors as detectors

    class FakeTPS(detectors.TPSLikeDetector):
        def search(self, lightcurve: LightCurve, context: DetectorContext) -> DetectionResult:
            return DetectionResult(
                success=True,
                best_period_days=2.0,
                best_epoch=1.0,
                best_duration_days=0.1,
                best_depth=None,
                raw_score=11.0,
                diagnostics={
                    "score_definition": "tps_like_mes",
                    "score_values": {
                        "tps_like_mes": 11.0,
                        "tps_like_robust_veto_score": 4.0,
                        "tps_like_event_consistency_score": 0.75,
                    },
                    "hardened_summary": {
                        "period_days": 4.0,
                        "epoch_days": 1.5,
                        "duration_hours": 2.4,
                        "event_observability_fraction": 0.8,
                    },
                },
            )

    runner = UnifiedPipelineRunner(
        unit_config(combinations=(PipelineSpec("raw", "tps_like"),)),
        detector_registry={"tps_like": FakeTPS()},
        characterizer=fake_characterizer,
    )
    result = runner.run_lightcurve(
        run_id="run",
        star_id="star",
        target_id="1",
        quarter=5,
        native=small_lightcurve(),
        injection_cases=(
            InjectionCase(
                injection_id="case",
                period_days=2.0,
                duration_days=0.1,
                depth=0.001,
                epoch_phase_fraction=0.2,
            ),
        ),
    )
    assert set(result.detection["score_definition"]) == {
        "tps_like_mes",
        "tps_like_robust_veto_score",
        "tps_like_event_consistency_score",
    }
    assert set(result.detection["score_name"]) == set(result.detection["score_definition"])
    assert result.detection.groupby(["injection_id", "treatment", "detector"]).size().iloc[0] == 3
    by_score = result.detection.set_index("score_definition")
    assert by_score.loc["tps_like_mes", "best_period_days"] == 2.0
    assert by_score.loc["tps_like_robust_veto_score", "best_period_days"] == 4.0
    assert bool(by_score.loc["tps_like_robust_veto_score", "harmonic_recovery"])
    assert not bool(by_score.loc["tps_like_robust_veto_score", "exact_recovery"])


def test_moving_block_null_scores_use_same_detector_schema():
    runner = UnifiedPipelineRunner(
        unit_config(),
        detector_registry={"probe": ProbeDetector()},
        characterizer=fake_characterizer,
    )
    null_scores = runner.run_null_scores(
        run_id="run",
        star_id="star",
        native=small_lightcurve(),
        n_trials=2,
    )
    assert_long_schema(null_scores, "null_score")
    assert set(null_scores["score_definition"]) == {"probe_score"}
    assert set(null_scores["score_name"]) == {"probe_score"}
    assert len(null_scores) == 2


def test_native_zero_injection_skips_batman(monkeypatch):
    import adaptive_transit.injection_plan as injection_plan

    monkeypatch.setattr(
        injection_plan,
        "inject_batman_transit",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("BATMAN should not run")),
    )
    realization = realize_injection(small_lightcurve(), native_zero_case())
    assert realization.batman_used is False
    assert not realization.in_transit.any()


def test_tps_like_raw_and_hardened_score_identities_are_distinct():
    detector = TPSLikeDetector()
    assert "tps_like_mes" in detector.score_definitions
    assert "tps_like_robust_veto_score" in detector.score_definitions
    assert "tps_like_event_consistency_score" in detector.score_definitions


def test_benchmark100_and_benchmark1000_use_same_runner_and_frozen_manifests():
    profile100 = benchmark_profile("benchmark100")
    profile1000 = benchmark_profile("benchmark1000")
    assert profile100.runner_module == profile1000.runner_module
    assert profile100.manifest_path.name == "kepler_q5_clean_100_star_manifest.csv"
    assert profile1000.manifest_path.name == "kepler_q5_clean_1000_star_manifest.csv"
    assert profile100.n_null_trials_per_star == 1000
    assert profile1000.n_null_trials_per_star == 1000
    assert profile100.period_match_tolerance_fraction == profile1000.period_match_tolerance_fraction == 0.02
    assert profile100.allowed_selection_groups == ("random_clean_q5_unstratified",)
    assert profile1000.allowed_selection_groups == (
        "development_reference_100",
        "random_clean_q5_population_expansion",
    )


def test_null_trial_count_is_part_of_config_identity():
    default = benchmark_profile("benchmark100")
    engineering = replace(default, n_null_trials_per_star=100)
    assert default.config_hash != engineering.config_hash


def test_config_hash_ignores_shard_output_directory():
    default = benchmark_profile("benchmark100")
    shard_1 = replace(default, output_dir=default.output_dir / "shard_01")
    shard_4 = replace(default, output_dir=default.output_dir / "shard_04")
    assert default.config_hash == shard_1.config_hash == shard_4.config_hash
    assert "output_dir" not in default.to_hash_payload()


def test_null_seed_depends_only_on_config_seed_star_and_trial():
    config = benchmark_profile("benchmark100")
    star = "kic_1234567_q5"
    trial = 42
    seed = stable_seed(config.null_generation_seed, star, trial)
    shard_seed = stable_seed(replace(config, output_dir=config.output_dir / "shard_02").null_generation_seed, star, trial)
    other_trial_seed = stable_seed(config.null_generation_seed, star, trial + 1)
    assert seed == shard_seed
    assert seed != other_trial_seed


def test_50_star_profile_is_not_active_scientific_default():
    assert ACTIVE_SCIENTIFIC_BENCHMARKS == ("benchmark100", "benchmark1000")
    assert "main50" in LEGACY_NONSCIENTIFIC_PROFILES
    with pytest.raises(ValueError):
        benchmark_profile("main50")


def test_resume_store_does_not_duplicate_rows_and_filters_config_hash(tmp_path):
    store = LongTableStore(tmp_path)
    row = {
        "run_id": "run",
        "config_hash": "aaa",
        "star_id": "star",
        "injection_id": "native_zero",
        "treatment": "raw",
        "detector": "bls",
        "score_definition": "bls_power",
        "raw_score": 1.0,
    }
    store.append_rows("detection", [row])
    store.append_rows("detection", [row])
    assert len(store.read("detection")) == 1
    assert store.completed_keys("detection", config_hash="bbb") == set()


def test_cli_resume_expected_keys_include_detector_score_definitions():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_adaptive_transit_benchmark.py"
    spec = importlib.util.spec_from_file_location("adaptive_benchmark_cli", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    config = unit_config(combinations=(PipelineSpec("raw", "tps_like"),))
    keys = module.expected_detection_keys(config, "run", "star", ("native_zero",))
    assert keys == {
        (
            "run",
            config.config_hash,
            "star",
            "native_zero",
            "raw",
            "tps_like",
            "tps_like_mes",
        ),
        (
            "run",
            config.config_hash,
            "star",
            "native_zero",
            "raw",
            "tps_like",
            "tps_like_robust_veto_score",
        ),
        (
            "run",
            config.config_hash,
            "star",
            "native_zero",
            "raw",
            "tps_like",
            "tps_like_event_consistency_score",
        ),
    }


def test_cli_dry_run_exposes_calibration_mode(capsys):
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_adaptive_transit_benchmark.py"
    spec = importlib.util.spec_from_file_location("adaptive_benchmark_cli_dry", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    rc = module.main(["--profile", "smoke", "--calibrate-fap", "--n-null-trials-per-star", "2", "--dry-run"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "mode=null_calibration" in captured
    assert "active_combinations=raw_bls,arima_bls" in captured


def test_cli_calibration_default_is_scientific_1000_nulls(capsys):
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_adaptive_transit_benchmark.py"
    spec = importlib.util.spec_from_file_location("adaptive_benchmark_cli_default_nulls", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    rc = module.main(["--profile", "benchmark100", "--calibrate-fap", "--dry-run"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "n_null_trials_per_star=1000" in captured


def test_benchmark100_four_shard_partition_reconstructs_frozen_manifest():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_adaptive_transit_benchmark.py"
    spec = importlib.util.spec_from_file_location("adaptive_benchmark_cli_shards", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    config = benchmark_profile("benchmark100")
    manifest = module.load_manifest(config.manifest_path, config.target_limit, config)
    summary = module.verify_shard_partition(manifest, num_shards=4)
    assert summary["unique_stars"] == 100
    assert summary["overlap_count"] == 0
    assert summary["missing_count"] == 0
    assert summary["unexpected_count"] == 0
    assert summary["reconstructs_manifest"] is True
    assert [(item["row_start_1based"], item["row_end_1based"], item["star_count"]) for item in summary["shards"]] == [
        (1, 25, 25),
        (26, 50, 25),
        (51, 75, 25),
        (76, 100, 25),
    ]


def test_cli_shard_dry_run_reports_separate_output_directory(capsys):
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_adaptive_transit_benchmark.py"
    spec = importlib.util.spec_from_file_location("adaptive_benchmark_cli_shard_dry", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    rc = module.main(["--profile", "benchmark100", "--num-shards", "4", "--shard-id", "2", "--dry-run"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "selected_target_count=25" in captured
    assert "shard_name=shard_02" in captured
    assert "shard_rows_1based=26-50" in captured
    assert "output_dir=" in captured
    assert "shard_02" in captured


def test_frozen_benchmark_manifests_pass_unified_guards():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_adaptive_transit_benchmark.py"
    spec = importlib.util.spec_from_file_location("adaptive_benchmark_cli_manifest", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    for name, expected_rows in (("benchmark100", 100), ("benchmark1000", 1000)):
        config = benchmark_profile(name)
        manifest = module.load_manifest(config.manifest_path, config.target_limit, config)
        assert len(manifest) == expected_rows
        assert manifest.duplicated(["target_id", "quarter"]).sum() == 0
        assert set(manifest["quarter"]) == {5}
        assert set(manifest["selection_group"]).issubset(set(config.allowed_selection_groups))


def test_cli_rejects_thresholds_from_different_config_hash(tmp_path):
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_adaptive_transit_benchmark.py"
    spec = importlib.util.spec_from_file_location("adaptive_benchmark_cli_thresholds", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    path = tmp_path / "thresholds.csv"
    pd.DataFrame(
        [
            {
                "config_hash": "not-this-config",
                "star_id": "star",
                "treatment": "raw",
                "detector": "bls",
                "score_definition": "bls_power",
                "fap_level": 0.01,
                "fap_threshold": 1.0,
            }
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="does not contain rows for config_hash"):
        module.load_thresholds(path, benchmark_profile("smoke"))


def test_shard_qc_and_merge_synthetic_outputs(tmp_path):
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_adaptive_transit_benchmark.py"
    spec = importlib.util.spec_from_file_location("adaptive_benchmark_cli_merge", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    manifest = pd.DataFrame(
        {
            "target_id": ["1001", "1002", "1003", "1004"],
            "quarter": [5, 5, 5, 5],
            "selection_group": ["unit"] * 4,
        }
    )
    config = AdaptiveTransitConfig(
        profile="unit",
        manifest_path=tmp_path / "manifest.csv",
        output_dir=tmp_path / "benchmark100",
        target_limit=4,
        strict_target_count=True,
        active_combinations=(PipelineSpec("raw", "bls"),),
        injection_period_grid=(2.0,),
        injection_duration_hours_grid=(2.0,),
        injection_depth_grid=(0.001,),
        epoch_phase_fraction_grid=(0.25,),
        include_native_zero_injection=False,
        n_null_trials_per_star=2,
        allowed_selection_groups=("unit",),
    )
    run_id = f"{config.profile}_{config.config_hash}"
    manifest.to_csv(config.manifest_path, index=False)

    for shard_id in range(1, 5):
        shard_dir = config.output_dir / module.shard_name(shard_id)
        shard = module.shard_manifest(manifest, num_shards=4, shard_id=shard_id)
        rows = module.shard_slices(len(manifest), 4)[shard_id - 1]
        module.write_shard_metadata(
            config,
            run_id,
            shard_dir,
            mode="benchmark",
            shard_id=shard_id,
            num_shards=4,
            shard_rows=rows,
        )
        star = module.star_id(shard.iloc[0]["target_id"], shard.iloc[0]["quarter"])
        injection_id = "batman_00000"
        pd.DataFrame([{"run_id": run_id, "config_hash": config.config_hash, "star_id": star}]).to_csv(
            shard_dir / "characterization.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "config_hash": config.config_hash,
                    "star_id": star,
                    "injection_id": injection_id,
                    "batman_used": True,
                }
            ]
        ).to_csv(shard_dir / "injection.csv", index=False)
        pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "config_hash": config.config_hash,
                    "star_id": star,
                    "injection_id": injection_id,
                    "treatment": "raw",
                }
            ]
        ).to_csv(shard_dir / "preservation.csv", index=False)
        pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "config_hash": config.config_hash,
                    "star_id": star,
                    "injection_id": injection_id,
                    "treatment": "raw",
                    "detector": "bls",
                    "score_definition": "bls_power",
                    "success": True,
                    "exact_recovery": True,
                    "harmonic_recovery": True,
                    "exact_period_error": 0.0,
                    "harmonic_period_error": 0.0,
                }
            ]
        ).to_csv(shard_dir / "detection.csv", index=False)
        pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "config_hash": config.config_hash,
                    "star_id": star,
                    "trial": trial,
                    "trial_seed": stable_seed(config.null_generation_seed, star, trial),
                    "treatment": "raw",
                    "detector": "bls",
                    "score_definition": "bls_power",
                    "score": 1.0 + trial,
                    "success": True,
                }
                for trial in range(2)
            ]
        ).to_csv(shard_dir / "null_score.csv", index=False)
        pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "config_hash": config.config_hash,
                    "star_id": star,
                    "treatment": "raw",
                    "detector": "bls",
                    "score_name": "bls_power",
                    "score_definition": "bls_power",
                    "fap_level": config.fap_level,
                    "fap_threshold": 2.0,
                    "null_trial_count": 2,
                }
            ]
        ).to_csv(shard_dir / "fap_thresholds.csv", index=False)

    summary = module.merge_shard_outputs(config.output_dir, config, run_id, manifest, num_shards=4)
    assert summary["unique_stars"] == 4
    assert summary["overlap_count"] == 0
    assert summary["missing_count"] == 0
    assert summary["unexpected_count"] == 0
    assert summary["config_hashes_identical"] is True
    assert pd.read_csv(config.output_dir / "detection.csv")["star_id"].nunique() == 4
    assert pd.read_csv(config.output_dir / "null_score.csv").shape[0] == 8
