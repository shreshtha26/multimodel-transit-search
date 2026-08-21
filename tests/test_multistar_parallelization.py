import importlib.util
import json
import sys
from pathlib import Path

def load_script(name, path):
    script_dir = str(Path(path).parent.resolve())
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def test_challenger_defaults_include_symmetric_pipelines():
    runner = load_script("run_multistar_challenger_benchmark_parallel", "scripts/run_multistar_challenger_benchmark.py")
    args = runner.default_settings("main")
    assert args.pipelines == runner.DEFAULT_PIPELINES
    assert set(args.pipelines) == {
        f"{branch}_{detector}"
        for branch in runner.BRANCHES
        for detector in runner.CORE_DETECTORS
    }
    for branch in runner.BRANCHES:
        for detector in runner.CORE_DETECTORS:
            assert f"{branch}_{detector}" in args.pipelines
        for detector in runner.CHALLENGER_DETECTORS:
            assert f"{branch}_{detector}" in runner.PIPELINE_DEFINITIONS
            assert f"{branch}_{detector}" not in args.pipelines
    assert args.reserve_cpu_cores == 2
    assert args.checkpoint_interval == 5
    assert args.prefetch_workers == 4

def test_legacy_challenger_cli_defaults_to_smoke_not_50_star_surface():
    runner = load_script("run_multistar_challenger_benchmark_default_profile", "scripts/run_multistar_challenger_benchmark.py")
    args = runner.parse_args([])
    assert args.profile == "smoke"
    assert args.target_limit == 2

def test_challenger_worker_count_leaves_two_cores(monkeypatch):
    runner = load_script("run_multistar_challenger_benchmark_cpu", "scripts/run_multistar_challenger_benchmark.py")
    monkeypatch.setattr(runner.os, "cpu_count", lambda: 10)
    args = runner.default_settings("main")
    assert runner.resolve_worker_count(args, 50) == 8
    args.max_workers = 6
    assert runner.resolve_worker_count(args, 50) == 6

def test_prepare_star_run_invalidates_stale_resume_files(tmp_path):
    runner = load_script("run_multistar_challenger_benchmark_resume", "scripts/run_multistar_challenger_benchmark.py")
    star_dir = tmp_path / "star"
    star_dir.mkdir()
    old_args = runner.default_settings("pilot")
    old_args.pipelines = ("raw_bls",)
    current_args = runner.default_settings("pilot")
    (star_dir / "run_config.json").write_text(json.dumps({"config_signature": runner.json_ready(runner.config_signature(old_args))}))
    for name in ("COMPLETE", "injections.csv", "star_summary.json", "failure.json"):
        (star_dir / name).write_text("stale")
    compatible = runner.prepare_star_run(star_dir, vars(current_args))
    assert not compatible
    for name in ("COMPLETE", "injections.csv", "star_summary.json", "failure.json"):
        assert not (star_dir / name).exists()
    assert runner.star_config_matches(star_dir, vars(current_args))

def test_calibration_defaults_use_safe_worker_resolution(monkeypatch):
    calibration = load_script("calibrate_multistar_challenger_benchmark_parallel", "scripts/calibrate_multistar_challenger_benchmark.py")
    monkeypatch.setattr(calibration.os, "cpu_count", lambda: 10)
    args = calibration.parse_args(["--profile", "pilot"])
    assert args.max_workers is None
    assert args.reserve_cpu_cores == 2
    assert args.checkpoint_interval == 5
    assert calibration.resolve_worker_count(args, 50) == 8

def test_characterization_uses_symmetric_detector_matched_lifts():
    analysis = load_script("analyze_multistar_characterization_effects_symmetric", "scripts/analyze_multistar_characterization_effects.py")
    import pandas as pd
    injections = pd.DataFrame([{"target_id": "1", "quarter": 5, "raw_bls_harmonic_rank1_matched": True, "raw_tcf_harmonic_rank1_matched": False, "arima_bls_harmonic_rank1_matched": False, "arima_tcf_harmonic_rank1_matched": True, "kalman_bls_harmonic_rank1_matched": True, "kalman_tcf_harmonic_rank1_matched": True, "gp_bls_harmonic_rank1_matched": False, "gp_tcf_harmonic_rank1_matched": True}])
    rates = analysis.aggregate_recovery_rates(injections).iloc[0]
    assert rates["arima_bls_lift"] == -1.0
    assert rates["arima_tcf_lift"] == 1.0
    assert rates["raw_best_harmonic_recovery_rate"] == 1.0
    assert rates["arima_best_pipeline_lift"] == 0.0

def test_characterization_remains_compatible_with_old_six_pipeline_outputs():
    analysis = load_script("analyze_multistar_characterization_effects_legacy", "scripts/analyze_multistar_characterization_effects.py")
    import pandas as pd
    injections = pd.DataFrame([{"target_id": "1", "quarter": 5, "raw_bls_harmonic_rank1_matched": False, "arima_tcf_harmonic_rank1_matched": True, "kalman_bls_harmonic_rank1_matched": False, "kalman_tcf_harmonic_rank1_matched": True, "gp_bls_harmonic_rank1_matched": False, "gp_tcf_harmonic_rank1_matched": False}])
    rates = analysis.aggregate_recovery_rates(injections).iloc[0]
    assert rates["raw_best_harmonic_recovery_rate"] == 0.0
    assert rates["arima_best_pipeline_lift"] == 1.0

def test_main_medium_search_resolution_overrides_high_defaults():
    runner = load_script("run_multistar_challenger_benchmark_medium", "scripts/run_multistar_challenger_benchmark.py")
    args = runner.parse_args(["--profile", "main", "--search-resolution", "medium"])
    assert args.search_resolution == "medium"
    assert args.n_periods == 5000
    assert args.n_coarse_periods == 2000
    assert args.n_refinement_regions == 18
    assert args.bls_oversample == 5
    assert args.kalman_injection_mode == "filter"
    assert args.gp_injection_mode == "filter"

def test_calibration_inherits_saved_benchmark_resolution(tmp_path):
    runner = load_script("run_multistar_challenger_benchmark_saved_config", "scripts/run_multistar_challenger_benchmark.py")
    args = runner.parse_args(["--profile", "main", "--search-resolution", "medium", "--output-dir", str(tmp_path)])
    runner.write_benchmark_config(args)
    calibration = load_script("calibrate_multistar_challenger_benchmark_saved_config", "scripts/calibrate_multistar_challenger_benchmark.py")
    calibration_args = calibration.parse_args(["--profile", "main", "--benchmark-dir", str(tmp_path)])
    assert calibration_args.search_resolution == "medium"
    assert calibration_args.n_periods == 5000
    assert calibration_args.n_coarse_periods == 2000
    assert calibration_args.pipelines == args.pipelines
    assert calibration_args.kalman_injection_mode == "filter"
    assert calibration_args.gp_injection_mode == "filter"

def test_detector_registry_adapts_tls_schema(monkeypatch):
    runner = load_script("run_multistar_challenger_benchmark_tls_registry", "scripts/run_multistar_challenger_benchmark.py")
    import numpy as np
    import pandas as pd

    def fake_tls(time, flux, **kwargs):
        return {
            "summary": {
                "period_days": 3.0,
                "duration_days": 0.2,
                "epoch_days": 1.0,
                "sde": 9.0,
                "snr": 7.0,
                "depth_raw": 0.001,
                "n_observations": int(np.isfinite(flux).sum()),
            },
            "periodogram": pd.DataFrame({"period_days": [2.0, 3.0, 4.0, 5.0, 6.0], "power": [1.0, 5.0, 1.0, 4.0, 1.0]}),
            "raw_result": object(),
        }

    monkeypatch.setattr(runner, "run_tls", fake_tls)
    args = vars(runner.default_settings("smoke"))
    time = np.linspace(0.0, 20.0, 500)
    flux = np.zeros(time.size)
    result = runner.run_detector_search("tls", time, flux, np.linspace(1.0, 5.0, 50), np.array([0.2]), args)
    assert result["summary"]["score"] == 9.0
    assert list(result["top_peaks"]["rank"]) == [1, 2]
    fields = runner.detector_result_fields("raw_tls", result, "tls", 3.0, args, 0.1)
    assert fields["raw_tls_score"] == 9.0
    assert fields["raw_tls_tls_snr"] == 7.0
    assert fields["raw_tls_harmonic_rank1_matched"]

def test_detector_registry_runs_tps_like_on_segment_grid():
    runner = load_script("run_multistar_challenger_benchmark_tps_registry", "scripts/run_multistar_challenger_benchmark.py")
    import numpy as np

    args = vars(runner.default_settings("smoke"))
    args.update({
        "min_period_days": 1.0,
        "max_period_days": 4.0,
        "top_k": 3,
        "tps_max_wavelet_level": 3,
        "tps_noise_window_cadences": 31,
        "tps_min_segment_cadences": 32,
        "min_transit_events": 2,
    })
    time = np.arange(0.0, 12.0, 0.1)
    flux = np.zeros(time.size)
    flux[np.arange(5, time.size, 20)] = -0.01
    segment_id = np.zeros(time.size, dtype=int)
    result = runner.run_detector_search("tps_like", time, flux, np.linspace(1.0, 4.0, 20), np.array([0.2]), args, segment_id=segment_id)
    assert result["summary"]["score"] == result["summary"]["mes"]
    assert {"period_days", "mes", "score"}.issubset(result["top_peaks"].columns)
