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
    assert args.pipelines == ("raw_bls", "raw_tcf", "arima_bls", "arima_tcf", "kalman_bls", "kalman_tcf", "gp_bls", "gp_tcf")
    assert args.reserve_cpu_cores == 2
    assert args.checkpoint_interval == 5
    assert args.prefetch_workers == 4

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
