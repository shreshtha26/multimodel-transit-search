import importlib.util
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


def test_run_experiment_builds_stage_settings(tmp_path):
    runner = load_script("run_experiment", "scripts/run_experiment.py")
    config = {"common_options": {"test_fraction": 0.2}, "stages": {"gap_mode_injection": {"output_subdir": "inj", "options": {"quality_policy": "default"}}}}

    settings = runner.stage_settings("gap_mode_injection", "11904151", 5, config, tmp_path)

    assert settings.target_id == "11904151"
    assert settings.quarter == 5
    assert settings.output_dir == tmp_path / "inj"
    assert settings.test_fraction == 0.2
    assert settings.quality_policy == "default"


def test_run_experiment_respects_requested_and_skipped_stages():
    runner = load_script("run_experiment", "scripts/run_experiment.py")
    config = {"stages": {"single_target_arima": {"enabled": True}, "gap_mode_comparison": {"enabled": False}, "gap_mode_injection": {"enabled": True}, "bls_baseline": {"enabled": False}}}

    stages = runner.selected_stages(config, stages=None, skip_stages=["single_target_arima"])

    assert stages == ["gap_mode_injection"]


def test_run_experiment_summarizes_bls_stage(tmp_path):
    runner = load_script("run_experiment", "scripts/run_experiment.py")
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "kic_11904151_q5_bls_summary.json").write_text('{"injected_period_days": 5.0, "recovered_period_days": 5.01, "period_matched": true}\n')

    summary = runner.summarize_stage("bls_baseline", tmp_path, "kic_11904151_q5")

    assert summary["injected_period_days"] == 5.0
    assert summary["recovered_period_days"] == 5.01
    assert summary["period_matched"] is True


def test_run_experiment_flattens_stage_summaries():
    runner = load_script("run_experiment", "scripts/run_experiment.py")
    record = {
        "target_id": "11904151",
        "quarter": 5,
        "config_path": "configs/phase2.yaml",
        "success": True,
        "stages": {"gap_mode_injection": {"return_code": 0, "success": True, "summary": {"injection_count": 27}}},
    }

    row = runner.flatten_record(record)

    assert row["gap_mode_injection_return_code"] == 0
    assert row["gap_mode_injection_injection_count"] == 27


def test_target_sample_summarizes_unified_record():
    sample = load_script("run_target_sample", "scripts/run_target_sample.py")
    target = {"target_id": 11904151, "quarter": 5, "name": "Kepler-10"}
    record = {
        "success": True,
        "stages": {
            "gap_mode_injection": {"summary": {"injection_count": 27, "best_median_snr_retention_mode": "full_grid_missing"}},
            "bls_baseline": {"summary": {"recovered_period_days": 5.0, "period_matched": True}},
        },
    }

    row = sample.target_summary_row(target, record)

    assert row["target_id"] == "11904151"
    assert row["gap_injection_count"] == 27
    assert row["gap_injection_best_snr_mode"] == "full_grid_missing"
    assert row["bls_recovered_period_days"] == 5.0
    assert row["bls_period_matched"] is True
