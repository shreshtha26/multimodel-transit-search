"""Run one configured experiment record."""

import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_ID = "11904151"
QUARTER = 5
CONFIG_PATH = Path("configs/phase2.yaml")
OUTPUT_DIR = Path("outputs/experiments")
STAGES = None
SKIP_STAGES = ()
CONTINUE_ON_ERROR = False
DRY_RUN = False
STAGE_ORDER = ("single_target_arima", "gap_mode_comparison", "gap_mode_injection", "bls_baseline")
STAGE_SPECS = {
    "single_target_arima": {"script": "run_single_target_arima.py", "output_subdir": "single_target"},
    "gap_mode_comparison": {"script": "run_gap_mode_comparison.py", "output_subdir": "gap_modes"},
    "gap_mode_injection": {"script": "run_gap_mode_injection_experiment.py", "output_subdir": "injections"},
    "bls_baseline": {"script": "run_bls_baseline.py", "output_subdir": "bls_baseline"},
}
BASE_DEFAULTS = {
    "orders": None,
    "test_fraction": 0.20,
    "acf_lags": 80,
    "transit_lag_min": 3,
    "transit_lag_max": 24,
    "stationarity_alpha": 0.05,
    "stationarity_min_observations": 24,
    "adf_regression": "c",
    "adf_autolag": "AIC",
    "kpss_regression": "c",
    "kpss_nlags": "auto",
    "fit_maxiter": 200,
    "require_finite_flux_error": False,
}
STAGE_DEFAULTS = {
    "single_target_arima": {
        "expanded_arima_grid": False,
        "max_p": 5,
        "max_d": 1,
        "max_q": 5,
        "max_total_order": None,
        "quality_policies": None,
        "stability_folds": 3,
        "stability_segments": 3,
        "scale_window": 96,
        "injection_depth": 0.001,
        "injection_duration_cadences": 6,
        "injection_depth_grid": None,
        "injection_duration_grid": None,
        "injection_centers_per_duration": 3,
        "injection_max_segments": 3,
        "injection_local_half_width_cadences": 24,
        "scan_stride": 10,
        "scan_max_centers": 250,
    },
    "gap_mode_comparison": {
        "quality_policies": None,
        "gap_modes": None,
        "correlogram_lags": 24,
        "interpolation_method": "linear",
        "max_interpolated_gap_cadences": 12,
        "edge_extrapolation": False,
    },
    "gap_mode_injection": {
        "quality_policy": "default",
        "gap_modes": None,
        "short_acf_lags": 24,
        "interpolation_method": "linear",
        "max_interpolated_gap_cadences": 12,
        "edge_extrapolation": False,
        "injection_depth_grid": (0.0005, 0.001, 0.002),
        "injection_duration_grid": (6,),
        "centers_per_duration": 3,
        "local_half_width_cadences": 24,
        "scan_stride": 20,
        "scan_max_centers": 120,
        "scale_window": 96,
        "false_alarm_rates": (0.10, 0.05, 0.01),
    },
    "bls_baseline": {
        "quality_policy": "default",
        "injection_period_days": 5.0,
        "injection_epoch_offset_days": 1.0,
        "injection_duration_hours": 4.0,
        "injection_depth": 0.001,
        "min_period_days": 1.0,
        "max_period_days": 15.0,
        "n_periods": 1000,
        "min_duration_hours": 1.5,
        "max_duration_hours": 10.0,
        "n_durations": 8,
        "objective": "snr",
        "top_k": 10,
        "period_match_tolerance_fraction": 0.02,
    },
}
ALIASES = {
    "single_target_arima": {"quality_policy": "quality_policies"},
    "gap_mode_comparison": {"quality_policy": "quality_policies"},
}


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def target_prefix(target_id, quarter):
    return f"kic_{str(target_id).replace('KIC', '').strip()}_q{quarter}"


def load_config(path):
    path = project_path(path)
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    if config.get("stages") is not None and not isinstance(config["stages"], dict):
        raise ValueError(f"{path} `stages` must be a mapping.")
    return config


def selected_stages(config, stages=None, skip_stages=()):
    stage_config = config.get("stages", {}) or {}
    skipped = set(skip_stages or ())
    stages = stages or STAGE_ORDER
    picked = [stage for stage in stages if stage not in skipped and stage_config.get(stage, {}).get("enabled", True)]
    if not picked:
        raise ValueError("No stages selected.")
    return picked


def stage_output_dir(base_output_dir, stage, config):
    stage_config = (config.get("stages", {}) or {}).get(stage, {})
    return Path(base_output_dir) / str(stage_config.get("output_subdir", STAGE_SPECS[stage]["output_subdir"]))


def normalized_options(stage, options):
    aliases = ALIASES.get(stage, {})
    clean = {}
    for key, value in (options or {}).items():
        clean[aliases.get(key, key)] = value
    return clean


def stage_settings(stage, target_id, quarter, config, base_output_dir):
    stage_config = (config.get("stages", {}) or {}).get(stage, {})
    settings = {**BASE_DEFAULTS, **STAGE_DEFAULTS[stage]}
    settings.update(normalized_options(stage, config.get("common_options")))
    settings.update(normalized_options(stage, stage_config.get("options")))
    settings.update({"target_id": str(target_id), "quarter": int(quarter), "output_dir": stage_output_dir(base_output_dir, stage, config)})
    return SimpleNamespace(**settings)


def load_stage(stage):
    script_path = Path(__file__).resolve().parent / STAGE_SPECS[stage]["script"]
    spec = importlib.util.spec_from_file_location(f"multimodel_{stage}", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def stage_output_paths(stage, output_dir, prefix):
    metrics = Path(output_dir) / "metrics"
    if stage == "single_target_arima":
        return {
            "phase1_completion": str(metrics / f"{prefix}_phase1_completion.json"),
            "recovery_summary": str(metrics / f"{prefix}_multi_injection_recovery_summary.json"),
        }
    if stage == "gap_mode_comparison":
        return {
            "comparison_csv": str(metrics / f"{prefix}_gap_mode_comparison.csv"),
            "report_json": str(metrics / f"{prefix}_gap_mode_report.json"),
            "plot_manifest": str(metrics / f"{prefix}_gap_mode_plot_manifest.csv"),
        }
    if stage == "gap_mode_injection":
        return {
            "summary_csv": str(metrics / f"{prefix}_gap_mode_injection_summary.csv"),
            "report_json": str(metrics / f"{prefix}_gap_mode_injection_report.json"),
            "results_csv": str(metrics / f"{prefix}_gap_mode_injection_results.csv"),
        }
    if stage == "bls_baseline":
        return {
            "summary_json": str(metrics / f"{prefix}_bls_summary.json"),
            "periodogram_csv": str(metrics / f"{prefix}_bls_periodogram.csv"),
            "top_peaks_csv": str(metrics / f"{prefix}_bls_top_peaks.csv"),
        }
    return {}


def summarize_stage(stage, output_dir, prefix):
    metrics = Path(output_dir) / "metrics"
    if stage == "single_target_arima":
        phase1 = read_json(metrics / f"{prefix}_phase1_completion.json")
        recovery = read_json(metrics / f"{prefix}_multi_injection_recovery_summary.json")
        return {
            "phase1_engineering_complete": phase1.get("phase1_engineering_complete"),
            "phase1_scientific_ready_for_phase2": phase1.get("phase1_scientific_ready_for_phase2"),
            "selected_quality_policy": phase1.get("selected_quality_policy"),
            "selected_mode": phase1.get("selected_mode"),
            "selected_order": phase1.get("selected_order"),
            "n_injections": recovery.get("n_injections"),
            "rank1_recovery_rate": recovery.get("rank1_recovery_rate"),
            "rank3_recovery_rate": recovery.get("rank3_recovery_rate"),
        }
    if stage == "gap_mode_comparison":
        report = read_json(metrics / f"{prefix}_gap_mode_report.json")
        best = report.get("best_available", {})
        return {
            "same_candidate_family_across_modes": report.get("same_candidate_family_across_modes"),
            "no_scientifically_acceptable_combination": report.get("no_scientifically_acceptable_combination"),
            "best_available_gap_mode": best.get("gap_mode"),
            "best_available_selected_order": best.get("selected_order"),
            "best_available_stationarity_conclusion": best.get("stationarity_conclusion"),
            "best_available_scientifically_acceptable": best.get("scientifically_acceptable"),
        }
    if stage == "gap_mode_injection":
        report = read_json(metrics / f"{prefix}_gap_mode_injection_report.json")
        return {
            "injection_count": report.get("injection_count"),
            "best_median_snr_retention_mode": report.get("best_median_snr_retention_mode"),
            "highest_spurious_peak_rate_mode": report.get("highest_spurious_peak_rate_mode"),
            "any_mode_top_recovers_all_at_far_0.01": report.get("any_mode_top_recovers_all_at_far_0.01"),
            "any_mode_scientifically_acceptable_before_injection": report.get("any_mode_scientifically_acceptable_before_injection"),
        }
    if stage == "bls_baseline":
        summary = read_json(metrics / f"{prefix}_bls_summary.json")
        return {
            "injected_period_days": summary.get("injected_period_days"),
            "recovered_period_days": summary.get("recovered_period_days"),
            "period_error_fraction": summary.get("period_error_fraction"),
            "period_matched": summary.get("period_matched"),
            "injected_beats_null": summary.get("injected_beats_null"),
        }
    return {}


def flatten_record(record):
    row = {"target_id": record["target_id"], "quarter": record["quarter"], "config_path": record["config_path"], "success": record["success"]}
    for stage, stage_record in record["stages"].items():
        row[f"{stage}_return_code"] = stage_record["return_code"]
        row[f"{stage}_success"] = stage_record["success"]
        for key, value in stage_record.get("summary", {}).items():
            row[f"{stage}_{key}"] = value
    return row


def save_record(record, output_dir, prefix):
    records_dir = Path(output_dir) / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    json_path = records_dir / f"{prefix}_experiment_record.json"
    csv_path = records_dir / f"{prefix}_experiment_record.csv"
    json_path.write_text(json.dumps(record, indent=2) + "\n")
    pd.DataFrame([flatten_record(record)]).to_csv(csv_path, index=False)
    return json_path, csv_path


def run_experiment(target_id=TARGET_ID, quarter=QUARTER, config_path=CONFIG_PATH, output_dir=OUTPUT_DIR, stages=STAGES, skip_stages=SKIP_STAGES, continue_on_error=CONTINUE_ON_ERROR, dry_run=DRY_RUN):
    config_path = project_path(config_path)
    output_dir = project_path(output_dir)
    config = load_config(config_path)
    prefix = target_prefix(target_id, quarter)
    record = {"target_id": str(target_id), "quarter": int(quarter), "config_path": str(config_path), "output_dir": str(output_dir), "dry_run": bool(dry_run), "success": True, "stages": {}}

    for stage in selected_stages(config, stages, skip_stages):
        settings = stage_settings(stage, target_id, quarter, config, output_dir)
        outputs = stage_output_paths(stage, settings.output_dir, prefix)
        if dry_run:
            print(f"{stage}: output_dir={settings.output_dir}")
            record["stages"][stage] = {"output_dir": str(settings.output_dir), "return_code": None, "success": None, "outputs": outputs, "summary": {}}
            continue

        start = time.perf_counter()
        try:
            return_code = int(load_stage(stage).main(settings))
        except Exception as exc:
            return_code = 1
            record["stages"][stage] = {
                "output_dir": str(settings.output_dir),
                "return_code": return_code,
                "success": False,
                "runtime_seconds": float(time.perf_counter() - start),
                "outputs": outputs,
                "summary": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            ok = return_code == 0
            record["stages"][stage] = {
                "output_dir": str(settings.output_dir),
                "return_code": return_code,
                "success": ok,
                "runtime_seconds": float(time.perf_counter() - start),
                "outputs": outputs,
                "summary": summarize_stage(stage, settings.output_dir, prefix) if ok else {},
            }
        print(f"{stage}: return_code={return_code}")
        if return_code != 0:
            record["success"] = False
            if not continue_on_error:
                break

    save_record(record, output_dir, prefix)
    return record


def main():
    record = run_experiment()
    prefix = target_prefix(TARGET_ID, QUARTER)
    print(f"Experiment record: {OUTPUT_DIR / 'records' / f'{prefix}_experiment_record.json'}")
    return 0 if record["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
