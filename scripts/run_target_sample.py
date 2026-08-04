"""Run the curated target sample through the unified experiment path."""

from pathlib import Path
import pandas as pd
import yaml
try:
    from scripts.run_experiment import project_path, run_experiment
except ModuleNotFoundError:
    from run_experiment import project_path, run_experiment

CONFIG_PATH = Path("configs/kepler_target_sample.yaml")
EXPERIMENT_CONFIG_PATH = None
OUTPUT_DIR = Path("outputs/target_sample")
LIMIT = None
TARGET_IDS = None
STAGES = None
SKIP_STAGES = ()
CONTINUE_ON_ERROR = False
DRY_RUN = False


def load_config(path):
    path = project_path(path)
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    if "targets" not in config or not isinstance(config["targets"], list):
        raise ValueError(f"{path} must contain a `targets` list.")
    return config


def target_prefix(target_id, quarter):
    return f"kic_{str(target_id).replace('KIC', '').strip()}_q{quarter}"


def selected_targets(config, target_ids=None, limit=None):
    targets = config["targets"]
    if target_ids:
        wanted = {str(target_id) for target_id in target_ids}
        targets = [target for target in targets if str(target["target_id"]) in wanted]
    if limit is not None:
        targets = targets[:limit]
    if not targets:
        raise ValueError("No targets selected.")
    return targets


def target_summary_row(target, record):
    stages = record.get("stages", {})
    phase1 = stages.get("single_target_arima", {}).get("summary", {})
    gap = stages.get("gap_mode_comparison", {}).get("summary", {})
    injection = stages.get("gap_mode_injection", {}).get("summary", {})
    bls = stages.get("bls_baseline", {}).get("summary", {})
    return {
        "target_id": str(target["target_id"]),
        "name": target.get("name", ""),
        "quarter": int(target.get("quarter", 5)),
        "experiment_success": record.get("success"),
        "phase1_engineering_complete": phase1.get("phase1_engineering_complete"),
        "phase1_scientific_ready_for_phase2": phase1.get("phase1_scientific_ready_for_phase2"),
        "selected_quality_policy": phase1.get("selected_quality_policy"),
        "selected_mode": phase1.get("selected_mode"),
        "selected_order": phase1.get("selected_order"),
        "gap_best_available_mode": gap.get("best_available_gap_mode"),
        "gap_best_available_order": gap.get("best_available_selected_order"),
        "gap_scientifically_acceptable": gap.get("best_available_scientifically_acceptable"),
        "gap_no_scientifically_acceptable_combination": gap.get("no_scientifically_acceptable_combination"),
        "gap_injection_count": injection.get("injection_count"),
        "gap_injection_best_snr_mode": injection.get("best_median_snr_retention_mode"),
        "gap_injection_far_0.01_top_recovery": injection.get("any_mode_top_recovers_all_at_far_0.01"),
        "bls_injected_period_days": bls.get("injected_period_days"),
        "bls_recovered_period_days": bls.get("recovered_period_days"),
        "bls_period_error_fraction": bls.get("period_error_fraction"),
        "bls_period_matched": bls.get("period_matched"),
        "bls_injected_beats_null": bls.get("injected_beats_null"),
    }


def run_target_sample(**options):
    config_path = options.get("config_path", CONFIG_PATH)
    experiment_config_path = options.get("experiment_config_path", EXPERIMENT_CONFIG_PATH)
    output_dir = options.get("output_dir", OUTPUT_DIR)
    limit = options.get("limit", LIMIT)
    target_ids = options.get("target_ids", TARGET_IDS)
    stages = options.get("stages", STAGES)
    skip_stages = options.get("skip_stages", SKIP_STAGES)
    continue_on_error = options.get("continue_on_error", CONTINUE_ON_ERROR)
    dry_run = options.get("dry_run", DRY_RUN)
    output_dir = project_path(output_dir)
    config = load_config(config_path)
    experiment_config_path = experiment_config_path or Path(config.get("experiment_config", "configs/phase2.yaml"))
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failed = False

    for target in selected_targets(config, target_ids, limit):
        target_id = str(target["target_id"])
        quarter = int(target.get("quarter", 5))
        try:
            record = run_experiment(target_id, quarter, experiment_config_path, output_dir, stages, skip_stages, continue_on_error, dry_run)
        except Exception as exc:
            record = {"success": False, "stages": {}, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(target_summary_row(target, record))
        print(f"{target_id} Q{quarter}: success={record.get('success')}")
        if not record.get("success"):
            failed = True
            if not continue_on_error:
                break

    summary = pd.DataFrame(rows)
    summary_path = metrics_dir / "target_sample_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Target-sample summary: {summary_path}")
    return summary, not failed


def main():
    _, ok = run_target_sample()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
