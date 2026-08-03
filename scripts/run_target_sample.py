"""Run the single-target multi-model-transit-search workflow over a configured target sample."""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
import pandas as pd
import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-model-transit-search over a Kepler target sample.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/kepler_target_sample.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/target_sample"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target-id", action="append", help="Only run specific target IDs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later targets if one target fails.",
    )
    return parser


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    if "targets" not in data or not isinstance(data["targets"], list):
        raise ValueError(f"{path} must contain a `targets` list.")
    return data


def target_prefix(target_id: str | int, quarter: int) -> str:
    clean_target = str(target_id).replace("KIC", "").strip()
    return f"kic_{clean_target}_q{quarter}"


def command_for_target(
    *,
    target: dict[str, Any],
    common_args: list[str],
    output_dir: Path,
) -> tuple[list[str], str, int]:
    target_id = str(target["target_id"])
    quarter = int(target.get("quarter", 5))
    command = [
        sys.executable,
        "scripts/run_single_target_arima.py",
        "--target-id",
        target_id,
        "--quarter",
        str(quarter),
        "--output-dir",
        str(output_dir),
        *common_args,
        *(str(value) for value in target.get("runner_args", [])),
    ]
    return command, target_id, quarter


def read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    common_args = [str(value) for value in config.get("runner_args", [])]
    requested_targets = set(args.target_id or [])

    targets = config["targets"]
    if requested_targets:
        targets = [target for target in targets if str(target["target_id"]) in requested_targets]
    if args.limit is not None:
        targets = targets[: args.limit]
    if not targets:
        raise ValueError("No targets selected.")

    metrics_dir = args.output_dir / "metrics"
    logs_dir = args.output_dir / "logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failed = False
    for target in targets:
        command, target_id, quarter = command_for_target(
            target=target,
            common_args=common_args,
            output_dir=args.output_dir,
        )
        prefix = target_prefix(target_id, quarter)
        log_path = logs_dir / f"{prefix}.log"

        if args.dry_run:
            print(" ".join(command))
            continue

        completed = subprocess.run(
            command,
            check=False,
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
        )
        log_path.write_text(completed.stdout + "\n" + completed.stderr)

        phase1 = read_json_if_present(metrics_dir / f"{prefix}_phase1_completion.json")
        recovery = read_json_if_present(metrics_dir / f"{prefix}_multi_injection_recovery_summary.json")
        row = {
            "target_id": target_id,
            "name": target.get("name", ""),
            "quarter": quarter,
            "return_code": completed.returncode,
            "log_path": str(log_path),
            "phase1_engineering_complete": phase1.get("phase1_engineering_complete"),
            "phase1_scientific_ready_for_phase2": phase1.get("phase1_scientific_ready_for_phase2"),
            "selected_quality_policy": phase1.get("selected_quality_policy"),
            "selected_mode": phase1.get("selected_mode"),
            "selected_order": phase1.get("selected_order"),
            "n_injections": recovery.get("n_injections"),
            "rank1_recovery_rate": recovery.get("rank1_recovery_rate"),
            "rank3_recovery_rate": recovery.get("rank3_recovery_rate"),
            "transit_preservation_failure_rate": recovery.get("transit_preservation_failure_rate"),
        }
        rows.append(row)
        print(f"{target_id} Q{quarter}: return_code={completed.returncode}")

        if completed.returncode != 0:
            failed = True
            if not args.continue_on_error:
                break

    if rows:
        summary = pd.DataFrame(rows)
        summary_path = metrics_dir / "target_sample_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Target-sample summary: {summary_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
