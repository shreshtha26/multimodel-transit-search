import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_adaptive_transit_benchmark_for_tests",
    ROOT / "scripts/run_adaptive_transit_benchmark.py",
)
benchmark = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC.loader is not None
SCRIPT_SPEC.loader.exec_module(benchmark)


def _manifest(path: Path) -> Path:
    manifest = path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "target_id": "1",
                "quarter": 5,
                "selection_group": "random_clean_q5_unstratified",
            }
        ]
    ).to_csv(manifest, index=False)
    return manifest


def test_data_fetch_failure_is_checkpointed_and_skipped_on_resume(monkeypatch, tmp_path):
    output_dir = tmp_path / "out"
    manifest = _manifest(tmp_path)
    calls = []

    def fail_loader(*_args, **_kwargs):
        calls.append("called")
        raise RuntimeError("unit fetch failure")

    monkeypatch.setattr(benchmark, "load_cached_kepler_pdcsap_frame", fail_loader)
    argv = [
        "--profile",
        "smoke",
        "--manifest-path",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--target-limit",
        "1",
        "--no-download",
    ]

    assert benchmark.main(argv) == 0
    assert calls == ["called"]
    status = pd.read_csv(output_dir / "run_status.csv")
    fetch = status[status["stage"].astype(str).eq("data_fetch")].iloc[0]
    assert fetch["status"] == "failed"
    assert "unit fetch failure" in fetch["error"]

    assert benchmark.main(argv) == 0
    assert calls == ["called"]
