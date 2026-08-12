from pathlib import Path
import importlib.util
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
spec = importlib.util.spec_from_file_location("run_multistar_challenger_benchmark", PROJECT_ROOT / "scripts/run_multistar_challenger_benchmark.py")
challenger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(challenger)


def args_for(path, *, selection_group=challenger.CLEAN_SELECTION_GROUP, require_catalog_clean=True):
    args = challenger.default_settings("smoke")
    args.manifest_path = Path(path)
    args.target_limit = 1
    args.strict_target_count = True
    args.target_ids = None
    args.selection_group = selection_group
    args.require_catalog_clean = require_catalog_clean
    args.stratified_pilot = False
    return args


def write_manifest(path, **overrides):
    row = {
        "target_id": "1234567",
        "quarter": 5,
        "selection_group": challenger.CLEAN_SELECTION_GROUP,
        "sample_stratum": "quiet_low_scatter",
        "koi_flag": False,
        "tce_flag": False,
        "confirmed_planet_flag": False,
        "eb_flag": False,
    }
    row.update(overrides)
    pd.DataFrame([row]).to_csv(path, index=False)


def test_clean_manifest_passes(tmp_path):
    path = tmp_path / "clean.csv"
    write_manifest(path)
    manifest = challenger.load_manifest(args_for(path))
    assert len(manifest) == 1
    assert manifest.iloc[0]["selection_group"] == challenger.CLEAN_SELECTION_GROUP


def test_catalog_contamination_fails_closed(tmp_path):
    path = tmp_path / "contaminated.csv"
    write_manifest(path, tce_flag=True)
    with pytest.raises(ValueError, match="cataloged KOI/TCE/confirmed-planet/EB"):
        challenger.load_manifest(args_for(path))


def test_missing_catalog_flags_fails_closed(tmp_path):
    path = tmp_path / "missing_flags.csv"
    pd.DataFrame([{"target_id": "1234567", "quarter": 5, "selection_group": challenger.CLEAN_SELECTION_GROUP}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing catalog contamination flags"):
        challenger.load_manifest(args_for(path))


def test_legacy_cohort_requires_explicit_opt_out(tmp_path):
    path = tmp_path / "legacy.csv"
    write_manifest(path, selection_group="confirmed_planet_host", confirmed_planet_flag=True)
    with pytest.raises(ValueError):
        challenger.load_manifest(args_for(path, selection_group="confirmed_planet_host", require_catalog_clean=True))
    manifest = challenger.load_manifest(args_for(path, selection_group="confirmed_planet_host", require_catalog_clean=False))
    assert manifest.iloc[0]["selection_group"] == "confirmed_planet_host"
