from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gp_bound_audit", ROOT / "scripts/audit_gp_length_scale_bound.py"
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_parse_bounds():
    assert MOD.parse_bounds("1.0,0.75,0.5") == [1.0, 0.75, 0.5]


def test_policy_name_is_stable():
    assert MOD.policy_name(1.0) == "minls_1d"
    assert MOD.policy_name(0.75) == "minls_0p75d"
    assert MOD.policy_name(0.5) == "minls_0p5d"
