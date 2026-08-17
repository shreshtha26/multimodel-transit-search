import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gp_audit", ROOT / "scripts/audit_gp_stability.py")
gp_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gp_audit)


def test_recommendation_prefers_clean_optimized_restart():
    frame = pd.DataFrame([
        {"policy": "optimized_r0", "optimized": True, "fit_completed": True, "converged": False, "bound_hit": False, "log_marginal_likelihood": 10.0},
        {"policy": "optimized_r2", "optimized": True, "fit_completed": True, "converged": True, "bound_hit": False, "log_marginal_likelihood": 9.0},
        {"policy": "fixed_kernel_3d", "optimized": False, "fit_completed": True, "converged": True, "bound_hit": False, "log_marginal_likelihood": 8.0},
    ])
    rec = gp_audit.choose_recommended_policy(frame)
    assert rec["recommended_policy"] == "optimized_r2"
    assert rec["recommendation_type"] == "converged_optimized"


def test_fixed_kernel_is_only_fallback_when_no_optimized_fit_converges():
    frame = pd.DataFrame([
        {"policy": "optimized_r0", "optimized": True, "fit_completed": True, "converged": False, "bound_hit": False, "log_marginal_likelihood": 10.0},
        {"policy": "optimized_r2", "optimized": True, "fit_completed": True, "converged": False, "bound_hit": False, "log_marginal_likelihood": 11.0},
        {"policy": "fixed_kernel_3d", "optimized": False, "fit_completed": True, "converged": True, "bound_hit": False, "log_marginal_likelihood": 8.0},
    ])
    rec = gp_audit.choose_recommended_policy(frame)
    assert rec["recommended_policy"] == "fixed_kernel_3d"
    assert rec["recommendation_type"] == "explicit_fixed_kernel_fallback"
