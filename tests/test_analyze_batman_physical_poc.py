from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_batman_physical_poc.py"
SPEC = importlib.util.spec_from_file_location("batman_poc_qc", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _frames():
    retention = []
    detectors = []
    base = []
    for target, stratum in [("1", "quiet"), ("2", "memory")]:
        for branch, converged in [
            ("raw", True),
            ("arima", target == "1"),
            ("kalman", True),
            ("gp", target == "1"),
        ]:
            base.append(
                {
                    "target_id": target,
                    "quarter": 5,
                    "sample_stratum": stratum,
                    "branch": branch,
                    "converged": converged,
                    "error": "",
                }
            )
            for case_index, period in enumerate([2.0, 5.0]):
                corr = {"raw": 1.0, "arima": 0.3, "kalman": 0.6, "gp": 0.95}[branch]
                snr = {"raw": 4.0, "arima": 5.0, "kalman": 7.0, "gp": 9.0}[branch]
                acf = {"raw": 0.8, "arima": 0.1, "kalman": 0.03, "gp": 0.2}[branch]
                retention.append(
                    {
                        "target_id": target,
                        "quarter": 5,
                        "sample_stratum": stratum,
                        "case_index": case_index,
                        "branch": branch,
                        "success": True,
                        "injected_period_days": period,
                        "requested_duration_hours": [2.0, 4.0][case_index],
                        "requested_depth": [0.0002, 0.0005][case_index],
                        "phase_fraction": 0.45,
                        "template_amplitude_ratio": corr,
                        "peak_depth_ratio": 0.9,
                        "template_energy_ratio": 0.8,
                        "template_correlation": corr,
                        "template_rmse_ppm": 20.0,
                        "oracle_signal_snr": snr,
                        "background_scale_ppm": 100.0,
                        "background_acf1": acf,
                    }
                )
                for detector in ["bls", "trapezoid", "tls"]:
                    exact = (
                        detector == "tls" and branch in {"kalman", "gp"}
                    ) or (
                        detector == "bls" and branch == "raw" and case_index == 0
                    )
                    detectors.append(
                        {
                            "target_id": target,
                            "quarter": 5,
                            "sample_stratum": stratum,
                            "case_index": case_index,
                            "branch": branch,
                            "detector": detector,
                            "success": True,
                            "exact_period_recovered": exact,
                            "harmonic_period_recovered": exact,
                            "period_exact_fractional_error": 0.0 if exact else 0.3,
                            "runtime_seconds": 0.1,
                        }
                    )
    return pd.DataFrame(retention), pd.DataFrame(detectors), pd.DataFrame(base)


def test_clean_fit_winners_exclude_optimizer_flagged_branch(tmp_path):
    retention, detectors, base = _frames()
    retention.to_csv(tmp_path / "physical_retention.csv", index=False)
    detectors.to_csv(tmp_path / "detector_results.csv", index=False)
    base.to_csv(tmp_path / "base_models.csv", index=False)

    assert MODULE.main(["--input-dir", str(tmp_path)]) == 0

    winners = pd.read_csv(tmp_path / "qc_analysis" / "per_star_winners.csv", dtype={"target_id": str})
    star1 = winners[winners["target_id"].eq("1")].iloc[0]
    star2 = winners[winners["target_id"].eq("2")].iloc[0]

    # Star 1 has clean GP and Kalman fits, so the TLS exact-recovery tie is retained.
    assert star1["tls_exact_winner_clean"] == "kalman|gp"
    assert star1["oracle_snr_winner_nonraw_clean"] == "gp"

    # Star 2 has optimizer-flagged GP/ARIMA fits; clean winners must exclude them.
    assert star2["tls_exact_winner_clean"] == "kalman"
    assert star2["oracle_snr_winner_nonraw_clean"] == "kalman"

    convergence = pd.read_csv(tmp_path / "qc_analysis" / "convergence_audit.csv")
    flagged = convergence[convergence["fit_status"].eq("optimizer_flagged")]
    assert set(flagged["branch"]) == {"arima", "gp"}


def test_case_clean_outputs_keep_partial_clean_samples(tmp_path):
    retention, detectors, base = _frames()
    retention.to_csv(tmp_path / "physical_retention.csv", index=False)
    detectors.to_csv(tmp_path / "detector_results.csv", index=False)
    base.to_csv(tmp_path / "base_models.csv", index=False)

    MODULE.main(["--input-dir", str(tmp_path)])
    clean = pd.read_csv(tmp_path / "qc_analysis" / "per_injection_branch_retention_clean.csv")

    # GP remains analyzable from its one clean star rather than being discarded
    # merely because another star's GP optimizer was flagged.
    gp_case0 = clean[(clean["branch"].eq("gp")) & (clean["case_index"].eq(0))].iloc[0]
    assert int(gp_case0["n_successful_stars"]) == 1


def test_gp_lower_bound_is_usable_boundary_limited():
    base = pd.DataFrame(
        [
            {
                "target_id": "42",
                "quarter": 5,
                "sample_stratum": "quiet",
                "branch": "gp",
                "converged": False,
                "error": "",
                "length_scale_days": 1.0,
            }
        ]
    )

    status = MODULE.build_model_status(base).iloc[0]
    assert status["fit_status"] == "boundary_limited"
    assert bool(status["fit_clean"]) is True
    assert bool(status["fit_interior"]) is False
    assert bool(status["boundary_limited_bool"]) is True
