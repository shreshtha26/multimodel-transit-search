import numpy as np
import pandas as pd

from adaptive_transit.detection.tps_like_hardening import (
    EventConsistencyConfig,
    candidate_event_metrics,
    harden_tps_like_result,
    standardize_candidate_table,
    _consistency_veto_diagnostics,
    _event_depth_chi2,
    _odd_even_depth_consistency,
    _huber_robust_event_snr,
)


def _synthetic_flux():
    rng = np.random.default_rng(123)
    time = np.arange(0.0, 30.0, 0.02)
    flux = 1.0 + rng.normal(0.0, 1.5e-4, size=len(time))

    # Repeated transit-like events at P=5 d.
    duration = 0.20
    for center in np.arange(1.0, 30.0, 5.0):
        flux[np.abs(time - center) <= duration / 2] -= 0.0020

    # One much deeper isolated dip that can inflate a wrong raw-MES candidate.
    flux[np.abs(time - 2.5) <= duration / 2] -= 0.0060
    return time, flux


def test_event_metrics_reward_repeated_events():
    time, flux = _synthetic_flux()
    good = candidate_event_metrics(
        time,
        flux,
        period_days=5.0,
        epoch_days=1.0,
        duration_hours=4.8,
    )
    bad = candidate_event_metrics(
        time,
        flux,
        period_days=7.0,
        epoch_days=2.5,
        duration_hours=4.8,
    )

    assert good["valid_event_count"] >= 5
    assert good["positive_event_fraction"] > bad["positive_event_fraction"]
    assert good["single_event_fraction"] < bad["single_event_fraction"]
    assert (
        good["leave_one_out_ratio_normalized"]
        > bad["leave_one_out_ratio_normalized"]
    )


def test_hardening_can_demote_single_event_dominated_raw_winner():
    time, flux = _synthetic_flux()
    result = {
        "summary": {
            "period_days": 7.0,
            "epoch_days": 2.5,
            "duration_hours": 4.8,
            "mes": 15.0,
            "max_ses": 14.0,
        },
        "periodogram": pd.DataFrame(
            [
                {
                    "period_days": 7.0,
                    "epoch_days": 2.5,
                    "duration_hours": 4.8,
                    "mes": 15.0,
                    "max_ses": 14.0,
                },
                {
                    "period_days": 5.0,
                    "epoch_days": 1.0,
                    "duration_hours": 4.8,
                    "mes": 12.0,
                    "max_ses": 5.0,
                },
            ]
        ),
    }

    hardened = harden_tps_like_result(result, time, flux)
    summary = hardened["summary"]

    assert np.isclose(summary["raw_top1_period_days"], 7.0)
    assert np.isclose(summary["period_days"], 5.0)
    assert summary["ranking_changed"]
    assert summary["raw_rank_of_selected_candidate"] == 2
    assert summary["event_consistency_score"] > 0
    ranking = hardened["ranking_table"]
    raw = ranking.loc[np.isclose(ranking["period_days"], 7.0)].iloc[0]
    assert not raw["transit_consistency_veto_pass"]
    assert "fails_" in raw["veto_reason"]


def test_periodogram_schema_accepts_epoch_phase_cadence():
    time = np.arange(100.0, 120.0, 0.02)
    periodogram = pd.DataFrame(
        [
            {
                "period_cadences": 250,
                "period_days": 5.0,
                "duration_cadences": 10,
                "duration_hours": 4.8,
                "epoch_phase_cadence": 50,
                "mes": 20.0,
            },
            {
                "period_cadences": 500,
                "period_days": 10.0,
                "duration_cadences": 20,
                "duration_hours": 9.6,
                "epoch_phase_cadence": 75,
                "mes": 18.0,
            },
        ]
    )

    candidates = standardize_candidate_table(
        periodogram,
        raw_summary={},
        time=time,
        top_n=10,
    )

    assert len(candidates) == 2
    assert np.isclose(candidates.iloc[0]["epoch_days"], 101.0, atol=1e-8)
    assert candidates.iloc[0]["source_index"] == 0
    assert candidates.iloc[0]["raw_rank"] == 1


def test_candidate_bank_collapses_adjacent_peak_samples_but_keeps_harmonic():
    time = np.arange(100.0, 130.0, 0.02)
    periodogram = pd.DataFrame(
        [
            {
                "period_days": 10.00,
                "duration_hours": 9.6,
                "epoch_phase_cadence": 100,
                "mes": 50.0,
            },
            {
                "period_days": 10.05,
                "duration_hours": 9.6,
                "epoch_phase_cadence": 103,
                "mes": 49.0,
            },
            {
                "period_days": 10.08,
                "duration_hours": 7.7,
                "epoch_phase_cadence": 106,
                "mes": 48.0,
            },
            {
                "period_days": 5.00,
                "duration_hours": 4.8,
                "epoch_phase_cadence": 80,
                "mes": 47.0,
            },
        ]
    )

    candidates = standardize_candidate_table(
        periodogram,
        raw_summary={},
        time=time,
        top_n=10,
        duplicate_period_fraction=0.01,
        duplicate_phase_duration_factor=0.75,
    )

    periods = candidates["period_days"].to_numpy()
    assert len(candidates) == 2
    assert np.any(np.isclose(periods, 10.0))
    assert np.any(np.isclose(periods, 5.0))
    assert set(candidates["source_index"].astype(int)) == {0, 3}


def test_event_depth_chi2_prefers_consistent_depths():
    errors = np.full(6, 0.2)
    consistent = _event_depth_chi2(
        np.array([2.0, 2.1, 1.9, 2.0, 2.05, 1.95]),
        errors,
    )
    inconsistent = _event_depth_chi2(
        np.array([2.0, 4.0, 0.2, 3.8, 0.3, 2.1]),
        errors,
    )
    assert consistent["event_depth_chi2_pvalue"] > 0.05
    assert inconsistent["event_depth_chi2_pvalue"] < 0.01
    assert consistent["event_depth_chi2_dof"] == 5


def test_consistent_repeated_transit_depths_pass_depth_chi2_veto():
    metrics = {
        "valid_event_count": 6,
        "event_observability_fraction": 1.0,
        "positive_event_fraction": 1.0,
        "single_event_fraction": 1.0 / 6.0,
        "leave_one_out_ratio_normalized": 1.0,
        "robust_event_snr": 10.0,
        "odd_even_tested": False,
        "odd_even_depth_pvalue": 1.0,
        **_event_depth_chi2(
            np.array([2.0, 2.1, 1.9, 2.0, 2.05, 1.95]),
            np.full(6, 0.2),
        ),
    }
    veto = _consistency_veto_diagnostics(
        metrics,
        config=EventConsistencyConfig(),
    )
    assert not veto["fails_depth_chi2"]
    assert veto["transit_consistency_veto_pass"]


def test_odd_even_depth_test_flags_alternating_depths():
    index = np.arange(8)
    errors = np.full(8, 0.1)
    depths = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    result = _odd_even_depth_consistency(index, depths, errors)
    assert result["odd_even_tested"]
    assert result["odd_even_depth_pvalue"] < 0.01
    assert result["odd_even_depth_difference_fraction"] > 0.5


def test_strong_alternating_depths_fail_odd_even_veto():
    time = np.arange(0.0, 45.0, 0.02)
    flux = np.ones_like(time)
    duration = 0.20
    for index, center in enumerate(np.arange(2.0, 45.0, 5.0)):
        depth = 0.001 if index % 2 == 0 else 0.004
        flux[np.abs(time - center) <= duration / 2] -= depth
    rng = np.random.default_rng(321)
    flux += rng.normal(0.0, 2.0e-5, size=len(flux))

    metrics = candidate_event_metrics(
        time,
        flux,
        period_days=5.0,
        epoch_days=2.0,
        duration_hours=4.8,
    )

    assert metrics["odd_even_tested"]
    assert metrics["fails_odd_even"]
    assert not metrics["diagnostic_consistency_veto_pass"]
    assert "fails_odd_even" in metrics["diagnostic_veto_reason"]


def test_too_few_odd_even_groups_do_not_create_false_veto():
    odd_even = _odd_even_depth_consistency(
        np.arange(4),
        np.array([1.0, 3.0, 1.0, 3.0]),
        np.full(4, 0.1),
        min_group_events=3,
    )
    metrics = {
        "valid_event_count": 4,
        "event_observability_fraction": 1.0,
        "positive_event_fraction": 1.0,
        "single_event_fraction": 0.25,
        "leave_one_out_ratio_normalized": 1.0,
        "event_depth_chi2_pvalue": 0.5,
        "event_depth_reduced_chi2": 1.0,
        "robust_event_snr": 6.0,
        **odd_even,
    }
    veto = _consistency_veto_diagnostics(
        metrics,
        config=EventConsistencyConfig(),
    )
    assert not odd_even["odd_even_tested"]
    assert not veto["fails_odd_even"]
    assert veto["transit_consistency_veto_pass"]


def test_huber_robust_event_snr_downweights_one_extreme_event():
    z = np.array([2.0, 2.2, 1.8, 2.1, 20.0])
    naive = np.sum(z) / np.sqrt(len(z))
    robust = _huber_robust_event_snr(z)
    assert robust["robust_event_snr"] < naive
    assert robust["robust_event_weight_min"] < 1.0


def test_genuine_repeated_transit_not_demoted_by_small_depth_variations():
    rng = np.random.default_rng(456)
    time = np.arange(0.0, 50.0, 0.02)
    flux = 1.0 + rng.normal(0.0, 1.0e-4, size=len(time))
    duration = 0.20
    depth_factors = np.array(
        [1.00, 0.92, 1.10, 0.97, 1.06, 0.95, 1.03, 0.99, 1.04, 0.96]
    )
    for depth_factor, center in zip(depth_factors, np.arange(1.0, 50.0, 5.0)):
        flux[np.abs(time - center) <= duration / 2] -= 0.0020 * depth_factor

    result = {
        "summary": {
            "period_days": 5.0,
            "epoch_days": 1.0,
            "duration_hours": 4.8,
            "mes": 30.0,
        },
        "periodogram": pd.DataFrame(
            [
                {
                    "period_days": 5.0,
                    "epoch_days": 1.0,
                    "duration_hours": 4.8,
                    "mes": 30.0,
                },
                {
                    "period_days": 10.0,
                    "epoch_days": 1.0,
                    "duration_hours": 4.8,
                    "mes": 25.0,
                },
            ]
        ),
    }

    hardened = harden_tps_like_result(result, time, flux)

    assert np.isclose(hardened["summary"]["period_days"], 5.0)
    assert hardened["summary"]["transit_consistency_veto_pass"]
    assert hardened["summary"]["selection_status"] == "raw_top1_mes_preserved"
    assert hardened["summary"]["raw_rank_of_selected_candidate"] == 1


def test_nan_chi2_pvalue_fails_closed():
    metrics = {
        "valid_event_count": 4,
        "event_observability_fraction": 1.0,
        "positive_event_fraction": 1.0,
        "single_event_fraction": 0.25,
        "leave_one_out_ratio_normalized": 1.0,
        "event_depth_chi2_pvalue": np.nan,
        "event_depth_reduced_chi2": np.nan,
        "odd_even_tested": False,
        "odd_even_depth_pvalue": 1.0,
        "robust_event_snr": 6.0,
    }
    veto = _consistency_veto_diagnostics(
        metrics,
        config=EventConsistencyConfig(),
    )
    assert veto["fails_depth_chi2"]
    assert not veto["transit_consistency_veto_pass"]


def test_no_veto_survivor_falls_back_deterministically_with_zero_score():
    time = np.arange(0.0, 20.0, 0.02)
    flux = np.ones_like(time)
    result = {
        "summary": {
            "period_days": 3.0,
            "epoch_days": 1.0,
            "duration_hours": 2.0,
            "mes": 20.0,
        },
        "periodogram": pd.DataFrame(
            [
                {
                    "period_days": 3.0,
                    "epoch_days": 1.0,
                    "duration_hours": 2.0,
                    "mes": 20.0,
                },
                {
                    "period_days": 4.0,
                    "epoch_days": 1.5,
                    "duration_hours": 2.0,
                    "mes": 18.0,
                },
            ]
        ),
    }

    hardened = harden_tps_like_result(result, time, flux)
    summary = hardened["summary"]

    assert not summary["any_candidate_survives_veto"]
    assert summary["selection_status"] == "no_veto_survivor_raw_mes_fallback"
    assert summary["robust_veto_score"] == 0.0
    assert summary["raw_rank_of_selected_candidate"] == 1
