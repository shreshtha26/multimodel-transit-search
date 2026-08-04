import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from adaptive_transit.noise_models.correlograms import save_correlogram_plot
from adaptive_transit.noise_models.gap_mode_comparison import (
    REQUIRED_GAP_COMPARISON_COLUMNS,
    candidate_family_table,
    gap_mode_summary_row,
    same_candidate_family_across_modes,
    validate_gap_comparison_schema,
)
from adaptive_transit.noise_models.stationarity import StationarityAssessment, UnitRootTestResult, interpret_stationarity_tests
from adaptive_transit.preprocessing.gap_modes import build_gap_mode_representations, gap_runs, interpolate_eligible_gaps
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve, regularize_cadence_grid


def _raw_gap_frame() -> pd.DataFrame:
    cadences = np.array([100, 101, 102, 105, 106, 107, 108, 112, 113])
    return pd.DataFrame(
        {
            "time": 100.0 + cadences * 0.020433,
            "flux": 1000.0 + np.arange(cadences.size, dtype=float),
            "flux_error": np.full(cadences.size, 0.1),
            "quality": np.zeros(cadences.size, dtype=int),
            "cadenceno": cadences,
        }
    )


def _test_result(test_name: str, reject_null: bool) -> UnitRootTestResult:
    return UnitRootTestResult(
        test_name=test_name,
        statistic=0.0,
        pvalue=0.01 if reject_null else 0.40,
        critical_values={},
        null_hypothesis="unit root / nonstationary" if test_name == "ADF" else "level stationary",
        reject_null=reject_null,
        alpha=0.05,
        n_observations=120,
        regression="c",
        lag_selection="AIC" if test_name == "ADF" else "auto",
        lags_used=1,
        status="ok",
        warning_messages=(),
    )


def _assessment(adf_rejects: bool = True, kpss_rejects: bool = True) -> StationarityAssessment:
    adf_result = _test_result("ADF", adf_rejects)
    kpss_result = _test_result("KPSS", kpss_rejects)
    conclusion, recommended_d, strength, agree, reasons = interpret_stationarity_tests(adf_result, kpss_result, original_series=True)
    return StationarityAssessment(
        original_adf=adf_result,
        original_kpss=kpss_result,
        differenced_adf=None,
        differenced_kpss=None,
        original_series_conclusion=conclusion,
        differenced_series_conclusion=None,
        recommended_d=recommended_d,
        recommendation_strength=strength,
        diagnostics_agree=agree,
        differencing_statistically_supported=conclusion == "nonstationary_supported",
        differencing_requires_review=conclusion != "nonstationary_supported",
        reason_codes=reasons,
        modelling_mode="full_grid_missing",
        observations_used=120,
        missing_observations=0,
        preprocessing_summary={},
    )


def _selected_row(**extra: object) -> pd.Series:
    row: dict[str, object] = {
        "p": 1,
        "d": 1,
        "q": 0,
        "selection_status": "valid_but_residual_autocorrelation_remains",
        "fit_metrics_trustworthy": True,
        "failure_reason": "",
        "statistical_validity_failed": False,
        "differencing_requires_review": True,
        "selected_model_differencing_requires_review": True,
        "selected_model_differencing_alignment": "unresolved",
        "differencing_statistically_supported": False,
        "whitening_constraint_failed": True,
        "variance_constraint_failed": True,
        "transit_preservation_diagnostics_available": False,
        "residual_acf_lag_1": -0.2,
        "max_abs_residual_acf_1_24": 0.3,
        "mean_abs_residual_acf_1_24": 0.1,
        "max_abs_residual_acf_transit_lags": 0.25,
        "minimum_ljung_box_p": 0.001,
        "test_RMSE": 0.01,
        "test_MAE": 0.008,
        "AIC": -100.0,
        "BIC": -90.0,
    }
    row.update(extra)
    return pd.Series(row)


def test_regular_cadence_grid_preserves_missing_cadences() -> None:
    regular = regularize_cadence_grid(_raw_gap_frame())

    assert regular["cadenceno"].tolist() == list(range(100, 114))
    assert regular.loc[regular["cadenceno"].isin([103, 104, 109, 110, 111]), "row_present"].eq(False).all()


def test_longest_contiguous_segment_and_gap_metadata() -> None:
    regular, _ = preprocess_pdcsap_light_curve(_raw_gap_frame(), quality_policy="default")
    representations = build_gap_mode_representations(regular, max_interpolated_gap_cadences=2)
    longest = representations["longest_contiguous"]
    full = representations["full_grid_missing"]

    assert longest.metadata["segment_start_cadenceno"] == 105
    assert longest.metadata["segment_end_cadenceno"] == 108
    assert longest.metadata["observations"] == 4
    assert longest.metadata["cadence_consistent"]
    assert np.isclose(longest.metadata["fraction_of_quarter_retained"], 4 / 14)
    assert full.metadata["total_grid_length"] == 14
    assert full.metadata["missing_cadences"] == 5
    assert full.metadata["gap_lengths"] == (2, 3)


def test_interpolation_fills_only_configured_interior_gaps() -> None:
    values = np.array([np.nan, 1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, 8.0, np.nan])

    filled, interpolated, metadata = interpolate_eligible_gaps(values, max_gap_cadences=2)

    np.testing.assert_allclose(filled[2:4], [2.0, 3.0])
    assert interpolated.tolist() == [False, False, True, True, False, False, False, False, False, False]
    assert np.isnan(filled[[0, 5, 6, 7, 9]]).all()
    assert metadata["interpolated_values"] == 2
    assert metadata["unfilled_long_gaps"] == 1
    assert metadata["unfilled_edge_gaps"] == 2
    assert metadata["edge_extrapolation_policy"] == "none"


def test_gap_runs_report_lengths() -> None:
    runs = gap_runs(np.array([False, True, True, False, True]))

    assert runs == [{"start_index": 1, "end_index": 2, "length": 2}, {"start_index": 4, "end_index": 4, "length": 1}]


def test_same_candidate_family_across_modes() -> None:
    table = candidate_family_table(((1, 0, 0), (1, 1, 0)), ("longest_contiguous", "full_grid_missing", "interpolated_full_grid"))

    assert same_candidate_family_across_modes(table)
    assert table.groupby("gap_mode").size().eq(2).all()


def test_correlogram_plots_generate_for_contiguous_data(tmp_path) -> None:
    values = np.sin(np.linspace(0.0, 4.0 * np.pi, 80))
    path = tmp_path / "acf.png"

    record = save_correlogram_plot(
        values,
        path,
        gap_mode="longest_contiguous",
        series_kind="modelling_series",
        plot_kind="acf",
        max_lag=12,
        cadence_days=0.020433,
        missing_observations_compressed=False,
        interpolation_used=False,
        missing_strategy="none",
        transit_lag_range=(3, 12),
        annotation="test",
    )

    assert record.generated
    assert path.exists()


def test_pacf_omitted_for_missing_valued_full_grid(tmp_path) -> None:
    values = np.arange(40, dtype=float)
    values[10:12] = np.nan

    record = save_correlogram_plot(
        values,
        tmp_path / "pacf.png",
        gap_mode="full_grid_missing",
        series_kind="modelling_series",
        plot_kind="pacf",
        max_lag=12,
        cadence_days=0.020433,
        missing_observations_compressed=False,
        interpolation_used=False,
        missing_strategy="none",
        transit_lag_range=(3, 12),
        annotation="test",
    )

    assert not record.generated
    assert "missing-valued" in record.reason


def test_gap_mode_comparison_schema() -> None:
    row = gap_mode_summary_row(
        quality_policy="default",
        selected=_selected_row(),
        assessment=_assessment(),
        metadata={
            "gap_mode": "full_grid_missing",
            "observations": 100,
            "missing_fraction": 0.10,
            "interpolated_fraction": 0.0,
            "gap_count": 2,
            "max_gap_length": 5,
            "total_grid_length": 110,
            "observed_cadences": 100,
            "missing_cadences": 10,
            "fraction_of_quarter_retained": 1.0,
            "ordinary_lags_meaningful": False,
        },
    )
    table = pd.DataFrame([row])

    validate_gap_comparison_schema(table)
    assert set(REQUIRED_GAP_COMPARISON_COLUMNS).issubset(table.columns)
    assert not table.loc[0, "scientifically_acceptable"]
    assert "residual_autocorrelation_remains" in table.loc[0, "failure_reasons"]


def test_compressing_gaps_can_change_apparent_lag_diagnostics() -> None:
    rng = np.random.default_rng(123)
    values = np.empty(500)
    noise = rng.normal(scale=1.0, size=500)
    values[0] = noise[0]
    for index in range(1, values.size):
        values[index] = 0.8 * values[index - 1] + noise[index]
    gapped = values.copy()
    gapped[1::2] = np.nan

    original_lag1 = acf(values, nlags=1, fft=True)[1]
    compressed_lag1 = acf(gapped[np.isfinite(gapped)], nlags=1, fft=True)[1]

    assert abs(original_lag1 - compressed_lag1) > 0.15
