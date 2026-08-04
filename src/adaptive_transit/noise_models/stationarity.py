"""ADF/KPSS stationarity diagnostics for ARIMA model-selection context."""

import warnings
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass(frozen=True)
class UnitRootTestResult:
    """Recorded result from one unit-root or stationarity test."""

    test_name: str
    statistic: float | None
    pvalue: float | None
    critical_values: dict[str, float]
    null_hypothesis: str
    reject_null: bool | None
    alpha: float
    n_observations: int
    regression: str | None
    lag_selection: str | None
    lags_used: int | None
    status: str
    warning_messages: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass(frozen=True)
class StationarityAssessment:
    """Joint ADF/KPSS assessment for one exact ARIMA modelling representation."""

    original_adf: UnitRootTestResult
    original_kpss: UnitRootTestResult
    differenced_adf: UnitRootTestResult | None
    differenced_kpss: UnitRootTestResult | None
    original_series_conclusion: str
    differenced_series_conclusion: str | None
    recommended_d: int | None
    recommendation_strength: str
    diagnostics_agree: bool
    differencing_statistically_supported: bool
    differencing_requires_review: bool
    reason_codes: tuple[str, ...]
    modelling_mode: str
    observations_used: int
    missing_observations: int
    preprocessing_summary: dict[str, object]

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _optional_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _clean_series(values: np.ndarray) -> tuple[np.ndarray, int, int]:
    series = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(series)
    return series[finite], int(series.size), int((~finite).sum())


def _unavailable_test(
    *,
    test_name: str,
    alpha: float,
    n_observations: int,
    regression: str | None,
    lag_selection: str | None,
    null_hypothesis: str,
    reason: str,
) -> UnitRootTestResult:
    return UnitRootTestResult(
        test_name=test_name,
        statistic=None,
        pvalue=None,
        critical_values={},
        null_hypothesis=null_hypothesis,
        reject_null=None,
        alpha=float(alpha),
        n_observations=int(n_observations),
        regression=regression,
        lag_selection=lag_selection,
        lags_used=None,
        status=reason,
        warning_messages=(),
    )


def run_adf_test(
    series: np.ndarray,
    *,
    alpha: float = 0.05,
    regression: str = "c",
    autolag: str | None = "AIC",
    min_observations: int = 24,
) -> UnitRootTestResult:
    """Run ADF with explicit assumptions and warning capture."""

    clean = np.asarray(series, dtype=float).reshape(-1)
    if clean.size < min_observations:
        return _unavailable_test(
            test_name="ADF",
            alpha=alpha,
            n_observations=clean.size,
            regression=regression,
            lag_selection=autolag,
            null_hypothesis="unit root / nonstationary",
            reason="insufficient_observations",
        )

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adf_output = adfuller(clean, regression=regression, autolag=autolag)
            statistic, pvalue, lags_used, nobs, critical_values = adf_output[:5]
        warning_messages = tuple(str(item.message) for item in caught)
        return UnitRootTestResult(
            test_name="ADF",
            statistic=_optional_float(statistic),
            pvalue=_optional_float(pvalue),
            critical_values={str(key): float(value) for key, value in critical_values.items()},
            null_hypothesis="unit root / nonstationary",
            reject_null=bool(float(pvalue) < alpha),
            alpha=float(alpha),
            n_observations=int(nobs),
            regression=regression,
            lag_selection=autolag,
            lags_used=int(lags_used),
            status="ok_with_warnings" if warning_messages else "ok",
            warning_messages=warning_messages,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics should report test failures.
        result = _unavailable_test(
            test_name="ADF",
            alpha=alpha,
            n_observations=clean.size,
            regression=regression,
            lag_selection=autolag,
            null_hypothesis="unit root / nonstationary",
            reason="execution_failed",
        )
        return UnitRootTestResult(**{**result.to_dict(), "warning_messages": (f"{type(exc).__name__}: {exc}",)})


def run_kpss_test(
    series: np.ndarray,
    *,
    alpha: float = 0.05,
    regression: str = "c",
    nlags: str | int = "auto",
    min_observations: int = 24,
) -> UnitRootTestResult:
    """Run KPSS with explicit assumptions and warning capture."""

    clean = np.asarray(series, dtype=float).reshape(-1)
    if clean.size < min_observations:
        return _unavailable_test(
            test_name="KPSS",
            alpha=alpha,
            n_observations=clean.size,
            regression=regression,
            lag_selection=str(nlags),
            null_hypothesis="level stationary",
            reason="insufficient_observations",
        )

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            statistic, pvalue, lags_used, critical_values = kpss(clean, regression=regression, nlags=nlags)
        warning_messages = tuple(str(item.message) for item in caught)
        return UnitRootTestResult(
            test_name="KPSS",
            statistic=_optional_float(statistic),
            pvalue=_optional_float(pvalue),
            critical_values={str(key): float(value) for key, value in critical_values.items()},
            null_hypothesis="level stationary",
            reject_null=bool(float(pvalue) < alpha),
            alpha=float(alpha),
            n_observations=int(clean.size),
            regression=regression,
            lag_selection=str(nlags),
            lags_used=int(lags_used),
            status="ok_with_warnings" if warning_messages else "ok",
            warning_messages=warning_messages,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics should report test failures.
        result = _unavailable_test(
            test_name="KPSS",
            alpha=alpha,
            n_observations=clean.size,
            regression=regression,
            lag_selection=str(nlags),
            null_hypothesis="level stationary",
            reason="execution_failed",
        )
        return UnitRootTestResult(**{**result.to_dict(), "warning_messages": (f"{type(exc).__name__}: {exc}",)})


def interpret_stationarity_tests(
    adf_result: UnitRootTestResult,
    kpss_result: UnitRootTestResult,
    *,
    original_series: bool,
) -> tuple[str, int | None, str, bool, tuple[str, ...]]:
    """Interpret ADF and KPSS jointly without treating either test as proof."""

    reason_codes: list[str] = []
    if adf_result.reject_null is None or kpss_result.reject_null is None:
        if adf_result.status == "insufficient_observations" or kpss_result.status == "insufficient_observations":
            return "insufficient_observations", None, "unresolved", False, ("INSUFFICIENT_OBSERVATIONS",)
        return "test_unavailable", None, "unresolved", False, ("TEST_UNAVAILABLE",)

    reason_codes.append("ADF_REJECTS_UNIT_ROOT" if adf_result.reject_null else "ADF_DOES_NOT_REJECT_UNIT_ROOT")
    reason_codes.append("KPSS_REJECTS_STATIONARITY" if kpss_result.reject_null else "KPSS_DOES_NOT_REJECT_STATIONARITY")
    if adf_result.warning_messages or kpss_result.warning_messages:
        reason_codes.append("TEST_EXECUTION_WARNING")

    if adf_result.reject_null and not kpss_result.reject_null:
        conclusion = "stationary_supported"
        recommended_d = 0 if original_series else None
        strength = "joint_tests_support_d0" if original_series else "diagnostic_only"
        reason_codes.append("ORIGINAL_STATIONARITY_SUPPORTED" if original_series else "DIFFERENCED_STATIONARITY_SUPPORTED")
        return conclusion, recommended_d, strength, True, tuple(reason_codes)

    if not adf_result.reject_null and kpss_result.reject_null:
        conclusion = "nonstationary_supported"
        recommended_d = 1 if original_series else None
        strength = "joint_tests_support_d1" if original_series else "diagnostic_only"
        reason_codes.append("ORIGINAL_NONSTATIONARITY_SUPPORTED" if original_series else "DIFFERENCED_NONSTATIONARITY_SUPPORTED")
        return conclusion, recommended_d, strength, True, tuple(reason_codes)

    if adf_result.reject_null and kpss_result.reject_null:
        reason_codes.append("CONFLICTING_STATIONARITY_TESTS")
        return "conflicting_rejections", None, "unresolved", False, tuple(reason_codes)

    reason_codes.append("INCONCLUSIVE_STATIONARITY_TESTS")
    return "inconclusive_low_power", None, "unresolved", False, tuple(reason_codes)


def assess_stationarity(
    values: np.ndarray,
    *,
    modelling_mode: str,
    preprocessing_summary: dict[str, object],
    alpha: float = 0.05,
    adf_regression: str = "c",
    adf_autolag: str | None = "AIC",
    kpss_regression: str = "c",
    kpss_nlags: str | int = "auto",
    min_observations: int = 24,
    gaps_compressed: bool = False,
    interpolated: bool = False,
    contiguous_segment_used: bool = False,
    series_representation: str = "finite_observed_values",
) -> StationarityAssessment:
    """Assess stationarity for the exact series representation used by ARIMA."""

    clean, original_count, missing_count = _clean_series(values)
    metadata = dict(preprocessing_summary)
    metadata.update(
        {
            "stationarity_original_observations": int(original_count),
            "stationarity_removed_nonfinite": int(missing_count),
            "stationarity_observations_used": int(clean.size),
            "stationarity_gaps_compressed": bool(gaps_compressed),
            "stationarity_interpolated": bool(interpolated),
            "stationarity_contiguous_segment_used": bool(contiguous_segment_used),
            "stationarity_series_representation": series_representation,
        }
    )

    original_adf = run_adf_test(clean, alpha=alpha, regression=adf_regression, autolag=adf_autolag, min_observations=min_observations)
    original_kpss = run_kpss_test(clean, alpha=alpha, regression=kpss_regression, nlags=kpss_nlags, min_observations=min_observations)
    original_conclusion, recommended_d, strength, agree, reason_codes = interpret_stationarity_tests(original_adf, original_kpss, original_series=True)

    differenced_adf: UnitRootTestResult | None = None
    differenced_kpss: UnitRootTestResult | None = None
    differenced_conclusion: str | None = None
    if clean.size > min_observations:
        differenced = np.diff(clean)
        differenced_adf = run_adf_test(differenced, alpha=alpha, regression=adf_regression, autolag=adf_autolag, min_observations=min_observations)
        differenced_kpss = run_kpss_test(differenced, alpha=alpha, regression=kpss_regression, nlags=kpss_nlags, min_observations=min_observations)
        differenced_conclusion, _, _, _, differenced_reasons = interpret_stationarity_tests(differenced_adf, differenced_kpss, original_series=False)
        diagnostic_reasons = []
        for code in differenced_reasons:
            if code.startswith("DIFFERENCED_"):
                diagnostic_reasons.append(code)
            elif code == "TEST_EXECUTION_WARNING":
                diagnostic_reasons.append("DIFFERENCED_TEST_EXECUTION_WARNING")
        reason_codes = tuple(dict.fromkeys((*reason_codes, *diagnostic_reasons)))

    if gaps_compressed:
        reason_codes = tuple(dict.fromkeys((*reason_codes, "GAP_COMPRESSED_SERIES_DIAGNOSTIC")))
    if interpolated:
        reason_codes = tuple(dict.fromkeys((*reason_codes, "INTERPOLATED_SERIES_DIAGNOSTIC")))
    if contiguous_segment_used:
        reason_codes = tuple(dict.fromkeys((*reason_codes, "CONTIGUOUS_SEGMENT_DIAGNOSTIC")))

    differencing_statistically_supported = original_conclusion == "nonstationary_supported"
    return StationarityAssessment(
        original_adf=original_adf,
        original_kpss=original_kpss,
        differenced_adf=differenced_adf,
        differenced_kpss=differenced_kpss,
        original_series_conclusion=original_conclusion,
        differenced_series_conclusion=differenced_conclusion,
        recommended_d=recommended_d,
        recommendation_strength=strength,
        diagnostics_agree=agree,
        differencing_statistically_supported=differencing_statistically_supported,
        differencing_requires_review=not differencing_statistically_supported,
        reason_codes=reason_codes,
        modelling_mode=modelling_mode,
        observations_used=int(clean.size),
        missing_observations=int(missing_count),
        preprocessing_summary=metadata,
    )


def candidate_differencing_alignment(candidate_d: int, assessment: StationarityAssessment) -> str:
    """Describe whether a candidate's differencing agrees with stationarity evidence."""

    if assessment.recommended_d is None:
        return "unresolved"
    if candidate_d == assessment.recommended_d:
        return "aligned_with_stationarity_evidence"
    return "conflicts_with_stationarity_evidence"


def candidate_family_role(candidate_d: int, assessment: StationarityAssessment) -> str:
    """Label the candidate's differencing family without deciding final rank."""

    if assessment.recommended_d is None:
        return "unresolved family"
    if candidate_d == assessment.recommended_d:
        return "stationarity-supported primary family"
    if candidate_d > assessment.recommended_d:
        return "differenced challenger family"
    return "unresolved family"


def stationarity_candidate_fields(assessment: StationarityAssessment, candidate_d: int) -> dict[str, object]:
    """Return flat fields that can be merged onto one candidate row."""

    alignment = candidate_differencing_alignment(candidate_d, assessment)
    differenced_candidate = candidate_d > 0
    return {
        **stationarity_report_fields(assessment),
        "candidate_d": int(candidate_d),
        "candidate_family_role": candidate_family_role(candidate_d, assessment),
        "candidate_differencing_alignment": alignment,
        "differencing_justified": bool(differenced_candidate and assessment.differencing_statistically_supported),
        "selected_model_differencing_alignment": alignment,
        "selected_model_differencing_requires_review": bool(differenced_candidate and alignment != "aligned_with_stationarity_evidence"),
    }


def stationarity_report_fields(assessment: StationarityAssessment) -> dict[str, object]:
    """Return the compact flat stationarity fields used in CSV and JSON reports."""

    diff_adf = assessment.differenced_adf
    diff_kpss = assessment.differenced_kpss
    return {
        "stationarity_diagnostics_available": bool(assessment.original_adf.reject_null is not None and assessment.original_kpss.reject_null is not None),
        "stationarity_alpha": float(assessment.original_adf.alpha),
        "original_adf_statistic": assessment.original_adf.statistic,
        "original_adf_pvalue": assessment.original_adf.pvalue,
        "original_adf_rejects_unit_root": assessment.original_adf.reject_null,
        "original_kpss_statistic": assessment.original_kpss.statistic,
        "original_kpss_pvalue": assessment.original_kpss.pvalue,
        "original_kpss_rejects_stationarity": assessment.original_kpss.reject_null,
        "original_series_stationarity_conclusion": assessment.original_series_conclusion,
        "differenced_adf_statistic": diff_adf.statistic if diff_adf else None,
        "differenced_adf_pvalue": diff_adf.pvalue if diff_adf else None,
        "differenced_kpss_statistic": diff_kpss.statistic if diff_kpss else None,
        "differenced_kpss_pvalue": diff_kpss.pvalue if diff_kpss else None,
        "differenced_series_stationarity_conclusion": assessment.differenced_series_conclusion,
        "recommended_d": assessment.recommended_d,
        "recommendation_strength": assessment.recommendation_strength,
        "differencing_statistically_supported": assessment.differencing_statistically_supported,
        "stationarity_reason_codes": "|".join(assessment.reason_codes),
        "stationarity_observations_used": assessment.observations_used,
        "stationarity_missing_observations": assessment.missing_observations,
        "stationarity_modelling_mode": assessment.modelling_mode,
    }
