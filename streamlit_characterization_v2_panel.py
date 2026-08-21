from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable
from html import escape
import re
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import lombscargle


# =============================================================================
# Stellar statistical characterisation — Streamlit scientific inspection panel
# =============================================================================
#
# Design goal:
#   preserve the "pick a star -> inspect the light curve -> inspect the statistics"
#   structure of the earlier dashboard, but replace the old three diagnostic plots
#   with the complete seven-domain characterisation workflow.
#
# The seven domains contain eleven canonical variables:
#
#   1. scatter amplitude
#   2. distribution shape
#   3. autocorrelation / memory
#   4. stationarity
#   5. spectral structure
#   6. periodicity / coherence
#   7. variability stability
#
# The diagnostic figures below are visual inspection aids derived from the same
# selected light curve. The scalar values shown beside them come from the saved
# validated characterisation outputs whenever available.
# =============================================================================


PLOT_CONFIG = {
    "displayModeBar": True,
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
}

CORE_TREATMENTS = ("raw", "arima", "kalman", "gp")
CORE_DETECTORS = ("bls", "tcf", "tps_like")
CORE_PIPELINES = tuple(f"{treatment}_{detector}" for treatment in CORE_TREATMENTS for detector in CORE_DETECTORS)
PIPELINE_LABELS = {
    "raw_bls": "Raw + BLS",
    "raw_tcf": "Raw + TCF",
    "raw_tps_like": "Raw + TPS-like",
    "arima_bls": "ARIMA + BLS",
    "arima_tcf": "ARIMA + TCF",
    "arima_tps_like": "ARIMA + TPS-like",
    "kalman_bls": "Kalman + BLS",
    "kalman_tcf": "Kalman + TCF",
    "kalman_tps_like": "Kalman + TPS-like",
    "gp_bls": "GP + BLS",
    "gp_tcf": "GP + TCF",
    "gp_tps_like": "GP + TPS-like",
}

CANONICAL_SCHEMA = [
    {
        "domain": "scatter_amplitude",
        "domain_label": "Scatter amplitude",
        "feature": "robust_scatter",
        "label": "Robust scatter",
        "source": "flux_robust_scale",
        "kind": "continuous",
        "unit": "relative flux",
        "question": "How large is the background variability?",
        "calculation": "Robust scale of the finite normalized-flux distribution (MAD-based scale used by the characterisation pipeline).",
        "rationale": "Stable amplitude descriptor that is less dominated by isolated excursions than ordinary standard deviation.",
    },
    {
        "domain": "distribution_shape",
        "domain_label": "Distribution shape",
        "feature": "skewness",
        "label": "Skewness",
        "source": "flux_skewness",
        "kind": "continuous",
        "unit": "dimensionless",
        "question": "Is the flux distribution asymmetric?",
        "calculation": "Sample skewness of the finite normalized-flux values.",
        "rationale": "Captures asymmetric variability that is invisible to a variance-only description.",
    },
    {
        "domain": "distribution_shape",
        "domain_label": "Distribution shape",
        "feature": "outlier_fraction",
        "label": "Outlier fraction",
        "source": "flux_outlier_fraction",
        "kind": "continuous",
        "unit": "fraction",
        "question": "How much of the light curve sits in robust tails?",
        "calculation": "Fraction of finite observations beyond the robust outlier threshold used by the characterisation pipeline.",
        "rationale": "Separates impulsive/tail-heavy behaviour from broadly distributed scatter.",
    },
    {
        "domain": "autocorrelation_memory",
        "domain_label": "Autocorrelation / memory",
        "feature": "acf_lag1",
        "label": "ACF lag 1",
        "source": "v2_acf_lag_1",
        "kind": "continuous",
        "unit": "correlation",
        "question": "How strongly does one cadence predict the next?",
        "calculation": "Normalized autocorrelation at a one-cadence lag using finite cadence pairs.",
        "rationale": "Direct measure of short-lag temporal dependence.",
    },
    {
        "domain": "autocorrelation_memory",
        "domain_label": "Autocorrelation / memory",
        "feature": "acf_timescale_days",
        "label": "ACF e-fold timescale",
        "source": "v2_acf_decay_e_days",
        "kind": "continuous",
        "unit": "days",
        "question": "For how long does the correlation persist?",
        "calculation": "First positive ACF decay time at which the correlation reaches the e-folding level, converted to days.",
        "rationale": "Adds a physical persistence timescale rather than relying on ACF(1) alone.",
    },
    {
        "domain": "stationarity",
        "domain_label": "Stationarity",
        "feature": "stationarity_state",
        "label": "Stationarity state",
        "source": "original_series_stationarity_conclusion",
        "kind": "categorical",
        "unit": "category",
        "question": "Are the statistical properties stable through the quarter?",
        "calculation": "Joint interpretation of ADF and KPSS tests on the original finite modelling series.",
        "rationale": "Avoids treating a single stationarity test as definitive and informs whether differencing/background assumptions are defensible.",
    },
    {
        "domain": "spectral_structure",
        "domain_label": "Spectral structure",
        "feature": "spectral_concentration",
        "label": "Spectral concentration",
        "source": "v2_spectral_concentration",
        "kind": "continuous",
        "unit": "fraction",
        "question": "Is variability power concentrated in a narrow frequency structure?",
        "calculation": "Dominant Lomb–Scargle peak power divided by total sampled spectral power.",
        "rationale": "Distinguishes concentrated/coherent spectral structure from broadband variability.",
    },
    {
        "domain": "spectral_structure",
        "domain_label": "Spectral structure",
        "feature": "harmonic_power_ratio",
        "label": "Harmonic power ratio",
        "source": "v2_spectral_harmonic_power_ratio",
        "kind": "continuous",
        "unit": "ratio",
        "question": "Does the dominant variability have harmonic support?",
        "calculation": "Strongest sampled power near f/2 or 2f divided by the dominant spectral-peak power.",
        "rationale": "Useful for non-sinusoidal periodic structure and rotation/star-spot review without treating it as an astrophysical classification.",
    },
    {
        "domain": "periodicity_coherence",
        "domain_label": "Periodicity / coherence",
        "feature": "dominant_period_days",
        "label": "Dominant period",
        "source": "v2_ls_dominant_period_days",
        "kind": "continuous",
        "unit": "days",
        "question": "What is the dominant coherent timescale?",
        "calculation": "Period corresponding to the strongest sampled Lomb–Scargle peak in the v2 period search.",
        "rationale": "Provides the characteristic periodic timescale used for coherence and review screens.",
    },
    {
        "domain": "periodicity_coherence",
        "domain_label": "Periodicity / coherence",
        "feature": "period_agreement_error",
        "label": "LS–ACF period agreement error",
        "source": "v2_ls_acf_period_relative_error",
        "kind": "continuous",
        "unit": "relative error",
        "question": "Do independent periodicity diagnostics agree?",
        "calculation": "Harmonic-aware relative disagreement between the Lomb–Scargle dominant period and the ACF period candidate.",
        "rationale": "A periodic peak is more convincing when independent time- and frequency-domain diagnostics support the same timescale.",
    },
    {
        "domain": "variance_evolution",
        "domain_label": "Variability stability",
        "feature": "segment_scale_variability",
        "label": "Segment-scale variability",
        "source": "v2_segment_scale_relative_mad",
        "kind": "continuous",
        "unit": "dimensionless",
        "question": "Does variability amplitude evolve across the quarter?",
        "calculation": "Robust dispersion of segment-by-segment scale estimates relative to the overall segment-scale level.",
        "rationale": "Separates a stable correlated process from one whose amplitude changes substantially with time.",
    },
]

CANONICAL_ORDER = [x["feature"] for x in CANONICAL_SCHEMA]
CONTINUOUS_FEATURES = [x["feature"] for x in CANONICAL_SCHEMA if x["kind"] == "continuous"]
FEATURE_META = {x["feature"]: x for x in CANONICAL_SCHEMA}

POPULATION_COLUMNS = [
    ("v2_amplitude_population_label", "Amplitude"),
    ("v2_memory_population_label", "Memory"),
]

BEHAVIOUR_ORDER = [
    "Quiet / low variability",
    "Low-scatter structured",
    "Short-memory stochastic",
    "Long-memory / correlated",
    "Coherent periodic",
    "Quasi-periodic / structured",
    "Evolving variability",
    "High-amplitude / high-scatter",
    "Tail-heavy / asymmetric",
    "Mixed / complex",
]

# Explicit v2 flags remain valuable evidence, but the dropdown is no longer
# populated only from these flags.  Stars without an explicit flag are assigned
# from the continuous seven-domain statistics relative to the 50-star sample.
EXPLICIT_BEHAVIOUR_FLAGS = [
    ("v2_coherent_periodic_candidate", "Coherent periodic"),
    ("v2_quasi_periodic_candidate", "Quasi-periodic / structured"),
    ("v2_evolving_variability_candidate", "Evolving variability"),
    ("v2_correlated_stochastic_candidate", "Long-memory / correlated"),
    ("v2_low_scatter_structured_candidate", "Low-scatter structured"),
    ("v2_quiet_candidate", "Quiet / low variability"),
]

REVIEW_FLAG_COLUMNS = [
    ("v2_rotation_spot_review_flag", "Rotation / star-spot review"),
    ("v2_pulsation_review_flag", "Pulsation review"),
]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def _normalize_target_id(value) -> str:
    text = str(value).upper().replace("KIC", "").strip()
    try:
        return str(int(float(text)))
    except Exception:
        return text


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"target_id": str})
    except Exception:
        return pd.DataFrame()


def _candidate_dirs(repo_root: Path, run_dir: Path | None) -> list[Path]:
    dirs: list[Path] = []
    if run_dir is not None:
        dirs.extend([run_dir / "metrics", run_dir / "characterization_analysis", run_dir])
    dirs.extend(
        [
            repo_root / "outputs" / "experiments" / "characterization_validation50" / "metrics",
            repo_root / "outputs" / "experiments" / "characterization_validation50",
            repo_root / "outputs" / "experiments" / "characterization_feature_audit" / "metrics",
            repo_root / "outputs" / "characterization",
            repo_root / "outputs" / "metrics",
            repo_root / "outputs",
            repo_root,
        ]
    )
    out: list[Path] = []
    seen = set()
    for d in dirs:
        key = str(d.resolve()) if d.exists() else str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _find_named_file(repo_root: Path, run_dir: Path | None, names: Iterable[str]) -> Path | None:
    names = list(names)
    for d in _candidate_dirs(repo_root, run_dir):
        for name in names:
            p = d / name
            if p.exists():
                return p

    for name in names:
        try:
            matches = list((repo_root / "outputs").rglob(name))
        except Exception:
            matches = []
        if matches:
            matches.sort(
                key=lambda p: (
                    "characterization_validation50" not in str(p),
                    -p.stat().st_mtime if p.exists() else 0,
                )
            )
            return matches[0]
    return None


def _read_named_csv(
    repo_root: Path, run_dir: Path | None, names: Iterable[str]
) -> tuple[pd.DataFrame, Path | None]:
    p = _find_named_file(repo_root, run_dir, names)
    return (_read_csv(p), p) if p is not None else (pd.DataFrame(), None)


@st.cache_data(show_spinner=False)
def _read_live_table(run_dir_text: str | None, table_name: str) -> pd.DataFrame:
    if not run_dir_text:
        return pd.DataFrame()
    run_dir = Path(run_dir_text)
    db_path = run_dir / "run_live.sqlite"
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as connection:
                return pd.read_sql_query(f'SELECT * FROM "{table_name}"', connection)
        except Exception:
            pass
    return _read_csv(run_dir / f"{table_name}.csv")


def _target_quarter_from_star_id(value) -> tuple[str, int | None]:
    match = re.match(r"kic_([^_]+)_q(\d+)", str(value), flags=re.I)
    if not match:
        return _normalize_target_id(value), None
    return _normalize_target_id(match.group(1)), int(match.group(2))


def _attach_target_quarter(frame: pd.DataFrame, characterization: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "target_id" not in out.columns:
        if not characterization.empty and {"star_id", "target_id"}.issubset(characterization.columns):
            keep = ["star_id", "target_id"] + (["quarter"] if "quarter" in characterization.columns else [])
            out = out.merge(characterization[keep].drop_duplicates("star_id"), on="star_id", how="left")
        elif "star_id" in out.columns:
            parsed = out["star_id"].map(_target_quarter_from_star_id)
            out["target_id"] = [item[0] for item in parsed]
            if "quarter" not in out.columns:
                out["quarter"] = [item[1] for item in parsed]
    if "target_id" in out.columns:
        out["target_id"] = out["target_id"].map(_normalize_target_id)
    if "quarter" in out.columns:
        out["quarter"] = pd.to_numeric(out["quarter"], errors="coerce").astype("Int64")
    return out


@st.cache_data(show_spinner=False)
def load_live_benchmark_bundle(run_dir_text: str | None) -> dict:
    characterization = _read_live_table(run_dir_text, "characterization")
    detection = _read_live_table(run_dir_text, "detection")
    injection = _read_live_table(run_dir_text, "injection")
    status = _read_live_table(run_dir_text, "run_status")
    thresholds = _read_live_table(run_dir_text, "fap_thresholds")

    characterization = _canonicalize(characterization)
    detection = _attach_target_quarter(detection, characterization)
    injection = _attach_target_quarter(injection, characterization)
    status = _attach_target_quarter(status, characterization)
    thresholds = _attach_target_quarter(thresholds, characterization)
    return {
        "characterization": characterization,
        "detection": detection,
        "injection": injection,
        "status": status,
        "thresholds": thresholds,
    }


def _first_existing_column(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    return next((c for c in names if c in df.columns), None)


def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    target_col = _first_existing_column(out, ["target_id", "kic", "kic_id", "kepid", "KIC", "target"])
    if target_col and target_col != "target_id":
        out = out.rename(columns={target_col: "target_id"})
    if "target_id" in out.columns:
        out["target_id"] = out["target_id"].map(_normalize_target_id)

    aliases = {
        "robust_scatter": ["robust_scatter", "flux_robust_scale"],
        "skewness": ["skewness", "flux_skewness"],
        "outlier_fraction": ["outlier_fraction", "flux_outlier_fraction"],
        "acf_lag1": ["acf_lag1", "v2_acf_lag_1", "acf_lag_1"],
        "acf_timescale_days": ["acf_timescale_days", "v2_acf_decay_e_days", "acf_decay_e_days"],
        "stationarity_state": ["stationarity_state", "original_series_stationarity_conclusion"],
        "spectral_concentration": ["spectral_concentration", "v2_spectral_concentration"],
        "harmonic_power_ratio": ["harmonic_power_ratio", "v2_spectral_harmonic_power_ratio"],
        "dominant_period_days": ["dominant_period_days", "v2_ls_dominant_period_days"],
        "period_agreement_error": ["period_agreement_error", "v2_ls_acf_period_relative_error"],
        "segment_scale_variability": ["segment_scale_variability", "v2_segment_scale_relative_mad"],
    }

    for canonical, candidates in aliases.items():
        if canonical not in out.columns:
            src = _first_existing_column(out, candidates)
            if src is not None:
                out[canonical] = out[src]

    for c in CONTINUOUS_FEATURES:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


@st.cache_data(show_spinner=False)
def load_characterization_bundle(repo_root_text: str, run_dir_text: str | None = None) -> dict:
    repo_root = Path(repo_root_text)
    run_dir = Path(run_dir_text) if run_dir_text else None

    canonical, _ = _read_named_csv(
        repo_root,
        run_dir,
        ["stellar_features_v2_50.csv", "stellar_features_v2.csv", "canonical_features.csv"],
    )
    master, _ = _read_named_csv(
        repo_root,
        run_dir,
        ["characterization_master_50.csv", "multistar_characterization_per_star.csv"],
    )
    redundancy, _ = _read_named_csv(repo_root, run_dir, ["canonical_redundancy_pairs.csv"])
    spearman, _ = _read_named_csv(repo_root, run_dir, ["canonical_spearman_correlation.csv"])

    if canonical.empty and not master.empty:
        canonical = master.copy()

    canonical = _canonicalize(canonical)
    master = _canonicalize(master)

    # Merge full diagnostic columns into the compact table for annotations only.
    if (
        not canonical.empty
        and not master.empty
        and "target_id" in canonical.columns
        and "target_id" in master.columns
    ):
        extra = [c for c in master.columns if c not in canonical.columns or c == "target_id"]
        if len(extra) > 1:
            canonical = canonical.merge(master[extra], on="target_id", how="left")

    return {
        "canonical": canonical,
        "master": master,
        "redundancy": redundancy,
        "spearman": spearman,
    }


def _candidate_light_curve_paths(
    repo_root: Path,
    run_dir: Path | None,
    target_id: str,
    quarter: int,
) -> list[Path]:
    key = f"kic_{_normalize_target_id(target_id)}_q{int(quarter)}"
    candidates: list[Path] = []

    if run_dir is not None:
        candidates.extend(
            [
                run_dir / "stars" / key / "regularized_light_curve.parquet",
                run_dir / "stars" / key / "characterization_input.parquet",
            ]
        )

    candidates.extend(
        [
            repo_root / "outputs" / "cache" / f"{key}_pdcsap.parquet",
            repo_root / "outputs" / "light_curve_cache" / f"{key}_pdcsap.parquet",
            repo_root / "outputs" / "experiments" / "characterization" / "processed" / f"{key}_characterization_input.parquet",
        ]
    )

    # Bounded fallbacks for the consolidated project tree.
    for root in (repo_root / "outputs", repo_root / "data"):
        if not root.exists():
            continue
        candidates.extend(root.glob(f"**/{key}_pdcsap.parquet"))
        candidates.extend(root.glob(f"**/{key}*/regularized_light_curve.parquet"))
        candidates.extend(root.glob(f"**/{key}*_characterization_input.parquet"))

    out: list[Path] = []
    seen = set()
    for p in candidates:
        key_text = str(p)
        if key_text not in seen:
            seen.add(key_text)
            out.append(p)
    return out


@st.cache_data(show_spinner=False)
def _load_light_curve(
    repo_root_text: str,
    run_dir_text: str | None,
    target_id: str,
    quarter: int,
) -> pd.DataFrame:
    repo_root = Path(repo_root_text)
    run_dir = Path(run_dir_text) if run_dir_text else None
    for p in _candidate_light_curve_paths(repo_root, run_dir, target_id, quarter):
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if not df.empty:
            return df
    return pd.DataFrame()


def _choose_time_flux(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if df.empty:
        return None, None

    time_col = _first_existing_column(df, ["time", "time_days", "bkjd"])
    flux_col = _first_existing_column(
        df,
        ["normalized_flux", "pdcsap_flux", "flux", "PDCSAP_FLUX", "sap_flux"],
    )

    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if time_col is None and numeric:
        time_col = numeric[0]
    if flux_col is None:
        flux_col = next((c for c in numeric if c != time_col), None)
    return time_col, flux_col


def _prepare_light_curve(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Load one light curve and put it on a unit-median relative-flux scale.

    Saved benchmark light curves may contain either already-normalized flux or raw
    PDCSAP counts.  All visual diagnostics need the same dimensionless flux scale,
    otherwise ppm plots can be wrong by orders of magnitude.
    """
    time_col, flux_col = _choose_time_flux(df)
    if time_col is None or flux_col is None:
        return np.array([]), np.array([])

    time = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    flux = pd.to_numeric(df[flux_col], errors="coerce").to_numpy(dtype=float)
    finite_time = np.isfinite(time)
    time = time[finite_time]
    flux = flux[finite_time]
    if time.size == 0:
        return time, flux

    order = np.argsort(time)
    time = time[order]
    flux = flux[order]

    finite_flux = flux[np.isfinite(flux)]
    if finite_flux.size:
        median_flux = float(np.median(finite_flux))
        # Raw Kepler PDCSAP values are usually large positive counts.
        # Already-normalized light curves sit close to unity and are left alone.
        if np.isfinite(median_flux) and median_flux != 0 and (median_flux > 2.0 or median_flux < -2.0):
            flux = flux / median_flux

    return time, flux


# -----------------------------------------------------------------------------
# Numerical helpers for visual diagnostics
# -----------------------------------------------------------------------------


def _num(value) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _robust_scale(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return np.nan
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad


def _relative_ppm(values: np.ndarray) -> np.ndarray:
    """Convert finite relative flux to ppm around its median."""
    x = np.asarray(values, dtype=float)
    finite = x[np.isfinite(x)]
    if not finite.size:
        return np.full_like(x, np.nan, dtype=float)
    med = float(np.median(finite))
    if not np.isfinite(med) or med == 0:
        return 1e6 * (x - float(np.nanmedian(x)))
    return 1e6 * (x / med - 1.0)


def _population_percentile(features: pd.DataFrame, feature: str, selected: str) -> float:
    if feature not in features.columns or "target_id" not in features.columns:
        return np.nan
    s = pd.to_numeric(features[feature], errors="coerce")
    target = features["target_id"].astype(str) == str(selected)
    if not target.any():
        return np.nan
    value = pd.to_numeric(features.loc[target, feature], errors="coerce").dropna()
    population = s.dropna()
    if value.empty or population.empty:
        return np.nan
    v = float(value.iloc[0])
    return 100.0 * ((population < v).sum() + 0.5 * (population == v).sum()) / len(population)


def _population_position(percentile: float, *, lower_is_more: bool = False) -> str:
    if not np.isfinite(percentile):
        return ""
    p = 100.0 - percentile if lower_is_more else percentile
    if p >= 75:
        return "high relative to this 50-star sample"
    if p <= 25:
        return "low relative to this 50-star sample"
    return "mid-range within this 50-star sample"


def _pretty_stationarity_state(value) -> str:
    state = str(value).strip().lower()
    mapping = {
        "stationary_supported": "Stationarity supported",
        "stationary": "Stationarity supported",
        "nonstationary_supported": "Non-stationarity supported",
        "non_stationary_supported": "Non-stationarity supported",
        "inconclusive": "Stationarity inconclusive",
        "mixed": "Stationarity evidence mixed",
    }
    if state in mapping:
        return mapping[state]
    if not state or state == "nan":
        return "Stationarity unavailable"
    return state.replace("_", " ").strip().capitalize()


def _stationarity_metric_value(value) -> str:
    """Short card label; keep the formal interpretation in the plot caption."""
    state = str(value).strip().lower()
    mapping = {
        "stationary_supported": "Supported",
        "stationary": "Supported",
        "nonstationary_supported": "Non-stationary",
        "non_stationary_supported": "Non-stationary",
        "inconclusive": "Inconclusive",
        "mixed": "Mixed evidence",
    }
    if state in mapping:
        return mapping[state]
    if not state or state == "nan":
        return "Unavailable"
    return state.replace("_", " ").strip().capitalize()


def _meaningful_label(value) -> bool:
    text = str(value).strip()
    return bool(text) and text.lower() not in {"—", "-", "nan", "none", "missing", "unavailable"}


def _percentile_rank(series: pd.Series) -> pd.Series:
    """Population-relative rank in [0, 1], preserving NaN for missing values."""
    x = pd.to_numeric(series, errors="coerce")
    return x.rank(method="average", pct=True)


def _derive_dominant_behaviours(features: pd.DataFrame) -> pd.Series:
    """Assign one interpretable statistical behaviour to every star.

    The labels are navigation summaries of the seven-domain characterisation,
    not astrophysical classes.  Thresholds are deliberately population-relative
    because this is a 50-star validation sample rather than a calibrated stellar
    taxonomy.

    Priority:
      1) retain any explicit vetted v2 candidate flag;
      2) otherwise use continuous population ranks from amplitude, distribution
         shape, memory, spectral/coherence and variance-evolution diagnostics;
      3) reserve "Mixed / complex" for genuinely non-dominant cases.
    """
    work = features.copy()

    def rank_of(column: str) -> pd.Series:
        if column not in work.columns:
            return pd.Series(np.nan, index=work.index, dtype=float)
        return _percentile_rank(work[column])

    p_scatter = rank_of("robust_scatter")
    p_acf1 = rank_of("acf_lag1")
    p_tau = rank_of("acf_timescale_days")
    p_spectral = rank_of("spectral_concentration")
    p_harmonic = rank_of("harmonic_power_ratio")
    p_evolution = rank_of("segment_scale_variability")
    p_outlier = rank_of("outlier_fraction")

    if "skewness" in work.columns:
        abs_skew = pd.to_numeric(work["skewness"], errors="coerce").abs()
        p_abs_skew = abs_skew.rank(method="average", pct=True)
    else:
        p_abs_skew = pd.Series(np.nan, index=work.index, dtype=float)

    # Lower LS-ACF disagreement is better coherence.
    if "period_agreement_error" in work.columns:
        p_agreement_error = rank_of("period_agreement_error")
        agreement_strength = 1.0 - p_agreement_error
    else:
        agreement_strength = pd.Series(np.nan, index=work.index, dtype=float)

    def row_max(*series_list: pd.Series) -> pd.Series:
        return pd.concat(series_list, axis=1).max(axis=1, skipna=True)

    memory_strength = row_max(p_acf1, p_tau)
    periodic_strength = pd.concat(
        [p_spectral, p_harmonic, agreement_strength], axis=1
    ).mean(axis=1, skipna=True)
    distribution_strength = row_max(p_abs_skew, p_outlier)

    labels = pd.Series("Mixed / complex", index=work.index, dtype=object)

    for idx, row in work.iterrows():
        # Explicit v2 flags win because they encode the vetted multivariate rules.
        explicit = None
        for column, label in EXPLICIT_BEHAVIOUR_FLAGS:
            if column in row.index and pd.notna(row[column]) and _boolish(row[column]):
                explicit = label
                break
        if explicit is not None:
            labels.at[idx] = explicit
            continue

        sc = p_scatter.at[idx]
        mem = memory_strength.at[idx]
        tau = p_tau.at[idx]
        spec = p_spectral.at[idx]
        per = periodic_strength.at[idx]
        dist = distribution_strength.at[idx]
        evol = p_evolution.at[idx]

        sc = 0.5 if pd.isna(sc) else float(sc)
        mem = 0.5 if pd.isna(mem) else float(mem)
        tau = 0.5 if pd.isna(tau) else float(tau)
        spec = 0.5 if pd.isna(spec) else float(spec)
        per = 0.5 if pd.isna(per) else float(per)
        dist = 0.5 if pd.isna(dist) else float(dist)
        evol = 0.5 if pd.isna(evol) else float(evol)

        # Very stable and low-amplitude backgrounds.
        if sc <= 0.25 and mem <= 0.50 and per <= 0.50 and evol <= 0.50 and dist <= 0.65:
            labels.at[idx] = "Quiet / low variability"

        # Low scatter but clear structure in another domain.
        elif sc <= 0.35 and max(mem, per, evol, dist) >= 0.65:
            labels.at[idx] = "Low-scatter structured"

        # Strong coherent periodicity: concentrated spectrum plus agreement across
        # time/frequency diagnostics.
        elif per >= 0.75 and spec >= 0.60:
            labels.at[idx] = "Coherent periodic"

        # Periodic structure is present, but not strong enough for the coherent class.
        elif per >= 0.65:
            labels.at[idx] = "Quasi-periodic / structured"

        # Background amplitude changes materially over the quarter.
        elif evol >= 0.80:
            labels.at[idx] = "Evolving variability"

        # Persistent temporal correlation without periodicity dominating.
        elif mem >= 0.75:
            labels.at[idx] = "Long-memory / correlated"

        # Strong non-Gaussian / asymmetric behaviour.
        elif dist >= 0.85:
            labels.at[idx] = "Tail-heavy / asymmetric"

        # Large background amplitude relative to the validation population.
        elif sc >= 0.75:
            labels.at[idx] = "High-amplitude / high-scatter"

        # Memory dies rapidly and no other domain dominates.
        elif tau <= 0.40 and per < 0.65 and evol < 0.80:
            labels.at[idx] = "Short-memory stochastic"

        else:
            labels.at[idx] = "Mixed / complex"

    return labels


def _review_flags_for_row(row: pd.Series) -> list[str]:
    """Return only the explicit astrophysical review-screen flags."""
    return [
        label
        for column, label in REVIEW_FLAG_COLUMNS
        if column in row.index and pd.notna(row[column]) and _boolish(row[column])
    ]


def _interpretation_text(domain: str, features: pd.DataFrame, row: pd.Series, selected: str, cadence: float = np.nan) -> str:
    if domain == "scatter":
        value = _format_value(row.get("robust_scatter", np.nan), "robust_scatter")
        p = _population_percentile(features, "robust_scatter", selected)
        return f"{value}; {_population_position(p)}." if np.isfinite(p) else f"Robust scatter: {value}."

    if domain == "distribution":
        skew = _num(row.get("skewness", np.nan))
        outlier = _num(row.get("outlier_fraction", np.nan))
        direction = "positive" if skew > 0 else "negative" if skew < 0 else "minimal"
        outlier_text = f"{100*outlier:.3f}%" if np.isfinite(outlier) else "—"
        return f"Flux distribution has {direction} asymmetry (skewness {skew:.3g}) with {outlier_text} of cadences in the robust tails." if np.isfinite(skew) else f"Robust-tail fraction: {outlier_text}."

    if domain == "acf":
        acf1 = _num(row.get("acf_lag1", np.nan))
        tau = _num(row.get("acf_timescale_days", np.nan))
        if np.isfinite(tau) and np.isfinite(cadence) and cadence > 0:
            cadences = tau / cadence
            return f"ACF(1) = {acf1:.3g}; correlation falls to the e-fold level after about {cadences:.1f} cadence(s) ({tau:.3g} d)."
        return f"ACF(1) = {acf1:.3g}; e-fold timescale = {tau:.3g} d."

    if domain == "stationarity":
        label = _pretty_stationarity_state(row.get("stationarity_state", ""))
        adf = _num(row.get("original_adf_pvalue", np.nan))
        kpss = _num(row.get("original_kpss_pvalue", np.nan))
        pp = _num(row.get("original_pp_pvalue", row.get("pp_pvalue", np.nan)))

        statements = []
        if np.isfinite(adf):
            if adf < 0.05:
                statements.append(f"ADF rejects a unit root (p={adf:.2g})")
            else:
                statements.append(f"ADF does not reject a unit root (p={adf:.2g})")
        if np.isfinite(kpss):
            if kpss < 0.05:
                statements.append(f"KPSS rejects stationarity (p={kpss:.2g})")
            else:
                statements.append(f"KPSS does not reject stationarity (p={kpss:.2g})")
        if np.isfinite(pp):
            if pp < 0.05:
                statements.append(f"PP rejects a unit root (p={pp:.2g})")
            else:
                statements.append(f"PP does not reject a unit root (p={pp:.2g})")

        if statements:
            return "; ".join(statements) + f" → {label.lower()}."
        return label + "."

    if domain == "spectral":
        concentration = _num(row.get("spectral_concentration", np.nan))
        harmonic = _num(row.get("harmonic_power_ratio", np.nan))
        p = _population_percentile(features, "spectral_concentration", selected)
        position = _population_position(p)
        return f"Spectral concentration = {concentration:.3g} ({position}); harmonic support ratio = {harmonic:.3g}."

    if domain == "periodicity":
        period = _num(row.get("dominant_period_days", np.nan))
        agreement = _num(row.get("period_agreement_error", np.nan))
        acf_period = _num(row.get("v2_acf_period_candidate_days", row.get("acf_period_candidate_days", np.nan)))
        if np.isfinite(agreement):
            acf_text = f"; ACF-supported period = {acf_period:.3g} d" if np.isfinite(acf_period) else ""
            return f"Dominant LS period = {period:.3g} d{acf_text}; harmonic-aware LS–ACF disagreement = {100*agreement:.1f}%. Lower disagreement means stronger cross-diagnostic agreement."
        return f"Dominant period = {period:.3g} d."

    if domain == "variance":
        value = _num(row.get("segment_scale_variability", np.nan))
        p = _population_percentile(features, "segment_scale_variability", selected)
        position = _population_position(p)
        return f"Segment-scale variability = {value:.3g}; {position}."

    return ""


def _median_cadence_days(time: np.ndarray) -> float:
    t = np.asarray(time, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return np.nan
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else np.nan


def _pairwise_acf(values: np.ndarray, max_lag: int = 700) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() < 8:
        return np.array([]), np.array([])

    mean = float(np.nanmean(x))
    scale = float(np.nanstd(x))
    if not np.isfinite(scale) or scale <= 0:
        return np.array([]), np.array([])

    z = (x - mean) / scale
    max_lag = min(int(max_lag), max(1, len(z) // 3))
    lags = np.arange(max_lag + 1)
    acf = np.full(max_lag + 1, np.nan, dtype=float)
    acf[0] = 1.0

    for lag in range(1, max_lag + 1):
        a = z[:-lag]
        b = z[lag:]
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() >= 8:
            acf[lag] = float(np.mean(a[mask] * b[mask]))

    return lags, acf


def _acf_lag_table(time: np.ndarray, flux: np.ndarray, max_lag: int = 20) -> pd.DataFrame:
    cadence = _median_cadence_days(time)
    lags, acf = _pairwise_acf(flux, max_lag=max_lag)
    if not lags.size:
        return pd.DataFrame(columns=["Lag", "Lag days", "ACF", "Role"])
    rows = []
    for lag, value in zip(lags[1 : int(max_lag) + 1], acf[1 : int(max_lag) + 1]):
        rows.append(
            {
                "Lag": int(lag),
                "Lag days": float(lag * cadence) if np.isfinite(cadence) else np.nan,
                "ACF": value,
                "Role": "canonical" if int(lag) == 1 else "diagnostic",
            }
        )
    return pd.DataFrame(rows)


def _lomb_scargle_visual(
    time: np.ndarray,
    flux: np.ndarray,
    n_frequency: int = 3200,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(time) & np.isfinite(flux)
    t = np.asarray(time[mask], dtype=float)
    y = np.asarray(flux[mask], dtype=float)
    if t.size < 24:
        return np.array([]), np.array([])

    t = t - np.nanmin(t)
    y = y - np.nanmedian(y)
    baseline = float(np.nanmax(t) - np.nanmin(t))
    cadence = _median_cadence_days(t)
    if not np.isfinite(baseline) or baseline <= 0 or not np.isfinite(cadence) or cadence <= 0:
        return np.array([]), np.array([])

    min_frequency = max(2.0 / baseline, 1e-4)
    max_frequency = min(0.5 / cadence, 24.0)
    if max_frequency <= min_frequency:
        return np.array([]), np.array([])

    freq = np.linspace(min_frequency, max_frequency, int(n_frequency))
    try:
        power = lombscargle(
            t,
            y,
            2.0 * np.pi * freq,
            precenter=True,
            normalize=True,
        )
    except Exception:
        return np.array([]), np.array([])
    return freq, np.asarray(power, dtype=float)


def _spectral_peak_snr_diagnostic(time: np.ndarray, flux: np.ndarray) -> float:
    _, power = _lomb_scargle_visual(time, flux)
    finite = power[np.isfinite(power)]
    if finite.size < 8:
        return np.nan
    med = float(np.median(finite))
    scale = _robust_scale(finite)
    peak = float(np.max(finite))
    if not np.isfinite(scale) or scale <= 0:
        return np.nan
    return float((peak - med) / scale)


def _segment_diagnostics(
    time: np.ndarray,
    flux: np.ndarray,
    n_segments: int = 12,
) -> pd.DataFrame:
    mask = np.isfinite(time) & np.isfinite(flux)
    t = np.asarray(time[mask], dtype=float)
    y = np.asarray(flux[mask], dtype=float)
    if t.size < 24:
        return pd.DataFrame()

    edges = np.linspace(float(t.min()), float(t.max()), int(n_segments) + 1)
    global_median = float(np.median(y))
    global_scale = _robust_scale(y)
    cadence = _median_cadence_days(t)

    rows = []
    for i in range(len(edges) - 1):
        if i == len(edges) - 2:
            m = (t >= edges[i]) & (t <= edges[i + 1])
        else:
            m = (t >= edges[i]) & (t < edges[i + 1])

        n = int(m.sum())
        width = float(edges[i + 1] - edges[i])
        expected = width / cadence if np.isfinite(cadence) and cadence > 0 else np.nan
        coverage = n / expected if np.isfinite(expected) and expected > 0 else np.nan
        gap_affected = bool(n < 8 or (np.isfinite(coverage) and coverage < 0.65))
        center = float(0.5 * (edges[i] + edges[i + 1]))

        if n >= 8:
            med = float(np.median(y[m]))
            scale = _robust_scale(y[m])
            median_shift = (
                (med - global_median) / global_scale
                if np.isfinite(global_scale) and global_scale > 0
                else np.nan
            )
            scale_ratio = (
                scale / global_scale
                if np.isfinite(scale) and np.isfinite(global_scale) and global_scale > 0
                else np.nan
            )
        else:
            med = np.nan
            scale = np.nan
            median_shift = np.nan
            scale_ratio = np.nan

        rows.append(
            {
                "time": float(np.median(t[m])) if n else center,
                "segment_start": float(edges[i]),
                "segment_end": float(edges[i + 1]),
                "median": med,
                "scale": scale,
                "median_shift_in_scale": median_shift,
                "scale_over_global": scale_ratio,
                "n": n,
                "coverage_fraction": coverage,
                "gap_affected": gap_affected,
            }
        )
    return pd.DataFrame(rows)


def _format_value(value, feature: str) -> str:
    if pd.isna(value):
        return "--"
    meta = FEATURE_META[feature]
    if meta["kind"] == "categorical":
        return str(value)
    x = _num(value)
    if not np.isfinite(x):
        return "--"
    if feature == "robust_scatter":
        return f"{1e6*x:.0f} ppm" if abs(x) < 0.1 else f"{x:.4g}"
    if meta["unit"] == "days":
        return f"{x:.3g} d"
    if meta["unit"] == "fraction":
        return f"{x:.4g}"
    return f"{x:.4g}"


def _boolish(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _style_figure(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=28, r=24, t=58, b=42),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(size=13, color="#344258"),
        title=dict(font=dict(size=17, color="#172033"), x=0.01),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(
        gridcolor="#e8edf3",
        zerolinecolor="#d9e1ea",
        automargin=True,
        showspikes=True,
        spikemode="across",
    )
    fig.update_yaxes(
        gridcolor="#e8edf3",
        zerolinecolor="#d9e1ea",
        automargin=True,
        showspikes=True,
        spikemode="across",
    )
    return fig


def _show_plot(fig: go.Figure | None, *, height: int = 360, key: str | None = None):
    if fig is None:
        return
    st.plotly_chart(
        _style_figure(fig, height=height),
        use_container_width=True,
        config=PLOT_CONFIG,
        key=key,
    )


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------


def _raw_light_curve_figure(time: np.ndarray, flux: np.ndarray, target: str) -> go.Figure | None:
    mask = np.isfinite(time) & np.isfinite(flux)
    if mask.sum() < 2:
        return None
    ppm = _relative_ppm(flux[mask])
    fig = go.Figure(
        go.Scattergl(
            x=time[mask],
            y=ppm,
            mode="markers",
            marker=dict(size=3, opacity=0.58),
            name="Relative flux",
            hovertemplate="Time %{x:.5f} d<br>Relative flux %{y:.1f} ppm<extra></extra>",
        )
    )
    fig.update_layout(title=f"KIC {target} · relative light curve", dragmode="zoom")
    fig.update_xaxes(title="Time (days)")
    fig.update_yaxes(title="Relative flux (ppm)")
    return fig


def _scatter_amplitude_figure(features: pd.DataFrame, selected: str) -> go.Figure | None:
    if "robust_scatter" not in features.columns or "target_id" not in features.columns:
        return None
    df = features[["target_id", "robust_scatter"]].copy()
    df["robust_scatter"] = pd.to_numeric(df["robust_scatter"], errors="coerce")
    df = df.dropna().sort_values("robust_scatter").reset_index(drop=True)
    if df.empty:
        return None
    df["Rank"] = np.arange(1, len(df) + 1)
    df["Robust scatter (ppm)"] = 1e6 * df["robust_scatter"]
    df["Star"] = "KIC " + df["target_id"].astype(str)

    other = df[df["target_id"].astype(str) != str(selected)]
    chosen = df[df["target_id"].astype(str) == str(selected)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=other["Rank"],
            y=other["Robust scatter (ppm)"],
            mode="markers",
            marker=dict(size=7, symbol="circle-open"),
            name="Other stars",
            text=other["Star"],
            hovertemplate="%{text}<br>Rank %{x}<br>Robust scatter %{y:.1f} ppm<extra></extra>",
        )
    )
    if not chosen.empty:
        fig.add_trace(
            go.Scatter(
                x=chosen["Rank"],
                y=chosen["Robust scatter (ppm)"],
                mode="markers+text",
                marker=dict(size=13, symbol="diamond"),
                text=["Selected"],
                textposition="top center",
                name="Selected star",
                hovertext=chosen["Star"],
                hovertemplate="%{hovertext}<br>Rank %{x}<br>Robust scatter %{y:.1f} ppm<extra></extra>",
            )
        )
    fig.update_layout(title="Robust scatter across characterised stars", legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(title="Stars ranked by robust scatter")
    fig.update_yaxes(title="Robust scatter (ppm)")
    return fig


def _distribution_shape_figure(
    flux: np.ndarray,
    row: pd.Series,
) -> go.Figure | None:
    x = np.asarray(flux, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return None

    ppm = _relative_ppm(x)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=ppm,
            nbinsx=70,
            histnorm="probability density",
            name="Flux distribution",
            opacity=0.82,
            hovertemplate="Flux offset %{x:.1f} ppm<br>Density %{y:.4g}<extra></extra>",
        )
    )
    # Keep the central distribution readable; extreme points are still retained
    # in the statistics and can be recovered with Plotly autoscale.
    finite_ppm = ppm[np.isfinite(ppm)]
    if finite_ppm.size >= 20:
        lo, hi = np.quantile(finite_ppm, [0.005, 0.995])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pad = 0.08 * (hi - lo)
            fig.update_xaxes(range=[float(lo - pad), float(hi + pad)])

    fig.update_layout(title="Normalized-flux distribution", showlegend=False)
    fig.update_xaxes(title="Relative flux (ppm)")
    fig.update_yaxes(title="Density")
    return fig


def _acf_figure(
    time: np.ndarray,
    flux: np.ndarray,
    row: pd.Series,
    display_lag_count: int | None = 20,
) -> go.Figure | None:
    cadence = _median_cadence_days(time)
    lags, acf = _pairwise_acf(flux)
    if not lags.size or not np.isfinite(cadence):
        return None

    lag_days = lags * cadence
    fig = go.Figure(
        go.Scatter(
            x=lag_days,
            y=acf,
            mode="lines",
            line=dict(width=2),
            name="ACF",
            hovertemplate="Lag %{x:.4f} d<br>ACF %{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(y=1.0 / np.e, line_dash="dot", line_width=1.2, annotation_text="1/e")
    tau = _num(row.get("acf_timescale_days", np.nan))
    if np.isfinite(tau):
        fig.add_vline(
            x=tau,
            line_dash="dash",
            line_width=2,
            annotation_text=f"e-fold {tau:.3g} d",
            annotation_position="top right",
        )

    full_max = float(np.nanmax(lag_days)) if lag_days.size else 1.0
    if display_lag_count is None:
        focus = full_max
    else:
        focus = min(full_max, max(float(display_lag_count) * cadence, cadence))
    fig.update_layout(title="Autocorrelation function")
    fig.update_xaxes(title="Lag (days)", range=[0, focus])
    fig.update_yaxes(title="Autocorrelation", range=[-0.15, 1.05])
    return fig


def _stationarity_figure(
    segments: pd.DataFrame,
    row: pd.Series,
) -> go.Figure | None:
    if segments.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=segments["time"],
            y=segments["median_shift_in_scale"],
            mode="lines+markers",
            name="Segment median",
            hovertemplate="Time %{x:.3f} d<br>Local level %{y:.3f} robust σ<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=1)
    fig.update_layout(title="Segment-level flux location")
    fig.update_xaxes(title="Time (days)")
    fig.update_yaxes(title="Local flux-level change (robust σ)")
    return fig


def _spectral_figure(
    time: np.ndarray,
    flux: np.ndarray,
    row: pd.Series,
) -> go.Figure | None:
    freq, power = _lomb_scargle_visual(time, flux)
    if not freq.size:
        return None

    period_axis = 1.0 / freq
    order = np.argsort(period_axis)
    period_axis = period_axis[order]
    power = power[order]

    period = _num(row.get("dominant_period_days", np.nan))
    baseline = float(np.nanmax(time) - np.nanmin(time)) if np.isfinite(time).sum() >= 2 else np.nan
    cadence = _median_cadence_days(time)
    min_period = max(0.2, 2.0 * cadence) if np.isfinite(cadence) else 0.2
    desired_max = max(30.0, 1.35 * period) if np.isfinite(period) and period > 0 else 30.0
    max_period = min(0.5 * baseline, desired_max) if np.isfinite(baseline) and baseline > 0 else desired_max
    if max_period <= min_period:
        max_period = float(np.nanmax(period_axis))

    view = np.isfinite(period_axis) & np.isfinite(power) & (period_axis >= min_period) & (period_axis <= max_period)
    if not view.any():
        view = np.isfinite(period_axis) & np.isfinite(power)

    fig = go.Figure(
        go.Scattergl(
            x=period_axis[view],
            y=power[view],
            mode="lines",
            line=dict(width=1.5),
            name="Lomb–Scargle power",
            hovertemplate="Period %{x:.4g} d<br>Power %{y:.5g}<extra></extra>",
        )
    )

    if np.isfinite(period) and period > 0:
        markers = (
            (0.5 * period, f"P/2 = {0.5 * period:.3g} d", "dot", 1.2),
            (period, f"P = {period:.3g} d", "dash", 2.0),
            (2.0 * period, f"2P = {2.0 * period:.3g} d", "dot", 1.2),
        )
        for px, label, dash, width in markers:
            if min_period <= px <= max_period:
                fig.add_vline(
                    x=px,
                    line_dash=dash,
                    line_width=width,
                    annotation_text=label,
                    annotation_position="top",
                )

    tick_candidates = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    tickvals = tick_candidates[(tick_candidates >= min_period) & (tick_candidates <= max_period)]
    ticktext = [f"{v:g}" for v in tickvals]

    fig.update_layout(title="Lomb–Scargle power by period")
    fig.update_xaxes(
        title="Period (days)",
        type="log",
        range=[np.log10(min_period), np.log10(max_period)],
        tickmode="array",
        tickvals=tickvals.tolist(),
        ticktext=ticktext,
    )
    fig.update_yaxes(title="Lomb–Scargle power")
    return fig


def _phase_fold_figure(
    time: np.ndarray,
    flux: np.ndarray,
    row: pd.Series,
) -> go.Figure | None:
    period = _num(row.get("dominant_period_days", np.nan))
    mask = np.isfinite(time) & np.isfinite(flux)
    if not np.isfinite(period) or period <= 0 or mask.sum() < 16:
        return None

    t = time[mask]
    y = flux[mask]
    phase = ((t - np.nanmin(t)) % period) / period
    ppm = _relative_ppm(y)

    order = np.argsort(phase)
    phase = phase[order]
    ppm = ppm[order]

    # Two cycles make repeatability visually testable: the pattern from 0–1
    # should recur identically from 1–2 if the fold is genuinely coherent.
    phase_two = np.concatenate([phase, phase + 1.0])
    ppm_two = np.concatenate([ppm, ppm])

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=phase_two,
            y=ppm_two,
            mode="markers",
            marker=dict(size=2.5, opacity=0.14),
            name="Cadences",
            hovertemplate="Phase %{x:.4f}<br>Relative flux %{y:.1f} ppm<extra></extra>",
        )
    )

    edges = np.linspace(0, 1, 41)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        m = (phase >= edges[i]) & (phase < edges[i + 1])
        if m.sum() >= 3:
            binned[i] = float(np.median(ppm[m]))
    finite = np.isfinite(binned)
    if finite.any():
        centers_two = np.concatenate([centers[finite], centers[finite] + 1.0])
        binned_two = np.concatenate([binned[finite], binned[finite]])
        fig.add_trace(
            go.Scatter(
                x=centers_two,
                y=binned_two,
                mode="lines+markers",
                line=dict(width=4.0),
                marker=dict(size=6),
                name="Phase-bin median",
                hovertemplate="Phase %{x:.3f}<br>Bin median %{y:.1f} ppm<extra></extra>",
            )
        )

    finite_ppm = ppm[np.isfinite(ppm)]
    if finite_ppm.size >= 20:
        lo, hi = np.quantile(finite_ppm, [0.005, 0.995])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pad = 0.08 * (hi - lo)
            fig.update_yaxes(range=[float(lo - pad), float(hi + pad)])

    fig.add_vline(x=1.0, line_dash="dot", line_width=1.0)
    fig.update_layout(title=f"Phase-folded light curve at P = {period:.3g} d")
    fig.update_xaxes(title="Phase (two repeated cycles)", range=[0, 2], tickvals=[0, 0.5, 1, 1.5, 2])
    fig.update_yaxes(title="Relative flux (ppm)")
    return fig


def _variance_evolution_figure(
    segments: pd.DataFrame,
    row: pd.Series,
) -> go.Figure | None:
    if segments.empty:
        return None

    fig = go.Figure(
        go.Scatter(
            x=segments["time"],
            y=segments["scale_over_global"],
            mode="lines+markers",
            connectgaps=False,
            line=dict(width=2.2),
            marker=dict(size=7),
            name="Segment robust scale",
            hovertemplate=(
                "Time %{x:.3f} d<br>Local/global robust scatter %{y:.3f}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=1.0, line_dash="dot", line_width=1.2)

    if "gap_affected" in segments.columns:
        gaps = segments[segments["gap_affected"].fillna(False).astype(bool)]
        for i, (_, gap) in enumerate(gaps.iterrows()):
            fig.add_vrect(
                x0=float(gap["segment_start"]),
                x1=float(gap["segment_end"]),
                opacity=0.10,
                line_width=0,
                annotation_text="Gap-affected" if i == 0 else None,
                annotation_position="top left",
            )

    fig.update_layout(title="Segment robust scatter through time")
    fig.update_xaxes(title="Time (days)")
    fig.update_yaxes(title="Local robust scatter / global robust scatter")
    return fig


def _recovery_figure(
    injections: pd.DataFrame,
    target: str,
    pipelines: Iterable[str],
    metric_suffix: str,
    pipeline_label: Callable[[str], str] | None,
) -> go.Figure | None:
    if injections.empty or "target_id" not in injections.columns:
        return None
    inj = injections.copy()
    inj["target_id"] = inj["target_id"].map(_normalize_target_id)
    star = inj[inj["target_id"] == target]
    if star.empty:
        return None

    rows = []
    for pipeline in pipelines:
        c = f"{pipeline}_{metric_suffix}"
        if c not in star.columns:
            continue
        rows.append(
            {
                "Pipeline": pipeline_label(pipeline) if pipeline_label else pipeline,
                "Recovery (%)": 100.0 * star[c].fillna(False).astype(bool).mean(),
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("Recovery (%)", ascending=False)
    fig = px.bar(
        df,
        x="Pipeline",
        y="Recovery (%)",
        text="Recovery (%)",
        title="How the search pipelines behave on this stellar background",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(yaxis_range=[0, 108])
    fig.update_xaxes(tickangle=-22)
    return fig


def _pipeline_id(treatment, detector) -> str:
    return f"{str(treatment)}_{str(detector)}"


def _pipeline_label(pipeline: str) -> str:
    return PIPELINE_LABELS.get(str(pipeline), str(pipeline).replace("_", " ").title())


def _long_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].astype(str).str.lower().isin({"true", "1", "1.0"})


def _star_pipeline_performance_table(
    detection: pd.DataFrame,
    injection: pd.DataFrame,
    selected: str,
) -> tuple[pd.DataFrame, str]:
    columns = [
        "pipeline",
        "Treatment",
        "Detector",
        "Completed injections",
        "Planned injections",
        "Recovery (%)",
        "Calibration",
    ]
    if detection.empty or "target_id" not in detection.columns:
        rows = [
            {
                "pipeline": pipeline,
                "Treatment": treatment.title() if treatment != "gp" else "GP",
                "Detector": "TPS-like" if detector == "tps_like" else detector.upper(),
                "Completed injections": 0,
                "Planned injections": 0,
                "Recovery (%)": np.nan,
                "Calibration": "PROVISIONAL - NOT COMMON-FAP CALIBRATED",
            }
            for treatment in CORE_TREATMENTS
            for detector in CORE_DETECTORS
            for pipeline in [_pipeline_id(treatment, detector)]
        ]
        return pd.DataFrame(rows, columns=columns), "PROVISIONAL - NOT COMMON-FAP CALIBRATED"

    det = detection.copy()
    det["target_id"] = det["target_id"].map(_normalize_target_id)
    det = det[det["target_id"].astype(str).eq(str(selected))]
    if det.empty:
        return pd.DataFrame(columns=columns), "PROVISIONAL - NOT COMMON-FAP CALIBRATED"

    inj = injection.copy() if injection is not None else pd.DataFrame()
    planned = 0
    if not inj.empty and "target_id" in inj.columns:
        inj["target_id"] = inj["target_id"].map(_normalize_target_id)
        inj = inj[inj["target_id"].astype(str).eq(str(selected))]
        if "injection_kind" in inj.columns:
            inj = inj[inj["injection_kind"].astype(str).ne("native")]
        elif "batman_used" in inj.columns:
            inj = inj[_long_bool(inj, "batman_used")]
        planned = int(inj["injection_id"].nunique()) if "injection_id" in inj.columns else int(len(inj))
    if planned == 0 and "injection_id" in det.columns:
        planned = int(det.loc[det["injection_id"].astype(str).ne("native_zero"), "injection_id"].nunique())

    if "above_threshold" not in det.columns and "passes_fap" in det.columns:
        det = det.rename(columns={"passes_fap": "above_threshold"})
    calibrated = "above_threshold" in det.columns and det["above_threshold"].notna().any()
    det["pipeline"] = det["treatment"].astype(str) + "_" + det["detector"].astype(str)
    det["success_bool"] = _long_bool(det, "success")
    det["harmonic_bool"] = _long_bool(det, "harmonic_recovery")
    det["exact_bool"] = _long_bool(det, "exact_recovery")
    det["threshold_bool"] = _long_bool(det, "above_threshold") if calibrated else True
    det["recovered_bool"] = det["success_bool"] & (det["harmonic_bool"] | det["exact_bool"]) & det["threshold_bool"]

    rows = []
    for treatment in CORE_TREATMENTS:
        for detector in CORE_DETECTORS:
            pipeline = _pipeline_id(treatment, detector)
            group = det[det["pipeline"].eq(pipeline)]
            completed = int(group["injection_id"].nunique()) if "injection_id" in group.columns else int(len(group) > 0)
            if group.empty or "injection_id" not in group.columns:
                recovery = np.nan
            else:
                by_injection = group.groupby("injection_id", dropna=False)["recovered_bool"].max()
                recovery = 100.0 * float(by_injection.mean()) if not by_injection.empty else np.nan
            rows.append(
                {
                    "pipeline": pipeline,
                    "Treatment": treatment.title() if treatment != "gp" else "GP",
                    "Detector": "TPS-like" if detector == "tps_like" else detector.upper(),
                    "Completed injections": completed,
                    "Planned injections": planned,
                    "Recovery (%)": recovery,
                    "Calibration": "Common-FAP calibrated" if calibrated else "PROVISIONAL - NOT COMMON-FAP CALIBRATED",
                }
            )
    label = "Common-FAP calibrated" if calibrated else "PROVISIONAL - NOT COMMON-FAP CALIBRATED"
    return pd.DataFrame(rows, columns=columns), label


def _pipeline_performance_figure(table: pd.DataFrame) -> go.Figure | None:
    if table.empty:
        return None
    matrix = table.pivot(index="Treatment", columns="Detector", values="Recovery (%)")
    treatment_order = ["Raw", "ARIMA", "Kalman", "GP"]
    detector_order = ["BLS", "TCF", "TPS-like"]
    matrix = matrix.reindex(index=treatment_order, columns=detector_order)
    text = matrix.applymap(lambda x: "" if pd.isna(x) else f"{x:.1f}%")
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(dtype=float),
            x=list(matrix.columns),
            y=list(matrix.index),
            text=text.to_numpy(),
            texttemplate="%{text}",
            colorscale="Viridis",
            zmin=0,
            zmax=100,
            colorbar=dict(title="Recovery %"),
            hovertemplate="%{y} x %{x}<br>Recovery %{z:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(title="Per-star core pipeline recovery")
    fig.update_xaxes(title="Detector")
    fig.update_yaxes(title="Treatment")
    return fig


def _star_progress_table(status: pd.DataFrame, detection: pd.DataFrame, injection: pd.DataFrame, selected: str) -> pd.DataFrame:
    rows = []
    if not status.empty and "target_id" in status.columns:
        s = status.copy()
        s["target_id"] = s["target_id"].map(_normalize_target_id)
        s = s[s["target_id"].astype(str).eq(str(selected))]
        for stage, group in s.groupby("stage", dropna=False):
            rows.append(
                {
                    "Stage": str(stage).replace("_", " ").title(),
                    "Completed/recorded units": int(len(group)),
                    "Latest status": str(group["status"].iloc[-1]) if "status" in group.columns and len(group) else "",
                    "Last update": str(group["updated_at"].iloc[-1]) if "updated_at" in group.columns and len(group) else "",
                }
            )
    if not any(row["Stage"] == "Detection" for row in rows) and not detection.empty and "target_id" in detection.columns:
        det = detection.copy()
        det["target_id"] = det["target_id"].map(_normalize_target_id)
        det = det[det["target_id"].astype(str).eq(str(selected))]
        if not det.empty:
            rows.append(
                {
                    "Stage": "Detection",
                    "Completed/recorded units": int(det[["injection_id", "treatment", "detector"]].drop_duplicates().shape[0]),
                    "Latest status": "partial",
                    "Last update": "",
                }
            )
    if not any(row["Stage"] == "Injection" for row in rows) and not injection.empty and "target_id" in injection.columns:
        inj = injection.copy()
        inj["target_id"] = inj["target_id"].map(_normalize_target_id)
        inj = inj[inj["target_id"].astype(str).eq(str(selected))]
        if not inj.empty:
            rows.append(
                {
                    "Stage": "Injection",
                    "Completed/recorded units": int(inj["injection_id"].nunique()) if "injection_id" in inj.columns else int(len(inj)),
                    "Latest status": "partial",
                    "Last update": "",
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Scientific summary table
# -----------------------------------------------------------------------------


def _selected_value(row: pd.Series, feature: str) -> str:
    return _format_value(row.get(feature, np.nan), feature)


def _feature_percentile_text(features: pd.DataFrame, feature: str, selected: str) -> str:
    meta = FEATURE_META[feature]
    if meta["kind"] != "continuous":
        return "n/a"
    percentile = _population_percentile(features, feature, selected)
    return f"{percentile:.1f}" if np.isfinite(percentile) else "Unavailable"


def _methods_table(row: pd.Series, features: pd.DataFrame, selected: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Domain": item["domain_label"],
                "Variable": item["label"],
                "Value": _selected_value(row, item["feature"]),
                "Population percentile": _feature_percentile_text(features, item["feature"], selected),
                "Definition / calculation": item["calculation"],
                "Scientific use": item["rationale"],
            }
            for item in CANONICAL_SCHEMA
        ]
    )


def _render_methods_table(row: pd.Series, features: pd.DataFrame, selected: str):
    """Wrapped HTML table: readable in a meeting without horizontal scrolling."""
    df = _methods_table(row, features, selected)
    header = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    body_rows = []
    for _, r in df.iterrows():
        cells = "".join(f"<td>{escape(str(r[c]))}</td>" for c in df.columns)
        body_rows.append(f"<tr>{cells}</tr>")
    html = f"""
    <style>
      .characterization-methods-wrap {{overflow-x:auto; margin-top:.35rem; margin-bottom:1rem;}}
      table.characterization-methods {{width:100%; border-collapse:collapse; table-layout:fixed; font-size:.86rem; background:white;}}
      table.characterization-methods th {{text-align:left; background:#f5f7fa; color:#344258; border:1px solid #dbe3ec; padding:.62rem .65rem; vertical-align:top;}}
      table.characterization-methods td {{border:1px solid #e1e7ee; padding:.6rem .65rem; vertical-align:top; line-height:1.38; white-space:normal; overflow-wrap:anywhere;}}
      table.characterization-methods th:nth-child(1), table.characterization-methods td:nth-child(1) {{width:13%;}}
      table.characterization-methods th:nth-child(2), table.characterization-methods td:nth-child(2) {{width:14%;}}
      table.characterization-methods th:nth-child(3), table.characterization-methods td:nth-child(3) {{width:9%;}}
      table.characterization-methods th:nth-child(4), table.characterization-methods td:nth-child(4) {{width:10%;}}
      table.characterization-methods th:nth-child(5), table.characterization-methods td:nth-child(5) {{width:27%;}}
      table.characterization-methods th:nth-child(6), table.characterization-methods td:nth-child(6) {{width:27%;}}
    </style>
    <div class="characterization-methods-wrap">
      <table class="characterization-methods">
        <thead><tr>{header}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main page
# -----------------------------------------------------------------------------


def render_characterization_v2_page(
    repo_root: Path,
    run_dir: Path | None = None,
    injections: pd.DataFrame | None = None,
    pipelines: Iterable[str] | None = None,
    metric_suffix: str | None = None,
    pipeline_label: Callable[[str], str] | None = None,
    header: Callable[[str, str], None] | None = None,
):
    """Render the intensive scientific characterisation page."""

    if header is not None:
        header(
            "Stars & statistics",
            "Pick a star, inspect the light curve, then interrogate the seven statistical domains used to characterise its stellar background.",
        )
    else:
        st.title("Stars & statistics")
        st.caption(
            "Pick a star, inspect the light curve, then interrogate the seven statistical domains used to characterise its stellar background."
        )

    bundle = load_characterization_bundle(
        str(repo_root),
        str(run_dir) if run_dir is not None else None,
    )
    live_bundle = load_live_benchmark_bundle(str(run_dir) if run_dir is not None else None)
    features = bundle["canonical"].copy()
    live_features = live_bundle["characterization"].copy()
    if not live_features.empty:
        if features.empty:
            features = live_features
        elif "target_id" in features.columns and "target_id" in live_features.columns:
            features = pd.concat([live_features, features], ignore_index=True, sort=False)
            subset = ["target_id"] + (["quarter"] if "quarter" in features.columns else [])
            features = features.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)

    if features.empty or "target_id" not in features.columns:
        st.error(
            "The validated stellar-characterisation table was not found. "
            "Run the 50-star characterisation/audit workflow first."
        )
        return

    features["target_id"] = features["target_id"].map(_normalize_target_id)

    # Navigation labels are derived from the already-validated v2 candidate flags.
    # They are used only to find interesting stars; they do not replace the
    # continuous canonical variables shown below.
    features["_dominant_statistical_behaviour"] = _derive_dominant_behaviours(
        features
    )
    features["_review_flags"] = features.apply(
        _review_flags_for_row,
        axis=1,
    )

    filter_behaviour_col, filter_review_col, selector_col = st.columns([1.25, 1.25, 1.0])

    behaviour_values = [
        str(x)
        for x in features["_dominant_statistical_behaviour"].dropna().unique()
        if _meaningful_label(x)
    ]
    behaviour_options = ["All behaviours"] + [
        label for label in BEHAVIOUR_ORDER if label in behaviour_values
    ]

    with filter_behaviour_col:
        behaviour_filter = st.selectbox(
            "Dominant statistical behaviour",
            behaviour_options,
            key="characterization_v2_behaviour_filter",
            help=(
                "A navigation summary derived from the seven-domain statistics relative to "
                "this 50-star validation sample, while retaining explicit vetted v2 flags when present. "
                "It is not an astrophysical classification."
            ),
        )

    filtered = features
    if behaviour_filter != "All behaviours":
        filtered = filtered[
            filtered["_dominant_statistical_behaviour"] == behaviour_filter
        ]

    review_labels_present = []
    for _, label in REVIEW_FLAG_COLUMNS:
        if filtered["_review_flags"].map(lambda flags: label in flags).any():
            review_labels_present.append(label)

    review_options = ["All review flags"]
    if filtered["_review_flags"].map(len).eq(0).any():
        review_options.append("No review flag")
    review_options.extend(review_labels_present)

    with filter_review_col:
        review_filter = st.selectbox(
            "Review flag",
            review_options,
            key="characterization_v2_review_filter",
            help=(
                "Explicit review screens for rotation/star-spot or pulsation-like structure. "
                "These flags indicate cases worth inspection; they are not classifications."
            ),
        )

    if review_filter == "No review flag":
        filtered = filtered[filtered["_review_flags"].map(len).eq(0)]
    elif review_filter != "All review flags":
        filtered = filtered[
            filtered["_review_flags"].map(lambda flags: review_filter in flags)
        ]

    ids = sorted(
        filtered["target_id"].dropna().astype(str).unique().tolist(),
        key=lambda x: int(x) if str(x).isdigit() else str(x),
    )

    if not ids:
        st.warning(
            "No stars match this behaviour/review combination. "
            "Choose a broader filter."
        )
        return

    # A filter change can invalidate the previously selected KIC. Reset only when
    # necessary so Streamlit never holds a value outside the current option list.
    saved_target = st.session_state.get("characterization_v2_target")
    if saved_target not in ids:
        st.session_state["characterization_v2_target"] = ids[0]

    with selector_col:
        selected = st.selectbox(
            "Star",
            ids,
            format_func=lambda x: f"KIC {x}",
            key="characterization_v2_target",
        )

    st.caption(
        f"Showing {len(ids)} of {features['target_id'].nunique()} characterised stars "
        "for the current filters. Dominant behaviour is a population-relative navigation "
        "summary of the seven statistical domains, not a physical stellar class."
    )

    star_rows = features[features["target_id"] == selected]
    if star_rows.empty:
        return
    row = star_rows.iloc[0]

    quarter = 5
    if "quarter" in row.index and pd.notna(row["quarter"]):
        try:
            quarter = int(float(row["quarter"]))
        except Exception:
            pass
    elif injections is not None and not injections.empty and "quarter" in injections.columns:
        tmp = injections.copy()
        tmp["target_id"] = tmp["target_id"].map(_normalize_target_id)
        q = pd.to_numeric(tmp.loc[tmp["target_id"] == selected, "quarter"], errors="coerce").dropna()
        if not q.empty:
            quarter = int(q.iloc[0])

    dominant_behaviour = str(
        row.get("_dominant_statistical_behaviour", "Mixed / complex")
    )
    active_review_flags = _review_flags_for_row(row)
    stationarity_label = _stationarity_metric_value(row.get("stationarity_state", ""))
    amplitude_label = str(row.get("v2_amplitude_population_label", "—"))
    memory_label = str(row.get("v2_memory_population_label", "—"))

    amplitude_title = "Amplitude population" if _meaningful_label(amplitude_label) else "Robust scatter"
    amplitude_value = (
        amplitude_label
        if _meaningful_label(amplitude_label)
        else _format_value(row.get("robust_scatter", np.nan), "robust_scatter")
    )
    memory_title = "Memory population" if _meaningful_label(memory_label) else "ACF memory"
    memory_value = (
        memory_label
        if _meaningful_label(memory_label)
        else _format_value(row.get("acf_timescale_days", np.nan), "acf_timescale_days")
    )

    m0, m1, m2, m3, m4 = st.columns(5)
    m0.metric("Star ID", f"KIC {selected}")
    m1.metric(amplitude_title, amplitude_value)
    m2.metric(memory_title, memory_value)
    m3.metric("Stationarity", stationarity_label)
    m4.metric("Dominant behaviour", dominant_behaviour)
    st.caption(
        "Review flag(s) for this star: "
        + (", ".join(active_review_flags) if active_review_flags else "none")
        + ". Population/behaviour labels are descriptive screening aids; "
        "review flags are not astrophysical classifications."
    )

    progress_table = _star_progress_table(
        live_bundle["status"],
        live_bundle["detection"],
        live_bundle["injection"],
        selected,
    )
    if not progress_table.empty:
        st.markdown("## Live completion state")
        st.dataframe(progress_table, hide_index=True, use_container_width=True)

    light_curve = _load_light_curve(
        str(repo_root),
        str(run_dir) if run_dir is not None else None,
        selected,
        quarter,
    )
    time, flux = _prepare_light_curve(light_curve)
    cadence = _median_cadence_days(time) if time.size else np.nan

    # ------------------------------------------------------------------
    # Raw light curve first.
    # ------------------------------------------------------------------
    st.markdown("## Light curve")
    if time.size and np.isfinite(flux).sum():
        _show_plot(
            _raw_light_curve_figure(time, flux, selected),
            height=390,
            key=f"raw_lc_{selected}",
        )
        st.caption("Toolbar: zoom, zoom out, box zoom, pan, autoscale and reset. All downstream visual diagnostics use the same unit-median relative-flux scale.")
    else:
        st.warning(
            "The selected star's saved light curve could not be located. "
            "Canonical values are still available, but the seven diagnostic figures need the regularized/PDCSAP light curve."
        )

    # ------------------------------------------------------------------
    # Seven scientific domains.
    # ------------------------------------------------------------------
    st.markdown("## Statistical characterisation")
    st.caption(
        "Seven scientific domains, eleven canonical variables. Each graph exposes the diagnostic visually; the line beneath it states the interpretation for the selected star."
    )

    segments = _segment_diagnostics(time, flux) if time.size else pd.DataFrame()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 1. Scatter amplitude")
        st.caption("How large is the stellar-background variability relative to the sample?")
        _show_plot(_scatter_amplitude_figure(features, selected), height=330, key=f"scatter_amplitude_{selected}")
        st.caption("**Interpretation:** " + _interpretation_text("scatter", features, row, selected, cadence))

    with c2:
        st.markdown("### 2. Distribution shape")
        st.caption("Is the normalized-flux distribution asymmetric or tail-heavy?")
        _show_plot(_distribution_shape_figure(flux, row) if time.size else None, height=330, key=f"distribution_shape_{selected}")
        st.caption("**Interpretation:** " + _interpretation_text("distribution", features, row, selected, cadence))

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### 3. Autocorrelation / memory")
        st.caption("How strongly does one cadence predict the next, and how quickly does that memory decay?")
        lag_window = st.radio(
            "ACF lag window",
            ["20", "10", "50", "Full"],
            horizontal=True,
            key=f"acf_lag_window_{selected}",
        )
        lag_count = None if lag_window == "Full" else int(lag_window)
        _show_plot(_acf_figure(time, flux, row, display_lag_count=lag_count) if time.size else None, height=345, key=f"acf_{selected}")
        lag_table = _acf_lag_table(time, flux, max_lag=50 if lag_count is None else lag_count) if time.size else pd.DataFrame()
        if not lag_table.empty:
            st.dataframe(
                lag_table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Lag days": st.column_config.NumberColumn(format="%.5f"),
                    "ACF": st.column_config.NumberColumn(format="%.4f"),
                },
            )
        st.caption("**Interpretation:** " + _interpretation_text("acf", features, row, selected, cadence))

    with c4:
        st.markdown("### 4. Stationarity")
        st.caption("Do the local flux level and the ADF/KPSS tests support a stable process?")
        _show_plot(_stationarity_figure(segments, row), height=345, key=f"stationarity_{selected}")
        st.caption("**Interpretation:** " + _interpretation_text("stationarity", features, row, selected, cadence))

    c5, c6 = st.columns(2)
    with c5:
        st.markdown("### 5. Spectral structure")
        st.caption("Where is the variability power concentrated in period space, and is there harmonic support?")
        _show_plot(_spectral_figure(time, flux, row) if time.size else None, height=355, key=f"spectral_{selected}")
        spectral_snr = _spectral_peak_snr_diagnostic(time, flux) if time.size else np.nan
        if np.isfinite(spectral_snr):
            st.caption(f"Diagnostic only: robust Lomb-Scargle peak prominence/SNR = {spectral_snr:.2f}.")
        st.caption("**Interpretation:** " + _interpretation_text("spectral", features, row, selected, cadence))

    with c6:
        st.markdown("### 6. Periodicity / coherence")
        st.caption("Does the light curve repeat coherently when folded at the dominant period?")
        _show_plot(_phase_fold_figure(time, flux, row) if time.size else None, height=355, key=f"periodicity_{selected}")
        st.caption("**Interpretation:** " + _interpretation_text("periodicity", features, row, selected, cadence))

    st.markdown("### 7. Variability stability")
    st.caption("Does the local variability amplitude remain stable through the quarter?")
    _show_plot(_variance_evolution_figure(segments, row), height=330, key=f"variance_evolution_{selected}")
    st.caption("**Interpretation:** " + _interpretation_text("variance", features, row, selected, cadence))

    # ------------------------------------------------------------------
    # One methods table, optimized for scientific discussion.
    # ------------------------------------------------------------------
    st.markdown("## Canonical variables and definitions")
    st.caption(
        "The eleven retained variables. The table gives the selected-star value, exact operational definition and why each statistic is scientifically useful."
    )
    _render_methods_table(row, features, selected)

    # ------------------------------------------------------------------
    # Per-star live pipeline performance.
    # ------------------------------------------------------------------
    performance_table, calibration_label = _star_pipeline_performance_table(
        live_bundle["detection"],
        live_bundle["injection"],
        selected,
    )
    if not performance_table.empty:
        st.markdown("## Per-star pipeline performance")
        if calibration_label != "Common-FAP calibrated":
            st.warning("PROVISIONAL - NOT COMMON-FAP CALIBRATED")
        else:
            st.caption("Common-FAP calibrated recovery for the selected star.")
        _show_plot(
            _pipeline_performance_figure(performance_table),
            height=360,
            key=f"pipeline_performance_{selected}",
        )
        st.dataframe(
            performance_table.drop(columns=["pipeline"]),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Recovery (%)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

    # ------------------------------------------------------------------
    # Scientific bridge back to legacy transit recovery — figure only.
    # ------------------------------------------------------------------
    if injections is not None and not injections.empty and pipelines and metric_suffix:
        st.markdown("## Characterisation and recovery")
        st.caption(
            "Star-level injection recovery connects the measured stellar-background properties to the practical question: which background treatments help or hurt weak-transit recovery for this star?"
        )
        _show_plot(
            _recovery_figure(injections, selected, pipelines, metric_suffix, pipeline_label),
            height=370,
            key=f"recovery_{selected}",
        )
