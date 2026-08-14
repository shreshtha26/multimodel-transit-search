from __future__ import annotations

import json
import re
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# Multi-model Transit Search — scientific demo dashboard
# =============================================================================

st.set_page_config(
    page_title="Multi-model Transit Search",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

PIPELINES = (
    "raw_bls",
    "raw_tcf",
    "arima_bls",
    "arima_tcf",
    "kalman_bls",
    "kalman_tcf",
    "gp_bls",
    "gp_tcf",
)

# Display fallback only; overridden from benchmark_config.json when available.
TOP_K_DISPLAY = 10

# Dependence-analysis defaults. A numerical winner is only called meaningful
# when it clears these percentage-point margins.
MIN_DOMINANCE_PP_DEFAULT = 10.0
MIN_COMPLEMENTARITY_PP_DEFAULT = 10.0
CONSENSUS_EASY_THRESHOLD = 0.80
CONSENSUS_HARD_THRESHOLD = 0.20

PIPELINE_META = {
    "raw_bls": ("Raw", "BLS"),
    "raw_tcf": ("Raw", "TCF"),
    "arima_bls": ("ARIMA", "BLS"),
    "arima_tcf": ("ARIMA", "TCF"),
    "kalman_bls": ("Kalman", "BLS"),
    "kalman_tcf": ("Kalman", "TCF"),
    "gp_bls": ("GP", "BLS"),
    "gp_tcf": ("GP", "TCF"),
}

PIPELINE_LABELS = {
    "raw_bls": "Raw + BLS",
    "raw_tcf": "Raw + TCF",
    "arima_bls": "ARIMA + BLS",
    "arima_tcf": "ARIMA + TCF",
    "kalman_bls": "Kalman + BLS",
    "kalman_tcf": "Kalman + TCF",
    "gp_bls": "Gaussian Process + BLS",
    "gp_tcf": "Gaussian Process + TCF",
}


def pipeline_label(value: str) -> str:
    value = str(value)
    return PIPELINE_LABELS.get(value, value.replace("_", " ").title())

FEATURE_LABELS = {
    "acf_lag_1": "ACF lag 1",
    "acf_timescale_days": "ACF timescale (d)",
    "acf_decay_e_days": "ACF e-fold decay (d)",
    "original_adf_pvalue": "ADF p-value",
    "original_kpss_pvalue": "KPSS p-value",
    "variance_drift": "Variance drift",
    "rolling_variance_max_to_median": "Rolling variance max/median",
    "spectral_strength": "Spectral strength",
    "dominant_period_days": "Dominant period (d)",
    "dominant_lomb_scargle_power": "Lomb–Scargle peak power",
    "spectral_entropy": "Spectral entropy",
    "flux_skewness": "Flux skewness",
    "flux_excess_kurtosis": "Excess kurtosis",
    "gap_fraction": "Gap fraction",
    "finite_observation_fraction": "Finite observation fraction",
    "flux_std": "Flux σ",
    "flux_robust_scale": "Robust scale",
}

REGIME_META = {
    "quiet_low_scatter": {
        "label": "Quiet / low-scatter",
        "question": "How much do extra background models help when the light curve is already comparatively quiet?",
        "expected_structure": "Low short-timescale scatter; useful as a clean reference regime.",
    },
    "gap_heavy": {
        "label": "Gap-heavy",
        "question": "Which models remain stable when the cadence series is fragmented or missing substantial observations?",
        "expected_structure": "Large missing-data fraction / fragmented cadence coverage.",
    },
    "high_scatter": {
        "label": "High-scatter",
        "question": "Which modelling strategy can improve detection when high-amplitude stochastic variability dominates?",
        "expected_structure": "Elevated flux dispersion / robust scale.",
    },
    "long_memory": {
        "label": "Long-memory / correlated",
        "question": "Do ARIMA, state-space, GP, wavelet or hybrid approaches help when correlations persist over longer lags?",
        "expected_structure": "Persistent ACF / longer characteristic correlation timescale.",
    },
    "smooth_background_dominant": {
        "label": "Smooth-background dominant",
        "question": "Can GP/state-space style smooth-background models remove slow variability without suppressing the transit?",
        "expected_structure": "Slow, coherent background structure relative to the transit duration.",
    },
}


def canonical_regime_key(value) -> str:
    """Normalize any saved label for the five benchmark background strata."""
    if value is None or pd.isna(value):
        return "unknown"
    raw = str(value).strip().lower()
    if not raw or raw in {"nan", "none", "unknown", "—"}:
        return "unknown"
    norm = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")

    aliases = {}
    for key, meta in REGIME_META.items():
        aliases[re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")] = key
        aliases[re.sub(r"[^a-z0-9]+", "_", meta["label"].lower()).strip("_")] = key

    # A few common manifest/display variants.
    aliases.update({
        "quiet_low_scatter": "quiet_low_scatter",
        "gap_heavy": "gap_heavy",
        "high_scatter": "high_scatter",
        "long_memory": "long_memory",
        "long_memory_correlated": "long_memory",
        "smooth_background": "smooth_background_dominant",
        "smooth_background_dominant": "smooth_background_dominant",
    })
    return aliases.get(norm, "unknown")


def regime_label(value: str) -> str:
    key = canonical_regime_key(value)
    if key == "unknown":
        return "Not loaded"
    return REGIME_META[key]["label"]


def regime_question(value: str) -> str:
    key = canonical_regime_key(value)
    return REGIME_META.get(key, {}).get(
        "question",
        "How does this statistical background regime change the relative performance of the candidate pipelines?",
    )


REGIME_COLUMN_CANDIDATES = (
    "selection_stratum",
    "statistical_stratum",
    "background_stratum",
    "sample_stratum",
    "target_stratum",
    "stratum",
    "background_regime",
    "regime",
    "selection_bucket",
    "selection_group",
)


def _known_regime_score(series: pd.Series) -> int:
    return int(series.dropna().map(canonical_regime_key).ne("unknown").sum())


def _best_regime_column(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None

    # First try the column names the project is expected to use.
    scored = []
    for c in REGIME_COLUMN_CANDIDATES:
        if c in df.columns:
            scored.append((_known_regime_score(df[c]), c))

    # If a manifest used a different column name, discover it from its values.
    # This only accepts columns containing the five known scientific strata.
    if not scored or max(score for score, _ in scored) == 0:
        for c in df.columns:
            if c in REGIME_COLUMN_CANDIDATES:
                continue
            s = df[c]
            if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
                score = _known_regime_score(s)
                if score:
                    scored.append((score, c))

    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1] if scored[0][0] > 0 else None


def attach_regime_metadata(
    df: pd.DataFrame,
    repo_root: Path,
    run_dir: Path,
    characterization: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach a clean `_regime_key` using the actual five-star stratification when available."""
    out = df.copy()
    if out.empty or "target_id" not in out.columns:
        out["_regime_key"] = "unknown"
        return out

    out["target_id"] = out["target_id"].map(normalize_target_id)

    # 1) Prefer a scientifically meaningful stratum already present in injections.
    col = _best_regime_column(out)
    if col:
        out["_regime_key"] = out[col].map(canonical_regime_key)
        return out

    # 2) Try characterization / target-selection tables.
    if characterization is not None and not characterization.empty and "target_id" in characterization.columns:
        cdf = characterization.copy()
        cdf["target_id"] = cdf["target_id"].map(normalize_target_id)
        ccol = _best_regime_column(cdf)
        if ccol:
            mapping = cdf[["target_id", ccol]].dropna().drop_duplicates("target_id")
            mapping = mapping.rename(columns={ccol: "_regime_key"})
            out = out.merge(mapping, on="target_id", how="left")
            if out["_regime_key"].notna().any():
                out["_regime_key"] = out["_regime_key"].map(canonical_regime_key)
                return out

    # 3) Try known manifest locations used by this project.
    manifest_candidates = [
        run_dir / "manifest.csv",
        run_dir / "benchmark_manifest.csv",
        repo_root / "configs" / "kepler_clean_background_manifest.csv",
        repo_root / "configs" / "kepler_clean_background_manifest_10star.csv",
    ]
    for p in manifest_candidates:
        if not p.exists():
            continue
        try:
            m = pd.read_csv(p, dtype={"target_id": str})
        except Exception:
            continue
        if "target_id" not in m.columns:
            # KIC is sometimes named differently in manifests.
            kic_col = next((c for c in ("kic", "kic_id", "kepid", "KIC") if c in m.columns), None)
            if kic_col:
                m = m.rename(columns={kic_col: "target_id"})
        if "target_id" not in m.columns:
            continue
        m["target_id"] = m["target_id"].map(normalize_target_id)
        mcol = _best_regime_column(m)
        if not mcol:
            continue
        mapping = m[["target_id", mcol]].dropna().drop_duplicates("target_id")
        mapping = mapping.rename(columns={mcol: "_regime_key"})
        out2 = out.merge(mapping, on="target_id", how="left")
        if out2["_regime_key"].notna().any():
            out2["_regime_key"] = out2["_regime_key"].map(canonical_regime_key)
            return out2

    # 4) Do not expose an internal population label as a "statistical regime".
    out["_regime_key"] = "unknown"
    return out

CSS = """
<style>
/* =========================================================
   FULL-SCREEN PRESENTATION LAYOUT
   Designed for laptop screen-sharing without browser zoom.
   ========================================================= */

html {
    font-size: 18px;
}

body {
    overflow-x: hidden;
}

/* Use almost the full main canvas instead of a narrow centered column */
.block-container {
    width: 100% !important;
    max-width: 1540px !important;
    padding-top: 1.6rem !important;
    padding-bottom: 4rem !important;
    padding-left: 2.4rem !important;
    padding-right: 2.4rem !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Main typography */
h1 {
    font-size: 3.0rem !important;
    line-height: 1.04 !important;
    letter-spacing: -0.025em !important;
    margin-bottom: 0.35rem !important;
}

h2 {
    font-size: 2.0rem !important;
    line-height: 1.12 !important;
    letter-spacing: -0.015em !important;
    margin-top: 1.6rem !important;
    margin-bottom: 0.75rem !important;
}

h3 {
    font-size: 1.45rem !important;
    line-height: 1.15 !important;
}

p, li, .stMarkdown {
    font-size: 1.08rem !important;
    line-height: 1.46 !important;
}

.small-note {
    color: #969696;
    font-size: 1.08rem !important;
    line-height: 1.4 !important;
    margin-bottom: 1.2rem !important;
}

/* =========================================================
   SIDEBAR — intentionally readable in screen share
   ========================================================= */
section[data-testid="stSidebar"] {
    min-width: 460px !important;
    max-width: 460px !important;
    width: 460px !important;
    border-right: 1px solid rgba(120,120,120,0.14);
}

section[data-testid="stSidebar"] .block-container {
    width: 100% !important;
    max-width: 460px !important;
    padding: 1.9rem 1.7rem 2.2rem 1.7rem !important;
}

section[data-testid="stSidebar"] h2 {
    font-size: 2.0rem !important;
    line-height: 1.1 !important;
}

section[data-testid="stSidebar"] h3 {
    font-size: 1.38rem !important;
    margin-top: 1.1rem !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    font-size: 1.14rem !important;
    line-height: 1.35 !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 0.58rem 0.45rem !important;
    margin: 0.12rem 0 !important;
    border-radius: 10px !important;
    min-height: 2.8rem !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(127,127,127,0.07);
}

/* The experiment dropdown should look like a real selector, not a code field */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    min-height: 3.35rem !important;
    border-radius: 12px !important;
    padding-left: 0.4rem !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] span {
    font-size: 1.08rem !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] button {
    font-size: 0.98rem !important;
}

/* =========================================================
   METRICS / CARDS
   ========================================================= */
[data-testid="stMetric"] {
    border: 1px solid rgba(120,120,120,0.15);
    border-radius: 16px;
    padding: 18px 20px !important;
    min-height: 132px !important;
    background: rgba(127,127,127,0.03);
}

[data-testid="stMetricLabel"] {
    font-size: 1.02rem !important;
    font-weight: 650 !important;
    line-height: 1.25 !important;
}

[data-testid="stMetricValue"] {
    font-size: 2.45rem !important;
    line-height: 1.05 !important;
    margin-top: 0.15rem !important;
}

[data-testid="stMetricDelta"] {
    font-size: 0.92rem !important;
}

/* =========================================================
   CONTROLS / TABLES / EXPANDERS
   ========================================================= */
div[data-testid="stExpander"] {
    border: 1px solid rgba(120,120,120,0.15);
    border-radius: 14px;
    margin-top: 0.55rem;
}

div[data-testid="stExpander"] summary {
    font-size: 1.0rem !important;
    min-height: 3rem !important;
}

div[data-baseweb="select"] > div {
    border-radius: 11px;
    min-height: 3rem;
}

div[data-baseweb="select"] span {
    font-size: 1rem !important;
}

.stButton > button {
    border-radius: 11px;
    min-height: 3rem;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

[data-testid="stDataFrame"] {
    font-size: 1rem !important;
}

button[data-baseweb="tab"] {
    font-size: 1.05rem !important;
    font-weight: 650 !important;
    padding-top: 0.75rem !important;
    padding-bottom: 0.75rem !important;
}

/* =========================================================
   CALLOUTS / METHOD CARDS
   ========================================================= */
.science-callout {
    border-left: 4px solid #888;
    padding: 0.55rem 0 0.55rem 1rem;
    margin: 0.7rem 0 1.1rem 0;
    font-size: 1.05rem;
}

.method-card {
    border: 1px solid rgba(120,120,120,0.15);
    border-radius: 15px;
    padding: 17px 18px;
    min-height: 132px;
    background: rgba(127,127,127,0.03);
    margin-bottom: 12px;
}

.method-title {
    font-size: 1.16rem;
    font-weight: 720;
    margin-bottom: 6px;
}

.method-note {
    font-size: 0.95rem;
    color: #929292;
    line-height: 1.35;
}

.pill-now, .pill-next {
    display: inline-block;
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 0.78rem;
    font-weight: 720;
    margin-bottom: 8px;
}

.pill-now { background: rgba(52,168,83,0.16); }
.pill-next { background: rgba(66,133,244,0.16); }

/* Plotly should occupy the available width */
[data-testid="stPlotlyChart"],
.js-plotly-plot,
.plot-container,
.svg-container {
    width: 100% !important;
}

/* Reduce unnecessary top/bottom chrome */
[data-testid="stHeader"] {
    height: 2.2rem;
}

/* Laptop-safe scaling */
@media (max-width: 1400px) {
    html { font-size: 17px; }

    section[data-testid="stSidebar"] {
        min-width: 420px !important;
        max-width: 420px !important;
        width: 420px !important;
    }

    section[data-testid="stSidebar"] .block-container {
        max-width: 420px !important;
    }

    .block-container {
        max-width: 100% !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
    }
}

@media (max-width: 1050px) {
    html { font-size: 16px; }

    section[data-testid="stSidebar"] {
        min-width: 360px !important;
        max-width: 360px !important;
        width: 360px !important;
    }

    section[data-testid="stSidebar"] .block-container {
        max-width: 360px !important;
    }

    .block-container {
        padding-left: 1.15rem !important;
        padding-right: 1.15rem !important;
    }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


st.markdown(
    """
<style>
/* =========================================================
   CLEAN SCIENTIFIC PRESENTATION THEME
   ========================================================= */
:root {
    --page: #f6f8fb;
    --surface: #ffffff;
    --surface-soft: #f0f4f8;
    --sidebar: #eef3f8;
    --border: #dbe3ec;
    --text: #172033;
    --muted: #66758a;
    --accent: #2563eb;
    --accent-soft: #e8f0ff;
    --teal: #0f9f8f;
    --violet: #7456d8;
    --amber: #b7791f;
    --amber-soft: #fff7df;
    --green: #15803d;
    --green-soft: #ecf8ef;
}

html,
body,
[data-testid="stAppViewContainer"] {
    background: var(--page) !important;
    color: var(--text) !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: var(--page) !important;
}

.block-container {
    max-width: 1380px !important;
    padding-top: 1.15rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Remove Streamlit top chrome / black strip */
header,
header[data-testid="stHeader"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stMainMenu"],
.stAppToolbar,
#MainMenu,
footer {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Full-label metric cards with hover info */
.info-card {
    background: #ffffff;
    border: 1px solid #dbe3ec;
    border-top: 4px solid #2563eb;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    min-height: 118px;
    width: 100%;
}
.info-card.teal { border-top-color: #0f9f8f; }
.info-card.violet { border-top-color: #7456d8; }
.info-card.amber { border-top-color: #b7791f; }

.info-card-label {
    color: #5b6a7f;
    font-size: 0.88rem;
    line-height: 1.24;
    font-weight: 650;
    white-space: normal;
    overflow: visible;
    text-overflow: clip;
    margin-bottom: 0.42rem;
}
.info-card-value {
    color: #172033;
    font-size: 2rem;
    line-height: 1.05;
    font-weight: 780;
}
.info-card-note {
    color: #6a788b;
    font-size: 0.78rem;
    line-height: 1.25;
    margin-top: 0.35rem;
}
.info-dot {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.05rem;
    height: 1.05rem;
    margin-left: 0.32rem;
    border: 1px solid #9badc1;
    border-radius: 999px;
    color: #60758c;
    font-size: 0.67rem;
    font-weight: 750;
    cursor: help;
    vertical-align: text-top;
}

/* Typography */
h1, h2, h3 {
    color: var(--text) !important;
    text-shadow: none !important;
}
h1 {
    font-size: 2.55rem !important;
    letter-spacing: -0.025em !important;
    margin-bottom: 0.18rem !important;
}
h2 {
    font-size: 1.55rem !important;
    margin-top: 1.15rem !important;
}
h3 {
    font-size: 1.16rem !important;
}
p, li, .stMarkdown {
    color: var(--text);
}
.small-note {
    color: var(--muted) !important;
}

/* Sidebar: readable but not dominant */
section[data-testid="stSidebar"] {
    min-width: 315px !important;
    max-width: 315px !important;
    width: 315px !important;
    background: var(--sidebar) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .block-container {
    width: 100% !important;
    max-width: 315px !important;
    padding: 1.25rem 1.05rem 1.5rem 1.05rem !important;
}
section[data-testid="stSidebar"] h2 {
    font-size: 1.52rem !important;
    color: var(--text) !important;
}
section[data-testid="stSidebar"] h3 {
    font-size: 1.08rem !important;
    color: #35435a !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #35435a !important;
    font-size: 0.96rem !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 0.42rem 0.35rem !important;
    margin: 0.05rem 0 !important;
    border-radius: 9px !important;
    border: 1px solid transparent !important;
    min-height: 2.35rem !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #e4ebf4 !important;
    border-color: #d5deea !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #ccd7e5 !important;
    min-height: 2.8rem !important;
    border-radius: 10px !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: #ffffff !important;
    color: #2f3b50 !important;
    border: 1px solid #ccd7e5 !important;
}

/* Controls */
div[data-baseweb="select"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}
.stButton > button {
    border-radius: 9px !important;
    min-height: 2.55rem !important;
    background: #ffffff !important;
    color: #273349 !important;
    border: 1px solid #ccd7e5 !important;
    box-shadow: none !important;
    transition: border-color 0.12s ease, background 0.12s ease;
}
.stButton > button:hover {
    transform: none !important;
    background: #f7faff !important;
    border-color: #7ea6ef !important;
    box-shadow: none !important;
}
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: #ffffff !important;
    border-color: var(--accent) !important;
}

/* Navigation + action buttons: explicitly override Streamlit theme colors.
   Newer Streamlit versions style primary/secondary buttons via stBaseButton test IDs,
   which can otherwise inherit dark/black theme backgrounds. */
div[data-testid="stButton"] button,
button[data-testid="stBaseButton-secondary"] {
    background: #ffffff !important;
    color: #273349 !important;
    border: 1px solid #ccd7e5 !important;
    box-shadow: none !important;
}
div[data-testid="stButton"] button:hover,
button[data-testid="stBaseButton-secondary"]:hover {
    background: #f7faff !important;
    color: #1f56c2 !important;
    border-color: #7ea6ef !important;
}
button[data-testid="stBaseButton-primary"] {
    background: #e8f0ff !important;
    color: #1f56c2 !important;
    border: 1px solid #bcd2ff !important;
    box-shadow: none !important;
}
button[data-testid="stBaseButton-primary"]:hover {
    background: #dce9ff !important;
    color: #194da8 !important;
    border-color: #8fb4fb !important;
}
div[data-testid="stButton"] button p,
div[data-testid="stButton"] button span,
button[data-testid^="stBaseButton"] p,
button[data-testid^="stBaseButton"] span {
    color: inherit !important;
}

/* Final light-theme override: Streamlit can inject dark button/summary styles after theme CSS. */
button[data-testid^="stBaseButton"],
div[data-testid="stButton"] > button,
div[data-testid="stButton"] button,
.stButton button {
    background: #ffffff !important;
    color: #21324d !important;
    border: 1px solid #c9d9ef !important;
    box-shadow: none !important;
}
button[data-testid^="stBaseButton"]:hover,
div[data-testid="stButton"] button:hover,
.stButton button:hover {
    background: #eef5ff !important;
    color: #174ea6 !important;
    border-color: #9ebff0 !important;
}
button[data-testid="stBaseButton-primary"],
div[data-testid="stButton"] button[kind="primary"],
.stButton button[kind="primary"] {
    background: #dfeaff !important;
    color: #174ea6 !important;
    border-color: #aac7f4 !important;
}
button[data-testid="stBaseButton-primary"] p,
button[data-testid="stBaseButton-primary"] span {
    color: #174ea6 !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: #e5efff !important;
    border-color: #bdd2f5 !important;
    color: #174ea6 !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) span,
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
    color: #174ea6 !important;
}

/* Expanders: never use the app/browser dark summary surface. */
div[data-testid="stExpander"],
div[data-testid="stExpander"] details {
    background: #ffffff !important;
    color: #21324d !important;
}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] details > summary {
    background: #edf5ff !important;
    color: #21324d !important;
    border-radius: 9px !important;
}
div[data-testid="stExpander"] summary:hover {
    background: #e2efff !important;
}
div[data-testid="stExpander"] summary p,
div[data-testid="stExpander"] summary span {
    color: #21324d !important;
}
div[data-testid="stExpander"] summary svg {
    color: #41668f !important;
    fill: #41668f !important;
}

/* Tabs: light blue/white in both selected and unselected states. */
button[data-baseweb="tab"],
button[role="tab"] {
    background: #ffffff !important;
    color: #40536e !important;
    border-radius: 8px 8px 0 0 !important;
}
button[data-baseweb="tab"][aria-selected="true"],
button[role="tab"][aria-selected="true"] {
    background: #e5efff !important;
    color: #174ea6 !important;
}
button[data-baseweb="tab"] p,
button[role="tab"] p { color: inherit !important; }

/* Small static tables follow the presentation palette. */
[data-testid="stTable"] table { background: #ffffff !important; color: #21324d !important; }
[data-testid="stTable"] th { background: #eaf3ff !important; color: #21324d !important; }
[data-testid="stTable"] td { background: #ffffff !important; color: #21324d !important; }

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
    border-radius: 12px !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
}

/* Custom overview */
.hero-shell {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin: 0 0 0.85rem 0 !important;
    box-shadow: none !important;
}
.hero-kicker {
    color: var(--accent) !important;
    font-size: 0.74rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 750;
    margin-bottom: 0.3rem;
}
.hero-title {
    color: var(--text) !important;
    font-size: 2.45rem !important;
    line-height: 1.03;
    font-weight: 800;
    margin-bottom: 0.25rem;
}
.hero-meta {
    color: var(--muted) !important;
    font-size: 0.98rem;
    max-width: 900px;
}

.headline-card {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: 4px solid var(--accent) !important;
    border-radius: 12px !important;
    padding: 0.85rem 1rem !important;
    min-height: 108px !important;
    box-shadow: none !important;
}
.headline-card.violet {
    border-top-color: var(--violet) !important;
}
.headline-card.green {
    border-top-color: var(--teal) !important;
}
.headline-label {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    font-weight: 650;
}
.headline-value {
    color: var(--text) !important;
    font-size: 2rem !important;
    line-height: 1.0;
    font-weight: 800;
    margin: 0.35rem 0;
}
.headline-delta {
    color: #53637a !important;
    font-size: 0.78rem !important;
    font-weight: 600;
}
.section-chip {
    background: var(--accent-soft) !important;
    color: #1f56c2 !important;
    border: 1px solid #cfe0ff !important;
    font-size: 0.7rem !important;
    padding: 0.26rem 0.52rem !important;
    margin-bottom: 0.35rem !important;
}

/* Alerts should inform, not dominate */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    box-shadow: none !important;
    padding: 0.65rem 0.8rem !important;
}
div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.86rem !important;
    line-height: 1.35 !important;
}
div[data-testid="stAlert"][kind="warning"] {
    background: var(--amber-soft) !important;
    border: 1px solid #efd99a !important;
    color: #654b16 !important;
}

/* Expanders and tables */
div[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stDataFrame"] {
    background: var(--surface) !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* Plot surfaces */
[data-testid="stPlotlyChart"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 0.2rem 0.3rem !important;
}

/* Method cards */
.method-card {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
}
.method-title {
    color: var(--text) !important;
}
.method-note {
    color: var(--muted) !important;
}
.pill-now {
    background: var(--green-soft) !important;
    color: var(--green) !important;
}
.pill-next {
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
}

/* separators */
hr {
    border-color: var(--border) !important;
}

@media (max-width: 1400px) {
    section[data-testid="stSidebar"] {
        min-width: 290px !important;
        max-width: 290px !important;
        width: 290px !important;
    }
    section[data-testid="stSidebar"] .block-container {
        max-width: 290px !important;
    }
    .block-container {
        max-width: 100% !important;
        padding-left: 1.4rem !important;
        padding-right: 1.4rem !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)



# =============================================================================
# Paths and data loading
# =============================================================================

def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in (start, *start.parents):
        if (p / "pyproject.toml").exists() and (p / "src").exists():
            return p
    return start


REPO_ROOT = find_repo_root(Path.cwd())
DEFAULT_RUN_ROOT = REPO_ROOT / "outputs" / "experiments" / "multistar_challenger_benchmark"


def normalize_target_id(value) -> str:
    text = str(value).upper().replace("KIC", "").strip()
    try:
        return str(int(float(text)))
    except Exception:
        return text


def star_key(target_id, quarter) -> str:
    return f"kic_{normalize_target_id(target_id)}_q{int(quarter)}"


def discover_runs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    runs = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if (p / "metrics").exists() or (p / "stars").exists() or (p / "benchmark_config.json").exists():
            runs.append(p)
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def experiment_label(name: str) -> str:
    """Convert internal run-folder IDs into presentation-friendly names."""
    raw = str(name).strip()
    special = {
        "clean_q5_50star": "Clean Kepler Q5 · 50-star benchmark",
        "clean_q5_10star": "Clean Kepler Q5 · 10-star benchmark",
        "clean_q5_5star": "Clean Kepler Q5 · 5-star benchmark",
        "runtime_test_medium": "Early 5-star runtime benchmark",
        "runtime_test_small": "Early small runtime test",
    }
    if raw in special:
        return special[raw]

    label = raw.replace("_", " ").replace("-", " ")
    label = re.sub(r"\bq(\d+)\b", lambda m: f"Q{m.group(1)}", label, flags=re.I)
    label = re.sub(r"\b(\d+)\s*star\b", r"\1-star", label, flags=re.I)
    label = re.sub(r"\s+", " ", label).strip()

    replacements = {
        "clean": "Clean",
        "runtime test": "Runtime test",
        "multistar": "Multi-star",
        "challenger": "challenger",
        "benchmark": "benchmark",
    }
    for old, new in replacements.items():
        label = re.sub(old, new, label, flags=re.I)

    return label[:1].upper() + label[1:] if label else raw


def compact_experiment_label(name: str) -> str:
    """Short version used in small UI controls."""
    raw = str(name).strip()
    special = {
        "clean_q5_50star": "Clean Q5 · 50 stars",
        "clean_q5_10star": "Clean Q5 · 10 stars",
        "clean_q5_5star": "Clean Q5 · 5 stars",
        "runtime_test_medium": "Early 5-star test",
        "runtime_test_small": "Early small test",
    }
    return special.get(raw, experiment_label(raw))


@st.cache_data(show_spinner=False)
def read_csv_safe(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"target_id": str})
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_json_safe(path_text: str) -> dict:
    path = Path(path_text)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def collect_injections(run_dir_text: str, include_partial: bool = True) -> pd.DataFrame:
    run_dir = Path(run_dir_text)
    global_path = run_dir / "metrics" / "multistar_challenger_injections.csv"
    if global_path.exists():
        df = pd.read_csv(global_path, dtype={"target_id": str})
        df["_source_state"] = "final_metrics"
        return df

    if not include_partial:
        return pd.DataFrame()

    frames = []
    stars_dir = run_dir / "stars"
    if stars_dir.exists():
        for star_dir in sorted(stars_dir.glob("kic_*_q*")):
            path = star_dir / "injections.csv"
            if not path.exists():
                continue
            try:
                frame = pd.read_csv(path, dtype={"target_id": str})
            except Exception:
                continue
            if frame.empty:
                continue
            frame["_source_state"] = "complete_star" if (star_dir / "COMPLETE").exists() else "checkpoint_partial"
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ("target_id", "quarter", "case_index") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False)
def collect_star_summaries(run_dir_text: str) -> pd.DataFrame:
    run_dir = Path(run_dir_text)
    global_path = run_dir / "metrics" / "multistar_challenger_star_summary.csv"
    if global_path.exists():
        return pd.read_csv(global_path, dtype={"target_id": str})

    rows = []
    stars_dir = run_dir / "stars"
    if not stars_dir.exists():
        return pd.DataFrame()
    for star_dir in sorted(stars_dir.glob("kic_*_q*")):
        path = star_dir / "star_summary.json"
        if not path.exists():
            continue
        try:
            row = json.loads(path.read_text())
        except Exception:
            continue
        row["_complete"] = (star_dir / "COMPLETE").exists()
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def collect_calibration(run_dir_text: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = Path(run_dir_text)
    thresholds = []
    nulls = []

    # Most current layout
    star_cal = run_dir / "star_calibration"
    candidate_roots = [star_cal]

    # Also allow nested calibration folders if the experiment layout changes.
    candidate_roots += [p for p in run_dir.glob("*calibration*") if p.is_dir() and p != star_cal]

    seen = set()
    for root in candidate_roots:
        if not root.exists():
            continue
        for star_dir in root.glob("kic_*_q*"):
            key = str(star_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            t = star_dir / "fap_thresholds.csv"
            n = star_dir / "null_trials.csv"
            if t.exists():
                try:
                    thresholds.append(pd.read_csv(t, dtype={"target_id": str}))
                except Exception:
                    pass
            if n.exists():
                try:
                    nulls.append(pd.read_csv(n, dtype={"target_id": str}))
                except Exception:
                    pass

    tdf = pd.concat(thresholds, ignore_index=True) if thresholds else pd.DataFrame()
    ndf = pd.concat(nulls, ignore_index=True) if nulls else pd.DataFrame()
    return tdf, ndf


@st.cache_data(show_spinner=False)
def collect_characterization(run_dir_text: str, repo_root_text: str) -> pd.DataFrame:
    run_dir = Path(run_dir_text)
    repo_root = Path(repo_root_text)

    preferred = [
        run_dir / "characterization_analysis" / "multistar_characterization_per_star.csv",
        run_dir / "metrics" / "multistar_characterization_per_star.csv",
    ]
    for p in preferred:
        if p.exists():
            try:
                return pd.read_csv(p, dtype={"target_id": str})
            except Exception:
                pass

    # Common target-selection feature files in this project.
    fallback = [
        repo_root / "outputs" / "target_selection" / "kepler_catalog_clean_candidate_features.csv",
        repo_root / "outputs" / "target_selection" / "kepler_catalog_clean_pool.csv",
    ]
    for p in fallback:
        if p.exists():
            try:
                df = pd.read_csv(p, dtype={"target_id": str})
                return df
            except Exception:
                pass

    return pd.DataFrame()


def available_pipelines(df: pd.DataFrame) -> list[str]:
    found = []
    for p in PIPELINES:
        if any(c.startswith(f"{p}_") for c in df.columns):
            found.append(p)
    return found


def harmonized_thresholds(thresholds: pd.DataFrame) -> pd.DataFrame:
    if thresholds.empty:
        return thresholds
    t = thresholds.copy()
    if "fap_level" in t.columns:
        t["fap_level"] = pd.to_numeric(t["fap_level"], errors="coerce")
        t = t[np.isclose(t["fap_level"], 0.01, equal_nan=False)]
    for c in ("target_id", "pipeline"):
        if c not in t.columns:
            return pd.DataFrame()
    t["target_id"] = t["target_id"].map(normalize_target_id)
    if "quarter" in t.columns:
        t["quarter"] = pd.to_numeric(t["quarter"], errors="coerce").astype("Int64")
    return t


def add_calibrated_columns(injections: pd.DataFrame, thresholds: pd.DataFrame, pipelines: Iterable[str]) -> pd.DataFrame:
    out = injections.copy()
    if out.empty:
        return out
    out["target_id"] = out["target_id"].map(normalize_target_id)

    t = harmonized_thresholds(thresholds)
    if t.empty:
        return out

    keys = ["target_id"]
    if "quarter" in out.columns and "quarter" in t.columns:
        out["quarter"] = pd.to_numeric(out["quarter"], errors="coerce").astype("Int64")
        keys.append("quarter")

    for p in pipelines:
        pthr = t[t["pipeline"] == p].copy()
        if pthr.empty or "score_threshold" not in pthr.columns:
            continue
        keep = keys + ["score_threshold"]
        pthr = pthr[keep].drop_duplicates(keys, keep="last").rename(columns={"score_threshold": f"{p}_fap01_threshold"})
        out = out.merge(pthr, on=keys, how="left")

        score_col = f"{p}_score"
        harmonic_col = f"{p}_harmonic_rank1_matched"
        exact_col = f"{p}_exact_rank1_matched"
        thr_col = f"{p}_fap01_threshold"
        if score_col in out.columns and harmonic_col in out.columns:
            score = pd.to_numeric(out[score_col], errors="coerce")
            thr = pd.to_numeric(out[thr_col], errors="coerce")
            out[f"{p}_fap01_detected"] = score > thr
            out[f"{p}_fap01_harmonic_recovered"] = (
                out[harmonic_col].fillna(False).astype(bool) & out[f"{p}_fap01_detected"].fillna(False)
            )
            if exact_col in out.columns:
                out[f"{p}_fap01_exact_recovered"] = (
                    out[exact_col].fillna(False).astype(bool) & out[f"{p}_fap01_detected"].fillna(False)
                )
    return out


def topk_recovered(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals.notna() & (vals >= 1)


def pipeline_summary(df: pd.DataFrame, pipelines: Iterable[str]) -> pd.DataFrame:
    rows = []
    for p in pipelines:
        branch, detector = PIPELINE_META.get(p, (p.split("_")[0], p.split("_")[-1]))
        row = {"pipeline": p, "branch": branch, "detector": detector, "n": len(df)}

        for suffix, label in (
            ("harmonic_rank1_matched", "harmonic_rank1"),
            ("exact_rank1_matched", "exact_rank1"),
            ("fap01_harmonic_recovered", "recovery_at_1pct_fap"),
            ("fap01_exact_recovered", "exact_recovery_at_1pct_fap"),
        ):
            c = f"{p}_{suffix}"
            if c in df.columns:
                s = df[c].fillna(False).astype(bool)
                row[label] = float(s.mean())

        top_cols = [f"{p}_exact_rank_topk", f"{p}_half_period_rank_topk", f"{p}_double_period_rank_topk"]
        existing = [c for c in top_cols if c in df.columns]
        if existing:
            top = pd.Series(False, index=df.index)
            for c in existing:
                top |= topk_recovered(df[c])
            row["harmonic_topk"] = float(top.mean())

        runtime_col = f"{p}_runtime_seconds"
        if runtime_col in df.columns:
            row["average_runtime_seconds"] = float(pd.to_numeric(df[runtime_col], errors="coerce").mean())

        rows.append(row)
    return pd.DataFrame(rows)


def union_rate(df: pd.DataFrame, pipelines: Iterable[str], suffix: str) -> tuple[float, int]:
    cols = [f"{p}_{suffix}" for p in pipelines if f"{p}_{suffix}" in df.columns]
    if not cols or df.empty:
        return np.nan, 0
    u = pd.Series(False, index=df.index)
    for c in cols:
        u |= df[c].fillna(False).astype(bool)
    return float(u.mean()), int(u.sum())


def raw_baseline_for(pipeline: str) -> str:
    detector = PIPELINE_META.get(pipeline, ("", ""))[1].lower()
    return f"raw_{detector}"


def unique_recoveries(df: pd.DataFrame, pipeline: str, suffix: str, pipelines: Iterable[str]) -> int:
    col = f"{pipeline}_{suffix}"
    if col not in df.columns:
        return 0
    current = df[col].fillna(False).astype(bool)
    others = pd.Series(False, index=df.index)
    for p in pipelines:
        if p == pipeline:
            continue
        c = f"{p}_{suffix}"
        if c in df.columns:
            others |= df[c].fillna(False).astype(bool)
    return int((current & ~others).sum())


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def _recovery_columns(df: pd.DataFrame, pipelines: Iterable[str], suffix: str) -> dict[str, str]:
    return {p: f"{p}_{suffix}" for p in pipelines if f"{p}_{suffix}" in df.columns}


def _classify_dependence(
    rates: dict[str, float],
    union_rate_value: float,
    min_dominance_pp: float,
    min_complementarity_pp: float,
) -> tuple[str, str, float, float, str, float, float]:
    """Return best, second, margin, gain, preference, spread, and structure label."""
    valid = [(p, float(v)) for p, v in rates.items() if np.isfinite(v)]
    if not valid:
        return "—", "—", np.nan, np.nan, "No meaningful preference", np.nan, "No strong structure"

    valid.sort(key=lambda x: x[1], reverse=True)
    best_p, best_v = valid[0]
    second_p, second_v = valid[1] if len(valid) > 1 else ("—", np.nan)
    margin_pp = 100.0 * (best_v - second_v) if np.isfinite(second_v) else np.nan
    gain_pp = 100.0 * (union_rate_value - best_v) if np.isfinite(union_rate_value) else np.nan
    vals = [v for _, v in valid]
    spread_pp = 100.0 * (max(vals) - min(vals)) if vals else np.nan
    meaningful = pipeline_label(best_p) if np.isfinite(margin_pp) and margin_pp >= min_dominance_pp else "No meaningful preference"

    if vals and min(vals) >= CONSENSUS_EASY_THRESHOLD:
        structure = "Consensus easy"
    elif vals and max(vals) <= CONSENSUS_HARD_THRESHOLD:
        structure = "Consensus hard"
    elif np.isfinite(margin_pp) and margin_pp >= min_dominance_pp:
        structure = "Pipeline dominance"
    elif np.isfinite(gain_pp) and gain_pp >= min_complementarity_pp:
        structure = "Complementary / disagreement"
    else:
        structure = "No strong structure"

    return best_p, second_p, margin_pp, gain_pp, meaningful, spread_pp, structure


def _group_dependence_summary(
    group: pd.DataFrame,
    pipelines: Iterable[str],
    suffix: str,
    min_dominance_pp: float,
    min_complementarity_pp: float,
) -> dict:
    recovery_cols = _recovery_columns(group, pipelines, suffix)
    rates: dict[str, float] = {}
    union_hit = pd.Series(False, index=group.index)
    for p, c in recovery_cols.items():
        hit = group[c].fillna(False).astype(bool)
        rates[p] = float(hit.mean())
        union_hit |= hit
    union_value = float(union_hit.mean()) if recovery_cols and len(group) else np.nan
    best_p, second_p, margin_pp, gain_pp, meaningful, spread_pp, structure = _classify_dependence(
        rates, union_value, min_dominance_pp, min_complementarity_pp
    )
    return {
        "rates": rates,
        "union": union_value,
        "best_pipeline": best_p,
        "second_pipeline": second_p,
        "margin_pp": margin_pp,
        "gain_pp": gain_pp,
        "meaningful": meaningful,
        "spread_pp": spread_pp,
        "structure": structure,
    }


def _weak_transit_subset(
    df: pd.DataFrame,
    mode: str = "Shallowest depth only",
    custom_depth_ppm: float = 500.0,
) -> tuple[pd.DataFrame, str]:
    depth_col = _first_existing_column(df, ("injected_depth", "depth", "transit_depth"))
    if depth_col is None or df.empty:
        return df.copy(), "Weak-depth field unavailable; using all loaded injections"

    depth = pd.to_numeric(df[depth_col], errors="coerce")
    levels = sorted(v for v in depth.dropna().unique() if np.isfinite(v))
    if not levels:
        return df.copy(), "Weak-depth values unavailable; using all loaded injections"

    if mode == "Bottom two depth levels" and len(levels) >= 2:
        keep_levels = levels[:2]
        out = df[depth.isin(keep_levels)].copy()
        label = "Bottom two depth levels: " + ", ".join(f"{1e6*v:.0f} ppm" for v in keep_levels)
        return out, label
    if mode == "Custom depth threshold":
        threshold = float(custom_depth_ppm) / 1e6
        out = df[depth <= threshold + 1e-15].copy()
        return out, f"Injected depth ≤ {custom_depth_ppm:.0f} ppm"

    shallow = levels[0]
    out = df[np.isclose(depth, shallow, rtol=0, atol=max(abs(shallow) * 1e-10, 1e-15))].copy()
    return out, f"Shallowest loaded depth: {1e6*shallow:.0f} ppm"


def build_star_dependence_table(
    df: pd.DataFrame,
    pipelines: Iterable[str],
    suffix: str,
    min_dominance_pp: float,
    min_complementarity_pp: float,
    weak_mode: str,
    custom_depth_ppm: float,
) -> tuple[pd.DataFrame, str]:
    weak, weak_label = _weak_transit_subset(df, weak_mode, custom_depth_ppm)
    if weak.empty or "target_id" not in weak.columns:
        return pd.DataFrame(), weak_label

    rows = []
    for target, g in weak.groupby("target_id", dropna=False):
        s = _group_dependence_summary(g, pipelines, suffix, min_dominance_pp, min_complementarity_pp)
        row = {
            "Star ID": f"KIC {normalize_target_id(target)}",
            "Background stratum": regime_label(g["_regime_key"].iloc[0]) if "_regime_key" in g.columns else "Unavailable",
            "N weak injections": int(len(g)),
        }
        for p, value in s["rates"].items():
            row[f"{pipeline_label(p)} (%)"] = 100.0 * value
        best_rate = s["rates"].get(s["best_pipeline"], np.nan)
        second_rate = s["rates"].get(s["second_pipeline"], np.nan)
        row.update({
            "Best numerical pipeline": pipeline_label(s["best_pipeline"]) if s["best_pipeline"] != "—" else "—",
            "Best recovery (%)": 100.0 * best_rate if np.isfinite(best_rate) else np.nan,
            "Second-best (%)": 100.0 * second_rate if np.isfinite(second_rate) else np.nan,
            "Dominance margin (pp)": s["margin_pp"],
            "Meaningful preference": s["meaningful"],
            "Multi-model union (%)": 100.0 * s["union"] if np.isfinite(s["union"]) else np.nan,
            "Multi-model gain (pp)": s["gain_pp"],
            "Recovery spread (pp)": s["spread_pp"],
            "Structure": s["structure"],
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Dominance margin (pp)", "Multi-model gain (pp)"], ascending=[False, False], na_position="last")
    return out, weak_label


def _star_level_for_combo(
    df: pd.DataFrame,
    group_cols: list[str],
    pipelines: Iterable[str],
    suffix: str,
) -> pd.DataFrame:
    recovery_cols = _recovery_columns(df, pipelines, suffix)
    if not recovery_cols or "target_id" not in df.columns:
        return pd.DataFrame()
    work = df[group_cols + ["target_id"] + list(recovery_cols.values())].copy()
    for c in recovery_cols.values():
        work[c] = work[c].fillna(False).astype(bool)
    # If a design combination is represented more than once for a star, count the star as
    # recovered when any corresponding injection is recovered rather than double-weighting it.
    return work.groupby(group_cols + ["target_id"], dropna=False, as_index=False)[list(recovery_cols.values())].max()


def _winner_unique_rescue_fraction(star_level: pd.DataFrame, winner: str, pipelines: Iterable[str], suffix: str) -> float:
    if star_level.empty or winner == "—":
        return np.nan
    win_col = f"{winner}_{suffix}"
    if win_col not in star_level.columns:
        return np.nan
    win = star_level[win_col].fillna(False).astype(bool)
    others = pd.Series(False, index=star_level.index)
    for p in pipelines:
        if p == winner:
            continue
        c = f"{p}_{suffix}"
        if c in star_level.columns:
            others |= star_level[c].fillna(False).astype(bool)
    return float((win & ~others).mean())


def _combo_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    period_col = _first_existing_column(df, ("injected_period_days", "injected_period", "period_days"))
    duration_col = _first_existing_column(df, ("injected_duration_hours", "injected_duration", "duration_hours"))
    depth_col = _first_existing_column(df, ("injected_depth", "depth", "transit_depth"))
    return period_col, duration_col, depth_col


def build_transit_dependence_table(
    df: pd.DataFrame,
    pipelines: Iterable[str],
    suffix: str,
    min_dominance_pp: float,
    min_complementarity_pp: float,
) -> pd.DataFrame:
    period_col, duration_col, depth_col = _combo_columns(df)
    combo_cols = [c for c in (period_col, duration_col, depth_col) if c]
    if len(combo_cols) < 3:
        return pd.DataFrame()
    star_level = _star_level_for_combo(df, combo_cols, pipelines, suffix)
    if star_level.empty:
        return pd.DataFrame()

    rows = []
    for key, g in star_level.groupby(combo_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        s = _group_dependence_summary(g, pipelines, suffix, min_dominance_pp, min_complementarity_pp)
        values = dict(zip(combo_cols, key))
        row = {
            "Period (d)": values.get(period_col, np.nan),
            "Duration (h)": values.get(duration_col, np.nan),
            "Depth (ppm)": 1e6 * float(values.get(depth_col, np.nan)) if pd.notna(values.get(depth_col, np.nan)) else np.nan,
            "N stars": int(g["target_id"].nunique()),
        }
        for p, value in s["rates"].items():
            row[f"{pipeline_label(p)} star recovery (%)"] = 100.0 * value
        best_rate = s["rates"].get(s["best_pipeline"], np.nan)
        second_rate = s["rates"].get(s["second_pipeline"], np.nan)
        row.update({
            "Best numerical pipeline": pipeline_label(s["best_pipeline"]) if s["best_pipeline"] != "—" else "—",
            "Best recovery (%)": 100.0 * best_rate if np.isfinite(best_rate) else np.nan,
            "Second-best (%)": 100.0 * second_rate if np.isfinite(second_rate) else np.nan,
            "Dominance margin (pp)": s["margin_pp"],
            "Meaningful preference": s["meaningful"],
            "Multi-model union (%)": 100.0 * s["union"] if np.isfinite(s["union"]) else np.nan,
            "Multi-model gain (pp)": s["gain_pp"],
            "Winner unique-rescue (%)": 100.0 * _winner_unique_rescue_fraction(g, s["best_pipeline"], pipelines, suffix),
            "Structure": s["structure"],
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Dominance margin (pp)", "Multi-model gain (pp)"], ascending=[False, False], na_position="last")
    return out


def build_interaction_dependence_table(
    df: pd.DataFrame,
    pipelines: Iterable[str],
    suffix: str,
    min_dominance_pp: float,
    min_complementarity_pp: float,
) -> pd.DataFrame:
    period_col, duration_col, depth_col = _combo_columns(df)
    if "_regime_key" not in df.columns or not all((period_col, duration_col, depth_col)):
        return pd.DataFrame()
    group_cols = ["_regime_key", period_col, duration_col, depth_col]
    star_level = _star_level_for_combo(df, group_cols, pipelines, suffix)
    if star_level.empty:
        return pd.DataFrame()

    # Original injection counts are reported separately so that percentages have context.
    original_counts = (
        df.groupby(group_cols, dropna=False)
        .size()
        .rename("_n_injections")
        .reset_index()
    )
    star_level = star_level.merge(original_counts, on=group_cols, how="left")

    rows = []
    for key, g in star_level.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        s = _group_dependence_summary(g, pipelines, suffix, min_dominance_pp, min_complementarity_pp)
        values = dict(zip(group_cols, key))
        row = {
            "Background stratum": regime_label(values.get("_regime_key", "unknown")),
            "Period (d)": values.get(period_col, np.nan),
            "Duration (h)": values.get(duration_col, np.nan),
            "Depth (ppm)": 1e6 * float(values.get(depth_col, np.nan)) if pd.notna(values.get(depth_col, np.nan)) else np.nan,
            "N stars": int(g["target_id"].nunique()),
            "N injections": int(pd.to_numeric(g["_n_injections"], errors="coerce").max()) if "_n_injections" in g.columns else int(len(g)),
        }
        for p, value in s["rates"].items():
            row[f"{pipeline_label(p)} recovery (%)"] = 100.0 * value
        best_rate = s["rates"].get(s["best_pipeline"], np.nan)
        row.update({
            "Best numerical pipeline": pipeline_label(s["best_pipeline"]) if s["best_pipeline"] != "—" else "—",
            "Dominance margin (pp)": s["margin_pp"],
            "Meaningful preference": s["meaningful"],
            "Multi-model union (%)": 100.0 * s["union"] if np.isfinite(s["union"]) else np.nan,
            "Multi-model gain (pp)": s["gain_pp"],
            "Winner unique-rescue (%)": 100.0 * _winner_unique_rescue_fraction(g, s["best_pipeline"], pipelines, suffix),
            "Structure": s["structure"],
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Dominance margin (pp)", "Multi-model gain (pp)"], ascending=[False, False], na_position="last")
    return out


def build_multimodel_gain_table(
    df: pd.DataFrame,
    pipelines: Iterable[str],
    suffix: str,
    best_pipeline: str,
) -> pd.DataFrame:
    recovery_cols = _recovery_columns(df, pipelines, suffix)
    if not recovery_cols or df.empty:
        return pd.DataFrame()
    union_hit = pd.Series(False, index=df.index)
    for c in recovery_cols.values():
        union_hit |= df[c].fillna(False).astype(bool)
    best_col = recovery_cols.get(best_pipeline)
    best_hit = df[best_col].fillna(False).astype(bool) if best_col else pd.Series(False, index=df.index)
    gain_mask = union_hit & ~best_hit
    gain_n = int(gain_mask.sum())

    rows = []
    for p, c in recovery_cols.items():
        hit = df[c].fillna(False).astype(bool)
        others = pd.Series(False, index=df.index)
        for op, oc in recovery_cols.items():
            if op != p:
                others |= df[oc].fillna(False).astype(bool)
        raw = raw_baseline_for(p)
        raw_col = recovery_cols.get(raw)
        delta_raw = np.nan
        if raw_col is not None:
            delta_raw = 100.0 * (hit.mean() - df[raw_col].fillna(False).astype(bool).mean())
        rescues_beyond_best = int((hit & ~best_hit).sum())
        pairwise_uplift = 100.0 * float((best_hit | hit).mean() - best_hit.mean()) if len(df) else np.nan
        rows.append({
            "Pipeline": pipeline_label(p),
            "Recovery (%)": 100.0 * float(hit.mean()),
            "Exclusive recoveries": int((hit & ~others).sum()),
            "Cases recovered when best fixed misses": rescues_beyond_best,
            "Coverage of multi-model gain cases (%)": 100.0 * rescues_beyond_best / gain_n if gain_n else 0.0,
            "Pairwise uplift over best fixed (pp)": pairwise_uplift,
            "Δ vs detector-matched raw (pp)": delta_raw,
        })
    return pd.DataFrame(rows).sort_values(
        ["Cases recovered when best fixed misses", "Exclusive recoveries"], ascending=[False, False]
    )


def _compact_preference_counts(table: pd.DataFrame, unit_name: str) -> tuple[pd.DataFrame, int, int]:
    """Condense a dependence table to at most five rows for the Overview page."""
    if table.empty or "Meaningful preference" not in table.columns:
        return pd.DataFrame(), 0, 0
    total = int(len(table))
    pref = table["Meaningful preference"].fillna("No meaningful preference").astype(str)
    clear = pref[pref != "No meaningful preference"]
    clear_n = int(len(clear))
    rows = []
    counts = clear.value_counts()
    for name, n in counts.head(3).items():
        rows.append({"Preference": name, unit_name: int(n), "% of total": 100.0 * n / total if total else np.nan})
    if len(counts) > 3:
        n_other = int(counts.iloc[3:].sum())
        rows.append({"Preference": "Other clear preferences", unit_name: n_other, "% of total": 100.0 * n_other / total if total else np.nan})
    no_pref = total - clear_n
    rows.append({"Preference": "No meaningful preference", unit_name: no_pref, "% of total": 100.0 * no_pref / total if total else np.nan})
    return pd.DataFrame(rows), clear_n, total


def _compact_transit_rows(table: pd.DataFrame, max_rows: int = 4) -> pd.DataFrame:
    """Most clearly separated transit morphologies, with only decision-relevant columns."""
    if table.empty:
        return pd.DataFrame()
    t = table.copy()
    if "Meaningful preference" in t.columns:
        clear = t[t["Meaningful preference"] != "No meaningful preference"].copy()
    else:
        clear = pd.DataFrame()
    if clear.empty:
        return pd.DataFrame([{
            "Transit morphology": "No clear preference",
            "Preferred method": "—",
            "Margin": "—",
            "Union gain": "—",
        }])
    clear = clear.sort_values(["Dominance margin (pp)", "Multi-model gain (pp)"], ascending=[False, False]).head(max_rows)
    rows = []
    for _, r in clear.iterrows():
        p = pd.to_numeric(pd.Series([r.get("Period (d)")]), errors="coerce").iloc[0]
        d = pd.to_numeric(pd.Series([r.get("Duration (h)")]), errors="coerce").iloc[0]
        dep = pd.to_numeric(pd.Series([r.get("Depth (ppm)")]), errors="coerce").iloc[0]
        morphology = " · ".join([
            f"{dep:.0f} ppm" if np.isfinite(dep) else "? ppm",
            f"{d:g} h" if np.isfinite(d) else "? h",
            f"{p:g} d" if np.isfinite(p) else "? d",
        ])
        rows.append({
            "Transit morphology": morphology,
            "Preferred method": r.get("Meaningful preference", "—"),
            "Margin": f"{float(r.get('Dominance margin (pp)', np.nan)):.1f} pp" if pd.notna(r.get("Dominance margin (pp)")) else "—",
            "Union gain": f"{float(r.get('Multi-model gain (pp)', np.nan)):.1f} pp" if pd.notna(r.get("Multi-model gain (pp)")) else "—",
        })
    return pd.DataFrame(rows)


def _compact_interaction_rows(table: pd.DataFrame, max_rows: int = 4) -> pd.DataFrame:
    """Most clearly separated background × transit combinations for the Overview page."""
    if table.empty or "Background stratum" not in table.columns:
        return pd.DataFrame()
    strata = table["Background stratum"].fillna("Unknown").astype(str)
    if strata.str.contains("unknown|unavailable", case=False, regex=True).all():
        return pd.DataFrame()
    clear = table[table["Meaningful preference"] != "No meaningful preference"].copy()
    if clear.empty:
        return pd.DataFrame([{
            "Background × transit": "No clear interaction preference",
            "Preferred method": "—",
            "Margin": "—",
            "Union gain": "—",
        }])
    clear = clear.sort_values(["Dominance margin (pp)", "Multi-model gain (pp)"], ascending=[False, False]).head(max_rows)
    rows = []
    for _, r in clear.iterrows():
        p = pd.to_numeric(pd.Series([r.get("Period (d)")]), errors="coerce").iloc[0]
        d = pd.to_numeric(pd.Series([r.get("Duration (h)")]), errors="coerce").iloc[0]
        dep = pd.to_numeric(pd.Series([r.get("Depth (ppm)")]), errors="coerce").iloc[0]
        label = f"{r.get('Background stratum', '—')} · {dep:.0f} ppm · {d:g} h · {p:g} d"
        rows.append({
            "Background × transit": label,
            "Preferred method": r.get("Meaningful preference", "—"),
            "Margin": f"{float(r.get('Dominance margin (pp)', np.nan)):.1f} pp" if pd.notna(r.get("Dominance margin (pp)")) else "—",
            "Union gain": f"{float(r.get('Multi-model gain (pp)', np.nan)):.1f} pp" if pd.notna(r.get("Multi-model gain (pp)")) else "—",
        })
    return pd.DataFrame(rows)


def _compact_gain_rows(gain_table: pd.DataFrame, best_label: str, max_rows: int = 4) -> pd.DataFrame:
    """Show only the branches that recover the most cases missed by the best fixed pipeline."""
    if gain_table.empty:
        return pd.DataFrame()
    g = gain_table.copy()
    if best_label:
        g = g[g["Pipeline"] != best_label]
    g = g.sort_values(["Cases recovered when best fixed misses", "Pairwise uplift over best fixed (pp)"], ascending=[False, False]).head(max_rows)
    out = pd.DataFrame({
        "Added branch": g["Pipeline"],
        "Rescued cases": g["Cases recovered when best fixed misses"].astype(int),
        "Gain coverage": g["Coverage of multi-model gain cases (%)"].map(lambda x: f"{x:.0f}%"),
        "Pairwise uplift": g["Pairwise uplift over best fixed (pp)"].map(lambda x: f"+{x:.1f} pp"),
    })
    return out.reset_index(drop=True)


def _render_compact_summary_table(df: pd.DataFrame) -> None:
    """Render a small light table with no horizontal or internal scrolling."""
    if df.empty:
        return
    show = df.copy()
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].map(lambda x: "—" if pd.isna(x) else f"{x:.1f}")
    html = show.to_html(index=False, border=0, classes="compact-summary-table", escape=True)
    st.markdown(html, unsafe_allow_html=True)


def _round_analysis_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if any(token in c for token in ("(%)", "(pp)")):
            out[c] = pd.to_numeric(out[c], errors="ignore")
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].round(1)
        elif c in {"Period (d)", "Duration (h)", "Depth (ppm)"}:
            vals = pd.to_numeric(out[c], errors="coerce")
            out[c] = vals.round(3 if c != "Depth (ppm)" else 0)
    return out


def _light_analysis_style(df: pd.DataFrame):
    """Keep analysis grids in the dashboard's light blue/white presentation palette."""
    return (
        df.style
        .set_properties(**{
            "background-color": "#ffffff",
            "color": "#21324d",
            "border-color": "#dbe7f5",
        })
        .set_table_styles([
            {"selector": "th", "props": [("background-color", "#eaf3ff"), ("color", "#21324d"), ("font-weight", "650")]},
            {"selector": "td", "props": [("background-color", "#ffffff"), ("color", "#21324d")]},
        ])
    )


def pct(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{100*x:.1f}%"


def expected_star_count(run_dir: Path, injections: pd.DataFrame) -> int:
    """Infer the intended benchmark size for presentation-status chips."""
    config = read_json_safe(str(run_dir / "benchmark_config.json"))

    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if str(key).lower() in {
                    "n_stars", "num_stars", "star_count", "target_count",
                    "n_targets", "num_targets", "final_size",
                }:
                    try:
                        value_i = int(value)
                        if value_i > 0:
                            return value_i
                    except Exception:
                        pass
                found = walk(value)
                if found:
                    return found
        elif isinstance(obj, list):
            for value in obj:
                found = walk(value)
                if found:
                    return found
        return None

    from_config = walk(config)
    if from_config:
        return int(from_config)

    match = re.search(r"(\d+)[_-]?star", run_dir.name, flags=re.I)
    if match:
        return int(match.group(1))

    return int(injections["target_id"].nunique()) if "target_id" in injections.columns else 0


def injection_grid_status(injections: pd.DataFrame, expected_stars: int) -> tuple[int, str]:
    """Return expected total cases and a human-readable hover description of the grid."""
    dims = [
        ("injected_period_days", "periods", "d", 1.0),
        ("injected_duration_hours", "durations", "h", 1.0),
        ("injected_depth", "depths", "ppm", 1e6),
    ]
    counts = []
    details = []
    for col, label, unit, scale in dims:
        if col not in injections.columns:
            continue
        vals = pd.to_numeric(injections[col], errors="coerce").dropna().unique()
        vals = np.sort(vals.astype(float))
        if len(vals) == 0:
            continue
        counts.append(len(vals))
        shown = vals * scale
        if len(shown) <= 10:
            values_text = ", ".join(f"{v:.4g}" for v in shown)
            details.append(f"{len(vals)} {label}: {values_text} {unit}")
        else:
            details.append(f"{len(vals)} {label}: {shown.min():.4g}–{shown.max():.4g} {unit}")

    grid_cases_per_star = int(np.prod(counts)) if counts else 0

    # A saved injection table can include an additional design dimension (for example,
    # multiple epochs/phases) that is not represented by the three headline columns above.
    # Infer the intended per-star case count from the fullest currently loaded star as a
    # safeguard, then never let the displayed progress exceed 100%.
    observed_cases_per_star = 0
    if not injections.empty and "target_id" in injections.columns:
        per_star_counts = injections.groupby(injections["target_id"].map(normalize_target_id)).size()
        if not per_star_counts.empty:
            observed_cases_per_star = int(per_star_counts.max())

    cases_per_star = max(grid_cases_per_star, observed_cases_per_star)
    expected_total = int(expected_stars * cases_per_star) if expected_stars and cases_per_star else len(injections)
    grid_text = " · ".join(details) if details else "Injection-grid details are not present in the loaded table."
    if cases_per_star:
        if observed_cases_per_star > grid_cases_per_star:
            grid_text = f"{cases_per_star} planned cases per star (inferred from the fullest loaded star) · " + grid_text
        else:
            grid_text = f"{cases_per_star} planned injections per star · " + grid_text
    return expected_total, grid_text


def star_pipeline_recovery_table(df: pd.DataFrame, pipelines: Iterable[str], suffix: str) -> pd.DataFrame:
    """Per-star recovery fractions used by the strong star-level uniqueness view."""
    if df.empty or "target_id" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["_target_norm"] = work["target_id"].map(normalize_target_id)
    rows = []
    for target, group in work.groupby("_target_norm"):
        row = {"target_id": target, "n_cases": len(group)}
        for pipeline in pipelines:
            col = f"{pipeline}_{suffix}"
            if col in group.columns:
                row[pipeline] = float(group[col].fillna(False).astype(bool).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def dominant_star_rows(
    df: pd.DataFrame,
    pipelines: Iterable[str],
    suffix: str,
    winner_min: float = 0.90,
    other_max: float = 0.60,
) -> pd.DataFrame:
    """Stars where one pipeline is very high while every alternative is substantially lower."""
    rates = star_pipeline_recovery_table(df, pipelines, suffix)
    if rates.empty:
        return rates
    rows = []
    for _, star in rates.iterrows():
        available = [p for p in pipelines if p in rates.columns and pd.notna(star.get(p, np.nan))]
        if len(available) < 2:
            continue
        for winner in available:
            winner_rate = float(star[winner])
            other_rates = {p: float(star[p]) for p in available if p != winner}
            strongest_other = max(other_rates.values()) if other_rates else np.nan
            if winner_rate >= winner_min and np.isfinite(strongest_other) and strongest_other <= other_max:
                strongest_name = max(other_rates, key=other_rates.get)
                rows.append({
                    "target_id": star["target_id"],
                    "n_cases": int(star["n_cases"]),
                    "pipeline": winner,
                    "winner_rate": winner_rate,
                    "strongest_other": strongest_name,
                    "strongest_other_rate": strongest_other,
                    "gap": winner_rate - strongest_other,
                })
    return pd.DataFrame(rows)


def infer_top_k(run_dir: Path, df: pd.DataFrame, pipelines: Iterable[str], fallback: int = 10) -> int:
    """Read the configured candidate-retention K when possible; otherwise use the verified project default."""
    config = read_json_safe(str(run_dir / "benchmark_config.json"))

    candidate_keys = {
        "top_k", "topk", "top_k_candidates", "n_top_candidates",
        "num_top_candidates", "candidate_top_k", "candidate_k",
    }

    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if str(key).lower() in candidate_keys:
                    try:
                        k = int(value)
                        if k > 0:
                            return k
                    except Exception:
                        pass
                found = walk(value)
                if found:
                    return found
        elif isinstance(obj, list):
            for value in obj:
                found = walk(value)
                if found:
                    return found
        return None

    configured = walk(config)
    if configured:
        return int(configured)

    # Rank columns can confirm a lower bound but not the configured ceiling.
    # Keep the project-level fallback when configuration metadata is absent.
    return int(fallback)



def _nested_config_value(obj, candidate_keys: set[str]):
    """Find a scalar config value by key, recursively, without depending on one config layout."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in candidate_keys and not isinstance(value, (dict, list)):
                return value
            found = _nested_config_value(value, candidate_keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _nested_config_value(value, candidate_keys)
            if found is not None:
                return found
    return None


def infer_period_match_tolerance(run_dir: Path, fallback: float = 0.02) -> float:
    config = read_json_safe(str(run_dir / "benchmark_config.json"))
    value = _nested_config_value(
        config,
        {"period_match_tolerance_fraction", "period_tolerance_fraction", "period_match_tolerance", "period_tolerance"},
    )
    try:
        value = float(value)
        if 0 < value < 1:
            return value
    except Exception:
        pass
    return float(fallback)


def benchmark_feature_frame(
    characterization: pd.DataFrame,
    star_summaries: pd.DataFrame,
    loaded_targets: Iterable[str],
) -> pd.DataFrame:
    """One numeric feature row per loaded benchmark star, combining characterization and saved summaries."""
    loaded = {normalize_target_id(x) for x in loaded_targets}
    frames = []
    for source in (characterization, star_summaries):
        if source is None or source.empty or "target_id" not in source.columns:
            continue
        frame = source.copy()
        frame["target_id"] = frame["target_id"].map(normalize_target_id)
        frame = frame[frame["target_id"].isin(loaded)].drop_duplicates("target_id")
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        overlap = [c for c in frame.columns if c != "target_id" and c in out.columns]
        frame = frame.drop(columns=overlap, errors="ignore")
        out = out.merge(frame, on="target_id", how="outer")
    return out


def dominance_feature_associations(
    characterization: pd.DataFrame,
    star_summaries: pd.DataFrame,
    loaded_targets: Iterable[str],
    unique_targets: Iterable[str],
    top_n: int = 6,
) -> pd.DataFrame:
    """Descriptive feature contrasts for strongly unique stars versus the other loaded benchmark stars."""
    feature_df = benchmark_feature_frame(characterization, star_summaries, loaded_targets)
    unique = {normalize_target_id(x) for x in unique_targets}
    if feature_df.empty or not unique:
        return pd.DataFrame()

    labels = dict(FEATURE_LABELS)
    labels.update({
        "robust_flux_scatter_ppm": "Robust flux scatter (ppm)",
        "lag_one_flux_acf": "Flux ACF lag 1",
        "finite_fraction": "Finite cadence fraction",
        "finite_observation_fraction": "Finite observation fraction",
        "baseline_days": "Observed baseline (d)",
        "n_finite_observations": "Finite cadences",
        "gap_fraction": "Gap fraction",
        "flux_std": "Flux σ",
        "flux_robust_scale": "Robust flux scale",
        "acf_lag_1": "ACF lag 1",
        "acf_timescale_days": "ACF timescale (d)",
        "variance_drift": "Variance drift",
        "spectral_strength": "Spectral strength",
        "spectral_entropy": "Spectral entropy",
        "flux_skewness": "Flux skewness",
        "flux_excess_kurtosis": "Excess kurtosis",
    })

    rows = []
    is_unique = feature_df["target_id"].isin(unique)
    for col, label in labels.items():
        if col not in feature_df.columns:
            continue
        vals = pd.to_numeric(feature_df[col], errors="coerce")
        a = vals[is_unique].dropna()
        b = vals[~is_unique].dropna()
        if len(a) < 1 or len(b) < 2:
            continue
        med_a, med_b = float(a.median()), float(b.median())
        pooled = vals.dropna()
        q25, q75 = pooled.quantile([0.25, 0.75]) if len(pooled) else (np.nan, np.nan)
        scale = float(q75 - q25) if np.isfinite(q75) and np.isfinite(q25) else np.nan
        if not np.isfinite(scale) or scale <= 0:
            scale = float(pooled.std(ddof=0)) if len(pooled) else np.nan
        effect = (med_a - med_b) / scale if np.isfinite(scale) and scale > 0 else np.nan
        rows.append({
            "Property": label,
            "Strongly-unique median": med_a,
            "Other loaded stars median": med_b,
            "Direction": "Higher" if med_a > med_b else ("Lower" if med_a < med_b else "Similar"),
            "Robust separation": effect,
            "Unique n": int(a.size),
            "Other n": int(b.size),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["_abs"] = out["Robust separation"].abs()
    return out.sort_values("_abs", ascending=False).head(top_n).drop(columns="_abs")


def star_model_evidence_tables(
    target: str,
    pipeline: str,
    all_star_cases: pd.DataFrame,
    characterization: pd.DataFrame,
    star_summaries: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compact pre/post diagnostics and saved model parameters for a selected strong-uniqueness star."""
    target = normalize_target_id(target)
    branch = PIPELINE_META.get(pipeline, (pipeline.split("_")[0], ""))[0].lower()

    source_rows = []
    for source in (characterization, star_summaries):
        if source is None or source.empty or "target_id" not in source.columns:
            continue
        f = source.copy()
        f["target_id"] = f["target_id"].map(normalize_target_id)
        hit = f[f["target_id"] == target]
        if not hit.empty:
            source_rows.append(hit.iloc[0])
    merged = {}
    for row in source_rows:
        for k, v in row.items():
            if k not in merged or pd.isna(merged[k]):
                merged[k] = v

    pre_candidates = [
        ("robust_flux_scatter_ppm", "Raw robust scatter", "ppm"),
        ("flux_std", "Raw flux σ", "relative flux"),
        ("flux_robust_scale", "Raw robust scale", "relative flux"),
        ("lag_one_flux_acf", "Raw ACF(1)", ""),
        ("acf_lag_1", "Raw ACF(1)", ""),
        ("gap_fraction", "Gap fraction", ""),
        ("finite_observation_fraction", "Finite observation fraction", ""),
    ]
    diag = []
    seen_labels = set()
    for col, label, unit in pre_candidates:
        if label in seen_labels or col not in merged:
            continue
        v = _num(merged.get(col))
        if np.isfinite(v):
            diag.append({"Stage": "Pre-fit / raw", "Metric": label, "Value": v, "Unit": unit})
            seen_labels.add(label)

    post_candidates = [
        (f"{branch}_residual_std", "Residual σ", "relative flux"),
        (f"{branch}_residual_acf1", "Residual ACF(1)", ""),
        (f"{branch}_depth_retention_fraction", "Depth retention", "fraction"),
        (f"{branch}_snr_retention_fraction", "SNR retention", "fraction"),
        (f"{branch}_local_snr", "Local SNR", ""),
        (f"{branch}_residual_depth", "Residual depth", "relative flux"),
    ]
    for col, label, unit in post_candidates:
        if col in all_star_cases.columns:
            vals = pd.to_numeric(all_star_cases[col], errors="coerce").dropna()
            if not vals.empty:
                diag.append({"Stage": "Post-fit / residual", "Metric": label, "Value": float(vals.median()), "Unit": unit})

    params = []
    param_keywords = (
        "order", "length_scale", "process_variance", "measurement_variance", "noise_variance",
        "training_points", "log_likelihood", "log_marginal", "converged", "runtime_seconds", "source",
    )
    # Saved star-level parameters first.
    for key, value in merged.items():
        kl = str(key).lower()
        if branch not in kl or not any(k in kl for k in param_keywords):
            continue
        if pd.isna(value):
            continue
        params.append({"Parameter / fit field": str(key), "Saved value": value})
    # Then injection-level parameters, summarized by median where numeric.
    for col in all_star_cases.columns:
        cl = str(col).lower()
        if branch not in cl or not any(k in cl for k in param_keywords):
            continue
        if any(r["Parameter / fit field"] == str(col) for r in params):
            continue
        vals = pd.to_numeric(all_star_cases[col], errors="coerce")
        if vals.notna().any():
            params.append({"Parameter / fit field": str(col), "Saved value": f"median {vals.median():.5g}"})
        else:
            nonnull = all_star_cases[col].dropna().astype(str)
            if not nonnull.empty:
                params.append({"Parameter / fit field": str(col), "Saved value": nonnull.iloc[0]})
    return pd.DataFrame(diag), pd.DataFrame(params[:10])


def infer_metric_suffix(df: pd.DataFrame, pipelines: Iterable[str]) -> tuple[str, str, bool]:
    if any(f"{p}_fap01_harmonic_recovered" in df.columns for p in pipelines):
        return "fap01_harmonic_recovered", "Recovery @ calibrated 1% FAP", True
    return "harmonic_rank1_matched", "Harmonic-aware rank-1 period recovery (uncalibrated)", False


# =============================================================================
# Light-curve lookup
# =============================================================================

def candidate_light_curve_paths(repo_root: Path, run_dir: Path, target_id, quarter) -> list[Path]:
    key = star_key(target_id, quarter)
    candidates = [
        run_dir / "stars" / key / "regularized_light_curve.parquet",
        repo_root / "outputs" / "cache" / f"{key}_pdcsap.parquet",
        repo_root / "outputs" / "light_curve_cache" / f"{key}_pdcsap.parquet",
    ]

    # Bounded recursive fallbacks.
    for root in (repo_root / "outputs", repo_root / "data"):
        if root.exists():
            candidates.extend(root.glob(f"**/{key}_pdcsap.parquet"))
            candidates.extend(root.glob(f"**/{key}*/regularized_light_curve.parquet"))

    dedup = []
    seen = set()
    for p in candidates:
        s = str(p)
        if s not in seen:
            seen.add(s)
            dedup.append(p)
    return dedup


@st.cache_data(show_spinner=False)
def load_light_curve(repo_root_text: str, run_dir_text: str, target_id: str, quarter: int):
    repo_root = Path(repo_root_text)
    run_dir = Path(run_dir_text)
    for p in candidate_light_curve_paths(repo_root, run_dir, target_id, quarter):
        if not p.exists():
            continue
        try:
            return pd.read_parquet(p), str(p)
        except Exception:
            pass
    return pd.DataFrame(), ""


def choose_time_flux(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if df.empty:
        return None, None

    time_candidates = ["time", "time_days", "bkjd"]
    flux_candidates = [
        "normalized_flux",
        "pdcsap_flux",
        "flux",
        "PDCSAP_FLUX",
        "sap_flux",
    ]

    time_col = next((c for c in time_candidates if c in df.columns), None)
    flux_col = next((c for c in flux_candidates if c in df.columns), None)

    if time_col is None:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        time_col = numeric[0] if numeric else None

    if flux_col is None:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != time_col]
        flux_col = numeric[0] if numeric else None

    return time_col, flux_col


def light_curve_plot(df: pd.DataFrame, title: str):
    time_col, flux_col = choose_time_flux(df)
    if time_col is None or flux_col is None:
        return None
    plot_df = df[[time_col, flux_col]].copy()
    plot_df[time_col] = pd.to_numeric(plot_df[time_col], errors="coerce")
    plot_df[flux_col] = pd.to_numeric(plot_df[flux_col], errors="coerce")
    plot_df = plot_df.dropna()
    fig = px.scatter(
        plot_df,
        x=time_col,
        y=flux_col,
        title=title,
        labels={time_col: "Time (days)", flux_col: "Normalized / PDCSAP flux"},
        opacity=0.55,
    )
    fig.update_traces(marker={"size": 3})
    yvals = plot_df[flux_col].to_numpy(dtype=float)
    if len(yvals) >= 20:
        lo, hi = np.nanpercentile(yvals, [0.5, 99.5])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pad = 0.08 * (hi - lo)
            fig.update_yaxes(range=[lo - pad, hi + pad])
    fig.update_layout(height=275, margin=dict(l=12, r=12, t=38, b=24))
    return fig


def unique_case_mask(df: pd.DataFrame, pipeline: str, suffix: str, pipelines: Iterable[str]) -> pd.Series:
    """Rows recovered by `pipeline` under the selected criterion and by no other loaded pipeline."""
    col = f"{pipeline}_{suffix}"
    if df.empty or col not in df.columns:
        return pd.Series(False, index=df.index)
    current = df[col].fillna(False).astype(bool)
    others = pd.Series(False, index=df.index)
    for other in pipelines:
        if other == pipeline:
            continue
        other_col = f"{other}_{suffix}"
        if other_col in df.columns:
            others |= df[other_col].fillna(False).astype(bool)
    return current & ~others


def _num(value):
    """Coerce one dashboard value to float without raising."""
    try:
        out = float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])
    except Exception:
        return np.nan
    return out


def _rank_text(value) -> str:
    v = _num(value)
    return str(int(v)) if np.isfinite(v) and v >= 1 else "—"


def selected_period_cell(row: pd.Series, pipeline: str, metric_suffix: str) -> str:
    """Compact, presentation-friendly description of what a pipeline selected for one injection."""
    recovered_p = _num(row.get(f"{pipeline}_recovered_period_days", np.nan))
    injected_p = _num(row.get("injected_period_days", np.nan))
    hit_col = f"{pipeline}_{metric_suffix}"
    recovered = bool(row.get(hit_col, False)) if pd.notna(row.get(hit_col, np.nan)) else False

    if not np.isfinite(recovered_p):
        return "— not returned"

    relation = ""
    exact = row.get(f"{pipeline}_exact_rank1_matched", np.nan)
    harmonic = row.get(f"{pipeline}_harmonic_rank1_matched", np.nan)
    if pd.notna(exact) and bool(exact):
        relation = "P"
    elif pd.notna(harmonic) and bool(harmonic) and np.isfinite(injected_p) and injected_p > 0:
        ratios = {"P/2": 0.5, "2P": 2.0}
        relation = min(ratios, key=lambda k: abs(recovered_p / injected_p - ratios[k]))
    elif pd.notna(harmonic) and bool(harmonic):
        relation = "harmonic"

    status = "✓" if recovered else "✗"
    rel = f" {relation}" if relation else ""
    return f"{recovered_p:.5g} d {status}{rel}"


def recovery_status_cell(row: pd.Series, pipeline: str, metric_suffix: str) -> str:
    hit_col = f"{pipeline}_{metric_suffix}"
    recovered = bool(row.get(hit_col, False)) if pd.notna(row.get(hit_col, np.nan)) else False
    if recovered:
        exact = row.get(f"{pipeline}_exact_rank1_matched", np.nan)
        if pd.notna(exact) and bool(exact):
            return "✓ Injected P"
        return "✓ P/2 or 2P"

    # Useful distinction when the period matched but the calibrated score did not clear threshold.
    harmonic = row.get(f"{pipeline}_harmonic_rank1_matched", np.nan)
    detected = row.get(f"{pipeline}_fap01_detected", np.nan)
    if metric_suffix.startswith("fap01") and pd.notna(harmonic) and bool(harmonic) and pd.notna(detected) and not bool(detected):
        return "✗ period match; below FAP"
    return "✗ Miss"


def topk_rank_cell(row: pd.Series, pipeline: str) -> str:
    return (
        f"P:{_rank_text(row.get(f'{pipeline}_exact_rank_topk', np.nan))} | "
        f"P/2:{_rank_text(row.get(f'{pipeline}_half_period_rank_topk', np.nan))} | "
        f"2P:{_rank_text(row.get(f'{pipeline}_double_period_rank_topk', np.nan))}"
    )


def full_detail_cell(row: pd.Series, pipeline: str, metric_suffix: str) -> str:
    period = selected_period_cell(row, pipeline, metric_suffix)
    score = _num(row.get(f"{pipeline}_score", np.nan))
    thr = _num(row.get(f"{pipeline}_fap01_threshold", np.nan))
    runtime = _num(row.get(f"{pipeline}_runtime_seconds", np.nan))
    bits = [period]
    if np.isfinite(score):
        bits.append(f"score {score:.4g}")
    if np.isfinite(thr):
        bits.append(f"thr {thr:.4g}")
    if np.isfinite(runtime):
        bits.append(f"{runtime:.3g}s")
    return " · ".join(bits)


def period_match_label(row: pd.Series, pipeline: str) -> str:
    exact = f"{pipeline}_exact_rank1_matched"
    harmonic = f"{pipeline}_harmonic_rank1_matched"
    if exact in row.index and pd.notna(row[exact]) and bool(row[exact]):
        return "Injected P · top-1"
    if harmonic in row.index and pd.notna(row[harmonic]) and bool(row[harmonic]):
        return "P/2 or 2P · top-1"

    top_cols = [
        f"{pipeline}_exact_rank_topk",
        f"{pipeline}_half_period_rank_topk",
        f"{pipeline}_double_period_rank_topk",
    ]
    if any(c in row.index and pd.notna(row[c]) and float(row[c]) >= 1 for c in top_cols):
        return f"P/2, P, 2P · top-{TOP_K_DISPLAY}"
    return "Miss"


def phase_folded_plot(df: pd.DataFrame, period_days: float, title: str):
    """Fold the saved source light curve at a period; this does not recreate the synthetic injection."""
    x, time_col, flux_col = _lc_numeric(df)
    if x.empty or not np.isfinite(period_days) or period_days <= 0:
        return None
    t = x[time_col].to_numpy(dtype=float)
    epoch = float(np.nanmin(t))
    phase = ((t - epoch + 0.5 * period_days) % period_days) / period_days - 0.5
    plot_df = pd.DataFrame({"phase": phase, "relative_flux": x[flux_col].to_numpy(dtype=float)})
    fig = px.scatter(
        plot_df,
        x="phase",
        y="relative_flux",
        title=title,
        labels={"phase": "Orbital phase", "relative_flux": "Relative flux"},
        opacity=0.45,
    )
    fig.update_traces(marker={"size": 3})
    fig.update_layout(height=275, margin=dict(l=12, r=12, t=38, b=24))
    return fig


def branch_series_column(df: pd.DataFrame, branch: str, kind: str) -> str | None:
    """Find an already-saved branch background/residual series without inventing one."""
    branch = branch.lower()
    if kind == "background":
        candidates = [
            f"{branch}_background", f"{branch}_background_flux", f"{branch}_trend",
            f"{branch}_mean", f"{branch}_fitted", f"{branch}_model",
        ]
    else:
        candidates = [
            f"{branch}_residual", f"{branch}_residual_flux", f"{branch}_innovations",
            f"{branch}_innovation", f"{branch}_detrended_flux",
        ]
    return next((c for c in candidates if c in df.columns), None)


def saved_series_plot(df: pd.DataFrame, value_col: str, title: str):
    time_col, _ = choose_time_flux(df)
    if time_col is None or value_col not in df.columns:
        return None
    plot_df = df[[time_col, value_col]].copy()
    plot_df[time_col] = pd.to_numeric(plot_df[time_col], errors="coerce")
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna()
    if plot_df.empty:
        return None
    fig = px.scatter(
        plot_df, x=time_col, y=value_col, title=title,
        labels={time_col: "Time (days)", value_col: value_col.replace("_", " ").title()},
        opacity=0.55,
    )
    fig.update_traces(marker={"size": 3})
    fig.update_layout(height=275, margin=dict(l=12, r=12, t=38, b=24))
    return fig


# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.markdown("## 🪐 Transit Search")
st.sidebar.caption("Kepler multi-model proof of concept")

run_root = DEFAULT_RUN_ROOT
runs = discover_runs(run_root)

st.sidebar.markdown("### Experiment")
if runs:
    run_names = [p.name for p in runs]
    preferred = next((i for i, n in enumerate(run_names) if n == "clean_q5_50star"), 0)
    selected_name = st.sidebar.selectbox(
        "Choose benchmark",
        run_names,
        index=preferred,
        format_func=compact_experiment_label,
        help="Select which saved or currently running benchmark to explore.",
    )
    RUN_DIR = run_root / selected_name
    st.sidebar.caption(experiment_label(selected_name))
else:
    RUN_DIR = run_root
    st.sidebar.info("No benchmark runs found yet.")

include_partial = True

NAV_OPTIONS = [
    "🏠 Overview",
    "📐 Recovery & scoring",
    "🧩 Unique Recovery",
    "⭐ Stars & statistics",
    "🔎 Injection explorer",
    "🎯 FAP calibration",
    "🧭 POC roadmap",
]

def navigate_to(label: str):
    st.session_state["page_nav"] = label

st.sidebar.markdown("### Explore")
page_label = st.sidebar.radio(
    "Explore",
    NAV_OPTIONS,
    label_visibility="collapsed",
    key="page_nav",
)

PAGE_MAP = {
    "🏠 Overview": "1",
    "📐 Recovery & scoring": "7",
    "🧩 Unique Recovery": "6",
    "⭐ Stars & statistics": "2",
    "🔎 Injection explorer": "3",
    "🎯 FAP calibration": "4",
    "🧭 POC roadmap": "5",
}
page = PAGE_MAP[page_label]

st.sidebar.divider()
if st.sidebar.button("↻ Refresh results", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

with st.sidebar.expander("Technical settings", expanded=False):
    include_partial = st.checkbox("Read live checkpoints", value=True)
    min_dominance_pp = float(st.number_input(
        "Meaningful preference margin (pp)",
        min_value=0.0, max_value=100.0, value=MIN_DOMINANCE_PP_DEFAULT, step=1.0,
        help="A numerical winner is called meaningful only when it exceeds the second-best pipeline by at least this many percentage points.",
    ))
    min_complementarity_pp = float(st.number_input(
        "Meaningful multi-model gain (pp)",
        min_value=0.0, max_value=100.0, value=MIN_COMPLEMENTARITY_PP_DEFAULT, step=1.0,
        help="Used only for the descriptive complementary/disagreement classification.",
    ))
    weak_transit_mode = st.selectbox(
        "Weak-transit subset",
        ["Shallowest depth only", "Bottom two depth levels", "Custom depth threshold"],
        index=0,
    )
    custom_weak_depth_ppm = float(st.number_input(
        "Custom weak-depth maximum (ppm)", min_value=1.0, value=500.0, step=50.0,
        disabled=weak_transit_mode != "Custom depth threshold",
    ))
    st.caption(f"Experiment: {experiment_label(RUN_DIR.name)}")
    st.caption(f"Folder ID: {RUN_DIR.name}")
    st.caption(f"Repository: {REPO_ROOT.name}")


# =============================================================================
# Load data
# =============================================================================

injections = collect_injections(str(RUN_DIR), include_partial=include_partial)
star_summaries = collect_star_summaries(str(RUN_DIR))
thresholds, null_trials = collect_calibration(str(RUN_DIR))
characterization = collect_characterization(str(RUN_DIR), str(REPO_ROOT))

injections = attach_regime_metadata(
    injections,
    repo_root=REPO_ROOT,
    run_dir=RUN_DIR,
    characterization=characterization,
)

pipelines = available_pipelines(injections)
injections = add_calibrated_columns(injections, thresholds, pipelines)
metric_suffix, metric_label, calibration_available = infer_metric_suffix(injections, pipelines)
TOP_K_DISPLAY = infer_top_k(RUN_DIR, injections, pipelines, fallback=TOP_K_DISPLAY)

if injections.empty and not page.startswith("5"):
    st.error(
        "No injection data found for this run. Select a run that has either "
        "`metrics/multistar_challenger_injections.csv` or per-star `injections.csv` checkpoints."
    )
    st.stop()


# =============================================================================
# Common header
# =============================================================================

def header(title: str, subtitle: str):
    st.title(title)
    st.markdown(f'<div class="small-note">{subtitle}</div>', unsafe_allow_html=True)





def presentation_plot(fig, height: int | None = None):
    """Compact Plotly styling for screen-share: less chrome, tighter framing, readable labels."""
    if fig is None:
        return None

    current_height = getattr(fig.layout, "height", None)
    if height is not None:
        final_height = int(height)
    elif current_height is None:
        final_height = 290
    else:
        # Keep charts compact even when individual constructors requested very tall figures.
        final_height = max(240, min(340, int(current_height)))

    fig.update_layout(
        height=final_height,
        autosize=True,
        font=dict(size=12, color="#344258"),
        title=dict(font=dict(size=15, color="#172033"), x=0.01, xanchor="left"),
        legend=dict(
            font=dict(size=11, color="#44546a"),
            bgcolor="rgba(0,0,0,0)",
            orientation="v",
        ),
        margin=dict(l=18, r=18, t=38, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        hovermode="closest",
    )
    fig.update_xaxes(
        title_font=dict(size=12, color="#526177"),
        tickfont=dict(size=10, color="#5e6e83"),
        automargin=True,
        gridcolor="#edf1f5",
        zerolinecolor="#d9e1ea",
    )
    fig.update_yaxes(
        title_font=dict(size=12, color="#526177"),
        tickfont=dict(size=10, color="#5e6e83"),
        automargin=True,
        gridcolor="#edf1f5",
        zerolinecolor="#d9e1ea",
    )
    return fig



def _lc_numeric(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, str | None]:
    time_col, flux_col = choose_time_flux(df)
    if time_col is None or flux_col is None:
        return pd.DataFrame(), None, None
    x = df[[time_col, flux_col]].copy()
    x[time_col] = pd.to_numeric(x[time_col], errors="coerce")
    x[flux_col] = pd.to_numeric(x[flux_col], errors="coerce")
    x = x.dropna().sort_values(time_col)
    if x.empty:
        return x, time_col, flux_col
    f = x[flux_col].to_numpy(dtype=float)
    med = np.nanmedian(f)
    if np.isfinite(med) and med != 0 and abs(med) > 100:
        x["_relative_flux"] = f / med - 1.0
    else:
        x["_relative_flux"] = f - med
    return x, time_col, "_relative_flux"


def derive_visual_stats(df: pd.DataFrame) -> dict:
    x, time_col, flux_col = _lc_numeric(df)
    if x.empty:
        return {}
    f = x[flux_col].to_numpy(dtype=float)
    t = x[time_col].to_numpy(dtype=float)
    out = {}
    out["Scatter"] = float(np.nanstd(f))
    med = float(np.nanmedian(f))
    mad = float(np.nanmedian(np.abs(f - med)))
    out["Robust scatter"] = 1.4826 * mad

    centered = f - np.nanmean(f)
    sigma = np.nanstd(centered)
    if np.isfinite(sigma) and sigma > 0:
        z = centered / sigma
        out["Skewness"] = float(np.nanmean(z ** 3))
        out["Excess kurtosis"] = float(np.nanmean(z ** 4) - 3.0)

    if len(f) > 2 and np.nanstd(f[:-1]) > 0 and np.nanstd(f[1:]) > 0:
        out["ACF(1)"] = float(np.corrcoef(f[:-1], f[1:])[0, 1])

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt):
        cadence = np.nanmedian(dt)
        if np.isfinite(cadence) and cadence > 0:
            missing = np.maximum(np.rint(dt / cadence).astype(int) - 1, 0).sum()
            out["Gap fraction"] = float(missing / max(len(f) + int(missing), 1))

    if len(f) >= 24:
        chunks = np.array_split(f, min(12, max(4, len(f) // 100)))
        vars_ = np.array([np.nanvar(c) for c in chunks if len(c) >= 3], dtype=float)
        vars_ = vars_[np.isfinite(vars_)]
        if len(vars_) and np.nanmedian(vars_) > 0:
            out["Variance drift"] = float(
                (np.nanmax(vars_) - np.nanmin(vars_)) / np.nanmedian(vars_)
            )
    return out


def acf_figure(df: pd.DataFrame, max_lags: int = 40):
    x, _, flux_col = _lc_numeric(df)
    if x.empty:
        return None
    f = x[flux_col].to_numpy(dtype=float)
    f = f - np.nanmean(f)
    denom = np.nansum(f * f)
    if not np.isfinite(denom) or denom <= 0:
        return None
    max_lags = min(max_lags, len(f) - 2)
    vals = []
    for lag in range(1, max_lags + 1):
        vals.append(np.nansum(f[:-lag] * f[lag:]) / denom)
    fig = px.bar(
        x=list(range(1, max_lags + 1)),
        y=vals,
        labels={"x": "Lag", "y": "Correlation"},
        title="Memory in the signal",
    )
    fig.update_layout(height=280, margin=dict(l=12, r=12, t=38, b=24))
    return fig


def distribution_figure(df: pd.DataFrame):
    x, _, flux_col = _lc_numeric(df)
    if x.empty:
        return None
    fig = px.histogram(
        x=x[flux_col],
        nbins=48,
        labels={"x": "Relative flux", "y": "Count"},
        title="Shape of the flux values",
    )
    fig.update_layout(height=280, margin=dict(l=12, r=12, t=38, b=24), showlegend=False)
    return fig


def rolling_variance_figure(df: pd.DataFrame):
    x, time_col, flux_col = _lc_numeric(df)
    if x.empty:
        return None
    n = len(x)
    window = max(25, min(180, n // 20))
    rv = x[flux_col].rolling(window=window, center=True, min_periods=max(8, window // 3)).var()
    plot = pd.DataFrame({"time": x[time_col], "variance": rv}).dropna()
    if plot.empty:
        return None
    fig = px.line(
        plot,
        x="time",
        y="variance",
        labels={"time": "Time (days)", "variance": "Rolling variance"},
        title="How stable is the noise?",
    )
    fig.update_layout(height=280, margin=dict(l=12, r=12, t=38, b=24))
    return fig


def spectrum_figure(df: pd.DataFrame):
    x, time_col, flux_col = _lc_numeric(df)
    if x.empty or len(x) < 20:
        return None
    t = x[time_col].to_numpy(dtype=float)
    f = x[flux_col].to_numpy(dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if not len(dt):
        return None
    cadence = float(np.nanmedian(dt))
    if not np.isfinite(cadence) or cadence <= 0:
        return None
    grid = np.arange(t[0], t[-1] + cadence / 2, cadence)
    if len(grid) < 20 or len(grid) > 200000:
        return None
    interp = np.interp(grid, t, f)
    interp = interp - np.mean(interp)
    power = np.abs(np.fft.rfft(interp)) ** 2
    freq = np.fft.rfftfreq(len(interp), d=cadence)
    keep = (freq > 0) & np.isfinite(power) & (power > 0)
    if not np.any(keep):
        return None
    sdf = pd.DataFrame({"frequency": freq[keep], "power": power[keep]})
    fig = px.line(
        sdf,
        x="frequency",
        y="power",
        log_y=True,
        labels={"frequency": "Frequency (1/day)", "power": "Power"},
        title="Where does variability live in frequency?",
    )
    fig.update_layout(height=280, margin=dict(l=12, r=12, t=38, b=24))
    return fig


def render_method_map():
    tabs = st.tabs(["Used now", "Planned next"])
    with tabs[0]:
        items = [
            ("ACF", "Memory", "Short-lag correlation"),
            ("ADF + KPSS", "Stationarity", "Trend / stationarity checks"),
            ("Variance drift", "Noise stability", "Changing noise amplitude"),
            ("PSD / spectral peak", "Frequency", "Periodic or colored structure"),
            ("Skew + kurtosis", "Distribution", "Asymmetry and heavy tails"),
            ("Gap metrics", "Missing data", "Cadence fragmentation"),
        ]
        cols = st.columns(3)
        for i, (title, tag, note) in enumerate(items):
            cols[i % 3].markdown(
                f"""<div class="method-card">
                <span class="pill-now">NOW · {tag}</span>
                <div class="method-title">{title}</div>
                <div class="method-note">{note}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    with tabs[1]:
        items = [
            ("PACF", "AR structure", "Cleaner autoregressive-order clues"),
            ("FFT / full PSD", "Frequency", "Richer colored-noise fingerprints"),
            ("Lomb–Scargle", "Periodicity", "Gap-friendly periodic structure"),
            ("Wavelets", "Time-frequency", "Nonstationary spectral changes"),
            ("Long-memory metrics", "Persistence", "Beyond short AR lags"),
            ("Change points", "Regime shifts", "Abrupt statistical changes"),
            ("Rolling stationarity", "Local behavior", "Stationarity through time"),
            ("Instrument diagnostics", "Artifacts", "Separate detector artifacts from stars"),
            ("Local morphology", "Transit safety", "Avoid modelling away sharp transit structure"),
        ]
        cols = st.columns(3)
        for i, (title, tag, note) in enumerate(items):
            cols[i % 3].markdown(
                f"""<div class="method-card">
                <span class="pill-next">NEXT · {tag}</span>
                <div class="method-title">{title}</div>
                <div class="method-note">{note}</div>
                </div>""",
                unsafe_allow_html=True,
            )



def render_info_card(
    container,
    label: str,
    value: str,
    explanation: str,
    note: str = "",
    accent: str = "blue",
):
    """Render a non-truncated metric card with a simple native hover tooltip."""
    accent_class = accent if accent in {"teal", "violet", "amber"} else ""
    note_html = f'<div class="info-card-note">{note}</div>' if note else ""
    container.markdown(
        f"""
        <div class="info-card {accent_class}">
            <div class="info-card-label">
                {label}
                <span class="info-dot" title="{explanation}">i</span>
            </div>
            <div class="info-card-value">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_drilldown(
    injections: pd.DataFrame,
    characterization: pd.DataFrame,
    pipeline: str,
    metric_suffix: str,
    metric_label: str,
    repo_root: Path,
    run_dir: Path,
):
    """Presentation-oriented pipeline -> star -> metrics drill-down."""
    branch, detector = PIPELINE_META.get(pipeline, ("", ""))
    hit_col = f"{pipeline}_{metric_suffix}"

    if hit_col not in injections.columns:
        st.info(f"No recovery column was found for {pipeline_label(pipeline)} in this run.")
        return

    st.divider()
    st.subheader(f"{pipeline_label(pipeline)}")
    st.caption(
        f"Background representation: **{branch}** · Transit detector: **{detector}**. "
        "Choose a star below to inspect where this pipeline succeeds or struggles."
    )

    # ------------------------------------------------------------------
    # Pipeline-level headline numbers
    # ------------------------------------------------------------------
    hits = injections[hit_col].fillna(False).astype(bool)
    score_col = f"{pipeline}_score"
    runtime_col = f"{pipeline}_runtime_seconds"

    c1, c2, c3, c4 = st.columns(4)

    if runtime_col in injections.columns:
        avg_runtime = pd.to_numeric(injections[runtime_col], errors="coerce").mean()
        avg_runtime_text = f"{avg_runtime:.3g} s" if pd.notna(avg_runtime) else "—"
    else:
        avg_runtime_text = "—"

    exact_current_col = (
        f"{pipeline}_fap01_exact_recovered"
        if metric_suffix == "fap01_harmonic_recovered" and f"{pipeline}_fap01_exact_recovered" in injections.columns
        else f"{pipeline}_exact_rank1_matched"
    )
    exact_current = (
        injections[exact_current_col].fillna(False).astype(bool).mean()
        if exact_current_col in injections.columns else np.nan
    )

    render_info_card(
        c1,
        "Recovery @ 1% FAP" if metric_suffix == "fap01_harmonic_recovered" else "P/2, P, 2P · top-1",
        f"{100*hits.mean():.1f}%",
        "Correct period/harmonic recovery using the dashboard's current criterion." if metric_suffix != "fap01_harmonic_recovered" else
        "Correct P/2, P or 2P top-ranked match with detector score above the empirical 1% false-alarm threshold.",
        note="Current benchmark metric",
        accent="teal",
    )
    render_info_card(
        c2,
        "Recovered injection cases",
        f"{int(hits.sum()):,} / {len(hits):,}",
        "Number of injection cases counted as recovered under the currently displayed recovery definition.",
    )
    render_info_card(
        c3,
        "Injected P · top-1" + (" @ 1% FAP" if metric_suffix == "fap01_harmonic_recovered" else ""),
        f"{100*exact_current:.1f}%" if np.isfinite(exact_current) else "—",
        "Highest-ranked candidate is the injected period itself" + (" and exceeds the empirical 1% FAP threshold." if metric_suffix == "fap01_harmonic_recovered" else "."),
        accent="violet",
    )
    render_info_card(
        c4,
        "Average runtime per case",
        avg_runtime_text,
        "Mean recorded wall-clock runtime for one injection case in this pipeline.",
        accent="amber",
    )

    # ------------------------------------------------------------------
    # Performance by statistical regime
    # ------------------------------------------------------------------
    if "_regime_key" in injections.columns and (injections["_regime_key"] != "unknown").any():
        regime_perf = (
            injections.assign(_hit=hits)
            .groupby("_regime_key", dropna=False)
            .agg(
                recovery=("_hit", "mean"),
                recovered=("_hit", "sum"),
                cases=("_hit", "size"),
                stars=("target_id", "nunique"),
            )
            .reset_index()
        )
        regime_perf["Light-curve regime"] = regime_perf["_regime_key"].astype(str).map(regime_label)
        regime_perf["Recovery %"] = 100 * regime_perf["recovery"]
        regime_perf = regime_perf.sort_values("Recovery %", ascending=True)

        fig = px.bar(
            regime_perf,
            x="Recovery %",
            y="Light-curve regime",
            orientation="h",
            text=regime_perf["Recovery %"].map(lambda x: f"{x:.1f}%"),
            title=f"{pipeline_label(pipeline)} by background stratum",
            hover_data={"recovered": True, "cases": True, "stars": True, "recovery": False, "_regime_key": False},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=330,
            margin=dict(l=20, r=70, t=65, b=25),
            xaxis_range=[0, min(100, max(5, regime_perf["Recovery %"].max() * 1.15))],
        )
        fig = presentation_plot(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})

    # ------------------------------------------------------------------
    # One row per star: rank best -> worst for this selected pipeline.
    # ------------------------------------------------------------------
    agg_dict = {
        "recovery": (hit_col, lambda s: s.fillna(False).astype(bool).mean()),
        "recovered": (hit_col, lambda s: s.fillna(False).astype(bool).sum()),
        "cases": (hit_col, "size"),
    }
    if runtime_col in injections.columns:
        agg_dict["average_runtime_s"] = (runtime_col, lambda s: pd.to_numeric(s, errors="coerce").mean())

    group_cols = ["target_id", "quarter"]
    if "_regime_key" in injections.columns:
        group_cols.append("_regime_key")

    star_perf = injections.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()
    star_perf["target_id"] = star_perf["target_id"].map(normalize_target_id)
    star_perf["Recovery %"] = 100 * star_perf["recovery"]
    if "_regime_key" in star_perf.columns:
        star_perf["Regime"] = star_perf["_regime_key"].astype(str).map(regime_label)
    else:
        star_perf["Regime"] = "—"
    star_perf = star_perf.sort_values(["Recovery %", "target_id"], ascending=[False, True])

    st.markdown("**Star-to-star spread for this pipeline**")
    best = star_perf.iloc[0]
    worst = star_perf.iloc[-1]

    known_regime_mask = (
        star_perf["Regime"].astype(str).ne("Not loaded")
        if "Regime" in star_perf.columns else pd.Series(False, index=star_perf.index)
    )
    has_regime_metadata = bool(known_regime_mask.any())
    if not has_regime_metadata:
        st.caption(
            "Background-stratum labels are not attached to the currently loaded result rows. "
            "Recovery values are valid; the stratum label is simply unavailable in this view."
        )

    # Keep the initial drill-down compact. Star details open only after the user chooses one.
    owner_key = "pipeline_star_owner"
    if st.session_state.get(owner_key) != pipeline:
        st.session_state[owner_key] = pipeline
        st.session_state["pipeline_star_target"] = None
        st.session_state["pipeline_star_quarter"] = None

    b1, b2 = st.columns(2)
    with b1:
        with st.container(border=True):
            st.caption("Best-performing star")
            st.markdown(f"### KIC {best['target_id']}")
            best_note = f"**{best['Recovery %']:.1f}% recovery** · {int(best['recovered'])}/{int(best['cases'])} injection cases"
            st.markdown(best_note)
            if has_regime_metadata and str(best.get("Regime", "")) != "Not loaded":
                st.caption(f"Background stratum: {best['Regime']}")
            if st.button(
                "Inspect light curve & metrics",
                key=f"inspect_best_{pipeline}_{best['target_id']}",
                use_container_width=True,
            ):
                st.session_state["pipeline_star_target"] = str(best["target_id"])
                st.session_state["pipeline_star_quarter"] = int(best["quarter"]) if pd.notna(best["quarter"]) else 5
                st.rerun()

    with b2:
        with st.container(border=True):
            st.caption("Worst-performing star")
            st.markdown(f"### KIC {worst['target_id']}")
            worst_note = f"**{worst['Recovery %']:.1f}% recovery** · {int(worst['recovered'])}/{int(worst['cases'])} injection cases"
            st.markdown(worst_note)
            if has_regime_metadata and str(worst.get("Regime", "")) != "Not loaded":
                st.caption(f"Background stratum: {worst['Regime']}")
            if st.button(
                "Inspect light curve & metrics",
                key=f"inspect_worst_{pipeline}_{worst['target_id']}",
                use_container_width=True,
            ):
                st.session_state["pipeline_star_target"] = str(worst["target_id"])
                st.session_state["pipeline_star_quarter"] = int(worst["quarter"]) if pd.notna(worst["quarter"]) else 5
                st.rerun()

    # One compact selector replaces the previous 48-button star atlas.
    picker_rows = star_perf.copy()
    picker_rows["_picker_label"] = picker_rows.apply(
        lambda r: (
            f"KIC {r['target_id']} — {r['Recovery %']:.1f}% recovery"
            + (f" — {r['Regime']}" if has_regime_metadata and str(r.get('Regime', '')) != 'Not loaded' else "")
        ),
        axis=1,
    )
    picker_options = ["Choose another star…"] + picker_rows["_picker_label"].tolist()
    picked = st.selectbox(
        "Inspect another star",
        picker_options,
        index=0,
        key=f"pipeline_star_picker_{pipeline}",
        help="Search or choose any star in this pipeline's ranking.",
    )
    if picked != "Choose another star…":
        prow = picker_rows[picker_rows["_picker_label"] == picked].iloc[0]
        picked_target = str(prow["target_id"])
        picked_quarter = int(prow["quarter"]) if pd.notna(prow["quarter"]) else 5
        if (
            st.session_state.get("pipeline_star_target") != picked_target
            or st.session_state.get("pipeline_star_quarter") != picked_quarter
        ):
            st.session_state["pipeline_star_target"] = picked_target
            st.session_state["pipeline_star_quarter"] = picked_quarter
            st.rerun()

    with st.expander("Optional: full star ranking", expanded=False):
        st.caption(
            "Audit view: every star ranked by this pipeline's recovery rate. "
            "Use it to see whether the best/worst examples are isolated or part of a broader performance spread."
        )
        show_cols = ["target_id", "quarter"]
        if has_regime_metadata:
            show_cols.append("Regime")
        show_cols += ["Recovery %", "recovered", "cases"]
        if "average_runtime_s" in star_perf.columns:
            show_cols.append("average_runtime_s")
        column_config = {
            "target_id": "KIC",
            "quarter": "Quarter",
            "Recovery %": st.column_config.NumberColumn("Recovery (%)", format="%.1f"),
            "recovered": "Recovered",
            "cases": "Cases",
            "average_runtime_s": st.column_config.NumberColumn("Average runtime (s)", format="%.3g"),
        }
        if has_regime_metadata:
            column_config["Regime"] = "Background stratum"
        st.dataframe(
            star_perf[show_cols],
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
        )

    target_value = st.session_state.get("pipeline_star_target")
    if not target_value:
        st.caption(
            "Choose the best star, worst star, or another KIC above to open its light curve and detailed metrics."
        )
        return

    target = str(st.session_state.get("pipeline_star_target"))
    target_row = star_perf[star_perf["target_id"].astype(str) == target].iloc[0]
    quarter = int(
        st.session_state.get(
            "pipeline_star_quarter",
            int(target_row["quarter"]) if pd.notna(target_row["quarter"]) else 5,
        )
    )
    target_regime = str(target_row["Regime"])

    target_heading = f"### KIC {target} · Q{quarter}"
    if target_regime != "Not loaded":
        target_heading += f" · {target_regime}"
    st.markdown(target_heading)

    target_cases = injections[
        (injections["target_id"].map(normalize_target_id) == target)
        & (pd.to_numeric(injections["quarter"], errors="coerce") == quarter)
    ].copy()

    # ------------------------------------------------------------------
    # Light curve + statistical characterization
    # ------------------------------------------------------------------
    left, right = st.columns([1.35, 1])

    with left:
        lc, lc_path = load_light_curve(str(repo_root), str(run_dir), target, quarter)
        if not lc.empty:
            fig = light_curve_plot(
                lc,
                f"KIC {target} Q{quarter} · {pipeline_label(pipeline)}",
            )
            fig = presentation_plot(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
        else:
            st.info(
                "No cached/regularized light-curve parquet was found for this KIC. "
                "Pipeline metrics are still available below."
            )

    with right:
        st.markdown("**Statistical characterization**")
        if characterization.empty:
            st.info("No characterization table was detected.")
        else:
            cdf = characterization.copy()
            if "target_id" in cdf.columns:
                cdf["target_id"] = cdf["target_id"].map(normalize_target_id)
            crow = cdf[cdf["target_id"] == target] if "target_id" in cdf.columns else pd.DataFrame()
            if crow.empty:
                st.info("This KIC was not found in the characterization table.")
            else:
                row = crow.iloc[0]
                feature_rows = []
                for c, label in FEATURE_LABELS.items():
                    if c in row.index and pd.notna(row[c]):
                        value = row[c]
                        if isinstance(value, (float, np.floating)):
                            value = f"{value:.5g}"
                        feature_rows.append({"Statistic": label, "Value": value})
                st.dataframe(pd.DataFrame(feature_rows), hide_index=True, use_container_width=True)

    # ------------------------------------------------------------------
    # Pipeline-specific metrics on the selected KIC
    # ------------------------------------------------------------------
    st.markdown(f"**{pipeline_label(pipeline)} on KIC {target}**")

    star_hits = target_cases[hit_col].fillna(False).astype(bool)
    exact_col = f"{pipeline}_exact_rank1_matched"
    harmonic_col = f"{pipeline}_harmonic_rank1_matched"
    top_cols = [
        f"{pipeline}_exact_rank_topk",
        f"{pipeline}_half_period_rank_topk",
        f"{pipeline}_double_period_rank_topk",
    ]

    mc1, mc2, mc3, mc4 = st.columns(4)

    harmonic_value = (
        f"{100*target_cases[harmonic_col].fillna(False).astype(bool).mean():.1f}%"
        if harmonic_col in target_cases.columns else "—"
    )
    exact_value = (
        f"{100*target_cases[exact_col].fillna(False).astype(bool).mean():.1f}%"
        if exact_col in target_cases.columns else "—"
    )

    existing_top = [c for c in top_cols if c in target_cases.columns]
    if existing_top:
        top = pd.Series(False, index=target_cases.index)
        for c in existing_top:
            top |= topk_recovered(target_cases[c])
        top_value = f"{100*top.mean():.1f}%"
    else:
        top_value = "—"

    render_info_card(
        mc1,
        "Displayed recovery metric",
        f"{100*star_hits.mean():.1f}%",
        "Recovery rate for this KIC using the metric currently selected for the dashboard.",
        accent="teal",
    )
    render_info_card(
        mc2,
        "P/2, P, 2P · top-1",
        harmonic_value,
        "The top-ranked candidate is the injected period or an accepted half/double-period alias.",
    )
    render_info_card(
        mc3,
        "Injected P · top-1",
        exact_value,
        "The highest-ranked candidate matches the injected period itself.",
        accent="violet",
    )
    render_info_card(
        mc4,
        f"P/2, P, 2P · top-{TOP_K_DISPLAY}",
        top_value,
        f"The injected period or an accepted half/double-period alias appears anywhere in the retained top-{TOP_K_DISPLAY} candidates.",
        accent="amber",
    )

    # Score distribution for this star/pipeline.
    if score_col in target_cases.columns:
        score_series = pd.to_numeric(target_cases[score_col], errors="coerce")
        plot_df = target_cases.copy()
        plot_df["_score"] = score_series
        plot_df["_recovered"] = star_hits.map({True: "Recovered", False: "Missed"})
        plot_df = plot_df.dropna(subset=["_score"])

        if not plot_df.empty:
            score_x = "case_index"
            score_x_label = "Injection case"
            if "injected_depth" in plot_df.columns:
                depth_num = pd.to_numeric(plot_df["injected_depth"], errors="coerce")
                plot_df["_depth_label"] = depth_num.map(
                    lambda x: f"{1e6*x:.0f} ppm" if np.isfinite(x) else "—"
                )
                score_x = "_depth_label"
                score_x_label = "Injected depth"
            fig = px.scatter(
                plot_df,
                x=score_x,
                y="_score",
                color="_recovered",
                hover_data=[
                    c for c in (
                        "case_index",
                        "injected_period_days",
                        "injected_duration_hours",
                        "injected_depth",
                        f"{pipeline}_recovered_period_days",
                    ) if c in plot_df.columns
                ],
                title=f"{pipeline_label(pipeline)} · injection scores for KIC {target}",
                labels={
                    "_score": "Detection score",
                    score_x: score_x_label,
                    "case_index": "Injection case",
                    "_recovered": "Outcome",
                },
            )
            finite_scores = plot_df["_score"].dropna().astype(float)
            if len(finite_scores):
                lo, hi = finite_scores.min(), finite_scores.max()
                pad = max((hi - lo) * 0.08, abs(hi) * 0.02, 0.5)
                fig.update_yaxes(range=[lo - pad, hi + pad])
            fig.update_traces(marker={"size": 7, "opacity": 0.72})
            fig.update_layout(height=300, title_text=f"{pipeline_label(pipeline)} · injection scores for KIC {target}")
            fig = presentation_plot(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})

    # Preservation / post-fit metrics available for the corresponding background branch.
    st.markdown("**Signal preservation / post-fit diagnostics**")
    background_key = branch.lower()
    diagnostic_candidates = [
        (f"{background_key}_residual_depth", "Residual depth"),
        (f"{background_key}_local_snr", "Local SNR"),
        (f"{background_key}_depth_retention_fraction", "Depth retention"),
        (f"{background_key}_snr_retention_fraction", "SNR retention"),
        (f"{background_key}_residual_acf1", "Residual ACF(1)"),
    ]
    diag_rows = []
    for col, label in diagnostic_candidates:
        if col in target_cases.columns:
            vals = pd.to_numeric(target_cases[col], errors="coerce")
            diag_rows.append(
                {
                    "Metric": label,
                    "Median": vals.median(),
                    "10th percentile": vals.quantile(0.10),
                    "90th percentile": vals.quantile(0.90),
                }
            )
    if diag_rows:
        st.dataframe(pd.DataFrame(diag_rows), hide_index=True, use_container_width=True)
    else:
        st.caption("No branch-specific preservation columns were present in this run output.")


# =============================================================================
# Page 1 — benchmark overview
# =============================================================================

if page.startswith("1"):
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-kicker">Kepler injection–recovery benchmark</div>
            <div class="hero-title">Multi-model transit search</div>
            <div class="hero-meta">
                <b>Model Proof Of Concept:</b> How does the statistical treatment of the stellar background affect weak-transit recovery, how complementary are the resulting detection methods, and is that complementarity structured enough to motivate a multi-model or adaptive search?
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    n_stars = injections["target_id"].nunique() if "target_id" in injections.columns else 0
    n_cases = len(injections)
    expected_stars = expected_star_count(RUN_DIR, injections)
    expected_cases, injection_hover = injection_grid_status(injections, expected_stars)

    run_incomplete = (
        (expected_stars > 0 and n_stars < expected_stars)
        or (expected_cases > 0 and n_cases < expected_cases)
        or (
            "_source_state" in injections.columns
            and (injections["_source_state"] == "checkpoint_partial").any()
        )
    )
    case_progress = min(100.0, (100 * n_cases / expected_cases)) if expected_cases else 100.0

    star_chip = (
        f"{n_stars} stars"
        if expected_stars <= 0 or n_stars >= expected_stars
        else f"{n_stars} / {expected_stars} stars loaded"
    )
    case_chip = (
        f"{n_cases:,} injection cases"
        if expected_cases <= 0 or n_cases >= expected_cases
        else f"{n_cases:,} / {expected_cases:,} injection cases loaded"
    )
    status_chip = (
        f"Run incomplete · {case_progress:.1f}% cases loaded"
        if run_incomplete
        else "Run complete"
    )
    status_style = (
        "background:#eef4ff;border:1px solid #cfe0ff;color:#2557b5;"
        if run_incomplete
        else "background:#ecf8ef;border:1px solid #cbe9d2;color:#176b35;"
    )
    injection_hover_safe = injection_hover.replace('"', '&quot;')

    st.markdown(
        f"""
        <div style="display:flex;flex-wrap:wrap;gap:0.45rem;margin:0.15rem 0 0.65rem 0;">
          <span title="Number of benchmark stars currently represented in the loaded results." style="background:#ffffff;border:1px solid #dbe3ec;border-radius:999px;padding:0.34rem 0.62rem;font-size:0.82rem;font-weight:650;color:#344258;cursor:help;">{star_chip}</span>
          <span title="Background-model + transit-detector combinations currently available in this run." style="background:#ffffff;border:1px solid #dbe3ec;border-radius:999px;padding:0.34rem 0.62rem;font-size:0.82rem;font-weight:650;color:#344258;cursor:help;">{len(pipelines)} pipelines tested</span>
          <span title="{injection_hover_safe}" style="background:#ffffff;border:1px solid #dbe3ec;border-radius:999px;padding:0.34rem 0.62rem;font-size:0.82rem;font-weight:650;color:#344258;cursor:help;">{case_chip} ⓘ</span>
          <span title="The dashboard can read completed stars plus live per-star checkpoints while the benchmark is still running." style="{status_style}border-radius:999px;padding:0.34rem 0.62rem;font-size:0.82rem;font-weight:650;cursor:help;">{status_chip}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if calibration_available:
        st.success(
            "1% FAP calibration is loaded. A recovery now requires both the correct harmonic-aware period match "
            "and a score above that star/pipeline's empirical 1% false-alarm threshold."
        )
    else:
        st.markdown(
            """
            <div style="background:#fff9e8;border:1px solid #efd99a;border-radius:10px;padding:0.62rem 0.78rem;margin:0.2rem 0 0.8rem 0;">
                <div style="font-size:0.88rem;font-weight:750;color:#654b16;margin-bottom:0.14rem;">In progress · 1% false-alarm calibration</div>
                <div style="font-size:0.84rem;line-height:1.35;color:#654b16;"><b>Preliminary benchmark — FAP calibration pending.</b> Values below are harmonic-aware rank-1 period recovery, not final FAP-controlled completeness.</div>
                <div style="font-size:0.78rem;line-height:1.32;color:#7a622d;margin-top:0.24rem;">Plain language: the current result asks whether the injected period (or its half/double-period harmonic) is the top-ranked candidate. The final calibration will additionally require that candidate to beat an empirical false-alarm threshold.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    summary = pipeline_summary(injections, pipelines)
    rate_col = (
        "recovery_at_1pct_fap"
        if calibration_available and "recovery_at_1pct_fap" in summary.columns
        else "harmonic_rank1"
    )

    union, union_n = union_rate(injections, pipelines, metric_suffix)
    if not summary.empty and rate_col in summary.columns:
        valid = summary.dropna(subset=[rate_col])
        if not valid.empty:
            best_row = valid.loc[valid[rate_col].idxmax()]
            best_rate = float(best_row[rate_col])
            best_name = str(best_row["pipeline"])
        else:
            best_rate, best_name = np.nan, "—"
    else:
        best_rate, best_name = np.nan, "—"

    uplift = union - best_rate if np.isfinite(union) and np.isfinite(best_rate) else np.nan

    st.markdown('<span class="section-chip">Headline result</span>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""
            <div class="headline-card" title="Best recovery when one fixed pipeline must be applied to every loaded injection case.">
                <div class="headline-label">Best fixed pipeline</div>
                <div class="headline-value">{pct(best_rate)}</div>
                <div class="headline-delta">{pipeline_label(best_name) if best_name != "—" else ""} · one branch used everywhere</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="headline-card violet" title="Multi-model union: an injection counts as recovered if any loaded pipeline succeeds. This is the combined recovery set, not yet a deployable selector.">
                <div class="headline-label">Multi-model union</div>
                <div class="headline-value">{pct(union)}</div>
                <div class="headline-delta">{f"{union_n:,} recovered · any branch may succeed" if np.isfinite(union) else ""}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="headline-card green" title="Gap between the multi-model union and the best fixed pipeline. It is the headroom an adaptive selector could try to capture, not a guaranteed realized gain.">
                <div class="headline-label">Potential adaptive gain</div>
                <div class="headline-value">{f"+{100*uplift:.1f} pp" if np.isfinite(uplift) else "—"}</div>
                <div class="headline-delta">multi-model union − best fixed pipeline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="science-callout" style="margin-bottom:0.45rem;">
            <b>Interpretation:</b> Different modelling assumptions are recovering different subsets of injected transits.
            Because the multi-model union is substantially higher than the best fixed pipeline, the next scientific question is:
            <b>Can we understand and predict where that complementarity comes from?</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_specs = [
        ("1 · Recovery & scoring", "What exactly counts as a recovery?", "📐 Recovery & scoring"),
        ("2 · Unique Recovery", "Where does one method clearly dominate?", "🧩 Unique Recovery"),
        ("3 · Stars & Statistics", "Does stellar background structure explain it?", "⭐ Stars & statistics"),
        ("4 · Injection Explorer", "Does period, duration or depth explain it?", "🔎 Injection explorer"),
        ("5 · FAP Calibration", "Do the gains survive controlled false alarms?", "🎯 FAP calibration"),
        ("→ Adaptive search", "How could a future selector retain the right branch?", "🧭 POC roadmap"),
    ]
    nav_cols = st.columns(6)
    for col, (label, description, target_page) in zip(nav_cols, nav_specs):
        with col:
            st.button(
                label,
                key=f"overview_nav_{target_page}",
                use_container_width=True,
                on_click=navigate_to,
                args=(target_page,),
                help=description,
                type="secondary",
            )
            st.caption(description)


    # -------------------------------------------------------------------------
    # Five compact overview analyses. The first page intentionally contains only
    # these five evidence blocks after the headline cards/navigation.
    # -------------------------------------------------------------------------

    # 1) Background treatment × detector interaction.
    chart = summary.dropna(subset=[rate_col]).copy()
    st.markdown("#### 1 · Background treatment × transit detector")
    if not chart.empty:
        chart["Recovery (%)"] = 100 * chart[rate_col]
        chart["Background treatment"] = chart["branch"].replace({"GP": "Gaussian Process"})
        chart["Detector"] = chart["detector"]
        chart["Pipeline"] = chart["pipeline"].map(pipeline_label)
        family_order = ["Raw", "ARIMA", "Gaussian Process", "Kalman"]
        family_rank = {name: i for i, name in enumerate(family_order)}
        chart["_family_rank"] = chart["Background treatment"].map(family_rank)
        chart = chart.sort_values(["Detector", "_family_rank"])
        fig = px.line(
            chart,
            x="Background treatment",
            y="Recovery (%)",
            color="Detector",
            markers=True,
            text=chart["Recovery (%)"].map(lambda x: f"{x:.1f}%"),
            category_orders={"Background treatment": family_order, "Detector": ["BLS", "TCF"]},
            hover_data={"Pipeline": True, "Recovery (%)": ":.1f"},
            labels={"Background treatment": "Background treatment", "Recovery (%)": "Recovery (%)"},
            color_discrete_map={"BLS": "#76b9ed", "TCF": "#2563eb"},
        )
        fig.update_traces(textposition="top center", line={"width": 3}, marker={"size": 9})
        fig.update_layout(
            title_text="",
            height=210,
            margin=dict(l=10, r=12, t=8, b=16),
            yaxis_range=[0, min(100, max(5, chart["Recovery (%)"].max() * 1.15))],
            legend_title_text="Transit detector",
        )
        fig = presentation_plot(fig, height=210)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
        st.caption("BLS and TCF respond differently to the background treatment; the compact summaries below test whether that difference has repeatable structure.")
    else:
        st.info("Pipeline recovery columns are not available for the interaction plot.")

    # Derive the three dependence analyses from existing injection-level outputs only.
    star_dep, weak_label = build_star_dependence_table(
        injections, pipelines, metric_suffix, min_dominance_pp, min_complementarity_pp,
        weak_transit_mode, custom_weak_depth_ppm,
    )
    transit_dep = build_transit_dependence_table(
        injections, pipelines, metric_suffix, min_dominance_pp, min_complementarity_pp,
    )
    interaction_dep = build_interaction_dependence_table(
        injections, pipelines, metric_suffix, min_dominance_pp, min_complementarity_pp,
    )
    gain_table = build_multimodel_gain_table(injections, pipelines, metric_suffix, best_name)

    st.markdown(
        """
        <style>
        table.compact-summary-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            background: #ffffff;
            border: 1px solid #d7e4f3;
            border-radius: 10px;
            overflow: hidden;
            margin: 0.25rem 0 0.55rem 0;
        }
        table.compact-summary-table th {
            background: #eaf2ff;
            color: #24456f;
            font-weight: 700;
            font-size: 0.76rem;
            line-height: 1.15;
            padding: 0.42rem 0.46rem;
            border-bottom: 1px solid #d7e4f3;
            text-align: left;
        }
        table.compact-summary-table td {
            color: #24334a;
            font-size: 0.77rem;
            line-height: 1.18;
            padding: 0.40rem 0.46rem;
            border-bottom: 1px solid #edf2f8;
            vertical-align: top;
            overflow-wrap: anywhere;
        }
        table.compact-summary-table tr:last-child td { border-bottom: none; }
        .overview-mini-title {
            font-size: 1rem;
            font-weight: 760;
            color: #172033;
            margin: 0 0 0.08rem 0;
        }
        .overview-mini-note {
            font-size: 0.78rem;
            line-height: 1.25;
            color: #66758a;
            margin-bottom: 0.20rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Where does model choice matter?")
    st.caption(
        f"Compact summaries only. A preference is flagged when the best method leads the second-best by at least {min_dominance_pp:.0f} percentage points."
    )

    # 2 + 3 side by side: star dependence and transit dependence.
    c_star, c_transit = st.columns(2, gap="medium")
    with c_star:
        star_summary, star_clear, star_total = _compact_preference_counts(star_dep, "Stars")
        st.markdown('<div class="overview-mini-title">2 · Star dependence — weak transits</div>', unsafe_allow_html=True)
        if star_total:
            st.markdown(
                f'<div class="overview-mini-note"><b>{star_clear}/{star_total}</b> stars show a clear pipeline preference · {weak_label}.</div>',
                unsafe_allow_html=True,
            )
            _render_compact_summary_table(star_summary)
        else:
            st.info("Star-level weak-transit summary unavailable.")

    with c_transit:
        if not transit_dep.empty and "Meaningful preference" in transit_dep.columns:
            transit_clear = int((transit_dep["Meaningful preference"] != "No meaningful preference").sum())
            transit_total = int(len(transit_dep))
        else:
            transit_clear = transit_total = 0
        st.markdown('<div class="overview-mini-title">3 · Transit dependence across stars</div>', unsafe_allow_html=True)
        if transit_total:
            st.markdown(
                f'<div class="overview-mini-note"><b>{transit_clear}/{transit_total}</b> period–duration–depth combinations show a clear preference. Top separations:</div>',
                unsafe_allow_html=True,
            )
            _render_compact_summary_table(_compact_transit_rows(transit_dep))
        else:
            st.info("Transit-morphology summary unavailable.")

    # 4 + 5 side by side: interaction and multi-model gain.
    c_interaction, c_gain = st.columns(2, gap="medium")
    with c_interaction:
        st.markdown('<div class="overview-mini-title">4 · Background × transit interaction</div>', unsafe_allow_html=True)
        compact_interaction = _compact_interaction_rows(interaction_dep)
        if compact_interaction.empty:
            st.markdown(
                '<div class="overview-mini-note"><b>Pending:</b> valid background-stratum labels are not attached, so this interaction cannot yet be separated from transit dependence.</div>',
                unsafe_allow_html=True,
            )
        else:
            interaction_clear = int((interaction_dep["Meaningful preference"] != "No meaningful preference").sum())
            interaction_total = int(len(interaction_dep))
            st.markdown(
                f'<div class="overview-mini-note"><b>{interaction_clear}/{interaction_total}</b> background × transit combinations show a clear preference. Top separations:</div>',
                unsafe_allow_html=True,
            )
            _render_compact_summary_table(compact_interaction)

    with c_gain:
        st.markdown('<div class="overview-mini-title">5 · What creates the multi-model gain?</div>', unsafe_allow_html=True)
        compact_gain = _compact_gain_rows(gain_table, pipeline_label(best_name) if best_name != "—" else "")
        if compact_gain.empty:
            st.info("Multi-model contribution summary unavailable.")
        else:
            st.markdown(
                f'<div class="overview-mini-note">Cases rescued after the best fixed branch (<b>{pipeline_label(best_name)}</b>) misses. Contributions overlap.</div>',
                unsafe_allow_html=True,
            )
            _render_compact_summary_table(compact_gain)

# =============================================================================
# Recovery & scoring — definitions and Kepler TPS comparison
# =============================================================================

elif page.startswith("7"):
    header(
        "Recovery & scoring",
        "What counts as a recovered injection, what each detector score means, and how this proof of concept relates to Kepler TPS.",
    )

    period_tol = infer_period_match_tolerance(RUN_DIR, fallback=0.02)
    config = read_json_safe(str(RUN_DIR / "benchmark_config.json"))
    bls_objective = _nested_config_value(config, {"bls_objective", "box_least_squares_objective"}) or "snr"

    st.markdown("### 1. What counts as a recovery?")
    st.markdown(
        f"""
        <div class="science-callout">
        For an injection with true period <b>P</b>, the top-ranked recovered period is compared with
        <b>P/2, P, and 2P</b>. Define the smallest relative period error as<br><br>
        <code>e = min(|P̂-P/2|/(P/2), |P̂-P|/P, |P̂-2P|/(2P))</code>.<br><br>
        The current period-match tolerance is <b>{100*period_tol:.1f}%</b>. Therefore the uncalibrated headline
        recovery is: <b>top-ranked candidate + harmonic-aware period error ≤ {100*period_tol:.1f}%</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    r1, r2, r3 = st.columns(3)
    render_info_card(
        r1,
        "Rank used in headline",
        "Top-1",
        f"Top-{TOP_K_DISPLAY} is retained as a diagnostic, but the headline recovery requires the correct period/harmonic to be the first-ranked candidate.",
    )
    render_info_card(
        r2,
        "Period error tolerance",
        f"±{100*period_tol:.1f}%",
        "This is an evaluation rule against the known injected truth. It is not a detection-significance threshold.",
        accent="violet",
    )
    render_info_card(
        r3,
        "Final significance gate",
        "1% FAP",
        "Once calibration is loaded, the matched top-ranked candidate must also exceed its empirical star-and-pipeline false-alarm threshold.",
        accent="teal",
    )

    st.markdown("### 2. What is the detection score?")
    left, right = st.columns(2)
    with left:
        st.markdown(
            f"""
            <div class="method-card">
                <div class="method-title">BLS score</div>
                <div class="method-note">
                The benchmark computes a Box Least Squares periodogram (configured objective: <b>{bls_objective}</b>),
                then robust-standardizes its periodogram power. The quantity stored/ranked as the BLS score is an
                SDE-like statistic:<br><br>
                <code>S_BLS = (power - median(power)) / [1.4826 × MAD(power)]</code><br><br>
                A larger value means the best box-like periodic dip stands farther above the typical periodogram background.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="method-card">
                <div class="method-title">TCF score</div>
                <div class="method-note">
                TCF evaluates the repeated ingress/egress edge pattern expected after the relevant time-series transformation.
                Each valid event gets an edge-consistency score based on the ingress-to-egress difference relative to a robust
                local standard error. Across repeated events, the benchmark combines them as:<br><br>
                <code>S_TCF = median(event scores) × √N_valid</code><br><br>
                A larger value means a more coherent repeated transit-edge pattern across the trial period.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        "BLS and TCF scores are different statistics and should not be compared numerically to one another. "
        "They become comparable at the decision level only after each pipeline is calibrated against its own null distribution."
    )

    st.markdown("### 3. What is the threshold?")
    st.markdown(
        """
        The dashboard uses two different thresholds, which should not be confused:

        **Period-match error threshold:** the recovered period must satisfy the relative-error rule above. This answers *did the search choose the injected signal (or an accepted harmonic)?*

        **False-alarm score threshold:** for each star and pipeline, null light curves are generated with a moving-block surrogate so short-range correlation and gaps are retained. The detector is rerun on each null realization, and the **1% FAP threshold is the empirical 99th percentile of the null maximum scores**. Final calibrated recovery requires both a period match and a score above this threshold.
        """
    )
    if calibration_available:
        st.success("This run has calibrated 1% FAP recovery columns loaded, so the dashboard can apply both conditions.")
    else:
        st.info("Current run status: FAP calibration is still pending, so the Overview is deliberately showing only the period-recovery condition.")

    st.markdown("### 4. How this compares with Kepler TPS")
    st.markdown(
        """
        <div style="background:#ffffff;border:1px solid #dbe7f5;border-radius:12px;overflow:hidden;">
          <div style="display:grid;grid-template-columns:1.1fr 1.7fr 1.7fr;background:#eaf3ff;color:#21324d;font-weight:700;padding:.65rem .8rem;gap:.8rem;">
            <div>Question</div><div>This proof of concept</div><div>Kepler TPS</div>
          </div>
          <div style="display:grid;grid-template-columns:1.1fr 1.7fr 1.7fr;padding:.7rem .8rem;gap:.8rem;border-top:1px solid #e6edf6;">
            <div><b>Noise/background handling</b></div><div>Explicit Raw / ARIMA / GP / Kalman alternatives are compared.</div><div>Adaptive wavelet-based whitening conditions the PDC flux for the transit search.</div>
          </div>
          <div style="display:grid;grid-template-columns:1.1fr 1.7fr 1.7fr;padding:.7rem .8rem;gap:.8rem;border-top:1px solid #e6edf6;">
            <div><b>Transit detector</b></div><div>BLS or TCF after each background treatment.</div><div>Wavelet-based matched filtering produces Single Event Statistics, which are combined over trial periods/phases into the Multiple Event Statistic (MES).</div>
          </div>
          <div style="display:grid;grid-template-columns:1.1fr 1.7fr 1.7fr;padding:.7rem .8rem;gap:.8rem;border-top:1px solid #e6edf6;">
            <div><b>Detection significance</b></div><div>Empirical per-star/per-pipeline 1% FAP score threshold (pending for the current overview).</div><div>Nominal MES threshold of 7.1σ, together with additional transit-consistency/veto tests in production pipeline runs.</div>
          </div>
          <div style="display:grid;grid-template-columns:1.1fr 1.7fr 1.7fr;padding:.7rem .8rem;gap:.8rem;border-top:1px solid #e6edf6;">
            <div><b>Injection recovery bookkeeping</b></div><div>Top-1 period within the configured tolerance of P, P/2 or 2P; then FAP gate when calibrated.</div><div>Kepler injection-efficiency studies used their own truth-matching rules; one published test used period within 3% plus epoch within 0.5 d, while integer aliases were flagged separately.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "The 1% empirical FAP threshold here is not numerically equivalent to Kepler's 7.1σ MES threshold. "
        "The useful comparison is architectural: both separate a search statistic from a significance/consistency decision, but the statistics and calibrations are different."
    )


# =============================================================================
# Unique Recovery — dedicated drill-down page
# =============================================================================

elif page.startswith("6"):
    header(
        "Unique Recovery",
        "Find the strongest cases where one pipeline clearly outperforms the alternatives, then inspect the underlying injections.",
    )

    criterion_text = (
        "Correct P/2, P or 2P top-ranked match + score above the star/pipeline empirical 1% FAP threshold"
        if calibration_available
        else "Correct P/2, P or 2P top-ranked match (uncalibrated)"
    )
    st.markdown(
        f'<div class="science-callout"><b>Recovery criterion:</b> {criterion_text}</div>',
        unsafe_allow_html=True,
    )

    explorer_mode = st.radio(
        "Explorer mode",
        ["Star-level dominance", "Exclusive injection cases", "Head-to-head"],
        horizontal=True,
        key="unique_explorer_mode",
        help=(
            "Star-level dominance is the strong uniqueness test: one pipeline has very high recovery for a star while all alternatives stay low. "
            "Exclusive injection cases instead counts single injection cases recovered by only one pipeline."
        ),
    )

    unique_counts = {
        p: int(unique_case_mask(injections, p, metric_suffix, pipelines).sum())
        for p in pipelines
    }
    comparison_pipeline = None
    dominance_lookup = {}

    if explorer_mode == "Star-level dominance":
        c1, c2 = st.columns(2)
        winner_min_pct = c1.slider(
            "Winner recovery must be at least",
            min_value=60,
            max_value=100,
            value=90,
            step=5,
            format="%d%%",
            key="dominance_winner_min",
        )
        other_max_pct = c2.slider(
            "Every other pipeline must be at most",
            min_value=0,
            max_value=90,
            value=60,
            step=5,
            format="%d%%",
            key="dominance_other_max",
        )
        dominance = dominant_star_rows(
            injections,
            pipelines,
            metric_suffix,
            winner_min=winner_min_pct / 100.0,
            other_max=other_max_pct / 100.0,
        )

        if dominance.empty:
            st.info(
                f"No stars currently meet the strong uniqueness rule: one pipeline ≥ {winner_min_pct}% recovery and every alternative ≤ {other_max_pct}%. "
                "You can relax the thresholds above to explore near-misses."
            )
            explorer_pipeline = pipelines[0] if pipelines else None
            subset = pd.DataFrame()
        else:
            dominance["Pipeline"] = dominance["pipeline"].map(pipeline_label)
            dominance["Star"] = dominance["target_id"].map(lambda x: f"KIC {x}")
            dominance["Winner recovery (%)"] = 100 * dominance["winner_rate"]
            dominance["Best alternative"] = dominance["strongest_other"].map(pipeline_label)
            dominance["Best alternative recovery (%)"] = 100 * dominance["strongest_other_rate"]
            dominance["Separation (pp)"] = 100 * dominance["gap"]
            total_unique_stars = int(dominance["target_id"].nunique())
            unique_ids = sorted(dominance["target_id"].astype(str).unique().tolist(), key=lambda x: int(x) if str(x).isdigit() else str(x))
            pipeline_counts = dominance.groupby("pipeline")["target_id"].nunique().sort_values(ascending=False)

            u1, u2 = st.columns([0.9, 2.1])
            render_info_card(
                u1,
                "Strongly unique stars",
                str(total_unique_stars),
                f"Rule: winner ≥ {winner_min_pct}% and every alternative ≤ {other_max_pct}%.",
                accent="teal",
            )
            with u2:
                st.markdown("**KIC IDs meeting the strong rule**")
                st.markdown(", ".join(f"`{k}`" for k in unique_ids))
                st.caption("Each listed star is assigned to the one pipeline that satisfies the strong-dominance criterion.")

            count_table = pd.DataFrame({
                "Dominant pipeline": [pipeline_label(p) for p in pipeline_counts.index],
                "Strongly unique stars": [int(pipeline_counts.loc[p]) for p in pipeline_counts.index],
                "KIC IDs": [
                    ", ".join(
                        dominance.loc[dominance["pipeline"] == p, "target_id"].astype(str).sort_values().tolist()
                    )
                    for p in pipeline_counts.index
                ],
            })
            st.table(count_table)

            with st.expander("Audit: full strong-dominance ranking", expanded=False):
                st.caption(
                    "A star counts once for a pipeline when that pipeline clears the high-recovery threshold across the star's injection grid and every competing pipeline remains below the low-recovery threshold."
                )
                st.dataframe(
                    dominance[["Star", "Pipeline", "Winner recovery (%)", "Best alternative", "Best alternative recovery (%)", "Separation (pp)", "n_cases"]]
                    .sort_values(["Separation (pp)", "Winner recovery (%)"], ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Winner recovery (%)": st.column_config.NumberColumn(format="%.1f"),
                        "Best alternative recovery (%)": st.column_config.NumberColumn(format="%.1f"),
                        "Separation (pp)": st.column_config.NumberColumn(format="%+.1f"),
                        "n_cases": "Injection cases",
                    },
                )
            explorer_pipeline = st.selectbox(
                "Dominant pipeline",
                pipeline_counts.index.tolist(),
                format_func=lambda p: f"{pipeline_label(p)} — {int(pipeline_counts.loc[p])} strongly unique star{'s' if int(pipeline_counts.loc[p]) != 1 else ''}",
                key="unique_explorer_pipeline",
            )
            selected_dom = dominance[dominance["pipeline"] == explorer_pipeline].copy()
            for _, rr in selected_dom.iterrows():
                dominance_lookup[str(rr["target_id"])] = rr

            selected_ids = selected_dom["target_id"].astype(str).tolist()
            st.markdown(f"#### What is visibly different about the {pipeline_label(explorer_pipeline)}-dominant stars?")
            st.markdown(
                f"**{len(selected_ids)} star{'s' if len(selected_ids) != 1 else ''}:** "
                + ", ".join(f"KIC {k}" for k in selected_ids)
            )
            loaded_targets = injections["target_id"].map(normalize_target_id).unique().tolist()
            assoc = dominance_feature_associations(
                characterization,
                star_summaries,
                loaded_targets=loaded_targets,
                unique_targets=selected_ids,
                top_n=6,
            )
            if not assoc.empty:
                show_assoc = assoc.copy()
                show_assoc["Strongly-unique median"] = show_assoc["Strongly-unique median"].map(lambda x: f"{x:.5g}")
                show_assoc["Other loaded stars median"] = show_assoc["Other loaded stars median"].map(lambda x: f"{x:.5g}")
                show_assoc["Robust separation"] = pd.to_numeric(show_assoc["Robust separation"], errors="coerce").map(lambda x: f"{x:+.2f}" if np.isfinite(x) else "—")
                st.table(show_assoc[["Property", "Strongly-unique median", "Other loaded stars median", "Direction", "Robust separation"]])
                st.caption(
                    "These are the largest descriptive contrasts among the currently loaded benchmark stars. "
                    "They identify properties worth testing as explanations/router features; they do not establish causal responsibility."
                )
            else:
                st.info("No comparable per-star characterization fields are currently attached, so a property-level dominance summary cannot yet be computed for this subset.")

            subset = injections[injections["target_id"].map(normalize_target_id).isin(selected_dom["target_id"].astype(str))].copy()
            subset["_target_norm"] = subset["target_id"].map(normalize_target_id)
            star_ids = selected_dom.sort_values("gap", ascending=False)["target_id"].astype(str).tolist()
            explorer_target = st.selectbox(
                "Star",
                star_ids,
                format_func=lambda k: (
                    f"KIC {k} — {100*float(dominance_lookup[k]['winner_rate']):.1f}% vs "
                    f"{100*float(dominance_lookup[k]['strongest_other_rate']):.1f}% best alternative"
                ),
                key="unique_explorer_star",
            )
            comparison_note = (
                f"{pipeline_label(explorer_pipeline)} reaches {100*float(dominance_lookup[explorer_target]['winner_rate']):.1f}% recovery, "
                f"while the strongest alternative ({pipeline_label(str(dominance_lookup[explorer_target]['strongest_other']))}) reaches only "
                f"{100*float(dominance_lookup[explorer_target]['strongest_other_rate']):.1f}%"
            )

    else:
        selectable = [p for p in pipelines if unique_counts.get(p, 0) > 0]
        if explorer_mode == "Exclusive injection cases" and not selectable:
            st.info("No exclusive injection recoveries are available under the currently selected recovery criterion.")
            explorer_pipeline = pipelines[0] if pipelines else None
            subset = pd.DataFrame()
        else:
            pipeline_options = selectable if explorer_mode == "Exclusive injection cases" else list(pipelines)
            explorer_pipeline = st.selectbox(
                "Pipeline",
                pipeline_options,
                format_func=lambda p: (
                    f"{pipeline_label(p)} — {unique_counts.get(p, 0)} exclusive injection cases"
                    if explorer_mode == "Exclusive injection cases"
                    else pipeline_label(p)
                ),
                key="unique_explorer_pipeline",
            )
            selected_col = f"{explorer_pipeline}_{metric_suffix}"
            subset = injections.copy()

            if explorer_mode == "Exclusive injection cases":
                subset = subset[unique_case_mask(subset, explorer_pipeline, metric_suffix, pipelines)].copy()
                comparison_note = "missed by every other loaded pipeline"
            else:
                comparator_options = [p for p in pipelines if p != explorer_pipeline]
                comparison_pipeline = st.selectbox(
                    "Compare against",
                    comparator_options,
                    format_func=pipeline_label,
                    key="unique_explorer_comparator",
                )
                outcome = st.radio(
                    "Show",
                    [
                        f"{pipeline_label(explorer_pipeline)} recovers / {pipeline_label(comparison_pipeline)} misses",
                        f"{pipeline_label(comparison_pipeline)} recovers / {pipeline_label(explorer_pipeline)} misses",
                        "Both recover",
                    ],
                    horizontal=True,
                    key="unique_explorer_outcome",
                )
                comp_col = f"{comparison_pipeline}_{metric_suffix}"
                a_hit = subset[selected_col].fillna(False).astype(bool) if selected_col in subset.columns else pd.Series(False, index=subset.index)
                b_hit = subset[comp_col].fillna(False).astype(bool) if comp_col in subset.columns else pd.Series(False, index=subset.index)
                if outcome.startswith(pipeline_label(explorer_pipeline)):
                    subset = subset[a_hit & ~b_hit].copy()
                    comparison_note = f"recovered here, while {pipeline_label(comparison_pipeline)} misses"
                elif outcome.startswith(pipeline_label(comparison_pipeline)):
                    subset = subset[b_hit & ~a_hit].copy()
                    comparison_note = f"missed here, while {pipeline_label(comparison_pipeline)} recovers"
                else:
                    subset = subset[a_hit & b_hit].copy()
                    comparison_note = f"recovered by both {pipeline_label(explorer_pipeline)} and {pipeline_label(comparison_pipeline)}"

            if not subset.empty:
                subset["_target_norm"] = subset["target_id"].map(normalize_target_id)
                star_counts = subset.groupby("_target_norm").size().sort_values(ascending=False)
                star_ids = star_counts.index.tolist()
                explorer_target = st.selectbox(
                    "Star",
                    star_ids,
                    format_func=lambda k: f"KIC {k} — {int(star_counts.loc[k])} matching case{'s' if int(star_counts.loc[k]) != 1 else ''}",
                    key="unique_explorer_star",
                )

    if subset.empty or explorer_pipeline is None:
        st.info("No cases are available for the current explorer selection.")
    else:
        # -------------------------------------------------------------
        # All injections for the selected star (normally 81 = 3×3×9).
        # This is deliberately built from the complete injection table,
        # not only from the currently filtered unique/head-to-head subset.
        # -------------------------------------------------------------
        all_star_cases = injections[injections["target_id"].map(normalize_target_id) == explorer_target].copy()
        if "case_index" in all_star_cases.columns:
            all_star_cases = all_star_cases.sort_values("case_index")
        all_star_cases = all_star_cases.reset_index(drop=True)

        # Star-level evidence first: the unique-recovery claim should be inspectable before
        # dropping into individual injection rows.
        if explorer_mode == "Star-level dominance" and not all_star_cases.empty:
            q_series = pd.to_numeric(all_star_cases.get("quarter", pd.Series([5])), errors="coerce").dropna()
            evidence_quarter = int(q_series.iloc[0]) if not q_series.empty else 5
            evidence_branch = PIPELINE_META.get(explorer_pipeline, (explorer_pipeline.split("_")[0], ""))[0]
            st.markdown(f"### Evidence for KIC {explorer_target} · {pipeline_label(explorer_pipeline)}")
            st.caption(comparison_note + ". Inspect the raw background structure and the saved model diagnostics before looking at individual injections.")

            raw_ev, model_ev, metric_ev = st.tabs(["Raw light curve", "Model pre-fit / post-fit", "Saved parameters & metrics"])
            evidence_lc, _ = load_light_curve(str(REPO_ROOT), str(RUN_DIR), explorer_target, evidence_quarter)

            with raw_ev:
                if not evidence_lc.empty:
                    fig = light_curve_plot(evidence_lc, f"KIC {explorer_target} Q{evidence_quarter} · raw source light curve")
                    fig = presentation_plot(fig, height=270)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
                    st.caption("Use this view to look for visible scatter, long-memory/smooth structure, gaps, changing variance, or other background morphology that could favor one treatment.")
                else:
                    st.info("The saved source light curve was not found for this star, but the scalar characterization and recovery diagnostics remain available.")

            with model_ev:
                if evidence_branch.lower() == "raw":
                    st.info("Raw is the reference branch, so there is no separate fitted background or residual model to display.")
                elif evidence_lc.empty:
                    st.info("A source light-curve file is required to display saved background/residual series for this star.")
                else:
                    bg_col = branch_series_column(evidence_lc, evidence_branch, "background")
                    resid_col = branch_series_column(evidence_lc, evidence_branch, "residual")
                    bcol, rcol = st.columns(2)
                    with bcol:
                        if bg_col:
                            fig = saved_series_plot(evidence_lc, bg_col, f"{evidence_branch} fitted background")
                            st.plotly_chart(presentation_plot(fig, height=250), use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
                        else:
                            st.info(f"The full {evidence_branch} fitted-background time series was not persisted for this run.")
                    with rcol:
                        if resid_col:
                            fig = saved_series_plot(evidence_lc, resid_col, f"{evidence_branch} residual")
                            st.plotly_chart(presentation_plot(fig, height=250), use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
                        else:
                            st.info(f"The full {evidence_branch} residual time series was not persisted for this run.")

            with metric_ev:
                evidence_diag, evidence_params = star_model_evidence_tables(
                    explorer_target, explorer_pipeline, all_star_cases, characterization, star_summaries
                )
                if not evidence_diag.empty:
                    st.markdown("**Pre-fit versus post-fit diagnostics**")
                    show_diag = evidence_diag.copy()
                    show_diag["Value"] = pd.to_numeric(show_diag["Value"], errors="coerce").map(lambda x: f"{x:.5g}" if np.isfinite(x) else "—")
                    st.table(show_diag)
                else:
                    st.info("No pre/post scalar diagnostics were found for this branch/star in the saved outputs.")
                if not evidence_params.empty:
                    st.markdown("**Saved model fit / parameter fields**")
                    st.table(evidence_params)
                else:
                    st.caption("No branch-specific model-parameter fields were persisted in the currently loaded summary/injection files.")

        # Mark which rows satisfy the explorer filter currently shown above.
        matching_case_ids = set()
        if "case_index" in subset.columns:
            matching_rows = subset[subset["_target_norm"] == explorer_target]
            matching_case_ids = set(pd.to_numeric(matching_rows["case_index"], errors="coerce").dropna().astype(int).tolist())

        st.markdown("### Injection-level evidence")
        show_all_injections = st.toggle(
            f"Show all {len(all_star_cases)} injection cases for KIC {explorer_target}",
            value=False,
            key=f"show_all_injections_{explorer_target}",
            help="Keep this closed during the main demo; open it when you want to audit every injected period/duration/depth combination and every pipeline's selected period.",
        )
        table_event = None
        table_view = "Selected periods"
        if show_all_injections:
            st.markdown(f"**All injections for KIC {explorer_target}**")
            st.caption(
                f"Showing {len(all_star_cases)} saved injection cases for this star. "
                "Each pipeline column tells you what period that pipeline actually selected. "
                "✓/✗ is evaluated using the recovery criterion shown at the top of this page."
            )

            table_view = st.radio(
                "Injection table view",
                ["Selected periods", "Recovery status", f"Top-{TOP_K_DISPLAY} ranks", "Full details"],
                horizontal=True,
                key=f"star_injection_table_view_{explorer_target}",
                help=(
                    "Selected periods shows the top-ranked period chosen by each pipeline. "
                    f"Recovery status reduces this to pass/fail. Top-{TOP_K_DISPLAY} ranks shows where P, P/2 and 2P appeared. "
                    "Full details also includes score, calibrated threshold when available, and runtime."
                ),
            )

            table_rows = []
            for _, rr in all_star_cases.iterrows():
                case_num = _num(rr.get("case_index", np.nan))
                injected_period = _num(rr.get("injected_period_days", np.nan))
                duration = _num(rr.get("injected_duration_hours", np.nan))
                dep = _num(rr.get("injected_depth", np.nan))
                case_int = int(case_num) if np.isfinite(case_num) else None
                out = {
                    "Case": case_int if case_int is not None else "—",
                    "Injected P (d)": injected_period,
                    "Duration (h)": duration,
                    "Depth (ppm)": 1e6 * dep if np.isfinite(dep) else np.nan,
                    "Matches current filter": "✓" if case_int in matching_case_ids else "",
                }
                for pp in pipelines:
                    label = pipeline_label(pp)
                    if table_view == "Selected periods":
                        out[label] = selected_period_cell(rr, pp, metric_suffix)
                    elif table_view == "Recovery status":
                        out[label] = recovery_status_cell(rr, pp, metric_suffix)
                    elif table_view == f"Top-{TOP_K_DISPLAY} ranks":
                        out[label] = topk_rank_cell(rr, pp)
                    else:
                        out[label] = full_detail_cell(rr, pp, metric_suffix)
                table_rows.append(out)

            injection_table_df = pd.DataFrame(table_rows)
            table_event = None
            try:
                table_event = st.dataframe(
                    injection_table_df,
                    hide_index=True,
                    use_container_width=True,
                    height=min(760, 42 * (len(injection_table_df) + 1)),
                    on_select="rerun",
                    selection_mode="single-row",
                    key=f"star_81_injection_table_{explorer_target}_{table_view}",
                    column_config={
                        "Injected P (d)": st.column_config.NumberColumn(format="%.5g"),
                        "Duration (h)": st.column_config.NumberColumn(format="%.3g"),
                        "Depth (ppm)": st.column_config.NumberColumn(format="%.0f"),
                    },
                )
            except TypeError:
                # Older Streamlit versions can still display the full table; the
                # injection selector below remains the drill-down control.
                st.dataframe(
                    injection_table_df,
                    hide_index=True,
                    use_container_width=True,
                    height=min(760, 42 * (len(injection_table_df) + 1)),
                    column_config={
                        "Injected P (d)": st.column_config.NumberColumn(format="%.5g"),
                        "Duration (h)": st.column_config.NumberColumn(format="%.3g"),
                        "Depth (ppm)": st.column_config.NumberColumn(format="%.0f"),
                    },
                )
                st.caption("Row-click selection requires a newer Streamlit version; use the Injection selector below to inspect a case.")

        # If supported, clicking a row sets the drill-down injection below.
        selected_table_row = None
        if table_event is not None:
            try:
                rows = list(table_event.selection.rows)
                if rows:
                    selected_table_row = int(rows[0])
            except Exception:
                pass

        def _case_label(i: int) -> str:
            r = all_star_cases.iloc[int(i)]
            case = r.get("case_index", i)
            period = _num(r.get("injected_period_days", np.nan))
            duration = _num(r.get("injected_duration_hours", np.nan))
            depth = _num(r.get("injected_depth", np.nan))
            ptxt = f"{period:.4g} d" if np.isfinite(period) else "—"
            dtxt = f"{duration:.3g} h" if np.isfinite(duration) else "—"
            depthtxt = f"{1e6*depth:.0f} ppm" if np.isfinite(depth) else "—"
            try:
                ctxt = str(int(float(case)))
            except Exception:
                ctxt = str(case)
            flag = " · matches filter" if np.isfinite(_num(case)) and int(_num(case)) in matching_case_ids else ""
            return f"case {ctxt} · P={ptxt} · duration={dtxt} · depth={depthtxt}{flag}"

        if selected_table_row is not None and 0 <= selected_table_row < len(all_star_cases):
            default_case_idx = selected_table_row
        else:
            # Default to the first case satisfying the unique/head-to-head filter
            # so the page initially demonstrates the selected complementarity result.
            default_case_idx = 0
            if matching_case_ids and "case_index" in all_star_cases.columns:
                numeric_cases = pd.to_numeric(all_star_cases["case_index"], errors="coerce")
                candidates = np.where(numeric_cases.isin(matching_case_ids))[0]
                if len(candidates):
                    default_case_idx = int(candidates[0])

        st.markdown("### Inspect one injection")
        explorer_case_idx = st.selectbox(
            "Injection",
            range(len(all_star_cases)),
            index=default_case_idx,
            format_func=_case_label,
            key=f"unique_explorer_case_{explorer_target}",
            help="Select any of the star's injections, or click a row in the table above when row selection is supported.",
        )
        erow = all_star_cases.iloc[int(explorer_case_idx)]
        equarter = int(pd.to_numeric(pd.Series([erow.get("quarter", 5)]), errors="coerce").fillna(5).iloc[0])
        injected_p = pd.to_numeric(pd.Series([erow.get("injected_period_days", np.nan)]), errors="coerce").iloc[0]
        duration_h = pd.to_numeric(pd.Series([erow.get("injected_duration_hours", np.nan)]), errors="coerce").iloc[0]
        depth = pd.to_numeric(pd.Series([erow.get("injected_depth", np.nan)]), errors="coerce").iloc[0]
        recovered_p = pd.to_numeric(
            pd.Series([erow.get(f"{explorer_pipeline}_recovered_period_days", np.nan)]), errors="coerce"
        ).iloc[0]

        regime_key = str(erow.get("_regime_key", "unknown"))
        outcome_text = period_match_label(erow, explorer_pipeline)

        m1, m2, m3, m4, m5 = st.columns([1.25, 1, 1, 1, 1.12])
        m1.markdown(
            f"""
            <div class="info-card">
                <div class="info-card-label">Star</div>
                <div class="info-card-value" style="font-size:1.55rem; white-space:nowrap;">KIC {explorer_target}</div>
                <div class="info-card-note">Quarter {equarter}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        m2.metric("Injected period", f"{injected_p:.4g} d" if np.isfinite(injected_p) else "—")
        m3.metric("Duration", f"{duration_h:.3g} h" if np.isfinite(duration_h) else "—")
        m4.metric("Depth", f"{1e6*depth:.0f} ppm" if np.isfinite(depth) else "—")
        m5.metric("Recovered period", f"{recovered_p:.5g} d" if np.isfinite(recovered_p) else "—")

        selected_case_num = _num(erow.get("case_index", np.nan))
        selected_matches_filter = (
            np.isfinite(selected_case_num) and int(selected_case_num) in matching_case_ids
        )
        if explorer_mode == "Star-level dominance":
            st.success(f"Strong star-level uniqueness: {comparison_note}.")
        elif explorer_mode == "Exclusive injection cases":
            if selected_matches_filter:
                st.success(
                    f"{pipeline_label(explorer_pipeline)} uniquely recovers this case ({outcome_text}); "
                    f"it is {comparison_note} at the same criterion."
                )
            else:
                st.info(
                    "This injection was selected from the star's full injection table. It is not one of the strict "
                    f"exclusive injection cases for {pipeline_label(explorer_pipeline)} under the current criterion; "
                    "the pipeline-by-pipeline result below shows what each method selected."
                )
        else:
            if selected_matches_filter:
                st.info(f"Selected head-to-head case: {comparison_note}.")
            else:
                st.info(
                    "This injection was selected from the star's full injection table and does not satisfy the current "
                    "head-to-head filter. The comparison below still shows every pipeline's result for this case."
                )

        branch, detector = PIPELINE_META.get(explorer_pipeline, ("", ""))
        lc, lc_path = load_light_curve(str(REPO_ROOT), str(RUN_DIR), explorer_target, equarter)

        raw_tab, background_tab, residual_tab, phase_tab = st.tabs(
            ["Raw light curve", f"{branch} background", f"{branch} residual", "Phase-folded"]
        )

        with raw_tab:
            if not lc.empty:
                fig = presentation_plot(light_curve_plot(lc, f"KIC {explorer_target} Q{equarter} · saved source light curve"))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
                st.caption("This is the saved source light curve. The synthetic injection itself is not reconstructed unless an injected time-series was persisted by the benchmark.")
            else:
                st.info("Raw light-curve plot not available yet because no cached/regularized light-curve file was found for this star in the current benchmark outputs.")

        with background_tab:
            if branch.lower() == "raw":
                st.info("No background plot is expected for the Raw branch — Raw is the reference light-curve representation and does not fit a separate background model.")
            elif not lc.empty:
                bg_col = branch_series_column(lc, branch, "background")
                if bg_col:
                    fig = presentation_plot(saved_series_plot(lc, bg_col, f"{branch} saved background · KIC {explorer_target}"))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
                else:
                    st.info(
                        f"{branch} background plot not available yet. The {branch} branch was run for this benchmark, "
                        "but the full fitted background time-series was not saved. The saved scalar diagnostics are shown below."
                    )
            else:
                st.info("Light-curve plot not available yet for this star because no cached/regularized light-curve file was found in this benchmark output.")

        with residual_tab:
            if not lc.empty:
                resid_col = branch_series_column(lc, branch, "residual")
                if resid_col:
                    fig = presentation_plot(saved_series_plot(lc, resid_col, f"{branch} saved residual · KIC {explorer_target}"))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
                else:
                    st.info(
                        f"{branch} residual plot not available yet. The branch was run, but the full residual/innovation "
                        "time-series was not saved for this benchmark. Use the saved depth/SNR-retention and residual-ACF diagnostics below."
                    )
            else:
                st.info("Residual plot not available yet because the source light-curve file for this star was not found in the saved benchmark outputs.")

        with phase_tab:
            if not lc.empty and np.isfinite(injected_p) and injected_p > 0:
                fig = phase_folded_plot(lc, float(injected_p), f"KIC {explorer_target} · source light curve folded at injected P={injected_p:.4g} d")
                fig = presentation_plot(fig)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
                st.caption("Folded at the known injected period for context. This view uses the saved source flux and does not claim to recreate an unsaved synthetic injection.")
            else:
                st.info("Phase-folded plot not available yet. It requires a saved source light curve and a finite injected period for this case.")

        st.markdown("**Same injection across every pipeline**")
        compare_rows = []
        for p in pipelines:
            hit_col = f"{p}_{metric_suffix}"
            if hit_col not in erow.index:
                continue
            recovered = bool(erow[hit_col]) if pd.notna(erow[hit_col]) else False
            exact_rank = pd.to_numeric(pd.Series([erow.get(f"{p}_exact_rank_topk", np.nan)]), errors="coerce").iloc[0]
            half_rank = pd.to_numeric(pd.Series([erow.get(f"{p}_half_period_rank_topk", np.nan)]), errors="coerce").iloc[0]
            double_rank = pd.to_numeric(pd.Series([erow.get(f"{p}_double_period_rank_topk", np.nan)]), errors="coerce").iloc[0]
            runtime = pd.to_numeric(pd.Series([erow.get(f"{p}_runtime_seconds", np.nan)]), errors="coerce").iloc[0]
            detected = erow.get(f"{p}_fap01_detected", np.nan)
            compare_rows.append(
                {
                    "Pipeline": pipeline_label(p),
                    "Recovered @ criterion": "✓" if recovered else "—",
                    "Period result": period_match_label(erow, p),
                    "Recovered P (d)": erow.get(f"{p}_recovered_period_days", np.nan),
                    f"Injected P top-{TOP_K_DISPLAY} rank": exact_rank,
                    f"P/2 top-{TOP_K_DISPLAY} rank": half_rank,
                    f"2P top-{TOP_K_DISPLAY} rank": double_rank,
                    "Above 1% FAP threshold": (
                        "✓" if pd.notna(detected) and bool(detected) else ("—" if pd.notna(detected) else "n/a")
                    ),
                    "Runtime (s)": runtime,
                }
            )
        comparison_df = pd.DataFrame(compare_rows)
        if not comparison_df.empty:
            comparison_df["_selected"] = comparison_df["Pipeline"].eq(pipeline_label(explorer_pipeline))
            comparison_df = comparison_df.sort_values(["_selected", "Recovered @ criterion"], ascending=[False, False]).drop(columns="_selected")
            st.dataframe(
                comparison_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Recovered P (d)": st.column_config.NumberColumn(format="%.5g"),
                    "Runtime (s)": st.column_config.NumberColumn(format="%.3f"),
                },
            )

        # Selected branch preservation diagnostics for the exact injection.
        branch_key = branch.lower()
        diag_map = [
            (f"{branch_key}_residual_depth", "Residual depth"),
            (f"{branch_key}_local_snr", "Local SNR"),
            (f"{branch_key}_depth_retention_fraction", "Depth retention"),
            (f"{branch_key}_snr_retention_fraction", "SNR retention"),
            (f"{branch_key}_residual_acf1", "Residual ACF(1)"),
        ]
        diag_rows = []
        for col, label in diag_map:
            if col in erow.index and pd.notna(erow[col]):
                diag_rows.append({"Diagnostic": label, "Value": erow[col]})
        if diag_rows:
            st.markdown(f"**Why might {branch} have helped? — saved preservation diagnostics**")
            st.dataframe(pd.DataFrame(diag_rows), hide_index=True, use_container_width=True)

        # Regime + characterization bridge to the future router.
        st.markdown("**Star context**")
        context_cols = st.columns([1, 2])
        if regime_key in REGIME_META:
            context_cols[0].metric("Background stratum", regime_label(regime_key))
        else:
            context_cols[0].metric("Background stratum", "Not attached")
            context_cols[0].caption("Recovery is still valid; only the optional selection-stratum label is unavailable in this loaded result view.")
        if not characterization.empty and "target_id" in characterization.columns:
            cdf = characterization.copy()
            cdf["target_id"] = cdf["target_id"].map(normalize_target_id)
            crow = cdf[cdf["target_id"] == explorer_target]
            if not crow.empty:
                rr = crow.iloc[0]
                compact_features = []
                for c in ("flux_std", "acf_lag_1", "acf_timescale_days", "variance_drift", "spectral_strength", "gap_fraction"):
                    if c in rr.index and pd.notna(rr[c]):
                        compact_features.append({"Statistic": FEATURE_LABELS.get(c, c), "Value": rr[c]})
                if compact_features:
                    context_cols[1].dataframe(pd.DataFrame(compact_features), hide_index=True, use_container_width=True)

        with st.expander("Pairwise complementarity — who recovers cases the other misses?", expanded=False):
            st.markdown(
                """
                **How to read this:** choose a row pipeline **A** and a column pipeline **B**. The cell is the number of injections
                recovered by **A** but missed by **B** under the same recovery criterion.

                This is a **pairwise** difference, not a strict unique-recovery count. A case in a cell may also be recovered by a third pipeline.
                Compare opposite cells to see which member of the pair contributes more additional recoveries.
                """
            )
            matrix = pd.DataFrame(index=pipelines, columns=pipelines, dtype=float)
            for pa in pipelines:
                ca = f"{pa}_{metric_suffix}"
                a = injections[ca].fillna(False).astype(bool) if ca in injections.columns else pd.Series(False, index=injections.index)
                for pb in pipelines:
                    if pa == pb:
                        matrix.loc[pa, pb] = np.nan
                        continue
                    cb = f"{pb}_{metric_suffix}"
                    b = injections[cb].fillna(False).astype(bool) if cb in injections.columns else pd.Series(False, index=injections.index)
                    matrix.loc[pa, pb] = int((a & ~b).sum())
            matrix.index = [pipeline_label(p) for p in matrix.index]
            matrix.columns = [pipeline_label(p) for p in matrix.columns]
            st.dataframe(matrix, use_container_width=True)


# =============================================================================
# Page 2 — stars & statistics
# =============================================================================

elif page.startswith("2"):
    header(
        "Stars & statistics",
        "Pick a star and see its statistical behavior visually — then compare how the search pipelines behave on it.",
    )

    valid_regimes = []
    if "_regime_key" in injections.columns:
        valid_regimes = [
            x for x in injections["_regime_key"].dropna().astype(str).unique().tolist()
            if x in REGIME_META
        ]

    # ------------------------------------------------------------------
    # Top visual: statistical background strata and recovery heatmap.
    # ------------------------------------------------------------------
    if valid_regimes:
        st.subheader("Five statistical background strata")

        card_cols = st.columns(min(5, len(valid_regimes)))
        for i, regime in enumerate(sorted(valid_regimes, key=lambda x: regime_label(x))):
            nstars = injections.loc[
                injections["_regime_key"].astype(str) == regime, "target_id"
            ].map(normalize_target_id).nunique()
            card_cols[i % len(card_cols)].metric(regime_label(regime), f"{nstars} stars")

        rows = []
        for regime, g in injections[injections["_regime_key"].isin(valid_regimes)].groupby("_regime_key"):
            for p in pipelines:
                c = f"{p}_{metric_suffix}"
                if c in g.columns:
                    rows.append(
                        {
                            "Background stratum": regime_label(regime),
                            "Pipeline": pipeline_label(p),
                            "Recovery": 100 * g[c].fillna(False).astype(bool).mean(),
                        }
                    )
        hdf = pd.DataFrame(rows)
        if not hdf.empty:
            pivot = hdf.pivot(index="Background stratum", columns="Pipeline", values="Recovery")
            fig = px.imshow(
                pivot,
                text_auto=".0f",
                aspect="auto",
                color_continuous_scale="Viridis",
                zmin=0,
                zmax=100,
                labels={"x": "", "y": "", "color": "Recovery %"},
                title="Which pipeline works best for which kind of star?",
            )
            fig.update_layout(
                height=350,
                margin=dict(l=15, r=15, t=60, b=35),
                xaxis_tickangle=-20,
            )
            fig = presentation_plot(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
    # else:
    #     st.info(
    #         "The current checkpoint files do not expose the five stratification labels directly. "
    #         "The app will pick them up automatically from the clean-background manifest when it is available."
    #     )

    # ------------------------------------------------------------------
    # Star picker — simple filter + KIC buttons, no giant table.
    # ------------------------------------------------------------------
    st.subheader("Choose a star")

    regime_filter_options = ["All background strata"] + sorted(valid_regimes, key=lambda x: regime_label(x))
    selected_regime = st.selectbox(
        "Background stratum",
        regime_filter_options,
        format_func=lambda x: x if x == "All background strata" else regime_label(x),
        label_visibility="collapsed",
    )

    atlas = injections.copy()
    if selected_regime != "All background strata":
        atlas = atlas[atlas["_regime_key"].astype(str) == selected_regime]

    atlas_cols = ["target_id", "quarter"]
    if "_regime_key" in atlas.columns:
        atlas_cols.append("_regime_key")
    atlas_targets = (
        atlas[atlas_cols]
        .drop_duplicates()
        .assign(target_id=lambda d: d["target_id"].map(normalize_target_id))
        .sort_values("target_id")
    )

    if atlas_targets.empty:
        st.warning("No stars match this filter.")
    else:
        if "atlas_target" not in st.session_state:
            st.session_state["atlas_target"] = str(atlas_targets.iloc[0]["target_id"])

        visible = set(atlas_targets["target_id"].astype(str))
        if str(st.session_state.get("atlas_target")) not in visible:
            st.session_state["atlas_target"] = str(atlas_targets.iloc[0]["target_id"])

        button_cols = st.columns(5)
        for i, (_, r) in enumerate(atlas_targets.iterrows()):
            tid = str(r["target_id"])
            q = int(r["quarter"]) if pd.notna(r["quarter"]) else 5
            if button_cols[i % 5].button(
                f"KIC {tid}",
                key=f"star_picker_{tid}_{q}",
                use_container_width=True,
                type="primary" if st.session_state.get("atlas_target") == tid else "secondary",
            ):
                st.session_state["atlas_target"] = tid
                st.session_state["atlas_quarter"] = q
                st.rerun()

        target = str(st.session_state["atlas_target"])
        target_rows = injections[injections["target_id"].map(normalize_target_id) == target]
        qvals = pd.to_numeric(target_rows["quarter"], errors="coerce").dropna().unique()
        quarter = int(st.session_state.get("atlas_quarter", qvals[0] if len(qvals) else 5))
        regime_vals = target_rows["_regime_key"].dropna().astype(str).unique() if "_regime_key" in target_rows.columns else []
        target_regime = regime_vals[0] if len(regime_vals) and regime_vals[0] in REGIME_META else None

        title_bits = [f"KIC {target}", f"Q{quarter}"]
        if target_regime:
            title_bits.append(regime_label(target_regime))
        st.markdown("### " + " · ".join(title_bits))

        lc, lc_path = load_light_curve(str(REPO_ROOT), str(RUN_DIR), target, quarter)

        # --------------------------------------------------------------
        # Visual statistical story.
        # --------------------------------------------------------------
        if not lc.empty:
            stats = derive_visual_stats(lc)

            # Small visual fingerprint cards.
            stat_cards = [
                ("Memory", stats.get("ACF(1)"), ""),
                ("Noise drift", stats.get("Variance drift"), ""),
                ("Skew", stats.get("Skewness"), ""),
                ("Heavy tails", stats.get("Excess kurtosis"), ""),
                ("Gaps", stats.get("Gap fraction"), "%"),
            ]
            cols = st.columns(5)
            for i, (label, value, suffix) in enumerate(stat_cards):
                if value is None or not np.isfinite(value):
                    shown = "—"
                elif suffix == "%":
                    shown = f"{100*value:.1f}%"
                else:
                    shown = f"{value:.3g}"
                cols[i].metric(label, shown)

            # Add saved stationarity tests when available, but keep them visual.
            if not characterization.empty and "target_id" in characterization.columns:
                cdf = characterization.copy()
                cdf["target_id"] = cdf["target_id"].map(normalize_target_id)
                crow = cdf[cdf["target_id"] == target]
                if not crow.empty:
                    r = crow.iloc[0]
                    adf = r.get("original_adf_pvalue", np.nan)
                    kpss = r.get("original_kpss_pvalue", np.nan)
                    if pd.notna(adf) or pd.notna(kpss):
                        a, b = st.columns(2)
                        a.metric("ADF p-value", f"{float(adf):.3g}" if pd.notna(adf) else "—")
                        b.metric("KPSS p-value", f"{float(kpss):.3g}" if pd.notna(kpss) else "—")

            # 2x2 visual grid + spectrum row.
            left, right = st.columns(2)
            with left:
                fig = light_curve_plot(lc, "Light curve")
                if fig:
                    fig.update_layout(height=300, title_font_size=15)
                    fig = presentation_plot(fig)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
            with right:
                fig = acf_figure(lc)
                if fig:
                    fig = presentation_plot(fig)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})

            left, right = st.columns(2)
            with left:
                fig = distribution_figure(lc)
                if fig:
                    fig = presentation_plot(fig)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
            with right:
                fig = rolling_variance_figure(lc)
                if fig:
                    fig = presentation_plot(fig)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})

            fig = spectrum_figure(lc)
            if fig:
                fig = presentation_plot(fig)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
        else:
            st.info(
                "No cached light-curve file was found for this KIC. "
                "The pipeline comparison below still works."
            )

        # --------------------------------------------------------------
        # Pipeline result on this star — visual bar, not a raw table.
        # --------------------------------------------------------------
        st.subheader("How did the pipelines do on this star?")
        g = target_rows[
            pd.to_numeric(target_rows["quarter"], errors="coerce") == quarter
        ].copy()

        perf_rows = []
        for p in pipelines:
            c = f"{p}_{metric_suffix}"
            if c in g.columns:
                perf_rows.append(
                    {
                        "Pipeline": pipeline_label(p),
                        "Recovery": 100 * g[c].fillna(False).astype(bool).mean(),
                    }
                )
        perf = pd.DataFrame(perf_rows).sort_values("Recovery", ascending=True)
        if not perf.empty:
            fig = px.bar(
                perf,
                x="Recovery",
                y="Pipeline",
                orientation="h",
                text=perf["Recovery"].map(lambda x: f"{x:.0f}%"),
                labels={"Recovery": "Recovery (%)", "Pipeline": ""},
                title=f"Recovery across the 81 injections for KIC {target}",
            )
            fig.update_traces(
                textposition="outside",
                cliponaxis=False,
            )
            fig.update_layout(
                height=350,
                margin=dict(l=15, r=95, t=55, b=30),
                xaxis_range=[0, min(105, max(10, perf["Recovery"].max() * 1.08))],
            )
            fig = presentation_plot(fig)
            # Restore only the extra right-side breathing room after the shared plot formatter.
            fig.update_layout(
                margin=dict(l=30, r=95, t=58, b=40),
                xaxis_range=[0, min(105, max(10, perf["Recovery"].max() * 1.08))],
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False, "displaylogo": False},
            )

        with st.expander("Detailed numbers for this star", expanded=False):
            detail_rows = []
            for p in pipelines:
                main = f"{p}_{metric_suffix}"
                if main not in g.columns:
                    continue
                detail_rows.append(
                    {
                        "Pipeline": pipeline_label(p),
                        "Recovery %": 100 * g[main].fillna(False).astype(bool).mean(),
                        "Injected P · top-1 %": (
                            100 * g[f"{p}_exact_rank1_matched"].fillna(False).astype(bool).mean()
                            if f"{p}_exact_rank1_matched" in g.columns else np.nan
                        ),
                        "Average runtime (s)": (
                            pd.to_numeric(g[f"{p}_runtime_seconds"], errors="coerce").mean()
                            if f"{p}_runtime_seconds" in g.columns else np.nan
                        ),
                    }
                )
            st.dataframe(pd.DataFrame(detail_rows), hide_index=True, use_container_width=True)

    # ------------------------------------------------------------------
    # Visual method map: what we use now vs what comes next.
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Statistical toolkit")
    render_method_map()


# =============================================================================
# Page 3 — injection explorer
# =============================================================================

elif page.startswith("3"):
    header(
        "Injection explorer",
        "Inspect success, rescue, and failure cases rather than relying only on aggregate recovery rates.",
    )

    pipeline = st.selectbox("Pipeline", pipelines, format_func=pipeline_label)
    raw = raw_baseline_for(pipeline)

    p_col = f"{pipeline}_{metric_suffix}"
    raw_col = f"{raw}_{metric_suffix}"

    view_options = ["All cases"]
    if p_col in injections.columns and raw_col in injections.columns and pipeline != raw:
        view_options += ["Challenger rescue", "Challenger hurts"]
    view = st.radio("Case type", view_options, horizontal=True)

    subset = injections.copy()
    if view == "Challenger rescue":
        subset = subset[
            subset[p_col].fillna(False).astype(bool)
            & ~subset[raw_col].fillna(False).astype(bool)
        ]
    elif view == "Challenger hurts":
        subset = subset[
            ~subset[p_col].fillna(False).astype(bool)
            & subset[raw_col].fillna(False).astype(bool)
        ]

    if subset.empty:
        st.info("No cases match this selection.")
    else:
        ids = sorted(subset["target_id"].map(normalize_target_id).unique())
        target = st.selectbox("Star", ids)
        star_cases = subset[subset["target_id"].map(normalize_target_id) == target].copy()
        star_cases = star_cases.sort_values("case_index" if "case_index" in star_cases.columns else star_cases.columns[0])

        labels = []
        for _, r in star_cases.iterrows():
            labels.append(
                f"case {int(r.get('case_index', 0))} · P={r.get('injected_period_days', np.nan):g} d · "
                f"D={r.get('injected_duration_hours', np.nan):g} h · depth={1e6*r.get('injected_depth', np.nan):.0f} ppm"
            )
        idx = st.selectbox("Injection", range(len(star_cases)), format_func=lambda i: labels[i])
        row = star_cases.iloc[int(idx)]
        quarter = int(row.get("quarter", 5))

        a, b, c, d = st.columns(4)
        a.metric("Injected period", f"{row.get('injected_period_days', np.nan):.4g} d")
        b.metric("Duration", f"{row.get('injected_duration_hours', np.nan):.3g} h")
        c.metric("Depth", f"{1e6*row.get('injected_depth', np.nan):.0f} ppm")

        recovered_period = pd.to_numeric(pd.Series([row.get(f"{pipeline}_recovered_period_days", np.nan)]), errors="coerce").iloc[0]
        d.metric("Recovered period", f"{recovered_period:.5g} d" if np.isfinite(recovered_period) else "—")

        left, right = st.columns([1.3, 1])
        with left:
            lc, lc_path = load_light_curve(str(REPO_ROOT), str(RUN_DIR), target, quarter)
            if not lc.empty:
                fig = light_curve_plot(lc, f"KIC {target} Q{quarter}")
                fig = presentation_plot(fig)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
            else:
                st.info("No local light-curve parquet was found for this star.")

        with right:
            details = []
            for p in pipelines:
                hit_col = f"{p}_{metric_suffix}"
                if hit_col not in row.index:
                    continue
                details.append(
                    {
                        "Pipeline": pipeline_label(p),
                        "Recovered": bool(row[hit_col]) if pd.notna(row[hit_col]) else False,
                        "Score": row.get(f"{p}_score", np.nan),
                        "Recovered P (d)": row.get(f"{p}_recovered_period_days", np.nan),
                        f"Top-{TOP_K_DISPLAY} exact rank": row.get(f"{p}_exact_rank_topk", np.nan),
                    }
                )
            st.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True)

        st.subheader("Background-model preservation diagnostics")
        diagnostics = []
        for branch in ("raw", "arima", "kalman", "gp"):
            diagnostics.append(
                {
                    "Branch": branch.upper(),
                    "Residual depth": row.get(f"{branch}_residual_depth", np.nan),
                    "Local SNR": row.get(f"{branch}_local_snr", np.nan),
                    "Depth retention": row.get(f"{branch}_depth_retention_fraction", np.nan),
                    "SNR retention": row.get(f"{branch}_snr_retention_fraction", np.nan),
                    "Residual ACF1": row.get(f"{branch}_residual_acf1", np.nan),
                }
            )
        st.dataframe(pd.DataFrame(diagnostics), hide_index=True, use_container_width=True)


# =============================================================================
# Page 4 — calibration
# =============================================================================

elif page.startswith("4"):
    header(
        "1% false-alarm calibration",
        "Compare each detector at a controlled false-alarm probability, not at an arbitrary raw score.",
    )

    if thresholds.empty or null_trials.empty:
        st.warning(
            "This run does not currently contain both `fap_thresholds.csv` and `null_trials.csv`. "
            "You can demo the benchmark, but do not claim 1% FAP recovery for this run until calibration is generated."
        )
    else:
        t = harmonized_thresholds(thresholds)
        cal_pipelines = [p for p in pipelines if p in set(t["pipeline"].astype(str))]
        pipeline = st.selectbox("Pipeline", cal_pipelines, format_func=pipeline_label)

        stars = sorted(t[t["pipeline"] == pipeline]["target_id"].map(normalize_target_id).unique())
        target = st.selectbox("Star", stars)
        tr = t[
            (t["pipeline"] == pipeline)
            & (t["target_id"].map(normalize_target_id) == target)
        ]
        if tr.empty:
            st.stop()
        threshold = float(pd.to_numeric(tr.iloc[0]["score_threshold"], errors="coerce"))
        quarter = int(tr.iloc[0].get("quarter", 5))

        score_col = f"{pipeline}_score"
        ns = null_trials[
            null_trials["target_id"].map(normalize_target_id) == target
        ].copy()

        star_inj = injections[
            (injections["target_id"].map(normalize_target_id) == target)
            & (pd.to_numeric(injections["quarter"], errors="coerce") == quarter)
        ].copy()

        fig = go.Figure()
        if score_col in ns.columns:
            null_scores = pd.to_numeric(ns[score_col], errors="coerce").dropna()
            fig.add_trace(
                go.Histogram(
                    x=null_scores,
                    name="Null scores",
                    opacity=0.65,
                    histnorm="probability density",
                    nbinsx=35,
                )
            )
        if score_col in star_inj.columns:
            inj_scores = pd.to_numeric(star_inj[score_col], errors="coerce").dropna()
            fig.add_trace(
                go.Histogram(
                    x=inj_scores,
                    name="Injected scores",
                    opacity=0.55,
                    histnorm="probability density",
                    nbinsx=35,
                )
            )
        fig.add_vline(x=threshold, line_dash="dash", annotation_text="1% FAP threshold")
        fig.update_layout(
            barmode="overlay",
            height=330,
            title=f"KIC {target} Q{quarter} · {pipeline_label(pipeline)} · null vs injected score",
            xaxis_title="Detection statistic / score",
            yaxis_title="Density",
        )
        fig = presentation_plot(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})

        rec_col = f"{pipeline}_fap01_harmonic_recovered"
        det_col = f"{pipeline}_fap01_detected"
        harmonic_col = f"{pipeline}_harmonic_rank1_matched"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("1% FAP threshold", f"{threshold:.4g}")
        if det_col in star_inj.columns:
            m2.metric("Scores above threshold", pct(star_inj[det_col].fillna(False).astype(bool).mean()))
        if harmonic_col in star_inj.columns:
            m3.metric("Correct period", pct(star_inj[harmonic_col].fillna(False).astype(bool).mean()))
        if rec_col in star_inj.columns:
            m4.metric("Recovery @ 1% FAP", pct(star_inj[rec_col].fillna(False).astype(bool).mean()))

        st.latex(
            r"R_{1\%} = \frac{\#\{\mathrm{correct\ harmonic\ period}\ \cap\ S > T_{0.99,\ null}\}}{N_{\mathrm{injections}}}"
        )

        st.subheader("Threshold table")
        display_t = t.copy()
        display_t = display_t[display_t["pipeline"].isin(cal_pipelines)]
        st.dataframe(display_t, use_container_width=True, hide_index=True)


# =============================================================================
# Page 5 — future path / POC
# =============================================================================

elif page.startswith("5"):
    header(
        "Future path / proof of concept",
        "A staged roadmap from interpretable statistical characterization and calibrated detectors to adaptive model selection, evidence fusion, and later learned representations.",
    )

    st.markdown(
        """
        <div class="science-callout">
        <b>POC hypothesis:</b> no single background model, whitening strategy, or transit detector should be assumed
        uniformly optimal. The future system should first understand the statistical regime of a light curve,
        then preserve the transit signal while suppressing the relevant background structure, and finally compare
        all candidate evidence at a matched false-alarm probability.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Current capabilities
    # -------------------------------------------------------------------------
    st.subheader("What exists now")
    current = pd.DataFrame(
        [
            ["Statistical characterization", "ACF / correlation timescale; ADF & KPSS; variance drift; spectral peak/strength; skewness/kurtosis; gap fraction", "Establishes that the background is not one homogeneous noise process and provides candidate routing features."],
            ["Raw branch", "Normalized Kepler PDCSAP flux", "Reference branch: establishes what can already be recovered without an added background model."],
            ["ARIMA", "Autoregressive / moving-average innovations", "Targets predictable short-memory temporal dependence and tests whether innovations expose transit edges."],
            ["Kalman / state space", "Local-level one-step residual", "Models an evolving latent background and naturally supports missing observations / gaps."],
            ["Gaussian Process", "Smooth correlated-background estimate and subtraction", "Flexible nonlinear correlated-noise model; current RBF case tests timescale separation from the transit."],
            ["BLS", "Periodic box least-squares search", "Strong geometric baseline for approximately box-shaped transits."],
            ["TCF", "Transit Comb Filter / edge-comb detector", "Designed to exploit ingress/egress-like structure that can remain after ARIMA-style transformations."],
            ["Injection–recovery", "Known synthetic transit grid injected into real Kepler backgrounds", "Provides controlled ground truth for completeness and comparative performance."],
            ["Null calibration", "Per-star/per-pipeline moving-block-surrogate nulls and empirical thresholds", "Lets pipelines be compared at a matched headline FAP such as 1%, instead of comparing incomparable raw score scales."],
        ],
        columns=["Capability", "Current implementation", "Scientific motivation"],
    )
    st.dataframe(current, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # Expanded statistical characterization
    # -------------------------------------------------------------------------
    st.subheader("1 · Expand statistical characterization")
    characterization_future = pd.DataFrame(
        [
            ["ACF + PACF", "Short/medium-lag temporal dependence and candidate AR order", "Separate simple autoregressive memory from smoother long-timescale correlation."],
            ["ADF + KPSS + rolling stationarity checks", "Global and local stationarity / trend behavior", "Avoid forcing one stationary model on a light curve whose statistical structure evolves."],
            ["FFT / PSD", "Frequency-domain power distribution", "Identify colored-noise structure, dominant frequencies, and broad spectral slopes; also motivates frequency-domain whitening/filter design."],
            ["Lomb–Scargle periodogram", "Periodic structure with irregular/missing cadence support", "Detect stellar/instrumental periodicities without first filling gaps onto a perfectly uniform grid."],
            ["Wavelet / time-frequency characterization", "How spectral content changes with time", "Kepler backgrounds can be nonstationary; a global PSD can hide time-localized changes."],
            ["Variance drift / rolling robust scale", "Heteroskedasticity and changing noise amplitude", "Tests whether a fixed-variance detector/background model is inappropriate."],
            ["Skewness, kurtosis, tail metrics / robust quantiles", "Non-Gaussianity and outliers", "Distinguish Gaussian-like noise from heavy-tailed or asymmetric regimes where least-squares assumptions can degrade."],
            ["Gap fraction + gap-length distribution", "Missing-data structure", "Quantify where interpolation-sensitive methods may fail and where state-space/GP handling may be advantageous."],
            ["Long-memory / Hurst-style descriptors", "Persistence beyond short AR lags", "Separate genuine long-range dependence from simple short-memory correlation."],
            ["Change-point / regime-shift metrics", "Abrupt statistical changes", "Flag light curves where a single global background model is structurally wrong."],
            ["Local morphology metrics", "Sharp excursions, transit-like edges, flare-like asymmetry", "Help prevent a background model from treating the transit itself as noise to be removed."],
            ["Instrumental contamination / quality diagnostics", "Known artifacts, discontinuities, quality-flag patterns", "Keep astrophysical variability and instrument behavior from being conflated."],
        ],
        columns=["Feature family", "What it measures", "Why it enters the POC"],
    )
    st.dataframe(characterization_future, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # Background / whitening branches
    # -------------------------------------------------------------------------
    st.subheader("2 · Expand background and whitening models")
    background_future = pd.DataFrame(
        [
            ["Kepler TPS-style wavelet whitening", "Whitening / reference", "Provides the most important Kepler-style comparator and addresses nonstationary colored noise with time-scale-local noise estimates.", "High"],
            ["TPS matched-filter reference", "Detection reference", "Creates a direct bridge from this POC to the Kepler detection philosophy rather than benchmarking only against BLS/TCF.", "High"],
            ["ARIMA + wavelet hybrid", "Hybrid background + whitening", "Tests whether short-memory autoregressive structure and time-localized colored noise are complementary.", "High"],
            ["Richer Kalman / state-space models", "Background model", "Local-linear-trend, AR-state, seasonal or stochastic components can represent evolving latent backgrounds beyond a local-level model.", "Medium"],
            ["GP kernel family", "Background model", "Compare RBF, Matérn and quasi-periodic kernels instead of assuming one covariance structure is universal.", "High"],
            ["Celerite / scalable GP variants", "Computational capability", "Makes richer correlated-noise models practical when moving from tens of stars toward much larger Kepler samples.", "Medium"],
            ["Frequency-domain / FFT whitening", "Whitening challenger", "Uses the measured PSD to suppress stationary colored noise efficiently; useful as a simpler contrast to wavelet whitening.", "Medium"],
            ["Robust / heavy-tail background likelihoods", "Noise model", "Reduce sensitivity to outliers and non-Gaussian tails that can distort Gaussian least-squares fits.", "Later"],
        ],
        columns=["Method", "Role", "Why add it", "POC priority"],
    )
    st.dataframe(background_future, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # Transit detection branches
    # -------------------------------------------------------------------------
    st.subheader("3 · Expand transit detectors")
    detector_future = pd.DataFrame(
        [
            ["BLS", "Retain as baseline", "Simple, interpretable, efficient detector for box-like transit geometry."],
            ["TCF", "Retain as transformed-domain detector", "Particularly relevant after ARIMA-like transformations that emphasize ingress/egress edges."],
            ["TLS", "Add physical-template detector", "Uses realistic limb-darkened transit shapes and tests whether physical morphology recovers cases BLS/TCF miss."],
            ["TPS / wavelet matched filter", "Add Kepler-style detector", "Tests a duration-template matched filter under adaptive whitening and provides a scientifically meaningful reference."],
            ["Generic matched filter after ARIMA / GP / Kalman", "Unify branch comparison", "Allows the same core detection statistic to be compared after different background representations."],
            ["Multi-duration / multi-template bank", "Detection capability", "Avoids assuming one transit duration or morphology and mirrors how a real search scans a family of templates."],
            ["Harmonic-aware candidate logic", "Candidate validation", "Prevents P/2 or 2P aliases from being counted as unrelated failures while still distinguishing exact from harmonic recovery."],
        ],
        columns=["Detector / capability", "Role", "Scientific motivation"],
    )
    st.dataframe(detector_future, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # Router / machine learning
    # -------------------------------------------------------------------------
    st.subheader("4 · Adaptive model selection: XGBoost and interpretable routers")
    router = pd.DataFrame(
        [
            ["Logistic regression / linear baseline", "Predict whether a challenger improves over raw", "Establish whether the relationship between statistical descriptors and model benefit is already simple/interpretable."],
            ["Decision tree", "Interpretable regime rules", "Produces human-readable thresholds such as 'high ACF + low gap fraction → prefer branch X'."],
            ["Random forest", "Nonlinear ensemble baseline", "Captures interactions among statistical features while remaining relatively robust and inspectable."],
            ["XGBoost", "Primary tabular router candidate", "Well suited to heterogeneous statistical features, nonlinear interactions, missing values and moderate sample sizes; can predict best branch or expected Δ recovery."],
            ["Probability calibration", "Router calibration", "The router should output calibrated confidence / expected benefit rather than an unqualified hard class."],
            ["Held-out-star validation", "Leakage control", "Entire stars—not injections from the same star—must be held out so the router generalizes to new light curves."],
        ],
        columns=["Model / capability", "Purpose", "Why it is useful"],
    )
    st.dataframe(router, use_container_width=True, hide_index=True)

    st.latex(
        r"X_i=[\mathrm{ACF}_1,\tau_{\mathrm{ACF}},p_{\mathrm{ADF}},p_{\mathrm{KPSS}},"
        r"\mathrm{PSD},\mathrm{variance\ drift},\mathrm{skew},\mathrm{kurtosis},"
        r"\mathrm{gap\ metrics},\ldots]"
    )
    st.latex(
        r"X_i \longrightarrow "
        r"[\Delta R_{\mathrm{ARIMA}},\Delta R_{\mathrm{Kalman}},\Delta R_{\mathrm{GP}},"
        r"\Delta R_{\mathrm{TPS/wavelet}},\ldots]"
    )
    st.caption(
        "The router can either choose the expected best branch or decide that several branches should be retained for evidence fusion."
    )

    # -------------------------------------------------------------------------
    # Neural networks / learned representations
    # -------------------------------------------------------------------------
    st.subheader("5 · Neural models — later POC challengers")
    neural = pd.DataFrame(
        [
            ["1D CNN", "Local morphology learner", "Can learn ingress/egress and local transit shapes directly from flux windows once there is enough training data."],
            ["TCN / dilated CNN", "Longer temporal context", "Extends convolutional models to wider receptive fields without losing temporal ordering."],
            ["Autoencoder / self-supervised encoder", "Representation learning", "Can learn background/variability embeddings from abundant unlabeled light curves before supervised transit training."],
            ["Transformer / attention", "Long-range context", "Potentially useful for long-range dependencies and context-dependent events, but computationally heavier and data hungry."],
            ["Mixture-of-experts / gated neural router", "Adaptive ensemble", "A future learned analogue of the statistical router: select or weight specialized experts based on the light curve."],
        ],
        columns=["Neural method", "Possible role", "Why / when it becomes useful"],
    )
    st.dataframe(neural, use_container_width=True, hide_index=True)
    st.info(
        "Neural models should come after the calibrated statistical baselines. Otherwise it becomes difficult to tell whether extra complexity is learning real transit evidence, background shortcuts, or star-specific leakage."
    )

    # -------------------------------------------------------------------------
    # Calibration / ensemble / validation
    # -------------------------------------------------------------------------
    st.subheader("6 · Calibration, evidence fusion, and validation")
    validation_future = pd.DataFrame(
        [
            ["End-to-end null refit / reselection", "Calibration", "Refit/reselect background models inside null trials so model-selection uncertainty is included in the false-alarm distribution."],
            ["Multiple FAP operating points", "Calibration", "Report 1%, 0.1% and potentially lower FAP operating points to show the completeness–reliability trade-off."],
            ["Calibrated evidence fusion", "Ensemble", "Combine branch evidence only after scores are placed on comparable false-alarm/probability scales."],
            ["Unique-recovery / overlap analysis", "Complementarity", "Quantify whether each new method adds genuinely new detections rather than duplicating existing successes."],
            ["Injection grid expansion", "Completeness", "Broaden period, depth, duration and stellar-regime coverage to map where each branch succeeds or fails."],
            ["Held-out-star and held-out-regime tests", "Generalization", "Demonstrate that any router/ensemble works on unseen stars and does not simply memorize the 50-star benchmark."],
            ["Known KOI / TCE recovery cross-check", "External validation", "After controlled clean-background injections, test behavior on real known transit signals and candidate populations."],
            ["Runtime / scalability profiling", "Engineering", "A scientifically better method still needs a tractable path from POC to Kepler-scale processing."],
        ],
        columns=["Capability", "Role", "Why it matters"],
    )
    st.dataframe(validation_future, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # Architecture
    # -------------------------------------------------------------------------
    st.subheader("Target POC architecture")
    st.code(
        """
Kepler pixels / light curve
          │
          ▼
preprocessing + quality handling + explicit gaps
          │
          ▼
statistical characterization Xᵢ
  ACF/PACF · ADF/KPSS · FFT/PSD · Lomb–Scargle
  wavelets · variance drift · tails · gaps · morphology
          │
          ├────────────┬────────────┬────────────┬──────────────┬───────────────┐
          ▼            ▼            ▼            ▼              ▼               ▼
         RAW         ARIMA        Kalman          GP         FFT/PSD       TPS/wavelet
          │            │            │             │          whitening       whitening
          │            └────── hybrid ARIMA + wavelet ─────────┘               │
          │                                                                      │
          ├──────── BLS ─────────────────────────────────────────────────────────┤
          ├──────── TCF ─────────────────────────────────────────────────────────┤
          ├──────── TLS ─────────────────────────────────────────────────────────┤
          └──────── matched-filter / TPS template bank ──────────────────────────┘
                                      │
                                      ▼
                          per-pipeline null calibration
                                      │
                                      ▼
                         calibrated candidate evidence
                                      │
                       ┌──────────────┴──────────────┐
                       ▼                             ▼
        interpretable / XGBoost router       calibrated evidence fusion
                       │                             │
                       └──────────────┬──────────────┘
                                      ▼
                         adaptive transit-search POC
                                      │
                         ┌────────────┴─────────────┐
                         ▼                          ▼
                 held-out-star tests        neural challengers
                                        CNN / TCN / attention /
                                          mixture-of-experts
        """
    )

    # -------------------------------------------------------------------------
    # Staged roadmap
    # -------------------------------------------------------------------------
    st.subheader("Suggested implementation order")
    roadmap = pd.DataFrame(
        [
            [1, "Finish 50-star benchmark + 1% FAP calibration", "Lock the empirical baseline and regime-dependent result."],
            [2, "TPS/wavelet comparator + TLS", "Add the two most scientifically important reference/challenger methods."],
            [3, "Expand characterization with PSD/FFT, PACF, wavelets, tails and gap descriptors", "Build a richer explanatory feature space before training a router."],
            [4, "ARIMA+wavelet, richer state-space, GP kernel comparison", "Test whether complementary background assumptions improve weak-signal recovery."],
            [5, "XGBoost + interpretable router on held-out stars", "Test the adaptive-selection hypothesis directly."],
            [6, "Calibrated evidence fusion", "Exploit unique recoveries when several branches are useful."],
            [7, "Neural challengers", "Only after a sufficiently large, leakage-controlled benchmark exists."],
            [8, "Scale + external validation", "Move from POC evidence to larger Kepler samples / known transit populations."],
        ],
        columns=["Stage", "Next capability", "Decision it answers"],
    )
    st.dataframe(roadmap, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Core future-path message for the demo**

        The objective is not to keep adding models indefinitely. Each addition tests a specific statistical hypothesis:

        **characterize the background → choose/construct the appropriate representation → detect with complementary
        transit templates → calibrate each path at a matched FAP → learn when each path helps → fuse only genuinely
        complementary evidence → validate on unseen stars.**
        """
    )


# =============================================================================
# Footer
# =============================================================================

st.divider()
st.caption(
    "Research prototype. Recovery, completeness, and false-alarm claims should be interpreted only for the "
    "selected experiment configuration and calibration procedure."
)
