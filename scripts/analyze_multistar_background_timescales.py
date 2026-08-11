"""Join cheap stellar-background time-scale features to the 50-star BLS/TCF benchmark."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.preprocessing.normalization import longest_contiguous_segment, preprocess_pdcsap_light_curve
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MULTISTAR_DIR = PROJECT_ROOT / "outputs/experiments/multistar_bls_tcf/optimized"
MANIFEST_PATH = MULTISTAR_DIR / "metrics/target_manifest_used.csv"
INJECTION_PATH = MULTISTAR_DIR / "metrics/multistar_bls_tcf_injections.csv"
STAR_SUMMARY_PATH = MULTISTAR_DIR / "metrics/multistar_star_summary.csv"
CACHE_DIR = PROJECT_ROOT / "outputs/cache/kepler_light_curves"
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/multistar_background_timescale"

def default_settings():
    return SimpleNamespace(manifest_path=MANIFEST_PATH, injection_path=INJECTION_PATH, star_summary_path=STAR_SUMMARY_PATH, cache_dir=CACHE_DIR, output_dir=OUTPUT_DIR, quality_policy="default", require_finite_flux_error=False, test_fraction=0.20, max_acf_tau_days=30.0, rolling_background_window_days=1.0, allow_download=False)

def parse_args():
    defaults = default_settings()
    parser = argparse.ArgumentParser(description="Analyze whether cheap background time-scale features predict BLS/TCF wins across stars.")
    parser.add_argument("--manifest-path", type=Path, default=defaults.manifest_path)
    parser.add_argument("--injection-path", type=Path, default=defaults.injection_path)
    parser.add_argument("--star-summary-path", type=Path, default=defaults.star_summary_path)
    parser.add_argument("--cache-dir", type=Path, default=defaults.cache_dir)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--max-acf-tau-days", type=float, default=defaults.max_acf_tau_days)
    parser.add_argument("--rolling-background-window-days", type=float, default=defaults.rolling_background_window_days)
    parser.add_argument("--allow-download", action="store_true")
    parsed = parser.parse_args()
    defaults.manifest_path = Path(parsed.manifest_path)
    defaults.injection_path = Path(parsed.injection_path)
    defaults.star_summary_path = Path(parsed.star_summary_path)
    defaults.cache_dir = Path(parsed.cache_dir)
    defaults.output_dir = Path(parsed.output_dir)
    defaults.max_acf_tau_days = float(parsed.max_acf_tau_days)
    defaults.rolling_background_window_days = float(parsed.rolling_background_window_days)
    defaults.allow_download = bool(parsed.allow_download)
    return defaults

def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def normalize_target_id(value):
    return str(value).replace("KIC", "").replace("kic", "").strip()

def cache_path(args, target_id, quarter):
    return Path(args.cache_dir) / f"kic_{normalize_target_id(target_id)}_q{int(quarter)}_pdcsap.parquet"

def load_frame(args, target_id, quarter):
    path = cache_path(args, target_id, quarter)
    if path.exists():
        return pd.read_parquet(path), True
    if not args.allow_download:
        raise FileNotFoundError(f"Cached light curve is missing: {path}. Rerun with --allow-download to fetch it.")
    light_curve = load_kepler_pdcsap(target_id, quarter)
    frame = light_curve.to_dataframe()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame, False

def robust_scale(values):
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        return float("nan")
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    return float(1.4826 * mad if mad > 0 else np.std(clean, ddof=1))

def first_crossing_lag_days(acf_values, cadence_days, threshold):
    for lag in range(1, len(acf_values)):
        value = float(acf_values[lag])
        if np.isfinite(value) and value <= float(threshold):
            return float(lag * cadence_days)
    return float("nan")

def integrated_positive_acf_days(acf_values, cadence_days):
    positive = []
    for value in acf_values[1:]:
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            break
        positive.append(value)
    return float(cadence_days * (1.0 + 2.0 * np.sum(positive))) if positive else float(cadence_days)

def acf_at_days(acf_values, cadence_days, days):
    lag = int(round(float(days) / float(cadence_days))) if cadence_days > 0 else 0
    if lag <= 0 or lag >= len(acf_values):
        return float("nan")
    return float(acf_values[lag])

def rolling_background_features(values, cadence_days, args):
    window = max(3, int(round(float(args.rolling_background_window_days) / float(cadence_days)))) if cadence_days > 0 else 3
    if window % 2 == 0:
        window += 1
    series = pd.Series(np.asarray(values, dtype=float))
    trend = series.rolling(window=window, center=True, min_periods=max(3, window // 2)).median().to_numpy(dtype=float)
    residual = np.asarray(values, dtype=float) - trend
    trend_scatter = robust_scale(trend)
    residual_scatter = robust_scale(residual)
    return {"rolling_background_window_cadences": int(window), "rolling_background_scatter_ppm": float(trend_scatter * 1.0e6), "rolling_short_residual_scatter_ppm": float(residual_scatter * 1.0e6), "rolling_background_to_short_scatter_ratio": float(trend_scatter / residual_scatter) if np.isfinite(trend_scatter) and np.isfinite(residual_scatter) and residual_scatter > 0 else float("nan")}

def star_background_features(row, args):
    target_id = normalize_target_id(row["target_id"])
    quarter = int(row["quarter"])
    raw, cache_hit = load_frame(args, target_id, quarter)
    regular, summary = preprocess_pdcsap_light_curve(raw, quality_policy=args.quality_policy, require_finite_flux_error=args.require_finite_flux_error, normalization_fit_fraction=1.0 - args.test_fraction)
    segment = longest_contiguous_segment(regular)
    time = segment["time"].to_numpy(dtype=float)
    flux = segment["normalized_flux"].to_numpy(dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    time = time[finite]
    flux = flux[finite]
    if len(flux) < 24:
        raise ValueError(f"Not enough finite points to estimate background features for KIC {target_id} Q{quarter}.")
    cadence_days = float(np.nanmedian(np.diff(time)))
    max_lag = max(2, min(len(flux) - 2, int(round(float(args.max_acf_tau_days) / cadence_days))))
    acf_values = acf(flux - np.nanmedian(flux), nlags=max_lag, fft=True, missing="none")
    tau_e = first_crossing_lag_days(acf_values, cadence_days, np.exp(-1.0))
    tau_half = first_crossing_lag_days(acf_values, cadence_days, 0.5)
    tau_zero = first_crossing_lag_days(acf_values, cadence_days, 0.0)
    integrated_tau = integrated_positive_acf_days(acf_values, cadence_days)
    low_frequency = rolling_background_features(flux, cadence_days, args)
    fields = {"target_id": target_id, "quarter": quarter, "selection_group": row.get("selection_group", ""), "light_curve_cache_hit": bool(cache_hit), "n_grid_observations": int(len(regular)), "n_usable_observations": int(regular["usable"].sum()), "longest_segment_length": int(len(segment)), "longest_segment_days": float(np.nanmax(time) - np.nanmin(time)), "median_cadence_days": cadence_days, "gap_fraction": float(1.0 - regular["usable"].sum() / len(regular)), "robust_flux_scatter_ppm": float(robust_scale(regular.loc[regular["usable"], "normalized_flux"].to_numpy(dtype=float)) * 1.0e6), "background_tau_acf_e_days": tau_e, "background_tau_acf_half_days": tau_half, "background_tau_acf_zero_days": tau_zero, "background_tau_integrated_positive_acf_days": integrated_tau, "background_acf_lag_1": acf_at_days(acf_values, cadence_days, cadence_days), "background_acf_6h": acf_at_days(acf_values, cadence_days, 0.25), "background_acf_1d": acf_at_days(acf_values, cadence_days, 1.0), "background_acf_2d": acf_at_days(acf_values, cadence_days, 2.0), "background_acf_5d": acf_at_days(acf_values, cadence_days, 5.0), "acf_max_lag_days": float(max_lag * cadence_days), "preprocessing_n_row_absent": int(summary.n_row_absent), "preprocessing_n_unusable_observed": int(summary.n_unusable_observed)}
    fields.update(low_frequency)
    return fields

def classify_winner(row, prefix):
    both = bool(row[f"{prefix}_both"])
    tcf_only = bool(row[f"{prefix}_tcf_only"])
    bls_only = bool(row[f"{prefix}_bls_only"])
    neither = bool(row[f"{prefix}_neither"])
    if both:
        return "both"
    if tcf_only:
        return "tcf_only"
    if bls_only:
        return "bls_only"
    if neither:
        return "neither"
    return "unknown"

def add_ratio_features(frame):
    result = frame.copy()
    duration_days = result["injected_duration_hours"].astype(float) / 24.0
    for tau_column in ("background_tau_acf_e_days", "background_tau_acf_half_days", "background_tau_integrated_positive_acf_days"):
        ratio_name = tau_column.replace("background_tau_", "background_to_transit_")
        inverse_name = tau_column.replace("background_tau_", "transit_to_background_")
        result[ratio_name] = result[tau_column].astype(float) / duration_days
        result[inverse_name] = duration_days / result[tau_column].astype(float)
    result["harmonic_winner"] = result.apply(lambda row: classify_winner(row, "harmonic"), axis=1)
    result["exact_winner"] = result.apply(lambda row: classify_winner(row, "exact"), axis=1)
    result["harmonic_bls_recovered"] = result["bls_harmonic_recovered"].astype(bool)
    result["harmonic_tcf_recovered"] = result["tcf_harmonic_recovered"].astype(bool)
    result["exact_bls_recovered"] = result["bls_exact_recovered"].astype(bool)
    result["exact_tcf_recovered"] = result["tcf_exact_recovered"].astype(bool)
    result["harmonic_tcf_beats_bls"] = result["harmonic_tcf_only"].astype(bool)
    result["harmonic_bls_beats_tcf"] = result["harmonic_bls_only"].astype(bool)
    result["harmonic_any_single_winner"] = result["harmonic_tcf_only"].astype(bool) | result["harmonic_bls_only"].astype(bool)
    return result

def qcut_labels(series, bins=4):
    clean = pd.to_numeric(series, errors="coerce")
    try:
        return pd.qcut(clean, q=int(bins), duplicates="drop")
    except ValueError:
        return pd.Series(["all"] * len(clean), index=series.index)

def outcome_summary(frame, group_column):
    grouped = frame.groupby(group_column, observed=False)
    return grouped.agg(injection_count=("success", "size"), star_count=("target_id", "nunique"), median_background_tau_acf_e_days=("background_tau_acf_e_days", "median"), median_background_to_transit_acf_e=("background_to_transit_acf_e_days", "median"), harmonic_bls_recovery_rate=("harmonic_bls_recovered", "mean"), harmonic_tcf_recovery_rate=("harmonic_tcf_recovered", "mean"), harmonic_union_recovery_rate=("harmonic_union", "mean"), harmonic_both_rate=("harmonic_both", "mean"), harmonic_bls_only_rate=("harmonic_bls_only", "mean"), harmonic_tcf_only_rate=("harmonic_tcf_only", "mean"), harmonic_neither_rate=("harmonic_neither", "mean"), exact_union_recovery_rate=("exact_union", "mean")).reset_index()

def star_outcome_summary(frame):
    grouped = frame.groupby(["target_id", "quarter"], as_index=False)
    return grouped.agg(injection_count=("success", "size"), harmonic_bls_recovery_rate=("harmonic_bls_recovered", "mean"), harmonic_tcf_recovery_rate=("harmonic_tcf_recovered", "mean"), harmonic_union_recovery_rate=("harmonic_union", "mean"), harmonic_bls_only_rate=("harmonic_bls_only", "mean"), harmonic_tcf_only_rate=("harmonic_tcf_only", "mean"), harmonic_neither_rate=("harmonic_neither", "mean"), exact_union_recovery_rate=("exact_union", "mean"), median_background_tau_acf_e_days=("background_tau_acf_e_days", "median"), median_background_to_transit_acf_e=("background_to_transit_acf_e_days", "median"), robust_flux_scatter_ppm=("robust_flux_scatter_ppm", "median"), gap_fraction=("gap_fraction", "median"), rolling_background_to_short_scatter_ratio=("rolling_background_to_short_scatter_ratio", "median"))

def correlation_table(frame, feature_columns, outcome_columns, level):
    rows = []
    for feature in feature_columns:
        for outcome in outcome_columns:
            subset = frame[[feature, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(subset) < 5 or subset[feature].nunique() < 2 or subset[outcome].nunique() < 2:
                spearman = float("nan")
                pearson = float("nan")
            else:
                spearman = float(subset[feature].corr(subset[outcome], method="spearman"))
                pearson = float(subset[feature].corr(subset[outcome], method="pearson"))
            rows.append({"level": level, "feature": feature, "outcome": outcome, "n": int(len(subset)), "spearman_correlation": spearman, "pearson_correlation": pearson})
    return pd.DataFrame(rows).sort_values("spearman_correlation", key=lambda item: item.abs(), ascending=False).reset_index(drop=True)

def plot_ratio_bins(by_ratio, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
    x = np.arange(len(by_ratio))
    labels = [str(value) for value in by_ratio["ratio_bin"]]
    axes[0][0].plot(x, by_ratio["harmonic_bls_recovery_rate"], marker="o", label="BLS")
    axes[0][0].plot(x, by_ratio["harmonic_tcf_recovery_rate"], marker="o", label="TCF")
    axes[0][0].plot(x, by_ratio["harmonic_union_recovery_rate"], marker="o", label="Union")
    axes[0][0].set_ylabel("harmonic recovery rate")
    axes[0][0].set_xticks(x)
    axes[0][0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0][0].legend()
    axes[0][1].plot(x, by_ratio["harmonic_bls_only_rate"], marker="o", label="BLS only")
    axes[0][1].plot(x, by_ratio["harmonic_tcf_only_rate"], marker="o", label="TCF only")
    axes[0][1].plot(x, by_ratio["harmonic_neither_rate"], marker="o", label="Neither")
    axes[0][1].set_ylabel("outcome fraction")
    axes[0][1].set_xticks(x)
    axes[0][1].set_xticklabels(labels, rotation=30, ha="right")
    axes[0][1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def run_analysis(args):
    manifest = pd.read_csv(args.manifest_path)
    injections = pd.read_csv(args.injection_path)
    star_summary = pd.read_csv(args.star_summary_path) if Path(args.star_summary_path).exists() else pd.DataFrame()
    feature_rows = [star_background_features(row, args) for row in manifest.to_dict(orient="records")]
    features = pd.DataFrame(feature_rows)
    injections["target_id"] = injections["target_id"].map(normalize_target_id)
    features["target_id"] = features["target_id"].map(normalize_target_id)
    joined = injections.merge(features, on=["target_id", "quarter", "selection_group"], how="left", validate="many_to_one", suffixes=("_saved", ""))
    joined = add_ratio_features(joined)
    joined["ratio_bin"] = qcut_labels(joined["background_to_transit_acf_e_days"], bins=4)
    joined["acf_half_ratio_bin"] = qcut_labels(joined["background_to_transit_acf_half_days"], bins=4)
    joined["integrated_acf_ratio_bin"] = qcut_labels(joined["background_to_transit_integrated_positive_acf_days"], bins=4)
    joined["tau_bin"] = qcut_labels(joined["background_tau_acf_e_days"], bins=4)
    by_ratio = outcome_summary(joined, "ratio_bin")
    by_acf_half_ratio = outcome_summary(joined, "acf_half_ratio_bin")
    by_integrated_ratio = outcome_summary(joined, "integrated_acf_ratio_bin")
    by_tau = outcome_summary(joined, "tau_bin")
    by_depth = outcome_summary(joined, "injected_depth")
    by_duration = outcome_summary(joined, "injected_duration_hours")
    by_period = outcome_summary(joined, "injected_period_days")
    by_star = star_outcome_summary(joined)
    feature_columns = ["background_tau_acf_e_days", "background_tau_acf_half_days", "background_tau_integrated_positive_acf_days", "background_to_transit_acf_e_days", "background_to_transit_acf_half_days", "background_to_transit_integrated_positive_acf_days", "robust_flux_scatter_ppm", "gap_fraction", "background_acf_6h", "background_acf_1d", "background_acf_2d", "rolling_background_to_short_scatter_ratio"]
    row_outcomes = ["harmonic_bls_recovered", "harmonic_tcf_recovered", "harmonic_union", "harmonic_bls_only", "harmonic_tcf_only", "harmonic_neither", "exact_union"]
    star_features = ["median_background_tau_acf_e_days", "median_background_to_transit_acf_e", "robust_flux_scatter_ppm", "gap_fraction", "rolling_background_to_short_scatter_ratio"]
    star_outcomes = ["harmonic_bls_recovery_rate", "harmonic_tcf_recovery_rate", "harmonic_union_recovery_rate", "harmonic_bls_only_rate", "harmonic_tcf_only_rate", "harmonic_neither_rate", "exact_union_recovery_rate"]
    row_correlations = correlation_table(joined, feature_columns, row_outcomes, "injection_row")
    star_correlations = correlation_table(by_star, star_features, star_outcomes, "star")
    correlations = pd.concat([row_correlations, star_correlations], ignore_index=True)
    summary = {"input_injection_path": str(args.injection_path), "input_manifest_path": str(args.manifest_path), "target_count": int(len(features)), "injection_count": int(len(joined)), "feature_policy": "ACF features are estimated on the longest contiguous usable normalized-flux segment after existing preprocessing; no gap interpolation is used.", "primary_ratio": "background_tau_acf_e_days / transit_duration_days", "harmonic_bls_recovery_rate": float(joined["harmonic_bls_recovered"].mean()), "harmonic_tcf_recovery_rate": float(joined["harmonic_tcf_recovered"].mean()), "harmonic_union_recovery_rate": float(joined["harmonic_union"].mean()), "median_background_tau_acf_e_days": float(features["background_tau_acf_e_days"].median()), "median_background_to_transit_acf_e": float(joined["background_to_transit_acf_e_days"].median()), "top_absolute_row_correlations": row_correlations.head(10).to_dict(orient="records"), "top_absolute_star_correlations": star_correlations.head(10).to_dict(orient="records")}
    return features, joined, by_ratio, by_acf_half_ratio, by_integrated_ratio, by_tau, by_depth, by_duration, by_period, by_star, correlations, star_summary, summary

def main(args=None):
    args = args or default_settings()
    metrics_dir = Path(args.output_dir) / "metrics"
    figures_dir = Path(args.output_dir) / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    features, joined, by_ratio, by_acf_half_ratio, by_integrated_ratio, by_tau, by_depth, by_duration, by_period, by_star, correlations, star_summary, summary = run_analysis(args)
    features_path = metrics_dir / "multistar_background_timescale_features.csv"
    joined_path = metrics_dir / "multistar_injections_with_background_timescales.csv"
    by_ratio_path = metrics_dir / "multistar_background_outcomes_by_ratio_bin.csv"
    summary_path = metrics_dir / "multistar_background_timescale_summary.json"
    features.to_csv(features_path, index=False)
    joined.to_csv(joined_path, index=False)
    by_ratio.to_csv(by_ratio_path, index=False)
    by_acf_half_ratio.to_csv(metrics_dir / "multistar_background_outcomes_by_acf_half_ratio_bin.csv", index=False)
    by_integrated_ratio.to_csv(metrics_dir / "multistar_background_outcomes_by_integrated_acf_ratio_bin.csv", index=False)
    by_tau.to_csv(metrics_dir / "multistar_background_outcomes_by_tau_bin.csv", index=False)
    by_depth.to_csv(metrics_dir / "multistar_background_outcomes_by_depth.csv", index=False)
    by_duration.to_csv(metrics_dir / "multistar_background_outcomes_by_duration.csv", index=False)
    by_period.to_csv(metrics_dir / "multistar_background_outcomes_by_period.csv", index=False)
    by_star.to_csv(metrics_dir / "multistar_background_outcomes_by_star.csv", index=False)
    correlations.to_csv(metrics_dir / "multistar_background_feature_correlations.csv", index=False)
    if not star_summary.empty:
        star_summary.to_csv(metrics_dir / "multistar_background_input_star_summary_snapshot.csv", index=False)
    summary_path.write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    plot_ratio_bins(by_ratio, figures_dir / "multistar_background_ratio_bin_recovery.png")
    print(f"Background features: {features_path}")
    print(f"Joined injection table: {joined_path}")
    print(f"Ratio-bin outcomes: {by_ratio_path}")
    print(f"Summary: {summary_path}")
    print(by_ratio.to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
