"""Relate multistar light-curve characterization to detector/background gains."""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

from adaptive_transit.data.kepler_io import load_kepler_pdcsap
from adaptive_transit.noise_models.characterization import (
    characterize_regularized_light_curve,
    json_ready,
)
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "outputs/experiments/multistar_challenger_benchmark/pilot"
CACHE_DIR = PROJECT_ROOT / "outputs/cache/kepler_light_curves"
OUTPUT_DIR = BENCHMARK_DIR / "characterization_analysis"
PIPELINES = ("raw_bls", "arima_tcf", "kalman_bls", "kalman_tcf", "gp_bls", "gp_tcf")
FAMILIES = ("arima", "kalman", "gp")


def default_settings():
    return SimpleNamespace(
        benchmark_dir=BENCHMARK_DIR,
        cache_dir=CACHE_DIR,
        output_dir=None,
        quality_policy="default",
        require_finite_flux_error=False,
        test_fraction=0.20,
        allow_download=False,
        max_workers=None,
        reserve_cpu_cores=2,
        acf_lags=80,
        rolling_window=96,
        spectral_frequencies=2000,
        stationarity_min_observations=24,
    )


def build_parser():
    defaults = default_settings()
    parser = argparse.ArgumentParser(description="Join multistar characterization features to branch recovery improvements.")
    parser.add_argument("--benchmark-dir", type=Path, default=defaults.benchmark_dir)
    parser.add_argument("--cache-dir", type=Path, default=defaults.cache_dir)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--quality-policy", default=defaults.quality_policy)
    parser.add_argument("--require-finite-flux-error", action="store_true")
    parser.add_argument("--test-fraction", type=float, default=defaults.test_fraction)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-workers", type=int, default=defaults.max_workers)
    parser.add_argument("--reserve-cpu-cores", type=int, default=defaults.reserve_cpu_cores)
    parser.add_argument("--acf-lags", type=int, default=defaults.acf_lags)
    parser.add_argument("--rolling-window", type=int, default=defaults.rolling_window)
    parser.add_argument("--spectral-frequencies", type=int, default=defaults.spectral_frequencies)
    parser.add_argument("--stationarity-min-observations", type=int, default=defaults.stationarity_min_observations)
    return parser


def normalize_target_id(value):
    return str(value).upper().replace("KIC", "").strip()


def star_prefix(target_id, quarter):
    return f"kic_{normalize_target_id(target_id)}_q{int(quarter)}"


def resolve_worker_count(max_workers, reserve_cpu_cores, task_count):
    available = os.cpu_count() or 1
    reserve = max(0, int(reserve_cpu_cores))
    default_workers = max(1, available - reserve)
    requested = default_workers if max_workers is None else int(max_workers)
    return max(1, min(requested, available, int(task_count)))


def metric_path(benchmark_dir, name):
    return Path(benchmark_dir) / "metrics" / name


def load_benchmark_tables(benchmark_dir):
    benchmark_dir = Path(benchmark_dir)
    manifest_path = metric_path(benchmark_dir, "target_manifest_used.csv")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Benchmark manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path, dtype={"target_id": str})
    manifest["target_id"] = manifest["target_id"].map(normalize_target_id)
    manifest["quarter"] = pd.to_numeric(manifest["quarter"], errors="raise").astype(int)
    if "selection_group" not in manifest.columns:
        manifest["selection_group"] = "unspecified"

    master_path = metric_path(benchmark_dir, "multistar_challenger_master_results.csv")
    injections_path = master_path if master_path.exists() else metric_path(benchmark_dir, "multistar_challenger_injections.csv")
    if not injections_path.exists():
        raise FileNotFoundError(f"Benchmark injections table is missing: {injections_path}")
    injections = pd.read_csv(injections_path, dtype={"target_id": str})
    injections["target_id"] = injections["target_id"].map(normalize_target_id)
    injections["quarter"] = pd.to_numeric(injections["quarter"], errors="raise").astype(int)

    star_summary_path = metric_path(benchmark_dir, "multistar_challenger_star_summary.csv")
    star_summary = pd.read_csv(star_summary_path, dtype={"target_id": str}) if star_summary_path.exists() else pd.DataFrame()
    if not star_summary.empty:
        star_summary["target_id"] = star_summary["target_id"].map(normalize_target_id)
        star_summary["quarter"] = pd.to_numeric(star_summary["quarter"], errors="raise").astype(int)
    return manifest, injections, star_summary, str(injections_path)


def cache_path(cache_dir, target_id, quarter):
    return Path(cache_dir) / f"{star_prefix(target_id, quarter)}_pdcsap.parquet"


def load_frame(cache_dir, target_id, quarter, allow_download):
    path = cache_path(cache_dir, target_id, quarter)
    if path.exists():
        return pd.read_parquet(path), True
    if not allow_download:
        raise FileNotFoundError(f"Cached light curve is missing: {path}")
    light_curve = load_kepler_pdcsap(target_id, quarter)
    frame = light_curve.to_dataframe()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame, False


def characterize_star_task(task):
    row, args = task
    target_id = normalize_target_id(row["target_id"])
    quarter = int(row["quarter"])
    frame, cache_hit = load_frame(args["cache_dir"], target_id, quarter, args["allow_download"])
    regular, preprocessing = preprocess_pdcsap_light_curve(
        frame,
        quality_policy=args["quality_policy"],
        require_finite_flux_error=args["require_finite_flux_error"],
        normalization_fit_fraction=1.0 - args["test_fraction"],
    )
    diagnostics = characterize_regularized_light_curve(
        regular,
        target_id=target_id,
        quarter=quarter,
        preprocessing_summary=preprocessing.to_dict(),
        acf_lags=args["acf_lags"],
        rolling_window=args["rolling_window"],
        spectral_frequencies=args["spectral_frequencies"],
        stationarity_min_observations=args["stationarity_min_observations"],
    )
    diagnostics["selection_group"] = str(row.get("selection_group", "unspecified"))
    diagnostics["light_curve_cache_hit"] = bool(cache_hit)
    return diagnostics


def characterization_worker_args(args):
    return {
        "cache_dir": str(args.cache_dir),
        "quality_policy": str(args.quality_policy),
        "require_finite_flux_error": bool(args.require_finite_flux_error),
        "test_fraction": float(args.test_fraction),
        "allow_download": bool(args.allow_download),
        "acf_lags": int(args.acf_lags),
        "rolling_window": int(args.rolling_window),
        "spectral_frequencies": int(args.spectral_frequencies),
        "stationarity_min_observations": int(args.stationarity_min_observations),
    }


def characterize_manifest(manifest, args):
    rows = manifest[["target_id", "quarter", "selection_group"]].drop_duplicates().to_dict(orient="records")
    worker_count = resolve_worker_count(args.max_workers, args.reserve_cpu_cores, len(rows))
    worker_args = characterization_worker_args(args)
    results = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(characterize_star_task, (row, worker_args)) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Characterizing stars"):
            results.append(future.result())
    features = pd.DataFrame(results)
    features["target_id"] = features["target_id"].map(normalize_target_id)
    features["quarter"] = pd.to_numeric(features["quarter"], errors="raise").astype(int)
    return features.sort_values(["target_id", "quarter"]).reset_index(drop=True), worker_count


def _mean_bool(frame, column):
    if column not in frame.columns:
        return float("nan")
    return float(frame[column].fillna(False).astype(bool).mean()) if len(frame) else float("nan")


def recovery_column(pipeline, injections):
    calibrated = f"{pipeline}_harmonic_recovered_star_fap"
    if calibrated in injections.columns:
        return calibrated
    return f"{pipeline}_harmonic_rank1_matched"


def exact_recovery_column(pipeline, injections):
    calibrated = f"{pipeline}_exact_recovered_star_fap"
    if calibrated in injections.columns:
        return calibrated
    return f"{pipeline}_exact_rank1_matched"


def aggregate_branch_diagnostics(injections):
    rows = []
    grouped = injections.groupby(["target_id", "quarter"], dropna=False)
    for (target_id, quarter), group in grouped:
        row = {"target_id": normalize_target_id(target_id), "quarter": int(quarter)}
        raw_abs_acf1 = pd.to_numeric(group.get("raw_residual_acf1"), errors="coerce").abs()
        row["raw_median_abs_residual_acf1"] = float(raw_abs_acf1.median())
        row["raw_median_local_snr"] = float(pd.to_numeric(group.get("raw_local_snr"), errors="coerce").median())
        for family in FAMILIES:
            branch_abs_acf1 = pd.to_numeric(group.get(f"{family}_residual_acf1"), errors="coerce").abs()
            row[f"{family}_median_abs_residual_acf1"] = float(branch_abs_acf1.median())
            row[f"{family}_median_snr_retention"] = float(pd.to_numeric(group.get(f"{family}_snr_retention_fraction"), errors="coerce").median())
            row[f"{family}_median_depth_retention"] = float(pd.to_numeric(group.get(f"{family}_depth_retention_fraction"), errors="coerce").median())
            row[f"{family}_whitening_abs_acf1_reduction"] = float((raw_abs_acf1 - branch_abs_acf1).median())
            row[f"{family}_snr_retention_below_half_rate"] = float((pd.to_numeric(group.get(f"{family}_snr_retention_fraction"), errors="coerce") < 0.5).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_recovery_rates(injections):
    rows = []
    grouped = injections.groupby(["target_id", "quarter"], dropna=False)
    for (target_id, quarter), group in grouped:
        row = {"target_id": normalize_target_id(target_id), "quarter": int(quarter), "injection_count": int(len(group))}
        for pipeline in PIPELINES:
            harmonic_column = recovery_column(pipeline, injections)
            exact_column = exact_recovery_column(pipeline, injections)
            row[f"{pipeline}_harmonic_recovery_rate"] = _mean_bool(group, harmonic_column)
            row[f"{pipeline}_exact_recovery_rate"] = _mean_bool(group, exact_column)
        rows.append(row)
    rates = pd.DataFrame(rows)
    rates["arima_improvement"] = rates["arima_tcf_harmonic_recovery_rate"] - rates["raw_bls_harmonic_recovery_rate"]
    rates["kalman_best_harmonic_recovery_rate"] = rates[["kalman_bls_harmonic_recovery_rate", "kalman_tcf_harmonic_recovery_rate"]].max(axis=1)
    rates["gp_best_harmonic_recovery_rate"] = rates[["gp_bls_harmonic_recovery_rate", "gp_tcf_harmonic_recovery_rate"]].max(axis=1)
    rates["kalman_improvement"] = rates["kalman_best_harmonic_recovery_rate"] - rates["raw_bls_harmonic_recovery_rate"]
    rates["gp_improvement"] = rates["gp_best_harmonic_recovery_rate"] - rates["raw_bls_harmonic_recovery_rate"]
    rates["best_challenger_harmonic_recovery_rate"] = rates[[
        "arima_tcf_harmonic_recovery_rate",
        "kalman_bls_harmonic_recovery_rate",
        "kalman_tcf_harmonic_recovery_rate",
        "gp_bls_harmonic_recovery_rate",
        "gp_tcf_harmonic_recovery_rate",
    ]].max(axis=1)
    rates["raw_bls_preferable_or_tied"] = rates["raw_bls_harmonic_recovery_rate"] >= rates["best_challenger_harmonic_recovery_rate"]
    rates["raw_bls_strictly_preferable"] = rates["raw_bls_harmonic_recovery_rate"] > rates["best_challenger_harmonic_recovery_rate"]
    return rates


def build_star_table(features, injections):
    branch = aggregate_branch_diagnostics(injections)
    rates = aggregate_recovery_rates(injections)
    table = features.merge(rates, on=["target_id", "quarter"], how="inner", validate="one_to_one")
    table = table.merge(branch, on=["target_id", "quarter"], how="left", validate="one_to_one")
    table["star"] = "KIC " + table["target_id"].astype(str) + " Q" + table["quarter"].astype(str)
    table["acf_timescale_days"] = table["acf_decay_e_days"].fillna(table["integrated_positive_acf_days"])
    table["spectral_strength"] = table["spectral_concentration"]
    table["variance_drift"] = table["rolling_variance_max_to_median"]
    front_columns = [
        "star",
        "target_id",
        "quarter",
        "selection_group",
        "acf_timescale_days",
        "acf_lag_1",
        "spectral_strength",
        "spectral_entropy",
        "variance_drift",
        "gap_fraction",
        "raw_bls_harmonic_recovery_rate",
        "arima_tcf_harmonic_recovery_rate",
        "arima_improvement",
        "kalman_best_harmonic_recovery_rate",
        "kalman_improvement",
        "gp_best_harmonic_recovery_rate",
        "gp_improvement",
        "arima_whitening_abs_acf1_reduction",
        "arima_median_snr_retention",
        "kalman_whitening_abs_acf1_reduction",
        "kalman_median_snr_retention",
        "gp_whitening_abs_acf1_reduction",
        "gp_median_snr_retention",
        "raw_bls_preferable_or_tied",
        "raw_bls_strictly_preferable",
    ]
    remaining = [column for column in table.columns if column not in front_columns]
    return table[front_columns + remaining].sort_values("target_id").reset_index(drop=True)


def finite_pairs(frame, feature, outcome):
    values = frame[[feature, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
    return values


def correlation_table(frame, feature_columns, outcome_columns):
    rows = []
    for feature in feature_columns:
        if feature not in frame.columns:
            continue
        for outcome in outcome_columns:
            if outcome not in frame.columns:
                continue
            if feature == outcome:
                continue
            values = finite_pairs(frame, feature, outcome)
            if len(values) < 4 or values[feature].nunique() < 2 or values[outcome].nunique() < 2:
                spearman = float("nan")
                spearman_pvalue = float("nan")
                pearson = float("nan")
                pearson_pvalue = float("nan")
            else:
                spearman_result = spearmanr(values[feature], values[outcome])
                pearson_result = pearsonr(values[feature], values[outcome])
                spearman = float(spearman_result.statistic)
                spearman_pvalue = float(spearman_result.pvalue)
                pearson = float(pearson_result.statistic)
                pearson_pvalue = float(pearson_result.pvalue)
            rows.append(
                {
                    "feature": feature,
                    "outcome": outcome,
                    "n": int(len(values)),
                    "spearman_correlation": spearman,
                    "spearman_pvalue": spearman_pvalue,
                    "pearson_correlation": pearson,
                    "pearson_pvalue": pearson_pvalue,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("spearman_correlation", key=lambda item: item.abs(), ascending=False).reset_index(drop=True)
    return result


def median_split_effect(frame, feature, outcome):
    values = finite_pairs(frame, feature, outcome)
    if len(values) < 4 or values[feature].nunique() < 2:
        return {"low_feature_mean": float("nan"), "high_feature_mean": float("nan"), "high_minus_low": float("nan"), "low_n": 0, "high_n": 0}
    threshold = float(values[feature].median())
    low = values[values[feature] <= threshold][outcome]
    high = values[values[feature] > threshold][outcome]
    return {
        "median_split_threshold": threshold,
        "low_feature_mean": float(low.mean()) if len(low) else float("nan"),
        "high_feature_mean": float(high.mean()) if len(high) else float("nan"),
        "high_minus_low": float(high.mean() - low.mean()) if len(low) and len(high) else float("nan"),
        "low_n": int(len(low)),
        "high_n": int(len(high)),
    }


def lookup_correlation(correlations, feature, outcome):
    if correlations.empty:
        return {}
    match = correlations[(correlations["feature"] == feature) & (correlations["outcome"] == outcome)]
    return match.iloc[0].to_dict() if len(match) else {}


def interpretation_from_effect(effect, *, positive_label, negative_label, flat_label="no clear split effect"):
    delta = effect.get("high_minus_low", float("nan"))
    if not np.isfinite(delta) or abs(delta) < 0.05:
        return flat_label
    return positive_label if delta > 0 else negative_label


def question_summary(star_table, correlations):
    questions = [
        {
            "question": "Do high-ACF stars benefit more from ARIMA?",
            "feature": "acf_lag_1",
            "outcome": "arima_improvement",
            "positive": "higher ACF stars show larger ARIMA lift",
            "negative": "higher ACF stars show lower ARIMA lift",
        },
        {
            "question": "Do smooth long-timescale stars benefit more from GP?",
            "feature": "integrated_positive_acf_days",
            "outcome": "gp_improvement",
            "positive": "longer ACF timescale stars show larger GP lift",
            "negative": "longer ACF timescale stars show lower GP lift",
        },
        {
            "question": "Do state-space-like variance/drift stars benefit more from Kalman?",
            "feature": "rolling_variance_max_to_median",
            "outcome": "kalman_improvement",
            "positive": "higher variance drift stars show larger Kalman lift",
            "negative": "higher variance drift stars show lower Kalman lift",
        },
        {
            "question": "Does high spectral concentration predict GP success?",
            "feature": "spectral_strength",
            "outcome": "gp_improvement",
            "positive": "higher spectral concentration tracks larger GP lift",
            "negative": "higher spectral concentration tracks lower GP lift",
        },
        {
            "question": "Does whitening improve ACF while damaging ARIMA transit SNR?",
            "feature": "arima_whitening_abs_acf1_reduction",
            "outcome": "arima_median_snr_retention",
            "positive": "larger ARIMA ACF reduction tracks higher transit SNR retention",
            "negative": "larger ARIMA ACF reduction tracks lower transit SNR retention",
        },
        {
            "question": "Does whitening improve ACF while damaging GP transit SNR?",
            "feature": "gp_whitening_abs_acf1_reduction",
            "outcome": "gp_median_snr_retention",
            "positive": "larger GP ACF reduction tracks higher transit SNR retention",
            "negative": "larger GP ACF reduction tracks lower transit SNR retention",
        },
        {
            "question": "Does whitening improve ACF while damaging Kalman transit SNR?",
            "feature": "kalman_whitening_abs_acf1_reduction",
            "outcome": "kalman_median_snr_retention",
            "positive": "larger Kalman ACF reduction tracks higher transit SNR retention",
            "negative": "larger Kalman ACF reduction tracks lower transit SNR retention",
        },
    ]
    rows = []
    for item in questions:
        effect = median_split_effect(star_table, item["feature"], item["outcome"])
        corr = lookup_correlation(correlations, item["feature"], item["outcome"])
        rows.append(
            {
                "question": item["question"],
                "feature": item["feature"],
                "outcome": item["outcome"],
                **effect,
                "spearman_correlation": corr.get("spearman_correlation", float("nan")),
                "spearman_pvalue": corr.get("spearman_pvalue", float("nan")),
                "interpretation": interpretation_from_effect(effect, positive_label=item["positive"], negative_label=item["negative"]),
            }
        )
    raw_count = int(star_table["raw_bls_preferable_or_tied"].fillna(False).astype(bool).sum()) if "raw_bls_preferable_or_tied" in star_table else 0
    strict_raw_count = int(star_table["raw_bls_strictly_preferable"].fillna(False).astype(bool).sum()) if "raw_bls_strictly_preferable" in star_table else 0
    rows.append(
        {
            "question": "Are there stars for which raw BLS is preferable?",
            "feature": "raw_bls_preferable_or_tied",
            "outcome": "best_challenger_harmonic_recovery_rate",
            "median_split_threshold": float("nan"),
            "low_feature_mean": float("nan"),
            "high_feature_mean": float("nan"),
            "high_minus_low": float("nan"),
            "low_n": 0,
            "high_n": int(raw_count),
            "spearman_correlation": float("nan"),
            "spearman_pvalue": float("nan"),
            "interpretation": f"raw BLS was preferable or tied for {raw_count} stars; strictly preferable for {strict_raw_count} stars",
        }
    )
    return pd.DataFrame(rows)


def feature_bin_summary(frame, features, outcomes):
    rows = []
    for feature in features:
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        if values.notna().sum() < 4 or values.nunique(dropna=True) < 2:
            continue
        threshold = float(values.median())
        labelled = frame.copy()
        labelled["feature_bin"] = np.where(values > threshold, "high", "low")
        for outcome in outcomes:
            if outcome not in labelled.columns:
                continue
            grouped = labelled.groupby("feature_bin", as_index=False).agg(
                star_count=("target_id", "size"),
                outcome_mean=(outcome, "mean"),
                outcome_median=(outcome, "median"),
            )
            for row in grouped.to_dict(orient="records"):
                rows.append({"feature": feature, "outcome": outcome, "threshold": threshold, **row})
    return pd.DataFrame(rows)


def write_markdown_report(path, summary, question_rows, top_table):
    lines = [
        "# Multistar Characterization Effects",
        "",
        f"Benchmark directory: `{summary['benchmark_dir']}`",
        f"Targets: {summary['target_count']}",
        f"Injections: {summary['injection_count']}",
        f"Recovery metric: {summary['recovery_metric']}",
        f"Characterization workers: {summary['characterization_workers']} with {summary['reserve_cpu_cores']} CPU cores reserved.",
        "",
        "## Question Summary",
        "",
    ]
    for row in question_rows.to_dict(orient="records"):
        lines.append(f"- {row['question']} {row['interpretation']}.")
    lines.extend(["", "## Star Table Preview", ""])
    preview_columns = [
        "star",
        "acf_timescale_days",
        "spectral_strength",
        "variance_drift",
        "gap_fraction",
        "gp_improvement",
        "kalman_improvement",
        "arima_improvement",
    ]
    lines.append(top_table[preview_columns].to_markdown(index=False))
    path.write_text("\n".join(lines) + "\n")


def build_outputs(args):
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else benchmark_dir / "characterization_analysis"
    manifest, injections, star_summary, source_path = load_benchmark_tables(benchmark_dir)
    features, worker_count = characterize_manifest(manifest, args)
    star_table = build_star_table(features, injections)

    feature_columns = [
        "acf_timescale_days",
        "acf_lag_1",
        "max_abs_acf_1_n",
        "integrated_positive_acf_days",
        "spectral_strength",
        "spectral_concentration",
        "spectral_entropy",
        "dominant_lomb_scargle_power",
        "rolling_variance_max_to_median",
        "rolling_mean_range_over_robust_scale",
        "gap_fraction",
        "flux_robust_scale",
        "original_adf_pvalue",
        "original_kpss_pvalue",
        "arima_whitening_abs_acf1_reduction",
        "kalman_whitening_abs_acf1_reduction",
        "gp_whitening_abs_acf1_reduction",
    ]
    outcome_columns = [
        "arima_improvement",
        "kalman_improvement",
        "gp_improvement",
        "arima_median_snr_retention",
        "kalman_median_snr_retention",
        "gp_median_snr_retention",
        "arima_whitening_abs_acf1_reduction",
        "kalman_whitening_abs_acf1_reduction",
        "gp_whitening_abs_acf1_reduction",
        "raw_bls_harmonic_recovery_rate",
    ]
    correlations = correlation_table(star_table, feature_columns, outcome_columns)
    questions = question_summary(star_table, correlations)
    bins = feature_bin_summary(
        star_table,
        [
            "acf_lag_1",
            "integrated_positive_acf_days",
            "spectral_strength",
            "rolling_variance_max_to_median",
            "gap_fraction",
        ],
        ["arima_improvement", "kalman_improvement", "gp_improvement", "raw_bls_harmonic_recovery_rate"],
    )
    raw_preferable = star_table[star_table["raw_bls_preferable_or_tied"].fillna(False).astype(bool)].copy()
    aggressive_whitening = star_table[
        (
            (star_table["arima_whitening_abs_acf1_reduction"] > star_table["arima_whitening_abs_acf1_reduction"].median())
            & (star_table["arima_median_snr_retention"] < 0.5)
        )
        | (
            (star_table["kalman_whitening_abs_acf1_reduction"] > star_table["kalman_whitening_abs_acf1_reduction"].median())
            & (star_table["kalman_median_snr_retention"] < 0.5)
        )
        | (
            (star_table["gp_whitening_abs_acf1_reduction"] > star_table["gp_whitening_abs_acf1_reduction"].median())
            & (star_table["gp_median_snr_retention"] < 0.5)
        )
    ].copy()

    recovery_metric = "per-star 1% FAP harmonic recovery" if any(f"{pipeline}_harmonic_recovered_star_fap" in injections.columns for pipeline in PIPELINES) else "rank-1 harmonic recovery"
    summary = {
        "benchmark_dir": str(benchmark_dir),
        "benchmark_injection_source": source_path,
        "target_count": int(star_table[["target_id", "quarter"]].drop_duplicates().shape[0]),
        "injection_count": int(len(injections)),
        "recovery_metric": recovery_metric,
        "characterization_workers": int(worker_count),
        "available_cpu_count": int(os.cpu_count() or 1),
        "reserve_cpu_cores": int(args.reserve_cpu_cores),
        "raw_bls_preferable_or_tied_star_count": int(len(raw_preferable)),
        "raw_bls_strictly_preferable_star_count": int(star_table["raw_bls_strictly_preferable"].fillna(False).astype(bool).sum()),
        "aggressive_whitening_low_snr_star_count": int(len(aggressive_whitening)),
        "mean_arima_improvement": float(star_table["arima_improvement"].mean()),
        "mean_kalman_improvement": float(star_table["kalman_improvement"].mean()),
        "mean_gp_improvement": float(star_table["gp_improvement"].mean()),
        "top_absolute_correlations": correlations.head(10).to_dict(orient="records"),
        "question_summary": questions.to_dict(orient="records"),
    }
    if not star_summary.empty:
        summary["benchmark_star_summary_rows"] = int(len(star_summary))

    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "multistar_characterization_features.csv", index=False)
    star_table.to_csv(output_dir / "multistar_characterization_per_star.csv", index=False)
    correlations.to_csv(output_dir / "multistar_characterization_correlations.csv", index=False)
    questions.to_csv(output_dir / "multistar_characterization_question_summary.csv", index=False)
    bins.to_csv(output_dir / "multistar_characterization_feature_bins.csv", index=False)
    raw_preferable.to_csv(output_dir / "multistar_characterization_raw_bls_preferable.csv", index=False)
    aggressive_whitening.to_csv(output_dir / "multistar_characterization_aggressive_whitening_low_snr.csv", index=False)
    (output_dir / "multistar_characterization_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    write_markdown_report(output_dir / "multistar_characterization_report.md", summary, questions, star_table)
    return output_dir, summary, star_table, questions


def main(args=None):
    args = args or build_parser().parse_args()
    output_dir, summary, star_table, questions = build_outputs(args)
    preview_columns = [
        "star",
        "acf_timescale_days",
        "spectral_strength",
        "variance_drift",
        "gap_fraction",
        "gp_improvement",
        "kalman_improvement",
        "arima_improvement",
    ]
    print(f"Output directory: {output_dir}")
    print(f"Recovery metric: {summary['recovery_metric']}")
    print(f"Characterization workers: {summary['characterization_workers']} (reserved cores: {summary['reserve_cpu_cores']})")
    print(star_table[preview_columns].to_string(index=False))
    print()
    print(questions[["question", "interpretation"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
