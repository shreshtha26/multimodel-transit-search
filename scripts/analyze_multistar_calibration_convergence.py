"""Analyze star-level FAP threshold convergence from stored challenger null trials."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_multistar_challenger_benchmark import DEFAULT_PIPELINES, PIPELINE_DEFINITIONS, default_settings, json_ready, normalize_target_id

def parse_levels(value):
    levels = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not levels or any(level < 1 for level in levels):
        raise argparse.ArgumentTypeError("Levels must be positive comma-separated integers.")
    return tuple(sorted(set(levels)))

def parse_pipelines(value):
    pipelines = tuple(item.strip() for item in str(value).split(",") if item.strip())
    invalid = sorted(set(pipelines).difference(PIPELINE_DEFINITIONS))
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown pipeline(s): {invalid}. Valid pipelines are {sorted(PIPELINE_DEFINITIONS)}.")
    return pipelines

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Measure how multi-star challenger FAP thresholds and recovery labels change with null-trial count.")
    parser.add_argument("--profile", choices=("pilot", "main", "smoke"), default="pilot")
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--levels", type=parse_levels, default=(10, 50, 100, 250, 500, 1000))
    parser.add_argument("--fap-level", type=float, default=0.01)
    parser.add_argument("--pipelines", type=parse_pipelines, default=DEFAULT_PIPELINES)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=90210)
    args = parser.parse_args(argv)
    defaults = default_settings(args.profile)
    args.benchmark_dir = Path(args.benchmark_dir) if args.benchmark_dir is not None else Path(defaults.output_dir)
    return args

def bool_series(values):
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.fillna(False).astype(str).str.lower().isin(("true", "1", "yes"))

def load_injections(args):
    path = Path(args.benchmark_dir) / "metrics/multistar_challenger_injections.csv"
    if not path.exists():
        raise FileNotFoundError(f"Injection table is missing: {path}")
    frame = pd.read_csv(path, dtype={"target_id": str})
    frame["target_id"] = frame["target_id"].map(normalize_target_id)
    frame["quarter"] = pd.to_numeric(frame["quarter"], errors="raise").astype(int)
    return frame

def load_null_trials(args):
    calibration_root = Path(args.benchmark_dir) / "star_calibration"
    tables = []
    for path in sorted(calibration_root.glob("kic_*_q*/null_trials.csv")):
        tables.append(pd.read_csv(path, dtype={"target_id": str}))
    if not tables:
        metrics_path = Path(args.benchmark_dir) / "metrics/multistar_challenger_star_null_trials.csv"
        if metrics_path.exists():
            tables.append(pd.read_csv(metrics_path, dtype={"target_id": str}))
    if not tables:
        raise FileNotFoundError(f"No null trial tables found under {calibration_root}")
    frame = pd.concat(tables, ignore_index=True)
    frame["target_id"] = frame["target_id"].map(normalize_target_id)
    frame["quarter"] = pd.to_numeric(frame["quarter"], errors="raise").astype(int)
    frame["trial"] = pd.to_numeric(frame["trial"], errors="raise").astype(int)
    return frame.sort_values(["target_id", "quarter", "trial"]).reset_index(drop=True)

def threshold_from_scores(scores, fap_level):
    scores = pd.to_numeric(pd.Series(scores), errors="coerce").dropna().to_numpy(dtype=float)
    if scores.size == 0:
        return float("nan")
    return float(np.quantile(scores, 1.0 - float(fap_level), method="higher"))

def bootstrap_threshold_interval(scores, fap_level, iterations, seed):
    scores = pd.to_numeric(pd.Series(scores), errors="coerce").dropna().to_numpy(dtype=float)
    if scores.size == 0 or int(iterations) < 1:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    thresholds = []
    for index in range(int(iterations)):
        sample = scores[rng.integers(0, scores.size, size=scores.size)]
        thresholds.append(threshold_from_scores(sample, fap_level))
    thresholds = np.asarray(thresholds, dtype=float)
    lower = float(np.nanquantile(thresholds, 0.025))
    upper = float(np.nanquantile(thresholds, 0.975))
    width = float(upper - lower)
    return lower, upper, width

def available_levels(null_trials, requested_levels):
    per_star_counts = null_trials.groupby(["target_id", "quarter"]).size()
    if per_star_counts.empty:
        return []
    minimum = int(per_star_counts.min())
    return [int(level) for level in requested_levels if int(level) <= minimum]

def prefix_integrity(null_trials, requested_levels):
    rows = []
    available = available_levels(null_trials, requested_levels)
    for (target_id, quarter), group in null_trials.groupby(["target_id", "quarter"], sort=False):
        trials = sorted(pd.to_numeric(group["trial"], errors="coerce").dropna().astype(int).unique())
        prefix_count = 0
        for expected in range(len(trials)):
            if expected in trials:
                prefix_count += 1
            else:
                break
        for level in requested_levels:
            rows.append({"target_id": target_id, "quarter": int(quarter), "requested_level": int(level), "has_complete_nested_prefix": bool(prefix_count >= int(level)), "available_prefix_count": int(prefix_count), "included_in_analysis": bool(int(level) in available and prefix_count >= int(level))})
    return pd.DataFrame(rows)

def threshold_records(null_trials, injections, args):
    rows = []
    labels = []
    available = available_levels(null_trials, args.levels)
    grouped_nulls = dict(tuple(null_trials.groupby(["target_id", "quarter"], sort=False)))
    grouped_injections = dict(tuple(injections.groupby(["target_id", "quarter"], sort=False)))
    for (target_id, quarter), null_group in grouped_nulls.items():
        injection_group = grouped_injections.get((target_id, quarter), pd.DataFrame())
        if injection_group.empty:
            continue
        ordered_nulls = null_group.sort_values("trial")
        for pipeline in args.pipelines:
            success_column = f"{pipeline}_success"
            score_column = f"{pipeline}_score"
            if score_column not in ordered_nulls.columns or score_column not in injection_group.columns:
                continue
            previous_threshold = float("nan")
            for level in available:
                subset = ordered_nulls.head(int(level))
                successful = subset[bool_series(subset[success_column])] if success_column in subset.columns else subset
                scores = pd.to_numeric(successful[score_column], errors="coerce")
                threshold = threshold_from_scores(scores, args.fap_level)
                bootstrap_seed = int(args.bootstrap_seed) + int(level) * 1009 + int(normalize_target_id(target_id)) + sum(ord(item) for item in pipeline)
                ci_lower, ci_upper, ci_width = bootstrap_threshold_interval(scores, args.fap_level, args.bootstrap_iterations, bootstrap_seed)
                threshold_change = float(threshold - previous_threshold) if np.isfinite(threshold) and np.isfinite(previous_threshold) else float("nan")
                threshold_change_fraction = float(threshold_change / previous_threshold) if np.isfinite(threshold_change) and previous_threshold != 0 and np.isfinite(previous_threshold) else float("nan")
                injection_scores = pd.to_numeric(injection_group[score_column], errors="coerce").to_numpy(dtype=float)
                passes = injection_scores >= threshold if np.isfinite(threshold) else np.zeros(len(injection_scores), dtype=bool)
                harmonic_rank1 = bool_series(injection_group[f"{pipeline}_harmonic_rank1_matched"])
                exact_rank1 = bool_series(injection_group[f"{pipeline}_exact_rank1_matched"])
                harmonic = harmonic_rank1.to_numpy(dtype=bool) & passes
                exact = exact_rank1.to_numpy(dtype=bool) & passes
                score_over = np.divide(injection_scores, threshold, out=np.full(len(injection_scores), np.nan, dtype=float), where=np.isfinite(threshold) & (threshold != 0))
                rows.append({"target_id": target_id, "quarter": int(quarter), "pipeline": pipeline, "branch": PIPELINE_DEFINITIONS[pipeline][0], "detector": PIPELINE_DEFINITIONS[pipeline][1], "null_level": int(level), "requested_fap_level": float(args.fap_level), "score_threshold": threshold, "threshold_bootstrap_ci95_lower": ci_lower, "threshold_bootstrap_ci95_upper": ci_upper, "threshold_bootstrap_ci95_width": ci_width, "threshold_bootstrap_ci95_width_fraction": float(ci_width / threshold) if np.isfinite(ci_width) and np.isfinite(threshold) and threshold != 0 else float("nan"), "bootstrap_iterations": int(args.bootstrap_iterations), "previous_available_threshold": previous_threshold, "threshold_change": threshold_change, "threshold_change_fraction": threshold_change_fraction, "requested_null_trials": int(level), "successful_null_trials": int(pd.to_numeric(scores, errors="coerce").notna().sum()), "injection_count": int(len(injection_group)), "detection_count": int(passes.sum()), "detection_rate": float(passes.mean()), "harmonic_recovery_count": int(harmonic.sum()), "harmonic_recovery_rate": float(harmonic.mean()), "exact_recovery_count": int(exact.sum()), "exact_recovery_rate": float(exact.mean()), "median_score_over_threshold": float(pd.Series(score_over).median())})
                for index, injection_row in injection_group.reset_index(drop=True).iterrows():
                    labels.append({"target_id": target_id, "quarter": int(quarter), "selection_group": str(injection_row.get("selection_group", "")), "case_index": int(injection_row["case_index"]), "injected_period_days": float(injection_row["injected_period_days"]), "injected_duration_hours": float(injection_row["injected_duration_hours"]), "injected_depth": float(injection_row["injected_depth"]), "epoch_phase_fraction": float(injection_row["epoch_phase_fraction"]), "pipeline": pipeline, "null_level": int(level), "score": float(injection_scores[index]) if np.isfinite(injection_scores[index]) else float("nan"), "score_threshold": threshold, "score_over_threshold": float(score_over[index]) if np.isfinite(score_over[index]) else float("nan"), "passes_threshold": bool(passes[index]), "harmonic_rank1_matched": bool(harmonic_rank1.iloc[index]), "harmonic_recovered": bool(harmonic[index]), "exact_rank1_matched": bool(exact_rank1.iloc[index]), "exact_recovered": bool(exact[index])})
                previous_threshold = threshold
    return pd.DataFrame(rows), pd.DataFrame(labels)

def pipeline_summary(labels, args):
    if labels.empty:
        return pd.DataFrame()
    grouped = labels.groupby(["null_level", "pipeline"], as_index=False)
    return grouped.agg(injection_count=("case_index", "size"), star_count=("target_id", "nunique"), detection_count=("passes_threshold", "sum"), detection_rate=("passes_threshold", "mean"), harmonic_recovery_count=("harmonic_recovered", "sum"), harmonic_recovery_rate=("harmonic_recovered", "mean"), exact_recovery_count=("exact_recovered", "sum"), exact_recovery_rate=("exact_recovered", "mean"), median_score_over_threshold=("score_over_threshold", "median"))

def label_changes(labels):
    if labels.empty:
        return pd.DataFrame()
    key_columns = ["target_id", "quarter", "case_index", "pipeline"]
    rows = []
    levels = sorted(labels["null_level"].unique())
    for previous, current in zip(levels[:-1], levels[1:]):
        left = labels[labels["null_level"] == previous][key_columns + ["passes_threshold", "harmonic_recovered", "exact_recovered"]].copy()
        right = labels[labels["null_level"] == current][key_columns + ["passes_threshold", "harmonic_recovered", "exact_recovered"]].copy()
        merged = left.merge(right, on=key_columns, suffixes=("_previous", "_current"), how="inner", validate="one_to_one")
        for pipeline, group in merged.groupby("pipeline"):
            rows.append({"pipeline": pipeline, "previous_null_level": int(previous), "current_null_level": int(current), "compared_injections": int(len(group)), "detection_label_changes": int((group["passes_threshold_previous"] != group["passes_threshold_current"]).sum()), "detection_label_change_fraction": float((group["passes_threshold_previous"] != group["passes_threshold_current"]).mean()), "harmonic_recovery_label_changes": int((group["harmonic_recovered_previous"] != group["harmonic_recovered_current"]).sum()), "harmonic_recovery_label_change_fraction": float((group["harmonic_recovered_previous"] != group["harmonic_recovered_current"]).mean()), "exact_recovery_label_changes": int((group["exact_recovered_previous"] != group["exact_recovered_current"]).sum()), "exact_recovery_label_change_fraction": float((group["exact_recovered_previous"] != group["exact_recovered_current"]).mean())})
    return pd.DataFrame(rows)

def combination_members(pipelines):
    named = [("raw_bls_union_gp_tcf", [pipeline for pipeline in ("raw_bls", "gp_tcf") if pipeline in pipelines]), ("existing_bls_tcf", [pipeline for pipeline in ("raw_bls", "arima_tcf") if pipeline in pipelines]), ("non_gp_union", [pipeline for pipeline in ("raw_bls", "arima_tcf", "kalman_bls", "kalman_tcf") if pipeline in pipelines]), ("gp_union", [pipeline for pipeline in ("gp_bls", "gp_tcf") if pipeline in pipelines]), ("all_pipelines", list(pipelines))]
    return [(name, members) for name, members in named if members]

def label_wide(labels, null_level, column):
    key_columns = ["target_id", "quarter", "case_index"]
    subset = labels[labels["null_level"] == int(null_level)][key_columns + ["pipeline", column]].copy()
    if subset.empty:
        return pd.DataFrame()
    wide = subset.pivot_table(index=key_columns, columns="pipeline", values=column, aggfunc="max", fill_value=False)
    return wide.astype(bool)

def combination_summary(labels, args):
    rows = []
    levels = sorted(labels["null_level"].unique()) if not labels.empty else []
    for level in levels:
        harmonic = label_wide(labels, level, "harmonic_recovered")
        exact = label_wide(labels, level, "exact_recovered")
        for name, members in combination_members(args.pipelines):
            harmonic_union = harmonic[members].any(axis=1) if not harmonic.empty else pd.Series(dtype=bool)
            exact_union = exact[members].any(axis=1) if not exact.empty else pd.Series(dtype=bool)
            rows.append({"null_level": int(level), "combination": name, "pipelines": ",".join(members), "injection_count": int(len(harmonic_union)), "harmonic_union_count": int(harmonic_union.sum()), "harmonic_union_rate": float(harmonic_union.mean()) if len(harmonic_union) else float("nan"), "exact_union_count": int(exact_union.sum()), "exact_union_rate": float(exact_union.mean()) if len(exact_union) else float("nan")})
    return pd.DataFrame(rows)

def combination_changes(labels, args):
    rows = []
    levels = sorted(labels["null_level"].unique()) if not labels.empty else []
    for previous, current in zip(levels[:-1], levels[1:]):
        previous_harmonic = label_wide(labels, previous, "harmonic_recovered")
        current_harmonic = label_wide(labels, current, "harmonic_recovered")
        previous_exact = label_wide(labels, previous, "exact_recovered")
        current_exact = label_wide(labels, current, "exact_recovered")
        for name, members in combination_members(args.pipelines):
            left_h = previous_harmonic[members].any(axis=1).rename("previous")
            right_h = current_harmonic[members].any(axis=1).rename("current")
            merged_h = pd.concat([left_h, right_h], axis=1, join="inner")
            left_e = previous_exact[members].any(axis=1).rename("previous")
            right_e = current_exact[members].any(axis=1).rename("current")
            merged_e = pd.concat([left_e, right_e], axis=1, join="inner")
            rows.append({"combination": name, "pipelines": ",".join(members), "previous_null_level": int(previous), "current_null_level": int(current), "compared_injections": int(len(merged_h)), "harmonic_union_label_changes": int((merged_h["previous"] != merged_h["current"]).sum()), "harmonic_union_label_change_fraction": float((merged_h["previous"] != merged_h["current"]).mean()) if len(merged_h) else float("nan"), "exact_union_label_changes": int((merged_e["previous"] != merged_e["current"]).sum()), "exact_union_label_change_fraction": float((merged_e["previous"] != merged_e["current"]).mean()) if len(merged_e) else float("nan")})
    return pd.DataFrame(rows)

def unique_summary(labels, args):
    rows = []
    levels = sorted(labels["null_level"].unique()) if not labels.empty else []
    for level in levels:
        harmonic = label_wide(labels, level, "harmonic_recovered")
        exact = label_wide(labels, level, "exact_recovered")
        for pipeline in args.pipelines:
            other = [item for item in args.pipelines if item != pipeline]
            harmonic_unique = harmonic[pipeline] & ~harmonic[other].any(axis=1) if not harmonic.empty else pd.Series(dtype=bool)
            exact_unique = exact[pipeline] & ~exact[other].any(axis=1) if not exact.empty else pd.Series(dtype=bool)
            rows.append({"null_level": int(level), "pipeline": pipeline, "injection_count": int(len(harmonic_unique)), "harmonic_unique_count": int(harmonic_unique.sum()), "harmonic_unique_rate": float(harmonic_unique.mean()) if len(harmonic_unique) else float("nan"), "exact_unique_count": int(exact_unique.sum()), "exact_unique_rate": float(exact_unique.mean()) if len(exact_unique) else float("nan")})
    return pd.DataFrame(rows)

def unique_changes(labels, args):
    rows = []
    levels = sorted(labels["null_level"].unique()) if not labels.empty else []
    for previous, current in zip(levels[:-1], levels[1:]):
        previous_harmonic = label_wide(labels, previous, "harmonic_recovered")
        current_harmonic = label_wide(labels, current, "harmonic_recovered")
        previous_exact = label_wide(labels, previous, "exact_recovered")
        current_exact = label_wide(labels, current, "exact_recovered")
        for pipeline in args.pipelines:
            other = [item for item in args.pipelines if item != pipeline]
            previous_h = (previous_harmonic[pipeline] & ~previous_harmonic[other].any(axis=1)).rename("previous")
            current_h = (current_harmonic[pipeline] & ~current_harmonic[other].any(axis=1)).rename("current")
            merged_h = pd.concat([previous_h, current_h], axis=1, join="inner")
            previous_e = (previous_exact[pipeline] & ~previous_exact[other].any(axis=1)).rename("previous")
            current_e = (current_exact[pipeline] & ~current_exact[other].any(axis=1)).rename("current")
            merged_e = pd.concat([previous_e, current_e], axis=1, join="inner")
            rows.append({"pipeline": pipeline, "previous_null_level": int(previous), "current_null_level": int(current), "compared_injections": int(len(merged_h)), "harmonic_unique_label_changes": int((merged_h["previous"] != merged_h["current"]).sum()), "harmonic_unique_label_change_fraction": float((merged_h["previous"] != merged_h["current"]).mean()) if len(merged_h) else float("nan"), "exact_unique_label_changes": int((merged_e["previous"] != merged_e["current"]).sum()), "exact_unique_label_change_fraction": float((merged_e["previous"] != merged_e["current"]).mean()) if len(merged_e) else float("nan")})
    return pd.DataFrame(rows)

def save_outputs(args, thresholds, labels, summary_table, changes, combination_table, combination_change_table, unique_table, unique_change_table, prefix_table, null_trials):
    metrics_dir = Path(args.benchmark_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = metrics_dir / "multistar_challenger_calibration_convergence_thresholds.csv"
    labels_path = metrics_dir / "multistar_challenger_calibration_convergence_labels.csv"
    summary_path = metrics_dir / "multistar_challenger_calibration_convergence_pipeline_summary.csv"
    changes_path = metrics_dir / "multistar_challenger_calibration_convergence_label_changes.csv"
    combination_path = metrics_dir / "multistar_challenger_calibration_convergence_union_summary.csv"
    combination_changes_path = metrics_dir / "multistar_challenger_calibration_convergence_union_changes.csv"
    unique_path = metrics_dir / "multistar_challenger_calibration_convergence_unique_summary.csv"
    unique_changes_path = metrics_dir / "multistar_challenger_calibration_convergence_unique_changes.csv"
    prefix_path = metrics_dir / "multistar_challenger_calibration_convergence_trial_prefix.csv"
    json_path = metrics_dir / "multistar_challenger_calibration_convergence_summary.json"
    thresholds.to_csv(thresholds_path, index=False)
    labels.to_csv(labels_path, index=False)
    summary_table.to_csv(summary_path, index=False)
    changes.to_csv(changes_path, index=False)
    combination_table.to_csv(combination_path, index=False)
    combination_change_table.to_csv(combination_changes_path, index=False)
    unique_table.to_csv(unique_path, index=False)
    unique_change_table.to_csv(unique_changes_path, index=False)
    prefix_table.to_csv(prefix_path, index=False)
    available = available_levels(null_trials, args.levels)
    complete_prefix_rows = int(prefix_table["included_in_analysis"].sum()) if "included_in_analysis" in prefix_table.columns else 0
    payload = {"benchmark_dir": str(args.benchmark_dir), "requested_levels": list(args.levels), "available_complete_levels": available, "missing_levels": [int(level) for level in args.levels if int(level) not in available], "level_policy": "nested_prefix_trials_0_to_level_minus_1", "fap_level": float(args.fap_level), "bootstrap_iterations": int(args.bootstrap_iterations), "pipeline_count": int(len(args.pipelines)), "pipelines": list(args.pipelines), "threshold_rows": int(len(thresholds)), "label_rows": int(len(labels)), "label_change_rows": int(len(changes)), "union_summary_rows": int(len(combination_table)), "union_change_rows": int(len(combination_change_table)), "unique_summary_rows": int(len(unique_table)), "unique_change_rows": int(len(unique_change_table)), "trial_prefix_rows_included": complete_prefix_rows, "interpretation": "Only nested prefix levels present for every calibrated star are included. A stable protocol requires threshold, recovery-label, union, and unique-recovery changes to become materially small across increasing null counts."}
    json_path.write_text(json.dumps(json_ready(payload), indent=2) + "\n")
    return thresholds_path, labels_path, summary_path, changes_path, combination_path, combination_changes_path, unique_path, unique_changes_path, prefix_path, json_path, payload

def run_analysis(args):
    injections = load_injections(args)
    null_trials = load_null_trials(args)
    thresholds, labels = threshold_records(null_trials, injections, args)
    summary_table = pipeline_summary(labels, args)
    changes = label_changes(labels)
    combination_table = combination_summary(labels, args)
    combination_change_table = combination_changes(labels, args)
    unique_table = unique_summary(labels, args)
    unique_change_table = unique_changes(labels, args)
    prefix_table = prefix_integrity(null_trials, args.levels)
    return (*save_outputs(args, thresholds, labels, summary_table, changes, combination_table, combination_change_table, unique_table, unique_change_table, prefix_table, null_trials), summary_table, changes)

def main(args=None):
    args = args or parse_args()
    thresholds_path, labels_path, summary_path, changes_path, combination_path, combination_changes_path, unique_path, unique_changes_path, prefix_path, json_path, payload, summary_table, changes = run_analysis(args)
    print(f"Convergence thresholds: {thresholds_path}")
    print(f"Convergence labels: {labels_path}")
    print(f"Convergence pipeline summary: {summary_path}")
    print(f"Convergence label changes: {changes_path}")
    print(f"Convergence union summary: {combination_path}")
    print(f"Convergence union changes: {combination_changes_path}")
    print(f"Convergence unique summary: {unique_path}")
    print(f"Convergence unique changes: {unique_changes_path}")
    print(f"Convergence trial prefix: {prefix_path}")
    print(f"Convergence summary: {json_path}")
    print(f"Available complete levels: {payload['available_complete_levels']}")
    if payload["missing_levels"]:
        print(f"Missing levels: {payload['missing_levels']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
