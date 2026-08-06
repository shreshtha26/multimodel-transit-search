"""Validate the multi-star candidate reranker with leakage and robustness checks."""
import argparse
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import train_candidate_rerankers as rerank


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_DIR = PROJECT_ROOT / "outputs/experiments/multistar_bls_tcf/optimized/metrics"
DEFAULT_RERANKER_CONFIG_PATH = PROJECT_ROOT / "configs/candidate_reranker_clean_v1.json"
MERGE_TOLERANCE_FRACTION = 0.002

COMMON_CANDIDATE_FEATURES = ["raw_candidate_rank"]
BLS_FEATURES = [
    "has_bls_candidate",
    "bls_present",
    "bls_rank",
    "bls_sde",
    "bls_score_relative_to_rank1",
    "tcf_period_delta_to_bls_best_fraction",
    "candidate_to_bls_best_harmonic_error_fraction",
    "candidate_to_bls_best_harmonic_factor",
]
TCF_FEATURES = [
    "has_tcf_candidate",
    "tcf_present",
    "has_tcf_event_diagnostics",
    "tcf_rank",
    "tcf_score",
    "tcf_score_relative_to_rank1",
    "bls_period_delta_to_tcf_best_fraction",
    "candidate_to_tcf_best_harmonic_error_fraction",
    "candidate_to_tcf_best_harmonic_factor",
    "tcf_valid_transit_events",
    "tcf_positive_transit_events",
    "tcf_positive_event_fraction",
    "tcf_median_event_score",
    "tcf_raw_pooled_score",
]
AGREEMENT_FEATURES = [
    "source_detector",
    "detector_agreement",
    "detector_count",
    "detector_candidate_period_delta_fraction",
    "detector_candidate_harmonic_error_fraction",
    "detector_candidate_best_harmonic_factor",
    "bls_tcf_rank1_harmonic_error_fraction",
    "bls_tcf_rank1_harmonic_factor",
    "candidate_duration_hours",
    "candidate_depth",
]
NOISE_FEATURES = [
    "selection_group",
    "noise_quartile",
    "robust_flux_scatter_ppm",
    "gap_fraction",
    "lag_one_flux_acf",
    "six_hour_scatter_proxy_ppm",
    "arima_converged",
]
CALIBRATION_FEATURES = [
    "tcf_global_empirical_p_value",
    "bls_global_empirical_p_value",
    "tcf_regime_empirical_p_value",
    "bls_regime_empirical_p_value",
]


def default_settings():
    return SimpleNamespace(
        metrics_dir=DEFAULT_METRICS_DIR,
        candidate_dataset=None,
        reranker_config=DEFAULT_RERANKER_CONFIG_PATH,
        n_splits=5,
        holdout_star_count=10,
        random_state=20260806,
        bootstrap_samples=2000,
        permutation_repeats=10,
        xgb_estimators=300,
        xgb_max_depth=3,
        xgb_learning_rate=0.05,
        n_jobs=4,
        fap_level=0.01,
    )


def build_parser():
    defaults = default_settings()
    parser = argparse.ArgumentParser(description="Validate the multi-star BLS/ARIMA-TCF candidate reranker result.")
    parser.add_argument("--metrics-dir", type=Path, default=defaults.metrics_dir)
    parser.add_argument("--candidate-dataset", type=Path, default=defaults.candidate_dataset)
    parser.add_argument("--reranker-config", type=Path, default=defaults.reranker_config)
    parser.add_argument("--n-splits", type=int, default=defaults.n_splits)
    parser.add_argument("--holdout-star-count", type=int, default=defaults.holdout_star_count)
    parser.add_argument("--random-state", type=int, default=defaults.random_state)
    parser.add_argument("--bootstrap-samples", type=int, default=defaults.bootstrap_samples)
    parser.add_argument("--permutation-repeats", type=int, default=defaults.permutation_repeats)
    parser.add_argument("--xgb-estimators", type=int, default=defaults.xgb_estimators)
    parser.add_argument("--xgb-max-depth", type=int, default=defaults.xgb_max_depth)
    parser.add_argument("--xgb-learning-rate", type=float, default=defaults.xgb_learning_rate)
    parser.add_argument("--n-jobs", type=int, default=defaults.n_jobs)
    parser.add_argument("--fap-level", type=float, default=defaults.fap_level)
    return parser


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def bool_series(series):
    if series.dtype == bool:
        return series
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})


def normalize_key_columns(frame):
    frame = frame.copy()
    if "target_id" in frame.columns:
        frame["target_id"] = frame["target_id"].map(rerank.normalize_target_id)
    if "quarter" in frame.columns:
        frame["quarter"] = pd.to_numeric(frame["quarter"], errors="raise").astype(int)
    return frame


def period_fractional_error(first, second):
    if pd.isna(first) or pd.isna(second):
        return np.nan
    first = float(first)
    second = float(second)
    if second <= 0 or not np.isfinite(first):
        return np.nan
    return float(abs(first - second) / second)


def periods_match(first, second, tolerance_fraction=MERGE_TOLERANCE_FRACTION):
    if pd.isna(first) or pd.isna(second):
        return False
    first = float(first)
    second = float(second)
    denominator = min(abs(first), abs(second))
    return denominator > 0 and abs(first - second) / denominator <= float(tolerance_fraction)


def detector_harmonic_error(candidate_period, reference_period):
    if pd.isna(candidate_period) or pd.isna(reference_period):
        return np.nan, np.nan
    factors = (0.5, 1.0, 2.0, 3.0)
    errors = {factor: period_fractional_error(candidate_period, float(reference_period) * factor) for factor in factors}
    best_factor = min(errors, key=errors.get)
    return float(errors[best_factor]), float(best_factor)


def empirical_p_value(score, null_scores):
    if pd.isna(score):
        return np.nan
    null_scores = np.asarray(null_scores, dtype=float)
    null_scores = null_scores[np.isfinite(null_scores)]
    if null_scores.size == 0:
        return np.nan
    return float((np.sum(null_scores >= float(score)) + 1.0) / (len(null_scores) + 1.0))


def existing_columns(columns, available):
    return [column for column in columns if column in available]


def feature_set_definitions(feature_columns):
    available = set(feature_columns)
    bls_only = existing_columns(COMMON_CANDIDATE_FEATURES + BLS_FEATURES, available)
    tcf_only = existing_columns(COMMON_CANDIDATE_FEATURES + TCF_FEATURES, available)
    detector = existing_columns(COMMON_CANDIDATE_FEATURES + BLS_FEATURES + TCF_FEATURES + AGREEMENT_FEATURES, available)
    detector_noise = existing_columns(
        COMMON_CANDIDATE_FEATURES + BLS_FEATURES + TCF_FEATURES + AGREEMENT_FEATURES + NOISE_FEATURES,
        available,
    )
    no_absolute_period = [column for column in feature_columns if column not in {"candidate_period_days", "bls_candidate_period_days", "tcf_candidate_period_days"}]
    return {
        "A_bls_only": bls_only,
        "B_tcf_only": tcf_only,
        "C_bls_tcf_detector": detector,
        "D_detector_plus_stellar_noise": detector_noise,
        "E_full_model": feature_columns,
        "F_full_without_absolute_period": no_absolute_period,
    }


def xgb_classifier(random_state, args, scale_pos_weight=1.0):
    if rerank.xgb is None:
        raise RuntimeError("xgboost is not installed.")
    return rerank.xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=args.xgb_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=int(random_state),
        n_jobs=args.n_jobs,
        tree_method="hist",
        verbosity=0,
    )


def permute_labels_within_groups(train_frame, rng):
    labels = train_frame["exact_match"].astype(int).copy()
    for _, index in train_frame.groupby("cv_group_id", sort=False).groups.items():
        values = labels.loc[index].to_numpy(copy=True)
        rng.shuffle(values)
        labels.loc[index] = values
    return labels


def fit_xgb_classifier_oof(frame, feature_columns, args, permute_seed=None):
    predictions = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold in sorted(frame["fold"].unique()):
        train_mask = frame["fold"] != fold
        test_mask = frame["fold"] == fold
        train_frame = frame.loc[train_mask].copy()
        y_train = train_frame["exact_match"].astype(int)
        if permute_seed is not None:
            rng = np.random.default_rng(int(permute_seed) + int(fold))
            y_train = permute_labels_within_groups(train_frame, rng)
        positives = int(y_train.sum())
        negatives = int(len(y_train) - positives)
        scale_pos_weight = negatives / positives if positives else 1.0
        model = Pipeline(
            [
                ("preprocessor", rerank.make_preprocessor(frame, feature_columns, scale_numeric=False)),
                ("classifier", xgb_classifier(args.random_state + int(fold), args, scale_pos_weight)),
            ]
        )
        model.fit(train_frame[feature_columns], y_train)
        predictions.loc[test_mask] = model.predict_proba(frame.loc[test_mask, feature_columns])[:, 1]
    return predictions.to_numpy()


def fit_logistic_predict(train_frame, test_frame, feature_columns, args):
    model = Pipeline(
        [
            ("preprocessor", rerank.make_preprocessor(train_frame, feature_columns, scale_numeric=True)),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=args.random_state,
                ),
            ),
        ]
    )
    model.fit(train_frame[feature_columns], train_frame["exact_match"].astype(int))
    return model.predict_proba(test_frame[feature_columns])[:, 1]


def fit_xgb_classifier_predict(train_frame, test_frame, feature_columns, args):
    y_train = train_frame["exact_match"].astype(int)
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / positives if positives else 1.0
    model = Pipeline(
        [
            ("preprocessor", rerank.make_preprocessor(train_frame, feature_columns, scale_numeric=False)),
            ("classifier", xgb_classifier(args.random_state, args, scale_pos_weight)),
        ]
    )
    model.fit(train_frame[feature_columns], y_train)
    return model.predict_proba(test_frame[feature_columns])[:, 1], model


def fit_xgb_ranker_predict(train_frame, test_frame, feature_columns, args):
    if rerank.xgb is None:
        return None
    train_frame = train_frame.copy()
    positive_group_mask = train_frame.groupby("cv_group_id")["exact_match"].transform("any")
    rank_train_frame = train_frame.loc[positive_group_mask].sort_values("cv_group_id").copy()
    if rank_train_frame.empty:
        return None
    preprocessor = rerank.make_preprocessor(train_frame, feature_columns, scale_numeric=False)
    preprocessor.fit(train_frame[feature_columns])
    x_train = preprocessor.transform(rank_train_frame[feature_columns])
    y_train = rank_train_frame["exact_match"].astype(int).to_numpy()
    group_sizes = rank_train_frame.groupby("cv_group_id", sort=False).size().astype(int).to_list()
    ranker = rerank.xgb.XGBRanker(
        objective="rank:pairwise",
        eval_metric="ndcg@10",
        n_estimators=args.xgb_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        tree_method="hist",
        verbosity=0,
    )
    ranker.fit(x_train, y_train, group=group_sizes, verbose=False)
    return ranker.predict(preprocessor.transform(test_frame[feature_columns]))


def make_cv_predictions(frame, feature_columns, args):
    prediction_frames = [
        rerank.make_detector_predictions(frame, "bls"),
        rerank.make_detector_predictions(frame, "tcf"),
        rerank.make_scored_predictions(frame, "raw_min_rank", -frame["raw_candidate_rank"].to_numpy()),
        rerank.make_scored_predictions(frame, "rule_based", rerank.deterministic_rule_scores(frame)),
    ]
    prediction_frames.append(rerank.make_scored_predictions(frame, "logistic_regression", rerank.fit_logistic_oof(frame, feature_columns, args)))
    xgb_scores = fit_xgb_classifier_oof(frame, feature_columns, args)
    prediction_frames.append(rerank.make_scored_predictions(frame, "xgboost_classifier", xgb_scores))
    ranker_scores = rerank.fit_xgb_ranker_oof(frame, feature_columns, args)
    if ranker_scores is not None:
        prediction_frames.append(rerank.make_scored_predictions(frame, "xgboost_pairwise_ranker", ranker_scores))
    return pd.concat(prediction_frames, ignore_index=True)


def model_metrics(predictions):
    return pd.DataFrame([rerank.compute_metrics(part, model) for model, part in predictions.groupby("model", sort=False)])


def group_key_frame(frame):
    return frame[rerank.GROUP_COLUMNS].drop_duplicates().sort_values(rerank.GROUP_COLUMNS).reset_index(drop=True)


def group_success_from_predictions(predictions, model, label_column="exact_match"):
    model_predictions = predictions[predictions["model"] == model].copy()
    groups = group_key_frame(model_predictions)
    rank1 = model_predictions[pd.to_numeric(model_predictions["reranked_rank"], errors="coerce") == 1].copy()
    rank1_success = rank1.groupby(rerank.GROUP_COLUMNS, sort=False)[label_column].first().map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
    )
    return rank1_success.reindex(pd.MultiIndex.from_frame(groups), fill_value=False)


def exact_rank_series(predictions, model):
    model_predictions = predictions[predictions["model"] == model].copy()
    groups = pd.MultiIndex.from_frame(group_key_frame(model_predictions))
    exact_rows = model_predictions[bool_series(model_predictions["exact_match"])]
    ranks = exact_rows.groupby(rerank.GROUP_COLUMNS, sort=False)["reranked_rank"].min()
    return ranks.reindex(groups)


def audit_zero_worsened(frame, predictions, metrics_dir):
    xgb_predictions = predictions[predictions["model"] == "xgboost_classifier"].copy()
    candidate_group_counts = frame.groupby(rerank.GROUP_COLUMNS, sort=False).size()
    xgb_group_counts = xgb_predictions.groupby(rerank.GROUP_COLUMNS, sort=False).size().reindex(candidate_group_counts.index, fill_value=0)
    xgb_rank1_counts = (
        xgb_predictions[pd.to_numeric(xgb_predictions["reranked_rank"], errors="coerce") == 1]
        .groupby(rerank.GROUP_COLUMNS, sort=False)
        .size()
        .reindex(candidate_group_counts.index, fill_value=0)
    )
    bls_rank = pd.to_numeric(frame["bls_rank"], errors="coerce")
    bls_rank1_counts = frame[bls_rank == 1].groupby(rerank.GROUP_COLUMNS, sort=False).size().reindex(candidate_group_counts.index, fill_value=0)
    exact = bool_series(frame["exact_match"])
    bls_success = (
        frame[(bls_rank == 1) & exact]
        .groupby(rerank.GROUP_COLUMNS, sort=False)
        .size()
        .reindex(candidate_group_counts.index, fill_value=0)
        .astype(bool)
    )
    xgb_success = group_success_from_predictions(predictions, "xgboost_classifier").reindex(candidate_group_counts.index, fill_value=False)
    rows = []
    for bls_value in [True, False]:
        for xgb_value in [True, False]:
            rows.append(
                {
                    "bls_exact": "Yes" if bls_value else "No",
                    "xgboost_exact": "Yes" if xgb_value else "No",
                    "count": int(((bls_success == bls_value) & (xgb_success == xgb_value)).sum()),
                }
            )
    contingency = pd.DataFrame(rows)
    audit = {
        "candidate_group_count": int(len(candidate_group_counts)),
        "xgboost_prediction_group_count": int((xgb_group_counts > 0).sum()),
        "groups_with_prediction_row_count_mismatch": int((xgb_group_counts != candidate_group_counts).sum()),
        "groups_with_missing_xgboost_rank1": int((xgb_rank1_counts == 0).sum()),
        "groups_with_duplicate_xgboost_rank1": int((xgb_rank1_counts > 1).sum()),
        "groups_with_missing_bls_rank1": int((bls_rank1_counts == 0).sum()),
        "groups_with_duplicate_bls_rank1": int((bls_rank1_counts > 1).sum()),
        "bls_exact_rank1_count": int(bls_success.sum()),
        "xgboost_exact_rank1_count": int(xgb_success.sum()),
        "improved_count": int((~bls_success & xgb_success).sum()),
        "worsened_count": int((bls_success & ~xgb_success).sum()),
        "same_exact_match_label_used": True,
    }
    contingency_path = metrics_dir / "candidate_reranker_bls_xgb_contingency.csv"
    audit_path = metrics_dir / "candidate_reranker_zero_worsened_audit.json"
    contingency.to_csv(contingency_path, index=False)
    audit_path.write_text(json.dumps(json_ready(audit), indent=2) + "\n")
    return contingency, audit, contingency_path, audit_path


def lock_holdout_stars(frame, metrics_dir, args):
    path = metrics_dir / "candidate_reranker_locked_holdout_stars.json"
    stars = np.array(sorted(frame["target_id"].unique()))
    if path.exists():
        payload = json.loads(path.read_text())
        holdout = [rerank.normalize_target_id(star) for star in payload["holdout_stars"]]
        return holdout, path
    if args.holdout_star_count >= len(stars):
        raise ValueError("Holdout star count must be smaller than the number of available stars.")
    rng = np.random.default_rng(args.random_state)
    holdout = sorted(rng.choice(stars, size=int(args.holdout_star_count), replace=False).tolist())
    payload = {
        "random_state": int(args.random_state),
        "holdout_star_count": int(args.holdout_star_count),
        "holdout_stars": holdout,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return holdout, path


def holdout_predictions(train_frame, test_frame, feature_columns, args):
    test_frame = test_frame.copy()
    test_frame["fold"] = 0
    prediction_frames = [
        rerank.make_detector_predictions(test_frame, "bls"),
        rerank.make_detector_predictions(test_frame, "tcf"),
        rerank.make_scored_predictions(test_frame, "raw_min_rank", -test_frame["raw_candidate_rank"].to_numpy()),
        rerank.make_scored_predictions(test_frame, "rule_based", rerank.deterministic_rule_scores(test_frame)),
    ]
    prediction_frames.append(rerank.make_scored_predictions(test_frame, "logistic_regression", fit_logistic_predict(train_frame, test_frame, feature_columns, args)))
    xgb_scores, xgb_model = fit_xgb_classifier_predict(train_frame, test_frame, feature_columns, args)
    prediction_frames.append(rerank.make_scored_predictions(test_frame, "xgboost_classifier", xgb_scores))
    ranker_scores = fit_xgb_ranker_predict(train_frame, test_frame, feature_columns, args)
    if ranker_scores is not None:
        prediction_frames.append(rerank.make_scored_predictions(test_frame, "xgboost_pairwise_ranker", ranker_scores))
    return pd.concat(prediction_frames, ignore_index=True), xgb_model


def development_and_holdout_validation(frame, feature_columns, metrics_dir, args):
    holdout_stars, holdout_path = lock_holdout_stars(frame, metrics_dir, args)
    development_frame = frame[~frame["target_id"].isin(holdout_stars)].copy()
    holdout_frame = frame[frame["target_id"].isin(holdout_stars)].copy()
    development_frame["fold"] = rerank.assign_target_folds(development_frame, args.n_splits, args.random_state)
    development_predictions = make_cv_predictions(development_frame, feature_columns, args)
    development_metrics = model_metrics(development_predictions)
    holdout_pred, _ = holdout_predictions(development_frame, holdout_frame, feature_columns, args)
    holdout_metrics = model_metrics(holdout_pred)
    development_predictions_path = metrics_dir / "candidate_reranker_development_oof_predictions.csv"
    development_metrics_path = metrics_dir / "candidate_reranker_development_cv_metrics.csv"
    holdout_predictions_path = metrics_dir / "candidate_reranker_locked_holdout_predictions.csv"
    holdout_metrics_path = metrics_dir / "candidate_reranker_locked_holdout_metrics.csv"
    development_predictions.to_csv(development_predictions_path, index=False)
    development_metrics.to_csv(development_metrics_path, index=False)
    holdout_pred.to_csv(holdout_predictions_path, index=False)
    holdout_metrics.to_csv(holdout_metrics_path, index=False)
    return {
        "holdout_stars": holdout_stars,
        "holdout_path": holdout_path,
        "development_frame": development_frame,
        "holdout_frame": holdout_frame,
        "development_predictions": development_predictions,
        "development_metrics": development_metrics,
        "holdout_predictions": holdout_pred,
        "holdout_metrics": holdout_metrics,
        "development_predictions_path": development_predictions_path,
        "development_metrics_path": development_metrics_path,
        "holdout_predictions_path": holdout_predictions_path,
        "holdout_metrics_path": holdout_metrics_path,
    }


def feature_ablation_validation(development_frame, feature_columns, metrics_dir, args):
    rows = []
    for name, columns in feature_set_definitions(feature_columns).items():
        if not columns:
            continue
        scores = fit_xgb_classifier_oof(development_frame, columns, args)
        predictions = rerank.make_scored_predictions(development_frame, name, scores)
        metrics = rerank.compute_metrics(predictions, name)
        metrics["feature_set"] = name
        metrics["feature_count"] = int(len(columns))
        rows.append(metrics)
    ablation = pd.DataFrame(rows)
    path = metrics_dir / "candidate_reranker_feature_ablation_metrics.csv"
    ablation.to_csv(path, index=False)
    return ablation, path


def permutation_validation(development_frame, feature_columns, metrics_dir, args):
    rows = []
    for repeat in range(int(args.permutation_repeats)):
        scores = fit_xgb_classifier_oof(development_frame, feature_columns, args, permute_seed=args.random_state + 10_000 + repeat)
        predictions = rerank.make_scored_predictions(development_frame, f"permuted_exact_labels_{repeat}", scores)
        metrics = rerank.compute_metrics(predictions, f"permuted_exact_labels_{repeat}")
        metrics["repeat"] = int(repeat)
        rows.append(metrics)
    permutation = pd.DataFrame(rows)
    summary = permutation.drop(columns=["model"]).agg(["mean", "std", "min", "max"]).reset_index().rename(columns={"index": "statistic"})
    path = metrics_dir / "candidate_reranker_label_permutation_metrics.csv"
    summary_path = metrics_dir / "candidate_reranker_label_permutation_summary.csv"
    permutation.to_csv(path, index=False)
    summary.to_csv(summary_path, index=False)
    return permutation, summary, path, summary_path


def group_level_model_table(predictions, candidate_frame):
    groups = pd.MultiIndex.from_frame(group_key_frame(candidate_frame))
    table = pd.DataFrame(index=groups).reset_index()
    for model in ["bls_rank", "raw_min_rank", "xgboost_classifier"]:
        rank = exact_rank_series(predictions, model).reindex(groups)
        table[f"{model}_exact_rank"] = rank.to_numpy()
        table[f"{model}_exact_at_1"] = (rank <= 1).fillna(False).to_numpy()
        table[f"{model}_reciprocal_rank"] = (1.0 / rank).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    bls_rank = pd.to_numeric(candidate_frame["bls_rank"], errors="coerce")
    tcf_rank = pd.to_numeric(candidate_frame["tcf_rank"], errors="coerce")
    exact = bool_series(candidate_frame["exact_match"])
    oracle = (
        candidate_frame[((bls_rank == 1) | (tcf_rank == 1)) & exact]
        .groupby(rerank.GROUP_COLUMNS, sort=False)
        .size()
        .reindex(groups, fill_value=0)
        .astype(bool)
    )
    table["oracle_rank1_union_exact_at_1"] = oracle.to_numpy()
    return table


def star_bootstrap_intervals(predictions, candidate_frame, metrics_dir, args):
    group_table = group_level_model_table(predictions, candidate_frame)
    stars = np.array(sorted(group_table["target_id"].unique()))
    rng = np.random.default_rng(args.random_state + 30_000)
    rows = []
    for sample_index in range(int(args.bootstrap_samples)):
        sampled_stars = rng.choice(stars, size=len(stars), replace=True)
        sampled = pd.concat([group_table[group_table["target_id"] == star] for star in sampled_stars], ignore_index=True)
        rows.append(
            {
                "sample": sample_index,
                "exact_recall_at_1": float(sampled["xgboost_classifier_exact_at_1"].mean()),
                "improvement_over_bls": float(
                    (sampled["xgboost_classifier_exact_at_1"].astype(int) - sampled["bls_rank_exact_at_1"].astype(int)).mean()
                ),
                "improvement_over_raw_min_rank": float(
                    (sampled["xgboost_classifier_exact_at_1"].astype(int) - sampled["raw_min_rank_exact_at_1"].astype(int)).mean()
                ),
                "improvement_over_oracle_rank1_union": float(
                    (
                        sampled["xgboost_classifier_exact_at_1"].astype(int)
                        - sampled["oracle_rank1_union_exact_at_1"].astype(int)
                    ).mean()
                ),
                "mean_reciprocal_rank": float(sampled["xgboost_classifier_reciprocal_rank"].mean()),
            }
        )
    bootstrap = pd.DataFrame(rows)
    ci_rows = []
    for metric in [
        "exact_recall_at_1",
        "improvement_over_bls",
        "improvement_over_raw_min_rank",
        "improvement_over_oracle_rank1_union",
        "mean_reciprocal_rank",
    ]:
        values = bootstrap[metric].to_numpy(dtype=float)
        ci_rows.append(
            {
                "metric": metric,
                "estimate": float(group_table_metric(group_table, metric)),
                "ci_lower_95": float(np.quantile(values, 0.025)),
                "ci_upper_95": float(np.quantile(values, 0.975)),
                "bootstrap_samples": int(args.bootstrap_samples),
                "resampling_unit": "target_id",
            }
        )
    ci = pd.DataFrame(ci_rows)
    samples_path = metrics_dir / "candidate_reranker_star_bootstrap_samples.csv"
    ci_path = metrics_dir / "candidate_reranker_star_bootstrap_ci.csv"
    bootstrap.to_csv(samples_path, index=False)
    ci.to_csv(ci_path, index=False)
    return ci, samples_path, ci_path


def group_table_metric(group_table, metric):
    if metric == "exact_recall_at_1":
        return group_table["xgboost_classifier_exact_at_1"].mean()
    if metric == "improvement_over_bls":
        return (group_table["xgboost_classifier_exact_at_1"].astype(int) - group_table["bls_rank_exact_at_1"].astype(int)).mean()
    if metric == "improvement_over_raw_min_rank":
        return (group_table["xgboost_classifier_exact_at_1"].astype(int) - group_table["raw_min_rank_exact_at_1"].astype(int)).mean()
    if metric == "improvement_over_oracle_rank1_union":
        return (
            group_table["xgboost_classifier_exact_at_1"].astype(int) - group_table["oracle_rank1_union_exact_at_1"].astype(int)
        ).mean()
    if metric == "mean_reciprocal_rank":
        return group_table["xgboost_classifier_reciprocal_rank"].mean()
    raise KeyError(metric)


def add_analysis_bins(group_frame):
    group_frame = group_frame.copy()
    specs = {
        "depth_bin": "injected_depth",
        "duration_bin": "injected_duration_hours",
        "period_bin": "injected_period_days",
    }
    for bin_column, value_column in specs.items():
        values = pd.to_numeric(group_frame[value_column], errors="coerce")
        group_frame[bin_column] = pd.qcut(values, q=min(4, values.nunique(dropna=True)), duplicates="drop").astype(str)
    return group_frame


def first_period_at_rank(frame, detector):
    rank_column = f"{detector}_rank"
    period_column = f"{detector}_candidate_period_days"
    rank = pd.to_numeric(frame[rank_column], errors="coerce")
    return frame[rank == 1].groupby(rerank.GROUP_COLUMNS, sort=False)[period_column].first()


def candidate_miss_analysis(frame, predictions, metrics_dir):
    work = frame.copy()
    work["exact_bool"] = bool_series(work["exact_match"])
    work["harmonic_bool"] = bool_series(work["harmonic_match"])
    work["bls_harmonic_candidate_row"] = (work["bls_present"].astype(bool)) & work["harmonic_bool"]
    work["tcf_harmonic_candidate_row"] = (work["tcf_present"].astype(bool)) & work["harmonic_bool"]
    group_meta_columns = [
        "target_id",
        "quarter",
        "injection_index",
        "selection_group",
        "injected_period_days",
        "injected_duration_hours",
        "injected_depth",
        "noise_quartile",
        "robust_flux_scatter_ppm",
        "gap_fraction",
    ]
    group_frame = work[group_meta_columns].drop_duplicates(rerank.GROUP_COLUMNS).set_index(rerank.GROUP_COLUMNS)
    grouped = work.groupby(rerank.GROUP_COLUMNS, sort=False).agg(
        candidate_count=("candidate_id", "size"),
        exact_candidate_set=("exact_bool", "any"),
        harmonic_candidate_set=("harmonic_bool", "any"),
        bls_harmonic_candidate=("bls_harmonic_candidate_row", "any"),
        tcf_harmonic_candidate=("tcf_harmonic_candidate_row", "any"),
    )
    group_frame = group_frame.join(grouped)
    group_frame = group_frame.join(first_period_at_rank(work, "bls").rename("bls_rank1_candidate_period_days"))
    group_frame = group_frame.join(first_period_at_rank(work, "tcf").rename("tcf_rank1_candidate_period_days"))

    xgb = predictions[predictions["model"] == "xgboost_classifier"].copy()
    xgb_rank1 = xgb[pd.to_numeric(xgb["reranked_rank"], errors="coerce") == 1].set_index(rerank.GROUP_COLUMNS)
    group_frame["xgboost_rank1_candidate_period_days"] = xgb_rank1["candidate_period_days"]
    group_frame["xgboost_exact_rank1"] = bool_series(xgb_rank1["exact_match"]).reindex(group_frame.index, fill_value=False)

    group_frame = group_frame.reset_index()
    group_frame = add_analysis_bins(group_frame)
    misses = group_frame[~group_frame["exact_candidate_set"]].copy()
    ranking_failures = group_frame[group_frame["exact_candidate_set"] & ~group_frame["xgboost_exact_rank1"]].copy()

    summary_rows = [
        {
            "stratification": "overall",
            "bucket": "all",
            "injection_groups": int(len(group_frame)),
            "candidate_generation_misses": int((~group_frame["exact_candidate_set"]).sum()),
            "miss_rate": float((~group_frame["exact_candidate_set"]).mean()),
            "bls_harmonic_candidate_rate_among_misses": float(misses["bls_harmonic_candidate"].mean()) if len(misses) else np.nan,
            "tcf_harmonic_candidate_rate_among_misses": float(misses["tcf_harmonic_candidate"].mean()) if len(misses) else np.nan,
        }
    ]
    for stratum in ["depth_bin", "duration_bin", "period_bin", "noise_quartile", "target_id"]:
        for bucket, bucket_frame in group_frame.groupby(stratum, dropna=False, sort=True):
            bucket_misses = bucket_frame[~bucket_frame["exact_candidate_set"]]
            summary_rows.append(
                {
                    "stratification": stratum,
                    "bucket": str(bucket),
                    "injection_groups": int(len(bucket_frame)),
                    "candidate_generation_misses": int(len(bucket_misses)),
                    "miss_rate": float(len(bucket_misses) / len(bucket_frame)) if len(bucket_frame) else np.nan,
                    "bls_harmonic_candidate_rate_among_misses": float(bucket_misses["bls_harmonic_candidate"].mean())
                    if len(bucket_misses)
                    else np.nan,
                    "tcf_harmonic_candidate_rate_among_misses": float(bucket_misses["tcf_harmonic_candidate"].mean())
                    if len(bucket_misses)
                    else np.nan,
                }
            )
    summary = pd.DataFrame(summary_rows)
    misses_path = metrics_dir / "candidate_generation_misses.csv"
    ranking_failures_path = metrics_dir / "candidate_ranking_failures.csv"
    summary_path = metrics_dir / "candidate_generation_miss_summary.csv"
    misses.to_csv(misses_path, index=False)
    ranking_failures.to_csv(ranking_failures_path, index=False)
    summary.to_csv(summary_path, index=False)
    return misses, ranking_failures, summary, misses_path, ranking_failures_path, summary_path


def base_unlabeled_candidate(kind, row, candidate_period):
    return {
        "dataset_kind": kind,
        "target_id": rerank.normalize_target_id(row["target_id"]),
        "quarter": int(row["quarter"]),
        "trial": int(row.get("trial", -1)),
        "selection_group": row.get("selection_group", "unspecified"),
        "candidate_period_days": float(candidate_period),
        "source_detector": "",
        "detector_agreement": False,
        "detector_count": 0,
        "bls_present": False,
        "tcf_present": False,
        "has_bls_candidate": False,
        "has_tcf_candidate": False,
        "has_tcf_event_diagnostics": False,
        "bls_candidate_period_days": np.nan,
        "tcf_candidate_period_days": np.nan,
        "bls_rank": np.nan,
        "tcf_rank": np.nan,
        "bls_sde": np.nan,
        "tcf_score": np.nan,
        "bls_score_relative_to_rank1": np.nan,
        "tcf_score_relative_to_rank1": np.nan,
        "detector_candidate_period_delta_fraction": np.nan,
        "detector_candidate_harmonic_error_fraction": np.nan,
        "detector_candidate_best_harmonic_factor": np.nan,
        "bls_period_delta_to_tcf_best_fraction": np.nan,
        "tcf_period_delta_to_bls_best_fraction": np.nan,
        "candidate_to_tcf_best_harmonic_error_fraction": np.nan,
        "candidate_to_tcf_best_harmonic_factor": np.nan,
        "candidate_to_bls_best_harmonic_error_fraction": np.nan,
        "candidate_to_bls_best_harmonic_factor": np.nan,
        "bls_tcf_rank1_harmonic_error_fraction": np.nan,
        "bls_tcf_rank1_harmonic_factor": np.nan,
        "tcf_valid_transit_events": np.nan,
        "tcf_positive_transit_events": np.nan,
        "tcf_positive_event_fraction": np.nan,
        "tcf_median_event_score": np.nan,
        "tcf_raw_pooled_score": np.nan,
        "candidate_duration_hours": np.nan,
        "candidate_depth": np.nan,
        "noise_quartile": row.get("noise_quartile", "unassigned"),
        "robust_flux_scatter_ppm": float(row.get("robust_flux_scatter_ppm", np.nan)),
        "gap_fraction": float(row.get("gap_fraction", np.nan)),
        "lag_one_flux_acf": float(row.get("lag_one_flux_acf", np.nan)),
        "six_hour_scatter_proxy_ppm": float(row.get("six_hour_scatter_proxy_ppm", np.nan)),
        "arima_converged": int(str(row.get("base_arima_converged", row.get("arima_converged", False))).lower() in {"true", "1", "yes"}),
        "tcf_global_empirical_p_value": np.nan,
        "bls_global_empirical_p_value": np.nan,
        "tcf_regime_empirical_p_value": np.nan,
        "bls_regime_empirical_p_value": np.nan,
        "raw_candidate_rank": 1.0,
    }


def add_unlabeled_detector_candidate(candidates, kind, row, detector, period, score, null_scores_by_detector, null_scores_by_detector_regime):
    if pd.isna(period) or pd.isna(score):
        return
    match_index = None
    for index, candidate in enumerate(candidates):
        if periods_match(candidate["candidate_period_days"], period):
            match_index = index
            break
    if match_index is None:
        candidates.append(base_unlabeled_candidate(kind, row, period))
        match_index = len(candidates) - 1
    candidate = candidates[match_index]
    candidate[f"{detector}_present"] = True
    candidate[f"has_{detector}_candidate"] = True
    candidate[f"{detector}_candidate_period_days"] = float(period)
    candidate[f"{detector}_rank"] = 1
    if detector == "bls":
        candidate["bls_sde"] = float(score)
        candidate["bls_score_relative_to_rank1"] = 1.0
        candidate["bls_global_empirical_p_value"] = empirical_p_value(score, null_scores_by_detector["bls"])
        candidate["bls_regime_empirical_p_value"] = empirical_p_value(
            score, null_scores_by_detector_regime["bls"].get(str(row.get("noise_quartile")), [])
        )
    else:
        candidate["tcf_score"] = float(score)
        candidate["tcf_score_relative_to_rank1"] = 1.0
        candidate["tcf_valid_transit_events"] = float(row.get("tcf_valid_transit_events", row.get("original_tcf_valid_transit_events", np.nan)))
        candidate["tcf_positive_event_fraction"] = float(
            row.get("tcf_positive_event_fraction", row.get("original_tcf_positive_event_fraction", np.nan))
        )
        candidate["tcf_raw_pooled_score"] = float(row.get("original_tcf_raw_pooled_score", np.nan))
        candidate["has_tcf_event_diagnostics"] = bool(np.isfinite(candidate["tcf_valid_transit_events"]))
        candidate["tcf_global_empirical_p_value"] = empirical_p_value(score, null_scores_by_detector["tcf"])
        candidate["tcf_regime_empirical_p_value"] = empirical_p_value(
            score, null_scores_by_detector_regime["tcf"].get(str(row.get("noise_quartile")), [])
        )


def finalize_unlabeled_candidates(candidates, bls_best_period, tcf_best_period):
    finalized = []
    for candidate in candidates:
        candidate = dict(candidate)
        candidate["detector_count"] = int(candidate["bls_present"]) + int(candidate["tcf_present"])
        candidate["detector_agreement"] = bool(candidate["detector_count"] == 2)
        if candidate["detector_agreement"]:
            candidate["source_detector"] = "both"
        elif candidate["bls_present"]:
            candidate["source_detector"] = "bls"
        elif candidate["tcf_present"]:
            candidate["source_detector"] = "tcf"
        candidate["bls_period_delta_to_tcf_best_fraction"] = period_fractional_error(candidate["candidate_period_days"], tcf_best_period)
        candidate["tcf_period_delta_to_bls_best_fraction"] = period_fractional_error(candidate["candidate_period_days"], bls_best_period)
        candidate["candidate_to_tcf_best_harmonic_error_fraction"], candidate["candidate_to_tcf_best_harmonic_factor"] = detector_harmonic_error(
            candidate["candidate_period_days"], tcf_best_period
        )
        candidate["candidate_to_bls_best_harmonic_error_fraction"], candidate["candidate_to_bls_best_harmonic_factor"] = detector_harmonic_error(
            candidate["candidate_period_days"], bls_best_period
        )
        candidate["bls_tcf_rank1_harmonic_error_fraction"], candidate["bls_tcf_rank1_harmonic_factor"] = detector_harmonic_error(
            bls_best_period, tcf_best_period
        )
        if candidate["detector_agreement"]:
            candidate["detector_candidate_period_delta_fraction"] = period_fractional_error(
                candidate["bls_candidate_period_days"], candidate["tcf_candidate_period_days"]
            )
            candidate["detector_candidate_harmonic_error_fraction"], candidate["detector_candidate_best_harmonic_factor"] = detector_harmonic_error(
                candidate["bls_candidate_period_days"], candidate["tcf_candidate_period_days"]
            )
        finalized.append(candidate)
    return finalized


def build_unlabeled_candidates(metrics_dir, null_scores_by_detector, null_scores_by_detector_regime):
    null_trials = pd.read_csv(metrics_dir / "multistar_null_trials.csv", dtype={"target_id": str})
    star_summaries = pd.read_csv(metrics_dir / "multistar_star_summary.csv", dtype={"target_id": str})
    null_trials = normalize_key_columns(null_trials)
    star_summaries = normalize_key_columns(star_summaries)
    summary_columns = [
        "target_id",
        "quarter",
        "selection_group",
        "noise_quartile",
        "robust_flux_scatter_ppm",
        "gap_fraction",
        "lag_one_flux_acf",
        "six_hour_scatter_proxy_ppm",
        "base_arima_converged",
    ]
    null_trials = null_trials.merge(star_summaries[summary_columns], on=["target_id", "quarter"], how="left", validate="many_to_one")
    rows = []
    for _, row in null_trials.iterrows():
        candidates = []
        if str(row.get("bls_success")).lower() in {"true", "1", "yes"}:
            add_unlabeled_detector_candidate(
                candidates,
                "null",
                row,
                "bls",
                row.get("bls_best_period_days"),
                row.get("bls_max_sde"),
                null_scores_by_detector,
                null_scores_by_detector_regime,
            )
        if str(row.get("tcf_success")).lower() in {"true", "1", "yes"}:
            add_unlabeled_detector_candidate(
                candidates,
                "null",
                row,
                "tcf",
                row.get("tcf_best_period_days"),
                row.get("tcf_max_score"),
                null_scores_by_detector,
                null_scores_by_detector_regime,
            )
        rows.extend(finalize_unlabeled_candidates(candidates, row.get("bls_best_period_days"), row.get("tcf_best_period_days")))

    for _, row in star_summaries.iterrows():
        candidates = []
        add_unlabeled_detector_candidate(
            candidates,
            "original",
            row,
            "bls",
            row.get("original_bls_period_days"),
            row.get("original_bls_sde"),
            null_scores_by_detector,
            null_scores_by_detector_regime,
        )
        add_unlabeled_detector_candidate(
            candidates,
            "original",
            row,
            "tcf",
            row.get("original_tcf_period_days"),
            row.get("original_tcf_score"),
            null_scores_by_detector,
            null_scores_by_detector_regime,
        )
        rows.extend(finalize_unlabeled_candidates(candidates, row.get("original_bls_period_days"), row.get("original_tcf_period_days")))

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    candidates["candidate_id"] = np.arange(len(candidates), dtype=int)
    return candidates


def null_score_distributions(metrics_dir):
    null_trials = pd.read_csv(metrics_dir / "multistar_null_trials.csv", dtype={"target_id": str})
    star_summaries = pd.read_csv(metrics_dir / "multistar_star_summary.csv", dtype={"target_id": str})
    null_trials = normalize_key_columns(null_trials)
    star_summaries = normalize_key_columns(star_summaries)
    null_trials = null_trials.merge(star_summaries[["target_id", "quarter", "noise_quartile"]], on=["target_id", "quarter"], how="left")
    tcf_success = bool_series(null_trials["tcf_success"])
    bls_success = bool_series(null_trials["bls_success"])
    global_scores = {
        "tcf": null_trials.loc[tcf_success, "tcf_max_score"].to_numpy(dtype=float),
        "bls": null_trials.loc[bls_success, "bls_max_sde"].to_numpy(dtype=float),
    }
    regime_scores = {"tcf": {}, "bls": {}}
    for noise_quartile, group in null_trials.groupby("noise_quartile", dropna=False):
        regime_scores["tcf"][str(noise_quartile)] = group.loc[bool_series(group["tcf_success"]), "tcf_max_score"].to_numpy(dtype=float)
        regime_scores["bls"][str(noise_quartile)] = group.loc[bool_series(group["bls_success"]), "bls_max_sde"].to_numpy(dtype=float)
    return global_scores, regime_scores


def rank_unlabeled_predictions(frame):
    frame = frame.copy()
    frame["reranked_rank"] = 999_999
    group_columns = ["dataset_kind", "target_id", "quarter", "trial"]
    for _, index in frame.groupby(group_columns, sort=False).groups.items():
        group = frame.loc[index]
        scores = pd.to_numeric(group["predicted_score"], errors="coerce").fillna(-np.inf).to_numpy()
        raw_rank = pd.to_numeric(group["raw_candidate_rank"], errors="coerce").fillna(999_999).to_numpy()
        candidate_id = pd.to_numeric(group["candidate_id"], errors="coerce").fillna(999_999).to_numpy()
        order = np.lexsort((candidate_id, raw_rank, -scores))
        ranks = np.empty(len(group), dtype=int)
        ranks[order] = np.arange(1, len(group) + 1)
        frame.loc[group.index, "reranked_rank"] = ranks
    return frame


def probability_threshold(values, fap_level):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.quantile(values, 1.0 - float(fap_level), method="higher"))


def reliability_curve(xgb_oof_predictions):
    work = xgb_oof_predictions.copy()
    work["exact_bool"] = bool_series(work["exact_match"])
    top = work[pd.to_numeric(work["reranked_rank"], errors="coerce") == 1].copy()
    top["score_bin"] = pd.cut(pd.to_numeric(top["predicted_score"], errors="coerce"), bins=np.linspace(0.0, 1.0, 11), include_lowest=True)
    curve = top.groupby("score_bin", dropna=False, observed=False).agg(
        candidate_count=("predicted_score", "size"),
        mean_predicted_score=("predicted_score", "mean"),
        observed_exact_rate=("exact_bool", "mean"),
    ).reset_index()
    curve["score_bin"] = curve["score_bin"].astype(str)
    brier_top = float(np.mean((pd.to_numeric(top["predicted_score"], errors="coerce").to_numpy() - top["exact_bool"].astype(float).to_numpy()) ** 2))
    brier_rows = float(
        np.mean(
            (
                pd.to_numeric(work["predicted_score"], errors="coerce").to_numpy()
                - work["exact_bool"].astype(float).to_numpy()
            )
            ** 2
        )
    )
    return curve, brier_top, brier_rows


def reranker_probability_calibration(frame, predictions, feature_columns, metrics_dir, args):
    null_scores, null_scores_regime = null_score_distributions(metrics_dir)
    unlabeled = build_unlabeled_candidates(metrics_dir, null_scores, null_scores_regime)
    if unlabeled.empty:
        raise RuntimeError("No null/original candidates are available for reranker calibration.")
    for column in feature_columns:
        if column not in unlabeled.columns:
            unlabeled[column] = np.nan
    _, model = fit_xgb_classifier_predict(frame, frame.iloc[:1].copy(), feature_columns, args)
    unlabeled["predicted_score"] = model.predict_proba(unlabeled[feature_columns])[:, 1]
    unlabeled = rank_unlabeled_predictions(unlabeled)

    null_top = unlabeled[(unlabeled["dataset_kind"] == "null") & (unlabeled["reranked_rank"] == 1)].copy()
    original_top = unlabeled[(unlabeled["dataset_kind"] == "original") & (unlabeled["reranked_rank"] == 1)].copy()
    threshold = probability_threshold(null_top["predicted_score"], args.fap_level)

    xgb_oof = predictions[predictions["model"] == "xgboost_classifier"].copy()
    injection_top = xgb_oof[pd.to_numeric(xgb_oof["reranked_rank"], errors="coerce") == 1].copy()
    injection_top["passes_reranker_global_fap"] = pd.to_numeric(injection_top["predicted_score"], errors="coerce") >= threshold
    injection_top["exact_bool"] = bool_series(injection_top["exact_match"])
    original_top["passes_reranker_global_fap"] = pd.to_numeric(original_top["predicted_score"], errors="coerce") >= threshold
    null_top["passes_reranker_global_fap"] = pd.to_numeric(null_top["predicted_score"], errors="coerce") >= threshold

    regime_rows = []
    for noise_quartile, null_group in null_top.groupby("noise_quartile", dropna=False):
        local_threshold = probability_threshold(null_group["predicted_score"], args.fap_level)
        injection_group = injection_top[injection_top["noise_quartile"].astype(str) == str(noise_quartile)].copy()
        original_group = original_top[original_top["noise_quartile"].astype(str) == str(noise_quartile)].copy()
        regime_rows.append(
            {
                "noise_quartile": noise_quartile,
                "null_trials": int(len(null_group)),
                "threshold": local_threshold,
                "observed_null_exceedance_rate": float((null_group["predicted_score"] >= local_threshold).mean())
                if np.isfinite(local_threshold)
                else np.nan,
                "injection_groups": int(len(injection_group)),
                "exact_recovery_at_threshold": float(
                    (bool_series(injection_group["exact_match"]) & (injection_group["predicted_score"] >= local_threshold)).mean()
                )
                if len(injection_group) and np.isfinite(local_threshold)
                else np.nan,
                "original_light_curves": int(len(original_group)),
                "original_candidate_fraction": float((original_group["predicted_score"] >= local_threshold).mean())
                if len(original_group) and np.isfinite(local_threshold)
                else np.nan,
            }
        )
    regime = pd.DataFrame(regime_rows)
    curve, brier_top, brier_rows = reliability_curve(xgb_oof)
    summary = {
        "calibration_status": "preliminary_rank1_only_not_final",
        "fap_level": float(args.fap_level),
        "global_probability_threshold": threshold,
        "null_trials": int(len(null_top)),
        "observed_null_exceedance_rate": float(null_top["passes_reranker_global_fap"].mean()),
        "injection_groups": int(len(injection_top)),
        "exact_recall_at_1_without_threshold": float(injection_top["exact_bool"].mean()),
        "exact_recovery_at_global_threshold": float(
            (injection_top["exact_bool"] & injection_top["passes_reranker_global_fap"]).mean()
        ),
        "candidate_fraction_at_global_threshold": float(injection_top["passes_reranker_global_fap"].mean()),
        "original_light_curves": int(len(original_top)),
        "original_candidate_fraction_at_global_threshold": float(original_top["passes_reranker_global_fap"].mean()),
        "top_candidate_brier_score": brier_top,
        "row_level_brier_score": brier_rows,
        "note": "Preliminary only: null/original calibration uses rank-1 detector candidates because this validation path does not regenerate top-k null/original candidate lists. Use run_multistar_reranker_topk_calibration.py for the matched top-k calibration experiment.",
    }

    candidates_path = metrics_dir / "reranker_null_original_candidate_scores.csv"
    top_path = metrics_dir / "reranker_null_original_top_scores.csv"
    regime_path = metrics_dir / "reranker_probability_calibration_by_noise_quartile.csv"
    curve_path = metrics_dir / "candidate_reranker_reliability_curve.csv"
    summary_path = metrics_dir / "reranker_probability_calibration_summary.json"
    unlabeled.to_csv(candidates_path, index=False)
    pd.concat([null_top, original_top], ignore_index=True).to_csv(top_path, index=False)
    regime.to_csv(regime_path, index=False)
    curve.to_csv(curve_path, index=False)
    summary_path.write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    return summary, candidates_path, top_path, regime_path, curve_path, summary_path


def load_existing_predictions(metrics_dir):
    path = metrics_dir / "candidate_reranker_oof_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run scripts/train_candidate_rerankers.py first; missing {path}")
    predictions = pd.read_csv(path, dtype={"target_id": str})
    return normalize_key_columns(predictions), path


def run_validation(args):
    metrics_dir = Path(args.metrics_dir)
    reranker_spec, reranker_config_path, reranker_config_sha256 = rerank.load_reranker_config(args.reranker_config)
    frame, candidate_path = rerank.load_candidates(args)
    predictions, predictions_path = load_existing_predictions(metrics_dir)
    feature_columns = rerank.leakage_safe_feature_columns(frame, reranker_spec)
    forbidden_overlap = sorted(set(feature_columns).intersection(rerank.FORBIDDEN_FEATURE_COLUMNS))
    injected_prefix_features = sorted(column for column in feature_columns if column.startswith("injected_"))
    if forbidden_overlap or injected_prefix_features:
        raise RuntimeError(f"Leakage features selected: {forbidden_overlap + injected_prefix_features}")

    contingency, audit, contingency_path, audit_path = audit_zero_worsened(frame, predictions, metrics_dir)
    holdout = development_and_holdout_validation(frame, feature_columns, metrics_dir, args)
    development_frame = holdout["development_frame"]
    ablation, ablation_path = feature_ablation_validation(development_frame, feature_columns, metrics_dir, args)
    permutation, permutation_summary, permutation_path, permutation_summary_path = permutation_validation(
        development_frame,
        feature_columns,
        metrics_dir,
        args,
    )
    bootstrap_ci, bootstrap_samples_path, bootstrap_ci_path = star_bootstrap_intervals(predictions, frame, metrics_dir, args)
    misses, ranking_failures, miss_summary, misses_path, ranking_failures_path, miss_summary_path = candidate_miss_analysis(
        frame, predictions, metrics_dir
    )
    calibration_summary, calibration_candidates_path, calibration_top_path, calibration_regime_path, reliability_path, calibration_summary_path = (
        reranker_probability_calibration(frame, predictions, feature_columns, metrics_dir, args)
    )

    summary = {
        "reranker_version": reranker_spec.get("version") if reranker_spec else "dynamic",
        "reranker_config": reranker_config_path,
        "reranker_config_sha256": reranker_config_sha256,
        "candidate_dataset": candidate_path,
        "existing_oof_predictions": predictions_path,
        "feature_count": int(len(feature_columns)),
        "forbidden_feature_overlap": forbidden_overlap,
        "injected_prefix_features": injected_prefix_features,
        "zero_worsened_audit": audit,
        "holdout_stars": holdout["holdout_stars"],
        "outputs": {
            "zero_worsened_contingency": contingency_path,
            "zero_worsened_audit": audit_path,
            "development_cv_metrics": holdout["development_metrics_path"],
            "development_oof_predictions": holdout["development_predictions_path"],
            "holdout_stars": holdout["holdout_path"],
            "holdout_metrics": holdout["holdout_metrics_path"],
            "holdout_predictions": holdout["holdout_predictions_path"],
            "feature_ablation_metrics": ablation_path,
            "label_permutation_metrics": permutation_path,
            "label_permutation_summary": permutation_summary_path,
            "star_bootstrap_samples": bootstrap_samples_path,
            "star_bootstrap_ci": bootstrap_ci_path,
            "candidate_generation_misses": misses_path,
            "candidate_ranking_failures": ranking_failures_path,
            "candidate_generation_miss_summary": miss_summary_path,
            "calibration_candidates": calibration_candidates_path,
            "calibration_top_scores": calibration_top_path,
            "calibration_by_noise_quartile": calibration_regime_path,
            "reliability_curve": reliability_path,
            "calibration_summary": calibration_summary_path,
        },
        "counts": {
            "candidate_generation_misses": int(len(misses)),
            "candidate_ranking_failures": int(len(ranking_failures)),
        },
        "calibration_summary": calibration_summary,
    }
    summary_path = metrics_dir / "candidate_reranker_validation_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    return {
        "summary_path": summary_path,
        "contingency": contingency,
        "audit": audit,
        "holdout_metrics": holdout["holdout_metrics"],
        "ablation": ablation,
        "permutation_summary": permutation_summary,
        "bootstrap_ci": bootstrap_ci,
        "miss_summary": miss_summary,
        "calibration_summary": calibration_summary,
    }


def main(args=None):
    args = args or build_parser().parse_args()
    result = run_validation(args)
    print(f"Validation summary: {result['summary_path']}")
    print("\nBLS vs XGBoost contingency:")
    print(result["contingency"].to_string(index=False))
    print("\nLocked holdout metrics:")
    print(
        result["holdout_metrics"][
            [
                "model",
                "exact_recall_at_1",
                "exact_recall_at_3",
                "exact_recall_at_5",
                "exact_recall_at_10",
                "harmonic_recall_at_1",
                "mean_reciprocal_rank",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print("\nFeature ablation:")
    print(result["ablation"][["feature_set", "feature_count", "exact_recall_at_1", "exact_recall_at_10", "mean_reciprocal_rank"]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nLabel permutation summary:")
    print(result["permutation_summary"][["statistic", "exact_recall_at_1", "exact_recall_at_10", "mean_reciprocal_rank"]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nStar bootstrap 95% CI:")
    print(result["bootstrap_ci"].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nCalibration summary:")
    print(json.dumps(json_ready(result["calibration_summary"]), indent=2))
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    raise SystemExit(main())
