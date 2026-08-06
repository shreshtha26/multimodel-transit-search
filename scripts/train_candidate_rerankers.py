"""Train grouped candidate rerankers for multi-star BLS/ARIMA-TCF candidates."""
import argparse
import hashlib
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency
    xgb = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_DIR = PROJECT_ROOT / "outputs/experiments/multistar_bls_tcf/optimized/metrics"
DEFAULT_RERANKER_CONFIG_PATH = PROJECT_ROOT / "configs/candidate_reranker_clean_v1.json"
GROUP_COLUMNS = ["target_id", "quarter", "injection_index"]
LABEL_COLUMNS = ["exact_match", "harmonic_match"]
KNOWN_CATEGORICAL_COLUMNS = ["source_detector", "selection_group", "noise_quartile"]
BOOLEAN_COLUMNS = [
    "detector_agreement",
    "bls_present",
    "tcf_present",
    "has_bls_candidate",
    "has_tcf_candidate",
    "has_tcf_event_diagnostics",
    "arima_converged",
]
FORBIDDEN_FEATURE_COLUMNS = {
    "candidate_id",
    "target_id",
    "quarter",
    "injection_index",
    "cv_group_id",
    "fold",
    "exact_match",
    "harmonic_match",
    "injected_period_days",
    "injected_epoch_days",
    "injected_depth",
    "injected_duration_hours",
    "epoch_phase_fraction",
    "candidate_period_error_fraction",
    "candidate_harmonic_error_fraction",
    "candidate_best_harmonic_factor",
    "in_transit_observation_count",
    "candidate_period_days",
    "bls_candidate_period_days",
    "tcf_candidate_period_days",
}
FORBIDDEN_FEATURE_PREFIXES = ("injected_",)


def default_settings():
    return SimpleNamespace(
        metrics_dir=DEFAULT_METRICS_DIR,
        candidate_dataset=None,
        reranker_config=DEFAULT_RERANKER_CONFIG_PATH,
        n_splits=5,
        random_state=42,
        xgb_estimators=300,
        xgb_max_depth=3,
        xgb_learning_rate=0.05,
        n_jobs=4,
    )


def build_parser():
    defaults = default_settings()
    parser = argparse.ArgumentParser(description="Train leakage-safe candidate rerankers with grouped CV.")
    parser.add_argument("--metrics-dir", type=Path, default=defaults.metrics_dir)
    parser.add_argument("--candidate-dataset", type=Path, default=defaults.candidate_dataset)
    parser.add_argument("--reranker-config", type=Path, default=defaults.reranker_config)
    parser.add_argument("--n-splits", type=int, default=defaults.n_splits)
    parser.add_argument("--random-state", type=int, default=defaults.random_state)
    parser.add_argument("--xgb-estimators", type=int, default=defaults.xgb_estimators)
    parser.add_argument("--xgb-max-depth", type=int, default=defaults.xgb_max_depth)
    parser.add_argument("--xgb-learning-rate", type=float, default=defaults.xgb_learning_rate)
    parser.add_argument("--n-jobs", type=int, default=defaults.n_jobs)
    return parser


def file_sha256(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reranker_config(path):
    if path is None:
        return None, None, None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Frozen reranker config not found: {path}")
    spec = json.loads(path.read_text())
    required = {"version", "feature_columns", "forbidden_feature_columns"}
    missing = required.difference(spec)
    if missing:
        raise ValueError(f"Reranker config {path} is missing keys: {sorted(missing)}")
    return spec, path, file_sha256(path)


def normalize_target_id(value):
    return str(value).upper().replace("KIC", "").strip()


def to_bool_series(series):
    if series.dtype == bool:
        return series
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})


def normalize_candidate_frame(frame):
    frame = frame.copy()
    frame["target_id"] = frame["target_id"].map(normalize_target_id)
    frame["quarter"] = pd.to_numeric(frame["quarter"], errors="raise").astype(int)

    if "injection_index" not in frame.columns:
        legacy_group_columns = [
            "target_id",
            "quarter",
            "injected_period_days",
            "injected_duration_hours",
            "injected_depth",
            "epoch_phase_fraction",
        ]
        if not set(legacy_group_columns).issubset(frame.columns):
            raise ValueError("Candidate dataset must contain injection_index or the legacy injected-parameter columns.")
        unique_injections = frame[legacy_group_columns].drop_duplicates().copy()
        unique_injections["injection_index"] = unique_injections.groupby(["target_id", "quarter"], sort=False).cumcount()
        frame = frame.merge(unique_injections, on=legacy_group_columns, how="left", validate="many_to_one")
    frame["injection_index"] = pd.to_numeric(frame["injection_index"], errors="raise").astype(int)

    for label in LABEL_COLUMNS:
        frame[label] = to_bool_series(frame[label])

    if "has_bls_candidate" not in frame.columns:
        frame["has_bls_candidate"] = frame.get("bls_present", pd.Series(False, index=frame.index))
    if "has_tcf_candidate" not in frame.columns:
        frame["has_tcf_candidate"] = frame.get("tcf_present", pd.Series(False, index=frame.index))
    if "has_tcf_event_diagnostics" not in frame.columns:
        frame["has_tcf_event_diagnostics"] = pd.to_numeric(
            frame.get("tcf_valid_transit_events", pd.Series(np.nan, index=frame.index)), errors="coerce"
        ).notna()

    for column in BOOLEAN_COLUMNS:
        if column in frame.columns:
            frame[column] = to_bool_series(frame[column]).astype(int)

    if "candidate_id" not in frame.columns:
        frame["candidate_id"] = np.arange(len(frame), dtype=int)
    frame["cv_group_id"] = frame[GROUP_COLUMNS].astype(str).agg("|".join, axis=1)
    frame["raw_candidate_rank"] = compute_raw_candidate_rank(frame)
    return frame


def compute_raw_candidate_rank(frame):
    rank_values = pd.concat(
        [
            pd.to_numeric(frame.get("bls_rank", pd.Series(np.nan, index=frame.index)), errors="coerce"),
            pd.to_numeric(frame.get("tcf_rank", pd.Series(np.nan, index=frame.index)), errors="coerce"),
        ],
        axis=1,
    )
    raw_rank = rank_values.min(axis=1, skipna=True)
    max_rank = rank_values.max(skipna=True).max()
    fill_rank = int(max_rank + 1) if np.isfinite(max_rank) else 999_999
    return raw_rank.fillna(fill_rank).astype(float)


def load_candidates(args):
    metrics_dir = Path(args.metrics_dir)
    candidate_path = Path(args.candidate_dataset) if args.candidate_dataset else metrics_dir / "multistar_candidate_reranking_dataset.csv"
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate dataset not found: {candidate_path}")
    frame = pd.read_csv(candidate_path, dtype={"target_id": str})
    return normalize_candidate_frame(frame), candidate_path


def leakage_safe_feature_columns(frame, reranker_spec=None):
    forbidden_columns = set(FORBIDDEN_FEATURE_COLUMNS)
    forbidden_prefixes = set(FORBIDDEN_FEATURE_PREFIXES)
    if reranker_spec is not None:
        forbidden_columns.update(reranker_spec.get("forbidden_feature_columns", []))
        forbidden_prefixes.update(reranker_spec.get("forbidden_feature_prefixes", []))
        columns = list(reranker_spec["feature_columns"])
        missing_columns = [column for column in columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Frozen reranker feature columns are missing from candidate dataset: {missing_columns}")
    else:
        columns = []
        for column in frame.columns:
            if column in forbidden_columns:
                continue
            if any(column.startswith(prefix) for prefix in forbidden_prefixes):
                continue
            columns.append(column)
    forbidden_used = sorted(set(columns).intersection(forbidden_columns))
    forbidden_used.extend(sorted(column for column in columns if any(column.startswith(prefix) for prefix in forbidden_prefixes)))
    if forbidden_used:
        version = reranker_spec.get("version", "dynamic") if reranker_spec else "dynamic"
        raise ValueError(f"Leakage columns selected as features for {version}: {forbidden_used}")
    return columns


def make_preprocessor(frame, feature_columns, scale_numeric):
    categorical_columns = [
        column
        for column in feature_columns
        if column in KNOWN_CATEGORICAL_COLUMNS or str(frame[column].dtype) in {"object", "category"}
    ]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def assign_target_folds(frame, n_splits, random_state):
    unique_targets = frame["target_id"].nunique()
    n_splits = min(int(n_splits), int(unique_targets))
    if n_splits < 2:
        raise ValueError("At least two target_id groups are required for grouped cross-validation.")
    try:
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    except TypeError:  # pragma: no cover - old scikit-learn compatibility
        splitter = GroupKFold(n_splits=n_splits)
    folds = np.full(len(frame), -1, dtype=int)
    y = frame["exact_match"].astype(int).to_numpy()
    groups = frame["target_id"].to_numpy()
    for fold, (_, test_index) in enumerate(splitter.split(frame, y, groups=groups)):
        folds[test_index] = fold
    if np.any(folds < 0):
        raise RuntimeError("Some rows were not assigned to a cross-validation fold.")
    for fold in sorted(np.unique(folds)):
        train_targets = set(frame.loc[folds != fold, "target_id"])
        test_targets = set(frame.loc[folds == fold, "target_id"])
        overlap = train_targets.intersection(test_targets)
        if overlap:
            raise RuntimeError(f"Fold {fold} leaks targets between train and test: {sorted(overlap)}")
    return folds


def detector_rank_scores(frame, detector):
    rank = pd.to_numeric(frame.get(f"{detector}_rank", pd.Series(np.nan, index=frame.index)), errors="coerce")
    if detector == "bls":
        strength = pd.to_numeric(frame.get("bls_sde", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    else:
        strength = pd.to_numeric(frame.get("tcf_score", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    return np.where(rank.notna(), -rank.to_numpy() + 1e-6 * strength.to_numpy(), -1e9)


def deterministic_rule_scores(frame):
    bls_rel = pd.to_numeric(frame.get("bls_score_relative_to_rank1", 0.0), errors="coerce").fillna(0.0)
    tcf_rel = pd.to_numeric(frame.get("tcf_score_relative_to_rank1", 0.0), errors="coerce").fillna(0.0)
    bls_sde = pd.to_numeric(frame.get("bls_sde", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    tcf_score = pd.to_numeric(frame.get("tcf_score", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    raw_rank = pd.to_numeric(frame["raw_candidate_rank"], errors="coerce").fillna(999_999)
    agreement = pd.to_numeric(frame.get("detector_agreement", 0), errors="coerce").fillna(0.0)
    has_bls = pd.to_numeric(frame.get("has_bls_candidate", 0), errors="coerce").fillna(0.0)
    has_tcf = pd.to_numeric(frame.get("has_tcf_candidate", 0), errors="coerce").fillna(0.0)
    tcf_events = pd.to_numeric(frame.get("tcf_positive_event_fraction", 0.0), errors="coerce").fillna(0.0)
    return (
        4.0 * agreement
        + 1.5 * has_bls
        + 0.8 * has_tcf
        + 1.2 * np.maximum(bls_rel, tcf_rel)
        + 0.15 * np.log1p(bls_sde)
        + 0.08 * np.log1p(tcf_score)
        + 0.4 * tcf_events
        - 0.15 * raw_rank
    ).to_numpy()


def add_rank_from_scores(frame, score_column, rank_column="reranked_rank"):
    frame = frame.copy()
    frame[rank_column] = 999_999
    for _, index in frame.groupby(GROUP_COLUMNS, sort=False).groups.items():
        group = frame.loc[index]
        scores = pd.to_numeric(group[score_column], errors="coerce").fillna(-np.inf).to_numpy()
        raw_rank = pd.to_numeric(group["raw_candidate_rank"], errors="coerce").fillna(999_999).to_numpy()
        candidate_id = pd.to_numeric(group["candidate_id"], errors="coerce").fillna(999_999).to_numpy()
        order = np.lexsort((candidate_id, raw_rank, -scores))
        ranks = np.empty(len(group), dtype=int)
        ranks[order] = np.arange(1, len(group) + 1)
        frame.loc[group.index, rank_column] = ranks
    return frame


def prediction_base_columns(frame):
    preferred = [
        "fold",
        "target_id",
        "quarter",
        "injection_index",
        "candidate_id",
        "candidate_period_days",
        "source_detector",
        "exact_match",
        "harmonic_match",
        "raw_candidate_rank",
        "bls_rank",
        "tcf_rank",
        "injected_period_days",
        "injected_depth",
        "injected_duration_hours",
        "noise_quartile",
    ]
    return [column for column in preferred if column in frame.columns]


def make_scored_predictions(frame, model_name, scores):
    predictions = frame[prediction_base_columns(frame)].copy()
    predictions["model"] = model_name
    predictions["predicted_score"] = scores
    predictions = add_rank_from_scores(predictions, "predicted_score")
    return predictions


def make_detector_predictions(frame, detector):
    predictions = frame[prediction_base_columns(frame)].copy()
    rank = pd.to_numeric(predictions[f"{detector}_rank"], errors="coerce")
    predictions["model"] = f"{detector}_rank"
    predictions["predicted_score"] = detector_rank_scores(frame, detector)
    predictions["reranked_rank"] = rank.fillna(999_999).astype(int)
    return predictions


def fit_logistic_oof(frame, feature_columns, args):
    predictions = np.full(len(frame), np.nan)
    for fold in sorted(frame["fold"].unique()):
        train = frame["fold"] != fold
        test = frame["fold"] == fold
        model = Pipeline(
            [
                ("preprocessor", make_preprocessor(frame, feature_columns, scale_numeric=True)),
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
        model.fit(frame.loc[train, feature_columns], frame.loc[train, "exact_match"].astype(int))
        predictions[test] = model.predict_proba(frame.loc[test, feature_columns])[:, 1]
    return predictions


def fit_xgb_classifier_oof(frame, feature_columns, args):
    if xgb is None:
        warnings.warn("xgboost is not installed; skipping XGBoost classifier.", RuntimeWarning)
        return None
    predictions = np.full(len(frame), np.nan)
    for fold in sorted(frame["fold"].unique()):
        train = frame["fold"] != fold
        test = frame["fold"] == fold
        y_train = frame.loc[train, "exact_match"].astype(int)
        positives = int(y_train.sum())
        negatives = int(len(y_train) - positives)
        scale_pos_weight = negatives / positives if positives else 1.0
        model = Pipeline(
            [
                ("preprocessor", make_preprocessor(frame, feature_columns, scale_numeric=False)),
                (
                    "classifier",
                    xgb.XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_estimators=args.xgb_estimators,
                        max_depth=args.xgb_max_depth,
                        learning_rate=args.xgb_learning_rate,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        scale_pos_weight=scale_pos_weight,
                        random_state=args.random_state + int(fold),
                        n_jobs=args.n_jobs,
                        tree_method="hist",
                        verbosity=0,
                    ),
                ),
            ]
        )
        model.fit(frame.loc[train, feature_columns], y_train)
        predictions[test] = model.predict_proba(frame.loc[test, feature_columns])[:, 1]
    return predictions


def fit_xgb_ranker_oof(frame, feature_columns, args):
    if xgb is None:
        warnings.warn("xgboost is not installed; skipping XGBoost ranker.", RuntimeWarning)
        return None
    predictions = np.full(len(frame), np.nan)
    for fold in sorted(frame["fold"].unique()):
        train = frame["fold"] != fold
        test = frame["fold"] == fold
        train_frame = frame.loc[train].copy()
        positive_group_mask = train_frame.groupby("cv_group_id")["exact_match"].transform("any")
        rank_train_frame = train_frame.loc[positive_group_mask].sort_values("cv_group_id").copy()
        if rank_train_frame.empty:
            warnings.warn(f"Fold {fold} has no positive ranking groups; skipping ranker fold.", RuntimeWarning)
            continue
        preprocessor = make_preprocessor(frame, feature_columns, scale_numeric=False)
        preprocessor.fit(train_frame[feature_columns])
        x_train = preprocessor.transform(rank_train_frame[feature_columns])
        y_train = rank_train_frame["exact_match"].astype(int).to_numpy()
        group_sizes = rank_train_frame.groupby("cv_group_id", sort=False).size().astype(int).to_list()
        ranker = xgb.XGBRanker(
            objective="rank:pairwise",
            eval_metric="ndcg@10",
            n_estimators=args.xgb_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=args.random_state + int(fold),
            n_jobs=args.n_jobs,
            tree_method="hist",
            verbosity=0,
        )
        ranker.fit(x_train, y_train, group=group_sizes, verbose=False)
        predictions[test] = ranker.predict(preprocessor.transform(frame.loc[test, feature_columns]))
    if np.isnan(predictions).any():
        warnings.warn("Some XGBoost ranker rows did not receive predictions.", RuntimeWarning)
    return predictions


def group_index(frame):
    return frame.groupby(GROUP_COLUMNS, sort=False).size().index


def best_label_ranks(frame, label_column):
    groups = group_index(frame)
    label_rows = frame[to_bool_series(frame[label_column])]
    best = label_rows.groupby(GROUP_COLUMNS, sort=False)["reranked_rank"].min()
    return best.reindex(groups)


def bls_rank1_success(frame):
    groups = group_index(frame)
    rank = pd.to_numeric(frame.get("bls_rank", pd.Series(np.nan, index=frame.index)), errors="coerce")
    success_rows = frame[(rank == 1) & to_bool_series(frame["exact_match"])]
    success = success_rows.groupby(GROUP_COLUMNS, sort=False).size()
    return success.reindex(groups, fill_value=0).astype(bool)


def compute_metrics(frame, model_name):
    groups = group_index(frame)
    n_groups = len(groups)
    exact_rank = best_label_ranks(frame, "exact_match")
    harmonic_rank = best_label_ranks(frame, "harmonic_match")
    exact_rank1 = exact_rank <= 1
    bls_success = bls_rank1_success(frame)
    finite_exact_rank = exact_rank.dropna()
    return {
        "model": model_name,
        "injection_groups": int(n_groups),
        "candidate_rows": int(len(frame)),
        "exact_recall_at_1": float((exact_rank <= 1).fillna(False).mean()),
        "exact_recall_at_3": float((exact_rank <= 3).fillna(False).mean()),
        "exact_recall_at_5": float((exact_rank <= 5).fillna(False).mean()),
        "exact_recall_at_10": float((exact_rank <= 10).fillna(False).mean()),
        "harmonic_recall_at_1": float((harmonic_rank <= 1).fillna(False).mean()),
        "mean_reciprocal_rank": float((1.0 / exact_rank.dropna()).sum() / n_groups),
        "median_exact_rank": float(finite_exact_rank.median()) if len(finite_exact_rank) else np.nan,
        "candidate_set_misses": int(exact_rank.isna().sum()),
        "cases_improved_over_bls": int((exact_rank1.fillna(False) & ~bls_success).sum()),
        "cases_worsened_relative_to_bls": int((~exact_rank1.fillna(False) & bls_success).sum()),
    }


def oracle_benchmarks(frame):
    groups = group_index(frame)
    rank_bls = pd.to_numeric(frame.get("bls_rank", pd.Series(np.nan, index=frame.index)), errors="coerce")
    rank_tcf = pd.to_numeric(frame.get("tcf_rank", pd.Series(np.nan, index=frame.index)), errors="coerce")
    exact = to_bool_series(frame["exact_match"])
    harmonic = to_bool_series(frame["harmonic_match"])
    bls_exact = frame[(rank_bls == 1) & exact].groupby(GROUP_COLUMNS, sort=False).size().reindex(groups, fill_value=0).astype(bool)
    tcf_exact = frame[(rank_tcf == 1) & exact].groupby(GROUP_COLUMNS, sort=False).size().reindex(groups, fill_value=0).astype(bool)
    bls_harmonic = frame[(rank_bls == 1) & harmonic].groupby(GROUP_COLUMNS, sort=False).size().reindex(groups, fill_value=0).astype(bool)
    tcf_harmonic = frame[(rank_tcf == 1) & harmonic].groupby(GROUP_COLUMNS, sort=False).size().reindex(groups, fill_value=0).astype(bool)
    exact_any = frame[exact].groupby(GROUP_COLUMNS, sort=False).size().reindex(groups, fill_value=0).astype(bool)
    harmonic_any = frame[harmonic].groupby(GROUP_COLUMNS, sort=False).size().reindex(groups, fill_value=0).astype(bool)
    return {
        "injection_groups": int(len(groups)),
        "bls_exact_rank1": float(bls_exact.mean()),
        "tcf_exact_rank1": float(tcf_exact.mean()),
        "oracle_rank1_union_exact": float((bls_exact | tcf_exact).mean()),
        "oracle_rank1_union_harmonic": float((bls_harmonic | tcf_harmonic).mean()),
        "candidate_set_ceiling_exact": float(exact_any.mean()),
        "candidate_set_ceiling_harmonic": float(harmonic_any.mean()),
        "candidate_set_exact_misses": int((~exact_any).sum()),
        "candidate_set_harmonic_misses": int((~harmonic_any).sum()),
    }


def add_regime_bins(frame):
    frame = frame.copy()
    metadata_columns = [
        column
        for column in ["injected_depth", "injected_duration_hours", "injected_period_days", "noise_quartile"]
        if column in frame.columns
    ]
    group_meta = frame[GROUP_COLUMNS + metadata_columns].drop_duplicates(GROUP_COLUMNS).copy()
    quantile_specs = {
        "depth_bin": "injected_depth",
        "duration_bin": "injected_duration_hours",
        "period_bin": "injected_period_days",
    }
    for output_column, source_column in quantile_specs.items():
        if source_column not in group_meta.columns:
            continue
        values = pd.to_numeric(group_meta[source_column], errors="coerce")
        if values.nunique(dropna=True) <= 1:
            group_meta[output_column] = "all"
            continue
        group_meta[output_column] = pd.qcut(values, q=min(4, values.nunique(dropna=True)), duplicates="drop").astype(str)
    merge_columns = GROUP_COLUMNS + [column for column in group_meta.columns if column.endswith("_bin")]
    return frame.merge(group_meta[merge_columns], on=GROUP_COLUMNS, how="left", validate="many_to_one")


def regime_metrics(predictions):
    predictions = add_regime_bins(predictions)
    rows = []
    strata = [
        ("depth", "depth_bin"),
        ("duration", "duration_bin"),
        ("period", "period_bin"),
        ("noise_quartile", "noise_quartile"),
    ]
    for model_name, model_frame in predictions.groupby("model", sort=False):
        for stratum_name, stratum_column in strata:
            if stratum_column not in model_frame.columns:
                continue
            for bucket, bucket_frame in model_frame.groupby(stratum_column, dropna=False, sort=True):
                metrics = compute_metrics(bucket_frame, model_name)
                metrics["stratification"] = stratum_name
                metrics["bucket"] = str(bucket)
                rows.append(metrics)
    return pd.DataFrame(rows)


def train_and_evaluate(args):
    reranker_spec, reranker_config_path, reranker_config_sha256 = load_reranker_config(args.reranker_config)
    frame, candidate_path = load_candidates(args)
    frame["fold"] = assign_target_folds(frame, args.n_splits, args.random_state)
    feature_columns = leakage_safe_feature_columns(frame, reranker_spec)

    prediction_frames = [
        make_detector_predictions(frame, "bls"),
        make_detector_predictions(frame, "tcf"),
        make_scored_predictions(frame, "raw_min_rank", -frame["raw_candidate_rank"].to_numpy()),
        make_scored_predictions(frame, "rule_based", deterministic_rule_scores(frame)),
    ]

    logistic_scores = fit_logistic_oof(frame, feature_columns, args)
    prediction_frames.append(make_scored_predictions(frame, "logistic_regression", logistic_scores))

    xgb_classifier_scores = fit_xgb_classifier_oof(frame, feature_columns, args)
    if xgb_classifier_scores is not None:
        prediction_frames.append(make_scored_predictions(frame, "xgboost_classifier", xgb_classifier_scores))

    xgb_ranker_scores = fit_xgb_ranker_oof(frame, feature_columns, args)
    if xgb_ranker_scores is not None:
        prediction_frames.append(make_scored_predictions(frame, "xgboost_pairwise_ranker", xgb_ranker_scores))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame([compute_metrics(part, name) for name, part in predictions.groupby("model", sort=False)])
    benchmarks = oracle_benchmarks(frame)
    fold_summary = (
        frame.groupby("fold")
        .agg(candidate_rows=("candidate_id", "size"), stars=("target_id", "nunique"), injection_groups=("cv_group_id", "nunique"))
        .reset_index()
    )
    regime = regime_metrics(predictions)

    metrics_dir = Path(args.metrics_dir)
    predictions_path = metrics_dir / "candidate_reranker_oof_predictions.csv"
    metrics_path = metrics_dir / "candidate_reranker_metrics.csv"
    regime_path = metrics_dir / "candidate_reranker_regime_metrics.csv"
    summary_path = metrics_dir / "candidate_reranker_summary.json"
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    regime.to_csv(regime_path, index=False)
    summary = {
        "reranker_version": reranker_spec.get("version") if reranker_spec else "dynamic",
        "reranker_config": str(reranker_config_path) if reranker_config_path else None,
        "reranker_config_sha256": reranker_config_sha256,
        "candidate_dataset": str(candidate_path),
        "feature_columns": feature_columns,
        "forbidden_feature_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "n_feature_columns": int(len(feature_columns)),
        "fold_summary": fold_summary.to_dict(orient="records"),
        "benchmarks": benchmarks,
        "outputs": {
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "regime_metrics": str(regime_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return metrics, benchmarks, predictions_path, metrics_path, regime_path, summary_path


def main(args=None):
    args = args or build_parser().parse_args()
    metrics, benchmarks, predictions_path, metrics_path, regime_path, summary_path = train_and_evaluate(args)
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Regime metrics: {regime_path}")
    print(f"Summary: {summary_path}")
    print(
        metrics[
            [
                "model",
                "exact_recall_at_1",
                "exact_recall_at_3",
                "exact_recall_at_5",
                "exact_recall_at_10",
                "harmonic_recall_at_1",
                "mean_reciprocal_rank",
                "cases_improved_over_bls",
                "cases_worsened_relative_to_bls",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print(
        "Benchmarks: "
        f"BLS rank-1={benchmarks['bls_exact_rank1']:.3f}, "
        f"TCF rank-1={benchmarks['tcf_exact_rank1']:.3f}, "
        f"oracle union rank-1={benchmarks['oracle_rank1_union_exact']:.3f}, "
        f"candidate ceiling={benchmarks['candidate_set_ceiling_exact']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
