"""Freeze the clean multi-star candidate reranker and its feature contract."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

import train_candidate_rerankers as rerank


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_DIR = PROJECT_ROOT / "outputs/experiments/multistar_bls_tcf/optimized/metrics"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs/experiments/multistar_bls_tcf/optimized/models"
DEFAULT_RERANKER_CONFIG_PATH = PROJECT_ROOT / "configs/candidate_reranker_clean_v1.json"


def default_settings():
    return SimpleNamespace(
        metrics_dir=DEFAULT_METRICS_DIR,
        model_dir=DEFAULT_MODEL_DIR,
        candidate_dataset=None,
        reranker_config=DEFAULT_RERANKER_CONFIG_PATH,
        random_state=42,
        xgb_estimators=300,
        xgb_max_depth=3,
        xgb_learning_rate=0.05,
        n_jobs=4,
    )


def build_parser():
    defaults = default_settings()
    parser = argparse.ArgumentParser(description="Train and save the frozen clean candidate reranker.")
    parser.add_argument("--metrics-dir", type=Path, default=defaults.metrics_dir)
    parser.add_argument("--model-dir", type=Path, default=defaults.model_dir)
    parser.add_argument("--candidate-dataset", type=Path, default=defaults.candidate_dataset)
    parser.add_argument("--reranker-config", type=Path, default=defaults.reranker_config)
    parser.add_argument("--random-state", type=int, default=defaults.random_state)
    parser.add_argument("--xgb-estimators", type=int, default=defaults.xgb_estimators)
    parser.add_argument("--xgb-max-depth", type=int, default=defaults.xgb_max_depth)
    parser.add_argument("--xgb-learning-rate", type=float, default=defaults.xgb_learning_rate)
    parser.add_argument("--n-jobs", type=int, default=defaults.n_jobs)
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


def build_final_xgb_classifier(frame, feature_columns, args):
    if rerank.xgb is None:
        raise RuntimeError("xgboost is not installed.")
    y_train = frame["exact_match"].astype(int)
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / positives if positives else 1.0
    model = Pipeline(
        [
            ("preprocessor", rerank.make_preprocessor(frame, feature_columns, scale_numeric=False)),
            (
                "classifier",
                rerank.xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=args.xgb_estimators,
                    max_depth=args.xgb_max_depth,
                    learning_rate=args.xgb_learning_rate,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=scale_pos_weight,
                    random_state=args.random_state,
                    n_jobs=args.n_jobs,
                    tree_method="hist",
                    verbosity=0,
                ),
            ),
        ]
    )
    model.fit(frame[feature_columns], y_train)
    return model, {"positive_rows": positives, "negative_rows": negatives, "scale_pos_weight": scale_pos_weight}


def freeze_reranker(args):
    spec, config_path, config_sha256 = rerank.load_reranker_config(args.reranker_config)
    frame, candidate_path = rerank.load_candidates(args)
    feature_columns = rerank.leakage_safe_feature_columns(frame, spec)
    model, class_balance = build_final_xgb_classifier(frame, feature_columns, args)
    version = spec["version"]

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{version}_xgboost_classifier.joblib"
    metadata_path = model_dir / f"{version}_metadata.json"
    feature_list_path = model_dir / f"{version}_feature_columns.txt"
    joblib.dump(model, model_path)
    feature_list_path.write_text("\n".join(feature_columns) + "\n")

    metadata = {
        "version": version,
        "primary_model": spec.get("primary_model", "xgboost_classifier"),
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "reranker_config": str(config_path),
        "reranker_config_sha256": config_sha256,
        "candidate_dataset": str(candidate_path),
        "candidate_dataset_sha256": rerank.file_sha256(candidate_path),
        "model_artifact": str(model_path),
        "model_artifact_sha256": rerank.file_sha256(model_path),
        "feature_list": str(feature_list_path),
        "feature_list_sha256": rerank.file_sha256(feature_list_path),
        "feature_columns": feature_columns,
        "forbidden_feature_columns": spec.get("forbidden_feature_columns", []),
        "forbidden_feature_prefixes": spec.get("forbidden_feature_prefixes", []),
        "training_rows": int(len(frame)),
        "injection_groups": int(frame["cv_group_id"].nunique()),
        "star_count": int(frame["target_id"].nunique()),
        "label_column": spec.get("label_column", "exact_match"),
        "group_columns": spec.get("group_columns", rerank.GROUP_COLUMNS),
        "split_unit": spec.get("split_unit", "target_id"),
        "model_params": {
            "n_estimators": int(args.xgb_estimators),
            "max_depth": int(args.xgb_max_depth),
            "learning_rate": float(args.xgb_learning_rate),
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
            "random_state": int(args.random_state),
        },
        "class_balance": class_balance,
        "validation_snapshot": spec.get("validation_snapshot", {}),
    }
    metadata_path.write_text(json.dumps(json_ready(metadata), indent=2) + "\n")
    return model_path, metadata_path, feature_list_path, metadata


def main(args=None):
    args = args or build_parser().parse_args()
    model_path, metadata_path, feature_list_path, metadata = freeze_reranker(args)
    print(f"Frozen model: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Feature list: {feature_list_path}")
    print(f"Version: {metadata['version']}")
    print(f"Features: {len(metadata['feature_columns'])}")
    print(f"Model SHA-256: {metadata['model_artifact_sha256']}")
    print(f"Config SHA-256: {metadata['reranker_config_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
