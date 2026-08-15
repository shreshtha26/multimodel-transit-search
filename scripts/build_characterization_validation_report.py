"""Build the 40-development + 10-validation characterization report.

This script is deliberately read-only with respect to the characterization
definitions. It does NOT refit, retune, or modify any v2 thresholds.

Inputs
------
Development diagnostics:
    outputs/experiments/characterization_population40/metrics/
        *_light_curve_diagnostics.csv

Validation diagnostics:
    outputs/experiments/characterization_validation10/metrics/
        *_light_curve_diagnostics.csv

Validation manifest:
    outputs/target_selection/kepler_characterization_validation10.csv

Development stratum metadata are recovered automatically, when possible, from
CSV files under outputs/ by matching target_id + quarter. You may also provide
an explicit development manifest with --development-manifest.

Outputs
-------
    outputs/experiments/characterization_validation50/

Important products:
    metrics/characterization_master_50.csv
    metrics/development_numeric_summary.csv
    metrics/validation_numeric_summary.csv
    metrics/validation_vs_development_robust_scores.csv
    metrics/validation_review_queue.csv
    metrics/characterization_validation_summary.txt
    plots/*.png

Interpretation
--------------
The 40-star development sample was deliberately stratified. The extra 10 stars
are an independent holdout and MUST NOT be used to retune the characterization
boundaries after inspecting their results. The purpose of this report is to
find failures/edge cases and assess whether the frozen definitions generalize.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DEV_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "characterization_population40"
    / "metrics"
)

DEFAULT_VAL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "characterization_validation10"
    / "metrics"
)

DEFAULT_VAL_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "target_selection"
    / "kepler_characterization_validation10.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "characterization_validation50"
)

TARGET_ID_ALIASES = (
    "target_id",
    "kepid",
    "kic",
    "KIC",
    "kepler_id",
)

QUARTER_ALIASES = (
    "quarter",
    "Quarter",
    "q",
)

STRATUM_ALIASES = (
    "sample_stratum",
    "stratum",
    "population_stratum",
)

COHERENCE_ALIASES = (
    "v2_coherent_periodic_candidate",
    "coherent_periodic_candidate_v2",
    "coherent_periodic_candidate",
)

STATIONARITY_ALIASES = (
    "stationarity",
    "stationarity_label",
    "stationarity_classification",
)

PERIOD_V1_ALIASES = (
    "dominant_period_days",
    "dominant_period_v1_days",
    "v1_dominant_period_days",
    "dominant_period_v1",
)

PERIOD_V2_ALIASES = (
    "v2_dominant_period_days",
    "dominant_period_v2_days",
    "dominant_period_v2",
)

LS_FAP_ALIASES = (
    "v2_ls_screening_fap",
    "ls_screening_fap_v2",
    "ls_fap_v2",
    "ls_screening_fap",
)


def _clean_target_id(value) -> str:
    text = str(value).strip()
    if text.upper().startswith("KIC"):
        text = text[3:].strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit():
        raise ValueError(f"Invalid Kepler target id: {value!r}")
    return text


def _extract_target_quarter_from_name(path: Path) -> tuple[str | None, int | None]:
    match = re.search(r"kic_(\d+)_q(\d+)_light_curve_diagnostics\.csv$", path.name)
    if not match:
        return None, None
    return match.group(1), int(match.group(2))


def _first_existing(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    for column in aliases:
        if column in frame.columns:
            return column
    return None


def _coerce_bool(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return np.nan


def _load_one_diagnostic(path: Path, sample_role: str) -> dict:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Diagnostic CSV is empty: {path}")

    # Most current diagnostic CSVs are one-row outputs. If a future version
    # writes key/value rows instead, pivot those into a single record.
    if len(frame) == 1:
        record = frame.iloc[0].to_dict()
    elif set(frame.columns) >= {"metric", "value"}:
        record = dict(zip(frame["metric"], frame["value"]))
    elif set(frame.columns) >= {"name", "value"}:
        record = dict(zip(frame["name"], frame["value"]))
    else:
        # Preserve information without guessing row semantics.
        record = frame.iloc[0].to_dict()
        record["_diagnostic_source_row_count"] = len(frame)

    file_target, file_quarter = _extract_target_quarter_from_name(path)

    target_column = next(
        (column for column in TARGET_ID_ALIASES if column in record),
        None,
    )
    quarter_column = next(
        (column for column in QUARTER_ALIASES if column in record),
        None,
    )

    target_id = (
        _clean_target_id(record[target_column])
        if target_column is not None and pd.notna(record[target_column])
        else file_target
    )
    quarter = (
        int(float(record[quarter_column]))
        if quarter_column is not None and pd.notna(record[quarter_column])
        else file_quarter
    )

    if target_id is None or quarter is None:
        raise ValueError(
            f"Could not determine target_id/quarter from diagnostic: {path}"
        )

    record["target_id"] = target_id
    record["quarter"] = int(quarter)
    record["sample_role"] = sample_role
    record["diagnostic_csv"] = str(path.relative_to(PROJECT_ROOT))
    return record


def load_diagnostics(directory: Path, sample_role: str) -> pd.DataFrame:
    paths = sorted(directory.glob("*_light_curve_diagnostics.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No *_light_curve_diagnostics.csv files found under {directory}"
        )

    records = [_load_one_diagnostic(path, sample_role) for path in paths]
    frame = pd.DataFrame(records)

    if frame.duplicated(["target_id", "quarter"]).any():
        duplicates = frame.loc[
            frame.duplicated(["target_id", "quarter"], keep=False),
            ["target_id", "quarter"],
        ]
        raise ValueError(
            "Duplicate diagnostic target-quarter rows found:\n"
            + duplicates.to_string(index=False)
        )
    return frame


def _canonical_manifest(frame: pd.DataFrame) -> pd.DataFrame | None:
    target_column = _first_existing(frame, TARGET_ID_ALIASES)
    if target_column is None:
        return None

    result = frame.copy()
    result["target_id"] = result[target_column].map(_clean_target_id)

    quarter_column = _first_existing(result, QUARTER_ALIASES)
    if quarter_column is None:
        result["quarter"] = 5
    else:
        result["quarter"] = pd.to_numeric(
            result[quarter_column], errors="coerce"
        ).fillna(5).astype(int)
    return result


def recover_development_strata(
    dev: pd.DataFrame,
    explicit_manifest: Path | None,
) -> pd.DataFrame:
    """Recover sample_stratum without hard-coding a historical manifest path."""
    result = dev.copy()
    result["sample_stratum"] = np.nan

    candidates: list[Path] = []
    if explicit_manifest is not None:
        candidates.append(explicit_manifest)
    else:
        candidates.extend(
            path
            for path in (PROJECT_ROOT / "outputs").rglob("*.csv")
            if "characterization_validation" not in str(path)
        )

    wanted = set(zip(result["target_id"].astype(str), result["quarter"].astype(int)))
    best_mapping: dict[tuple[str, int], str] = {}
    best_path: Path | None = None

    for path in candidates:
        try:
            frame = pd.read_csv(path, nrows=5000)
        except Exception:
            continue

        canonical = _canonical_manifest(frame)
        if canonical is None:
            continue

        stratum_column = _first_existing(canonical, STRATUM_ALIASES)
        if stratum_column is None:
            continue

        mapping = {}
        for _, row in canonical.iterrows():
            if pd.isna(row[stratum_column]):
                continue
            key = (str(row["target_id"]), int(row["quarter"]))
            if key in wanted:
                mapping[key] = str(row[stratum_column])

        if len(mapping) > len(best_mapping):
            best_mapping = mapping
            best_path = path

        if len(best_mapping) == len(wanted):
            break

    if best_mapping:
        result["sample_stratum"] = [
            best_mapping.get((str(t), int(q)), np.nan)
            for t, q in zip(result["target_id"], result["quarter"])
        ]
        print(
            f"Recovered sample_stratum for "
            f"{result['sample_stratum'].notna().sum()}/{len(result)} "
            f"development stars"
            + (f" from {best_path}" if best_path else "")
        )
    else:
        print(
            "Warning: development sample_stratum could not be recovered. "
            "The report will still be generated."
        )
    return result


def normalize_known_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    mappings = {
        "stationarity_label": STATIONARITY_ALIASES,
        "coherent_periodic_candidate_v2": COHERENCE_ALIASES,
        "dominant_period_v1_days": PERIOD_V1_ALIASES,
        "dominant_period_v2_days": PERIOD_V2_ALIASES,
        "ls_screening_fap_v2": LS_FAP_ALIASES,
    }

    for canonical_name, aliases in mappings.items():
        source = _first_existing(result, aliases)
        if source is not None:
            result[canonical_name] = result[source]

    if "coherent_periodic_candidate_v2" in result.columns:
        result["coherent_periodic_candidate_v2"] = result[
            "coherent_periodic_candidate_v2"
        ].map(_coerce_bool)

    for column in (
        "dominant_period_v1_days",
        "dominant_period_v2_days",
        "ls_screening_fap_v2",
    ):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    return result


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded_patterns = (
        "target_id",
        "quarter",
        "fap",
        "p_value",
        "pvalue",
        "_p_",
        "return_code",
        "seed",
        "count",
    )

    columns = []
    for column in frame.columns:
        if column in {
            "dominant_period_v1_days",
            "dominant_period_v2_days",
        }:
            columns.append(column)
            continue

        if any(pattern in column.lower() for pattern in excluded_patterns):
            continue

        # Boolean diagnostic/classification flags are categorical, even though
        # pandas can coerce them to numeric-looking values.  Do not feed them
        # into quantiles, robust-z calculations, or continuous distributions.
        if pd.api.types.is_bool_dtype(frame[column].dtype):
            continue

        converted = pd.to_numeric(frame[column], errors="coerce")
        if pd.api.types.is_bool_dtype(converted.dtype):
            continue

        if converted.notna().sum() >= max(5, int(0.5 * len(frame))):
            columns.append(column)

    # Deduplicate while preserving order.
    return list(dict.fromkeys(columns))


def robust_validation_scores(
    dev: pd.DataFrame,
    val: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    rows = []

    for column in columns:
        dev_values = (
            pd.to_numeric(dev[column], errors="coerce")
            .dropna()
            .astype(float)
        )
        if len(dev_values) < 5:
            continue

        median = float(dev_values.median())
        mad = float(np.median(np.abs(dev_values - median)))

        # Normal-consistent MAD scaling. If MAD collapses, use IQR as a stable
        # fallback instead of manufacturing infinite z-scores.
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 0:
            q1, q3 = np.nanpercentile(dev_values, [25, 75])
            iqr = float(q3 - q1)
            scale = iqr / 1.349 if iqr > 0 else np.nan

        for _, row in val.iterrows():
            value = pd.to_numeric(
                pd.Series([row.get(column)]), errors="coerce"
            ).iloc[0]
            value = float(value) if pd.notna(value) else np.nan
            robust_z = (
                (float(value) - median) / scale
                if pd.notna(value) and np.isfinite(scale) and scale > 0
                else np.nan
            )
            rows.append(
                {
                    "target_id": row["target_id"],
                    "quarter": row["quarter"],
                    "metric": column,
                    "value": value,
                    "development_median": median,
                    "development_robust_scale": scale,
                    "robust_z": robust_z,
                    "abs_robust_z": abs(robust_z) if pd.notna(robust_z) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def build_review_queue(
    master: pd.DataFrame,
    robust_scores: pd.DataFrame,
    z_threshold: float,
) -> pd.DataFrame:
    val = master.loc[master["sample_role"] == "validation"].copy()
    review: dict[tuple[str, int], list[str]] = {
        (str(row["target_id"]), int(row["quarter"])): []
        for _, row in val.iterrows()
    }

    if not robust_scores.empty:
        extreme = robust_scores.loc[
            robust_scores["abs_robust_z"] >= float(z_threshold)
        ].sort_values("abs_robust_z", ascending=False)
        for _, row in extreme.iterrows():
            key = (str(row["target_id"]), int(row["quarter"]))
            review[key].append(
                f"development-range outlier: {row['metric']} "
                f"(robust z={row['robust_z']:.2f})"
            )

    if {
        "dominant_period_v1_days",
        "dominant_period_v2_days",
    }.issubset(val.columns):
        p1 = pd.to_numeric(val["dominant_period_v1_days"], errors="coerce")
        p2 = pd.to_numeric(val["dominant_period_v2_days"], errors="coerce")
        relative_change = (p2 - p1).abs() / p1.abs().replace(0, np.nan)
        val["_period_relative_change"] = relative_change

        for _, row in val.loc[relative_change >= 0.20].iterrows():
            key = (str(row["target_id"]), int(row["quarter"]))
            review[key].append(
                "v1-v2 dominant-period disagreement "
                f"({100 * row['_period_relative_change']:.1f}%)"
            )

    if "stationarity_label" in val.columns:
        for _, row in val.iterrows():
            label = str(row.get("stationarity_label", "")).lower()
            if (
                "conflict" in label
                or "inconclusive" in label
                or "nonstationary" in label
            ):
                key = (str(row["target_id"]), int(row["quarter"]))
                review[key].append(f"stationarity={row['stationarity_label']}")

    if {
        "ls_screening_fap_v2",
        "coherent_periodic_candidate_v2",
    }.issubset(val.columns):
        for _, row in val.iterrows():
            fap = row.get("ls_screening_fap_v2")
            coherent = row.get("coherent_periodic_candidate_v2")
            if pd.notna(fap) and float(fap) < 1e-6 and coherent is False:
                key = (str(row["target_id"]), int(row["quarter"]))
                review[key].append(
                    "very significant LS peak but coherence criterion is false"
                )

    rows = []
    for (target_id, quarter), reasons in review.items():
        rows.append(
            {
                "target_id": target_id,
                "quarter": quarter,
                "needs_visual_review": bool(reasons),
                "n_review_reasons": len(reasons),
                "review_reasons": " | ".join(reasons),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["needs_visual_review", "n_review_reasons", "target_id"],
        ascending=[False, False, True],
    )


def _summary_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        values = (
            pd.to_numeric(frame[column], errors="coerce")
            .dropna()
            .astype(float)
        )
        if values.empty:
            continue
        rows.append(
            {
                "metric": column,
                "n": len(values),
                "median": values.median(),
                "q25": values.quantile(0.25),
                "q75": values.quantile(0.75),
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def _plot_distribution(
    dev: pd.DataFrame,
    val: pd.DataFrame,
    column: str,
    output_path: Path,
) -> None:
    dev_values = (
        pd.to_numeric(dev[column], errors="coerce")
        .dropna()
        .astype(float)
    )
    val_values = (
        pd.to_numeric(val[column], errors="coerce")
        .dropna()
        .astype(float)
    )
    if len(dev_values) < 3 or len(val_values) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(dev_values, bins="auto", alpha=0.45, label="Development (40)")
    ax.scatter(
        val_values,
        np.zeros(len(val_values)),
        marker="|",
        s=180,
        label="Validation (10)",
    )
    ax.set_xlabel(column.replace("_", " "))
    ax.set_ylabel("Development count")
    ax.set_title(f"Development distribution with validation overlay: {column}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _choose_scatter_metrics(columns: list[str]) -> list[str]:
    priorities = (
        "robust_scatter",
        "scatter",
        "acf1",
        "acf_1",
        "memory",
        "variance_drift",
        "spectral_concentration",
        "coherence",
        "skew",
        "kurt",
    )
    chosen = []
    for token in priorities:
        matches = [
            column
            for column in columns
            if token in column.lower() and column not in chosen
        ]
        chosen.extend(matches[:1])
    return chosen[:6]


def _plot_pair(
    dev: pd.DataFrame,
    val: pd.DataFrame,
    x: str,
    y: str,
    output_path: Path,
) -> None:
    dx = pd.to_numeric(dev[x], errors="coerce")
    dy = pd.to_numeric(dev[y], errors="coerce")
    vx = pd.to_numeric(val[x], errors="coerce")
    vy = pd.to_numeric(val[y], errors="coerce")

    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    valid_dev = dx.notna() & dy.notna()
    if "sample_stratum" in dev.columns and dev["sample_stratum"].notna().any():
        for label, group in dev.loc[valid_dev].groupby("sample_stratum"):
            ax.scatter(
                pd.to_numeric(group[x], errors="coerce"),
                pd.to_numeric(group[y], errors="coerce"),
                alpha=0.7,
                label=f"Development: {label}",
            )
    else:
        ax.scatter(dx[valid_dev], dy[valid_dev], alpha=0.7, label="Development")

    valid_val = vx.notna() & vy.notna()
    ax.scatter(
        vx[valid_val],
        vy[valid_val],
        marker="x",
        s=70,
        linewidths=1.8,
        label="Validation",
    )

    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    ax.set_title(f"{y} vs {x}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_text_summary(
    master: pd.DataFrame,
    review_queue: pd.DataFrame,
    robust_scores: pd.DataFrame,
    output_path: Path,
) -> None:
    dev = master.loc[master["sample_role"] == "development"]
    val = master.loc[master["sample_role"] == "validation"]

    lines = [
        "CHARACTERIZATION VALIDATION SUMMARY",
        "===================================",
        "",
        f"Development stars: {len(dev)}",
        f"Independent validation stars: {len(val)}",
        f"Total stars: {len(master)}",
        "",
        "IMPORTANT:",
        "The validation set is a holdout. Do not retune characterization",
        "thresholds solely to make these ten stars fit the development sample.",
        "",
    ]

    if "stationarity_label" in master.columns:
        lines.append("Stationarity labels:")
        lines.append(
            master.groupby(["sample_role", "stationarity_label"])
            .size()
            .to_string()
        )
        lines.append("")

    if "coherent_periodic_candidate_v2" in master.columns:
        lines.append("Coherent periodic candidate counts:")
        lines.append(
            master.groupby(
                ["sample_role", "coherent_periodic_candidate_v2"],
                dropna=False,
            ).size().to_string()
        )
        lines.append("")

    needs_review = review_queue.loc[review_queue["needs_visual_review"]]
    lines.append(
        f"Validation stars flagged for visual review: "
        f"{len(needs_review)}/{len(review_queue)}"
    )
    if not needs_review.empty:
        lines.append(
            needs_review[
                ["target_id", "quarter", "review_reasons"]
            ].to_string(index=False)
        )
    lines.append("")

    if not robust_scores.empty:
        extreme = robust_scores.loc[
            robust_scores["abs_robust_z"] >= 3.5
        ].sort_values("abs_robust_z", ascending=False)
        lines.append(
            f"Validation metric values beyond |robust z| >= 3.5: {len(extreme)}"
        )
        if not extreme.empty:
            lines.append(
                extreme[
                    ["target_id", "metric", "value", "robust_z"]
                ].head(30).to_string(index=False)
            )

    output_path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine the frozen 40-star development characterization with the "
            "10-star independent validation characterization and produce a "
            "read-only validation report."
        )
    )
    parser.add_argument("--development-dir", type=Path, default=DEFAULT_DEV_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=DEFAULT_VAL_MANIFEST,
    )
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=None,
        help=(
            "Optional explicit 40-star manifest containing target_id, quarter, "
            "and sample_stratum. If omitted, the script searches outputs/*.csv."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--outlier-z", type=float, default=3.5)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    dev = load_diagnostics(args.development_dir, "development")
    val = load_diagnostics(args.validation_dir, "validation")

    if len(dev) != 40:
        print(f"Warning: expected 40 development diagnostics; found {len(dev)}.")
    if len(val) != 10:
        raise RuntimeError(
            f"Expected 10 independent validation diagnostics; found {len(val)}."
        )

    dev = recover_development_strata(dev, args.development_manifest)

    # Validation stars remain unstratified by design.
    val["sample_stratum"] = np.nan
    if args.validation_manifest.exists():
        validation_manifest = _canonical_manifest(
            pd.read_csv(args.validation_manifest)
        )
        if validation_manifest is not None and "sample_role" in validation_manifest:
            # No scientific feature columns are imported here; manifest use is
            # only an identity/selection cross-check.
            expected = set(
                zip(
                    validation_manifest["target_id"].astype(str),
                    validation_manifest["quarter"].astype(int),
                )
            )
            observed = set(zip(val["target_id"].astype(str), val["quarter"].astype(int)))
            if expected != observed:
                raise RuntimeError(
                    "Validation diagnostics do not match the frozen validation manifest."
                )

    master = pd.concat([dev, val], ignore_index=True, sort=False)
    master = normalize_known_fields(master)

    if master.duplicated(["target_id", "quarter"]).any():
        duplicates = master.loc[
            master.duplicated(["target_id", "quarter"], keep=False),
            ["target_id", "quarter", "sample_role"],
        ]
        raise RuntimeError(
            "Development/validation overlap detected:\n"
            + duplicates.to_string(index=False)
        )

    output_dir = args.output_dir
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    master = master.sort_values(
        ["sample_role", "target_id"],
        kind="stable",
    ).reset_index(drop=True)
    master.to_csv(metrics_dir / "characterization_master_50.csv", index=False)

    numeric_columns = numeric_feature_columns(master)
    dev_norm = master.loc[master["sample_role"] == "development"].copy()
    val_norm = master.loc[master["sample_role"] == "validation"].copy()

    dev_summary = _summary_numeric(dev_norm, numeric_columns)
    val_summary = _summary_numeric(val_norm, numeric_columns)
    dev_summary.to_csv(
        metrics_dir / "development_numeric_summary.csv",
        index=False,
    )
    val_summary.to_csv(
        metrics_dir / "validation_numeric_summary.csv",
        index=False,
    )

    robust_scores = robust_validation_scores(
        dev_norm,
        val_norm,
        numeric_columns,
    )
    robust_scores.to_csv(
        metrics_dir / "validation_vs_development_robust_scores.csv",
        index=False,
    )

    review_queue = build_review_queue(
        master,
        robust_scores,
        z_threshold=args.outlier_z,
    )
    review_queue.to_csv(
        metrics_dir / "validation_review_queue.csv",
        index=False,
    )

    write_text_summary(
        master,
        review_queue,
        robust_scores,
        metrics_dir / "characterization_validation_summary.txt",
    )

    # Generate a small, review-oriented plot set rather than dozens of plots.
    preferred_metrics = _choose_scatter_metrics(numeric_columns)
    for index, column in enumerate(preferred_metrics[:6], start=1):
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", column)
        _plot_distribution(
            dev_norm,
            val_norm,
            column,
            plots_dir / f"{index:02d}_distribution_{safe}.png",
        )

    if len(preferred_metrics) >= 2:
        pair_index = 1
        for i in range(min(4, len(preferred_metrics))):
            for j in range(i + 1, min(4, len(preferred_metrics))):
                x = preferred_metrics[i]
                y = preferred_metrics[j]
                _plot_pair(
                    dev_norm,
                    val_norm,
                    x,
                    y,
                    plots_dir / f"pair_{pair_index:02d}_{x}_vs_{y}.png",
                )
                pair_index += 1

    print()
    print("Characterization validation report complete.")
    print(f"Development diagnostics loaded: {len(dev_norm)}")
    print(f"Independent validation diagnostics loaded: {len(val_norm)}")
    print(f"Total master rows: {len(master)}")
    print(f"Numeric metrics evaluated: {len(numeric_columns)}")
    print(
        "Validation stars requiring review: "
        f"{int(review_queue['needs_visual_review'].sum())}/{len(review_queue)}"
    )
    print()
    print(
        "Master table:",
        metrics_dir / "characterization_master_50.csv",
    )
    print(
        "Review queue:",
        metrics_dir / "validation_review_queue.csv",
    )
    print(
        "Summary:",
        metrics_dir / "characterization_validation_summary.txt",
    )
    print("Plots:", plots_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
