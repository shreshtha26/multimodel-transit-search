"""Build population-relative stellar variability profiles for model selection.

Usage examples
--------------
After a multi-star benchmark:

    python scripts/build_stellar_variability_profiles.py \
        --input outputs/experiments/multistar_challenger_benchmark/main/metrics/multistar_challenger_star_summary.csv

Or after collecting individual characterization CSVs:

    python scripts/build_stellar_variability_profiles.py \
        --input-dir outputs/experiments/characterization/metrics

The output contains two separate products:
1. `stellar_variability_profiles.csv`: interpretation / screening labels.
2. `stellar_model_selection_features.csv`: continuous features only, intended as
   X_star for XGBoost / RF / neural-network routing experiments.

Important scientific assumption
-------------------------------
Population labels are RELATIVE TO THE INPUT POPULATION.  Re-running this script
on a different stellar sample can move q25/q75 boundaries.  Save the emitted
JSON thresholds with every experiment and never describe them as universal
astrophysical class boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adaptive_transit.noise_models.stellar_variability import (
    apply_population_variability_boundaries,
    model_selection_feature_frame,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build population-relative stellar variability profiles.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="One consolidated characterization/star-summary CSV.")
    source.add_argument("--input-dir", type=Path, help="Directory containing *_light_curve_diagnostics.csv files.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/stellar_variability_profiles/metrics"))
    return parser


def load_records(input_path: Path | None, input_dir: Path | None) -> pd.DataFrame:
    if input_path is not None:
        frame = pd.read_csv(input_path, dtype={"target_id": str})
        if frame.empty:
            raise ValueError(f"No rows found in {input_path}")
        return frame

    paths = sorted(Path(input_dir).glob("*_light_curve_diagnostics.csv")) if input_dir is not None else []
    if not paths:
        raise FileNotFoundError(f"No *_light_curve_diagnostics.csv files found in {input_dir}")
    frames = [pd.read_csv(path, dtype={"target_id": str}) for path in paths]
    return pd.concat(frames, ignore_index=True)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    records = load_records(args.input, args.input_dir)
    profiles, thresholds = apply_population_variability_boundaries(records)
    features = model_selection_feature_frame(profiles)

    # Retain identifiers beside X_star so joins to injection-level Y outcomes are
    # deterministic.  Selection-group labels are retained only as provenance;
    # they are NOT model features.
    identifiers = [column for column in ("target_id", "quarter", "selection_group") if column in profiles.columns]
    feature_output = pd.concat([profiles[identifiers].reset_index(drop=True), features.reset_index(drop=True)], axis=1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "stellar_variability_profiles.csv"
    feature_path = output_dir / "stellar_model_selection_features.csv"
    threshold_path = output_dir / "stellar_variability_population_boundaries.json"

    profiles.to_csv(profile_path, index=False)
    feature_output.to_csv(feature_path, index=False)
    threshold_path.write_text(json.dumps(thresholds, indent=2) + "\n")

    print(f"Profiles: {profile_path}")
    print(f"Model-selection features: {feature_path}")
    print(f"Population boundaries: {threshold_path}")
    print(f"Stars: {len(profiles)}")
    if "v2_quiet_candidate" in profiles:
        print(f"Quiet candidates: {int(profiles['v2_quiet_candidate'].fillna(False).sum())}")
    if "v2_low_scatter_structured_candidate" in profiles:
        print(f"Low-scatter but structured: {int(profiles['v2_low_scatter_structured_candidate'].fillna(False).sum())}")
    if "v2_rotation_spot_review_flag" in profiles:
        print(f"Rotation/spot review flags: {int(profiles['v2_rotation_spot_review_flag'].fillna(False).sum())}")
    if "v2_pulsation_review_flag" in profiles:
        print(f"Pulsation review flags: {int(profiles['v2_pulsation_review_flag'].fillna(False).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
