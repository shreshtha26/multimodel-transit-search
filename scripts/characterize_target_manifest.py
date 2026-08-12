"""Compute cheap Q5 background features for a target-selection candidate manifest."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from tqdm.auto import tqdm

from analyze_multistar_background_timescales import star_background_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs/target_selection/kepler_catalog_clean_candidates_q5.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/target_selection/kepler_catalog_clean_candidate_features.csv"
DEFAULT_CACHE = PROJECT_ROOT / "outputs/cache/kepler_light_curves"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Characterize a clean target-selection pool before choosing the final benchmark cohort.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-acf-tau-days", type=float, default=30.0)
    parser.add_argument("--rolling-background-window-days", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.manifest_path.exists():
        raise FileNotFoundError(f"Candidate manifest not found: {args.manifest_path}. Run scripts/build_clean_kepler_manifest.py first.")
    manifest = pd.read_csv(args.manifest_path, dtype={"target_id": str})
    if args.limit is not None:
        manifest = manifest.head(int(args.limit)).copy()
    feature_args = SimpleNamespace(
        cache_dir=args.cache_dir,
        allow_download=bool(args.allow_download),
        quality_policy="default",
        require_finite_flux_error=False,
        test_fraction=0.20,
        max_acf_tau_days=float(args.max_acf_tau_days),
        rolling_background_window_days=float(args.rolling_background_window_days),
    )
    rows = []
    metadata_columns = [column for column in ("selection_group", "sample_stratum", "koi_flag", "tce_flag", "confirmed_planet_flag", "eb_flag", "provenance") if column in manifest.columns]
    for row in tqdm(manifest.to_dict(orient="records"), desc="Characterize clean candidates", unit="star"):
        metadata = {column: row.get(column) for column in metadata_columns}
        try:
            features = star_background_features(row, feature_args)
            rows.append({**metadata, **features, "status": "success", "error": ""})
        except Exception as exc:
            rows.append({"target_id": str(row["target_id"]), "quarter": int(row["quarter"]), **metadata, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    output = pd.DataFrame(rows)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)
    successes = int((output["status"] == "success").sum()) if len(output) else 0
    print(f"Characterized {successes}/{len(output)} targets -> {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
