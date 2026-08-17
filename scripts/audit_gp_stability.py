#!/usr/bin/env python3
"""Audit GP optimizer stability on base stars without rerunning injections/detectors.

The production GP model is intentionally left unchanged by this patch.  This
script asks a narrower question first: do modest optimizer restarts remove the
ConvergenceWarnings while leaving the fitted background/residual behavior
consistent?  A fixed-kernel fit is included only as a deterministic fallback
benchmark; it is not silently substituted for an optimized GP.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs/kepler_clean_background_manifest_10star.csv"
DEFAULT_CACHE = PROJECT_ROOT / "outputs/cache/kepler_light_curves"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/experiments/batman_physical_detection_poc/pilot10/qc_analysis/gp_stability"

POLICIES = (
    {"policy": "optimized_r0", "optimize_kernel": True, "n_restarts_optimizer": 0},
    {"policy": "optimized_r2", "optimize_kernel": True, "n_restarts_optimizer": 2},
    {"policy": "fixed_kernel_3d", "optimize_kernel": False, "n_restarts_optimizer": 0},
)


def normalize_target_id(value) -> str:
    return str(value).upper().replace("KIC", "").strip()


def star_prefix(target_id, quarter) -> str:
    return f"kic_{normalize_target_id(target_id)}_q{int(quarter)}"


def robust_scale(values) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    med = float(np.median(x))
    scale = float(1.4826 * np.median(np.abs(x - med)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(x, ddof=1))
    return scale


def acf1(values) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size < 3:
        return float("nan")
    mask = np.isfinite(x[:-1]) & np.isfinite(x[1:])
    if mask.sum() < 3:
        return float("nan")
    a, b = x[:-1][mask], x[1:][mask]
    if np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def finite_corr(a, b) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.std(a[mask]) <= 0 or np.std(b[mask]) <= 0:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def load_manifest(path: Path, target_limit: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"target_id", "quarter"}.issubset(frame.columns):
        raise ValueError("Manifest requires target_id and quarter.")
    frame = frame.copy()
    frame["target_id"] = frame["target_id"].map(normalize_target_id)
    frame["quarter"] = pd.to_numeric(frame["quarter"], errors="raise").astype(int)
    if "sample_stratum" not in frame:
        frame["sample_stratum"] = "unspecified"
    return frame.drop_duplicates(["target_id", "quarter"]).head(int(target_limit)).reset_index(drop=True)


def load_frame(target_id, quarter, cache_dir: Path, no_download: bool):
    path = Path(cache_dir) / f"{star_prefix(target_id, quarter)}_pdcsap.parquet"
    if path.exists():
        return pd.read_parquet(path), True
    if no_download:
        raise FileNotFoundError(path)
    from adaptive_transit.data.kepler_io import load_kepler_pdcsap
    frame = load_kepler_pdcsap(target_id, quarter).to_dataframe()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame, False


def fit_policy(time, flux, args, policy: dict) -> tuple[dict, object | None]:
    from adaptive_transit.noise_models.gp import fit_smooth_gp_background
    started = perf_counter()
    try:
        fitted = fit_smooth_gp_background(
            time,
            flux,
            max_train_points=args.gp_max_train_points,
            length_scale_days=args.gp_length_scale_days,
            min_length_scale_days=args.gp_min_length_scale_days,
            max_length_scale_days=args.gp_max_length_scale_days,
            measurement_noise_fraction=args.gp_measurement_noise_fraction,
            n_restarts_optimizer=policy["n_restarts_optimizer"],
            random_seed=args.random_seed,
            optimize_kernel=policy["optimize_kernel"],
        )
        p = fitted.parameters
        bound_hit = bool(
            p.get("length_scale_at_lower_bound", False)
            or p.get("length_scale_at_upper_bound", False)
            or p.get("signal_variance_at_lower_bound", False)
            or p.get("signal_variance_at_upper_bound", False)
        )
        row = {
            "policy": policy["policy"],
            "optimized": bool(policy["optimize_kernel"]),
            "n_restarts_optimizer": int(policy["n_restarts_optimizer"]),
            "fit_completed": True,
            "converged": bool(fitted.converged),
            "optimizer_warning_count": int(p.get("optimizer_warning_count", 0)),
            "optimizer_warning_message": str(p.get("optimizer_warning_message", "")),
            "bound_hit": bound_hit,
            "length_scale_days": float(p["length_scale_days"]),
            "signal_variance": float(p["signal_variance"]),
            "log_marginal_likelihood": float(fitted.log_marginal_likelihood),
            "residual_scale_ppm": float(1e6 * robust_scale(fitted.residuals)),
            "abs_residual_acf1": abs(acf1(fitted.residuals)),
            "runtime_seconds": float(perf_counter() - started),
            "error": "",
        }
        return row, fitted
    except Exception as exc:
        return {
            "policy": policy["policy"],
            "optimized": bool(policy["optimize_kernel"]),
            "n_restarts_optimizer": int(policy["n_restarts_optimizer"]),
            "fit_completed": False,
            "converged": False,
            "optimizer_warning_count": np.nan,
            "optimizer_warning_message": "",
            "bound_hit": np.nan,
            "length_scale_days": np.nan,
            "signal_variance": np.nan,
            "log_marginal_likelihood": np.nan,
            "residual_scale_ppm": np.nan,
            "abs_residual_acf1": np.nan,
            "runtime_seconds": float(perf_counter() - started),
            "error": f"{type(exc).__name__}: {exc}",
        }, None


def choose_recommended_policy(group: pd.DataFrame) -> dict:
    """Prefer a converged optimized fit, then non-bound fit, then larger LML.

    Fixed-kernel is used only when no optimized fit converged.  This makes the
    fallback explicit rather than silently redefining the GP model.
    """
    g = group.copy()
    optimized = g[g["optimized"].fillna(False).astype(bool) & g["fit_completed"].fillna(False).astype(bool)]
    clean = optimized[optimized["converged"].fillna(False).astype(bool)]
    if not clean.empty:
        clean = clean.assign(_bound=clean["bound_hit"].fillna(True).astype(bool))
        clean = clean.sort_values(["_bound", "log_marginal_likelihood"], ascending=[True, False])
        row = clean.iloc[0]
        return {"recommended_policy": row["policy"], "recommendation_type": "converged_optimized"}
    fixed = g[(~g["optimized"].fillna(True).astype(bool)) & g["fit_completed"].fillna(False).astype(bool)]
    if not fixed.empty:
        return {"recommended_policy": fixed.iloc[0]["policy"], "recommendation_type": "explicit_fixed_kernel_fallback"}
    if not optimized.empty:
        row = optimized.sort_values("log_marginal_likelihood", ascending=False).iloc[0]
        return {"recommended_policy": row["policy"], "recommendation_type": "optimizer_flagged_no_clean_fit"}
    return {"recommended_policy": "none", "recommendation_type": "all_fits_failed"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit GP optimizer stability on POC base stars only.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-limit", type=int, default=10)
    parser.add_argument("--quality-policy", default="default")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--gp-max-train-points", type=int, default=512)
    parser.add_argument("--gp-length-scale-days", type=float, default=3.0)
    parser.add_argument("--gp-min-length-scale-days", type=float, default=1.0)
    parser.add_argument("--gp-max-length-scale-days", type=float, default=30.0)
    parser.add_argument("--gp-measurement-noise-fraction", type=float, default=0.20)
    parser.add_argument("--random-seed", type=int, default=123)
    args = parser.parse_args(argv)

    from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest_path, args.target_limit)
    rows = []
    for star in tqdm(manifest.to_dict(orient="records"), desc="GP stability audit"):
        target_id, quarter = star["target_id"], int(star["quarter"])
        frame, from_cache = load_frame(target_id, quarter, args.cache_dir, args.no_download)
        regular, _ = preprocess_pdcsap_light_curve(frame, quality_policy=args.quality_policy)
        time = regular["time"].to_numpy(dtype=float)
        flux = regular["normalized_flux"].to_numpy(dtype=float)

        fitted_models = {}
        star_rows = []
        for policy in POLICIES:
            row, fitted = fit_policy(time, flux, args, policy)
            row.update({
                "target_id": target_id,
                "quarter": quarter,
                "sample_stratum": str(star.get("sample_stratum", "unspecified")),
                "from_cache": bool(from_cache),
            })
            star_rows.append(row)
            fitted_models[policy["policy"]] = fitted

        baseline = fitted_models.get("optimized_r0")
        for row in star_rows:
            fitted = fitted_models.get(row["policy"])
            if baseline is not None and fitted is not None:
                row["residual_corr_vs_optimized_r0"] = finite_corr(baseline.residuals, fitted.residuals)
                row["background_corr_vs_optimized_r0"] = finite_corr(baseline.background_mean, fitted.background_mean)
            else:
                row["residual_corr_vs_optimized_r0"] = np.nan
                row["background_corr_vs_optimized_r0"] = np.nan
        rows.extend(star_rows)

    detail = pd.DataFrame(rows)
    detail.to_csv(args.output_dir / "gp_stability_per_star_policy.csv", index=False)

    summary = detail.groupby("policy", dropna=False).agg(
        n_stars=("target_id", "nunique"),
        fit_completion_rate=("fit_completed", "mean"),
        convergence_rate=("converged", "mean"),
        bound_hit_rate=("bound_hit", "mean"),
        median_length_scale_days=("length_scale_days", "median"),
        median_residual_scale_ppm=("residual_scale_ppm", "median"),
        median_abs_residual_acf1=("abs_residual_acf1", "median"),
        median_runtime_seconds=("runtime_seconds", "median"),
        median_residual_corr_vs_r0=("residual_corr_vs_optimized_r0", "median"),
        median_background_corr_vs_r0=("background_corr_vs_optimized_r0", "median"),
    ).reset_index()
    summary.to_csv(args.output_dir / "gp_stability_policy_summary.csv", index=False)

    recommendations = []
    for (target_id, quarter), group in detail.groupby(["target_id", "quarter"], dropna=False):
        rec = choose_recommended_policy(group)
        recommendations.append({"target_id": target_id, "quarter": quarter, **rec})
    recommendations = pd.DataFrame(recommendations)
    recommendations.to_csv(args.output_dir / "gp_policy_recommendations.csv", index=False)

    payload = {
        "purpose": "diagnose_optimizer_stability_before_changing_production_gp_defaults",
        "policies": list(POLICIES),
        "note": "fixed_kernel_3d is an explicit fallback benchmark, not an automatic production replacement",
    }
    (args.output_dir / "gp_stability_config.json").write_text(json.dumps(payload, indent=2) + "\n")

    print("\n=== GP STABILITY POLICY SUMMARY ===\n")
    print(summary.to_string(index=False))
    print("\n=== PER-STAR RECOMMENDATION ===\n")
    print(recommendations.to_string(index=False))
    print(f"\nWrote: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
