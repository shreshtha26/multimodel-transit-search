#!/usr/bin/env python3
"""Audit GP lower length-scale bounds without changing production GP defaults.

Why this exists
---------------
The current GP POC uses an RBF lower length-scale bound of 1.0 day.  In the
10-star stability audit, every optimizer-flagged fit landed exactly on that
lower bound and optimizer restarts reproduced the same solution.  This script
therefore tests whether a modest relaxation of that bound improves the GP fit
while preserving the injected BATMAN transit.

This is a sensitivity audit only.  It does NOT modify gp.py, rerun BLS/TLS,
or declare any candidate bound to be the production default.

Policies tested by default
--------------------------
1.00 d, 0.75 d, 0.50 d

The shortest default candidate (0.50 d = 12 h) remains longer than the longest
BATMAN POC duration (8 h).  That is only a conservative screening rule; the
actual signal-transfer metrics remain the scientific check.

Outputs
-------
gp_length_scale_bound_per_star.csv
gp_length_scale_bound_signal_transfer.csv
gp_length_scale_bound_policy_summary.csv
gp_length_scale_bound_duration_summary.csv
gp_length_scale_bound_config.json
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
DEFAULT_CASES = PROJECT_ROOT / "configs/batman_physical_poc_cases.csv"
DEFAULT_CACHE = PROJECT_ROOT / "outputs/cache/kepler_light_curves"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/experiments/batman_physical_detection_poc/pilot10/qc_analysis/gp_length_scale_bound"
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


def load_manifest(path: Path, target_limit: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"target_id", "quarter"}.issubset(frame.columns):
        raise ValueError("Manifest requires target_id and quarter.")
    frame = frame.copy()
    frame["target_id"] = frame["target_id"].map(normalize_target_id)
    frame["quarter"] = pd.to_numeric(frame["quarter"], errors="raise").astype(int)
    if "sample_stratum" not in frame:
        frame["sample_stratum"] = "unspecified"
    return (
        frame.drop_duplicates(["target_id", "quarter"])
        .head(int(target_limit))
        .reset_index(drop=True)
    )


def load_cases(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"period_days", "duration_hours", "depth", "phase_fraction"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Case file missing columns: {missing}")
    frame = frame.copy().reset_index(drop=True)
    if "case_index" not in frame:
        frame["case_index"] = np.arange(len(frame), dtype=int)
    if "case_label" not in frame:
        frame["case_label"] = frame["case_index"].map(lambda x: f"case_{x}")
    return frame


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


def signal_transfer_metrics(original_template, transmitted_template, base_residuals):
    truth = np.asarray(original_template, dtype=float).reshape(-1)
    transmitted = np.asarray(transmitted_template, dtype=float).reshape(-1)
    background = np.asarray(base_residuals, dtype=float).reshape(-1)

    finite = np.isfinite(truth) & np.isfinite(transmitted)
    if finite.sum() < 20:
        return {k: np.nan for k in (
            "template_amplitude_ratio",
            "peak_depth_ratio",
            "template_energy_ratio",
            "template_correlation",
            "template_rmse_ppm",
            "oracle_signal_snr",
        )}

    threshold = max(1.0e-12, 1.0e-6 * float(np.nanmax(np.abs(truth[finite]))))
    active = finite & (truth < -threshold)
    if active.sum() < 2:
        return {k: np.nan for k in (
            "template_amplitude_ratio",
            "peak_depth_ratio",
            "template_energy_ratio",
            "template_correlation",
            "template_rmse_ppm",
            "oracle_signal_snr",
        )}

    denom = float(np.dot(truth[finite], truth[finite]))
    amp = float(np.dot(transmitted[finite], truth[finite]) / denom) if denom > 0 else np.nan

    truth_peak = float(np.max(-truth[active]))
    trans_peak = float(np.max(-transmitted[active]))
    peak = trans_peak / truth_peak if truth_peak > 0 else np.nan

    truth_energy = float(np.linalg.norm(truth[finite]))
    trans_energy = float(np.linalg.norm(transmitted[finite]))
    energy = trans_energy / truth_energy if truth_energy > 0 else np.nan

    corr = (
        float(np.corrcoef(truth[active], transmitted[active])[0, 1])
        if np.std(truth[active]) > 0 and np.std(transmitted[active]) > 0
        else np.nan
    )
    rmse_ppm = float(1e6 * np.sqrt(np.mean((transmitted[finite] - truth[finite]) ** 2)))

    bg_scale = robust_scale(background)
    oracle = trans_energy / bg_scale if np.isfinite(bg_scale) and bg_scale > 0 else np.nan

    return {
        "template_amplitude_ratio": amp,
        "peak_depth_ratio": peak,
        "template_energy_ratio": energy,
        "template_correlation": corr,
        "template_rmse_ppm": rmse_ppm,
        "oracle_signal_snr": oracle,
    }


def policy_name(bound: float) -> str:
    return f"minls_{bound:g}d".replace(".", "p")


def parse_bounds(text: str) -> list[float]:
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("At least one lower bound is required.")
    if any(x <= 0 for x in values):
        raise ValueError("All lower bounds must be positive.")
    return values


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit GP lower length-scale sensitivity and BATMAN signal transfer."
    )
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-limit", type=int, default=10)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--quality-policy", default="default")
    parser.add_argument("--min-length-scale-grid", default="1.0,0.75,0.5")
    parser.add_argument("--initial-length-scale-days", type=float, default=3.0)
    parser.add_argument("--max-length-scale-days", type=float, default=30.0)
    parser.add_argument("--gp-max-train-points", type=int, default=512)
    parser.add_argument("--gp-measurement-noise-fraction", type=float, default=0.20)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--impact-parameter", type=float, default=0.3)
    parser.add_argument("--limb-u1", type=float, default=0.30)
    parser.add_argument("--limb-u2", type=float, default=0.20)
    parser.add_argument("--supersample-factor", type=int, default=7)
    args = parser.parse_args(argv)

    bounds = parse_bounds(args.min_length_scale_grid)
    cases = load_cases(args.case_file)
    max_duration_days = float(pd.to_numeric(cases["duration_hours"]).max()) / 24.0

    from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve
    from adaptive_transit.noise_models.gp import (
        apply_prepared_smooth_gp_filter,
        fit_smooth_gp_background,
        prepare_smooth_gp_filter,
    )
    from adaptive_transit.injections.batman import inject_batman_transit

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest_path, args.target_limit)

    star_rows = []
    transfer_rows = []

    for star in tqdm(manifest.to_dict(orient="records"), desc="GP lower-bound audit"):
        target_id = star["target_id"]
        quarter = int(star["quarter"])
        frame, from_cache = load_frame(target_id, quarter, args.cache_dir, args.no_download)
        regular, _ = preprocess_pdcsap_light_curve(
            frame,
            quality_policy=args.quality_policy,
            require_finite_flux_error=False,
            normalization_fit_fraction=1.0,
        )
        time = regular["time"].to_numpy(dtype=float)
        flux = regular["normalized_flux"].to_numpy(dtype=float)
        finite = np.isfinite(time) & np.isfinite(flux)
        t_min = float(np.min(time[finite]))

        for lower in bounds:
            started = perf_counter()
            policy = policy_name(lower)
            row = {
                "target_id": target_id,
                "quarter": quarter,
                "sample_stratum": str(star.get("sample_stratum", "unspecified")),
                "policy": policy,
                "min_length_scale_days": float(lower),
                "max_poc_transit_duration_days": max_duration_days,
                "lower_bound_to_max_duration_ratio": float(lower / max_duration_days),
                "from_cache": bool(from_cache),
            }

            try:
                fitted = fit_smooth_gp_background(
                    time,
                    flux,
                    max_train_points=args.gp_max_train_points,
                    length_scale_days=args.initial_length_scale_days,
                    min_length_scale_days=float(lower),
                    max_length_scale_days=args.max_length_scale_days,
                    measurement_noise_fraction=args.gp_measurement_noise_fraction,
                    n_restarts_optimizer=0,
                    random_seed=args.random_seed,
                    optimize_kernel=True,
                )
                prepared = prepare_smooth_gp_filter(time, fitted)
                p = fitted.parameters

                row.update(
                    fit_completed=True,
                    converged=bool(fitted.converged),
                    length_scale_days=float(p["length_scale_days"]),
                    length_scale_at_lower_bound=bool(
                        p.get("length_scale_at_lower_bound", False)
                    ),
                    signal_variance=float(p["signal_variance"]),
                    log_marginal_likelihood=float(fitted.log_marginal_likelihood),
                    residual_scale_ppm=float(1e6 * robust_scale(fitted.residuals)),
                    abs_residual_acf1=abs(acf1(fitted.residuals)),
                    optimizer_warning_count=int(p.get("optimizer_warning_count", 0)),
                    optimizer_warning_message=str(
                        p.get("optimizer_warning_message", "")
                    ),
                    runtime_seconds=float(perf_counter() - started),
                    error="",
                )
                star_rows.append(row)

                base_residuals = np.asarray(fitted.residuals, dtype=float)
                for _, case in cases.iterrows():
                    period = float(case["period_days"])
                    duration_days = float(case["duration_hours"]) / 24.0
                    depth = float(case["depth"])
                    phase = float(case["phase_fraction"])
                    epoch = t_min + phase * period

                    injected, template, _, truth = inject_batman_transit(
                        time,
                        flux,
                        period_days=period,
                        epoch_days=epoch,
                        duration_days=duration_days,
                        depth=depth,
                        impact_parameter=args.impact_parameter,
                        limb_darkening_coefficients=(args.limb_u1, args.limb_u2),
                        supersample_factor=args.supersample_factor,
                    )
                    processed = apply_prepared_smooth_gp_filter(
                        injected, prepared
                    ).residuals
                    transmitted = np.asarray(processed, dtype=float) - base_residuals
                    metrics = signal_transfer_metrics(
                        template, transmitted, base_residuals
                    )
                    transfer_rows.append(
                        {
                            "target_id": target_id,
                            "quarter": quarter,
                            "sample_stratum": str(
                                star.get("sample_stratum", "unspecified")
                            ),
                            "policy": policy,
                            "min_length_scale_days": float(lower),
                            "case_index": int(case["case_index"]),
                            "case_label": str(case["case_label"]),
                            "period_days": period,
                            "duration_hours": float(case["duration_hours"]),
                            "depth": depth,
                            "phase_fraction": phase,
                            "radius_ratio": float(truth.radius_ratio),
                            **metrics,
                        }
                    )
            except Exception as exc:
                row.update(
                    fit_completed=False,
                    converged=False,
                    length_scale_days=np.nan,
                    length_scale_at_lower_bound=np.nan,
                    signal_variance=np.nan,
                    log_marginal_likelihood=np.nan,
                    residual_scale_ppm=np.nan,
                    abs_residual_acf1=np.nan,
                    optimizer_warning_count=np.nan,
                    optimizer_warning_message="",
                    runtime_seconds=float(perf_counter() - started),
                    error=f"{type(exc).__name__}: {exc}",
                )
                star_rows.append(row)

    stars = pd.DataFrame(star_rows)
    transfers = pd.DataFrame(transfer_rows)

    stars.to_csv(args.output_dir / "gp_length_scale_bound_per_star.csv", index=False)
    transfers.to_csv(
        args.output_dir / "gp_length_scale_bound_signal_transfer.csv", index=False
    )

    policy_summary = (
        stars.groupby(["policy", "min_length_scale_days"], dropna=False)
        .agg(
            n_stars=("target_id", "nunique"),
            fit_completion_rate=("fit_completed", "mean"),
            convergence_rate=("converged", "mean"),
            lower_bound_hit_rate=("length_scale_at_lower_bound", "mean"),
            median_fitted_length_scale_days=("length_scale_days", "median"),
            median_log_marginal_likelihood=("log_marginal_likelihood", "median"),
            median_residual_scale_ppm=("residual_scale_ppm", "median"),
            median_abs_residual_acf1=("abs_residual_acf1", "median"),
            median_runtime_seconds=("runtime_seconds", "median"),
        )
        .reset_index()
    )

    if not transfers.empty:
        trans_summary = (
            transfers.groupby(["policy", "min_length_scale_days"], dropna=False)
            .agg(
                n_injections=("case_index", "size"),
                median_template_amplitude_ratio=(
                    "template_amplitude_ratio",
                    "median",
                ),
                median_peak_depth_ratio=("peak_depth_ratio", "median"),
                median_template_energy_ratio=("template_energy_ratio", "median"),
                median_template_correlation=("template_correlation", "median"),
                median_template_rmse_ppm=("template_rmse_ppm", "median"),
                median_oracle_signal_snr=("oracle_signal_snr", "median"),
            )
            .reset_index()
        )
        policy_summary = policy_summary.merge(
            trans_summary,
            on=["policy", "min_length_scale_days"],
            how="left",
            validate="one_to_one",
        )

        duration_summary = (
            transfers.groupby(
                ["policy", "min_length_scale_days", "duration_hours"],
                dropna=False,
            )
            .agg(
                n_injections=("case_index", "size"),
                median_template_amplitude_ratio=(
                    "template_amplitude_ratio",
                    "median",
                ),
                median_peak_depth_ratio=("peak_depth_ratio", "median"),
                median_template_correlation=("template_correlation", "median"),
                median_template_rmse_ppm=("template_rmse_ppm", "median"),
                median_oracle_signal_snr=("oracle_signal_snr", "median"),
            )
            .reset_index()
        )
    else:
        duration_summary = pd.DataFrame()

    policy_summary.to_csv(
        args.output_dir / "gp_length_scale_bound_policy_summary.csv", index=False
    )
    duration_summary.to_csv(
        args.output_dir / "gp_length_scale_bound_duration_summary.csv", index=False
    )

    config = {
        "purpose": "sensitivity_audit_only_do_not_change_production_gp_defaults",
        "min_length_scale_grid_days": bounds,
        "max_poc_transit_duration_days": max_duration_days,
        "note": (
            "Candidate lower bounds are judged jointly on optimizer behavior, "
            "background whitening, and BATMAN signal transfer."
        ),
    }
    (args.output_dir / "gp_length_scale_bound_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )

    print("\n=== GP LOWER-BOUND POLICY SUMMARY ===\n")
    print(policy_summary.to_string(index=False))

    if not duration_summary.empty:
        longest = duration_summary[
            np.isclose(
                pd.to_numeric(duration_summary["duration_hours"]),
                float(pd.to_numeric(cases["duration_hours"]).max()),
            )
        ]
        print("\n=== LONGEST-TRANSIT (8 h) SIGNAL TRANSFER ===\n")
        print(longest.to_string(index=False))

    print(
        "\nInterpretation rule: a lower bound is not acceptable merely because "
        "convergence or likelihood improves. It must also preserve the BATMAN "
        "transit, especially the longest 8 h cases."
    )
    print(f"\nWrote: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
