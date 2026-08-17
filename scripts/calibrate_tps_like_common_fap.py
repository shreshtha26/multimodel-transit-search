#!/usr/bin/env python3
"""Common empirical-FAP calibration for raw vs hardened TPS-like ranking.

"Common" here means:
1. identical native preprocessed star,
2. identical fixed TPS-like wavelet noise model,
3. identical moving-block surrogate realization for each trial,
4. identical search grid/configuration,
5. identical nominal FAP target.

Raw MES and the event-consistency score retain separate numerical thresholds
because they are different statistics with different scales.

This script also performs the *true* native zero-injection control by searching
the original preprocessed flux directly.  It never calls BATMAN with an
artificially tiny depth.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import run_tps_like_batman_poc as base
from adaptive_transit.detection.common_fap import (
    calibration_row,
    empirical_p_value,
)
from adaptive_transit.detection.false_alarm import moving_block_surrogate
from adaptive_transit.detection.tps_like import (
    prepare_tps_like_noise_model,
    run_tps_like_search,
)
from adaptive_transit.detection.tps_like_hardening import (
    DEFAULT_EVENT_CONSISTENCY_CONFIG,
    config_dict,
    harden_tps_like_result,
)
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HARDENED_DIR = (
    base.DEFAULT_PHYSICAL_POC / "tps_like_comparator_hardened"
)
DEFAULT_OUTPUT = DEFAULT_HARDENED_DIR / "common_fap"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-path", type=Path, default=base.DEFAULT_MANIFEST
    )
    parser.add_argument("--cache-dir", type=Path, default=base.DEFAULT_CACHE)
    parser.add_argument(
        "--hardened-results-dir",
        type=Path,
        default=DEFAULT_HARDENED_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-limit", type=int, default=10)
    parser.add_argument("--quality-policy", default="default")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-resume", action="store_true")

    parser.add_argument("--n-null-trials-per-star", type=int, default=100)
    parser.add_argument("--fap-level", type=float, default=0.01)
    parser.add_argument("--null-block-size-cadences", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    parser.add_argument(
        "--minimum-success-fraction", type=float, default=0.90
    )
    return parser.parse_args(argv)


def _load_search_settings(hardened_results_dir: Path, cli) -> SimpleNamespace:
    defaults = base.parse_args([])
    config_path = Path(hardened_results_dir) / "run_config.json"
    saved = {}
    if config_path.exists():
        saved = json.loads(config_path.read_text())

    keys = [
        "wavelet",
        "max_wavelet_level",
        "noise_window_cadences",
        "min_segment_cadences",
        "min_events",
        "min_period_days",
        "max_period_days",
        "duration_hours_grid",
    ]
    data = {key: getattr(defaults, key) for key in keys}
    for key in keys:
        if key in saved:
            data[key] = saved[key]

    data["duration_hours_grid"] = tuple(
        float(value) for value in data["duration_hours_grid"]
    )
    data["cache_dir"] = Path(cli.cache_dir)
    data["quality_policy"] = cli.quality_policy
    data["no_download"] = bool(cli.no_download)
    return SimpleNamespace(**data)


def _search_once(time, flux, segment_id, prepared, settings):
    result = run_tps_like_search(
        time,
        flux,
        segment_id,
        prepared_noise_model=prepared,
        min_period_days=settings.min_period_days,
        max_period_days=settings.max_period_days,
        duration_hours_grid=settings.duration_hours_grid,
        wavelet=settings.wavelet,
        max_level=settings.max_wavelet_level,
        noise_window_cadences=settings.noise_window_cadences,
        min_segment_cadences=settings.min_segment_cadences,
        min_events=settings.min_events,
    )
    hardened = harden_tps_like_result(
        result,
        time,
        flux,
        config=DEFAULT_EVENT_CONSISTENCY_CONFIG,
    )
    raw = result["summary"]
    hard = hardened["summary"]
    return {
        "raw_top1_mes": float(raw["mes"]),
        "raw_top1_period_days": float(raw["period_days"]),
        "raw_top1_epoch_days": float(raw["epoch_days"]),
        "raw_top1_duration_hours": float(raw["duration_hours"]),
        "event_consistency_score": float(hard["event_consistency_score"]),
        "robust_veto_score": float(hard["robust_veto_score"]),
        "hardened_period_days": float(hard["period_days"]),
        "hardened_epoch_days": float(hard["epoch_days"]),
        "hardened_duration_hours": float(hard["duration_hours"]),
        "selected_raw_mes": float(hard["selected_raw_mes"]),
        "consistency_weight": float(hard["consistency_weight"]),
        "valid_event_count": int(hard["valid_event_count"]),
        "event_observability_fraction": float(
            hard["event_observability_fraction"]
        ),
        "positive_event_fraction": float(hard["positive_event_fraction"]),
        "single_event_fraction": float(hard["single_event_fraction"]),
        "leave_one_out_ratio_normalized": float(
            hard["leave_one_out_ratio_normalized"]
        ),
        "event_consistent_flag": bool(hard["event_consistent_flag"]),
        "transit_consistency_veto_pass": bool(
            hard["transit_consistency_veto_pass"]
        ),
        "diagnostic_consistency_veto_pass": bool(
            hard["diagnostic_consistency_veto_pass"]
        ),
        "event_depth_chi2_pvalue": float(hard["event_depth_chi2_pvalue"]),
        "event_depth_reduced_chi2": float(hard["event_depth_reduced_chi2"]),
        "odd_even_tested": bool(hard["odd_even_tested"]),
        "odd_even_depth_pvalue": float(hard["odd_even_depth_pvalue"]),
        "odd_even_depth_z": float(hard["odd_even_depth_z"]),
        "odd_even_depth_difference_fraction": float(
            hard["odd_even_depth_difference_fraction"]
        ),
        "robust_event_snr": float(hard["robust_event_snr"]),
        "fails_min_events": bool(hard["fails_min_events"]),
        "fails_observability": bool(hard["fails_observability"]),
        "fails_positive_events": bool(hard["fails_positive_events"]),
        "fails_single_event_dominance": bool(
            hard["fails_single_event_dominance"]
        ),
        "fails_leave_one_out": bool(hard["fails_leave_one_out"]),
        "fails_depth_chi2": bool(hard["fails_depth_chi2"]),
        "fails_odd_even": bool(hard["fails_odd_even"]),
        "fails_robust_sign": bool(hard["fails_robust_sign"]),
        "veto_reason": str(hard["veto_reason"]),
        "diagnostic_veto_reason": str(hard["diagnostic_veto_reason"]),
        "any_candidate_survives_veto": bool(hard["any_candidate_survives_veto"]),
        "selection_status": str(hard["selection_status"]),
        "ranking_changed": bool(hard["ranking_changed"]),
    }


def _trial_seed_table(seed: int, target_id: str, quarter: int, n_trials: int):
    try:
        target_number = int(str(target_id).replace("KIC", "").strip())
    except ValueError:
        target_number = abs(hash(str(target_id))) % (2**31 - 1)

    seq = np.random.SeedSequence(
        [int(seed), int(target_number), int(quarter), 91173]
    )
    children = seq.spawn(int(n_trials))
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in children
    ]


def _calibrate_star(payload):
    row, cli_dict, search_dict = payload
    cli = SimpleNamespace(**cli_dict)
    settings = SimpleNamespace(**search_dict)
    settings.cache_dir = Path(settings.cache_dir)
    settings.duration_hours_grid = tuple(settings.duration_hours_grid)

    target_id = base.normalize_target_id(row["target_id"])
    quarter = int(row["quarter"])
    stratum = str(row.get("sample_stratum", "unspecified"))
    star_key = base.star_prefix(target_id, quarter)
    star_dir = Path(cli.output_dir) / "stars" / star_key
    star_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = star_dir / "shared_null_trials.csv"

    load_args = SimpleNamespace(
        cache_dir=Path(cli.cache_dir),
        no_download=bool(cli.no_download),
    )
    frame, _ = base.load_frame(target_id, quarter, load_args)
    regular, _ = preprocess_pdcsap_light_curve(
        frame,
        quality_policy=settings.quality_policy,
        require_finite_flux_error=False,
        normalization_fit_fraction=1.0,
    )
    time = regular["time"].to_numpy(dtype=float)
    flux = regular["normalized_flux"].to_numpy(dtype=float)
    segment_id = regular["segment_id"].to_numpy(dtype=int)

    prepared = prepare_tps_like_noise_model(
        flux,
        segment_id,
        wavelet=settings.wavelet,
        max_level=settings.max_wavelet_level,
        noise_window_cadences=settings.noise_window_cadences,
        min_segment_cadences=settings.min_segment_cadences,
    )

    # True native zero-injection control: no BATMAN call at all.
    native = _search_once(time, flux, segment_id, prepared, settings)

    seeds = _trial_seed_table(
        cli.seed, target_id, quarter, cli.n_null_trials_per_star
    )
    seed_by_trial = dict(enumerate(seeds))

    if checkpoint.exists() and not cli.no_resume:
        existing = pd.read_csv(checkpoint)
    else:
        existing = pd.DataFrame()

    completed = set()
    if not existing.empty and "trial_index" in existing:
        completed = set(
            pd.to_numeric(existing["trial_index"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )

    rows = existing.to_dict(orient="records") if not existing.empty else []
    pending = [
        trial_index
        for trial_index in range(cli.n_null_trials_per_star)
        if trial_index not in completed
    ]

    for count, trial_index in enumerate(pending, start=1):
        trial_seed = seed_by_trial[trial_index]
        rng = np.random.default_rng(trial_seed)
        common = {
            "target_id": target_id,
            "quarter": quarter,
            "sample_stratum": stratum,
            "trial_index": int(trial_index),
            "trial_seed": int(trial_seed),
            "null_block_size_cadences": int(
                cli.null_block_size_cadences
            ),
        }
        try:
            surrogate = moving_block_surrogate(
                flux,
                block_size=int(cli.null_block_size_cadences),
                rng=rng,
            )
            scored = _search_once(
                time, surrogate, segment_id, prepared, settings
            )
            rows.append(
                {
                    **common,
                    "success": True,
                    "error": "",
                    **scored,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **common,
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        if count % 5 == 0 or count == len(pending):
            pd.DataFrame(rows).sort_values("trial_index").to_csv(
                checkpoint, index=False
            )

    trials = pd.DataFrame(rows).sort_values("trial_index").reset_index(drop=True)
    success_mask = (
        trials["success"].astype(str).str.lower().isin({"true", "1"})
        if "success" in trials
        else pd.Series(False, index=trials.index)
    )
    valid = trials.loc[success_mask].copy()
    success_fraction = (
        len(valid) / cli.n_null_trials_per_star
        if cli.n_null_trials_per_star > 0
        else 0.0
    )
    if success_fraction < cli.minimum_success_fraction:
        raise RuntimeError(
            f"{star_key}: only {len(valid)}/{cli.n_null_trials_per_star} "
            "null trials succeeded; refusing threshold calibration."
        )

    raw_cal = calibration_row(
        valid["raw_top1_mes"],
        method="tps_like_raw_top1",
        score_name="raw_top1_mes",
        fap_level=cli.fap_level,
        requested_trials=cli.n_null_trials_per_star,
    )
    hard_cal = calibration_row(
        valid["robust_veto_score"],
        method="tps_like_chi2_robust_hardened",
        score_name="robust_veto_score",
        fap_level=cli.fap_level,
        requested_trials=cli.n_null_trials_per_star,
    )
    for item in (raw_cal, hard_cal):
        item.update(
            {
                "target_id": target_id,
                "quarter": quarter,
                "sample_stratum": stratum,
            }
        )

    native_row = {
        "target_id": target_id,
        "quarter": quarter,
        "sample_stratum": stratum,
        **native,
        "raw_score_threshold": raw_cal["score_threshold"],
        "hardened_score_threshold": hard_cal["score_threshold"],
        "hardened_score_name": "robust_veto_score",
        "raw_empirical_p_value": empirical_p_value(
            native["raw_top1_mes"], valid["raw_top1_mes"]
        ),
        "hardened_empirical_p_value": empirical_p_value(
            native["robust_veto_score"],
            valid["robust_veto_score"],
        ),
        "raw_passes_fap": bool(
            empirical_p_value(
                native["raw_top1_mes"], valid["raw_top1_mes"]
            )
            <= cli.fap_level
        ),
        "hardened_passes_fap": bool(
            empirical_p_value(
                native["robust_veto_score"], valid["robust_veto_score"]
            )
            <= cli.fap_level
        ),
    }
    pd.DataFrame([native_row]).to_csv(
        star_dir / "native_zero_injection.csv", index=False
    )
    pd.DataFrame([raw_cal, hard_cal]).to_csv(
        star_dir / "fap_thresholds.csv", index=False
    )

    # The seeds are the reproducible shared-null catalogue.  Other raw-flux
    # detectors can regenerate exactly the same surrogate trials.
    seed_manifest = pd.DataFrame(
        {
            "target_id": target_id,
            "quarter": quarter,
            "trial_index": np.arange(cli.n_null_trials_per_star, dtype=int),
            "trial_seed": seeds,
            "null_block_size_cadences": int(
                cli.null_block_size_cadences
            ),
        }
    )
    seed_manifest.to_csv(
        star_dir / "shared_null_trial_manifest.csv", index=False
    )

    return {
        "thresholds": [raw_cal, hard_cal],
        "native": native_row,
        "trials_path": str(checkpoint),
        "seed_manifest_path": str(
            star_dir / "shared_null_trial_manifest.csv"
        ),
    }


def _threshold_lookup(thresholds: pd.DataFrame, method: str):
    subset = thresholds.loc[thresholds["method"] == method].copy()
    return {
        (str(row["target_id"]), int(row["quarter"])): float(
            row["score_threshold"]
        )
        for _, row in subset.iterrows()
    }


def _null_lookup(null_trials: pd.DataFrame, score_column: str):
    lookup = {}
    for (target_id, quarter), group in null_trials.groupby(
        ["target_id", "quarter"], dropna=False
    ):
        success = group["success"].astype(str).str.lower().isin(
            {"true", "1"}
        )
        values = pd.to_numeric(
            group.loc[success, score_column], errors="coerce"
        ).dropna()
        lookup[(str(target_id), int(quarter))] = values.to_numpy(dtype=float)
    return lookup


def calibrate_injections(hardened_dir, output_dir, thresholds, null_trials):
    path = Path(hardened_dir) / "tps_like_hardened_results.csv"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()

    injections = pd.read_csv(path)
    if injections.empty:
        return injections, pd.DataFrame()

    injections["target_id"] = injections["target_id"].map(
        base.normalize_target_id
    )
    injections["quarter"] = pd.to_numeric(
        injections["quarter"], errors="raise"
    ).astype(int)

    raw_thresholds = _threshold_lookup(
        thresholds, "tps_like_raw_top1"
    )
    hard_thresholds = _threshold_lookup(
        thresholds, "tps_like_chi2_robust_hardened"
    )
    raw_nulls = _null_lookup(null_trials, "raw_top1_mes")
    hard_nulls = _null_lookup(null_trials, "robust_veto_score")

    raw_threshold_column = []
    hard_threshold_column = []
    raw_p = []
    hard_p = []

    for _, row in injections.iterrows():
        key = (str(row["target_id"]), int(row["quarter"]))
        raw_thr = raw_thresholds.get(key, np.nan)
        hard_thr = hard_thresholds.get(key, np.nan)
        raw_threshold_column.append(raw_thr)
        hard_threshold_column.append(hard_thr)

        raw_score = float(row.get("raw_top1_mes", np.nan))
        hard_score = float(row.get("robust_veto_score", np.nan))
        raw_p.append(
            empirical_p_value(raw_score, raw_nulls[key])
            if key in raw_nulls and np.isfinite(raw_score)
            else np.nan
        )
        hard_p.append(
            empirical_p_value(hard_score, hard_nulls[key])
            if key in hard_nulls and np.isfinite(hard_score)
            else np.nan
        )

    injections["raw_fap_threshold"] = raw_threshold_column
    injections["hardened_fap_threshold"] = hard_threshold_column
    injections["raw_empirical_p_value"] = raw_p
    injections["hardened_empirical_p_value"] = hard_p
    # Use empirical p-values for the actual decision.  This is robust to the
    # point mass at zero introduced by hard vetoes, where threshold comparisons
    # alone can otherwise mishandle ties.
    injections["raw_passes_fap"] = (
        pd.to_numeric(injections["raw_empirical_p_value"], errors="coerce")
        <= float(thresholds["fap_level"].iloc[0])
    )
    injections["hardened_passes_fap"] = (
        pd.to_numeric(injections["hardened_empirical_p_value"], errors="coerce")
        <= float(thresholds["fap_level"].iloc[0])
    )
    injections["raw_fap_harmonic_recovered"] = (
        injections["raw_top1_harmonic_period_recovered"]
        .astype(str)
        .str.lower()
        .isin({"true", "1"})
        & injections["raw_passes_fap"]
    )
    injections["hardened_fap_harmonic_recovered"] = (
        injections["harmonic_period_recovered"]
        .astype(str)
        .str.lower()
        .isin({"true", "1"})
        & injections["hardened_passes_fap"]
    )

    injections.to_csv(
        Path(output_dir) / "calibrated_injection_results.csv",
        index=False,
    )

    success = injections["success"].astype(str).str.lower().isin(
        {"true", "1"}
    )
    valid = injections.loc[success].copy()
    if valid.empty:
        return injections, pd.DataFrame()

    summary = pd.DataFrame(
        [
            {
                "method": "tps_like_raw_top1",
                "n_cases": len(valid),
                "uncalibrated_harmonic_recovery": float(
                    valid["raw_top1_harmonic_period_recovered"]
                    .astype(str)
                    .str.lower()
                    .isin({"true", "1"})
                    .mean()
                ),
                "fraction_above_fap_threshold": float(
                    valid["raw_passes_fap"].mean()
                ),
                "harmonic_recovery_at_common_fap": float(
                    valid["raw_fap_harmonic_recovered"].mean()
                ),
                "median_score": float(valid["raw_top1_mes"].median()),
                "median_threshold": float(
                    valid["raw_fap_threshold"].median()
                ),
            },
            {
                "method": "tps_like_chi2_robust_hardened",
                "n_cases": len(valid),
                "uncalibrated_harmonic_recovery": float(
                    valid["harmonic_period_recovered"]
                    .astype(str)
                    .str.lower()
                    .isin({"true", "1"})
                    .mean()
                ),
                "fraction_above_fap_threshold": float(
                    valid["hardened_passes_fap"].mean()
                ),
                "harmonic_recovery_at_common_fap": float(
                    valid["hardened_fap_harmonic_recovered"].mean()
                ),
                "median_score": float(
                    valid["robust_veto_score"].median()
                ),
                "median_threshold": float(
                    valid["hardened_fap_threshold"].median()
                ),
                "ranking_changed_fraction": float(
                    valid["ranking_changed"]
                    .astype(str)
                    .str.lower()
                    .isin({"true", "1"})
                    .mean()
                ),
            },
        ]
    )
    summary.to_csv(
        Path(output_dir) / "common_fap_summary.csv", index=False
    )
    return injections, summary


def main(argv=None):
    args = parse_args(argv)
    if not 0 < args.fap_level < 1:
        raise ValueError("--fap-level must lie between 0 and 1.")
    if args.n_null_trials_per_star < 10:
        raise ValueError("--n-null-trials-per-star must be at least 10.")
    if args.null_block_size_cadences < 2:
        raise ValueError("--null-block-size-cadences must be at least 2.")

    args.hardened_results_dir = Path(args.hardened_results_dir)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = base.load_manifest(args.manifest_path, args.target_limit)
    settings = _load_search_settings(args.hardened_results_dir, args)

    config = {
        "scientific_status": "empirical_common_FAP_chi2_robust_validation_POC",
        "definition_of_common": [
            "same native preprocessed star",
            "same fixed TPS-like wavelet noise model",
            "same moving-block surrogate realization per star/trial",
            "same search grid and duration bank",
            "same nominal FAP level",
            "separate numerical threshold for each score scale",
        ],
        "fap_level": args.fap_level,
        "n_null_trials_per_star": args.n_null_trials_per_star,
        "null_block_size_cadences": args.null_block_size_cadences,
        "seed": args.seed,
        "minimum_success_fraction": args.minimum_success_fraction,
        "search_settings": vars(settings),
        "event_consistency": config_dict(
            DEFAULT_EVENT_CONSISTENCY_CONFIG
        ),
        "native_zero_injection_definition": (
            "search original preprocessed stellar flux directly; BATMAN skipped"
        ),
    }
    (args.output_dir / "calibration_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n"
    )

    cli_dict = vars(args).copy()
    cli_dict["cache_dir"] = str(args.cache_dir)
    cli_dict["hardened_results_dir"] = str(args.hardened_results_dir)
    cli_dict["output_dir"] = str(args.output_dir)

    search_dict = vars(settings).copy()
    search_dict["cache_dir"] = str(settings.cache_dir)
    search_dict["duration_hours_grid"] = list(
        settings.duration_hours_grid
    )

    payloads = [
        (row, cli_dict, search_dict)
        for row in manifest.to_dict(orient="records")
    ]

    outputs = []
    workers = max(1, min(int(args.max_workers), len(payloads)))
    if workers == 1:
        for payload in tqdm(payloads, desc="Common-FAP stars"):
            outputs.append(_calibrate_star(payload))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_calibrate_star, payload): payload[0]
                for payload in payloads
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Common-FAP stars",
            ):
                outputs.append(future.result())

    thresholds = pd.DataFrame(
        [row for output in outputs for row in output["thresholds"]]
    )
    native = pd.DataFrame([output["native"] for output in outputs])

    null_frames = [
        pd.read_csv(output["trials_path"]) for output in outputs
    ]
    null_trials = pd.concat(null_frames, ignore_index=True)
    null_trials["target_id"] = null_trials["target_id"].map(
        base.normalize_target_id
    )

    seed_frames = [
        pd.read_csv(output["seed_manifest_path"]) for output in outputs
    ]
    shared_manifest = pd.concat(seed_frames, ignore_index=True)

    thresholds.to_csv(args.output_dir / "fap_thresholds.csv", index=False)
    native.to_csv(
        args.output_dir / "native_zero_injection_results.csv", index=False
    )
    null_trials.to_csv(
        args.output_dir / "shared_null_trials.csv", index=False
    )
    shared_manifest.to_csv(
        args.output_dir / "shared_null_trial_manifest.csv", index=False
    )

    _, summary = calibrate_injections(
        args.hardened_results_dir,
        args.output_dir,
        thresholds,
        null_trials,
    )

    report_lines = [
        "TPS-LIKE EVENT-CONSISTENCY COMMON-FAP VALIDATION",
        "=" * 60,
        f"Stars: {len(manifest)}",
        f"Null trials requested per star: {args.n_null_trials_per_star}",
        f"Nominal FAP: {100 * args.fap_level:.3f}%",
        f"Moving-block size: {args.null_block_size_cadences} cadences",
        "",
        "TRUE NATIVE ZERO-INJECTION CONTROL",
        f"Raw native controls above threshold: {int(native['raw_passes_fap'].sum())}/{len(native)}",
        f"Hardened native controls above threshold: {int(native['hardened_passes_fap'].sum())}/{len(native)}",
    ]
    if not summary.empty:
        report_lines.extend(
            [
                "",
                "INJECTION RECOVERY AT COMMON FAP",
            ]
        )
        for _, row in summary.iterrows():
            report_lines.append(
                f"{row['method']}: "
                f"{row['harmonic_recovery_at_common_fap']:.3f}"
            )
    report_lines.extend(
        [
            "",
            "Interpretation:",
            "- thresholds are star-specific and score-specific;",
            "- both methods use the same null realization per star/trial;",
            "- chi2/odd-even/robust vetoes are TPS-like POC tests, not Kepler SOC tests;",
            "- native zero-injection means BATMAN was skipped entirely.",
        ]
    )
    report = "\n".join(report_lines) + "\n"
    (args.output_dir / "validation_summary.txt").write_text(report)
    print("\n" + report)
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
