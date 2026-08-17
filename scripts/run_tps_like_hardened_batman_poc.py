#!/usr/bin/env python3
"""Run event-consistency re-ranking on the existing TPS-like BATMAN POC.

This is intentionally an A/B extension of run_tps_like_batman_poc.py:
the original raw-MES result is preserved, while the top raw candidates are
re-ranked using repeated-event consistency measured in the injected light curve.

The event-consistency score is *not* called MES and is not treated as a Kepler
SOC robust statistic.  Its significance is established later by the common-FAP
calibration script.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import run_tps_like_batman_poc as base
from adaptive_transit.detection.tps_like_hardening import (
    DEFAULT_EVENT_CONSISTENCY_CONFIG,
    config_dict,
    harden_tps_like_result,
)
from adaptive_transit.injections.batman import inject_batman_transit
from adaptive_transit.preprocessing.normalization import preprocess_pdcsap_light_curve


DEFAULT_HARDENED_OUTPUT = (
    base.DEFAULT_PHYSICAL_POC / "tps_like_comparator_hardened_v3"
)


def _bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _raw_summary_from_row(row: pd.Series) -> dict:
    mapping = {
        "period_days": row.get("recovered_period_days", np.nan),
        "epoch_days": row.get("recovered_epoch_days", np.nan),
        "duration_hours": row.get("recovered_duration_hours", np.nan),
        "mes": row.get("mes", np.nan),
        "max_ses": row.get("max_ses", np.nan),
        "observed_event_count": row.get("observed_event_count", np.nan),
        "expected_event_count": row.get("expected_event_count", np.nan),
        "observability_fraction": row.get("observability_fraction", np.nan),
        "period_cadences": row.get("period_cadences", np.nan),
        "duration_cadences": row.get("duration_cadences", np.nan),
    }
    return mapping


def _ensure_raw_periodograms(row, cases, args):
    """Run the original search once, forcing periodogram persistence if needed."""

    raw_rows, noise = base.process_star(row, cases, args)
    target_id = base.normalize_target_id(row["target_id"])
    quarter = int(row["quarter"])
    star_dir = Path(args.output_dir) / "stars" / base.star_prefix(target_id, quarter)

    successful_case_ids = [
        int(r["case_index"])
        for _, r in raw_rows.iterrows()
        if _bool(r.get("success", False))
    ]
    missing = [
        case_index
        for case_index in successful_case_ids
        if not (star_dir / f"case_{case_index:02d}_periodogram.csv").exists()
    ]
    if missing:
        forced = copy.copy(args)
        forced.no_resume = True
        forced.save_periodograms = True
        raw_rows, noise = base.process_star(row, cases, forced)
    return raw_rows, noise


def harden_star(row, cases, args):
    target_id = base.normalize_target_id(row["target_id"])
    quarter = int(row["quarter"])
    stratum = str(row.get("sample_stratum", "unspecified"))
    star_dir = Path(args.output_dir) / "stars" / base.star_prefix(target_id, quarter)
    star_dir.mkdir(parents=True, exist_ok=True)

    raw_rows, noise = _ensure_raw_periodograms(row, cases, args)

    frame, _ = base.load_frame(target_id, quarter, args)
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

    rows = []
    for _, case in cases.iterrows():
        case_index = int(case["case_index"])
        raw_match = raw_rows.loc[
            pd.to_numeric(raw_rows["case_index"], errors="coerce") == case_index
        ]
        if raw_match.empty:
            rows.append(
                {
                    "target_id": target_id,
                    "quarter": quarter,
                    "sample_stratum": stratum,
                    "case_index": case_index,
                    "success": False,
                    "error": "Missing raw TPS-like result row.",
                }
            )
            continue

        raw_row = raw_match.iloc[0]
        if not _bool(raw_row.get("success", False)):
            failed = raw_row.to_dict()
            failed["pipeline"] = "tps_like_event_consistency_hardened"
            rows.append(failed)
            continue

        period = float(case["period_days"])
        duration_hours = float(case["duration_hours"])
        depth = float(case["depth"])
        phase = float(case["phase_fraction"])
        epoch = t_min + phase * period

        try:
            injected, _, _, _ = inject_batman_transit(
                time,
                flux,
                period_days=period,
                epoch_days=epoch,
                duration_days=duration_hours / 24.0,
                depth=depth,
                impact_parameter=args.impact_parameter,
                limb_darkening_coefficients=(args.limb_u1, args.limb_u2),
                supersample_factor=args.supersample_factor,
            )

            periodogram_path = star_dir / f"case_{case_index:02d}_periodogram.csv"
            periodogram = pd.read_csv(periodogram_path)
            hardened = harden_tps_like_result(
                {
                    "summary": _raw_summary_from_row(raw_row),
                    "periodogram": periodogram,
                },
                time,
                injected,
                config=DEFAULT_EVENT_CONSISTENCY_CONFIG,
            )
            summary = hardened["summary"]

            hard_error, hard_exact, hard_harmonic = base.period_match(
                summary["period_days"],
                period,
                args.period_match_tolerance_fraction,
            )
            raw_error, raw_exact, raw_harmonic = base.period_match(
                summary["raw_top1_period_days"],
                period,
                args.period_match_tolerance_fraction,
            )

            out = raw_row.to_dict()
            out.update(
                {
                    "pipeline": "tps_like_chi2_robust_hardened",
                    "detector": "tps_like_chi2_robust",
                    "success": True,
                    "error": "",
                    # Hardened selected ephemeris.
                    "recovered_period_days": float(summary["period_days"]),
                    "recovered_epoch_days": float(summary["epoch_days"]),
                    "recovered_duration_hours": float(summary["duration_hours"]),
                    "period_exact_fractional_error": float(hard_error),
                    "exact_period_recovered": bool(hard_exact),
                    "harmonic_period_recovered": bool(hard_harmonic),
                    # The raw MES of the selected candidate is still MES.
                    "mes": float(summary["selected_raw_mes"]),
                    "max_ses": float(summary["selected_max_ses"])
                    if np.isfinite(summary["selected_max_ses"])
                    else np.nan,
                    # New ranking statistic: do not label this MES.
                    "event_consistency_score": float(
                        summary["event_consistency_score"]
                    ),
                    "robust_veto_score": float(summary["robust_veto_score"]),
                    "consistency_weight": float(summary["consistency_weight"]),
                    "valid_event_count": int(summary["valid_event_count"]),
                    "expected_event_count_event_check": int(
                        summary["expected_event_count_event_check"]
                    ),
                    "event_observability_fraction": float(
                        summary["event_observability_fraction"]
                    ),
                    "positive_event_fraction": float(
                        summary["positive_event_fraction"]
                    ),
                    "single_event_fraction": float(
                        summary["single_event_fraction"]
                    ),
                    "anti_dominance_score": float(
                        summary["anti_dominance_score"]
                    ),
                    "combined_event_snr": summary["combined_event_snr"],
                    "leave_one_out_combined_snr_min": summary[
                        "leave_one_out_combined_snr_min"
                    ],
                    "leave_one_out_ratio": float(
                        summary["leave_one_out_ratio"]
                    ),
                    "leave_one_out_ratio_normalized": float(
                        summary["leave_one_out_ratio_normalized"]
                    ),
                    "median_event_depth": summary["median_event_depth"],
                    "event_depth_relative_mad": summary[
                        "event_depth_relative_mad"
                    ],
                    "event_depth_weighted_mean": summary[
                        "event_depth_weighted_mean"
                    ],
                    "event_depth_weighted_mean_error": summary[
                        "event_depth_weighted_mean_error"
                    ],
                    "event_depth_chi2": summary["event_depth_chi2"],
                    "event_depth_chi2_dof": summary["event_depth_chi2_dof"],
                    "event_depth_reduced_chi2": summary[
                        "event_depth_reduced_chi2"
                    ],
                    "event_depth_chi2_pvalue": summary[
                        "event_depth_chi2_pvalue"
                    ],
                    "odd_even_tested": summary["odd_even_tested"],
                    "odd_event_count": summary["odd_event_count"],
                    "even_event_count": summary["even_event_count"],
                    "odd_depth": summary["odd_depth"],
                    "even_depth": summary["even_depth"],
                    "odd_even_depth_difference": summary[
                        "odd_even_depth_difference"
                    ],
                    "odd_even_depth_difference_fraction": summary[
                        "odd_even_depth_difference_fraction"
                    ],
                    "odd_even_depth_z": summary["odd_even_depth_z"],
                    "odd_even_depth_pvalue": summary[
                        "odd_even_depth_pvalue"
                    ],
                    "robust_event_snr": summary["robust_event_snr"],
                    "robust_event_weight_min": summary[
                        "robust_event_weight_min"
                    ],
                    "robust_event_effective_count": summary[
                        "robust_event_effective_count"
                    ],
                    "event_consistent_flag": bool(
                        summary["event_consistent_flag"]
                    ),
                    "transit_consistency_veto_pass": bool(
                        summary["transit_consistency_veto_pass"]
                    ),
                    "diagnostic_consistency_veto_pass": bool(
                        summary["diagnostic_consistency_veto_pass"]
                    ),
                    "fails_min_events": bool(summary["fails_min_events"]),
                    "fails_observability": bool(summary["fails_observability"]),
                    "fails_positive_events": bool(
                        summary["fails_positive_events"]
                    ),
                    "fails_single_event_dominance": bool(
                        summary["fails_single_event_dominance"]
                    ),
                    "fails_leave_one_out": bool(summary["fails_leave_one_out"]),
                    "fails_depth_chi2": bool(summary["fails_depth_chi2"]),
                    "fails_odd_even": bool(summary["fails_odd_even"]),
                    "fails_robust_sign": bool(summary["fails_robust_sign"]),
                    "veto_reason": summary["veto_reason"],
                    "diagnostic_veto_reason": summary[
                        "diagnostic_veto_reason"
                    ],
                    "any_candidate_survives_veto": bool(
                        summary["any_candidate_survives_veto"]
                    ),
                    "selection_status": summary["selection_status"],
                    "raw_rank_of_selected_candidate": int(
                        summary["raw_rank_of_selected_candidate"]
                    ),
                    "ranking_changed": bool(summary["ranking_changed"]),
                    # Preserve the original top-1 explicitly for the A/B test.
                    "raw_top1_period_days": float(
                        summary["raw_top1_period_days"]
                    ),
                    "raw_top1_epoch_days": float(
                        summary["raw_top1_epoch_days"]
                    ),
                    "raw_top1_duration_hours": float(
                        summary["raw_top1_duration_hours"]
                    ),
                    "raw_top1_mes": float(summary["raw_top1_mes"]),
                    "raw_top1_max_ses": float(summary["raw_top1_max_ses"])
                    if np.isfinite(summary["raw_top1_max_ses"])
                    else np.nan,
                    "raw_top1_period_exact_fractional_error": float(raw_error),
                    "raw_top1_exact_period_recovered": bool(raw_exact),
                    "raw_top1_harmonic_period_recovered": bool(raw_harmonic),
                }
            )
            rows.append(out)

            hardened["ranking_table"].to_csv(
                star_dir / f"case_{case_index:02d}_hardened_ranking.csv",
                index=False,
            )
        except Exception as exc:
            failed = raw_row.to_dict()
            failed.update(
                {
                    "pipeline": "tps_like_chi2_robust_hardened",
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            rows.append(failed)

    out = pd.DataFrame(rows)
    out.to_csv(star_dir / "tps_like_hardened_results.csv", index=False)
    return out, noise


def write_summaries(results: pd.DataFrame, output_dir: Path):
    valid = results.loc[results["success"].map(_bool)].copy()
    if valid.empty:
        return

    rows = []
    rows.append(
        {
            "method": "tps_like_raw_top1",
            "n_cases": int(len(valid)),
            "exact_period_recovery": float(
                valid["raw_top1_exact_period_recovered"].mean()
            ),
            "harmonic_period_recovery": float(
                valid["raw_top1_harmonic_period_recovered"].mean()
            ),
            "median_score": float(valid["raw_top1_mes"].median()),
            "score_name": "raw_top1_mes",
        }
    )
    rows.append(
        {
            "method": "tps_like_chi2_robust_hardened",
            "n_cases": int(len(valid)),
            "exact_period_recovery": float(valid["exact_period_recovered"].mean()),
            "harmonic_period_recovery": float(
                valid["harmonic_period_recovered"].mean()
            ),
            "median_score": float(valid["robust_veto_score"].median()),
            "score_name": "robust_veto_score",
            "ranking_changed_fraction": float(valid["ranking_changed"].mean()),
            "event_consistent_flag_fraction": float(
                valid["event_consistent_flag"].mean()
            ),
            "transit_consistency_veto_pass_fraction": float(
                valid["transit_consistency_veto_pass"].mean()
            ),
        }
    )
    pd.DataFrame(rows).to_csv(
        output_dir / "summary_tps_like_hardening.csv", index=False
    )

    by_case = (
        valid.groupby(
            ["injected_period_days", "requested_duration_hours", "requested_depth"],
            dropna=False,
        )
        .agg(
            n_cases=("case_index", "size"),
            raw_harmonic_recovery=("raw_top1_harmonic_period_recovered", "mean"),
            hardened_harmonic_recovery=("harmonic_period_recovered", "mean"),
            ranking_changed_fraction=("ranking_changed", "mean"),
            median_raw_mes=("raw_top1_mes", "median"),
            median_event_consistency_score=("event_consistency_score", "median"),
            median_robust_veto_score=("robust_veto_score", "median"),
            veto_pass_fraction=("transit_consistency_veto_pass", "mean"),
            median_depth_chi2_pvalue=("event_depth_chi2_pvalue", "median"),
            median_odd_even_pvalue=("odd_even_depth_pvalue", "median"),
            median_robust_event_snr=("robust_event_snr", "median"),
            median_single_event_fraction=("single_event_fraction", "median"),
            median_leave_one_out_ratio=(
                "leave_one_out_ratio_normalized",
                "median",
            ),
        )
        .reset_index()
    )
    by_case.to_csv(
        output_dir / "summary_by_injection_regime_hardened.csv", index=False
    )

    valid.loc[valid["ranking_changed"].map(_bool)].to_csv(
        output_dir / "ranking_changes.csv", index=False
    )


def main(argv=None):
    args = base.parse_args(argv)
    if Path(args.output_dir) == Path(base.DEFAULT_OUTPUT):
        args.output_dir = DEFAULT_HARDENED_OUTPUT
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Candidate re-ranking needs candidate ephemerides, so keep periodograms.
    args.save_periodograms = True

    manifest = base.load_manifest(args.manifest_path, args.target_limit)
    cases = base.load_cases(args.case_file, args.case_limit)
    manifest.to_csv(args.output_dir / "manifest_used.csv", index=False)
    cases.to_csv(args.output_dir / "cases_used.csv", index=False)

    config = vars(args).copy()
    config = {
        key: (
            str(value)
            if isinstance(value, Path)
            else list(value)
            if isinstance(value, tuple)
            else value
        )
        for key, value in config.items()
    }
    config.update(
        {
            "method_label": "tps_like_event_consistency_hardened",
            "scientific_status": "POC_chi2_robust_veto_not_full_Kepler_TPS_pending_FAP",
            "event_consistency": config_dict(DEFAULT_EVENT_CONSISTENCY_CONFIG),
            "important_note": (
                "chi2/odd-even/robust vetoes are POC approximations; the "
                "headline robust_veto_score must be judged only after empirical "
                "null calibration and must not be called Kepler SOC TPS"
            ),
        }
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n"
    )

    frames = []
    noise_rows = []
    for row in tqdm(
        manifest.to_dict(orient="records"),
        desc="TPS-like event-consistency hardening",
    ):
        try:
            result, noise = harden_star(row, cases, args)
            frames.append(result)
            noise_rows.append(
                {key: value for key, value in noise.items() if key != "preprocessing"}
            )
        except Exception as exc:
            noise_rows.append(
                {
                    "target_id": base.normalize_target_id(row["target_id"]),
                    "quarter": int(row["quarter"]),
                    "sample_stratum": str(
                        row.get("sample_stratum", "unspecified")
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                args.output_dir / "tps_like_hardened_results.csv", index=False
            )
        pd.DataFrame(noise_rows).to_csv(
            args.output_dir / "tps_like_star_noise_models.csv", index=False
        )

    results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not results.empty:
        write_summaries(results, args.output_dir)

    print("\nTPS-like event-consistency hardening complete.")
    print(f"Result rows: {len(results)}")
    if not results.empty:
        valid = results.loc[results["success"].map(_bool)]
        print(f"Successful rows: {len(valid)}")
        if not valid.empty:
            print(
                "Raw harmonic recovery: "
                f"{valid['raw_top1_harmonic_period_recovered'].mean():.3f}"
            )
            print(
                "Hardened harmonic recovery: "
                f"{valid['harmonic_period_recovered'].mean():.3f}"
            )
            print(
                "Ranking changed: "
                f"{valid['ranking_changed'].mean():.3f}"
            )
    print(f"Output: {args.output_dir}")
    print(
        "\nNext: run calibrate_tps_like_common_fap.py.  Do not interpret "
        "robust_veto_score as significant before that calibration."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
