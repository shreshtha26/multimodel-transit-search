#!/usr/bin/env python3
"""Analyze the TPS-like true zero-injection native-background control."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_transit.tps_null_audit import (
    build_zero_injection_table,
    compare_zero_to_injected,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARATOR = (
    PROJECT_ROOT
    / "outputs/experiments/batman_physical_detection_poc/pilot10/tps_like_comparator"
)
DEFAULT_RAW_NULL = DEFAULT_COMPARATOR / "tps_like_zero_injection_raw/tps_like_results.csv"
DEFAULT_INJECTED = DEFAULT_COMPARATOR / "tps_like_results.csv"
DEFAULT_OUTPUT = DEFAULT_COMPARATOR / "zero_injection_audit"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-null-path", type=Path, default=DEFAULT_RAW_NULL)
    parser.add_argument("--injected-path", type=Path, default=DEFAULT_INJECTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--period-tolerance-fraction", type=float, default=0.02)
    parser.add_argument("--max-realized-depth", type=float, default=1e-12)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_null = pd.read_csv(args.raw_null_path)
    injected = pd.read_csv(args.injected_path)

    zero = build_zero_injection_table(
        raw_null,
        max_realized_depth=args.max_realized_depth,
    )
    cases, stars = compare_zero_to_injected(
        zero,
        injected,
        tolerance_fraction=args.period_tolerance_fraction,
    )

    zero_path = args.output_dir / "tps_like_zero_injection.csv"
    case_path = args.output_dir / "tps_like_null_vs_injected_cases.csv"
    star_path = args.output_dir / "tps_like_null_vs_star_summary.csv"
    summary_path = args.output_dir / "tps_like_zero_injection_summary.txt"

    zero.to_csv(zero_path, index=False)
    cases.to_csv(case_path, index=False)
    stars.to_csv(star_path, index=False)

    persistent = stars.loc[stars["persistent_injected_period_flag"]].copy()
    n_persistent = int(len(persistent))
    n_persistent_null_match = int(persistent["null_matches_persistent_period"].sum())
    all_case_null_match = float(cases["winner_matches_null_period"].mean())
    max_realized = float(
        pd.to_numeric(zero["zero_injection_realized_depth"], errors="coerce").abs().max()
    )

    lines = [
        "TPS-LIKE TRUE ZERO-INJECTION / NATIVE-BACKGROUND CONTROL",
        "=" * 72,
        "",
        "Scope:",
        "  Same 10-star preprocessing, wavelet-noise model, and TPS-like search",
        "  settings as the BATMAN comparator, but BATMAN is skipped entirely and",
        "  the original preprocessed stellar flux is searched. The audit also",
        "  verifies the explicit zero-injection flag and zero realized depth.",
        "  This is a native-background diagnostic, not a randomized FAP calibration.",
        "",
        f"Stars: {len(zero)}",
        f"Successful injected comparison cases: {len(cases)}",
        f"Maximum realized injected depth in control: {max_realized:.3e}",
        f"Period-match tolerance: {args.period_tolerance_fraction:.3f}",
        f"Injected winners matching each star's null winner: {all_case_null_match:.3f}",
        f"Persistent injected-period stars: {n_persistent}/{len(stars)}",
        (
            "Persistent stars whose persistent period matches the null winner: "
            f"{n_persistent_null_match}/{n_persistent}"
            if n_persistent
            else "Persistent stars whose persistent period matches the null winner: 0/0"
        ),
        "",
        "Per-star comparison:",
    ]

    for row in stars.itertuples(index=False):
        lines.append(
            "  KIC {target}: null={null:.6f} d, persistent={persistent:.6f} d, "
            "persistent_flag={flag}, null_match={match}, "
            "injected_winners_matching_null={fraction:.3f}, "
            "null_MES={null_mes:.3f}, median_injected_MES={inj_mes:.3f}, "
            "median_delta_MES={delta:.3f}".format(
                target=row.target_id,
                null=row.null_recovered_period_days,
                persistent=row.persistent_injected_period_days,
                flag=bool(row.persistent_injected_period_flag),
                match=bool(row.null_matches_persistent_period),
                fraction=row.fraction_injected_winners_matching_null_period,
                null_mes=row.null_mes,
                inj_mes=row.median_injected_mes,
                delta=row.median_injected_minus_null_mes,
            )
        )

    lines.extend(
        [
            "",
            "Interpretation guardrail:",
            "  A null-period match supports the interpretation that the injected",
            "  search is being dominated by a star-specific/background winner.",
            "  It is still not a detection veto and does not establish a 1% FAP",
            "  threshold. Candidate-ranking changes must wait for null/FAP",
            "  validation and event-consistency testing.",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines))

    print("\nTPS-like true zero-injection control audit complete.")
    print(f"Stars: {len(zero)}")
    print(f"Injected comparison cases: {len(cases)}")
    print(f"Maximum realized control depth: {max_realized:.3e}")
    print(f"Injected winners matching null winner: {all_case_null_match:.3f}")
    print(
        "Persistent-period stars matching null winner: "
        f"{n_persistent_null_match}/{n_persistent}"
    )
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
