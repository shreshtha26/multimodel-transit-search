# Multi-Star Challenger Benchmark

## Active Scientific Benchmark

Active scientific runs now use the unified long-format adaptive-transit runner:

```text
Core pipelines: raw/arima/kalman/gp x bls/tcf/tps_like
```

TLS and trapezoid remain available as challenger detectors, but they are not active by default.

```bash
python scripts/run_adaptive_transit_benchmark.py --profile benchmark100 --calibrate-fap --n-null-trials-per-star 1000
python scripts/run_adaptive_transit_benchmark.py --profile benchmark100 --thresholds-path outputs/experiments/adaptive_transit/benchmark100/fap_thresholds.csv
```

The final population-scale run uses the same runner and differs by manifest and scale:

```bash
python scripts/run_adaptive_transit_benchmark.py --profile benchmark1000 --calibrate-fap --n-null-trials-per-star 1000
python scripts/run_adaptive_transit_benchmark.py --profile benchmark1000 --thresholds-path outputs/experiments/adaptive_transit/benchmark1000/fap_thresholds.csv
```

The scientific common-FAP setting is 1000 moving-block null trials per star. Smaller null counts, including 100-null runs, are engineering tests only and must not be used for final 1% FAP claims.

The legacy wide-table challenger runner remains for historical reproduction, but `pilot`, `main`, and 50-star profiles are no longer active scientific benchmark profiles.

## Legacy Wide-Table Runner

This benchmark extends the single-target raw, ARIMA, Kalman, and GP branches across multiple Kepler Quarter 5 stellar backgrounds.

The scientific purpose is to test whether cheap stellar-background properties predict which preprocessing/detector pipeline wins.

## Runner

```bash
python scripts/run_multistar_challenger_benchmark.py --profile pilot
python scripts/run_multistar_challenger_benchmark.py --profile main
```

Profiles:

| Profile | Targets | Injection grid | Purpose |
| ------- | ------: | -------------- | ------- |
| `smoke` | 2 | 1 case/star | validate multiprocessing and outputs |
| `pilot` | 10 | reduced grid | validate runtime and cross-pipeline joins |
| `main` | 50 | 81 cases/star | full multi-background benchmark |

Default pipelines:

```text
raw_bls
arima_tcf
kalman_bls
kalman_tcf
gp_bls
gp_tcf
```

The pilot profile uses the background time-scale audit table when available to sample across quiet, high-scatter, long-ACF, and gap-heavy targets.

## Parallelism

The runner parallelizes across stars with process workers:

```bash
python scripts/run_multistar_challenger_benchmark.py --profile pilot --max-workers 6
```

Each worker caps numerical-library thread pools to one thread so multiple star workers do not oversubscribe the machine.

Per-star checkpoints are saved under:

```text
outputs/experiments/multistar_challenger_benchmark/<profile>/stars/kic_<id>_q<quarter>/
```

The final per-star checkpoint is:

```text
COMPLETE
```

Rerunning the same command resumes compatible completed stars and partial `injections.csv` files.

## Per-Star FAP Calibration

The rank-1 injection benchmark is followed by per-star calibration:

```bash
python scripts/calibrate_multistar_challenger_benchmark.py --profile pilot --n-null-trials-per-star 10 --max-workers 6
```

The calibration script uses moving-block surrogates from each branch's fitted series:

| Pipeline | Null source |
| -------- | ----------- |
| `raw_bls` | normalized flux |
| `arima_tcf` | ARIMA innovations |
| `kalman_bls` | Kalman residuals |
| `kalman_tcf` | Kalman residuals |
| `gp_bls` | GP residuals |
| `gp_tcf` | GP residuals |

Thresholds are estimated separately for each star and pipeline. The calibration therefore compares performance at the same nominal false-alarm probability rather than at the same numerical detector score.

The currently stored pilot calibration uses only 10 null trials per star:

```text
10 stars × 10 null trials/star = 100 null rows
```

This is enough to validate the output contract, but too small for a final 1% tail estimate. Treat these as 10-null engineering-calibration results only.

The convergence analyzer recomputes thresholds and labels at every complete null-count level available:

```bash
python scripts/analyze_multistar_calibration_convergence.py --profile pilot
```

It uses nested prefixes of the same stored null sequence:

```text
10-null estimate   = trials 0-9
50-null estimate   = trials 0-49
100-null estimate  = trials 0-99
250-null estimate  = trials 0-249
500-null estimate  = trials 0-499
1000-null estimate = trials 0-999
```

It writes threshold movement, bootstrap threshold uncertainty, recovery-label changes, union-label changes, and unique-recovery-label changes for:

```text
10
50
100
250
500
1000
```

Only levels present for every calibrated star are included. After each longer calibration run, rerun the analyzer.

The default bootstrap interval uses 500 resamples of the stored null scores. This is a finite-null uncertainty diagnostic, not a replacement for enough null trials.

Incremental convergence sequence:

```bash
python scripts/calibrate_multistar_challenger_benchmark.py --profile pilot --n-null-trials-per-star 50 --max-workers 6 --no-download
python scripts/analyze_multistar_calibration_convergence.py --profile pilot
python scripts/calibrate_multistar_challenger_benchmark.py --profile pilot --n-null-trials-per-star 100 --max-workers 6 --no-download
python scripts/analyze_multistar_calibration_convergence.py --profile pilot
python scripts/calibrate_multistar_challenger_benchmark.py --profile pilot --n-null-trials-per-star 250 --max-workers 6 --no-download
python scripts/analyze_multistar_calibration_convergence.py --profile pilot
python scripts/calibrate_multistar_challenger_benchmark.py --profile pilot --n-null-trials-per-star 500 --max-workers 6 --no-download
python scripts/analyze_multistar_calibration_convergence.py --profile pilot
python scripts/calibrate_multistar_challenger_benchmark.py --profile pilot --n-null-trials-per-star 1000 --max-workers 6 --no-download
python scripts/analyze_multistar_calibration_convergence.py --profile pilot
```

## Outputs

Global metrics are written to:

```text
outputs/experiments/multistar_challenger_benchmark/<profile>/metrics/
```

Important files:

```text
target_manifest_used.csv
target_execution_status.csv
multistar_challenger_injections.csv
multistar_challenger_star_summary.csv
multistar_challenger_pipeline_summary.csv
multistar_challenger_pairwise_overlap.csv
multistar_challenger_combinations.csv
multistar_challenger_by_depth.csv
multistar_challenger_by_duration.csv
multistar_challenger_by_period.csv
multistar_challenger_by_star.csv
multistar_challenger_summary.json
```

After calibration, additional files are written:

```text
multistar_challenger_star_null_trials.csv
multistar_challenger_star_fap_thresholds.csv
multistar_challenger_master_results.csv
multistar_challenger_star_fap_pipeline_summary.csv
multistar_challenger_star_fap_combinations.csv
multistar_challenger_star_fap_by_depth.csv
multistar_challenger_star_fap_by_duration.csv
multistar_challenger_star_fap_by_period.csv
multistar_challenger_star_fap_by_background_ratio.csv
multistar_challenger_star_fap_summary.json
multistar_challenger_calibration_convergence_thresholds.csv
multistar_challenger_calibration_convergence_labels.csv
multistar_challenger_calibration_convergence_pipeline_summary.csv
multistar_challenger_calibration_convergence_label_changes.csv
multistar_challenger_calibration_convergence_union_summary.csv
multistar_challenger_calibration_convergence_union_changes.csv
multistar_challenger_calibration_convergence_unique_summary.csv
multistar_challenger_calibration_convergence_unique_changes.csv
multistar_challenger_calibration_convergence_trial_prefix.csv
multistar_challenger_calibration_convergence_summary.json
```

## Scientific Caveat

The first runner reports rank-1 injection recovery and overlap. The calibration runner adds star-level FAP columns to the master table.

The 10-null pilot calibration should not be used as a final false-alarm estimate. The 50-star benchmark should use a much larger null count per star or another explicitly justified star/regime-level calibration design.
