# Gaussian Process Background Baseline

Status: completed single-target challenger branch for KIC 11904151, Quarter 5.

This branch adds Gaussian Process regression as a background-noise challenger to ARIMA and Kalman. It does not replace raw BLS, ARIMA-TCF, Kalman, the injection grid, null generation, or the frozen reranker.

## Model

The first GP baseline estimates a smooth latent background function:

```text
background(t) ~ GP(mean, covariance)
normalized_flux_t = background(t) + measurement_noise_t
```

The covariance is intentionally simple:

```text
constant_amplitude * RBF(time_length_scale)
```

The RBF length scale controls how quickly the background can vary. The lower length-scale bound is set longer than the transit durations in the 81-case benchmark, so the GP is discouraged from fitting individual transit dips.

Measurement noise is represented by a fixed nugget variance in the GP fit. Larger nugget variance makes the background smoother and leaves more short-timescale structure in the residuals.

The detector input is:

```text
residual_t = normalized_flux_t - GP_posterior_mean(time_t)
```

Residuals may contain transit evidence if the GP background captures long-timescale stellar or instrumental variability while leaving short-duration dips behind.

## Anchor-Point Approximation

Exact GP fitting over every Kepler cadence for every injection would be cubic in the number of cadences and is not practical for the repeated null and injection workflows.

The baseline therefore fits the GP on a deterministic, evenly spaced subset of finite observed cadences and predicts the background on the full time grid.

This is a computational approximation, not a new scientific detector. The saved model summaries explicitly record:

```text
anchor_point_approximation = true
training_point_count
max_train_points
length_scale_days
measurement_noise_variance
```

## Missing Cadences

Missing or unusable flux values are not used for fitting and do not produce residual values.

If a row has finite time but missing flux, the GP can produce a background prediction at that time, but the residual remains `NaN`.

If the time value itself is missing, the GP does not predict the background at that row.

No flux values are silently interpolated before detector input.

## Difference From ARIMA and Kalman

ARIMA models linear serial dependence in a discrete cadence sequence and emits one-step-ahead innovations.

Kalman estimates an explicit latent state with process and measurement noise and emits one-step residuals.

The GP branch estimates a smooth covariance-based background directly over time. It is two-sided: the posterior mean at one cadence can depend on observations before and after that cadence.

This means GP residuals are not causal forecasting innovations. They are background-subtracted residuals for detector comparison.

## Current Experiment Flow

```text
Kepler PDCSAP
-> existing preprocessing
-> normalized flux
-> smooth anchor-point GP background
-> residuals
├── BLS
└── TCF
```

The first scripts use the same single-target injection philosophy as Kalman:

```text
KIC 11904151
Kepler Quarter 5
periods: 2, 5, 10 days
durations: 2, 4, 8 hours
depths: 200, 500, 1000 ppm
epoch phases: 0.15, 0.45, 0.75
```

## Parallelism

The expensive null and injection-grid loops run with process-level parallelism.

The scripts also cap common BLAS/OpenMP thread pools to one thread per worker:

```text
OMP_NUM_THREADS
OPENBLAS_NUM_THREADS
MKL_NUM_THREADS
NUMEXPR_NUM_THREADS
```

This avoids oversubscribing the CPU when several detector workers are running at once.

Both long-running scripts checkpoint row-level results to their final CSV paths as each worker result finishes. If a run is interrupted, rerunning the same command resumes completed trials or injection cases by default. Use `--no-resume` only when intentionally replacing an existing partial result.

## Commands

```bash
python scripts/run_gp_baseline.py
python scripts/run_gp_null_calibration.py
python scripts/run_gp_injection_grid.py
python scripts/compare_gp_recovery_overlap.py
python scripts/run_gp_timescale_sensitivity.py
python scripts/run_gp_multiduration_timescale_sensitivity.py
python scripts/analyze_gp_remaining_misses.py
```

Run the null calibration before the injection grid because the injection grid reads the GP-specific 1% FAP thresholds.

The default null calibration uses 1,000 surrogate trials and the default injection grid uses all 81 cases. Worker count can be controlled without changing source:

```bash
python scripts/run_gp_null_calibration.py --n-jobs 8
python scripts/run_gp_injection_grid.py --n-jobs 8
```

The overlap comparison should be run after the GP injection grid. It joins raw BLS, existing TCF, Kalman, and GP injection rows using the exact injection parameter tuple.

The time-scale sensitivity script is a mechanism test, not a new calibrated recovery run. It varies the GP RBF length scale relative to one fixed injected transit duration and reports transit retention, residual diagnostics, and detector scores without applying configuration-specific FAP thresholds.

The multi-duration sensitivity script repeats that mechanism test for 2-hour, 4-hour, and 8-hour transits and aggregates by the dimensionless ratio `ell_GP / transit_duration`. The remaining-miss analyzer reads the saved overlap table and does not rerun detectors.

## Outputs

```text
outputs/experiments/gp_baseline/
outputs/experiments/gp_null_calibration/
outputs/experiments/gp_injection_grid/
outputs/experiments/gp_recovery_overlap/
outputs/experiments/gp_timescale_sensitivity/
outputs/experiments/gp_multiduration_timescale_sensitivity/
```

The baseline saves model diagnostics, residual parquet files, BLS periodograms, TCF periodograms, and top peaks.

The null calibration saves detector-specific GP-BLS and GP-TCF threshold tables.

The injection grid saves per-injection recovery rows and grouped recovery summaries by depth, duration, and period.

The overlap comparison saves pairwise and union recovery summaries against raw BLS, existing TCF, Kalman-BLS, and Kalman-TCF.

The time-scale sensitivity experiment saves a preservation-sorted table, a ratio-sorted table, transit-window samples, and a diagnostic plot comparing injected flux, GP background, and GP residuals around selected transit windows.

The multi-duration time-scale experiment saves the full duration/ratio table, an aggregate-by-ratio table, a JSON summary, and a ratio-collapse plot. The remaining-miss analyzer saves the 4 all-method misses and binned summaries by depth, duration, and period.

## Completed KIC 11904151 Quarter 5 Result

The completed null calibration used 1,000 successful surrogate trials:

| Pipeline | 1% FAP threshold |
| -------- | ---------------: |
| GP -> BLS |         0.001683 |
| GP -> TCF |        15.774957 |

On the matched 81-injection grid:

| Metric | Result |
| ------ | -----: |
| GP -> BLS harmonic-aware recovery at 1% FAP | 73/81 = 90.1% |
| GP -> TCF harmonic-aware recovery at 1% FAP | 58/81 = 71.6% |
| GP -> BLS exact recovery at 1% FAP | 53/81 = 65.4% |
| GP -> TCF exact recovery at 1% FAP | 23/81 = 28.4% |
| Median depth retention | 85.1% |
| Median SNR retention | 153.6% |

The overlap analysis gives:

| Combination | Recovery |
| ----------- | -------: |
| Raw BLS | 67/81 = 82.7% |
| Existing TCF | 64/81 = 79.0% |
| Kalman-TCF | 67/81 = 82.7% |
| Existing BLS union TCF | 72/81 = 88.9% |
| Existing BLS union TCF union GP-TCF | 75/81 = 92.6% |
| All six methods | 77/81 = 95.1% |

GP-TCF contributes 3 unique recoveries beyond the existing non-GP methods. GP-BLS contributes 2 unique recoveries beyond the existing non-GP methods. The GP-TCF unique recoveries are all shallow 2-day, 2-hour, 200 ppm injections.

The GP-BLS result is the strongest standalone result on this single-target 81-injection benchmark. This should not be generalized beyond this benchmark until the same branch is tested on more stars and broader injection distributions.

## Time-Scale-Separation Hypothesis

The current working mechanism is:

```text
transit duration << GP background time scale
```

This should be treated as a hypothesis supported by the current outputs, not as a demonstrated universal mechanism. The controlled sensitivity script varies only the GP background responsiveness for the same injected transit.

For the default KIC 11904151 Quarter 5 5-day, 4-hour, 1000 ppm injection:

| GP length scale / transit duration | Background absorption | Depth retention | SNR retention | BLS period | TCF period |
| ---------------------------------: | --------------------: | --------------: | ------------: | ---------: | ---------: |
| 0.5 | 62.8% | 35.5% | 69.0% | 4.994 d | 5.000 d |
| 1.0 | 59.0% | 39.4% | 68.1% | 4.994 d | 5.000 d |
| 2.0 | 34.0% | 63.4% | 103.1% | 4.994 d | 2.501 d |
| 5.0 | 15.5% | 85.9% | 129.4% | 4.994 d | 2.501 d |
| 10.0 | 8.7% | 95.7% | 146.6% | 4.994 d | 2.501 d |
| 20.0 | 2.3% | 101.3% | 185.2% | 4.994 d | 2.501 d |

The pattern is consistent with short GP length scales absorbing transit morphology and longer GP length scales preserving short dips. The high-retention configurations also retain substantial residual autocorrelation, so the scientific trade-off is not simply maximization of smoothness or residual whiteness.

The multi-duration test repeats the same fixed-ratio experiment for 2-hour, 4-hour, and 8-hour transits. Across 18 fixed-length-scale configurations:

```text
Spearman ratio vs depth retention:        +0.825
Spearman ratio vs SNR retention:          +0.555
Spearman ratio vs background absorption:  -0.837
Spearman ratio vs residual ACF:           +0.724
```

Median behavior by ratio:

| ell_GP / transit duration | Background absorption | Depth retention | SNR retention | Max residual ACF 1-24 |
| ------------------------: | --------------------: | --------------: | ------------: | --------------------: |
| 0.5 | 62.8% | 35.5% | 69.0% | 0.628 |
| 1.0 | 59.0% | 39.4% | 68.1% | 0.649 |
| 2.0 | 34.0% | 63.4% | 103.1% | 0.715 |
| 5.0 | 15.4% | 86.7% | 129.4% | 0.771 |
| 10.0 | 4.8% | 95.8% | 162.5% | 0.790 |
| 20.0 | 1.7% | 100.1% | 172.9% | 0.803 |

The curves are not perfectly duration-invariant, especially at low ratios, but the direction is consistent across the tested durations. This strengthens the hypothesis that time-scale separation is the controlling quantity.

## Remaining Misses

After combining raw BLS, existing TCF, Kalman-BLS, Kalman-TCF, GP-BLS, and GP-TCF, 4 of 81 injections remain missed.

Those misses are concentrated in one regime:

```text
period: 10 days
depth: 200 ppm
duration: 2 hours for 3 cases, 4 hours for 1 case
```

Their median GP depth retention is 85.5% and median GP SNR retention is 166.4%. That means the remaining failures are better interpreted as shallow long-period candidate-generation or scoring failures, not GP transit erasure.

## Scientific Caution

GPs can fit away transits if the kernel is too flexible. A high likelihood or low residual variance is not evidence of superiority if depth and SNR retention collapse.

The relevant comparison is transit-search suitability:

```text
residual whitening
variance stability
transit depth retention
transit SNR retention
period recovery
recovery at calibrated FAP
unique recoveries beyond existing BLS, TCF, and Kalman unions
```

Do not claim GP is better until the null-calibrated injection-recovery and overlap outputs support it.
