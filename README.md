# multimodel-transit-search

`multimodel-transit-search` is a reproducible research prototype for benchmarking complementary approaches to exoplanet transit detection in Kepler light curves.

The current implementation contains four detector/background branches and a candidate-level machine-learning reranker:

```text
Kepler PDCSAP flux
│
├── statistical characterization
│   ├── sampling/gap diagnostics
│   ├── ADF/KPSS stationarity tests
│   ├── ACF/Ljung-Box temporal-dependence metrics
│   ├── rolling mean/variance drift
│   ├── Lomb-Scargle spectral structure
│   └── distribution and quality-flag summaries
│
├── direct normalized-flux branch
│   └── Box Least Squares (BLS)
│
├── autoregressive-transformation branch
│   ├── ARIMA diagnostics and model selection
│   ├── one-step-ahead innovations
│   └── periodic Transit Comb Filter (TCF)
│
├── state-space challenger branch
│   ├── local-level Kalman background estimate
│   ├── one-step residuals
│   ├── residual BLS
│   └── residual TCF
│
└── Gaussian Process challenger branch
    ├── smooth anchor-point GP background estimate
    ├── background-subtracted residuals
    ├── residual BLS
    └── residual TCF
             │
             ▼
      detector candidate sets
             │
             ▼
      BLS / TCF candidate merge
             │
             ▼
   leakage-audited candidate reranker
             │
             ├── logistic regression
             ├── XGBoost classifier
             └── XGBoost pairwise ranker
             │
             ▼
      frozen clean_reranker_v1
             │
             ▼
   top-k null/original calibration
```

The longer-term goal is to benchmark statistical, probabilistic, machine-learning, and deep-learning approaches at controlled false-alarm rates, characterize where each method succeeds or fails, and eventually combine complementary evidence in a calibrated adaptive ensemble.

> **Research status:** active prototype. The repository contains a single-target methodological benchmark and a 50-star Kepler Quarter 5 pilot. The candidate reranker is frozen as `clean_reranker_v1`, but neither the detector pipeline nor the reranker should yet be interpreted as a validated astrophysical transit-search system or catalog generator.

## Research Questions

The current work is organized around seven questions:

1. Can an ARIMA-family model reduce predictable correlated variability while retaining transit information in a detectable form?
2. How sensitive is ARIMA selection and transit preservation to the treatment of missing Kepler cadences?
3. How does direct BLS detection compare with TCF detection on ARIMA-transformed light curves?
4. Can a simple Kalman/state-space background model preserve transit evidence differently from the current ARIMA branch?
5. Can a smooth Gaussian Process background model remove correlated variability while preserving transit morphology better than ARIMA or Kalman?
6. Do BLS and TCF provide complementary candidate information across multiple stars and noise regimes?
7. Can a leakage-audited candidate reranker combine those candidates while generalizing to previously unseen targets and maintaining an empirically calibrated false-alarm threshold?

## Experimental Design

### Single-target benchmark

The original methodological benchmark uses:

| Item                         | Value                           |
| ---------------------------- | ------------------------------- |
| Target                       | KIC 11904151                    |
| Quarter                      | Kepler Quarter 5                |
| Flux product                 | PDCSAP flux                     |
| Quality policy               | Lightkurve default quality mask |
| Usable observations          | 4,486                           |
| Regular cadence-grid length  | 4,634                           |
| Explicit missing cadences    | 148                             |
| Injection periods            | 2, 5, 10 days                   |
| Injection durations          | 2, 4, 8 hours                   |
| Injection depths             | 200, 500, 1,000 ppm             |
| Epoch phases                 | 0.15, 0.45, 0.75                |
| Injection cases per detector | 81                              |

Synthetic box-shaped transits are injected into an observed Kepler PDCSAP light curve.

The benchmark is therefore:

```text
real Kepler background variability and measurement noise
+
synthetic known transit signal
```

rather than a fully simulated light curve.

This allows detection performance to be evaluated against a known injected truth while retaining realistic stellar and instrumental structure.

### 50-star pilot

The current multi-star benchmark uses:

| Item                     | Value                         |
| ------------------------ | ----------------------------- |
| Targets                  | 50 Kepler target-quarter rows |
| Quarter                  | Kepler Quarter 5              |
| Flux product             | PDCSAP flux                   |
| Injection periods        | 2, 5, 10 days                 |
| Injection durations      | 2, 4, 8 hours                 |
| Injection depths         | 500, 1,000 ppm                |
| Epoch phases             | 0.45                          |
| Injections               | 900 total                     |
| Injections per target    | 18                            |
| Null trials              | 400 total                     |
| Null trials per target   | 8                             |
| Detector FAP target      | 1%                            |
| Candidate reranker       | `clean_reranker_v1`           |
| Reranker split unit      | `target_id`                   |
| Frozen reranker features | 44                            |

The 50-star run completed successfully for all 50 requested targets.

> **Cohort clarification:** `configs/kepler_50_star_manifest.csv` is a legacy engineering/stress-test cohort (one reference target plus rows labelled primarily as `confirmed_planet_host`), not a catalog-clean background sample. Weak injected signals in that cohort can compete with pre-existing astrophysical transit/eclipse signals. Its aggregate injection-recovery rates therefore must not be presented as clean-background recovery rates. The manifest and historical outputs are retained unchanged for reproducibility.

A separate catalog-clean workflow is now defined in [`docs/clean_injection_benchmark.md`](docs/clean_injection_benchmark.md). It excludes cataloged KOI/TCE/confirmed-planet/EB associations before light-curve characterization, freezes a separate `configs/kepler_clean_background_manifest.csv`, and keeps known-signal systems for dedicated positive-control/stress-test analyses.

The reduced multi-star injection grid is an engineering pilot designed to test generalization, calibration, and candidate combination. It is not intended to approximate the full Kepler planet population.

## Data and Preprocessing

The current preprocessing pipeline includes:

* Kepler PDCSAP loading with `lightkurve`;
* configurable Kepler quality-mask policies;
* regular cadence-grid reconstruction;
* explicit observed, usable, missing, and interpolated-cadence masks;
* chronological normalization intended to avoid future-data leakage;
* cached light curves for multi-star experiments;
* machine-readable CSV, JSON, Parquet, and PNG outputs.

## Gap Representations

ARIMA diagnostics are evaluated under three representations of the same light curve.

### `longest_contiguous`

Uses the longest uninterrupted usable segment.

Advantages:

* conventional cadence-lag ACF/PACF interpretation is easier;
* no flux values are invented across gaps.

Limitation:

* a substantial fraction of the quarter can be discarded.

### `full_gap` / `full_grid_missing`

Preserves the complete regular cadence grid and represents missing observations as `NaN`.

This is the default full-light-curve representation.

The name `full_gap` appears in earlier single-target outputs, while the gap-comparison workflow uses `full_grid_missing` for the equivalent representation.

### `interpolated_full_grid`

Interpolates only eligible short interior gaps.

This is treated as a challenger representation, not as ground truth.

A better forecast metric under interpolation is not by itself evidence that the representation is scientifically preferable.

## ARIMA Branch

For observed normalized flux (y_t), the one-step-ahead innovation is conceptually

```text
e_t = y_t - y_hat(t | t - 1)
```

where `y_hat(t | t - 1)` is the prediction made using information available before cadence `t`.

The intended interpretation is:

```text
predictable stellar/instrumental variation
            ↓
       ARIMA prediction

unpredicted variation + model error + possible transit evidence
            ↓
          innovation
```

ARIMA is therefore used as an intermediate signal transformation rather than as the final scientific forecasting product.

### Implemented ARIMA diagnostics

The branch includes:

* configurable `(p, d, q)` candidate grids;
* convergence checks;
* numerical-validity checks;
* coefficient-boundary and stability checks;
* ADF stationarity testing;
* KPSS stationarity testing;
* explicit differencing diagnostics;
* AIC and BIC;
* RMSE and MAE;
* negative log-score diagnostics;
* baseline forecasting comparisons;
* residual autocorrelation analysis;
* Ljung-Box tests;
* rolling-variance diagnostics;
* ARCH-style variance diagnostics;
* chronological and segment stability checks;
* transit-preservation experiments;
* transformed-template matching;
* gap-representation experiments.

### ARIMA selection policy

Candidates are not selected solely because they minimize AIC, BIC, RMSE, or MAE.

The intended hierarchy is approximately:

```text
fit and numerical validity
→ residual whitening
→ variance stability
→ transit preservation
→ transit distortion
→ baseline-relative forecasting
→ model complexity
→ forecast metrics
→ information criteria
```

A model with attractive AIC/BIC or forecast error is therefore not scientifically preferred if it is unstable, non-converged, leaves strong residual dependence, or destroys transit information.

### Current ARIMA status

For the default full-grid missing-data representation, the current best-available candidate is:

```text
ARIMA(1,1,0)
```

This should **not** be interpreted as a validated optimal noise model.

Important unresolved issues include:

* ADF and KPSS provide conflicting evidence about stationarity;
* the physical/statistical justification for `d=1` remains unresolved;
* residual autocorrelation remains;
* residual variance is unstable;
* the selected order is sensitive to gap representation;
* direct box-shaped transit preservation is weak;
* ARIMA convergence remains problematic in subsequent TCF experiments.

The ARIMA work should therefore currently be interpreted as a completed methodological branch with unresolved scientific adequacy, not as evidence that `ARIMA(1,1,0)` is the correct Kepler noise model.

## BLS Branch

The direct detector branch applies Astropy Box Least Squares to normalized PDCSAP flux.

Implemented functionality includes:

* period-grid search;
* duration-grid search;
* periodic box-transit injection;
* top-k candidate extraction;
* moving-block null surrogates;
* empirical false-alarm calibration;
* harmonic-aware period matching;
* injection-recovery analysis by period, duration, and depth.

### Single-target BLS benchmark

Across the 81-case injection grid:

| Metric                               | Result |
| ------------------------------------ | -----: |
| Harmonic-aware rank-1 period match   |  88.9% |
| Detection rate at 1% FAP             |  82.7% |
| Recovery rate at 1% FAP              |  82.7% |
| Detection rate at 0.1% FAP           |  74.1% |
| Recovery rate at 0.1% FAP            |  74.1% |
| Median period error                  |  0.12% |
| Median recovered / injected depth    |  88.5% |
| Median recovered / injected duration | 110.6% |

The BLS null distribution uses 1,000 moving-block surrogate trials.

```text
1% FAP threshold:   59.8864
0.1% FAP threshold: 74.3911
```

These thresholds are specific to this preprocessing, search grid, statistic, target, and null-generation procedure.

## ARIMA-TCF Branch

The second detector searches the ARIMA innovation series rather than the original normalized flux.

The workflow is:

```text
PDCSAP flux
→ normalization
→ ARIMA transformation
→ one-step-ahead innovations
→ periodic ingress/egress comb search
→ event-consistency scoring
→ TCF candidate periods
```

Implemented functionality includes:

* periodic TCF-style ingress/egress matching;
* coarse-to-fine period search;
* multiple duration trials;
* minimum repeated-transit-event constraints;
* event-consistency scoring;
* harmonic-aware matching;
* top-10 candidate diagnostics;
* moving-block null calibration;
* parallel null evaluation.

### Single-case TCF example

For a 5-day, 4-hour, 1,000-ppm injection:

```text
injected period:          5.0000 days
recovered period:         4.9998 days
period matched:           True
event-consistent score:   86.3140
```

The fitted ARIMA model in this experiment did **not** converge.

The result is therefore useful evidence that the search code can locate the injected periodicity after the transformation, but it is not evidence that the underlying ARIMA model is statistically satisfactory.

### 81-case TCF benchmark

| Metric                                  |  Result |
| --------------------------------------- | ------: |
| ARIMA convergence rate                  |   48.1% |
| Harmonic-aware rank-1 period match      |   79.0% |
| Exact rank-1 period match               |   65.4% |
| Exact injected period present in top 10 |  100.0% |
| Score exceeds 1% FAP threshold          |  100.0% |
| Harmonic-aware recovery at 1% FAP       |   79.0% |
| Exact recovery at 1% FAP                |   65.4% |
| Median exact-period rank when present   |       1 |
| Median event-consistent score           | 30.8609 |

The detector-conditional 1% threshold is:

```text
15.5860
```

This threshold was generated from 1,000 moving-block surrogate realizations of the fitted ARIMA innovations.

Importantly, this is a **detector-conditional calibration**.

It does not propagate the uncertainty that would arise from independently refitting and reselecting ARIMA for every null realization.

That distinction should be retained when interpreting the TCF results.

## Kalman State-Space Branch

The Kalman branch is a new challenger background model, not a replacement for ARIMA.

The first model is intentionally simple:

```text
background_t = background_t-1 + process_noise
normalized_flux_t = background_t + measurement_noise
```

The hidden state is the slowly drifting stellar/instrumental background level. The observation equation says that normalized PDCSAP flux is that background plus measurement noise.

Missing cadences are handled explicitly. The filter predicts through a gap but skips the observation update, so missing flux values are not silently interpolated.

The detector input is the one-step residual:

```text
residual_t = normalized_flux_t - E[normalized_flux_t | previous cadences]
```

Those residuals are then passed through the existing BLS and TCF detector implementations:

```text
PDCSAP flux
→ normalization
→ local-level Kalman background
→ one-step residuals
├── BLS
└── TCF
```

The Kalman branch reports likelihood-style fit diagnostics, residual whitening and variance diagnostics, transit depth retention, and transit SNR retention. A model should not be considered better merely because it predicts flux more accurately if it suppresses injected transits.

See [Kalman State-Space Baseline](docs/kalman_state_space_baseline.md) for the model assumptions and run sequence.

### Single-target Kalman benchmark

The current saved repository outputs should be treated as the authoritative single-target comparison for KIC 11904151, Quarter 5.

The primary recovery definition is harmonic-aware: periods at the injected value or simple harmonics such as `P/2` and `2P` are counted as successful recovery of the same periodic phenomenon. Exact-period recovery is retained as a stricter diagnostic.

Across the matched 81-injection grid:

| Pipeline                     | Recovery |
| ---------------------------- | -------: |
| Raw flux -> BLS              |    67/81 |
| Existing TCF, harmonic-aware |    64/81 |
| Existing TCF, exact period   |    53/81 |
| Kalman residuals -> BLS      |    64/81 |
| Kalman residuals -> TCF      |    67/81 |
| Raw BLS union Kalman-TCF     |    71/81 |
| Existing BLS union TCF       |    72/81 |
| All four methods             |    72/81 |

The corresponding rates are:

```text
Raw -> BLS:                         0.827
Existing TCF, harmonic-aware:       0.790
Existing TCF, exact period:         0.654
Kalman -> BLS:                      0.790
Kalman -> TCF:                      0.827
Raw BLS union Kalman-TCF:           0.877
Existing BLS union TCF:             0.889
All four methods combined:          0.889
```

The row-level overlap between raw BLS and Kalman-TCF is:

| Case                         | Count |
| ---------------------------- | ----: |
| Both recovered               |    63 |
| Kalman-TCF only              |     4 |
| Raw BLS only                 |     4 |
| Neither recovered            |    10 |

Kalman-TCF therefore changes the error pattern and provides pairwise complementarity with raw BLS. However, it does not expand the best current single-target ensemble: all Kalman recoveries are already contained within the existing BLS-TCF union on this 81-injection benchmark.

The current Kalman conclusion is:

> State-space/Kalman preprocessing produced competitive standalone transit recovery, with Kalman-TCF recovering 82.7% of injections at the calibrated 1% false-alarm rate. Relative to raw-flux BLS, Kalman-TCF recovered four additional cases while missing four BLS detections, demonstrating complementary detector behavior. However, adding the Kalman pipelines did not improve the overall BLS-TCF union recovery of 88.9% on this benchmark.

This distinction should be retained for future challenger models:

```text
standalone performance
pairwise complementarity
incremental ensemble value beyond the best existing union
```

## Gaussian Process Branch

The GP branch is a new probabilistic, covariance-based background model. It is not a transit detector and does not replace ARIMA, Kalman, BLS, TCF, or the frozen reranker.

The first GP baseline estimates a smooth latent background:

```text
background(t) ~ GP(mean, covariance)
normalized_flux_t = background(t) + measurement_noise_t
```

The implemented covariance is deliberately simple:

```text
constant_amplitude * RBF(time_length_scale)
```

The lower length-scale bound is longer than the injected transit durations in the 81-case benchmark, so the GP is discouraged from fitting individual transit dips. The detector input is the background-subtracted residual:

```text
residual_t = normalized_flux_t - GP_posterior_mean(time_t)
```

Because exact GP fitting on every cadence for every injection would be too expensive, the baseline uses a deterministic anchor-point approximation: it fits the GP on evenly spaced finite observed cadences and predicts the background on the full time grid. Missing flux values are not used for fitting and do not produce detector residuals.

The null and injection-grid scripts use process-level parallelism for the expensive detector loops and cap BLAS/OpenMP thread pools to avoid CPU oversubscription.

See [Gaussian Process Background Baseline](docs/gaussian_process_baseline.md) for the model assumptions and run sequence.

Completed KIC 11904151 Quarter 5 GP outputs should be interpreted as a smooth anchor-point GP background experiment, not full exact GP regression on every cadence. Using 1,000 null surrogates, the calibrated 1% FAP thresholds were:

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

The main ensemble result is that GP adds real unique recoveries beyond the existing non-GP union:

| Combination | Recovery |
| ----------- | -------: |
| Raw BLS | 67/81 = 82.7% |
| Existing TCF | 64/81 = 79.0% |
| Kalman-TCF | 67/81 = 82.7% |
| Existing BLS union TCF | 72/81 = 88.9% |
| Existing BLS union TCF union GP-TCF | 75/81 = 92.6% |
| All six methods | 77/81 = 95.1% |

GP-TCF contributes 3 unique recoveries beyond the existing non-GP methods, and GP-BLS contributes 2 unique recoveries beyond the existing non-GP methods. The GP-TCF-only recoveries are all shallow 2-day, 2-hour, 200 ppm injections, while GP-BLS supplies the strongest standalone GP recovery rate on this benchmark.

The current physical/statistical explanation should be treated as a hypothesis supported by the saved outputs, not yet as a demonstrated universal mechanism:

```text
transit duration << GP background time scale
```

To test this directly, `scripts/run_gp_timescale_sensitivity.py` varies the fixed GP RBF length scale relative to one injected transit duration while keeping the light curve, injection, and detector grids fixed. For the KIC 11904151 Quarter 5 5-day, 4-hour, 1000 ppm injection, the result is:

| GP length scale / transit duration | Background absorption | Depth retention | SNR retention | BLS period | TCF period |
| ---------------------------------: | --------------------: | --------------: | ------------: | ---------: | ---------: |
| 0.5 | 62.8% | 35.5% | 69.0% | 4.994 d | 5.000 d |
| 1.0 | 59.0% | 39.4% | 68.1% | 4.994 d | 5.000 d |
| 2.0 | 34.0% | 63.4% | 103.1% | 4.994 d | 2.501 d |
| 5.0 | 15.5% | 85.9% | 129.4% | 4.994 d | 2.501 d |
| 10.0 | 8.7% | 95.7% | 146.6% | 4.994 d | 2.501 d |
| 20.0 | 2.3% | 101.3% | 185.2% | 4.994 d | 2.501 d |

This is consistent with the idea that very responsive GP backgrounds absorb transit morphology, while smoother backgrounds preserve short dips. The same table also shows the whitening-versus-preservation trade-off: residual autocorrelation remains substantial for the smoother high-retention configurations, so the result should not be reduced to "smoother GP is always better."

The multi-duration version repeats the same controlled test for 2-hour, 4-hour, and 8-hour transits using the dimensionless ratio `ell_GP / transit_duration`. It is still a mechanism test, not a FAP-calibrated recovery experiment. Across 18 fixed-length-scale configurations:

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

This strengthens the interpretation that time-scale separation is the controlling variable, while preserving the caveat that high-retention settings leave correlated structure in the residuals.

The 4 cases missed by all six saved methods are:

```text
period: 10 days
depth: 200 ppm
duration: 2 hours for 3 cases, 4 hours for 1 case
```

Their median GP depth retention is 85.5% and median GP SNR retention is 166.4%, so the remaining failures are better interpreted as shallow long-period candidate-generation or scoring failures, not GP transit erasure.

## 50-Star BLS / ARIMA-TCF Pilot

The multi-star experiment extends both detectors across 50 Kepler Quarter 5 target-quarter rows.

Across 900 synthetic injections, before false-alarm filtering:

| Metric                            | Result |
| --------------------------------- | -----: |
| BLS exact rank-1 recovery         |  63.4% |
| TCF exact rank-1 recovery         |  31.3% |
| Exact rank-1 union                |  74.4% |
| BLS exact candidate-set recall    |  68.7% |
| TCF exact candidate-set recall    |  42.0% |
| BLS + TCF exact candidate ceiling |  83.2% |

The important observation is not that TCF outperforms BLS—it does not in this pilot.

Instead, TCF contributes candidate periods that BLS does not always rank highly.

This creates a candidate-set ceiling substantially above either detector's rank-1 performance and motivates candidate-level combination.

### Pooled detector-level 1% FAP

Using pooled null maxima across the 50 selected stars:

| Metric                                        | Result |
| --------------------------------------------- | -----: |
| BLS exact recovery                            |  52.7% |
| TCF exact recovery                            |  13.0% |
| BLS + TCF exact union recovery                |  58.3% |
| Original-light-curve BLS threshold exceedance |  20.0% |
| Original-light-curve TCF threshold exceedance |  10.0% |

Simple noise-quartile thresholds did not solve the cross-star calibration problem.

This is one reason the project moved from direct detector-score comparison toward candidate-level reranking.

### Background time-scale audit

The first multi-star mechanism audit joins cheap stellar-background features onto the saved 50-star injection rows:

```bash
python scripts/analyze_multistar_background_timescales.py
```

The script estimates ACF-based background timescales on the longest contiguous usable segment of each preprocessed light curve, then evaluates whether background-to-transit time-scale ratios and other cheap features predict BLS-only, TCF-only, both, and neither outcomes. It does not rerun detectors.

The strongest cheap predictor in the current audit is flux scatter:

```text
Spearman robust_flux_scatter_ppm vs harmonic BLS recovery:  -0.622
Spearman robust_flux_scatter_ppm vs harmonic union recovery: -0.554
Spearman robust_flux_scatter_ppm vs harmonic neither rate:   +0.554
```

The ACF e-folding ratio also carries signal:

```text
Spearman background_tau_acf_e / transit duration vs harmonic BLS recovery: -0.555
Spearman background_tau_acf_e / transit duration vs harmonic TCF-only:     +0.341
```

By quartile of `background_tau_acf_e_days / transit_duration_days`:

| Ratio bin | BLS recovery | TCF recovery | Union recovery | BLS-only | TCF-only | Neither |
| --------- | -----------: | -----------: | -------------: | -------: | -------: | ------: |
| lowest | 83.8% | 13.2% | 83.8% | 70.6% | 0.0% | 16.2% |
| low-mid | 68.5% | 15.8% | 70.3% | 54.5% | 1.8% | 29.7% |
| high-mid | 50.9% | 12.7% | 53.5% | 40.8% | 2.6% | 46.5% |
| highest | 6.8% | 29.7% | 33.8% | 4.1% | 27.0% | 66.2% |

This is the first multi-background evidence that cheap light-curve properties can explain detector outcome patterns. It supports adding physically meaningful background features to a later adaptive router, while also showing that high-ratio/high-scatter regimes increase the overall miss rate rather than simply switching cleanly from BLS to TCF.

### Multi-background challenger benchmark

The next multi-star runner is built for a staged clean-background 5 → 10 → 50-star benchmark. The default manifest is `configs/kepler_clean_background_manifest.csv`, and the runner fails closed unless the selected rows are explicitly `catalog_clean_background` with false KOI/TCE/confirmed-planet/EB flags. Build and freeze that manifest first; see [`docs/clean_injection_benchmark.md`](docs/clean_injection_benchmark.md).

Example main run after the clean manifest has been built:

```bash
python scripts/run_multistar_challenger_benchmark.py \
  --profile main \
  --manifest-path configs/kepler_clean_background_manifest.csv \
  --selection-group catalog_clean_background \
  --target-limit 50
```

Known-planet/EB or other contaminated cohorts require the explicit `--allow-contaminated-cohort` opt-out and should use a separate output directory.

The main profile runs 50 targets with the full representative 81-case injection grid:

```text
50 stars × 81 injections = 4050 injected cases
```

The default challenger pipelines are:

```text
raw_bls
raw_tcf
arima_bls
arima_tcf
kalman_bls
kalman_tcf
gp_bls
gp_tcf
```

The runner parallelizes across stars using worker processes, caps numerical-library thread pools inside each worker, checkpoints every star to `stars/kic_<id>_q<quarter>/injections.csv`, and resumes completed compatible work by default.

Current output namespace:

```text
outputs/experiments/multistar_challenger_benchmark/pilot/
outputs/experiments/multistar_challenger_benchmark/main/
```

Key summary files:

```text
metrics/multistar_challenger_injections.csv
metrics/multistar_challenger_base_candidates.csv
metrics/multistar_challenger_pipeline_summary.csv
metrics/multistar_challenger_pairwise_overlap.csv
metrics/multistar_challenger_combinations.csv
metrics/multistar_challenger_failure_modes.csv
metrics/multistar_challenger_by_stratum.csv
metrics/multistar_challenger_summary.json
```

This runner reports both rank-1 and top-k injection recovery. It also searches each unmodified light curve before injection, stores each pipeline's base top-k candidates, and separates ranking failures from candidate-generation failures and preserved pre-existing rank-1 competition. FAP-calibrated claims still require matched null calibration for the same multi-star challenger candidate-generation settings.

Per-star branch-conditional FAP calibration is run as a second stage:

```bash
python scripts/calibrate_multistar_challenger_benchmark.py --profile pilot --n-null-trials-per-star 10
```

The current stored 10-star pilot calibration is a **10-null engineering calibration result, not an inferential 1% FAP result**:

```text
10 stars × 10 null trials/star = 100 star-conditional null rows
```

It does not reuse KIC 11904151's numeric thresholds. Each star and pipeline receives its own threshold from moving-block null surrogates of the relevant branch series.

Current 10-null engineering-calibration recovery at the nominal star-level threshold:

| Pipeline | Harmonic recovery | Exact recovery |
| -------- | ----------------: | -------------: |
| raw_bls | 24/80 = 30.0% | 24/80 = 30.0% |
| arima_tcf | 53/80 = 66.2% | 28/80 = 35.0% |
| kalman_bls | 53/80 = 66.2% | 53/80 = 66.2% |
| kalman_tcf | 62/80 = 77.5% | 35/80 = 43.8% |
| gp_bls | 50/80 = 62.5% | 50/80 = 62.5% |
| gp_tcf | 50/80 = 62.5% | 33/80 = 41.2% |
| all pipelines | 70/80 = 87.5% | 68/80 = 85.0% |

The calibrated master table is:

```text
outputs/experiments/multistar_challenger_benchmark/pilot/metrics/multistar_challenger_master_results.csv
```

The convergence analyzer tracks threshold movement and recovery-label changes as more null trials become available:

```bash
python scripts/analyze_multistar_calibration_convergence.py --profile pilot
```

The analyzer uses nested prefixes of the same stored null sequence:

```text
10-null estimate   = trials 0-9
50-null estimate   = trials 0-49
100-null estimate  = trials 0-99
250-null estimate  = trials 0-249
500-null estimate  = trials 0-499
1000-null estimate = trials 0-999
```

It reports threshold movement, bootstrap threshold uncertainty, per-pipeline recovery-label changes, union-label changes, and unique-recovery-label changes. The default threshold bootstrap uses 500 resamples of the stored null scores.

To fill the pilot convergence grid incrementally:

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

The calibration command resumes prior null trials with the same deterministic seeds, so the later levels extend the same star-level null sequence rather than starting over.

Current convergence files include:

```text
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

## Candidate Reranking

The BLS and TCF top candidates are merged into candidate groups and described using detector, consistency, and light-curve diagnostics.

The current frozen model contract is:

```text
clean_reranker_v1
```

Configuration:

```text
configs/candidate_reranker_clean_v1.json
```

The reranker uses 44 features.

### Leakage controls

The model deliberately excludes fields that reveal the experimental answer or allow the restricted injection grid to become a shortcut.

Forbidden features include:

* `target_id`;
* injection index;
* injected period;
* injected depth;
* injected duration;
* injected epoch;
* epoch phase;
* exact-match label;
* harmonic-match label;
* candidate period error relative to injected truth;
* absolute candidate-period fields.

Absolute candidate periods are excluded because the pilot injections occur only at 2, 5, and 10 days. Allowing the model to use absolute candidate period would make it possible to learn the experimental design instead of learning detector behavior.

Cross-validation is grouped by:

```text
target_id
```

rather than by candidate row.

Candidate rows from the same star therefore cannot appear simultaneously in training and validation folds.

## Candidate-Reranker Results

Across all 900 injection groups using grouped out-of-fold predictions:

| Model                     | Exact Recall@1 |  Recall@3 |  Recall@5 | Recall@10 | Harmonic Recall@1 |       MRR |
| ------------------------- | -------------: | --------: | --------: | --------: | ----------------: | --------: |
| BLS rank                  |          63.4% |     67.2% |     68.7% |     68.7% |             65.6% |     0.655 |
| TCF rank                  |          31.3% |     38.7% |     42.0% |     42.0% |             58.7% |     0.353 |
| Raw minimum detector rank |          63.4% |     77.0% |     79.8% |     83.2% |             65.6% |     0.709 |
| Logistic regression       |          72.8% |     76.8% |     79.8% |     83.2% |             79.2% |     0.756 |
| XGBoost classifier        |      **75.1%** |     79.3% | **81.6%** |     83.2% |             79.1% |     0.777 |
| XGBoost pairwise ranker   |      **75.6%** | **79.6%** |     80.4% |     83.2% |         **81.6%** | **0.779** |

The XGBoost classifier improves exact rank-1 selection over BLS in 108 injection cases and worsens it in 3 cases in the grouped out-of-fold analysis.

The pairwise ranker has slightly higher rank-1 performance, but the XGBoost classifier is retained as the primary frozen reranker because it produces candidate probabilities that can be calibrated against null trials.

### Validation checks

The current validation workflow includes:

* grouped cross-validation by unseen `target_id`;
* a locked 10-star holdout;
* label-permutation testing;
* feature ablation;
* star-level bootstrap confidence intervals;
* candidate-generation miss analysis;
* candidate-ranking failure analysis.

Selected validation results:

```text
label-permutation Recall@1 mean:          13.5%

star-bootstrap XGBoost Recall@1 95% CI:  67.0% – 82.9%

bootstrap improvement over BLS 95% CI:   +6.2 to +17.6 percentage points

candidate-generation misses:             151

XGBoost ranking failures:                 73
```

### Locked 10-star holdout

The validation workflow also reports performance on 180 injection cases from 10 held-out stars.

| Model                   | Exact Recall@1 |
| ----------------------- | -------------: |
| BLS                     |          50.6% |
| TCF                     |          27.2% |
| Logistic regression     |          62.2% |
| XGBoost classifier      |      **62.8%** |
| XGBoost pairwise ranker |          62.2% |

On this holdout, the XGBoost classifier improves 22 cases relative to BLS and worsens none.

This is encouraging evidence of generalization across targets, but the holdout is still small and comes from the same restricted experimental design.

It should not yet be treated as a population-level estimate of Kepler performance.

## Feature Ablation

Ablation on the 40-star development split gives:

| Feature set                       | Exact Recall@1 |
| --------------------------------- | -------------: |
| BLS only                          |          71.0% |
| TCF only                          |          75.0% |
| BLS + TCF detector features       |          78.1% |
| Detector + stellar-noise features |          78.2% |
| Full clean feature set            |          77.8% |

The result supports the central candidate-combination hypothesis:

> BLS and TCF contain complementary ranking information, while the current stellar-noise features add relatively little beyond the detector features in this pilot.

## Corrected Reranker False-Alarm Calibration

An earlier calibration used only detector rank-1 candidates.

That result is retained only as a diagnostic and is explicitly labelled:

```text
preliminary_rank1_only_not_final
```

It should not be used as the final reranker false-alarm calibration because injected examples were reranked over top-k candidates while the null/original calibration considered only rank-1 detector outputs.

The corrected experiment applies the same candidate-generation logic to null and original light curves:

```text
null/original light curve
→ BLS top-k candidates
→ TCF top-k candidates
→ candidate merge
→ frozen clean_reranker_v1 scoring
→ maximum reranker probability per light curve
→ empirical null threshold
```

with:

```text
top_k = 10
```

### Corrected calibration result

| Metric                                    |   Result |
| ----------------------------------------- | -------: |
| Null light curves                         |      400 |
| Original light curves                     |       50 |
| Raw detector candidate rows               |    9,000 |
| Merged candidate rows                     |    8,828 |
| 1% FAP probability threshold              | 0.992460 |
| Observed null exceedance                  |     1.0% |
| Injection exact recovery at threshold     |    40.2% |
| Original light curves exceeding threshold |   1 / 50 |

The single original light curve above the threshold is:

```text
KIC 2557816
Quarter 5
BLS-sourced candidate period ≈ 9.37946 days
```

This object is a **candidate**, not a demonstrated false positive or planet.

An original Kepler light curve can contain astrophysical periodicity, eclipsing binaries, stellar activity, instrumental structure, or a real planet.

## What the Current Results Establish

The current experiments support several engineering conclusions:

1. A reproducible BLS injection-recovery baseline is operational.
2. Periodic TCF-style searching of ARIMA innovations is operational.
3. BLS is substantially stronger than the current TCF implementation as a standalone exact-period detector across the 50-star pilot.
4. TCF nevertheless contributes complementary candidate periods.
5. Candidate-level combination can outperform either detector's raw rank-1 choice.
6. A leakage-audited XGBoost reranker improves candidate ordering on unseen target groups in the current pilot.
7. The reranker can be calibrated using top-k null candidates under a matched candidate-generation procedure.
8. ARIMA model adequacy and convergence remain major unresolved issues.

## What the Current Results Do Not Establish

The repository does **not** yet demonstrate that:

* ARIMA is the optimal Kepler background model;
* `ARIMA(1,1,0)` is generally appropriate for Kepler light curves;
* ARIMA-TCF outperforms BLS;
* the candidate reranker will generalize to the full Kepler population;
* the current 1% threshold is precisely estimated in the distribution tail;
* the model can distinguish planets from astrophysical false positives;
* the model has a dedicated known-planet recovery benchmark against cataloged periods/ephemerides;
* the model handles realistic limb-darkened transit morphology;
* performance is robust across quarters, cadence modes, stellar populations, or broad orbital-period distributions;
* the pipeline is suitable for catalog production.

Important limitations include:

* only 50 targets in the multi-star pilot;
* only Quarter 5;
* a restricted 2/5/10-day injection-period design;
* box-shaped synthetic transits;
* only two injection depths in the multi-star run;
* only 400 multi-star null realizations;
* unresolved ARIMA non-convergence;
* limited tail statistics for 1% FAP calibration;
* no completed dedicated known-planet recovery benchmark against cataloged truth;
* no completed dedicated eclipsing-binary or astrophysical false-positive benchmark;
* no external population-scale validation set.

## Repository Structure

```text
multimodel-transit-search/
├── configs/
│   ├── candidate_reranker_clean_v1.json
│   └── kepler_target_sample.yaml
│
├── docs/
│   └── scientific and dataset notes
│
├── notebooks/
│   └── exploratory analysis
│
├── outputs/
│   ├── experiments/
│   │   ├── bls_baseline/
│   │   ├── bls_injection_grid/
│   │   ├── tcf_baseline/
│   │   ├── tcf_null_calibration/
│   │   ├── tcf_injection_grid/
│   │   └── multistar_bls_tcf/
│   ├── gap_modes/
│   ├── injections/
│   ├── metrics/
│   └── processed/
│
├── scripts/
│   └── executable experiment and validation workflows
│
├── src/adaptive_transit/
│   ├── data/
│   ├── detection/
│   ├── injections/
│   ├── noise_models/
│   ├── preprocessing/
│   └── transit_models/
│
├── tests/
├── pyproject.toml
└── README.md
```

The Python distribution is named:

```text
multi-model-transit-search
```

while the import package is:

```text
adaptive_transit
```

## Installation

Python 3.11 or later is required.

```bash
git clone https://github.com/shreshtha26/multimodel-transit-search.git
cd multimodel-transit-search

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Machine-learning experiments require:

```bash
python -m pip install -e ".[ml]"
```

Additional optional dependency groups are available for deep learning, notebooks, parallel processing, and experiment tracking.

The first Kepler run requires network access so that `lightkurve` can retrieve the requested data.

## Running the Experiments

Run scripts from the repository root.

### ARIMA prototype

```bash
python scripts/run_single_target_arima.py
python scripts/run_gap_mode_comparison.py
python scripts/run_gap_mode_injection_experiment.py
```

### BLS

```bash
python scripts/run_bls_baseline.py
python scripts/run_bls_injection_grid.py
python scripts/analyse_bls_injection_grid.py
```

### ARIMA-TCF

```bash
python scripts/run_tcf_baseline.py
python scripts/run_tcf_null_calibration.py
python scripts/run_tcf_injection_grid.py
```

The TCF null calibration uses multiprocessing and should be run as a script rather than from an interactive notebook cell.

### Kalman State-Space Challenger

```bash
python scripts/run_kalman_baseline.py
python scripts/run_kalman_null_calibration.py
python scripts/run_kalman_injection_grid.py
python scripts/compare_kalman_recovery_overlap.py
```

Run the Kalman null calibration before the Kalman injection grid. The injection grid reads the Kalman-specific BLS and TCF 1% FAP thresholds.

The overlap comparison script does not rerun detectors. It joins the saved BLS, TCF, and Kalman injection grids on the exact injection parameter tuple and writes comparison-ready recovery-overlap tables.

### Gaussian Process Background Challenger

```bash
python scripts/run_gp_baseline.py
python scripts/run_gp_null_calibration.py
python scripts/run_gp_injection_grid.py
python scripts/compare_gp_recovery_overlap.py
python scripts/run_gp_timescale_sensitivity.py
python scripts/run_gp_multiduration_timescale_sensitivity.py
python scripts/analyze_gp_remaining_misses.py
```

Run the GP null calibration before the GP injection grid. The injection grid reads the GP-specific BLS and TCF 1% FAP thresholds.

The GP null and injection scripts run worker processes for expensive detector evaluations and cap numerical library thread pools inside each worker.

Both scripts checkpoint completed rows to their final CSV outputs and resume by default when rerun with the same settings.

The GP time-scale sensitivity script is not a new FAP-calibrated recovery experiment. It is a controlled mechanism test that varies the GP background length scale relative to a fixed injected transit duration and reports transit retention, residual diagnostics, and detector scores.

The multi-duration sensitivity script repeats that mechanism test for 2-hour, 4-hour, and 8-hour transits and aggregates results against the dimensionless ratio `ell_GP / transit_duration`. The remaining-miss analyzer reads the saved overlap table and does not rerun any detectors.

### BLS / TCF comparison

```bash
python scripts/compare_bls_tcf.py
```

### 50-star pilot

```bash
python scripts/run_multistar_bls_tcf.py
python scripts/analyze_multistar_regime_calibration.py
python scripts/analyze_multistar_background_timescales.py
python scripts/run_multistar_challenger_benchmark.py --profile pilot
python scripts/calibrate_multistar_challenger_benchmark.py --profile pilot --n-null-trials-per-star 10
python scripts/analyze_multistar_calibration_convergence.py --profile pilot
python scripts/run_multistar_challenger_benchmark.py --profile main
python scripts/build_multistar_candidate_dataset.py
```

The optimized workflow caches downloaded light curves and base ARIMA fits and supports resumable per-target execution.

### Candidate reranking

```bash
python scripts/train_candidate_rerankers.py
python scripts/validate_candidate_reranker_result.py
python scripts/freeze_candidate_reranker.py
```

Frozen configuration:

```text
configs/candidate_reranker_clean_v1.json
```

### Corrected top-k calibration

```bash
python scripts/run_multistar_reranker_topk_calibration.py
```

This is the reranker calibration that should be used for current false-alarm analysis.

The earlier rank-1-only calibration is preliminary.

### Tests

```bash
pytest
```

## Important Outputs

### ARIMA

```text
outputs/metrics/kic_11904151_q5_arima_candidates.csv
outputs/metrics/kic_11904151_q5_stationarity_diagnostics.csv
outputs/metrics/kic_11904151_q5_phase1_completion.json
outputs/gap_modes/metrics/kic_11904151_q5_gap_mode_comparison.csv
outputs/gap_modes/metrics/kic_11904151_q5_gap_mode_report.json
```

### BLS

```text
outputs/experiments/bls_baseline/metrics/kic_11904151_q5_bls_summary.json
outputs/experiments/bls_baseline/metrics/kic_11904151_q5_bls_fap_thresholds.csv
outputs/experiments/bls_injection_grid/metrics/kic_11904151_q5_bls_injection_grid.csv
outputs/experiments/bls_injection_grid/metrics/kic_11904151_q5_bls_injection_grid_summary.json
```

### TCF

```text
outputs/experiments/tcf_baseline/metrics/kic_11904151_q5_tcf_summary.json
outputs/experiments/tcf_null_calibration/metrics/kic_11904151_q5_tcf_fap_thresholds.csv
outputs/experiments/tcf_null_calibration/metrics/kic_11904151_q5_tcf_null_calibration_summary.json
outputs/experiments/tcf_injection_grid/metrics/kic_11904151_q5_tcf_injection_grid.csv
outputs/experiments/tcf_injection_grid/metrics/kic_11904151_q5_tcf_injection_grid_summary.json
```

### Kalman

```text
outputs/experiments/kalman_baseline/metrics/kic_11904151_q5_kalman_summary.json
outputs/experiments/kalman_baseline/metrics/kic_11904151_q5_kalman_model_diagnostics.csv
outputs/experiments/kalman_null_calibration/metrics/kic_11904151_q5_kalman_fap_thresholds.csv
outputs/experiments/kalman_null_calibration/metrics/kic_11904151_q5_kalman_null_calibration_summary.json
outputs/experiments/kalman_injection_grid/metrics/kic_11904151_q5_kalman_injection_grid.csv
outputs/experiments/kalman_injection_grid/metrics/kic_11904151_q5_kalman_injection_grid_summary.json
outputs/experiments/kalman_recovery_overlap/metrics/kic_11904151_q5_kalman_recovery_overlap.csv
outputs/experiments/kalman_recovery_overlap/metrics/kic_11904151_q5_kalman_pairwise_overlap.csv
outputs/experiments/kalman_recovery_overlap/metrics/kic_11904151_q5_kalman_combination_overlap.csv
outputs/experiments/kalman_recovery_overlap/metrics/kic_11904151_q5_kalman_recovery_overlap_summary.json
```

### Gaussian Process

```text
outputs/experiments/gp_baseline/metrics/kic_11904151_q5_gp_summary.json
outputs/experiments/gp_baseline/metrics/kic_11904151_q5_gp_model_diagnostics.csv
outputs/experiments/gp_null_calibration/metrics/kic_11904151_q5_gp_fap_thresholds.csv
outputs/experiments/gp_null_calibration/metrics/kic_11904151_q5_gp_null_calibration_summary.json
outputs/experiments/gp_injection_grid/metrics/kic_11904151_q5_gp_injection_grid.csv
outputs/experiments/gp_injection_grid/metrics/kic_11904151_q5_gp_injection_grid_summary.json
outputs/experiments/gp_recovery_overlap/metrics/kic_11904151_q5_gp_recovery_overlap.csv
outputs/experiments/gp_recovery_overlap/metrics/kic_11904151_q5_gp_pairwise_overlap.csv
outputs/experiments/gp_recovery_overlap/metrics/kic_11904151_q5_gp_combination_overlap.csv
outputs/experiments/gp_recovery_overlap/metrics/kic_11904151_q5_gp_recovery_overlap_summary.json
outputs/experiments/gp_timescale_sensitivity/metrics/kic_11904151_q5_gp_timescale_sensitivity_summary.csv
outputs/experiments/gp_timescale_sensitivity/metrics/kic_11904151_q5_gp_timescale_sensitivity_by_ratio.csv
outputs/experiments/gp_timescale_sensitivity/processed/kic_11904151_q5_gp_timescale_transit_window_samples.csv
outputs/experiments/gp_timescale_sensitivity/figures/kic_11904151_q5_gp_timescale_transit_window_diagnostics.png
outputs/experiments/gp_multiduration_timescale_sensitivity/metrics/kic_11904151_q5_gp_multiduration_timescale_sensitivity.csv
outputs/experiments/gp_multiduration_timescale_sensitivity/metrics/kic_11904151_q5_gp_multiduration_timescale_by_ratio.csv
outputs/experiments/gp_multiduration_timescale_sensitivity/figures/kic_11904151_q5_gp_multiduration_timescale_ratio_collapse.png
outputs/experiments/gp_recovery_overlap/metrics/kic_11904151_q5_gp_remaining_all_method_misses.csv
outputs/experiments/gp_recovery_overlap/metrics/kic_11904151_q5_gp_remaining_all_method_misses_summary.json
```

### Multi-star benchmark and reranker

```text
outputs/experiments/multistar_bls_tcf/optimized/metrics/multistar_summary.json
outputs/experiments/multistar_bls_tcf/optimized/metrics/multistar_candidate_reranking_dataset.csv
outputs/experiments/multistar_bls_tcf/optimized/metrics/candidate_reranker_metrics.csv
outputs/experiments/multistar_bls_tcf/optimized/metrics/candidate_reranker_validation_summary.json
outputs/experiments/multistar_bls_tcf/optimized/metrics/candidate_reranker_locked_holdout_metrics.csv
outputs/experiments/multistar_bls_tcf/optimized/models/clean_reranker_v1_metadata.json
outputs/experiments/multistar_bls_tcf/optimized/models/clean_reranker_v1_feature_columns.txt
outputs/experiments/multistar_bls_tcf/optimized/models/clean_reranker_v1_xgboost_classifier.joblib
outputs/experiments/multistar_background_timescale/metrics/multistar_background_timescale_features.csv
outputs/experiments/multistar_background_timescale/metrics/multistar_injections_with_background_timescales.csv
outputs/experiments/multistar_background_timescale/metrics/multistar_background_feature_correlations.csv
outputs/experiments/multistar_background_timescale/metrics/multistar_background_timescale_summary.json
outputs/experiments/multistar_background_timescale/figures/multistar_background_ratio_bin_recovery.png
```

### Top-k reranker calibration

```text
outputs/experiments/multistar_bls_tcf/optimized/reranker_topk_calibration/reranker_topk_probability_calibration_summary.json
outputs/experiments/multistar_bls_tcf/optimized/reranker_topk_calibration/reranker_topk_detector_candidates.csv
outputs/experiments/multistar_bls_tcf/optimized/reranker_topk_calibration/reranker_topk_merged_candidates.csv
outputs/experiments/multistar_bls_tcf/optimized/reranker_topk_calibration/reranker_topk_scored_candidates.csv
outputs/experiments/multistar_bls_tcf/optimized/reranker_topk_calibration/reranker_topk_top_scores.csv
```

## Reproducibility Rules

The experiments follow several rules that should remain explicit as the project expands:

* injection and null random seeds must be recorded;
* false-alarm thresholds are detector- and configuration-specific;
* thresholds must be recalibrated when preprocessing, search grids, score definitions, candidate generation, or null procedures change;
* candidate-reranker validation must split by `target_id`, not candidate row;
* answer-derived injection fields must not enter model features;
* the frozen feature contract is versioned;
* modifying the feature contract creates a new reranker version;
* null and injected cases should use equivalent candidate-generation logic when calibrating the reranker;
* summary numbers should not be generalized beyond the experiment that generated them.

## Next Scientific Milestones

The immediate priorities are:

1. Expand beyond the current 50-star sample.
2. Increase the number of null trials substantially for more reliable tail calibration.
3. Replace the fixed 2/5/10-day injection grid with a broader sampled period distribution.
4. Investigate the 151 candidate-generation misses before adding more complex classifiers.
5. Improve and scientifically resolve ARIMA convergence and differencing behavior.
6. Compare alternative background models, including robust biweight/spline detrending and probabilistic/state-space approaches.
7. Add transit-shaped detectors such as TLS.
8. Introduce realistic limb-darkened transit injections.
9. Test on known Kepler planets and astrophysical false positives.
10. Characterize performance across stellar variability, noise level, depth, duration, period, gaps, and cadence.
11. Validate `clean_reranker_v1` on a substantially larger untouched target population.
12. Only after candidate generation is robust, evaluate CNN/TCN, probabilistic, and attention-based morphology models and a higher-level adaptive ensemble.

## Current Conclusion

The project has progressed from a single-target ARIMA experiment into a working multi-detector candidate-generation and reranking prototype.

The current evidence indicates that:

```text
BLS
→ strongest standalone detector in the present pilot

ARIMA-TCF
→ weaker standalone detector
→ contributes complementary candidate periods

Kalman-TCF
→ competitive standalone detector on the 81-case benchmark
→ pairwise-complementary with raw BLS
→ no added union recovery beyond the existing BLS-TCF union in the current saved outputs

GP-BLS / GP-TCF
→ smooth anchor-point covariance-background challenger branch
→ GP-BLS recovers 90.1% at calibrated 1% FAP on the 81-case benchmark
→ GP-TCF contributes 3 unique recoveries beyond the existing non-GP union
→ all six current methods reach 95.1% union recovery on KIC 11904151 Quarter 5

multi-star background features
→ flux scatter and ACF time-scale ratios predict detector outcome patterns
→ support a future adaptive router as a scientific hypothesis, not a decorative model layer

BLS + TCF candidate set
→ higher recovery ceiling than either detector alone

leakage-audited XGBoost reranker
→ improves rank-1 candidate selection on unseen target groups

top-k null calibration
→ provides a matched empirical threshold for the frozen reranker
```

At the same time, ARIMA convergence, limited population size, restricted injections, thin false-alarm tails, and the absence of known-planet/false-positive validation remain substantial scientific limitations.

> **The current result is evidence that heterogeneous transit detectors can provide complementary candidate information that a leakage-controlled reranker can exploit. It is not yet evidence of a validated replacement for established Kepler transit-search pipelines.**

## Documentation

* [Phase 1: Single-Target ARIMA Prototype](docs/phase1_single_target_arima.md)
* [Kepler Dataset Strategy](docs/kepler_dataset_strategy.md)
* [Kalman State-Space Baseline](docs/kalman_state_space_baseline.md)
* [Gaussian Process Background Baseline](docs/gaussian_process_baseline.md)
* [Multi-Star Background Time-Scale Audit](docs/multistar_background_timescale_audit.md)
* [Multi-Star Challenger Benchmark](docs/multistar_challenger_benchmark.md)
