# multimodel-transit-search

`multimodel-transit-search` is a reproducible research prototype for benchmarking complementary approaches to exoplanet transit detection in Kepler light curves.

The current implementation contains two detector branches and a candidate-level machine-learning reranker:

```text
Kepler PDCSAP flux
│
├── direct normalized-flux branch
│   └── Box Least Squares (BLS)
│
└── autoregressive-transformation branch
    ├── ARIMA diagnostics and model selection
    ├── one-step-ahead innovations
    └── periodic Transit Comb Filter (TCF)
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

The current work is organized around five questions:

1. Can an ARIMA-family model reduce predictable correlated variability while retaining transit information in a detectable form?
2. How sensitive is ARIMA selection and transit preservation to the treatment of missing Kepler cadences?
3. How does direct BLS detection compare with TCF detection on ARIMA-transformed light curves?
4. Do BLS and TCF provide complementary candidate information across multiple stars and noise regimes?
5. Can a leakage-audited candidate reranker combine those candidates while generalizing to previously unseen targets and maintaining an empirically calibrated false-alarm threshold?

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
* the model has been validated on confirmed Kepler planets;
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
* no known-planet benchmark;
* no eclipsing-binary or astrophysical false-positive benchmark;
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

### BLS / TCF comparison

```bash
python scripts/compare_bls_tcf.py
```

### 50-star pilot

```bash
python scripts/run_multistar_bls_tcf.py
python scripts/analyze_multistar_regime_calibration.py
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
