# Multi-Model Transit Search

A reproducible research prototype for comparing exoplanet transit-search methods on Kepler light curves.

The project currently implements and evaluates two detector branches plus a frozen candidate reranker:

```text
Kepler PDCSAP flux
├── direct normalized-flux branch
│   └── Box Least Squares (BLS)
└── autoregressive-transformation branch
    ├── ARIMA diagnostics and model selection
    ├── one-step-ahead innovations
    └── periodic Transit Comb Filter (TCF)

BLS/TCF candidate tables
└── leakage-audited XGBoost reranker
    ├── grouped validation by unseen target_id
    ├── frozen clean_v1 feature contract
    └── top-k null/original false-alarm calibration
```

The longer-term goal is to compare statistical, machine-learning, and deep-learning detectors at matched false-alarm rates and then combine their evidence using an adaptive ensemble.

> **Research status:** active research prototype with one single-target benchmark and one 50-star Kepler pilot. The clean reranker is frozen as `clean_reranker_v1`, but the pipeline is still not a production transit-search system or a validated astrophysical catalog generator.

## Current Research Questions

The repository is being developed around four questions:

1. Can an ARIMA-family model reduce predictable correlated variability without destroying transit information?
2. How sensitive is ARIMA selection to the representation of missing Kepler cadences?
3. How does direct BLS detection compare with ARIMA-transformed TCF detection on the same injection grid?
4. Can all methods eventually be compared at controlled empirical false-alarm rates across many stars and noise regimes?
5. Can a leakage-audited candidate reranker combine BLS and TCF candidates without overfitting to target identity or injected-period metadata?

## Current Scope

The original single-target benchmark uses:

| Item                         | Current value                   |
| ---------------------------- | ------------------------------- |
| Target                       | KIC 11904151                    |
| Quarter                      | Kepler Quarter 5                |
| Flux product                 | PDCSAP flux                     |
| Quality policy               | Lightkurve default quality mask |
| Usable observations          | 4,486                           |
| Regular cadence-grid length  | 4,634                           |
| Explicit missing cadences    | 148                             |
| Injection periods            | 2, 5, and 10 days               |
| Injection durations          | 2, 4, and 8 hours               |
| Injection depths             | 200, 500, and 1,000 ppm         |
| Epoch phases                 | 0.15, 0.45, and 0.75            |
| Injection cases per detector | 81                              |

The synthetic box transits are added to a real Kepler PDCSAP light curve. Therefore, this is a **real-noise, synthetic-signal injection-recovery benchmark**, not a purely simulated dataset.

The current multi-star pilot uses:

| Item                         | Current value                                            |
| ---------------------------- | -------------------------------------------------------- |
| Targets                      | 50 Kepler target-quarter rows                            |
| Quarter                      | Kepler Quarter 5                                         |
| Flux product                 | PDCSAP flux                                              |
| Injection periods            | 2, 5, and 10 days                                        |
| Injection durations          | 2, 4, and 8 hours                                        |
| Injection depths             | 500 and 1,000 ppm                                        |
| Epoch phases                 | 0.45                                                     |
| Injection cases              | 900 total, 18 per target                                 |
| Null trials                  | 400 total, 8 per target                                  |
| Candidate reranker version   | `clean_reranker_v1`                                      |
| Reranker split policy        | grouped by `target_id`, never by candidate row           |
| Frozen reranker feature set  | 44 non-leaking detector/noise/candidate-consistency fields |

The multi-star pilot deliberately excludes answer-derived fields such as injected period, injected depth, injected duration, epoch phase, exact-match labels, harmonic labels, target ID, and absolute candidate period from model features.

## Implemented Components

### Data and preprocessing

* Kepler PDCSAP loading with Lightkurve.
* Configurable quality filtering.
* Construction of a regular cadence grid.
* Explicit masks for observed, usable, missing, and interpolated cadences.
* Leakage-aware normalization using a chronological fitting fraction.
* Machine-readable CSV, JSON, and Parquet outputs.

### Gap-representation experiments

Three representations are compared:

```text
longest_contiguous
    Uses only the longest uninterrupted usable segment.

full_grid_missing
    Preserves the complete regular cadence grid and leaves missing cadences as NaN.
    This is the default explicit gap representation.

interpolated_full_grid
    Interpolates only eligible short interior gaps.
    This is treated as a challenger representation rather than ground truth.
```

Earlier single-target outputs use the label `full_gap` for the full-grid missing-data representation.

### ARIMA branch

* Candidate fitting over explicit `(p, d, q)` grids.
* Convergence and numerical-validity checks.
* ADF and KPSS stationarity diagnostics.
* Residual ACF and Ljung-Box whitening diagnostics.
* Rolling-variance and ARCH-based variance diagnostics.
* Coefficient-boundary and stability checks.
* Forecast baselines and fit metrics.
* Transit-preservation experiments.
* ARIMA-transformed template matching.
* Gap-mode comparison.
* Best-available versus scientifically acceptable model-selection semantics.

### BLS branch

* Astropy Box Least Squares period search.
* Period and duration grids.
* Synthetic periodic box-transit injection.
* Moving-block null surrogates.
* Empirical 1% and 0.1% false-alarm thresholds.
* Injection-recovery summaries by depth, duration, and period.

### ARIMA-TCF branch

* Fixed ARIMA transformation to one-step-ahead innovations.
* Periodic ingress-egress comb matching.
* Event-consistent scoring across repeated transit events.
* Minimum event-count and event-consistency constraints.
* Harmonic-aware period matching.
* Exact-period top-10 rank diagnostics.
* Coarse-to-fine period search.
* Parallel moving-block null calibration.
* Injection-recovery summaries by depth, duration, and period.

### Multi-star candidate reranking

* Parallel 50-star BLS/ARIMA-TCF benchmark with resumable per-star checkpoints.
* Cached Kepler light curves and cached base ARIMA fits.
* Merged BLS/TCF candidate rows for reranking.
* Explicit missingness indicators for detector-specific diagnostics.
* Frozen `clean_reranker_v1` feature list and model metadata.
* Logistic-regression, XGBoost classifier, and XGBoost pairwise-ranker comparisons.
* Grouped cross-validation by unseen `target_id`.
* Label-permutation validation, feature ablation, star-level bootstrap confidence intervals, and miss analysis.
* Corrected top-k null/original calibration for reranker scores.

## Current Results

### 1. ARIMA model validation

For the default full-grid missing-data representation, the current highest-ranked admissible model is:

```text
ARIMA(1,1,0)
```

It is **not scientifically accepted as a final noise model**.

The main unresolved issues are:

* ADF and KPSS give conflicting evidence for the undifferenced series.
* The statistical need for `d=1` remains unresolved.
* Residual autocorrelation remains.
* Residual variance is unstable.
* The selected model fails the current transit-preservation constraint.

The engineering workflow for the single-target ARIMA prototype is complete, but the saved phase report marks it as not scientifically ready for scale-up.

### 2. BLS baseline

The single 5-day, 4-hour, 1,000-ppm injection is recovered at approximately 4.994 days and exceeds both calibrated thresholds.

The 81-case injection grid reports:

| Metric                             | Result |
| ---------------------------------- | -----: |
| Harmonic-aware rank-1 period match |  88.9% |
| Detection rate at 1% FAP           |  82.7% |
| Recovery rate at 1% FAP            |  82.7% |
| Detection rate at 0.1% FAP         |  74.1% |
| Recovery rate at 0.1% FAP          |  74.1% |
| Median period error                |  0.12% |
| Median recovered/injected depth    |  88.5% |
| Median recovered/injected duration | 110.6% |

The BLS false-alarm thresholds are based on 1,000 moving-block null trials:

```text
1% FAP threshold:   59.8864
0.1% FAP threshold: 74.3911
```

### 3. ARIMA-TCF baseline

For the single 5-day, 4-hour, 1,000-ppm injection:

```text
injected period:            5.0000 days
recovered period:           4.9998 days
period matched:             True
event-consistent score:     86.3140
original-light-curve score: 22.4209
```

The current detector-conditional 1% TCF threshold is:

```text
15.5860
```

This threshold is calibrated on moving-block surrogates of the fitted ARIMA innovations. It is therefore conditional on the fitted ARIMA transformation and does not yet include uncertainty from refitting ARIMA for every null trial.

The 81-case TCF injection grid reports:

| Metric                                  |  Result |
| --------------------------------------- | ------: |
| ARIMA convergence rate                  |   48.1% |
| Harmonic-aware rank-1 period match      |   79.0% |
| Exact rank-1 period match               |   65.4% |
| Exact injected period present in top 10 |  100.0% |
| Detection rate at 1% FAP                |  100.0% |
| Harmonic-aware recovery at 1% FAP       |   79.0% |
| Exact recovery at 1% FAP                |   65.4% |
| Median exact-period rank when present   |       1 |
| Median event-consistent score           | 30.8609 |

### 4. 50-star BLS/ARIMA-TCF pilot

The 50-star pilot is an optimized real-noise, synthetic-signal experiment over 900 injections and 400 null trials. BLS generalized more strongly than TCF, while TCF still supplied complementary candidate periods.

Before false-alarm filtering:

| Metric                           | Result |
| -------------------------------- | -----: |
| BLS exact rank-1 recovery        |  63.4% |
| TCF exact rank-1 recovery        |  31.3% |
| Oracle exact rank-1 union        |  74.4% |
| BLS exact candidate-set recall   |  68.7% |
| TCF exact candidate-set recall   |  42.0% |
| BLS + TCF exact candidate ceiling |  83.2% |

At the pooled detector-level 1% FAP threshold:

| Metric                       | Result |
| ---------------------------- | -----: |
| BLS exact recovery           |  52.7% |
| TCF exact recovery           |  13.0% |
| BLS + TCF exact recovery     |  58.3% |
| Original TCF candidate rate  |  10.0% |
| Original BLS candidate rate  |  20.0% |

Noise-quartile raw-score thresholds did not solve cross-star calibration. The simple regime threshold produced 56.8% exact union recovery, which is slightly below the pooled threshold result.

### 5. Frozen clean reranker

The candidate reranker is frozen as:

```text
clean_reranker_v1
```

The frozen feature contract lives in:

```text
configs/candidate_reranker_clean_v1.json
```

The clean feature list excludes injected-period metadata, labels, target identity, injection index, and absolute candidate-period fields. Absolute candidate periods were excluded because the pilot injection grid contains only 2, 5, and 10 day periods.

Out-of-fold grouped validation across all 50 stars:

| Model                  | Exact Recall@1 | Exact Recall@3 | Exact Recall@5 | Exact Recall@10 | Harmonic Recall@1 | MRR   |
| ---------------------- | -------------: | -------------: | -------------: | --------------: | ----------------: | ----: |
| BLS rank-1             |          63.4% |          67.2% |          68.7% |           68.7% |             65.6% | 0.655 |
| TCF rank-1             |          31.3% |          38.7% |          42.0% |           42.0% |             58.7% | 0.353 |
| Raw minimum rank       |          63.4% |          77.0% |          79.8% |           83.2% |             65.6% | 0.709 |
| Logistic regression    |          72.8% |          76.8% |          79.8% |           83.2% |             79.2% | 0.756 |
| XGBoost classifier     |          75.1% |          79.3% |          81.6% |           83.2% |             79.1% | 0.777 |
| XGBoost pairwise ranker |          75.6% |          79.6% |          80.4% |           83.2% |             81.6% | 0.779 |

The XGBoost classifier contingency against BLS rank-1 is:

| BLS exact | XGBoost exact | Count |
| --------- | ------------- | ----: |
| Yes       | Yes           |   568 |
| Yes       | No            |     3 |
| No        | Yes           |   108 |
| No        | No            |   221 |

Validation checks:

| Check                                      | Result |
| ------------------------------------------ | -----: |
| Label-permutation exact Recall@1 mean      |  13.5% |
| Star-bootstrap XGBoost Recall@1 95% CI     | 67.0-82.9% |
| Star-bootstrap improvement over BLS 95% CI | +6.2 to +17.6 percentage points |
| Candidate-generation misses                |    151 |
| XGBoost classifier ranking failures        |     73 |

Feature ablation on the 40-star development split shows that BLS and TCF provide complementary information:

| Feature set                     | Exact Recall@1 |
| ------------------------------- | -------------: |
| BLS only                        |          71.0% |
| TCF only                        |          75.0% |
| BLS + TCF detector features     |          78.1% |
| Detector + stellar-noise fields |          78.2% |
| Full clean model                |          77.8% |

### 6. Reranker score calibration

The initial rank-1-only reranker calibration is retained only as a preliminary diagnostic. It is explicitly labelled:

```text
preliminary_rank1_only_not_final
```

The corrected calibration reruns null and original-data candidate generation with top-k detector outputs before applying the frozen reranker.

Corrected top-k calibration:

| Metric                             | Result |
| ---------------------------------- | -----: |
| Null light curves                  |    400 |
| Original light curves              |     50 |
| Raw detector top-k rows            |  9,000 |
| Merged candidate rows              |  8,828 |
| Reranker probability threshold     | 0.992460 |
| Observed null exceedance           |   1.0% |
| OOF exact recovery at threshold    |  40.2% |
| Original candidate fraction        |   2.0% |

The one original light curve exceeding the corrected threshold is KIC 2557816, Quarter 5, with a BLS-sourced candidate period of 9.37946 days.

### What these results do not establish

The BLS and TCF numbers should not yet be interpreted as a final head-to-head comparison because:

* the 50-star experiment is still a pilot, not a population-scale survey;
* the injection grid is small and contains only box-shaped signals;
* the optimized multi-star run uses a reduced grid relative to the earlier 81-case single-star experiments;
* the frozen reranker has been validated by grouped cross-validation and a locked 10-star holdout, but it still needs a larger independent target sample;
* the corrected reranker calibration uses 400 null light curves, which is thin for tail estimation at 1% FAP;
* original-light-curve candidates are significant candidates, not proven false positives, because real planets, eclipsing binaries, stellar periodicity, and instrumental signals may be present;
* no known-planet population or astrophysical false-positive benchmark has been evaluated.

## Scientific Selection Policy

ARIMA is treated as an intermediate signal transformation, not as the final forecasting product.

Candidate selection therefore follows the broad hierarchy:

```text
fit validity
-> residual whitening
-> variance stability
-> transit preservation
-> transit distortion
-> baseline-relative forecasting
-> model complexity
-> forecast metrics
-> information criteria
```

A model with attractive AIC, BIC, RMSE, or MAE is not accepted if it fails convergence, numerical validity, whitening, variance, or transit-preservation checks.

## Repository Structure

```text
multimodel-transit-search/
├── configs/                         configuration assets
├── docs/                            scientific and dataset notes
├── notebooks/                       exploratory analyses
├── outputs/                         committed metrics, figures, and processed artifacts
│   ├── experiments/
│   │   ├── bls_baseline/
│   │   ├── bls_injection_grid/
│   │   ├── tcf_baseline/
│   │   ├── tcf_null_calibration/
│   │   └── tcf_injection_grid/
│   ├── gap_modes/
│   ├── injections/
│   ├── metrics/
│   └── processed/
├── scripts/                         executable experiment workflows
├── src/adaptive_transit/
│   ├── data/                        Kepler data loading
│   ├── detection/                   BLS, TCF, matched filters, and false-alarm tools
│   ├── injections/                  synthetic transit injection
│   ├── noise_models/                ARIMA fitting, selection, and diagnostics
│   └── preprocessing/               cadence grids, masks, and normalization
├── tests/                           automated tests
├── pyproject.toml
└── README.md
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

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Development dependencies can be installed with:

```bash
python -m pip install -e ".[dev]"
```

Optional extras are available for later ML, deep-learning, parallel, tracking, and notebook work:

```bash
python -m pip install -e ".[ml]"
python -m pip install -e ".[dl]"
python -m pip install -e ".[parallel]"
python -m pip install -e ".[tracking]"
python -m pip install -e ".[notebooks]"
```

The multi-star reranker workflow requires the ML extra because it uses XGBoost:

```bash
python -m pip install -e ".[ml]"
```

The first Kepler run requires network access so Lightkurve can retrieve the target light curve.

## Running the Current Experiments

The experiment scripts currently use values defined in their `default_settings()` functions rather than a complete command-line interface.

Run them from the repository root.

### ARIMA prototype

```bash
python scripts/run_single_target_arima.py
python scripts/run_gap_mode_comparison.py
python scripts/run_gap_mode_injection_experiment.py
```

### BLS baseline and injection grid

The BLS baseline creates the null calibration files required by the injection-grid script.

```bash
python scripts/run_bls_baseline.py
python scripts/run_bls_injection_grid.py
python scripts/analyse_bls_injection_grid.py
```

### ARIMA-TCF baseline and injection grid

Run the TCF null calibration before the TCF injection grid.

```bash
python scripts/run_tcf_baseline.py
python scripts/run_tcf_null_calibration.py
python scripts/run_tcf_injection_grid.py
```

The TCF null calibration uses multiprocessing. Run it as a script from the repository root rather than from an interactive notebook cell.

### Multi-star BLS/ARIMA-TCF pilot

The optimized 50-star pilot is resumable and uses local light-curve and ARIMA caches:

```bash
python scripts/run_multistar_bls_tcf.py
python scripts/analyze_multistar_regime_calibration.py
python scripts/build_multistar_candidate_dataset.py
```

The default optimized profile uses 50 targets, 18 injections per target, 8 null trials per target, and grouped target-level execution.

### Frozen candidate reranker

Train and validate the clean reranker with grouped folds by `target_id`:

```bash
python scripts/train_candidate_rerankers.py
python scripts/validate_candidate_reranker_result.py
```

Freeze the final clean XGBoost classifier:

```bash
python scripts/freeze_candidate_reranker.py
```

The frozen feature contract is:

```text
configs/candidate_reranker_clean_v1.json
```

The frozen model artifacts are written to:

```text
outputs/experiments/multistar_bls_tcf/optimized/models/
```

### Corrected top-k reranker calibration

The matched calibration experiment regenerates top-k null and original-data detector candidates and applies the frozen reranker:

```bash
python scripts/run_multistar_reranker_topk_calibration.py
```

This is the calibration result to use for reranker false-alarm analysis. The rank-1-only calibration in the validation script is preliminary and is labelled as such in its summary JSON.

### Tests

```bash
pytest
```

## Important Output Files

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

### Multi-star reranker

```text
outputs/experiments/multistar_bls_tcf/optimized/metrics/multistar_summary.json
outputs/experiments/multistar_bls_tcf/optimized/metrics/multistar_candidate_reranking_dataset.csv
outputs/experiments/multistar_bls_tcf/optimized/metrics/candidate_reranker_metrics.csv
outputs/experiments/multistar_bls_tcf/optimized/metrics/candidate_reranker_validation_summary.json
outputs/experiments/multistar_bls_tcf/optimized/models/clean_reranker_v1_metadata.json
outputs/experiments/multistar_bls_tcf/optimized/models/clean_reranker_v1_feature_columns.txt
outputs/experiments/multistar_bls_tcf/optimized/models/clean_reranker_v1_xgboost_classifier.joblib
outputs/experiments/multistar_bls_tcf/optimized/reranker_topk_calibration/reranker_topk_probability_calibration_summary.json
outputs/experiments/multistar_bls_tcf/optimized/reranker_topk_calibration/reranker_topk_scored_candidates.csv
outputs/experiments/multistar_bls_tcf/optimized/reranker_topk_calibration/reranker_topk_top_scores.csv
```

## Next Scientific Milestones

1. Expand the multi-star pilot beyond 50 targets and use more null trials per noise regime for stronger tail calibration.
2. Replace the fixed three-period injection grid with a broader sampled period distribution so absolute period cannot act as an experimental-design shortcut.
3. Save top-k detector diagnostics for injected, null, and original cases using the same schema in future runs.
4. Validate `clean_reranker_v1` on a larger untouched target set before treating it as a final estimate.
5. Investigate the 151 current candidate-generation misses by period, duration, depth, noise regime, gap fraction, and detector harmonic behavior.
6. Improve candidate generation before adding deep-learning morphology models.
7. Stabilize ARIMA fitting and define a scientifically defensible response to non-convergence.
8. Compare additional background models such as robust biweight/spline detrending and Gaussian-process or state-space models.
9. Add TLS and other transit-shaped statistical detectors.
10. Evaluate all detectors at matched false-alarm rates and characterize performance by depth, duration, period, stellar variability, gaps, and cadence.

## Documentation

* [Phase 1: Single-Target ARIMA Prototype](docs/phase1_single_target_arima.md)
* [Kepler Dataset Strategy](docs/kepler_dataset_strategy.md)

## Reproducibility Notes

* Random seeds should be recorded for all null and injection experiments.
* Null thresholds are detector- and search-configuration-specific.
* A threshold must be recalibrated whenever the preprocessing, period grid, duration grid, score, search strategy, or null-generation procedure changes.
* The frozen reranker feature list is versioned in `configs/candidate_reranker_clean_v1.json`; changing it creates a new model version, not a silent update.
* Candidate reranker validation must split by `target_id`, not by candidate row.
* The rank-1-only reranker calibration is preliminary; use the top-k null/original calibration output for reranker false-alarm analysis.
* Committed summary files represent specific experiment configurations and should not be generalized beyond their target, quarter, and injection grid.
* Scientific conclusions should be based on larger population-scale, out-of-sample results rather than the current 50-star pilot alone.
