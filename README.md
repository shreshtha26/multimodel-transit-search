# Multi-Model Transit Search

A reproducible research prototype for comparing exoplanet transit-search methods on Kepler light curves.

The project currently implements and evaluates two detector branches:

```text
Kepler PDCSAP flux
├── direct normalized-flux branch
│   └── Box Least Squares (BLS)
└── autoregressive-transformation branch
    ├── ARIMA diagnostics and model selection
    ├── one-step-ahead innovations
    └── periodic Transit Comb Filter (TCF)
```

The longer-term goal is to compare statistical, machine-learning, and deep-learning detectors at matched false-alarm rates and then combine their evidence using an adaptive ensemble.

> **Research status:** active single-target prototype. The current results demonstrate an end-to-end benchmarking framework, not a validated production transit-search pipeline or evidence that one method is scientifically superior.

## Current Research Questions

The repository is being developed around four questions:

1. Can an ARIMA-family model reduce predictable correlated variability without destroying transit information?
2. How sensitive is ARIMA selection to the representation of missing Kepler cadences?
3. How does direct BLS detection compare with ARIMA-transformed TCF detection on the same injection grid?
4. Can all methods eventually be compared at controlled empirical false-alarm rates across many stars and noise regimes?

## Current Scope

The committed benchmark currently uses:

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

### What these results do not establish

The BLS and TCF numbers should not yet be interpreted as a final head-to-head comparison because:

* only one target and one quarter are used;
* the detectors currently use different period-grid resolutions;
* TCF uses a coarse-to-fine search while BLS evaluates its configured grid directly;
* the TCF threshold is conditional on a fitted ARIMA transformation;
* ARIMA convergence is currently poor across the TCF injection grid;
* accepted harmonic matches and exact-period matches are reported separately for TCF;
* the injection grid is small and contains only box-shaped signals;
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

## Next Scientific Milestones

1. Stabilize ARIMA fitting and define a scientifically defensible response to non-convergence.
2. Validate coarse-to-fine TCF against exhaustive search across injected and no-injection cases.
3. Run BLS and TCF with harmonized period grids, search ranges, recovery definitions, and null-calibration scopes.
4. Scale the experiment to a multi-target Kepler sample.
5. Add target-level parallel execution with deterministic random seeds and failure logging.
6. Compare additional background models such as robust biweight/spline detrending and Gaussian-process or state-space models.
7. Add TLS and other transit-shaped statistical detectors.
8. Introduce feature-based ML and one-dimensional deep-learning morphology models only after the statistical benchmark is stable.
9. Evaluate all detectors at matched false-alarm rates and characterize performance by depth, duration, period, stellar variability, gaps, and cadence.
10. Train an adaptive ensemble or model router using out-of-sample detector evidence.

## Documentation

* [Phase 1: Single-Target ARIMA Prototype](docs/phase1_single_target_arima.md)
* [Kepler Dataset Strategy](docs/kepler_dataset_strategy.md)

## Reproducibility Notes

* Random seeds should be recorded for all null and injection experiments.
* Null thresholds are detector- and search-configuration-specific.
* A threshold must be recalibrated whenever the preprocessing, period grid, duration grid, score, search strategy, or null-generation procedure changes.
* Committed summary files represent specific experiment configurations and should not be generalized beyond their target, quarter, and injection grid.
* Scientific conclusions should be based on population-scale, out-of-sample results rather than the current single-target prototype.
