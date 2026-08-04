# multimodel-transit-search

`multimodel-transit-search` is a research prototype for benchmarking multiple approaches to exoplanet transit detection in Kepler light curves.

The long-term goal is to compare classical statistical detectors, ARIMA/ARMA transformed-template methods, BLS- and TCF-style searches, probabilistic models, machine-learning classifiers, deep-learning morphology models, and adaptive ensembles using reproducible injection-recovery experiments.

> **Status:** Phase 1 engineering prototype. The current repository validates the ARIMA noise-model and transformed-template branch on one Kepler target and quarter. It is not yet a complete or scientifically validated transit-search system.

## Scientific Question

The immediate Phase 1 question is:

> Can an ARIMA-family model reduce predictable correlated variability in a Kepler PDCSAP light curve while retaining transit evidence in a form that remains detectable downstream?

```text
Kepler PDCSAP flux
-> quality filtering and explicit cadence grid
-> gap-aware light-curve representation
-> leakage-free normalization
-> ARIMA candidate fitting and diagnostics
-> one-step-ahead innovations
-> synthetic transit-injection tests
-> ARIMA-transformed-template matched filtering
-> scientific-readiness report
```

For observed normalized flux `y_t`, the innovation is

```text
e_t = y_t - y_hat(t | t - 1)
```

The prediction represents variability that the model considers predictable from the past. The innovation contains unpredictable variation, model error, and any transit evidence not absorbed by the model.

## Implemented in Phase 1

The current codebase includes:

- Kepler PDCSAP loading through `lightkurve`;
- configurable Kepler quality-mask policies;
- regular cadence-grid construction with explicit gap and usability masks;
- chronological, leakage-free normalization;
- longest-contiguous and full-grid-with-missing-values ARIMA representations;
- a challenger workflow for short-gap interpolation;
- configurable ARIMA order grids;
- convergence, numerical-validity, and coefficient-boundary checks;
- ADF and KPSS stationarity diagnostics;
- explicit `d=0` versus `d=1` differencing-alignment reporting;
- AIC, BIC, RMSE, MAE, and negative-log-score diagnostics;
- mean, median, and persistence forecast baselines;
- residual ACF and Ljung-Box whitening diagnostics;
- rolling-variance and ARCH-style variance-stability diagnostics;
- chronological-prefix and segment-level stability checks;
- synthetic box-transit injection;
- transit depth, SNR, timing, and morphology-preservation measurements;
- ARIMA-transformed-template matched filtering;
- blind single-event template scans;
- small multi-injection recovery experiments; and
- CSV, JSON, Parquet, and PNG experiment artifacts.

## Current Verified Result

The latest supplied successful single-target run used:

```text
target:             KIC 11904151
quarter:            Q5
raw cadences:       4492
quality policy:     default
selected mode:      full_gap
selected model:     ARIMA(1,1,0)
selection status:   valid_but_residual_autocorrelation_remains
```

### Stationarity

```text
ADF p-value:                      0.00134143
KPSS p-value:                     0.01
joint conclusion:                conflicting_rejections
recommended d:                   unresolved
selected differencing alignment: unresolved
differencing requires review:    True
```

ADF rejects a unit root, while KPSS rejects level stationarity. These conflicting results do **not** justify ordinary differencing. The selected `d=1` model is therefore the best-available admissible full-gap candidate under the current ranking rules, not a validated final noise model.

### Residual and transit diagnostics

The selected model remains scientifically inadequate because:

- residual autocorrelation remains;
- residual variance is unstable;
- the selected order changes with the gap representation;
- the selected model fails the current direct transit-preservation constraint; and
- the experiment is limited to one target, one quarter, and a small synthetic injection grid.

For the selected `ARIMA(1,1,0)` candidate, the current candidate table reports approximately:

```text
depth-retention fraction: 0.138
SNR-retention fraction:   0.094
transit-preservation test: failed
```

The limited transformed-template multi-injection scan nevertheless reports:

```text
rank-1 recovery rate: 1.000
```

These results answer different questions. Direct retention asks whether the injected transit remains box-like in the innovations. Transformed-template recovery asks whether the expected ARIMA-distorted shape can still identify the injected location. The current result suggests that transformed-template matching can recover the tested injections even though the original box morphology is strongly distorted.

This is encouraging engineering evidence, not population-level proof of improved transit detection.

### Phase status

```text
Phase 1 engineering complete:                 True
ARIMA branch accepted as final noise model:   False
```

“Engineering complete” means that the planned diagnostics and artifacts for the single-target prototype were generated. It does not mean that the selected ARIMA model passed the scientific acceptance criteria.

## Model-Selection Policy

ARIMA candidates are not ranked only by forecast accuracy.

```text
fit and numerical validity
-> residual-whitening adequacy
-> variance stability
-> transit preservation
-> transit distortion and transformed-template behavior
-> baseline-relative forecasting
-> model complexity
-> RMSE / MAE / negative log score
-> AIC / BIC
```

A non-converged or unstable model may still print attractive AIC, BIC, or forecast metrics. Those values are retained for debugging but are not considered trustworthy enough to make that candidate the winner.

`statsmodels` convergence warnings are therefore expected during candidate-grid evaluation and are intentionally surfaced.

## Gap Representations

### `longest_contiguous`

Uses only the longest uninterrupted usable segment.

- Cadence-lag ACF/PACF interpretation is defensible.
- Much of the quarter may be discarded.
- The latest run selected a different model than the full-gap representation, demonstrating gap sensitivity.

### `full_gap` / `full_grid_missing`

Preserves the regular cadence grid and leaves missing cadences as `NaN`.

- This is the default scientific representation.
- State-space ARIMA handles missing observations without inventing flux values.
- The single-target runner calls this mode `full_gap`.
- The dedicated gap-comparison runner calls the equivalent representation `full_grid_missing`.

### `interpolated_full_grid`

Fills only eligible short interior gaps under the configured interpolation policy.

- This is a challenger representation, not the default.
- Better forecast metrics do not establish better scientific validity.
- Results must be labelled as interpolation-dependent.

The latest verified orchestration completed all three current stages:

```text
single_target_arima:   success
gap_mode_comparison:  success
gap_mode_injection:   success
```

The current gap-mode comparison reports:

```text
longest_contiguous:     ARIMA(1,0,0), stationarity_supported, recommended d=0
full_grid_missing:      ARIMA(1,1,0), conflicting_rejections, recommended d unresolved
interpolated_full_grid: ARIMA(1,1,1), conflicting_rejections, recommended d unresolved
```

None of these representation/model combinations is scientifically accepted yet. The change in selected order across modes is useful evidence that gap handling affects the ARIMA decision.

## Installation

Python 3.11 or newer is required.

```bash
git clone https://github.com/shreshtha26/multimodel-transit-search.git
cd multimodel-transit-search

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Optional dependency groups are available for machine learning, deep learning, notebooks, parallel processing, and experiment tracking. These packages support the planned architecture; installing them does not mean that the corresponding model branches are already implemented.

## Running the Prototype

Unified experiment run:

```bash
PYTHONPATH=src python scripts/run_experiment.py
```

The default constants in `scripts/run_experiment.py` use KIC 11904151, Quarter 5, `configs/phase2.yaml`, and `outputs/experiments/`.

Run a stage directly only when debugging that specific stage:

```bash
PYTHONPATH=src python scripts/run_single_target_arima.py
PYTHONPATH=src python scripts/run_gap_mode_comparison.py
PYTHONPATH=src python scripts/run_gap_mode_injection_experiment.py
```

Configured target sample:

```bash
PYTHONPATH=src python scripts/run_target_sample.py
```

Tests:

```bash
PYTHONPATH=src pytest
```

## Outputs

The unified experiment writes:

```text
outputs/experiments/
├── single_target/   ARIMA candidate diagnostics and single-target artifacts
├── gap_modes/       gap-representation comparison tables and ACF/PACF plots
├── injections/      gap-mode injection and transformed-template diagnostics
└── records/         one JSON/CSV orchestration record
```

The single-target stage layout is:

```text
outputs/experiments/single_target/
├── figures/     diagnostic plots
├── metrics/     candidate tables, stationarity, injections, scans, and reports
└── processed/   regularized light curve and innovations
```

Important artifacts include:

```text
metrics/kic_11904151_q5_arima_candidates.csv
metrics/kic_11904151_q5_stationarity_diagnostics.csv
metrics/kic_11904151_q5_transit_preservation.csv
metrics/kic_11904151_q5_transformed_template_match.csv
metrics/kic_11904151_q5_template_scan.csv
metrics/kic_11904151_q5_multi_injection_recovery.csv
metrics/kic_11904151_q5_phase1_completion.json
processed/kic_11904151_q5_regularized_light_curve.parquet
processed/kic_11904151_q5_innovations.parquet
```

```text
CSV      -> compact diagnostic and result tables
JSON     -> structured summaries and scientific-readiness reports
Parquet  -> typed time-series data for downstream analysis
PNG      -> visual scientific diagnostics
```

## Repository Structure

```text
configs/                  experiment and target-sample configuration
docs/                     scientific and implementation notes
notebooks/                exploratory analysis
outputs/                  generated experiment artifacts
scripts/                  command-line experiment runners
src/adaptive_transit/     reusable package code
tests/                    automated tests
pyproject.toml            package metadata and dependencies
```

The distribution name in `pyproject.toml` is `multi-model-transit-search`; the import package is `adaptive_transit`.

## Planned Multi-Model Architecture

The planned system separates background/noise modelling from transit detection.

```text
input light curve
-> preprocessing, quality, cadence, and gap features
-> parallel background/noise models
-> residual or detrended representations
-> parallel transit detectors
-> candidate-level features and diagnostics
-> adaptive selector or calibrated ensemble
-> final transit score and explanation
```

| Memory family | Background/noise branch | Transit branch |
|---|---|---|
| Linear | ARIMA/ARMA, linear state-space models | BLS, linear matched filters |
| Probabilistic | Gaussian processes, Kalman models | Bayesian transit models |
| Nonlinear | nonlinear autoregression and engineered temporal models | XGBoost/LightGBM candidate classifiers |
| Convolutional | 1D CNN or temporal convolutional network | local-window or phase-folded morphology CNN |
| Attention | time-series Transformer or patch encoder | attention-based event classifier |

XGBoost is most naturally used as a candidate classifier or meta-selector over engineered light-curve, detector, and diagnostic features. It should not be assumed to choose the correct time-series model directly from raw flux without intermediate features and validation.

## Current Limitations

Not yet implemented or validated:

- complete PDCSAP + BLS and robust-detrending + BLS baselines;
- transformed-template TCF search;
- population-scale injection-recovery benchmarking;
- matched empirical false-alarm-rate comparison;
- systematic tests across stellar variability, depth, duration, cadence, and gap regimes;
- feature-based ML, convolutional, probabilistic, and attention branches;
- calibrated uncertainty and out-of-distribution diagnostics; and
- an adaptive model selector or ensemble.

## Immediate Next Step

```text
Kepler PDCSAP
-> documented baseline preprocessing
-> Box Least Squares period search
-> injection recovery
-> empirical false-alarm calibration
```

This establishes a standard transit-search reference before expanding the ARIMA-transformed-template branch or training an adaptive ensemble.

## Current Conclusion

Phase 1 has built a reproducible single-target experiment framework and shown that an ARIMA-transformed template can recover the tested injected locations despite substantial distortion of the original box-shaped transit.

However, the selected `ARIMA(1,1,0)` model is not scientifically accepted because differencing remains unresolved, residual dependence remains, variance is unstable, and direct transit preservation is weak.

> The ARIMA branch is implemented and diagnostically informative, but it has not yet been validated as a superior or generally reliable transit-search preprocessing method.

## Documentation

- [Phase 1: Single-Target ARIMA Prototype](docs/phase1_single_target_arima.md)
- [Kepler Dataset Strategy](docs/kepler_dataset_strategy.md)
