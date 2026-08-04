# multi-model-transit-search

`multimodel-transit-search`, is a multimodel
exoplanet transit-search benchmark. The long-term goal is to compare statistical
detectors, ARIMA/ARMA-transformed-template methods, BLS/TCF-style searches,
machine-learning classifiers, deep-learning morphology models, and adaptive
ensembles on reproducible Kepler injection-recovery experiments.

The current implementation is still focused on validating the ARIMA/ARMA
noise-model branch. It should be read as a Phase 1 single-target science-data
prototype, not as a completed transit-search system.

## Current Objective

The immediate objective is to determine whether a simple ARIMA-family model can reduce correlated Kepler PDCSAP variability while preserving transit evidence in a form that can be searched downstream.

Current Phase 1 target:

```text
Kepler PDCSAP flux
-> explicit cadence and gap representation
-> ARIMA candidate diagnostics
-> one-step-ahead innovations
-> transit-preservation checks
-> ARIMA-transformed-template matched-filter prototype
```

## Implemented Phase 1 Components

Implemented package functionality includes:

- Kepler PDCSAP loading and quality filtering.
- Regular cadence-grid construction with explicit gap and usability masks.
- Leakage-free flux normalization on a chronological fit fraction.
- Full-grid-with-missing-values and longest-contiguous-segment ARIMA representations.
- ARIMA candidate fitting over explicit order grids.
- Convergence, fit-validity, and coefficient-boundary checks.
- Residual whitening diagnostics using ACF and Ljung-Box tests.
- Residual ACF diagnostics through lag 24, including transit-relevant lag summaries.
- Variance-stability diagnostics using rolling variance and ARCH testing.
- Simple forecast baselines for mean, median, and persistence comparisons.
- Basic synthetic box-transit preservation diagnostics.
- Fixed-parameter ARIMA application to injected light curves and transit templates.
- Scientific model-selection hierarchy with best-available versus scientifically acceptable semantics.
- ADF/KPSS stationarity diagnostics for the exact series representation supplied to ARIMA selection.
- `d=0` versus `d=1` candidate-family and differencing-alignment reporting.
- Machine-readable CSV, JSON, and Parquet outputs for the single-target and smoke workflows.

Primary Stage 1 reference:

- [Phase 1: Single-Target ARIMA Prototype](docs/phase1_single_target_arima.md)
- [Kepler Dataset Strategy](docs/kepler_dataset_strategy.md)

## Current Scientific Result

For the default single-target run, the current best-available full-gap model is:

```text
quality policy: default
mode: full_gap
model: ARIMA(1,1,0)
status: valid_but_residual_autocorrelation_remains
```

It is not scientifically accepted as a final noise model because:

- stationarity evidence is conflicting;
- ordinary differencing remains unresolved;
- residual autocorrelation remains;
- variance instability remains;
- downstream transit-recovery performance has not yet been established at controlled false-alarm rates.

The current full-gap stationarity result is:

```text
ADF rejects unit-root null
KPSS rejects level-stationarity null
joint conclusion: conflicting_rejections
recommended d: unresolved
selected-model differencing alignment: unresolved
differencing requires review: True
```

Do not interpret this as statistical justification for `d=1`. The selected
ARIMA(1,1,0) is the best-available admissible full-gap candidate under the
current hierarchy, not a scientifically accepted final model.

## Invalid Candidates

Non-converged or unstable candidates may still report attractive AIC, BIC, RMSE,
MAE, or negative-log-score values. Those values are recorded for diagnostics
only. They are not trusted for model selection and cannot make an invalid
candidate the winner.

For example, an ARIMA candidate can appear strong under forecast-fit metrics while failing convergence or stability checks. In this project, fit validity is evaluated before forecast metrics.

## Selection Hierarchy

Current ARIMA selection is hierarchical. The broad order is:

```text
fit validity
-> whitening constraints
-> variance stability
-> transit preservation
-> transit distortion
-> baseline-relative forecasting
-> model complexity
-> forecast metrics
-> information criteria
```

RMSE, MAE, negative log score, AIC, and BIC are diagnostics or late tie-breakers.
They are not the primary scientific objective because ARIMA is an intermediate
noise transformation, not the final forecasting product.

The scientific objective is:

```text
correlated light curve
-> whitened noise
-> detectable transformed transit evidence
```

## Outputs

The default run writes:

```text
outputs/metrics/kic_11904151_q5_arima_candidates.csv
outputs/metrics/kic_11904151_q5_stationarity_diagnostics.csv
outputs/metrics/kic_11904151_q5_phase1_completion.json
outputs/processed/kic_11904151_q5_regularized_light_curve.parquet
outputs/processed/kic_11904151_q5_innovations.parquet
```

The smoke workflow writes the same artifact family under:

```text
outputs/smoke/
```

## Current Limitations

- Single-target, single-quarter prototype.
- Gap representation is still under investigation.
- No complete BLS benchmark yet.
- No transformed-template TCF benchmark yet.
- No false-alarm-controlled method comparison yet.
- No population-scale injection benchmark yet.
- No feature-based ML, deep-learning morphology model, or adaptive ensemble has been implemented yet.

## Next Implementation Phase

The immediate next task is:

```text
Compare longest-contiguous, full-grid-with-missing-values,
and interpolated gap-handling representations.
```

Later phases should add reproducible transit injection, BLS, transformed-template
TCF, empirical false-alarm calibration, and larger injection-recovery benchmarks.
Those are planned scientific extensions, not completed functionality.
