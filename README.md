# multimodel-transit-search

`multimodel-transit-search` is a multimodel benchmark for exoplanet transit detection. Its long-term goal is to compare statistical detectors, ARIMA/ARMA transformed-template methods, BLS- and TCF-style searches, machine-learning classifiers, deep-learning morphology models, and adaptive ensembles using reproducible Kepler injection-recovery experiments.

The current implementation focuses on validating the ARIMA/ARMA noise-model branch. It should be treated as a **Phase 1 single-target science-data prototype**, not as a completed transit-search system.

## Current Objective

The immediate objective is to determine whether a simple ARIMA-family model can reduce correlated variability in Kepler PDCSAP light curves while preserving transit evidence in a form suitable for downstream detection.

The current Phase 1 workflow is:

```text
Kepler PDCSAP flux
-> explicit cadence and gap representation
-> ARIMA candidate diagnostics
-> one-step-ahead innovations
-> transit-preservation checks
-> ARIMA-transformed-template matched-filter prototype
```

## Implemented Phase 1 Components

The package currently includes:

* Kepler PDCSAP light-curve loading and quality filtering.
* Regular cadence-grid construction with explicit gap and usability masks.
* Leakage-free flux normalization using a chronological fitting fraction.
* Full-grid-with-missing-values and longest-contiguous-segment ARIMA representations.
* ARIMA candidate fitting over explicit order grids.
* Convergence, fit-validity, coefficient-boundary, stationarity, and invertibility checks.
* Residual-whitening diagnostics using autocorrelation functions and Ljung–Box tests.
* Residual ACF diagnostics through lag 24, including summaries over transit-relevant lag ranges.
* Variance-stability diagnostics using rolling variance and ARCH testing.
* Simple forecast baselines based on the mean, median, and persistence.
* Basic synthetic box-transit preservation diagnostics.
* Fixed-parameter ARIMA application to injected light curves and transit templates.
* A scientific model-selection hierarchy that distinguishes the best-available model from a scientifically acceptable model.
* ADF and KPSS stationarity diagnostics for the exact series representation supplied to ARIMA selection.
* Candidate-family and differencing-alignment reporting for `d=0` and `d=1`.
* Machine-readable CSV, JSON, and Parquet outputs for the main single-target and smoke workflows.

Primary Phase 1 references:

* [Phase 1: Single-Target ARIMA Prototype](docs/phase1_single_target_arima.md)
* [Kepler Dataset Strategy](docs/kepler_dataset_strategy.md)

## Current Scientific Result

For the default single-target run, the current best-available full-gap model is:

```text
quality policy: default
mode: full_gap
model: ARIMA(1,1,0)
status: valid_but_residual_autocorrelation_remains
```

This model is **not scientifically accepted** as the final noise model because:

* the stationarity evidence is conflicting;
* the need for ordinary differencing remains unresolved;
* residual autocorrelation remains;
* residual variance is unstable;
* downstream transit-recovery performance has not yet been evaluated at controlled false-alarm rates.

The current full-gap stationarity result is:

```text
ADF: rejects the unit-root null
KPSS: rejects the level-stationarity null
joint conclusion: conflicting_rejections
recommended d: unresolved
selected-model differencing alignment: unresolved
differencing requires review: True
```

This result must not be interpreted as statistical justification for `d=1`.

`ARIMA(1,1,0)` is the highest-ranked admissible full-gap candidate under the current selection hierarchy. It is not yet a scientifically acceptable final model.

## Invalid Candidates

Non-converged, unstable, or otherwise invalid candidates may still report attractive values for:

```text
AIC
BIC
RMSE
MAE
negative log score
```

These values are retained for diagnostic purposes only. They are not considered trustworthy for model selection and cannot make an invalid candidate the winner.

For example, an ARIMA model may appear strong under forecast-fit metrics while failing convergence or numerical-stability checks. In this project, fit validity is evaluated before forecasting or information-criterion metrics.

## Model-Selection Hierarchy

ARIMA candidate selection follows a scientific hierarchy rather than a forecast-only ranking.

The broad order is:

```text
fit validity
-> residual-whitening constraints
-> variance stability
-> transit preservation
-> transit distortion
-> baseline-relative forecasting
-> model complexity
-> forecast metrics
-> information criteria
```

RMSE, MAE, negative log score, AIC, and BIC are therefore used as diagnostics or late-stage tie-breakers.

They are not the primary scientific objective because ARIMA is an intermediate noise transformation rather than the final forecasting product.

The intended scientific transformation is:

```text
correlated light curve
-> reduced predictable noise
-> preserved and detectable transformed transit evidence
```

## Outputs

The default single-target workflow writes:

```text
outputs/metrics/kic_11904151_q5_arima_candidates.csv
outputs/metrics/kic_11904151_q5_stationarity_diagnostics.csv
outputs/metrics/kic_11904151_q5_phase1_completion.json
outputs/processed/kic_11904151_q5_regularized_light_curve.parquet
outputs/processed/kic_11904151_q5_innovations.parquet
```

The smoke workflow writes the equivalent artifact family under:

```text
outputs/smoke/
```

These files serve different purposes:

```text
CSV      -> compact tables for manual inspection
JSON     -> structured run and scientific-readiness reports
Parquet  -> typed machine-readable data for downstream analysis
```

## Current Limitations

The current implementation remains limited in several important ways:

* It is a single-target, single-quarter prototype.
* The effect of gap representation is still under investigation.
* A complete BLS benchmark has not yet been implemented.
* A complete transformed-template TCF benchmark has not yet been implemented.
* Detection methods have not yet been compared at controlled false-alarm rates.
* A population-scale injection-recovery benchmark has not yet been performed.
* Feature-based machine learning has not yet been implemented.
* Deep-learning transit-morphology models have not yet been implemented.
* The adaptive multimodel ensemble has not yet been implemented.

## Next Implementation Phase

The immediate next task is to compare three explicit gap-handling representations:

```text
longest contiguous segment
full cadence grid with missing observations
interpolated full cadence grid
```

This comparison will determine whether the current preference for `d=1`, the selected ARIMA order, and the remaining residual autocorrelation depend materially on how gaps are represented.

Later phases will add:

```text
reproducible synthetic transit injection
PDCSAP + BLS baseline
robust detrending + BLS baseline
ARIMA-transformed-template TCF or matched filtering
empirical false-alarm calibration
multi-quarter testing
multi-target injection-recovery benchmarks
```

These are planned scientific extensions and should not be interpreted as completed functionality.
