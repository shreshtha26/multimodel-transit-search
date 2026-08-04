# multimodel-transit-search

`multimodel-transit-search` is a multimodel benchmark for exoplanet transit
detection. Its long-term goal is to compare statistical detectors,
ARIMA/ARMA transformed-template methods, BLS- and TCF-style searches,
machine-learning classifiers, deep-learning morphology models, and adaptive
ensembles using reproducible Kepler injection-recovery experiments.

The current implementation focuses on validating the ARIMA/ARMA noise-model
branch. It should be treated as a **Phase 1 single-target science-data
prototype**, not as a completed transit-search system.

## Current Objective

The immediate objective is to determine whether a simple ARIMA-family model can
reduce correlated variability in Kepler PDCSAP light curves while preserving
transit evidence in a form suitable for downstream detection.

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

- Kepler PDCSAP loading and quality filtering.
- Regular cadence-grid construction with explicit gap and usability masks.
- Leakage-free flux normalization on a chronological fit fraction.
- Gap-mode comparison across longest-contiguous, full-grid-with-missing-values, and interpolated full-grid representations.
- ARIMA candidate fitting over explicit order grids.
- Convergence, fit-validity, and coefficient-boundary checks.
- Residual whitening diagnostics using ACF and Ljung-Box tests.
- Residual ACF diagnostics through lag 24, including transit-relevant lag summaries.
- Variance-stability diagnostics using rolling variance and ARCH testing.
- Simple forecast baselines for mean, median, and persistence comparisons.
- Basic synthetic box-transit preservation diagnostics.
- Fixed-parameter ARIMA application to injected light curves and transit templates.
- Gap-mode transit-injection experiment with transformed-template depth, SNR retention, ingress/egress distortion, spurious peaks, and empirical FAR thresholds.
- Scientific model-selection hierarchy with best-available versus scientifically acceptable semantics.
- ADF/KPSS stationarity diagnostics for the exact series representation supplied to ARIMA selection.
- `d=0` versus `d=1` candidate-family and differencing-alignment reporting.
- ACF/PACF plot generation for gap representations where cadence-lag assumptions are defensible.
- Machine-readable CSV, JSON, and Parquet outputs for the single-target and smoke workflows.

Primary Phase 1 references:

* [Phase 1: Single-Target ARIMA Prototype](docs/phase1_single_target_arima.md)
* [Kepler Dataset Strategy](docs/kepler_dataset_strategy.md)

## Gap-Mode Comparison

The current default representation is `full_grid_missing`: the regular Kepler
cadence grid is preserved and missing cadences remain `NaN`. Statsmodels'
state-space ARIMA machinery handles those missing observations during fitting;
the pipeline does not invent flux values for the default mode.

The gap comparison now evaluates three explicit representations:

```text
longest_contiguous
    Uses only the longest uninterrupted usable cadence segment.
    Ordinary ACF/PACF lag interpretation is defensible, but most of the quarter
    is discarded.

full_grid_missing
    Preserves the full regular cadence grid and keeps missing cadences as NaN.
    It is the default modelling representation. Ordinary PACF is omitted for
    the missing-valued modelling series because compressing gaps would change
    the meaning of cadence lags.

interpolated_full_grid
    Fills only eligible short interior gaps with configured linear interpolation.
    It is a challenger mode, not the default. Any ACF/PACF plots are labelled as
    interpolation-dependent, and long gaps are not silently filled.
```

For the current default target and quarter, the gap-mode comparison reports:

```text
longest_contiguous:     ARIMA(2,1,0), stationarity_supported, recommended d=0, differencing conflict
full_grid_missing:      ARIMA(1,1,0), conflicting_rejections, recommended d unresolved
interpolated_full_grid: ARIMA(1,1,0), conflicting_rejections, recommended d unresolved
```

No gap representation/model combination is scientifically acceptable yet.
Residual autocorrelation and variance instability remain in every mode, and
transit-recovery performance has not yet been benchmarked at controlled
false-alarm rates. Interpolation can improve forecast metrics, but that is not
treated as scientific evidence that the interpolated mode is better.

## Current Injection Result

The first gap-mode injection experiment uses shared synthetic transit centers
from the longest clean segment and evaluates the selected ARIMA model for each
gap representation. It measures transformed-template depth retention, matched
filter SNR retention, ingress/egress distortion, spurious residual peaks, and
single-light-curve empirical false-alarm thresholds.

Current default run:

```text
full_grid_missing:      median SNR retention 0.63, recovery at FAR 1% = 1.00
interpolated_full_grid: median SNR retention 0.63, recovery at FAR 1% = 1.00
longest_contiguous:     median SNR retention 0.58, recovery at FAR 1% = 1.00
```

This is a narrow controlled-injection result, not a population-scale benchmark.
The full-grid and interpolated modes preserve detectability for these injected
events, but their transformed templates are strongly ingress/egress dominated,
which is expected for differenced ARIMA models and still needs downstream
detector benchmarking.

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

These values are retained for diagnostic purposes only. They are not considered
trustworthy for model selection and cannot make an invalid candidate the winner.

For example, an ARIMA model may appear strong under forecast-fit metrics while
failing convergence or numerical-stability checks. In this project, fit validity
is evaluated before forecasting or information-criterion metrics.

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

The default single-target run writes:

```text
outputs/metrics/kic_11904151_q5_arima_candidates.csv
outputs/metrics/kic_11904151_q5_stationarity_diagnostics.csv
outputs/metrics/kic_11904151_q5_phase1_completion.json
outputs/processed/kic_11904151_q5_regularized_light_curve.parquet
outputs/processed/kic_11904151_q5_innovations.parquet
```

The gap-mode comparison run writes:

```text
outputs/gap_modes/metrics/kic_11904151_q5_gap_mode_comparison.csv
outputs/gap_modes/metrics/kic_11904151_q5_gap_mode_report.json
outputs/gap_modes/metrics/kic_11904151_q5_gap_mode_plot_manifest.csv
outputs/gap_modes/processed/kic_11904151_q5_gap_mode_comparison.parquet
outputs/injections/metrics/kic_11904151_q5_gap_mode_injection_summary.csv
outputs/injections/metrics/kic_11904151_q5_gap_mode_injection_report.json
outputs/injections/processed/kic_11904151_q5_gap_mode_injection_results.parquet
```

The smoke workflow writes the same artifact family under:

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

- Single-target, single-quarter prototype.
- Gap representation is now explicitly compared, but none of the tested modes is scientifically accepted.
- No complete BLS benchmark yet.
- No transformed-template TCF benchmark yet.
- No false-alarm-controlled method comparison yet.
- No population-scale injection benchmark yet.
- No feature-based ML, deep-learning morphology model, or adaptive ensemble has been implemented yet.

## Next Implementation Phase

The immediate next task is:

```text
Implement the PDCSAP + BLS baseline.
```

Later phases should add biweight + BLS, ARIMA-transformed-template TCF,
survey-scale empirical false-alarm calibration, and larger injection-recovery
benchmarks. Those are planned scientific extensions, not completed functionality.
