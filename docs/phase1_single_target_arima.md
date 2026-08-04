# multi-model-transit-search Phase 1: Single-Target ARIMA Prototype

Status: engineering-complete as a single-target prototype; not yet a final adequate noise model.

This phase uses one Kepler PDCSAP quarter to build and audit the first multi-model-transit-search
noise-model and detection experiment:

```text
PDCSAP flux
-> explicit cadence grid and masks
-> leakage-free normalization
-> ARIMA candidate comparison
-> hierarchical ARIMA selection constraints
-> one-step-ahead innovations
-> rolling-scale standardized innovations
-> synthetic transit preservation test
-> ARIMA-transformed transit template
-> blind single-event matched-filter scan
-> multi-injection recovery summary
```

## Latest Full Run

Command:

```bash
.venv/bin/python scripts/run_single_target_arima.py --target-id 11904151 --quarter 5
```

Selected baseline after hierarchical constraints:

```text
quality policy: default
mode: full_gap
order: ARIMA(1, 1, 0)
```

The default run evaluates all default ARIMA orders, compares strict/default/
permissive quality masks, fits full-quarter NaN-gap models, fits contiguous
segment models, records residual diagnostics, compares simple baselines, checks
coefficient boundaries, tests order stability, injects a synthetic transit, and
runs a blind transformed-template scan.

The selector does not treat RMSE/MAE as the primary objective. It first checks
fit validity, residual-whitening constraints, variance stability and injected
transit preservation. Forecast metrics, information criteria and simplicity are
used only as tie-breakers among models in the same constraint tier.

Non-converged candidates can still emit RMSE, AIC and BIC values, but those
values are treated as diagnostics from a failed fit rather than trustworthy
selection evidence. Differenced candidates are also flagged for review because
first differencing changes transit morphology; preservation must be judged in
the transformed-template space used by the detector.

Stationarity diagnostics are now run in package code on the exact modelling
series used by each ARIMA representation. For the current default full-gap
series, ADF rejects the unit-root null and KPSS rejects the level-stationarity
null:

```text
original full-gap conclusion: conflicting_rejections
recommended d: unresolved
selected-model differencing alignment: unresolved
differencing requires review: True
```

This does not statistically justify `d=1`. It means ARIMA(1, 1, 0) remains a
best-available full-gap candidate under the current hierarchy, not a final
scientifically accepted noise model.

## Completion Criteria

The machine-readable completion report is written to:

```text
outputs/metrics/kic_11904151_q5_phase1_completion.json
```

All required implementation criteria are true:

```text
preprocessing is explicit and reproducible
missing cadences remain explicit
normalization is leakage-free
variance behavior is characterized
order selection runs across folds and segments
full-quarter NaN-gap ARIMA is evaluated
simple baselines are compared
coefficient boundary diagnostics are recorded
residual failure modes are documented
standardized innovations are saved
injected transit preservation is measured
ARIMA-transformed-template matching is measured
blind single-event scan is measured
hierarchical model selection is explicit
stationarity diagnostics are recorded
```

## Scientific Result

The selected full-quarter ARIMA baseline still has documented limitations:

```text
selected status: valid_but_residual_autocorrelation_remains
stationarity evidence is conflicting
differencing remains unresolved
residual autocorrelation remains
variance instability remains
raw innovation-space transit preservation fails
```

However, the transformed-template detector behaves as intended:

```text
raw flux + unchanged box score:              32.77
ARIMA innovations + unchanged box score:      9.09
ARIMA innovations + transformed template:    20.90
```

The blind scan searched 251 trial centers and recovered the injected center as
rank 1:

```text
injected center: 16518
best detected center: 16518
best injected-neighborhood rank: 1
```

## Phase 1 Conclusion

Phase 1 is engineering-complete because the project now has a reproducible
single-target pipeline that exposes ARIMA's noise-model limitations, ranks
candidate noise models with an explicit hierarchy, and demonstrates that
ARIMA-transformed templates can recover an injected transit in a blind
single-event scan.

It is not yet a final adequate ARIMA noise model because the selected full-gap
candidate still has unresolved differencing evidence and fails whitening and
variance-stability constraints. That is the correct scientific conclusion from
this run.

The package now includes a dedicated gap-mode comparison that evaluates:

```text
longest contiguous segment
full cadence grid with missing values
interpolated cadence grid
```

For the current target, the longest contiguous segment supports `d=0` by ADF/KPSS
but still selects a differenced ARIMA challenger, while full-grid and
interpolated modes keep the unresolved stationarity conclusion and select
ARIMA(1, 1, 0). No gap mode is scientifically acceptable yet.

The repository now includes the first narrow gap-mode injection experiment. It
injects shared synthetic transit centers into each gap representation and
measures transformed-template depth retention, matched-filter SNR retention,
ingress/egress distortion, spurious residual peaks and empirical
single-light-curve false-alarm thresholds.

Current result:

```text
full_grid_missing:      median SNR retention 0.63, FAR 1% recovery 1.00
interpolated_full_grid: median SNR retention 0.63, FAR 1% recovery 1.00
longest_contiguous:     median SNR retention 0.58, FAR 1% recovery 1.00
```

This does not make the ARIMA model scientifically acceptable. It means the
selected ARIMA transformations preserve detectability for this narrow injected
sample while still distorting transit morphology, especially at ingress and
egress. The next implementation phase should be the PDCSAP + BLS baseline.

The code now includes the first selected-model recovery grid over multiple
synthetic transit depths, durations, and clean injection centers. The output is:

```text
outputs/metrics/kic_11904151_q5_multi_injection_recovery.csv
outputs/metrics/kic_11904151_q5_multi_injection_recovery_summary.json
```
