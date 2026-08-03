# multi-model-transit-search Phase 1: Single-Target ARIMA Prototype

Status: complete as a single-target prototype.

This phase uses one Kepler PDCSAP quarter to build and audit the first multi-model-transit-search
noise-model and detection experiment:

```text
PDCSAP flux
-> explicit cadence grid and masks
-> leakage-free normalization
-> ARIMA candidate comparison
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

Selected baseline:

```text
quality policy: permissive
mode: full_gap
order: ARIMA(1, 1, 0)
```

The default run evaluates all default ARIMA orders, compares strict/default/
permissive quality masks, fits full-quarter NaN-gap models, fits contiguous
segment models, records residual diagnostics, compares simple baselines, checks
coefficient boundaries, tests order stability, injects a synthetic transit, and
runs a blind transformed-template scan.

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
```

## Scientific Result

The selected ARIMA baseline still has documented limitations:

```text
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

Phase 1 is complete because the project now has a reproducible single-target
pipeline that exposes ARIMA's noise-model limitations while also demonstrating
that ARIMA-transformed templates can recover an injected transit in a blind
single-event scan.

The next phase should scale this from one injected event to injection-recovery
experiments across transit depths, durations, periods, targets, and noise
regimes.

The code now includes the first selected-model recovery grid over multiple
synthetic transit depths, durations, and clean injection centers. The output is:

```text
outputs/metrics/kic_11904151_q5_multi_injection_recovery.csv
outputs/metrics/kic_11904151_q5_multi_injection_recovery_summary.json
```
