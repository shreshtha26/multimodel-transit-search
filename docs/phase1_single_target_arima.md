# Multi-Model Transit Search Phase 1: Single-Target ARIMA Prototype

> **Status:** Historical Phase 1 record. Engineering-complete as a single-target prototype, but not a scientifically accepted final ARIMA noise model. Later BLS, ARIMA-TCF, multi-star, and reranker work is summarized in the [main README](../README.md).

This phase uses one Kepler PDCSAP quarter to build and audit the first ARIMA-based noise-model and transit-preservation workflow in `multi-model-transit-search`:

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

The purpose of this phase was not to establish ARIMA as the final Kepler background model. It was to determine whether an ARIMA-family transformation could reduce predictable structure while preserving transit information in a form that remained detectable downstream.

## Reference Target

```text
target: KIC 11904151
quarter: Kepler Quarter 5
flux product: PDCSAP
quality policy: default
gap representation: full_gap
selected order: ARIMA(1, 1, 0)
```

## Reproducing the Phase 1 Workflows

Run the individual Phase 1 scripts from the repository root:

```bash
python scripts/run_single_target_arima.py
python scripts/run_gap_mode_comparison.py
python scripts/run_gap_mode_injection_experiment.py
```

The same stages can also be called through the unified experiment runner by selecting them explicitly:

```python
from pathlib import Path
from scripts.run_experiment import run_experiment

run_experiment(
    target_id="11904151",
    quarter=5,
    output_dir=Path("outputs/experiments"),
    stages=["single_target_arima", "gap_mode_comparison", "gap_mode_injection"],
)
```

Explicit stage selection matters because the unified runner is also used by later project stages.

## ARIMA Selection Policy

The selector does not treat RMSE, MAE, AIC, or BIC as the primary scientific objective.

The intended hierarchy is approximately:

```text
fit and numerical validity
-> residual whitening
-> variance stability
-> transit preservation
-> transit distortion
-> baseline-relative forecasting
-> model complexity
-> forecast metrics
-> information criteria
```

Non-converged candidates can still emit RMSE, AIC, and BIC values, but those values are treated as diagnostics from an unsuccessful fit rather than as trustworthy selection evidence.

Differenced candidates are also flagged for review because differencing changes transit morphology. Transit preservation must therefore be judged in the transformed space used by the downstream detector, not only by inspecting the original box shape.

## Stationarity and Differencing

Stationarity diagnostics are run on the actual modelling series used by each ARIMA representation.

For the reference full-gap series, the saved Phase 1 result is:

```text
original full-gap conclusion: conflicting_rejections
recommended d: unresolved
selected-model differencing alignment: unresolved
differencing requires review: True
```

ADF rejects the unit-root null while KPSS rejects the level-stationarity null.

This does **not** statistically justify `d=1`.

It means that `ARIMA(1,1,0)` remains the best-available full-gap candidate under the current selection hierarchy, not a scientifically validated final noise model.

## Completion Criteria

The machine-readable completion report is written to:

```text
outputs/metrics/kic_11904151_q5_phase1_completion.json
```

The Phase 1 engineering criteria include:

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

These criteria describe implementation completeness. They do not imply scientific adequacy of the selected ARIMA model.

## Scientific Result

The selected full-quarter ARIMA baseline retains important limitations:

```text
selected status: valid_but_residual_autocorrelation_remains
stationarity evidence is conflicting
differencing remains unresolved
residual autocorrelation remains
variance instability remains
raw innovation-space transit preservation fails
```

The key Phase 1 observation was that the transit template must be transformed consistently with the data transformation.

For the reference injection:

```text
raw flux + unchanged box score:             32.77
ARIMA innovations + unchanged box score:     9.09
ARIMA innovations + transformed template:   20.90
```

The unchanged box performs poorly in innovation space because ARIMA changes the morphology of the transit signal. Applying the same transformation to the template recovers part of the lost matched-filter response.

The blind single-event scan searched 251 trial centers and recovered the injected center at rank 1:

```text
injected center: 16518
best detected center: 16518
best injected-neighborhood rank: 1
```

This demonstrates that transformed-template matching is operational for the reference injection. It does not establish population-level detection performance.

## Gap-Representation Experiment

The Phase 1 gap-mode comparison evaluates:

```text
longest_contiguous
full_grid_missing
interpolated_full_grid
```

For the reference target:

- the longest contiguous representation gives stationarity evidence supporting `d=0`, but the model-selection procedure still chooses a differenced ARIMA challenger;
- the full-grid missing-data and interpolated representations retain unresolved stationarity evidence and select `ARIMA(1,1,0)`;
- no gap representation is considered scientifically satisfactory under the current diagnostics.

The narrow gap-mode injection experiment uses shared synthetic transit centers across representations and measures:

- transformed-template depth retention;
- matched-filter SNR retention;
- ingress/egress distortion;
- spurious residual peaks;
- empirical single-light-curve false-alarm thresholds.

Saved reference results:

```text
full_grid_missing:      median SNR retention 0.63, FAR 1% recovery 1.00
interpolated_full_grid: median SNR retention 0.63, FAR 1% recovery 1.00
longest_contiguous:     median SNR retention 0.58, FAR 1% recovery 1.00
```

These numbers apply only to the narrow Phase 1 injection experiment.

They do not make the ARIMA model scientifically acceptable. They show that the selected transformations preserved detectability for those injections while still distorting transit morphology, particularly around ingress and egress.

## Multi-Injection Recovery Output

The selected-model recovery grid over multiple synthetic transit depths, durations, and clean injection centers writes:

```text
outputs/metrics/kic_11904151_q5_multi_injection_recovery.csv
outputs/metrics/kic_11904151_q5_multi_injection_recovery_summary.json
```

## Phase 1 Conclusion

Phase 1 is engineering-complete because it provides a reproducible single-target workflow that:

1. makes cadence gaps and preprocessing explicit;
2. compares ARIMA candidates under a documented hierarchy;
3. exposes unresolved stationarity, whitening, and variance-stability problems;
4. measures transit preservation rather than relying only on forecasting metrics;
5. demonstrates why the transit template must be transformed together with the data;
6. recovers a reference injected event in a blind transformed-template scan.

The correct scientific conclusion is therefore:

> `ARIMA(1,1,0)` is a best-available Phase 1 challenger under the tested full-gap setup, not a validated optimal Kepler noise model.

## Relationship to Later Project Stages

Phase 1 is now a historical foundation rather than the current endpoint of the repository.

Subsequent work has implemented:

```text
direct PDCSAP + BLS baseline
-> BLS injection-recovery and empirical null calibration
-> periodic TCF search on ARIMA innovations
-> 50-star BLS/ARIMA-TCF pilot
-> merged detector candidate sets
-> leakage-audited candidate reranking
-> frozen clean_reranker_v1
-> matched top-k null/original reranker calibration
```

For the current project architecture, results, limitations, and next milestones, see the [main README](../README.md). For how target samples are organized and scaled, see [Kepler Dataset Strategy](kepler_dataset_strategy.md).
