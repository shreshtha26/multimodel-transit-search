# Kalman State-Space Baseline

Status: new single-target challenger branch for KIC 11904151, Quarter 5.

This branch adds a deliberately simple local-level state-space model as a background-noise challenger to ARIMA. It does not replace ARIMA, BLS, TCF, the injection grid, null generation, or the frozen reranker.

## Model

The hidden state is the slowly varying background flux level:

```text
background_t = background_{t-1} + process_noise_t
```

The observation equation is:

```text
normalized_flux_t = background_t + measurement_noise_t
```

Process noise controls how quickly the estimated background is allowed to drift from cadence to cadence. Larger process variance lets the background follow stellar or instrumental variability more aggressively, but can also suppress transit signals by tracking them.

Measurement noise controls how much the filter trusts each observed cadence. Larger measurement variance makes the background smoother and leaves more short-timescale structure in the residuals.

The detector input is the one-step residual:

```text
residual_t = normalized_flux_t - E[normalized_flux_t | previous cadences]
```

This residual may contain transit evidence because the predicted background is based on the previous filtered state, not on a transit-aware model.

## Missing Cadences

Missing or unusable cadences are not interpolated by the Kalman model.

For a missing cadence, the filter performs the prediction step but skips the observation update. The predicted background remains available, while the residual at that cadence is `NaN`.

## Difference From ARIMA

ARIMA models serial dependence directly in the observed series through autoregressive, differencing, and moving-average terms.

The Kalman branch instead defines an explicit latent background state and separates:

```text
state evolution noise
observation noise
one-step residuals
```

This makes the background interpretation simpler, but the first model is intentionally crude. It is only a baseline challenger, not a full astrophysical stellar-variability model.

## Current Experiment Flow

```text
Kepler PDCSAP
-> existing preprocessing
-> normalized flux
-> local-level Kalman background
-> residuals
-> BLS
-> TCF
```

The first scripts use the same target and injection philosophy as the existing single-target experiments:

```text
KIC 11904151
Kepler Quarter 5
periods: 2, 5, 10 days
durations: 2, 4, 8 hours
depths: 200, 500, 1000 ppm
epoch phases: 0.15, 0.45, 0.75
```

## Commands

```bash
python scripts/run_kalman_baseline.py
python scripts/run_kalman_null_calibration.py
python scripts/run_kalman_injection_grid.py
```

Run the null calibration before the injection grid because the injection grid reads the Kalman-specific 1% FAP thresholds.

To diagnose whether the local-level state is absorbing transits, run:

```bash
python scripts/run_kalman_sensitivity.py
```

This keeps the baseline outputs untouched and writes a separate process-noise sensitivity experiment.

## Outputs

```text
outputs/experiments/kalman_baseline/
outputs/experiments/kalman_null_calibration/
outputs/experiments/kalman_injection_grid/
outputs/experiments/kalman_sensitivity/
```

The baseline saves model diagnostics, residual parquet files, BLS periodograms, TCF periodograms, and top peaks.

The null calibration saves detector-specific Kalman-BLS and Kalman-TCF threshold tables.

The injection grid saves per-injection recovery rows and grouped recovery summaries by depth, duration, and period.

The sensitivity run saves:

```text
outputs/experiments/kalman_sensitivity/metrics/kic_11904151_q5_kalman_sensitivity_summary.csv
outputs/experiments/kalman_sensitivity/metrics/kic_11904151_q5_kalman_sensitivity_metadata.json
outputs/experiments/kalman_sensitivity/processed/kic_11904151_q5_kalman_transit_window_samples.csv
outputs/experiments/kalman_sensitivity/figures/kic_11904151_q5_kalman_transit_window_diagnostics.png
```

## Scientific Caution

The Kalman model should not be ranked as better than ARIMA or raw BLS from likelihood alone.

The relevant comparison is transit-search suitability:

```text
residual whitening
variance stability
transit depth retention
transit SNR retention
period recovery
recovery at calibrated FAP
```

Do not claim a winner until the null-calibrated injection-recovery outputs support it.
