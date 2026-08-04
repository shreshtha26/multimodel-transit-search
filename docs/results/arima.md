# ARIMA Phase 1 Snapshot v1

## Scope

This experiment evaluates whether an ARIMA-family background model can reduce predictable Kepler PDCSAP variability while preserving transit evidence for downstream transformed-template detection.

## Dataset

* Target: KIC 11904151
* Quarter: 5
* Loaded cadences: 4,492
* Selected quality policy: default
* Selected gap representation: full gap

## Selected Model

* Selected order: ARIMA(1,1,0)
* Model status: valid but residual autocorrelation remains
* Fit converged: yes
* Differencing requires review: yes
* Selected order is gap-sensitive: yes

The model is the highest-ranked admissible candidate tested for the selected representation. It is not established as an optimal or scientifically validated background model.

## Stationarity

* ADF p-value: 0.00134143
* KPSS p-value: 0.01
* Conclusion: conflicting rejections
* Recommended differencing order: unresolved

The evidence does not clearly justify whether differencing with `d=1` is scientifically appropriate.

## Residual Diagnostics

The selected model does not fully whiten the light curve.

Observed limitations include:

* Residual autocorrelation remains
* Ljung–Box tests reject white residuals
* Residual variance is unstable
* Model selection changes with gap representation

Therefore, the innovations still contain predictable temporal structure.

## Transit Preservation

For the selected ARIMA(1,1,0) full-gap model:

* Transit depth retention: approximately 13.8%
* Transit SNR retention: approximately 9.4%
* Transit-preservation constraint failed

The original box-shaped transit is strongly altered by the ARIMA transformation.

This does not necessarily mean the transit evidence is destroyed. It means the signal must be searched using an ARIMA-transformed template rather than the original box template.

## Transformed-Template Result

* Multi-injection rank-1 recovery rate: 100%

This result is encouraging, but it is not yet a controlled false-alarm benchmark. Rank-1 recovery alone does not establish scientific detection performance.

## Completion Status

* Phase 1 engineering complete: yes
* Phase 1 scientifically ready for Phase 2: no

## Scientific Conclusion

ARIMA(1,1,0) can be used as the current engineering model for producing innovations and testing transformed-template detection.

However, it should not yet be presented as the scientifically optimal noise model because:

* Differencing remains unresolved
* Residual autocorrelation remains
* Residual variance is unstable
* Transit depth and SNR retention are low
* Model selection is sensitive to gap treatment
* False-alarm-controlled periodic TCF validation has not yet been completed

## Current Role in the Project

The current pipeline is:

```text
Kepler PDCSAP light curve
→ explicit cadence and gap handling
→ ARIMA(1,1,0)
→ one-step-ahead innovations
→ transformed transit template
→ periodic TCF
```

## Next Required Test

The next scientific benchmark is:

```text
Standard light curve → BLS
versus
ARIMA innovations → TCF
```

Both detectors must be evaluated using independently calibrated 1% false-alarm thresholds and the same injection grid.

## Status Label

Use:

```text
ARIMA Phase 1 engineering snapshot
```

Do not yet use:

```text
Frozen scientific ARIMA baseline
```
