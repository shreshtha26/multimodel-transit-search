# BLS Baseline v1

## Configuration

- Target: KIC 11904151
- Quarter: 5
- Null trials: 1,000
- False-alarm level: 1%
- Block size: 24 cadences
- Random seed: 123
- Period grid: 1–15 days
- Number of trial periods: 1,000
- Duration grid: 1.5–10 hours
- Number of durations: 8
- Objective: BLS SNR
- Period-match tolerance: 2%, including harmonics

## Overall Results

- Injection count: 81
- Period-match rate: 88.9%
- Recovery rate at 1% FAP: 82.7%

A recovery requires both:

1. The recovered period matches the injected period or an accepted harmonic.
2. The maximum BLS power exceeds the calibrated 1% FAP threshold.

## Strong Regimes

BLS recovered all tested injections with:

- Transit depth of 500 ppm or 1,000 ppm
- Transit duration of 8 hours

## Weak Regime

The main weak regime was:

- Transit depth: 200 ppm
- Transit duration: 2 hours
- Period: 5–10 days

All failed injections occurred at the shallowest tested depth of 200 ppm.

## Failure Analysis

Fourteen of 81 injections failed recovery at 1% FAP.

Nine of these failures selected the same competing period:

- 11.720721 days

This peak appears to dominate the BLS search when the injected transit is too weak.

Some injections recovered the correct period but remained below the 1% FAP threshold. These represent insufficient detection significance rather than incorrect period selection.

## Scientific Conclusion

BLS provides a strong baseline for moderate and deep transits but loses sensitivity for shallow, short-duration and relatively long-period signals.

The frozen BLS benchmark is:

- Period-match rate: 88.9%
- Recovery rate at 1% FAP: 82.7%

The 0.1% FAP result is not treated as final because 1,000 null trials are insufficient for stable calibration of the 0.1% tail.

## Next Comparison

The next benchmark will compare:

- Standard light curve → BLS
- ARIMA innovations → TCF

Each detector will use its own independently calibrated 1% FAP threshold.