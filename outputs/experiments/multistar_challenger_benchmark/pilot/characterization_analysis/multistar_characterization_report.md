# Multistar Characterization Effects

Benchmark directory: `outputs/experiments/multistar_challenger_benchmark/pilot`
Targets: 10
Injections: 80
Recovery metric: per-star 1% FAP harmonic recovery
Characterization workers: 8 with 2 CPU cores reserved.

## Question Summary

- Do high-ACF stars benefit more from ARIMA? higher ACF stars show larger ARIMA lift.
- Do smooth long-timescale stars benefit more from GP? longer ACF timescale stars show larger GP lift.
- Do state-space-like variance/drift stars benefit more from Kalman? higher variance drift stars show larger Kalman lift.
- Does high spectral concentration predict GP success? no clear split effect.
- Does whitening improve ACF while damaging ARIMA transit SNR? larger ARIMA ACF reduction tracks higher transit SNR retention.
- Does whitening improve ACF while damaging GP transit SNR? larger GP ACF reduction tracks higher transit SNR retention.
- Does whitening improve ACF while damaging Kalman transit SNR? larger Kalman ACF reduction tracks higher transit SNR retention.
- Are there stars for which raw BLS is preferable? raw BLS was preferable or tied for 3 stars; strictly preferable for 0 stars.

## Star Table Preview

| star            |   acf_timescale_days |   spectral_strength |   variance_drift |   gap_fraction |   gp_improvement |   kalman_improvement |   arima_improvement |
|:----------------|---------------------:|--------------------:|-----------------:|---------------:|-----------------:|---------------------:|--------------------:|
| KIC 1026957 Q5  |            3.08416   |          0.739133   |          5.75311 |      0.0321536 |            0.25  |                1     |               0.75  |
| KIC 11086270 Q5 |            0.0204343 |          0.00911846 |          7.13514 |      0.0319379 |            0     |                0     |              -0.375 |
| KIC 1161345 Q5  |            1.14434   |          0.402662   |         11.0216  |      0.0321536 |            0.5   |                0.25  |               0.125 |
| KIC 11904151 Q5 |            1.78217   |          0.243993   |          7.95972 |      0.0319379 |            0     |                0     |              -0.125 |
| KIC 2302548 Q5  |            2.60134   |          0.294578   |         25.3192  |      0.0321536 |            0.75  |                0.25  |               0.625 |
| KIC 2438513 Q5  |            2.79084   |          0.584337   |          6.45936 |      0.0321536 |            0.5   |                0.625 |               0.5   |
| KIC 2442448 Q5  |            0.0204345 |          0.0100228  |          1.9436  |      0.0321536 |            0     |                0     |              -0.375 |
| KIC 2581554 Q5  |            1.36911   |          0.66866    |          8.60771 |      0.0319379 |            0     |                0.625 |               0.625 |
| KIC 2692377 Q5  |            3.05545   |          0.336949   |         21.1005  |      0.0321536 |            1     |                1     |               0.875 |
| KIC 2832589 Q5  |            3.03545   |          0.780416   |          8.49981 |      0.0321536 |            0.375 |                0.5   |               0.125 |
