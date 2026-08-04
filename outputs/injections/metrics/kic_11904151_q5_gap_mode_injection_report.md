# Gap-Mode Transit-Injection Experiment

This experiment injects known box transits into the same Kepler light curve representation used by each selected ARIMA gap-mode model.
It does not run BLS, TLS, TCF, GP, ML, or false-alarm calibration beyond empirical single-light-curve null thresholds.

## Selected Gap-Mode Models

| gap_mode               | selected_order   | stationarity_conclusion   |   recommended_d | selected_differencing_alignment      | scientifically_acceptable   |
|:-----------------------|:-----------------|:--------------------------|----------------:|:-------------------------------------|:----------------------------|
| longest_contiguous     | ARIMA(2, 1, 0)   | stationary_supported      |               0 | conflicts_with_stationarity_evidence | False                       |
| full_grid_missing      | ARIMA(1, 1, 0)   | conflicting_rejections    |             nan | unresolved                           | False                       |
| interpolated_full_grid | ARIMA(1, 1, 0)   | conflicting_rejections    |             nan | unresolved                           | False                       |

## Injection Summary

| gap_mode               |   n_injections |   median_depth_retention_fraction |   median_snr_retention_fraction |   median_ingress_egress_distortion_fraction |   median_best_spurious_statistic |   spurious_peak_exceeds_injected_rate |   recovery_rate_at_far_0.1 |   top_recovery_rate_at_far_0.1 |   median_spurious_peaks_above_far_0.1 |   recovery_rate_at_far_0.05 |   top_recovery_rate_at_far_0.05 |   median_spurious_peaks_above_far_0.05 |   recovery_rate_at_far_0.01 |   top_recovery_rate_at_far_0.01 |   median_spurious_peaks_above_far_0.01 |
|:-----------------------|---------------:|----------------------------------:|--------------------------------:|--------------------------------------------:|---------------------------------:|--------------------------------------:|---------------------------:|-------------------------------:|--------------------------------------:|----------------------------:|--------------------------------:|---------------------------------------:|----------------------------:|--------------------------------:|---------------------------------------:|
| full_grid_missing      |              9 |                           1.05731 |                        0.626767 |                                    1        |                          6.22842 |                                     0 |                          1 |                              1 |                                    12 |                           1 |                               1 |                                      6 |                           1 |                               1 |                                      2 |
| interpolated_full_grid |              9 |                           1.05817 |                        0.627859 |                                    1        |                          3.06882 |                                     0 |                          1 |                              1 |                                    12 |                           1 |                               1 |                                      6 |                           1 |                               1 |                                      2 |
| longest_contiguous     |              9 |                           1.01031 |                        0.580748 |                                    0.775587 |                          1.62711 |                                     0 |                          1 |                              1 |                                     1 |                           1 |                               1 |                                      1 |                           1 |                               1 |                                      1 |

## Interpretation

- Any mode scientifically acceptable before injection? False
- Any mode has perfect top recovery at FAR 0.01? True
- Best median SNR retention mode: interpolated_full_grid
- Highest spurious-peak-exceeds-injected rate mode: full_grid_missing
