# Gap-Mode ARIMA Comparison

This report compares explicit gap representations only. It does not add BLS, TCF, GP, ML, or false-alarm calibration.

## Selected Models

```text
quality_policy               gap_mode selected_order stationarity_conclusion  recommended_d    selected_differencing_alignment  residual_autocorrelation_remaining  variance_instability  scientifically_acceptable
       default     longest_contiguous ARIMA(1, 0, 0)    stationary_supported            0.0 aligned_with_stationarity_evidence                                True                  True                      False
       default      full_grid_missing ARIMA(1, 1, 0)  conflicting_rejections            NaN                         unresolved                                True                  True                      False
       default interpolated_full_grid ARIMA(1, 1, 1)  conflicting_rejections            NaN                         unresolved                                True                  True                      False
```

## Questions

- Does selected d change by gap mode? True
- Does selected ARIMA order change? True
- Does the stationarity conclusion change? True
- Does residual whitening improve? False
- Does variance stability improve? False
- Does interpolation artificially improve fit metrics? lower_rmse_than_full_grid_missing
- Does the longest contiguous segment favour d=0? True
- Is any mode scientifically acceptable? False

## Best Available

Best available mode/model: longest_contiguous ARIMA(1, 0, 0)

If all modes remain scientifically inadequate, the best available row is diagnostic only.

No scientifically acceptable gap representation/model combination was found.
