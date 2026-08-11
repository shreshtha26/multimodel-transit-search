# Multi-Star Background Time-Scale Audit

Status: exploratory 50-star mechanism analysis using saved BLS/ARIMA-TCF outputs.

This audit asks whether cheap stellar-background features help explain which detector wins across multiple stellar backgrounds. It does not rerun BLS, TCF, ARIMA, Kalman, or GP detectors.

## Method

The script reads the optimized 50-star outputs:

```text
outputs/experiments/multistar_bls_tcf/optimized/metrics/target_manifest_used.csv
outputs/experiments/multistar_bls_tcf/optimized/metrics/multistar_bls_tcf_injections.csv
```

For each target, it loads the cached PDCSAP light curve and applies the existing preprocessing. Background features are estimated on the longest contiguous usable normalized-flux segment.

No gap interpolation is used.

The primary cheap time-scale proxy is:

```text
background_tau_acf_e_days
```

defined as the first ACF crossing below `exp(-1)`. The main dimensionless feature is:

```text
background_tau_acf_e_days / transit_duration_days
```

The script also saves half-life ACF and integrated-positive-ACF variants because the e-folding proxy is coarse when many stars decorrelate within one cadence.

## Command

```bash
python scripts/analyze_multistar_background_timescales.py
```

## Outputs

```text
outputs/experiments/multistar_background_timescale/metrics/multistar_background_timescale_features.csv
outputs/experiments/multistar_background_timescale/metrics/multistar_injections_with_background_timescales.csv
outputs/experiments/multistar_background_timescale/metrics/multistar_background_outcomes_by_ratio_bin.csv
outputs/experiments/multistar_background_timescale/metrics/multistar_background_outcomes_by_acf_half_ratio_bin.csv
outputs/experiments/multistar_background_timescale/metrics/multistar_background_outcomes_by_integrated_acf_ratio_bin.csv
outputs/experiments/multistar_background_timescale/metrics/multistar_background_feature_correlations.csv
outputs/experiments/multistar_background_timescale/metrics/multistar_background_timescale_summary.json
outputs/experiments/multistar_background_timescale/figures/multistar_background_ratio_bin_recovery.png
```

## Current Result

Across 50 stars and 900 injections, the strongest cheap predictor in this audit is flux scatter:

```text
Spearman robust_flux_scatter_ppm vs harmonic BLS recovery:  -0.622
Spearman robust_flux_scatter_ppm vs harmonic union recovery: -0.554
Spearman robust_flux_scatter_ppm vs harmonic neither rate:   +0.554
```

Background time-scale ratios also carry useful signal:

```text
Spearman background_tau_acf_e / transit duration vs harmonic BLS recovery: -0.555
Spearman background_tau_acf_e / transit duration vs harmonic TCF-only:     +0.341
```

By quartile of `background_tau_acf_e_days / transit_duration_days`:

| Ratio bin | BLS recovery | TCF recovery | Union recovery | BLS-only | TCF-only | Neither |
| --------- | -----------: | -----------: | -------------: | -------: | -------: | ------: |
| lowest | 83.8% | 13.2% | 83.8% | 70.6% | 0.0% | 16.2% |
| low-mid | 68.5% | 15.8% | 70.3% | 54.5% | 1.8% | 29.7% |
| high-mid | 50.9% | 12.7% | 53.5% | 40.8% | 2.6% | 46.5% |
| highest | 6.8% | 29.7% | 33.8% | 4.1% | 27.0% | 66.2% |

This suggests that high-scatter, slower-correlated backgrounds are where BLS loses its advantage and TCF becomes relatively more useful. However, the overall neither rate also rises sharply, so this is not simply a TCF-win regime.

## Interpretation

The result supports the idea that an eventual adaptive selector should receive physically meaningful light-curve features:

```text
noise amplitude
gap fraction
background autocorrelation
background-to-transit time-scale ratio
low-frequency-to-short-timescale scatter ratio
```

This does not yet prove a final routing rule. It shows that cheap background characteristics are associated with detector outcome patterns across multiple stellar backgrounds, which is the prerequisite for a scientifically justified adaptive router.
