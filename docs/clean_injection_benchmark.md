# Clean-background injection benchmark

This workflow separates three scientific questions that must not share one recovery denominator:

1. **Catalog-clean background injection recovery** — primary benchmark for weak synthetic transits.
2. **Known-planet positive controls** — recover cataloged real signals from the unmodified light curve.
3. **Known-signal / eclipsing-binary stress tests** — study competition, harmonics, iterative-search behavior, and false positives.

`catalog_clean_background` means *no match in the catalog snapshots used by the target-selection script for KOIs, DR25 TCEs, confirmed Kepler names, or the Kepler eclipsing-binary catalog*. It does **not** mean that the star is known to be planet-free.

## Preserve the legacy cohort

Do not overwrite `configs/kepler_50_star_manifest.csv` or prior output directories. They are needed to reproduce the original engineering pilot and reranker work.

The challenger runner now defaults to `configs/kepler_clean_background_manifest.csv` and refuses to run a clean injection benchmark unless all required catalog-contamination flags are present and false.

To deliberately run the legacy known-signal cohort, opt out explicitly, for example:

```bash
python scripts/run_multistar_challenger_benchmark.py \
  --profile main \
  --manifest-path configs/kepler_50_star_manifest.csv \
  --selection-group confirmed_planet_host \
  --allow-contaminated-cohort \
  --output-dir outputs/experiments/multistar_challenger_benchmark/legacy_known_host_stress_test
```

## 1. Build a catalog-clean candidate pool

This downloads/caches the official catalog snapshots used for exclusions, writes a full clean pool, and draws a deterministic candidate set for Q5 characterization:

```bash
python scripts/build_clean_kepler_manifest.py --quarter 5 --candidate-limit 250
```

Key outputs:

```text
outputs/target_selection/kepler_catalog_clean_pool.csv
outputs/target_selection/kepler_catalog_clean_candidates_q5.csv
outputs/target_selection/catalog_clean_selection_sources.json
outputs/target_selection/catalog_cache/
```

The source-record JSON is part of the provenance contract. Keep it with any published benchmark output.

## 2. Characterize the candidate backgrounds

Use cached Q5 light curves when available and allow downloads for candidates not already cached:

```bash
python scripts/characterize_target_manifest.py \
  --manifest-path outputs/target_selection/kepler_catalog_clean_candidates_q5.csv \
  --allow-download
```

Output:

```text
outputs/target_selection/kepler_catalog_clean_candidate_features.csv
```

Failed/missing-Q5 targets remain in this file with `status=failed`; they are not eligible for the final benchmark manifest.

## 3. Freeze the final 50-star clean manifest

Re-run the builder with the feature file. It selects non-overlapping extremes across five cheap background regimes and preserves all catalog-clean flags:

```bash
python scripts/build_clean_kepler_manifest.py \
  --quarter 5 \
  --candidate-limit 250 \
  --feature-path outputs/target_selection/kepler_catalog_clean_candidate_features.csv \
  --final-size 50
```

Output:

```text
configs/kepler_clean_background_manifest.csv
```

The five current `sample_stratum` labels are:

```text
quiet_low_scatter
high_scatter
long_memory
smooth_background_dominant
gap_heavy
```

These are sampling labels, not claims that each star belongs to a unique physical variability class.

## 4. Run the staged clean benchmark: 5 -> 10 -> 50

Use separate output directories so every stage remains inspectable.

### Five-star validation

```bash
python scripts/run_multistar_challenger_benchmark.py \
  --profile main \
  --manifest-path configs/kepler_clean_background_manifest.csv \
  --selection-group catalog_clean_background \
  --target-limit 5 \
  --output-dir outputs/experiments/multistar_challenger_benchmark/clean_q5_5star
```

Inspect the files below before scaling:

```text
metrics/target_manifest_used.csv
metrics/multistar_challenger_base_candidates.csv
metrics/multistar_challenger_pipeline_summary.csv
metrics/multistar_challenger_combinations.csv
metrics/multistar_challenger_failure_modes.csv
metrics/multistar_challenger_by_star.csv
metrics/multistar_challenger_by_stratum.csv
```

### Ten-star validation

```bash
python scripts/run_multistar_challenger_benchmark.py \
  --profile main \
  --manifest-path configs/kepler_clean_background_manifest.csv \
  --selection-group catalog_clean_background \
  --target-limit 10 \
  --output-dir outputs/experiments/multistar_challenger_benchmark/clean_q5_10star
```

### Fifty-star main injection benchmark

Run this only after the 5- and 10-star outputs are scientifically sensible:

```bash
python scripts/run_multistar_challenger_benchmark.py \
  --profile main \
  --manifest-path configs/kepler_clean_background_manifest.csv \
  --selection-group catalog_clean_background \
  --target-limit 50 \
  --output-dir outputs/experiments/multistar_challenger_benchmark/clean_q5_50star
```

## 5. New diagnostic outputs

For every star, the runner now searches the **unmodified light curve before injection** through every active branch/detector and saves its top candidates:

```text
stars/kic_<id>_q<quarter>/base_light_curve_candidates.csv
metrics/multistar_challenger_base_candidates.csv
```

Each injection row also records:

```text
<pipeline>_harmonic_topk_matched
<pipeline>_base_rank1_period_days
<pipeline>_matches_base_rank1
<pipeline>_failure_mode
```

Failure modes are deliberately separated:

- `rank1_recovery`: injected period or accepted harmonic is rank 1;
- `ranking_failure`: injected period/harmonic exists in top-k but is not rank 1;
- `base_rank1_competition`: rank 1 remains matched to that pipeline's pre-injection rank-1 candidate;
- `candidate_generation_failure`: injected period/harmonic is absent from top-k and rank 1 is not simply the preserved base candidate;
- `pipeline_error`: detector/branch execution failed.

`multistar_challenger_combinations.csv` reports both rank-1 and top-k harmonic unions. The difference is the ensemble-level ranking gap and is the candidate-reranker opportunity.

## 6. FAP calibration comes after the clean injection benchmark

Do not interpret the injection-only summary as a false-alarm-controlled recovery claim. Once the 50-star clean benchmark is frozen, calibrate *that exact benchmark directory*:

```bash
python scripts/calibrate_multistar_challenger_benchmark.py \
  --profile main \
  --benchmark-dir outputs/experiments/multistar_challenger_benchmark/clean_q5_50star \
  --n-null-trials-per-star 1000
```

Use the existing convergence analyzer to decide whether the 1% tail estimate is stable enough for the claim you want to make; do not infer tail precision only from the nominal number of nulls.

## Interpretation contract

For the primary benchmark, report:

- individual-pipeline rank-1 recovery;
- individual-pipeline top-k recovery;
- all-pipeline rank-1 and top-k unions;
- ranking-gap counts/rates;
- failure-mode breakdowns;
- performance by depth, duration, period, star, and background stratum;
- FAP-controlled recovery only after matched null calibration.

Do not merge known-planet hosts or eclipsing binaries back into the clean-background denominator. Analyze them separately as positive controls or stress tests.
