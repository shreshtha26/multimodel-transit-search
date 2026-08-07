# Kepler Dataset Strategy

`multi-model-transit-search` should not store the full Kepler archive inside the repository.

Kepler light curves are externally maintained science products, so the reproducible data strategy is:

```text
versioned target manifest/config
-> Lightkurve/MAST retrieval on demand
-> explicit preprocessing
-> optional local cache
-> per-target experiment outputs
-> aggregate benchmark tables
```

The repository currently contains **two target-sample layers with different purposes**. They should not be treated as interchangeable.

## 1. Early Curated Target Sample

The original curated sample is configured in:

```text
configs/kepler_target_sample.yaml
```

It contains six known transiting systems in Quarter 5:

```text
Kepler-10
Kepler-110
Kepler-11
Kepler-9
Kepler-20
Kepler-22
```

This sample was introduced to move the early ARIMA/transformed-template work beyond a single target while keeping iteration inexpensive.

It is still useful for running the unified early-stage workflow, but it is **not the current primary multi-star benchmark**.

Run it from the repository root with:

```bash
python scripts/run_target_sample.py
```

or from Python:

```python
from scripts.run_target_sample import run_target_sample

run_target_sample(continue_on_error=True)
```

The batch summary is written to:

```text
outputs/target_sample/metrics/target_sample_summary.csv
```

The runner uses:

```text
configs/phase2.yaml
```

and delegates each target to the unified `scripts/run_experiment.py` path.

## 2. Current 50-Star Multi-Detector Pilot

The current multi-star BLS/ARIMA-TCF pilot uses:

```text
configs/kepler_50_star_manifest.csv
```

The manifest contains 50 unique Kepler Quarter 5 target-quarter rows:

```text
1 reference target
49 rows labelled confirmed_planet_host
```

The current optimized pilot is configured in `scripts/run_multistar_bls_tcf.py`.

Its default experimental design includes:

```text
50 targets
18 synthetic injections per target
900 injections total
8 null trials per target
400 null trials total
BLS candidate generation
ARIMA(1,1,0) innovation transformation
TCF candidate generation
per-target resumable execution
cached Kepler light curves
cached base ARIMA fits
```

The optimized injection grid is:

```text
periods:   2, 5, 10 days
durations: 2, 4, 8 hours
depths:    500, 1000 ppm
phase:     0.45
```

Run the pilot from the repository root with:

```bash
python scripts/run_multistar_bls_tcf.py
```

The default manifest path is:

```text
configs/kepler_50_star_manifest.csv
```

and the default output root is:

```text
outputs/experiments/multistar_bls_tcf/optimized/
```

This 50-star experiment is the current engineering pilot used for BLS/TCF comparison and candidate-reranker development.

It is still not a population-scale Kepler benchmark.

## Why the Two Samples Both Remain

The two configurations serve different purposes:

| Sample | File | Purpose |
|---|---|---|
| Early curated sample | `configs/kepler_target_sample.yaml` | Small multi-target extension of the Phase 1/unified experiment workflow |
| Current 50-star pilot | `configs/kepler_50_star_manifest.csv` | Multi-detector BLS/ARIMA-TCF benchmarking, candidate generation, and reranker development |

Keeping both is useful as long as their roles are documented clearly.

The six-star YAML should not be described as the current primary sample, and the 50-star manifest should not be presented as a final representative Kepler population.

## Manifest Design

A scalable target manifest should contain, at minimum:

```text
target_id
quarter
```

Optional metadata can include fields such as:

```text
selection_group
scientific role
known-system label
sample stratum
provenance
```

Experiment-specific truth variables should be kept separate from model features when they could leak the answer into downstream machine-learning models.

In particular, candidate reranking must not use target identity or injected-truth fields as predictive features.

## Data Retrieval and Caching

The multi-star runner loads Kepler PDCSAP light curves through the package data loader and can cache the retrieved light-curve frames locally.

The current default cache location is:

```text
outputs/cache/kepler_light_curves/
```

Caching reduces repeated MAST downloads during resumed or repeated experiments.

Cached data should be treated as a reproducibility and performance aid, not as the canonical source of the Kepler archive.

The canonical source remains the external Kepler/MAST product identified by target, quarter, and preprocessing policy.

## Reproducibility Requirements

Each benchmark should record enough information to reconstruct the analyzed sample and search configuration.

At minimum, preserve:

- target manifest or configuration;
- target ID and quarter;
- flux product;
- quality-mask policy;
- preprocessing settings;
- injection configuration and random seed;
- null-generation procedure and random seed;
- detector period and duration grids;
- ARIMA order and fitting settings when applicable;
- candidate `top_k`;
- false-alarm calibration settings;
- software/configuration version associated with the outputs.

False-alarm thresholds are configuration-specific and should be recalibrated whenever the preprocessing, detector statistic, period grid, duration grid, candidate-generation procedure, or null-generation method changes.

## Scaling Beyond 50 Targets

The intended scaling path is now:

```text
single reference target
-> six-target curated development sample
-> 50-star multi-detector pilot
-> larger stratified untouched target sample
-> hundreds to thousands of targets
-> multiple quarters / cadence regimes
-> parallel or HPC/cloud execution
```

The next larger sample should not simply add arbitrary targets.

It should be designed to test generalization across relevant regimes such as:

- stellar variability;
- photometric scatter;
- gap fraction;
- transit depth;
- transit duration;
- orbital period;
- quarter;
- cadence mode;
- known planets and astrophysical false positives.

## Storage Policy

Do not commit bulk downloaded Kepler light curves to Git.

Prefer:

```text
small versioned manifests/configs
+ code that retrieves the specified science products
+ local caches excluded from source control
+ compact committed metrics and summaries
```

Do not attempt to download every Kepler target and every quarter to a laptop as the next step.

Population-scale execution should use controlled parallelism and, when necessary, HPC or cloud resources.

## Current Conclusion

The repository has moved beyond the original six-target development sample.

The current data hierarchy is:

```text
KIC 11904151 single-target methodological reference
-> six-target curated early development sample
-> 50-star Quarter 5 BLS/ARIMA-TCF pilot
-> future larger untouched and stratified validation samples
```

This structure preserves reproducibility while allowing the project to scale without treating the current 50-star pilot as representative of the full Kepler population.

For current detector, reranker, calibration, and performance results, see the [main README](../README.md). For the historical ARIMA foundation, see [Phase 1: Single-Target ARIMA Prototype](phase1_single_target_arima.md).
