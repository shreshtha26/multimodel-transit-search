# Kepler Dataset Strategy

multi-model-transit-search should not store the full Kepler archive inside this repository. Kepler
light curves are large, remotely versioned science products, so the reproducible
approach is:

```text
target catalog/config
-> Lightkurve/MAST download on demand
-> explicit preprocessing
-> per-target outputs
-> aggregate result table
```

## Current Target Sample

The current robust sample is configured in:

```text
configs/kepler_target_sample.yaml
```

It includes known Kepler transiting systems such as Kepler-10, Kepler-110,
Kepler-11, Kepler-9, Kepler-20, and Kepler-22. This is deliberately larger than
the original single-target experiment but still small enough to iterate on.

Run it with:

```bash
.venv/bin/python scripts/run_target_sample.py \
  --config configs/kepler_target_sample.yaml \
  --continue-on-error
```

The batch summary is written to:

```text
outputs/target_sample/metrics/target_sample_summary.csv
```

## Scaling To More Kepler Data

To scale beyond this sample, create a larger YAML/CSV-derived config containing:

```text
target_id
quarter
scientific label or role
optional per-target runner args
```

Then run the same batch script. The code does not need to change.

Do not immediately run every Kepler target and every quarter on a laptop. The
full Kepler long-cadence archive is an HPC/cloud batch job. The right scaling
path is:

```text
single target
-> curated target sample
-> hundreds of targets
-> full quarter grid
-> parallel/HPC execution
```

This keeps multi-model-transit-search reproducible while avoiding accidental multi-terabyte local
downloads.
