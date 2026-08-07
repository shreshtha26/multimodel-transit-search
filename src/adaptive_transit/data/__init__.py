"""Data-loading helpers for Kepler light curves.
I start with Kepler long-cadence PDCSAP light curves from MAST through Lightkurve.
For each target-quarter I retain the time, PDCSAP flux and uncertainty, quality information
and cadence numbers, then explicitly reconstruct the cadence structure before doing any
statistical modelling or transit search."""
