"""Kepler light-curve loading utilities.
This module loads PDCSAP flux for one target and one quarter at a time. It is not a full-archive ingestion system."""

from dataclasses import dataclass
from typing import Any
import lightkurve as lk
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class KeplerLightCurve:
    """In-memory representation of one Kepler PDCSAP light curve."""
    target_id: str
    quarter: int
    time: np.ndarray
    flux: np.ndarray
    flux_error: np.ndarray
    quality: np.ndarray
    cadenceno: np.ndarray
    metadata: dict[str, Any]

    def to_dataframe(self):
        return pd.DataFrame({"time": self.time, "flux": self.flux, "flux_error": self.flux_error, "quality": self.quality, "cadenceno": self.cadenceno})

def format_kepler_target(target_id):
    """Normalize a target identifier into the format expected by Lightkurve."""
    target = str(target_id).strip()
    if target.upper().startswith("KIC"):
        target = target[3:].strip()
    return f"KIC {target}"

def load_kepler_pdcsap(target_id, quarter, *, author="Kepler", cadence="long"):
    """Download one Kepler PDCSAP light curve with Lightkurve.
    No quality masking, gap handling, cleaning, or normalization is performed here.
    Those operations are deferred to `preprocess_pdcsap_light_curve` so the preprocessing choices remain explicit and testable."""
    query = format_kepler_target(target_id)
    search = lk.search_lightcurve(query, mission="Kepler", quarter=quarter, author=author, cadence=cadence)
    if len(search) == 0:
        raise RuntimeError(f"No Kepler light curve found for target={query!r}, quarter={quarter}, author={author!r}, cadence={cadence!r}.")
    light_curve = search[0].download(quality_bitmask="none")
    if light_curve is None:
        raise RuntimeError(f"Lightkurve did not download a light curve for {query}.")
    # Cast FITS integer columns to native int64 for downstream NumPy/pandas operations.
    return KeplerLightCurve(target_id=query, quarter=quarter, time=np.asarray(light_curve.time.value, dtype=float),
                            flux=np.asarray(light_curve.pdcsap_flux.value, dtype=float),
                            flux_error=np.asarray(light_curve.pdcsap_flux_err.value, dtype=float),
                            quality=np.asarray(light_curve.quality.value, dtype=np.int64),
                            cadenceno=np.asarray(light_curve.cadenceno.value, dtype=np.int64),
                            metadata={"author": author, "cadence": cadence, "mission": "Kepler", "search_result": str(search[0])})
