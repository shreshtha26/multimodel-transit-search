"""Cadence-aware quality masking, gap representation, and normalization for Kepler PDCSAP light curves.
This module converts loaded PDCSAP samples into an explicit cadence grid while preserving missing and unusable cadences for downstream analysis."""

from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from lightkurve.utils import KeplerQualityFlags
REQUIRED_PDCSAP_COLUMNS = ("time", "flux", "flux_error", "quality", "cadenceno")
PERMISSIVE_QUALITY_BITMASK = 2 | 4 | 8 | 256 | 16384 | 32768 | 65536

@dataclass(frozen=True)
class PreprocessingSummary:
    """Audit record for one PDCSAP preprocessing pass."""
    quality_policy: str
    quality_rejection_bitmask: int
    n_raw: int
    n_cadence_grid: int
    n_row_absent: int
    n_usable: int
    n_unusable_observed: int
    median_flux: float
    normalization_fit_fraction: float
    normalization_fit_count: int
    quality_zero_fraction_raw: float
    quality_ok_fraction_raw: float
    flux_error_finite_fraction_observed: float
    median_cadence_days: float
    gap_count: int
    max_gap_cadences: int
    max_gap_days: float
    segment_count: int
    longest_segment_length: int

    def to_dict(self):
        return asdict(self)

def validate_pdcsap_frame(frame):
    """Fail explicitly if the expected PDCSAP columns are missing."""
    missing = [column for column in REQUIRED_PDCSAP_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required PDCSAP columns: {missing}")

def obvious_valid_mask(frame, *, require_quality_zero=True, require_finite_flux_error=False):
    """Return a simple finite-data mask retained for compatibility with older callers.
    The main preprocessing path uses the named quality-policy machinery below."""
    validate_pdcsap_frame(frame)
    mask = np.isfinite(frame["time"].to_numpy()) & np.isfinite(frame["flux"].to_numpy())
    if require_quality_zero:
        mask &= frame["quality"].to_numpy() == 0
    if require_finite_flux_error:
        mask &= np.isfinite(frame["flux_error"].to_numpy())
    return mask

def quality_rejection_bitmask(quality_policy):
    """Return the Kepler quality bits rejected by a named policy."""
    policy = str(quality_policy).lower()
    if policy == "strict":
        return -1
    if policy == "none":
        return 0
    if policy == "permissive":
        return PERMISSIVE_QUALITY_BITMASK
    if policy in {"default", "hard", "hardest"}:
        return int(KeplerQualityFlags.OPTIONS[policy])
    raise ValueError("quality_policy must be one of: strict, permissive, default, hard, hardest, none.")

def quality_ok_mask(quality, quality_policy):
    """Return True for cadences that pass the selected Kepler quality policy."""
    policy = str(quality_policy).lower()
    quality_values = pd.to_numeric(pd.Series(quality), errors="coerce").to_numpy()
    if policy == "none":
        return np.ones(quality_values.shape, dtype=bool)
    if policy == "strict":
        return np.isfinite(quality_values) & (quality_values == 0)
    bitmask = quality_rejection_bitmask(policy)
    finite = np.isfinite(quality_values)
    ok = np.zeros(quality_values.shape, dtype=bool)
    quality_int = quality_values[finite].astype(np.int64)
    ok[finite] = (quality_int & bitmask) == 0
    return ok

def regularize_cadence_grid(frame):
    """Reindex the light curve onto its complete cadence-number grid.
    Cadences absent from the downloaded file are inserted as explicit rows with NaN scientific values and row_present=False; no flux values are imputed."""
    validate_pdcsap_frame(frame)
    raw = frame.copy()
    if raw.empty:
        raise ValueError("Cannot build a cadence grid from an empty PDCSAP frame.")
    if raw["cadenceno"].isna().any():
        raise ValueError("cadenceno contains missing values; cannot build cadence grid.")
    if raw["cadenceno"].duplicated().any():
        duplicated = raw.loc[raw["cadenceno"].duplicated(), "cadenceno"].head().tolist()
        raise ValueError(f"Duplicate cadenceno values are not supported: {duplicated}")
    raw["cadenceno"] = raw["cadenceno"].astype(np.int64)
    raw["row_present"] = True
    start = int(raw["cadenceno"].min())
    stop = int(raw["cadenceno"].max())
    full_index = pd.Index(np.arange(start, stop + 1, dtype=np.int64), name="cadenceno")
    regular = raw.set_index("cadenceno").sort_index().reindex(full_index).reset_index()
    regular["row_present"] = regular["row_present"].eq(True)
    return regular

def summarize_cadence_gaps(regular):
    """Summarize contiguous runs of absent or unusable cadences without filling them."""
    usable = regular["usable"].to_numpy(dtype=bool)
    if len(usable) == 0:
        return float("nan"), 0, 0, float("nan")
    usable_rows = regular.loc[usable, ["cadenceno", "time"]].sort_values("cadenceno")
    cadence_numbers = usable_rows["cadenceno"].to_numpy(dtype=float)
    times = usable_rows["time"].to_numpy(dtype=float)
    if len(usable_rows) < 2:
        median_cadence = float("nan")
    else:
        cadence_steps = np.diff(cadence_numbers)
        time_steps = np.diff(times)
        valid_steps = np.isfinite(cadence_steps) & np.isfinite(time_steps) & (cadence_steps > 0) & (time_steps > 0)
        per_cadence_deltas = time_steps[valid_steps] / cadence_steps[valid_steps]
        median_cadence = float(np.median(per_cadence_deltas)) if len(per_cadence_deltas) else float("nan")
    missing_or_unusable = ~usable
    gap_count = 0
    max_gap_cadences = 0
    current_gap = 0
    for is_gap in missing_or_unusable:
        if is_gap:
            current_gap += 1
            max_gap_cadences = max(max_gap_cadences, current_gap)
        elif current_gap:
            gap_count += 1
            current_gap = 0
    if current_gap:
        gap_count += 1
    max_gap_days = float(max_gap_cadences * median_cadence) if np.isfinite(median_cadence) else float("nan")
    return median_cadence, gap_count, max_gap_cadences, max_gap_days

def assign_segment_ids(regular):
    """Label contiguous usable cadence runs; unusable rows receive segment_id=-1."""
    usable = regular["usable"].to_numpy(dtype=bool)
    starts = usable & np.r_[True, ~usable[:-1]]
    segment_ids = np.cumsum(starts) - 1
    segment_ids[~usable] = -1
    return pd.Series(segment_ids.astype(np.int64), index=regular.index)

def segment_lengths(regular):
    """Return usable-cadence counts for each contiguous segment, largest first."""
    usable = regular.loc[regular["segment_id"] >= 0]
    if usable.empty:
        return pd.Series(dtype=np.int64)
    return usable.groupby("segment_id").size().sort_values(ascending=False)

def longest_contiguous_segment(regular):
    """Return the longest contiguous usable segment for gap-free model comparisons."""
    lengths = segment_lengths(regular)
    if lengths.empty:
        raise ValueError("No usable contiguous segment is available.")
    segment_id = int(lengths.index[0])
    return regular.loc[regular["segment_id"] == segment_id].copy().reset_index(drop=True)

def summarize_gaps(time):
    """Estimate gaps from an already-filtered time vector.
    This helper is retained for compatibility; cadence-aware code should prefer summarize_cadence_gaps on the explicit cadence grid."""
    if len(time) < 2:
        return float("nan"), 0, float("nan")
    deltas = np.diff(np.sort(time))
    finite_deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if len(finite_deltas) == 0:
        return float("nan"), 0, float("nan")
    median_cadence = float(np.median(finite_deltas))
    gap_threshold = 1.5 * median_cadence
    large_gaps = finite_deltas[finite_deltas > gap_threshold]
    max_gap = float(np.max(large_gaps)) if len(large_gaps) else 0.0
    return median_cadence, int(len(large_gaps)), max_gap

def preprocess_pdcsap_light_curve(frame, *, quality_policy="strict", require_quality_zero=None, require_finite_flux_error=False, normalization_fit_fraction=1.00):
    """Build the cadence grid, apply the selected validity policy, and normalize PDCSAP flux.
    The normalization median is fitted only on the leading configured fraction of usable cadences so train/holdout workflows can avoid normalization leakage."""

    validate_pdcsap_frame(frame)
    if not 0.0 < normalization_fit_fraction <= 1.0:
        raise ValueError("normalization_fit_fraction must be in (0, 1].")
    if require_quality_zero is not None:
        quality_policy = "strict" if require_quality_zero else "none"
    quality_policy = str(quality_policy).lower()
    bitmask = quality_rejection_bitmask(quality_policy)
    raw = frame.copy()
    regular = regularize_cadence_grid(raw)
    regular["finite_time"] = np.isfinite(regular["time"].to_numpy(dtype=float))
    regular["finite_flux"] = np.isfinite(regular["flux"].to_numpy(dtype=float))
    regular["finite_flux_error"] = np.isfinite(regular["flux_error"].to_numpy(dtype=float))
    regular["quality_ok"] = quality_ok_mask(regular["quality"], quality_policy)
    regular["usable"] = regular["row_present"] & regular["finite_time"] & regular["finite_flux"] & regular["quality_ok"]
    if require_finite_flux_error:
        regular["usable"] &= regular["finite_flux_error"]
    regular["observed_mask"] = regular["usable"]
    regular["gap_reason"] = "usable"
    regular.loc[~regular["row_present"], "gap_reason"] = "cadence_absent_from_file"
    regular.loc[regular["row_present"] & ~regular["finite_time"], "gap_reason"] = "nonfinite_time"
    regular.loc[regular["row_present"] & regular["finite_time"] & ~regular["finite_flux"], "gap_reason"] = "nonfinite_flux"
    regular.loc[regular["row_present"] & regular["finite_time"] & regular["finite_flux"] & ~regular["quality_ok"], "gap_reason"] = "quality_flagged"
    if require_finite_flux_error:
        regular.loc[regular["row_present"] & regular["finite_time"] & regular["finite_flux"] & regular["quality_ok"] & ~regular["finite_flux_error"], "gap_reason"] = "nonfinite_flux_error"
    usable_indices = regular.index[regular["usable"]].to_numpy()
    if len(usable_indices) == 0:
        raise ValueError("No valid cadences remain after preprocessing.")
    normalization_count = max(1, int(np.floor(len(usable_indices) * normalization_fit_fraction)))
    normalization_indices = usable_indices[:normalization_count]
    median_flux = float(np.nanmedian(regular.loc[normalization_indices, "flux"].to_numpy(dtype=float)))
    if not np.isfinite(median_flux) or median_flux == 0.0:
        raise ValueError(f"Cannot normalize with median_flux={median_flux!r}.")
    regular["normalization_fit"] = False
    regular.loc[normalization_indices, "normalization_fit"] = True
    regular["normalized_flux"] = np.nan
    regular.loc[regular["usable"], "normalized_flux"] = regular.loc[regular["usable"], "flux"] / median_flux - 1.0
    regular["segment_id"] = assign_segment_ids(regular)
    lengths = segment_lengths(regular)
    median_cadence, gap_count, max_gap_cadences, max_gap_days = summarize_cadence_gaps(regular)
    raw_quality = raw["quality"].to_numpy()
    quality_zero_fraction = float(np.mean(raw_quality == 0))
    quality_ok_fraction = float(np.mean(quality_ok_mask(raw["quality"], quality_policy)))
    row_present = regular["row_present"].to_numpy(dtype=bool)
    flux_error_valid_fraction = float(regular.loc[row_present, "finite_flux_error"].mean()) if row_present.any() else float("nan")
    summary = PreprocessingSummary(quality_policy=quality_policy, quality_rejection_bitmask=int(bitmask), n_raw=int(len(raw)), n_cadence_grid=int(len(regular)), n_row_absent=int((~regular["row_present"]).sum()), n_usable=int(regular["usable"].sum()), n_unusable_observed=int((regular["row_present"] & ~regular["usable"]).sum()), median_flux=median_flux, normalization_fit_fraction=float(normalization_fit_fraction), normalization_fit_count=int(normalization_count), quality_zero_fraction_raw=quality_zero_fraction, quality_ok_fraction_raw=quality_ok_fraction, flux_error_finite_fraction_observed=flux_error_valid_fraction, median_cadence_days=median_cadence, gap_count=gap_count, max_gap_cadences=int(max_gap_cadences), max_gap_days=max_gap_days, segment_count=int(len(lengths)), longest_segment_length=int(lengths.iloc[0]) if not lengths.empty else 0)
    return regular, summary
