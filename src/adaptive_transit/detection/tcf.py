"""Periodic Transit Comb Filter search on ARIMA innovations."""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def default_period_grid(time, min_period_days=1.0, max_period_days=15.0, n_periods=1000):
    time = np.asarray(time, dtype=float)
    finite_time = time[np.isfinite(time)]
    if finite_time.size < 2:
        raise ValueError("At least two finite time values are required.")
    if min_period_days <= 0 or max_period_days <= min_period_days:
        raise ValueError("Invalid period range.")
    if n_periods < 2:
        raise ValueError("n_periods must be at least 2.")
    return np.linspace(float(min_period_days), float(max_period_days), int(n_periods))

def default_duration_grid(min_duration_hours=1.5, max_duration_hours=10.0, n_durations=8):
    if min_duration_hours <= 0 or max_duration_hours <= min_duration_hours:
        raise ValueError("Invalid duration range.")
    if n_durations < 2:
        raise ValueError("n_durations must be at least 2.")
    return np.linspace(float(min_duration_hours), float(max_duration_hours), int(n_durations)) / 24.0

def median_cadence(time):
    time = np.asarray(time, dtype=float)
    finite_time = np.sort(np.unique(time[np.isfinite(time)]))
    differences = np.diff(finite_time)
    differences = differences[np.isfinite(differences) & (differences > 0)]
    if differences.size == 0:
        raise ValueError("A positive cadence could not be estimated.")
    return float(np.median(differences))

def robust_scale(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("At least two finite values are required.")
    median = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - median)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(values, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Innovation scale is not positive.")
    return scale

def circular_window_sum(values, half_width):
    values = np.asarray(values, dtype=float)
    half_width = int(half_width)
    if half_width <= 0:
        return values.copy()
    if values.size <= 2 * half_width:
        return np.full(values.shape, np.sum(values), dtype=float)
    kernel = np.ones(2 * half_width + 1, dtype=float)
    padded = np.concatenate([values[-half_width:], values, values[:half_width]])
    return np.convolve(padded, kernel, mode="valid")

def circular_window_sum_rows(values, half_width):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional array.")
    half_width = int(half_width)
    if half_width <= 0:
        return values.copy()
    n_bins = int(values.shape[1])
    if n_bins <= 2 * half_width:
        return np.repeat(np.sum(values, axis=1, keepdims=True), n_bins, axis=1)
    kernel = np.ones(2 * half_width + 1, dtype=float)
    padded = np.concatenate([values[:, -half_width:], values, values[:, :half_width]], axis=1)
    return np.vstack([np.convolve(row, kernel, mode="valid") for row in padded])

def fit_arima_innovations(flux, order=(1, 1, 0), maxiter=200):
    series = np.asarray(flux, dtype=float).reshape(-1)
    if np.isfinite(series).sum() < 24:
        raise ValueError("At least 24 finite flux values are required.")
    model = ARIMA(series, order=tuple(order), trend="n", enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(method_kwargs={"maxiter": int(maxiter)})
    innovations = np.asarray(fitted.resid, dtype=float).reshape(-1)
    innovations[~np.isfinite(series)] = np.nan
    burn = int(getattr(fitted, "loglikelihood_burn", 0))
    if burn > 0:
        innovations[:burn] = np.nan
    mle_retvals = getattr(fitted, "mle_retvals", {})
    converged = bool(mle_retvals.get("converged", True))
    warnflag = mle_retvals.get("warnflag")
    iterations = mle_retvals.get("iterations")
    function_calls = mle_retvals.get("fcalls")
    summary = {"order": tuple(order), "converged": converged, "aic": float(fitted.aic), "bic": float(fitted.bic), "finite_innovation_count": int(np.isfinite(innovations).sum()), "optimizer_warnflag": int(warnflag) if warnflag is not None else None, "optimizer_iterations": int(iterations) if iterations is not None else None, "optimizer_function_calls": int(function_calls) if function_calls is not None else None}
    return {"innovations": innovations, "fit": fitted, "summary": summary}

def score_period(time, innovations, period, duration_grid, scale, cadence, edge_width_cadences=0, min_edge_observations=4, min_transit_events=3, min_event_consistency_fraction=0.60):
    finite = np.isfinite(time) & np.isfinite(innovations)
    time_values = np.asarray(time, dtype=float)[finite]
    innovation_values = np.asarray(innovations, dtype=float)[finite]
    empty = {"period": float(period), "score": np.nan, "raw_pooled_score": np.nan, "epoch": np.nan, "duration": np.nan, "edge_amplitude": np.nan, "n_edge_observations": 0, "n_valid_transit_events": 0, "n_positive_transit_events": 0, "positive_event_fraction": np.nan, "median_event_score": np.nan}
    if time_values.size < min_edge_observations:
        return empty
    start_time = float(np.min(time_values))
    n_phase_bins = max(int(np.round(float(period) / float(cadence))), 4)
    phase_bin_width = float(period) / float(n_phase_bins)
    elapsed_cycles = (time_values - start_time) / float(period)
    cycle_indices = np.floor(elapsed_cycles).astype(int)
    phases = np.mod(time_values - start_time, float(period))
    phase_indices = np.floor(phases / phase_bin_width).astype(int)
    phase_indices = np.clip(phase_indices, 0, n_phase_bins - 1)
    n_cycles = int(np.max(cycle_indices)) + 1
    flat_indices = cycle_indices * n_phase_bins + phase_indices
    event_bin_sums = np.bincount(flat_indices, weights=innovation_values, minlength=n_cycles * n_phase_bins).astype(float).reshape(n_cycles, n_phase_bins)
    event_bin_counts = np.bincount(flat_indices, minlength=n_cycles * n_phase_bins).astype(float).reshape(n_cycles, n_phase_bins)
    event_bin_sums = circular_window_sum_rows(event_bin_sums, edge_width_cadences)
    event_bin_counts = circular_window_sum_rows(event_bin_counts, edge_width_cadences)
    best = empty.copy()
    phase_positions = np.arange(n_phase_bins)
    for duration in duration_grid:
        duration = float(duration)
        if duration <= 0 or duration >= period:
            continue
        duration_bins = max(int(np.round(duration / phase_bin_width)), 1)
        if duration_bins >= n_phase_bins:
            continue
        egress_positions = phase_positions + duration_bins
        egress_phase_indices = np.mod(egress_positions, n_phase_bins)
        egress_cycle_offsets = (egress_positions // n_phase_bins).astype(int)
        ingress_sums = event_bin_sums
        ingress_counts = event_bin_counts
        egress_sums = np.zeros_like(event_bin_sums)
        egress_counts = np.zeros_like(event_bin_counts)
        same_cycle = egress_cycle_offsets == 0
        next_cycle = egress_cycle_offsets == 1
        egress_sums[:, same_cycle] = event_bin_sums[:, egress_phase_indices[same_cycle]]
        egress_counts[:, same_cycle] = event_bin_counts[:, egress_phase_indices[same_cycle]]
        if n_cycles > 1 and np.any(next_cycle):
            egress_sums[:-1, next_cycle] = event_bin_sums[1:, egress_phase_indices[next_cycle]]
            egress_counts[:-1, next_cycle] = event_bin_counts[1:, egress_phase_indices[next_cycle]]
        valid_events = (ingress_counts > 0) & (egress_counts > 0)
        event_differences = np.full_like(event_bin_sums, np.nan)
        event_standard_errors = np.full_like(event_bin_sums, np.nan)
        event_scores = np.full_like(event_bin_sums, np.nan)
        event_differences[valid_events] = egress_sums[valid_events] / egress_counts[valid_events] - ingress_sums[valid_events] / ingress_counts[valid_events]
        event_standard_errors[valid_events] = float(scale) * np.sqrt(1.0 / ingress_counts[valid_events] + 1.0 / egress_counts[valid_events])
        event_scores[valid_events] = event_differences[valid_events] / event_standard_errors[valid_events]
        n_valid_events = np.sum(valid_events, axis=0)
        n_positive_events = np.sum(event_scores > 0, axis=0)
        positive_event_fractions = np.divide(n_positive_events, n_valid_events, out=np.full(n_phase_bins, np.nan, dtype=float), where=n_valid_events > 0)
        median_event_scores = np.ma.median(np.ma.masked_invalid(event_scores), axis=0).filled(np.nan)
        median_event_amplitudes = np.ma.median(np.ma.masked_invalid(event_differences), axis=0).filled(np.nan)
        total_ingress_counts = np.sum(np.where(valid_events, ingress_counts, 0.0), axis=0)
        total_egress_counts = np.sum(np.where(valid_events, egress_counts, 0.0), axis=0)
        total_ingress_sums = np.sum(np.where(valid_events, ingress_sums, 0.0), axis=0)
        total_egress_sums = np.sum(np.where(valid_events, egress_sums, 0.0), axis=0)
        pooled_ingress_means = np.divide(total_ingress_sums, total_ingress_counts, out=np.full(n_phase_bins, np.nan, dtype=float), where=total_ingress_counts > 0)
        pooled_egress_means = np.divide(total_egress_sums, total_egress_counts, out=np.full(n_phase_bins, np.nan, dtype=float), where=total_egress_counts > 0)
        pooled_differences = pooled_egress_means - pooled_ingress_means
        inverse_ingress_counts = np.divide(1.0, total_ingress_counts, out=np.full(n_phase_bins, np.nan, dtype=float), where=total_ingress_counts > 0)
        inverse_egress_counts = np.divide(1.0, total_egress_counts, out=np.full(n_phase_bins, np.nan, dtype=float), where=total_egress_counts > 0)
        pooled_standard_errors = float(scale) * np.sqrt(inverse_ingress_counts + inverse_egress_counts)
        raw_pooled_scores = pooled_differences / pooled_standard_errors
        event_consistent_scores = median_event_scores * np.sqrt(n_valid_events)
        total_edge_observations = total_ingress_counts + total_egress_counts
        valid_phases = (n_valid_events >= int(min_transit_events)) & (total_edge_observations >= int(min_edge_observations)) & (positive_event_fractions >= float(min_event_consistency_fraction)) & np.isfinite(event_consistent_scores)
        scores = np.full(n_phase_bins, np.nan, dtype=float)
        scores[valid_phases] = event_consistent_scores[valid_phases]
        if not np.isfinite(scores).any():
            continue
        best_index = int(np.nanargmax(scores))
        best_score = float(scores[best_index])
        if np.isfinite(best["score"]) and best_score <= best["score"]:
            continue
        ingress_phase = (best_index + 0.5) * phase_bin_width
        center_phase = np.mod(ingress_phase + 0.5 * duration, period)
        best = {"period": float(period), "score": best_score, "raw_pooled_score": float(raw_pooled_scores[best_index]), "epoch": float(start_time + center_phase), "duration": duration, "edge_amplitude": float(median_event_amplitudes[best_index]), "n_edge_observations": int(total_edge_observations[best_index]), "n_valid_transit_events": int(n_valid_events[best_index]), "n_positive_transit_events": int(n_positive_events[best_index]), "positive_event_fraction": float(positive_event_fractions[best_index]), "median_event_score": float(median_event_scores[best_index])}
    return best

def select_top_peaks(periodogram, top_k=10, separation_fraction=0.01):
    required_columns = {"period", "score"}
    missing_columns = required_columns.difference(periodogram.columns)
    if missing_columns:
        raise ValueError(f"Periodogram is missing columns: {sorted(missing_columns)}")
    working = periodogram.copy()
    working["periodogram_index"] = np.arange(len(working))
    working = working[np.isfinite(working["period"]) & np.isfinite(working["score"])].reset_index(drop=True)
    if working.empty:
        return pd.DataFrame()
    scores = working["score"].to_numpy(dtype=float)
    if scores.size < 3:
        candidate_positions = np.arange(scores.size)
    else:
        interior_positions = np.flatnonzero((scores[1:-1] >= scores[:-2]) & (scores[1:-1] >= scores[2:])) + 1
        boundary_positions = []
        if scores[0] >= scores[1]:
            boundary_positions.append(0)
        if scores[-1] >= scores[-2]:
            boundary_positions.append(scores.size - 1)
        candidate_positions = np.unique(np.concatenate([interior_positions, np.asarray(boundary_positions, dtype=int)]))
    if candidate_positions.size == 0:
        candidate_positions = np.asarray([int(np.nanargmax(scores))])
    ordered_positions = candidate_positions[np.argsort(scores[candidate_positions])[::-1]]
    selected = []
    for position in ordered_positions:
        row = working.iloc[int(position)].to_dict()
        period = float(row["period"])
        too_close = any(abs(period - item["period_days"]) / item["period_days"] <= float(separation_fraction) for item in selected)
        if too_close:
            continue
        row["rank"] = len(selected) + 1
        row["period_days"] = period
        selected.append(row)
        if len(selected) >= int(top_k):
            break
    return pd.DataFrame(selected)

def evaluate_period_grid(time, innovations, period_grid, duration_grid, scale, cadence, edge_width_cadences, min_edge_observations, min_transit_events, min_event_consistency_fraction):
    rows = [score_period(time, innovations, period, duration_grid, scale, cadence, edge_width_cadences=edge_width_cadences, min_edge_observations=min_edge_observations, min_transit_events=min_transit_events, min_event_consistency_fraction=min_event_consistency_fraction) for period in period_grid]
    return pd.DataFrame(rows)

def run_tcf(time, innovations, period_grid, duration_grid, edge_width_cadences=0, min_edge_observations=4, min_transit_events=3, min_event_consistency_fraction=0.60, top_k=10, search_mode="coarse_to_fine", n_coarse_periods=2000, n_refinement_regions=20, refinement_half_width_points=30):
    time = np.asarray(time, dtype=float).reshape(-1)
    innovations = np.asarray(innovations, dtype=float).reshape(-1)
    period_grid = np.asarray(period_grid, dtype=float).reshape(-1)
    duration_grid = np.asarray(duration_grid, dtype=float).reshape(-1)
    if time.shape != innovations.shape:
        raise ValueError("time and innovations must have the same shape.")
    if period_grid.size == 0 or duration_grid.size == 0:
        raise ValueError("Period and duration grids must not be empty.")
    if search_mode not in {"exhaustive", "coarse_to_fine"}:
        raise ValueError("search_mode must be exhaustive or coarse_to_fine.")
    finite = np.isfinite(time) & np.isfinite(innovations)
    if finite.sum() < 24:
        raise ValueError("At least 24 finite innovation values are required.")
    cadence = median_cadence(time[finite])
    scale = robust_scale(innovations[finite])
    if search_mode == "exhaustive" or period_grid.size <= int(n_coarse_periods):
        evaluated_positions = np.arange(period_grid.size)
        periodogram = evaluate_period_grid(time, innovations, period_grid, duration_grid, scale, cadence, edge_width_cadences, min_edge_observations, min_transit_events, min_event_consistency_fraction)
        coarse_period_count = int(period_grid.size)
        refined_period_count = 0
    else:
        coarse_period_count = min(int(n_coarse_periods), int(period_grid.size))
        coarse_positions = np.unique(np.round(np.linspace(0, period_grid.size - 1, coarse_period_count)).astype(int))
        coarse_grid = period_grid[coarse_positions]
        coarse_periodogram = evaluate_period_grid(time, innovations, coarse_grid, duration_grid, scale, cadence, edge_width_cadences, min_edge_observations, min_transit_events, min_event_consistency_fraction)
        coarse_top_k = max(int(top_k), int(n_refinement_regions))
        coarse_top_peaks = select_top_peaks(coarse_periodogram, top_k=coarse_top_k, separation_fraction=0.01)
        refinement_positions = []
        for period in coarse_top_peaks["period_days"].head(int(n_refinement_regions)):
            center_position = int(np.argmin(np.abs(period_grid - float(period))))
            lower_position = max(0, center_position - int(refinement_half_width_points))
            upper_position = min(period_grid.size, center_position + int(refinement_half_width_points) + 1)
            refinement_positions.extend(range(lower_position, upper_position))
        refinement_positions = np.asarray(refinement_positions, dtype=int)
        evaluated_positions = np.unique(np.concatenate([coarse_positions, refinement_positions]))
        additional_positions = np.setdiff1d(evaluated_positions, coarse_positions, assume_unique=True)
        if additional_positions.size > 0:
            additional_grid = period_grid[additional_positions]
            additional_periodogram = evaluate_period_grid(time, innovations, additional_grid, duration_grid, scale, cadence, edge_width_cadences, min_edge_observations, min_transit_events, min_event_consistency_fraction)
            periodogram = pd.concat([coarse_periodogram, additional_periodogram], ignore_index=True)
        else:
            periodogram = coarse_periodogram
        periodogram = periodogram.sort_values("period").reset_index(drop=True)
        coarse_period_count = int(coarse_positions.size)
        refined_period_count = int(additional_positions.size)
    top_peaks = select_top_peaks(periodogram, top_k=top_k, separation_fraction=0.01)
    if top_peaks.empty:
        raise ValueError("TCF did not produce any finite peaks.")
    summary = top_peaks.iloc[0].to_dict()
    summary["cadence_days"] = cadence
    summary["innovation_scale"] = scale
    summary["min_transit_events"] = int(min_transit_events)
    summary["min_event_consistency_fraction"] = float(min_event_consistency_fraction)
    summary["search_mode"] = search_mode
    summary["requested_period_count"] = int(period_grid.size)
    summary["evaluated_period_count"] = int(len(periodogram))
    summary["coarse_period_count"] = int(coarse_period_count)
    summary["refined_period_count"] = int(refined_period_count)
    search_summary = {"search_mode": search_mode, "requested_period_count": int(period_grid.size), "evaluated_period_count": int(len(periodogram)), "coarse_period_count": int(coarse_period_count), "refined_period_count": int(refined_period_count), "n_refinement_regions": int(n_refinement_regions), "refinement_half_width_points": int(refinement_half_width_points)}
    return {"summary": summary, "periodogram": periodogram, "top_peaks": top_peaks, "search_summary": search_summary}
def period_match_fraction(recovered_period, injected_period):
    recovered_period = float(recovered_period)
    injected_period = float(injected_period)
    if not np.isfinite(recovered_period) or not np.isfinite(injected_period) or injected_period <= 0:
        return np.inf
    harmonic_periods = [0.5 * injected_period, injected_period, 2.0 * injected_period]
    errors = [abs(recovered_period - period) / period for period in harmonic_periods]
    return float(min(errors))

def peak_records(peaks):
    if isinstance(peaks, pd.DataFrame):
        return peaks.to_dict(orient="records")
    return list(peaks)

def matching_peak_rank(peaks, target_period_days, tolerance_fraction=0.02):
    target_period_days = float(target_period_days)
    if not np.isfinite(target_period_days) or target_period_days <= 0:
        return None
    for fallback_rank, peak in enumerate(peak_records(peaks), start=1):
        period = float(peak.get("period_days", peak.get("period", np.nan)))
        rank = int(peak.get("rank", fallback_rank))
        error = abs(period - target_period_days) / target_period_days
        if np.isfinite(error) and error <= float(tolerance_fraction):
            return rank
    return None

def harmonic_peak_rank(peaks, target_period_days, ratio, tolerance_fraction=0.02):
    target_period_days = float(target_period_days)
    ratio = float(ratio)
    harmonic_period = target_period_days * ratio
    if not np.isfinite(harmonic_period) or harmonic_period <= 0:
        return None
    return matching_peak_rank(peaks, harmonic_period, tolerance_fraction=tolerance_fraction)