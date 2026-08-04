import numpy as np
from adaptive_transit.detection.bls import period_match_fraction, run_bls
from adaptive_transit.injections.synthetic import inject_periodic_box_transit

def test_bls_recovers_injected_period():
    rng = np.random.default_rng(123)
    time = np.arange(0.0, 30.0, 0.0204)
    flux = rng.normal(0.0, 0.001, time.size)
    flux_error = np.full(time.size, 0.001)
    injected, _, _ = inject_periodic_box_transit(time, flux, period_days=5.0, epoch_days=1.2, duration_days=0.2, depth=0.01)
    period_grid = np.linspace(2.0, 8.0, 500)
    duration_grid = np.asarray([0.12, 0.2, 0.28])
    result = run_bls(time, injected, flux_error, period_grid, duration_grid, top_k=5)
    summary = result["summary"]
    assert period_match_fraction(summary["period"], 5.0) < 0.01
    assert summary["depth"] > 0.005
    assert len(result["top_peaks"]) == 5

def test_bls_allows_missing_values():
    time = np.arange(0.0, 20.0, 0.0204)
    flux = np.zeros(time.size)
    flux[40:45] = np.nan
    injected, _, _ = inject_periodic_box_transit(time, flux, period_days=4.0, epoch_days=1.0, duration_days=0.2, depth=0.01)
    result = run_bls(time, injected, None, np.linspace(2.0, 6.0, 300), np.asarray([0.2]), top_k=3)
    assert result["summary"]["n_observations"] == int(np.isfinite(injected).sum())
    assert len(result["top_peaks"]) == 3
