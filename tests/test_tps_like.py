import numpy as np

from adaptive_transit.detection.tps_like import (
    _centered_square_pulse,
    combine_periodic_events,
    duration_hours_to_cadences,
    period_grid_to_unique_cadences,
)


def test_periodic_combination_prefers_true_cadence_period():
    n = 120
    numerator = np.zeros(n, dtype=float)
    denominator = np.ones(n, dtype=float)
    numerator[np.arange(3, n, 10)] = 5.0

    candidates = []
    for period in (8, 9, 10, 11, 12):
        candidate = combine_periodic_events(
            numerator,
            denominator,
            period,
            duration_cadences=3,
            min_events=3,
        )
        candidates.append(candidate)

    best = max(candidates, key=lambda row: row.mes)
    assert best.period_cadences == 10
    assert best.epoch_phase_cadence == 3
    assert best.observed_event_count == 12


def test_duration_and_period_grid_are_cadence_quantized():
    cadence_days = 30.0 / 60.0 / 24.0
    assert duration_hours_to_cadences(3.0, cadence_days) == 6
    periods = period_grid_to_unique_cadences(1.0, 2.0, cadence_days)
    assert periods[0] >= 48
    assert periods[-1] <= 96
    assert np.all(np.diff(periods) == 1)


def test_square_pulse_has_requested_negative_width():
    pulse = _centered_square_pulse(101, 7)
    assert np.sum(pulse < 0) == 7
    assert np.all(pulse[pulse < 0] == -1.0)
    assert np.isclose(np.sum(pulse), -7.0)
