import numpy as np
import pytest

from adaptive_transit.transit_models.box import box_transit_mask, box_transit_template
from adaptive_transit.transit_models.periodic import periodic_box_transit_template, transit_center_times


def test_box_transit_mask_uses_requested_cadence_count():
    cadence = np.arange(20)

    mask = box_transit_mask(cadence, center_cadenceno=10, duration_cadences=4)

    assert cadence[mask].tolist() == [9, 10, 11, 12]


def test_box_transit_template_rejects_non_finite_parameters():
    cadence = np.arange(5)

    with pytest.raises(ValueError, match="duration_cadences"):
        box_transit_template(cadence, center_cadenceno=2, duration_cadences=np.nan, depth=0.01)
    with pytest.raises(ValueError, match="center_cadenceno"):
        box_transit_template(cadence, center_cadenceno=np.inf, duration_cadences=2, depth=0.01)
    with pytest.raises(ValueError, match="depth"):
        box_transit_template(cadence, center_cadenceno=2, duration_cadences=2, depth=np.inf)


def test_box_transit_template_marks_negative_dip():
    cadence = np.arange(6)

    template, in_transit = box_transit_template(cadence, center_cadenceno=2, duration_cadences=3, depth=0.01)

    assert cadence[in_transit].tolist() == [1, 2, 3]
    assert np.all(template[in_transit] == -0.01)
    assert np.all(template[~in_transit] == 0.0)


def test_periodic_box_template_marks_repeated_transits():
    time = np.linspace(0.0, 20.0, 1000)
    template, in_transit = periodic_box_transit_template(time, period_days=5.0, epoch_days=1.0, duration_days=0.2, depth=0.01)
    centers = transit_center_times(time, period_days=5.0, epoch_days=1.0, duration_days=0.2)
    assert len(centers) == 4
    assert in_transit.sum() > 0
    assert np.all(template[in_transit] == -0.01)
    assert np.all(template[~in_transit] == 0.0)
