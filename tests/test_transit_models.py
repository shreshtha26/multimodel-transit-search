import numpy as np
from adaptive_transit.transit_models.periodic import periodic_box_transit_template, transit_center_times

def test_periodic_box_template_marks_repeated_transits():
    time = np.linspace(0.0, 20.0, 1000)
    template, in_transit = periodic_box_transit_template(time, period_days=5.0, epoch_days=1.0, duration_days=0.2, depth=0.01)
    centers = transit_center_times(time, period_days=5.0, epoch_days=1.0, duration_days=0.2)
    assert len(centers) == 4
    assert in_transit.sum() > 0
    assert np.all(template[in_transit] == -0.01)
    assert np.all(template[~in_transit] == 0.0)
