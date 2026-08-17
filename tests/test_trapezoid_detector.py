import numpy as np
import pandas as pd

from adaptive_transit.detection.trapezoid import (
    periodic_trapezoid_shape,
    run_bls_seeded_trapezoid,
)


def test_bls_seeded_trapezoid_refines_known_shape():
    time = np.linspace(0.0, 30.0, 3000)
    period = 3.5
    epoch = 1.0
    duration = 0.20
    shape = periodic_trapezoid_shape(
        time,
        period_days=period,
        epoch_days=epoch,
        duration_days=duration,
        ingress_fraction=0.20,
    )
    flux = -1.0e-3 * shape
    bls_result = {
        "top_peaks": pd.DataFrame(
            [
                {
                    "period": period,
                    "transit_time": epoch,
                    "duration": duration,
                    "power": 10.0,
                }
            ]
        )
    }
    result = run_bls_seeded_trapezoid(
        time,
        flux,
        bls_result,
        duration_grid=np.array([duration]),
        ingress_fractions=(0.10, 0.20, 0.30),
        top_k_periods=1,
        phase_offset_fractions=(0.0,),
    )
    best = result["summary"]
    assert abs(best["period_days"] - period) < 1.0e-12
    assert abs(best["duration_days"] - duration) < 1.0e-12
    assert abs(best["ingress_fraction"] - 0.20) < 1.0e-12
    assert abs(best["depth"] - 1.0e-3) < 1.0e-8
