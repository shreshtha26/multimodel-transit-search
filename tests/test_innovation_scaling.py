import numpy as np

from adaptive_transit.noise_models.scaling import (
    standardize_innovations,
    trailing_robust_scale,
)


def test_trailing_robust_scale_returns_positive_scale() -> None:
    rng = np.random.default_rng(12)
    values = rng.normal(size=100)
    values[20:22] = np.nan

    scale = trailing_robust_scale(values, window=20)

    assert scale.shape == values.shape
    assert np.all(np.isfinite(scale))
    assert np.all(scale > 0)


def test_standardize_innovations_preserves_unusable_as_nan() -> None:
    innovations = np.array([1.0, 2.0, np.nan, 4.0])
    scale = np.ones(4)
    usable = np.array([True, True, False, False])

    standardized = standardize_innovations(innovations, scale, usable)

    np.testing.assert_allclose(standardized[:2], [1.0, 2.0])
    assert np.isnan(standardized[2])
    assert np.isnan(standardized[3])
