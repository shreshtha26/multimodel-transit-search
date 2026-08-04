import numpy as np
from adaptive_transit.detection.false_alarm import moving_block_surrogate, empirical_fap


def test_block_surrogate_preserves_missing_mask():
    flux = np.arange(100, dtype=float)
    flux[20:25] = np.nan

    surrogate = moving_block_surrogate(
        flux,
        block_size=10,
        rng=np.random.default_rng(123))

    assert surrogate.shape == flux.shape
    assert np.array_equal(np.isnan(surrogate), np.isnan(flux))


def test_block_surrogate_is_reproducible():
    flux = np.arange(100, dtype=float)

    first = moving_block_surrogate(
        flux,
        block_size=10,
        rng=np.random.default_rng(123))

    second = moving_block_surrogate(
        flux,
        block_size=10,
        rng=np.random.default_rng(123))

    assert np.array_equal(first, second)


def test_empirical_fap():
    null_powers = np.asarray([1.0, 2.0, 3.0, 5.0])

    result = empirical_fap(
        observed_power=4.0,
        null_max_powers=null_powers)

    assert np.isclose(result, 0.4)


def test_empirical_fap_rejects_empty_input():
    try:
        empirical_fap(
            observed_power=4.0,
            null_max_powers=np.asarray([]))
    except ValueError:
        return

    raise AssertionError("Expected ValueError for empty null powers.")