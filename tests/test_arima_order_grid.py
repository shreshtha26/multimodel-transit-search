from __future__ import annotations

import pytest

from adaptive_transit.noise_models.arima import generate_arima_orders


def test_generate_arima_orders_excludes_zero_order_by_default() -> None:
    orders = generate_arima_orders(max_p=1, max_d=1, max_q=1)

    assert (0, 0, 0) not in orders
    assert (1, 0, 0) in orders
    assert (0, 1, 1) in orders


def test_generate_arima_orders_applies_complexity_cap() -> None:
    orders = generate_arima_orders(max_p=3, max_d=1, max_q=3, max_total_order=2)

    assert (1, 1, 0) in orders
    assert (2, 0, 0) in orders
    assert (2, 1, 0) not in orders
    assert all(sum(order) <= 2 for order in orders)


def test_generate_arima_orders_rejects_negative_bounds() -> None:
    with pytest.raises(ValueError):
        generate_arima_orders(max_p=-1, max_d=1, max_q=1)
