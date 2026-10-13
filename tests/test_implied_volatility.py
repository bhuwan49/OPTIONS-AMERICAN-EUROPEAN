# -*- coding: utf-8 -*-

import pytest

import optnlib
from optnlib.exceptions import InvalidInputError
from optnlib.implied_volatility import implied_volatility


def test_european_round_trip():
    price = optnlib.european_price("c", 100, 100, 1.0, 0.05, 0.2)
    recovered = implied_volatility(price, "c", 100, 100, 1.0, 0.05)
    assert recovered == pytest.approx(0.2, abs=1e-6)


def test_european_round_trip_with_yield():
    price = optnlib.european_price("p", 105, 100, 0.75, 0.03, 0.35, q=0.02)
    recovered = implied_volatility(price, "p", 105, 100, 0.75, 0.03, q=0.02)
    assert recovered == pytest.approx(0.35, abs=1e-6)


def test_american_round_trip_bs2002():
    price = optnlib.american_price("p", 100, 100, 1.0, 0.05, 0.25)
    recovered = implied_volatility(
        price, "p", 100, 100, 1.0, 0.05, style="american", method="bs2002"
    )
    assert recovered == pytest.approx(0.25, abs=1e-5)


def test_american_round_trip_tree():
    price = optnlib.american_price("p", 100, 100, 1.0, 0.05, 0.25, method="lr")
    recovered = implied_volatility(
        price, "p", 100, 100, 1.0, 0.05, style="american", method="lr"
    )
    assert recovered == pytest.approx(0.25, abs=1e-4)


def test_implied_volatility_with_dividends():
    price = optnlib.european_price("c", 100, 100, 1.0, 0.05, 0.3, divs=[(0.5, 2.0)])
    recovered = implied_volatility(
        price, "c", 100, 100, 1.0, 0.05, divs=[(0.5, 2.0)]
    )
    assert recovered == pytest.approx(0.3, abs=1e-5)


def test_implied_volatility_rejects_zero_time():
    with pytest.raises(InvalidInputError):
        implied_volatility(5.0, "c", 100, 100, 0.0, 0.05)


def test_implied_volatility_rejects_negative_price():
    with pytest.raises(InvalidInputError):
        implied_volatility(-1.0, "c", 100, 100, 1.0, 0.05)
