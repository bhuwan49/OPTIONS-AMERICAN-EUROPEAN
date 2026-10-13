# -*- coding: utf-8 -*-

import pytest

import optnlib
from optnlib.exceptions import InvalidInputError


def test_public_api_surface():
    for name in (
        "price", "european_price", "american_price", "black_scholes",
        "black_scholes_merton", "black", "implied_volatility", "greeks",
        "delta", "gamma", "theta", "vega", "rho",
    ):
        assert hasattr(optnlib, name), name


def test_unified_price_dispatches():
    euro = optnlib.price("c", 100, 100, 1.0, 0.05, 0.2, style="european")
    amer = optnlib.price("p", 100, 100, 1.0, 0.05, 0.2, style="american")
    assert euro == pytest.approx(optnlib.black_scholes("c", 100, 100, 1.0, 0.05, 0.2))
    assert amer == pytest.approx(optnlib.american_price("p", 100, 100, 1.0, 0.05, 0.2))


def test_unified_price_american_method():
    lr = optnlib.price("p", 100, 100, 1.0, 0.05, 0.2, style="american", method="lr")
    assert lr == pytest.approx(
        optnlib.american_price("p", 100, 100, 1.0, 0.05, 0.2, method="lr")
    )


def test_aliases():
    assert optnlib.black_scholes("c", 100, 100, 1.0, 0.05, 0.2) == optnlib.european.price(
        "c", 100, 100, 1.0, 0.05, 0.2
    )


def test_flag_aliases():
    assert optnlib.black_scholes("call", 100, 100, 1.0, 0.05, 0.2) == optnlib.black_scholes(
        "c", 100, 100, 1.0, 0.05, 0.2
    )
    assert optnlib.black_scholes("Put", 100, 100, 1.0, 0.05, 0.2) == optnlib.black_scholes(
        "p", 100, 100, 1.0, 0.05, 0.2
    )


def test_invalid_strike():
    with pytest.raises(InvalidInputError):
        optnlib.black_scholes("c", 100, 0, 1.0, 0.05, 0.2)


def test_invalid_spot():
    with pytest.raises(InvalidInputError):
        optnlib.black_scholes("c", -1, 100, 1.0, 0.05, 0.2)


def test_invalid_time():
    with pytest.raises(InvalidInputError):
        optnlib.black_scholes("c", 100, 100, -1.0, 0.05, 0.2)


def test_invalid_volatility():
    with pytest.raises(InvalidInputError):
        optnlib.black_scholes("c", 100, 100, 1.0, 0.05, -0.2)


def test_invalid_flag():
    with pytest.raises(ValueError):
        optnlib.black_scholes("x", 100, 100, 1.0, 0.05, 0.2)


def test_invalid_style():
    with pytest.raises(ValueError):
        optnlib.price("c", 100, 100, 1.0, 0.05, 0.2, style="bermudan")
