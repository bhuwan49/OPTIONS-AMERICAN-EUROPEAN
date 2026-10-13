# -*- coding: utf-8 -*-

import math

import pytest

import optnlib
from optnlib import european


def test_black_scholes_known_call():
    # Hull, Options, Futures and Other Derivatives (textbook value).
    assert european.black_scholes("c", 100, 100, 1.0, 0.05, 0.2) == pytest.approx(
        10.4506, abs=1e-4
    )


def test_black_scholes_known_put():
    assert european.black_scholes("p", 100, 100, 1.0, 0.05, 0.2) == pytest.approx(
        5.5735, abs=1e-4
    )


def test_black_scholes_put_call_parity():
    S, K, t, r, sigma = 100.0, 95.0, 0.75, 0.04, 0.25
    call = european.black_scholes("c", S, K, t, r, sigma)
    put = european.black_scholes("p", S, K, t, r, sigma)
    assert call - put == pytest.approx(S - K * math.exp(-r * t), abs=1e-10)


def test_black_scholes_merton_reduces_to_bs():
    a = european.black_scholes("c", 100, 100, 1.0, 0.05, 0.2)
    b = european.black_scholes_merton("c", 100, 100, 1.0, 0.05, 0.2, q=0.0)
    assert a == pytest.approx(b, abs=1e-12)


def test_black_scholes_merton_known_value():
    assert european.black_scholes_merton(
        "c", 100, 100, 1.0, 0.05, 0.2, 0.03
    ) == pytest.approx(8.6525, abs=1e-4)


def test_black_scholes_merton_parity_with_yield():
    S, K, t, r, q, sigma = 100.0, 90.0, 1.25, 0.03, 0.02, 0.3
    call = european.black_scholes_merton("c", S, K, t, r, sigma, q)
    put = european.black_scholes_merton("p", S, K, t, r, sigma, q)
    expected = S * math.exp(-q * t) - K * math.exp(-r * t)
    assert call - put == pytest.approx(expected, abs=1e-10)


def test_black76_matches_bsm_forward():
    F, K, t, r, sigma = 105.0, 100.0, 1.0, 0.04, 0.2
    b76 = european.black("c", F, K, t, r, sigma)
    # A forward of F corresponds to S = F e^{-r t} with no yield.
    bsm = european.black_scholes("c", F * math.exp(-r * t), K, t, r, sigma)
    assert b76 == pytest.approx(bsm, abs=1e-10)


def test_intrinsic_at_zero_time():
    assert european.price("c", 110, 100, 0.0, 0.05, 0.2) == pytest.approx(10.0)
    assert european.price("p", 90, 100, 0.0, 0.05, 0.2) == pytest.approx(10.0)
    assert european.price("c", 90, 100, 0.0, 0.05, 0.2) == 0.0


def test_zero_volatility_deterministic():
    val = european.price("c", 100, 90, 1.0, 0.05, 0.0)
    assert val == pytest.approx(100 - 90 * math.exp(-0.05), abs=1e-12)


def test_forward_and_escrowed_models_agree_at_ex_date():
    # A single dividend at the very end is (almost) model independent.
    common = dict(flag="c", S=100, K=100, t=1.0, r=0.05, sigma=0.2)
    divs = [(1.0, 2.0)]
    fwd = european.price(divs=divs, dividend_model="forward", **common)
    esc = european.price(divs=divs, dividend_model="escrowed", **common)
    assert fwd == pytest.approx(esc, abs=1e-8)


def test_discrete_dividend_lowers_call():
    no_div = european.price("c", 100, 100, 1.0, 0.05, 0.2)
    with_div = european.price("c", 100, 100, 1.0, 0.05, 0.2, divs=[(0.5, 3.0)])
    assert with_div < no_div
