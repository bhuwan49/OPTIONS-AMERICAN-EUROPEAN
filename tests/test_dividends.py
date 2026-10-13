# -*- coding: utf-8 -*-

import math

import pytest

from optnlib import dividends
from optnlib.exceptions import InvalidInputError


def test_normalize_sorts_and_filters():
    out = dividends.normalize_divs([(0.75, 1.0), (0.25, 2.0), (2.0, 3.0)], t=1.0)
    assert out == [(0.25, 2.0), (0.75, 1.0)]


def test_normalize_rejects_nonpositive_ex_time():
    with pytest.raises(InvalidInputError):
        dividends.normalize_divs([(0.0, 1.0)], t=1.0)


def test_normalize_skips_zero_amount():
    assert dividends.normalize_divs([(0.5, 0.0)], t=1.0) == []


def test_pv_discrete():
    val = dividends.pv_discrete([(0.5, 2.0), (1.0, 3.0)], r=0.05, t=1.0)
    expected = 2.0 * math.exp(-0.05 * 0.5) + 3.0 * math.exp(-0.05 * 1.0)
    assert val == pytest.approx(expected, abs=1e-12)


def test_adjusted_forward():
    S, t, r, q = 100.0, 1.0, 0.05, 0.0
    divs = [(0.5, 2.0)]
    fwd = dividends.adjusted_forward(S, t, r, q, divs)
    expected = S * math.exp(r * t) - 2.0 * math.exp(r * 0.5)
    assert fwd == pytest.approx(expected, abs=1e-12)


def test_adjusted_spot():
    val = dividends.adjusted_spot(100.0, 0.05, [(0.5, 2.0)], 1.0)
    assert val == pytest.approx(100.0 - 2.0 * math.exp(-0.05 * 0.5), abs=1e-12)


def test_effective_yield_reduces_to_q():
    assert dividends.effective_yield(100, 1.0, 0.05, 0.03, None) == pytest.approx(0.03)


def test_effective_yield_matches_constant_forward():
    # A discrete dividend and the equivalent continuous yield must produce the
    # same forward.
    S, t, r, q = 100.0, 1.0, 0.05, 0.0
    divs = [(0.5, 3.0)]
    q_eff = dividends.effective_yield(S, t, r, q, divs, "forward")
    fwd_div = dividends.adjusted_forward(S, t, r, q, divs)
    fwd_yield = S * math.exp((r - q_eff) * t)
    assert fwd_div == pytest.approx(fwd_yield, abs=1e-10)


def test_validate_model():
    assert dividends.validate_model("forward") == "forward"
    assert dividends.validate_model("escrowed") == "escrowed"
    with pytest.raises(InvalidInputError):
        dividends.validate_model("bad")


def test_escrowed_yield_matches_escrowed_forward():
    S, t, r, q = 100.0, 1.0, 0.05, 0.0
    divs = [(0.5, 3.0)]
    q_eff = dividends.effective_yield(S, t, r, q, divs, "escrowed")
    pv = dividends.pv_discrete(divs, r, t)
    fwd_escrowed = (S - pv) * math.exp(r * t)
    assert S * math.exp((r - q_eff) * t) == pytest.approx(fwd_escrowed, abs=1e-10)
