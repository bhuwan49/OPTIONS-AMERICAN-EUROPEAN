# -*- coding: utf-8 -*-

import pytest

import optnlib
from optnlib import greeks as greeks_fn
from optnlib.greeks import analytical_greeks, numerical_greeks


def test_known_black_scholes_greeks():
    d = greeks_fn("c", 100, 100, 1.0, 0.05, 0.2)
    assert d["delta"] == pytest.approx(0.6368, abs=1e-3)
    assert d["gamma"] == pytest.approx(0.01876, abs=1e-4)
    assert d["vega"] == pytest.approx(0.3752, abs=1e-3)
    assert d["rho"] == pytest.approx(0.5323, abs=1e-3)
    assert d["theta"] == pytest.approx(-0.01757, abs=1e-4)


def test_analytical_matches_numerical_european():
    ana = analytical_greeks("c", 100, 100, 1.0, 0.05, 0.2, 0.03)
    num = numerical_greeks("c", 100, 100, 1.0, 0.05, 0.2, q=0.03)
    for key in ana:
        assert ana[key] == pytest.approx(num[key], rel=1e-3, abs=1e-6)


def test_delta_put_call_relationship():
    call = greeks_fn("c", 100, 100, 1.0, 0.05, 0.2)["delta"]
    put = greeks_fn("p", 100, 100, 1.0, 0.05, 0.2)["delta"]
    # With no dividends, delta_call - delta_put = 1.
    assert call - put == pytest.approx(1.0, abs=1e-10)


def test_gamma_equals_for_call_and_put():
    gc = greeks_fn("c", 100, 105, 0.5, 0.03, 0.25)["gamma"]
    gp = greeks_fn("p", 100, 105, 0.5, 0.03, 0.25)["gamma"]
    assert gc == pytest.approx(gp, abs=1e-12)


def test_vega_equals_for_call_and_put():
    vc = greeks_fn("c", 100, 105, 0.5, 0.03, 0.25)["vega"]
    vp = greeks_fn("p", 100, 105, 0.5, 0.03, 0.25)["vega"]
    assert vc == pytest.approx(vp, abs=1e-12)


def test_american_greeks_numerical():
    d = greeks_fn("p", 100, 100, 1.0, 0.05, 0.2, style="american", method="lr")
    assert d["delta"] < 0.0
    assert d["gamma"] > 0.0


def test_greeks_with_discrete_dividend_use_numerical():
    d = greeks_fn("c", 100, 100, 1.0, 0.05, 0.2, divs=[(0.5, 3.0)])
    assert d["delta"] > 0.0
    assert d["gamma"] > 0.0


def test_individual_helpers_match_dict():
    kwargs = dict(q=0.02)
    assert optnlib.delta("p", 100, 100, 1.0, 0.05, 0.2, **kwargs) == pytest.approx(
        greeks_fn("p", 100, 100, 1.0, 0.05, 0.2, **kwargs)["delta"]
    )
    assert optnlib.gamma("p", 100, 100, 1.0, 0.05, 0.2, **kwargs) == pytest.approx(
        greeks_fn("p", 100, 100, 1.0, 0.05, 0.2, **kwargs)["gamma"]
    )
