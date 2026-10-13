# -*- coding: utf-8 -*-

import math

import pytest

from optnlib import american


def test_american_put_ge_european():
    from optnlib import european

    euro = european.black_scholes("p", 100, 100, 1.0, 0.05, 0.2)
    for method in ("bs2002", "crr", "lr"):
        amer = american.price("p", 100, 100, 1.0, 0.05, 0.2, method=method)
        assert amer >= euro - 1e-9


def test_american_put_ge_intrinsic():
    val = american.price("p", 80, 100, 1.0, 0.05, 0.2, method="lr")
    assert val >= 100 - 80


def test_american_call_no_dividend_equals_european():
    from optnlib import european

    euro = european.black_scholes("c", 100, 100, 1.0, 0.05, 0.2)
    assert american.price("c", 100, 100, 1.0, 0.05, 0.2, method="lr") == pytest.approx(
        euro, abs=1e-3
    )
    assert american.price(
        "c", 100, 100, 1.0, 0.05, 0.2, method="bs2002"
    ) == pytest.approx(euro, abs=1e-6)


def test_bs2002_call_matches_lr():
    for S in (80, 90, 100, 110, 120):
        a = american.price("c", S, 100, 1.0, 0.05, 0.2, q=0.03, method="bs2002")
        t = american.price("c", S, 100, 1.0, 0.05, 0.2, q=0.03, method="lr")
        assert a == pytest.approx(t, rel=5e-3)


def test_bs2002_put_is_lower_bound():
    # BS2002 is a lower bound; it must not exceed the tree value.
    for S in (80, 100, 120):
        a = american.price("p", S, 100, 1.0, 0.05, 0.2, q=0.03, method="bs2002")
        t = american.price("p", S, 100, 1.0, 0.05, 0.2, q=0.03, method="lr")
        assert a <= t + 1e-6


def test_crr_converges_to_lr():
    crr = american.price("p", 100, 100, 1.0, 0.05, 0.2, method="crr", n_steps=2000)
    lr = american.price("p", 100, 100, 1.0, 0.05, 0.2, method="lr", n_steps=501)
    assert crr == pytest.approx(lr, abs=2e-3)


def test_put_call_symmetry():
    # P(S, K, r, q) == C(K, S, rate=r-b, yield=r) with b = r - q.
    S, K, t, r, q, sigma = 100.0, 110.0, 0.5, 0.04, 0.02, 0.25
    put = american.price("p", S, K, t, r, sigma, q=q, method="bs2002")
    call = american.price("c", K, S, t, r - (r - q), sigma, q=r, method="bs2002")
    assert put == pytest.approx(call, abs=1e-8)


def test_zero_interest_rate_put_equals_european():
    from optnlib import european

    euro = european.black_scholes("p", 100, 100, 1.0, 0.0, 0.2)
    amer = american.price("p", 100, 100, 1.0, 0.0, 0.2, method="bs2002")
    assert amer == pytest.approx(euro, abs=1e-6)


def test_invalid_method():
    from optnlib.exceptions import InvalidInputError

    with pytest.raises(InvalidInputError):
        american.price("p", 100, 100, 1.0, 0.05, 0.2, method="nope")


def test_discrete_dividend_lowers_american_call():
    no_div = american.price("c", 100, 100, 1.0, 0.05, 0.2, method="lr")
    with_div = american.price(
        "c", 100, 100, 1.0, 0.05, 0.2, divs=[(0.5, 5.0)], method="lr"
    )
    assert with_div < no_div
