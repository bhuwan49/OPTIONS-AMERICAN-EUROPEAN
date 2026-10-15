# -*- coding: utf-8 -*-

"""Tests for :mod:`optnlib.hedging`."""

import math

import numpy as np
import pytest

import optnlib
from optnlib import hedging as h
from optnlib.exceptions import InvalidInputError

MKT = dict(S=100.0, t=0.25, r=0.05, sigma=0.25)


def _short_call(**kw):
    p = dict(MKT)
    p.update(kw)
    return h.OptionLeg(quantity=-10, flag="c", K=105, multiplier=100,
                       name="short call", **p)


def _stock(qty=500.0):
    return h.Underlying(quantity=qty, S=100.0, name="stock")


def _hedge_basket(multiplier=100.0):
    return [
        h.Underlying(S=100.0, name="stock"),
        h.OptionLeg(flag="c", K=100, multiplier=multiplier, name="H1", **MKT),
        h.OptionLeg(flag="p", K=110, multiplier=multiplier, name="H2", **MKT),
    ]


# -- instruments -------------------------------------------------------------


def test_underlying_stock_greeks():
    u = h.Underlying(quantity=1.0, S=100.0)
    assert u.unit_greeks() == {"delta": 1.0, "gamma": 0.0, "theta": 0.0,
                               "vega": 0.0, "rho": 0.0}
    assert u.trade_unit == "shares"


def test_option_leg_unit_greeks_match_library():
    leg = h.OptionLeg(flag="c", K=105, **MKT)
    got = leg.unit_greeks()
    want = optnlib.greeks("c", 100, 105, 0.25, 0.05, 0.25)
    for g in ("delta", "gamma", "theta", "vega", "rho"):
        assert got[g] == pytest.approx(want[g], rel=1e-9, abs=1e-12)


def test_contract_and_position_scaling():
    leg = h.OptionLeg(quantity=-10, flag="c", K=105, multiplier=100, **MKT)
    ug = leg.unit_greeks()
    cg = leg.contract_greeks()
    pg = leg.position_greeks()
    for g in ("delta", "gamma", "theta", "vega", "rho"):
        assert cg[g] == pytest.approx(100.0 * ug[g])
        assert pg[g] == pytest.approx(-10.0 * 100.0 * ug[g])


# -- portfolio ---------------------------------------------------------------


def test_portfolio_greeks_and_value_additive():
    pf = h.Portfolio([_short_call(), _stock(500)])
    manual = {}
    for g in ("delta", "gamma", "theta", "vega", "rho"):
        manual[g] = sum(leg.position_greeks()[g] for leg in pf.legs)
    for g in manual:
        assert pf.greeks()[g] == pytest.approx(manual[g])
    assert pf.value() == pytest.approx(sum(l.value() for l in pf.legs))


def test_scenario_and_attribution_agree_for_small_shock():
    pf = h.Portfolio([_short_call(), _stock(500)])
    attr = pf.pnl_attribution(spot_shock=0.1, vol_shock=0.001, days=1.0)
    actual = pf.scenario(spot_shock=0.1, vol_shock=0.001, days=1.0)
    assert attr["actual"] == pytest.approx(actual)
    assert attr["total"] == pytest.approx(actual, abs=5e-1)
    assert attr["error"] == pytest.approx(attr["total"] - attr["actual"])


def test_portfolio_describe_returns_string():
    pf = h.Portfolio([_short_call(), _stock(500)])
    text = pf.describe()
    assert "Portfolio" in text and "delta" in text
    assert str(pf) == text


# -- solver ------------------------------------------------------------------


def test_stock_row_is_unit_delta_row():
    pf = h.Portfolio([_short_call(), _stock(500)])
    plan = h.hedge(pf, _hedge_basket())
    # row 0 = delta, col 0 = underlying
    assert plan.matrix[0, 0] == pytest.approx(1.0)
    # row 1 = gamma, row 2 = vega: stock contributes nothing
    assert plan.matrix[1, 0] == pytest.approx(0.0)
    assert plan.matrix[2, 0] == pytest.approx(0.0)


def test_three_by_three_neutralizes_delta_gamma_vega():
    # Options sharing spot/vol/expiry make the vega row collinear with the
    # gamma row (vega/gamma = S**2 * sigma * t), so the matrix is rank-2 and
    # the auto solver falls back to a minimum-norm least-squares hedge.
    pf = h.Portfolio([_short_call(), _stock(500)])
    plan = h.hedge(pf, _hedge_basket(), greeks=("delta", "gamma", "vega"))
    assert plan.method == "lstsq"
    assert plan.rank == 2 and plan.min_dim == 3
    for g in ("delta", "gamma", "vega"):
        assert plan.residual[g] == pytest.approx(0.0, abs=1e-6)
    # A w = rhs to least-squares accuracy
    assert np.allclose(plan.matrix.dot(plan.quantities), plan.rhs, atol=1e-6)


def test_well_conditioned_square_system_uses_solve():
    # Distinct expiries break the vega/gamma collinearity -> full rank.
    pf = h.Portfolio([_short_call(), _stock(500)])
    basket = [
        h.Underlying(S=100.0, name="stock"),
        h.OptionLeg(flag="c", K=100, multiplier=100, name="H1", **MKT),
        h.OptionLeg(flag="p", K=95, multiplier=100, name="H2",
                    **dict(MKT, t=0.35)),
    ]
    plan = h.hedge(pf, basket, greeks=("delta", "gamma", "vega"))
    assert plan.method == "solve"
    assert plan.rank == 3 and plan.min_dim == 3
    for g in ("delta", "gamma", "vega"):
        assert plan.residual[g] == pytest.approx(0.0, abs=1e-8)


def test_rank_deficient_inconsistent_returns_bounded_lstsq():
    # Portfolio 3M/25%-vol, hedge options 6M/28%-vol: rank-2 AND inconsistent,
    # so gamma/vega cannot be neutralized. Weights must stay bounded (the old
    # naive solve returned ~1e18 offsets) and a warning must be reported.
    pf = h.Portfolio([
        h.OptionLeg(quantity=-10, flag="c", K=105, multiplier=100, name="sc",
                    **MKT),
        _stock(500),
    ])
    basket = [
        h.Underlying(S=100.0, name="stock"),
        h.OptionLeg(flag="c", K=100, multiplier=100, name="H1",
                    S=100.0, t=0.50, r=0.05, sigma=0.28),
        h.OptionLeg(flag="p", K=95, multiplier=100, name="H2",
                    S=100.0, t=0.50, r=0.05, sigma=0.28),
    ]
    plan = h.hedge(pf, basket, greeks=("delta", "gamma", "vega"))
    assert plan.method == "lstsq"
    assert plan.rank == 2 and plan.min_dim == 3
    assert np.all(np.isfinite(plan.quantities))
    assert np.max(np.abs(plan.quantities)) < 1e6
    # delta is in the span, so it is neutralized; gamma/vega are not.
    assert plan.residual["delta"] == pytest.approx(0.0, abs=1e-6)
    assert abs(plan.residual["gamma"]) > 1.0
    assert "Warning" in plan.describe()


def test_strict_raises_on_rank_deficient():
    pf = h.Portfolio([_short_call(), _stock(500)])
    with pytest.raises(InvalidInputError):
        h.hedge(pf, _hedge_basket(), greeks=("delta", "gamma", "vega"),
                strict=True)


def test_method_override_forces_pinv():
    pf = h.Portfolio([_short_call(), _stock(500)])
    plan = h.hedge(pf, _hedge_basket(), greeks=("delta", "gamma", "vega"),
                   method="pinv")
    assert plan.method == "pinv"
    for g in ("delta", "gamma", "vega"):
        assert plan.residual[g] == pytest.approx(0.0, abs=1e-6)


def test_unknown_method_raises():
    pf = h.Portfolio([_short_call()])
    with pytest.raises(InvalidInputError):
        h.hedge(pf, _hedge_basket(), method="bogus")


def test_delta_hedge_uses_negative_of_portfolio_delta():
    pf = h.Portfolio([_short_call(), _stock(500)])
    plan = h.delta_hedge(pf)
    assert plan.quantities[0] == pytest.approx(-pf.delta())
    assert plan.residual["delta"] == pytest.approx(0.0, abs=1e-9)


def test_multiplier_scaling_is_consistent():
    pf = h.Portfolio([_short_call(), _stock(500)])
    pf_g = pf.greeks()
    plan1 = h.hedge(pf, _hedge_basket(multiplier=1.0))
    plan100 = h.hedge(pf, _hedge_basket(multiplier=100.0))
    # Whatever the contract size, the hedge legs reproduce the same position
    # greeks (i.e. they neutralize the portfolio's constrained greeks).
    for g in ("delta", "gamma", "vega"):
        assert plan1.to_portfolio().greeks()[g] == pytest.approx(-pf_g[g], abs=1e-6)
        assert plan100.to_portfolio().greeks()[g] == pytest.approx(-pf_g[g], abs=1e-6)


def test_underdetermined_uses_lstsq_and_neutralizes():
    pf = h.Portfolio([_short_call(), _stock(500)])
    plan = h.hedge(pf, [h.Underlying(S=100.0),
                        h.OptionLeg(flag="c", K=100, **MKT)],
                   greeks=("delta",))
    assert plan.method == "lstsq"
    assert plan.residual["delta"] == pytest.approx(0.0, abs=1e-9)


def test_singular_matrix_raises_when_strict():
    pf = h.Portfolio([_short_call(), _stock(500)])
    dup = h.OptionLeg(flag="c", K=100, name="A", **MKT)
    dup2 = h.OptionLeg(flag="c", K=100, name="B", **MKT)
    with pytest.raises(InvalidInputError):
        h.hedge(pf, [dup, dup2], greeks=("delta", "gamma"), strict=True)
    # Non-strict default degrades gracefully to a least-squares hedge.
    plan = h.hedge(pf, [dup, dup2], greeks=("delta", "gamma"))
    assert plan.method == "lstsq"


def test_unknown_greek_and_empty_inputs_raise():
    pf = h.Portfolio([_short_call()])
    with pytest.raises(InvalidInputError):
        h.hedge(pf, [h.Underlying(S=100.0)], greeks=("bogus",))
    with pytest.raises(InvalidInputError):
        h.hedge(pf, [], greeks=("delta",))


# -- reporting ---------------------------------------------------------------


def test_describe_and_to_dict():
    pf = h.Portfolio([_short_call(), _stock(500)])
    plan = h.hedge(pf, _hedge_basket())
    text = plan.describe()
    assert "Hedging plan" in text
    assert "Required trades" in text
    assert "Summary:" in text
    assert str(plan) == text
    d = plan.to_dict()
    assert set(d) == {"stock", "H1", "H2"}
    assert d["stock"] == pytest.approx(float(plan.quantities[0]))


def test_to_portfolio_round_trips_weights():
    pf = h.Portfolio([_short_call(), _stock(500)])
    plan = h.hedge(pf, _hedge_basket())
    hedge_pf = plan.to_portfolio()
    assert len(hedge_pf) == 3
    for leg, w in zip(hedge_pf.legs, plan.quantities):
        assert leg.quantity == pytest.approx(float(w))


def test_describe_decimals_option():
    pf = h.Portfolio([_short_call(), _stock(500)])
    plan = h.hedge(pf, _hedge_basket())
    text = plan.describe(decimals=2)
    assert "5.32" in text and "5.38" in text
    # rank/condition diagnostics are surfaced
    assert "rank 2/3" in text and "cond" in text
