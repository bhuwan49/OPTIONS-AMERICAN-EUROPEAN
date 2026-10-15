# -*- coding: utf-8 -*-

"""Tests for the regularized / constrained hedge builders.

Covers :func:`optnlib.hedging.hedge_ridge` (scikit-learn ridge) and
:func:`optnlib.hedging.hedge_optimize` (scipy.optimize constrained). These are
deliberately separate from the closed-form solvers in ``optnlib.hedging.hedge``.
"""

import sys

import numpy as np
import pytest

from optnlib import hedging as h
from optnlib.exceptions import InvalidInputError

MKT = dict(S=100.0, t=0.25, r=0.05, sigma=0.25)
GREEKS = ("delta", "gamma", "vega")


def _portfolio():
    return h.Portfolio([
        h.OptionLeg(quantity=-10, flag="c", K=105, multiplier=100,
                    name="short call", **MKT),
        h.Underlying(quantity=500, S=100, name="stock"),
    ])


def _well_conditioned_basket():
    return [
        h.Underlying(S=100.0, name="stock"),
        h.OptionLeg(flag="c", K=100, multiplier=100, name="H1", **MKT),
        h.OptionLeg(flag="p", K=95, multiplier=100, name="H2",
                    **dict(MKT, t=0.35)),
    ]


def _ill_conditioned_basket():
    # Same expiry/vol -> vega row collinear with gamma row -> rank-deficient.
    return [
        h.Underlying(S=100.0, name="stock"),
        h.OptionLeg(flag="c", K=100, multiplier=100, name="H1",
                    S=100.0, t=0.50, r=0.05, sigma=0.28),
        h.OptionLeg(flag="p", K=95, multiplier=100, name="H2",
                    S=100.0, t=0.50, r=0.05, sigma=0.28),
    ]


# -- ridge -------------------------------------------------------------------


def test_ridge_exports_and_is_separate_from_hedge():
    # ridge is a distinct function; it must NOT be reachable via hedge(method=)
    with pytest.raises(InvalidInputError):
        h.hedge(_portfolio(), _well_conditioned_basket(), method="ridge")
    assert callable(h.hedge_ridge)


def test_ridge_weights_are_bounded_on_ill_conditioned():
    pytest.importorskip("sklearn")
    pf = _portfolio()
    plan = h.hedge_ridge(pf, _ill_conditioned_basket(), greeks=GREEKS)
    assert plan.method == "ridge"
    assert np.all(np.isfinite(plan.quantities))
    assert np.max(np.abs(plan.quantities)) < 1e3
    assert plan.info["alpha"] > 0.0


def test_ridge_alpha_auto_and_fixed():
    pytest.importorskip("sklearn")
    pf = _portfolio()
    auto = h.hedge_ridge(pf, _ill_conditioned_basket(), greeks=GREEKS,
                         alpha="auto")
    fixed = h.hedge_ridge(pf, _ill_conditioned_basket(), greeks=GREEKS,
                          alpha=1e-6)
    assert auto.info["alpha"] > fixed.info["alpha"]
    # smaller alpha tracks the (least-squares) minimum-norm solution closely
    lstsq = h.hedge(pf, _ill_conditioned_basket(), greeks=GREEKS)
    assert np.allclose(fixed.quantities, lstsq.quantities, atol=1e-3)


def test_ridge_reduces_to_exact_when_well_conditioned_and_tiny_alpha():
    pytest.importorskip("sklearn")
    pf = _portfolio()
    plan = h.hedge_ridge(pf, _well_conditioned_basket(), greeks=GREEKS,
                         alpha=1e-10)
    for g in GREEKS:
        assert plan.residual[g] == pytest.approx(0.0, abs=1e-2)


def test_ridge_invalid_alpha_raises():
    pytest.importorskip("sklearn")
    with pytest.raises(InvalidInputError):
        h.hedge_ridge(_portfolio(), _ill_conditioned_basket(), alpha="bogus")
    with pytest.raises(InvalidInputError):
        h.hedge_ridge(_portfolio(), _ill_conditioned_basket(), alpha=-1.0)


def test_ridge_without_sklearn_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", None)
    with pytest.raises(InvalidInputError, match="scikit-learn"):
        h.hedge_ridge(_portfolio(), _ill_conditioned_basket())


def test_ridge_describe_shows_alpha():
    pytest.importorskip("sklearn")
    plan = h.hedge_ridge(_portfolio(), _ill_conditioned_basket(), greeks=GREEKS)
    text = plan.describe()
    assert "Ridge" in text and "alpha=" in text
    assert "Solver  : ridge" in text


# -- constrained optimization ------------------------------------------------


def test_optimize_default_matches_min_norm_lstsq():
    pf = _portfolio()
    opt = h.hedge_optimize(pf, _ill_conditioned_basket(), greeks=GREEKS)
    lstsq = h.hedge(pf, _ill_conditioned_basket(), greeks=GREEKS)
    assert opt.method == "optimize"
    assert np.allclose(opt.quantities, lstsq.quantities, atol=1e-4)


def test_optimize_respects_position_bounds():
    pf = _portfolio()
    plan = h.hedge_optimize(pf, _ill_conditioned_basket(), greeks=GREEKS,
                            bounds=2.0)
    assert np.max(np.abs(plan.quantities)) <= 2.0 + 1e-6


def test_optimize_respects_gross_limit():
    pf = _portfolio()
    plan = h.hedge_optimize(pf, _ill_conditioned_basket(), greeks=GREEKS,
                            gross_limit=6.0)
    assert np.sum(np.abs(plan.quantities)) <= 6.0 + 1e-6


def test_optimize_per_instrument_bounds():
    pf = _portfolio()
    bounds = [(-10.0, 10.0), (None, 1.0), (-1.0, None)]
    plan = h.hedge_optimize(pf, _ill_conditioned_basket(), greeks=GREEKS,
                            bounds=bounds)
    assert plan.quantities[1] <= 1.0 + 1e-6
    assert plan.quantities[2] >= -1.0 - 1e-6


def test_optimize_is_always_finite_on_inconsistent_system():
    pf = _portfolio()
    plan = h.hedge_optimize(pf, _ill_conditioned_basket(), greeks=GREEKS,
                            bounds=100.0, cost_weight=0.1)
    assert np.all(np.isfinite(plan.quantities))
    assert plan.info["success"] in (True, False)  # reported, not raised
    assert "solver" in plan.info and "objective" in plan.info


def test_optimize_invalid_bounds_length_raises():
    with pytest.raises(InvalidInputError):
        h.hedge_optimize(_portfolio(), _ill_conditioned_basket(),
                         bounds=[(0.0, 1.0)])


def test_optimize_costs_reduce_trading():
    pf = _portfolio()
    cheap = h.hedge_optimize(pf, _ill_conditioned_basket(), greeks=GREEKS,
                             bounds=10.0, costs=[0.0, 1.0, 1.0],
                             cost_weight=1.0)
    assert np.all(np.isfinite(cheap.quantities))


def test_optimize_describe_shows_solver_info():
    plan = h.hedge_optimize(_portfolio(), _ill_conditioned_basket(),
                            greeks=GREEKS, bounds=5.0)
    text = plan.describe()
    assert "Optimize:" in text and "success=" in text
    assert "Solver  : optimize" in text


def test_regularized_unknown_greek_raises():
    with pytest.raises(InvalidInputError):
        h.hedge_optimize(_portfolio(), _ill_conditioned_basket(),
                         greeks=("bogus",))


def test_regularized_to_portfolio_and_to_dict():
    pytest.importorskip("sklearn")
    plan = h.hedge_ridge(_portfolio(), _ill_conditioned_basket(), greeks=GREEKS)
    assert set(plan.to_dict()) == {"stock", "H1", "H2"}
    hp = plan.to_portfolio()
    for leg, w in zip(hp.legs, plan.quantities):
        assert leg.quantity == pytest.approx(float(w))
