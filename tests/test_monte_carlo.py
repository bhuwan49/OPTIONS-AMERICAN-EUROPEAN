# -*- coding: utf-8 -*-

import numpy as np
import pytest

import optnlib
from optnlib import monte_carlo as mc
from optnlib.exceptions import InvalidInputError


def test_public_api_surface():
    assert hasattr(optnlib, "monte_carlo")
    assert hasattr(optnlib, "monte_carlo_price")
    assert optnlib.monte_carlo_price is mc.price


def test_simulate_shape_and_starting_price():
    paths = mc.simulate(100.0, 1.0, 0.05, 0.2, n_steps=16, n_sims=25,
                        seed=7)
    assert paths.shape == (25, 17)
    assert np.all(paths[:, 0] == 100.0)


def test_simulate_is_reproducible():
    kwargs = dict(S=100.0, t=1.0, r=0.05, sigma=0.2, n_steps=12,
                  n_sims=50)
    first = mc.simulate(seed=123, **kwargs)
    second = mc.simulate(seed=123, **kwargs)
    other = mc.simulate(seed=124, **kwargs)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_call_price_converges_to_black_scholes():
    ks = dict(flag="c", S=100, K=100, t=1.0, r=0.05, sigma=0.2)
    simulated = mc.price(n_steps=16, n_sims=200_000, seed=42, **ks)
    analytic = optnlib.black_scholes("c", 100, 100, 1.0, 0.05, 0.2)
    assert simulated == pytest.approx(analytic, abs=0.15)


def test_put_price_converges_to_black_scholes():
    simulated = mc.price("p", 100, 100, 1.0, 0.05, 0.2,
                         n_steps=16, n_sims=200_000, seed=42)
    analytic = optnlib.black_scholes("p", 100, 100, 1.0, 0.05, 0.2)
    assert simulated == pytest.approx(analytic, abs=0.15)


def test_zero_time_returns_intrinsic():
    assert mc.price("p", 80, 100, 0.0, 0.05, 0.2) == 20.0
    assert mc.price("c", 120, 100, 0.0, 0.05, 0.2) == 20.0


def test_zero_volatility_is_deterministic():
    S, K, t, r, q = 100.0, 90.0, 1.0, 0.05, 0.03
    expected = np.exp(-r * t) * max(S * np.exp((r - q) * t) - K, 0.0)
    got = mc.price("c", S, K, t, r, 0.0, q=q, n_steps=8, n_sims=10, seed=1)
    assert got == pytest.approx(expected, rel=1e-12)


def test_dividend_yield_lowers_call():
    no_yield = mc.price("c", 100, 100, 1.0, 0.05, 0.0, q=0.0,
                        n_steps=8, n_sims=10, seed=1)
    with_yield = mc.price("c", 100, 100, 1.0, 0.05, 0.0, q=0.05,
                          n_steps=8, n_sims=10, seed=1)
    assert with_yield < no_yield


@pytest.mark.parametrize("kwargs", [
    {"n_steps": 0},
    {"n_steps": -1},
    {"n_steps": 1.5},
    {"n_sims": 0},
    {"n_sims": -10},
    {"n_sims": 2.5},
])
def test_invalid_path_parameters_raise(kwargs):
    with pytest.raises(InvalidInputError):
        mc.price("c", 100, 100, 1.0, 0.05, 0.2, **kwargs)
    with pytest.raises(InvalidInputError):
        mc.simulate(100, 1.0, 0.05, 0.2, **kwargs)


def _mpl():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    return matplotlib


def test_plot_paths_saves_figure(tmp_path):
    _mpl()
    paths = mc.simulate(100.0, 1.0, 0.05, 0.2, n_steps=20, n_sims=10,
                        seed=3)
    out = tmp_path / "monte_carlo_paths.png"
    fig = mc.plot_paths(paths, K=100.0, t=1.0, save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig.__class__.__name__ == "Figure"


def test_plot_paths_rejects_non_2d():
    _mpl()
    with pytest.raises(InvalidInputError):
        mc.plot_paths(np.array([100.0, 101.0, 102.0]))
