# -*- coding: utf-8 -*-

import math

import pytest

import optnlib
from optnlib import stochvol as sv


def test_heston_fourier_matches_monte_carlo():
    model = sv.Heston(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    fourier = model.price("c", 100.0, 100.0, 1.0, 0.05)
    simulation = model.price_mc(
        "c", 100.0, 100.0, 1.0, 0.05, n_steps=64, n_sims=100_000, seed=42)
    assert abs(fourier - simulation.price) < 4.0 * simulation.std_error + 0.02


def test_heston_put_call_parity():
    model = sv.Heston()
    call = model.price("c", 100.0, 105.0, 1.0, 0.05, q=0.03)
    put = model.price("p", 100.0, 105.0, 1.0, 0.05, q=0.03)
    expected = 100.0 * math.exp(-0.03) - 105.0 * math.exp(-0.05)
    assert call - put == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("flag", ["c", "p"])
def test_heston_reduces_to_black_scholes(flag):
    model = sv.Heston(v0=0.04, kappa=2.0, theta=0.04, xi=0.0, rho=0.0)
    expected = optnlib.black_scholes(flag, 100.0, 100.0, 1.0, 0.05, 0.2)
    assert model.price(flag, 100.0, 100.0, 1.0, 0.05) == pytest.approx(
        expected, abs=1e-9)


@pytest.mark.parametrize("model", [
    sv.HullWhite(v0=0.04, mu_v=0.0, sigma_v=0.0, rho=0.0),
    sv.SteinStein(sigma0=0.2, kappa=2.0, theta=0.2, xi=0.0, rho=0.0),
    sv.Scott(v0=0.04, kappa=2.0, theta=0.04, xi=0.0, rho=0.0),
    sv.ThreeHalves(v0=0.04, kappa=2.0, theta=0.04, xi=0.0, rho=0.0),
])
def test_classical_models_reduce_to_black_scholes(model):
    expected = optnlib.black_scholes("c", 100.0, 100.0, 1.0, 0.05, 0.2)
    simulated = model.price(
        "c", 100.0, 100.0, 1.0, 0.05, n_steps=32, n_sims=100_000, seed=42)
    assert simulated == pytest.approx(expected, abs=0.25)


def test_merton_closed_form_matches_fourier():
    model = sv.Merton(sigma=0.2, lam=0.5, mu_j=-0.1, sigma_j=0.15)
    closed = model.price("c", 100.0, 100.0, 1.0, 0.05, method="closed-form")
    fourier = model.price("c", 100.0, 100.0, 1.0, 0.05)
    assert closed == pytest.approx(fourier, rel=1e-9, abs=1e-9)


def test_merton_matches_monte_carlo():
    model = sv.Merton(sigma=0.2, lam=0.5, mu_j=-0.1, sigma_j=0.15)
    closed = model.price("c", 100.0, 100.0, 1.0, 0.05, method="closed-form")
    simulation = model.price_mc(
        "c", 100.0, 100.0, 1.0, 0.05, n_steps=64, n_sims=100_000, seed=42)
    assert abs(closed - simulation.price) < 4.0 * simulation.std_error + 0.02


def test_merton_without_jumps_is_black_scholes():
    model = sv.Merton(sigma=0.2, lam=0.0)
    expected = optnlib.black_scholes("c", 100.0, 100.0, 1.0, 0.05, 0.2)
    assert model.price("c", 100.0, 100.0, 1.0, 0.05) == pytest.approx(
        expected, abs=1e-9)
    assert model.price("c", 100.0, 100.0, 1.0, 0.05,
                       method="closed-form") == pytest.approx(
        expected, abs=1e-9)


def test_kou_matches_monte_carlo():
    model = sv.Kou(sigma=0.2, lam=0.5, p=0.4, eta1=10.0, eta2=5.0)
    fourier = model.price("c", 100.0, 100.0, 1.0, 0.05)
    simulation = model.price_mc(
        "c", 100.0, 100.0, 1.0, 0.05, n_steps=64, n_sims=100_000, seed=42)
    assert abs(fourier - simulation.price) < 4.0 * simulation.std_error + 0.02


def test_bates_matches_monte_carlo():
    model = sv.Bates(lam=0.5, mu_j=-0.1, sigma_j=0.15)
    fourier = model.price("c", 100.0, 100.0, 1.0, 0.05)
    simulation = model.price_mc(
        "c", 100.0, 100.0, 1.0, 0.05, n_steps=64, n_sims=100_000, seed=42)
    assert abs(fourier - simulation.price) < 4.0 * simulation.std_error + 0.02


def test_bates_without_jumps_matches_heston():
    heston = sv.Heston()
    bates = sv.Bates(lam=0.0)
    assert bates.price("c", 100.0, 100.0, 1.0, 0.05) == pytest.approx(
        heston.price("c", 100.0, 100.0, 1.0, 0.05), abs=1e-9)


def test_sabr_hagan_matches_monte_carlo():
    model = sv.SABR(alpha=0.3, beta=1.0, rho=-0.4, nu=0.5)
    hagan = model.price("c", 100.0, 100.0, 1.0, 0.05)
    simulation = model.price_mc(
        "c", 100.0, 100.0, 1.0, 0.05, n_steps=64, n_sims=100_000, seed=42)
    assert abs(hagan - simulation.price) < 4.0 * simulation.std_error + 0.02


def test_sabr_price_uses_black76_with_hagan_volatility():
    model = sv.SABR(alpha=0.3, beta=1.0, rho=-0.4, nu=0.5)
    forward = 100.0 * math.exp(0.05)
    volatility = model.implied_vol(forward, 100.0, 1.0)
    expected = optnlib.black("c", forward, 100.0, 1.0, 0.05, volatility)
    assert model.price("c", 100.0, 100.0, 1.0, 0.05) == pytest.approx(
        expected, abs=1e-12)


def test_sabr_implied_vol_and_zero_expiry():
    model = sv.SABR(alpha=0.3, beta=1.0, rho=-0.4, nu=0.5)
    assert model.implied_vol(100.0, 100.0, 1.0) > 0.0
    assert model.implied_vol(100.0, 100.0, 0.0) == 0.0


def test_mc_result_confidence_interval():
    result = sv.Heston().price_mc(
        "c", 100.0, 100.0, 1.0, 0.05, n_steps=8, n_sims=1000, seed=1)
    low, high = result.confidence_interval
    assert low < result.price < high
    assert float(result) == result.price


def test_heston_scheme_choices_are_accepted():
    model = sv.Heston()
    for scheme in ("qe", "euler", "milstein"):
        result = model.price_mc(
            "c", 100.0, 100.0, 0.5, 0.05, n_steps=8, n_sims=100,
            scheme=scheme, seed=2)
        assert result.price > 0.0
