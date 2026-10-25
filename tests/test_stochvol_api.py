# -*- coding: utf-8 -*-

import numpy as np
import pytest

import optnlib
from optnlib import stochvol as sv
from optnlib.exceptions import InvalidInputError


MODELS = [
    sv.Heston(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7),
    sv.HullWhite(v0=0.04, mu_v=0.0, sigma_v=0.3, rho=-0.7),
    sv.SteinStein(sigma0=0.2, kappa=2.0, theta=0.2, xi=0.3, rho=-0.7),
    sv.Scott(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7),
    sv.ThreeHalves(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7),
    sv.SABR(alpha=0.3, beta=1.0, rho=-0.4, nu=0.5),
    sv.Merton(sigma=0.2, lam=0.5, mu_j=-0.1, sigma_j=0.15),
    sv.Kou(sigma=0.2, lam=0.5, p=0.4, eta1=10.0, eta2=5.0),
    sv.Bates(lam=0.5, mu_j=-0.1, sigma_j=0.15),
]


def test_public_api_surface():
    assert hasattr(optnlib, "stochvol")
    assert optnlib.stochvol is sv
    for name in ("Heston", "HullWhite", "SteinStein", "Scott", "ThreeHalves",
                 "SABR", "Merton", "Kou", "Bates"):
        assert hasattr(sv, name)


def test_zero_time_prices_are_intrinsic():
    for model in MODELS:
        assert model.price("p", 80.0, 100.0, 0.0, 0.05) == pytest.approx(20.0)
        assert model.price_mc("c", 120.0, 100.0, 0.0, 0.05).price == 20.0


def test_heston_simulation_shape_and_reproducibility():
    model = sv.Heston()
    first = model.simulate(100.0, 0.5, 0.05, n_steps=12, n_sims=50, seed=7)
    second = model.simulate(100.0, 0.5, 0.05, n_steps=12, n_sims=50, seed=7)
    assert first.spots.shape == (50, 13)
    assert first.variances.shape == (50, 13)
    assert np.array_equal(first.spots, second.spots)
    assert np.array_equal(first.variances, second.variances)
    assert first.spots[0, 0] == 100.0
    assert first.variances[0, 0] == pytest.approx(0.04)
    assert np.all(first.variances >= 0.0)


def test_jump_models_do_not_reports_variance_state():
    merton = sv.Merton()
    result = merton.simulate(100.0, 0.2, 0.05, n_steps=5, n_sims=10, seed=1)
    assert result.spots.shape == (10, 6)
    assert result.variances is None
    with pytest.raises(InvalidInputError):
        result.plot_variance()


def test_variance_models_keep_variance_nonnegative():
    for model in (sv.Heston(), sv.Bates(), sv.ThreeHalves(),
                  sv.HullWhite(), sv.Scott(), sv.SteinStein()):
        result = model.simulate(100.0, 0.5, 0.05, n_steps=20, n_sims=100,
                                seed=11)
        assert np.all(result.variances >= 0.0)


def test_invalid_model_parameters_raise():
    with pytest.raises(InvalidInputError):
        sv.Heston(v0=0.0)
    with pytest.raises(InvalidInputError):
        sv.Heston(xi=-0.1)
    with pytest.raises(InvalidInputError):
        sv.Heston(rho=1.5)
    with pytest.raises(InvalidInputError):
        sv.HullWhite(sigma_v=-0.1)
    with pytest.raises(InvalidInputError):
        sv.SteinStein(sigma0=0.0)
    with pytest.raises(InvalidInputError):
        sv.Scott(theta=0.0)
    with pytest.raises(InvalidInputError):
        sv.ThreeHalves(xi=-0.1)
    with pytest.raises(InvalidInputError):
        sv.SABR(alpha=0.0)
    with pytest.raises(InvalidInputError):
        sv.SABR(beta=1.5)
    with pytest.raises(InvalidInputError):
        sv.SABR(rho=1.0)
    with pytest.raises(InvalidInputError):
        sv.Merton(sigma=-0.1)
    with pytest.raises(InvalidInputError):
        sv.Kou(eta1=1.0)
    with pytest.raises(InvalidInputError):
        sv.Kou(eta2=0.0)
    with pytest.raises(InvalidInputError):
        sv.Bates(sigma_j=-0.1)


def test_invalid_scheme_raises():
    with pytest.raises(InvalidInputError):
        sv.Heston().simulate(100.0, 1.0, 0.05, scheme="not-a-scheme")


def test_invalid_method_raises():
    with pytest.raises(InvalidInputError):
        sv.Heston().price("c", 100.0, 100.0, 1.0, 0.05, method="unknown")


def test_fourier_unsupported_model_raises():
    with pytest.raises(NotImplementedError):
        sv.HullWhite().price("c", 100.0, 100.0, 1.0, 0.05, method="fourier")


def test_feller_flag():
    assert sv.Heston(xi=0.1).feller_satisfied
    assert not sv.Heston(v0=0.01, kappa=0.1, theta=0.01, xi=1.0).feller_satisfied
