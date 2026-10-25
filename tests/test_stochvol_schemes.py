# -*- coding: utf-8 -*-

import numpy as np

from optnlib.stochvol import schemes


def test_euler_and_milstein_variance_stay_nonnegative():
    rng = np.random.default_rng(1)
    v = np.array([0.0, 0.001, 0.04, 0.25])
    dw = 2.0 * np.sqrt(0.01) * rng.standard_normal(v.size)
    for step in (schemes.euler_variance, schemes.milstein_variance):
        result = step(v, 2.0, 0.04, 0.5, 0.01, dw)
        assert np.all(result >= 0.0)


def test_qe_variance_stays_nonnegative():
    rng = np.random.default_rng(2)
    v = np.array([0.0, 0.001, 0.04, 0.25])
    z = 3.0 * rng.standard_normal(v.size)
    result = schemes.qe_variance(v, 2.0, 0.04, 0.5, 0.01, z, rng)
    assert np.all(result >= 0.0)


def test_qe_variance_with_no_vol_of_vol_is_deterministic():
    rng = np.random.default_rng(3)
    v = np.array([0.04, 0.09])
    z = rng.standard_normal(v.size)
    result = schemes.qe_variance(v, 2.0, 0.04, 0.0, 0.1, z, rng)
    expected = 0.04 + (v - 0.04) * np.exp(-2.0 * 0.1)
    assert np.allclose(result, expected)


def test_lognormal_exact_stays_positive():
    rng = np.random.default_rng(4)
    z = rng.standard_normal(100)
    result = schemes.lognormal_exact(np.full(100, 0.04), 0.1, 0.3, 0.05, z)
    assert np.all(result > 0.0)


def test_compound_poisson_zero_intensity_returns_zero():
    rng = np.random.default_rng(5)
    jumps = schemes.compound_poisson_jumps(
        rng, 10, 0.0, 0.1, lambda size: np.ones(size))
    assert np.all(jumps == 0.0)


def test_merton_and_kou_jump_samplers_have_expected_shape():
    rng = np.random.default_rng(6)
    merton = schemes.merton_jumps(rng, 100, 5.0, 0.1, 0.0, 0.2)
    kou = schemes.kou_jumps(rng, 100, 5.0, 0.1, 0.5, 10.0, 5.0)
    assert merton.shape == (100,)
    assert kou.shape == (100,)
    assert np.all(np.isfinite(merton))
    assert np.all(np.isfinite(kou))
