# -*- coding: utf-8 -*-

"""
optnlib.stochvol.schemes
~~~~~~~~~~~~~~~~~~~~~~~~

Discretization and exact-step helpers used by the stochastic-volatility models.
The routines are deliberately small and operate on NumPy arrays so that the
model classes can share the same numerical building blocks.
"""

import math

import numpy as np

__all__ = [
    "euler_variance",
    "milstein_variance",
    "qe_variance",
    "ou_exact",
    "lognormal_exact",
    "compound_poisson_jumps",
    "merton_jumps",
    "kou_jumps",
]


def euler_variance(v, kappa, theta, xi, dt, dw):
    """Full-truncation Euler step for a CIR variance process."""
    v = np.maximum(np.asarray(v, dtype=float), 0.0)
    next_v = v + kappa * (theta - v) * dt + xi * np.sqrt(v) * dw
    return np.maximum(next_v, 0.0)


def milstein_variance(v, kappa, theta, xi, dt, dw):
    """Full-truncation Milstein step for a CIR variance process."""
    v = np.maximum(np.asarray(v, dtype=float), 0.0)
    next_v = (v + kappa * (theta - v) * dt + xi * np.sqrt(v) * dw
              + 0.25 * xi * xi * (dw * dw - dt))
    return np.maximum(next_v, 0.0)


def qe_variance(v, kappa, theta, xi, dt, z, rng):
    """Andersen's quadratic-exponential step for a CIR variance process.

    The QE scheme is the preferred Heston variance discretization because it
    preserves non-negativity without the bias of simple truncation.
    """
    v = np.maximum(np.asarray(v, dtype=float), 0.0)
    size = v.shape

    if xi <= 1e-14:
        return np.maximum(theta + (v - theta) * math.exp(-kappa * dt), 0.0)

    if kappa <= 1e-14:
        dw = math.sqrt(dt) * z
        return np.maximum(v + xi * np.sqrt(v) * dw, 0.0)

    exp_k = math.exp(-kappa * dt)
    mean = theta + (v - theta) * exp_k
    variance = (v * xi * xi * exp_k * (1.0 - exp_k) / kappa
                + theta * xi * xi * (1.0 - exp_k) ** 2 / (2.0 * kappa))
    variance = np.maximum(variance, 0.0)
    safe_mean = np.maximum(mean, 1e-14)
    psi = variance / (safe_mean * safe_mean)

    lower = psi <= 1.5
    next_v = np.empty(size, dtype=float)

    if np.any(lower):
        psis = psi[lower]
        b2 = 2.0 / psis - 1.0 + np.sqrt(2.0 / psis) * np.sqrt(
            np.maximum(2.0 / psis - 1.0, 0.0))
        a = mean[lower] / (1.0 + b2)
        b = np.sqrt(b2)
        next_v[lower] = a * (b + z[lower]) ** 2

    upper = ~lower
    if np.any(upper):
        psis = psi[upper]
        p = (psis - 1.0) / (psis + 1.0)
        beta = (1.0 - p) / np.maximum(mean[upper], 1e-14)
        uniform = rng.random(np.count_nonzero(upper))
        values = np.log((1.0 - p) / (1.0 - uniform)) / beta
        next_v[upper] = np.where(uniform <= p, 0.0, values)

    return np.maximum(next_v, 0.0)


def ou_exact(y, kappa, theta, xi, dt, z):
    """Exact Gaussian step for an Ornstein-Uhlenbeck process."""
    if kappa <= 1e-14:
        return y + xi * math.sqrt(dt) * z
    decay = math.exp(-kappa * dt)
    scale = xi * math.sqrt((1.0 - decay * decay) / (2.0 * kappa))
    return theta + (y - theta) * decay + scale * z


def lognormal_exact(v, mu, sigma, dt, z):
    """Exact step for a geometric Brownian motion / lognormal state."""
    v = np.asarray(v, dtype=float)
    result = v * np.exp((mu - 0.5 * sigma * sigma) * dt
                        + sigma * math.sqrt(dt) * z)
    return np.maximum(result, 0.0)


def compound_poisson_jumps(rng, n_sims, lam, dt, sampler):
    """Simulate the sum of compound-Poisson jumps over one time step.

    ``sampler(size)`` returns the jump sizes for ``size`` individual jumps.
    """
    if lam <= 0.0:
        return np.zeros(n_sims, dtype=float)
    counts = rng.poisson(lam * dt, size=n_sims)
    total = int(np.sum(counts))
    if total == 0:
        return np.zeros(n_sims, dtype=float)
    locations = np.repeat(np.arange(n_sims), counts)
    values = sampler(total)
    jumps = np.zeros(n_sims, dtype=float)
    np.add.at(jumps, locations, values)
    return jumps


def merton_jumps(rng, n_sims, lam, dt, mu_j, sigma_j):
    """Normal compound-Poisson jumps used by Merton and Bates models."""
    def sampler(size):
        if sigma_j == 0.0:
            return np.full(size, mu_j, dtype=float)
        return rng.normal(mu_j, sigma_j, size=size)

    return compound_poisson_jumps(rng, n_sims, lam, dt, sampler)


def kou_jumps(rng, n_sims, lam, dt, p, eta1, eta2):
    """Double-exponential compound-Poisson jumps used by the Kou model."""
    def sampler(size):
        signs = rng.random(size) < p
        upward = rng.exponential(1.0 / eta1, size=size)
        downward = -rng.exponential(1.0 / eta2, size=size)
        return np.where(signs, upward, downward)

    return compound_poisson_jumps(rng, n_sims, lam, dt, sampler)
