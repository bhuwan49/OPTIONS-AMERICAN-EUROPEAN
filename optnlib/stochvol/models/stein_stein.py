# -*- coding: utf-8 -*-

"""
optnlib.stochvol.models.stein_stein
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stein-Stein stochastic-volatility model:

``dS/S = (r - q) dt + |Y| dW_S``

``dY = kappa (theta - Y) dt + xi dW_Y``

``dW_S dW_Y = rho dt``

Here ``Y`` is the signed volatility state and ``theta`` is its long-run level.
"""

import math

import numpy as np

from ..base import (
    SimulationResult,
    StochasticVolModel,
    correlated_normals,
    nonnegative,
    positive,
    validate_correlation,
    validate_market,
    validate_simulation,
)
from ..schemes import ou_exact

__all__ = ["SteinStein"]


class SteinStein(StochasticVolModel):
    """Stein-Stein stochastic-volatility model."""

    name = "Stein-Stein"
    default_scheme = "exact"
    schemes = ("exact", "euler")

    def __init__(self, sigma0=0.2, kappa=2.0, theta=0.2, xi=0.3, rho=-0.7):
        self.sigma0 = positive(sigma0, "sigma0")
        self.kappa = positive(kappa, "kappa")
        self.theta = positive(theta, "theta")
        self.xi = nonnegative(xi, "xi")
        self.rho = validate_correlation(rho)
        self._repr_params = (
            ("sigma0", self.sigma0), ("kappa", self.kappa),
            ("theta", self.theta), ("xi", self.xi), ("rho", self.rho),
        )

    def simulate(self, S, t, r, q=0.0, n_steps=252, n_sims=10_000,
                 scheme=None, seed=None):
        """Simulate spot and signed-volatility paths."""
        scheme = self._scheme(scheme)
        S, t, r, q = validate_market(S, t, r, q)
        n_steps, n_sims = validate_simulation(n_steps, n_sims)
        times = np.linspace(0.0, t, n_steps + 1)
        spots = np.empty((n_sims, n_steps + 1), dtype=float)
        variances = np.empty((n_sims, n_steps + 1), dtype=float)
        state = np.empty((n_sims, n_steps + 1), dtype=float)
        spots[:, 0] = S
        state[:, 0] = self.sigma0
        variances[:, 0] = self.sigma0 * self.sigma0
        if t == 0.0:
            return SimulationResult(spots, variances, times, scheme, self.name)

        rng = np.random.default_rng(seed)
        z1, z2 = correlated_normals(rng, n_sims, n_steps, self.rho)
        dt = t / n_steps
        sqrt_dt = math.sqrt(dt)

        for step in range(n_steps):
            y = state[:, step]
            if scheme == "exact":
                next_y = ou_exact(y, self.kappa, self.theta, self.xi, dt,
                                  z2[:, step])
            else:
                next_y = (y + self.kappa * (self.theta - y) * dt
                          + self.xi * sqrt_dt * z2[:, step])
            state[:, step + 1] = next_y
            variances[:, step + 1] = next_y * next_y
            volatility = np.abs(y)
            log_increment = ((r - q - 0.5 * volatility * volatility) * dt
                             + volatility * sqrt_dt * z1[:, step])
            spots[:, step + 1] = spots[:, step] * np.exp(log_increment)

        return SimulationResult(spots, variances, times, scheme, self.name)
