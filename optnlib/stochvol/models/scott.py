# -*- coding: utf-8 -*-

"""
optnlib.stochvol.models.scott
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scott's exponential-Ornstein-Uhlenbeck stochastic-volatility model:

``dS/S = (r - q) dt + exp(Y / 2) dW_S``

``dY = kappa (log(theta) - Y) dt + xi dW_Y``

``dW_S dY = rho dt``

``theta`` is the long-run variance and ``v0`` is the initial variance.
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

__all__ = ["Scott"]


class Scott(StochasticVolModel):
    """Scott stochastic-volatility model."""

    name = "Scott"
    default_scheme = "exact"
    schemes = ("exact", "euler")

    def __init__(self, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7):
        self.v0 = positive(v0, "v0")
        self.kappa = positive(kappa, "kappa")
        self.theta = positive(theta, "theta")
        self.xi = nonnegative(xi, "xi")
        self.rho = validate_correlation(rho)
        self._repr_params = (
            ("v0", self.v0), ("kappa", self.kappa), ("theta", self.theta),
            ("xi", self.xi), ("rho", self.rho),
        )

    def simulate(self, S, t, r, q=0.0, n_steps=252, n_sims=10_000,
                 scheme=None, seed=None):
        """Simulate spot and variance paths."""
        scheme = self._scheme(scheme)
        S, t, r, q = validate_market(S, t, r, q)
        n_steps, n_sims = validate_simulation(n_steps, n_sims)
        times = np.linspace(0.0, t, n_steps + 1)
        spots = np.empty((n_sims, n_steps + 1), dtype=float)
        variances = np.empty((n_sims, n_steps + 1), dtype=float)
        state = np.empty((n_sims, n_steps + 1), dtype=float)
        spots[:, 0] = S
        variances[:, 0] = self.v0
        state[:, 0] = math.log(self.v0)
        if t == 0.0:
            return SimulationResult(spots, variances, times, scheme, self.name)

        rng = np.random.default_rng(seed)
        z1, z2 = correlated_normals(rng, n_sims, n_steps, self.rho)
        dt = t / n_steps
        sqrt_dt = math.sqrt(dt)
        log_theta = math.log(self.theta)

        for step in range(n_steps):
            y = state[:, step]
            if scheme == "exact":
                next_y = ou_exact(y, self.kappa, log_theta, self.xi, dt,
                                  z2[:, step])
            else:
                next_y = (y + self.kappa * (log_theta - y) * dt
                          + self.xi * sqrt_dt * z2[:, step])
            state[:, step + 1] = next_y
            v = np.exp(next_y)
            variances[:, step + 1] = v
            log_increment = ((r - q - 0.5 * v) * dt
                             + np.sqrt(v) * sqrt_dt * z1[:, step])
            spots[:, step + 1] = spots[:, step] * np.exp(log_increment)

        return SimulationResult(spots, variances, times, scheme, self.name)
