# -*- coding: utf-8 -*-

"""
optnlib.stochvol.models.three_halves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 3/2 stochastic-volatility model:

``dS/S = (r - q) dt + sqrt(v) dW_S``

``dv = kappa v (theta - v) dt + xi v^(3/2) dW_v``

``dW_S dW_v = rho dt``
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

__all__ = ["ThreeHalves"]


class ThreeHalves(StochasticVolModel):
    """3/2 stochastic-volatility model."""

    name = "3/2"
    default_scheme = "euler"
    schemes = ("euler", "milstein")

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
        spots[:, 0] = S
        variances[:, 0] = self.v0
        if t == 0.0:
            return SimulationResult(spots, variances, times, scheme, self.name)

        rng = np.random.default_rng(seed)
        z1, z2 = correlated_normals(rng, n_sims, n_steps, self.rho)
        dt = t / n_steps
        sqrt_dt = math.sqrt(dt)

        for step in range(n_steps):
            v = np.maximum(variances[:, step], 0.0)
            dw = sqrt_dt * z2[:, step]
            next_v = (v + self.kappa * v * (self.theta - v) * dt
                      + self.xi * np.power(v, 1.5) * dw)
            if scheme == "milstein":
                next_v = next_v + 0.75 * self.xi * self.xi * v * v * (
                    dw * dw - dt)
            next_v = np.maximum(next_v, 0.0)
            variances[:, step + 1] = next_v
            log_increment = ((r - q - 0.5 * v) * dt
                             + np.sqrt(v) * sqrt_dt * z1[:, step])
            spots[:, step + 1] = spots[:, step] * np.exp(log_increment)

        return SimulationResult(spots, variances, times, scheme, self.name)
