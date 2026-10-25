# -*- coding: utf-8 -*-

"""
optnlib.stochvol.models.hull_white
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hull-White (1987) stochastic-volatility model with geometric variance:

``dS/S = (r - q) dt + sqrt(v) dW_S``

``dv = mu_v v dt + sigma_v v dW_v``

``dW_S dW_v = rho dt``
"""

import math

import numpy as np

from ..base import (
    SimulationResult,
    StochasticVolModel,
    correlated_normals,
    finite,
    nonnegative,
    positive,
    validate_correlation,
    validate_market,
    validate_simulation,
)
from ..schemes import lognormal_exact

__all__ = ["HullWhite"]


class HullWhite(StochasticVolModel):
    """Hull-White (1987) geometric-variance stochastic-volatility model."""

    name = "Hull-White"
    default_scheme = "lognormal"
    schemes = ("lognormal", "euler")

    def __init__(self, v0=0.04, mu_v=0.0, sigma_v=0.3, rho=-0.7):
        self.v0 = positive(v0, "v0")
        self.mu_v = finite(mu_v, "mu_v")
        self.sigma_v = nonnegative(sigma_v, "sigma_v")
        self.rho = validate_correlation(rho)
        self._repr_params = (
            ("v0", self.v0), ("mu_v", self.mu_v),
            ("sigma_v", self.sigma_v), ("rho", self.rho),
        )

    def simulate(self, S, t, r, q=0.0, n_steps=252, n_sims=10_000,
                 scheme=None, seed=None):
        """Simulate spot and geometric-variance paths."""
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
            v = variances[:, step]
            if scheme == "lognormal":
                next_v = lognormal_exact(
                    v, self.mu_v, self.sigma_v, dt, z2[:, step])
            else:
                next_v = np.maximum(
                    v + self.mu_v * v * dt
                    + self.sigma_v * v * sqrt_dt * z2[:, step],
                    0.0,
                )
            variances[:, step + 1] = next_v
            log_increment = ((r - q - 0.5 * v) * dt
                             + np.sqrt(v) * sqrt_dt * z1[:, step])
            spots[:, step + 1] = spots[:, step] * np.exp(log_increment)

        return SimulationResult(spots, variances, times, scheme, self.name)
