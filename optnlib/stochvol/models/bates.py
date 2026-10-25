# -*- coding: utf-8 -*-

"""
optnlib.stochvol.models.bates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bates (1996) model: Heston stochastic volatility plus normally distributed
log jumps in the underlying.
"""

import math

import numpy as np

from .heston import Heston
from ..base import (
    SimulationResult,
    correlated_normals,
    finite,
    nonnegative,
    validate_market,
    validate_simulation,
)
from ..schemes import euler_variance, merton_jumps, milstein_variance, qe_variance

__all__ = ["Bates"]


class Bates(Heston):
    """Bates stochastic-volatility jump-diffusion model."""

    name = "Bates"
    default_scheme = "qe"
    schemes = ("qe", "euler", "milstein")
    has_cf = True

    def __init__(self, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                 lam=0.5, mu_j=-0.1, sigma_j=0.15):
        super().__init__(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
        self.lam = nonnegative(lam, "lam")
        self.mu_j = finite(mu_j, "mu_j")
        self.sigma_j = nonnegative(sigma_j, "sigma_j")
        self._repr_params = self._repr_params + (
            ("lam", self.lam), ("mu_j", self.mu_j),
            ("sigma_j", self.sigma_j),
        )

    @property
    def jump_compensator(self):
        """``m = E[e^J] - 1``."""
        return math.exp(self.mu_j + 0.5 * self.sigma_j * self.sigma_j) - 1.0

    def characteristic_function(self, u, t, r, q=0.0):
        """Characteristic function of ``log(S_t / S_0)``."""
        u = complex(u)
        iu = 1j * u
        jump_cf = np.exp(iu * self.mu_j - 0.5 * self.sigma_j ** 2 * u * u)
        jump_factor = np.exp(
            self.lam * t * (jump_cf - 1.0 - iu * self.jump_compensator))
        return super().characteristic_function(u, t, r, q=q) * jump_factor

    def simulate(self, S, t, r, q=0.0, n_steps=252, n_sims=10_000,
                 scheme=None, seed=None):
        """Simulate Bates spot and variance paths."""
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
        compensator = self.jump_compensator

        for step in range(n_steps):
            v = variances[:, step]
            if scheme == "qe":
                next_v = qe_variance(
                    v, self.kappa, self.theta, self.xi, dt, z2[:, step], rng)
            elif scheme == "milstein":
                next_v = milstein_variance(
                    v, self.kappa, self.theta, self.xi, dt,
                    sqrt_dt * z2[:, step])
            else:
                next_v = euler_variance(
                    v, self.kappa, self.theta, self.xi, dt,
                    sqrt_dt * z2[:, step])
            variances[:, step + 1] = next_v
            jumps = merton_jumps(
                rng, n_sims, self.lam, dt, self.mu_j, self.sigma_j)
            log_increment = ((r - q - 0.5 * v - self.lam * compensator) * dt
                             + np.sqrt(v) * sqrt_dt * z1[:, step] + jumps)
            spots[:, step + 1] = spots[:, step] * np.exp(log_increment)

        return SimulationResult(spots, variances, times, scheme, self.name)
