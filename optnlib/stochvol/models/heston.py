# -*- coding: utf-8 -*-

"""
optnlib.stochvol.models.heston
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Heston (1993) stochastic-volatility model:

``dS/S = (r - q) dt + sqrt(v) dW_S``

``dv = kappa (theta - v) dt + xi sqrt(v) dW_v``

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
from ..schemes import euler_variance, milstein_variance, qe_variance

__all__ = ["Heston"]


class Heston(StochasticVolModel):
    """Heston stochastic-volatility model."""

    name = "Heston"
    default_scheme = "qe"
    schemes = ("qe", "euler", "milstein")
    has_cf = True

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

    @property
    def feller_satisfied(self):
        """Whether ``2 kappa theta >= xi^2`` holds."""
        return 2.0 * self.kappa * self.theta >= self.xi * self.xi

    def characteristic_function(self, u, t, r, q=0.0):
        """Characteristic function of ``log(S_t / S_0)``."""
        u = complex(u)
        t = float(t)
        r = float(r)
        q = float(q)
        iu = 1j * u

        if self.xi == 0.0:
            return np.exp(iu * (r - q) * t - 0.5 * self.v0 * t * (u * u + iu))

        root = np.sqrt((self.rho * self.xi * iu - self.kappa) ** 2
                       + self.xi * self.xi * (iu + u * u))
        denominator = self.kappa - self.rho * self.xi * iu + root
        g = (self.kappa - self.rho * self.xi * iu - root) / denominator
        exp_root = np.exp(-root * t)
        c_term = (self.kappa * self.theta / (self.xi * self.xi)) * (
            (self.kappa - self.rho * self.xi * iu - root) * t
            - 2.0 * np.log((1.0 - g * exp_root) / (1.0 - g))
        )
        d_term = ((self.kappa - self.rho * self.xi * iu - root)
                  / (self.xi * self.xi)) * (
            (1.0 - exp_root) / (1.0 - g * exp_root)
        )
        return np.exp(iu * (r - q) * t + c_term + d_term * self.v0)

    def simulate(self, S, t, r, q=0.0, n_steps=252, n_sims=10_000,
                 scheme=None, seed=None):
        """Simulate Heston spot and variance paths."""
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
            log_increment = ((r - q - 0.5 * v) * dt
                             + np.sqrt(v) * sqrt_dt * z1[:, step])
            spots[:, step + 1] = spots[:, step] * np.exp(log_increment)

        return SimulationResult(spots, variances, times, scheme, self.name)
