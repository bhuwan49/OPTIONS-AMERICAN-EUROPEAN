# -*- coding: utf-8 -*-

"""
optnlib.stochvol.models.merton
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Merton (1976) jump-diffusion model:

``dS/S = (r - q - lambda m) dt + sigma dW + (e^J - 1) dN``

where ``J ~ Normal(mu_j, sigma_j^2)`` and ``m = E[e^J] - 1``.
"""

import math

import numpy as np

from ..base import (
    SimulationResult,
    StochasticVolModel,
    finite,
    nonnegative,
    validate_market,
    validate_simulation,
)
from ...core import intrinsic, norm_cdf
from ...helpers import validate_option
from ..schemes import merton_jumps

__all__ = ["Merton"]


class Merton(StochasticVolModel):
    """Merton jump-diffusion model."""

    name = "Merton jump diffusion"
    default_scheme = "exact"
    schemes = ("exact",)
    has_cf = True

    def __init__(self, sigma=0.2, lam=0.5, mu_j=-0.1, sigma_j=0.15):
        self.sigma = nonnegative(sigma, "sigma")
        self.lam = nonnegative(lam, "lam")
        self.mu_j = finite(mu_j, "mu_j")
        self.sigma_j = nonnegative(sigma_j, "sigma_j")
        self._repr_params = (
            ("sigma", self.sigma), ("lam", self.lam),
            ("mu_j", self.mu_j), ("sigma_j", self.sigma_j),
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
        return np.exp(
            iu * (r - q - self.lam * self.jump_compensator) * t
            - 0.5 * self.sigma ** 2 * t * (u * u + iu)
            + self.lam * t * (jump_cf - 1.0)
        )

    def price_closed_form(self, flag, S, K, t, r, q=0.0):
        """Price a European option using the Merton jump series."""
        flag = validate_option(flag, S, K, t, r, self.sigma, q=q)
        S, K, t, r, q = map(float, (S, K, t, r, q))
        if t == 0.0:
            return intrinsic(flag, S, K)

        lam_t = self.lam * t
        compensator = self.jump_compensator
        max_jumps = int(lam_t + 10.0 * math.sqrt(max(lam_t, 0.0)) + 20.0)
        max_jumps = min(max(max_jumps, 10), 100)
        total = 0.0
        for n in range(max_jumps + 1):
            weight = math.exp(-lam_t) * lam_t ** n / math.factorial(n)
            if weight == 0.0:
                continue
            mean_n = ((r - q - self.lam * compensator
                       - 0.5 * self.sigma ** 2) * t + n * self.mu_j)
            variance_n = self.sigma ** 2 * t + n * self.sigma_j ** 2
            if variance_n <= 1e-18:
                terminal = S * math.exp(mean_n)
                if flag == "c":
                    value = max(terminal - K, 0.0)
                else:
                    value = max(K - terminal, 0.0)
                total += weight * math.exp(-r * t) * value
                continue
            root = math.sqrt(variance_n)
            d1 = (math.log(S / K) + mean_n + variance_n) / root
            d2 = d1 - root
            forward_value = S * math.exp(mean_n + 0.5 * variance_n)
            if flag == "c":
                value = (forward_value * norm_cdf(d1) - K * norm_cdf(d2))
            else:
                value = (K * norm_cdf(-d2) - forward_value * norm_cdf(-d1))
            total += weight * math.exp(-r * t) * value
        return total

    def simulate(self, S, t, r, q=0.0, n_steps=252, n_sims=10_000,
                 scheme=None, seed=None):
        """Simulate Merton jump-diffusion paths."""
        scheme = self._scheme(scheme)
        S, t, r, q = validate_market(S, t, r, q)
        n_steps, n_sims = validate_simulation(n_steps, n_sims)
        times = np.linspace(0.0, t, n_steps + 1)
        spots = np.empty((n_sims, n_steps + 1), dtype=float)
        spots[:, 0] = S
        if t == 0.0:
            return SimulationResult(spots, None, times, scheme, self.name)

        rng = np.random.default_rng(seed)
        dt = t / n_steps
        sqrt_dt = math.sqrt(dt)
        compensator = self.jump_compensator
        log_spots = np.full(n_sims, math.log(S), dtype=float)

        for step in range(n_steps):
            z = rng.standard_normal(n_sims)
            jumps = merton_jumps(
                rng, n_sims, self.lam, dt, self.mu_j, self.sigma_j)
            log_spots += ((r - q - self.lam * compensator
                           - 0.5 * self.sigma ** 2) * dt
                          + self.sigma * sqrt_dt * z + jumps)
            spots[:, step + 1] = np.exp(log_spots)

        return SimulationResult(spots, None, times, scheme, self.name)
