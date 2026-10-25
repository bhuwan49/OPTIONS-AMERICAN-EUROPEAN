# -*- coding: utf-8 -*-

"""
optnlib.stochvol.models.kou
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kou (2002) double-exponential jump-diffusion model:

``dS/S = (r - q - lambda m) dt + sigma dW + (e^J - 1) dN``

where ``J`` is double exponential: upward jumps occur with probability ``p``
and mean ``1/eta1``; downward jumps occur with probability ``1-p`` and mean
``1/eta2``.
"""

import math

import numpy as np

from ..base import (
    SimulationResult,
    StochasticVolModel,
    finite,
    nonnegative,
    probability,
    validate_market,
    validate_simulation,
)
from ..schemes import kou_jumps
from ...exceptions import InvalidInputError

__all__ = ["Kou"]


class Kou(StochasticVolModel):
    """Kou double-exponential jump-diffusion model."""

    name = "Kou jump diffusion"
    default_scheme = "exact"
    schemes = ("exact",)
    has_cf = True

    def __init__(self, sigma=0.2, lam=0.5, p=0.4, eta1=10.0, eta2=5.0):
        self.sigma = nonnegative(sigma, "sigma")
        self.lam = nonnegative(lam, "lam")
        self.p = probability(p, "p")
        self.eta1 = finite(eta1, "eta1")
        self.eta2 = finite(eta2, "eta2")
        if self.eta1 <= 1.0:
            raise InvalidInputError("eta1 must be strictly greater than 1")
        if self.eta2 <= 0.0:
            raise InvalidInputError("eta2 must be strictly positive")
        self._repr_params = (
            ("sigma", self.sigma), ("lam", self.lam), ("p", self.p),
            ("eta1", self.eta1), ("eta2", self.eta2),
        )

    @property
    def jump_compensator(self):
        """``m = E[e^J] - 1``."""
        return (self.p * self.eta1 / (self.eta1 - 1.0)
                + (1.0 - self.p) * self.eta2 / (self.eta2 + 1.0) - 1.0)

    def characteristic_function(self, u, t, r, q=0.0):
        """Characteristic function of ``log(S_t / S_0)``."""
        u = complex(u)
        iu = 1j * u
        jump_cf = (self.p * self.eta1 / (self.eta1 - iu)
                   + (1.0 - self.p) * self.eta2 / (self.eta2 + iu))
        return np.exp(
            iu * (r - q - self.lam * self.jump_compensator) * t
            - 0.5 * self.sigma ** 2 * t * (u * u + iu)
            + self.lam * t * (jump_cf - 1.0)
        )

    def simulate(self, S, t, r, q=0.0, n_steps=252, n_sims=10_000,
                 scheme=None, seed=None):
        """Simulate Kou jump-diffusion paths."""
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
            jumps = kou_jumps(
                rng, n_sims, self.lam, dt, self.p, self.eta1, self.eta2)
            log_spots += ((r - q - self.lam * compensator
                           - 0.5 * self.sigma ** 2) * dt
                          + self.sigma * sqrt_dt * z + jumps)
            spots[:, step + 1] = np.exp(log_spots)

        return SimulationResult(spots, None, times, scheme, self.name)
