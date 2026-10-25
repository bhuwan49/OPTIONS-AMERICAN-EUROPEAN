# -*- coding: utf-8 -*-

"""
optnlib.stochvol.models.sabr
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SABR stochastic-volatility model with Hagan's asymptotic European implied
volatility formula:

``dF = alpha F^beta dW_F``

``d alpha = nu alpha dW_alpha``

``dW_F dW_alpha = rho dt``

``price`` uses Hagan's formula. ``price_mc`` simulates the forward and is
intended as a diagnostic rather than as a replacement for the asymptotic
formula.
"""

import math

import numpy as np

from ..base import (
    MCResult,
    SimulationResult,
    StochasticVolModel,
    correlated_normals,
    finite,
    nonnegative,
    positive,
    validate_market,
    validate_simulation,
)
from ...core import intrinsic
from ...european import black
from ...exceptions import InvalidInputError
from ...helpers import validate_option

__all__ = ["SABR"]


class SABR(StochasticVolModel):
    """SABR stochastic-volatility model."""

    name = "SABR"
    default_scheme = "euler"
    schemes = ("euler",)

    def __init__(self, alpha=0.3, beta=1.0, rho=-0.4, nu=0.5):
        self.alpha = positive(alpha, "alpha")
        self.beta = finite(beta, "beta")
        if self.beta < 0.0 or self.beta > 1.0:
            raise InvalidInputError("beta must be between 0 and 1")
        self.rho = finite(rho, "rho")
        if self.rho <= -1.0 or self.rho >= 1.0:
            raise InvalidInputError("SABR rho must be strictly between -1 and 1")
        self.nu = nonnegative(nu, "nu")
        self._repr_params = (
            ("alpha", self.alpha), ("beta", self.beta),
            ("rho", self.rho), ("nu", self.nu),
        )

    def implied_vol(self, F, K, t):
        """Hagan implied volatility for a European option on forward ``F``."""
        F = positive(F, "forward F")
        K = positive(K, "strike K")
        t = float(t)
        if t < 0.0:
            raise InvalidInputError("time to expiry t must be >= 0")
        if t == 0.0:
            return 0.0

        one_minus_beta = 1.0 - self.beta
        log_fk = math.log(F / K)
        fk_power = (F * K) ** (0.5 * one_minus_beta)
        correction = (1.0 + (one_minus_beta ** 2 / 24.0) * log_fk ** 2
                      + (one_minus_beta ** 4 / 1920.0) * log_fk ** 4)
        a_term = self.alpha / (fk_power * correction)

        z = (self.nu / self.alpha) * fk_power * log_fk
        if abs(z) < 1e-10:
            z_over_x = 1.0
        else:
            root = math.sqrt(max(1.0 - 2.0 * self.rho * z + z * z, 0.0))
            x = math.log((root + z - self.rho) / (1.0 - self.rho))
            z_over_x = z / x if abs(x) > 1e-14 else 1.0

        b_term = 1.0 + (
            (one_minus_beta ** 2 / 24.0) * self.alpha ** 2
            / (F * K) ** one_minus_beta
            + 0.25 * self.rho * self.beta * self.nu * self.alpha / fk_power
            + (2.0 - 3.0 * self.rho ** 2) / 24.0 * self.nu ** 2
        ) * t
        return max(a_term * z_over_x * b_term, 0.0)

    def price(self, flag, S, K, t, r, q=0.0, method="hagan",
              n_steps=252, n_sims=10_000, scheme=None, seed=None):
        """Price a European option with Hagan's formula or Monte Carlo."""
        flag = validate_option(flag, S, K, t, r, 0.0, q=q)
        S, K, t, r, q = map(float, (S, K, t, r, q))
        if t == 0.0:
            return intrinsic(flag, S, K)
        if str(method).lower() in ("mc", "monte-carlo", "simulation"):
            return self.price_mc(
                flag, S, K, t, r, q=q, n_steps=n_steps, n_sims=n_sims,
                scheme=scheme, seed=seed,
            ).price
        forward = S * math.exp((r - q) * t)
        volatility = self.implied_vol(forward, K, t)
        return black(flag, forward, K, t, r, volatility)

    def simulate(self, S, t, r=0.0, q=0.0, n_steps=252, n_sims=10_000,
                 scheme=None, seed=None):
        """Simulate the SABR forward and volatility paths.

        ``S`` is interpreted as the initial forward for this model.
        """
        scheme = self._scheme(scheme)
        S, t, _r, _q = validate_market(S, t, r, q)
        n_steps, n_sims = validate_simulation(n_steps, n_sims)
        times = np.linspace(0.0, t, n_steps + 1)
        forwards = np.empty((n_sims, n_steps + 1), dtype=float)
        variances = np.empty((n_sims, n_steps + 1), dtype=float)
        forward = np.full(n_sims, S, dtype=float)
        alpha = np.full(n_sims, self.alpha, dtype=float)
        forwards[:, 0] = forward
        variances[:, 0] = alpha * alpha
        if t == 0.0:
            return SimulationResult(forwards, variances, times, scheme, self.name)

        rng = np.random.default_rng(seed)
        z1, z2 = correlated_normals(rng, n_sims, n_steps, self.rho)
        dt = t / n_steps
        sqrt_dt = math.sqrt(dt)

        for step in range(n_steps):
            if self.beta == 1.0:
                next_forward = forward * np.exp(
                    -0.5 * alpha * alpha * dt + alpha * sqrt_dt * z1[:, step])
            else:
                next_forward = np.maximum(
                    forward + alpha * np.power(forward, self.beta) * sqrt_dt
                    * z1[:, step],
                    0.0,
                )
            next_alpha = alpha * np.exp(
                -0.5 * self.nu * self.nu * dt + self.nu * sqrt_dt * z2[:, step])
            forward = next_forward
            alpha = next_alpha
            forwards[:, step + 1] = forward
            variances[:, step + 1] = alpha * alpha

        return SimulationResult(forwards, variances, times, scheme, self.name)

    def price_mc(self, flag, S, K, t, r, q=0.0, n_steps=252, n_sims=10_000,
                 scheme=None, seed=None):
        """Price a European option by simulating SABR forward paths."""
        flag = validate_option(flag, S, K, t, r, 0.0, q=q)
        S, K, t, r, q = map(float, (S, K, t, r, q))
        if t == 0.0:
            return MCResult(intrinsic(flag, S, K), 0.0, n_sims, 0, "exact")
        forward = S * math.exp((r - q) * t)
        simulation = self.simulate(
            forward, t, 0.0, 0.0, n_steps=n_steps, n_sims=n_sims,
            scheme=scheme, seed=seed,
        )
        terminal = simulation.spots[:, -1]
        if flag == "c":
            payoff = np.maximum(terminal - K, 0.0)
        else:
            payoff = np.maximum(K - terminal, 0.0)
        discounted = math.exp(-r * t) * payoff
        price = float(np.mean(discounted))
        std_error = (float(np.std(discounted, ddof=1) / math.sqrt(n_sims))
                     if n_sims > 1 else 0.0)
        return MCResult(price, std_error, n_sims, n_steps, simulation.scheme)
