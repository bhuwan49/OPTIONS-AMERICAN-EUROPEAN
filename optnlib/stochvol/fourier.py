# -*- coding: utf-8 -*-

"""
optnlib.stochvol.fourier
~~~~~~~~~~~~~~~~~~~~~~~~

Fourier pricing for European options from a characteristic function of the
log-return ``log(S_t / S_0)``. The implementation uses two Gil-Pelaez
probabilities, one under the risk-neutral measure and one under the
exponential tilt by the terminal stock price.
"""

import math

import numpy as np
from scipy.integrate import quad

from ..core import intrinsic
from ..helpers import validate_option

__all__ = ["price_from_cf"]


def _probability_above(cf, k, t, r, q, tilt=False):
    phi_minus_i = None
    if tilt:
        phi_minus_i = cf(-1j, t, r, q)
        if abs(phi_minus_i) < 1e-300:
            raise ValueError("characteristic function has a zero at -i")

    def integrand(u):
        numerator = cf(u - 1j, t, r, q) if tilt else cf(u, t, r, q)
        value = np.exp(-1j * u * k) * numerator / (1j * u)
        real = float(np.real(value))
        if tilt:
            real /= float(np.real(phi_minus_i))
        return real

    integral, _error = quad(
        integrand, 1e-10, np.inf, epsabs=1e-10, epsrel=1e-10, limit=300
    )
    return 0.5 + integral / math.pi


def price_from_cf(cf, flag, S, K, t, r, q=0.0):
    """Price a European call or put from a characteristic function.

    ``cf(u, t, r, q)`` must return ``E[exp(i u log(S_t / S_0))]`` under the
    risk-neutral measure.
    """
    flag = validate_option(flag, S, K, t, r, 0.0, q=q)
    S = float(S)
    K = float(K)
    t = float(t)
    r = float(r)
    q = float(q)

    if t == 0.0:
        return intrinsic(flag, S, K)

    k = math.log(K / S)
    probability = _probability_above(cf, k, t, r, q, tilt=False)
    tilted_probability = _probability_above(cf, k, t, r, q, tilt=True)
    probability = min(max(probability, 0.0), 1.0)
    tilted_probability = min(max(tilted_probability, 0.0), 1.0)

    if flag == "c":
        price = (S * math.exp(-q * t) * tilted_probability
                 - K * math.exp(-r * t) * probability)
    else:
        price = (K * math.exp(-r * t) * (1.0 - probability)
                 - S * math.exp(-q * t) * (1.0 - tilted_probability))
    return max(float(price), 0.0)
