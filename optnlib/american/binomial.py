# -*- coding: utf-8 -*-

"""
optnlib.american.binomial
~~~~~~~~~~~~~~~~~~~~~~~~~

Binomial tree American (and optionally European) option pricing using the
Cox-Ross-Rubinstein (CRR) and Leisen-Reimer (LR) lattices.

Continuous dividend yields enter through the cost of carry ``b = r - q``.
Discrete cash dividends are handled with the "drop" method: the dividend amount
is removed from every node's stock price at the ex-dividend step.
"""

import math

import numpy as np

from ..core import norm_cdf
from ..dividends import normalize_divs


def _sign(value):
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def _odd(n):
    n = abs(int(n))
    if n % 2 == 0:
        n += 1
    return n


def _cumulative_dividends(divs_norm, n_steps, dt):
    """Cumulative cash dividends paid up to each time step (for the drop model)."""
    cumulative = np.zeros(n_steps + 1)
    if not divs_norm:
        return cumulative
    for step in range(1, n_steps + 1):
        t_step = step * dt
        cumulative[step] = sum(
            amount for ex_time, amount in divs_norm if ex_time <= t_step
        )
    return cumulative


def _backward_induction(flag, S, K, t, r, p, u, d, df, n_steps, american,
                        cumulative):
    z = 1.0 if flag == "c" else -1.0

    spot = S * d ** n_steps * (u / d) ** np.arange(n_steps + 1)
    term = spot - cumulative[n_steps]
    values = np.maximum(z * (term - K), 0.0)

    for step in range(n_steps - 1, -1, -1):
        values = df * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            spot = S * d ** step * (u / d) ** np.arange(step + 1)
            spot = spot - cumulative[step]
            exercise = np.maximum(z * (spot - K), 0.0)
            values = np.maximum(values, exercise)
    return float(values[0])


def crr_price(flag, S, K, t, r, sigma, q=0.0, divs=None, n_steps=500,
              american=True):
    """Cox-Ross-Rubinstein binomial tree price."""
    divs_norm = normalize_divs(divs, t) if divs else []
    dt = t / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    df = math.exp(-r * dt)
    cumulative = _cumulative_dividends(divs_norm, n_steps, dt)
    return _backward_induction(flag, S, K, t, r, p, u, d, df, n_steps, american,
                               cumulative)


def lr_price(flag, S, K, t, r, sigma, q=0.0, divs=None, n_steps=101,
             american=True):
    """Leisen-Reimer binomial tree price (uses the Peizer-Pratt inversion)."""
    n_steps = _odd(n_steps)
    divs_norm = normalize_divs(divs, t) if divs else []

    d1 = (math.log(S / K) + ((r - q) + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    scale = n_steps + 1.0 / 3.0 + 0.1 / (n_steps + 1.0)
    width = n_steps + 1.0 / 6.0
    hd1 = 0.5 + _sign(d1) * math.sqrt(
        0.25 - 0.25 * math.exp(-(d1 / scale) ** 2 * width)
    )
    hd2 = 0.5 + _sign(d2) * math.sqrt(
        0.25 - 0.25 * math.exp(-(d2 / scale) ** 2 * width)
    )

    dt = t / n_steps
    p = hd2
    b = r - q
    u = math.exp(b * dt) * hd1 / hd2
    d = (math.exp(b * dt) - p * u) / (1.0 - p)
    df = math.exp(-r * dt)
    cumulative = _cumulative_dividends(divs_norm, n_steps, dt)
    return _backward_induction(flag, S, K, t, r, p, u, d, df, n_steps, american,
                               cumulative)


def price(flag, S, K, t, r, sigma, q=0.0, divs=None, method="lr",
          n_steps=None, american=True):
    """Binomial tree price dispatching on ``method`` (``'crr'`` or ``'lr'``)."""
    if method == "crr":
        return crr_price(
            flag, S, K, t, r, sigma, q=q, divs=divs,
            n_steps=500 if n_steps is None else n_steps, american=american,
        )
    if method == "lr":
        return lr_price(
            flag, S, K, t, r, sigma, q=q, divs=divs,
            n_steps=101 if n_steps is None else n_steps, american=american,
        )
    raise ValueError("method must be 'crr' or 'lr'; got %r" % (method,))
