# -*- coding: utf-8 -*-

"""
optnlib.european
~~~~~~~~~~~~~~~~

European option pricing: Black-Scholes (no dividend), Black-Scholes-Merton
(continuous dividend yield), Black-76 (on a forward), and unified handling of
continuous or discrete dividends.
"""

import math

from .core import d1, d2, intrinsic, norm_cdf
from .dividends import (
    adjusted_forward,
    adjusted_spot,
    normalize_divs,
    validate_model,
)
from .helpers import validate_option


def black_scholes(flag, S, K, t, r, sigma):
    """Black-Scholes option price (no dividend yield).

    :param flag: ``'c'`` or ``'p'``.
    :param S: underlying spot price.
    :param K: strike price (> 0).
    :param t: time to expiry in years.
    :param r: annualized continuously compounded risk-free rate.
    :param sigma: annualized volatility.
    """
    return black_scholes_merton(flag, S, K, t, r, sigma, q=0.0)


def black_scholes_merton(flag, S, K, t, r, sigma, q=0.0):
    """Black-Scholes-Merton option price with continuous dividend yield ``q``.

    :param q: annualized continuous dividend yield (default 0).
    """
    flag = validate_option(flag, S, K, t, r, sigma, q=q)
    S, K, t, r, sigma, q = map(float, (S, K, t, r, sigma, q))

    if t == 0.0:
        return intrinsic(flag, S, K)
    if sigma == 0.0:
        forward_intrinsic = (
            S * math.exp(-q * t) - K * math.exp(-r * t)
            if flag == "c"
            else K * math.exp(-r * t) - S * math.exp(-q * t)
        )
        return max(forward_intrinsic, 0.0)

    d_1 = d1(S, K, t, r, sigma, q)
    d_2 = d2(S, K, t, r, sigma, q)
    disc_s = S * math.exp(-q * t)
    disc_k = K * math.exp(-r * t)

    if flag == "c":
        return disc_s * norm_cdf(d_1) - disc_k * norm_cdf(d_2)
    return disc_k * norm_cdf(-d_2) - disc_s * norm_cdf(-d_1)


def black(flag, F, K, t, r, sigma):
    """Black-76 option price on a forward ``F``.

    :param F: forward price of the underlying (> 0).
    """
    if flag not in ("c", "p", "call", "put", "C", "P", "Call", "Put"):
        raise ValueError("flag must be 'c' or 'p'; got %r" % (flag,))
    call = flag in ("c", "call", "C", "Call")
    F, K, t, r, sigma = map(float, (F, K, t, r, sigma))
    if K <= 0.0:
        raise ValueError("strike K must be strictly positive")
    if t == 0.0:
        return max(F - K, 0.0) if call else max(K - F, 0.0)
    if sigma == 0.0:
        value = max(F - K, 0.0) if call else max(K - F, 0.0)
        return math.exp(-r * t) * value

    d_1 = (math.log(F / K) + 0.5 * sigma * sigma * t) / (sigma * math.sqrt(t))
    d_2 = d_1 - sigma * math.sqrt(t)
    if call:
        return math.exp(-r * t) * (F * norm_cdf(d_1) - K * norm_cdf(d_2))
    return math.exp(-r * t) * (K * norm_cdf(-d_2) - F * norm_cdf(-d_1))


def price(flag, S, K, t, r, sigma, q=0.0, divs=None, dividend_model="forward"):
    """European option price with continuous and/or discrete dividends.

    :param q: continuous dividend yield (default 0).
    :param divs: discrete schedule ``[(ex_time, amount), ...]`` (default None).
    :param dividend_model: ``'forward'`` (default) or ``'escrowed'``.
    """
    flag = validate_option(flag, S, K, t, r, sigma, q=q, divs=divs)
    if t == 0.0:
        return intrinsic(flag, S, K)

    schedule = normalize_divs(divs, t)
    if not schedule:
        return black_scholes_merton(flag, S, K, t, r, sigma, q)

    model = validate_model(dividend_model)
    if model == "forward":
        forward = adjusted_forward(S, t, r, q, divs)
        return black(flag, forward, K, t, r, sigma)

    spot = adjusted_spot(S, r, divs, t)
    if spot <= 0.0:
        raise ValueError("escrowed spot is non-positive; dividends too large")
    return black_scholes_merton(flag, spot, K, t, r, sigma, q)
