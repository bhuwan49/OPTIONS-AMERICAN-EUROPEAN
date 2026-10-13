# -*- coding: utf-8 -*-

"""
optnlib.core
~~~~~~~~~~~~

Low level building blocks shared by every pricing engine: the standard normal
distribution, the Black-Scholes ``d1``/``d2`` terms, discount factors and the
option intrinsic value.
"""

import math

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x):
    """Cumulative standard normal distribution ``N(x)``."""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def norm_pdf(x):
    """Standard normal probability density ``n(x)``."""
    x = float(x)
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def d1(S, K, t, r, sigma, q=0.0):
    """Black-Scholes-Merton ``d1`` term.

    ``d1 = (ln(S/K) + (r - q + sigma**2/2) t) / (sigma sqrt(t))``
    """
    S = float(S)
    K = float(K)
    t = float(t)
    numerator = math.log(S / K) + (float(r) - float(q) + 0.5 * float(sigma) ** 2) * t
    denominator = float(sigma) * math.sqrt(t)
    return numerator / denominator


def d2(S, K, t, r, sigma, q=0.0):
    """Black-Scholes-Merton ``d2`` term: ``d1 - sigma sqrt(t)``."""
    return d1(S, K, t, r, sigma, q) - float(sigma) * math.sqrt(float(t))


def discount_factor(r, t):
    """Continuous discount factor ``exp(-r t)``."""
    return math.exp(-float(r) * float(t))


def intrinsic(flag, S, K):
    """Intrinsic value of an option."""
    if _is_call(flag):
        return max(float(S) - float(K), 0.0)
    return max(float(K) - float(S), 0.0)


def _is_call(flag):
    flag = _normalize_flag(flag)
    return flag == "c"


def _normalize_flag(flag):
    if flag in ("c", "call", "C", "Call"):
        return "c"
    if flag in ("p", "put", "P", "Put"):
        return "p"
    raise ValueError("flag must be 'c' (call) or 'p' (put); got %r" % (flag,))
