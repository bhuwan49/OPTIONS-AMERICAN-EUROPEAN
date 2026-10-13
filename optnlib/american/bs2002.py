# -*- coding: utf-8 -*-

"""
optnlib.american.bs2002
~~~~~~~~~~~~~~~~~~~~~~~

Bjerksund-Stensland (2002) closed-form approximation for American options on
an underlying with a continuous dividend yield.

Reference
---------
Bjerksund, P. and Stensland, G., "Closed Form Valuation of American Options",
2002. https://core.ac.uk/download/pdf/30824897.pdf

The implementation follows the published algorithm, including the modified
(split) early exercise boundary and the bivariate normal terms that arise from
the two-period structure.
"""

import math

from ..core import norm_cdf
from ..european import black_scholes_merton

_GOLDEN = 0.5 * (math.sqrt(5.0) - 1.0)

try:  # scipy provides the Gauss-Legendre nodes used by the bivariate normal CDF
    from numpy.polynomial.legendre import leggauss as _leggauss

    _GL_NODES, _GL_WEIGHTS = _leggauss(20)
except Exception:  # pragma: no cover - numpy is a hard dependency
    _GL_NODES = _GL_WEIGHTS = None


def _phi_1993(s, t, gamma, h, x, r, b, sigma):
    """The Phi function of the Bjerksund-Stensland approximation."""
    kappa = 2.0 * b / sigma ** 2 + (2.0 * gamma - 1.0)
    d1 = -(math.log(s / h) + (b + (gamma - 0.5) * sigma ** 2) * t) / (
        sigma * math.sqrt(t)
    )
    d2 = -(math.log(x * x / (s * h)) + (b + (gamma - 0.5) * sigma ** 2) * t) / (
        sigma * math.sqrt(t)
    )
    lambd = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sigma ** 2
    return (
        math.exp(lambd * t)
        * s ** gamma
        * (norm_cdf(d1) - (x / s) ** kappa * norm_cdf(d2))
    )


def _cbnd(dh, dk, rho):
    """Bivariate normal term ``P(X > dh, Y > dk)`` via Gauss-Legendre quadrature.

    Ported from the Genz algorithm used by tf-quant-finance.
    """
    h = -dh
    k = -dk
    hk = h * k
    hs = (h * h + k * k) / 2.0
    asr = math.asin(rho)

    total = 0.0
    for node, weight in zip(_GL_NODES, _GL_WEIGHTS):
        sn = math.sin(asr * (node + 1.0) / 2.0)
        total += weight * math.exp((hk * sn - hs) / (1.0 - sn * sn))
    bvn = total * asr / (4.0 * math.pi) + norm_cdf(-h) * norm_cdf(-k)
    return float(bvn)


def _psi_2002(s, t, gamma, h, x2, x1, t1, r, b, sigma):
    """The Psi function of the two-period Bjerksund-Stensland (2002) model."""
    rt1 = sigma * math.sqrt(t1)
    rt = sigma * math.sqrt(t)
    d1 = -(math.log(s / x1) + (b + (gamma - 0.5) * sigma ** 2) * t1) / rt1
    d2 = -(math.log(x2 ** 2 / (s * x1)) + (b + (gamma - 0.5) * sigma ** 2) * t1) / rt1
    d3 = -(math.log(s / x1) - (b + (gamma - 0.5) * sigma ** 2) * t1) / rt1
    d4 = -(math.log(x2 ** 2 / (s * x1)) - (b + (gamma - 0.5) * sigma ** 2) * t1) / rt1
    f1 = -(math.log(s / h) + (b + (gamma - 0.5) * sigma ** 2) * t) / rt
    f2 = -(math.log(x2 ** 2 / (s * h)) + (b + (gamma - 0.5) * sigma ** 2) * t) / rt
    f3 = -(math.log(x1 ** 2 / (s * h)) + (b + (gamma - 0.5) * sigma ** 2) * t) / rt
    f4 = -(
        math.log((s * x1 ** 2) / (h * x2 ** 2))
        + (b + (gamma - 0.5) * sigma ** 2) * t
    ) / rt
    kappa = 2.0 * b / sigma ** 2 + (2.0 * gamma - 1.0)
    lambd = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sigma ** 2
    rho = math.sqrt(t1 / t)
    return (
        math.exp(lambd * t)
        * s ** gamma
        * (
            _cbnd(d1, f1, rho)
            - (x2 / s) ** kappa * _cbnd(d2, f2, rho)
            - (x1 / s) ** kappa * _cbnd(d3, f3, -rho)
            + (x1 / x2) ** kappa * _cbnd(d4, f4, -rho)
        )
    )


def _boundaries(k, t, r, b, sigma, beta):
    t1 = _GOLDEN * t
    if r - b == 0.0:
        b0 = k
    else:
        b0 = max(k, (r / (r - b)) * k)
    b_inf = beta / (beta - 1.0) * k
    scale = k ** 2 / ((b_inf - b0) * b0)
    ht1 = -(b * (t - t1) + 2.0 * sigma * math.sqrt(t - t1)) * scale
    ht2 = -(b * t + 2.0 * sigma * math.sqrt(t)) * scale
    x1 = b0 + (b_inf - b0) * (1.0 - math.exp(ht1))
    x2 = b0 + (b_inf - b0) * (1.0 - math.exp(ht2))
    return t1, x1, x2


def _beta(r, b, sigma):
    return 0.5 - b / sigma ** 2 + math.sqrt(
        (b / sigma ** 2 - 0.5) ** 2 + 2.0 * r / sigma ** 2
    )


def _call_2002(s, k, t, r, b, sigma):
    """Bjerksund-Stensland (2002) value of an American call."""
    beta = _beta(r, b, sigma)
    t1, x1, x2 = _boundaries(k, t, r, b, sigma, beta)
    alpha1 = (x1 - k) * x1 ** (-beta)
    alpha2 = (x2 - k) * x2 ** (-beta)

    if s >= x2:
        return s - k

    return (
        alpha2 * s ** beta
        - alpha2 * _phi_1993(s, t1, beta, x2, x2, r, b, sigma)
        + _phi_1993(s, t1, 1.0, x2, x2, r, b, sigma)
        - _phi_1993(s, t1, 1.0, x1, x2, r, b, sigma)
        - k * _phi_1993(s, t1, 0.0, x2, x2, r, b, sigma)
        + k * _phi_1993(s, t1, 0.0, x1, x2, r, b, sigma)
        + alpha1 * _phi_1993(s, t1, beta, x1, x2, r, b, sigma)
        - alpha1 * _psi_2002(s, t, beta, x1, x2, x1, t1, r, b, sigma)
        + _psi_2002(s, t, 1.0, x1, x2, x1, t1, r, b, sigma)
        - _psi_2002(s, t, 1.0, k, x2, x1, t1, r, b, sigma)
        - k * _psi_2002(s, t, 0.0, x1, x2, x1, t1, r, b, sigma)
        + k * _psi_2002(s, t, 0.0, k, x2, x1, t1, r, b, sigma)
    )


def price(flag, S, K, t, r, sigma, q=0.0):
    """Bjerksund-Stensland (2002) American option price.

    :param flag: ``'c'`` or ``'p'``.
    :param S: underlying spot price.
    :param K: strike price.
    :param t: time to expiry in years.
    :param r: annualized continuously compounded risk-free rate.
    :param sigma: annualized volatility.
    :param q: annualized continuous dividend yield (default 0).
    """
    is_call = flag in ("c", "call", "C", "Call")
    S, K, t, r, sigma, q = map(float, (S, K, t, r, sigma, q))

    if t <= 0.0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)

    # Deterministic limit: the approximation's boundary terms overflow as the
    # volatility collapses, so price the no-volatility value directly. The
    # error introduced is negligible for sigma below this threshold.
    if sigma <= 1e-4:
        euro = black_scholes_merton("c" if is_call else "p", S, K, t, r, 0.0, q)
        intrinsic = S - K if is_call else K - S
        return max(euro, max(intrinsic, 0.0))

    b = r - q

    if is_call:
        if b >= r:
            # Early exercise is never optimal: fall back to the European value.
            return black_scholes_merton("c", S, K, t, r, sigma, q)
        if _beta(r, b, sigma) <= 1.0:
            # Infinite exercise boundary: early exercise is not optimal.
            return black_scholes_merton("c", S, K, t, r, sigma, q)
        return _call_2002(S, K, t, r, b, sigma)

    # American put via the Bjerksund-Stensland put-call transformation.
    if _beta(r - b, -b, sigma) <= 1.0:
        return black_scholes_merton("p", S, K, t, r, sigma, q)
    return _call_2002(K, S, t, r - b, -b, sigma)
