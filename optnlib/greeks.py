# -*- coding: utf-8 -*-

"""
optnlib.greeks
~~~~~~~~~~~~~~

Option greeks.

European options without discrete dividends use the analytical
Black-Scholes-Merton formulas. Everything else (American options, discrete
dividends) uses central finite differences ("bump and revalue").

Scaling follows common trading convention (and ``py_vollib``):

* ``theta`` is reported per calendar day (the annual derivative divided by 365),
* ``vega``  is reported per 1% change in volatility (the raw derivative x 0.01),
* ``rho``   is reported per 1% change in the interest rate (raw derivative x 0.01),
* ``delta`` and ``gamma`` are reported raw.
"""

import math

from .core import d1, d2, norm_cdf, norm_pdf
from .helpers import validate_option

_BUMP_S = 1e-4
_BUMP_T = 1e-4
_BUMP_SIGMA = 1e-4
_BUMP_R = 1e-4


def _european_price(flag, S, K, t, r, sigma, q, divs, dividend_model):
    from .european import price

    return price(flag, S, K, t, r, sigma, q=q, divs=divs,
                 dividend_model=dividend_model)


def _american_price(flag, S, K, t, r, sigma, q, divs, method, dividend_model,
                    n_steps):
    from .american import price

    return price(flag, S, K, t, r, sigma, q=q, divs=divs, method=method,
                 dividend_model=dividend_model, n_steps=n_steps)


def _price(flag, S, K, t, r, sigma, q, divs, style, method, dividend_model,
           n_steps):
    if style == "american":
        return _american_price(flag, S, K, t, r, sigma, q, divs, method,
                               dividend_model, n_steps)
    return _european_price(flag, S, K, t, r, sigma, q, divs, dividend_model)


def _is_call(flag):
    return flag in ("c", "call", "C", "Call")


# ---------------------------------------------------------------------------
# Analytical greeks (European, continuous yield only)


def analytical_greeks(flag, S, K, t, r, sigma, q=0.0):
    """Analytical Black-Scholes-Merton greeks for a European option."""
    flag = validate_option(flag, S, K, t, r, sigma, q=q)
    S, K, t, r, sigma, q = map(float, (S, K, t, r, sigma, q))

    if t == 0.0 or sigma == 0.0:
        call = _is_call(flag)
        delta = (1.0 if S > K else 0.0) if call else (-1.0 if S < K else 0.0)
        return {"delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0,
                "rho": 0.0}

    d_1 = d1(S, K, t, r, sigma, q)
    d_2 = d2(S, K, t, r, sigma, q)
    disc_q = math.exp(-q * t)
    disc_r = math.exp(-r * t)
    pdf_d1 = norm_pdf(d_1)

    gamma = disc_q * pdf_d1 / (S * sigma * math.sqrt(t))
    vega = S * disc_q * pdf_d1 * math.sqrt(t)
    common = -S * disc_q * pdf_d1 * sigma / (2.0 * math.sqrt(t))

    if _is_call(flag):
        delta = disc_q * norm_cdf(d_1)
        theta = common - r * K * disc_r * norm_cdf(d_2) + q * S * disc_q * norm_cdf(d_1)
        rho = K * t * disc_r * norm_cdf(d_2)
    else:
        delta = -disc_q * norm_cdf(-d_1)
        theta = common + r * K * disc_r * norm_cdf(-d_2) - q * S * disc_q * norm_cdf(-d_1)
        rho = -K * t * disc_r * norm_cdf(-d_2)

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta / 365.0,
        "vega": vega * 0.01,
        "rho": rho * 0.01,
    }


# ---------------------------------------------------------------------------
# Numerical greeks (bump and revalue)


def numerical_greeks(flag, S, K, t, r, sigma, q=0.0, divs=None,
                     style="european", method="bs2002",
                     dividend_model="forward", n_steps=None):
    """Finite difference greeks for European or American options."""
    flag = validate_option(flag, S, K, t, r, sigma, q=q, divs=divs)
    S, K, t, r, sigma, q = map(float, (S, K, t, r, sigma, q))

    def value(S_, t_, sigma_, r_):
        return _price(flag, S_, K, t_, r_, sigma_, q, divs, style, method,
                      dividend_model, n_steps)

    # Delta and gamma in the spot dimension (central differences).
    ds = _BUMP_S * max(S, 1.0)
    up = value(S + ds, t, sigma, r)
    down = value(S - ds, t, sigma, r)
    delta = (up - down) / (2.0 * ds)
    gamma = (up - 2.0 * value(S, t, sigma, r) + down) / (ds * ds)

    # Theta: derivative with respect to time to expiry, negated.
    dt = _BUMP_T * max(t, 1e-3)
    if t > dt:
        theta = -(value(S, t + dt, sigma, r) - value(S, t - dt, sigma, r)) / (2.0 * dt)
    else:
        theta = -(value(S, t + dt, sigma, r) - value(S, t, sigma, r)) / dt
    theta = theta / 365.0

    # Vega: derivative with respect to volatility, scaled per 1%.
    dsigma = _BUMP_SIGMA
    if sigma > dsigma:
        vega = (value(S, t, sigma + dsigma, r) - value(S, t, sigma - dsigma, r)) / (
            2.0 * dsigma
        )
    else:
        vega = (value(S, t, sigma + dsigma, r) - value(S, t, sigma, r)) / dsigma
    vega = vega * 0.01

    # Rho: derivative with respect to the rate, scaled per 1%.
    dr = _BUMP_R
    rho = (value(S, t, sigma, r + dr) - value(S, t, sigma, r - dr)) / (2.0 * dr)
    rho = rho * 0.01

    return {"delta": float(delta), "gamma": float(gamma), "theta": float(theta),
            "vega": float(vega), "rho": float(rho)}


# ---------------------------------------------------------------------------
# Public dispatch


def _resolve_type(greeks_type, style, divs):
    if greeks_type is not None:
        return greeks_type
    if style == "european" and not divs:
        return "analytical"
    return "numerical"


def greeks(flag, S, K, t, r, sigma, q=0.0, divs=None, style="european",
           method="bs2002", dividend_model="forward", n_steps=None,
           greeks_type=None):
    """Return a dict of ``delta, gamma, theta, vega, rho``.

    Analytical formulas are used for European options with no discrete
    dividends; otherwise finite differences are used. Pass
    ``greeks_type='analytical'`` or ``'numerical'`` to force a choice.
    """
    kind = _resolve_type(greeks_type, style, divs)
    if kind == "analytical":
        return analytical_greeks(flag, S, K, t, r, sigma, q)
    return numerical_greeks(
        flag, S, K, t, r, sigma, q=q, divs=divs, style=style, method=method,
        dividend_model=dividend_model, n_steps=n_steps,
    )


def delta(flag, S, K, t, r, sigma, **kwargs):
    """Option delta."""
    return greeks(flag, S, K, t, r, sigma, **kwargs)["delta"]


def gamma(flag, S, K, t, r, sigma, **kwargs):
    """Option gamma."""
    return greeks(flag, S, K, t, r, sigma, **kwargs)["gamma"]


def theta(flag, S, K, t, r, sigma, **kwargs):
    """Option theta, per calendar day."""
    return greeks(flag, S, K, t, r, sigma, **kwargs)["theta"]


def vega(flag, S, K, t, r, sigma, **kwargs):
    """Option vega, per 1% change in volatility."""
    return greeks(flag, S, K, t, r, sigma, **kwargs)["vega"]


def rho(flag, S, K, t, r, sigma, **kwargs):
    """Option rho, per 1% change in the risk-free rate."""
    return greeks(flag, S, K, t, r, sigma, **kwargs)["rho"]
