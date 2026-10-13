# -*- coding: utf-8 -*-

"""
optnlib
~~~~~~~

European and American option pricing, implied volatility and greeks, with
support for continuous dividend yields and discrete cash dividend schedules.

Quick start
-----------

::

    import optnlib

    # European Black-Scholes
    optnlib.black_scholes('c', 100, 100, 1.0, 0.05, 0.2)

    # American put (Bjerksund-Stensland 2002 approximation)
    optnlib.american_price('p', 100, 100, 1.0, 0.05, 0.2)

    # Unified entry point
    optnlib.price('p', 100, 100, 1.0, 0.05, 0.2, style='american', method='lr')

See ``EXAMPLES.md`` for runnable examples covering every feature.
"""

from . import american, core, dividends, european, exceptions, hedging
from .american import price as american_price
from .european import (
    black,
    black_scholes,
    black_scholes_merton,
    price as european_price,
)
from .greeks import delta, gamma, greeks, rho, theta, vega
from .implied_volatility import implied_volatility

__version__ = "0.1.0"


def price(flag, S, K, t, r, sigma, q=0.0, divs=None, style="european",
          method=None, dividend_model="forward", n_steps=None):
    """Unified option price for European and American options.

    :param flag: ``'c'`` or ``'p'``.
    :param S: underlying spot price.
    :param K: strike price (> 0).
    :param t: time to expiry in years.
    :param r: annualized continuously compounded risk-free rate.
    :param sigma: annualized volatility.
    :param q: annualized continuous dividend yield (default 0).
    :param divs: discrete schedule ``[(ex_time, amount), ...]`` (default None).
    :param style: ``'european'`` (default) or ``'american'``.
    :param method: American engine ``'bs2002'``/``'crr'``/``'lr'`` (ignored for
        European). Defaults to ``'bs2002'`` for American options.
    :param dividend_model: ``'forward'`` (default) or ``'escrowed'``.
    :param n_steps: number of tree steps when ``method`` is ``'crr'`` or ``'lr'``.
    """
    if style == "american":
        return american_price(
            flag, S, K, t, r, sigma, q=q, divs=divs,
            method="bs2002" if method is None else method,
            dividend_model=dividend_model, n_steps=n_steps,
        )
    if style not in ("european", "euro", "eu"):
        raise ValueError("style must be 'european' or 'american'; got %r" % (style,))
    return european_price(flag, S, K, t, r, sigma, q=q, divs=divs,
                          dividend_model=dividend_model)


__all__ = [
    "price",
    "european_price",
    "american_price",
    "black_scholes",
    "black_scholes_merton",
    "black",
    "implied_volatility",
    "greeks",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "european",
    "american",
    "core",
    "dividends",
    "exceptions",
    "hedging",
]
