# -*- coding: utf-8 -*-

"""
optnlib.implied_volatility
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implied volatility for European and American options.

The implied volatility is the volatility that reproduces an observed option
price. For European options the underlying model is priced directly; for
American options the chosen American engine is used (and the result therefore
depends on that engine).
"""

from scipy.optimize import brentq

from .exceptions import InvalidInputError
from .helpers import validate_option

_LOWER = 1e-8
_UPPER = 5.0


def _solve_price(func, target):
    """Find ``sigma`` such that ``func(sigma) == target`` on a wide bracket."""
    lo, hi = _LOWER, _UPPER
    f_lo = func(lo) - target
    f_hi = func(hi) - target
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo * f_hi > 0.0:
        raise InvalidInputError(
            "option price is outside the range reproducible by the model "
            "(no implied volatility exists)"
        )
    return brentq(lambda s: func(s) - target, lo, hi, xtol=1e-12, rtol=1e-12,
                  maxiter=200)


def implied_volatility(price, flag, S, K, t, r, q=0.0, divs=None,
                       style="european", method="bs2002",
                       dividend_model="forward", n_steps=None):
    """Implied volatility.

    :param price: observed option price.
    :param flag: ``'c'`` or ``'p'``.
    :param S: underlying spot price.
    :param K: strike price (> 0).
    :param t: time to expiry in years (> 0).
    :param r: annualized continuously compounded risk-free rate.
    :param q: annualized continuous dividend yield (default 0).
    :param divs: discrete schedule ``[(ex_time, amount), ...]`` (default None).
    :param style: ``'european'`` (default) or ``'american'``.
    :param method: American engine when ``style='american'``.
    :param dividend_model: ``'forward'`` (default) or ``'escrowed'``.
    :param n_steps: tree steps when ``style='american'`` and ``method`` is a tree.
    """
    flag = validate_option(flag, S, K, t, r, sigma=0.0, q=q, divs=divs)
    price = float(price)

    if float(t) <= 0.0:
        raise InvalidInputError("implied volatility requires t > 0")
    if price < 0.0:
        raise InvalidInputError("option price must be non-negative")

    if style == "american":
        from .american import price as american_price

        def model(sigma):
            return american_price(flag, S, K, t, r, sigma, q=q, divs=divs,
                                  method=method, dividend_model=dividend_model,
                                  n_steps=n_steps)
    else:
        from .european import price as european_price

        def model(sigma):
            return european_price(flag, S, K, t, r, sigma, q=q, divs=divs,
                                  dividend_model=dividend_model)

    return _solve_price(model, price)
