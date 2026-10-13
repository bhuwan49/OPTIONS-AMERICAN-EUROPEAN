# -*- coding: utf-8 -*-

"""
optnlib.american
~~~~~~~~~~~~~~~~

American option pricing.

Engines
-------
``'bs2002'``
    Bjerksund-Stensland (2002) closed-form approximation (default, fast).
``'crr'``
    Cox-Ross-Rubinstein binomial tree.
``'lr'``
    Leisen-Reimer binomial tree (accurate, recommended over ``'crr'``).

Discrete cash dividends are supported under two models:

``'forward'``
    Dividends drop the tree nodes at their ex-dates (timing aware); for the
    Bjerksund-Stensland engine they are converted to an equivalent continuous
    yield.
``'escrowed'``
    The present value of the dividends is removed from the spot up front.
"""

from ..core import intrinsic
from ..dividends import adjusted_spot, effective_yield, validate_model
from ..exceptions import InvalidInputError
from ..helpers import validate_option
from . import binomial, bs2002

_METHODS = ("bs2002", "crr", "lr")


def price(flag, S, K, t, r, sigma, q=0.0, divs=None, method="bs2002",
          dividend_model="forward", n_steps=None):
    """American option price.

    :param flag: ``'c'`` or ``'p'``.
    :param S: underlying spot price.
    :param K: strike price (> 0).
    :param t: time to expiry in years.
    :param r: annualized continuously compounded risk-free rate.
    :param sigma: annualized volatility.
    :param q: annualized continuous dividend yield (default 0).
    :param divs: discrete schedule ``[(ex_time, amount), ...]`` (default None).
    :param method: ``'bs2002'`` (default), ``'crr'`` or ``'lr'``.
    :param dividend_model: ``'forward'`` (default) or ``'escrowed'``.
    :param n_steps: number of tree steps (binomial engines only).
    """
    flag = validate_option(flag, S, K, t, r, sigma, q=q, divs=divs)
    model = validate_model(dividend_model)

    if method not in _METHODS:
        raise InvalidInputError(
            "method must be one of %s; got %r" % (", ".join(_METHODS), method)
        )

    if float(t) == 0.0:
        return intrinsic(flag, S, K)

    # Near-zero volatility: the tree degenerates and the Bjerksund-Stensland
    # boundary terms overflow, so both fall back to a deterministic value.
    if float(sigma) <= 1e-4:
        return bs2002.price(flag, S, K, t, r, sigma, q)

    # The approximation only understands a continuous yield: fold the discrete
    # schedule into an equivalent yield for whichever dividend model was chosen.
    if method == "bs2002":
        q_eff = effective_yield(S, t, r, q, divs, model)
        return bs2002.price(flag, S, K, t, r, sigma, q_eff)

    # Binomial engines.
    if model == "escrowed" and divs:
        spot = adjusted_spot(S, r, divs, t)
        if spot <= 0.0:
            raise InvalidInputError(
                "escrowed spot is non-positive; dividends too large"
            )
        return binomial.price(
            flag, spot, K, t, r, sigma, q=q, divs=None,
            method=method, n_steps=n_steps,
        )

    return binomial.price(
        flag, S, K, t, r, sigma, q=q, divs=divs,
        method=method, n_steps=n_steps,
    )


__all__ = ["price", "bs2002", "binomial"]
