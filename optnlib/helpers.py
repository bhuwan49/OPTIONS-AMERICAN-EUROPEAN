# -*- coding: utf-8 -*-

"""
optnlib.helpers
~~~~~~~~~~~~~~~

Input validation shared by the pricing engines.
"""

from .core import _normalize_flag
from .exceptions import InvalidInputError


def validate_option(flag, S, K, t, r, sigma, q=0.0, divs=None):
    """Validate the common option parameters.

    Returns the normalized ``flag`` (``'c'`` or ``'p'``).
    """
    flag = _normalize_flag(flag)

    if float(K) <= 0.0:
        raise InvalidInputError("strike K must be strictly positive; got %r" % (K,))
    if float(S) <= 0.0:
        raise InvalidInputError("spot S must be strictly positive; got %r" % (S,))
    if float(t) < 0.0:
        raise InvalidInputError("time to expiry t must be >= 0; got %r" % (t,))
    if float(sigma) < 0.0:
        raise InvalidInputError("volatility sigma must be >= 0; got %r" % (sigma,))
    if divs is not None:
        for item in divs:
            try:
                ex_time, amount = item
            except (TypeError, ValueError):
                raise InvalidInputError(
                    "each dividend must be a (ex_time, amount) pair; got %r" % (item,)
                )
            if float(ex_time) <= 0.0:
                raise InvalidInputError(
                    "dividend ex-time must be > 0; got %r" % (ex_time,)
                )
    return flag
