# -*- coding: utf-8 -*-

"""
optnlib.dividends
~~~~~~~~~~~~~~~~~

Handling of continuous dividend yields and discrete cash dividend schedules.

A discrete schedule is a sequence of ``(ex_time, amount)`` pairs, where
``ex_time`` is the (future) time of the ex-dividend date measured in years and
``amount`` is the cash amount paid per share. Only dividends whose ex-time lies
strictly inside ``(0, t]`` affect the option.

Two models are supported, matching common market practice:

``'forward'``
    Each discrete cash amount is treated as a forward cashflow. The pricing
    forward is ``S e^{(r-q)t} - sum_i D_i e^{r (t - t_i)}``.

``'escrowed'``
    The present value of the discrete dividends is removed from the spot:
    ``S - sum_i D_i e^{-r t_i}``.
"""

import math

from .exceptions import InvalidInputError

_VALID_MODELS = ("forward", "escrowed")


def normalize_divs(divs, t):
    """Return a sorted list of ``(ex_time, amount)`` pairs inside ``(0, t]``."""
    if divs is None:
        return []
    out = []
    for ex_time, amount in divs:
        ex_time = float(ex_time)
        amount = float(amount)
        if amount == 0.0:
            continue
        if ex_time <= 0.0:
            raise InvalidInputError(
                "dividend ex-times must be strictly positive; got %r" % (ex_time,)
            )
        if ex_time > float(t):
            continue
        out.append((ex_time, amount))
    out.sort(key=lambda item: item[0])
    return out


def validate_model(dividend_model):
    if dividend_model not in _VALID_MODELS:
        raise InvalidInputError(
            "dividend_model must be one of %s; got %r"
            % (", ".join(_VALID_MODELS), dividend_model)
        )
    return dividend_model


def pv_discrete(divs, r, t):
    """Present value of the discrete dividends: ``sum_i D_i e^{-r t_i}``."""
    schedule = normalize_divs(divs, t)
    return sum(amount * math.exp(-float(r) * ex_time) for ex_time, amount in schedule)


def adjusted_forward(S, t, r, q, divs):
    """Dividend adjusted forward ``S e^{(r-q)t} - sum_i D_i e^{r (t - t_i)}``."""
    schedule = normalize_divs(divs, t)
    forward = float(S) * math.exp((float(r) - float(q)) * float(t))
    for ex_time, amount in schedule:
        forward -= amount * math.exp(float(r) * (float(t) - ex_time))
    return forward


def adjusted_spot(S, r, divs, t):
    """Escrowed spot ``S - sum_i D_i e^{-r t_i}``."""
    return float(S) - pv_discrete(divs, r, t)


def effective_yield(S, t, r, q, divs, dividend_model="forward"):
    """Return the continuous yield ``q_eff`` equivalent to the dividend handling.

    Used by engines that only accept a continuous dividend yield (for example
    the Bjerksund-Stensland approximation).
    """
    validate_model(dividend_model)
    schedule = normalize_divs(divs, t)
    if not schedule:
        return float(q)
    t = float(t)
    S = float(S)
    r = float(r)
    q = float(q)
    if t <= 0.0:
        return q
    if dividend_model == "forward":
        forward = S * math.exp((r - q) * t) - sum(
            amount * math.exp(r * (t - ex_time)) for ex_time, amount in schedule
        )
    else:  # escrowed
        pv = sum(amount * math.exp(-r * ex_time) for ex_time, amount in schedule)
        forward = (S - pv) * math.exp((r - q) * t)
    ratio = forward / S
    if ratio <= 0.0:
        raise InvalidInputError(
            "dividends are too large for the spot: equivalent yield is undefined"
        )
    return r - math.log(ratio) / t
