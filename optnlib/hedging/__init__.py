# -*- coding: utf-8 -*-

"""
optnlib.hedging
~~~~~~~~~~~~~~~

Greeks-based hedging and risk management for option/underlying portfolios.

Build a :class:`Portfolio` from :class:`OptionLeg` / :class:`Underlying`
positions, then neutralize delta/gamma/vega (or any set of greeks) against a
basket of hedge instruments::

    from optnlib import hedging as h
    from optnlib import european_price

    pf = h.Portfolio([
        h.OptionLeg(quantity=-10, flag='c', S=100, K=105, t=0.25, r=0.05,
                    sigma=0.25, multiplier=100),
        h.Underlying(quantity=500, S=100),
    ])

    plan = h.hedge(pf, [
        h.Underlying(S=100),
        h.OptionLeg(flag='c', S=100, K=100, t=0.25, r=0.05, sigma=0.25,
                    multiplier=100),
        h.OptionLeg(flag='p', S=100, K=110, t=0.25, r=0.05, sigma=0.25,
                    multiplier=100),
    ], greeks=('delta', 'gamma', 'vega'))

    print(plan)
"""

from .hedge import HedgeResult, delta_hedge, hedge
from .instruments import OptionLeg, Underlying
from .portfolio import Portfolio
from .regularized import hedge_optimize, hedge_ridge

__all__ = [
    "OptionLeg",
    "Underlying",
    "Portfolio",
    "HedgeResult",
    "hedge",
    "delta_hedge",
    "hedge_ridge",
    "hedge_optimize",
]
