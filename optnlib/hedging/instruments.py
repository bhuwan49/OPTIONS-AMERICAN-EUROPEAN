# -*- coding: utf-8 -*-

"""
optnlib.hedging.instruments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Position-level building blocks for hedging: option legs and the (linear)
underlying.

A "leg" bundles a per-unit market instrument with a signed ``quantity`` and a
contract ``multiplier`` so that risk can be aggregated at the position level.

Units
-----
* ``unit_greeks``     greeks of **one** option/share (per-share greeks).
* ``contract_greeks`` greeks of **one traded unit** (contract) of the leg,
  i.e. ``unit_greeks * multiplier``. These are the matrix columns used by the
  hedge solver, so the solved weights are numbers of contracts/shares.
* ``position_greeks`` greeks of the whole leg, i.e. ``contract_greeks * quantity``.
* ``value``           mark-to-market value of the whole leg.

The underlying is a degenerate leg whose only non-zero greek is delta (= 1).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..american import price as _american_price
from ..european import price as _european_price
from ..greeks import greeks as _greeks

_GREEKS = ("delta", "gamma", "theta", "vega", "rho")


def _pick(value, default):
    return default if value is None else value


def _price_option(flag, S, K, t, r, sigma, q, divs, style, method,
                  dividend_model, n_steps):
    if style == "american":
        return _american_price(
            flag, S, K, t, r, sigma, q=q, divs=divs, method=method,
            dividend_model=dividend_model, n_steps=n_steps,
        )
    return _european_price(
        flag, S, K, t, r, sigma, q=q, divs=divs,
        dividend_model=dividend_model,
    )


def _is_call(flag):
    return str(flag).lower().startswith("c")


@dataclass
class OptionLeg:
    """A (possibly multi-contract) option position.

    ``quantity`` is the number of contracts (negative = short). ``multiplier``
    is the number of underlying shares per contract (100 for standard equity
    options, 1 for per-share maths).
    """

    quantity: float = 1.0
    flag: str = "c"
    S: float = 100.0
    K: float = 100.0
    t: float = 1.0
    r: float = 0.0
    sigma: float = 0.2
    q: float = 0.0
    divs: Optional[List[Tuple[float, float]]] = field(default=None)
    style: str = "european"
    method: str = "bs2002"
    dividend_model: str = "forward"
    n_steps: Optional[int] = None
    multiplier: float = 1.0
    name: Optional[str] = None

    # -- pricing ---------------------------------------------------------
    def reprice(self, S=None, t=None, sigma=None, r=None):
        """Price one share of the option, optionally with bumped parameters."""
        return _price_option(
            self.flag,
            _pick(S, self.S),
            self.K,
            _pick(t, self.t),
            _pick(r, self.r),
            _pick(sigma, self.sigma),
            self.q, self.divs, self.style, self.method, self.dividend_model,
            self.n_steps,
        )

    def price(self):
        """Mark-to-market price of one share of the option."""
        return self.reprice()

    # -- greeks ----------------------------------------------------------
    def unit_greeks(self, S=None, t=None, sigma=None, r=None):
        """Per-share greeks of the option."""
        g = _greeks(
            self.flag, _pick(S, self.S), self.K, _pick(t, self.t),
            _pick(r, self.r), _pick(sigma, self.sigma), q=self.q, divs=self.divs,
            style=self.style, method=self.method,
            dividend_model=self.dividend_model, n_steps=self.n_steps,
        )
        return {k: float(g[k]) for k in _GREEKS}

    def contract_greeks(self, S=None, t=None, sigma=None, r=None):
        """Greeks of one traded contract (``unit_greeks * multiplier``)."""
        ug = self.unit_greeks(S=S, t=t, sigma=sigma, r=r)
        return {k: ug[k] * self.multiplier for k in _GREEKS}

    def position_greeks(self, S=None, t=None, sigma=None, r=None):
        """Greeks of the whole leg (``contract_greeks * quantity``)."""
        cg = self.contract_greeks(S=S, t=t, sigma=sigma, r=r)
        return {k: cg[k] * self.quantity for k in _GREEKS}

    # -- value -----------------------------------------------------------
    def value(self):
        """Mark-to-market value of the whole leg."""
        return self.quantity * self.multiplier * self.price()

    # -- labels ----------------------------------------------------------
    @property
    def label(self):
        kind = "Call" if _is_call(self.flag) else "Put"
        return "%s K=%g t=%g" % (kind, self.K, self.t)

    @property
    def trade_unit(self):
        return "contracts" if self.multiplier != 1.0 else "units"


@dataclass
class Underlying:
    """A linear position in the underlying asset (no curvature, no vol risk)."""

    quantity: float = 1.0
    S: float = 100.0
    multiplier: float = 1.0
    name: Optional[str] = None
    t: float = 0.0
    r: float = 0.0
    sigma: float = 0.0

    def reprice(self, S=None, t=None, sigma=None, r=None):
        return _pick(S, self.S)

    def price(self):
        return self.S

    def unit_greeks(self, S=None, t=None, sigma=None, r=None):
        return {"delta": 1.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0,
                "rho": 0.0}

    def contract_greeks(self, S=None, t=None, sigma=None, r=None):
        return {k: v * self.multiplier for k, v in self.unit_greeks().items()}

    def position_greeks(self, S=None, t=None, sigma=None, r=None):
        return {k: v * self.quantity for k, v
                in self.contract_greeks().items()}

    def value(self):
        return self.quantity * self.multiplier * self.price()

    @property
    def label(self):
        return "Underlying (S=%g)" % (self.S,)

    @property
    def trade_unit(self):
        return "shares"
