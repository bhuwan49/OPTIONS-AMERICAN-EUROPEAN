# -*- coding: utf-8 -*-

"""
optnlib.hedging.portfolio
~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`Portfolio` is an ordered collection of legs (options and/or the
underlying). It aggregates value and greeks, and provides full-revaluation
scenario analysis plus a Taylor ("greeks") P&L attribution.

Sign/shock conventions
----------------------
``scenario`` and ``pnl_attribution`` take **absolute** shocks:

* ``spot_shock``  change in the underlying price (e.g. ``-5.0`` dollars),
* ``vol_shock``   change in volatility (e.g. ``0.02`` = +2 vol points),
* ``days``        calendar days of time decay (positive = time passes),
* ``rate_shock``  change in the rate (e.g. ``0.01`` = +1%).

Turning a relative move into a shock is the caller's job, e.g.
``spot_shock = 0.05 * S``.
"""

from .instruments import Underlying

_GREEKS = ("delta", "gamma", "theta", "vega", "rho")


class Portfolio:
    """An ordered bag of option/underlying legs."""

    def __init__(self, legs=None):
        self.legs = list(legs) if legs else []

    # -- construction ----------------------------------------------------
    def add(self, leg):
        self.legs.append(leg)
        return leg

    def __len__(self):
        return len(self.legs)

    def __iter__(self):
        return iter(self.legs)

    def __repr__(self):
        return "Portfolio(%d legs)" % (len(self.legs),)

    # -- market helpers --------------------------------------------------
    def spot(self):
        """Reference spot (that of the first leg; all legs share one spot)."""
        return self.legs[0].S if self.legs else 0.0

    # -- value & greeks --------------------------------------------------
    def value(self):
        """Mark-to-market value of the whole portfolio."""
        return float(sum(leg.value() for leg in self.legs))

    def greeks(self, S=None, t=None, sigma=None, r=None):
        """Aggregated portfolio greeks as a dict."""
        total = {k: 0.0 for k in _GREEKS}
        for leg in self.legs:
            pg = leg.position_greeks(S=S, t=t, sigma=sigma, r=r)
            for k in _GREEKS:
                total[k] += pg[k]
        return total

    def delta(self):
        return self.greeks()["delta"]

    def gamma(self):
        return self.greeks()["gamma"]

    def theta(self):
        return self.greeks()["theta"]

    def vega(self):
        return self.greeks()["vega"]

    def rho(self):
        return self.greeks()["rho"]

    def dollar_greeks(self, spot=None):
        """Greeks in dollar terms.

        * ``delta`` dollars per +1.0 move in spot,
        * ``gamma`` dollars per +1% move in spot,
        * ``vega``  dollars per +1 vol point,
        * ``theta`` dollars per calendar day,
        * ``rho``   dollars per +1% rate move.
        """
        s = self.spot() if spot is None else float(spot)
        g = self.greeks()
        return {
            "delta": g["delta"] * s,
            "gamma": g["gamma"] * s * s * 0.01,
            "vega": g["vega"],
            "theta": g["theta"],
            "rho": g["rho"],
        }

    # -- scenario & attribution ------------------------------------------
    def scenario(self, spot_shock=0.0, vol_shock=0.0, days=0.0,
                 rate_shock=0.0):
        """Full-revaluation P&L for the given absolute shocks."""
        base = self.value()
        shifted = 0.0
        for leg in self.legs:
            S = leg.S + spot_shock
            sigma = leg.sigma + vol_shock
            t = max(leg.t - days / 365.0, 0.0)
            r = leg.r + rate_shock
            shifted += leg.quantity * leg.multiplier * leg.reprice(
                S=S, t=t, sigma=sigma, r=r)
        return float(shifted - base)

    def pnl_attribution(self, spot_shock=0.0, vol_shock=0.0, days=0.0,
                        rate_shock=0.0):
        """Taylor P&L attribution (delta + gamma + vega + theta + rho).

        Returns a dict with the per-greek contributions, the Taylor ``total``,
        the full-revaluation ``actual`` P&L and the attribution ``error``.
        """
        g = self.greeks()
        dS = float(spot_shock)
        delta_pnl = g["delta"] * dS
        gamma_pnl = 0.5 * g["gamma"] * dS * dS
        vega_pnl = g["vega"] * (vol_shock / 0.01)
        theta_pnl = g["theta"] * days
        rho_pnl = g["rho"] * (rate_shock / 0.01)
        total = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + rho_pnl
        actual = self.scenario(spot_shock, vol_shock, days, rate_shock)
        return {
            "delta": float(delta_pnl),
            "gamma": float(gamma_pnl),
            "vega": float(vega_pnl),
            "theta": float(theta_pnl),
            "rho": float(rho_pnl),
            "total": float(total),
            "actual": float(actual),
            "error": float(total - actual),
        }

    # -- reporting -------------------------------------------------------
    def describe(self):
        """Human-readable summary of the portfolio."""
        g = self.greeks()
        header = "Portfolio  ({} leg{})   value = {:,.2f}".format(
            len(self.legs), "" if len(self.legs) == 1 else "s", self.value())
        lines = [header]
        for leg in self.legs:
            label = leg.name or leg.label
            lines.append("    %-28s qty = %+g" % (label, leg.quantity))
        lines.append(
            "    delta = %+.6f  gamma = %+.6f  theta = %+.6f  "
            "vega = %+.6f  rho = %+.6f"
            % (g["delta"], g["gamma"], g["theta"], g["vega"], g["rho"])
        )
        return "\n".join(lines)

    def __str__(self):
        return self.describe()
