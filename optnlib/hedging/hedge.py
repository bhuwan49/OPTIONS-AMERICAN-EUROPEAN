# -*- coding: utf-8 -*-

"""
optnlib.hedging.hedge
~~~~~~~~~~~~~~~~~~~~~

Solve the hedge-weight linear system

.. math::

    A \\, w = -b ,

where

* ``A`` has one **row per greek** and one **column per hedge instrument**
  (each entry is that instrument's greek *per traded unit*),
* ``b`` is the **portfolio** greek vector, and
* ``w`` is the vector of signed quantities (contracts/shares) to trade.

For the canonical delta/gamma/vega neutralization with
``[underlying, H1, H2]`` this is the familiar 3x3 system::

    [ delta_u  delta_H1  delta_H2 ] [ w_u  ]   [ -delta_pf ]
    [ gamma_u  gamma_H1  gamma_H2 ] [ w_H1 ] = [ -gamma_pf ]
    [ vega_u   vega_H1   vega_H2  ] [ w_H2 ]   [ -vega_pf  ]

with ``delta_u = 1`` and ``gamma_u = vega_u = 0``.

A square system is solved exactly with :func:`numpy.linalg.solve`; an
over/under-determined system is solved in the least-squares sense with
:func:`numpy.linalg.lstsq` (minimum-norm weights when under-determined).
"""

from dataclasses import replace

import numpy as np

from ..exceptions import InvalidInputError
from .instruments import Underlying
from .portfolio import Portfolio

_ALL_GREEKS = ("delta", "gamma", "theta", "vega", "rho")


def _fmt(x, decimals=None):
    if decimals is None:
        s = "{:,.6f}".format(float(x)).rstrip("0").rstrip(".")
        return s if s not in ("", "-0", "-") else "0"
    return "{:,.{d}f}".format(float(x), d=decimals)


class HedgeResult:
    """Outcome of a hedge solve.

    Attributes
    ----------
    instruments : list
        Hedge instruments, in matrix column order.
    quantities : numpy.ndarray
        Signed number of units to trade per instrument (+ long, - short).
    greeks : tuple of str
        The greeks that were constrained (matrix rows).
    matrix : numpy.ndarray
        The ``A`` matrix (rows = greeks, columns = instruments).
    rhs : numpy.ndarray
        The right-hand side ``target - b``.
    residual : dict
        Post-hedge residual per constrained greek (target is ~0).
    unused : list of str
        Greeks that were *not* constrained (no DOF left for them).
    method : str
        ``'solve'`` or ``'lstsq'``.
    """

    def __init__(self, instruments, quantities, greeks, matrix, rhs,
                 residual, portfolio_greeks, target, unused, method):
        self.instruments = instruments
        self.quantities = np.asarray(quantities, dtype=float)
        self.greeks = tuple(greeks)
        self.matrix = np.asarray(matrix, dtype=float)
        self.rhs = np.asarray(rhs, dtype=float)
        self.residual = dict(residual)
        self.portfolio_greeks = dict(portfolio_greeks)
        self.target = dict(target)
        self.unused = list(unused)
        self.method = method

    # -- programmatic ----------------------------------------------------
    def to_dict(self):
        """Map instrument label -> signed quantity."""
        return {inst.name or inst.label: float(q)
                for inst, q in zip(self.instruments, self.quantities)}

    def to_portfolio(self):
        """The hedge trades as a :class:`Portfolio` of legs."""
        return Portfolio([
            replace(inst, quantity=float(q))
            for inst, q in zip(self.instruments, self.quantities)
        ])

    def __getitem__(self, instrument):
        """Signed quantity for a given instrument instance or its index."""
        if isinstance(instrument, int):
            return float(self.quantities[instrument])
        for inst, q in zip(self.instruments, self.quantities):
            if inst is instrument:
                return float(q)
        raise KeyError(instrument)

    # -- reporting -------------------------------------------------------
    def _summary(self, decimals=None):
        parts = []
        for inst, w in zip(self.instruments, self.quantities):
            if abs(w) < 1e-12:
                continue
            direction = "LONG" if w > 0 else "SHORT"
            name = inst.name or inst.label
            parts.append("%s %s %s (%s)"
                         % (direction, _fmt(abs(w), decimals),
                            inst.trade_unit, name))
        return ", ".join(parts) if parts else "no trades required"

    def describe(self, decimals=None):
        """Formatted hedging report (also returned by ``str(result)``)."""
        lines = []
        lines.append("Hedging plan  (neutralize: %s)" % ", ".join(self.greeks))
        lines.append("=" * 62)
        lines.append("Portfolio before hedge:")
        for g in self.greeks:
            lines.append("    %-6s = %+.6f" % (g, self.portfolio_greeks[g]))
        lines.append("")
        lines.append("Required trades  (+ = go long, - = go short):")
        lines.append("-" * 62)
        for inst, w in zip(self.instruments, self.quantities):
            if w > 1e-12:
                direction = "LONG"
            elif w < -1e-12:
                direction = "SHORT"
            else:
                direction = "FLAT"
            name = inst.name or inst.label
            lines.append("  %-30s %-6s %14s %s"
                         % (name, direction, _fmt(abs(w), decimals),
                            inst.trade_unit))
        lines.append("")
        lines.append("Residual greeks after hedge (target = 0):")
        for g in self.greeks:
            lines.append("  %-6s = %+.6f" % (g, self.residual[g]))
        if self.unused:
            lines.append("Not neutralized (no degrees of freedom): "
                         + ", ".join(self.unused))
        lines.append("")
        lines.append("Summary: " + self._summary(decimals))
        return "\n".join(lines)

    def __str__(self):
        return self.describe()


def hedge(portfolio, instruments, greeks=("delta", "gamma", "vega"),
          target=None):
    """Neutralize (or set to ``target``) the portfolio's greeks.

    :param portfolio: :class:`Portfolio` to hedge.
    :param instruments: sequence of hedge instruments (options and/or the
        underlying). The number of instruments should normally equal the
        number of greeks; otherwise a least-squares solution is returned.
    :param greeks: which greeks to constrain, e.g. ``('delta','gamma','vega')``.
    :param target: optional dict of desired post-hedge greek values
        (default: zeros).
    :returns: :class:`HedgeResult`.
    """
    instruments = list(instruments)
    greeks = tuple(greeks)
    if not instruments:
        raise InvalidInputError("at least one hedge instrument is required")
    if not greeks:
        raise InvalidInputError("at least one greek must be specified")
    for g in greeks:
        if g not in _ALL_GREEKS:
            raise InvalidInputError(
                "unknown greek %r; choose from %s" % (g, _ALL_GREEKS))

    pf_greeks = portfolio.greeks()
    b = np.array([pf_greeks[g] for g in greeks], dtype=float)
    A = np.array([[inst.contract_greeks()[g] for inst in instruments]
                  for g in greeks], dtype=float)

    tgt = np.zeros(len(greeks))
    if target is not None:
        tgt = np.array([float(target.get(g, 0.0)) for g in greeks])
    rhs = tgt - b

    if A.shape[0] == A.shape[1]:
        try:
            w = np.linalg.solve(A, rhs)
            method = "solve"
        except np.linalg.LinAlgError:
            raise InvalidInputError(
                "hedge matrix is singular: the chosen instruments do not span "
                "the requested greeks %s (e.g. duplicate or linearly dependent "
                "exposures)" % (greeks,))
    else:
        w, _res, _rank, _sv = np.linalg.lstsq(A, rhs, rcond=None)
        method = "lstsq"

    new_greeks = b + A.dot(w)
    residual = {g: float(new_greeks[i] - tgt[i]) for i, g in enumerate(greeks)}
    target_map = {g: float(tgt[i]) for i, g in enumerate(greeks)}
    unused = [g for g in _ALL_GREEKS if g not in greeks]

    return HedgeResult(
        instruments=instruments, quantities=w, greeks=greeks, matrix=A,
        rhs=rhs, residual=residual, portfolio_greeks=pf_greeks,
        target=target_map, unused=unused, method=method,
    )


def delta_hedge(portfolio, underlying=None):
    """Neutralize only delta using a single underlying position.

    If ``underlying`` is omitted, a flat :class:`Underlying` at the
    portfolio's reference spot is used (so only its weight matters).
    """
    if underlying is None:
        underlying = Underlying(quantity=0.0, S=portfolio.spot())
    return hedge(portfolio, [underlying], greeks=("delta",))
