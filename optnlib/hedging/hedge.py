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

Solver selection
----------------
The matrix is first **row-scaled** (each greek row divided by its norm)
because greeks live on very different scales. It is then **diagnosed**:
its rank and condition number are computed. Only a square, full-rank,
well-conditioned system is solved exactly with :func:`numpy.linalg.solve`.
Anything else -- over/under-determined, rank-deficient or ill-conditioned --
falls back to :func:`numpy.linalg.lstsq` (minimum-norm weights), because
``solve`` on such a matrix silently returns huge, mutually-offsetting
positions. This mirrors the "auto-dispatcher" in the ``greek_hedging``
reference notebook.

A useful consequence: for Black-Scholes, ``vega / gamma = S**2 * sigma * t``
is the *same* for every option sharing spot, vol and expiry. So two hedge
options with identical expiry/vol have collinear (gamma, vega) columns and the
delta/gamma/vega system is rank-deficient. Use options with different expiries
or vols if you need to neutralize gamma and vega independently.
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
        ``'solve'``, ``'lstsq'`` or ``'pinv'``.
    rank : int or None
        Numerical rank of the (optionally row-scaled) matrix.
    condition : float or None
        Condition number of the (optionally row-scaled) matrix.
    min_dim : int or None
        ``min(matrix.shape)`` -- the largest rank the system could have.
    scaled : bool
        Whether the matrix was row-scaled before solving.
    """

    def __init__(self, instruments, quantities, greeks, matrix, rhs,
                 residual, portfolio_greeks, target, unused, method,
                 rank=None, condition=None, min_dim=None, scaled=False):
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
        self.rank = rank
        self.condition = condition
        self.min_dim = min_dim
        self.scaled = scaled

    @property
    def full_rank(self):
        """Whether the matrix has full (numerical) rank."""
        return (self.rank is not None and self.min_dim is not None
                and self.rank == self.min_dim)

    @property
    def well_conditioned(self):
        """Whether the matrix condition number is finite."""
        return self.condition is not None and np.isfinite(self.condition)

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
        lines.append(self._solver_line())
        warning = self._warning()
        if warning:
            lines.append(warning)
        lines.append("")
        lines.append("Summary: " + self._summary(decimals))
        return "\n".join(lines)

    def _solver_line(self):
        bits = []
        if self.rank is not None and self.min_dim is not None:
            bits.append("rank %d/%d" % (self.rank, self.min_dim))
        if self.condition is not None:
            bits.append("cond %.2e" % self.condition)
        if self.scaled:
            bits.append("row-scaled")
        detail = ("  (" + ", ".join(bits) + ")") if bits else ""
        return "Solver  : %s%s" % (self.method, detail)

    def _warning(self):
        """A plain-language note when the matrix was not solved exactly."""
        if self.full_rank and self.well_conditioned:
            return None
        resid_norm = max((abs(v) for v in self.residual.values()), default=0.0)
        rhs_norm = max((abs(v) for v in self.rhs), default=0.0)
        if resid_norm <= 1e-6 * (1.0 + rhs_norm):
            return (
                "Note    : hedge matrix is rank-deficient (rank %d/%d); the "
                "target is\n          still reachable, so the minimum-norm "
                "weights below are an exact\n          hedge (many equivalent "
                "solutions exist)." % (self.rank, self.min_dim))
        return (
            "Warning : hedge matrix is rank-deficient/ill-conditioned -- these\n"
            "          instruments cannot fully neutralize %s;\n"
            "          minimum-norm weights shown, residual risk remains."
            % ", ".join(self.greeks))

    def __str__(self):
        return self.describe()


def _diagnose(A_work):
    """Return ``(rank, min_dim, condition)`` of a (scaled) matrix."""
    if A_work.size == 0:
        return 0, 0, 0.0
    min_dim = int(min(A_work.shape))
    try:
        rank = int(np.linalg.matrix_rank(A_work))
    except np.linalg.LinAlgError:
        rank = 0
    try:
        condition = float(np.linalg.cond(A_work))
    except np.linalg.LinAlgError:
        condition = np.inf
    return rank, min_dim, condition


def hedge(portfolio, instruments, greeks=("delta", "gamma", "vega"),
          target=None, method="auto", cond_tol=1e8, use_scaling=True,
          strict=False):
    """Neutralize (or set to ``target``) the portfolio's greeks.

    :param portfolio: :class:`Portfolio` to hedge.
    :param instruments: sequence of hedge instruments (options and/or the
        underlying). The number of instruments should normally equal the
        number of greeks; otherwise a least-squares solution is returned.
    :param greeks: which greeks to constrain, e.g. ``('delta','gamma','vega')``.
    :param target: optional dict of desired post-hedge greek values
        (default: zeros).
    :param method: ``'auto'`` (default), ``'solve'``, ``'lstsq'`` or
        ``'pinv'``. ``'auto'`` uses :func:`numpy.linalg.solve` only for a
        square, full-rank, well-conditioned matrix and otherwise falls back to
        the minimum-norm least-squares solution.
    :param cond_tol: condition-number threshold above which a matrix is
        considered ill-conditioned (default ``1e8``).
    :param use_scaling: row-scale the matrix before diagnosing/solving
        (default ``True``); greeks live on very different scales.
    :param strict: if ``True``, raise :class:`InvalidInputError` when the
        system cannot be solved exactly (rank-deficient or ill-conditioned)
        instead of returning a least-squares hedge.
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
    if method not in ("auto", "solve", "lstsq", "pinv"):
        raise InvalidInputError(
            "unknown method %r; choose from 'auto', 'solve', 'lstsq', 'pinv'"
            % (method,))

    pf_greeks = portfolio.greeks()
    b = np.array([pf_greeks[g] for g in greeks], dtype=float)
    A = np.array([[inst.contract_greeks()[g] for inst in instruments]
                  for g in greeks], dtype=float)

    tgt = np.zeros(len(greeks))
    if target is not None:
        tgt = np.array([float(target.get(g, 0.0)) for g in greeks])
    rhs = tgt - b

    # -- row-scale so greeks on different scales are comparable ----------
    A_work, rhs_work = A, rhs
    if use_scaling and A.shape[0] > 0:
        row_norms = np.linalg.norm(A, axis=1)
        row_norms = np.maximum(row_norms, 1e-12)
        D = np.diag(1.0 / row_norms)
        A_work = D @ A
        rhs_work = D @ rhs

    rank, min_dim, condition = _diagnose(A_work)
    square = A_work.shape[0] == A_work.shape[1]
    full_rank = min_dim > 0 and rank == min_dim
    well_conditioned = np.isfinite(condition) and condition < cond_tol

    if method == "auto":
        chosen = "solve" if (square and full_rank and well_conditioned) \
            else "lstsq"
    else:
        chosen = method

    if strict and not (full_rank and well_conditioned):
        raise InvalidInputError(
            "hedge matrix is singular or ill-conditioned: the chosen "
            "instruments do not span the requested greeks %s (rank %d/%d, "
            "cond %.2e); use different expiries/vols or pass strict=False to "
            "get a least-squares hedge" % (greeks, rank, min_dim, condition))

    if chosen == "solve":
        try:
            w = np.linalg.solve(A_work, rhs_work)
        except np.linalg.LinAlgError:
            w, _res, _rank, _sv = np.linalg.lstsq(A_work, rhs_work, rcond=None)
            chosen = "lstsq"
    elif chosen == "pinv":
        w = np.linalg.pinv(A_work) @ rhs_work
    else:
        w, _res, _rank, _sv = np.linalg.lstsq(A_work, rhs_work, rcond=None)

    new_greeks = b + A.dot(w)
    residual = {g: float(new_greeks[i] - tgt[i]) for i, g in enumerate(greeks)}
    target_map = {g: float(tgt[i]) for i, g in enumerate(greeks)}
    unused = [g for g in _ALL_GREEKS if g not in greeks]

    return HedgeResult(
        instruments=instruments, quantities=w, greeks=greeks, matrix=A,
        rhs=rhs, residual=residual, portfolio_greeks=pf_greeks,
        target=target_map, unused=unused, method=chosen,
        rank=rank, condition=condition, min_dim=min_dim, scaled=bool(use_scaling),
    )


def delta_hedge(portfolio, underlying=None):
    """Neutralize only delta using a single underlying position.

    If ``underlying`` is omitted, a flat :class:`Underlying` at the
    portfolio's reference spot is used (so only its weight matters).
    """
    if underlying is None:
        underlying = Underlying(quantity=0.0, S=portfolio.spot())
    return hedge(portfolio, [underlying], greeks=("delta",))
