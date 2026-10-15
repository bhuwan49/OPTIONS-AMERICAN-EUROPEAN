# -*- coding: utf-8 -*-

"""
optnlib.hedging.regularized
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regularized and constrained hedge builders.

These are **separate** methods from the closed-form solvers in
:mod:`optnlib.hedging.hedge` (``solve`` / ``lstsq`` / ``pinv``). They are
designed for the cases where a raw linear solve produces huge, mutually
offsetting trades or fails outright:

* :func:`hedge_ridge` -- Tikhonov / ridge regression
  (``scikit-learn``) adds a penalty ``alpha * ||w||^2`` so the solution is
  always unique and the trade sizes stay bounded, even for a singular or
  ill-conditioned greek matrix;
* :func:`hedge_optimize` -- constrained numerical optimization
  (``scipy.optimize.minimize``) minimizes the (row-scaled) hedging error plus
  optional transaction costs subject to **position limits** and a **gross
  exposure** cap. By construction the weights can never explode.

Both accept a :class:`~optnlib.hedging.portfolio.Portfolio` and the same
``instruments`` / ``greeks`` / ``target`` arguments as
:func:`optnlib.hedging.hedge`, and return a
:class:`~optnlib.hedging.hedge.HedgeResult`.
"""

import numpy as np

from ..exceptions import InvalidInputError
from .hedge import _ALL_GREEKS, HedgeResult

__all__ = ["hedge_ridge", "hedge_optimize"]

_RIDGE_EXTRA_MSG = (
    "hedge_ridge requires scikit-learn; install it with "
    "'pip install scikit-learn' or 'pip install optnlib[ridge]'"
)


def _build_system(portfolio, instruments, greeks, target):
    """Assemble ``(A, b, rhs, pf_greeks, target_map, unused)``.

    ``A`` has one row per greek and one column per instrument. The system to
    solve is ``A @ w = rhs`` with ``rhs = target - b``.
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

    target_map = {g: float(tgt[i]) for i, g in enumerate(greeks)}
    unused = [g for g in _ALL_GREEKS if g not in greeks]
    return instruments, greeks, A, b, rhs, pf_greeks, target_map, unused


def _scale(A, rhs):
    """Row-scale the system so greeks on different scales are comparable."""
    if A.shape[0] == 0:
        return A, rhs
    row_norms = np.maximum(np.linalg.norm(A, axis=1), 1e-12)
    D = np.diag(1.0 / row_norms)
    return D @ A, D @ rhs


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


def _finalize(instruments, greeks, A, b, rhs, pf_greeks, target_map, unused,
              w, method, use_scaling, info):
    """Compute residuals and wrap the solution in a :class:`HedgeResult`."""
    w = np.asarray(w, dtype=float)
    if w.shape != (A.shape[1],):
        raise InvalidInputError("solver returned weights of unexpected shape")
    if not np.all(np.isfinite(w)):
        raise InvalidInputError(
            "%s failed to produce a finite hedge; check the instruments, "
            "bounds and costs" % method)

    A_work, _ = _scale(A, rhs) if use_scaling else (A, rhs)
    rank, min_dim, condition = _diagnose(A_work)

    new_greeks = b + A.dot(w)
    residual = {g: float(new_greeks[i] - target_map[g])
                for i, g in enumerate(greeks)}
    return HedgeResult(
        instruments=instruments, quantities=w, greeks=greeks, matrix=A,
        rhs=rhs, residual=residual, portfolio_greeks=pf_greeks,
        target=target_map, unused=unused, method=method,
        rank=rank, condition=condition, min_dim=min_dim,
        scaled=bool(use_scaling), info=info,
    )


def hedge_ridge(portfolio, instruments, greeks=("delta", "gamma", "vega"),
                target=None, alpha="auto", use_scaling=True):
    """Ridge-regularized hedge (Tikhonov) using ``scikit-learn``.

    Minimizes the (optionally row-scaled) hedging error plus a penalty on the
    trade sizes::

        w = argmin ||A w - rhs||_2^2 + alpha * ||w||_2^2

    The penalty makes the problem strictly convex, so a **unique** solution
    exists even when the greek matrix is singular -- and the weights stay
    bounded instead of exploding. This is a separate method from the
    closed-form solvers in :func:`optnlib.hedging.hedge`.

    :param portfolio: :class:`Portfolio` to hedge.
    :param instruments: hedge instruments (options and/or the underlying).
    :param greeks: which greeks to constrain.
    :param target: optional dict of desired post-hedge greek values.
    :param alpha: ridge penalty. ``'auto'`` (default) selects it by
        leave-one-out generalized cross-validation (``RidgeCV``); pass a float
        for a fixed penalty.
    :param use_scaling: row-scale the system before fitting (default ``True``).
    :returns: :class:`HedgeResult` (``method='ridge'``).
    """
    try:
        from sklearn.linear_model import Ridge, RidgeCV
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise InvalidInputError(_RIDGE_EXTRA_MSG) from exc

    (instruments, greeks, A, b, rhs, pf_greeks,
     target_map, unused) = _build_system(portfolio, instruments, greeks, target)

    A_work, rhs_work = _scale(A, rhs) if use_scaling else (A, rhs)

    if isinstance(alpha, str):
        if alpha != "auto":
            raise InvalidInputError(
                "alpha must be a number or 'auto', got %r" % (alpha,))
        model = RidgeCV(alphas=np.logspace(-8, 1, 40), fit_intercept=False)
    else:
        a = float(alpha)
        if a < 0:
            raise InvalidInputError("alpha must be non-negative")
        model = Ridge(alpha=a, fit_intercept=False)

    model.fit(A_work, rhs_work)
    w = np.asarray(model.coef_, dtype=float)
    chosen_alpha = float(getattr(model, "alpha_", getattr(model, "alpha", 0.0)))

    return _finalize(instruments, greeks, A, b, rhs, pf_greeks, target_map,
                     unused, w, "ridge", use_scaling,
                     info={"alpha": chosen_alpha})


def hedge_optimize(portfolio, instruments, greeks=("delta", "gamma", "vega"),
                   target=None, use_scaling=True, bounds=None,
                   gross_limit=None, costs=None, cost_weight=0.0,
                   method="SLSQP", maxiter=500, ftol=1e-12, x0=None):
    """Constrained hedge via ``scipy.optimize.minimize``.

    Minimizes the (optionally row-scaled) hedging error plus a transaction
    cost term::

        min_w  ||A_s w - rhs_s||_2^2 + cost_weight * sum_j c_j |w_j|

    subject to per-instrument **position bounds** and an optional **gross
    exposure** cap ``sum_j |w_j| <= gross_limit``. Because the weights are
    bounded by construction, the trade sizes can never explode and the solver
    always returns a usable answer -- even for an inconsistent system. This is
    a separate method from the closed-form solvers in
    :func:`optnlib.hedging.hedge`.

    :param portfolio: :class:`Portfolio` to hedge.
    :param instruments: hedge instruments (options and/or the underlying).
    :param greeks: which greeks to constrain.
    :param target: optional dict of desired post-hedge greek values.
    :param use_scaling: row-scale the system before optimizing (default
        ``True``).
    :param bounds: position limits. ``None`` = unbounded; a scalar ``L`` means
        ``-L <= w_j <= L`` for every instrument; or a sequence of
        ``(lo, hi)`` pairs (``None`` entries allowed) per instrument.
    :param gross_limit: optional cap on ``sum_j |w_j|``.
    :param costs: optional per-instrument cost coefficients ``c_j``.
    :param cost_weight: multiplier on the cost term (default ``0.0``).
    :param method: ``scipy.optimize.minimize`` method (default ``'SLSQP'``).
    :param maxiter: maximum optimizer iterations.
    :param ftol: objective tolerance.
    :param x0: optional starting point (default: zeros).
    :returns: :class:`HedgeResult` (``method='optimize'``).
    """
    from scipy.optimize import minimize

    (instruments, greeks, A, b, rhs, pf_greeks,
     target_map, unused) = _build_system(portfolio, instruments, greeks, target)

    A_work, rhs_work = _scale(A, rhs) if use_scaling else (A, rhs)
    n = A.shape[1]

    if bounds is None:
        box = [(-np.inf, np.inf)] * n
    elif np.isscalar(bounds):
        limit = abs(float(bounds))
        box = [(-limit, limit)] * n
    else:
        box = list(bounds)
        if len(box) != n:
            raise InvalidInputError(
                "bounds must have one entry per instrument (%d), got %d"
                % (n, len(box)))

    cost = np.zeros(n) if costs is None else np.asarray(costs, dtype=float)
    if cost.shape != (n,):
        raise InvalidInputError(
            "costs must have one entry per instrument (%d)" % (n,))

    def objective(w):
        err = A_work @ w - rhs_work
        return float(err @ err) + cost_weight * float(cost @ np.abs(w))

    def gradient(w):
        err = A_work @ w - rhs_work
        return 2.0 * (A_work.T @ err) + cost_weight * cost * np.sign(w)

    constraints = []
    if gross_limit is not None:
        gl = float(gross_limit)
        constraints.append({
            "type": "ineq",
            "fun": lambda w: gl - float(np.sum(np.abs(w))),
            "jac": lambda w: -np.sign(w),
        })

    start = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float)
    res = minimize(objective, start, jac=gradient, method=method,
                   bounds=box, constraints=constraints or (),
                   options={"maxiter": int(maxiter), "ftol": float(ftol)})

    info = {
        "solver": method,
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(getattr(res, "nit", 0)),
        "objective": float(res.fun),
    }
    return _finalize(instruments, greeks, A, b, rhs, pf_greeks, target_map,
                     unused, res.x, "optimize", use_scaling, info=info)
