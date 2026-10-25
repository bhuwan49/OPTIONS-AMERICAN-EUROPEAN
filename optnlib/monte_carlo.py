# -*- coding: utf-8 -*-

"""
optnlib.monte_carlo
~~~~~~~~~~~~~~~~~~~

Monte Carlo pricing for European options under geometric Brownian motion.

The simulation uses the exact GBM discretization

``S_{t+dt} = S_t exp((r - q - sigma**2 / 2) dt + sigma sqrt(dt) Z)``

with ``Z`` standard normal. ``n_steps`` controls the number of time steps per
path and ``n_sims`` the number of simulated paths. Each path starts at ``S``.

Plotting is optional and requires matplotlib; it is imported lazily so pricing
works with numpy alone.
"""

import math

import numpy as np

from .core import intrinsic
from .exceptions import InvalidInputError
from .helpers import validate_option

__all__ = ["simulate", "price", "plot_paths"]

_DEFAULT_STEPS = 252
_DEFAULT_SIMS = 10_000


def _positive_int(value, name):
    if isinstance(value, bool):
        raise InvalidInputError(
            "%s must be a positive integer; got %r" % (name, value))
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        raise InvalidInputError(
            "%s must be a positive integer; got %r" % (name, value))
    if as_int != value or as_int < 1:
        raise InvalidInputError(
            "%s must be a positive integer; got %r" % (name, value))
    return as_int


def _validate_market(S, t, r, sigma, q):
    S = float(S)
    t = float(t)
    r = float(r)
    sigma = float(sigma)
    q = float(q)
    if S <= 0.0:
        raise InvalidInputError(
            "spot S must be strictly positive; got %r" % (S,))
    if t < 0.0:
        raise InvalidInputError(
            "time to expiry t must be >= 0; got %r" % (t,))
    if sigma < 0.0:
        raise InvalidInputError(
            "volatility sigma must be >= 0; got %r" % (sigma,))
    return S, t, r, sigma, q


def simulate(S, t, r, sigma, q=0.0, n_steps=_DEFAULT_STEPS,
             n_sims=_DEFAULT_SIMS, seed=None):
    """Simulate GBM paths for the underlying.

    :param S: initial underlying price (> 0).
    :param t: time horizon in years (>= 0).
    :param r: annualized continuously compounded risk-free rate.
    :param sigma: annualized volatility (>= 0).
    :param q: annualized continuous dividend yield (default 0).
    :param n_steps: number of time steps per path (default 252).
    :param n_sims: number of simulated paths (default 10,000).
    :param seed: optional seed for ``numpy.random.default_rng``.
    :returns: array of shape ``(n_sims, n_steps + 1)``; column 0 is ``S``.
    """
    S, t, r, sigma, q = _validate_market(S, t, r, sigma, q)
    n_steps = _positive_int(n_steps, "n_steps")
    n_sims = _positive_int(n_sims, "n_sims")

    paths = np.empty((n_sims, n_steps + 1), dtype=float)
    paths[:, 0] = S

    if t == 0.0:
        return paths

    dt = t / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    diffusion = sigma * math.sqrt(dt)
    rng = np.random.default_rng(seed)
    increments = drift + diffusion * rng.standard_normal((n_sims, n_steps))
    paths[:, 1:] = S * np.exp(np.cumsum(increments, axis=1))
    return paths


def price(flag, S, K, t, r, sigma, q=0.0, n_steps=_DEFAULT_STEPS,
          n_sims=_DEFAULT_SIMS, seed=None):
    """Monte Carlo price of a European call or put.

    :param flag: ``'c'`` or ``'p'``.
    :param S: underlying spot price.
    :param K: strike price (> 0).
    :param t: time to expiry in years.
    :param r: annualized continuously compounded risk-free rate.
    :param sigma: annualized volatility.
    :param q: annualized continuous dividend yield (default 0).
    :param n_steps: number of time steps per path (default 252).
    :param n_sims: number of simulated paths (default 10,000).
    :param seed: optional seed for reproducible results.
    :returns: discounted mean terminal payoff.
    """
    flag = validate_option(flag, S, K, t, r, sigma, q=q)
    n_steps = _positive_int(n_steps, "n_steps")
    n_sims = _positive_int(n_sims, "n_sims")

    S = float(S)
    K = float(K)
    t = float(t)
    r = float(r)

    if t == 0.0:
        return intrinsic(flag, S, K)

    terminal = simulate(
        S, t, r, sigma, q=q, n_steps=n_steps, n_sims=n_sims, seed=seed
    )[:, -1]
    if flag == "c":
        payoff = np.maximum(terminal - K, 0.0)
    else:
        payoff = np.maximum(K - terminal, 0.0)
    return float(math.exp(-r * t) * np.mean(payoff))


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on env
        raise InvalidInputError(
            "matplotlib is required for plotting; install it with "
            "'pip install matplotlib' or 'pip install optnlib[viz]'"
        ) from exc
    return plt


def plot_paths(paths, K=None, t=None, n_paths=100, ax=None, save=None,
               show=False, title=None, figsize=(10, 5)):
    """Plot simulated paths with matplotlib.

    :param paths: array returned by :func:`simulate`, shape
        ``(n_sims, n_steps + 1)``.
    :param K: optional strike price; drawn as a horizontal dashed line.
    :param t: optional time horizon; when given, the x-axis is in years.
    :param n_paths: maximum number of paths to draw (default 100).
    :param ax: existing matplotlib ``Axes`` to draw on.
    :param save: path to write the figure to.
    :param show: display the figure interactively.
    :returns: the matplotlib ``Figure``.
    """
    plt = _require_matplotlib()

    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[0] < 1 or paths.shape[1] < 1:
        raise InvalidInputError(
            "paths must be a 2-D array with shape (n_sims, n_steps + 1)")

    n_paths = _positive_int(n_paths, "n_paths")
    n_show = min(n_paths, paths.shape[0])

    if t is None:
        x = np.arange(paths.shape[1])
        xlabel = "Step"
    else:
        t = float(t)
        if t < 0.0:
            raise InvalidInputError("time horizon t must be >= 0; got %r" % (t,))
        x = np.linspace(0.0, t, paths.shape[1])
        xlabel = "Time (years)"

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for i in range(n_show):
        ax.plot(x, paths[i], linewidth=0.8, alpha=0.45, color="#1f77b4")
    if K is not None:
        ax.axhline(float(K), color="black", linestyle="--", linewidth=1.2,
                   label="Strike")
        ax.legend(frameon=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Underlying price")
    ax.set_title(title or "Monte Carlo simulated paths")
    ax.grid(True, alpha=0.3)
    if own:
        fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
