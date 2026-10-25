# -*- coding: utf-8 -*-

"""
optnlib.stochvol.plot
~~~~~~~~~~~~~~~~~~~~~

Optional matplotlib plotting helpers for simulated stochastic-volatility paths.
"""

import numpy as np

from ..exceptions import InvalidInputError
from .base import positive_int

__all__ = ["plot_paths", "plot_variance_paths"]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise InvalidInputError(
            "matplotlib is required for plotting; install it with "
            "'pip install matplotlib' or 'pip install optnlib[viz]'"
        ) from exc
    return plt


def _time_axis(length, t):
    if t is None:
        return np.arange(length), "Step"
    t = float(t)
    if t < 0.0:
        raise InvalidInputError("time horizon t must be >= 0; got %r" % (t,))
    return np.linspace(0.0, t, length), "Time (years)"


def plot_paths(paths, K=None, t=None, n_paths=100, ax=None, save=None,
               show=False, title=None, figsize=(10, 5)):
    """Plot simulated spot paths with matplotlib."""
    plt = _require_matplotlib()
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[0] < 1 or paths.shape[1] < 1:
        raise InvalidInputError(
            "paths must be a 2-D array with shape (n_sims, n_steps + 1)")
    n_paths = positive_int(n_paths, "n_paths")
    x, xlabel = _time_axis(paths.shape[1], t)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for row in paths[:min(n_paths, paths.shape[0])]:
        ax.plot(x, row, linewidth=0.8, alpha=0.4, color="#1f77b4")
    if K is not None:
        ax.axhline(float(K), color="black", linestyle="--", linewidth=1.2,
                   label="Strike")
        ax.legend(frameon=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Underlying price")
    ax.set_title(title or "Stochastic-volatility simulated paths")
    ax.grid(True, alpha=0.3)
    if own:
        fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_variance_paths(variances, t=None, n_paths=100, ax=None, save=None,
                        show=False, title=None, figsize=(10, 5)):
    """Plot simulated variance paths with matplotlib."""
    plt = _require_matplotlib()
    variances = np.asarray(variances, dtype=float)
    if variances.ndim != 2 or variances.shape[0] < 1 or variances.shape[1] < 1:
        raise InvalidInputError(
            "variances must be a 2-D array with shape (n_sims, n_steps + 1)")
    n_paths = positive_int(n_paths, "n_paths")
    x, xlabel = _time_axis(variances.shape[1], t)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for row in variances[:min(n_paths, variances.shape[0])]:
        ax.plot(x, row, linewidth=0.8, alpha=0.4, color="#d62728")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Variance")
    ax.set_title(title or "Stochastic-volatility variance paths")
    ax.grid(True, alpha=0.3)
    if own:
        fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
