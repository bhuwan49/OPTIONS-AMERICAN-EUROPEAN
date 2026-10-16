# -*- coding: utf-8 -*-

"""
optnlib.volatility.plot
~~~~~~~~~~~~~~~~~~~~~~~~

Plotting for volatility smiles and surfaces.

Two backends are supported and selected with ``backend=``:

* ``'matplotlib'`` -- static figures (``.png``/``.pdf`` ...), uses the
  ``mpl_toolkits.mplot3d`` toolkit for the surface;
* ``'plotly'`` -- interactive HTML surfaces/smiles.

The 3-D time axis can show **full calendar dates** (``time_axis='date'``,
default, ISO ``YYYY-MM-DD``), days-to-expiry (``'dte'``) or decimal years
(``'years'``).
"""

from datetime import timedelta

import numpy as np

from ..exceptions import InvalidInputError

__all__ = ["plot_smile", "plot_surface"]

_TIME_AXES = ("date", "dte", "years")


def _require(module, extra):
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on env
        raise InvalidInputError(
            "%s is required for plotting; install it with "
            "'pip install %s' or 'pip install optnlib[%s]'"
            % (module, module.split(".")[0], extra)) from exc


def _as_surface(surface):
    from .surface import VolSurface

    if isinstance(surface, VolSurface):
        return surface
    return VolSurface(surface)


def _pick_expiries(surface, expiries, n_expiries):
    if expiries is not None:
        return [surface._nearest_expiry(e) for e in expiries]
    exps = surface.expiries
    if n_expiries is None or n_expiries >= len(exps):
        return exps
    idx = np.linspace(0, len(exps) - 1, int(n_expiries)).round().astype(int)
    return [exps[i] for i in sorted(set(idx.tolist()))]


def _time_coord(surface, t1d, time_axis):
    """Return the numeric/date coordinate for the time axis."""
    if time_axis == "date":
        return [surface.reference_date + timedelta(days=float(d)) for d in t1d]
    if time_axis == "years":
        return np.asarray(t1d, dtype=float) / 365.0
    return np.asarray(t1d, dtype=float)


# -- smile -------------------------------------------------------------------


def plot_smile(surface, expiries=None, n_expiries=6, backend="matplotlib",
               ax=None, save=None, show=False, title=None, moneyness=False):
    """Plot the volatility smile (IV vs strike) for several expiries.

    :param surface: :class:`VolSurface` or a tidy IV DataFrame.
    :param expiries: explicit expiry dates/strings to show; defaults to a
        spread of ``n_expiries`` slices.
    :param backend: ``'matplotlib'`` or ``'plotly'``.
    :param ax: existing matplotlib ``Axes`` to draw on.
    :param moneyness: plot ``K/S`` on the x-axis instead of the strike.
    :param save: path to write the figure to.
    :param show: display the figure interactively.
    :returns: the matplotlib ``Figure`` or the plotly ``go.Figure``.
    """
    surface = _as_surface(surface)
    chosen = _pick_expiries(surface, expiries, n_expiries)
    spot = surface.spot
    ticker = surface.ticker or "underlying"
    title = title or "Volatility smile%s" % (
        " - %s" % ticker if surface.ticker else "")
    xlabel = "Moneyness (K/S)" if moneyness else "Strike"

    series = []
    for exp in chosen:
        sub = surface.smile(exp)
        x = sub["moneyness"].to_numpy() if moneyness else \
            sub["strike"].to_numpy()
        series.append((exp, x, sub["iv"].to_numpy()))

    if backend == "plotly":
        return _smile_plotly(series, xlabel, title, spot, moneyness,
                             save, show)
    if backend != "matplotlib":
        raise InvalidInputError("backend must be 'matplotlib' or 'plotly'")
    return _smile_mpl(series, xlabel, title, spot, moneyness, ax, save, show)


def _smile_mpl(series, xlabel, title, spot, moneyness, ax, save, show):
    plt = _require("matplotlib.pyplot", "viz")

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(9, 5.5))
    else:
        fig = ax.figure

    colors = plt.cm.viridis(np.linspace(0, 0.9, max(len(series), 1)))
    for (exp, x, iv), color in zip(series, colors):
        ax.plot(x, iv, marker="o", markersize=3, linewidth=1.8, color=color,
                label=exp.isoformat())
    atm = (1.0 if moneyness else spot)
    ax.axvline(atm, color="grey", linestyle="--", linewidth=1, alpha=0.7,
               label="ATM")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Implied volatility")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Expiry", fontsize=8, ncol=2)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _smile_plotly(series, xlabel, title, spot, moneyness, save, show):
    go = _require("plotly.graph_objects", "viz")
    fig = go.Figure()
    for exp, x, iv in series:
        fig.add_trace(go.Scatter(x=x, y=iv, mode="lines+markers",
                                 name=exp.isoformat()))
    atm = 1.0 if moneyness else spot
    fig.add_vline(x=atm, line_dash="dash", line_color="grey",
                  annotation_text="ATM")
    fig.update_layout(title=title, xaxis_title=xlabel,
                      yaxis_title="Implied volatility",
                      template="plotly_white", hovermode="x unified")
    _plotly_save(fig, save)
    if show:
        fig.show()
    return fig


# -- surface -----------------------------------------------------------------


def plot_surface(surface, backend="matplotlib", time_axis="date",
                 x_axis="moneyness", n_dte=40, n_moneyness=60, method="linear",
                 cmap="viridis", save=None, show=False, title=None,
                 elev=25, azim=-60):
    """Plot the 3-D implied-volatility surface.

    :param surface: :class:`VolSurface` or a tidy IV DataFrame.
    :param time_axis: ``'date'`` (default, full ISO dates), ``'dte'`` or
        ``'years'``.
    :param x_axis: ``'moneyness'`` (default) or ``'strike'``.
    :param n_dte, n_moneyness: grid resolution.
    :param method: interpolation method (``'linear'``/``'cubic'``/``'nearest'``).
    :returns: the matplotlib ``Figure`` or the plotly ``go.Figure``.
    """
    surface = _as_surface(surface)
    if time_axis not in _TIME_AXES:
        raise InvalidInputError(
            "time_axis must be one of %s" % (_TIME_AXES,))
    if x_axis not in ("moneyness", "strike"):
        raise InvalidInputError("x_axis must be 'moneyness' or 'strike'")

    grid = surface.to_grid(n_dte=n_dte, n_moneyness=n_moneyness,
                           method=method)
    x1d = grid["moneyness"] * surface.spot if x_axis == "strike" \
        else grid["moneyness"]
    tcoord = _time_coord(surface, grid["dte"], time_axis)
    IV = grid["iv"]
    xlabel = "Strike" if x_axis == "strike" else "Moneyness (K/S)"
    ylabel = {"date": "Expiry", "dte": "Days to expiry",
              "years": "Time to expiry (years)"}[time_axis]
    ticker = surface.ticker or "underlying"
    title = title or "Volatility surface%s" % (
        " - %s" % ticker if surface.ticker else "")

    if backend == "plotly":
        return _surface_plotly(x1d, tcoord, IV, xlabel, ylabel, title,
                               time_axis, cmap, save, show)
    if backend != "matplotlib":
        raise InvalidInputError("backend must be 'matplotlib' or 'plotly'")
    return _surface_mpl(x1d, tcoord, IV, xlabel, ylabel, title, time_axis,
                        cmap, elev, azim, save, show)


def _surface_mpl(x1d, tcoord, IV, xlabel, ylabel, title, time_axis, cmap,
                 elev, azim, save, show):
    plt = _require("matplotlib.pyplot", "viz")

    if time_axis == "date":
        import matplotlib.dates as mdates

        y = mdates.date2num(tcoord)
    else:
        y = np.asarray(tcoord, dtype=float)

    XX, YY = np.meshgrid(x1d, y)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(XX, YY, IV, cmap=cmap, linewidth=0,
                           antialiased=True, alpha=0.95)
    if time_axis == "date":
        import matplotlib.dates as mdates

        ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.set_yticks(y[:: max(1, len(y) // 8)])
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_zlabel("Implied volatility", labelpad=8)
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, shrink=0.6, aspect=12, label="Implied volatility")
    for lbl in ax.get_yticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
    try:
        fig.subplots_adjust(left=0.02, right=0.88, top=0.94, bottom=0.06)
    except Exception:  # pragma: no cover - best effort
        pass
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _surface_plotly(x1d, tcoord, IV, xlabel, ylabel, title, time_axis, cmap,
                    save, show):
    go = _require("plotly.graph_objects", "viz")
    if time_axis == "date":
        import pandas as pd

        y = pd.to_datetime([d.isoformat() for d in tcoord])
    else:
        y = np.asarray(tcoord, dtype=float)
    fig = go.Figure(data=[go.Surface(
        x=np.asarray(x1d), y=y, z=IV, colorscale=cmap,
        colorbar=dict(title="IV"))])
    fig.update_layout(
        title=title, template="plotly_white",
        scene=dict(xaxis_title=xlabel, yaxis_title=ylabel,
                   zaxis_title="Implied volatility"),
    )
    if time_axis == "date":
        fig.update_layout(scene=dict(yaxis=dict(tickformat="%Y-%m-%d")))
    _plotly_save(fig, save)
    if show:
        fig.show()
    return fig


def _plotly_save(fig, save):
    if not save:
        return
    if save.endswith(".html"):
        fig.write_html(save)
    else:
        try:
            fig.write_image(save)
        except Exception as exc:  # pragma: no cover - needs kaleido
            print("Could not write image (%s); saving HTML instead. "
                  "Install 'kaleido' for static export." % exc)
            fig.write_html(save + ".html")
