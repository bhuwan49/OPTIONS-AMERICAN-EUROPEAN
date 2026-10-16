# -*- coding: utf-8 -*-

"""
optnlib.volatility.plot
~~~~~~~~~~~~~~~~~~~~~~~~

Plotting for volatility smiles and surfaces.

Two backends are supported and selected with ``backend=``:

* ``'plotly'`` -- interactive figures (recommended, default when available);
* ``'matplotlib'`` -- static figures (``.png``/``.pdf`` ...).

``backend=None`` (the default) picks plotly when it is importable and falls
back to matplotlib otherwise.

The surface is drawn with **time on the x-axis and strike on the y-axis**. The
time axis is labelled in **decimal years** to expiry; with the plotly backend
the hover box additionally shows the calendar date of each point. A red marker
is drawn at the at-the-money point (``strike == spot``) at the middle tenor.
"""

from datetime import timedelta

import numpy as np

from ..exceptions import InvalidInputError

__all__ = ["plot_smile", "plot_surface"]

_BACKENDS = ("matplotlib", "plotly")


def _require(module, extra):
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on env
        raise InvalidInputError(
            "%s is required for plotting; install it with "
            "'pip install %s' or 'pip install optnlib[%s]'"
            % (module, module.split(".")[0], extra)) from exc


def _resolve_backend(backend):
    """Pick a backend; ``None``/``'auto'`` prefers plotly then matplotlib."""
    import importlib.util

    if backend in (None, "auto"):
        if importlib.util.find_spec("plotly") is not None:
            return "plotly"
        return "matplotlib"
    if backend not in _BACKENDS:
        raise InvalidInputError(
            "backend must be 'matplotlib', 'plotly' or None; got %r"
            % (backend,))
    return backend


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


def _years(dte1d):
    """Decimal years to expiry from a 1-D dte array."""
    return np.asarray(dte1d, dtype=float) / 365.0


def _dates(surface, dte1d):
    """Calendar dates for a 1-D dte array."""
    return [surface.reference_date + timedelta(days=float(d)) for d in dte1d]


def _smile_series(surface, expiries):
    series = []
    for exp in expiries:
        sub = surface.smile(exp)
        dte = int(round(float(sub["dte"].iloc[0])))
        series.append((exp, dte, sub["strike"].to_numpy(),
                       sub["iv"].to_numpy()))
    return series


def _atm_point(surface, grid, method):
    """Mid-tenor at-the-money point ``(tenor_years, strike, iv)`` or ``None``."""
    dte1d = np.asarray(grid["dte"], dtype=float)
    if dte1d.size == 0:
        return None
    dte_mid = float(dte1d[len(dte1d) // 2])
    strike = float(surface.spot)
    try:
        iv = float(surface.iv_moneyness(1.0, dte_mid, method=method))
    except InvalidInputError:
        m1d = np.asarray(grid["moneyness"], dtype=float)
        j = int(np.argmin(np.abs(m1d - 1.0)))
        i = len(dte1d) // 2
        iv = float(np.asarray(grid["iv"], dtype=float)[i, j])
    if not np.isfinite(iv):
        return None
    return dte_mid / 365.0, strike, iv


# -- smile -------------------------------------------------------------------


def plot_smile(surface, expiries=None, n_expiries=5, backend=None, ax=None,
               save=None, show=False, title=None, layout="grid", ncols=None,
               figsize=None, width=1000, height=700, markers=True):
    """Plot the volatility smile (IV vs strike price).

    :param surface: :class:`VolSurface` or a tidy IV DataFrame.
    :param expiries: explicit expiry dates/strings to show; defaults to a
        spread of ``n_expiries`` slices.
    :param layout: ``'grid'`` (default) draws one panel per expiry;
        ``'overlay'`` draws all expiries on a single axes.
    :param backend: ``None`` (auto), ``'matplotlib'`` or ``'plotly'``.
    :param ax: existing matplotlib ``Axes`` to draw on (``overlay`` only).
    :param ncols: number of grid columns (``layout='grid'``).
    :param save: path to write the figure to.
    :param show: display the figure interactively.
    :returns: the matplotlib ``Figure`` or the plotly ``go.Figure``.
    """
    surface = _as_surface(surface)
    backend = _resolve_backend(backend)
    if layout not in ("grid", "overlay"):
        raise InvalidInputError("layout must be 'grid' or 'overlay'")

    chosen = _pick_expiries(surface, expiries, n_expiries)
    series = _smile_series(surface, chosen)
    spot = surface.spot
    ticker = surface.ticker
    title = title or "Volatility smile%s" % (
        " - %s" % ticker if ticker else "")
    xlabel = "Strike Price"

    if backend == "plotly":
        if layout == "grid":
            fig = _smile_plotly_grid(series, xlabel, title, spot, markers,
                                     width, height)
        else:
            fig = _smile_plotly_overlay(series, xlabel, title, spot, markers,
                                        width, height)
        _plotly_save(fig, save)
        if show:
            fig.show()
        return fig

    if layout == "grid":
        fig = _smile_mpl_grid(series, xlabel, title, spot, markers, ncols,
                              figsize)
    else:
        fig = _smile_mpl_overlay(series, xlabel, title, spot, markers, ax,
                                 figsize)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        _require("matplotlib.pyplot", "viz").show()
    return fig


def _smile_mpl_overlay(series, xlabel, title, atm, markers, ax, figsize):
    plt = _require("matplotlib.pyplot", "viz")

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=figsize or (11, 6))
    else:
        fig = ax.figure

    colors = plt.cm.viridis(np.linspace(0, 0.9, max(len(series), 1)))
    for (exp, dte, x, iv), color in zip(series, colors):
        ax.plot(x, iv, marker="o" if markers else None, markersize=3,
                linewidth=1.8, color=color,
                label="%s (%d dte)" % (exp.isoformat(), dte))
    ax.axvline(atm, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Implied volatility")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Expiry", fontsize=8, loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False)
    if own:
        fig.tight_layout()
    return fig


def _smile_mpl_grid(series, xlabel, title, atm, markers, ncols, figsize):
    plt = _require("matplotlib.pyplot", "viz")

    n = len(series)
    ncols = int(ncols) if ncols else min(3, n)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or
                             (3.6 * ncols, 3.0 * nrows), sharex=True,
                             squeeze=False)
    axes = axes.ravel()
    for ax, (exp, dte, x, iv) in zip(axes, series):
        ax.plot(x, iv, marker="o" if markers else None, markersize=3,
                linewidth=1.6, color="#1f77b4")
        ax.axvline(atm, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title("%s  (%d dte)" % (exp.isoformat(), dte), fontsize=9)
        ax.grid(True, alpha=0.3)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[:n]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("IV")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _smile_plotly_overlay(series, xlabel, title, atm, markers, width, height):
    go = _require("plotly.graph_objects", "viz")
    fig = go.Figure()
    for exp, dte, x, iv in series:
        fig.add_trace(go.Scatter(x=x, y=iv,
                                 mode="lines+markers" if markers else "lines",
                                 name="%s (%d dte)" % (exp.isoformat(), dte)))
    fig.add_vline(x=atm, line_dash="dash", line_color="grey",
                  annotation_text="ATM")
    fig.update_layout(title=title, xaxis_title=xlabel,
                      yaxis_title="Implied volatility",
                      template="plotly_white", hovermode="x unified",
                      width=width, height=height,
                      legend=dict(title="Expiry"))
    return fig


def _smile_plotly_grid(series, xlabel, title, atm, markers, width, height):
    go = _require("plotly.graph_objects", "viz")
    subplots = _require("plotly.subplots", "viz").make_subplots
    n = len(series)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    titles = ["%s (%d dte)" % (e.isoformat(), d) for e, d, _, _ in series]
    fig = subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                   shared_xaxes=True, horizontal_spacing=0.06,
                   vertical_spacing=0.10)
    for i, (exp, dte, x, iv) in enumerate(series):
        r, c = divmod(i, ncols)
        fig.add_trace(go.Scatter(x=x, y=iv,
                                 mode="lines+markers" if markers else "lines",
                                 line=dict(color="#1f77b4"),
                                 showlegend=False), row=r + 1, col=c + 1)
        fig.add_vline(x=atm, line_dash="dash", line_color="grey",
                      row=r + 1, col=c + 1)
    fig.update_layout(title=title, template="plotly_white",
                      width=width, height=height)
    for r in range(1, nrows + 1):
        for c in range(1, ncols + 1):
            if (r - 1) * ncols + (c - 1) < n:
                fig.update_xaxes(title_text=xlabel, row=r, col=c)
                fig.update_yaxes(title_text="Implied vol", row=r, col=c)
    return fig


# -- surface -----------------------------------------------------------------


def plot_surface(surface, backend=None, n_dte=40, n_moneyness=60,
                 method="linear", cmap="viridis", save=None, show=False,
                 title=None, elev=22, azim=-135, show_contours=False,
                 width=950, height=760, figsize=(11, 8), decimate=None,
                 atm_marker=True):
    """Plot the 3-D implied-volatility surface (time on x, strike on y).

    The time axis is labelled in **decimal years**; the plotly hover box also
    shows the calendar date of each point. If ``atm_marker`` is set, a red dot
    marks the at-the-money point (``strike == spot``) at the middle tenor.

    :param surface: :class:`VolSurface` or a tidy IV DataFrame.
    :param backend: ``None`` (auto), ``'matplotlib'`` or ``'plotly'``.
    :param show_contours: draw z-axis contour projections (plotly) / floor
        contours (matplotlib).
    :param atm_marker: mark the at-the-money point with a red dot.
    :param decimate: decimate the grid for the matplotlib backend (int stride);
        ``None`` picks a sensible default.
    :returns: the matplotlib ``Figure`` or the plotly ``go.Figure``.
    """
    surface = _as_surface(surface)
    backend = _resolve_backend(backend)

    grid = surface.to_grid(n_dte=n_dte, n_moneyness=n_moneyness, method=method)
    strike1d = np.asarray(grid["moneyness"], dtype=float) * surface.spot
    time1d = _years(grid["dte"])
    IV = grid["iv"]
    atm = _atm_point(surface, grid, method) if atm_marker else None

    xlabel = "Time to expiry (years)"
    ylabel = "Strike Price"
    ticker = surface.ticker
    title = title or "Volatility surface%s" % (
        " - %s" % ticker if ticker else "")

    if backend == "plotly":
        dates = [d.isoformat() for d in _dates(surface, grid["dte"])]
        fig = _surface_plotly(time1d, strike1d, IV, xlabel, ylabel, title, cmap,
                              show_contours, width, height, atm, dates)
        _plotly_save(fig, save)
        if show:
            fig.show()
        return fig

    fig = _surface_mpl(time1d, strike1d, IV, xlabel, ylabel, title, cmap, elev,
                       azim, show_contours, figsize, decimate, atm)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        _require("matplotlib.pyplot", "viz").show()
    return fig


def _decimate(arr, stride):
    return arr[::stride] if arr.ndim == 1 else arr[::stride, ::stride]


def _surface_mpl(time1d, strike1d, IV, xlabel, ylabel, title, cmap, elev, azim,
                 show_contours, figsize, decimate, atm):
    plt = _require("matplotlib.pyplot", "viz")

    time_num = np.asarray(time1d, dtype=float)
    strike_num = np.asarray(strike1d, dtype=float)

    MM, TT = np.meshgrid(strike_num, time_num)   # (n_dte, n_moneyness)
    ZZ = np.asarray(IV, dtype=float)
    XX, YY = TT, MM

    stride = max(1, len(time_num) // 45) if decimate is None else int(decimate)
    if stride > 1:
        XX, YY, ZZ = _decimate(XX, stride), _decimate(YY, stride), \
            _decimate(ZZ, stride)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(XX, YY, ZZ, cmap=cmap, linewidth=0,
                           antialiased=True, alpha=0.95)
    if show_contours:
        ax.contour(XX, YY, ZZ, zdir="z", offset=float(np.nanmin(ZZ)),
                   cmap=cmap, linewidths=0.8)
    if atm is not None:
        ax.scatter([atm[0]], [atm[1]], [atm[2]], color="red", s=70,
                   depthshade=False, label="ATM", zorder=10)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_zlabel("Implied volatility", labelpad=8)
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect((len(time_num), len(strike_num),
                           0.6 * max(len(time_num), len(strike_num))))
    except Exception:  # pragma: no cover - best effort
        pass
    ax.set_facecolor("white")
    if atm is not None:
        ax.legend(loc="upper left", frameon=False)
    cb = fig.colorbar(surf, shrink=0.6, aspect=14, pad=0.08)
    cb.set_label("Implied volatility")
    return fig


def _surface_plotly(time1d, strike1d, IV, xlabel, ylabel, title, cmap,
                    show_contours, width, height, atm, dates):
    go = _require("plotly.graph_objects", "viz")

    x = np.asarray(time1d, dtype=float)
    y = np.asarray(strike1d, dtype=float)
    # plotly expects z (and customdata) with shape (len(y), len(x))
    z = np.asarray(IV, dtype=float).T
    customdata = np.tile(np.asarray(dates, dtype=object), (len(y), 1))

    trace_kwargs = {}
    if show_contours:
        trace_kwargs["contours"] = dict(
            z=dict(show=True, usecolormap=True, project=dict(z=True)))

    surface = go.Surface(
        x=x, y=y, z=z, customdata=customdata, colorscale=cmap, **trace_kwargs,
        hovertemplate=("Date: %{customdata}<br>Tenor: %{x:.3f} yr<br>"
                       "Strike Price: %{y:.2f}<br>IV: %{z:.4f}<extra></extra>"),
        colorbar=dict(title="Implied Volatility", thickness=16, len=0.75))

    fig = go.Figure(data=[surface])
    if atm is not None:
        fig.add_trace(go.Scatter3d(
            x=[atm[0]], y=[atm[1]], z=[atm[2]], mode="markers+text",
            marker=dict(size=6, color="red"), text=["ATM"],
            textposition="top center", name="ATM",
            hovertemplate=("ATM<br>Strike Price: %{y:.2f}<br>"
                           "IV: %{z:.4f}<extra></extra>")))
    fig.update_layout(
        title=title, template="plotly_white", autosize=False,
        width=width, height=height,
        margin=dict(l=60, r=30, b=50, t=70),
        scene=dict(
            xaxis_title=xlabel, yaxis_title=ylabel,
            zaxis_title="Implied volatility",
            yaxis=dict(tickformat=",.0f"),
            aspectratio=dict(x=1.4, y=1.4, z=0.7),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
    )
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
