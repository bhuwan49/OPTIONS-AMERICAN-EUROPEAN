# -*- coding: utf-8 -*-

"""
optnlib.volatility
~~~~~~~~~~~~~~~~~~~

Volatility smile and 3-D volatility surface from market option chains.

Typical use::

    from optnlib import volatility as v

    # fetch + compute + plot in one call; keeps the queryable surface
    surf = v.volatility_surface("AAPL", strike_range=(0.8, 1.2),
                                dte_range=(7, 180), backend="plotly",
                                save="AAPL_surface.html")

    surf.iv("2026-03-20", 195.0)      # IV at an exact date + strike
    surf.iv_moneyness(1.05, dte=45)   # IV at a moneyness + days-to-expiry
    surf.smile("2026-03-20")          # IV vs strike at one expiry

    # smile only
    v.volatility_smile("SPY", backend="matplotlib", save="spy_smile.png")

    # data-only (no plotting)
    chain = v.fetch_chain("NVDA")
    frame = v.compute_iv_surface(chain)

    # offline demo (no network)
    surf = v.volatility_surface(source="model", backend="plotly",
                                save="model_surface.html")

The market-data and plotting dependencies are optional and imported lazily;
install them with ``pip install optnlib[vol]``.
"""

from ..exceptions import InvalidInputError
from .market import export_chain, fetch_chain, model_chain
from .plot import plot_smile, plot_surface
from .surface import VolSurface, compute_iv_surface

__all__ = [
    "VolSurface",
    "compute_iv_surface",
    "fetch_chain",
    "export_chain",
    "model_chain",
    "plot_smile",
    "plot_surface",
    "load_surface",
    "volatility_smile",
    "volatility_surface",
]


def load_surface(ticker=None, chain=None, source="fetch", strike_range=(0.8, 1.2),
                 dte_range=(7, 365), moneyness=True, r=None, q=None,
                 max_expiries=None, reference_date=None, price_col="mid",
                 style="european", method="bs2002", model_kwargs=None):
    """Acquire a chain (or accept one) and return a :class:`VolSurface`.

    :param ticker: symbol used when ``chain`` is not supplied.
    :param chain: an existing tidy chain DataFrame; skips the fetch.
    :param source: ``'fetch'`` (live Yahoo via yfinance) or ``'model'``
        (offline synthetic chain).
    :param model_kwargs: extra keyword arguments for :func:`model_chain`.
    """
    if source not in ("fetch", "model"):
        raise InvalidInputError("source must be 'fetch' or 'model'")

    if chain is None:
        if source == "model":
            chain = model_chain(**(model_kwargs or {}))
        else:
            if not ticker:
                raise InvalidInputError("a ticker is required to fetch a chain")
            chain = fetch_chain(ticker, strike_range=strike_range,
                                dte_range=dte_range, moneyness=moneyness,
                                r=r, q=q, max_expiries=max_expiries,
                                reference_date=reference_date)

    frame = compute_iv_surface(chain, r=r, q=q, price_col=price_col,
                               style=style, method=method,
                               reference_date=reference_date)
    return VolSurface(frame)


def volatility_smile(ticker=None, chain=None, source="fetch",
                     strike_range=(0.8, 1.2), dte_range=(7, 365),
                     moneyness=True, r=None, q=None, max_expiries=None,
                     reference_date=None, plot=True, backend="matplotlib",
                     save=None, show=False, n_expiries=6, title=None,
                     model_kwargs=None):
    """Build a :class:`VolSurface` and (by default) plot its smile.

    Returns the :class:`VolSurface` so it can be queried directly; the figure,
    when drawn, is available as ``surface.figure``.
    """
    surf = load_surface(ticker=ticker, chain=chain, source=source,
                        strike_range=strike_range, dte_range=dte_range,
                        moneyness=moneyness, r=r, q=q,
                        max_expiries=max_expiries,
                        reference_date=reference_date,
                        model_kwargs=model_kwargs)
    surf.figure = None
    if plot:
        surf.figure = plot_smile(surf, n_expiries=n_expiries, backend=backend,
                                 save=save, show=show, title=title,
                                 moneyness=moneyness)
    return surf


def volatility_surface(ticker=None, chain=None, source="fetch",
                       strike_range=(0.8, 1.2), dte_range=(7, 365),
                       moneyness=True, r=None, q=None, max_expiries=None,
                       reference_date=None, plot=True, backend="matplotlib",
                       time_axis="date", x_axis="moneyness", n_dte=40,
                       n_moneyness=60, interp="linear", save=None, show=False,
                       title=None, model_kwargs=None):
    """Build a :class:`VolSurface` and (by default) plot the 3-D surface.

    Returns the :class:`VolSurface` so it can be queried directly; the figure,
    when drawn, is available as ``surface.figure``.
    """
    surf = load_surface(ticker=ticker, chain=chain, source=source,
                        strike_range=strike_range, dte_range=dte_range,
                        moneyness=moneyness, r=r, q=q,
                        max_expiries=max_expiries,
                        reference_date=reference_date,
                        model_kwargs=model_kwargs)
    surf.figure = None
    if plot:
        surf.figure = plot_surface(surf, backend=backend, time_axis=time_axis,
                                   x_axis=x_axis, n_dte=n_dte,
                                   n_moneyness=n_moneyness, method=interp,
                                   save=save, show=show, title=title)
    return surf
