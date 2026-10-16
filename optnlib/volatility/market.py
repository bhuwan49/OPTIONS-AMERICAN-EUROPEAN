# -*- coding: utf-8 -*-

"""
optnlib.volatility.market
~~~~~~~~~~~~~~~~~~~~~~~~~~

Market-data access for the volatility smile / surface tools.

Three providers are offered:

* :func:`fetch_chain` -- live option chains from Yahoo Finance via the optional
  ``yfinance`` package (lazy import);
* :func:`export_chain` -- write every expiry's calls/puts into an Excel
  workbook (one sheet pair per expiry), using ``pandas`` + ``openpyxl``;
* :func:`model_chain` -- an **offline** synthetic chain priced with optnlib's
  own Black-Scholes engine, for demos and tests that must not touch the network.

All providers return a *tidy* :class:`pandas.DataFrame` with one row per
contract and the columns::

    expiry, dte, strike, flag, bid, ask, last, mid, volume, open_interest,
    iv_yahoo

plus metadata in ``frame.attrs`` (``ticker``, ``spot``, ``r``, ``q``,
``reference_date``).
"""

from datetime import date, datetime, timedelta

import numpy as np

from ..exceptions import InvalidInputError

__all__ = ["fetch_chain", "export_chain", "model_chain", "CHAIN_COLUMNS"]

CHAIN_COLUMNS = [
    "expiry", "dte", "strike", "flag", "bid", "ask", "last", "mid",
    "volume", "open_interest", "iv_yahoo",
]

_DEFAULT_RATE = 0.045


def _require(module, extra):
    """Import an optional dependency or raise a helpful error."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on env
        raise InvalidInputError(
            "%s is required for this feature; install it with "
            "'pip install %s' or 'pip install optnlib[%s]'"
            % (module, module.split(".")[0], extra)
        ) from exc


def _to_date(value):
    """Coerce a date / datetime / ISO string to a :class:`datetime.date`."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value.strip()[:10])
    raise InvalidInputError("cannot interpret %r as a date" % (value,))


def _strike_mask(strikes, spot, strike_range, moneyness):
    """Boolean mask keeping strikes inside ``strike_range``.

    ``moneyness=True`` treats the bounds as multiples of ``spot`` (e.g.
    ``(0.8, 1.2)``); ``moneyness=False`` treats them as absolute strikes (e.g.
    ``(80, 120)``).
    """
    if strike_range is None:
        return np.ones(len(strikes), dtype=bool)
    lo, hi = strike_range
    if moneyness:
        lo, hi = lo * spot, hi * spot
    return (np.asarray(strikes) >= lo) & (np.asarray(strikes) <= hi)


def _tidy(frame, expiry, dte, spot, strike_range, moneyness):
    """Normalise one expiry's calls/puts into the tidy chain schema."""
    import pandas as pd

    if frame is None or len(frame) == 0:
        return None

    df = frame.copy()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    if strike_range is not None:
        df = df[_strike_mask(df["strike"].to_numpy(), spot, strike_range,
                             moneyness)]
    if df.empty:
        return None

    bid = pd.to_numeric(df.get("bid"), errors="coerce").fillna(0.0)
    ask = pd.to_numeric(df.get("ask"), errors="coerce").fillna(0.0)
    last = pd.to_numeric(df.get("lastPrice"), errors="coerce").fillna(0.0)
    mid = np.where((bid > 0) & (ask > 0), (bid + ask) / 2.0, last)

    out = pd.DataFrame({
        "expiry": expiry,
        "dte": int(dte),
        "strike": df["strike"].to_numpy(),
        "flag": df["flag"].to_numpy(),
        "bid": bid.to_numpy(),
        "ask": ask.to_numpy(),
        "last": last.to_numpy(),
        "mid": mid,
        "volume": pd.to_numeric(df.get("volume"), errors="coerce")
                    .fillna(0).to_numpy(),
        "open_interest": pd.to_numeric(df.get("openInterest"),
                                       errors="coerce").fillna(0).to_numpy(),
        "iv_yahoo": pd.to_numeric(df.get("impliedVolatility"),
                                  errors="coerce").to_numpy(),
    })
    return out


def _resolve_rate(ticker_obj, r):
    if r is not None:
        return float(r)
    try:
        import yfinance as yf

        irx = yf.Ticker("^IRX").fast_info["last_price"]
        if irx and np.isfinite(irx):
            return float(irx) / 100.0
    except Exception:  # pragma: no cover - network dependent
        pass
    return _DEFAULT_RATE


def _resolve_yield(ticker_obj, q):
    if q is not None:
        return float(q)
    try:
        info = ticker_obj.info
        qy = info.get("dividendYield")
        if qy:
            return float(qy)
    except Exception:  # pragma: no cover - network dependent
        pass
    return 0.0


def fetch_chain(ticker, strike_range=None, dte_range=None, moneyness=True,
                r=None, q=None, max_expiries=None, reference_date=None):
    """Fetch a tidy option chain for ``ticker`` from Yahoo Finance.

    :param ticker: Yahoo ticker symbol, e.g. ``'AAPL'``.
    :param strike_range: ``(low, high)`` strike filter. With
        ``moneyness=True`` (default) the bounds are **moneyness** multiples of
        spot, e.g. ``(0.8, 1.2)`` keeps 80%-120% of spot. With
        ``moneyness=False`` the bounds are **absolute strike prices**, e.g.
        ``(80, 120)``.
    :param dte_range: ``(min_days, max_days)`` to expiry (inclusive).
    :param moneyness: interpret ``strike_range`` as multiples of spot
        (``True``, default) or as absolute strikes (``False``).
    :param r: risk-free rate; ``None`` tries ``^IRX``, else ``0.045``.
    :param q: dividend yield; ``None`` tries the ticker's metadata, else ``0``.
    :param max_expiries: keep at most this many (nearest) expiries.
    :param reference_date: "today" used for DTE; defaults to the real today.
    :returns: tidy :class:`pandas.DataFrame` (see module docstring).
    """
    import pandas as pd

    yf = _require("yfinance", "market")

    ref = _to_date(reference_date) if reference_date is not None else date.today()
    ticker = ticker.strip().upper()
    tk = yf.Ticker(ticker)

    expiries = list(getattr(tk, "options", []) or [])
    if not expiries:
        raise InvalidInputError(
            "no option expiries found for ticker %r" % (ticker,))

    chosen = []
    for iso in expiries:
        d = _to_date(iso)
        dte = (d - ref).days
        if dte < 0:
            continue
        if dte_range is not None and not (dte_range[0] <= dte <= dte_range[1]):
            continue
        chosen.append((d, dte))
    chosen.sort(key=lambda x: x[1])
    if max_expiries is not None:
        chosen = chosen[:int(max_expiries)]
    if not chosen:
        raise InvalidInputError(
            "no expiries for %r fall inside dte_range=%r"
            % (ticker, dte_range))

    try:
        spot = float(tk.fast_info["last_price"])
    except Exception:  # pragma: no cover - network dependent
        spot = float(tk.info.get("regularMarketPrice"))
    if not (spot and np.isfinite(spot) and spot > 0):
        raise InvalidInputError("could not determine a spot price for %r"
                                % (ticker,))

    rate = _resolve_rate(tk, r)
    div_yield = _resolve_yield(tk, q)

    parts = []
    for d, dte in chosen:
        chain = tk.option_chain(d.isoformat())
        for side in ("calls", "puts"):
            sub = getattr(chain, side)
            sub = sub.copy()
            sub["flag"] = "c" if side == "calls" else "p"
            tidy = _tidy(sub, d, dte, spot, strike_range, moneyness)
            if tidy is not None:
                parts.append(tidy)

    if not parts:
        raise InvalidInputError("no contracts matched the requested filters")

    out = pd.concat(parts, ignore_index=True)
    out.attrs.update(ticker=ticker, spot=spot, r=rate, q=div_yield,
                     reference_date=ref)
    return out


def _clean_timezones(df):
    """Drop timezone info from datetime columns so Excel can store them."""
    import pandas as pd

    if df is None or df.empty:
        return df
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)
    return df


def export_chain(ticker, output_filename=None, dte_range=None, r=None, q=None,
                 reference_date=None):
    """Export every expiry's calls and puts to an Excel workbook.

    Writes one sheet pair per expiry (``{date}_Calls`` / ``{date}_Puts``) with
    timezones stripped so Excel can store the datetimes. Returns the output
    filename.
    """
    pd = _require("pandas", "market")
    _require("openpyxl", "market")
    yf = _require("yfinance", "market")

    ref = _to_date(reference_date) if reference_date is not None else date.today()
    ticker = ticker.strip().upper()
    tk = yf.Ticker(ticker)
    expiries = list(getattr(tk, "options", []) or [])
    if not expiries:
        raise InvalidInputError(
            "no option data found for ticker %r" % (ticker,))

    if output_filename is None:
        output_filename = "%s_options.xlsx" % ticker

    with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
        for iso in expiries:
            d = _to_date(iso)
            dte = (d - ref).days
            if dte_range is not None and not (dte_range[0] <= dte <= dte_range[1]):
                continue
            try:
                chain = tk.option_chain(d.isoformat())
                calls = _clean_timezones(chain.calls.copy())
                puts = _clean_timezones(chain.puts.copy())
                calls.to_excel(writer, sheet_name="%s_Calls" % iso, index=False)
                puts.to_excel(writer, sheet_name="%s_Puts" % iso, index=False)
            except Exception as exc:  # pragma: no cover - network dependent
                print("    [ERROR] %s: %s" % (iso, exc))
    return output_filename


def _model_iv(moneyness, dte, base_vol, skew, curvature, term_slope):
    """A smooth synthetic implied-vol surface (deterministic)."""
    m = np.asarray(moneyness, dtype=float)
    t = np.maximum(np.asarray(dte, dtype=float), 1.0) / 365.0
    iv = (base_vol
          + skew * (m - 1.0)
          + curvature * (m - 1.0) ** 2
          + term_slope * (np.sqrt(t) - np.sqrt(0.25)))
    return np.maximum(iv, 1e-4)


def model_chain(spot=100.0, r=_DEFAULT_RATE, q=0.0, dtes=(30, 60, 90, 180, 365),
                strikes=None, n_strikes=11, moneyness_range=(0.7, 1.3),
                base_vol=0.20, skew=-0.35, curvature=0.9, term_slope=0.05,
                spread=0.0, reference_date=None):
    """Build an **offline** synthetic option chain with a known IV surface.

    Prices are produced with optnlib's own Black-Scholes engine, so the
    implied volatility recovered from ``mid`` equals the model IV exactly (up
    to root-finder tolerance). Useful for demos and for tests that must not
    depend on the network.

    :param dtes: days-to-expiry for each slice.
    :param strikes: explicit strike list; otherwise ``n_strikes`` are spread
        across ``moneyness_range`` (multiples of ``spot``).
    :param base_vol, skew, curvature, term_slope: parameters of
        ``iv(m, t) = base + skew*(m-1) + curv*(m-1)^2 + term*(sqrt(t)-0.5)``.
    :param spread: absolute bid/ask half-spread; ``mid`` is unchanged.
    :returns: tidy :class:`pandas.DataFrame` (see module docstring).
    """
    import pandas as pd

    from ..european import price as european_price

    ref = _to_date(reference_date) if reference_date is not None else date.today()
    if strikes is None:
        strikes = np.round(np.linspace(moneyness_range[0], moneyness_range[1],
                                       int(n_strikes)) * float(spot), 2)
    strikes = np.asarray(strikes, dtype=float)

    rows = []
    for dte in dtes:
        expiry = ref + timedelta(days=int(dte))
        t = dte / 365.0
        for K in strikes:
            m = K / float(spot)
            iv = float(_model_iv(m, dte, base_vol, skew, curvature, term_slope))
            for flag in ("c", "p"):
                px = float(european_price(flag, spot, K, t, r, iv, q=q))
                rows.append({
                    "expiry": expiry,
                    "dte": int(dte),
                    "strike": float(K),
                    "flag": flag,
                    "bid": max(px - spread, 0.0),
                    "ask": px + spread,
                    "last": px,
                    "mid": px,
                    "volume": 0,
                    "open_interest": 0,
                    "iv_yahoo": iv,
                })

    out = pd.DataFrame(rows, columns=CHAIN_COLUMNS)
    out.attrs.update(ticker="MODEL", spot=float(spot), r=float(r),
                     q=float(q), reference_date=ref)
    return out
