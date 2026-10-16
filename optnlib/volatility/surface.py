# -*- coding: utf-8 -*-

"""
optnlib.volatility.surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Turn a tidy option chain into an implied-volatility surface and query it.

* :func:`compute_iv_surface` computes an implied vol per contract (using
  :func:`optnlib.implied_volatility`) and returns a tidy DataFrame.
* :class:`VolSurface` wraps that DataFrame, builds an interpolator over
  ``(log-moneyness, days-to-expiry)`` and exposes point queries such as
  ``surface.iv("2026-03-20", 195.0)``.
"""

from datetime import date, datetime, timedelta

import numpy as np

from ..exceptions import InvalidInputError

__all__ = ["VolSurface", "compute_iv_surface"]

_SURFACE_COLUMNS = ["expiry", "dte", "t", "strike", "flag", "moneyness",
                    "log_moneyness", "mid", "iv"]

_DATE_TOL = 0.5  # days


def _to_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value.strip()[:10])
    raise InvalidInputError("cannot interpret %r as a date" % (value,))


def compute_iv_surface(chain, spot=None, r=None, q=None, price_col="mid",
                       style="european", method="bs2002",
                       reference_date=None, errors="drop", min_price=1e-8):
    """Compute implied volatilities for every contract in ``chain``.

    :param chain: tidy chain DataFrame (see
        :mod:`optnlib.volatility.market`) or any DataFrame with columns
        ``expiry``/``dte``, ``strike``, ``flag`` and ``price_col``.
    :param spot, r, q: market inputs; default to ``chain.attrs`` when omitted.
    :param price_col: column holding the option price (default ``'mid'``).
    :param errors: ``'drop'`` (default) removes contracts whose price admits no
        implied vol; ``'nan'`` keeps them with ``iv = NaN``.
    :returns: tidy :class:`pandas.DataFrame` with ``iv`` plus surface
        coordinates; market inputs are copied into ``frame.attrs``.
    """
    import pandas as pd

    from ..implied_volatility import implied_volatility

    if not isinstance(chain, pd.DataFrame):
        raise InvalidInputError("chain must be a pandas DataFrame")

    attrs = getattr(chain, "attrs", {}) or {}
    if spot is None:
        spot = attrs.get("spot")
    if r is None:
        r = attrs.get("r", 0.0)
    if q is None:
        q = attrs.get("q", 0.0)
    if reference_date is None:
        reference_date = attrs.get("reference_date")
    if spot is None:
        raise InvalidInputError("spot is required (pass spot= or set attrs)")
    spot = float(spot)
    r = float(r)
    q = float(q)

    for col in ("expiry", "strike", "flag", price_col):
        if col not in chain.columns:
            raise InvalidInputError("chain is missing required column %r" % (col,))

    if errors not in ("drop", "nan"):
        raise InvalidInputError("errors must be 'drop' or 'nan'")

    df = chain.copy()
    if "dte" in df.columns:
        dte = pd.to_numeric(df["dte"], errors="coerce").to_numpy(dtype=float)
    else:
        if reference_date is None:
            raise InvalidInputError(
                "chain has no 'dte' column and no reference_date")
        ref = _to_date(reference_date)
        dte = np.array([(_to_date(e) - ref).days for e in df["expiry"]],
                       dtype=float)

    t = dte / 365.0
    strikes = pd.to_numeric(df["strike"], errors="coerce").to_numpy(dtype=float)
    prices = pd.to_numeric(df[price_col], errors="coerce").to_numpy(dtype=float)
    flags = df["flag"].to_numpy()
    expiries = df["expiry"].to_numpy()

    ivs = np.full(len(df), np.nan)
    for i in range(len(df)):
        if not np.isfinite(prices[i]) or prices[i] <= min_price:
            continue
        if not np.isfinite(strikes[i]) or strikes[i] <= 0:
            continue
        if not np.isfinite(t[i]) or t[i] <= 0:
            continue
        try:
            ivs[i] = implied_volatility(
                prices[i], flags[i], spot, strikes[i], t[i], r, q=q,
                style=style, method=method)
        except InvalidInputError:
            ivs[i] = np.nan

    df["dte"] = dte
    df["t"] = t
    df["mid"] = prices
    df["iv"] = ivs
    df["moneyness"] = strikes / spot
    with np.errstate(divide="ignore", invalid="ignore"):
        df["log_moneyness"] = np.log(strikes / spot)

    bad = ~np.isfinite(ivs)
    if errors == "drop":
        df = df[~bad].reset_index(drop=True)
    if df.empty:
        raise InvalidInputError(
            "no contract produced a valid implied volatility; check the "
            "market inputs (spot/rate/yield) and prices")

    ref_attr = _to_date(reference_date) if reference_date is not None \
        else attrs.get("reference_date")
    keep = [c for c in _SURFACE_COLUMNS if c in df.columns]
    rest = [c for c in df.columns if c not in keep]
    df = df[keep + rest]
    df.attrs.update(spot=spot, r=r, q=q, reference_date=ref_attr,
                    ticker=attrs.get("ticker"))
    return df


class VolSurface:
    """An implied-volatility surface with point/slice queries.

    Built from a tidy DataFrame produced by :func:`compute_iv_surface` (or any
    DataFrame with ``expiry``/``dte``, ``strike``/``moneyness`` and ``iv``).

    Coordinates are ``(log-moneyness, days-to-expiry)``.

    :param frame: tidy implied-vol DataFrame.
    :param spot: underlying spot (defaults to ``frame.attrs['spot']``).
    :param rate, div_yield: market inputs (default to ``frame.attrs``).
    :param reference_date: "today" used to convert an expiry date to DTE;
        defaults to ``frame.attrs['reference_date']`` or the earliest expiry.
    """

    def __init__(self, frame, spot=None, rate=None, div_yield=None,
                 reference_date=None):
        import pandas as pd

        if not isinstance(frame, pd.DataFrame):
            raise InvalidInputError("frame must be a pandas DataFrame")
        attrs = getattr(frame, "attrs", {}) or {}

        self.spot = float(spot if spot is not None else attrs.get("spot"))
        self.rate = float(rate if rate is not None else attrs.get("r", 0.0))
        self.div_yield = float(
            div_yield if div_yield is not None else attrs.get("q", 0.0))
        ref = reference_date if reference_date is not None \
            else attrs.get("reference_date")
        self.ticker = attrs.get("ticker")

        df = frame.copy()
        if "dte" not in df.columns:
            if ref is None:
                ref = _to_date(df["expiry"].min())
            ref_d = _to_date(ref)
            df["dte"] = [(_to_date(e) - ref_d).days for e in df["expiry"]]
        if "moneyness" not in df.columns:
            df["moneyness"] = df["strike"] / self.spot
        if "log_moneyness" not in df.columns:
            df["log_moneyness"] = np.log(df["moneyness"])
        df = df[np.isfinite(df["iv"].to_numpy(dtype=float))].reset_index(drop=True)
        self.raw_frame = df.copy()

        if "flag" in df.columns:
            is_call = df["flag"].astype(str).str.lower().str.startswith("c")
            otm = np.where(is_call, df["strike"] >= self.spot,
                           df["strike"] < self.spot)
            df = (df.assign(_prefer=~otm)
                    .sort_values("_prefer")
                    .drop_duplicates(subset=["expiry", "strike"], keep="first")
                    .drop(columns="_prefer")
                    .sort_values(["dte", "strike"])
                    .reset_index(drop=True))

        self.frame = df
        self._m = df["log_moneyness"].to_numpy(dtype=float)
        self._dte = df["dte"].to_numpy(dtype=float)
        self._iv = df["iv"].to_numpy(dtype=float)

        if ref is None:
            ref = _to_date(df["expiry"].min())
        self.reference_date = _to_date(ref)

        self._min_x, self._max_x = float(self._m.min()), float(self._m.max())
        self._min_dte = float(self._dte.min())
        self._max_dte = float(self._dte.max())
        self.expiries = sorted({_to_date(e) for e in df["expiry"]})
        self.strikes = np.unique(df["strike"].to_numpy(dtype=float))
        self._interp = self._build_interpolators()

    # -- interpolation ---------------------------------------------------
    def _build_interpolators(self):
        pts = np.column_stack([self._m, self._dte])
        vals = self._iv
        uniq_dte = np.unique(self._dte)
        uniq_m = np.unique(self._m)
        if len(uniq_dte) >= 2 and len(uniq_m) >= 3 and len(vals) >= 4:
            from scipy.interpolate import (
                CloughTocher2DInterpolator,
                LinearNDInterpolator,
                NearestNDInterpolator,
            )
            try:
                cubic = CloughTocher2DInterpolator(pts, vals)
            except Exception:  # pragma: no cover - depends on point layout
                cubic = None
            return {
                "linear": LinearNDInterpolator(pts, vals),
                "nearest": NearestNDInterpolator(pts, vals),
                "cubic": cubic,
            }
        return None  # fall back to 1-D slice interpolation

    def _check_bounds(self, x, y):
        span_dte = self._max_dte - self._min_dte
        span_x = self._max_x - self._min_x
        if not (self._min_dte - _DATE_TOL <= y <= self._max_dte + _DATE_TOL):
            raise InvalidInputError(
                "expiry is outside the surface: dte=%.2f not in [%.2f, %.2f]"
                % (y, self._min_dte, self._max_dte))
        if not (self._min_x - 1e-9 - span_x <= x <= self._max_x + 1e-9 + span_x):
            raise InvalidInputError(
                "strike is far outside the surface: log-moneyness=%.4f not in "
                "[%.4f, %.4f]" % (x, self._min_x, self._max_x))

    def _query_many(self, xs, ys, method):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if method not in ("linear", "nearest", "cubic"):
            raise InvalidInputError(
                "method must be 'linear', 'cubic' or 'nearest'")
        if self._interp is not None:
            interp = self._interp.get(method)
            if interp is None and method == "cubic":
                raise InvalidInputError(
                    "cubic interpolation is unavailable for this surface; "
                    "use method='linear' or 'nearest'")
            pts = np.column_stack([xs.ravel(), ys.ravel()])
            out = np.asarray(interp(pts), dtype=float).ravel()
            return out.reshape(xs.shape)
        return self._query_1d(xs, ys, method)

    def _query_1d(self, xs, ys, method):
        xs = np.atleast_1d(np.asarray(xs, dtype=float))
        ys = np.atleast_1d(np.asarray(ys, dtype=float))
        out = np.full(xs.shape, np.nan)
        uniq = np.unique(self._dte)
        for j, y in enumerate(ys):
            if np.min(np.abs(uniq - y)) > _DATE_TOL:
                continue
            slice_dte = uniq[np.argmin(np.abs(uniq - y))]
            mask = np.abs(self._dte - slice_dte) <= _DATE_TOL
            order = np.argsort(self._m[mask])
            mx = self._m[mask][order]
            mv = self._iv[mask][order]
            out[j] = np.interp(xs[j], mx, mv)
        return out

    def iv(self, expiry, strike, method="linear"):
        """Interpolated IV for a calendar **expiry** date and **strike**.

        A point that lands exactly on a computed contract returns its exact IV.
        Points outside the available data raise :class:`InvalidInputError`.
        """
        d = _to_date(expiry)
        dte = (d - self.reference_date).days
        if dte < 0:
            raise InvalidInputError(
                "expiry %s precedes the reference date %s"
                % (d.isoformat(), self.reference_date.isoformat()))
        strike = float(strike)
        if strike <= 0:
            raise InvalidInputError("strike must be positive")
        x = float(np.log(strike / self.spot))
        self._check_bounds(x, dte)
        val = float(self._query_many(x, dte, method))
        if not np.isfinite(val):
            raise InvalidInputError(
                "point (expiry=%s, strike=%.4f) lies outside the convex hull "
                "of available contracts" % (d.isoformat(), strike))
        return val

    def iv_moneyness(self, moneyness, dte, method="linear"):
        """Interpolated IV for a **moneyness** multiple and **DTE** in days."""
        moneyness = float(moneyness)
        if moneyness <= 0:
            raise InvalidInputError("moneyness must be positive")
        x = float(np.log(moneyness))
        y = float(dte)
        self._check_bounds(x, y)
        val = float(self._query_many(x, y, method))
        if not np.isfinite(val):
            raise InvalidInputError(
                "point (moneyness=%.4f, dte=%.1f) lies outside the convex hull "
                "of available contracts" % (moneyness, y))
        return val

    # -- slices ----------------------------------------------------------
    def _nearest_expiry(self, expiry):
        if expiry is None:
            return self.expiries[0]
        d = _to_date(expiry)
        if d in self.expiries:
            return d
        dtes = np.array([(e - self.reference_date).days for e in self.expiries])
        y = (d - self.reference_date).days
        return self.expiries[int(np.argmin(np.abs(dtes - y)))]

    def smile(self, expiry=None):
        """IV versus strike for a single (nearest) expiry slice."""
        exp = self._nearest_expiry(expiry)
        mask = np.array([_to_date(e) == exp for e in self.frame["expiry"]])
        sub = self.frame[mask].sort_values("strike")
        if sub.empty:
            raise InvalidInputError("no data at expiry %s" % exp.isoformat())
        return sub[["expiry", "dte", "strike", "moneyness", "flag", "iv"]] \
            .reset_index(drop=True)

    def term_structure(self, strike=None):
        """IV versus expiry for a single (nearest) strike."""
        if strike is None:
            strike = float(self.frame["strike"].median())
        strike = float(strike)
        k = self.strikes[int(np.argmin(np.abs(self.strikes - strike)))]
        sub = self.frame[self.frame["strike"] == k].sort_values("dte")
        if sub.empty:
            raise InvalidInputError("no data at strike %s" % k)
        return sub[["expiry", "dte", "strike", "moneyness", "flag", "iv"]] \
            .reset_index(drop=True)

    def to_grid(self, n_dte=40, n_moneyness=60, method="linear",
                moneyness_range=None, dte_range=None):
        """Evaluate the surface on a regular grid (for plotting).

        :returns: dict with 1-D ``moneyness``, ``dte``, ``dates`` and the 2-D
            ``iv`` array (shape ``(n_dte, n_moneyness)``).
        """
        if moneyness_range is None:
            moneyness_range = (float(np.exp(self._min_x)),
                               float(np.exp(self._max_x)))
        if dte_range is None:
            dte_range = (self._min_dte, self._max_dte)
        m1d = np.linspace(moneyness_range[0], moneyness_range[1],
                          int(n_moneyness))
        t1d = np.linspace(dte_range[0], dte_range[1], int(n_dte))
        MM, TT = np.meshgrid(np.log(m1d), t1d)
        IV = self._query_many(MM, TT, method)
        dates = [self.reference_date + timedelta(days=float(d)) for d in t1d]
        return {"moneyness": m1d, "dte": t1d, "dates": dates, "iv": IV}

    def __repr__(self):
        return ("VolSurface(ticker=%r, %d contracts, %d expiries, "
                "%d strikes)" % (self.ticker, len(self.frame),
                                 len(self.expiries), len(self.strikes)))
