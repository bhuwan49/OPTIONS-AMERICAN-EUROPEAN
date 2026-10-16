# -*- coding: utf-8 -*-

"""
optnlib.volatility.cli
~~~~~~~~~~~~~~~~~~~~~~~

Command-line interface::

    python -m optnlib.volatility --ticker AAPL
    python -m optnlib.volatility --ticker SPY --mode surface --backend plotly \
        --save-dir ./out --show
    python -m optnlib.volatility --ticker NVDA --query 2026-03-20 195
    python -m optnlib.volatility --source model --mode both
"""

import argparse
import os
import sys

from ..exceptions import InvalidInputError


def _build_parser():
    p = argparse.ArgumentParser(
        prog="optnlib.volatility",
        description="Build and plot an implied-volatility smile / surface.")
    p.add_argument("--ticker", "-t", default=None,
                   help="underlying symbol (prompted for if omitted)")
    p.add_argument("--source", choices=("fetch", "model"), default="fetch",
                   help="data source: live Yahoo or offline synthetic")
    p.add_argument("--mode", choices=("smile", "surface", "both"),
                   default="both", help="what to draw")
    p.add_argument("--strike-range", nargs=2, type=float, metavar=("LO", "HI"),
                   default=None,
                   help="strike window as moneyness multiples (default 0.8 1.2)")
    p.add_argument("--absolute-strikes", action="store_true",
                   help="interpret --strike-range as absolute strikes")
    p.add_argument("--dte-range", nargs=2, type=int, metavar=("MIN", "MAX"),
                   default=None, help="days-to-expiry window (default 7 365)")
    p.add_argument("--r", type=float, default=0.045, help="risk-free rate")
    p.add_argument("--q", type=float, default=0.0, help="dividend yield")
    p.add_argument("--max-expiries", type=int, default=None,
                   help="cap the number of expiries fetched")
    p.add_argument("--reference-date", default=None,
                   help="ISO date treated as 'today' (default: real today)")
    p.add_argument("--backend", choices=("matplotlib", "plotly"),
                   default="matplotlib")
    p.add_argument("--time-axis", choices=("date", "dte", "years"),
                   default="date")
    p.add_argument("--n-expiries", type=int, default=8,
                   help="smile slices to draw")
    p.add_argument("--save-dir", default=".", help="output directory")
    p.add_argument("--show", action="store_true", help="display interactively")
    p.add_argument("--no-plot", action="store_true",
                   help="compute only, do not draw")
    p.add_argument("--query", nargs=2, metavar=("DATE", "STRIKE"),
                   help="print the IV at this ISO date and strike and exit")
    return p


def _prompt(text, default):
    try:
        raw = input("%s [%s]: " % (text, default)).strip()
    except EOFError:  # pragma: no cover - non-interactive
        raw = ""
    return raw or default


def main(argv=None):
    args = _build_parser().parse_args(argv)

    ticker = args.ticker
    if ticker is None and args.source == "fetch":
        ticker = _prompt("Ticker", "AAPL").upper()

    strike_range = tuple(args.strike_range) if args.strike_range else (0.8, 1.2)
    dte_range = tuple(args.dte_range) if args.dte_range else (7, 365)

    from . import load_surface

    model_kwargs = None
    if args.source == "model" and args.reference_date:
        model_kwargs = {"reference_date": args.reference_date}

    try:
        surf = load_surface(
            ticker=ticker, source=args.source, strike_range=strike_range,
            dte_range=dte_range, moneyness=not args.absolute_strikes,
            r=args.r, q=args.q, max_expiries=args.max_expiries,
            reference_date=args.reference_date, model_kwargs=model_kwargs)
    except InvalidInputError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    if args.query:
        date_str, strike_str = args.query
        try:
            iv = surf.iv(date_str, float(strike_str))
        except InvalidInputError as exc:
            print("error: %s" % exc, file=sys.stderr)
            return 2
        print("IV(%s, %s) = %.6f" % (date_str, strike_str, iv))
        return 0

    if args.no_plot:
        print(surf)
        return 0

    from .plot import plot_smile, plot_surface

    os.makedirs(args.save_dir, exist_ok=True)
    ext = "html" if args.backend == "plotly" else "png"
    tag = (ticker or "model").replace("/", "_")
    made = []

    if args.mode in ("smile", "both"):
        path = os.path.join(args.save_dir, "%s_smile.%s" % (tag, ext))
        plot_smile(surf, backend=args.backend, save=path, show=args.show,
                   n_expiries=args.n_expiries)
        made.append(path)
    if args.mode in ("surface", "both"):
        path = os.path.join(args.save_dir, "%s_surface.%s" % (tag, ext))
        plot_surface(surf, backend=args.backend, time_axis=args.time_axis,
                     save=path, show=args.show)
        made.append(path)

    for path in made:
        print("wrote %s" % path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
