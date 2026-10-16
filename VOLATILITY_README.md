# optnlib.volatility — volatility smiles & surfaces

`optnlib.volatility` turns an option chain into an **implied-volatility smile**
and a **3-D volatility surface**, and lets you query the interpolated IV at any
expiry date and strike.

The pipeline has four stages, each of which you can use on its own:

| stage | function | what it does |
|---|---|---|
| 1. data | `fetch_chain(...)` / `model_chain(...)` | live (Yahoo) or synthetic tidy chain |
| 2. IV   | `compute_iv_surface(...)` | one implied vol per contract |
| 3. model| `VolSurface(...)` | interpolate + query (`iv`, `smile`, ...) |
| 4. plot | `plot_smile(...)` / `plot_surface(...)` | matplotlib or interactive plotly |

The convenience wrappers `volatility_smile(...)` and `volatility_surface(...)`
run all four stages in one call and return the queryable `VolSurface`.

> Data and plotting dependencies are **optional**: `pandas`, `yfinance`,
> `openpyxl`, `matplotlib` and `plotly`. Install them with
> `pip install "optnlib[vol]"` (or the narrower `[market]` / `[viz]`). Base
> `optnlib` still only requires `numpy` + `scipy`; all imports are lazy.

---

## Quick start

### Live data (needs internet + `yfinance`)

```python
from optnlib import volatility as v

surf = v.volatility_surface("AAPL", strike_range=(0.8, 1.2),
                            dte_range=(7, 180), backend="plotly",
                            save="AAPL_surface.html", show=False)
print(surf)                       # VolSurface(ticker='AAPL', <n> contracts, ...)

surf.iv("2026-03-20", 195.0)      # IV at an exact expiry date and strike
surf.iv_moneyness(1.05, dte=45)   # IV at a moneyness multiple and days-to-expiry
```

### Offline (no network, no data files)

```python
from optnlib import volatility as v

surf = v.volatility_surface(source="model", backend="plotly",
                            save="model_surface.html", show=False)
```

`source="model"` builds a synthetic chain with a smooth, known IV surface and
prices it with optnlib's own Black-Scholes engine — handy for demos and tests.

### Command line

```bash
# fetch + plot both smile and surface (prompts for the ticker if omitted)
python -m optnlib.volatility

# explicit ticker, interactive HTML, custom window
python -m optnlib.volatility --ticker SPY --mode surface \
    --backend plotly --save-dir ./out --show

# exact IV lookup for a date + strike, then exit
python -m optnlib.volatility --ticker NVDA --query 2026-03-20 195

# offline demo
python -m optnlib.volatility --source model --mode both
```

The installed console script `optnlib-volatility` is equivalent to
`python -m optnlib.volatility`.

| flag | default | meaning |
|---|---|---|
| `--ticker`, `-t` | prompt / `AAPL` | underlying symbol |
| `--source` | `fetch` | `fetch` (Yahoo) or `model` (synthetic) |
| `--mode` | `both` | `smile`, `surface` or `both` |
| `--strike-range LO HI` | `0.8 1.2` | moneyness window (or absolute strikes with `--absolute-strikes`) |
| `--dte-range MIN MAX` | `7 365` | days-to-expiry window |
| `--r`, `--q` | `0.045`, `0.0` | rate and dividend yield |
| `--max-expiries` | — | cap the number of expiries |
| `--reference-date` | today | ISO date treated as "today" |
| `--backend` | `matplotlib` | `matplotlib` (png/pdf) or `plotly` (html) |
| `--time-axis` | `date` | `date` (ISO), `dte` or `years` |
| `--n-expiries` | `8` | smile slices to draw |
| `--save-dir` | `.` | where figures are written |
| `--query DATE STRIKE` | — | print the IV for that date and strike and exit |
| `--no-plot` | — | compute only, print a summary |
| `--show` | — | display interactively |

---

## The chain (stage 1)

Both providers return a **tidy** `pandas.DataFrame`, one row per contract:

```
expiry, dte, strike, flag, bid, ask, last, mid, volume, open_interest, iv_yahoo
```

plus market metadata in `frame.attrs` (`ticker`, `spot`, `r`, `q`,
`reference_date`).

### `fetch_chain(ticker, strike_range=None, dte_range=None, moneyness=True, r=None, q=None, max_expiries=None, reference_date=None)`

Pulls live chains from Yahoo Finance. `strike_range=(lo, hi)` is interpreted as
**moneyness** multiples of spot when `moneyness=True` (default); `r=None` tries
the 13-week T-bill (`^IRX`) and `q=None` the ticker's dividend yield, falling
back to `0.045` / `0`.

```python
chain = v.fetch_chain("AAPL", strike_range=(0.8, 1.2), dte_range=(7, 180),
                      max_expiries=6)
```

### `model_chain(...)`

Builds a synthetic chain with a parametric IV surface

```
iv(m, t) = base_vol + skew·(m − 1) + curvature·(m − 1)² + term_slope·(√t − √0.25)
```

where `m = K/S` is moneyness and `t` is time to expiry in years. Because the
prices come from optnlib's own Black-Scholes engine, the IV recovered in stage 2
matches the model IV to root-finder accuracy.

```python
chain = v.model_chain(spot=100.0, r=0.045, dtes=(30, 60, 90, 180, 365),
                      n_strikes=11, moneyness_range=(0.7, 1.3),
                      base_vol=0.20, skew=-0.35, curvature=0.9,
                      term_slope=0.05, reference_date="2026-01-05")
```

### `export_chain(ticker, output_filename=None, dte_range=None, r=None, q=None, reference_date=None)`

Writes every expiry's calls and puts to an Excel workbook (one `{date}_Calls` /
`{date}_Puts` sheet pair per expiry), with timezones stripped so Excel can store
the datetimes.

---

## Implied vols (stage 2)

`compute_iv_surface(chain, spot=None, r=None, q=None, price_col='mid', style='european', method='bs2002', reference_date=None, errors='drop')`
computes an implied vol per contract with `optnlib.implied_volatility`, using
`t = dte/365`, and appends `iv`, `t`, `moneyness` and `log_moneyness` columns.
`spot`, `r`, `q` and `reference_date` default to the chain's `attrs`.

- `errors='drop'` (default) removes contracts whose price admits no implied vol;
- `errors='nan'` keeps them with `iv = NaN`.

```python
frame = v.compute_iv_surface(chain)
# columns: expiry, dte, t, strike, flag, moneyness, log_moneyness, mid, iv, ...
```

---

## Querying the surface (stage 3)

`VolSurface(frame, spot=None, rate=None, div_yield=None, reference_date=None)`
builds a `(log-moneyness, days-to-expiry)` interpolation over the data. When the
same `(expiry, strike)` appears as both a call and a put, the **out-of-the-money**
contract is preferred, so each expiry slice has one IV per strike.

Coordinates used for interpolation:

- **x** = `log_moneyness = ln(K / S)`,
- **y** = `dte` (calendar days to expiry).

### Point queries

```python
surf.iv(expiry, strike, method="linear")     # calendar date (str/date/datetime) + strike
surf.iv_moneyness(moneyness, dte, method="linear")   # K/S multiple + days-to-expiry
```

A query that lands exactly on a computed contract returns that contract's exact
IV. Points outside the available data raise `optnlib.exceptions.InvalidInputError`
with a clear message (out-of-range DTE, out-of-range strike, expiry before the
reference date, etc.).

```python
surf.iv("2026-02-04", 90.0)     # 0.2338   (30 DTE)
surf.iv("2026-02-04", 100.0)    # 0.1893   (at the money)
surf.iv("2026-02-04", 110.0)    # 0.1640

surf.iv_moneyness(1.05, dte=45)    # 0.1774
surf.iv_moneyness(0.95, dte=180)   # 0.2302
```

> `method` selects the interpolator: `"linear"` (default, Delaunay linear ND),
> `"cubic"` (Clough–Tocher) or `"nearest"`. For a surface with a single expiry
> (a pure smile) the interpolation falls back to 1-D over `log_moneyness`.

### Slices

```python
surf.smile("2026-02-04")   # -> DataFrame: expiry, dte, strike, moneyness, flag, iv
surf.term_structure(100.0) # -> DataFrame: expiry, dte, strike, moneyness, flag, iv
```

```python
# term structure at strike 100 (model surface)
#    expiry   dte       iv
# 2026-02-04  30  0.189335
# 2026-03-06  60  0.195272
# 2026-04-05  90  0.199828
# 2026-07-04 180  0.210112
# 2027-01-05 365  0.225000
```

### Grid

```python
grid = surf.to_grid(n_dte=40, n_moneyness=60, method="linear")
# dict: moneyness (1-D), dte (1-D days), dates (1-D datetime), iv (2-D [dte, moneyness])
```

---

## Plotting (stage 4)

```python
v.plot_smile(surface, expiries=None, n_expiries=6, backend="matplotlib",
             save=None, show=False, moneyness=False, title=None)
v.plot_surface(surface, backend="matplotlib", time_axis="date",
               x_axis="moneyness", n_dte=40, n_moneyness=60, method="linear",
               cmap="viridis", save=None, show=False, title=None)
```

Both accept either a `VolSurface` or a tidy IV DataFrame and return the figure.

- `backend="matplotlib"` → static image (write `.png`/`.pdf`/… via `save=`);
- `backend="plotly"` → interactive figure (write `.html`; static `.png` needs
  `kaleido`).

**Time axis** — `time_axis="date"` (default) labels the 3-D time axis with full
ISO calendar dates (`YYYY-MM-DD`); `"dte"` uses days-to-expiry and `"years"`
decimal years. `x_axis="moneyness"` (default) or `"strike"`.

```python
v.plot_smile(surf, backend="matplotlib", save="smile.png", moneyness=True)
v.plot_surface(surf, backend="plotly", time_axis="date",
               x_axis="strike", save="surface.html")
```

---

## One-call wrappers

`volatility_smile(...)` and `volatility_surface(...)` acquire data, compute IVs
and plot in a single call, returning the `VolSurface` so you can keep querying
it. The figure is stored on `surface.figure` (or `None` when `plot=False`).

```python
surf = v.volatility_surface("AAPL", strike_range=(0.8, 1.2),
                            dte_range=(7, 180), backend="plotly",
                            save="AAPL_surface.html", plot=True)
print(surf.iv("2026-03-20", 195.0))
```

Or build without plotting:

```python
surf = v.load_surface("AAPL", dte_range=(7, 180))
```

---

## API summary

| function | returns |
|---|---|
| `fetch_chain(...)` | tidy chain DataFrame (live) |
| `export_chain(...)` | writes Excel, returns filename |
| `model_chain(...)` | tidy chain DataFrame (synthetic) |
| `compute_iv_surface(...)` | tidy IV DataFrame |
| `VolSurface(...)` | queryable surface object |
| `surf.iv(expiry, strike)` / `surf.iv_moneyness(m, dte)` | interpolated IV (float) |
| `surf.smile(expiry)` / `surf.term_structure(strike)` | slice DataFrames |
| `surf.to_grid(...)` | dict of grid arrays |
| `plot_smile(...)` / `plot_surface(...)` | figure |
| `load_surface(...)` | `VolSurface` |
| `volatility_smile(...)` / `volatility_surface(...)` | `VolSurface` (+ `surface.figure`) |

---

## Notes & caveats

- The surface uses whichever contracts the chain provides; wide bid/ask spreads,
  stale quotes and near-zero prices can inject noise into the IVs. Filter the
  chain (e.g. by volume/open interest) before computing the surface if needed.
- yfinance data is delayed and occasionally incomplete; `source="model"` is the
  deterministic, network-free alternative for demos and tests.
- Out-of-the-money options are preferred when a strike has both a call and a put,
  which is the usual convention for a clean smile.
- Querying far outside the traded grid raises rather than extrapolating.
