# optnlib — implied volatility smiles & surfaces

`optnlib.volatility` is the implied-volatility toolkit of `optnlib`. It

- **fetches live option chains** from Yahoo Finance (via `yfinance`),
- **inverts each contract's price into an implied volatility** using `optnlib`'s
  own Black-Scholes / American engines,
- **builds a queryable volatility surface**, and
- **plots the smile and the 3-D surface** (interactive plotly by default, with a
  matplotlib fallback) and exports IV tables.

An **offline** synthetic chain (`source="model"`) is also available for demos and
tests that must not touch the network — see the last section.

This guide goes **chronologically from the basics** (one call) to every feature
in detail. All examples use **live tickers**; output values are not hard-coded
because they depend on the current market — run the snippets to see them.

> Optional dependencies are imported lazily. Install everything with
> `pip install "optnlib[vol]"` (`[market]` + `[viz]`); the base package only needs
> `numpy` + `scipy`.

---

## 0. Install & import

```bash
pip install "optnlib[vol]"
```

```python
from optnlib import volatility as v
```

`v` exposes: `fetch_chain`, `export_chain`, `model_chain`, `compute_iv_surface`,
`VolSurface`, `surface_table`, `plot_smile`, `plot_surface`, `load_surface`,
`volatility_smile`, `volatility_surface`.

Error handling uses `optnlib.exceptions.InvalidInputError`:

```python
from optnlib.exceptions import InvalidInputError
```

---

## 1. Quick start (one call)

The fastest path: fetch a live chain, compute IVs, build the surface and plot it
— all in one call.

```python
from optnlib import volatility as v

# plotly by default; time in decimal years on X, strike on Y
surf = v.volatility_surface("AAPL", strike_range=(0.8, 1.2), dte_range=(7, 365))
v.surface_table(surf)   # strike x tenor IV matrix (tenor columns are ISO dates)
```

---

## 2. The pipeline

`optnlib.volatility` is a four-stage pipeline. The one-call wrappers run all four;
you can also run each stage yourself for full control.

1. **Get a chain** — `fetch_chain(...)` (live) or `model_chain(...)` (offline).
   Returns a *tidy* `pandas.DataFrame`: one row per contract.
2. **Compute IVs** — `compute_iv_surface(chain, ...)` adds an `iv` column per
   contract (using `optnlib.implied_volatility`).
3. **Build the surface** — `VolSurface(frame, ...)` interpolates the IVs over
   `(log-moneyness, days-to-expiry)` and exposes point/slice queries.
4. **Query / plot** — `surf.iv(...)`, `surf.smile(...)`, `surface_table(...)`,
   `plot_smile(...)`, `plot_surface(...)`.

```python
from optnlib import volatility as v

chain = v.fetch_chain("AAPL", strike_range=(0.8, 1.2), dte_range=(7, 365))  # 1
frame = v.compute_iv_surface(chain)                                        # 2
surf  = v.VolSurface(frame)                                                # 3
print(surf.iv(frame["expiry"].iloc[0], frame["strike"].iloc[0]))           # 4
v.plot_surface(surf)
```

---

## 3. Getting a live chain — `fetch_chain`

```python
chain = v.fetch_chain(
    "AAPL",
    strike_range=(0.8, 1.2),   # see below: moneyness or absolute strikes
    dte_range=(7, 365),        # days-to-expiry window (None = all expiries)
    moneyness=True,            # interpret strike_range as multiples of spot
    r=None,                    # risk-free rate (None -> ^IRX, else 0.045)
    q=None,                    # dividend yield (None -> ticker metadata, else 0)
    max_expiries=None,         # keep at most this many nearest expiries
    reference_date=None,       # "today" for DTE (None -> the real today)
)
```

### 3.1 `strike_range` — moneyness vs absolute strikes

- **Moneyness (default).** With `moneyness=True`, the bounds are multiples of the
  spot price. `(0.8, 1.2)` keeps strikes between 80% and 120% of spot.
  ```python
  chain = v.fetch_chain("AAPL", strike_range=(0.8, 1.2))            # 80%-120% of spot
  ```
- **Absolute strikes.** With `moneyness=False`, the bounds are real strike prices.
  ```python
  chain = v.fetch_chain("AAPL", strike_range=(180, 240), moneyness=False)
  ```

> If `strike_range` is `None` (the `fetch_chain` default), **all** strikes are kept.

### 3.2 `dte_range` — expiry window

`(min_days, max_days)` inclusive, in calendar days from `reference_date`:

```python
chain = v.fetch_chain("AAPL", dte_range=(7, 45))     # next ~6 weeks
chain = v.fetch_chain("AAPL", dte_range=None)        # all expiries, incl. LEAPS
```

The high-level wrappers (`volatility_surface`, `volatility_smile`, `load_surface`)
also default to `dte_range=None`, so they include **all** expiries unless you cap
them.

### 3.3 Rates, dividends and reference date

`r` and `q` default to `None`, which tries the market:

- `r=None` → the `^IRX` 13-week T-bill yield, falling back to `0.045`.
- `q=None` → the ticker's dividend yield, falling back to `0.0`.
- `reference_date=None` → the real today (used to convert dates into DTE).

Override them explicitly when you need a fixed scenario:

```python
chain = v.fetch_chain("MSFT", r=0.05, q=0.008, reference_date="2026-01-05")
```

### 3.4 Limiting the number of expiries

```python
chain = v.fetch_chain("SPY", max_expiries=6)   # the 6 nearest expiries
```

### 3.5 The tidy schema

Every provider returns the same columns:

```
expiry, dte, strike, flag, bid, ask, last, mid, volume, open_interest, iv_yahoo
```

plus market metadata in `chain.attrs`:

```python
chain.attrs   # {'ticker': 'AAPL', 'spot': 231.4, 'r': 0.043, 'q': 0.005,
              #  'reference_date': datetime.date(2026, 10, 16)}
```

`flag` is `'c'` for calls, `'p'` for puts. `mid` is `(bid + ask) / 2` when both
quotes exist, otherwise the last price. `iv_yahoo` is Yahoo's own IV estimate,
provided for reference only — `compute_iv_surface` recomputes IVs with
`optnlib`.

---

## 4. Exporting raw chains to Excel — `export_chain`

Dump every expiry's calls and puts straight to an `.xlsx` workbook (one sheet
pair per expiry: `{date}_Calls` / `{date}_Puts`). Returns the filename.

```python
v.export_chain("AAPL", output_filename="aapl_options.xlsx", dte_range=(7, 180))
# -> writes {'{expiry}_Calls', '{expiry}_Puts'} sheets; returns 'aapl_options.xlsx'
```

Requires `openpyxl` (in the `market` extra). Timezones are stripped so Excel can
store the datetimes.

---

## 5. Computing implied volatilities — `compute_iv_surface`

Turn a chain (or any DataFrame with `expiry`/`dte`, `strike`, `flag` and a price
column) into IVs:

```python
frame = v.compute_iv_surface(
    chain,
    spot=None,          # default: chain.attrs['spot']
    r=None, q=None,     # default: chain.attrs['r'] / ['q']
    price_col="mid",    # price column to invert
    style="european",   # or "american"
    method="bs2002",    # American engine when style='american'
    reference_date=None,
    errors="drop",      # 'drop' (default) or 'nan' for un-invertible contracts
    min_price=1e-8,     # ignore prices below this
)
```

The result is a tidy IV DataFrame with the surface coordinates:

```
expiry, dte, t, strike, flag, moneyness, log_moneyness, mid, iv
```

- `t` is time to expiry in years; `moneyness = strike / spot`;
  `log_moneyness = ln(strike / spot)`.
- Contracts whose price admits no implied vol (deep ITM, stale/zero quotes) are
  handled by `errors`:

```python
frame_drop = v.compute_iv_surface(chain, errors="drop")   # removes bad rows (default)
frame_nan  = v.compute_iv_surface(chain, errors="nan")    # keeps them with iv = NaN
```

### 5.1 IV options: `price_col`, `style`, `method`

```python
# Use a different price column, e.g. the last trade
frame = v.compute_iv_surface(chain, price_col="last")

# American IVs (early exercise) — pick the same engine you price with
frame = v.compute_iv_surface(chain, style="american", method="lr")
```

`style` is `'european'` (default) or `'american'`. `method` selects the American
engine (`'bs2002'` default, `'crr'`, `'lr'`) and is ignored for European options.

---

## 6. Building the surface — `VolSurface`

```python
surf = v.VolSurface(frame, spot=None, rate=None, div_yield=None,
                    reference_date=None)
```

`VolSurface` interpolates the IVs in `(log-moneyness, days-to-expiry)`. When both
a call and a put exist at the same `(expiry, strike)`, the **out-of-the-money**
contract is preferred, which keeps the smiles clean.

Useful attributes:

| attribute | meaning |
|---|---|
| `surf.spot`, `surf.rate`, `surf.div_yield` | market inputs used |
| `surf.reference_date` | "today" used for DTE |
| `surf.ticker` | source ticker (`None` for synthetic chains) |
| `surf.expiries` | sorted list of `datetime.date` expiries |
| `surf.strikes` | sorted array of strikes |
| `surf.raw_frame` | the full tidy IV frame (before OTM de-duplication) |
| `surf.frame` | the de-duplicated frame used for interpolation |
| `surf.figure` | last figure drawn by a wrapper (set by `volatility_*`) |

```python
print(surf)   # VolSurface(ticker='AAPL', 980 contracts, 18 expiries, 25 strikes)
```

---

## 7. Querying the surface

All queries interpolate the surface; a point that lands exactly on a computed
contract returns its exact IV. **Points outside the traded grid raise
`InvalidInputError`** — there is no extrapolation.

### 7.1 By expiry date + strike — `iv`

```python
# expiry as an ISO string or datetime.date, strike as a price
iv = surf.iv("2026-12-18", 230.0)
```

### 7.2 By moneyness + days-to-expiry — `iv_moneyness`

```python
iv = surf.iv_moneyness(1.05, dte=45)   # K/S = 1.05, 45 days out
```

### 7.3 Interpolation methods

Both queries accept `method`:

- `"linear"` — **default**, uses a linear Delaunay interpolation (or a 1-D slice
  interpolation for a single-expiry surface).
- `"cubic"` — Clough–Tocher 2-D interpolation (smoother); may be unavailable for
  some surfaces (raises `InvalidInputError`).
- `"nearest"` — nearest-neighbour lookup.

```python
surf.iv("2026-12-18", 230.0, method="cubic")
surf.iv_moneyness(1.05, dte=45, method="nearest")
```

Handle out-of-range points:

```python
from optnlib.exceptions import InvalidInputError

try:
    surf.iv("2029-01-19", 500.0)
except InvalidInputError as exc:
    print(exc)   # e.g. "expiry is outside the surface: dte=... not in [...]"
```

### 7.4 Slices — `smile` and `term_structure`

```python
# IV versus strike for a single (nearest) expiry
smile = surf.smile("2026-12-18")
# columns: expiry, dte, strike, moneyness, flag, iv

# IV versus expiry for a single (nearest) strike
term = surf.term_structure(230.0)
# columns: expiry, dte, strike, moneyness, flag, iv
```

`smile()` with no argument uses the first expiry; `term_structure()` with no
argument uses the median strike. Both snap to the nearest available
expiry/strike.

### 7.5 Full grid — `to_grid`

```python
grid = surf.to_grid(n_dte=40, n_moneyness=60, method="linear")
# grid["moneyness"] (1-D), grid["dte"] (1-D), grid["dates"] (1-D),
# grid["iv"] -> 2-D array of shape (n_dte, n_moneyness)
```

Optional `moneyness_range` and `dte_range` narrow the grid; both default to the
surface's full extent.

---

## 8. IV tables

### 8.1 `to_matrix` — strike x tenor DataFrame

```python
mat = surf.to_matrix(time_axis="date", method="linear")
```

- **Rows** are strikes (index name `Strike`).
- **Columns** are expiries. `time_axis` labels them as:
  - `"date"` — **default**; ISO expiry dates (`2026-12-18`), column name `Expiry`;
  - `"dte"` — days-to-expiry (int), column name `DTE (days)`;
  - `"years"` — decimal years (rounded to 3 dp), column name `Tenor (years)`.

```python
mat = surf.to_matrix(time_axis="years")   # decimal-year columns
```

### 8.2 `surface_table` — print and return

```python
v.surface_table(surf, time_axis="date", method="linear", decimals=4,
                print_table=True)
```

Prints the rounded matrix to stdout and returns the DataFrame. Pass
`print_table=False` to only return it.

```python
mat = v.surface_table(surf, print_table=False)   # no stdout print
```

### 8.3 Export a table to Excel

Every table is a `pandas.DataFrame`, so use pandas directly (needs `openpyxl`):

```python
v.surface_table(surf, print_table=False).to_excel("iv_matrix.xlsx")
chain.to_excel("aapl_chain.xlsx", index=False)
surf.smile("2026-12-18").to_excel("smile.xlsx", index=False)
```

> For a ready-made workbook of the **raw** chain, use `export_chain(...)` (§4).

---

## 9. Plotting

Both plot functions accept a `VolSurface` or a tidy IV DataFrame and return the
figure. `backend=None` (default) picks **plotly** if it is importable, otherwise
matplotlib; you can force `backend="plotly"` or `"matplotlib"`.

### 9.1 The smile — `plot_smile`

```python
v.plot_smile(surf, layout="grid", n_expiries=5, ncols=None, backend=None,
             save=None, show=False, title=None, figsize=None, width=1000,
             height=700, markers=True)
```

- `layout="grid"` — **default**; one panel per expiry, `ncols` columns.
- `layout="overlay"` — all selected expiries on a single axes. Pass `ax=` to draw
  onto an existing matplotlib `Axes`.
- `expiries` — explicit list of expiry dates; otherwise a spread of `n_expiries`
  slices is chosen.
- The x-axis is always **Strike Price**.

```python
v.plot_smile(surf, layout="overlay")                 # all smiles on one axes
v.plot_smile(surf, expiries=["2026-11-20", "2026-12-18"])
```

### 9.2 The 3-D surface — `plot_surface`

```python
v.plot_surface(surf, backend=None, n_dte=40, n_moneyness=60, method="linear",
               cmap="viridis", save=None, show=False, title=None,
               elev=22, azim=-135, show_contours=False, width=950, height=760,
               figsize=(11, 8), decimate=None, atm_marker=True)
```

- **Time is on X (decimal years)**, **strike on Y**, IV on Z.
- With plotly, the **hover box shows the calendar date** of each point in
  addition to the decimal-year coordinate.
- `atm_marker=True` (default) draws a **red dot** at the at-the-money point
  (`strike == spot`) at the middle tenor.
- `show_contours=True` adds z-axis contour projections (plotly) / floor contours
  (matplotlib).
- `decimate` (matplotlib backend) strides the grid for speed; `None` picks a
  sensible default.

```python
v.plot_surface(surf, show_contours=True, atm_marker=True)
```

### 9.3 Backends, saving and showing

- `backend=None` → plotly if importable, else matplotlib.
- Invalid backend names raise `InvalidInputError`.
- `save="figure.html"` writes an interactive plotly HTML; `save="figure.png"`
  needs `kaleido` installed. matplotlib writes png/svg/pdf directly.

```python
v.plot_smile(surf, backend="matplotlib", save="smile.png")
v.plot_surface(surf, save="surface.html", show=False)
```

---

## 10. One-call wrappers

These run the whole pipeline. They accept the fetch/model parameters plus plot
parameters and return a `VolSurface` (the figure, when drawn, is stored at
`surf.figure`).

| function | what it does | default plot |
|---|---|---|
| `load_surface(...)` | fetch/synthesize + compute IVs + build the surface | none |
| `volatility_smile(...)` | `load_surface` then `plot_smile` | smile grid |
| `volatility_surface(...)` | `load_surface` then `plot_surface` | 3-D surface |

```python
# No plot — just the queryable surface
surf = v.load_surface("AAPL", strike_range=(0.8, 1.2), dte_range=(7, 365))

# Smile grid (layout='grid' default) with matplotlib
surf = v.volatility_smile("SPY", backend="matplotlib", layout="grid",
                          n_expiries=6, save="spy_smile.png")

# 3-D surface with contours and an ATM marker
surf = v.volatility_surface("AAPL", dte_range=(7, 365),
                            show_contours=True, atm_marker=True,
                            save="aapl_surface.html")
```

`volatility_surface` plot-specific params: `show_contours`, `atm_marker`,
`n_dte`, `n_moneyness`, `interp` (the interpolation `method`). `volatility_smile`
plot-specific params: `layout`, `n_expiries`. Both take `backend`, `save`,
`show`, `title`.

To build a surface from an existing chain instead of fetching:

```python
chain = v.fetch_chain("AAPL", strike_range=(0.8, 1.2))
surf  = v.volatility_surface(chain=chain)
```

---

## 11. Offline synthetic chain — `model_chain`

Everything above works identically on an **offline** synthetic chain whose IV
surface is known exactly (prices come from `optnlib`'s own Black-Scholes engine,
so recovered IVs match the model to root-finder tolerance). This is the only
non-live section, and it is the recommended way to try the API without the
network.

```python
from optnlib import volatility as v
import numpy as np

# 1) Build a synthetic chain (deterministic, no network).
chain = v.model_chain(
    spot=100.0,
    r=0.045,
    q=0.0,
    dtes=(30, 60, 90, 180, 365),     # days-to-expiry per slice
    strikes=None,                    # explicit list, or None to auto-generate
    n_strikes=11,                    # used when strikes is None
    moneyness_range=(0.7, 1.3),      # auto strikes span 70%-130% of spot
    base_vol=0.20,                   # iv(m, t) = base + skew*(m-1)
    skew=-0.35,                      #          + curvature*(m-1)^2
    curvature=0.9,                   #          + term_slope*(sqrt(t)-0.5)
    term_slope=0.05,
    spread=0.0,                      # absolute bid/ask half-spread
    reference_date="2026-01-05",
)

# 2) Wrap into a surface (or use source="model" on the wrappers).
frame = v.compute_iv_surface(chain)
surf  = v.VolSurface(frame)

# 3) Query it.
print(surf)                                  # VolSurface(ticker='MODEL', ...)
iv = surf.iv_moneyness(1.0, dte=30)          # ~ base_vol
smile = surf.smile("2026-02-04")             # IV vs strike at the 30d expiry
term  = surf.term_structure(100.0)

# 4) Table + plot.
v.surface_table(surf, time_axis="date", decimals=4)
v.plot_surface(surf, show_contours=True, atm_marker=True)

# Equivalent one-call form:
surf = v.volatility_surface(source="model", save="model_surface.html")
v.volatility_smile(source="model", backend="matplotlib", save="model_smile.png")
```

Extra keyword arguments for the wrappers are forwarded via `model_kwargs`:

```python
surf = v.volatility_surface(source="model",
                            model_kwargs={"spot": 50.0, "base_vol": 0.30,
                                          "skew": -0.6})
```

---

## 12. API reference

### 12.1 Market data

| function | signature | returns |
|---|---|---|
| `fetch_chain` | `(ticker, strike_range=None, dte_range=None, moneyness=True, r=None, q=None, max_expiries=None, reference_date=None)` | tidy chain DataFrame (live) |
| `export_chain` | `(ticker, output_filename=None, dte_range=None, r=None, q=None, reference_date=None)` | writes `.xlsx`, returns filename |
| `model_chain` | `(spot=100.0, r=0.045, q=0.0, dtes=(30,60,90,180,365), strikes=None, n_strikes=11, moneyness_range=(0.7,1.3), base_vol=0.20, skew=-0.35, curvature=0.9, term_slope=0.05, spread=0.0, reference_date=None)` | tidy chain DataFrame (offline) |
| `compute_iv_surface` | `(chain, spot=None, r=None, q=None, price_col="mid", style="european", method="bs2002", reference_date=None, errors="drop", min_price=1e-8)` | tidy IV DataFrame |

### 12.2 Surface object

| member | description |
|---|---|
| `VolSurface(frame, spot=None, rate=None, div_yield=None, reference_date=None)` | build the surface |
| `surf.iv(expiry, strike, method="linear")` | interpolated IV (float) |
| `surf.iv_moneyness(moneyness, dte, method="linear")` | interpolated IV (float) |
| `surf.smile(expiry=None)` | IV-vs-strike slice DataFrame |
| `surf.term_structure(strike=None)` | IV-vs-expiry slice DataFrame |
| `surf.to_grid(n_dte=40, n_moneyness=60, method="linear", moneyness_range=None, dte_range=None)` | dict of grid arrays |
| `surf.to_matrix(time_axis="date", method="linear")` | strike x tenor DataFrame |
| `surface_table(surface, time_axis="date", method="linear", decimals=4, print_table=True)` | IV matrix DataFrame |

### 12.3 Wrappers & plotting

| function | key params |
|---|---|
| `load_surface(ticker=None, chain=None, source="fetch", strike_range=(0.8,1.2), dte_range=None, moneyness=True, r=None, q=None, max_expiries=None, reference_date=None, price_col="mid", style="european", method="bs2002", model_kwargs=None)` | no plot; returns `VolSurface` |
| `volatility_smile(...)` | + `plot, backend, layout, save, show, n_expiries, title` |
| `volatility_surface(...)` | + `plot, backend, show_contours, atm_marker, n_dte, n_moneyness, interp, save, show, title` |
| `plot_smile(surface, expiries=None, n_expiries=5, backend=None, ax=None, save=None, show=False, title=None, layout="grid", ncols=None, figsize=None, width=1000, height=700, markers=True)` | smile (grid/overlay) |
| `plot_surface(surface, backend=None, n_dte=40, n_moneyness=60, method="linear", cmap="viridis", save=None, show=False, title=None, elev=22, azim=-135, show_contours=False, width=950, height=760, figsize=(11,8), decimate=None, atm_marker=True)` | 3-D surface |

---

## 13. Notes & gotchas

- **No extrapolation.** IV queries outside the traded grid raise
  `InvalidInputError` (convex-hull bounds). Stay inside the expiries/strikes you
  fetched.
- **OTM preference.** When a call and a put share an `(expiry, strike)`, the
  out-of-the-money contract wins, so a surface mixes calls and puts.
- **`strike_range`.** Moneyness multiples of spot by default
  (`moneyness=True`, e.g. `(0.8, 1.2)`); pass `moneyness=False` for absolute
  strikes (e.g. `(180, 240)`).
- **`dte_range`.** `fetch_chain` and the wrappers default to `None` = **all**
  expiries returned by Yahoo (including LEAPS). Cap it (e.g. `(7, 365)`) for
  faster, smaller surfaces.
- **Tables vs the plot axis.** `surface_table` / `to_matrix` label tenor columns
  with **ISO dates by default** (`time_axis="date"`); use `"years"` or `"dte"`
  for decimals/days. The **3-D surface axis stays in decimal years** with the
  date shown on hover.
- **Live data quality.** Yahoo chains can be delayed/incomplete, with stale bids
  or missing IVs. `compute_iv_surface(errors="drop")` (default) removes bad rows;
  use `errors="nan"` to keep them, or `source="model"` for deterministic runs.
- **Backends & saving.** `backend=None` auto-selects plotly if available else
  matplotlib. `save="x.html"` for interactive plotly; `save="x.png"` needs
  `kaleido`.
- **Optional deps.** Base `optnlib` needs only `numpy` + `scipy`. Live fetching
  needs `yfinance` + `pandas`; Excel export needs `openpyxl`; plotting needs
  `matplotlib`/`plotly`. Install with `pip install "optnlib[vol]"`.
- **Validation.** Inconsistent inputs (bad `flag`, non-positive strike, negative
  price, invalid `method`/`layout`/`style`/`errors`) raise `InvalidInputError`
  from `optnlib.exceptions`.
