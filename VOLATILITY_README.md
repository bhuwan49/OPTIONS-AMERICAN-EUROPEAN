# optnlib.volatility — volatility smiles & surfaces

`optnlib.volatility` builds implied-volatility smiles and 3-D surfaces from option chains. It works with live Yahoo Finance data or a built-in synthetic chain for offline demos.

## Quick start (copy-paste ready)

### 1) Build surface, plot, show matrix

```python
from optnlib import volatility as v

# Build surface + plot in one go (plotly by default, time in decimal years on X, strike on Y)
surf = v.volatility_surface("AAPL", strike_range=(0.8, 1.2), dte_range=(7, 180))
v.surface_table(surf)  # strike × tenor IV matrix
```

### 2) Query IVs and slices

```python
# Interpolated IV
surf.iv("2026-03-20", 195.0)      # expiry date (YYYY-MM-DD) + strike price
surf.iv_moneyness(1.05, dte=45)   # moneyness K/S + days-to-expiry

# Slices
surf.smile("2026-03-20")          # IV vs strike for nearest expiry
surf.term_structure(195.0)        # IV vs expiry for nearest strike
```

### 3) Plot smiles and surface explicitly

```python
# Smile (grid by default)
v.plot_smile(surf, layout="grid")                # plotly grid of per-expiry smiles

# 3-D surface: time on X (years), strike on Y, red ATM dot at mid tenor
v.plot_surface(surf, show_contours=True, atm_marker=True)
```

### 4) Offline (no network)

```python
from optnlib import volatility as v

surf = v.volatility_surface(source="model")
v.surface_table(surf)                     # columns are ISO expiry dates (default)
v.surface_table(surf, time_axis="years")  # or decimal-year columns
v.plot_surface(surf)
```

### 5) Step-by-step (no one-call wrapper)

```python
from optnlib import volatility as v

# 1) chain
chain = v.fetch_chain("AAPL", strike_range=(0.8, 1.2), dte_range=(7, 180))
# strike_range is moneyness multiples by default; use absolute strikes with:
# chain = v.fetch_chain("AAPL", strike_range=(80, 120), moneyness=False)

# 2) IVs
frame = v.compute_iv_surface(chain)

# 3) surface
surf = v.VolSurface(frame)

# 4) query/plot
print(surf.iv(chain["expiry"].iloc[0], chain["strike"].iloc[0]))
v.plot_smile(surf, layout="grid")
v.plot_surface(surf)
```

## Basics: from chain to surface

1. **Get chain** — `fetch_chain("TICKER", ...)` (live) or `model_chain(...)` (synthetic). Returns a tidy DataFrame with `expiry, dte, strike, flag, bid/ask/mid, ...` and `attrs` (`spot, r, q, reference_date, ticker`).
2. **Compute IVs** — `compute_iv_surface(chain, ...)` adds `iv`, `t`, `moneyness`, `log_moneyness`. Drops contracts that fail to invert IV by default (`errors='drop'`).
3. **Build surface** — `VolSurface(frame, ...)` interpolates in `(log-moneyness, days-to-expiry)`. Out-of-the-money contracts are preferred when both call/put exist at the same (expiry, strike).
4. **Plot/query** — `plot_smile`, `plot_surface`, or use one-call wrappers `volatility_smile`/`volatility_surface`.

## Plotting

- **Smile**: `v.plot_smile(surf, layout="grid", n_expiries=5, backend=None)`
  - `layout="grid"` (default, one panel per expiry) or `"overlay"` (all on one axes). Axis is **Strike Price**.
- **Surface**: `v.plot_surface(surf, n_dte=40, n_moneyness=60, show_contours=False, atm_marker=True)`
  - Time is on **X** (decimal years), strike on **Y**. With plotly, hovering shows the calendar date. `backend=None` picks plotly if available, else matplotlib. Save via `save="file.html"` or `save="file.png"` (png needs `kaleido`).

```python
v.plot_smile(surf, layout="grid")                # plotly grid of smiles
v.plot_surface(surf, show_contours=True, atm_marker=True)   # 3-D surface
```

## Advanced parameters

| function | key params (chronology) | notes |
|---|---|---|
| `fetch_chain(ticker, ...)` | `strike_range`, `dte_range`, `moneyness=True`, `r`, `q`, `max_expiries`, `reference_date` | live via yfinance; falls back to IRX/div yield if None. `strike_range` = moneyness multiples when `moneyness=True` (e.g. `(0.8, 1.2)`), or **absolute strikes** when `moneyness=False` (e.g. `(80, 120)`). |
| `model_chain(...)` | `spot,r,dtes,n_strikes,moneyness_range,base_vol,skew,curvature,term_slope,reference_date` | synthetic surface for offline/tests; recovered IVs match model. |
| `compute_iv_surface(chain, ...)` | `spot,r,q,price_col="mid"`, `style="european"`, `method="bs2002"`, `errors="drop"\|"nan"`, `min_price` | adds `iv,t,moneyness,log_moneyness`. |
| `VolSurface(frame, ...)` | `spot,rate,div_yield,reference_date` | interpolates `(log_moneyness,dte)` with `linear`/`cubic`/`nearest`; prefers OTM when call+put at same strike/expiry. |
| `surf.iv(expiry,strike, method=...)` / `surf.iv_moneyness(m,dte, method=...)` | `method="linear"|"cubic"|"nearest"` | raises if outside traded grid (no extrapolation). |
| `surf.smile(expiry)` / `surf.term_structure(strike)` | — | returns slice DataFrames. |
| `surf.to_grid(n_dte,n_moneyness,method,...)` | grid resolution + `method` | for plotting/inspection. |
| `surf.to_matrix(time_axis="date"\|"years"\|"dte", method="linear")` | columns by tenor (**date** default) | strike index (rows). |
| `surface_table(..., time_axis="date", decimals=4, print_table=True)` | same as above | prints table unless `print_table=False`. |
| `plot_smile(..., layout="grid"\|"overlay", n_expiries=5, backend=None, ...)` | layout + selection | strike-only axis. |
| `plot_surface(..., n_dte=40,n_moneyness=60,show_contours=False,atm_marker=True,backend=None,save,show,...)` | grid, contours, ATM marker | **time on X (years)**, **strike on Y**; plotly hover shows date. |
| `volatility_smile` / `volatility_surface` | pass through fetch/model + plot params | one-call; return `VolSurface`, figure stored at `.figure`. `load_surface` = no plot. |

## API summary

| function | returns |
|---|---|
| `fetch_chain(...)` | tidy chain DataFrame (live) |
| `export_chain(...)` | writes Excel, returns filename |
| `model_chain(...)` | tidy chain DataFrame (synthetic) |
| `compute_iv_surface(...)` | tidy IV DataFrame |
| `VolSurface(...)` | queryable surface object |
| `surf.iv(expiry,strike)` / `surf.iv_moneyness(m,dte)` | interpolated IV (float) |
| `surf.smile(expiry)` / `surf.term_structure(strike)` | slice DataFrames |
| `surf.to_grid(...)` | dict of grid arrays |
| `surf.to_matrix(time_axis, method)` | strike × tenor IV DataFrame |
| `surface_table(...)` | IV matrix DataFrame (prints unless `print_table=False`) |
| `plot_smile(...)` / `plot_surface(...)` | figure |
| `load_surface(...)` | `VolSurface` |
| `volatility_smile(...)` / `volatility_surface(...)` | `VolSurface` (+ `.figure`) |

## Notes

- Interpolation in `(log_moneyness, days-to-expiry)`. No extrapolation — out of range raises `InvalidInputError`.
- `strike_range=(low, high)` is read as **moneyness** multiples of spot by default (`moneyness=True`), e.g. `(0.8, 1.2)`. Pass `moneyness=False` to use **absolute strike prices**, e.g. `(80, 120)`.
- `dte_range=None` (the default for `fetch_chain` and the wrappers) keeps **all** expiries the provider returns, including LEAPS; pass `(min_days, max_days)` to cap the horizon.
- Tables (`surface_table`, `to_matrix`) label tenor columns with **ISO expiry dates** by default; use `time_axis="years"` or `"dte"` for decimals/days. The 3-D surface axis stays in decimal years with the date shown on hover.
- yfinance data can be delayed/incomplete; use `source="model"` for deterministic offline runs.
- OTM preference keeps smiles clean when both call and put exist at the same strike/expiry.
- Optional deps imported lazily. Install with `pip install "optnlib[vol]"` (`[market]`/`[viz]` also fine).
