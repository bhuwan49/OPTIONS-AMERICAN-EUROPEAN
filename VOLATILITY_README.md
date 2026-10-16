# optnlib.volatility — volatility smiles & surfaces

optnlib.volatility builds implied-volatility smiles and 3-D surfaces from option chains. It works with live Yahoo Finance data or a built-in synthetic chain for offline demos.

## Quick start

`python
from optnlib import volatility as v

# Build surface + plot in one go (plotly by default, years axis, strike on Y)
surf = v.volatility_surface("AAPL", strike_range=(0.8, 1.2), dte_range=(7, 180))
v.surface_table(surf)  # strike × tenor IV matrix
`

`python
# Query interpolated IV
surf.iv("2026-03-20", 195.0)      # expiry date + strike price
surf.iv_moneyness(1.05, dte=45)   # moneyness + days-to-expiry

# Inspect slices
surf.smile("2026-03-20")          # IV vs strike for nearest expiry
surf.term_structure(195.0)        # IV vs expiry for nearest strike
`

`python
# Offline (no network)
surf = v.volatility_surface(source="model")
`

## Basics: from chain to surface

1. **Get chain** — etch_chain("TICKER", ...) (live) or model_chain(...) (synthetic). Returns a tidy DataFrame with expiry, dte, strike, flag, bid/ask/mid, ... and ttrs (spot, r, q, reference_date, ticker).
2. **Compute IVs** — compute_iv_surface(chain, ...) adds iv, 	, moneyness, log_moneyness. Drops contracts that fail to invert IV by default (errors='drop').
3. **Build surface** — VolSurface(frame, ...) interpolates in (log-moneyness, days-to-expiry). Out-of-the-money contracts are preferred when both call/put exist at the same (expiry,strike).
4. **Plot/query** — plot_smile, plot_surface, or use one-call wrappers olatility_smile/olatility_surface.

## Plotting

- **Smile**: .plot_smile(surf, layout="grid", n_expiries=5, backend=None)  
  layout="grid" (default, one panel per expiry) or "overlay" (all on one axes). Axis is **Strike Price**.
- **Surface**: .plot_surface(surf, n_dte=40, n_moneyness=60, show_contours=False, atm_marker=True)  
  Time is on **X** (decimal years), strike on **Y**. With plotly, hovering shows the calendar date. ackend=None picks plotly if available, else matplotlib. Save via save="file.html" or save="file.png" (png needs kaleido).

`python
v.plot_smile(surf, layout="grid")                # plotly grid of smiles
v.plot_surface(surf, show_contours=True)         # 3-D surface with contours + ATM dot
`

## Advanced parameters

| function | key params (chronology) | notes |
|---|---|---|
| etch_chain | strike_range (moneyness multiples if moneyness=True), dte_range, max_expiries, , q, eference_date | live via yfinance; falls back to IRX/div yield if None. |
| model_chain | spot, r, dtes, n_strikes, moneyness_range, base_vol, skew, curvature, term_slope, reference_date | synthetic surface for offline testing; prices match recovered IVs. |
| compute_iv_surface | price_col='mid', style='european', method='bs2002', errors='drop'|'nan', min_price | inverts per contract; copies attrs from chain. |
| VolSurface | spot, rate, div_yield, reference_date | builds (log-moneyness,dte) interpolators (linear default, cubic, 
earest); prefers OTM. |
| iv/iv_moneyness | method='linear'|'cubic'|'nearest' | raises InvalidInputError if outside traded grid (no extrapolation). |
| 	o_matrix/surface_table | 	ime_axis='years'|'date'|'dte', method, decimals, print_table | strike×tenor matrix; surface_table prints by default. |
| plot_smile | layout, n_expiries, expiries, backend, ax, markers, save, show | strike-only; grid is cleanest for multiple expiries. |
| plot_surface | 
_dte, n_moneyness, method, cmap, show_contours, atm_marker, elev, azim, decimate, backend, save, show | years on X, strike on Y; date in plotly hover. |
| olatility_smile/olatility_surface | pass through fetch/model + plot params; return VolSurface (figure at .figure) | one-call wrappers. load_surface builds without plotting. |

## API summary

| function | returns |
|---|---|
| etch_chain(...) | tidy chain DataFrame (live) |
| export_chain(...) | writes Excel, returns filename |
| model_chain(...) | tidy chain DataFrame (synthetic) |
| compute_iv_surface(...) | tidy IV DataFrame |
| VolSurface(...) | queryable surface object |
| surf.iv(expiry, strike) / surf.iv_moneyness(m, dte) | interpolated IV (float) |
| surf.smile(expiry) / surf.term_structure(strike) | slice DataFrames |
| surf.to_grid(...) | dict of grid arrays |
| surf.to_matrix(time_axis, method) | strike × tenor IV DataFrame |
| surface_table(...) | IV matrix DataFrame (prints unless print_table=False) |
| plot_smile(...) / plot_surface(...) | figure |
| load_surface(...) | VolSurface |
| olatility_smile(...) / olatility_surface(...) | VolSurface (+ surface.figure) |

## Notes

- Interpolation is in (log-moneyness, days-to-expiry). Points outside the convex hull raise rather than extrapolating.
- yfinance data can be delayed/incomplete — use source="model" for deterministic, offline runs.
- OTM preference keeps smiles clean when both calls and puts exist at the same strike/expiry.
- All optional deps imported lazily; install extras via pip install "optnlib[vol]".
