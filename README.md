# optnlib

European and American option pricing, Monte Carlo simulation, implied
volatility and greeks, with support for continuous dividend yields and discrete
cash dividend schedules.

`optnlib` is a small, dependency-light library (`numpy` + `scipy` only) built on
top of the modeling conventions of [`py_vollib`](https://github.com/vollib/py_vollib),
but as a standalone package with no third-party pricing dependency.

## Hedging & risk management

`optnlib.hedging` turns a portfolio of options + the underlying into a set of
**signed trades** that neutralize delta / gamma / vega (or any chosen set of
greeks) by solving the hedge matrix equation `A · w = −b`:

```python
from optnlib import hedging as h

MKT = dict(S=100.0, t=0.25, r=0.05, sigma=0.25)
pf = h.Portfolio([
    h.OptionLeg(quantity=-10, flag='c', K=105, multiplier=100,
                name='short call', **MKT),
    h.Underlying(quantity=500, S=100, name='stock'),
])

plan = h.hedge(pf, [
    h.Underlying(S=100, name='stock'),
    h.OptionLeg(flag='c', K=100, multiplier=100, name='H1', **MKT),
    h.OptionLeg(flag='p', K=110, multiplier=100, name='H2', **MKT),
], greeks=('delta', 'gamma', 'vega'))

print(plan.describe(decimals=2))
#   stock   SHORT    0.007 shares
#   H1      LONG     5.320 contracts
#   H2      LONG     5.380 contracts
```

For hard cases — rank-deficient / ill-conditioned systems where a raw solve
returns huge offsetting trades — use the separate regularized builders
`h.hedge_ridge(...)` (scikit-learn ridge, tunable `alpha`) or
`h.hedge_optimize(...)` (position bounds, gross-exposure cap, transaction
costs).

- **Full guide:** [`HEDGING_README.md`](HEDGING_README.md)
- **Worked examples:** [`HEDGING_EXAMPLES.md`](HEDGING_EXAMPLES.md)
- **Regularized & constrained:** [`HEDGING_REGULARIZED.md`](HEDGING_REGULARIZED.md)

## Volatility smiles & surfaces

`optnlib.volatility` fetches an option chain (live Yahoo via `yfinance`, or an
offline synthetic chain), computes implied vols, and plots the **smile** and the
**3-D surface** (interactive plotly by default, matplotlib fallback). The smile
is drawn as a grid of per-expiry panels; the surface puts time on the x-axis
(decimal years, with the calendar date shown on hover) and can be queried at any
expiry and strike:

```python
from optnlib import volatility as v

surf = v.volatility_surface("AAPL", strike_range=(0.8, 1.2),
                            dte_range=(7, 180), save="AAPL_surface.html")

surf.iv("2026-03-20", 195.0)      # interpolated IV at an exact date + strike
surf.iv_moneyness(1.05, dte=45)   # IV at a moneyness + days-to-expiry
surf.smile("2026-03-20")          # IV vs strike at one expiry
surf.term_structure(195.0)        # IV vs expiry at one strike
v.surface_table(surf)             # strike x tenor IV matrix (ISO expiry columns)

# strike_range is moneyness multiples by default; use absolute strikes with
# v.volatility_surface("AAPL", strike_range=(80, 120), moneyness=False, ...)

# offline demo (no network needed)
v.volatility_surface(source="model", save="model_surface.html")
```

Install the extras with `pip install "optnlib[vol]"`.

- **Full guide:** [`VOLATILITY_README.md`](VOLATILITY_README.md)
- **Implied volatility guide (smiles & surfaces):** [`IMPLIED_VOLATILITY_README.md`](IMPLIED_VOLATILITY_README.md)

## Features

- **European options** — Black-Scholes, Black-Scholes-Merton (continuous yield),
  and Black-76 (options on a forward).
- **American options** — Bjerksund-Stensland (2002) closed-form approximation
  (fast, default), plus Cox-Ross-Rubinstein and Leisen-Reimer binomial trees.
- **Monte Carlo** — European option pricing by GBM path simulation, with
  configurable time steps and path count, plus direct matplotlib path
  visualization.
- **Dividends** — continuous yield, and discrete cash schedules modeled either
  as *forward* cashflows (default) or *escrowed* present values. Tree engines
  drop discrete dividends at their ex-dates.
- **Implied volatility** — for both European and American options.
- **Greeks** — analytical BSM greeks for European options; central-difference
  numerical greeks for American options and discrete dividends.
- **Unified API** — one `price(...)` and one `greeks(...)` entry point, plus
  vollib-style aliases (`black_scholes`, `black_scholes_merton`, `black`).
- **Hedging & risk** — `optnlib.hedging`: portfolios, dollar greeks, scenario
  stress testing, Taylor P&L attribution, and delta/gamma/vega (or arbitrary
  greek) hedge solving with descriptive long/short trade output.
- **Volatility smiles & surfaces** — `optnlib.volatility`: fetch or synthesize an
  option chain, compute implied vols, and plot/query per-expiry smiles and 3-D
  surfaces (plotly or matplotlib, decimal-years axis + date-on-hover), with a strike × tenor IV
  matrix.

## Installation

```bash
pip install .
```

Optional extras: `[market]` (pandas/yfinance/openpyxl for option data),
`[viz]` (matplotlib/plotly for plotting), `[vol]` (both), `[ridge]`
(scikit-learn for regularized hedging).

For development (tests):

```bash
pip install -e ".[dev]"
```

Requires Python `>=3.9,<3.15`.

## Quick start

```python
import optnlib

# European Black-Scholes
optnlib.black_scholes('c', 100, 100, 1.0, 0.05, 0.2)
# 10.450583572185565

# Black-Scholes-Merton with a continuous dividend yield
optnlib.black_scholes_merton('c', 100, 100, 1.0, 0.05, 0.2, q=0.03)
# 8.652528553942709

# American put, default Bjerksund-Stensland (2002) engine
optnlib.american_price('p', 100, 100, 1.0, 0.05, 0.2)
# 6.045010205142354

# Unified entry point
optnlib.price('p', 100, 100, 1.0, 0.05, 0.2, style='american', method='lr')
# 6.087222149481556

# Explicit European API with a discrete dividend
optnlib.european_price('c', 100, 100, 1.0, 0.05, 0.2, divs=[(0.5, 2.0)])
# 9.244683623230566

# Monte Carlo European call (252 steps, 10,000 paths)
optnlib.monte_carlo_price('c', 100, 100, 1.0, 0.05, 0.2, seed=42)
# 10.557482096687831

# Simulate paths and plot them with matplotlib
paths = optnlib.monte_carlo.simulate(100, 1.0, 0.05, 0.2, seed=42)
optnlib.monte_carlo.plot_paths(paths, K=100, t=1.0, save='mc_paths.png')
```

See [`EXAMPLES.md`](EXAMPLES.md) for 23 runnable examples covering every feature.

## API overview

### Pricing

| Function | Description |
| --- | --- |
| `price(flag, S, K, t, r, sigma, q=0.0, divs=None, style='european', method=None, dividend_model='forward', n_steps=None)` | Unified European/American price |
| `european_price(flag, S, K, t, r, sigma, q=0.0, divs=None, dividend_model='forward')` | European price |
| `american_price(flag, S, K, t, r, sigma, q=0.0, divs=None, method='bs2002', dividend_model='forward', n_steps=None)` | American price |
| `black_scholes(flag, S, K, t, r, sigma)` | Black-Scholes |
| `black_scholes_merton(flag, S, K, t, r, sigma, q=0.0)` | Black-Scholes-Merton |
| `black(flag, F, K, t, r, sigma)` | Black-76 on a forward `F` |

### Monte Carlo

| Function | Description |
| --- | --- |
| `monte_carlo.price(flag, S, K, t, r, sigma, q=0.0, n_steps=252, n_sims=10000, seed=None)` | European Monte Carlo price under GBM |
| `monte_carlo.simulate(S, t, r, sigma, q=0.0, n_steps=252, n_sims=10000, seed=None)` | Simulated GBM paths, shape `(n_sims, n_steps + 1)` |
| `monte_carlo.plot_paths(paths, K=None, t=None, n_paths=100, save=None, show=False)` | Matplotlib path visualization |

`optnlib.monte_carlo_price(...)` is a top-level alias for
`optnlib.monte_carlo.price(...)`.

### Implied volatility

```python
optnlib.implied_volatility(price, flag, S, K, t, r, q=0.0, divs=None,
                           style='european', method='bs2002',
                           dividend_model='forward', n_steps=None)
```

### Greeks

```python
optnlib.greeks(flag, S, K, t, r, sigma, q=0.0, divs=None, style='european',
               method='bs2002', dividend_model='forward', n_steps=None,
               greeks_type=None)   # -> {'delta', 'gamma', 'theta', 'vega', 'rho'}

optnlib.delta(...)   # individual helpers
optnlib.gamma(...)
optnlib.theta(...)
optnlib.vega(...)
optnlib.rho(...)
```

Analytical BSM greeks are used for European options with no discrete dividends;
everything else uses central finite differences. Pass
`greeks_type='analytical'` or `'numerical'` to force a choice.

**Scaling** (following common trading convention and `py_vollib`):

- `delta`, `gamma` — raw.
- `theta` — per calendar day (annual derivative / 365).
- `vega` — per 1% change in volatility (raw derivative × 0.01).
- `rho` — per 1% change in the interest rate (raw derivative × 0.01).

## Parameters

- `flag` — `'c'`/`'call'` or `'p'`/`'put'` (case-insensitive).
- `S` — underlying spot price (`> 0`). For `black`, the forward `F`.
- `K` — strike price (`> 0`).
- `t` — time to expiry in years (`>= 0`); `t = 0` returns intrinsic value.
- `r` — annualized continuously compounded risk-free rate.
- `sigma` — annualized volatility (`>= 0`).
- `q` — annualized continuous dividend yield (default `0`).
- `divs` — discrete cash schedule `[(ex_time, amount), ...]`, where `ex_time`
  is in years and must be strictly positive; entries with `ex_time > t` are
  ignored (default `None`).
- `method` — American engine: `'bs2002'` (default), `'crr'` or `'lr'`.
- `dividend_model` — `'forward'` (default) or `'escrowed'`.
- `n_steps` — tree steps for `'crr'`/`'lr'` (defaults: `500` CRR, `101` LR),
  or Monte Carlo time steps per path (default `252`).
- `n_sims` — Monte Carlo paths to simulate (default `10,000`).
- `seed` — optional NumPy generator seed for reproducible Monte Carlo results.

## Dividend models

- **`forward`** — each discrete cash amount is a forward cashflow. The pricing
  forward is `S·e^{(r−q)t} − Σ D_i·e^{r(t−t_i)}`. Binomial engines instead drop
  the cash amount from every node at the ex-date step (timing aware).
- **`escrowed`** — the present value of the discrete dividends is removed from
  the spot up front: `S − Σ D_i·e^{−r·t_i}`.

The Bjerksund-Stensland engine only accepts a continuous yield, so discrete
schedules are folded into an equivalent yield `q_eff` consistent with the
chosen model.

## Turning real dividend data into `q` and `divs`

**Continuous yield `q`** — the annualized, continuously compounded dividend
yield. Given the annual cash dividend per share `D` (or an analyst's dividend
yield `y = D/S`), the usual conversions are:

- Proportional yield: `q ≈ D / S`.
- If the quote is a simple annual rate `y`, `q = ln(1 + y)`.
- Exactly, over a horizon `t` with a discrete schedule:
  `q = r − ln(F / S) / t`, where
  `F = S·e^{r·t} − Σ D_i·e^{r·(t − t_i)}` is the dividend-adjusted forward.
  This is exactly the equivalent yield `optnlib` builds internally when it
  folds `divs` into `q` for the BS2002 engine.

```python
import optnlib

S, D = 100.0, 4.0            # $4.00 cash dividends per share per year
q = D / S                    # 0.04
optnlib.black_scholes_merton('c', 100, 100, 1.0, 0.05, 0.2, q=q)
# 8.102643534463212
```

**Discrete schedule `divs`** — a list of `(ex_time, amount)` pairs, where
`ex_time` is the time **in years** from now to the ex-dividend date and
`amount` is the cash amount per share. Convert calendar dates with a day count
such as ACT/365:

```python
from datetime import date

today, expiry = date(2026, 1, 1), date(2027, 1, 1)
t = (expiry - today).days / 365                       # 1.0
divs = [
    ((date(2026, 4, 15) - today).days / 365, 1.25),   # 0.2849
    ((date(2026, 10, 15) - today).days / 365, 1.25),  # 0.7863
]
optnlib.european_price('c', 100, 100, t, 0.05, 0.2, divs=divs)
# 8.957235470754656
```

Rules: `ex_time` must be `> 0`; entries with `ex_time > t` are ignored; a
dividend exactly at `t` is included.

## American engines

| Method | Description | Default steps |
| --- | --- | --- |
| `bs2002` | Bjerksund-Stensland (2002) closed-form approximation. Fast; a lower bound for puts. | n/a |
| `crr` | Cox-Ross-Rubinstein binomial tree. | 500 |
| `lr` | Leisen-Reimer binomial tree (Peizer-Pratt inversion). Accurate; recommended tree. | 101 |

## Monte Carlo

`optnlib.monte_carlo` prices European calls and puts by simulating geometric
Brownian motion paths under the risk-neutral measure with continuous yield `q`.
Each path uses the exact step

`S_next = S * exp((r - q - sigma**2 / 2) * dt + sigma * sqrt(dt) * Z)`.

`n_steps` sets the number of time steps and `n_sims` the number of paths; both
default to `252` and `10,000`. Pass `seed` for reproducible results.

```python
from optnlib import monte_carlo as mc

price = mc.price('c', 100, 100, 1.0, 0.05, 0.2,
                 n_steps=252, n_sims=100_000, seed=42)

paths = mc.simulate(100, 1.0, 0.05, 0.2, n_steps=252, n_sims=10_000, seed=42)
mc.plot_paths(paths, K=100, t=1.0, save='mc_paths.png')
```

Pricing uses numpy only. Plotting requires matplotlib (`pip install
"optnlib[viz]"`) and is imported lazily, so the pricing API works without it.
Monte Carlo currently covers European options with continuous yields only;
American early exercise and discrete dividend schedules use the other engines.

## Testing

```bash
python -m pytest
```

## License

MIT. See [`LICENSE`](LICENSE).
