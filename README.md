# optnlib

European and American option pricing, implied volatility and greeks, with
support for continuous dividend yields and discrete cash dividend schedules.

`optnlib` is a small, dependency-light library (`numpy` + `scipy` only) built on
top of the modeling conventions of [`py_vollib`](https://github.com/vollib/py_vollib),
but as a standalone package with no third-party pricing dependency.

## Features

- **European options** — Black-Scholes, Black-Scholes-Merton (continuous yield),
  and Black-76 (options on a forward).
- **American options** — Bjerksund-Stensland (2002) closed-form approximation
  (fast, default), plus Cox-Ross-Rubinstein and Leisen-Reimer binomial trees.
- **Dividends** — continuous yield, and discrete cash schedules modeled either
  as *forward* cashflows (default) or *escrowed* present values. Tree engines
  drop discrete dividends at their ex-dates.
- **Implied volatility** — for both European and American options.
- **Greeks** — analytical BSM greeks for European options; central-difference
  numerical greeks for American options and discrete dividends.
- **Unified API** — one `price(...)` and one `greeks(...)` entry point, plus
  vollib-style aliases (`black_scholes`, `black_scholes_merton`, `black`).

## Installation

```bash
pip install .
```

For development (tests):

```bash
pip install -e ".[dev]"
```

Requires Python `>=3.9,<3.13`.

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
```

See [`EXAMPLES.md`](EXAMPLES.md) for 18 runnable examples covering every feature.

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
- `n_steps` — tree steps for `'crr'`/`'lr'` (defaults: `500` CRR, `101` LR).

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

## Testing

```bash
python -m pytest
```

## License

MIT. See [`LICENSE`](LICENSE).
