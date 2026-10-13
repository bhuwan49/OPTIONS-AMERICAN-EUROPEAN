# optnlib.hedging

Greeks-based hedging and risk management for option/underlying portfolios.

This subpackage answers one question in the form of a matrix equation:

> Given a portfolio and a set of hedge instruments, **how much of each
> instrument do I buy (long) or sell (short) to neutralize a chosen set of
> greeks?**

It is kept deliberately separate from the pricing core (`optnlib.price`,
`optnlib.greeks`, ...). See the repository root `README.md` for the pricing
library and `EXAMPLES.md` for pricing examples.

## Install / import

`optnlib` must be importable (e.g. `pip install -e .` from the repo root), then:

```python
from optnlib import hedging as h
```

## The math

For a portfolio you want to make **delta-, gamma- and vega-neutral** using the
underlying stock plus two hedging options `H1`, `H2`, you solve

```
A · w = −b
```

```
⎡ Δ_stock  Δ_H1  Δ_H2 ⎤ ⎡ w_stock ⎤   ⎡ −Δ_portfolio ⎤
⎢ Γ_stock  Γ_H1  Γ_H2 ⎥ ⎢ w_H1    ⎥ = ⎢ −Γ_portfolio ⎥
⎣ 𝒱_stock  𝒱_H1  𝒱_H2 ⎦ ⎣ w_H2    ⎦   ⎣ −𝒱_portfolio ⎦
```

Structural facts built in for you:

* the underlying's higher-order greeks are **zero**: `Γ_stock = 𝒱_stock = 0`,
* the underlying's delta is **one**: `Δ_stock = 1.0`,
* the solved weights `w` are **signed quantities** to trade: `+` = go long,
  `−` = go short.

The implementation is general:

* **one row per greek**, **one column per hedge instrument**, each entry being
  that instrument's greek *per traded unit*;
* a **square** system is solved exactly with `numpy.linalg.solve`;
* an **over/under-determined** system (more instruments than greeks, or vice
  versa) falls back to `numpy.linalg.lstsq` (minimum-norm weights when
  under-determined);
* a **singular** matrix raises `optnlib.exceptions.InvalidInputError` with an
  explanatory message.

You can target any ordering/subset of `('delta', 'gamma', 'theta', 'vega',
'rho')`, and you can set non-zero post-hedge targets via `target=`.

## Quick start

```python
from optnlib import hedging as h

MKT = dict(S=100.0, t=0.25, r=0.05, sigma=0.25)

pf = h.Portfolio([
    h.OptionLeg(quantity=-10, flag='c', K=105, multiplier=100,
                name='short call', **MKT),
    h.Underlying(quantity=500, S=100, name='stock'),
])

plan = h.hedge(
    pf,
    [
        h.Underlying(S=100, name='stock'),
        h.OptionLeg(flag='c', K=100, multiplier=100, name='H1', **MKT),
        h.OptionLeg(flag='p', K=110, multiplier=100, name='H2', **MKT),
    ],
    greeks=('delta', 'gamma', 'vega'),
)

print(plan.describe(decimals=2))
```

```
Hedging plan  (neutralize: delta, gamma, vega)
==============================================================
Portfolio before hedge:
    delta  = +90.107422
    gamma  = -31.097792
    vega   = -194.361201

Required trades  (+ = go long, - = go short):
--------------------------------------------------------------
  stock                          LONG           446.52 shares
  H1                             LONG             2.18 contracts
  H2                             LONG             9.09 contracts

Residual greeks after hedge (target = 0):
  delta  = -0.000000
  gamma  = +0.000000
  vega   = +0.000000
Not neutralized (no degrees of freedom): theta, rho

Summary: LONG 446.52 shares (stock), LONG 2.18 contracts (H1), LONG 9.09 contracts (H2)
```

## Concepts

### `OptionLeg`

An option position. Key fields:

| field | meaning |
|---|---|
| `quantity` | number of contracts (negative = short) |
| `flag` | `'c'` / `'p'` |
| `S, K, t, r, sigma, q` | standard option parameters |
| `divs` | optional discrete dividend schedule `[(ex_time, amount), ...]` |
| `style` | `'european'` (default) / `'american'` |
| `method` | American engine `'bs2002'` (default) / `'crr'` / `'lr'` |
| `dividend_model` | `'forward'` (default) / `'escrowed'` |
| `multiplier` | shares per contract (100 for standard equity options, 1 for per-share) |
| `name` | short label used in reports (e.g. `'H1'`) |

Greek accessors:

* `unit_greeks()` – per-share greeks,
* `contract_greeks()` – per traded contract (`unit_greeks × multiplier`); these
  are the matrix columns,
* `position_greeks()` – the whole leg (`contract_greeks × quantity`).

### `Underlying`

A linear position. Its only non-zero greek is `delta = 1`; `gamma = vega =
theta = rho = 0`. Its trade unit is `shares`.

### `Portfolio`

Aggregates legs and exposes risk:

```python
pf.value()                 # mark-to-market value
pf.greeks()                # {'delta','gamma','theta','vega','rho'}
pf.dollar_greeks()         # $ delta/1.0 spot, $ gamma/1% spot, $ vega/vol pt, $ theta/day, $ rho/1% rate
pf.scenario(spot_shock=5.0, vol_shock=0.01, days=1.0, rate_shock=0.0)
pf.pnl_attribution(spot_shock=5.0, vol_shock=0.01, days=1.0)
pf.describe()
```

**Shock conventions** (absolute values):

| argument | meaning |
|---|---|
| `spot_shock` | change in the underlying price, e.g. `-5.0` dollars |
| `vol_shock` | change in volatility, e.g. `0.02` = +2 vol points |
| `days` | calendar days of time decay (positive = time passes) |
| `rate_shock` | change in rate, e.g. `0.01` = +1% |

For a *relative* spot move, compute it yourself: `spot_shock = 0.05 * S`.

`pnl_attribution` returns the per-greek Taylor contributions plus the full
revaluation P&L and the attribution error:

```python
{'delta': ..., 'gamma': ..., 'vega': ..., 'theta': ..., 'rho': ...,
 'total': ..., 'actual': ..., 'error': ...}
```

### `hedge()` and `delta_hedge()`

```python
plan = h.hedge(portfolio, instruments, greeks=('delta','gamma','vega'), target=None)
plan = h.delta_hedge(portfolio, underlying=None)   # shorthand for a single-stock delta hedge
```

### `HedgeResult`

| member | meaning |
|---|---|
| `quantities` | numpy array of signed units to trade |
| `matrix` | the `A` matrix (rows = greeks, cols = instruments) |
| `rhs` | right-hand side `target − b` |
| `residual` | post-hedge residual per constrained greek (≈ 0) |
| `unused` | greeks that were not constrained |
| `method` | `'solve'` or `'lstsq'` |
| `describe(decimals=None)` / `str(plan)` | formatted report |
| `to_dict()` | `{instrument_name: signed_quantity}` |
| `to_portfolio()` | the hedge trades as a `Portfolio` |

## Reporting

`plan.describe(decimals=None)` (and `print(plan)`) produce the table shown
above. Instrument rows read `LONG`/`SHORT`, the absolute quantity, and the
trade unit (`shares` for the underlying, `contracts` when an option's
`multiplier != 1`, else `units`). The one-line `Summary:` lists the trades in
plain language.

`portfolio.describe()` gives an analogous portfolio-level summary.

## Notes & caveats

* Greeks for European options with no discrete dividends are analytical; for
  American options and discrete dividends they are central finite differences,
  so they carry a small bump-and-revalue error.
* This solves for a **local, instantaneous** neutralization. Because greeks
  change as spot/vol/time move, a hedge must be rebalanced. Use
  `Portfolio.scenario` / `pnl_attribution` to see the residual risk.
* If you supply fewer greeks than instruments the system is under-determined
  and the reported weights are the minimum-norm solution; add `target=` if you
  want specific non-zero post-hedge exposures.

See `HEDGING_EXAMPLES.md` (repo root) for runnable end-to-end examples.
