# optnlib.hedging — Examples

Runnable examples for the hedging subpackage. Pricing examples live in the
repository root `EXAMPLES.md`; these focus on portfolio risk and hedging.

```python
from optnlib import hedging as h

MKT = dict(S=100.0, t=0.25, r=0.05, sigma=0.25)
```

---

## 1. Portfolio value and greeks

```python
pf = h.Portfolio([
    h.OptionLeg(quantity=-10, flag='c', K=105, multiplier=100,
                name='short call', **MKT),
    h.Underlying(quantity=500, S=100, name='stock'),
])

print(pf.describe())
print(pf.greeks())
```

```
Portfolio  (2 legs)   value = 46,560.11
    short call                   qty = -10
    stock                        qty = +500
    delta = +90.107422  gamma = -31.097792  theta = +31.768570  vega = -194.361201  rho = -93.873408

{'delta': 90.10742165144086, 'gamma': -31.097792192168026,
 'theta': 31.76857049030826, 'vega': -194.36120120105016,
 'rho': -93.87340844550042}
```

---

## 2. Dollar greeks

```python
print(pf.dollar_greeks())
```

```
{'delta': 9010.742165144085,   # $ per +1.0 spot move
 'gamma': -3109.7792192168026, # $ per +1% spot move
 'vega': -194.36120120105016,  # $ per +1 vol point
 'theta': 31.76857049030826,   # $ per calendar day
 'rho': -93.87340844550042}    # $ per +1% rate move
```

---

## 3. Delta hedge with the underlying

```python
plan = h.delta_hedge(pf)
print(plan.describe())
```

```
Hedging plan  (neutralize: delta)
==============================================================
Portfolio before hedge:
    delta  = +90.107422

Required trades  (+ = go long, - = go short):
--------------------------------------------------------------
  Underlying (S=100)             SHORT       90.107422 shares

Residual greeks after hedge (target = 0):
  delta  = +0.000000
Not neutralized (no degrees of freedom): gamma, theta, vega, rho

Summary: SHORT 90.107422 shares (Underlying (S=100))
```

`plan.quantities[0]` equals `-pf.delta()`.

---

## 4. The 3×3 delta-gamma-vega hedge (underlying + H1 + H2)

```python
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
  stock                          SHORT        0.007288 shares
  H1                             LONG         5.319956 contracts
  H2                             LONG         5.379814 contracts

Residual greeks after hedge (target = 0):
  delta  = +0.000000
  gamma  = +0.000000
  vega   = +0.000000
Not neutralized (no degrees of freedom): theta, rho

Solver  : lstsq  (rank 2/3, cond 1.86e+19, row-scaled)
Note    : hedge matrix is rank-deficient (rank 2/3); the target is
          still reachable, so the minimum-norm weights below are an exact
          hedge (many equivalent solutions exist).

Summary: SHORT 0.007288 shares (stock), LONG 5.319956 contracts (H1), LONG 5.379814 contracts (H2)
```

The stock's row of the matrix is `[1, 0, 0]`:

```python
print(plan.matrix[0, 0], plan.matrix[1, 0], plan.matrix[2, 0])  # 1.0  0.0  0.0
```

Residual is machine-zero because `A · w = −b` is solved exactly, and the hedge
legs reproduce the portfolio's greeks:

```python
plan.to_portfolio().greeks()['delta']  # ≈ -90.107  (i.e. cancels the portfolio)
print(plan.to_dict())                  # {'stock': -0.00729, 'H1': 5.320, 'H2': 5.380}
```

---

## 5. A 2×2 hedge (delta + vega only)

Constraints change the stock weight — the option now also carries delta.

```python
plan = h.hedge(
    pf,
    [
        h.Underlying(S=100, name='stock'),
        h.OptionLeg(flag='c', K=100, multiplier=100, name='H1', **MKT),
    ],
    greeks=('delta', 'vega'),
)
print(plan.describe(decimals=2))
```

```
Hedging plan  (neutralize: delta, vega)
==============================================================
Portfolio before hedge:
    delta  = +90.107422
    vega   = -194.361201

Required trades  (+ = go long, - = go short):
--------------------------------------------------------------
  stock                          SHORT          647.50 shares
  H1                             LONG             9.87 contracts

Residual greeks after hedge (target = 0):
  delta  = +0.000000
  vega   = +0.000000
Not neutralized (no degrees of freedom): gamma, theta, rho

Summary: SHORT 647.50 shares (stock), LONG 9.87 contracts (H1)
```

---

## 6. Over-determined system (`lstsq`)

More instruments than greeks → least-squares weights.

```python
plan = h.hedge(pf, [
    h.Underlying(S=100, name='stock'),
    h.OptionLeg(flag='c', K=100, multiplier=100, name='H1', **MKT),
    h.OptionLeg(flag='p', K=110, multiplier=100, name='H2', **MKT),
], greeks=('delta', 'gamma'))

print(plan.method)                 # 'lstsq'
print(plan.residual)               # {'delta': 0.0, 'gamma': 0.0}
print(plan.describe(decimals=2))
```

```
Hedging plan  (neutralize: delta, gamma)
==============================================================
Portfolio before hedge:
    delta  = +90.107422
    gamma  = -31.097792

Required trades  (+ = go long, - = go short):
--------------------------------------------------------------
  stock                          SHORT        0.007288 shares
  H1                             LONG         5.319956 contracts
  H2                             LONG         5.379814 contracts

Residual greeks after hedge (target = 0):
  delta  = -0.000000
  gamma  = +0.000000
Not neutralized (no degrees of freedom): theta, vega, rho
```

---

## 7. Hedging an American option

American greeks use finite differences; the same API applies.

```python
short_put = h.OptionLeg(quantity=-5, flag='p', K=100, multiplier=100,
                        name='short put', S=100, t=0.5, r=0.05, sigma=0.3,
                        style='american', method='bs2002')
pf = h.Portfolio([short_put])

plan = h.hedge(pf, [
    h.Underlying(S=100, name='stock'),
    h.OptionLeg(flag='p', K=95, multiplier=100, name='H1', S=100, t=0.5,
                r=0.05, sigma=0.3, style='american'),
], greeks=('delta', 'gamma'))

print(plan.describe(decimals=2))
```

```
Hedging plan  (neutralize: delta, gamma)
==============================================================
Portfolio before hedge:
    delta  = +214.562636
    gamma  = -9.904805

Required trades  (+ = go long, - = go short):
--------------------------------------------------------------
  stock                          SHORT           30.82 shares
  H1                             LONG             5.53 contracts

Residual greeks after hedge (target = 0):
  delta  = +0.000000
  gamma  = +0.000000
Not neutralized (no degrees of freedom): theta, vega, rho
```

---

## 8. Scenario / stress testing

Full revaluation P&L for absolute shocks.

```python
print(pf.scenario(spot_shock=5.0))     # spot +5 dollars
print(pf.scenario(spot_shock=-10.0))   # spot -10 dollars
print(pf.scenario(vol_shock=0.02))     # vol +2 points
print(pf.scenario(days=1.0))           # one calendar day of decay
```

---

## 9. P&L attribution

Taylor breakdown vs. full revaluation.

```python
attr = pf.pnl_attribution(spot_shock=2.0, vol_shock=0.01, days=1.0)
for k, v in attr.items():
    print(f"{k:>7}: {v:+.4f}")
```

```
  delta: +180.2148
  gamma:  -62.1956
   vega: -194.3612
  theta:  +31.7686
    rho:   -0.0000
  total:  -44.5734
 actual:  -50.1853
  error:   +5.6119   # higher-order terms not captured by a Taylor expansion
```

---

## 10. Contract multipliers / per-share mode

Set `multiplier=1` to work in per-share units. The solved option weights scale
with the multiplier, and the hedge legs still reproduce the same position
greeks.

```python
plan_per_share  = h.hedge(pf, [
    h.Underlying(S=100, name='stock'),
    h.OptionLeg(flag='c', K=100, multiplier=1,   name='H1', **MKT),
    h.OptionLeg(flag='p', K=110, multiplier=1,   name='H2', **MKT),
])
plan_per_100    = h.hedge(pf, [
    h.Underlying(S=100, name='stock'),
    h.OptionLeg(flag='c', K=100, multiplier=100, name='H1', **MKT),
    h.OptionLeg(flag='p', K=110, multiplier=100, name='H2', **MKT),
])

for g in ('delta', 'gamma', 'vega'):
    assert plan_per_share.to_portfolio().greeks()[g] == \
           plan_per_100.to_portfolio().greeks()[g]
```

---

## 11. Targeting a residual exposure

Pass `target=` to keep (rather than zero) specific exposures.

```python
plan = h.hedge(
    pf,
    [h.Underlying(S=100, name='stock')],
    greeks=('delta',),
    target={'delta': 25.0},   # finish with +25 delta, not 0
)
print(plan.residual)          # {'delta': ~0.0}   -> hit the target
```

---

## 12. Programmatic use

```python
plan = h.hedge(pf, [
    h.Underlying(S=100, name='stock'),
    h.OptionLeg(flag='c', K=100, multiplier=100, name='H1', **MKT),
    h.OptionLeg(flag='p', K=110, multiplier=100, name='H2', **MKT),
])

weights = plan.to_dict()            # {'stock': ..., 'H1': ..., 'H2': ...}
hedge_legs = plan.to_portfolio()    # the trades as a Portfolio
print(hedge_legs.describe())
```

---

## Errors to expect

```python
from optnlib.exceptions import InvalidInputError

# duplicate / linearly dependent instruments (when strict):
h.hedge(pf, [
    h.OptionLeg(flag='c', K=100, name='A', **MKT),
    h.OptionLeg(flag='c', K=100, name='B', **MKT),
], greeks=('delta', 'gamma'), strict=True)
# InvalidInputError: hedge matrix is singular or ill-conditioned: ...

# unknown greek:
h.hedge(pf, [h.Underlying(S=100)], greeks=('delta_v2',))
# InvalidInputError: unknown greek 'delta_v2'; ...
```
