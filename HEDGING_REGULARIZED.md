# optnlib.hedging — regularized & constrained hedging

Two **extra** hedge builders that sit alongside the closed-form solvers
(`solve` / `lstsq` / `pinv`) described in [`HEDGING_README.md`](HEDGING_README.md):

| function | method | idea |
|---|---|---|
| `hedge_ridge(...)` | `'ridge'` | Tikhonov / ridge regression (`scikit-learn`) |
| `hedge_optimize(...)` | `'optimize'` | constrained optimization (`scipy.optimize`) |

They exist for the cases where a raw linear solve produces **huge,
mutually-offsetting trades** or fails outright. Both are kept deliberately
separate from `hedge()` — they are not reachable via `hedge(method=...)`.

```python
from optnlib import hedging as h
```

> `hedge_ridge` needs `scikit-learn` (`pip install optnlib[ridge]`). The base
> `optnlib` install still only requires `numpy` + `scipy`; the import is lazy.
> `hedge_optimize` uses `scipy.optimize`, already a core dependency.

---

## Why bother? A motivating example

Take a short-call portfolio and hedge it with the underlying plus two options
that share the same *expiry and vol*:

```python
pf = h.Portfolio([
    h.OptionLeg(quantity=-10, flag='c', K=105, multiplier=100,
                S=100, t=0.25, r=0.05, sigma=0.25, name='short call'),
    h.Underlying(quantity=500, S=100, name='stock'),
])

instruments = [
    h.Underlying(S=100, name='stock'),
    h.OptionLeg(flag='c', K=100, multiplier=100, S=100, t=0.50, r=0.05,
                sigma=0.28, name='H1'),
    h.OptionLeg(flag='p', K=95, multiplier=100, S=100, t=0.50, r=0.05,
                sigma=0.28, name='H2'),
]

greeks = ('delta', 'gamma', 'vega')
```

For Black–Scholes, `vega / gamma = S² σ t` is the **same** for every option
sharing spot, vol and expiry. H1 and H2 share both, so the vega row is a scalar
multiple of the gamma row and the 3×3 matrix is **rank-2**. A naive
`numpy.linalg.solve` does not warn — it silently returns garbage:

```
unscaled solve  ->  w = [-3.30e18, 3.54e16, -3.88e16]     # nonsense
```

`hedge()` already guards this (it falls back to `lstsq`). Ridge and constrained
optimization give you two more knobs:

* **ridge** trades a little hedging accuracy for much smaller trades, and has a
  closed form even when the matrix is singular;
* **optimize** lets you impose hard **position limits** and a **gross-exposure**
  cap, and to weigh **transaction costs** — so the trades can *never* explode.

---

## 1. Ridge hedging — `hedge_ridge`

Ridge (Tikhonov) solves

```
w = argmin  ||A w − rhs||²  +  α ||w||²
```

The penalty makes the problem strictly convex, so a **unique** solution exists
even when `A` is singular, and large offsetting weights are discouraged.

```python
plan = h.hedge_ridge(
    pf, instruments,
    greeks=('delta', 'gamma', 'vega'),
    alpha='auto',        # or a number, e.g. 1e-4
)
print(plan)
```

`alpha='auto'` (default) picks the penalty by leave-one-out generalized
cross-validation (`sklearn.linear_model.RidgeCV`). Pass a float for a fixed
penalty.

### The accuracy ↔ trade-size dial

On the ill-conditioned example above (row-scaled system):

| `alpha` | max &#124;w&#124; | &#124;&#124;w&#124;&#124;₂ | scaled residual |
|---|---|---|---|
| `'auto'` (≈0.41) | 6.57 | 7.34 | 99.0 |
| `1e-1` | 8.22 | 8.87 | 107.1 |
| `1e-2` | 8.90 | 9.48 | 119.0 |
| `1e-4` | 8.98 | 9.55 | 120.8 |
| `1e-6` | 8.98 | 9.55 | 120.8 |

Reading it: as `alpha → 0` the weights approach the `lstsq` minimum-norm
solution and the residual flattens at the **irreducible** level (the system is
inconsistent — gamma/vega cannot both be neutralized). `alpha='auto'` chooses
the *smallest* trades, accepting more residual. This is exactly the trade-off
ridge is meant to expose: **pick the point you are comfortable with**.

Useful `plan` fields: `plan.quantities`, `plan.residual`, `plan.info['alpha']`,
`plan.method == 'ridge'`.

---

## 2. Constrained hedging — `hedge_optimize`

`hedge_optimize` minimizes the (row-scaled) hedging error plus an optional
transaction-cost term

```
min_w  ||A_s w − rhs_s||²  +  cost_weight · Σ cⱼ |wⱼ|
```

subject to **position bounds** and an optional **gross-exposure** cap
`Σ|wⱼ| ≤ gross_limit`. Because the weights are bounded by construction, the
trades can never explode and the solver **always** returns a usable answer,
even for an inconsistent system.

```python
plan = h.hedge_optimize(
    pf, instruments,
    greeks=('delta', 'gamma', 'vega'),
    bounds=5.0,             # |w_j| <= 5 for every instrument
    gross_limit=20.0,       # sum |w_j| <= 20
    costs=[0.01, 0.05, 0.05],   # per-unit cost: stock is cheapest
    cost_weight=50.0,       # how much cost matters vs. remaining risk
)
print(plan)
```

```text
Solver  : optimize  (rank 2/3, cond 6.70e+18, row-scaled)
Optimize: SLSQP, success=True, iters=..., objective=...
```

* `bounds=None` (default) → unbounded; the optimizer then reproduces the
  default `lstsq` minimum-norm hedge exactly.
* `bounds=L` (scalar) → symmetric cap `±L` on every instrument.
* `bounds=[(lo, hi), ...]` (per instrument, `None` allowed) for asymmetric caps.
* `costs` + `cost_weight` penalize trading expensive instruments.
* `method='SLSQP'` (default), plus `maxiter`, `ftol`, `x0`.

`plan.info` reports `solver`, `success`, `message`, `nit`, `objective`.

For an inconsistent system, a tight bound deliberately leaves residual greek
risk (it is reported honestly in `plan.residual`); loosen `bounds` / lower
`cost_weight` to neutralize more.

---

## 3. Comparison

All five hedge routes on the same (rank-deficient) problem:

| route | call | safe when singular? | small trades? | extra features |
|---|---|---|---|---|
| `solve` | `h.hedge(...)` (auto) | ❌ (guarded → falls back) | — | fastest |
| `lstsq` | `h.hedge(...)` (auto) | ✅ min-norm | ✅ | any shape |
| `pinv` | `h.hedge(..., method='pinv')` | ✅ | ✅ | reusable operator |
| `ridge` | `h.hedge_ridge(...)` | ✅ unique | ✅✅ (tunable α) | α dial, sklearn |
| `optimize` | `h.hedge_optimize(...)` | ✅ always finite | ✅✅ | bounds, gross cap, costs |

Rule of thumb: use plain `h.hedge(...)` first — it already detects rank
deficiency and ill-conditioning. Reach for `hedge_ridge` when you want
explicitly smaller trades, and for `hedge_optimize` when you have *real-world
limits* (position caps, gross limits, transaction costs) the closed forms
cannot express.

---

## 4. Notes & caveats

* Both functions row-scale the matrix (`use_scaling=True`, default) before
  solving, so greeks on wildly different scales are treated comparably; pass
  `use_scaling=False` to disable.
* `alpha='auto'` optimizes a *prediction* criterion (GCV). On very small
  greek systems it can prefer noticeably more shrinkage than a trading desk
  would. Inspect the sweep table above and pass an explicit `alpha` if you want
  to stay closer to the exact (minimum-norm) hedge.
* The reported `residual` is always in the original, **unscaled** units.
* These are still *local, instantaneous* neutralizations; rebalance as
  spot/vol/time move (see `Portfolio.scenario` / `pnl_attribution`).

See [`HEDGING_EXAMPLES.md`](HEDGING_EXAMPLES.md) for the closed-form solver
examples and [`HEDGING_README.md`](HEDGING_README.md) for the full guide.
