# Model Selection Guide

This guide answers one question: **which implemented `optnlib` model should I
use?** It covers the deterministic European/American engines, Monte Carlo, and
the Phase 1 stochastic-volatility and jump models. It does not replace the API
guides.

Use it before writing a pricer. If a model is not listed here, it is not
implemented in the current version.

## Quick decision tree

```text
Is the option American?
|
+-- Yes
|   |
|   +-- Discrete cash dividends?
|   |   |
|   |   +-- Yes --> Leisen-Reimer (`lr`) tree
|   |   +-- No  --> Need maximum accuracy?
|   |               +-- Yes --> Leisen-Reimer (`lr`)
|   |               +-- No  --> Bjerksund-Stensland 2002 (`bs2002`)
|   |
|   +-- Need a benchmark / convergence check?
|       --> Cox-Ross-Rubinstein (`crr`)
|
+-- No (European)
    |
    +-- Known forward / futures option?
    |   --> Black-76 (`black`)
    |
    +-- Discrete cash dividends?
    |   --> `european_price(..., divs=...)`
    |
    +-- Only a single implied volatility, no smile needed?
    |   --> Black-Scholes or Black-Scholes-Merton
    |
    +-- Need a smile or skew?
        |
        +-- Fast forward-smile fit, especially rates/FX?
        |   --> SABR (Hagan)
        |
        +-- Need persistent stochastic variance?
        |   |
        |   +-- Equity/index-style correlated variance?
        |   |   --> Heston (Fourier by default)
        |   |
        |   +-- Need a different variance dynamic?
        |       --> Hull-White, Stein-Stein, Scott, or 3/2 (Monte Carlo)
        |
        +-- Need jumps?
            |
            +-- Symmetric Gaussian jumps?
            |   --> Merton (Fourier or closed form)
            |
            +-- Asymmetric double-exponential jumps?
            |   --> Kou (Fourier)
            |
            +-- Jumps plus stochastic variance?
                --> Bates (Fourier)
```

## First question: exercise style

| Requirement | Use | Do not use |
| --- | --- | --- |
| European exercise, no early-exercise premium | `black_scholes`, `black_scholes_merton`, `black`, `european_price` | American engines |
| American exercise, continuous yield | `american_price(..., method='bs2002')` as the fast default | European closed forms |
| American exercise, high accuracy | `american_price(..., method='lr')` | `bs2002` as a precision benchmark |
| American exercise, validation / convergence | `american_price(..., method='crr')` | `crr` for routine production pricing |
| American exercise with discrete cash dividends | `lr` or `crr` with `divs=...` | `bs2002` as the final answer |

`bs2002` is fast and useful for screening, but it is an approximation and a
lower bound for American puts. Use `lr` when the early-exercise premium matters,
and `crr` when you need a slow, transparent benchmark.

## Second question: dividends

| Dividend situation | Model |
| --- | --- |
| No dividend | Black-Scholes |
| Continuous yield `q` | Black-Scholes-Merton or `european_price` |
| Known forward / futures price | Black-76 |
| Discrete cash schedule, European | `european_price(..., divs=...)` |
| Discrete cash schedule, American | `american_price(..., divs=..., method='lr'|'crr')` |
| Discrete schedule treated as escrowed cash | Set `dividend_model='escrowed'` |
| Only a continuous yield is available | BSM, `european_price`, or any American engine |

For discrete dividends, choose `dividend_model='forward'` when the listed cash
amounts are forward cashflows and `'escrowed'` when they are present values
removed from spot. The binomial engines drop cash at the ex-date step; the
BS2002 engine folds the schedule into an equivalent continuous yield, so it is
less suitable when dividend timing drives the early-exercise decision.

## Third question: do you need a smile?

If you have one volatility number per option and no smile requirement, stop at
Black-Scholes/BSM/Black-76. Stochastic-volatility and jump models add parameters
that must be estimated or calibrated; they are not automatically better for a
single price.

| Need | Preferred model | Why |
| --- | --- | --- |
| No smile; quote is a single implied vol | Black-Scholes / BSM / Black-76 | Exact under the model, fastest, fewest assumptions |
| Fast European smile, especially forward/rates/FX | SABR (Hagan) | Closed-form implied vol; easy per-expiry fit |
| Equity/index smile with persistent variance | Heston | Fourier pricing, correlated variance, mean reversion |
| Simple geometric variance process | Hull-White (1987) | Straightforward stochastic-vol dynamics |
| Signed mean-reverting volatility state | Stein-Stein | Volatility itself mean-reverts |
| Log-variance mean reversion | Scott | Positive variance by construction |
| Strong vol-of-vol / steep skew | 3/2 | Different variance scaling; Monte Carlo only |
| Sudden gaps or jump risk | Merton or Kou | Explicit jump component |
| Jumps plus stochastic variance | Bates | Both effects in one model |

## Model comparison

| Model | Best for | Main limitation | Default pricing |
| --- | --- | --- | --- |
| Black-Scholes | European, no dividends, single vol | No smile, no early exercise | Closed form |
| Black-Scholes-Merton | European, continuous yield | No smile, no early exercise | Closed form |
| Black-76 | Forward/futures options | No smile, no early exercise | Closed form |
| `european_price` | European with continuous or discrete dividends | No smile, no early exercise | Closed form |
| `bs2002` | Fast American approximation | Approximate; lower bound for puts | Closed form |
| `lr` | Accurate American trees | Slower; step-count sensitivity | Tree |
| `crr` | American benchmark/validation | Slowest tree; oscillation at low steps | Tree |
| `monte_carlo` | European GBM path simulation | No American exercise, no discrete dividends, sampling error | Monte Carlo |
| Heston | Correlated stochastic variance, equity smile | Calibration and Feller sensitivity | Fourier |
| Hull-White | Simple geometric variance | No mean reversion; MC only | Monte Carlo |
| Stein-Stein | Mean-reverting signed volatility | MC only; no closed form here | Monte Carlo |
| Scott | Positive log-variance dynamics | MC only; no closed form here | Monte Carlo |
| 3/2 | Steep skew / strong vol-of-vol | MC only; discretization-sensitive | Monte Carlo |
| SABR | Fast forward smile | Hagan is asymptotic, not exact | Hagan formula |
| Merton | Gaussian jump risk | Symmetric jumps may not fit skew alone | Fourier / closed form |
| Kou | Asymmetric jump risk | More parameters; calibration needed | Fourier |
| Bates | Jumps plus stochastic variance | Richest model, hardest to calibrate | Fourier |

## Choosing a stochastic-volatility model

### Heston

Choose Heston when you need a **dynamic, mean-reverting variance process** with
correlation between spot and variance, and you want a fast European price.
Typical use: equity/index smile, variance-risk sensitivity, scenario analysis.

Use Heston when:

- The smile persists across maturities.
- Spot/variance correlation is important.
- You need Fourier pricing or simulated variance paths.
- You can calibrate `v0`, `kappa`, `theta`, `xi`, and `rho`.

Avoid Heston when:

- You only need one implied vol.
- You need American exercise (not supported in Phase 1).
- The market is dominated by sudden jumps rather than diffusion.
- You cannot calibrate and are guessing parameters.

### Hull-White (1987)

Choose Hull-White when you want a simple stochastic-variance model with
geometric variance diffusion and no mean reversion. It is useful for teaching,
quick experiments, and comparing against Heston.

Avoid it when you need long-run variance mean reversion or a closed-form price;
Phase 1 provides Monte Carlo only.

### Stein-Stein

Choose Stein-Stein when the volatility state itself is signed and mean-reverting
and you want a simple alternative to Heston. It is Monte Carlo only here.

Avoid it when you need exact Fourier pricing or a model with a well-known
closed-form European price.

### Scott

Choose Scott when you want mean-reverting **log variance**, which keeps variance
positive by construction. It is Monte Carlo only here.

Avoid it when you need a fast closed-form price or a standard calibration
workflow.

### 3/2

Choose the 3/2 model when you need a variance process with a different
vol-of-vol scaling and steeper skew than Heston. It is Monte Carlo only here and
more sensitive to discretization.

Avoid it for routine production pricing unless you have a specific reason and
have checked convergence.

### SABR

Choose SABR when you need a **fast implied-vol smile** and a model that is easy
to fit per expiry. It is especially common for rates and FX forward smiles.

Use SABR when:

- You need an implied vol, not a full dynamic hedge model.
- You are fitting a single expiry or a small set of expiries.
- You can choose `beta` deliberately and check the fit.

Avoid SABR when:

- You need an exact arbitrage-free dynamic model.
- You need long-dated far-OTM accuracy without checking the asymptotic formula.
- You need American exercise.

SABR `price` converts spot to a forward and uses Black-76 with Hagan implied
volatility. `price_mc` is a diagnostic, not a calibration substitute.

## Choosing a jump model

### Merton

Choose Merton when jumps are approximately symmetric in log price and you want a
well-known closed-form/Fourier benchmark. It is a good first jump model.

Avoid Merton when the market smile is strongly asymmetric and jumps are
predominantly downward; use Kou or Bates instead.

### Kou

Choose Kou when you need **asymmetric upward and downward jumps** with
double-exponential tails. It is useful for equity skew and gap risk.

Avoid Kou when you do not have enough option data to identify the jump
parameters, or when a pure stochastic-volatility model already fits the smile.

### Bates

Choose Bates when you need **both stochastic variance and jumps**. It is the
richest Phase 1 model and the most demanding to calibrate.

Avoid Bates when a simpler model fits, when calibration data is sparse, or when
you need fast, robust parameter estimation without a dedicated calibration
workflow.

## Monte Carlo vs Fourier vs closed form

| Need | Use |
| --- | --- |
| European price under Heston, Merton, Kou, or Bates | Fourier default |
| Merton price with the jump series | `method='closed-form'` |
| Paths, exposures, or path-dependent experiments | `simulate` / `price_mc` |
| American exercise | Trees or BS2002; not the stochastic-volatility module |
| European GBM with no smile | `monte_carlo` or the closed forms |
| Reproducible simulation | Always pass `seed` |

Fourier pricing is generally much faster and more accurate than Monte Carlo for
the models that support it. Use Monte Carlo when you need simulated paths or
when no characteristic function is available. Never compare a Fourier price and
a Monte Carlo price without accounting for the Monte Carlo standard error.

## Simulation scheme selection

| Model | Recommended scheme | When to change |
| --- | --- | --- |
| Heston | `qe` | `milstein` for high vol-of-vol; `euler` for a baseline |
| Bates | `qe` | Same as Heston |
| Hull-White | `lognormal` | `euler` only for comparison |
| Stein-Stein | `exact` | `euler` only for comparison |
| Scott | `exact` | `euler` only for comparison |
| 3/2 | `euler` | `milstein` if convergence checks support it |
| SABR | `euler` | Reduce `dt` when `nu` is large |
| Merton | `exact` | Do not replace exact jumps with Euler |
| Kou | `exact` | Do not replace exact jumps with Euler |

For Heston and Bates, check the Feller condition and variance positivity. If the
variance frequently hits zero, QE or Milstein with a smaller step is usually
better than plain Euler.

## Speed and accuracy hierarchy

From fastest/least flexible to slowest/most flexible:

1. **Black-Scholes / BSM / Black-76** — closed form, exact under assumptions.
2. **BS2002** — fast American approximation.
3. **SABR Hagan** — fast asymptotic European smile.
4. **Fourier Heston / Merton / Kou / Bates** — fast under the model.
5. **Leisen-Reimer tree** — accurate American pricing.
6. **CRR tree** — transparent benchmark, slower.
7. **Monte Carlo** — flexible but sampling error; stochastic-volatility MC needs
   many paths and steps.

This ordering is not a ranking of model quality. A faster model can be the right
choice when its assumptions match the question.

## Use-case map

| Situation | Recommended starting point |
| --- | --- |
| Plain European equity call/put | Black-Scholes |
| European index option with continuous yield | Black-Scholes-Merton |
| Option on a future or known forward | Black-76 |
| European option with discrete cash dividends | `european_price(..., divs=...)` |
| American equity put, fast risk screen | `bs2002` |
| American equity put, accurate price | `lr` |
| American option with discrete dividends | `lr` with `divs=...` |
| Validate an American tree | `crr` |
| European smile in rates/FX | SABR |
| Equity/index stochastic volatility | Heston |
| Simple stochastic-variance experiment | Hull-White or Stein-Stein |
| Positive log-variance experiment | Scott |
| Strong vol-of-vol / steep skew | 3/2 |
| Symmetric jump risk | Merton |
| Asymmetric jump risk | Kou |
| Jumps plus stochastic variance | Bates |
| Path simulation for risk/stress | `monte_carlo` or `stochvol.simulate` |
| Implied vol from a price | `implied_volatility` |
| Greeks for a European option | `greeks` analytic BSM where applicable |
| Greeks for American/discrete dividends | `greeks` numerical |
| Hedge a portfolio | `hedging` |
| Build a smile/surface | `volatility` |

## Model risk checklist

Before trusting a price:

1. **Check the exercise style.** A European model cannot price an American
   option, and a stochastic-volatility model in Phase 1 cannot either.
2. **Check dividends.** A discrete cash schedule is not a continuous yield; use
   the dividend-aware engine and the correct dividend model.
3. **Check the pricing method.** Compare Fourier and Monte Carlo within the
   Monte Carlo standard error.
4. **Check limiting cases.** Heston with zero vol-of-vol should match
   Black-Scholes; Merton with zero jump intensity should match Black-Scholes;
   Bates with zero jumps should match Heston.
5. **Check convergence.** Increase `n_steps` and `n_sims`; if the answer moves
   materially, the discretization or sampling error is too large.
6. **Check parameter plausibility.** Heston `v0` and `theta` should be positive
   and `xi` nonnegative; SABR `alpha` positive, `beta` in `[0, 1]`, `|rho| < 1`,
   `nu >= 0`.
7. **Check Feller and variance paths.** A violated Feller condition is not
   automatically fatal, but variance hitting zero needs scheme attention.
8. **Check implied-vol consistency.** When computing implied vol, use the same
   `style` and `method` as the price.

## What is not implemented

Do not select these from the current library:

- Double Heston.
- Bergomi or rough Bergomi.
- Rough Heston.
- Local stochastic volatility (LSV).
- Variance Gamma, NIG, or CGMY.
- American stochastic-volatility pricing.
- Calibration helpers for Heston, Bates, SABR, Merton, or Kou.
- Discrete dividend schedules inside `stochvol`.

These are Phase 2 items. For now, use the closest Phase 1 model and document the
approximation.

## Where to go next

- [`README.md`](README.md) — installation, quick start, unified API.
- [`STOCHASTIC_VOLATILITY_README.md`](STOCHASTIC_VOLATILITY_README.md) — model
  equations, schemes, pricing methods, examples.
- [`MONTE_CARLO_README.md`](MONTE_CARLO_README.md) — GBM simulation and
  implementation notes.
- [`IMPLIED_VOLATILITY_README.md`](IMPLIED_VOLATILITY_README.md) — implied vol
  and smile construction.
- [`VOLATILITY_README.md`](VOLATILITY_README.md) — live/synthetic surfaces.
- [`HEDGING_README.md`](HEDGING_README.md) — portfolio hedging.
- [`EXAMPLES.md`](EXAMPLES.md) — runnable examples.
