# optnlib.monte_carlo — Monte Carlo option pricing

`optnlib.monte_carlo` prices options by simulating risk-neutral geometric
Brownian motion (GBM) paths, evaluating the payoff, and discounting the average
payoff back to today.

The current implementation covers **European vanilla calls and puts with a
continuous dividend yield**. It is intentionally small: simulation, pricing,
and matplotlib path visualization. More advanced option families are described
below with their current status so you can see where they fit and what would be
needed to add them.

> **Current scope:** European calls and puts, GBM, continuous yield `q`,
> configurable `n_steps` and `n_sims`, reproducible `seed`, and optional
> matplotlib path plots. American exercise, discrete cash dividends,
> path-dependent payoffs, and multi-asset options are **not implemented yet**.

## Install / import

```bash
pip install .
```

For plotting, install the visualization extra:

```bash
pip install "optnlib[viz]"
```

Then import the module:

```python
from optnlib import monte_carlo as mc
```

The top-level convenience alias is also available:

```python
import optnlib

optnlib.monte_carlo_price(
    'c', 100, 100, 1.0, 0.05, 0.2, seed=42
)
```

## Option types at a glance (before pricing)

Monte Carlo is a general numerical method: simulate paths, compute the payoff,
discount the average. Different option families differ in **when exercise is
allowed** and **how the payoff is calculated**. The list below covers the
common families; many real products combine several of these features.

### By exercise style

| Type | Brief explanation | Monte Carlo implication | Status |
| --- | --- | --- | --- |
| **European** | Exercise is allowed only at expiry. | Payoff depends on the terminal price `S_T`; discount its average. | **Supported** for calls and puts. |
| **American** | Exercise is allowed at any time up to expiry. | Requires an optimal early-exercise policy, typically Longstaff-Schwartz regression. | Not yet supported. |
| **Bermudan** | Exercise is allowed only on a finite set of dates. | Requires backward induction across the exercise dates. | Not yet supported. |

### By payoff direction

| Type | Payoff at expiry | Brief explanation | Status |
| --- | --- | --- | --- |
| **Call** | `max(S_T - K, 0)` | Right to buy the underlying at strike `K`. | **Supported.** |
| **Put** | `max(K - S_T, 0)` | Right to sell the underlying at strike `K`. | **Supported.** |

### By payoff shape

| Type | Brief explanation | Status |
| --- | --- | --- |
| **Digital / binary** | Pays a fixed cash amount (cash-or-nothing) or the underlying value (asset-or-nothing) if a condition is met. | Not yet supported. |
| **Gap** | Pays only if the terminal price crosses a trigger, with a payoff measured from a different strike. | Not yet supported. |
| **Power** | Payoff depends on the underlying raised to a power, such as `max(S_T**p - K, 0)`. | Not yet supported. |
| **Capped / floored** | The payoff is limited above or below by a cap or floor. | Not yet supported. |

### Path-dependent options

| Type | Brief explanation | Monte Carlo implication | Status |
| --- | --- | --- | --- |
| **Asian** | Payoff uses the average underlying price over the life of the option instead of only `S_T`. | Requires the full simulated path and an averaging rule (arithmetic or geometric). | Not yet supported. |
| **Barrier** | The option is activated (knock-in) or cancelled (knock-out) when the price crosses a barrier. | Requires checking the path for barrier crossing. | Not yet supported. |
| **Lookback** | Payoff uses the maximum or minimum price reached over the path. | Requires path extrema. | Not yet supported. |
| **Cliquet / ratchet** | A series of forward-starting returns with caps and floors; the payoff accumulates over reset dates. | Requires path segments and reset-date cashflows. | Not yet supported. |

### Multi-asset options

| Type | Brief explanation | Monte Carlo implication | Status |
| --- | --- | --- | --- |
| **Spread** | Payoff depends on the difference between two assets, e.g. `max(S1_T - S2_T - K, 0)`. | Simulate correlated assets. | Not yet supported. |
| **Basket** | Payoff depends on a weighted sum of several assets. | Simulate the joint distribution of the assets. | Not yet supported. |
| **Rainbow / best-of / worst-of** | Payoff depends on the best or worst performer in a group. | Simulate correlated assets and compare them at expiry. | Not yet supported. |
| **Quanto** | Payoff is converted from one currency to another at a fixed rate. | Requires an additional FX process and correlation assumptions. | Not yet supported. |

### Structured and multi-stage options

| Type | Brief explanation | Status |
| --- | --- | --- |
| **Compound** | An option whose underlying is another option; exercise decisions occur at multiple dates. | Not yet supported. |
| **Chooser** | The holder chooses at a future date whether the option becomes a call or a put. | Not yet supported. |
| **Forward-start** | The strike is set at a future date rather than today. | Not yet supported. |
| **Auto-callable** | The option terminates early when a trigger is hit, often with a coupon. | Not yet supported. |

### Dividend variants

| Type | Brief explanation | Status |
| --- | --- | --- |
| **Continuous dividend yield** | The underlying is assumed to pay dividends continuously at rate `q`. | **Supported** through the `q` parameter. |
| **Discrete cash dividends** | Specific cash amounts are paid on specific ex-dividend dates. | Not yet supported in the Monte Carlo engine. Use the tree engines for discrete schedules. |
| **Dividend futures / forwards** | Pricing is performed on a dividend-adjusted forward. | Use `optnlib.black` for the deterministic Black-76 route; not part of this Monte Carlo module. |

### What the status means

- **Supported** means the function described in this README can price that
  option today.
- **Not yet supported** means the current API does not implement it. The
  roadmap section lists the natural extension points.

For European vanillas, the deterministic Black-Scholes/BSM engines are faster
and more accurate. Monte Carlo becomes especially useful when a payoff depends
on the whole path or on several correlated assets.

## Quick start

### Price a European call

```python
from optnlib import monte_carlo as mc

call = mc.price(
    'c', 100, 100, 1.0, 0.05, 0.2,
    n_steps=252, n_sims=10_000, seed=42,
)
print(call)
# 10.557482096687831
```

For comparison, the analytic Black-Scholes value is:

```python
import optnlib

optnlib.black_scholes('c', 100, 100, 1.0, 0.05, 0.2)
# 10.450583572185565
```

The Monte Carlo estimate differs from the analytic value because it is based on
a finite sample of random paths.

### Price a European put

```python
put = mc.price(
    'p', 100, 100, 1.0, 0.05, 0.2,
    n_steps=252, n_sims=10_000, seed=42,
)
print(put)
# 5.500993316830674
```

The analytic Black-Scholes put value is `5.573526022256971`.

### Simulate and plot paths

```python
paths = mc.simulate(
    100, 1.0, 0.05, 0.2,
    n_steps=252, n_sims=1_000, seed=42,
)
print(paths.shape)
# (1000, 253)

fig = mc.plot_paths(
    paths, K=100, t=1.0,
    n_paths=100, save='mc_paths.png',
)
# Returns a matplotlib Figure and saves mc_paths.png
```

`paths[:, 0]` is the initial price `S`. Each row is one simulated path and each
column is one time step.

## How the pricing works

Under the risk-neutral measure with continuous dividend yield `q`, the
underlying follows

```text
dS_t = (r - q) S_t dt + sigma S_t dW_t
```

The simulator uses the exact GBM step

```text
S_{t+dt} = S_t * exp((r - q - sigma**2 / 2) * dt + sigma * sqrt(dt) * Z)
```

where `Z` is a standard normal random variable and `dt = t / n_steps`. The
terminal value is therefore lognormal:

```text
S_T = S_0 * exp((r - q - sigma**2 / 2) * T + sigma * sqrt(T) * Z)
```

For a European option with payoff `payoff(S_T)`, the Monte Carlo estimator is

```text
V_MC = exp(-r * T) * (1 / N) * sum_i payoff(S_T_i)
```

where `N = n_sims`. The estimate has sampling error

```text
SE = std(discounted_payoffs, ddof=1) / sqrt(N)
```

so the error shrinks like `1 / sqrt(N)`: quadrupling the number of simulations
halves the standard error.

### `n_steps` versus `n_sims`

- **`n_sims`** is the number of independent paths. It controls sampling error.
- **`n_steps`** is the number of time steps in each path. It controls path
  resolution and the number of random increments.

For a European vanilla priced from the terminal value only, the exact GBM
discretization gives the correct terminal distribution for any `n_steps`.
Therefore `n_steps` does **not** remove pricing bias for European calls and
puts; it mainly controls the shape of the simulated paths and becomes important
for path-dependent extensions such as Asian or barrier options.

`n_steps=252` means roughly daily steps over one year. `n_steps=1` stores only
the initial and terminal prices.

## API reference

### `simulate`

```python
simulate(S, t, r, sigma, q=0.0, n_steps=252, n_sims=10_000, seed=None)
```

Simulate GBM paths for the underlying.

| Parameter | Meaning |
| --- | --- |
| `S` | Initial underlying price (`> 0`). |
| `t` | Time horizon in years (`>= 0`). |
| `r` | Annualized continuously compounded risk-free rate. |
| `sigma` | Annualized volatility (`>= 0`). |
| `q` | Annualized continuous dividend yield (default `0.0`). |
| `n_steps` | Number of time steps per path (default `252`). |
| `n_sims` | Number of simulated paths (default `10,000`). |
| `seed` | Optional seed for `numpy.random.default_rng`. Use a value for reproducibility. |

Returns a NumPy array of shape `(n_sims, n_steps + 1)`. Column `0` contains the
initial price `S`; the remaining columns are the simulated prices at each time
step.

`n_steps` and `n_sims` must be positive integers. Invalid values raise
`optnlib.exceptions.InvalidInputError`.

### `price`

```python
price(flag, S, K, t, r, sigma, q=0.0, n_steps=252, n_sims=10_000, seed=None)
```

Price a European call or put.

| Parameter | Meaning |
| --- | --- |
| `flag` | `'c'`/`'call'` for a call or `'p'`/`'put'` for a put. |
| `S` | Initial underlying price (`> 0`). |
| `K` | Strike price (`> 0`). |
| `t` | Time to expiry in years (`>= 0`). |
| `r` | Annualized continuously compounded risk-free rate. |
| `sigma` | Annualized volatility (`>= 0`). |
| `q` | Annualized continuous dividend yield (default `0.0`). |
| `n_steps` | Number of time steps per path (default `252`). |
| `n_sims` | Number of simulated paths (default `10,000`). |
| `seed` | Optional seed for reproducible results. |

Returns the discounted mean terminal payoff as a `float`.

Special cases:

- `t == 0` returns the intrinsic value immediately, without simulating.
- `sigma == 0` produces deterministic paths and a deterministic discounted
  payoff.
- The estimator is based only on the terminal price for European calls and puts.

### `plot_paths`

```python
plot_paths(paths, K=None, t=None, n_paths=100, ax=None, save=None,
           show=False, title=None, figsize=(10, 5))
```

Plot simulated paths with matplotlib.

| Parameter | Meaning |
| --- | --- |
| `paths` | Array returned by `simulate`, shape `(n_sims, n_steps + 1)`. |
| `K` | Optional strike price; drawn as a horizontal dashed line. |
| `t` | Optional time horizon in years. When supplied, the x-axis is in years. |
| `n_paths` | Maximum number of paths to draw (default `100`). |
| `ax` | Optional existing matplotlib `Axes`. |
| `save` | Optional path to write the figure to. |
| `show` | If `True`, call `matplotlib.pyplot.show()`. |
| `title` | Optional plot title. |
| `figsize` | Figure size (default `(10, 5)`). |

Returns the matplotlib `Figure`.

Matplotlib is imported lazily. Pricing works with NumPy alone; installing
`optnlib[viz]` provides matplotlib for plotting.

### Top-level alias

```python
import optnlib

optnlib.monte_carlo_price(...)  # same as optnlib.monte_carlo.price(...)
```

## Worked examples

### European call with a continuous dividend yield

```python
from optnlib import monte_carlo as mc

call_q = mc.price(
    'c', 100, 100, 1.0, 0.05, 0.2,
    q=0.03, seed=42,
)
print(call_q)
# 8.754457408861414

# Analytic comparison
import optnlib
optnlib.black_scholes_merton('c', 100, 100, 1.0, 0.05, 0.2, q=0.03)
# 8.652528553942709
```

A higher dividend yield lowers the expected growth rate of the underlying and
therefore lowers the call value.

### Reproducible simulations

```python
a = mc.simulate(100, 1.0, 0.05, 0.2, n_steps=12, n_sims=100, seed=7)
b = mc.simulate(100, 1.0, 0.05, 0.2, n_steps=12, n_sims=100, seed=7)
c = mc.simulate(100, 1.0, 0.05, 0.2, n_steps=12, n_sims=100, seed=8)

assert (a == b).all()       # same seed: identical paths
assert not (a == c).all()   # different seed: different paths
```

The seed only controls the pseudo-random sequence. It does not change the
model, the expected price, or the accuracy of the estimator.

### Estimate the standard error

`price()` returns only the point estimate. To compute the sampling error, use
the simulated terminal prices directly:

```python
import numpy as np
from optnlib import monte_carlo as mc

S, K, t, r, sigma = 100, 100, 1.0, 0.05, 0.2
paths = mc.simulate(S, t, r, sigma, seed=42)
discounted = np.exp(-r * t) * np.maximum(paths[:, -1] - K, 0.0)

estimate = discounted.mean()
standard_error = discounted.std(ddof=1) / np.sqrt(discounted.size)
print(estimate, standard_error)
# 10.557482096687831 0.1494107528299081

print(estimate - 1.96 * standard_error, estimate + 1.96 * standard_error)
# 10.264637021141212 10.85032717223445
```

The analytic Black-Scholes value `10.450583572185565` lies inside the
approximate 95% confidence interval.

### Plot on an existing matplotlib axes

```python
import matplotlib.pyplot as plt
from optnlib import monte_carlo as mc

paths = mc.simulate(100, 1.0, 0.05, 0.2, n_steps=252, n_sims=500, seed=42)

fig, ax = plt.subplots(figsize=(11, 5))
mc.plot_paths(paths, K=100, t=1.0, n_paths=50, ax=ax)
ax.set_title('GBM paths with a 100 strike')
fig.savefig('custom_mc_paths.png', dpi=150, bbox_inches='tight')
```

## Accuracy, steps, simulations, and memory

### Choosing `n_sims`

| `n_sims` | Typical use |
| --- | --- |
| `1,000` | Quick demos and plots. |
| `10,000` | Default; fast estimates with visible sampling noise. |
| `100,000` | Routine pricing and comparisons. |
| `1,000,000` | Higher accuracy; slower and more memory-intensive. |

Because the standard error scales as `1 / sqrt(n_sims)`, reducing the error by a
factor of 10 requires about 100 times as many paths.

### Choosing `n_steps`

- `252` — daily steps over a one-year horizon (the default).
- `12` — monthly steps.
- `1` — only the initial and terminal prices.
- More steps improve path resolution for visualization and future
  path-dependent payoffs, but do not improve the terminal distribution of a
  European vanilla under exact GBM simulation.

### Memory

The simulated array has shape `(n_sims, n_steps + 1)` and uses float64 values.
Approximate memory is:

```text
n_sims * (n_steps + 1) * 8 bytes
```

Examples:

- `10,000 x 253` ≈ 20 MB.
- `100,000 x 253` ≈ 200 MB.
- `1,000,000 x 253` ≈ 2 GB.

If you only need the terminal value for a European vanilla, use a small
`n_steps` to reduce memory. `price()` still allocates the full path array
internally because it uses `simulate()`.

## Limitations and roadmap

The following features are natural extensions but are **not currently
implemented**:

- **American options** — Longstaff-Schwartz regression estimates the optimal
  early-exercise policy. This is approximate and has a small downward bias.
- **Bermudan options** — backward induction over the allowed exercise dates.
- **Discrete cash dividends** — subtract each cash amount from every path at
  its ex-dividend step.
- **Custom payoff callbacks** — accept a user-provided function such as
  `payoff(paths)` so path-dependent payoffs can be priced without changing the
  simulator.
- **Asian, barrier, and lookback options** — evaluate the payoff from the full
  simulated path.
- **Multi-asset options** — simulate correlated assets using a correlation
  matrix or Cholesky decomposition.
- **Variance reduction** — antithetic variates, control variates, importance
  sampling, and stratification.
- **Quasi-Monte Carlo** — low-discrepancy sequences for faster convergence in
  some dimensions.
- **Monte Carlo Greeks** — bump-and-reprice, pathwise derivatives, or
  likelihood-ratio methods.
- **Standard error and confidence intervals in `price()`** — currently these
  must be computed from simulated paths manually.

The deterministic engines in the rest of `optnlib` remain the recommended choice
for European and American vanilla pricing where they apply.

## Testing

Run the Monte Carlo tests:

```bash
python -m pytest tests/test_monte_carlo.py -q
```

Run the full suite:

```bash
python -m pytest -q
```

The Monte Carlo tests use fixed seeds so results are reproducible.

## See also

- Root pricing guide: [`README.md`](README.md)
- General examples: [`EXAMPLES.md`](EXAMPLES.md)
- Source: [`optnlib/monte_carlo.py`](optnlib/monte_carlo.py)
- Tests: [`tests/test_monte_carlo.py`](tests/test_monte_carlo.py)
