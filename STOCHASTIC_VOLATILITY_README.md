# Stochastic Volatility & Jump Models

`optnlib.stochvol` adds stochastic-volatility and jump-diffusion models to
`optnlib`. The module shares one interface across every model:

- `model.price(...)` — European price.
- `model.price_mc(...)` — Monte Carlo price with a standard error.
- `model.simulate(...)` — simulated spot and variance paths.
- `model.characteristic_function(...)` — available for Fourier-priced models.
- `model.price_fourier(...)` — Fourier price where available.
- `model.price_closed_form(...)` — closed form where available.

Phase 1 is implemented. Phase 2 advanced models are listed at the end.

## Included models

| Class | Model | Default pricing | Variance state |
| --- | --- | --- | --- |
| `Heston` | Heston (1993) | Fourier | Yes |
| `HullWhite` | Hull-White (1987), geometric variance | Monte Carlo | Yes |
| `SteinStein` | Stein-Stein (1991) | Monte Carlo | Yes |
| `Scott` | Scott exponential OU | Monte Carlo | Yes |
| `ThreeHalves` | 3/2 stochastic volatility | Monte Carlo | Yes |
| `SABR` | SABR with Hagan implied vol | Hagan formula | Yes |
| `Merton` | Merton (1976) jump diffusion | Fourier | No |
| `Kou` | Kou (2002) double-exponential jumps | Fourier | No |
| `Bates` | Bates (1996), Heston plus jumps | Fourier | Yes |

## Quick start

```python
from optnlib import stochvol as sv

heston = sv.Heston(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

# Fourier price (default because Heston has a characteristic function)
call = heston.price('c', S=100, K=100, t=1.0, r=0.05)

# Monte Carlo price plus standard error
mc = heston.price_mc('c', S=100, K=100, t=1.0, r=0.05,
                     n_steps=252, n_sims=100_000, seed=42)
print(mc.price, mc.std_error, mc.confidence_interval)

# Simulated paths
paths = heston.simulate(100, 1.0, 0.05, n_steps=252, n_sims=10_000, seed=42)
paths.plot(K=100, t=1.0, save='heston_paths.png')
paths.plot_variance(save='heston_variance.png')
```

## Model catalogue

### Heston

```text
dS/S = (r - q) dt + sqrt(v) dW_S
dv   = kappa (theta - v) dt + xi sqrt(v) dW_v
dW_S dW_v = rho dt
```

Parameters: `v0` initial variance, `kappa` mean-reversion speed, `theta`
long-run variance, `xi` vol-of-vol, `rho` correlation.

`Heston.feller_satisfied` reports whether `2 kappa theta >= xi^2`. The model
still simulates when the condition fails, but the variance can hit zero and the
QE scheme is recommended.

### Hull-White (1987)

```text
dS/S = (r - q) dt + sqrt(v) dW_S
dv   = mu_v v dt + sigma_v v dW_v
dW_S dW_v = rho dt
```

This is the stochastic-volatility model from Hull and White (1987), with
geometric variance diffusion. Parameters: `v0`, `mu_v`, `sigma_v`, `rho`.

### Stein-Stein

```text
dS/S = (r - q) dt + |Y| dW_S
dY   = kappa (theta - Y) dt + xi dW_Y
dW_S dY = rho dt
```

`Y` is a signed volatility state and `theta` is its long-run level. Parameters:
`sigma0`, `kappa`, `theta`, `xi`, `rho`.

### Scott

```text
dS/S = (r - q) dt + exp(Y / 2) dW_S
dY   = kappa (log(theta) - Y) dt + xi dW_Y
dW_S dY = rho dt
```

`theta` is the long-run variance and `v0` is the initial variance. Parameters:
`v0`, `kappa`, `theta`, `xi`, `rho`.

### 3/2 model

```text
dS/S = (r - q) dt + sqrt(v) dW_S
dv   = kappa v (theta - v) dt + xi v^(3/2) dW_v
dW_S dW_v = rho dt
```

Parameters: `v0`, `kappa`, `theta`, `xi`, `rho`. Phase 1 uses Euler or
Milstein with full truncation for the variance.

### SABR

```text
dF     = alpha F^beta dW_F
dalpha = nu alpha dW_alpha
dW_F dW_alpha = rho dt
```

Parameters: `alpha`, `beta`, `rho`, `nu`. `price` uses Hagan's asymptotic
implied-volatility formula and Black-76. `price_mc` simulates the forward and
is a diagnostic; SABR is not a complete martingale model in the same sense as
Heston.

`SABR.implied_vol(F, K, t)` takes a **forward** `F`.

### Merton jump diffusion

```text
dS/S = (r - q - lambda m) dt + sigma dW + (e^J - 1) dN
J ~ Normal(mu_j, sigma_j^2)
m = E[e^J] - 1
```

Parameters: `sigma`, `lam`, `mu_j`, `sigma_j`. Pricing is available by Fourier,
the Merton jump series (`price_closed_form`), and Monte Carlo.

### Kou jump diffusion

```text
dS/S = (r - q - lambda m) dt + sigma dW + (e^J - 1) dN
```

`J` is double exponential: upward jumps occur with probability `p` and mean
`1/eta1`; downward jumps occur with probability `1-p` and mean `1/eta2`.
Parameters: `sigma`, `lam`, `p`, `eta1`, `eta2`, with `eta1 > 1` and
`eta2 > 0`.

### Bates

Bates combines Heston stochastic variance with Merton normal log jumps.
Parameters are the Heston parameters plus `lam`, `mu_j`, and `sigma_j`.

## Simulation schemes

The scheme argument is optional; each model has a default.

| Scheme | Where it is used | Notes |
| --- | --- | --- |
| `euler` | Heston, Bates, Hull-White, Stein-Stein, Scott, 3/2, SABR | Simple Euler-Maruyama. Variance schemes use full truncation. |
| `milstein` | Heston, Bates, 3/2 | Adds the Milstein correction; useful when vol-of-vol is high. |
| `qe` | Heston, Bates | Andersen quadratic-exponential scheme for CIR variance. Keeps variance non-negative and is the recommended Heston default. |
| `lognormal` | Hull-White | Exact lognormal step for geometric variance. |
| `exact` | Stein-Stein, Scott, Merton, Kou | Exact Gaussian OU steps or exact compound-Poisson jumps. |

### Which scheme should I use?

1. **Heston or Bates variance:** use `qe` unless you are studying discretization
   bias. Use `milstein` for stronger convergence when `xi` is high. Use `euler`
   for a quick baseline.
2. **Stein-Stein or Scott:** use the default exact OU step.
3. **Hull-White:** use the default exact lognormal step.
4. **3/2:** use `euler` for stability; `milstein` adds a correction but the
   variance can still be sensitive to step size.
5. **SABR:** Euler/log-Euler is the only Phase 1 scheme; reduce `dt` when `nu`
   is large.
6. **Merton and Kou:** jumps are sampled exactly, so there is no jump
   discretization bias.

### Scheme support matrix

| Model | `euler` | `milstein` | `qe` | `lognormal` | `exact` |
| --- | --- | --- | --- | --- | --- |
| Heston | Yes | Yes | Yes | No | No |
| Bates | Yes | Yes | Yes | No | No |
| Hull-White | Yes | No | No | Yes | No |
| Stein-Stein | Yes | No | No | No | Yes |
| Scott | Yes | No | No | No | Yes |
| ThreeHalves | Yes | Yes | No | No | No |
| SABR | Yes | No | No | No | No |
| Merton | No | No | No | No | Yes |
| Kou | No | No | No | No | Yes |

## Pricing methods

### Fourier pricing

Heston, Merton, Kou, and Bates expose a characteristic function of
`log(S_t / S_0)`. `price` uses the default `method='auto'`, which selects
Fourier pricing for these models. Calls and puts are both supported, including
a continuous dividend yield `q`.

```python
from optnlib import stochvol as sv

model = sv.Merton(sigma=0.2, lam=0.5, mu_j=-0.1, sigma_j=0.15)
fourier = model.price('c', 100, 100, 1.0, 0.05)
closed = model.price('c', 100, 100, 1.0, 0.05, method='closed-form')
```

Merton also provides `price_closed_form(...)`, the standard Poisson jump series.

### Monte Carlo pricing

`price_mc` returns an `MCResult`:

```python
result = sv.Heston().price_mc(
    'c', 100, 100, 1.0, 0.05,
    n_steps=252, n_sims=100_000, scheme='qe', seed=42,
)

result.price          # estimate
result.std_error      # standard error
result.confidence_interval
```

Monte Carlo error decreases like `1 / sqrt(n_sims)`. Increase `n_sims` for
precision and `n_steps` when discretization bias matters.

## SABR

```python
sabr = sv.SABR(alpha=0.3, beta=1.0, rho=-0.4, nu=0.5)

# Hagan price (spot is converted to a forward)
price = sabr.price('c', 100, 100, 1.0, 0.05)

# Hagan implied volatility on a forward
iv = sabr.implied_vol(F=105.127, K=100, t=1.0)
```

SABR's Hagan formula is asymptotic. It is accurate for moderate strikes and
tenors. Monte Carlo is useful for a sanity check, but it does not replace a
proper calibration or an arbitrage-free treatment of the SABR dynamics.

## Visualization

```python
paths = sv.Bates().simulate(100, 1.0, 0.05, n_steps=252, n_sims=10_000, seed=42)

paths.plot(K=100, t=1.0, n_paths=100, save='bates_paths.png')
paths.plot_variance(n_paths=100, save='bates_variance.png')
```

Plotting uses matplotlib lazily. Install it with `pip install "optnlib[viz]"` if
needed.

## Validation and limitations

- Phase 1 prices **European options only**. American stochastic-volatility
  pricing is Phase 2.
- Discrete dividend schedules are not supported by `stochvol` yet; use the
  continuous yield `q`.
- Heston, Bates, Merton, and Kou use Fourier pricing by default. Hull-White,
  Stein-Stein, Scott, and 3/2 use Monte Carlo; SABR uses Hagan's formula.
- SABR pricing is asymptotic, not an exact solution.
- The 3/2 model uses a truncated Euler/Milstein discretization and can be
  sensitive to step size for extreme parameters.
- The Hull-White implementation here uses geometric variance, not an
  interest-rate Hull-White model.
- Calibration to market implied-vol surfaces is Phase 2.
- No rough volatility, LSV, Bergomi, Double Heston, or Levy models are included
  in Phase 1.

## Phase 2 roadmap

- Double Heston.
- Bergomi and rough Bergomi.
- Rough Heston.
- LSV.
- Variance Gamma, NIG, and CGMY.
- Calibration to `optnlib.volatility` surfaces.
- American stochastic-volatility pricing.
- Additional variance-reduction and Greeks estimators.

## References

- Heston, S. (1993). A closed-form solution for options with stochastic
  volatility with applications to bond and currency options.
- Hull, J., and White, A. (1987). The pricing of options on assets with
  stochastic volatilities.
- Stein, E., and Stein, J. (1991). Stock price distributions with stochastic
  volatility: an analytic approach.
- Scott, L. (1987). Option pricing when the variance changes randomly.
- Hagan, P. et al. (2002). Managing smile risk.
- Merton, R. (1976). Option pricing when underlying stock returns are
  discontinuous.
- Kou, S. (2002). A jump-diffusion model for option pricing.
- Bates, D. (1996). Jumps and stochastic volatility.
- Andersen, L. (2008). Simple and efficient simulation of the Heston
  stochastic volatility model.
