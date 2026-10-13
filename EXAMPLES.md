# optnlib examples

Eighteen copy-pasteable examples. Each shows a **Parameters** block, the code,
and the expected output in a comment.

> All values below were produced with `optnlib==0.1.0` on CPython 3.14.
> Floating point output can differ in the last few digits across platforms.

---

## Example 1 — European call (Black-Scholes)

**Parameters**

- `flag = 'c'`
- `S = 100`, `K = 100`
- `t = 1.0`, `r = 0.05`, `sigma = 0.2`

```python
import optnlib

optnlib.black_scholes('c', 100, 100, 1.0, 0.05, 0.2)
# 10.450583572185565
```

---

## Example 2 — European put (Black-Scholes)

**Parameters**

- `flag = 'p'`
- `S = 100`, `K = 100`
- `t = 1.0`, `r = 0.05`, `sigma = 0.2`

```python
import optnlib

optnlib.black_scholes('p', 100, 100, 1.0, 0.05, 0.2)
# 5.573526022256971
```

---

## Example 3 — European call with a continuous dividend yield (Black-Scholes-Merton)

**Parameters**

- `flag = 'c'`
- `S = 100`, `K = 100`
- `t = 1.0`, `r = 0.05`, `sigma = 0.2`
- `q = 0.03`

```python
import optnlib

optnlib.black_scholes_merton('c', 100, 100, 1.0, 0.05, 0.2, q=0.03)
# 8.652528553942709
```

---

## Example 4 — Black-76 on a forward (options on futures)

**Parameters**

- `flag = 'c'`
- `F = 105` (forward), `K = 100`
- `t = 1.0`, `r = 0.04`, `sigma = 0.2`

```python
import optnlib

optnlib.black('c', 105, 100, 1.0, 0.04, 0.2)
# 10.477979035159038
```

---

## Example 5 — European call with discrete dividends (forward model, default)

**Parameters**

- `flag = 'c'`
- `S = 100`, `K = 100`
- `t = 1.0`, `r = 0.05`, `sigma = 0.2`
- `divs = [(0.25, 1.5), (0.75, 1.5)]` (two 1.5 cash dividends)
- `dividend_model = 'forward'` (default)

```python
import optnlib

optnlib.price('c', 100, 100, 1.0, 0.05, 0.2, divs=[(0.25, 1.5), (0.75, 1.5)])
# 8.669502736032905
```

---

## Example 6 — European call with discrete dividends (escrowed model)

**Parameters**

- `flag = 'c'`
- `S = 100`, `K = 100`
- `t = 1.0`, `r = 0.05`, `sigma = 0.2`
- `divs = [(0.25, 1.5), (0.75, 1.5)]`
- `dividend_model = 'escrowed'`

```python
import optnlib

optnlib.price('c', 100, 100, 1.0, 0.05, 0.2,
              divs=[(0.25, 1.5), (0.75, 1.5)],
              dividend_model='escrowed')
# 8.669502736032896
```

---

## Example 7 — Implied volatility (European)

**Parameters**

- `price = 10.450583572185565` (the Example 1 call)
- `flag = 'c'`, `S = 100`, `K = 100`, `t = 1.0`, `r = 0.05`
- `style = 'european'` (default)

```python
import optnlib

optnlib.implied_volatility(10.450583572185565, 'c', 100, 100, 1.0, 0.05)
# 0.20000000000000037
```

---

## Example 8 — All European greeks at once

**Parameters**

- `flag = 'c'`, `S = 100`, `K = 100`, `t = 1.0`, `r = 0.05`, `sigma = 0.2`
- Analytical BSM greeks are used (no discrete dividends)

```python
import optnlib

optnlib.greeks('c', 100, 100, 1.0, 0.05, 0.2)
# {'delta': 0.6368306511756191, 'gamma': 0.018762017345846895,
#  'theta': -0.01757267820941972, 'vega': 0.3752403469169379,
#  'rho': 0.5323248154537634}
```

---

## Example 9 — Individual greek helpers

**Parameters**

- `flag = 'c'`, `S = 100`, `K = 100`, `t = 1.0`, `r = 0.05`, `sigma = 0.2`

```python
import optnlib

optnlib.delta('c', 100, 100, 1.0, 0.05, 0.2)   # 0.6368306511756191
optnlib.vega('c', 100, 100, 1.0, 0.05, 0.2)    # 0.3752403469169379  (per 1% vol)
optnlib.theta('c', 100, 100, 1.0, 0.05, 0.2)   # -0.01757267820941972 (per day)
```

---

## Example 10 — American put (Bjerksund-Stensland 2002, default)

**Parameters**

- `flag = 'p'`, `S = 100`, `K = 100`, `t = 1.0`, `r = 0.05`, `sigma = 0.2`
- `method = 'bs2002'` (default)

```python
import optnlib

optnlib.american_price('p', 100, 100, 1.0, 0.05, 0.2)
# 6.045010205142354
```

---

## Example 11 — American put (Leisen-Reimer binomial tree)

**Parameters**

- `flag = 'p'`, `S = 100`, `K = 100`, `t = 1.0`, `r = 0.05`, `sigma = 0.2`
- `method = 'lr'`, `n_steps = 101` (default)

```python
import optnlib

optnlib.american_price('p', 100, 100, 1.0, 0.05, 0.2, method='lr')
# 6.087222149481556
```

---

## Example 12 — American put (Cox-Ross-Rubinstein, custom step count)

**Parameters**

- `flag = 'p'`, `S = 100`, `K = 100`, `t = 1.0`, `r = 0.05`, `sigma = 0.2`
- `method = 'crr'`, `n_steps = 1000`

```python
import optnlib

optnlib.american_price('p', 100, 100, 1.0, 0.05, 0.2, method='crr', n_steps=1000)
# 6.0895952829782045
```

---

## Example 13 — American call with a continuous yield (early exercise matters)

**Parameters**

- `flag = 'c'`, `S = 100`, `K = 90`, `t = 1.0`, `r = 0.05`, `sigma = 0.2`
- `q = 0.06` (yield exceeds half the rate, so early exercise can be optimal)

```python
import optnlib

optnlib.american_price('c', 100, 90, 1.0, 0.05, 0.2, q=0.06)
# 12.65910794518857
```

---

## Example 14 — American call with a discrete dividend (forward model, LR tree)

**Parameters**

- `flag = 'c'`, `S = 100`, `K = 100`, `t = 1.0`, `r = 0.05`, `sigma = 0.2`
- `divs = [(0.5, 3.0)]`
- `method = 'lr'`, `dividend_model = 'forward'` (default)

```python
import optnlib

optnlib.american_price('c', 100, 100, 1.0, 0.05, 0.2,
                       divs=[(0.5, 3.0)], method='lr')
# 8.942755140161186
```

---

## Example 15 — American put with discrete dividends (LR tree)

**Parameters**

- `flag = 'p'`, `S = 100`, `K = 100`, `t = 1.0`, `r = 0.05`, `sigma = 0.2`
- `divs = [(0.25, 2.0), (0.75, 2.0)]`
- `method = 'lr'`

```python
import optnlib

optnlib.american_price('p', 100, 100, 1.0, 0.05, 0.2,
                       divs=[(0.25, 2.0), (0.75, 2.0)], method='lr')
# 7.785160801247057
```

---

## Example 16 — Implied volatility (American)

**Parameters**

- `price = 7.927966446637392` (American put from Example 10's parameters, sigma 0.25)
- `flag = 'p'`, `S = 100`, `K = 100`, `t = 1.0`, `r = 0.05`
- `style = 'american'`, `method = 'bs2002'` (must match the engine used to price)

```python
import optnlib

price = optnlib.american_price('p', 100, 100, 1.0, 0.05, 0.25)   # 7.927966446637392
optnlib.implied_volatility(price, 'p', 100, 100, 1.0, 0.05,
                           style='american', method='bs2002')
# 0.2500000000000281
```

---

## Example 17 — American greeks (numerical)

**Parameters**

- `flag = 'p'`, `S = 100`, `K = 100`, `t = 1.0`, `r = 0.05`, `sigma = 0.2`
- `style = 'american'`, `method = 'lr'` (finite differences)

```python
import optnlib

optnlib.greeks('p', 100, 100, 1.0, 0.05, 0.2, style='american', method='lr')
# {'delta': -0.41060684385501034, 'gamma': 0.020623874998904057,
#  'theta': -0.006119630163917259, 'vega': 0.3749275723659906,
#  'rho': -0.3031935656154783}
```

---

## Example 18 — Unified `price()` and input validation

**Parameters**

- European call: `S = 100`, `K = 100`, `t = 1.0`, `r = 0.05`, `sigma = 0.2`
- American put: same parameters, `method = 'lr'`
- Bad input: a negative spot should raise `InvalidInputError`

```python
import optnlib
from optnlib.exceptions import InvalidInputError

optnlib.price('c', 100, 100, 1.0, 0.05, 0.2, style='european')
# 10.450583572185565

optnlib.price('p', 100, 100, 1.0, 0.05, 0.2, style='american', method='lr')
# 6.087222149481556

try:
    optnlib.price('c', -100, 100, 1.0, 0.05, 0.2)
except InvalidInputError as exc:
    print(exc)   # spot S must be strictly positive; got -100
```
