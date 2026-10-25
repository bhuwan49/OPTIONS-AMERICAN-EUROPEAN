# -*- coding: utf-8 -*-

"""
optnlib.stochvol.base
~~~~~~~~~~~~~~~~~~~~~

Shared building blocks for the stochastic-volatility models: parameter
validation, simulation result containers, and the common Monte Carlo pricing
entry points.
"""

from dataclasses import dataclass
import math

import numpy as np

from ..core import intrinsic
from ..exceptions import InvalidInputError
from ..helpers import validate_option

__all__ = [
    "SimulationResult",
    "MCResult",
    "StochasticVolModel",
    "positive",
    "nonnegative",
    "finite",
    "probability",
    "positive_int",
    "validate_market",
    "validate_simulation",
]

DEFAULT_STEPS = 252
DEFAULT_SIMS = 10_000


@dataclass
class SimulationResult:
    """Container returned by ``model.simulate(...)``.

    ``spots`` always has shape ``(n_sims, n_steps + 1)``. ``variances`` has the
    same shape for variance models, or is ``None`` when the model has no
    independent variance state.
    """

    spots: np.ndarray
    variances: object
    times: np.ndarray
    scheme: str
    model: str

    def plot(self, K=None, n_paths=100, save=None, show=False, ax=None):
        """Plot a sample of the simulated spot paths."""
        from .plot import plot_paths

        return plot_paths(self.spots, K=K, t=float(self.times[-1]),
                          n_paths=n_paths, save=save, show=show, ax=ax,
                          title="%s simulated paths" % self.model)

    def plot_variance(self, n_paths=100, save=None, show=False, ax=None):
        """Plot a sample of the simulated variance paths."""
        from .plot import plot_variance_paths

        if self.variances is None:
            raise InvalidInputError(
                "%s does not have an independent variance state" % self.model)
        return plot_variance_paths(self.variances, t=float(self.times[-1]),
                                   n_paths=n_paths, save=save, show=show,
                                   ax=ax, title="%s variance paths" % self.model)


@dataclass
class MCResult:
    """Monte Carlo estimate with its standard error."""

    price: float
    std_error: float
    n_sims: int
    n_steps: int
    scheme: str

    def __float__(self):
        return float(self.price)

    @property
    def confidence_interval(self):
        """Approximate 95%% confidence interval for the price."""
        return (self.price - 1.96 * self.std_error,
                self.price + 1.96 * self.std_error)


def positive(value, name):
    """Return ``value`` as a finite float strictly greater than zero."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise InvalidInputError(
            "%s must be a positive number; got %r" % (name, value))
    if not math.isfinite(result) or result <= 0.0:
        raise InvalidInputError(
            "%s must be a positive finite number; got %r" % (name, value))
    return result


def nonnegative(value, name):
    """Return ``value`` as a finite float greater than or equal to zero."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise InvalidInputError(
            "%s must be a non-negative number; got %r" % (name, value))
    if not math.isfinite(result) or result < 0.0:
        raise InvalidInputError(
            "%s must be a non-negative finite number; got %r" % (name, value))
    return result


def finite(value, name):
    """Return ``value`` as a finite float."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise InvalidInputError(
            "%s must be a finite number; got %r" % (name, value))
    if not math.isfinite(result):
        raise InvalidInputError(
            "%s must be a finite number; got %r" % (name, value))
    return result


def probability(value, name, strict=False):
    """Return a valid probability in ``[0, 1]`` (or ``(0, 1)``)."""
    result = finite(value, name)
    if strict:
        if result <= 0.0 or result >= 1.0:
            raise InvalidInputError(
                "%s must be strictly between 0 and 1; got %r" % (name, value))
    elif result < 0.0 or result > 1.0:
        raise InvalidInputError(
            "%s must be between 0 and 1; got %r" % (name, value))
    return result


def positive_int(value, name):
    """Return a positive integer, rejecting bools and floats with fractions."""
    if isinstance(value, bool):
        raise InvalidInputError(
            "%s must be a positive integer; got %r" % (name, value))
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise InvalidInputError(
            "%s must be a positive integer; got %r" % (name, value))
    if result != value or result < 1:
        raise InvalidInputError(
            "%s must be a positive integer; got %r" % (name, value))
    return result


def validate_correlation(rho):
    """Validate a correlation coefficient."""
    rho = finite(rho, "rho")
    if rho < -1.0 or rho > 1.0:
        raise InvalidInputError("rho must be between -1 and 1; got %r" % (rho,))
    return rho


def validate_market(S, t, r, q):
    """Validate and normalize the common market arguments."""
    return (positive(S, "spot S"), nonnegative(t, "time to expiry t"),
            finite(r, "risk-free rate r"), finite(q, "dividend yield q"))


def validate_simulation(n_steps, n_sims):
    """Validate and normalize the simulation resolution arguments."""
    return positive_int(n_steps, "n_steps"), positive_int(n_sims, "n_sims")


def correlated_normals(rng, n_sims, n_steps, rho):
    """Return two standard-normal arrays with correlation ``rho``."""
    z1 = rng.standard_normal((n_sims, n_steps))
    z2_independent = rng.standard_normal((n_sims, n_steps))
    z2 = rho * z1 + math.sqrt(max(1.0 - rho * rho, 0.0)) * z2_independent
    return z1, z2


class StochasticVolModel:
    """Base class for the Phase 1 stochastic-volatility models."""

    name = "stochastic-volatility model"
    default_scheme = "euler"
    schemes = ("euler",)
    has_cf = False

    def _scheme(self, scheme):
        if scheme is None:
            scheme = self.default_scheme
        scheme = str(scheme).lower().replace("_", "-")
        aliases = {"full-truncation": "euler", "fulltruncation": "euler"}
        scheme = aliases.get(scheme, scheme)
        if scheme not in self.schemes:
            raise InvalidInputError(
                "%s supports schemes %s; got %r"
                % (self.name, ", ".join(self.schemes), scheme))
        return scheme

    def simulate(self, S, t, r, q=0.0, n_steps=DEFAULT_STEPS, n_sims=DEFAULT_SIMS,
                 scheme=None, seed=None):
        """Simulate the model. Implemented by every concrete model."""
        raise NotImplementedError

    def characteristic_function(self, u, t, r, q=0.0):
        """Characteristic function of ``log(S_t / S_0)`` when available."""
        raise NotImplementedError(
            "%s does not provide a characteristic function" % self.name)

    def price_fourier(self, flag, S, K, t, r, q=0.0):
        """Price a European option using the model characteristic function."""
        if not self.has_cf:
            raise NotImplementedError(
                "%s does not provide Fourier pricing" % self.name)
        from .fourier import price_from_cf

        return price_from_cf(self.characteristic_function, flag, S, K, t, r, q)

    def price_mc(self, flag, S, K, t, r, q=0.0, n_steps=DEFAULT_STEPS,
                 n_sims=DEFAULT_SIMS, scheme=None, seed=None):
        """Price a European option by Monte Carlo and return ``MCResult``."""
        flag = validate_option(flag, S, K, t, r, 0.0, q=q)
        S = positive(S, "spot S")
        K = positive(K, "strike K")
        t = nonnegative(t, "time to expiry t")
        r = finite(r, "risk-free rate r")
        q = finite(q, "dividend yield q")
        n_steps = positive_int(n_steps, "n_steps")
        n_sims = positive_int(n_sims, "n_sims")

        if t == 0.0:
            return MCResult(intrinsic(flag, S, K), 0.0, n_sims, 0, "exact")

        simulation = self.simulate(
            S, t, r, q=q, n_steps=n_steps, n_sims=n_sims,
            scheme=scheme, seed=seed,
        )
        terminal = simulation.spots[:, -1]
        if flag == "c":
            payoff = np.maximum(terminal - K, 0.0)
        else:
            payoff = np.maximum(K - terminal, 0.0)
        discounted = math.exp(-r * t) * payoff
        price = float(np.mean(discounted))
        if n_sims > 1:
            std_error = float(np.std(discounted, ddof=1) / math.sqrt(n_sims))
        else:
            std_error = 0.0
        return MCResult(price, std_error, n_sims, n_steps, simulation.scheme)

    def price(self, flag, S, K, t, r, q=0.0, method="auto", n_steps=DEFAULT_STEPS,
              n_sims=DEFAULT_SIMS, scheme=None, seed=None):
        """Price a European option.

        ``method='auto'`` uses Fourier pricing when a characteristic function
        is available, otherwise Monte Carlo. Use ``method='mc'`` to force a
        simulation and ``method='fourier'`` to force Fourier pricing.
        """
        method = str(method).lower().replace("_", "-")
        if method in ("auto", "default"):
            if self.has_cf:
                return self.price_fourier(flag, S, K, t, r, q=q)
            method = "mc"
        if method in ("fourier", "cf", "analytic"):
            return self.price_fourier(flag, S, K, t, r, q=q)
        if method in ("closed-form", "closedform"):
            closed_form = getattr(self, "price_closed_form", None)
            if closed_form is None:
                raise NotImplementedError(
                    "%s does not provide a closed-form price" % self.name)
            return closed_form(flag, S, K, t, r, q=q)
        if method in ("mc", "monte-carlo", "simulation"):
            return self.price_mc(
                flag, S, K, t, r, q=q, n_steps=n_steps, n_sims=n_sims,
                scheme=scheme, seed=seed,
            ).price
        raise InvalidInputError(
            "method must be 'auto', 'mc', 'fourier', or 'closed-form'; got %r"
            % (method,))

    def __repr__(self):
        params = getattr(self, "_repr_params", None)
        if not params:
            return "%s()" % self.__class__.__name__
        body = ", ".join("%s=%r" % item for item in params)
        return "%s(%s)" % (self.__class__.__name__, body)
