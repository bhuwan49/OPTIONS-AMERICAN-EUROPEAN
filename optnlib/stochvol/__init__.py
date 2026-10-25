# -*- coding: utf-8 -*-

"""
optnlib.stochvol
~~~~~~~~~~~~~~~~

Stochastic-volatility and jump-diffusion models with Monte Carlo simulation,
Fourier pricing where available, and optional path visualization.

Phase 1 models
--------------

- :class:`Heston`
- :class:`HullWhite`
- :class:`SteinStein`
- :class:`Scott`
- :class:`ThreeHalves`
- :class:`SABR`
- :class:`Merton`
- :class:`Kou`
- :class:`Bates`

Quick start
-----------

::

    from optnlib import stochvol as sv

    model = sv.Heston(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    model.price('c', 100, 100, 1.0, 0.05)

    paths = model.simulate(100, 1.0, 0.05, n_steps=252, n_sims=10_000,
                           seed=42)
    paths.plot(K=100, t=1.0)
"""

from . import fourier, plot, schemes
from .base import MCResult, SimulationResult, StochasticVolModel
from .models import (
    Bates,
    Heston,
    HullWhite,
    Kou,
    Merton,
    SABR,
    Scott,
    SteinStein,
    ThreeHalves,
)

__all__ = [
    "Bates",
    "Heston",
    "HullWhite",
    "Kou",
    "Merton",
    "SABR",
    "Scott",
    "SteinStein",
    "ThreeHalves",
    "StochasticVolModel",
    "SimulationResult",
    "MCResult",
    "schemes",
    "fourier",
    "plot",
]
