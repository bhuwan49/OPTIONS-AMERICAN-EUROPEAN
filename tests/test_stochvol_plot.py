# -*- coding: utf-8 -*-

import numpy as np
import pytest

from optnlib import stochvol as sv
from optnlib.exceptions import InvalidInputError


def _mpl():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    return matplotlib


def test_simulation_result_plot_saves_figure(tmp_path):
    _mpl()
    result = sv.Heston().simulate(100.0, 0.2, 0.05, n_steps=10, n_sims=10,
                                  seed=1)
    output = tmp_path / "stochvol_paths.png"
    figure = result.plot(K=100.0, save=str(output), n_paths=5)
    assert output.exists() and output.stat().st_size > 0
    assert figure.__class__.__name__ == "Figure"


def test_simulation_result_variance_plot_saves_figure(tmp_path):
    _mpl()
    result = sv.Heston().simulate(100.0, 0.2, 0.05, n_steps=10, n_sims=10,
                                  seed=1)
    output = tmp_path / "stochvol_variance.png"
    figure = result.plot_variance(save=str(output), n_paths=5)
    assert output.exists() and output.stat().st_size > 0
    assert figure.__class__.__name__ == "Figure"


def test_plot_rejects_non_2d():
    _mpl()
    with pytest.raises(InvalidInputError):
        sv.plot.plot_paths(np.array([100.0, 101.0]))
