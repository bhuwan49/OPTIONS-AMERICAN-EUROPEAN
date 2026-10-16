# -*- coding: utf-8 -*-

"""Offline tests for the optnlib.volatility subpackage."""

from datetime import date

import numpy as np
import pytest

from optnlib import volatility as v
from optnlib.exceptions import InvalidInputError

REF = "2026-01-05"


@pytest.fixture
def chain():
    return v.model_chain(reference_date=REF)


@pytest.fixture
def surface(chain):
    return v.VolSurface(v.compute_iv_surface(chain))


def test_model_chain_schema(chain):
    assert list(chain.columns) == v.market.CHAIN_COLUMNS
    assert chain.attrs["spot"] == 100.0
    assert chain.attrs["reference_date"] == date(2026, 1, 5)
    assert set(chain["flag"].unique()) == {"c", "p"}
    assert chain["dte"].min() == 30 and chain["dte"].max() == 365


def test_compute_iv_surface_recovers_model_vol(chain):
    frame = v.compute_iv_surface(chain)
    assert {"iv", "moneyness", "log_moneyness", "t"} <= set(frame.columns)
    assert frame["iv"].notna().all()
    # recovered IV must match the model IV stored in iv_yahoo
    assert np.allclose(frame["iv"], frame["iv_yahoo"], atol=1e-4)
    assert frame.attrs["spot"] == 100.0


def test_compute_iv_surface_nan_mode_keeps_rows(chain):
    bad = chain.copy()
    bad.loc[bad.index[0], "mid"] = 0.0
    dropped = v.compute_iv_surface(bad, errors="drop")
    kept = v.compute_iv_surface(bad, errors="nan")
    assert len(kept) == len(bad)
    assert len(dropped) == len(bad) - 1
    assert len(kept) - len(dropped) == 1


def test_exact_grid_query_matches_direct(surface):
    frame = surface.frame
    exp = surface.expiries[0]
    strike = float(surface.strikes[len(surface.strikes) // 2])
    direct = float(frame[(frame.expiry == exp)
                         & (frame.strike == strike)]["iv"].iloc[0])
    assert surface.iv(exp, strike) == pytest.approx(direct, abs=1e-10)
    # string and datetime inputs accepted
    assert surface.iv(exp.isoformat(), strike) == pytest.approx(direct, abs=1e-10)


def test_iv_moneyness_query(surface):
    val = surface.iv_moneyness(1.0, dte=30)
    assert 0.0 < val < 2.0


def test_out_of_range_dte_raises(surface):
    with pytest.raises(InvalidInputError, match="outside the surface"):
        surface.iv("2030-01-01", 100.0)


def test_out_of_range_strike_raises(surface):
    with pytest.raises(InvalidInputError, match="outside the surface"):
        surface.iv(surface.expiries[0], 1.0)


def test_expiry_before_reference_raises(surface):
    with pytest.raises(InvalidInputError, match="precedes"):
        surface.iv("2020-01-01", 100.0)


def test_bad_method_raises(surface):
    with pytest.raises(InvalidInputError, match="method"):
        surface.iv_moneyness(1.0, 30, method="nope")


def test_smile_and_term_structure(surface):
    sm = surface.smile(surface.expiries[-1])
    assert list(sm["strike"]) == sorted(sm["strike"])
    assert len(sm) == len(surface.strikes)  # both calls and puts per strike
    ts = surface.term_structure(100.0)
    assert list(ts["dte"]) == sorted(ts["dte"])
    assert len(ts) == len(surface.expiries)


def test_to_grid_shape(surface):
    grid = surface.to_grid(n_dte=7, n_moneyness=9)
    assert grid["iv"].shape == (7, 9)
    assert len(grid["dates"]) == 7
    assert all(hasattr(d, "isoformat") for d in grid["dates"])


def test_repr(surface):
    assert "VolSurface" in repr(surface)
    assert "MODEL" in repr(surface)


def test_load_surface_model_source():
    surf = v.load_surface(source="model",
                          model_kwargs={"reference_date": REF})
    assert isinstance(surf, v.VolSurface)
    assert surf.spot == 100.0


def test_load_surface_requires_ticker_for_fetch():
    with pytest.raises(InvalidInputError, match="ticker"):
        v.load_surface(source="fetch")


def test_load_surface_unknown_source():
    with pytest.raises(InvalidInputError, match="source"):
        v.load_surface(chain=v.model_chain(reference_date=REF), source="bogus")


# -- plotting (matplotlib Agg / plotly) -------------------------------------


def _mpl():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    return matplotlib


def test_plot_smile_matplotlib(surface, tmp_path):
    _mpl()
    out = tmp_path / "smile.png"
    fig = v.plot_smile(surface, backend="matplotlib", save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig.__class__.__name__ == "Figure"


def test_plot_surface_matplotlib_time_axis_date(surface, tmp_path):
    _mpl()
    out = tmp_path / "surf.png"
    fig = v.plot_surface(surface, backend="matplotlib", time_axis="date",
                         save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig.__class__.__name__ == "Figure"


@pytest.mark.parametrize("axis", ["dte", "years"])
def test_plot_surface_time_axes(surface, tmp_path, axis):
    _mpl()
    fig = v.plot_surface(surface, backend="matplotlib", time_axis=axis,
                         n_dte=6)
    assert fig.__class__.__name__ == "Figure"


def test_plot_surface_bad_time_axis(surface):
    _mpl()
    with pytest.raises(InvalidInputError, match="time_axis"):
        v.plot_surface(surface, time_axis="decade")


def test_plot_plotly_html(surface, tmp_path):
    pytest.importorskip("plotly")
    out = tmp_path / "surf.html"
    fig = v.plot_surface(surface, backend="plotly", save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig.__class__.__name__ == "Figure"


def test_plot_smile_plotly(surface):
    pytest.importorskip("plotly")
    fig = v.plot_smile(surface, backend="plotly")
    assert fig.__class__.__name__ == "Figure"


def test_volatility_surface_high_level(tmp_path):
    _mpl()
    out = tmp_path / "hh.png"
    surf = v.volatility_surface(source="model", backend="matplotlib",
                                save=str(out),
                                model_kwargs={"reference_date": REF})
    assert isinstance(surf, v.VolSurface)
    assert surf.figure is not None
    assert out.exists()


# -- CLI ---------------------------------------------------------------------


def test_cli_no_plot(capsys):
    from optnlib.volatility import cli

    rc = cli.main(["--source", "model", "--reference-date", REF, "--no-plot"])
    assert rc == 0
    assert "VolSurface" in capsys.readouterr().out


def test_cli_query(capsys):
    from optnlib.volatility import cli

    rc = cli.main(["--source", "model", "--reference-date", REF,
                   "--query", "2026-02-04", "100.0"])
    assert rc == 0
    assert "IV(2026-02-04, 100.0)" in capsys.readouterr().out


def test_cli_query_out_of_range(capsys):
    from optnlib.volatility import cli

    rc = cli.main(["--source", "model", "--reference-date", REF,
                   "--query", "2030-01-01", "100.0"])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_cli_writes_files(tmp_path):
    _mpl()
    from optnlib.volatility import cli

    rc = cli.main(["--source", "model", "--reference-date", REF,
                   "--mode", "both", "--backend", "matplotlib",
                   "--save-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "model_smile.png").exists()
    assert (tmp_path / "model_surface.png").exists()
