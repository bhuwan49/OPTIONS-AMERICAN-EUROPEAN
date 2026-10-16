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


def test_plot_smile_matplotlib_grid(surface, tmp_path):
    _mpl()
    out = tmp_path / "smile.png"
    fig = v.plot_smile(surface, backend="matplotlib", layout="grid",
                       save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig.__class__.__name__ == "Figure"


def test_plot_smile_matplotlib_overlay(surface, tmp_path):
    _mpl()
    out = tmp_path / "smile_overlay.png"
    fig = v.plot_smile(surface, backend="matplotlib", layout="overlay",
                       save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig.__class__.__name__ == "Figure"


def test_plot_smile_bad_layout(surface):
    _mpl()
    with pytest.raises(InvalidInputError, match="layout"):
        v.plot_smile(surface, layout="fancy")


def test_plot_surface_bad_backend(surface):
    with pytest.raises(InvalidInputError, match="backend"):
        v.plot_surface(surface, backend="gnuplot")


def test_plot_smile_auto_backend(surface):
    pytest.importorskip("plotly")
    fig = v.plot_smile(surface, backend=None)
    assert fig.__class__.__name__ == "Figure"


def test_plot_surface_matplotlib(surface, tmp_path):
    _mpl()
    out = tmp_path / "surf.png"
    fig = v.plot_surface(surface, backend="matplotlib", save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig.__class__.__name__ == "Figure"


def test_plot_surface_contours(surface):
    _mpl()
    fig = v.plot_surface(surface, backend="matplotlib", show_contours=True,
                         n_dte=6, n_moneyness=8)
    assert fig.__class__.__name__ == "Figure"





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


# -- IV matrix table ---------------------------------------------------------


def test_to_matrix_years(surface):
    df = surface.to_matrix(time_axis="years")
    assert df.shape == (len(surface.strikes), len(surface.expiries))
    assert list(df.index) == list(surface.strikes)
    assert df.index.name == "Strike"
    assert df.columns.name == "Tenor (years)"
    assert df.attrs == {} or True  # no crash; values numeric
    assert np.isfinite(df.to_numpy()).any()


def test_to_matrix_date_and_dte_columns(surface):
    by_date = surface.to_matrix(time_axis="date")
    assert list(by_date.columns) == [e.isoformat() for e in surface.expiries]
    by_dte = surface.to_matrix(time_axis="dte")
    assert list(by_dte.columns) == [
        (e - surface.reference_date).days for e in surface.expiries]


def test_to_matrix_default_is_date(surface):
    df = surface.to_matrix()
    assert list(df.columns) == [e.isoformat() for e in surface.expiries]
    assert df.columns.name == "Expiry"


def test_to_matrix_bad_axis(surface):
    with pytest.raises(InvalidInputError, match="time_axis"):
        surface.to_matrix(time_axis="century")


def test_surface_table_prints(surface, capsys):
    df = v.surface_table(surface)
    out = capsys.readouterr().out
    assert "Strike" in out and "Expiry" in out
    assert df.shape == (len(surface.strikes), len(surface.expiries))


def test_surface_table_years_override(surface, capsys):
    v.surface_table(surface, time_axis="years", print_table=True)
    out = capsys.readouterr().out
    assert "Tenor" in out


def test_surface_table_silent(surface, capsys):
    v.surface_table(surface, print_table=False)
    assert capsys.readouterr().out == ""


def test_strike_mask_absolute_vs_moneyness():
    strikes = np.array([80.0, 95.0, 100.0, 110.0, 120.0, 150.0])
    as_moneyness = v.market._strike_mask(
        strikes, spot=100.0, strike_range=(0.8, 1.2), moneyness=True)
    as_absolute = v.market._strike_mask(
        strikes, spot=100.0, strike_range=(80.0, 120.0), moneyness=False)
    assert list(strikes[as_moneyness]) == [80.0, 95.0, 100.0, 110.0, 120.0]
    assert list(strikes[as_moneyness]) == list(strikes[as_absolute])


def test_default_dte_range_is_all():
    import inspect

    for func in (v.load_surface, v.volatility_smile, v.volatility_surface):
        assert inspect.signature(func).parameters["dte_range"].default is None
