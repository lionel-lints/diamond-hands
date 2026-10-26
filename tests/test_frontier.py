from math import comb

import numpy as np
import pandas as pd
import pytest

import frontier as F
import style_windows as SW


def test_grid_weights_sum_to_one_and_cover_all_mixes():
    g = list(F.grid(0.05))
    assert len(g) == comb(20 + 3, 3)  # non-negative 4-asset mixes in 5% steps
    assert all(w.sum() == pytest.approx(1) and (w >= 0).all() for w in g)
    assert any(np.allclose(w, [1, 0, 0, 0]) for w in g)


def test_pareto_keeps_only_non_dominated_points():
    df = pd.DataFrame({"ret": [0.05, 0.04, 0.07, 0.06, 0.08], "risk": [0.10, 0.12, 0.15, 0.16, 0.20]})
    front = F.pareto(df, "ret", "risk")
    # 0.04@0.12 is dominated by 0.05@0.10; 0.06@0.16 by 0.07@0.15
    assert front.ret.tolist() == [0.05, 0.07, 0.08]


def synthetic_prices(monthly):
    idx = pd.period_range("2000-01", periods=len(next(iter(monthly.values()))), freq="M").to_timestamp()
    return pd.DataFrame({F.ASSETS[k]: np.cumprod(np.r_[1, 1 + np.array(v)])[1:] for k, v in monthly.items()}, index=idx)


def test_stats_on_constant_returns():
    n = 120
    r = 0.01
    noise = {"us": [r] * n, "intl": [r] * n, "bonds": [r] * n, "tips": [r] * n}
    prices = synthetic_prices(noise)
    rets, cov, mu = F.load_inputs(prices)
    s = F.stats(np.array([1, 0, 0, 0]), rets, cov, mu)
    years = len(rets) / 12
    assert s["hist_cagr"] == pytest.approx((1 + r) ** (len(rets) / years) - 1)
    assert s["vol"] == pytest.approx(0, abs=1e-12)
    assert s["max_dd"] == pytest.approx(0)
    assert s["fwd_return"] == pytest.approx(F.FORWARD_GEO["us"])


def test_stats_max_drawdown():
    up_down = [0.10, -0.5, 0.20] + [0.0] * 20
    flat = [0.0] * 23
    prices = synthetic_prices({"us": up_down, "intl": flat, "bonds": flat, "tips": flat})
    rets, cov, mu = F.load_inputs(prices)
    s = F.stats(np.array([1, 0, 0, 0]), rets, cov, mu)
    # first return is dropped by pct_change; path from there: -50%, +20% -> trough -50%
    assert s["max_dd"] == pytest.approx(-0.5)
    s_half = F.stats(np.array([.5, .5, 0, 0]), rets, cov, mu)
    assert s_half["max_dd"] == pytest.approx(-0.25)


def test_window_returns():
    idx = ["2000-03", "2002-09", "2007-10", "2009-02", "2020-01", "2020-03", "2021-12", "2022-12", "2026-10"]
    d = pd.DataFrame({"A": [100, 50, 100, 25, 100, 80, 100, 90, 180]}, index=idx)
    out = SW.window_returns(d).A
    assert out["Dot-com 2000-03→2002-09"] == pytest.approx(-0.5)
    assert out["GFC 2007-10→2009-02"] == pytest.approx(-0.75)
    assert out["AI run 2023-01→now"] == pytest.approx(1.0)  # 2022-12 -> latest (2026-10)
