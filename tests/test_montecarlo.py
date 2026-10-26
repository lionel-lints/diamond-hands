import numpy as np
import pandas as pd
import pytest

import montecarlo as M


def const_paths(r, n_paths=3, n_months=24, n_blocks=2):
    return np.full((n_paths, n_months, n_blocks), r)


def test_bootstrap_indices_are_contiguous_blocks_and_wrap():
    rng = np.random.default_rng(0)
    idx = M.bootstrap_indices(n_hist=30, n_months=25, n_paths=50, block_len=12, rng=rng)
    assert idx.shape == (50, 25) and idx.min() >= 0 and idx.max() < 30
    steps = (idx[:, 1:12] - idx[:, :11]) % 30
    assert (steps == 1).all()  # within a block, consecutive months (mod wrap)


def test_recenter_hits_target_geometric_mean_and_keeps_shape():
    rng = np.random.default_rng(1)
    h = pd.DataFrame({"A": rng.normal(0.01, 0.04, 400), "B": rng.normal(0.0, 0.02, 400)})
    out = M.recenter(h, {"A": 0.05, "B": None})
    g = np.exp(np.log1p(out.A).mean() * 12) - 1
    assert g == pytest.approx(0.05, abs=1e-9)
    assert np.corrcoef(h.A, out.A)[0, 1] > 0.999
    pd.testing.assert_series_equal(out.B, h.B)


def test_cma_targets_small_premium_is_one_point_in_every_run(monkeypatch):
    # cma.wide() is in percent; SMALL_PREMIUM is a decimal (0.01 = 1pp). The median/p25 runs
    # once added it before converting, which made the premium 0.01pp.
    w = pd.DataFrame({"us_large": [5.0, 7.0], "intl_dev": [6.0, 8.0], "us_bonds": [4.0, 5.0]},
                     index=["A", "B"]).reindex(columns=M.cma.CLASSES)
    monkeypatch.setattr(M.cma, "wide", lambda: w)
    for which in ["median", "p25", "A"]:
        t = M.cma_targets(["SP500", "SCV", "SMALL_BLEND"], which, current_tbill=0.04)
        assert t["SCV"] - t["SP500"] == pytest.approx(M.SMALL_PREMIUM)
        assert t["SMALL_BLEND"] - t["SP500"] == pytest.approx(M.SMALL_PREMIUM)
    assert M.cma_targets(["SP500"], "median", 0.04)["SP500"] == pytest.approx(0.06)


def test_simulate_buy_and_hold_constant_returns():
    start = pd.Series({"TOTAL_US": 600.0, "INTL": 400.0})
    sim = M.simulate(start, const_paths(0.01), np.zeros((3, 24)), rebal_blocks=set())
    assert sim["nominal"][:, -1] == pytest.approx(np.full(3, 1000 * 1.01 ** 24))
    assert sim["real"][:, -1] == pytest.approx(sim["nominal"][:, -1])


def test_simulate_deflates_by_inflation():
    start = pd.Series({"CASH": 100.0})
    sim = M.simulate(start, const_paths(0.0, n_blocks=1), np.full((3, 24), 0.01), rebal_blocks=set())
    assert sim["real"][:, -1] == pytest.approx(np.full(3, 100 / 1.01 ** 24))


def test_rebalancing_respects_bands():
    start = pd.Series({"TOTAL_US": 500.0, "BOND": 500.0})
    R = np.zeros((1, 6, 2))
    R[0, 0, 0] = 0.15   # stocks +15% -> share 53.5%: inside the 5pp band, no rebalance
    sim = M.simulate(start, R, np.zeros((1, 6)), rebal_blocks={"TOTAL_US", "BOND"}, rebalance_every=6)
    assert sim["nominal"][0, -1] == pytest.approx(1075)
    R[0, 0, 0] = 0.30   # +30% -> 56.5% share: outside the band -> reset to 50/50 at month 6
    sim2 = M.simulate(start, R, np.zeros((1, 6)), rebal_blocks={"TOTAL_US", "BOND"}, rebalance_every=6)
    assert sim2["nominal"][0, -1] == pytest.approx(1150)  # value unchanged by rebalancing itself


def test_dip_buying_deploys_reserve_in_tranches():
    start = pd.Series({"TOTAL_US": 600.0, "INTL": 0.0, "CASH": 400.0})  # all 400 is reserve
    R = np.zeros((1, 3, 3))
    R[0, 0, :2] = -0.25   # reference index -25%: first trigger (20%) fires, second (30%) doesn't
    sim = M.simulate(start, R, np.zeros((1, 3)), rebal_blocks={"TOTAL_US"}, reserve=400.0,
                     dips=[(0.20, 0.5), (0.30, 0.5)], ref_weights={"TOTAL_US": 0.6, "INTL": 0.4})
    # after month 1: stocks 450, cash 400 -> 200 moved into stocks
    assert sim["nominal"][0, 1] == pytest.approx(850)
    R2 = R.copy()
    R2[0, 1, :2] = 0.10   # stocks +10% next month: 650*1.1 + 200 cash
    sim2 = M.simulate(start, R2, np.zeros((1, 3)), rebal_blocks={"TOTAL_US"}, reserve=400.0,
                      dips=[(0.20, 0.5), (0.30, 0.5)], ref_weights={"TOTAL_US": 0.6, "INTL": 0.4})
    assert sim2["nominal"][0, 2] == pytest.approx(650 * 1.1 + 200)


def test_contributions_go_to_targets():
    start = pd.Series({"TOTAL_US": 750.0, "BOND": 250.0})
    sim = M.simulate(start, const_paths(0.0), np.zeros((3, 24)), rebal_blocks={"TOTAL_US", "BOND"},
                     contribution=100.0)
    assert sim["nominal"][0, -1] == pytest.approx(1000 + 24 * 100)


def test_path_metrics():
    x = np.array([[100, 120, 90, 95, 130, 125.0]])
    assert M.max_drawdown(x)[0] == pytest.approx(1 - 90 / 120)
    assert M.longest_underwater_months(x)[0] == 2  # months at 90 and 95 below the 120 peak
    bench = np.array([[100, 100, 100, 100, 100, 100.0]])
    # 2-month windows: 90/100, 95/120, 130/90, 125/95 -> worst is 95/120 - 1 vs. a flat benchmark
    assert M.worst_rolling_lag(x, bench, window=2)[0] == pytest.approx(95 / 120 - 1)


def test_streams_stop_after_their_months_and_index_with_inflation():
    start = pd.Series({"TOTAL_US": 1000.0})
    infl = np.full((1, 24), 0.01)
    fixed = [{"monthly_usd": 100.0, "months": 12, "indexed": False}]
    sim = M.simulate(start, np.zeros((1, 24, 1)), infl, {"TOTAL_US"}, streams=fixed)
    assert sim["nominal"][0, -1] == pytest.approx(1000 + 12 * 100)
    indexed = [{"monthly_usd": 100.0, "months": None, "indexed": True}]
    sim = M.simulate(start, np.zeros((1, 24, 1)), infl, {"TOTAL_US"}, streams=indexed)
    expected = 1000 + sum(100 * 1.01 ** (t + 1) for t in range(24))
    assert sim["nominal"][0, -1] == pytest.approx(expected)
    # in real terms an indexed stream adds a constant 100/month
    assert sim["real"][0, -1] == pytest.approx(expected / 1.01 ** 24)


def test_twr_excludes_contributions_so_drawdowns_stay_visible():
    start = pd.Series({"TOTAL_US": 1000.0})
    R = np.zeros((1, 3, 1))
    R[0, 0, 0] = -0.5  # crash in month 1, flat after
    streams = [{"monthly_usd": 1000.0, "months": None, "indexed": False}]
    sim = M.simulate(start, R, np.zeros((1, 3)), {"TOTAL_US"}, streams=streams)
    assert sim["nominal"][0, -1] == pytest.approx(500 + 3000)   # wealth recovered thanks to savings...
    assert sim["twr"][0, -1] == pytest.approx(0.5)               # ...but the investments are still -50%
    assert M.max_drawdown(sim["twr"])[0] == pytest.approx(0.5)
    assert M.max_drawdown(sim["nominal"])[0] < 0.5


def test_buy_and_hold_contributions_go_pro_rata_to_held_blocks():
    start = pd.Series({"TOTAL_US": 300.0, "INTL": 100.0, "CASH": 600.0})
    streams = [{"monthly_usd": 100.0, "months": None, "indexed": False}]
    sim = M.simulate(start, np.zeros((1, 2, 3)), np.zeros((1, 2)), rebal_blocks=set(), streams=streams,
                     contrib_blocks={"TOTAL_US", "INTL"})
    assert sim["nominal"][0, -1] == pytest.approx(1000 + 200)   # money lands somewhere (not dropped)


def test_contributions_without_destination_are_not_silently_lost():
    """Before the fix, a buy-and-hold portfolio had no rebalance blocks, so streams went nowhere."""
    start = pd.Series({"TOTAL_US": 1000.0})
    streams = [{"monthly_usd": 100.0, "months": None, "indexed": False}]
    sim = M.simulate(start, np.zeros((1, 3, 1)), np.zeros((1, 3)), rebal_blocks=set(), streams=streams,
                     contrib_blocks={"TOTAL_US"})
    assert sim["nominal"][0, -1] == pytest.approx(1300)
