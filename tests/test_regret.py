import numpy as np
import pandas as pd
import pytest

import regret as G


def test_constant_mix_path_rebalances_monthly():
    hist = pd.DataFrame({"A": [0.10, -0.10], "B": [0.0, 0.0]})
    p = G.constant_mix_path(pd.Series({"A": 50.0, "B": 50.0}), hist)
    assert p.tolist() == pytest.approx([1.05, 1.05 * 0.95])  # 50/50 restored before month 2


def test_lag_stats_worst_and_longest_behind():
    idx = pd.period_range("2000-01", periods=6, freq="M").astype(str)
    port = np.array([1.0, 0.88, 0.8, 0.85, 1.2, 1.3])  # windows: -12%, -9.1%, +6%, +41%, +8%
    bench = np.ones(6)
    diff, st = G.lag_stats(port, bench, idx, lag_months=1)
    assert st["worst_5y_lag"] == pytest.approx(0.88 - 1)
    assert st["worst_5y_lag_ending"] == "2000-02"
    assert st["longest_behind_months"] == 2    # two consecutive negative windows
    assert st["pct_windows_behind_10pp"] == pytest.approx(1 / 5)
    assert st["pct_windows_behind_5pp"] == pytest.approx(2 / 5)


def test_no_scv_twin_moves_scv_into_us():
    t = {"us": 0.5, "intl": 0.3, "scv": 0.1, "bonds": 0.1, "reserve_usd": "new_taxable"}
    tw = G.no_scv_twin(t)
    assert tw["us"] == pytest.approx(0.6) and "scv" not in tw and tw["reserve_usd"] == "new_taxable"
    assert t["scv"] == 0.1  # original untouched
