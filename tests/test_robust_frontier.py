import numpy as np
import pandas as pd
import pytest

import robust_frontier as RF


def test_grid_targets_are_valid_and_unique():
    mixes = list(RF.grid())
    names = [n for n, _, _ in mixes]
    assert len(names) == len(set(names))
    for name, meta, t in mixes:
        assert sum(v for k, v in t.items() if k != "reserve_usd") == pytest.approx(1.0), name
        stocks = t["us"] + t["intl"] + t["scv"]
        assert stocks == pytest.approx(meta["stocks"])
        assert t["intl"] == pytest.approx(meta["stocks"] * meta["intl_of_stocks"])


def test_shortfall_is_zero_on_frontier_and_measures_double_dominance():
    m10 = np.array([2.0, 2.2, 1.9, 2.1])
    p3 = np.array([0.90, 0.85, 0.88, 0.89])
    s = RF.shortfall(m10, p3)
    assert s[0] == 0 and s[1] == 0          # frontier: nothing beats them on both
    assert s[2] == pytest.approx(min(2.0 / 1.9, 0.90 / 0.88) - 1)  # beaten by mix 0 on both
    assert s[3] == 0                         # 2.1/0.89 sits between 0 and 1: on the frontier


def test_dominators_require_beating_in_every_run():
    df = pd.DataFrame([
        {"mix": "a", "run": "r1", "m10": 2.0, "p3": 0.9}, {"mix": "a", "run": "r2", "m10": 2.0, "p3": 0.9},
        {"mix": "b", "run": "r1", "m10": 2.1, "p3": 0.95}, {"mix": "b", "run": "r2", "m10": 2.1, "p3": 0.95},
        {"mix": "c", "run": "r1", "m10": 2.1, "p3": 0.95}, {"mix": "c", "run": "r2", "m10": 1.9, "p3": 0.95},
    ])
    assert RF.dominators(df, "a") == ["b"]   # c loses on m10 in r2


def test_ips_like_name_maps_candidates_onto_the_grid():
    t = {"us": .54, "intl": .27, "scv": .09, "bonds": .05, "tips": .05}
    assert RF.ips_like_name(t) == "90% stk | 30% intl | 10% SCV | 50% TIPS"
    assert RF.ips_like_name({"us": .7, "intl": .3}) == "100% stk | 30% intl | 0% SCV | 0% TIPS"
