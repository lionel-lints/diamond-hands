import pandas as pd
import pytest

import scenario_matrix as SM


def test_worst_best_orders_outcomes():
    t = pd.DataFrame({"A": [-0.3, 0.2, -0.1], "B": [-0.05, 0.1, -0.4]}, index=["x", "y", "z"])
    r = {row["blend"]: row for row in SM.worst_best(t)}
    assert r["A"]["worst"][0] == ("x", -0.3) and r["A"]["best"][0] == ("y", 0.2)
    assert r["B"]["worst"][0] == ("z", -0.4)


def test_belief_rows_weight_and_plausible_worst():
    t = pd.DataFrame({"A": [-0.5, 0.2, -0.1], "B": [-0.1, 0.1, -0.2]},
                     index=["scenario: crash (1 yr)", "scenario: boom (3 yr)", "history (real worst): dip"])
    br = SM.belief_rows(t, {"crash": 0, "boom": 3, "dip": 2})
    assert br.loc["A", "belief-weighted average"] == pytest.approx((0.2 * 3 - 0.1 * 2) / 5)
    # the crash is weighted 0, so it's not "plausible": A's plausible worst is the dip
    assert br.loc["A", "worst among plausible (weight >= 2)"] == pytest.approx(-0.1)
    assert SM.belief_rows(t, {}) is None
