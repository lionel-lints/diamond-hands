import pandas as pd
import pytest

import scenario_matrix as SM
import user_profile as P


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


def test_variations_without_an_ips_target_use_the_first_candidate(fake_root):
    prof = P.current()
    prof.settings.pop("ips_target")
    first = next(iter(prof.candidates.values()))
    v = SM.variations(prof)
    assert len(v) == 3
    stocks = 1 - sum(first.get(k, 0) for k in ("bonds", "tips", "tips_short"))
    at80 = v["target shape @ 80% stocks"]
    assert at80["intl"] == pytest.approx(0.8 * first.get("intl", 0) / stocks)
