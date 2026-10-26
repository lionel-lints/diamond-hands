import numpy as np
import pandas as pd
import pytest

import compare as C


def test_cost_metrics_blend_by_dollars_and_skip_free():
    funds = pd.DataFrame([{"fund": "VTI", "usd": 750.0, "free": False}, {"fund": "VXUS", "usd": 250.0, "free": False},
                          {"fund": "(bank)", "usd": 1000.0, "free": True}])
    blended, coverage, n = C.cost_metrics(funds, {"VTI": 0.03, "VXUS": 0.05})
    assert blended == pytest.approx((750 * 0.03 + 250 * 0.05 + 0) / 2000)
    assert coverage == 1.0 and n == 2


def test_cost_metrics_reports_unknown_er_coverage():
    funds = pd.DataFrame([{"fund": "VTI", "usd": 500.0, "free": False}, {"fund": "MYSTERY", "usd": 500.0, "free": False}])
    blended, coverage, n = C.cost_metrics(funds, {"VTI": 0.03})
    assert blended == pytest.approx(0.03) and coverage == 0.5 and n == 2
    funds.loc[1, "usd"] = 1500.0
    assert C.cost_metrics(funds, {"VTI": 0.03})[1] == pytest.approx(0.25)  # dollar-weighted, not row count


def test_lookup_er_prefix_match_is_unique_only():
    ers = {"State Street Equity 500 Index Fund": 0.02, "Acme Fund A": 0.1, "Acme Fund B": 0.2}
    assert C.lookup_er("State Street Equity 500 Index", ers) == 0.02
    assert np.isnan(C.lookup_er("Acme Fund", ers))   # ambiguous -> unknown, not a guess


def test_top10_share():
    prof = {"A": pd.Series({f"s{i}": 1 / 20 for i in range(20)}), "META": pd.Series({"META": 1.0})}
    blocks = pd.Series({"A": 1000.0, "META": 100.0})
    # top 10: META 100 + nine issuers at 50 -> 550 of 1100
    assert C.top10_share(blocks, prof) == pytest.approx(550 / 1100)


@pytest.mark.parametrize("metric,v,out", [("MC: 10-yr median wealth", 2.071, "2.07x"),
                                          ("cost: blended expense ratio (%)", 0.0442, "0.044%"),
                                          ("reverse: liquid -40% breach share", 0.094, "9.4%"),
                                          ("scenario: ai_unwind_severe", -0.277, "-28%"),
                                          ("regret: SCV alone", float("nan"), "—")])
def test_fmt_cell(metric, v, out):
    assert C.fmt_cell(metric, v) == out
