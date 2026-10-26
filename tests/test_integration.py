"""End-to-end runs against the real data/ folder. Outputs go to a temp dir, never to output/."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import allocate as A
import frontier as F
import lookthrough as L
import reverse_stress as R
import stress as S
import user_profile as P
from conftest import REAL_ROOT, holdings_total

pytestmark = pytest.mark.integration


def test_lookthrough_conserves_every_dollar(real_root_sandbox):
    L.main([])
    detail = pd.read_csv(real_root_sandbox / "output/example/lookthrough_detail.csv", keep_default_na=False)
    assert detail.dollars.sum() == pytest.approx(holdings_total(), abs=1)


def test_lookthrough_shares_are_consistent(real_root_sandbox):
    L.main([])
    out = real_root_sandbox / "output/example"
    summ = pd.read_csv(out / "exposure_summary.csv", index_col=[0, 1]).pct
    for group in ["sleeve_pct_networth", "region_pct_stocks", "us_size_pct_us"]:
        assert summ[group].sum() == pytest.approx(1, abs=0.001), group
    ai = pd.read_csv(out / "ai_exposure.csv")
    assert ai.cum_pct_stocks.is_monotonic_increasing
    assert 0 < ai.pct_stocks.iloc[0] < 1


def test_every_equity_row_is_classified(real_root_sandbox):
    L.main([])
    d = pd.read_csv(real_root_sandbox / "output/example/lookthrough_detail.csv", keep_default_na=False)
    eq = d[d.sleeve.isin(["equity", "small_value"])]
    assert (eq.region != "").all()
    assert set(eq["size"]) <= {"large", "mid", "small", "intl"}


def test_every_basket_ticker_is_found_or_known_missing(real_root_sandbox):
    x = L.tag(L.explode(L.load_positions()))
    basket = pd.read_csv(REAL_ROOT / "data/ai_basket.csv", dtype=str).fillna("")
    found = set(x.Ticker[x.ai_tier != ""])
    missing = {t for t in basket.ticker if t} - found
    assert missing <= {"ARM"}, f"basket tickers not in any fund: {missing}"


def test_stress_portfolios_all_have_same_net_worth(real_root_sandbox):
    ports = A.portfolios()
    assert ports["0 Today"].sum() == pytest.approx(holdings_total(), abs=1)
    for name, b in ports.items():
        assert b.sum() == pytest.approx(ports["0 Today"].sum(), abs=1), name


def test_real_allocations_hit_targets_and_locate_assets(real_root_sandbox):
    state = A.account_state()
    for name, target in P.current().candidates.items():
        fills, summary = A.allocate(target, state)
        # Off-target only if frozen holdings (taxable/HSA, never sold) already exceed a class's
        # target; the overage then shows up as a shortfall elsewhere and nets to zero.
        frozen = state["fixed_blocks"].drop(list(P.current().satellites), errors="ignore")
        frozen_cls = frozen.groupby(frozen.index.map(A.B.asset_class)).sum()
        over = summary[summary.gap_usd > 1]
        for cls, row in over.iterrows():
            assert frozen_cls.get(cls, 0) >= row.target_usd - 1, (name, cls, summary)
        assert abs(summary.gap_usd.sum()) <= 1, (name, summary)
        # Asset location: bonds/TIPS only in tax-advantaged accounts, TIPS only in the 401(k)
        assert set(fills[fills["class"].isin(["bonds", "tips"])].account) <= {"workplace_401k", "roth_ira"}, name
        assert set(fills[fills["class"] == "tips"].account) <= {"workplace_401k"}, name
        # Every 401(k) fund is an index fund from the plan menu
        menu = pd.read_csv(P.current().plan_menu_path)
        k = fills[fills.account == "workplace_401k"].fund
        assert set(k) <= set(menu[menu.management == "index"].fund_name), name


def test_stress_main_outputs_are_sane(real_root_sandbox):
    S.main([])
    out = real_root_sandbox / "output/example"
    mix = pd.read_csv(out / "stress_sleeve_mix.csv", index_col=0)
    # Sleeves (including post-IPO lab exposure) sum to 100% of net worth
    for name, total in mix.sum().items():
        assert total == pytest.approx(1, abs=0.001), name
    hyp = pd.read_csv(out / "stress_hypothetical.csv")
    assert set(hyp.scenario) == set(pd.read_csv(REAL_ROOT / "data/scenarios.csv").scenario)
    piv = hyp.pivot(index="scenario", columns="portfolio", values="pct_liquid")
    up = ["ai_keeps_ripping", "lost_decade_2000s"]  # lost decade is positive nominal (10 yrs of income)
    mixed = [s for s in piv.index if s.endswith("_local") and s.startswith("japan")]  # sign depends on intl/bond share
    assert (piv.loc[up] > 0).all().all()
    assert (piv.drop(index=up + mixed) < 0).all().all()
    # Japan: the more international + bonds, the better (Defensive beats Aggressive in every Japan variant)
    for s in [x for x in piv.index if x.startswith("japan")]:
        assert piv.at[s, "Defensive 75/25"] > piv.at[s, "Aggressive 90/10"], s
    # More defensive (more bonds, more SCV) should lose less in a severe AI unwind
    sev = piv.loc["ai_unwind_severe"]
    assert sev["Defensive 75/25"] > sev["SCV 80/20"] > sev["Three-fund 80/20"] > sev["Aggressive 90/10"]
    rep = pd.read_csv(out / "stress_replay.csv")
    assert (rep.trough_pct <= 0).all() and (rep.trough_pct > -1).all()


def test_reverse_stress_main_outputs_are_sane(real_root_sandbox, monkeypatch):
    monkeypatch.setattr(R, "N_WORLDS", 5000)  # fast; main() reads the global at call time
    R.main([])
    out = pd.read_csv(real_root_sandbox / "output/example/reverse_stress_summary.csv")
    assert len(out) == (1 + len(P.current().candidates)) * len(R.failures())
    assert out.pct_worlds.between(0, 1).all()
    # Aggressive (90/10) should breach the drawdown failures at least as often as Defensive (75/25)
    for f in ["liquid -40%", "net worth -30%"]:
        p = out[out.failure == f].set_index("portfolio").pct_worlds
        assert p["Aggressive 90/10"] >= p["Defensive 75/25"], f


def test_frontier_front_is_non_dominated():
    prices = pd.read_csv(REAL_ROOT / "data/prices/monthly_adjclose.csv", index_col=0, parse_dates=True)
    rets, cov, mu = F.load_inputs(prices)
    rows = [{**dict(zip(F.ASSETS, w)), **F.stats(w, rets, cov, mu)} for w in F.grid(0.1)]
    df = pd.DataFrame(rows)
    front = F.pareto(df, "fwd_return", "vol")
    for _, p in front.iterrows():
        dominated = (df.fwd_return > p.fwd_return + 1e-9) & (df.vol < p.vol - 1e-9)
        assert not dominated.any()
    assert np.isclose(front.iloc[-1][list(F.ASSETS)].sum(), 1)


@pytest.mark.parametrize("script", ["style_windows.py"])
def test_read_only_scripts_run_from_cli(script):
    r = subprocess.run([sys.executable, str(REAL_ROOT / "scripts" / script)], capture_output=True, text=True, cwd=REAL_ROOT)
    assert r.returncode == 0, r.stderr
    assert "Dot-com" in r.stdout


@pytest.mark.network
def test_live_ishares_download_is_csv(tmp_path, monkeypatch):
    import fetch_fund_holdings as FH

    monkeypatch.setattr(FH, "ROOT", tmp_path)
    monkeypatch.setattr(FH, "OUT", tmp_path)
    monkeypatch.setattr(FH, "ISHARES", {"IVV": FH.ISHARES["IVV"]})
    FH.main()
    f = next(tmp_path.glob("IVV_*.csv"))
    L_lines = f.read_text().splitlines()
    assert any(l.startswith("Ticker,") for l in L_lines)


def test_401k_menu_is_well_formed():
    menu = pd.read_csv(REAL_ROOT / "profiles/example/plan_menu.csv", dtype=str).fillna("")
    assert menu.fund_name.is_unique
    assert set(menu.management) <= {"index", "active"}
    # Every non-target-date index fund maps to a model block; active funds are never modeled
    idx = menu[(menu.management == "index") & (menu.asset_class != "target date")]
    assert (idx.model_block != "").all(), idx[idx.model_block == ""].fund_name.tolist()
    assert (menu[menu.management == "active"].model_block.isin(["", "STABLE"])).all()
    er = pd.to_numeric(menu.expense_ratio_pct)
    assert er.notna().all() and er.between(0, 2).all()
    assert (menu[menu.management == "index"].expense_ratio_pct.astype(float) <= 0.15).all()



def test_reverse_stress_ranking_is_robust_to_factor_and_loading_choices(real_root_sandbox):
    """The candidate ranking on liquid -40% must not depend on the factor/loading design."""
    prof = S.block_profiles()
    ports = A.portfolios()
    names = ["Defensive 75/25", "SCV 80/20", "Three-fund 80/20", "Aggressive 90/10"]
    ports = {k: ports[k] for k in names}
    fail = {"liquid -40%": R.failures()["liquid -40%"]}
    for corr in [None, R.load_factor_corr()]:
        for loadings in [None, R.load_fitted_loadings()]:
            w = R.sample_worlds(20_000, seed=11, corr=corr, loadings=loadings, carry=R.current_carry())
            p = R.breach_summary(ports, w, prof, fail).set_index("portfolio").pct_worlds
            assert list(p.sort_values().index) == names, p


def test_calibrated_worlds_match_history_on_bonds_in_crashes(real_root_sandbox):
    """History (VBIIX, 1997-2026): bonds averaged +3.8% in 12-mo windows with stocks < -20%.
    The default calibrated worlds should not have bonds losing money on average there."""
    w = R.sample_worlds(50_000, seed=12, corr=R.load_factor_corr(), loadings=R.load_fitted_loadings(),
                        carry=R.current_carry())
    assert w.bonds[w.f_equity < -0.2].mean() > -0.01


def test_example_profile_runs_end_to_end(example_sandbox):
    """The engine works for someone else: different accounts, plan menu, direct stock (AAPL)."""
    L.main([])
    A.main([])
    S.main([])
    R.main([])
    out = example_sandbox / "output/example"
    for f in ["lookthrough_detail", "allocations", "stress_hypothetical", "reverse_stress_summary"]:
        assert (out / f"{f}.csv").exists(), f
    state = A.account_state()
    for name, target in P.current().candidates.items():
        fills, summary = A.allocate(target, state)
        # Off-target only if frozen holdings (taxable/HSA, never sold) already exceed a class's
        # target; the overage then shows up as a shortfall elsewhere and nets to zero.
        frozen = state["fixed_blocks"].drop(list(P.current().satellites), errors="ignore")
        frozen_cls = frozen.groupby(frozen.index.map(A.B.asset_class)).sum()
        over = summary[summary.gap_usd > 1]
        for cls, row in over.iterrows():
            assert frozen_cls.get(cls, 0) >= row.target_usd - 1, (name, cls, summary)
        assert abs(summary.gap_usd.sum()) <= 1, (name, summary)
        assert set(fills[fills["class"].isin(["bonds", "tips"])].account) <= {"workplace_401k", "roth_ira"}
    ports = A.portfolios()
    assert all(abs(b.sum() - holdings_total()) < 1 for b in ports.values())
    assert "AAPL" in S.block_profiles()  # direct stock handled generically


def test_scenarios_file_is_consistent():
    """Every scenario states its horizon; cash earns T-bills over it; short TIPS never earn
    less than cash; every sleeve column is present."""
    sc = pd.read_csv(REAL_ROOT / "data/scenarios.csv").set_index("scenario")
    assert sc.horizon_years.notna().all() and (sc.horizon_years > 0).all()
    assert (sc.tips_short >= sc.cash - 1e-9).all()
    assert (sc.cash > 0).all()
    for s in ["ai_unwind_credit_fed_cuts", "ai_unwind_credit_sticky_inflation", "lost_decade_2000s"]:
        assert s in sc.index
    # The two credit variants differ in whether bonds hedge: the whole point of the pair
    assert sc.at["ai_unwind_credit_fed_cuts", "bonds"] > 0 > sc.at["ai_unwind_credit_sticky_inflation", "bonds"]
    sleeves = set(R.LOADINGS)
    assert sleeves <= set(sc.columns), sleeves - set(sc.columns)


def test_montecarlo_runs_and_is_sane(real_root_sandbox):
    import montecarlo as M
    df = M.run(P.current(), n_paths=200, runs=[("hist 1990-2026", None), ("CMA median", "median")])
    assert set(df.run) == {"hist 1990-2026", "CMA median"}
    assert len(df) == 2 * (1 + len(P.current().candidates))
    assert (df.real_x_10y_p05 <= df.real_x_10y_p50).all()
    for c in ["p_maxdd_30", "p_maxdd_40", "p_underwater_after_5y_real", "p_lag_5y"]:
        assert df[c].between(0, 1).all(), c
    assert (df.p_maxdd_40 <= df.p_maxdd_30).all()
    assert (df.lag_threshold == P.current().failures["lag_vs_benchmark"]).all()  # label matches the test run
    h = df[df.run == "hist 1990-2026"].set_index("portfolio")
    # Today gets the same savings: its 10-yr wealth can't be inflated or deflated by a missing stream
    assert h.real_x_10y_p50["0 Today (buy & hold)"] > 1.5
    assert h.p_maxdd_40["Defensive 75/25"] < h.p_maxdd_40["Aggressive 90/10"]


def test_cma_forecast_applies_only_over_its_horizon(real_root_sandbox):
    import montecarlo as M
    assert M.cma_horizon_months("median") == 120 and M.cma_horizon_months("GMO") == 84


def test_robust_frontier_runs_on_a_small_grid(real_root_sandbox, monkeypatch):
    import robust_frontier as RF
    monkeypatch.setattr(RF, "STOCK_SHARES", [0.8, 0.9])
    monkeypatch.setattr(RF, "INTL_OF_STOCKS", [0.3])
    monkeypatch.setattr(RF, "SCV_OF_STOCKS", [0.0, 0.2])
    monkeypatch.setattr(RF, "TIPS_OF_FIXED", [0.5])
    df = RF.evaluate(P.current(), list(RF.grid()), n_paths=200, runs=[("hist 1990-2026", None), ("CMA median", "median")])
    assert len(df) == 4 * 2
    assert (df.alloc_gap_usd <= 1).all()          # every mix is buyable in the real accounts
    assert (df.groupby("run").shortfall.min() == 0).all()   # each run has a frontier


def test_regret_runs_and_scv_regret_grows_with_tilt(real_root_sandbox):
    import regret as G
    df = G.run(P.current()).set_index("portfolio")
    sweep = df[df.index.str.startswith("sweep") & (df.alloc_gap_usd <= 1)]   # buyable mixes only
    assert len(sweep) >= 4
    s = sweep[sweep.scv_only_worst_ending != ""].scv_only_worst_5y
    assert (s.diff().dropna() < 0).all()   # each step up in SCV deepens its worst 5-yr lag
    assert df.loc["Aggressive 90/10"].scv_within_lag_limit in (True, False)


def test_comparison_table_covers_every_candidate(real_root_sandbox):
    import compare as C
    t = C.build(P.current(), n_paths=200)
    assert list(t.columns) == ["0 Today"] + list(P.current().candidates)
    assert t.notna().loc[[m for m in t.index if not m.startswith("regret")]].all().all()
    assert (t.loc["cost: blended expense ratio (%)"] < 0.15).all()  # index funds only
    assert (t.loc["cost: ER coverage of liquid $"] > 0.99).all()  # every fund has a sourced ER


def test_scenario_matrix_covers_scenarios_and_blends(real_root_sandbox):
    import scenario_matrix as SM
    t, ports = SM.outcomes(P.current())
    sc = pd.read_csv(REAL_ROOT / "data/scenarios.csv")
    assert len([i for i in t.index if i.startswith("scenario:")]) == len(sc)
    assert t.notna().all().all()
    run = [i for i in t.index if "AI run" in i][0]
    assert (t.loc[run] > 0).all()        # a gain episode shows its gain, not a zero "worst point"
    # more stocks -> worse Japan-contagion outcome along the target's shape
    j = [i for i in t.index if "japan_us_to_trough_contagion" in i][0]
    assert t.at[j, "target shape @ 70% stocks"] > t.at[j, "target shape @ 80% stocks"] > t.at[j, "target shape @ 100% stocks"]


def test_indicators_measure_offline(real_root_sandbox, monkeypatch):
    import indicators as I
    monkeypatch.setattr(I, "ROOT", real_root_sandbox)
    monkeypatch.setattr(I, "CAPE_CACHE", REAL_ROOT / "data/prices/cape.csv")
    m = I.measure(offline=True)
    assert set(m) == {"cape", "sp500_top10", "core_ai_us_market", "intl_favored_share"}
    assert 5 < m["cape"][0] < 80 and 0 < m["sp500_top10"][0] < 1 and 0 < m["core_ai_us_market"][0] < 1
    cfg = P.current().settings["review_indicators"]
    assert set(cfg) == set(m) and all(c["shrink_below"] < c["baseline"] for c in cfg.values())
