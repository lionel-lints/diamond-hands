import math

import pandas as pd
import pytest

import reverse_stress as R
import stress as S


def test_sample_worlds_reproducible_and_bounded():
    a = R.sample_worlds(500, seed=1)
    b = R.sample_worlds(500, seed=1)
    pd.testing.assert_frame_equal(a, b)
    sleeves = a[list(R.LOADINGS)]
    assert list(a.columns) == list(R.LOADINGS) + [f"f_{k}" for k in R.FACTOR_RANGE]
    assert (sleeves >= R.BOUNDS[0]).all().all() and (sleeves <= R.BOUNDS[1]).all().all()
    for k, (lo, hi) in R.FACTOR_RANGE.items():
        assert a[f"f_{k}"].between(lo, hi).all(), k


def test_sample_worlds_cash_is_flat():
    w = R.sample_worlds(200, seed=2)
    assert (w["cash"] == 0).all()


def test_portfolio_results_hand_checked(fake_root):
    # Fake TOTAL_US = 50% ai_core / 25% us_large_other / 25% us_smid, with 0.1% carved out to labs_direct.
    blocks = pd.Series({"TOTAL_US": 1000.0, "PRIVATE_RE": 1000.0})
    worlds = pd.DataFrame([{"ai_core": -0.10, "us_large_other": -0.10, "us_smid": -0.10,
                            "labs_direct": -0.10, "private_re": -0.50}])
    r = R.portfolio_results(worlds, blocks, S.block_profiles()).iloc[0]
    # P&L: -10% of 1000 on liquid, -500 on real estate
    assert r.net_start == 2000 and r.liquid_start == 1000
    assert r.net_after == pytest.approx(1400)
    assert r.liquid_after == pytest.approx(900)


def test_describe_world_ranks_by_dollar_move(fake_root):
    sl = pd.Series({"ai_core": 500.0, "us_large_other": 250.0, "us_smid": 250.0, "labs_direct": 1.0})
    shocks = pd.Series({"ai_core": -0.50, "us_large_other": -0.90, "us_smid": 0.0, "labs_direct": 0.0})
    # ai_core moves $250, us_large_other $225: ai_core first despite smaller % shock
    assert R.describe_world(shocks, sl, top=2) == "ai_core -50%, us_large_other -90%"


def test_reference_return_hand_checked(fake_root):
    # Fake REFERENCE = 0.6 TOTAL_US (50% ai_core) + 0.4 INTL (60% ai_core, 40% intl_dev)
    worlds = pd.DataFrame([{"ai_core": -0.50}, {"intl_dev": -0.10}])
    ref = R.reference_return(worlds, S.block_profiles())
    assert ref[0] == pytest.approx(0.6 * 0.5 * -0.5 + 0.4 * 0.6 * -0.5)
    assert ref[1] == pytest.approx(0.4 * 0.4 * -0.10)


def test_breach_summary_counts_and_picks_mildest_by_reference(fake_root):
    blocks = pd.Series({"TOTAL_US": 1000.0})
    worlds = pd.DataFrame([
        {"ai_core": -0.10},                                            # no breach
        {"ai_core": -0.85},                                            # breach; ref -0.459
        {"ai_core": -0.85, "us_large_other": -0.85, "us_smid": -0.85},  # breach; ref much worse
    ])
    failures = {"liquid -40%": lambda r: r.liquid_after <= 0.60 * r.liquid_start}
    out = R.breach_summary({"p": blocks}, worlds, S.block_profiles(), failures).iloc[0]
    assert out.n_breaches == 2 and out.pct_worlds == pytest.approx(2 / 3)
    assert out.mildest_ref_return == pytest.approx(0.6 * 0.5 * -0.85 + 0.4 * 0.6 * -0.85)
    assert out.mildest_world.startswith("ai_core -85%")
    assert out.mean_ai_core == pytest.approx(-0.85)
    assert math.isnan(out.pct_bonds_negative)  # no bonds sleeve in this toy world
    assert out.mildest_factors == ""  # toy worlds carry no factor columns


def test_mildest_is_comparable_across_portfolios(fake_root):
    """A more defensive portfolio can only break in worlds at least as bad on the yardstick."""
    worlds = R.sample_worlds(20_000, seed=3)
    prof = S.block_profiles()
    ports = {"stocks": pd.Series({"TOTAL_US": 1000.0}),
             "half_bonds": pd.Series({"TOTAL_US": 500.0, "CASH": 500.0})}
    failures = {"-30%": lambda r: r.liquid_after <= 0.70 * r.liquid_start}
    out = R.breach_summary(ports, worlds, prof, failures).set_index("portfolio")
    assert out.n_breaches["half_bonds"] < out.n_breaches["stocks"]
    assert out.mildest_ref_return["half_bonds"] < out.mildest_ref_return["stocks"]


def test_uniform_shock_loses_that_share_of_non_cash(fake_root):
    blocks = pd.Series({"TOTAL_US": 1000.0, "INTL": 500.0, "CASH": 300.0, "PRIVATE_RE": 200.0})
    worlds = pd.DataFrame([{k: -0.2 for k in R.LOADINGS if k != "cash"}])
    r = R.portfolio_results(worlds, blocks, S.block_profiles()).iloc[0]
    assert r.net_after == pytest.approx(2000 - 0.2 * 1700)
    assert r.liquid_after == pytest.approx(1800 - 0.2 * 1500)


def test_bond_loadings_hedge_pure_growth_scares():
    """Documented assumption: with rates and credit unchanged, bonds rise when equities fall."""
    assert R.LOADINGS["bonds"]["equity"] < 0
    assert R.LOADINGS["bonds"]["rates"] < 0  # inflation/tightening hurts bonds
    assert R.LOADINGS["bonds"]["credit"] <= 0  # flight to quality: credit stress lifts Treasuries


def test_bonds_hedge_on_average_in_ai_crash_worlds():
    """History check: bonds rose in every 12-mo window with stocks < -20% (VBMFX, 1997-2026)."""
    w = R.sample_worlds(50_000, seed=4)
    assert w.bonds[w.ai_core < -0.5].mean() > 0


def test_breach_summary_handles_never_breaching(fake_root):
    blocks = pd.Series({"TOTAL_US": 1000.0})
    worlds = pd.DataFrame([{"ai_core": -0.85}])
    failures = {"liquid -99%": lambda r: r.liquid_after <= 0.01 * r.liquid_start}
    out = R.breach_summary({"p": blocks}, worlds, S.block_profiles(), failures).iloc[0]
    assert out.n_breaches == 0 and out.pct_worlds == 0
    assert math.isnan(out.mildest_ref_return)


def test_benchmark_blocks_swaps_only_stocks(fake_root):
    blocks = pd.Series({"TOTAL_US": 300.0, "SCV": 100.0, "META": 100.0, "BOND": 400.0, "CASH": 50.0, "PRIVATE_RE": 50.0})
    b = R.benchmark_blocks(blocks)
    assert b.sum() == pytest.approx(blocks.sum())
    assert b["TOTAL_US"] == pytest.approx(500 * 0.6) and b["INTL"] == pytest.approx(500 * 0.4)
    assert "SCV" not in b and "META" not in b
    for k in ["BOND", "CASH", "PRIVATE_RE"]:
        assert b[k] == blocks[k]


def test_benchmark_has_zero_lag_against_itself(fake_root):
    prof = S.block_profiles()
    blocks = R.benchmark_blocks(pd.Series({"TOTAL_US": 600.0, "INTL": 400.0, "BOND": 200.0}))
    r = R.portfolio_results(R.sample_worlds(1000, seed=5), blocks, prof)
    assert (r.liquid_after - r.bench_liquid_after).abs().max() == pytest.approx(0, abs=1e-6)


def test_lag_failure_trips_when_tilt_underperforms(fake_root):
    # All-SCV portfolio vs. a 60/40 benchmark; AI rips +50% while SCV is flat.
    blocks = pd.Series({"SCV": 1000.0})
    worlds = pd.DataFrame([{"ai_core": 0.50}])
    r = R.portfolio_results(worlds, blocks, S.block_profiles())
    # Benchmark ai_core $ = 600*0.5 (less $0.30 carved out to labs: 0.1% of 600 x 50%) + 400*0.6
    assert r.bench_liquid_after.iloc[0] == pytest.approx(1000 + 0.5 * (300 - 0.3 + 240))
    assert bool(R.failures()["lags market 10pp"](r).iloc[0])


def test_stocks_failure_ignores_bonds_and_cash(fake_root):
    blocks = pd.Series({"TOTAL_US": 1000.0, "BOND": 1000.0, "CASH": 1000.0})
    worlds = pd.DataFrame([{"ai_core": -0.85, "us_large_other": -0.85, "us_smid": -0.85, "labs_direct": -0.85, "bonds": 0.10}])
    r = R.portfolio_results(worlds, blocks, S.block_profiles()).iloc[0]
    assert r.stocks_start == pytest.approx(1000)
    assert r.stocks_after == pytest.approx(150)
    assert r.liquid_after == pytest.approx(3000 - 850 + 100)
    assert bool(R.failures()["stocks -50%"](pd.DataFrame([r])).iloc[0])
    assert not bool(R.failures()["liquid -40%"](pd.DataFrame([r])).iloc[0])


def test_stocks_failure_never_trips_without_stocks(fake_root):
    r = R.portfolio_results(pd.DataFrame([{"bonds": -0.5}]), pd.Series({"BOND": 100.0}), S.block_profiles())
    assert not R.failures()["stocks -50%"](r).any()


def test_relative_return_stats_ordering(fake_root):
    st = R.relative_return_stats(R.sample_worlds(2000, seed=6), pd.Series({"SCV": 500.0, "TOTAL_US": 500.0}), S.block_profiles())
    assert st["worst"] <= st["p05"] <= st["best"]


def test_correlated_uniforms_hit_target_correlation():
    import numpy as np
    target = np.array([[1, .7, .3], [.7, 1, .5], [.3, .5, 1]])
    u = R.correlated_uniforms(np.random.default_rng(0), 50_000, target)
    assert (u > 0).all() and (u < 1).all()
    assert np.allclose(np.corrcoef(u.T), target, atol=0.03)  # copula keeps rank correlation ~ target


def test_correlated_worlds_keep_factor_ranges_and_correlate():
    corr = [[1, .7, 0, 0, 0], [.7, 1, 0, 0, 0], [0, 0, 1, 0, .5], [0, 0, 0, 1, 0], [0, 0, .5, 0, 1]]
    w = R.sample_worlds(20_000, seed=7, corr=corr)
    for k, (lo, hi) in R.FACTOR_RANGE.items():
        assert w[f"f_{k}"].between(lo, hi).all(), k
    assert w.f_equity.corr(w.f_ai) > 0.6 and abs(w.f_equity.corr(w.f_rates)) < 0.05


def test_carry_shifts_only_the_named_sleeves():
    a = R.sample_worlds(2000, seed=8)
    b = R.sample_worlds(2000, seed=8, carry={"bonds": 0.05, "cash": 0.04})
    assert (b.cash == 0.04).all()
    unclipped = a.bonds.between(R.BOUNDS[0] + 0.05, R.BOUNDS[1] - 0.05)
    assert ((b.bonds - a.bonds)[unclipped] - 0.05).abs().max() < 1e-12
    pd.testing.assert_series_equal(a.ai_core, b.ai_core)


def test_current_carry_reads_latest_yields(fake_root, monkeypatch):
    monkeypatch.setattr(R, "ROOT", fake_root)
    (fake_root / "data/prices").mkdir(exist_ok=True)
    (fake_root / "data/prices/macro_monthly.csv").write_text("m,gs10,tb3m\n2026-08,4.0,3.0\n2026-09,5.0,\n")
    c = R.current_carry()
    assert c["bonds"] == pytest.approx(0.05) and c["cash"] == pytest.approx(0.03)  # last non-missing


def test_fitted_loadings_override_only_sleeves_with_history(fake_root, monkeypatch):
    monkeypatch.setattr(R, "ROOT", fake_root)
    (fake_root / "data/loadings_fitted.csv").write_text("sleeve,equity,ai,rates,credit\nbonds,0.01,0,-0.74,0.4\n")
    L = R.load_fitted_loadings()
    assert L["bonds"] == {"equity": 0.01, "ai": 0.0, "rates": -0.74, "credit": 0.4}
    assert L["ai_core"] == R.LOADINGS["ai_core"]  # no history -> assumed
    assert R.LOADINGS["bonds"]["rates"] == -0.6  # original untouched


@pytest.mark.parametrize("argv,must,must_not", [
    ([], ["copula-correlated", "fitted from history", "10-yr yield"], ["independent", "carry: off"]),
    (["--factors", "independent", "--loadings", "assumed", "--no-carry"],
     ["factors: independent", "hand-set assumed", "carry: off"], ["copula", "fitted"]),
])
def test_run_header_describes_the_actual_run(argv, must, must_not):
    """The printed description must match the flags (it once said 'independent' while
    running correlated factors by default)."""
    h = R.run_header(R.parse_args(argv))
    for m in must:
        assert m in h, (m, h)
    for m in must_not:
        assert m not in h, (m, h)


def test_module_docstring_matches_defaults():
    doc = R.__doc__
    assert R.DEFAULT_FACTORS == "historical" and "--factors historical" in doc
    assert R.DEFAULT_LOADINGS == "fitted" and "--loadings fitted" in doc
    assert "never shocked" not in doc  # cash earns carry


def test_plausibility_flags_out_of_range_and_clip():
    sl = pd.Series({"ai_core": 100.0, "bonds": 50.0, "crypto": 0.0})
    world = pd.Series({"f_equity": -0.55, "f_ai": 0.2, "ai_core": R.BOUNDS[0], "bonds": 0.02, "crypto": R.BOUNDS[0]})
    f = R.plausibility_flags(world, sl, {"equity": (-0.47, 0.59), "ai": (-0.47, 1.10)})
    assert "equity -0.55 outside history" in f
    assert "ai_core at clip" in f
    assert "crypto" not in f  # not held -> not flagged
    calm = world.copy()
    calm["f_equity"], calm["ai_core"] = -0.1, -0.5
    assert R.plausibility_flags(calm, sl, {"equity": (-0.47, 0.59)}) == ""


def test_describe_world_can_restrict_to_stock_sleeves():
    sl = pd.Series({"private_re": 1000.0, "ai_core": 100.0})
    shocks = pd.Series({"private_re": -0.4, "ai_core": -0.5})
    assert R.describe_world(shocks, sl).startswith("private_re")
    assert R.describe_world(shocks, sl, sleeves=R.STOCK_SLEEVES) == "ai_core -50%"


def test_private_re_is_never_fitted():
    """Public REITs are not a proxy for unlevered at-cost housing deals (calibrate.REFERENCE_ONLY)."""
    import calibrate as C
    assert "private_re" in C.REFERENCE_ONLY
    fitted = pd.read_csv(R.ROOT / "data/loadings_fitted.csv", index_col=0)
    assert "private_re" not in fitted.index
    assert R.load_fitted_loadings()["private_re"] == R.LOADINGS["private_re"]



def test_tips_earn_the_inflation_surprise_only():
    a = R.sample_worlds(3000, seed=9)
    b = R.sample_worlds(3000, seed=9, expected_inflation=0.024)
    surprise = a.f_inflation - 0.024
    ok = a.tips.between(R.BOUNDS[0] + 0.2, R.BOUNDS[1] - 0.2)
    assert ((b.tips - a.tips)[ok] - surprise[ok]).abs().max() < 1e-12
    pd.testing.assert_series_equal(a.bonds, b.bonds)  # nominal bonds don't accrue inflation
    pd.testing.assert_series_equal(a.cash, b.cash)


def test_real_failure_deflates_by_world_inflation(fake_root):
    blocks = pd.Series({"CASH": 1000.0})
    worlds = pd.DataFrame([{"cash": 0.0, "f_inflation": 0.50}, {"cash": 0.0, "f_inflation": 0.10}])
    r = R.portfolio_results(worlds, blocks, S.block_profiles())
    assert r.net_after_real.tolist() == pytest.approx([1000 / 1.5, 1000 / 1.1])
    fail = {"real": lambda r: r.net_after_real <= 0.70 * r.net_start}
    assert R.breach_summary({"p": blocks}, worlds, S.block_profiles(), fail).n_breaches.iloc[0] == 1


def test_real_failure_only_when_profile_asks(fake_root):
    import user_profile as P
    assert not any("real" in k for k in R.failures())
    P.current().settings["failures"]["real_networth_drawdown"] = 0.30
    assert "net worth -30% real" in R.failures()


def test_inflation_range_matches_history_not_widened():
    """A widened uniform inflation range made 5% the median world (see FACTOR_RANGE comment)."""
    lo, hi = R.FACTOR_RANGE["inflation"]
    hist = R.load_factor_hist_range()["inflation"]
    assert abs(lo - hist[0]) < 0.01 and abs(hi - hist[1]) < 0.01


def test_variants_table_runs_all_four_combinations(fake_root, monkeypatch):
    calls = []
    def fake_worlds(factors, loadings, no_carry):
        calls.append((factors, loadings, no_carry))
        shock = -0.85 if factors == "historical" else -0.10  # only historical worlds breach
        return pd.DataFrame([{"ai_core": shock}, {"ai_core": -0.05}])
    monkeypatch.setattr(R, "build_worlds", fake_worlds)
    failures = {"liquid -40%": lambda r: r.liquid_after <= 0.60 * r.liquid_start}
    out = R.variants_table({"p": pd.Series({"TOTAL_US": 1000.0})}, S.block_profiles(), failures)
    assert calls == [(f, l, False) for f, l in R.VARIANTS]
    assert list(out.columns) == ["variant", "failure", "portfolio", "breach", "cushion", "flags"]
    assert list(out.variant) == ["independent/assumed", "independent/fitted", "historical/assumed", "historical/fitted"]
    assert list(out.breach) == [0.0, 0.0, 0.5, 0.5]
    assert out.cushion.iloc[:2].isna().all() and out.cushion.iloc[2:].notna().all()
