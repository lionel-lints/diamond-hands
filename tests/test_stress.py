import pandas as pd
import pytest

import stress as S


def row(**kw):
    base = {"ai_tier": "", "sleeve": "equity", "region": "US", "size": "large"}
    return pd.Series({**base, **kw})


@pytest.mark.parametrize("kw,expected", [
    ({"ai_tier": "core", "region": "EM"}, "ai_core"),
    ({"ai_tier": "adjacent_semis"}, "ai_adjacent"),
    ({"ai_tier": "adjacent_infra"}, "ai_adjacent"),
    ({"ai_tier": "narrative"}, "ai_adjacent"),
    ({"ai_tier": "big_tech_other"}, "us_large_other"),
    ({"sleeve": "small_value", "size": "small"}, "us_small_value"),
    ({"size": "mid"}, "us_smid"),
    ({"size": "small"}, "us_smid"),
    ({"region": "Developed ex-US", "size": "intl"}, "intl_dev"),
    ({"region": "EM", "size": "intl"}, "em"),
])
def test_sleeve_of(kw, expected):
    assert S.sleeve_of(row(**kw)) == expected


def test_block_profiles_sum_to_one_and_match_hand_values(fake_root):
    prof = S.block_profiles()
    for block, p in prof.items():
        assert p.sum() == pytest.approx(1.0), block
    assert prof["TOTAL_US"].to_dict() == pytest.approx({"ai_core": .50, "us_large_other": .25, "us_smid": .25})
    assert prof["SCV"].to_dict() == pytest.approx({"us_small_value": 1.0})
    assert prof["SMALL_BLEND"].to_dict() == pytest.approx({"us_small_value": .5, "us_smid": .5})
    assert prof["INTL"].to_dict() == pytest.approx({"ai_core": .6, "intl_dev": .4})


def test_today_blocks(fake_root):
    b = S.today_blocks()
    assert b.to_dict() == {"CASH": 400, "CRYPTO": 50, "INTL": 500, "META": 200,
                           "PRIVATE_RE": 1000, "SMALL_BLEND": 400, "TOTAL_US": 1000}


def test_today_blocks_rejects_unmapped(fake_root):
    with open(fake_root / "profiles/test/holdings/a_holdings.csv", "a") as f:
        f.write("taxable,taxable,V,Mystery Fund,MYST,,,,10,2026-10-20,\n")
    with pytest.raises(ValueError, match="MYST"):
        S.today_blocks()


def test_to_sleeves_conserves_dollars_and_adds_lab_exposure(fake_root):
    prof = S.block_profiles()
    blocks = pd.Series({"TOTAL_US": 1000.0, "BOND": 500.0})
    s = S.to_sleeves(blocks, prof)
    # LabA weight = 7.2e11 * 0.10 / 72e12 = 0.1%; LabB excluded from total-market funds
    assert s["labs_direct"] == pytest.approx(1.0)
    assert s.sum() == pytest.approx(1500)  # lab dollars displace, never add
    assert s["ai_core"] == pytest.approx(500 - 0.5)  # carved out of TOTAL_US sleeves pro rata
    assert s["bonds"] == pytest.approx(500)  # non-TOTAL_US blocks untouched


def test_hypothetical_pnl_hand_checked(fake_root):
    prof = S.block_profiles()
    blocks = pd.Series({"TOTAL_US": 1000.0, "PRIVATE_RE": 1000.0})
    out = S.hypothetical({"p": blocks}, prof).iloc[0]
    # -10% on 1000 stocks (labs included in the 1000), -50% on 1000 RE
    assert out.pnl == pytest.approx(-100 - 500)
    assert out.pct_networth == pytest.approx(-600 / 2000)
    assert out.pct_liquid == pytest.approx(-100 / 1000)


@pytest.fixture
def replay_prices(fake_root, monkeypatch):
    idx = ["2020-01", "2020-02", "2020-03", "2020-04"]
    (fake_root / "data/prices").mkdir()
    pd.DataFrame({
        "US": [100, 90, 80, 120],
        "TIPS_NEW": [None, None, 100, 110],  # preferred series, starts late
        "TIPS_OLD": [100, 100, 100, 100],    # fallback before the preferred one exists
        "UP": [100, 101, 102, 103],
    }, index=idx).to_csv(fake_root / "data/prices/replay_monthly.csv")
    monkeypatch.setattr(S.B, "REPLAY_SERIES", {"TOTAL_US": ["US"], "TIPS": ["TIPS_NEW", "TIPS_OLD"],
                                             "INTL": ["UP"], "CASH": [], "PRIVATE_RE": ["US"]})
    monkeypatch.setattr(S, "EPISODES", {"ep": ("2020-01", "2020-04"), "to_latest": ("2020-02", None)})


def test_replay_trough_end_and_fallback(replay_prices):
    blocks = pd.Series({"TOTAL_US": 100.0, "TIPS": 100.0, "PRIVATE_RE": 5000.0})
    r = S.replay({"p": blocks}).set_index("episode").loc["ep"]
    # TIPS: fallback flat for Feb/Mar, then preferred +10% in Apr -> 110.
    # Path: Feb 90+100, Mar 80+100, Apr 120+110 vs start 200. PRIVATE_RE ignored (illiquid).
    assert r.trough_pct == pytest.approx(-0.10)
    assert r.end_pct == pytest.approx(230 / 200 - 1)
    assert r.trough_usd == pytest.approx(-20)


def test_replay_trough_is_zero_when_never_below_start(replay_prices):
    r = S.replay({"p": pd.Series({"INTL": 100.0})}).set_index("episode")
    assert r.trough_pct["ep"] == 0
    assert r.end_pct["ep"] == pytest.approx(0.03)


def test_replay_open_ended_window_runs_to_latest(replay_prices):
    r = S.replay({"p": pd.Series({"INTL": 101.0})}).set_index("episode")
    assert r.end_pct["to_latest"] == pytest.approx(103 / 101 - 1)


def test_replay_cash_is_flat(replay_prices):
    r = S.replay({"p": pd.Series({"CASH": 100.0})}).set_index("episode")
    assert r.end_pct["ep"] == 0 and r.trough_pct["ep"] == 0


def test_em_block_drops_korea(fake_root):
    # Fake IEMG = TSMC 60% + SK Hynix 40% (Korea); FTSE EM has no Korea -> 100% TSMC (ai_core)
    prof = S.block_profiles()
    assert prof["EM"].to_dict() == pytest.approx({"ai_core": 1.0})


@pytest.fixture
def long_history(replay_prices, fake_root):
    """Adds FF and FRED files: a 4-month window where ETFs don't exist yet."""
    pd.DataFrame({"us_mkt": [0.0, -0.10, 0.0, 0.10], "us_small_value": [0.0] * 4, "us_small_growth": [0.0] * 4,
                  "us_big_growth": [0.0] * 4, "dev_mkt": [float("nan")] * 4},
                 index=["2020-01", "2020-02", "2020-03", "2020-04"]).to_csv(fake_root / "data/prices/ff_monthly.csv")
    pd.DataFrame({"cpi": [100.0, 110.0, 110.0, 121.0], "gs10": [5.0, 5.0, 6.0, 6.0], "tb3m": [12.0] * 4},
                 index=["2020-01", "2020-02", "2020-03", "2020-04"]).to_csv(fake_root / "data/prices/macro_monthly.csv")


def test_replay_returns_builds_synthetic_series(long_history):
    rets, cpi = S.replay_returns()
    assert rets.loc["2020-02", "SYN_TBILL"] == pytest.approx(0.01)            # 12%/12, last month's rate
    assert rets.loc["2020-03", "SYN_TSY"] == pytest.approx(0.05 / 12 - 6 * 0.01)  # carry - duration x dy
    assert rets.loc["2020-02", "FF_us_mkt"] == pytest.approx(-0.10)
    assert cpi.loc["2020-04"] == 121.0


def test_replay_reports_real_returns_and_series_used(long_history, monkeypatch):
    monkeypatch.setattr(S.B, "REPLAY_SERIES", {"CASH": ["SYN_TBILL"], "INTL": ["NOPE", "FF_dev_mkt", "FF_us_mkt"]})
    monkeypatch.setattr(S, "EPISODES", {"ep": ("2020-01", "2020-04")})
    r = S.replay({"p": pd.Series({"CASH": 100.0, "INTL": 100.0})}).iloc[0]
    # INTL falls back past an empty dev series to the US market (-10% then +10%); cash +1%/mo
    cash_end, intl_end = 100 * 1.01 ** 3, 100 * 0.9 * 1.0 * 1.1
    assert r.end_pct == pytest.approx((cash_end + intl_end) / 200 - 1)
    assert r.end_pct_real == pytest.approx((cash_end + intl_end) / 1.21 / 200 - 1)
    assert "INTL=FF_us_mkt" in r.series_used and "CASH=SYN_TBILL" in r.series_used


def test_stocks_on_fallback_flags_missing_and_stale_series(fake_root):
    d = fake_root / "data/prices"
    d.mkdir(exist_ok=True)
    pd.DataFrame({"TOTAL_US": [100, 100, 100], "STALE": [100, None, None]},
                 index=["2020-02", "2020-03", "2020-04"]).to_csv(d / "replay_monthly.csv")
    blocks = {"META": 1.0, "STALE": 1.0, "TOTAL_US": 1.0}  # META: no column; STALE: ends early
    assert sorted(S.stocks_on_fallback(blocks)) == ["META", "STALE"]


def test_every_block_has_a_pre_1990_fallback():
    """No block should silently go flat in the 1970s episodes (it once did for international)."""
    long = {"FF_us_mkt", "FF_small_blend", "FF_us_small_value", "FF_us_big_growth", "SYN_TSY", "SYN_TBILL"}
    exempt = {"PRIVATE_RE", "CRYPTO"}  # excluded from the replay / didn't exist
    for block, chain in S.B.REPLAY_SERIES.items():
        if block not in exempt:
            assert set(chain) & long, block
    assert set(S.B.DIRECT_STOCK_FALLBACK) & long
