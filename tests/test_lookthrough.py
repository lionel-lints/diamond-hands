import pandas as pd
import pytest

import lookthrough as L


@pytest.mark.parametrize("raw,expected", [("1,234.50", 1234.5), ("-", 0.0), ("", 0.0), (" 7 ", 7.0), (3, 3.0)])
def test_num_parses_ishares_numbers(raw, expected):
    assert L.num(raw) == expected


def test_load_fund_drops_cash_rows_and_normalizes(fake_root):
    f = L.load_fund("ITOT")
    assert "USD" not in set(f.Ticker)
    assert f.w.sum() == pytest.approx(1.0)
    assert f.set_index("Ticker").w["NVDA"] == pytest.approx(0.50)


@pytest.mark.parametrize("etf", ["IXUS", "ISCV"])
def test_load_fund_handles_header_variants(fake_root, etf):
    f = L.load_fund(etf)
    assert f.w.sum() == pytest.approx(1.0)
    assert list(f.columns) == ["Ticker", "Name", "Sector", "Location", "w"]


def test_load_fund_picks_latest_file(fake_root):
    newer = fake_root / "data/fund_holdings/IVV_2026-11-01.csv"
    newer.write_text((fake_root / "data/fund_holdings/ISCG_2026-10-20.csv").read_text())
    assert set(L.load_fund("IVV").Ticker) == {"SGRO"}


def test_load_positions_skips_template_and_blank_values(fake_root):
    pos = L.load_positions()
    assert "ZZZ" not in set(pos.key)
    assert "Checking" not in set(pos.key)
    assert pos.value_usd.sum() == 3550
    # key falls back to fund name when ticker is blank
    assert "Vanguard Morningstar Small Cap Index Admiral" in set(pos.key)


def test_size_buckets_follow_cap_rank_cutoffs(fake_root):
    s = L.size_buckets()
    assert s.to_dict() == {"NVDA": "large", "AAPL": "large", "XYZ": "mid", "SMOL": "small"}


def test_explode_conserves_dollars(fake_root):
    x = L.explode(L.load_positions())
    assert x.dollars.sum() == pytest.approx(3550)
    by_src = x.groupby("source").dollars.sum()
    assert by_src["VTI"] == pytest.approx(1000)
    assert by_src["Vanguard Morningstar Small Cap Index Admiral"] == pytest.approx(400)


def test_explode_splits_small_blend_into_value_and_growth(fake_root):
    x = L.explode(L.load_positions())
    sc = x[x.source == "Vanguard Morningstar Small Cap Index Admiral"].set_index("Ticker")
    assert sc.dollars["SVAL"] == pytest.approx(200)
    assert sc.sleeve["SVAL"] == "small_value"
    assert sc.dollars["SGRO"] == pytest.approx(200)
    assert sc.sleeve["SGRO"] == "equity"


def _rows(**kw):
    base = {"ticker": None, "fund_name": "?", "block": ""}
    return pd.DataFrame([{**base, **kw}])


def test_classify_order_explicit_catalog_then_plan_menu(fake_root):
    import user_profile as P
    prof = P.current()
    (prof.root / "plan_menu.csv").write_text("fund_name,model_block\nAcme 500 Index Fund,SP500\n")
    assert L.classify(_rows(ticker="VTI")).tolist() == ["TOTAL_US"]                  # catalog by ticker
    assert L.classify(_rows(fund_name="Acme 500 Index Fund")).tolist() == ["SP500"]   # plan menu by name
    assert L.classify(_rows(ticker="VTI", block="BOND")).tolist() == ["BOND"]         # explicit wins
    assert L.classify(_rows(ticker="AAPL", block="STOCK")).tolist() == ["AAPL"]       # direct stock -> ticker


@pytest.mark.parametrize("kw,match", [
    ({"ticker": "MYST", "fund_name": "Mystery Fund"}, "can't classify 'Mystery Fund'"),
    ({"fund_name": "X", "block": "BONDZ"}, "unknown block 'BONDZ'"),
    ({"fund_name": "X", "block": "STOCK"}, "marked STOCK but has no ticker"),
])
def test_classify_errors_name_the_problem(fake_root, kw, match):
    import user_profile as P
    with pytest.raises(P.ProfileError, match=match):
        L.classify(_rows(**kw))


def test_load_positions_rejects_unknown_accounts(fake_root):
    import user_profile as P
    with open(P.current().holdings_dir / "a_holdings.csv", "a") as f:
        f.write("brokerage2,taxable,V,Vanguard Total Stock Market ETF,VTI,,,,10,2026-10-20,\n")
    with pytest.raises(P.ProfileError, match="brokerage2"):
        L.load_positions()


def test_tag_matches_basket_by_ticker_and_by_name(fake_root):
    x = L.tag(L.explode(L.load_positions()))
    tiers = x.groupby("Name").ai_tier.first()
    assert tiers["NVIDIA"] == "core"
    assert tiers["TAIWAN SEMICONDUCTOR MANUFACTURING"] == "core"  # name match (no ticker in basket)
    assert tiers["META PLATFORMS INC CL A"] == "core"  # direct stock, ticker match
    assert tiers["APPLE"] == "big_tech_other"
    assert tiers["NESTLE SA"] == ""


def test_tag_regions_and_sizes(fake_root):
    x = L.tag(L.explode(L.load_positions())).set_index("Name")
    assert x.region["TAIWAN SEMICONDUCTOR MANUFACTURING"] == "EM"
    assert x.region["NESTLE SA"] == "Developed ex-US"
    assert x["size"]["NESTLE SA"] == "intl"
    assert x.region["Rental property LP"] == ""
    assert x["size"]["SMALL VALUE CO"] == "small"  # not in ITOT -> defaults to small


def test_tag_gives_direct_stock_its_real_sector(fake_root):
    x = L.tag(L.explode(L.load_positions()))
    meta = x[x.Ticker == "META"]
    assert (meta.Sector != "Direct stock").all()


def test_tag_aliases_google_share_classes(fake_root):
    x = pd.DataFrame([{"Ticker": "GOOG", "Name": "ALPHABET C", "Sector": "Communication", "Location": "United States",
                       "dollars": 1.0, "account": "", "source": "", "sleeve": "equity"}])
    assert L.tag(x).issuer.iloc[0] == "GOOGL"


def test_main_ai_core_share_hand_checked(fake_root, capsys):
    L.main([])
    detail = pd.read_csv(fake_root / "output/test/lookthrough_detail.csv", keep_default_na=False)
    eq = detail[detail.region != ""]
    # core = NVDA 500 (VTI) + META 200 + TSMC 300 (VXUS); stocks = 1000 + 200 + 500 + 400
    assert eq[eq.ai_tier == "core"].dollars.sum() == pytest.approx(1000)
    assert eq.dollars.sum() == pytest.approx(2100)
    ai = pd.read_csv(fake_root / "output/test/ai_exposure.csv").set_index("tier")
    assert ai.pct_stocks["core"] == pytest.approx(1000 / 2100)
    assert "Net worth tracked: $3,550" in capsys.readouterr().out
