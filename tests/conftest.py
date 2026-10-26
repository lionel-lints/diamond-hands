"""Shared fixtures: a tiny fake repo whose numbers are easy to check by hand.

Fake ITOT (total US) weights: NVDA 50%, AAPL 25%, XYZ 17%, SMOL 8% (+ a cash row that must be ignored).
Cap-rank cut points: starting cumulative weight 0 / .50 -> large, .75 -> mid, .92 -> small.
"""
import shutil
import textwrap
from pathlib import Path

import pandas as pd
import pytest

import blocks as B
import lookthrough as L
import reverse_stress as R
import stress as S
import user_profile as P

REAL_ROOT = Path(__file__).resolve().parent.parent

PROFILE_TOML = """
name = "Test"
horizon_years = 3
emergency_fund = 100
satellites = ["META", "CRYPTO", "PRIVATE_RE"]
ips_target = "t1"

[failures]
liquid_drawdown = 0.40
networth_drawdown = 0.30
lag_vs_benchmark = 0.10
stocks_drawdown = 0.50

[candidates.t1]
us = 0.55
intl = 0.25
scv = 0.05
bonds = 0.15
"""

ACCOUNTS_TOML = """
[accounts]
"401k" = { mode = "movable" }
roth_ira = { mode = "movable" }
taxable = { mode = "frozen", cash_is_new_money = true }
re = { mode = "frozen" }
rh = { mode = "frozen" }
bank = { mode = "bank" }

[new_money]
name = "taxable_new"

[funds."401k"]
us = [["SSSYX", 1.0]]
intl = [["BTMKX", 0.75], ["VEMAX", 0.25]]
scv = [["VSIAX", 1.0]]
bonds = [["VBILX", 1.0]]
tips = [["FIPDX", 1.0]]

[funds.roth_ira]
us = [["VTI", 1.0]]
intl = [["VXUS", 1.0]]
scv = [["VBR", 1.0]]
bonds = [["BND", 1.0]]
tips = [["VAIPX", 1.0]]
tips_short = [["VTIP", 1.0]]
reserve = [["VMFXX", 1.0]]

[funds.taxable_new]
us = [["VTI", 1.0]]
intl = [["VXUS", 1.0]]
reserve = [["VCTXX", 1.0]]

[location]
fill_order = ["tips_short", "bonds", "tips", "reserve", "scv", "intl", "us"]

[location.order]
tips_short = ["roth_ira"]
bonds = ["401k", "roth_ira"]
tips = ["401k", "roth_ira"]
reserve = ["taxable_new", "roth_ira"]
scv = ["roth_ira", "401k"]
intl = ["taxable_new", "roth_ira", "401k"]
us = ["401k", "roth_ira", "taxable_new"]
"""

ISHARES_PREAMBLE = 'iShares Fake ETF\nFund Holdings as of,"Oct 19, 2026"\n\n'
STD_HEADER = "Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Quantity,Price,Location,Exchange,Currency,FX Rate,Market Currency,Accrual Date"


def std_row(tk, name, sector, mv, loc="United States", asset="Equity"):
    return f'"{tk}","{name}","{sector}","{asset}","{mv:,.2f}","0","0","0","0","{loc}","X","USD","1","USD","-"'


def write_std(path, rows):
    path.write_text(ISHARES_PREAMBLE + STD_HEADER + "\n" + "\n".join(rows) + "\n")


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    fh = tmp_path / "data/fund_holdings"
    fh.mkdir(parents=True)
    write_std(fh / "ITOT_2026-10-20.csv", [
        std_row("NVDA", "NVIDIA", "Information Technology", 5000),
        std_row("AAPL", "APPLE", "Information Technology", 2500),
        std_row("XYZ", "XYZ CORP", "Industrials", 1700),
        std_row("SMOL", "SMOL INC", "Financials", 800),
        std_row("USD", "USD CASH", "Cash and/or Derivatives", 999, asset="Cash"),
    ])
    write_std(fh / "IVV_2026-10-20.csv", [
        std_row("NVDA", "NVIDIA", "Information Technology", 6000),
        std_row("AAPL", "APPLE", "Information Technology", 4000),
    ])
    write_std(fh / "ISCG_2026-10-20.csv", [std_row("SGRO", "SMALL GROWTH CO", "Health Care", 100)])
    write_std(fh / "IMCB_2026-10-20.csv", [std_row("XYZ", "XYZ CORP", "Industrials", 100)])
    write_std(fh / "EFA_2026-10-20.csv", [std_row("NESN", "NESTLE SA", "Consumer Staples", 100, loc="Switzerland")])
    write_std(fh / "IEMG_2026-10-20.csv", [
        std_row("2330", "TAIWAN SEMICONDUCTOR MANUFACTURING", "Information Technology", 600, loc="Taiwan"),
        std_row("000660", "SK HYNIX", "Information Technology", 400, loc="Korea (South)"),  # dropped: FTSE EM has no Korea
    ])
    # IXUS has an extra Type column; ISCV names its weight column "Market Weight". Both seen in real files.
    (fh / "IXUS_2026-10-20.csv").write_text(ISHARES_PREAMBLE + textwrap.dedent("""\
        Ticker,Name,Type,Sector,Asset Class,Market Value,Weight (%),Notional Value,Quantity,Price,Location,Exchange,Currency,FX Rate,Market Currency,Accrual Date
        "2330","TAIWAN SEMICONDUCTOR MANUFACTURING","EQUITY","Information Technology","Equity","600.00","60","0","0","0","Taiwan","TWSE","USD","1","TWD","-"
        "NESN","NESTLE SA","EQUITY","Consumer Staples","Equity","400.00","40","0","0","0","Switzerland","SIX","USD","1","CHF","-"
        """))
    (fh / "ISCV_2026-10-20.csv").write_text(ISHARES_PREAMBLE + textwrap.dedent("""\
        Ticker,Name,Type,Sector,Asset Class,Market Value,Notional Value,Quantity,Price,Location,Exchange,Currency,FX Rate,Market Currency,Accrual Date,Market Weight,Notional Weight
        "SVAL","SMALL VALUE CO","EQUITY","Financials","Equity","100.00","0","0","0","United States","NYSE","USD","1","USD","-","100","100"
        """))

    # Fake profile "test": same accounts and dollars the hand-checked tests rely on.
    prof = tmp_path / "profiles/test"
    h = prof / "holdings"
    h.mkdir(parents=True)
    cols = "account,account_type,institution,fund_name,ticker,block,shares,price,value_usd,as_of_date,notes"
    (h / "a_holdings.csv").write_text(cols + "\n" + "\n".join([
        "taxable,taxable,V,Vanguard Total Stock Market ETF,VTI,,,,1000,2026-10-20,",
        "taxable,taxable,V,Vanguard Total Intl Stock ETF,VXUS,,,,500,2026-10-20,",
        "taxable,taxable,V,Meta Platforms Inc Cl A,META,STOCK,,,200,2026-10-20,",
        "roth_ira,roth,V,Vanguard Federal Money Market,VMFXX,,,,100,2026-10-20,",
    ]) + "\n")
    (h / "b_holdings.csv").write_text(cols + "\n" + "\n".join([
        "401k,pre-tax,Acme,Vanguard Morningstar Small Cap Index Admiral,,SMALL_BLEND,,,400,2026-10-20,",
        "bank,cash,,High-yield savings,,CASH,,,300,2026-10-20,",
        "re,private,,Rental property LP,,PRIVATE_RE,,,1000,2026-10-20,",
        "rh,taxable,Broker,Ethereum,ETH,,,,50,2026-10-20,",
        "bank,cash,,Checking,,CASH,,,,2026-10-20,TODO blank value must be skipped",
    ]) + "\n")
    (h / "_template.csv").write_text(cols + "\nshould,be,ignored,X,ZZZ,,,,123,,\n")
    (prof / "profile.toml").write_text(PROFILE_TOML)
    (prof / "accounts.toml").write_text(ACCOUNTS_TOML)
    shutil.copy(REAL_ROOT / "data/fund_catalog.csv", tmp_path / "data/fund_catalog.csv")

    (tmp_path / "data/ai_basket.csv").write_text(textwrap.dedent("""\
        ticker,name_match,tier,rationale
        NVDA,NVIDIA,core,x
        META,META PLATFORMS,core,x
        ,TAIWAN SEMICONDUCTOR,core,x
        AAPL,APPLE,big_tech_other,x
        """))
    (tmp_path / "data/private_ai.csv").write_text(
        "lab,assumed_valuation_usd,float_post_lockup,in_total_market_funds,in_sp500_fund,notes\n"
        "LabA,7.2e11,0.10,yes,no,w = 7.2e10/72e12 = 0.1%\n"
        "LabB,9.9e12,0.50,no,no,excluded from total market funds\n")
    (tmp_path / "data/scenarios.csv").write_text(
        "scenario,ai_core,ai_adjacent,us_large_other,us_smid,us_small_value,intl_dev,em,bonds,tips,cash,private_re,crypto,labs_direct,description\n"
        "flat_minus10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-50,-10,-10,everything -10% except RE -50%\n")

    _point_engine_at(monkeypatch, tmp_path)
    P.use(prof)
    return tmp_path


def _point_engine_at(monkeypatch, root):
    """Shared data from `root`/data, outputs to `root`/output; restore the active profile after."""
    for mod in (L, S, R, B):
        monkeypatch.setattr(mod, "ROOT", root)
    monkeypatch.setattr(P, "OUTPUT_ROOT", root / "output")
    monkeypatch.setattr(P, "_current", None)


@pytest.fixture
def real_root_sandbox(tmp_path, monkeypatch):
    """Real shared data/ (symlinked, read-only use) + the fictional example profile, with
    outputs redirected to tmp so the real output/ is untouched."""
    (tmp_path / "data").symlink_to(REAL_ROOT / "data")
    _point_engine_at(monkeypatch, tmp_path)
    P.use(REAL_ROOT / "profiles/example")
    return tmp_path


@pytest.fixture
def example_sandbox(tmp_path, monkeypatch):
    """Real shared data + the fictional example profile."""
    (tmp_path / "data").symlink_to(REAL_ROOT / "data")
    _point_engine_at(monkeypatch, tmp_path)
    P.use(REAL_ROOT / "profiles/example")
    return tmp_path


def holdings_total(profile=None):
    d = (profile or P.current()).holdings_dir
    frames = [pd.read_csv(f) for f in d.glob("*.csv") if not f.name.startswith("_")]
    return pd.concat(frames).value_usd.sum()
