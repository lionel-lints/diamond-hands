"""Shared model vocabulary: building blocks, how each is modeled, and how holdings map to them.

A *block* is a kind of fund (TOTAL_US, SCV, BOND, ...) or one directly held stock (block =
its ticker). Blocks are the same for every user; which funds a user owns or can buy is
profile data (profiles/<name>/), mapped to blocks through data/fund_catalog.csv, the
profile's plan menu, or an explicit `block` column in the holdings file.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Equity blocks: decomposed into sleeves via proxy-ETF holdings (data/fund_holdings/).
BLOCK_PROXY = {"TOTAL_US": {"ITOT": 1}, "SP500": {"IVV": 1}, "INTL": {"IXUS": 1},
               "SMALL_BLEND": {"ISCV": 0.5, "ISCG": 0.5}, "SCV": {"ISCV": 1},
               "MID": {"IMCB": 1}, "INTL_DEV": {"EFA": 1}, "EM": {"IEMG": 1}}
# Countries a block's real fund excludes but its proxy holds. Vanguard EM tracks FTSE, which
# classifies South Korea as developed; MSCI EAFE (EFA) has no Korea either (assumption A7).
BLOCK_EXCLUDE_LOCATIONS = {"EM": {"Korea (South)"}}
# Non-equity blocks map to a single sleeve.
BLOCK_FIXED_SLEEVE = {"BOND": "bonds", "TIPS": "tips", "TIPS_SHORT": "tips_short", "STABLE": "cash",
                      "CASH": "cash", "PRIVATE_RE": "private_re", "CRYPTO": "crypto"}
KNOWN_BLOCKS = set(BLOCK_PROXY) | set(BLOCK_FIXED_SLEEVE)
DIRECT_STOCK = "STOCK"  # holdings `block` value for a directly held stock; its block becomes its ticker

# Historical return series per block; later entries are fallbacks before the preferred series
# exists (assumption S4). Fund tickers come from data/prices/replay_monthly.csv; FF_* from the
# Ken French library (ff_monthly.csv, 1926-); SYN_* are built from FRED yields
# (stress.replay_returns): SYN_TBILL = 3-mo T-bill income, SYN_TSY = 10-yr Treasury carry
# minus 6 x yield change (a duration-6 Treasury proxy: no credit risk, so it overstates the
# bond hedge in credit crises: 2008 +14.5% vs. VBIIX +5%).
REPLAY_SERIES = {
    "TOTAL_US": ["VTSMX", "FF_us_mkt"], "SP500": ["VFINX", "FF_us_mkt"],
    # International before 1990 (no Ken French developed ex-US data): US market as a stand-in,
    # not flat. EAFE fell about as much as the US in 1973-74 and beat it later in the 1970s, so
    # this is roughly right for crashes and conservative for the stagflation decade (judgment).
    "INTL": ["VGTSX", "FF_dev_mkt", "FF_us_mkt"], "SMALL_BLEND": ["NAESX", "FF_small_blend"],
    "SCV": ["VISVX", "FF_us_small_value"], "TIPS": ["VIPSX", "VBMFX", "SYN_TSY"],
    "PRIVATE_RE": ["VGSIX"], "CRYPTO": ["ETH-USD"],
    # Cash and stable value earn T-bills (was 0%; assumption S5).
    "CASH": ["SYN_TBILL"], "STABLE": ["SYN_TBILL"],
    # Bonds live in the 401(k), whose option is the intermediate-term index (VBILX ~ VBIIX).
    "BOND": ["VBIIX", "VBMFX", "SYN_TSY"],
    "MID": ["VIMSX", "NAESX", "FF_us_mkt"], "INTL_DEV": ["VTMGX", "VGTSX", "FF_dev_mkt", "FF_us_mkt"],
    # No emerging-markets series before 1994; earlier windows use developed ex-US, then US (judgment).
    "EM": ["VEIEX", "FF_dev_mkt", "FF_us_mkt"],
    # Short TIPS (VTIP) only exist from 2012; earlier windows use T-bills, the closer analog
    # (little duration) than intermediate TIPS (assumption S12).
    "TIPS_SHORT": ["VTIP", "SYN_TBILL"],
}
DIRECT_STOCK_FALLBACK = ["VIGRX", "FF_us_big_growth"]

# Asset class of each block, for allocation targets. Direct stocks count as "us".
BLOCK_CLASS = {"TOTAL_US": "us", "SP500": "us", "MID": "us", "SMALL_BLEND": "us",
               "INTL": "intl", "INTL_DEV": "intl", "EM": "intl", "SCV": "scv",
               "BOND": "bonds", "TIPS": "tips", "TIPS_SHORT": "tips_short",
               "CASH": "cash", "STABLE": "cash"}

# Look-through sleeve label for non-equity holdings (lookthrough.explode).
NON_EQUITY_LABEL = {"CASH": "cash", "STABLE": "cash", "PRIVATE_RE": "private_real_estate",
                    "CRYPTO": "crypto", "BOND": "bonds", "TIPS": "tips", "TIPS_SHORT": "tips"}


def is_direct_stock(block):
    return block not in KNOWN_BLOCKS


def replay_series(block):
    return REPLAY_SERIES[block] if block in REPLAY_SERIES else [block] + DIRECT_STOCK_FALLBACK


def asset_class(block):
    return "us" if is_direct_stock(block) else BLOCK_CLASS.get(block, "other")


def catalog(root=None):
    """data/fund_catalog.csv as {key: block}; key is a ticker or exact fund name."""
    c = pd.read_csv((root or ROOT) / "data/fund_catalog.csv", dtype=str)
    return dict(zip(c.key, c.block))
