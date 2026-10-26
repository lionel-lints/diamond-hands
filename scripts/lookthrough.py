"""Look-through exposure across every account: region, size, sector, top names, AI basket.

Each holding is classified into a block (blocks.py), every fund block is replaced by its
proxy ETF's published holdings (data/fund_holdings/) scaled to the dollars held, and direct
stocks pass straight through. Holdings come from the active profile (user_profile.py).
Run: uv run --with pandas python scripts/lookthrough.py [--profile NAME]
"""
import csv
import io
import sys
from pathlib import Path

import pandas as pd

import blocks as B
import user_profile as P

ROOT = Path(__file__).resolve().parent.parent  # shared data

EM = {
    "China", "Taiwan", "India", "Korea (South)", "Brazil", "Saudi Arabia", "South Africa",
    "Mexico", "Indonesia", "Thailand", "Malaysia", "United Arab Emirates", "Qatar", "Kuwait",
    "Poland", "Turkey", "Chile", "Philippines", "Greece", "Peru", "Hungary", "Czech Republic",
    "Colombia", "Egypt",
}
ISSUER_ALIASES = {"GOOG": "GOOGL"}


def num(s):
    s = str(s).replace(",", "").strip()
    return float(s) if s not in ("", "-") else 0.0


def classify(df, profile=None):
    """Block for each holding: an explicit `block` column wins (STOCK -> the ticker), then the
    shared fund catalog by ticker or fund name, then the profile's plan menu by fund name."""
    profile = profile or P.current()
    cat = B.catalog(ROOT)
    menu = P.plan_menu_blocks(profile)
    explicit = df["block"].fillna("") if "block" in df else pd.Series("", index=df.index)

    def one(i, r):
        e = explicit[i]
        if e == B.DIRECT_STOCK:
            if not isinstance(r.ticker, str) or not r.ticker:
                raise P.ProfileError(f"{profile.holdings_dir}: {r.fund_name!r} is marked STOCK but has no ticker")
            return r.ticker
        if e:
            if e not in B.KNOWN_BLOCKS:
                raise P.ProfileError(f"{profile.holdings_dir}: unknown block {e!r} for {r.fund_name!r}; use one of "
                                     f"{sorted(B.KNOWN_BLOCKS)} or STOCK")
            return e
        for k in (r.ticker, r.fund_name):
            if isinstance(k, str) and k in cat:
                return cat[k]
        if r.fund_name in menu:
            return menu[r.fund_name]
        raise P.ProfileError(f"{profile.holdings_dir}: can't classify {r.fund_name!r} (ticker {r.ticker!r}). "
                             f"Add a `block` column value, a data/fund_catalog.csv row, or a plan_menu.csv entry.")

    return pd.Series([one(i, r) for i, r in df.iterrows()], index=df.index)


def load_positions(profile=None):
    profile = profile or P.current()
    files = [f for f in sorted(profile.holdings_dir.glob("*.csv")) if not f.name.startswith("_")]
    if not files:
        raise P.ProfileError(f"{profile.holdings_dir}: no holdings CSVs")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["key"] = df.ticker.fillna(df.fund_name)
    df = df[df.value_usd.notna() & (df.value_usd != 0)].copy()
    unknown = set(df.account) - set(profile.accounts)
    if unknown:
        raise P.ProfileError(f"{profile.root / 'accounts.toml'}: holdings use accounts not listed in [accounts]: {sorted(unknown)}")
    df["block"] = classify(df, profile)
    return df


def load_fund(tk):
    f = max((ROOT / "data/fund_holdings").glob(f"{tk}_*.csv"))
    lines = f.read_text(encoding="utf-8-sig").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Ticker,"))
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    df = pd.DataFrame(rows)
    df = df[df["Asset Class"] == "Equity"].copy()
    df["w"] = df["Market Value"].map(num)  # weight column name differs between iShares files
    df["w"] /= df.w.sum()
    return df[["Ticker", "Name", "Sector", "Location", "w"]]


def size_buckets():
    """Morningstar-style cut on US cap rank: top 70% of market cap large, next 20% mid, rest small."""
    itot = load_fund("ITOT").sort_values("w", ascending=False)
    cum = itot.w.cumsum()
    size = pd.Series(pd.cut(cum - itot.w, [-1, 0.70, 0.90, 2], labels=["large", "mid", "small"]).astype(str).values, index=itot.Ticker)
    return size[~size.index.duplicated()]


def explode(positions):
    """One row per (position, underlying stock) with dollar exposure."""
    rows = []
    for _, p in positions.iterrows():
        k, v, block = p.key, p.value_usd, p.block
        if block in B.BLOCK_PROXY:
            for etf, share in B.BLOCK_PROXY[block].items():
                h = load_fund(etf)
                rows.append(h.assign(dollars=h.w * v * share, account=p.account, source=k,
                                     sleeve="small_value" if etf == "ISCV" else "equity"))
        elif B.is_direct_stock(block):
            rows.append(pd.DataFrame([{"Ticker": block, "Name": p.fund_name.upper(), "Sector": "Direct stock",
                                       "Location": "United States", "dollars": v, "account": p.account,
                                       "source": k, "sleeve": "equity"}]))
        else:
            sleeve = B.NON_EQUITY_LABEL[block]
            rows.append(pd.DataFrame([{"Ticker": "", "Name": k, "Sector": sleeve, "Location": "",
                                       "dollars": v, "account": p.account, "source": k, "sleeve": sleeve}]))
    return pd.concat(rows, ignore_index=True)


def ai_basket_path():
    return P.current().ai_basket_path or ROOT / "data/ai_basket.csv"


def tag(x):
    basket = pd.read_csv(ai_basket_path(), dtype=str).fillna("")
    by_ticker = {r.ticker: r.tier for r in basket.itertuples() if r.ticker}
    by_name = [(r.name_match, r.tier) for r in basket.itertuples() if not r.ticker]
    meta_sector = load_fund("ITOT").drop_duplicates("Ticker").set_index("Ticker").Sector

    def tier(r):
        if r.Ticker in by_ticker:
            return by_ticker[r.Ticker]
        return next((t for n, t in by_name if str(r.Name).upper().startswith(n)), "")

    x["ai_tier"] = x.apply(tier, axis=1)
    x["issuer"] = x.Ticker.replace(ISSUER_ALIASES).where(x.Ticker != "", x.Name)
    x.loc[x.Sector == "Direct stock", "Sector"] = x.Ticker.map(meta_sector)
    x["region"] = x.Location.map(lambda l: "" if not l else "US" if l == "United States" else "EM" if l in EM else "Developed ex-US")
    sizes = size_buckets()
    x["size"] = ""
    us = x.region == "US"
    x.loc[us, "size"] = x.loc[us, "Ticker"].map(sizes).fillna("small")
    x.loc[x.region.isin(["Developed ex-US", "EM"]), "size"] = "intl"
    return x


def table(x, col, base):
    t = x.groupby(col).dollars.sum().sort_values(ascending=False)
    return pd.DataFrame({"dollars": t.round(0), "pct": (t / base).round(4)})


def main(argv=None):
    P.from_argv(sys.argv[1:] if argv is None else argv)
    OUT = P.current().output_dir
    print(P.current().summary(), "\n")
    pos = load_positions()
    x = tag(explode(pos))
    OUT.mkdir(parents=True, exist_ok=True)
    x.to_csv(OUT / "lookthrough_detail.csv", index=False)

    total = x.dollars.sum()
    liquid = x[x.sleeve != "private_real_estate"].dollars.sum()
    eq = x[x.region != ""]
    stocks = eq.dollars.sum()
    pct = lambda d, b: f"${d:>11,.0f}  {d / b:6.1%}"

    print(f"Net worth tracked: ${total:,.0f}   liquid: ${liquid:,.0f}   stocks: ${stocks:,.0f}\n")
    print("By sleeve (% of net worth):")
    print(table(x, "sleeve", total).to_string(), "\n")
    print("Stocks by region (% of stocks):")
    print(table(eq, "region", stocks).to_string(), "\n")
    print("US stocks by size (% of US stocks):")
    us = eq[eq.region == "US"]
    print(table(us, "size", us.dollars.sum()).to_string(), "\n")
    print("Stocks by sector (% of stocks):")
    print(table(eq, "Sector", stocks).head(12).to_string(), "\n")

    top = eq.groupby("issuer").agg(dollars=("dollars", "sum"), name=("Name", "first")).sort_values("dollars", ascending=False)
    print("Top 15 companies (% of stocks / % of liquid):")
    for iss, r in top.head(15).iterrows():
        print(f"  {r['name'][:34]:34} ${r.dollars:>9,.0f}  {r.dollars / stocks:5.1%}  {r.dollars / liquid:5.1%}")
    t10 = top.head(10).dollars.sum()
    print(f"  Top 10 = {t10 / stocks:.1%} of stocks, {t10 / liquid:.1%} of liquid\n")

    print("AI basket:")
    tiers = ["core", "adjacent_semis", "adjacent_infra", "narrative", "big_tech_other"]
    cum = 0
    rows = []
    for t in tiers:
        d = eq[eq.ai_tier == t].dollars.sum()
        cum += d
        rows.append({"tier": t, "dollars": round(d), "pct_stocks": d / stocks, "cum_pct_stocks": cum / stocks,
                     "pct_liquid": d / liquid, "cum_pct_liquid": cum / liquid})
        print(f"  {t:15} {pct(d, stocks)} of stocks | cumulative {cum / stocks:6.1%} of stocks, {cum / liquid:6.1%} of liquid, {cum / total:6.1%} of net worth")
    pd.DataFrame(rows).to_csv(OUT / "ai_exposure.csv", index=False)

    print("\nCore AI by source:")
    print(table(eq[eq.ai_tier == "core"], "source", eq[eq.ai_tier == "core"].dollars.sum()).to_string())

    matched = eq[eq.ai_tier != ""].groupby("issuer").Name.first()
    missing = set(pd.read_csv(ai_basket_path(), dtype=str).fillna("").ticker) - set(matched.index) - {"", "GOOG"}
    print("\nBasket names matched:", len(matched), "| tickers not found in any fund:", sorted(missing) or "none")

    summary = pd.concat({
        "sleeve_pct_networth": table(x, "sleeve", total).pct,
        "region_pct_stocks": table(eq, "region", stocks).pct,
        "us_size_pct_us": table(us, "size", us.dollars.sum()).pct,
        "sector_pct_stocks": table(eq, "Sector", stocks).pct,
    })
    summary.to_csv(OUT / "exposure_summary.csv", header=["pct"])


if __name__ == "__main__":
    main()
