"""Stress tests: today's portfolio and candidate allocations under (1) hypothetical sleeve shocks
from data/scenarios.csv and (2) replays of real historical episodes.

Portfolios are expressed in "building blocks" (TOTAL_US, INTL, SCV, ...). Each block is
decomposed into sleeves (ai_core, us_small_value, ...) via the proxy-ETF look-through.
Run: uv run --with pandas python scripts/stress.py [--profile NAME]
"""
import sys
from pathlib import Path

import pandas as pd

import blocks as B
import lookthrough as L
import user_profile as P

ROOT = Path(__file__).resolve().parent.parent  # shared data

US_FLOAT_CAP = 72e12  # approx. float-adjusted US market cap (Advisor Perspectives / Mercer, 2026)

EPISODES = {
    # Pre-1990s windows run on Ken French / synthetic FRED series (see blocks.REPLAY_SERIES).
    "1973-74 bear": ("1972-12", "1974-09"),
    "Stagflation 1973-81": ("1972-12", "1981-12"),
    "Dot-com 2000-02": ("2000-03", "2002-09"),
    "GFC 2007-09": ("2007-10", "2009-02"),
    "COVID 2020": ("2020-01", "2020-03"),
    "Inflation 2022": ("2021-12", "2022-12"),
    "AI run 2023-now": ("2022-12", None),
}

# Candidates are profile data (profile.toml); allocate.py turns them into real funds per account.


def sleeve_of(r):
    if r.ai_tier == "core":
        return "ai_core"
    if r.ai_tier in ("adjacent_semis", "adjacent_infra", "narrative"):
        return "ai_adjacent"
    if r.sleeve == "small_value":
        return "us_small_value"
    if r.region == "US":
        return "us_large_other" if r["size"] == "large" else "us_smid"
    return {"Developed ex-US": "intl_dev", "EM": "em"}[r.region]


def direct_stock_sleeves(tickers):
    """Sleeve for each directly held stock, from the AI basket tier and US size bucket."""
    if not tickers:
        return {}
    x = L.tag(pd.DataFrame([{"Ticker": t, "Name": t, "Sector": "", "Location": "United States", "dollars": 1.0,
                             "account": "", "source": t, "sleeve": "equity"} for t in tickers]))
    return {r.Ticker: sleeve_of(r) for _, r in x.iterrows()}


def block_profiles(direct_stocks=None):
    """Sleeve mix per $1 of each block: fund blocks via proxy look-through, fixed blocks map
    to one sleeve, and each direct stock (default: those in the active profile's holdings)."""
    if direct_stocks is None:
        direct_stocks = sorted({b for b in L.load_positions().block if B.is_direct_stock(b)})
    prof = {}
    for block, proxies in B.BLOCK_PROXY.items():
        parts = []
        for etf, share in proxies.items():
            h = L.load_fund(etf)
            h = h[~h.Location.isin(B.BLOCK_EXCLUDE_LOCATIONS.get(block, set()))]
            h = h.assign(w=h.w / h.w.sum())
            parts.append(h.assign(dollars=h.w * share, account="", source=etf,
                                  sleeve="small_value" if etf == "ISCV" else "equity"))
        x = L.tag(pd.concat(parts, ignore_index=True))
        x["s"] = x.apply(sleeve_of, axis=1)
        prof[block] = x.groupby("s").dollars.sum()
    for block, s in B.BLOCK_FIXED_SLEEVE.items():
        prof[block] = pd.Series({s: 1.0})
    for ticker, s in direct_stock_sleeves(direct_stocks).items():
        prof[ticker] = pd.Series({s: 1.0})
    return prof


def today_blocks():
    pos = L.load_positions()
    return pos.groupby("block").value_usd.sum()


def to_sleeves(blocks, prof):
    s = pd.concat([prof[k] * v for k, v in blocks.items()]).groupby(level=0).sum()
    labs = pd.read_csv(ROOT / "data/private_ai.csv")
    lab_w = (labs.assumed_valuation_usd * labs.float_post_lockup / US_FLOAT_CAP)[labs.in_total_market_funds == "yes"].sum()
    # Post-IPO the labs displace other stocks in total-market funds, so carve their dollars
    # out of TOTAL_US's existing sleeves pro rata (dollars are conserved, not added on top).
    lab_usd = blocks.get("TOTAL_US", 0) * lab_w
    if lab_usd:
        s = s.sub(prof["TOTAL_US"] * lab_usd, fill_value=0)
    s["labs_direct"] = lab_usd
    return s


def hypothetical(portfolios, prof):
    sc = pd.read_csv(ROOT / "data/scenarios.csv").set_index("scenario")
    shocks = sc.drop(columns=["description", "horizon_years"], errors="ignore") / 100
    rows = []
    for name, blocks in portfolios.items():
        sl = to_sleeves(blocks, prof)
        liquid = blocks.drop("PRIVATE_RE", errors="ignore").sum()
        for scen, sh in shocks.iterrows():
            pnl = (sl * sh.reindex(sl.index).fillna(0)).sum()
            liq_pnl = pnl - sl.get("private_re", 0) * sh.private_re
            rows.append({"portfolio": name, "scenario": scen, "pnl": pnl, "pct_networth": pnl / blocks.sum(),
                         "pct_liquid": liq_pnl / liquid})
    return pd.DataFrame(rows)


def replay_returns():
    """Monthly returns for every replay series (fund tickers, FF_*, SYN_*) and the CPI index,
    on one YYYY-MM index. FF/macro files are optional (tests run without them)."""
    px = pd.read_csv(ROOT / "data/prices/replay_monthly.csv", index_col=0)
    rets = px.pct_change(fill_method=None)
    cpi = None
    ffp, mp = ROOT / "data/prices/ff_monthly.csv", ROOT / "data/prices/macro_monthly.csv"
    extra = {}
    if ffp.exists():
        ff = pd.read_csv(ffp, index_col=0)
        for c in ["us_mkt", "us_small_value", "us_big_growth", "dev_mkt"]:
            extra[f"FF_{c}"] = ff[c]
        extra["FF_small_blend"] = (ff.us_small_value + ff.us_small_growth) / 2
    if mp.exists():
        m = pd.read_csv(mp, index_col=0)
        y = m.gs10 / 100
        extra["SYN_TSY"] = y.shift(1) / 12 - 6.0 * y.diff()
        extra["SYN_TBILL"] = (m.tb3m / 1200).shift(1)
        cpi = m.cpi
    if extra:
        rets = rets.join(pd.DataFrame(extra), how="outer")
    return rets.sort_index(), cpi


def replay(portfolios):
    rets, cpi = replay_returns()
    rows = []
    for ep, (a, b) in EPISODES.items():
        r = rets.loc[a:b].iloc[1:]
        series, used = {}, {}
        for block in {b for blocks in portfolios.values() for b in blocks.index}:
            s = pd.Series(0.0, index=r.index)
            for t in reversed(B.replay_series(block)):  # later entries are fallbacks
                if t in r:
                    s = r[t].where(r[t].notna(), s)
            series[block] = s.fillna(0)
            # which series covered most of the window (reported, so fallbacks are visible)
            used[block] = next((t for t in B.replay_series(block) if t in r and r[t].notna().mean() > 0.5), "none (flat)")
        deflator = None
        if cpi is not None and a in cpi.index:
            c = cpi.reindex(r.index)
            deflator = (c / cpi[a]).ffill()
        for name, blocks in portfolios.items():
            w = blocks.drop("PRIVATE_RE", errors="ignore")  # liquid only; private marks don't trade
            path = sum(((1 + series[k]).cumprod() * v for k, v in w.items()))
            start = w.sum()
            row = {"portfolio": name, "episode": ep, "trough_pct": min(path.min(), start) / start - 1,
                   "end_pct": path.iloc[-1] / start - 1, "trough_usd": min(path.min(), start) - start,
                   "trough_pct_real": float("nan"), "end_pct_real": float("nan"),
                   "series_used": "; ".join(f"{k}={used[k]}" for k in sorted(w.index))}
            if deflator is not None and deflator.notna().all():
                real = path / deflator
                row["trough_pct_real"] = min(real.min(), start) / start - 1
                row["end_pct_real"] = real.iloc[-1] / start - 1
            rows.append(row)
    return pd.DataFrame(rows)


def stocks_on_fallback(blocks):
    """Direct stocks in `blocks` whose own series is missing or stale in replay_monthly.csv;
    those replay through DIRECT_STOCK_FALLBACK (assumption S4), not their own history."""
    px = pd.read_csv(ROOT / "data/prices/replay_monthly.csv", index_col=0)
    last = px.index[-1]
    out = []
    for b in blocks:
        if not B.is_direct_stock(b):
            continue
        if b not in px.columns or px[b].dropna().empty or px[b].dropna().index[-1] < last:
            out.append(b)
    return out


def main(argv=None):
    import allocate  # local import: allocate imports this module

    P.from_argv(sys.argv[1:] if argv is None else argv)
    OUT = P.current().output_dir
    EMERGENCY_FUND = P.current().emergency_fund
    print(P.current().summary(), "\n")
    prof = block_profiles()
    portfolios = allocate.portfolios()

    print("Sleeve mix (% of net worth):")
    mix = pd.DataFrame({k: to_sleeves(b, prof) / b.sum() for k, b in portfolios.items()}).fillna(0)
    print((mix * 100).round(1).to_string(), "\n")

    hyp = hypothetical(portfolios, prof)
    rep = replay(portfolios)
    OUT.mkdir(parents=True, exist_ok=True)
    mix.to_csv(OUT / "stress_sleeve_mix.csv")
    hyp.to_csv(OUT / "stress_hypothetical.csv", index=False)
    rep.to_csv(OUT / "stress_replay.csv", index=False)

    fmt_usd = lambda v: f"{v / 1000:+,.0f}K"
    sc = pd.read_csv(ROOT / "data/scenarios.csv").set_index("scenario")
    if "horizon_years" in sc:
        print("Scenario horizons (years; shocks are cumulative over the horizon, cash earns T-bills):",
              ", ".join(f"{k} {v:g}" for k, v in sc.horizon_years.items()), "\n")
    print("HYPOTHETICAL shocks: $ change in net worth (incl. private real estate)")
    print(hyp.pivot(index="scenario", columns="portfolio", values="pnl").map(fmt_usd).to_string(), "\n")
    print("HYPOTHETICAL shocks: % change in liquid portfolio (excl. private real estate)")
    print(hyp.pivot(index="scenario", columns="portfolio", values="pct_liquid").map("{:+.1%}".format).to_string(), "\n")
    print("REPLAY of real episodes: worst point, % of liquid portfolio (monthly data)")
    print(rep.pivot(index="episode", columns="portfolio", values="trough_pct").map("{:+.1%}".format).to_string(), "\n")
    print("REPLAY: return over full episode window")
    print(rep.pivot(index="episode", columns="portfolio", values="end_pct").map("{:+.1%}".format).to_string(), "\n")
    print("REPLAY after inflation (CPI-deflated): worst point / full window")
    print(rep.pivot(index="episode", columns="portfolio", values="trough_pct_real").map("{:+.1%}".format).to_string(), "\n")
    print(rep.pivot(index="episode", columns="portfolio", values="end_pct_real").map("{:+.1%}".format).to_string(), "\n")
    ips = P.current().ips_target or list(portfolios)[-1]
    print(f"Series used per block ({ips}):")
    for _, r in rep[rep.portfolio == ips].iterrows():
        print(f"  {r.episode:22} {r.series_used}")
    print()
    fallback = stocks_on_fallback({b for blocks in portfolios.values() for b in blocks.index})
    if fallback:
        print(f"⚠ {', '.join(sorted(fallback))}: no recent own-price history in "
              "data/prices/replay_monthly.csv, so the replay used the fallback series "
              "(VIGRX/FF_us_big_growth) instead. Run fetch_prices.py to add them.\n")
    print(f"Dip-buying dry powder above ${EMERGENCY_FUND:,} emergency fund (cash beyond EF):")
    for k, b in portfolios.items():
        print(f"  {k:22} ${b.get('CASH', 0) - EMERGENCY_FUND:>10,.0f}")


if __name__ == "__main__":
    main()
