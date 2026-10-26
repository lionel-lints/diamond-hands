"""One table scoring every candidate on every model.

Rows are metrics, columns are the profile's candidates (plus today's portfolio). Each metric
comes straight from the model that owns it, run fresh, so the table can't go stale:
  exposure        stocks % of liquid, core AI % of stocks, top-10 companies % of stocks
  scenarios       stress.py named scenarios, % of liquid (lost decade also after inflation)
  replay          stress.py real-terms worst point in historical episodes
  reverse stress  reverse_stress.py default model: breach share and cushion
  Monte Carlo     montecarlo.py: 3-yr 5th pct (worst of three runs), 10-yr median and
                  P(drawdown > 40%) under the 16-firm median forecast
  regret          regret.py: SCV tilt alone and the whole tilt package, worst 5-yr lag
  cost            blended expense ratio over liquid holdings, number of distinct funds

Run: uv run --with pandas --with numpy python scripts/compare.py [--profile NAME] [--paths N]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import allocate as A
import blocks as B
import lookthrough as L
import montecarlo as M
import regret as G
import reverse_stress as R
import stress as S
import user_profile as P

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ["ai_unwind_severe", "ai_unwind_credit_sticky_inflation", "broad_recession", "stagflation",
             "ai_keeps_ripping", "japan_us_to_trough_local", "japan_us_23y_contagion"]
EPISODES = ["1973-74 bear", "Dot-com 2000-02", "GFC 2007-09", "Inflation 2022"]
MC_RUNS = [("hist 1954-2026", None), ("CMA median", "median"), ("CMA GMO", "GMO")]
LOST_DECADE_CPI = 1.28  # CPI +28% over 2000-2009 (data/scenarios.csv lost_decade_2000s)


def issuer_profiles(direct_stocks):
    """Per block, issuer weights per $1 (proxy look-through), for top-10 concentration."""
    prof = {}
    for block, proxies in B.BLOCK_PROXY.items():
        parts = []
        for etf, share in proxies.items():
            h = L.load_fund(etf)
            h = h[~h.Location.isin(B.BLOCK_EXCLUDE_LOCATIONS.get(block, set()))]
            parts.append(h.assign(dollars=h.w / h.w.sum() * share, account="", source=etf, sleeve="equity"))
        x = L.tag(pd.concat(parts, ignore_index=True))
        prof[block] = x.groupby("issuer").dollars.sum()
    for t in direct_stocks:
        prof[t] = pd.Series({t: 1.0})
    return prof


def top10_share(blocks, iprof):
    parts = [iprof[b] * v for b, v in blocks.items() if b in iprof]
    if not parts:
        return np.nan
    s = pd.concat(parts).groupby(level=0).sum()
    return s.nlargest(10).sum() / s.sum()


def expense_ratios(profile):
    """{fund key or name: ER %} from the shared catalog and the profile's plan menu."""
    cat = pd.read_csv(ROOT / "data/fund_catalog.csv", dtype=str).fillna("")
    out = {r.key: float(r.expense_ratio_pct) for r in cat.itertuples() if r.expense_ratio_pct}
    if profile.plan_menu_path:
        menu = pd.read_csv(profile.plan_menu_path, dtype=str).fillna("")
        for r in menu.itertuples():
            if r.expense_ratio_pct:
                out[r.fund_name] = float(r.expense_ratio_pct)
                if r.ticker:
                    out.setdefault(r.ticker, float(r.expense_ratio_pct))
    return out


def holdings_by_fund(name, target, state, profile, pos):
    """Liquid dollars per fund: kept positions (frozen accounts, excluding re-spent cash and
    banks) plus the allocator's fills. Banks / private RE / direct stocks / crypto cost 0."""
    acc = profile.accounts
    mode = pos.account.map(lambda a: acc[a]["mode"])
    new_cash = (mode == "frozen") & (pos.block == "CASH") & pos.account.map(lambda a: acc[a].get("cash_is_new_money", False))
    if target is None:  # today: everything as held
        kept = pos
        fills = pd.DataFrame(columns=["fund", "usd"])
    else:
        kept = pos[(mode == "frozen") & ~new_cash]
        fills, _ = A.allocate(target, state, profile)
    kept = kept[kept.block != "PRIVATE_RE"]
    # Free: bank cash, uninvested cash without a fund ticker (e.g. a rollover in transit),
    # crypto, direct stocks.
    rows = [{"fund": k, "usd": v, "free": m == "bank" or b == "CRYPTO" or B.is_direct_stock(b)
             or (b == "CASH" and not isinstance(t, str))}
            for k, v, b, m, t in zip(kept.key, kept.value_usd, kept.block, mode[kept.index], kept.ticker)]
    rows += [{"fund": f, "usd": u, "free": False} for f, u in zip(fills.fund, fills.usd)]
    if target is not None:
        rows.append({"fund": "(bank: emergency fund)", "usd": state["emergency_fund"], "free": True})
    return pd.DataFrame(rows)


def lookup_er(name, ers):
    """ER by exact key, else by a plan-menu name the holding's name is a prefix of
    (exports often drop a trailing "Fund": "Acme 500 Index")."""
    if name in ers:
        return ers[name]
    hits = [v for k, v in ers.items() if isinstance(name, str) and len(name) > 8 and k.startswith(name)]
    return hits[0] if len(hits) == 1 else np.nan


def cost_metrics(funds, ers):
    """(dollar-weighted ER over funds with a known ER, share of dollars with a known ER, fund count)."""
    f = funds.copy()
    f["er"] = [0.0 if free else lookup_er(k, ers) for k, free in zip(f.fund, f.free)]
    known = f.er.notna()
    blended = (f.usd[known] * f.er[known]).sum() / f.usd[known].sum()
    coverage = f.usd[known].sum() / f.usd.sum() if f.usd.sum() else 1.0
    n_funds = f[~f.free].fund.nunique()
    return blended, coverage, n_funds


def build(profile, n_paths=1000):
    state = A.account_state(profile=profile)
    pos = L.load_positions(profile)
    ports = A.portfolios(profile)
    prof = S.block_profiles()
    directs = sorted({b for b in pos.block if B.is_direct_stock(b)})
    iprof = issuer_profiles(directs)
    ers = expense_ratios(profile)
    targets = {"0 Today": None, **profile.candidates}

    hyp = S.hypothetical(ports, prof).set_index(["portfolio", "scenario"]).pct_liquid
    rep = S.replay(ports).set_index(["portfolio", "episode"]).trough_pct_real
    worlds = R.sample_worlds(R.N_WORLDS, R.SEED, corr=R.load_factor_corr(), loadings=R.load_fitted_loadings(),
                             carry=R.current_carry(), expected_inflation=R.current_expected_inflation())
    fails = R.failures(profile)
    rs = R.breach_summary(ports, worlds, prof, fails).set_index(["portfolio", "failure"])
    mc = M.run(profile, n_paths, runs=MC_RUNS).set_index(["portfolio", "run"])
    rg = G.run(profile).set_index("portfolio")

    liq_fail = next(k for k in fails if k.startswith("liquid"))
    real_fail = next((k for k in fails if k.endswith("real")), None)
    rows = {}
    for name, blocks in ports.items():
        mcname = M.TODAY if name == "0 Today" else name
        sl = S.to_sleeves(blocks, prof)
        stocks = sl.reindex(R.STOCK_SLEEVES).fillna(0).sum()
        liquid = blocks.drop("PRIVATE_RE", errors="ignore").sum()
        blended, coverage, n_funds = cost_metrics(holdings_by_fund(name, targets[name], state, profile, pos), ers)
        r = {
            "stocks % of liquid": stocks / liquid,
            "core AI % of stocks": sl.get("ai_core", 0) / stocks,
            "top-10 companies % of stocks": top10_share(blocks.drop(["CASH", "PRIVATE_RE", "CRYPTO", "BOND", "TIPS",
                                                                     "TIPS_SHORT", "STABLE"], errors="ignore"), iprof),
        }
        for s in SCENARIOS:
            r[f"scenario: {s}"] = hyp[(name, s)]
        r["scenario: lost_decade_2000s (real)"] = (1 + hyp[(name, "lost_decade_2000s")]) / LOST_DECADE_CPI - 1
        for e in EPISODES:
            r[f"replay real worst: {e}"] = rep[(name, e)]
        r[f"reverse: {liq_fail} breach share"] = rs.loc[(name, liq_fail)].pct_worlds
        r[f"reverse: {liq_fail} cushion (global stocks)"] = rs.loc[(name, liq_fail)].mildest_ref_return
        if real_fail:
            r[f"reverse: {real_fail} breach share"] = rs.loc[(name, real_fail)].pct_worlds
        r["MC: 3-yr 5th pct wealth (worst of 3 runs)"] = min(mc.loc[(mcname, run)].real_x_3y_p05 for run, _ in MC_RUNS)
        r["MC: 10-yr median wealth (CMA median)"] = mc.loc[(mcname, "CMA median")].real_x_10y_p50
        r["MC: P(drawdown > 40%) (CMA median)"] = mc.loc[(mcname, "CMA median")].p_maxdd_40
        if name in rg.index:
            has_scv = rg.loc[name].scv_only_worst_ending != ""
            r["regret: SCV alone, worst 5-yr lag"] = rg.loc[name].scv_only_worst_5y if has_scv else np.nan
            r["regret: whole tilt, worst 5-yr lag"] = rg.loc[name].worst_5y_lag
        r["cost: blended expense ratio (%)"] = blended
        r["cost: ER coverage of liquid $"] = coverage
        r["cost: distinct funds"] = n_funds
        rows[name] = r
    return pd.DataFrame(rows)


def fmt_cell(metric, v):
    if pd.isna(v):
        return "—"
    if "wealth" in metric:
        return f"{v:.2f}x"
    if "expense ratio" in metric:
        return f"{v:.3f}%"
    if "distinct funds" in metric:
        return f"{v:.0f}"
    if "breach share" in metric or "P(" in metric or "% of" in metric or "coverage" in metric:
        return f"{v:.0%}" if abs(v) >= 0.1 or "breach" not in metric else f"{v:.1%}"
    return f"{v:+.0%}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profile")
    p.add_argument("--paths", type=int, default=1000)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    P.use(args.profile) if args.profile else P.current()
    profile = P.current()
    print(profile.summary(), "\n")
    t = build(profile, args.paths)
    out = profile.output_dir
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "comparison.csv")
    shown = t.copy().astype(object)
    for m in t.index:
        shown.loc[m] = [fmt_cell(m, v) for v in t.loc[m]]
    short = {c: c.replace(", short TIPS", " sTIPS").replace(" + reserve", "+res")
             for c in shown.columns}
    pd.set_option("display.width", 250)
    print(shown.rename(columns=short).to_string())
    print(f"\nIPS target: {profile.ips_target}. Signs: scenario/replay/regret are % changes (negative = loss or lag).")


if __name__ == "__main__":
    main()
