"""Regret / tracking error: how long and how badly would a tilt have made you
trail a plain index portfolio? This sets a tilt size I won't abandon.

For each portfolio (the profile's candidates, plus a sweep of SCV share), compare it with its
benchmark: the same portfolio with every stock position swapped for the 60/40 US/intl index
(reverse_stress.benchmark_blocks), so the stock/bond split is identical and only the tilts
differ. Both are held at constant weights (monthly rebalanced) through actual history,
1954-2026 (pre-1990 international = US, S16), using the replay series.

Two comparisons per portfolio:
  vs. benchmark  the whole stock tilt package (intl share, SCV, META, US overweight)
  vs. no-SCV twin  the SCV tilt alone: the same portfolio with its SCV moved into US stocks.
                   This is the number that sizes the SCV tilt, and it is checked against the
                   profile's lag threshold ([failures] lag_vs_benchmark).

Per comparison, over every rolling 5-year window: the worst cumulative lag and when it ended,
the share of windows trailing by more than 5pp and 10pp, and the longest stretch of
consecutive months in which the trailing-5-year return was behind the benchmark. Plus each
portfolio's lag in the named `ai_keeps_ripping` scenario (stress.py).

Run: uv run --with pandas --with numpy python scripts/regret.py [--profile NAME]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import allocate as A
import montecarlo as M
import reverse_stress as R
import stress as S
import user_profile as P

ROOT = Path(__file__).resolve().parent.parent
WINDOW = ("1954-01", "2026-09")
LAG_MONTHS = 60
SCV_SWEEP = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


def constant_mix_path(weights, hist):
    """Growth of $1 held at constant weights (monthly rebalanced)."""
    w = weights.reindex(hist.columns).fillna(0)
    return (1 + hist.values @ (w.values / w.sum())).cumprod()


def lag_stats(port, bench, index, lag_months=LAG_MONTHS):
    """Rolling-window cumulative return of port minus bench, and summary statistics."""
    rp = port[lag_months:] / port[:-lag_months]
    rb = bench[lag_months:] / bench[:-lag_months]
    diff = pd.Series(rp - rb, index=index[lag_months:])
    behind = (diff < 0).astype(int).values
    longest, run = 0, 0
    for b in behind:
        run = run + 1 if b else 0
        longest = max(longest, run)
    return diff, {
        "worst_5y_lag": diff.min(), "worst_5y_lag_ending": str(diff.idxmin()),
        "best_5y_lead": diff.max(),
        "pct_windows_behind_5pp": (diff <= -0.05).mean(), "pct_windows_behind_10pp": (diff <= -0.10).mean(),
        "longest_behind_months": longest,
        "annualized_vs_bench": (port[-1] ** (12 / len(port)) - bench[-1] ** (12 / len(bench))),
    }


def portfolios_to_test(profile):
    """The profile's candidates plus an SCV sweep around the IPS target's stock share."""
    out = dict(profile.candidates)
    base = profile.candidates.get(profile.ips_target) or next(iter(profile.candidates.values()))
    fixed = {k: base.get(k, 0) for k in ("bonds", "tips", "tips_short")}
    stocks = 1 - sum(fixed.values())
    intl = base.get("intl", 0) / stocks
    for v in SCV_SWEEP:
        out[f"sweep: {stocks:.0%} stk, SCV {v:.0%} of stocks"] = {
            "us": stocks * (1 - intl - v), "intl": stocks * intl, "scv": stocks * v, **fixed,
            "reserve_usd": base.get("reserve_usd", 0)}
    return out


def no_scv_twin(target):
    """Same target with the SCV share moved into US stocks."""
    t = dict(target)
    t["us"] = t.get("us", 0) + t.pop("scv", 0)
    return t


def run(profile, window=WINDOW):
    state = A.account_state(profile=profile)
    def build(target):
        fills, summary = A.allocate(target, state, profile)
        return A.to_blocks(fills, state).drop("PRIVATE_RE", errors="ignore"), summary.gap_usd.abs().max()

    built, gaps = {}, {}
    for name, target in portfolios_to_test(profile).items():
        blocks, gaps[name] = build(target)
        twin = build(no_scv_twin(target))[0] if target.get("scv", 0) > 0 else None
        built[name] = (blocks, R.benchmark_blocks(blocks), twin)
    all_blocks = sorted({b for p, bm, tw in built.values()
                         for b in set(p.index) | set(bm.index) | (set(tw.index) if tw is not None else set())})
    hist, _ = M.block_history(all_blocks, window)
    prof = S.block_profiles()
    sc = pd.read_csv(ROOT / "data/scenarios.csv").set_index("scenario")
    rip = sc.loc[["ai_keeps_ripping"]]
    rows = []
    lag_limit = profile.failures["lag_vs_benchmark"]
    for name, (blocks, bench, twin) in built.items():
        path = constant_mix_path(blocks, hist)
        _, st = lag_stats(path, constant_mix_path(bench, hist), hist.index)
        scv = {"scv_only_worst_5y": 0.0, "scv_only_worst_ending": "", "scv_only_longest_behind_months": 0,
               "scv_only_annualized": 0.0}
        if twin is not None:
            _, t = lag_stats(path, constant_mix_path(twin, hist), hist.index)
            scv = {"scv_only_worst_5y": t["worst_5y_lag"], "scv_only_worst_ending": t["worst_5y_lag_ending"],
                   "scv_only_longest_behind_months": t["longest_behind_months"],
                   "scv_only_annualized": t["annualized_vs_bench"]}
        scv["scv_within_lag_limit"] = scv["scv_only_worst_5y"] > -lag_limit
        shocks = rip.drop(columns=["description", "horizon_years"], errors="ignore").iloc[0] / 100
        sl, bl = S.to_sleeves(blocks, prof), S.to_sleeves(bench, prof)
        rip_gap = ((sl * shocks.reindex(sl.index).fillna(0)).sum() / sl.sum()
                   - (bl * shocks.reindex(bl.index).fillna(0)).sum() / bl.sum())
        # alloc_gap_usd > 0: frozen holdings keep the target out of reach (e.g. a large SCV share
        # needs less US stock than taxable already holds), so the tilt tested is smaller than named
        rows.append({"portfolio": name, **st, **scv, "ai_rips_vs_bench": rip_gap, "alloc_gap_usd": gaps[name]})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profile")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    P.use(args.profile) if args.profile else P.current()
    profile = P.current()
    print(profile.summary())
    print(f"Constant-mix replay {WINDOW[0]}..{WINDOW[1]} vs. the same portfolio with stocks swapped for "
          f"60/40 US/intl; rolling {LAG_MONTHS // 12}-year windows\n")
    df = run(profile)
    out = profile.output_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "regret.csv", index=False)
    fmt = {"worst_5y_lag": "{:+.1%}", "best_5y_lead": "{:+.1%}", "pct_windows_behind_5pp": "{:.0%}",
           "pct_windows_behind_10pp": "{:.0%}", "longest_behind_months": "{:.0f}", "annualized_vs_bench": "{:+.2%}",
           "ai_rips_vs_bench": "{:+.1%}"}
    fmt.update({"scv_only_worst_5y": "{:+.1%}", "scv_only_annualized": "{:+.2%}", "scv_only_longest_behind_months": "{:.0f}"})
    t = df.set_index("portfolio")
    print("Whole tilt package vs. benchmark:")
    print(t[["worst_5y_lag", "worst_5y_lag_ending", "best_5y_lead", "pct_windows_behind_5pp", "pct_windows_behind_10pp",
             "longest_behind_months", "annualized_vs_bench", "ai_rips_vs_bench"]]
          .apply(lambda c: c.map(fmt[c.name].format) if c.name in fmt else c).to_string())
    lim = profile.failures["lag_vs_benchmark"]
    print(f"\nSCV tilt alone vs. the same portfolio without it (profile lag limit: {lim:.0%}):")
    s = t[t.scv_only_worst_ending != ""]
    print(s[["scv_only_worst_5y", "scv_only_worst_ending", "scv_only_longest_behind_months", "scv_only_annualized",
             "scv_within_lag_limit"]].apply(lambda c: c.map(fmt[c.name].format) if c.name in fmt else c).to_string())
    print("\nworst_5y_lag: worst 5-yr cumulative return vs. benchmark (ending month shown); longest_behind_months: "
          "longest run of months with the trailing 5-yr return behind; ai_rips_vs_bench: the ai_keeps_ripping scenario")


if __name__ == "__main__":
    main()
