"""Robust frontier: is there a mix near the hand-picked candidates that does
better under EVERY return assumption, not just one?

A grid of allocations (stock share x intl share of stocks x SCV share of stocks x bonds/TIPS
split) is turned into real funds per account by allocate.py (taxable frozen, asset location),
then simulated by montecarlo.py under each history window and forecast set, with the
profile's contributions and dip rule. All mixes share the same random paths (common random
numbers), so differences come from the mix, not from luck. Every mix carries the profile's
reserve policy (the IPS target's reserve_usd), so grid rows and the profile's candidates are
evaluated under the same cash/dip-buying treatment.

Two objectives, matching the 3-year risk window and longer growth:
  p3   5th-percentile real wealth at 3 years (higher = safer)
  m10  median real wealth at 10 years (higher = more growth)
A mix's shortfall in a run = how much some other mix beats it on BOTH objectives at once
(min of the two relative improvements; 0 if nothing dominates it). "Robust" = shortfall
<= TOLERANCE in every run. The key question for a candidate: does any mix dominate it in all
runs?

Run: uv run --with pandas --with numpy python scripts/robust_frontier.py [--profile NAME] [--paths N]
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import allocate as A
import montecarlo as M
import reverse_stress as R
import user_profile as P

ROOT = Path(__file__).resolve().parent.parent

STOCK_SHARES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
INTL_OF_STOCKS = [0.20, 0.30, 0.40]
SCV_OF_STOCKS = [0.0, 0.10, 0.15, 0.20, 0.25]
TIPS_OF_FIXED = [0.0, 0.5]
TOLERANCE = 0.01   # within 1% of a dominating mix on both objectives counts as "on the frontier"
HORIZON_MONTHS = 120
N_PATHS = 1000
RUNS = [("hist 1990-2026", None), ("hist 1954-2026", None), ("CMA median", "median"),
        ("CMA 25th pct", "p25"), ("CMA GMO", "GMO"), ("CMA BlackRock", "BlackRock")]


def grid(reserve_usd=0):
    """The allocation grid. reserve_usd: the cash reserve every mix holds (the profile's
    reserve policy by default, so candidates and grid mixes are comparable)."""
    for s, i, v, t in itertools.product(STOCK_SHARES, INTL_OF_STOCKS, SCV_OF_STOCKS, TIPS_OF_FIXED):
        fixed = 1 - s
        if fixed == 0 and t > 0:
            continue  # TIPS split is meaningless with no fixed income
        target = {"us": s * (1 - i - v), "intl": s * i, "scv": s * v,
                  "bonds": fixed * (1 - t), "tips": fixed * t, "reserve_usd": reserve_usd}
        name = f"{s:.0%} stk | {i:.0%} intl | {v:.0%} SCV | {t:.0%} TIPS"
        yield name, {"stocks": s, "intl_of_stocks": i, "scv_of_stocks": v, "tips_of_fixed": t}, target


def shortfall(m10, p3):
    """Per mix: the largest margin by which another mix beats it on both objectives."""
    m, p = np.asarray(m10)[:, None], np.asarray(p3)[:, None]
    gain = np.minimum(m.T / m - 1, p.T / p - 1)  # [i, j]: j vs i, the smaller of the two gains
    np.fill_diagonal(gain, -np.inf)
    return np.maximum(gain.max(1), 0.0)


def evaluate(profile, mixes, n_paths=N_PATHS, runs=RUNS, seed=M.SEED):
    """Rows: mix x run with p3, m10, P(DD>40% within 10 yrs)."""
    state = A.account_state(profile=profile)
    hold = set(profile.satellites) | {"CASH", "PRIVATE_RE"}
    built = {}
    for name, meta, target in mixes:
        fills, summary = A.allocate(target, state, profile)
        blocks = A.to_blocks(fills, state).drop("PRIVATE_RE", errors="ignore")
        reserve = fills[fills["class"] == "reserve"].usd.sum()
        built[name] = (meta, blocks, set(blocks.index) - hold, reserve, summary.gap_usd.abs().max())
    all_blocks = sorted({b for _, blocks, _, _, _ in built.values() for b in blocks.index} | set(M.REFERENCE_BLOCKS))
    tbill = R.current_carry()["cash"]
    rows = []
    for run_name, cma_set in runs:
        window = M.WINDOWS.get(run_name, M.WINDOWS["hist 1990-2026"])
        raw, infl = M.block_history(all_blocks, window)
        hist = M.recenter(raw, M.cma_targets(all_blocks, cma_set, tbill)) if cma_set else raw
        idx = M.bootstrap_indices(len(hist), HORIZON_MONTHS, n_paths, M.BLOCK_MONTHS, np.random.default_rng(seed))
        cut = M.cma_horizon_months(cma_set) if cma_set else HORIZON_MONTHS
        H = np.concatenate([hist.values[idx][:, :cut], raw.values[idx][:, cut:]], axis=1)
        I = infl.values[idx]
        for name, (meta, blocks, rebal, reserve, gap) in built.items():
            cols = [all_blocks.index(b) for b in blocks.index]
            sim = M.simulate(blocks, H[:, :, cols], I, rebal, reserve, profile.dip_buying if reserve else (),
                             ref_weights=dict(R.REFERENCE), streams=profile.contributions())
            start = blocks.sum()
            rows.append({"mix": name, **meta, "run": run_name, "alloc_gap_usd": gap,
                         "p3": np.quantile(sim["real"][:, 36] / start, 0.05),
                         "m10": np.median(sim["real"][:, 120] / start),
                         "p_dd40_10y": (M.max_drawdown(sim["twr"]) > 0.40).mean()})
    df = pd.DataFrame(rows)
    df["shortfall"] = df.groupby("run", group_keys=False).apply(
        lambda d: pd.Series(shortfall(d.m10.values, d.p3.values), index=d.index))
    return df


def dominators(df, mix):
    """Mixes that beat `mix` on both objectives in every run."""
    me = df[df.mix == mix].set_index("run")[["m10", "p3"]]
    out = []
    for other, d in df[df.mix != mix].groupby("mix"):
        d = d.set_index("run")[["m10", "p3"]].reindex(me.index)
        if ((d.m10 > me.m10) & (d.p3 > me.p3)).all():
            out.append(other)
    return out


def ips_like_name(target):
    """Grid name closest to a profile candidate (for locating it on the grid)."""
    s = 1 - target.get("bonds", 0) - target.get("tips", 0) - target.get("tips_short", 0)
    i = target.get("intl", 0) / s
    v = target.get("scv", 0) / s
    fixed = 1 - s
    t = (target.get("tips", 0) / fixed) if fixed else 0.0
    pick = lambda xs, x: min(xs, key=lambda y: abs(y - x))
    s, i, v, t = pick(STOCK_SHARES, s), pick(INTL_OF_STOCKS, i), pick(SCV_OF_STOCKS, v), pick(TIPS_OF_FIXED, t)
    if s == 1.0:
        t = 0.0
    return f"{s:.0%} stk | {i:.0%} intl | {v:.0%} SCV | {t:.0%} TIPS"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profile")
    p.add_argument("--paths", type=int, default=N_PATHS)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    P.use(args.profile) if args.profile else P.current()
    profile = P.current()
    print(profile.summary())
    base = profile.candidates.get(profile.ips_target) or next(iter(profile.candidates.values()), {})
    reserve_usd = base.get("reserve_usd", 0)
    mixes = list(grid(reserve_usd))
    print(f"{len(mixes)} mixes x {len(RUNS)} runs x {args.paths:,} paths (common random numbers), 10-yr horizon, "
          f"contributions ${profile.monthly_contribution:,.0f}/mo, reserve_usd={reserve_usd!r}\n")
    df = evaluate(profile, mixes, args.paths)
    out = profile.output_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "robust_frontier.csv", index=False)

    worst = df.groupby("mix").agg(max_shortfall=("shortfall", "max"), m10_min=("m10", "min"), p3_min=("p3", "min"),
                                  m10_med=("m10", "median"), p3_med=("p3", "median"), dd40_med=("p_dd40_10y", "median"),
                                  stocks=("stocks", "first"))
    robust = worst[worst.max_shortfall <= TOLERANCE].sort_values(["stocks", "m10_med"])
    print(f"Robust mixes (within {TOLERANCE:.0%} of the frontier in all {len(RUNS)} runs): {len(robust)} of {len(worst)}")
    fmt = {"max_shortfall": "{:.1%}", "m10_min": "{:.2f}x", "p3_min": "{:.2f}x", "m10_med": "{:.2f}x",
           "p3_med": "{:.2f}x", "dd40_med": "{:.0%}"}
    show = robust.drop(columns="stocks")
    print(show.apply(lambda c: c.map(fmt[c.name].format)).to_string() if len(show) else "  (none)", "\n")

    print("Where the profile's candidates sit (nearest grid mix):")
    for name, target in profile.candidates.items():
        g = ips_like_name(target)
        w = worst.loc[g]
        dom = dominators(df, g)
        print(f"  {name:26} ~ {g:38} worst shortfall {w.max_shortfall:5.1%} | "
              f"robust: {'yes' if w.max_shortfall <= TOLERANCE else 'no'} | beaten in ALL runs by {len(dom)} mix(es)"
              + (f": {', '.join(dom[:3])}{'...' if len(dom) > 3 else ''}" if dom else ""))


if __name__ == "__main__":
    main()
