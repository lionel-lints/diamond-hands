"""Which scenario hurts each blend most? (A view across existing models, not a new model.)

For a set of blends (the profile's candidates plus a few systematic variations of the IPS
target), compute the liquid-portfolio outcome in every named scenario (stress.py,
data/scenarios.csv) and every historical replay (real terms, worst point), then list each
blend's worst and best outcomes. The point is to pick the blend whose bad outcomes live in
scenarios I find least likely, and whose good outcomes live in ones I find most likely.

Optional profile input `[scenario_beliefs]` (scenario = weight, e.g. 0 to 3) adds two rows:
the belief-weighted average outcome, and the worst outcome among scenarios weighted >= 2
("plausible to me").

Caveat: named scenarios are cumulative over different horizons (1 to 23 years) and mostly
nominal (lost decade and Japan are effectively real); replays are real (CPI-deflated).
Compare blends within a row, not rows with each other.

Run: uv run --with pandas --with numpy python scripts/scenario_matrix.py [--profile NAME]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import allocate as A
import stress as S
import user_profile as P

ROOT = Path(__file__).resolve().parent.parent
TOP = 3
# Replay rows show the real worst point, except where the episode's question is how it ENDED:
# a gain episode (its worst point is just the start) and a decade-long one (its trough
# duplicates the 1973-74 row).
REPLAY_FULL_WINDOW = {"AI run 2023-now", "Stagflation 1973-81"}


def variations(profile):
    """The IPS target's structure at other stock shares (intl/SCV variations are profile candidates)."""
    base = profile.candidates[profile.ips_target]
    s0 = 1 - sum(base.get(k, 0) for k in ("bonds", "tips", "tips_short"))
    intl, scv = base.get("intl", 0) / s0, base.get("scv", 0) / s0
    tips_share = base.get("tips", 0) / (1 - s0) if s0 < 1 else 0.5

    def mix(stocks, intl_of_stocks, scv_of_stocks):
        fixed = 1 - stocks
        return {"us": stocks * (1 - intl_of_stocks - scv_of_stocks), "intl": stocks * intl_of_stocks,
                "scv": stocks * scv_of_stocks, "bonds": fixed * (1 - tips_share), "tips": fixed * tips_share,
                "reserve_usd": base.get("reserve_usd", 0)}

    out = {}
    for stocks in (0.70, 0.80, 1.00):
        out[f"target shape @ {stocks:.0%} stocks"] = mix(stocks, intl, scv)
    return out


def outcomes(profile):
    """(outcome table: rows = scenario/episode labels, cols = blends; horizon per row)."""
    state = A.account_state(profile=profile)
    targets = {**profile.candidates, **variations(profile)}
    ports = {"0 Today": S.today_blocks()}
    for name, t in targets.items():
        fills, _ = A.allocate(t, state, profile)
        ports[name] = A.to_blocks(fills, state)
    prof = S.block_profiles()
    sc = pd.read_csv(ROOT / "data/scenarios.csv").set_index("scenario")
    hyp = S.hypothetical(ports, prof).pivot(index="scenario", columns="portfolio", values="pct_liquid")
    hyp.index = [f"scenario: {s} ({sc.at[s, 'horizon_years']:g} yr)" for s in hyp.index]
    r = S.replay(ports)
    r["value"] = np.where(r.episode.isin(REPLAY_FULL_WINDOW), r.end_pct_real, r.trough_pct_real)
    rep = r.pivot(index="episode", columns="portfolio", values="value")
    rep.index = [f"history ({'real, full period' if e in REPLAY_FULL_WINDOW else 'real worst'}): {e}"
                 for e in rep.index]
    table = pd.concat([hyp, rep])[list(ports)]
    return table, ports


def worst_best(table):
    rows = []
    for blend in table.columns:
        col = table[blend].dropna().sort_values()
        rows.append({"blend": blend,
                     "worst": [(i, col[i]) for i in col.index[:TOP]],
                     "best": [(i, col[i]) for i in col.index[::-1][:TOP]]})
    return rows


def belief_rows(table, beliefs):
    """Belief-weighted mean and worst-among-plausible (weight >= 2), per blend."""
    key = {lbl: lbl.split(": ", 1)[1].split(" (")[0] for lbl in table.index}
    w = pd.Series({lbl: float(beliefs.get(key[lbl], 0)) for lbl in table.index})
    if w.sum() == 0:
        return None
    weighted = (table.mul(w, axis=0).sum() / w.sum())
    plausible = table[w >= 2]
    return pd.DataFrame({"belief-weighted average": weighted,
                         "worst among plausible (weight >= 2)": plausible.min() if len(plausible) else np.nan,
                         "which": plausible.idxmin() if len(plausible) else ""})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profile")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    P.use(args.profile) if args.profile else P.current()
    profile = P.current()
    print(profile.summary(), "\n")
    table, _ = outcomes(profile)
    out = profile.output_dir
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "scenario_matrix.csv")
    print("Liquid-portfolio outcome by scenario (rows) and blend (columns); each blend's worst is marked *:\n")
    shown = table.map(lambda v: f"{v:+.0%}")
    for b in table.columns:
        i = table[b].idxmin()
        shown.at[i, b] = shown.at[i, b] + "*"
    pd.set_option("display.width", 300)
    print(shown.to_string(), "\n")
    print("Each blend's Achilles heel (worst) and best case:")
    for r in worst_best(table):
        w = "; ".join(f"{lbl.split(': ', 1)[1]} {v:+.0%}" for lbl, v in r["worst"])
        b = "; ".join(f"{lbl.split(': ', 1)[1]} {v:+.0%}" for lbl, v in r["best"])
        print(f"  {r['blend']}\n     worst: {w}\n     best:  {b}")
    beliefs = profile.settings.get("scenario_beliefs", {})
    br = belief_rows(table, beliefs) if beliefs else None
    if br is not None:
        print("\nUsing the profile's [scenario_beliefs]:")
        print(br.to_string())
    else:
        print("\n(No [scenario_beliefs] in profile.toml: add `scenario = weight` lines to weight scenarios "
              "by how likely they seem to me.)")


if __name__ == "__main__":
    main()
