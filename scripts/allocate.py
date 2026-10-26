"""Turn whole-portfolio targets into per-account fund dollars, then into model blocks.

Everything personal comes from the active profile (user_profile.py):
  profile.toml   candidate targets (shares of the investable pool), satellites, emergency fund
  accounts.toml  each account's mode, the funds it can use per asset class, location order

The investable pool is liquid money minus the emergency fund, the satellites, and frozen cash
that isn't new money (e.g. an IRA stub). Accounts by mode:

  frozen    holdings kept (e.g. taxable with unknown cost basis); with cash_is_new_money,
            its cash is spent as new money
  bank      cash; the emergency fund comes from here, the rest is new money
  movable   fully re-allocatable (no tax on trades)

Asset location fills each class through [location.order] in [location] fill_order. If frozen
holdings already exceed a target, that's reported as a gap, not fixed.

Run: uv run --with pandas python scripts/allocate.py [--profile NAME]
"""
import sys
from pathlib import Path

import pandas as pd

import blocks as B
import lookthrough as L
import stress as S
import user_profile as P

ROOT = Path(__file__).resolve().parent.parent

CLASSES = P.CLASSES


def fund_block(fund, profile=None):
    """Model block of a fund named in accounts.toml: catalog ticker/name, then plan menu."""
    profile = profile or P.current()
    cat = B.catalog(L.ROOT)
    if fund in cat:
        return cat[fund]
    menu = P.plan_menu_blocks(profile)
    if fund in menu:
        return menu[fund]
    raise P.ProfileError(f"{profile.root / 'accounts.toml'}: fund {fund!r} is not in data/fund_catalog.csv "
                         f"or the profile's plan_menu.csv")


def account_state(pos=None, profile=None):
    """Split today's positions into fixed blocks (frozen + satellites + stubs), the
    emergency fund, and the capacity of each allocatable account."""
    profile = profile or P.current()
    pos = L.load_positions(profile) if pos is None else pos
    acc = profile.accounts
    acct = pos.account.map(lambda a: acc[a].get("merge_into", a))
    mode = pos.account.map(lambda a: acc[a]["mode"])
    new_cash = (mode == "frozen") & (pos.block == "CASH") & pos.account.map(lambda a: acc[a].get("cash_is_new_money", False))
    bank = pos[mode == "bank"].value_usd.sum()
    ef = profile.emergency_fund
    fixed = pos[(mode == "frozen") & ~new_cash]
    capacity = {a: pos[(acct == a) & (mode == "movable")].value_usd.sum() for a in profile.allocatable_accounts()}
    capacity[profile.new_money_account] = pos[new_cash].value_usd.sum() + max(bank - ef, 0)
    return {
        "fixed_blocks": fixed.groupby("block").value_usd.sum(),
        "emergency_fund": min(bank, ef),
        "capacity": capacity,
    }


def allocate(target, state, profile=None):
    """Fill accounts to hit target class shares of the pool. Returns (fills, summary)."""
    profile = profile or P.current()
    funds = profile.account_funds()
    new_money = profile.new_money_account
    fixed = state["fixed_blocks"]
    frozen = fixed.drop(list(profile.satellites), errors="ignore")
    frozen_cls = frozen.groupby(frozen.index.map(B.asset_class)).sum()
    frozen_pool = frozen_cls.drop(["cash", "other"], errors="ignore")  # cash stubs stay out of the pool
    cap = dict(state["capacity"])
    pool = frozen_pool.sum() + sum(cap.values())

    reserve_usd = target.get("reserve_usd", 0)
    if reserve_usd == "new_taxable":
        reserve_usd = cap[new_money]
    shares = {c: target.get(c, 0) for c in CLASSES if c != "reserve"}
    assert abs(sum(shares.values()) - 1) < 1e-9, f"target shares must sum to 1: {shares}"
    want = {c: s * (pool - reserve_usd) for c, s in shares.items()}
    want["reserve"] = reserve_usd
    need = {c: want[c] - frozen_pool.get(c, 0) for c in want}

    fills = []

    def put(account, cls, usd):
        if cls not in funds.get(account, {}):
            raise P.ProfileError(f"{profile.root / 'accounts.toml'}: [funds.{account}] has no {cls!r} fund")
        for fund, share in funds[account][cls]:
            fills.append({"account": account, "class": cls, "fund": fund, "block": fund_block(fund, profile),
                          "usd": usd * share})
        cap[account] -= usd

    for cls in profile.fill_order:
        for account in profile.location_order.get(cls, []):
            if need[cls] <= 1e-6:
                break
            if cap[account] <= 1e-6:
                continue
            usd = min(need[cls], cap[account])
            put(account, cls, usd)
            need[cls] -= usd
    for account, left in list(cap.items()):  # any leftover capacity goes to US stocks
        if left > 1e-6:
            put(account, "us", left)

    fills = pd.DataFrame(fills).groupby(["account", "class", "fund", "block"], as_index=False).usd.sum()
    got = fills.groupby("class").usd.sum().reindex(CLASSES).fillna(0) + frozen_pool.reindex(CLASSES).fillna(0)
    summary = pd.DataFrame({"target_usd": pd.Series(want).reindex(CLASSES).fillna(0), "actual_usd": got})
    summary["gap_usd"] = summary.actual_usd - summary.target_usd
    return fills, summary


def to_blocks(fills, state):
    """Whole-portfolio model blocks: fixed holdings + emergency fund + allocated funds."""
    b = state["fixed_blocks"].copy()
    b["CASH"] = b.get("CASH", 0) + state["emergency_fund"]
    return b.add(fills.groupby("block").usd.sum(), fill_value=0)


def portfolios(profile=None):
    """Today's portfolio plus every candidate, as model blocks (used by stress/reverse_stress)."""
    profile = profile or P.current()
    state = account_state(profile=profile)
    out = {"0 Today": S.today_blocks()}
    for name, target in profile.candidates.items():
        fills, _ = allocate(target, state)
        out[name] = to_blocks(fills, state)
    return out


def main(argv=None):
    P.from_argv(sys.argv[1:] if argv is None else argv)
    profile = P.current()
    OUT = profile.output_dir
    print(profile.summary(), "\n")
    state = account_state()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, target in profile.candidates.items():
        fills, summary = allocate(target, state)
        rows.append(fills.assign(candidate=name))
        print(f"== {name} ==")
        print(fills.assign(usd=fills.usd.round(0)).to_string(index=False))
        off = summary[summary.gap_usd.abs() > 1]
        if len(off):
            print("  Off target (frozen holdings can't be sold):")
            print(off.round(0).to_string())
        print()
    pd.concat(rows).to_csv(OUT / "allocations.csv", index=False)


if __name__ == "__main__":
    main()
