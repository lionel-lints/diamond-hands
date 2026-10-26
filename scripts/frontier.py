"""Risk/return frontier for the liquid portfolio: US stocks, international stocks, bonds, TIPS.

Two views, because history alone overfits:
  history  - actual monthly returns of Vanguard index mutual funds since TIPS fund inception (2000-06)
  forward  - Vanguard VCMM 10-yr return midpoints (2026) with historical volatility/correlation

Run: uv run --with pandas --with numpy python scripts/frontier.py
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ASSETS = {"us": "VTSMX", "intl": "VGTSX", "bonds": "VBMFX", "tips": "VIPSX"}

# Vanguard 2026 outlook, 10-yr annualized nominal, midpoint of published range.
# TIPS has no published figure in our source; assumed ~= nominal bonds.
FORWARD_GEO = {"us": 0.052, "intl": 0.059, "bonds": 0.043, "tips": 0.042}



def load_inputs(prices):
    """Monthly returns, annualized covariance, and forward arithmetic means for ASSETS."""
    rets = prices[list(ASSETS.values())].pct_change().dropna()
    rets.columns = list(ASSETS)
    cov = rets.cov().values * 12
    mu_fwd_arith = np.array([FORWARD_GEO[a] for a in ASSETS]) + np.diag(cov) / 2
    return rets, cov, mu_fwd_arith


def stats(w, rets, cov, mu_fwd_arith):
    r = rets.values @ w
    growth = np.cumprod(1 + r)
    years = len(r) / 12
    dd = growth / np.maximum.accumulate(np.r_[1.0, growth])[1:] - 1  # peak includes starting wealth
    roll12 = pd.Series(1 + r).rolling(12).apply(np.prod, raw=True) - 1
    vol = np.sqrt(w @ cov @ w)
    return {
        "hist_cagr": growth[-1] ** (1 / years) - 1,
        "vol": vol,
        "max_dd": dd.min(),
        "worst_12m": roll12.min(),
        "fwd_return": w @ mu_fwd_arith - vol**2 / 2,
    }


def grid(step=0.05):
    n = round(1 / step)
    for us, intl, bonds in itertools.product(range(n + 1), repeat=3):
        tips = n - us - intl - bonds
        if tips >= 0:
            yield np.array([us, intl, bonds, tips]) / n


def pareto(df, ret_col, risk_col):
    df = df.sort_values([risk_col, ret_col], ascending=[True, False])
    keep, best = [], -np.inf
    for i, row in df.iterrows():
        if row[ret_col] > best + 1e-9:
            keep.append(i)
            best = row[ret_col]
    return df.loc[keep]



def main():
    prices = pd.read_csv(ROOT / "data/prices/monthly_adjclose.csv", index_col=0, parse_dates=True)
    rets, cov, mu_fwd_arith = load_inputs(prices)

    rows = []
    for w in grid():
        s = stats(w, rets, cov, mu_fwd_arith)
        rows.append({**dict(zip(ASSETS, w)), "stocks": w[0] + w[1], **s})
    df = pd.DataFrame(rows)
    df["intl_share_of_stocks"] = np.where(df.stocks > 0, df.intl / df.stocks.replace(0, np.nan), np.nan)

    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "frontier_grid.csv", index=False)

    pct = lambda x: f"{x:6.1%}"
    cols = ["us", "intl", "bonds", "tips", "hist_cagr", "vol", "max_dd", "worst_12m", "fwd_return"]

    print(f"History window: {rets.index[0]:%Y-%m} to {rets.index[-1]:%Y-%m} ({len(rets)/12:.1f} yrs)\n")
    print("Asset stats:")
    for a in ASSETS:
        w = np.eye(4)[list(ASSETS).index(a)]
        s = stats(w, rets, cov, mu_fwd_arith)
        print(f"  {a:6} hist {pct(s['hist_cagr'])}  vol {pct(s['vol'])}  maxDD {pct(s['max_dd'])}  fwd {pct(s['fwd_return'])}")
    print("\nCorrelations:\n", rets.corr().round(2), "\n")

    for label, ret_col, risk_col in [
        ("HISTORY: return vs max drawdown", "hist_cagr", "max_dd"),
        ("FORWARD: return vs volatility", "fwd_return", "vol"),
    ]:
        d = df.copy()
        if risk_col == "max_dd":
            d["risk"] = -d.max_dd
            front = pareto(d, ret_col, "risk")
        else:
            front = pareto(d, ret_col, risk_col)
        front.to_csv(out / f"pareto_{ret_col}.csv", index=False)
        print(f"== Pareto front, {label} ({len(front)} pts) ==")
        print(front[cols].to_string(index=False, formatters={c: pct for c in cols}), "\n")

    print("== Candidates with intl = 30% of stocks, bonds/TIPS split 50/50 ==")
    cands = []
    for stocks in [1.0, 0.9, 0.8, 0.7, 0.6]:
        fi = 1 - stocks
        w = np.array([stocks * 0.7, stocks * 0.3, fi / 2, fi / 2])
        cands.append({"mix": f"{stocks:.0%}/{fi:.0%}", **dict(zip(ASSETS, w)), **stats(w, rets, cov, mu_fwd_arith)})
    cands = pd.DataFrame(cands)
    cands.to_csv(out / "candidates.csv", index=False)
    print(cands[["mix"] + cols].to_string(index=False, formatters={c: pct for c in cols}))


if __name__ == "__main__":
    main()
