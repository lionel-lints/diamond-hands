"""Calibrate reverse_stress.LOADINGS against history.

For each sleeve with a usable history, regress its 12-month returns on historical proxies
of the model's four factors and compare the fitted loadings with the assumed ones.

Factor proxies (12-month changes, overlapping monthly windows), scaled to the model's units:
  equity  global stocks: 60% US total market + 40% international (VTSMX / VGTSX)
  ai      Nasdaq-100 minus S&P 500 (^NDX price index - VFINX); a tech/growth proxy, not "AI"
  rates   +10 x change in the 10-yr Treasury yield (+1pp yield -> +0.10), so an intermediate
          bond fund with ~6-yr duration should load about -0.6
  credit  -8 x change in the Baa-Treasury spread (+1pp spread -> -0.08), negative = stress

Long-history check (1954-): the same regression on Ken French US series, with the ai proxy
replaced by large growth minus market.

Caveats: overlapping windows overstate significance, so read point estimates and R^2, not
t-stats. The AI sleeves (ai_core, ai_adjacent, labs_direct) have no pre-AI history and stay
judgment, and so does crypto. Private real estate is regressed on public REITs (VGSIX) for
reference only and NOT written to loadings_fitted.csv: REITs are levered, marked daily
commercial real estate, while the profile's private_re is unlevered-at-cost housing deals
(REFERENCE_ONLY below). Using the REIT fit (credit 1.37) pinned private_re at the -85% clip in
moderately bad worlds and collapsed net-worth cushions by ~20pp; that was an artifact.

The 12-month CPI inflation rate is a fifth factor for correlations and ranges only (it
isn't a regressor; sleeves feel it through its correlation with rates, and TIPS through
inflation accrual in reverse_stress.py).

Also writes data/factor_hist_range.csv: the min/max of each factor proxy over the fit window,
so reverse_stress.py can flag worlds that extrapolate beyond calibrated history.

Run: uv run --with pandas --with numpy python scripts/calibrate.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import reverse_stress as R

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
FACTORS = ["equity", "ai", "rates", "credit"]
RATES_SCALE = 10.0
CREDIT_SCALE = -8.0
FLAG_DIFF = 0.30  # flag a loading if |fitted - assumed| exceeds this, or the sign flips
REFERENCE_ONLY = {"private_re"}  # fitted and printed, but not used by reverse_stress.py

SLEEVE_PROXY = {  # sleeve -> price series in replay_monthly.csv
    "us_large_other": "VFINX",  # S&P 500 (includes AI names; nearest long proxy)
    "us_smid": "NAESX",
    "us_small_value": "VISVX",
    "intl_dev": "VTMGX",
    "em": "VEIEX",
    "bonds": "VBIIX",
    "tips": "VIPSX",
    "tips_short": "VTIP",
    "private_re": "VGSIX",  # public REITs: far more volatile than private marks
}


def twelve_month(prices):
    return prices / prices.shift(12) - 1


def factor_frame(px, macro):
    r12 = twelve_month(px)
    f = pd.DataFrame({
        "equity": 0.6 * r12["VTSMX"] + 0.4 * r12["VGTSX"],
        "ai": r12["^NDX"] - r12["VFINX"],
        "rates": RATES_SCALE * macro.gs10.diff(12) / 100,
        "credit": CREDIT_SCALE * macro.baa10y.diff(12) / 100,
        # Not a regressor (FACTORS); used for the factor correlation and historical range only.
        "inflation": macro.cpi.pct_change(12, fill_method=None),
    })
    return f, r12


def ols(y, X):
    """Least squares with intercept; returns (coefs by column, R^2, n)."""
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    A = np.column_stack([np.ones(len(d)), d[X.columns].values])
    beta, *_ = np.linalg.lstsq(A, d.y.values, rcond=None)
    resid = d.y.values - A @ beta
    r2 = 1 - resid.var() / d.y.values.var()
    return pd.Series(beta[1:], index=X.columns), r2, len(d)


def calibrate(px, macro, loadings=None):
    loadings = R.LOADINGS if loadings is None else loadings
    f, r12 = factor_frame(px, macro)
    rows = []
    for sleeve, series in SLEEVE_PROXY.items():
        if series not in r12:
            continue
        beta, r2, n = ols(r12[series], f[FACTORS])
        for fac in FACTORS:
            assumed = loadings[sleeve][fac]
            fitted = beta[fac]
            flag = (abs(fitted - assumed) > FLAG_DIFF) or (np.sign(fitted) != np.sign(assumed)
                                                         and max(abs(fitted), abs(assumed)) > 0.1)
            rows.append({"sleeve": sleeve, "proxy": series, "factor": fac, "assumed": assumed,
                         "fitted": fitted, "r2": r2, "n_months": n, "flag": bool(flag)})
    return pd.DataFrame(rows)


def calibrate_long(ff, macro):
    """1954- check on Ken French US series; ai proxy = large growth minus market."""
    g = (1 + ff).cumprod()
    r12 = g / g.shift(12) - 1
    f = pd.DataFrame({
        "equity": r12.us_mkt,
        "ai": r12.us_big_growth - r12.us_mkt,
        "rates": RATES_SCALE * macro.gs10.diff(12) / 100,
        "credit": CREDIT_SCALE * macro.baa10y.diff(12) / 100,
    })
    out = {}
    for col, sleeve in [("us_small_value", "us_small_value"), ("us_big_value", "us_large_other")]:
        beta, r2, n = ols(r12[col], f[FACTORS])
        out[sleeve] = {**beta.round(2).to_dict(), "r2": round(r2, 2), "n": n}
    return pd.DataFrame(out).T


def load():
    px = pd.read_csv(ROOT / "data/prices/replay_monthly.csv", index_col=0)
    px.index = pd.PeriodIndex(px.index, freq="M")
    macro = pd.read_csv(ROOT / "data/prices/macro_monthly.csv", index_col=0)
    macro.index = pd.PeriodIndex(macro.index, freq="M")
    ff = pd.read_csv(ROOT / "data/prices/ff_monthly.csv", index_col=0)
    ff.index = pd.PeriodIndex(ff.index, freq="M")
    return px, macro, ff


def main():
    px, macro, ff = load()
    cal = calibrate(px, macro)
    OUT.mkdir(exist_ok=True)
    cal.to_csv(OUT / "calibration.csv", index=False)
    # Inputs reverse_stress.py can use instead of its hand-set assumptions (see its --factors /
    # --loadings options): fitted loadings for sleeves with history, and factor correlations.
    fitted = cal.pivot(index="sleeve", columns="factor", values="fitted")[FACTORS].round(3)
    fitted.drop(index=list(REFERENCE_ONLY), errors="ignore").to_csv(ROOT / "data/loadings_fitted.csv")
    f, _ = factor_frame(px, macro)
    f = f[FACTORS + ["inflation"]].dropna()
    f.corr().round(3).to_csv(ROOT / "data/factor_corr.csv")
    pd.DataFrame({"min": f.min(), "max": f.max()}).round(3).to_csv(ROOT / "data/factor_hist_range.csv")
    wide = cal.pivot(index="sleeve", columns="factor", values=["assumed", "fitted"]).round(2)
    meta = cal.groupby("sleeve").agg(r2=("r2", "first"), n=("n_months", "first"), flags=("flag", "sum"))
    print("Fitted vs. assumed loadings (12-month windows, ~1997-2026; see docstring for proxies)\n")
    for sleeve in SLEEVE_PROXY:
        if sleeve not in meta.index:
            continue
        c = cal[cal.sleeve == sleeve].set_index("factor")
        cells = "  ".join(f"{fac}: {c.assumed[fac]:+.2f} → {c.fitted[fac]:+.2f}{'*' if c.flag[fac] else ' '}"
                          for fac in FACTORS)
        note = "  (reference only: not used)" if sleeve in REFERENCE_ONLY else ""
        print(f"  {sleeve:15} {cells}   R² {meta.r2[sleeve]:.2f}  n={meta.n[sleeve]}{note}")
    print("\n  * = sign flip or |fitted - assumed| > 0.30\n")
    print("Long-history check (Ken French, 1954-; ai proxy = large growth - market):")
    print(calibrate_long(ff, macro).to_string())


if __name__ == "__main__":
    main()
