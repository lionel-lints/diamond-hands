"""Reverse stress test: search random "worlds" for the ones that break each portfolio.

scenarios.csv asks "given this named future, what's the loss?" This asks the reverse:
"given a definition of failure, what do the futures that cause it look like?" You don't
have to name the future -- only say what failure means. The point is to find each
allocation's breaking points, not to predict anything.

Each world is a 12-month move. Sleeve shock = sum(loading x factor) + noise + carry, clipped
to BOUNDS. Four factors (equity, ai, rates, credit) are drawn from uniform ranges
(FACTOR_RANGE, deliberately wider than history). By default they're correlated as in
1997-2026 via a Gaussian copula (--factors historical), and sleeves with usable history use
fitted loadings (--loadings fitted, from calibrate.py). --factors independent /
--loadings assumed are the original hand-set design, kept for sensitivity runs. Uniform
marginals mean extreme worlds are NOT rare here, even with historical correlation: breach
shares rank allocations, they are not probabilities.

Under fitted loadings the bond fund (VBIIX, the 401(k)'s option) FALLS in credit stress
(credit +0.40); under assumed loadings it rises (-0.2). That disagreement is assumption R11.
Cash and short TIPS earn the T-bill rate and bonds/TIPS the 10-yr yield as carry. Nothing
is inflation-adjusted yet.

Outputs, per portfolio x failure condition: how often it breaks, the *mildest* world
that breaks it, and what the breaking worlds have in common (mean AI-basket shock, how
often bonds failed to hedge). "Mildest" is measured on a portfolio-independent yardstick
-- the return of a plain global stock index (REFERENCE) in that world -- so candidates can
be compared: "breaks only if global stocks fall worse than X%". Each mildest world is
checked for plausibility: factor draws outside their calibrated historical range
(data/factor_hist_range.csv) and sleeves pinned at a clip bound are flagged, so an
extrapolated world isn't mistaken for a realistic one.

Run: uv run --with pandas --with numpy python scripts/reverse_stress.py [--profile NAME]
     [--factors historical|independent] [--loadings fitted|assumed] [--no-carry]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import blocks as B
import stress as S
import user_profile as P

ROOT = Path(__file__).resolve().parent.parent

N_WORLDS = 100_000
SEED = 42



def failures(profile=None):
    """Failure conditions from the profile's [failures] thresholds. Each takes the DataFrame
    from portfolio_results (net_/liquid_/stocks_ start & after, bench_liquid_after) and
    returns a bool mask: two absolute-loss tests, one relative (tilt regret), one behavioral."""
    f = (profile or P.current()).failures
    liq, net, lag, nerve = f["liquid_drawdown"], f["networth_drawdown"], f["lag_vs_benchmark"], f["stocks_drawdown"]
    out = {
        f"liquid -{liq:.0%}": lambda r: r.liquid_after <= (1 - liq) * r.liquid_start,
        f"net worth -{net:.0%}": lambda r: r.net_after <= (1 - net) * r.net_start,
        f"lags market {lag * 100:.0f}pp": lambda r: (r.liquid_after - r.bench_liquid_after) <= -lag * r.liquid_start,
        f"stocks -{nerve:.0%}": lambda r: (r.stocks_start > 0) & (r.stocks_after <= (1 - nerve) * r.stocks_start),
    }
    if "real_networth_drawdown" in f:  # optional: net worth after inflation
        real = f["real_networth_drawdown"]
        out[f"net worth -{real:.0%} real"] = lambda r: r.net_after_real <= (1 - real) * r.net_start
    return out


# World generator: 4 regime factors per world (independent, or copula-correlated), then
# sleeve shock = sum(loading * factor) + idiosyncratic noise + carry, clipped to BOUNDS.
FACTOR_RANGE = {
    "equity": (-0.60, 0.50),   # global risk appetite: crashes to melt-ups
    "ai": (-0.70, 0.60),       # AI-specific repricing on top of equities
    "rates": (-0.20, 0.25),    # positive = inflation/tightening; negative = flight to quality
    "credit": (-0.25, 0.05),   # negative = credit/liquidity stress (AI buildout is debt-financed)
    "inflation": (-0.02, 0.09),  # realized CPI over the 12-month world = the 1997-2026 range. Unlike
                                 # the other factors this one is NOT widened: a uniform -2%..12% put the
                                 # median world at 5% inflation (vs. 2.4% historical median and expected)
                                 # and 43% of worlds above 6%. 1970s-style inflation lives in the named
                                 # stagflation scenario instead. Still skewed: median 3.5%.
}
# Inflation reaches stocks/bonds only through its correlation with the other factors (no
# loading); TIPS and short TIPS additionally earn the inflation SURPRISE: realized minus the
# market's expectation (10-yr breakeven, FRED T10YIE). All results are also reported in real
# terms (deflated by the world's inflation).
INFLATION_LINKED = ("tips", "tips_short")
LOADINGS = {
    #                equity    ai   rates  credit
    "ai_core":       {"equity": 0.8, "ai": 0.7, "rates": -0.3, "credit": 0.3},
    "ai_adjacent":   {"equity": 0.8, "ai": 0.6, "rates": -0.3, "credit": 0.4},
    "labs_direct":   {"equity": 0.5, "ai": 1.2, "rates": -0.3, "credit": 0.5},
    "us_large_other": {"equity": 0.9, "ai": 0.1, "rates": -0.2, "credit": 0.2},
    "us_smid":       {"equity": 1.0, "ai": 0.1, "rates": -0.2, "credit": 0.4},
    "us_small_value": {"equity": 0.9, "ai": 0.0, "rates": -0.1, "credit": 0.3},
    "intl_dev":      {"equity": 0.9, "ai": 0.1, "rates": -0.2, "credit": 0.2},
    "em":            {"equity": 1.0, "ai": 0.1, "rates": -0.2, "credit": 0.3},
    # Bonds: hedge in growth scares (negative equity loading), hurt by inflation/tightening.
    # Credit stress -> flight to quality: the total bond fund is mostly Treasuries/agencies,
    # so it rallies (VBMFX: never negative over 12 months when stocks fell >20%, 1997-2026;
    # +6% in the 2008 window). The "bonds fail with credit" view lives in the named
    # ai_unwind_credit scenario, not in every random world.
    "bonds":         {"equity": -0.15, "ai": 0.0, "rates": -0.6, "credit": -0.2},
    "tips":          {"equity": -0.05, "ai": 0.0, "rates": -0.25, "credit": 0.0},
    # Short TIPS (VTIP): little duration, so small rate sensitivity (2022: about -3% vs. -12%).
    "tips_short":    {"equity": -0.02, "ai": 0.0, "rates": -0.08, "credit": 0.0},
    "cash":          {"equity": 0.0, "ai": 0.0, "rates": 0.0, "credit": 0.0},
    "private_re":    {"equity": 0.3, "ai": 0.0, "rates": -0.2, "credit": 0.8},
    "crypto":        {"equity": 1.5, "ai": 0.8, "rates": -0.3, "credit": 0.3},
}
# Each world is a 12-MONTH move (matches calibrate.py's 12-month windows). Multi-year paths,
# e.g. the 3-year AI risk window, are the Monte Carlo's job.
HORIZON_YEARS = 1
# Carry: income known in advance, added to the shock for one horizon. Bonds/TIPS earn about
# the current 10-yr yield, cash/short TIPS about the T-bill rate (latest FRED values, see
# current_carry). Stocks get no drift: their return isn't known ex ante.
CARRY_SOURCE = {"bonds": "gs10", "tips": "gs10", "tips_short": "tb3m", "cash": "tb3m"}
IDIO = 0.05                # sleeve-specific noise, +/- 5pp
BOUNDS = (-0.85, 1.00)     # clip: nothing loses more than 85% or gains more than 100%

# Yardstick for "how bad was this world": a plain global stock index (VT-like).
REFERENCE = {"TOTAL_US": 0.6, "INTL": 0.4}

# Defaults for the factor structure and loadings (see calibrate.py).
# Since 2026-10-22: factors drawn with their historical correlation (equity-credit 0.71,
# rates-credit 0.50 in 1997-2026) and history-fitted loadings for sleeves that have history.
# The original independent/assumed design stays available as a sensitivity check; the
# candidate ranking is the same under all four combinations.
DEFAULT_FACTORS = "historical"
DEFAULT_LOADINGS = "fitted"

# Benchmark for "lags market": same portfolio, but every stock position (fund blocks and
# direct stocks, e.g. a SCV tilt or META) swapped for REFERENCE. Bonds, TIPS, cash, real
# estate, crypto unchanged, so the test isolates *which stocks I hold*, not the stock/bond split.
STOCK_SLEEVES = ["ai_core", "ai_adjacent", "labs_direct", "us_large_other", "us_smid",
                 "us_small_value", "intl_dev", "em"]


def correlated_uniforms(rng, n, corr):
    """Gaussian copula: n draws of len(corr) uniforms on (0, 1) with the given correlation."""
    z = rng.standard_normal((n, len(corr))) @ np.linalg.cholesky(np.asarray(corr)).T
    return 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))


def current_carry():
    """Latest yields from data/prices/macro_monthly.csv, as one-horizon carry per sleeve."""
    m = pd.read_csv(ROOT / "data/prices/macro_monthly.csv", index_col=0)
    last = {c: m[c].dropna().iloc[-1] / 100 for c in set(CARRY_SOURCE.values())}
    return {sleeve: last[src] * HORIZON_YEARS for sleeve, src in CARRY_SOURCE.items()}


def current_expected_inflation():
    """Latest 10-yr breakeven inflation (decimal), from data/prices/macro_monthly.csv."""
    m = pd.read_csv(ROOT / "data/prices/macro_monthly.csv", index_col=0)
    return m["t10yie"].dropna().iloc[-1] / 100 * HORIZON_YEARS


def sample_worlds(n=N_WORLDS, seed=SEED, corr=None, loadings=None, carry=None, expected_inflation=None):
    """n random worlds: one column per sleeve (plus the f_* factor draws), one row per world.

    corr: None draws factors independently (the original design). A 4x4 matrix (FACTOR_RANGE
    order) draws them with that correlation via a Gaussian copula; each factor keeps its
    uniform range, so worlds still go beyond history, but stay historically coherent.
    loadings: overrides LOADINGS (e.g. fitted ones from calibrate.py).
    carry: {sleeve: return} added before clipping; None means no carry (pure shocks).
    expected_inflation: if given, inflation-linked sleeves earn (realized - expected)."""
    loadings = LOADINGS if loadings is None else loadings
    carry = carry or {}
    rng = np.random.default_rng(seed)
    if corr is None:
        factors = {k: rng.uniform(lo, hi, n) for k, (lo, hi) in FACTOR_RANGE.items()}
    else:
        u = correlated_uniforms(rng, n, corr)
        factors = {k: lo + u[:, i] * (hi - lo) for i, (k, (lo, hi)) in enumerate(FACTOR_RANGE.items())}
    worlds = {}
    for sleeve, sl_loadings in loadings.items():
        shock = sum(c * factors[f] for f, c in sl_loadings.items())
        if any(sl_loadings.values()):
            shock = shock + rng.uniform(-IDIO, IDIO, n)
        shock = shock + carry.get(sleeve, 0.0)
        if expected_inflation is not None and sleeve in INFLATION_LINKED and "inflation" in factors:
            shock = shock + (factors["inflation"] - expected_inflation)
        worlds[sleeve] = np.clip(np.broadcast_to(shock, n), *BOUNDS)
    worlds.update({f"f_{k}": v for k, v in factors.items()})
    return pd.DataFrame(worlds)


def reference_return(worlds, prof):
    """Per-world return of the REFERENCE global stock index (portfolio-independent)."""
    w = pd.concat([prof[k] * v for k, v in REFERENCE.items()]).groupby(level=0).sum()
    return pd.Series(worlds.reindex(columns=w.index).fillna(0).values @ w.values, index=worlds.index)


def stock_blocks(blocks):
    return [b for b in blocks.index if b in B.BLOCK_PROXY or B.is_direct_stock(b)]


def benchmark_blocks(blocks):
    """The same portfolio with every stock position swapped for REFERENCE (equal dollars)."""
    sb = stock_blocks(blocks)
    stocks = blocks.reindex(sb).fillna(0).sum()
    b = blocks.drop(sb, errors="ignore").copy()
    for k, v in REFERENCE.items():
        b[k] = b.get(k, 0) + stocks * v
    return b


def _pnl(worlds, blocks, prof):
    """Per-world dollar P&L: total, real-estate part, and stock part."""
    sl = S.to_sleeves(blocks, prof)
    sh = worlds.reindex(columns=sl.index).fillna(0)
    stock_sl = sl.reindex(STOCK_SLEEVES).fillna(0)
    return (sh.values @ sl.values,
            sh["private_re"].values * sl["private_re"] if "private_re" in sl.index else 0,
            worlds.reindex(columns=STOCK_SLEEVES).fillna(0).values @ stock_sl.values,
            stock_sl.sum())


def portfolio_results(worlds, blocks, prof):
    """Per-world outcome for one portfolio: net/liquid/stocks after shock, and the
    benchmark's liquid value after the same shock."""
    pnl, re_pnl, stock_pnl, stocks_start = _pnl(worlds, blocks, prof)
    b_pnl, b_re_pnl, _, _ = _pnl(worlds, benchmark_blocks(blocks), prof)
    net_start = blocks.sum()
    liquid_start = net_start - blocks.get("PRIVATE_RE", 0)
    infl = worlds["f_inflation"].values if "f_inflation" in worlds else 0.0
    net_after = net_start + pnl
    return pd.DataFrame({
        "net_start": net_start,
        "liquid_start": liquid_start,
        "net_after": net_after,
        "net_after_real": net_after / (1 + infl),
        "liquid_after": liquid_start + pnl - re_pnl,
        "stocks_start": stocks_start,
        "stocks_after": stocks_start + stock_pnl,
        "bench_liquid_after": liquid_start + b_pnl - b_re_pnl,
        "inflation": infl,
    }, index=worlds.index)


def relative_return_stats(worlds, blocks, prof):
    """How far the portfolio's liquid return drifts from its benchmark across all worlds."""
    r = portfolio_results(worlds, blocks, prof)
    rel = (r.liquid_after - r.bench_liquid_after) / r.liquid_start
    return {"worst": rel.min(), "p05": rel.quantile(0.05), "best": rel.max()}


def describe_world(shocks, sl, top=3, sleeves=None):
    """One-line summary of a world: the sleeves with the biggest dollar moves (optionally
    only among `sleeves`, e.g. stock sleeves for a stocks-only failure)."""
    shocks = shocks.fillna(0)
    sl = sl if sleeves is None else sl.reindex(sleeves).dropna()
    moved = (shocks.reindex(sl.index).fillna(0) * sl).abs().sort_values(ascending=False)
    return ", ".join(f"{s} {shocks.get(s, 0):+.0%}" for s in moved.index[:top])


def load_factor_hist_range():
    p = ROOT / "data/factor_hist_range.csv"
    if not p.exists():
        return None
    r = pd.read_csv(p, index_col=0)
    return {k: (r.at[k, "min"], r.at[k, "max"]) for k in r.index}


def plausibility_flags(world, sl, hist_range=None):
    """Why a world may be implausible: factor draws outside the calibrated historical range,
    and held sleeves pinned at a clip bound. Empty string = nothing flagged."""
    flags = []
    if hist_range:
        for k, (lo, hi) in hist_range.items():
            v = world.get(f"f_{k}")
            if v is not None and not (lo <= v <= hi):
                flags.append(f"{k} {v:+.2f} outside history [{lo:+.2f}, {hi:+.2f}]")
    held = [s for s in sl.index if sl[s] > 0 and s in world]
    for s in held:
        if world[s] <= BOUNDS[0] + 1e-9 or world[s] >= BOUNDS[1] - 1e-9:
            flags.append(f"{s} at clip {world[s]:+.0%}")
    return "; ".join(flags)


def breach_summary(portfolios, worlds, prof, failures=None):
    """One row per (portfolio, failure): breach rate, mildest breaking world, and what
    the breaking worlds have in common."""
    failures = globals()["failures"]() if failures is None else failures
    ref = reference_return(worlds, prof)
    hist_range = load_factor_hist_range()
    rows = []
    for pname, blocks in portfolios.items():
        sl = S.to_sleeves(blocks, prof)
        res = portfolio_results(worlds, blocks, prof)
        for fname, cond in failures.items():
            hit = cond(res)
            row = {"portfolio": pname, "failure": fname,
                   "n_breaches": int(hit.sum()), "pct_worlds": float(hit.mean()),
                   "mildest_ref_return": np.nan, "mildest_world": "", "mildest_factors": "",
                   "mildest_flags": "",
                   "mean_ai_core": np.nan, "mean_bonds": np.nan, "pct_bonds_negative": np.nan,
                   "pct_breaches_up_market": np.nan}
            if hit.any():
                mild_i = ref[hit].idxmax()  # breaching world where global stocks did best
                row["mildest_ref_return"] = ref[mild_i]
                stocks_only = fname.startswith("stocks ")
                row["mildest_world"] = describe_world(worlds.loc[mild_i], sl,
                                                      sleeves=STOCK_SLEEVES if stocks_only else None)
                row["mildest_flags"] = plausibility_flags(worlds.loc[mild_i], sl, hist_range)
                if all(f"f_{k}" in worlds for k in FACTOR_RANGE):
                    row["mildest_factors"] = ", ".join(f"{k} {worlds.at[mild_i, 'f_' + k]:+.2f}" for k in FACTOR_RANGE)
                row["pct_breaches_up_market"] = float((ref[hit] > 0).mean())
                bw = worlds.loc[res.index[hit]]
                row["mean_ai_core"] = bw.ai_core.mean() if "ai_core" in bw else np.nan
                row["mean_bonds"] = bw.bonds.mean() if "bonds" in bw else np.nan
                row["pct_bonds_negative"] = float((bw.bonds < 0).mean()) if "bonds" in bw else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def load_fitted_loadings():
    """LOADINGS with sleeves that have history replaced by calibrate.py's fitted values.
    AI sleeves, cash and crypto keep their assumed loadings (no usable history)."""
    fitted = pd.read_csv(ROOT / "data/loadings_fitted.csv", index_col=0)
    out = {k: dict(v) for k, v in LOADINGS.items()}
    for sleeve, row in fitted.iterrows():
        out[sleeve] = {f: float(row[f]) for f in FACTOR_RANGE if f in row.index}  # no inflation loading
    return out


def load_factor_corr():
    c = pd.read_csv(ROOT / "data/factor_corr.csv", index_col=0)
    return c.loc[list(FACTOR_RANGE), list(FACTOR_RANGE)].values


VARIANTS = [("independent", "assumed"), ("independent", "fitted"),
            ("historical", "assumed"), ("historical", "fitted")]


def build_worlds(factors=DEFAULT_FACTORS, loadings=DEFAULT_LOADINGS, no_carry=False, n=N_WORLDS, seed=SEED):
    """Worlds for one factor x loading choice (the CLI flags)."""
    return sample_worlds(n, seed,
                         corr=load_factor_corr() if factors == "historical" else None,
                         loadings=load_fitted_loadings() if loadings == "fitted" else None,
                         carry=None if no_carry else current_carry(),
                         expected_inflation=None if no_carry else current_expected_inflation())


def variants_table(portfolios, prof, failures=None, no_carry=False):
    """Breach share, cushion and plausibility flags per portfolio x failure under all four
    factor x loading combinations (README "sensitivity habit"; reverse_stress_variants.csv)."""
    rows = []
    for factors, loadings in VARIANTS:
        s = breach_summary(portfolios, build_worlds(factors, loadings, no_carry), prof, failures)
        rows.append(pd.DataFrame({"variant": f"{factors}/{loadings}", "failure": s.failure,
                                  "portfolio": s.portfolio, "breach": s.pct_worlds,
                                  "cushion": s.mildest_ref_return, "flags": s.mildest_flags}))
    return pd.concat(rows, ignore_index=True)


def run_header(args):
    """Describe the run exactly as configured (a test pins this to the flags)."""
    factors = ("copula-correlated (historical 1997-2026 correlation, uniform ranges)"
               if args.factors == "historical" else "independent (uniform ranges)")
    loadings = ("fitted from history where available (calibrate.py)" if args.loadings == "fitted"
                else "hand-set assumed (LOADINGS)")
    carry = ("off" if args.no_carry else "bonds/TIPS at 10-yr yield, cash/short TIPS at T-bill; "
             f"TIPS earn inflation minus {current_expected_inflation():.2%} expected")
    return (f"{N_WORLDS:,} random {HORIZON_YEARS}-year worlds (seed {SEED}) | factors: {factors} | "
            f"loadings: {loadings} | carry: {carry}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--factors", choices=["independent", "historical"], default=DEFAULT_FACTORS,
                   help="historical = correlation from 1997-2026 history; extremes are still NOT rare "
                        "(uniform ranges), so breach shares are not probabilities")
    p.add_argument("--loadings", choices=["assumed", "fitted"], default=DEFAULT_LOADINGS)
    p.add_argument("--no-carry", action="store_true", help="pure shocks: no bond/cash income")
    p.add_argument("--profile", help="profile name or folder (default: see user_profile.py)")
    return p.parse_args(argv)


def main(argv=None):
    import allocate

    args = parse_args(sys.argv[1:] if argv is None else argv)
    P.use(args.profile) if args.profile else P.current()
    OUT = P.current().output_dir
    print(P.current().summary())
    prof = S.block_profiles()
    portfolios = allocate.portfolios()
    worlds = build_worlds(args.factors, args.loadings, args.no_carry)
    summary = breach_summary(portfolios, worlds, prof)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "reverse_stress_summary.csv", index=False)
    variants = variants_table(portfolios, prof, no_carry=args.no_carry)
    variants.to_csv(OUT / "reverse_stress_variants.csv", index=False)

    print(run_header(args))
    print(f"Inflation across worlds: median {worlds.f_inflation.median():+.1%}, "
          f"above 6% in {(worlds.f_inflation > 0.06).mean():.0%} of worlds")
    print(f"Baseline across ALL worlds: ai_core avg {worlds.ai_core.mean():+.0%}, "
          f"bonds down in {(worlds.bonds < 0).mean():.0%} (compare to 'bonds down' among breaking worlds below)\n")
    for fname in failures():
        print(f"== Failure: {fname} ==")
        sub = summary[summary.failure == fname]
        if (sub.n_breaches == 0).all():
            print(f"  no portfolio breaks this in any world (one-period model; see drift table below)\n")
            continue
        for _, r in sub.iterrows():
            if r.n_breaches == 0:
                print(f"  {r['portfolio']:22} never breaks")
            else:
                print(f"  {r['portfolio']:22} breaks in {r.pct_worlds:5.1%} of worlds | "
                      f"breaking worlds: ai_core avg {r.mean_ai_core:+.0%}, "
                      f"bonds down in {r.pct_bonds_negative:.0%}, global stocks up in {r.pct_breaches_up_market:.0%} | "
                      f"mildest breach (global stocks {r.mildest_ref_return:+.0%}): {r.mildest_world} [{r.mildest_factors}]"
                      + (f"\n  {'':22} ⚠ implausible? {r.mildest_flags}" if r.mildest_flags else ""))
        print()
    print("Drift vs. benchmark (same portfolio, stocks swapped for the 60/40 global index), "
          "liquid return difference across all worlds:")
    for pname, blocks in portfolios.items():
        st = relative_return_stats(worlds, blocks, prof)
        print(f"  {pname:22} worst {st['worst']:+.1%} | 5th pct {st['p05']:+.1%} | best {st['best']:+.1%}")
    print(f"  (one-period model; a {P.current().failures['lag_vs_benchmark']:.0%} lag needs multi-year compounding"
          " -> regret model)")
    print(f"\nCushion spread across the four factor x loading variants (global-stock return of the mildest "
          f"breach; written to reverse_stress_variants.csv):")
    for fname, sub in variants.groupby("failure", sort=False):
        if sub.cushion.isna().all():
            continue
        print(f"== {fname} ==")
        for pname, d in sub.groupby("portfolio", sort=False):
            if d.cushion.notna().any():
                print(f"  {pname:30} cushion {d.cushion.min():+.0%} to {d.cushion.max():+.0%} "
                      f"(spread {d.cushion.max() - d.cushion.min():.0%}) | breach {d.breach.min():.1%} to {d.breach.max():.1%}")


if __name__ == "__main__":
    main()
