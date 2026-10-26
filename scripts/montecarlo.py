"""Monte Carlo of the liquid portfolio along multi-year paths.

Paths are built by a block bootstrap of monthly historical returns: 12-month blocks of all
blocks' returns AND CPI, taken together from the same calendar months, so crashes stay
clustered and cross-asset behavior (e.g. stocks and bonds falling together in 2022) is
preserved. Two history windows: 1990-2026 (every series genuine) and 1954-2026 (adds the
1970s via Ken French / synthetic FRED stand-ins; international = US before 1990, S16).

Policy along each path, from the profile and IPS:
  - rebalance every REBALANCE_EVERY months, only if a block drifts beyond its band:
    +/-5pp, or 25% of target for targets under 20% (IPS sec. 5)
  - dip buying: deploy the reserve in tranches as a 60/40 US/intl index falls from its high
  - monthly contributions, invested at target
Satellites (META, crypto, ...) and the emergency fund are held, not rebalanced. Private real
estate is excluded: illiquid, unmarked, can't be rebalanced or sold into a crash.

Forecast re-centering (3.6): each block's history is rescaled so its geometric mean matches a
capital-market-assumption set (data/cma.csv) while its volatility and correlations stay
historical. Runs: history as-is, the 16-firm median, the pessimistic 25th percentile, and
the two bookends (GMO, BlackRock). A forecast applies only over its own horizon (10 years;
GMO 7); later months use the un-recentered 1990-2026 history. Extending a 7-10 year
mean-reversion call over 30 years would turn it into a permanent trend.

Run: uv run --with pandas --with numpy python scripts/montecarlo.py [--profile NAME] [--paths N]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import blocks as B
import cma
import reverse_stress as R
import stress as S
import user_profile as P

ROOT = Path(__file__).resolve().parent.parent

N_PATHS = 2000
SEED = 7
BLOCK_MONTHS = 12
HORIZONS_YEARS = [3, 10, 20, 30]
REBALANCE_EVERY = 6
LAG_WINDOW_MONTHS = 60    # "lags the benchmark by >= X over any 5-year window"...
LAG_SEARCH_YEARS = 10     # ...within the first 10 years (over 30 years, ~300 overlapping windows
                          # make almost any tilt trip it eventually)
WINDOWS = {"hist 1990-2026": ("1990-08", "2026-09"), "hist 1954-2026": ("1954-01", "2026-09")}
REFERENCE_BLOCKS = list(R.REFERENCE)  # always simulated, for the dip trigger and lag benchmark

# CMA asset class used to re-center each block; direct stocks use us_large.
CMA_CLASS = {"TOTAL_US": "us_large", "SP500": "us_large", "MID": "us_large",
             "SMALL_BLEND": "us_small", "SCV": "us_small_value", "INTL": "intl_dev",
             "INTL_DEV": "intl_dev", "EM": "em", "BOND": "us_bonds", "TIPS": "tips",
             "TIPS_SHORT": "cash", "CASH": "cash", "STABLE": "cash"}
# When a firm has no forecast for a class: small/SCV = that firm's us_large + 1pp (judgment;
# only GMO and Research Affiliates publish one), TIPS = bonds, cash = current T-bill.
SMALL_PREMIUM = 0.01


# --- data -----------------------------------------------------------------------------
def block_history(block_names, window):
    """Monthly returns (months x blocks) for `window` via the replay chains, plus monthly CPI
    inflation. Blocks with no data at all in a month are flat (crypto before 2017)."""
    rets, cpi = S.replay_returns()
    a, b = window
    r = rets.loc[a:b]
    out = {}
    for block in block_names:
        s = pd.Series(np.nan, index=r.index)
        for t in reversed(B.replay_series(block)):
            if t in r:
                s = r[t].where(r[t].notna(), s)
        out[block] = s.fillna(0.0)
    hist = pd.DataFrame(out)
    infl = cpi.pct_change(fill_method=None).reindex(hist.index)
    keep = infl.notna()
    return hist[keep], infl[keep]


def bootstrap_indices(n_hist, n_months, n_paths, block_len, rng):
    """(paths x months) indices into history: consecutive `block_len`-month runs from random
    starting months, wrapping around the end (circular block bootstrap)."""
    n_blocks = -(-n_months // block_len)
    starts = rng.integers(0, n_hist, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_len)[None, None, :]) % n_hist
    return idx.reshape(n_paths, -1)[:, :n_months]


def recenter(hist, target_geo_annual):
    """Rescale each column so its annualized geometric mean equals the target, keeping its
    monthly volatility pattern and its correlation with the other columns."""
    out = hist.copy()
    for col, g in target_geo_annual.items():
        if col not in out or g is None or np.isnan(g):
            continue
        hist_g = np.exp(np.log1p(out[col]).mean()) - 1
        tgt = (1 + g) ** (1 / 12) - 1
        out[col] = (1 + out[col]) * (1 + tgt) / (1 + hist_g) - 1
    return out


def cma_targets(block_names, which, current_tbill):
    """{block: annual nominal geometric target} for a CMA run: 'median', 'p25', or a firm."""
    w = cma.wide() / 100  # percent -> decimal first: SMALL_PREMIUM is a decimal
    if which in ("median", "p25"):
        q = 0.5 if which == "median" else 0.25
        row = w.quantile(q)
        row["us_small"] = row["us_large"] + SMALL_PREMIUM  # most firms don't publish these;
        row["us_small_value"] = row["us_large"] + SMALL_PREMIUM  # don't let 1-2 firms set them
    else:
        row = w.loc[which].copy()
    if np.isnan(row.get("us_small", np.nan)):
        row["us_small"] = row["us_large"] + SMALL_PREMIUM
    if np.isnan(row.get("us_small_value", np.nan)):
        row["us_small_value"] = row["us_small"]
    if np.isnan(row.get("tips", np.nan)):
        row["tips"] = row["us_bonds"]
    if np.isnan(row.get("cash", np.nan)):
        row["cash"] = current_tbill
    out = {}
    for b in block_names:
        cls = CMA_CLASS.get(b, "us_large" if B.is_direct_stock(b) else None)
        out[b] = row[cls] if cls else None
    return out


def cma_horizon_months(which):
    """Forecast horizon of a CMA run: the firm's stated horizon, or 10 years for median/p25."""
    if which in ("median", "p25"):
        return 120
    d = cma.load()
    return int(d[d.firm == which].horizon_years.max() * 12)


# --- simulation -----------------------------------------------------------------------
def bands(target_share):
    """Rebalancing band per block: 5pp, or 25% of target for targets under 20%."""
    return np.where(target_share >= 0.20, 0.05, 0.25 * target_share)


def simulate(start, R_paths, infl_paths, rebal_blocks, reserve=0.0, dips=(), contribution=0.0,
             ref_weights=None, rebalance_every=REBALANCE_EVERY, streams=(), contrib_blocks=None):
    """Simulate dollar paths.

    start: pd.Series of starting dollars per block (liquid only). R_paths: (paths, months,
    blocks) monthly returns in start.index order. rebal_blocks: blocks kept at their starting
    shares (by bands); others drift. reserve: cash (inside the CASH block) deployable by
    `dips` = [(drawdown, fraction)] of the reference index (ref_weights over start.index).
    contribution: flat nominal $/month forever. streams: profile contribution streams
    ({monthly_usd, months, indexed}); indexed ones grow with the path's realized inflation.
    New money goes to rebal_blocks at target weights, or, if contrib_blocks is given (a
    buy-and-hold portfolio with nothing to rebalance), to those blocks pro rata to start.
    Returns dict of (paths, months+1) arrays: nominal and real WEALTH (includes new money),
    the time-weighted return index twr / twr_real (investment performance only: contributions
    excluded, so they can't hide a drawdown), and the reference index level.
    """
    n_paths, n_months, n_blocks = R_paths.shape
    names = list(start.index)
    v = np.tile(start.values.astype(float), (n_paths, 1))
    rb = np.array([n in rebal_blocks for n in names])
    tgt = np.where(rb, start.values, 0.0)
    tgt = tgt / tgt.sum() if tgt.sum() > 0 else tgt
    band = bands(tgt)
    if contrib_blocks is not None:
        cb = np.array([n in contrib_blocks for n in names])
        cw = np.where(cb, start.values, 0.0)
        cw = cw / cw.sum() if cw.sum() > 0 else cw
    else:
        cb, cw = rb, tgt
    cash_i = names.index("CASH") if "CASH" in names else None
    res = np.full(n_paths, float(reserve))
    fired = np.zeros((n_paths, len(dips)), dtype=bool)
    refw = np.zeros(n_blocks) if ref_weights is None else np.array([ref_weights.get(n, 0.0) for n in names])
    ref, ref_peak = np.ones(n_paths), np.ones(n_paths)
    deflator = np.ones(n_paths)
    nominal = np.empty((n_paths, n_months + 1))
    real = np.empty_like(nominal)
    ref_level = np.empty_like(nominal)
    twr = np.empty_like(nominal)
    twr_real = np.empty_like(nominal)
    nominal[:, 0] = real[:, 0] = v.sum(1)
    twr[:, 0] = twr_real[:, 0] = 1.0
    ref_level[:, 0] = 1.0
    unit = np.ones(n_paths)
    for t in range(n_months):
        r = R_paths[:, t, :]
        before = v.sum(1)
        v = v * (1 + r)
        unit = unit * np.where(before > 0, v.sum(1) / np.where(before > 0, before, 1), 1.0)
        if cash_i is not None:  # reserve sits in cash and earns cash returns
            res = res * (1 + r[:, cash_i])
        if refw.any():
            ref = ref * (1 + r @ refw)
            ref_peak = np.maximum(ref_peak, ref)
        deflator = deflator * (1 + infl_paths[:, t])
        add = np.full(n_paths, float(contribution))
        for st in streams:
            if st["monthly_usd"] and (st["months"] is None or t < st["months"]):
                add = add + st["monthly_usd"] * (deflator if st["indexed"] else 1.0)
        if add.any():
            v[:, cb] += add[:, None] * cw[cb][None, :]
        # dip buying: move reserve cash into the rebalanced blocks at target weights
        if dips and cash_i is not None:
            dd = 1 - ref / ref_peak
            for k, (trig, frac) in enumerate(dips):
                go = (~fired[:, k]) & (dd >= trig)
                if go.any():
                    amt = np.minimum(frac * reserve, res[go])
                    amt = np.minimum(amt, v[go, cash_i])
                    v[go, cash_i] -= amt
                    v[np.ix_(go, rb)] += amt[:, None] * tgt[rb][None, :]
                    res[go] -= amt
                    fired[go, k] = True
        if rebalance_every and (t + 1) % rebalance_every == 0 and rb.any():
            pot = v[:, rb].sum(1)
            share = v[:, rb] / pot[:, None]
            off = (np.abs(share - tgt[rb][None, :]) > band[rb][None, :]).any(1)
            if off.any():
                v[np.ix_(off, rb)] = pot[off, None] * tgt[rb][None, :]
        nominal[:, t + 1] = v.sum(1)
        real[:, t + 1] = nominal[:, t + 1] / deflator
        twr[:, t + 1] = unit
        twr_real[:, t + 1] = unit / deflator
        ref_level[:, t + 1] = ref
    return {"nominal": nominal, "real": real, "twr": twr, "twr_real": twr_real, "ref": ref_level}


# --- metrics ---------------------------------------------------------------------------
def max_drawdown(x):
    peak = np.maximum.accumulate(x, axis=1)
    return (1 - x / peak).max(1)


def longest_underwater_months(x):
    """Longest run of months spent below a previous peak, per path."""
    below = x < np.maximum.accumulate(x, axis=1) - 1e-9
    best = np.zeros(x.shape[0], dtype=int)
    run = np.zeros(x.shape[0], dtype=int)
    for t in range(x.shape[1]):
        run = np.where(below[:, t], run + 1, 0)
        best = np.maximum(best, run)
    return best


def worst_rolling_lag(x, bench, window=LAG_WINDOW_MONTHS):
    """Worst `window`-month cumulative return of x minus that of bench, per path."""
    if x.shape[1] <= window:
        return np.full(x.shape[0], np.nan)
    rx = x[:, window:] / x[:, :-window]
    rb = bench[:, window:] / bench[:, :-window]
    return (rx - rb).min(1)


def summarize(sim, bench_sim, start_total, lag_threshold):
    real = sim["real"] / start_total
    out = {}
    for h in HORIZONS_YEARS:
        m = h * 12
        if m < real.shape[1]:
            w = real[:, m]
            out[f"real_x_{h}y_p50"] = np.median(w)
            out[f"real_x_{h}y_p05"] = np.quantile(w, 0.05)
    # Risk metrics use the time-weighted index (investment performance), not wealth: new
    # contributions would otherwise mask drawdowns and shorten time underwater.
    dd = max_drawdown(sim["twr"])
    out["p_maxdd_30"] = (dd > 0.30).mean()
    out["p_maxdd_40"] = (dd > 0.40).mean()
    uw = longest_underwater_months(sim["twr_real"])
    out["underwater_months_p50"] = np.median(uw)
    out["underwater_months_p95"] = np.quantile(uw, 0.95)
    if real.shape[1] > 60:
        out["p_underwater_after_5y_real"] = (sim["twr_real"][:, 60] < 1).mean()
        out["p_wealth_below_start_5y_real"] = (real[:, 60] < 1).mean()
    if bench_sim is not None:
        m = LAG_SEARCH_YEARS * 12 + 1
        lag = worst_rolling_lag(sim["twr"][:, :m], bench_sim["twr"][:, :m])
        out["p_lag_5y"] = (lag <= -lag_threshold).mean()
        out["lag_threshold"] = lag_threshold
        out["worst_5y_lag_p50"] = np.nanmedian(lag)
    return out


# --- driver ----------------------------------------------------------------------------
def candidate_inputs(profile):
    """Per portfolio: starting liquid blocks, the blocks to rebalance, and the reserve."""
    import allocate as A
    state = A.account_state(profile=profile)
    out = {"0 Today (buy & hold)": (S.today_blocks().drop("PRIVATE_RE", errors="ignore"), set(), 0.0)}
    hold = set(profile.satellites) | {"CASH", "PRIVATE_RE"}
    for name, target in profile.candidates.items():
        fills, _ = A.allocate(target, state, profile)
        blocks = A.to_blocks(fills, state).drop("PRIVATE_RE", errors="ignore")
        reserve = fills[fills["class"] == "reserve"].usd.sum()
        out[name] = (blocks, set(blocks.index) - hold, reserve)
    return out


def run(profile, n_paths=N_PATHS, seed=SEED, runs=None):
    inputs = candidate_inputs(profile)
    all_blocks = sorted({b for blocks, _, _ in inputs.values() for b in blocks.index} | set(REFERENCE_BLOCKS))
    tbill = R.current_carry()["cash"]
    runs = runs or [("hist 1990-2026", None), ("hist 1954-2026", None), ("CMA median", "median"),
                    ("CMA 25th pct", "p25"), ("CMA GMO", "GMO"), ("CMA BlackRock", "BlackRock")]
    n_months = max(HORIZONS_YEARS) * 12
    rows = []
    for run_name, cma_set in runs:
        window = WINDOWS.get(run_name, WINDOWS["hist 1990-2026"])
        raw, infl = block_history(all_blocks, window)
        hist = recenter(raw, cma_targets(all_blocks, cma_set, tbill)) if cma_set else raw
        rng = np.random.default_rng(seed)
        idx = bootstrap_indices(len(hist), n_months, n_paths, BLOCK_MONTHS, rng)
        # forecast months first, then un-recentered history (same sampled months)
        cut = cma_horizon_months(cma_set) if cma_set else n_months
        Hfull = np.concatenate([hist.values[idx][:, :cut], raw.values[idx][:, cut:]], axis=1)
        I = infl.values
        for name, (blocks, rebal, reserve) in inputs.items():
            cols = [all_blocks.index(b) for b in blocks.index]
            Rp = Hfull[:, :, cols]
            ref_w = {k: v for k, v in R.REFERENCE.items()}
            policy_rebal = rebal if "Today" not in name else set()
            # Same savings for every portfolio, including today's (the fair counterfactual for
            # "should I rebalance?"). Today has nothing to rebalance, so its new money is added
            # to what it already holds (not cash, not satellites), pro rata.
            streams = profile.contributions()
            today = "Today" in name
            hold_out = set(profile.satellites) | {"CASH"}
            sim = simulate(blocks, Rp, I[idx], policy_rebal, reserve, profile.dip_buying if reserve else (),
                           ref_weights=ref_w, streams=streams,
                           contrib_blocks=(set(blocks.index) - hold_out) if today else None)
            bench = R.benchmark_blocks(blocks)
            bcols = [all_blocks.index(b) for b in bench.index]
            bench_rebal = (set(bench.index) - set(profile.satellites) - {"CASH"}) if "Today" not in name else set()
            bsim = simulate(bench, Hfull[:, :, bcols], I[idx], bench_rebal, reserve,
                            profile.dip_buying if reserve else (), ref_weights=ref_w, streams=streams,
                            contrib_blocks=(set(bench.index) - hold_out) if today else None)
            rows.append({"run": run_name, "portfolio": name,
                         **summarize(sim, bsim, blocks.sum(), profile.failures["lag_vs_benchmark"])})
    return pd.DataFrame(rows)


def describe_streams(profile):
    parts = []
    for c in profile.contributions():
        if c["monthly_usd"]:
            span = "ongoing" if c["months"] is None else f"{c['months'] / 12:g} yrs"
            idx = ", inflation-indexed" if c["indexed"] else ""
            parts.append(f"{c['name']}: ${c['monthly_usd']:,.0f}/mo, {span}{idx}")
    return "; ".join(parts) or "none"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profile")
    p.add_argument("--paths", type=int, default=N_PATHS)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    P.use(args.profile) if args.profile else P.current()
    profile = P.current()
    print(profile.summary())
    print(f"{args.paths:,} paths x {max(HORIZONS_YEARS)} yrs, {BLOCK_MONTHS}-month block bootstrap | "
          f"rebalance check every {REBALANCE_EVERY} mo (bands 5pp / 25% rel.) | dip rule {profile.dip_buying} | "
          f"contributions ${profile.monthly_contribution:,.0f}/mo today ({describe_streams(profile)}) | "
          f"liquid only (private RE excluded)\n")
    df = run(profile, args.paths)
    out = profile.output_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "montecarlo_summary.csv", index=False)
    show = {"real_x_3y_p05": "3y real p5", "real_x_10y_p50": "10y real p50", "real_x_10y_p05": "10y real p5",
            "real_x_30y_p50": "30y real p50", "p_maxdd_40": "P(DD>40%)", "underwater_months_p95": "underwater p95 (mo)",
            "p_underwater_after_5y_real": "P(under at 5y, real)",
            "p_lag_5y": f"P(5y lag>={profile.failures['lag_vs_benchmark']:.0%})"}
    for run_name, d in df.groupby("run", sort=False):
        print(f"== {run_name} ==  (real p = wealth incl. contributions as a multiple of today's liquid; "
              "DD / underwater / lag = investment returns excluding new money)")
        t = d.set_index("portfolio")[list(show)].rename(columns=show)
        fmt = {c: ("{:.2f}x" if "real p" in c else "{:.0f}" if "mo)" in c else "{:.0%}") for c in t.columns}
        print(t.apply(lambda col: col.map(fmt[col.name].format)).to_string(), "\n")


if __name__ == "__main__":
    main()
