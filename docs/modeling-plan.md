# Modeling plan: scenarios, stress tests, and a robust allocation

The approach behind the scripts, model by model. Each model's assumptions, with evidence and
sensitivity, are in [`assumptions.md`](assumptions.md) under the same letter.

## What the modeling has to answer

1. **How much AI am I actually exposed to today?** Across every account, through every fund,
   plus any stocks held directly.
2. **What happens to the whole portfolio in each scenario?** That includes AI unwinds, broad
   crashes, inflation shocks, and the scenario where AI keeps ripping.
3. **Which allocation is good enough everywhere?** We want the one that holds up under every
   return assumption, not the one that's optimal under a single assumption.
4. **How much regret would I feel?** If AI keeps winning, how far behind the plain market
   would each candidate fall, and for how long?

**Why not a single efficient frontier?** The first version here (`scripts/frontier.py`, kept
for reference) used one 26-year history and one forecast, left out direct stocks, cash and
real estate, ran no AI scenario, and optimized for a single "best" point, which just chases
whatever return assumption goes in. Everything below exists to fix one of those.

## Sleeves (the building blocks)

Every script first converts "dollars in funds" into "dollars in sleeves".

| Sleeve | Proxy for history | Notes |
|---|---|---|
| US large: AI basket | Look-through weights × stock returns | Defined in `data/ai_basket.csv` (core + adjacent tiers) |
| US large: everything else | S&P 500 minus the AI basket | |
| US mid/small blend | Vanguard small cap index (NAESX) | |
| US small value | Vanguard small value (VISVX) / Fama-French small value back to 1927 | |
| International developed | EAFE / Fama-French developed ex-US (1990+) | |
| Emerging markets | Vanguard EM (VEIEX, 1994+) | |
| US bonds | Vanguard total / intermediate bond (VBMFX, VBIIX) | |
| TIPS, short TIPS | Vanguard TIPS (VIPSX, 2000+), VTIP (2012+) | |
| Cash | T-bills | |
| Private real estate | Hand-set loadings, held at cost | Can't rebalance it or sell it in a crash |
| Direct stocks | Their own price history | Counted in the AI basket if listed there |
| Crypto | ETH history from 2017 | Usually immaterial |
| Frontier labs (direct) | `data/private_ai.csv` | Index inclusion after a hypothetical IPO |

## Data inputs

- `profiles/<name>/holdings/`: your account exports.
- `data/fund_holdings/`: published holdings of the iShares proxy ETFs
  (`fetch_fund_holdings.py`). Vanguard and most 401(k) funds block scripted downloads, so
  each block maps to an iShares ETF tracking the same or a near-identical index.
- `data/prices/`: monthly returns (`fetch_prices.py`, `fetch_history.py`).
  - Index mutual funds via yfinance.
  - The Ken French Data Library for US size/value back to 1926 and international since
    1990, which covers the 1929–32 crash, the 1970s and 1973–74.
  - FRED: CPI, 10-yr yield, Baa spread, T-bills, breakeven inflation.
- `data/cma.csv`: 10-year capital market assumptions from 16 firms, dated and sourced, so
  no single forecast drives the answer.

## Models

### A. Look-through exposure (`lookthrough.py`)
- **Shows:** total exposure by region, size/style and sector, plus top-10 names and AI
  basket %.
- **Covers:** liquid assets, then the whole net worth.

### B. Historical scenario replay (`stress.py`)
Each candidate is run through real episodes using each sleeve's actual returns, with
fallbacks to Ken French and synthetic FRED series before fund inception:

| Episode | Why it matters |
|---|---|
| 2000–02 dot-com | The closest analog to an AI unwind: a concentrated, expensive tech leadership deflating |
| 2007–09 GFC | A broad crash where nothing in stocks helped |
| 2020 COVID | A fast crash with a fast recovery, which tests a buy-the-dip rule |
| 2022 inflation | Stocks and bonds fell together, which tests TIPS |
| 1973–74 / 1970s | Stagflation (Ken French and synthetic series only) |
| 2023–now AI run | The regret case: how much each tilt cost |

**Outputs:** peak-to-trough drop, dollar loss on today's balances, and which series each
block used.

### C. Hypothetical shock scenarios (`data/scenarios.csv` + `stress.py`)
Explicit, editable shocks per sleeve, each with a horizon. They are assumptions, not
predictions. Examples:

| Scenario | AI basket | Rest of US large | Small value | Intl | Bonds |
|---|---|---|---|---|---|
| AI unwind, mild | −40% | −15% | −5% | −15% | +5% |
| AI unwind, severe | −65% | −30% | −15% | −30% | +10% |
| Broad recession | −45% | −45% | −50% | −45% | +8% |
| Stagflation | −30% | −25% | −20% | −20% | −15% |
| AI keeps ripping | +80% | +20% | +10% | +15% | +3% |

The file also has a credit-stress fork (the Fed can cut vs. inflation keeps it from cutting),
the actual 2000–09 lost decade, and Japan-style 1989–2012 paths for US equity.

**Frontier-lab boom-bust (Anthropic / OpenAI IPO).** Modeled through two channels.
1. **Direct.** Each lab enters an index fund only under that fund's rules (e.g. Morningstar
   total-market indexes after the 5th trading day; the S&P 500 needs 12 months of trading
   and GAAP profitability). Weights follow the float, not the headline valuation.
   Assumptions are in `data/private_ai.csv`.
2. **Indirect, the bigger channel.** The labs are major customers of, and partly owned by,
   the AI basket. A post-IPO bust reprices those stakes and contracts, which shows up as an
   AI-unwind shock on the basket.

### D. Monte Carlo (`montecarlo.py`)
- **Method:** circular block bootstrap of monthly history (12-month blocks, so crashes stay
  clustered), over 1990– and 1954–.
- **Re-centering:** each block's mean is re-centered to a forecast set over that forecast's
  own horizon; one run per firm, plus median, 25th percentile and history.
- **Behavior included:** the profile's contributions, band rebalancing, and the dip-buying
  rule.
- **Outputs:** real wealth at 3/10/20/30 years (median and 5th percentile), probability of
  30%/40% drawdowns, time underwater, probability of a 5-yr lag versus the benchmark.

### E. Robust frontier (`robust_frontier.py`)
- ~195 mixes (stock share × intl share × SCV share × TIPS share) are run through the real
  allocator and the Monte Carlo under 6 return sets, with common random numbers.
- Two objectives: 3-yr 5th-percentile real wealth (risk) and 10-yr median (growth).
- The question: does any mix beat a candidate on both, in **every** run?

### F. Regret / tracking error (`regret.py`)
- For each candidate versus the plain market (a 60/40 US/intl index plus the same bonds):
  the worst 5-year lag and the longest stretch behind, 1954–2026.
- Also the SCV tilt alone versus a no-SCV twin, which sizes a tilt you won't abandon.

### G. Reverse stress test (`reverse_stress.py`)
- **The idea:** instead of naming futures, define failure and search 100K random worlds for
  the ones that cause it. The next crisis won't look like the last one; you define failure,
  you don't predict the future.
- **Worlds:** 5 factors (equity, AI, rates, credit, inflation), correlated as in 1997–2026 by
  default, mapped to sleeve shocks via loadings fitted by `calibrate.py`. See the README for
  how to read (and not misread) the output.

## Candidate allocations

Candidates live in the profile (`[candidates.*]` in `profile.toml`). The example profile's
are typical shapes worth comparing:

| Name | Stocks/bonds | Intl % of stocks | SCV % of stocks |
|---|---|---|---|
| Three-fund 80/20 | 80/20 | 30% | 0% |
| SCV 80/20 | 80/20 | 30% | 15% |
| Defensive 75/25 | 75/25 | 35% | 20% |
| Aggressive 90/10 | 90/10 | 30% | 10% |

`compare.py` scores every candidate (plus "0 Today") on: AI basket %, top-10 %, every
scenario and replay, reverse-stress breach rates and cushions, Monte Carlo percentiles,
regret, and blended expense ratio.

## Cross-checks with outside tools
- Backtest candidates in **testfol.io** or **Portfolio Visualizer**. Results should be in the
  same ballpark (see [`cross-checks.md`](cross-checks.md)).
- Check the look-through against your broker's portfolio analysis, where it has one.
