# Diamond Hands 💎🙌

A scenario-modeling toolkit for answering two questions about your own portfolio:

1. **How concentrated am I, really?** Look through every fund in every account to the
   underlying companies, regions and sizes, with a focus on exposure to AI-driven names
   (NVIDIA, Microsoft, Alphabet, Amazon, Meta, Broadcom, TSMC, ...).
2. **Which allocation holds up if that concentration unwinds, and also if it doesn't?**
   Score candidate allocations against named scenarios, real historical episodes,
   100K randomized "worlds", a Monte Carlo of multi-year paths, and regret against the
   plain index.

It is built to be driven together with [Claude Code](https://claude.com/claude-code): you
describe your accounts and goals, and Claude builds your profile, runs the models and
explains the output. Run `/setup-profile` in Claude Code to start.

> **Not financial advice.** This is a hobby tool that friends share. It makes many
> simplifying assumptions (see [`docs/assumptions.md`](docs/assumptions.md) and
> [Known limitations](#known-limitations)). Use it to sharpen your own thinking, check
> anything important with a professional, and decide for yourself.

- [`docs/modeling-plan.md`](docs/modeling-plan.md): the modeling approach (models A–G)
- [`docs/assumptions.md`](docs/assumptions.md): every assumption, with its evidence and sensitivity
- [`docs/cross-checks.md`](docs/cross-checks.md): results checked against an independent tool
- [`AGENTS.md`](AGENTS.md) (also `CLAUDE.md`): orientation for Claude and other AI agents

## Quick start

```sh
git clone https://github.com/lionel-lints/diamond-hands.git && cd diamond-hands
# needs uv (https://docs.astral.sh/uv/); Python dependencies are pinned in uv.lock
uv run python -m pytest                     # ~10s
uv run python scripts/lookthrough.py        # runs on the fictional example profile
```

Then, in Claude Code, run `/setup-profile`. Or set a profile up by hand (see [Profiles](#profiles)).

**Your data stays local.** Everything under `profiles/` except `example/` is gitignored, as are
all outputs. You can fork the repo, even publicly, and keep your holdings off GitHub. Check
`git status` before you push anything.

## Layout

The engine and shared market data are the same for everyone. Each person's inputs live in a
**profile** folder.

```
profiles/
  example/           # fictional person: a working template, and what the tests run on
    profile.toml     # horizon, emergency fund, satellites, failure thresholds, AI cap,
                     #   candidate allocations, chosen IPS target, savings
    accounts.toml    # accounts + mode (movable/frozen/bank), funds per asset class,
                     #   asset-location order
    plan_menu.csv    # employer 401(k) menu (fund -> model block, expense ratios)
    holdings/        # account exports (no account numbers); optional `block` column
    ips.md           # template: your investment policy statement
    decisions.md     # template: your dated decision log
  <you>/             # yours (gitignored)
  _holdings_template.csv
data/                # shared
  fund_catalog.csv   # ticker / fund name -> model block (extend it for new funds)
  fund_holdings/     # published holdings of proxy ETFs (iShares), dated files
  prices/            # monthly prices, Ken French factors, FRED macro series
  ai_basket.csv      # default AI tickers by tier (a profile may override)
  scenarios.csv      # named hypothetical shocks per sleeve
  private_ai.csv     # frontier-lab IPO assumptions
  cma.csv            # 10-yr capital market assumptions from 16 firms
  loadings_fitted.csv, factor_corr.csv   # written by calibrate.py
scripts/             # engine; blocks.py = shared model vocabulary, user_profile.py = loader
tests/               # unit (fake profile), integration (example profile), opt-in network
output/<profile>/    # per-profile results (gitignored); output/ root = shared model results
```

## Profiles

Every script runs on one profile, chosen by:
1. `--profile <name|path>`
2. `$PORTFOLIO_PROFILE`
3. the name in `.portfolio-profile` (a gitignored file at the repo root)
4. `example`

The profile's inputs are printed at the top of every run, so the assumptions behind a result
are always visible.

**To set up a new person** (or let `/setup-profile` do it):
1. `cp -r profiles/example profiles/<you>` and `echo <you> > .portfolio-profile`.
2. Put holdings exports in `holdings/` using the columns in `profiles/_holdings_template.csv`.
   - Funds whose ticker or name is in `data/fund_catalog.csv` (or your `plan_menu.csv`) are
     classified automatically.
   - Otherwise set `block` yourself: `CASH`, `PRIVATE_RE`, `SP500`, ... (see `scripts/blocks.py`).
   - Use `STOCK` for an individual stock (including employer stock or RSUs).
3. List every account in `accounts.toml` with its mode, the funds it can buy per asset class,
   and the location order.
4. If you have an employer plan, list its menu in `plan_menu.csv` (fund name, model block,
   expense ratio). Your plan's fee disclosure (the "404a-5 notice") has all of it.
5. Set your horizon, emergency fund, failure thresholds, candidates and savings streams
   (`[[contributions]]`: dollars per year or `"max:401k_employee"`, `"max:ira"` from
   `data/contribution_limits.csv`) in `profile.toml`.
6. Run `lookthrough.py`, then `allocate.py`, `stress.py`, `reverse_stress.py`, `compare.py`.

Mistakes fail loudly with the file and the problem, e.g. an unlisted account, a fund that
isn't in the catalog, or shares that don't sum to 1.

## What it models

- **Your whole balance sheet, by sleeve.** Every fund is looked through its proxy ETF into
  ~13 sleeves. Individual stocks, crypto and private real estate are carried as
  "satellites": counted in exposure and net worth, held unchanged in every candidate.
- **Real buyable allocations.** Candidates are filled with the funds each account can
  actually buy, using your asset-location order. Accounts are `movable` (fully
  re-allocatable), `frozen` (held as-is, e.g. taxable with large gains) or `bank` (cash:
  emergency fund first, the rest is new money).
- **Risk, five ways.** Named shocks (AI unwinds, recession, stagflation, Japan-style lost
  decades, a frontier-lab IPO boom-bust), historical replays back to 1926, a reverse stress
  test over 100K random worlds, a block-bootstrap Monte Carlo with your contributions,
  rebalancing bands and dip-buying rule, and regret versus the plain index.
- **Return uncertainty.** Forecasts from 16 firms (`data/cma.csv`), run separately, so no
  single forecast drives the answer.

## Known limitations

These things are **not modeled yet**. If your situation depends on one, read the results with
that in mind, or extend the engine (see [Contributing](#contributing)).

- **Partial selling in taxable accounts.** A taxable account is either `frozen` (never sold)
  or fully `movable`, which ignores taxes. There is no cost basis, capital-gains budget or
  tax-lot selection. If most of your money is in taxable accounts, `frozen` leaves little to
  rebalance and `movable` overstates your flexibility.
- **Private-company equity.** There's no block for illiquid startup stock or options (e.g.
  equity in a private employer). The closest stand-ins are `PRIVATE_RE` (illiquid, held at
  cost) or a direct `STOCK` of a comparable public company, and neither is right.
- **Your job as an AI exposure.** Human capital isn't modeled. Contributions continue in
  every scenario, even ones where an AI bust would put your income at risk. If you work in
  tech, your paycheck is correlated with the AI basket. The only lever today is your
  emergency fund and failure thresholds.
- **Employer stock is treated like any public stock.** Public employer stock and RSUs work as
  `STOCK` satellites (and count toward AI exposure if they're in `data/ai_basket.csv`), but
  vesting schedules, lockups and concentration rules aren't modeled.
- **Taxes, withdrawals and decumulation.** No tax drag, no required withdrawals, no
  retirement-income phase. This is an accumulation-phase tool.
- **US-centric.** US account types, IRS contribution limits, USD returns.
- **Fund coverage.** Funds are mapped to ~13 index-like blocks via iShares proxies. Active
  funds, sector funds, target-date funds and bond types beyond intermediate core/TIPS need a
  catalog entry, and may need a new block.
- **Monthly data.** Drawdowns read ~3–4pp shallower than daily statements
  ([cross-checks](docs/cross-checks.md)).

## Contributing

Fork it and make it fit your situation. If you build something others could use (one of the
limitations above, a new fund block, a new scenario), open a pull request. Please:
- keep personal data out of the PR (profiles and outputs are gitignored, keep it that way)
- add tests that run on the fake profile (`tests/conftest.py`) or the example profile
- update `docs/assumptions.md` in the same PR as any change to an assumption

## The pipeline

Every script first converts "dollars in funds" into "dollars in ~13 sleeves" (ai_core,
us_large_other, us_smid, us_small_value, intl_dev, em, bonds, tips, cash, private_re,
crypto, labs_direct) via the proxy-ETF look-through. All analysis is arithmetic on those
sleeve dollars.

| Script | Question | Method |
|---|---|---|
| `lookthrough.py` | What do I actually own? | Funds → contents: region, size, sector, top names, AI % |
| `allocate.py` | What exactly do I buy, where? | Candidate targets → real funds per account (taxable frozen, asset location) |
| `stress.py` | What happens in futures I named? | Applies `scenarios.csv` shocks; replays real episodes (2000–02, 2008, 2020, 2022, 2023–now) |
| `frontier.py` | Return/vol tradeoff of US/intl/bonds/TIPS? | Grid over historical + forward returns |
| `reverse_stress.py` | **What would have to happen for me to fail?** | Searches 100K random worlds for the ones breaching failure conditions |
| `montecarlo.py` | **What happens along real multi-year paths?** | 12-month block bootstrap of history (1990– and 1954–), re-centered to 16-firm forecasts; rebalancing bands, dip rule, contributions; real wealth at 3/10/20/30 yrs, drawdowns, time underwater, 5-yr lag |
| `robust_frontier.py` | Is there a better mix near my candidates under EVERY forecast? | ~195 mixes through the allocator + Monte Carlo under 6 return sets (common random numbers); 3-yr 5th pct vs. 10-yr median; which mixes beat a candidate in all runs |
| `regret.py` | How long and how badly would a tilt have made me trail? | 1954–2026 constant-mix replay vs. the plain-index benchmark, and the SCV tilt alone vs. a no-SCV twin; worst 5-yr lag, longest stretch behind, checked against the profile's lag limit |
| `compare.py` | **One table: every candidate on every model** | Runs exposure, scenarios, replay, reverse stress, Monte Carlo, regret, and costs fresh → `output/<profile>/comparison.csv` |
| `scenario_matrix.py` | **Which scenario hurts each blend most?** | Every named scenario + replay × candidates and stock/intl variations; each blend's worst and best; optional `[scenario_beliefs]` weights |
| `indicators.py` | **Is the concentration behind my hedges still there?** | US CAPE (live), S&P top-10 share, core AI share of the US market, forecast consensus vs. the profile's `[review_indicators]` baseline/thresholds (IPS §9) |
| `calibrate.py` | Do the reverse-stress assumptions match history? | Fits sleeve loadings and factor correlations (shared, not per profile) |

## Running

Per profile (add `--profile <name>` to override the default):

```sh
uv run python scripts/lookthrough.py
uv run python scripts/allocate.py
uv run python scripts/stress.py
uv run python scripts/reverse_stress.py   # --factors/--loadings/--no-carry for sensitivity
uv run python scripts/montecarlo.py       # --paths N
```

Shared (same for everyone):

```sh
uv run python scripts/frontier.py
uv run python scripts/calibrate.py        # refreshes data/loadings_fitted.csv, data/factor_corr.csv
uv run python -m pytest                   # tests
```

Refresh data before big decisions:
- `fetch_fund_holdings.py` (iShares CSVs)
- `fetch_prices.py` (yfinance; also fetches the active profile's directly held stocks)
- `fetch_history.py` (Ken French + FRED)

Re-run look-through + stress at each rebalance check.

## The reverse stress test: how to use it (and how not to)

`stress.py` starts from a scenario and computes a loss. `reverse_stress.py` starts from a
loss and finds the scenarios.

**Failure** is defined in the profile (`profile.toml [failures]`). The example's:
- liquid −40%
- net worth −30%
- **lags the market by ≥10pp**: versus the same portfolio with all stocks swapped for a 60/40
  US/intl index (tilt regret, in up *or* down markets)
- **stocks alone −50%**: the "losing your nerve" test

**How worlds are built.** It generates 100K random **12-month** "worlds":
1. It rolls 5 factors per world (*equity*, *ai*, *rates*, *credit*, *inflation*), each
   uniform over a range. Stress sides are deliberately wider than history, and the *ai*
   upside reaches the 1999–2000 historical maximum (+110%) so melt-up lag failures are in
   range (`FACTOR_RANGE`). Inflation is kept to its 1997–2026 range: a wide uniform range
   made 5% the typical world (assumption R13).
2. **By default the factors are correlated as in 1997–2026** (Gaussian copula,
   `--factors historical`): equity–credit 0.71, rates–credit 0.50. Without that,
   implausible combinations dominate, such as a credit crunch while stocks rally.
3. Each sleeve's shock is loadings × factors + noise + carry. Bonds/TIPS earn the 10-yr
   yield and cash the T-bill rate. TIPS also earn the inflation surprise (realized minus the
   2.37% breakeven). Every world's inflation deflates results for the real-terms failure
   (`real_networth_drawdown` in the profile).
4. **By default, sleeves with usable history use loadings fitted by `calibrate.py`**
   (`--loadings fitted`).
   - AI sleeves, crypto and private real estate keep hand-set loadings.
   - Real estate keeps hand-set loadings because public REITs are no proxy for unlevered
     at-cost housing deals.
5. `--factors independent --loadings assumed` is the original design, kept for sensitivity.

**Bonds depend on that choice** (assumption R11):
- **Fitted** on an intermediate-term bond fund (VBIIX), bonds **fall** in credit stress
  (credit +0.40).
- **Assumed**, they rise (−0.2).
- History supports both readings in parts: Treasuries rally in panics, while intermediate
  funds hold corporates.
- The named `ai_unwind_credit_*` scenario pair is the tiebreaker.

**What it reports.** Per portfolio × failure:
- how many worlds break it
- the **mildest** world that breaks it, measured on a yardstick that doesn't depend on the
  portfolio: how a plain global stock index (60% US / 40% intl, `REFERENCE`) did in that world
- what the breaking worlds have in common
- a **plausibility flag (⚠)** on that mildest world when any factor is outside its
  calibrated historical range (`data/factor_hist_range.csv`) or a held sleeve is pinned at
  the −85% / +100% clip

### Reading the output

```
alt: Three-fund 80/20  breaks in 10.7% of worlds | breaking worlds: ai_core avg -74%,
                    bonds down in 50% | mildest breach (global stocks -44%): ai_core -85%, ...
                    [equity -0.17, ai -0.64, rates +0.20, credit -0.23]
                    ⚠ implausible? ai -0.64 outside history [-0.47, +1.10]; ai_core at clip -85%
```

- **breaks in X% of worlds** — share of the explored world-space that breaches. Compare
  this *across portfolios*; the ranking is the signal.
- **breaking worlds: ai_core avg / bonds down** — the signature of the worlds that break
  you. Compare "bonds down" to the all-worlds baseline printed at the top before reading
  anything into it.
- **mildest breach** — of all the worlds that break you, the one where global stocks did
  *best*. Read it as "this allocation breaks only if global stocks fall worse than X%".
  The bracket shows that world's factor draws, so you can see *why* it broke (e.g. rates
  up + credit down = bonds failed too). This is the decision-relevant row, and unlike the
  old gross-move metric it is comparable across candidates.
- **⚠ implausible?** — the mildest world relies on extrapolation: a factor beyond anything
  in 1997–2026, or a sleeve pinned at the clip. Discount its level. Look at the next-mildest
  unflagged breaches in `output/<profile>/reverse_stress_summary.csv`, or tighten
  `FACTOR_RANGE` and rerun.

### How to misread it (don't)

1. **"10.7%" is NOT a probability.** The factors are uniform over wide ranges. Even with
   historical *correlation*, extreme worlds are over-represented relative to reality:
   `--factors historical` means historically *correlated*, not historically *likely*. It
   means "10.7% of the possibility space I explored", not "a 10.7% chance this happens".
   Never quote these as odds. Use them to rank allocations, which *is* robust: the ranking
   on liquid −40% and net worth −30% is identical under all 4 factor × loading combinations.
2. **Mildest ≠ most likely.** The mildest breach is the best global-stock outcome that
   still breaks you — a measure of how much cushion exists, not of how likely that world
   is. (An earlier version used "gross dollars moved"; that was circular — any world that
   breaks "−30%" moves ≥30% — and printed the threshold back for every portfolio.)
3. **It's one period, not a path.** No rebalancing, contributions, dip-buying, or
   recovery. That's the Monte Carlo model's job (see `docs/modeling-plan.md` §D).
4. **The dice are assumptions.** The ranges, the correlation, and the loadings (fitted or
   hand-set) encode how sleeves move together, e.g. whether bonds hedge. Different
   choices give different *levels*. Before quoting a cushion, check its spread across the
   four variants (`output/<profile>/reverse_stress_variants.csv`, written by every run, with
   the spread printed at the end). In one real profile, cushions spread 4–6pp for 90/10
   options and up to ~11pp for 80/20 ones.

### The decision rule

For each candidate, look at the **mildest breaking world** and ask: *is that a world I
find plausible?* If breaking an allocation requires ai_core −80% with everything else
halved and you consider that truly implausible, you can hold it. If the mildest breach
is a world that keeps you up at night, go more defensive. That judgment — not the breach
percentages — is what this tool is for.

### The sensitivity habit

Before trusting any conclusion, argue with the knobs at the top of the file:
- `profile.toml [failures]` — define failure honestly (e.g. "liquid < 2 years of expenses")
- `FACTOR_RANGE` — how extreme the dice get
- `--factors` / `--loadings` / `--no-carry` — the four-way sensitivity; `LOADINGS` for the
  hand-set values (e.g. make bonds more credit-driven if you think the next AI unwind
  transmits through debt markets)

Change one assumption, rerun, and check whether the **ranking of candidates survives**.
Conclusions that survive knob-twisting are the ones to act on; conclusions that flip are
telling you which assumption actually drives the answer.

## Conventions

- Keep your own `profiles/<you>/decisions.md`: log every change of substance, dated, with why
  (template in `profiles/example/`).
- Never put account numbers in holdings files.

## License

[MIT](LICENSE). Market data in `data/` comes from public sources (Ken French Data Library,
FRED, iShares, Yahoo Finance, and published forecasts cited in each file) and remains
subject to their terms.
