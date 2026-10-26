# Assumptions register

Every assumption the models rely on, in one place. If a result depends on it, it belongs here.
Update this file in the same commit as any change to an assumption.

**Personal inputs vs. model assumptions.** Everything personal lives in a
profile (`profiles/<name>/`, see `scripts/user_profile.py`): horizon, emergency fund,
satellites, failure thresholds, AI cap, candidates (`profile.toml`); accounts, their modes,
funds and asset-location order (`accounts.toml`); holdings and plan menu. Rows below marked
**(profile)** are set per profile; they're listed here so their effect on results is visible.
Every script prints the active profile's inputs at the top of its output. Everything else is a
shared model assumption.

**Status**
- **verified**: checked against data or a primary source
- **sourced**: taken from a cited third party; not independently checked
- **judgment**: a deliberate modeling choice
- **unchecked**: a guess that still needs calibration

**Sensitivity**
- **tested**: we changed it and looked at the effect
- **low / high**: expected effect, not yet tested
- **?**: unknown

Last full review: 2026-10-22.

## Cross-cutting

| ID | Assumption | Where | Value | Evidence | Status | Sensitivity |
|---|---|---|---|---|---|---|
| X1 | **(profile)** Planning horizon for "what AI does next" | `profile.toml horizon_years`; `reverse_stress.HORIZON_YEARS` (model: 1-yr worlds) | per profile (example: 5 years); reverse-stress worlds are **12-month** moves | The profile's stated horizon. Reverse stress is calibrated on 12-mo windows, so its world = 1 year; 3-year paths are the Monte Carlo's job | judgment | high |
| X2 | Scenario and replay returns are nominal; Monte Carlo and reverse stress also report real (inflation-adjusted) results | all scripts | — | — | judgment | high in stagflation |
| X3 | No taxes, trading costs, or withdrawals; fees appear only as `compare.py`'s blended expense ratio | all scripts | — | Index ERs are typically 0.02–0.13%, so ignoring fees costs little; taxes matter for taxable accounts (see README, Known limitations) | judgment | low (fees), high (taxes when selling in taxable) |
| X4 | **(profile)** Emergency fund is a hard floor, never invested | `profile.toml emergency_fund` | per profile | The profile's decision | judgment | low |
| X5 | Shared fund catalog holds only public funds; plan-specific share classes stay in the profile | `data/fund_catalog.csv` vs. `profiles/<name>/plan_menu.csv` | catalog: public tickers with retail or blank ERs; plan menu: employer-menu names, blocks, ERs | Keeps one person's 401(k) menu out of the shared, published files | judgment | low |

## Look-through (model A, `lookthrough.py`)

| ID | Assumption | Where | Value | Evidence | Status | Sensitivity |
|---|---|---|---|---|---|---|
| A1 | iShares proxy ETFs stand in for the funds a profile holds | `lookthrough.PROXIES` | VTI/VTSAX/FZROX→ITOT; S&P 500 fund→IVV; VXUS/VFWAX→IXUS; Vanguard small cap→50% ISCV + 50% ISCG | Vanguard blocks scripted holdings downloads. Proxies track the same or near-identical indexes. VFWAX excludes small caps that IXUS includes | sourced | low (~±1pp on AI %) |
| A2 | Holdings snapshot is current | `data/fund_holdings/*_2026-10-20.csv` | as of 2026-10-19 | iShares files | verified | low until markets move a lot; re-fetch before decisions |
| A3 | US size buckets by cap rank: top 70% of cap is large, next 20% mid, rest small | `lookthrough.size_buckets` | 70/90 cut | Morningstar-style definition, applied to ITOT weights | judgment | low |
| A4 | The AI basket defines "AI exposure" | `data/ai_basket.csv` (shared default; a profile may override with its own `ai_basket.csv`) | 45 names in 5 tiers | A judgment call (edit it, or override per profile); e.g. Tesla is "narrative", Micron and SK Hynix are "core" | judgment | high: it *is* the metric |
| A5 | Name matching for international basket names is by name prefix | `lookthrough.tag` | — | Tiny false positives (Tencent Music, Alibaba Health: ~$7) | verified | low |
| A6 | Private real estate is valued at cost | holdings | the holdings file's `value_usd` | No marks available | judgment | ? (true value unknown) |
| A7 | South Korea and Canada are in "international" | IXUS proxy | — | IXUS includes them. **An EAFE + FTSE EM pair (common in 401(k) menus) excludes both** (MSCI EAFE has no Canada or Korea; FTSE EM has no Korea) | verified | low–medium: Korea holds SK Hynix and Samsung (AI memory) |

## Named scenarios and replay (models B, C, `stress.py`)

| ID | Assumption | Where | Value | Evidence | Status | Sensitivity |
|---|---|---|---|---|---|---|
| S1 | **(profile)** Candidates are built per account from real funds | `profiles/<name>/accounts.toml` `[funds]`, `[location]` | per profile: real funds per account and asset class | Candidates use the funds each account can actually buy, not one total-market fund everywhere | verified | low–medium |
| S2 | Scenario shock sizes per sleeve | `data/scenarios.csv` | e.g. ai_unwind_severe: ai_core −65%, SCV −15%, bonds +10% | Every scenario now states `horizon_years`; cash earns T-bills (3.94%) compounded over it; short TIPS ≥ cash. New scenarios are calibrated to history: credit forks on the 21 credit-stress windows; `lost_decade_2000s` is the actual 2000–09 decade by sleeve proxy. The six original scenarios' stock and bond numbers were set before horizons existed and are still rough | new: sourced; original: unchecked | high: these are the scenario | Japan-style scenarios (added after the model freeze, as an extra lens): US equity sleeves follow Japan 1989–2003 / –2012. Market level = Nikkei price (−80% / −76%) + ~0.8%/yr dividends. Style spreads from Ken French Japan, in USD (roughly currency-neutral as *spreads*). Deflation regime for bonds and cash, and rest-of-world returns, are judgment.
| S3 | Private real estate falls 10–35% in crashes | `data/scenarios.csv` | −35% in a broad recession | REITs fell 65% in 2008. Leveraged syndicates can do worse, or go to zero | unchecked | high in dollars for real-estate-heavy profiles |
| S4 | Replay return series per block, with fallbacks back to 1926 | `blocks.REPLAY_SERIES` | Vanguard funds → Ken French (`FF_*`) → synthetic FRED (`SYN_*`); every replay row reports which series it used (`series_used`) | Fund histories via yfinance; Ken French; FRED | sourced | low after 1990; **medium for 1970s windows** (S15–S17) |
| S5 | Cash and stable value earn 3-month T-bills in the replay | `REPLAY_SERIES["CASH"] = ["SYN_TBILL"]` | FRED TB3MS | Was 0% until 2026-10-22 (understated cash-heavy portfolios, e.g. "Today") | verified | low |
| S6 | Replay excludes private real estate (liquid only) | `stress.replay` | — | Private marks don't trade | judgment | — |
| S7 | Replay is buy-and-hold within each episode (weights drift) | `stress.replay` | — | No rebalancing mid-crash | judgment | low–medium |
| S8 | Monthly data | `data/prices/*` | month-end | **Measured 2026-10-22** (docs/cross-checks.md): monthly max drawdown is ~3.6–4pp shallower than daily (2008–09: −50% month-end vs. −53.6% daily, testfol.io). Returns and calendar years match an independent tool within ~0.1–0.5pp | verified | low for returns; **drawdowns understated by ~3–4pp** |
| S9 | Bond proxy is the total bond fund (VBMFX) | `stress`, `frontier`, `reverse_stress` | VBMFX | Many 401(k)s hold an **intermediate-term** fund (e.g. VBILX): longer duration, fell ~13% in 2022 | sourced | low–medium |
| S10 | Lab IPO assumptions | `data/private_ai.csv`, `stress.US_FLOAT_CAP` | Anthropic $2T, OpenAI $1.4T, 20% float post-lockup; US float cap $72T; in total-market funds, not the S&P 500 for ≥12 months | Reported targets (unconfirmed); index rules from Vanguard, Mercer, Morningstar | sourced | low (direct exposure is a fraction of a percent for index investors) |
| S11 | **(profile)** Satellites (direct stocks, crypto, private RE) are held unchanged in every candidate | `profile.toml satellites` | — | The profile's decision | judgment | — |

| S12 | Short TIPS before 2012 replay as T-bills | `REPLAY_SERIES["TIPS_SHORT"]` | VTIP → SYN_TBILL | Little duration, so T-bills are the closer analog (was: intermediate TIPS) | judgment | low (variant not chosen) |
| S14 | **(profile)** Frozen accounts: nothing sold | `accounts.toml` `mode = "frozen"` | — | Typical reason: taxable with unknown or large embedded gains | judgment | medium: limits how close targets can get if frozen holdings exceed them |
| S15 | Synthetic Treasury series for bonds/TIPS before 1986 | `stress.replay_returns` `SYN_TSY` | 10-yr yield carry − 6 × Δyield | Correlation 0.66 with VBIIX monthly (1997–2026). No credit risk: overstates the 2008 hedge (+14.5% vs. +5%), understates 2022 (−10% vs. −13%) | sourced/judgment | medium for 1970s |
| S16 | International before 1990 replays as the **US market** | `REPLAY_SERIES["INTL"/"INTL_DEV"/"EM"]` → `FF_us_mkt` | — | No Ken French developed ex-US data before 1990. It was first left *flat*, which made intl a perfect 1970s hedge (caught by review of the output). EAFE fell about like the US in 1973–74 and beat it later in the decade: about right for the crash, conservative for the decade | judgment | medium for 1970s |
| S17 | TIPS before 1997 replay as nominal Treasuries (they didn't exist) | same | — | Understates TIPS's inflation protection in the 1970s windows | judgment | medium for stagflation episodes |

## Monte Carlo (model D, `montecarlo.py`)

| ID | Assumption | Where | Value | Evidence | Status | Sensitivity |
|---|---|---|---|---|---|---|
| M1 | Paths = circular block bootstrap of monthly history; all blocks and CPI drawn from the same months | `BLOCK_MONTHS`, `WINDOWS` | 12-month blocks; 1990-08…2026-09 and 1954-01…2026-09 | Keeps crash clustering and cross-asset co-movement. 1954– relies on stand-ins (S15–S17) | judgment | **high**: window choice moves 30-yr medians ~30% |
| M2 | Liquid portfolio only; private RE excluded | `candidate_inputs` | — | Illiquid, unmarked, can't be rebalanced or sold into a crash | judgment | — |
| M3 | Rebalancing at the block level, ignoring taxes and trading costs; satellites and the emergency fund aren't rebalanced | `simulate` | every 6 months; ±5pp, or 25% relative for targets under 20% | Common IPS bands. In reality only tax-advantaged accounts rebalance tax-free | judgment | low–medium |
| M4 | **(profile)** Contributions: named streams | `profile.toml [[contributions]]`, IRS limits in `data/contribution_limits.csv` | per profile: dollars per year or `max:<limit>`; IRS-max streams are inflation-indexed (the limits are indexed), fixed-dollar streams aren't unless `indexed = true` | The Monte Carlo invests contributions at target weights across the portfolio, not per account | judgment | high for 10–30 yr wealth; none for the risk metrics (M10) |
| M11 | Every portfolio, including "0 Today", gets the same contribution streams; Today's go to what it already holds (no rebalancing) | `run` (`contrib_blocks`) | — | Without this, Today got no contributions while candidates did, which made rebalancing look like it nearly doubled the money | verified | — |
| M10 | Risk metrics (drawdowns, time underwater, under-at-5y, 5-yr lag) use the time-weighted return index, excluding new money; wealth multiples include contributions | `simulate` (`twr`, `twr_real`), `summarize` | — | With large contributions flowing in, wealth-based drawdowns understated losses (in one profile, P(under at 5y) went 15% → 4%) | verified | — |
| M5 | **(profile)** Dip rule: deploy the reserve as a 60/40 US/intl index falls from its high; **never rebuilt in the simulation** | `profile.toml [dip_buying]`, `simulate` (`fired` never resets) | per profile (example: all at −25%) | An IPS may say to rebuild the reserve from new savings after a recovery; the Monte Carlo doesn't. That's conservative: it understates the rule's benefit in multi-crash paths | judgment | low for small reserves |
| M6 | Forecast re-centering: each block's history rescaled to a CMA geometric mean, over the forecast's own horizon only; history after | `recenter`, `cma_targets`, `cma_horizon_months` | median, 25th pct (10 yr); GMO (7 yr); BlackRock (10 yr) | Applying GMO's 7-yr call for 30 yrs made US stocks a permanent −5.6%/yr and Three-fund end at 0.36× real (caught in the first run) | judgment | high |
| M7 | Small cap / small value forecasts where firms don't publish them = that firm's US large + 1pp; also used in the median and 25th-pct runs | `SMALL_PREMIUM` | +1pp | Only GMO and RA publish small/SCV, and they disagree wildly (GMO SCV − large = +8.9pp). **Generous next to the real fund:** VISVX beat VTSMX by only +0.1%/yr over 1998–2026, and Ken French's paper small value beat VISVX by 1.3%/yr. Until 2026-10-23 a units bug applied it as +0.01pp in the median/25th-pct runs | judgment | **tested** (BlackRock run, a 90/10 SCV-tilted candidate): SCV tilt adds +3.0% to 10-yr median wealth at +1pp, +1.6% at 0, −1.2% at −2pp |
| M8 | "Lags the benchmark ≥10pp over any 5-yr window", searched over the first 10 years | `LAG_WINDOW_MONTHS`, `LAG_SEARCH_YEARS` | 60-mo windows, first 10 yrs | Over 30 years, ~300 overlapping windows trip it for almost any tilt | judgment | medium |
| M9 | Crypto is flat before 2017 (no history) | `block_history` | — | Tiny weight (<0.3% of liquid) | judgment | low |

## Robust frontier (model E, `robust_frontier.py`)

| ID | Assumption | Where | Value | Evidence | Status | Sensitivity |
|---|---|---|---|---|---|---|
| E1 | Objectives: 3-yr 5th-pct real wealth (the risk window) and 10-yr median real wealth | `evaluate` | — | A 3-year risk window plus growth (constants in `robust_frontier.py`) | judgment | — |
| E2 | Grid | `STOCK_SHARES`, `INTL_OF_STOCKS`, `SCV_OF_STOCKS`, `TIPS_OF_FIXED` | stocks 70–100%, intl 20–40%, SCV 0–25%, TIPS 0/50% of fixed | SCV sits at the 25% edge, so it was extended to 50% at 90% stocks: consensus forecasts peak at 20–25% and decline by 50%; history and GMO keep rising | judgment | medium |
| E3 | "Beaten" = better on both objectives in all 6 runs; margins under ~0.5% are noise | `dominators`, `TOLERANCE` | 1% | A seed change shrank one candidate's best dominator margin from 0.5% to 0.2% | judgment | — |

## Regret (model F, `regret.py`)

| ID | Assumption | Where | Value | Evidence | Status | Sensitivity |
|---|---|---|---|---|---|---|
| F4 | Regret = rolling 5-yr cumulative return vs. (a) the plain-index benchmark and (b) the same portfolio without SCV, constant mix, monthly rebalanced, 1954–2026 | `regret.py` | — | The SCV-alone comparison sizes the tilt; the benchmark comparison mixes in the intl/US/META tilts | judgment | pre-1990 intl = US (S16) affects (a), not (b) |
| F5 | **(profile)** Tolerable lag = the reverse-stress lag threshold | `profile.toml [failures] lag_vs_benchmark` | per profile (example: 10pp over 5 yrs) | The profile's stated threshold | judgment | **decides the SCV size** |

## Frontier (`frontier.py`, v1, superseded by E)

| ID | Assumption | Where | Value | Evidence | Status | Sensitivity |
|---|---|---|---|---|---|---|
| F1 | History window | `data/prices/monthly_adjclose.csv` | 2000-07 to 2026-10 | Limited by TIPS fund inception | sourced | **high**: history says 0% international |
| F2 | Forward returns | `frontier.FORWARD_GEO` (v1: Vanguard only); `data/cma.csv` (16 firms, used by the Monte Carlo and robust frontier) | v1: US 5.2%, intl 5.9%, bonds 4.3%, TIPS 4.2%. 16-firm medians: US large 6.1%, intl dev 7.1%, EM 7.5%, US bonds 4.7% | v1 = Vanguard midpoints. `data/cma.csv` mostly from the PortfolioLab compilation, cross-checked on primary sources for 5 firms. GMO converted from 7-yr real (+2.5% inflation, assumed). 13 of 14 firms favor intl dev over US large (median +1.1pp/yr; range −1.2 to +6.4). US large forecasts span −5.6% to 9.0%; bonds only 4.3–5.1%. Small value has 1 forecast (GMO) | sourced | **high**: the forecast choice drives the frontier |
| F3 | Covariances are constant | `frontier.load_inputs` | historical | Stock–bond correlation flipped positive in 2022 | judgment | medium |

## Reverse stress test (model G, `reverse_stress.py`)

| ID | Assumption | Where | Value | Evidence | Status | Sensitivity |
|---|---|---|---|---|---|---|
| R1 | One period, no path | whole model | — | No rebalancing, dip-buying, or recovery. Paths are the Monte Carlo's job | judgment | high for "lags market" (multi-year) |
| R2 | Factor draws: uniform ranges, **historically correlated** (Gaussian copula) | `FACTOR_RANGE`, `data/factor_corr.csv`, `DEFAULT_FACTORS` | ranges: equity −60%…+50%; ai −70%…+60%; rates −20%…+25%; credit −25%…+5%. Correlation 1997–2026: equity–credit 0.71, rates–credit 0.50, equity–rates 0.35, equity–ai 0.37 | **verified** (calibrate.py). Independent draws had been an unstated departure from history. Ranges remain judgment: wider than history on purpose; credit range is skewed toward stress. Breach shares are **not probabilities** | correlation verified; ranges judgment | **tested**: candidate ranking is identical under all 4 factor × loading combinations; cushions move up to ~11pp |
| R3 | Sleeve loadings on factors | `data/loadings_fitted.csv` (default) / `LOADINGS` (assumed) | fitted by OLS on 12-mo windows, ~1997–2026 (calibrate.py; R² 0.39–0.94) | **verified** for 8 sleeves with history (private RE fitted for reference only, R12). Findings vs. assumed: SCV falls when tech leads (ai −0.40, assumed 0): the hedge is *stronger* than assumed. SCV/EM/REITs are very credit-sensitive (+0.55/+1.14/+1.37): small value is *not* a credit-crisis hedge. Intermediate TIPS are rate-sensitive (−0.66 vs. −0.25). **AI sleeves, cash, crypto stay judgment** (no pre-AI history) | verified (8) / judgment (AI, crypto, private RE) | tested: candidate ranking identical in all 4 variants for liquid −40% and net worth −30%; cushion spread up to ~11pp | **Caveat for SCV-tilted portfolios:** their reverse-stress cushion *levels* depend on one coefficient, SCV's fitted ai loading (−0.405), applied far outside its fitted range in the mildest-breach worlds (e.g. ai +0.52 with SCV −81%; ⚠-flagged). Cross-check with the named scenarios and the replay, which don't use it.
| R4 | Sleeve-specific noise | `IDIO` | ±5pp uniform | Guess | unchecked | low |
| R5 | Shock bounds | `BOUNDS` | −85% … +100% | Guess | unchecked | low |
| R6 | Carry: bonds/TIPS earn the current 10-yr yield, cash and short TIPS the T-bill rate, for one horizon; stocks get no drift | `CARRY_SOURCE`, `current_carry()` | 4.99% / 3.94% (FRED, 2026-09) | Yields are known in advance; without carry the fitted model had bonds −5% in crashes vs. +3.8% in history. Now about 0% (still conservative: model crashes have harsher credit stress than history) | verified | medium |
| R10 | Factor proxies and their scaling | `calibrate.py` | equity = 60/40 US/intl; ai = Nasdaq-100 − S&P 500; rates = +10 × Δ10-yr yield; credit = −8 × ΔBaa spread | Proxies are imperfect: "ai" is tech/growth, and the 1999–2002 dot-com cycle dominates it | judgment | medium |
| R11 | **Bond response to credit stress: the two loadings disagree on its sign** | `data/loadings_fitted.csv` (default) vs. `LOADINGS["bonds"]`; named scenarios `ai_unwind_credit_fed_cuts` / `_sticky_inflation` | fitted credit **+0.40** (holding rates fixed, bonds fall) vs. assumed **−0.20** (they rise) | **Both right in their own terms.** The fitted value is a partial effect. In all 21 historical 12-mo windows with Baa spreads +1pp, intermediate bonds still *rose* (+5.3% avg, worst −3.5%) because yields fell ~1pp. So the real question is whether the Fed can cut. That's split into two named scenarios. **The stakes scale with the bond share:** under 1% of liquid for a 90/10, ~2% for 80/20–75/25 | resolved as a named fork | high for model levels; **low stakes for stock-heavy allocations** |
| R12 | Private real estate uses hand-set loadings, not a REIT fit | `calibrate.REFERENCE_ONLY` | equity 0.3, ai 0, rates −0.2, credit 0.8 | Public REITs (credit 1.37) are levered and marked daily. Private real estate here means housing deals valued at cost. The REIT fit pinned private_re at −85% in moderate worlds and cut net-worth cushions ~20pp | judgment | high in dollars for real-estate-heavy profiles; still unchecked against any private-RE data |
| R13 | Inflation is a 5th factor: realized CPI over the 12-month world, uniform on the **1997–2026 range** (−2%…+9%), copula-correlated (rates 0.46, others ≤0.11) | `FACTOR_RANGE["inflation"]`, `data/factor_corr.csv` | median world 3.5% (historical median 2.4%; expected 2.37%) | **Not widened like the other factors**: −2%…+12% made 5% the median world and put 43% of worlds above 6%. 1970s-style inflation lives in the named `stagflation` scenario. No sleeve loads on inflation directly: it acts through its correlation with rates, through TIPS accrual (R14), and through deflating all results to real terms | judgment (range) / verified (correlation) | **medium**: real-terms cushions are ~6pp worse than nominal |
| R14 | TIPS and short TIPS earn the inflation *surprise* (realized − expected), on top of carry | `INFLATION_LINKED`, `current_expected_inflation()` | expected = 10-yr breakeven 2.37% (FRED T10YIE, 2026-10) | Carry (10-yr yield) already includes expected inflation; only the surprise is extra | sourced | low |
