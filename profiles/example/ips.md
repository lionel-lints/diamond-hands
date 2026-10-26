# Investment Policy Statement: <target name>

A one-page promise to your future self, written while calm, so the plan survives a crash.
Copy this file to `profiles/<you>/ips.md` and fill it in. Section numbers matter: the
scripts refer to §5 (rebalancing) and §9 (review triggers).

Adopted: <date> · Review by: <date, also `review_by` in profile.toml>

## 1. Purpose and scope

- What this money is for, and when you'll need it.
- Which accounts are in scope (yours only? a partner's?). Anything deliberately out of scope
  (a house purchase, private deals, employer equity).
- Emergency fund: $<amount>, never invested (`emergency_fund` in profile.toml).

## 2. Risk tolerance (stated)

- The largest drop you'd hold through without selling: <e.g. −30%>.
- Your failure definitions (`[failures]` in profile.toml) and why each number.
- How your income is exposed to the same risks as your portfolio (e.g. working in tech
  during an AI unwind). The models don't capture this; say how you're accounting for it.

## 3. Target allocation

| Asset class | Target | Funds (per account) |
|---|---|---|
| US stocks | | |
| International stocks | | |
| US small value | | |
| Bonds | | |
| TIPS | | |

Why this target over the other candidates (cite `output/<you>/comparison.csv`).

## 4. AI concentration cap

- Core AI basket ≤ <x>% of stocks (`ai_cap_core_pct_stocks`), and what you'll do if it's breached.
- Direct holdings (satellites) and why you keep them.

## 5. Rebalancing, contributions, and dip-buying

- Bands: rebalance when a class drifts ±5pp, or 25% of target for targets under 20%.
- Where new contributions go.
- Dip-buying rule (`[dip_buying]`): triggers, fractions, and how the reserve is rebuilt.

## 6. If I'm wrong and AI keeps ripping (pre-committed)

- How far behind the plain index you'll tolerate (`lag_vs_benchmark`) before reconsidering.
- What you will NOT do (e.g. chase after a 50% run-up).

## 7. Account-by-account implementation

| Account | Mode | Trades |
|---|---|---|
| | | |

(From `output/<you>/allocations.csv`.)

## 8. What the models say

Key numbers for the chosen target vs. the runners-up: worst scenario, replay troughs,
reverse-stress cushion, Monte Carlo 5th percentile, worst 5-yr regret.

## 9. Review triggers and open items

- Scheduled full review by <date>.
- Indicators (`indicators.py`, `[review_indicators]`): baseline and the level that would
  justify shrinking the hedges.
- Life events that force a review (job change, home purchase, inheritance, ...).
