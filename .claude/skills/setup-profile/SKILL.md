---
name: setup-profile
description: Interview a new user about their accounts, holdings, goals and risk tolerance, and build their profile in profiles/<name>/ from their brokerage exports. Use when someone new wants to start, says "set me up", or asks to create or update their profile.
---

# Set up a profile

Goal: a validated `profiles/<name>/` that runs through every model, built from a short
interview and the user's own account exports. Read `README.md` (Profiles, Known limitations)
and `profiles/example/` first; the example is the format reference for every file.

Keep it conversational: a few questions at a time, plain language, no jargon without a
one-line explanation. The user may not be a finance professional. Remind them once, early,
that this is a modeling tool, not advice, and that their profile stays on their machine
(gitignored).

## 1. Name and scaffold

- Ask what to call the profile (a first name or handle; lowercase, no spaces).
- `cp -r profiles/example profiles/<name>`, delete the example holdings file, and
  `echo <name> > .portfolio-profile`.
- Confirm with `git status` that `profiles/<name>/` does not show up (it's gitignored).

## 2. Accounts and holdings

Ask for every account: workplace plans (401(k), 403(b), 457), IRAs (traditional/Roth),
HSA, taxable brokerage, bank cash, and anything else (real estate, private deals, crypto,
employer equity). Then ask them to drop exports (CSV or PDF statements) in
`profiles/<name>/holdings/`, or to paste positions.

For each position, write a row in the `profiles/_holdings_template.csv` format:
- **Remove account numbers.** Name accounts by role (`workplace_401k`, `roth_ira`,
  `brokerage`, `savings`), not by number.
- Funds with a ticker in `data/fund_catalog.csv` classify automatically. Workplace funds
  without tickers must match a `fund_name` in `plan_menu.csv`.
- A fund that isn't in either: find its index and expense ratio from the fund company's
  page, add a catalog row with the right block and a source URL, and tell the user (they
  may want to PR it upstream). If no block fits (sector, active, target-date funds), say
  so, agree on the closest block, and note it in `decisions.md`. Never invent a block name:
  any value not in `scripts/blocks.py` is treated as a directly held stock's ticker.
- Individual stocks: `block = STOCK`, ticker set. Bank cash: `CASH`. Real estate:
  `PRIVATE_RE` at cost. Crypto: `CRYPTO`.

## 3. Employer plan menu (if any)

Ask for the plan's fee disclosure (the "404a-5 notice") or the fund list. Fill
`plan_menu.csv`: fund name exactly as on statements, index vs. active, asset class,
`model_block`, expense ratio. Active funds get no block.

## 4. Account modes and fund choices (`accounts.toml`)

For each account decide, with the user:
- `movable`: can be fully re-allocated without tax (401(k), IRA, HSA).
- `frozen`: held as-is. The usual choice for taxable accounts with large gains. Ask about
  embedded gains; if they're small, `movable` may be honest.
- `bank`: cash; the emergency fund comes from here first, the rest is new money.

Then the funds each account would actually use per class (`[funds.<account>]`) and the
asset-location order (`[location]`: bonds/TIPS in tax-advantaged first, etc.).

**Mostly-taxable users:** explain that the engine can't do partial, tax-aware selling yet.
`frozen` leaves little to rebalance, so the decision is mostly about new money; `movable`
overstates flexibility. Record the choice and why in `decisions.md`.

## 5. Goals, risk and failure (`profile.toml`)

Ask, and translate the answers into settings:
- Horizon for this decision (`horizon_years`), emergency fund (`emergency_fund`).
- "What drop would you hold through without selling?" and "What would count as failure?"
  → `[failures]` (liquid, net worth, stocks-only drawdowns; lag behind the market).
- Holdings to keep unchanged regardless (`satellites`).
- Savings per year by account (`[[contributions]]`, `max:401k_employee` / `max:ira` for IRS
  maximums, employer match as dollars).
- Optional: an AI cap (`ai_cap_core_pct_stocks`) and a dip-buying rule (`[dip_buying]`).

**Job and income exposure:** ask where they work and how their income would fare in an AI
bust. The models don't capture human capital. If their pay or equity is tied to AI (big
tech, AI startups, dev tools), say so plainly. Suggest the levers that exist: a larger
emergency fund, stricter failure thresholds, a more defensive candidate. Public employer
stock or RSUs go in as `STOCK` and count toward AI exposure if they're in
`data/ai_basket.csv`. Private employer equity has no good representation; agree on one and
note it in `ips.md` §2.

## 6. Candidates

Start from the example's four shapes (Three-fund 80/20, SCV 80/20, Defensive 75/25,
Aggressive 90/10) and adjust to their stated risk. 3–6 candidates is plenty. Shares must sum
to 1 across `us, intl, scv, bonds, tips, tips_short, reserve`.

## 7. Validate and first run

```sh
uv run --with pandas python -c "import sys; sys.path.insert(0, 'scripts'); import user_profile as P; print(P.load('<name>').summary())"
uv run --with pandas python scripts/lookthrough.py
uv run --with pandas python scripts/allocate.py
uv run --with pandas --with numpy python scripts/compare.py
```

Fix any error (they name the file and the problem). Check that the look-through total
matches the sum of their statements. If their direct stocks have no price history, run
`fetch_prices.py`.

Walk them through the look-through first (where their AI exposure actually comes from),
then the comparison table. Explain what each row means before what it implies.

## 8. Write it down

Start `profiles/<name>/decisions.md` (template in `profiles/example/`) with today's scope,
the account modes, any limitation workarounds, and the candidates. Offer to draft `ips.md`
once they've picked a target.
