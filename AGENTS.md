# AGENTS.md

Orientation for Claude Code and other AI agents working in this repo. (`CLAUDE.md` is a
symlink to this file.)

## What this is

A scenario-modeling toolkit people use to measure AI-concentration risk across all their
accounts and choose a rebalancing allocation. Read `README.md` first: layout, profiles, the
pipeline table, and how to read the reverse stress test. Methodology is in
`docs/modeling-plan.md`, every assumption in `docs/assumptions.md`.

The engine (`scripts/`) and shared market data (`data/`) are the same for everyone. Each
person's inputs are a **profile** in `profiles/<name>/` (gitignored, except `example/`).

## Ground rules

- **Never commit or push personal data.** Profiles other than `example/` and all of
  `output/` are gitignored; keep it that way. Before any commit, run `git status` and make
  sure nothing under `profiles/<someone>/` or `output/` is staged. Never `git add -f` them.
- **Never write account numbers** into holdings files. Redact them from broker exports.
- **Not financial advice.** Present results as model output with their assumptions, not as
  recommendations. The user decides. When a result depends on a known limitation (README,
  "Known limitations"), say so.
- **Respect the limitations.** If the user's situation hits one (mostly-taxable money,
  private employer equity, income tied to AI), tell them plainly what the model can't see,
  pick the least-wrong representation, and write it down in their `decisions.md`.
- **Assumptions are tracked.** If you change a shared assumption (anything in `data/` or a
  constant in `scripts/`), update `docs/assumptions.md` in the same change.

## Working with a user

- New user, or a "help me set up": run the `/setup-profile` skill
  (`.claude/skills/setup-profile/SKILL.md`).
- Which profile is active: `cat .portfolio-profile` (falls back to `example`). Every script
  takes `--profile <name>` and prints the profile's inputs first.
- Typical session: refresh data if stale → `lookthrough.py` → `allocate.py` → `compare.py`
  → dig into whichever model drives the difference between candidates.
- Help the user record decisions in `profiles/<name>/decisions.md` and their policy in
  `profiles/<name>/ips.md` (templates in `profiles/example/`).

## Commands

```sh
uv run --with pandas --with pytest --with numpy --with requests python -m pytest   # all tests, ~10s
uv run --with pandas --with pytest --with numpy --with requests python -m pytest -m network  # live downloads
uv run --with pandas python scripts/lookthrough.py [--profile NAME]
uv run --with pandas python scripts/allocate.py
uv run --with pandas python scripts/stress.py
uv run --with pandas --with numpy python scripts/reverse_stress.py
uv run --with pandas --with numpy python scripts/montecarlo.py
uv run --with pandas --with numpy python scripts/compare.py         # one table, every model
uv run --with yfinance --with pandas python scripts/fetch_prices.py  # refresh prices (+ profile's stocks)
uv run --with pandas --with requests python scripts/fetch_fund_holdings.py
```

Validate a profile without running a model:

```sh
uv run --with pandas python -c "import sys; sys.path.insert(0, 'scripts'); import user_profile as P; print(P.load('NAME').summary())"
```

## Code conventions

- `scripts/blocks.py` is the shared vocabulary (blocks, sleeves, replay series, proxies);
  `scripts/user_profile.py` loads and validates profiles with errors that name the file.
- Tests: unit tests run on a fake repo built in `tests/conftest.py`; integration tests run
  the real `data/` with `profiles/example`. New features need tests on one of those, never
  on a real person's profile.
- Match the surrounding style: short docstrings that say what question a function answers,
  comments that cite assumption IDs (e.g. `assumption S12`).
