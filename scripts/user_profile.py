"""The active user profile: one person's inputs, kept apart from the shared engine and data.

A profile is a folder (profiles/<name>/, or any path) with:
  profile.toml    personal settings: horizon, emergency fund, satellites, failure
                  thresholds, AI cap, candidate allocations, the chosen IPS target
  accounts.toml   each account's mode, the funds it can use per asset class, and the
                  asset-location order
  holdings/*.csv  holdings exports (files starting with "_" are ignored)
  plan_menu.csv   optional employer-plan menu (fund_name, model_block, ...)
  ai_basket.csv   optional; falls back to the shared data/ai_basket.csv

Which profile: `--profile <name|path>` on any script, else $PORTFOLIO_PROFILE, else the
name in .portfolio-profile at the repo root, else "example".
Everything is validated on load; errors name the file and the problem.
"""
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = ROOT / "profiles"
OUTPUT_ROOT = ROOT / "output"
DEFAULT_FILE = ROOT / ".portfolio-profile"
ENV_VAR = "PORTFOLIO_PROFILE"

CLASSES = ["us", "intl", "scv", "bonds", "tips", "tips_short", "reserve"]
ACCOUNT_MODES = {"movable", "frozen", "bank"}
FAILURE_KEYS = ["liquid_drawdown", "networth_drawdown", "lag_vs_benchmark", "stocks_drawdown"]
OPTIONAL_FAILURE_KEYS = ["real_networth_drawdown"]


class ProfileError(ValueError):
    pass


@dataclass
class Profile:
    root: Path
    settings: dict
    accounts_cfg: dict

    @property
    def name(self):
        return self.root.name

    @property
    def display_name(self):
        return self.settings.get("name", self.name)

    @property
    def holdings_dir(self):
        return self.root / "holdings"

    @property
    def plan_menu_path(self):
        p = self.root / "plan_menu.csv"
        return p if p.exists() else None

    @property
    def ai_basket_path(self):
        p = self.root / "ai_basket.csv"
        return p if p.exists() else None

    @property
    def output_dir(self):
        return OUTPUT_ROOT / self.name

    # --- settings -------------------------------------------------------------------
    @property
    def horizon_years(self):
        return self.settings["horizon_years"]

    @property
    def emergency_fund(self):
        return self.settings["emergency_fund"]

    @property
    def satellites(self):
        return set(self.settings.get("satellites", []))

    @property
    def ai_cap(self):
        return self.settings.get("ai_cap_core_pct_stocks")

    @property
    def failures(self):
        return self.settings["failures"]

    @property
    def candidates(self):
        return self.settings.get("candidates", {})

    def contributions(self):
        """Resolved [[contributions]] streams: name, account, monthly_usd (this year's dollars),
        months (None = for the whole simulation), indexed (grows with inflation)."""
        limits = contribution_limits()
        out = []
        for c in self.settings.get("contributions", []):
            annual = c["annual"]
            if isinstance(annual, str):
                annual = limits[annual.split(":", 1)[1]]
            years = c.get("years")
            out.append({"name": c["name"], "account": c["account"], "monthly_usd": annual / 12,
                        "months": None if years is None else int(round(years * 12)),
                        "indexed": c.get("indexed", isinstance(c["annual"], str))})
        return out

    @property
    def monthly_contribution(self):
        """Total of all streams today, in today's dollars per month."""
        return sum(c["monthly_usd"] for c in self.contributions())

    @property
    def dip_buying(self):
        """[(drawdown trigger, fraction of the reserve)], or [] for no rule."""
        d = self.settings.get("dip_buying", {})
        return list(zip(d.get("triggers", []), d.get("fractions", [])))

    @property
    def ips_target(self):
        return self.settings.get("ips_target")

    # --- accounts -------------------------------------------------------------------
    @property
    def accounts(self):
        return self.accounts_cfg["accounts"]

    @property
    def new_money_account(self):
        return self.accounts_cfg.get("new_money", {}).get("name", "taxable_new")

    def account_funds(self):
        """{account: {class: [(fund, share), ...]}} for movable accounts and new money."""
        return {a: {c: [tuple(f) for f in funds] for c, funds in classes.items()}
                for a, classes in self.accounts_cfg.get("funds", {}).items()}

    @property
    def location_order(self):
        return self.accounts_cfg["location"]["order"]

    @property
    def fill_order(self):
        return self.accounts_cfg["location"]["fill_order"]

    def allocatable_accounts(self):
        """Movable accounts (after merges) in file order, then the new-money account."""
        out = []
        for a, cfg in self.accounts.items():
            if cfg["mode"] == "movable":
                target = cfg.get("merge_into", a)
                if target not in out:
                    out.append(target)
        return out + [self.new_money_account]

    def summary(self):
        f = self.failures
        return (f"Profile: {self.display_name} ({self.root}) | horizon {self.horizon_years} yr | "
                f"emergency fund ${self.emergency_fund:,.0f} | AI cap {self.ai_cap} | failures: "
                f"liquid -{f['liquid_drawdown']:.0%}, net worth -{f['networth_drawdown']:.0%}, "
                f"lag {f['lag_vs_benchmark']:.0%}, stocks -{f['stocks_drawdown']:.0%}"
                + (f", net worth -{f['real_networth_drawdown']:.0%} real" if "real_networth_drawdown" in f else ""))


def _read_toml(path):
    if not path.exists():
        raise ProfileError(f"missing {path}")
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def validate(p):
    s, where = p.settings, p.root / "profile.toml"
    for key in ["horizon_years", "emergency_fund", "failures"]:
        if key not in s:
            raise ProfileError(f"{where}: missing `{key}`")
    for key in FAILURE_KEYS:
        if key not in s["failures"]:
            raise ProfileError(f"{where}: [failures] missing `{key}`")
    unknown = set(s["failures"]) - set(FAILURE_KEYS) - set(OPTIONAL_FAILURE_KEYS)
    if unknown:
        raise ProfileError(f"{where}: [failures] has unknown keys {sorted(unknown)}")
    for name, target in p.candidates.items():
        bad = set(target) - set(CLASSES) - {"reserve_usd"}
        if bad:
            raise ProfileError(f"{where}: candidate {name!r} has unknown classes {sorted(bad)}; use {CLASSES}")
        shares = sum(v for k, v in target.items() if k != "reserve_usd")
        if abs(shares - 1) > 1e-9:
            raise ProfileError(f"{where}: candidate {name!r} shares sum to {shares:.4f}, not 1")
    limits = contribution_limits()
    known_accounts = set(p.accounts) | {p.new_money_account}
    for c in s.get("contributions", []):
        for key in ["name", "account", "annual"]:
            if key not in c:
                raise ProfileError(f"{where}: a [[contributions]] entry is missing `{key}`")
        if c["account"] not in known_accounts:
            raise ProfileError(f"{where}: contribution {c['name']!r} goes to unknown account {c['account']!r}")
        a = c["annual"]
        if isinstance(a, str):
            if not a.startswith("max:") or a.split(":", 1)[1] not in limits:
                raise ProfileError(f"{where}: contribution {c['name']!r}: annual must be a number or "
                                   f"'max:<limit>' with a limit in data/contribution_limits.csv {sorted(limits)}")
        elif a < 0:
            raise ProfileError(f"{where}: contribution {c['name']!r} is negative")
    if "monthly_contribution" in s:
        raise ProfileError(f"{where}: `monthly_contribution` was replaced by [[contributions]] streams")
    d = s.get("dip_buying", {})
    if len(d.get("triggers", [])) != len(d.get("fractions", [])):
        raise ProfileError(f"{where}: [dip_buying] needs one fraction per trigger")
    if sum(d.get("fractions", [])) > 1 + 1e-9:
        raise ProfileError(f"{where}: [dip_buying] fractions sum to more than 1")
    if p.ips_target and p.ips_target not in p.candidates:
        raise ProfileError(f"{where}: ips_target {p.ips_target!r} is not one of the candidates")

    where = p.root / "accounts.toml"
    for a, cfg in p.accounts.items():
        if cfg.get("mode") not in ACCOUNT_MODES:
            raise ProfileError(f"{where}: account {a!r} needs mode = one of {sorted(ACCOUNT_MODES)}")
        if "merge_into" in cfg and cfg["merge_into"] not in p.accounts:
            raise ProfileError(f"{where}: account {a!r} merges into unknown account {cfg['merge_into']!r}")
    funds = p.account_funds()
    allocatable = set(p.allocatable_accounts())
    for a, classes in funds.items():
        if a not in allocatable:
            raise ProfileError(f"{where}: [funds.{a}] is not a movable account or the new-money account")
        for c, fl in classes.items():
            if abs(sum(sh for _, sh in fl) - 1) > 1e-9:
                raise ProfileError(f"{where}: [funds.{a}] {c} shares must sum to 1")
    missing = set(CLASSES) - set(p.fill_order)
    if missing:
        raise ProfileError(f"{where}: [location] fill_order is missing {sorted(missing)}")
    for c, accts in p.location_order.items():
        for a in accts:
            if c not in funds.get(a, {}):
                raise ProfileError(f"{where}: [location.order] sends {c!r} to {a!r}, but [funds.{a}] has no {c!r} fund")


def resolve(name_or_path):
    p = Path(name_or_path)
    if p.is_dir():
        return p.resolve()
    if (PROFILES_DIR / str(name_or_path)).is_dir():
        return (PROFILES_DIR / str(name_or_path)).resolve()
    raise ProfileError(f"no profile {name_or_path!r} (looked for a folder at {p} and {PROFILES_DIR / str(name_or_path)})")


def load(name_or_path):
    root = resolve(name_or_path)
    p = Profile(root, _read_toml(root / "profile.toml"), _read_toml(root / "accounts.toml"))
    validate(p)
    return p


_current = None


def default_name():
    if os.environ.get(ENV_VAR):
        return os.environ[ENV_VAR]
    if DEFAULT_FILE.exists() and DEFAULT_FILE.read_text().strip():
        return DEFAULT_FILE.read_text().strip()
    return "example"


def use(name_or_path):
    global _current
    _current = load(name_or_path)
    return _current


def current():
    return _current if _current is not None else use(default_name())


def from_argv(argv):
    """Strip `--profile X` / `--profile=X` from argv, activate it, return the remaining args."""
    argv = list(argv)
    for i, a in enumerate(argv):
        if a == "--profile" and i + 1 < len(argv):
            use(argv[i + 1])
            return argv[:i] + argv[i + 2:]
        if a.startswith("--profile="):
            use(a.split("=", 1)[1])
            return argv[:i] + argv[i + 1:]
    current()
    return argv


def contribution_limits(year=None):
    """{limit: usd} from data/contribution_limits.csv for `year` (default: latest official year)."""
    d = pd.read_csv(ROOT / "data/contribution_limits.csv")
    if year is None:
        year = d[d.status == "official"].year.max()
    return dict(zip(d[d.year == year].limit, d[d.year == year].usd.astype(float)))


def plan_menu_blocks(p=None):
    """{fund_name: model_block} from the profile's plan menu (empty if none)."""
    p = p or current()
    if not p.plan_menu_path:
        return {}
    m = pd.read_csv(p.plan_menu_path, dtype=str).fillna("")
    return {r.fund_name: r.model_block for r in m.itertuples() if r.model_block}
