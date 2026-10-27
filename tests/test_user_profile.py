"""Profile loading, selection, and validation errors (each should name the file and problem)."""
import pytest

import user_profile as P
from conftest import ACCOUNTS_TOML, PROFILE_TOML, REAL_ROOT


def write_profile(tmp_path, profile=PROFILE_TOML, accounts=ACCOUNTS_TOML):
    d = tmp_path / "p"
    (d / "holdings").mkdir(parents=True)
    (d / "profile.toml").write_text(profile)
    (d / "accounts.toml").write_text(accounts)
    return d


def test_shipped_profiles_load_and_validate():
    for name in ["example"]:
        p = P.load(REAL_ROOT / "profiles" / name)
        assert p.candidates and p.failures and p.emergency_fund > 0, name


def test_allocatable_accounts_merge_and_order(tmp_path):
    acc = ACCOUNTS_TOML.replace('"401k" = { mode = "movable" }',
                                '"401k" = { mode = "movable" }\nrollover = { mode = "movable", merge_into = "401k" }')
    p = P.load(write_profile(tmp_path, accounts=acc))
    assert p.allocatable_accounts() == ["401k", "roth_ira", "taxable_new"]


@pytest.mark.parametrize("old,new,match", [
    ("emergency_fund = 100\n", "", "missing `emergency_fund`"),
    ("stocks_drawdown = 0.50\n", "", r"\[failures\] missing `stocks_drawdown`"),
    ("bonds = 0.15\n", "bonds = 0.25\n", "shares sum to 1.1000"),
    ("scv = 0.05\n", "scv = 0.05\ngold = 0.0\n", r"unknown classes \['gold'\]"),
    ('ips_target = "t1"', 'ips_target = "nope"', "ips_target 'nope' is not one of the candidates"),
    ("stocks_drawdown = 0.50\n", "stocks_drawdown = 0.50\nreal_drawdown = 0.3\n", r"unknown keys \['real_drawdown'\]"),
])
def test_profile_toml_errors(tmp_path, old, new, match):
    assert old in PROFILE_TOML
    with pytest.raises(P.ProfileError, match=match):
        P.load(write_profile(tmp_path, profile=PROFILE_TOML.replace(old, new)))


@pytest.mark.parametrize("old,new,match", [
    ('roth_ira = { mode = "movable" }', 'roth_ira = { mode = "sometimes" }', "needs mode"),
    ('bonds = ["401k", "roth_ira"]', 'bonds = ["401k", "taxable_new"]', r"sends 'bonds' to 'taxable_new'"),
    ('intl = [["VTMGX", 0.75], ["VEMAX", 0.25]]', 'intl = [["VTMGX", 0.75], ["VEMAX", 0.2]]', "shares must sum to 1"),
    ('fill_order = ["tips_short", ', 'fill_order = [', r"fill_order is missing \['tips_short'\]"),
    ('[funds.taxable_new]\nus = [["VTI", 1.0]]\n', '[funds.taxable_new]\n', r"\[funds.taxable_new\] needs a `us` fund"),
])
def test_accounts_toml_errors(tmp_path, old, new, match):
    assert old in ACCOUNTS_TOML
    with pytest.raises(P.ProfileError, match=match):
        P.load(write_profile(tmp_path, accounts=ACCOUNTS_TOML.replace(old, new)))


def test_selection_order(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "DEFAULT_FILE", tmp_path / ".portfolio-profile")
    monkeypatch.delenv(P.ENV_VAR, raising=False)
    assert P.default_name() == "example"
    (tmp_path / ".portfolio-profile").write_text("alex\n")
    assert P.default_name() == "alex"
    monkeypatch.setenv(P.ENV_VAR, "someone")
    assert P.default_name() == "someone"


def test_from_argv_strips_and_activates(monkeypatch):
    monkeypatch.setattr(P, "_current", None)
    rest = P.from_argv(["--factors", "independent", "--profile", "example", "-x"])
    assert rest == ["--factors", "independent", "-x"] and P.current().name == "example"
    rest = P.from_argv([f"--profile={REAL_ROOT / 'profiles/example'}"])
    assert rest == [] and P.current().name == "example"


def test_missing_profile_is_a_clear_error():
    with pytest.raises(P.ProfileError, match="no profile 'nobody'"):
        P.load("nobody")


CONTRIB = """
[[contributions]]
name = "401k"
account = "401k"
annual = "max:401k_employee"
years = 10

[[contributions]]
name = "fixed"
account = "roth_ira"
annual = 1200
"""


def test_contribution_streams_resolve_limits_and_years(tmp_path):
    p = P.load(write_profile(tmp_path, profile=PROFILE_TOML + CONTRIB))
    c = {x["name"]: x for x in p.contributions()}
    assert c["401k"]["monthly_usd"] == pytest.approx(24_500 / 12)  # 2026 official IRS limit
    assert c["401k"]["months"] == 120 and c["401k"]["indexed"] is True
    assert c["fixed"]["monthly_usd"] == 100 and c["fixed"]["months"] is None and c["fixed"]["indexed"] is False
    assert p.monthly_contribution == pytest.approx(24_500 / 12 + 100)


@pytest.mark.parametrize("old,new,match", [
    ('annual = "max:401k_employee"', 'annual = "max:401k_dream"', "max:<limit>"),
    ('account = "roth_ira"', 'account = "piggy_bank"', "unknown account 'piggy_bank'"),
    ("annual = 1200", "annual = -5", "is negative"),
])
def test_contribution_errors(tmp_path, old, new, match):
    with pytest.raises(P.ProfileError, match=match):
        P.load(write_profile(tmp_path, profile=PROFILE_TOML + CONTRIB.replace(old, new)))


def test_old_scalar_contribution_key_is_rejected(tmp_path):
    bad = PROFILE_TOML.replace('ips_target = "t1"', 'ips_target = "t1"\nmonthly_contribution = 100')
    with pytest.raises(P.ProfileError, match="replaced by"):
        P.load(write_profile(tmp_path, profile=bad))


def test_candidate_negative_share_is_rejected(tmp_path):
    # Shares still sum to 1, but shorts aren't modeled
    bad = PROFILE_TOML.replace("us = 0.55", "us = 0.85").replace("bonds = 0.15", "bonds = -0.15")
    with pytest.raises(P.ProfileError, match="negative share"):
        P.load(write_profile(tmp_path, profile=bad))


def test_candidate_reserve_usd_is_validated(tmp_path):
    d = write_profile(tmp_path, profile=PROFILE_TOML.replace("bonds = 0.15", 'bonds = 0.15\nreserve_usd = "everywhere"'))
    with pytest.raises(P.ProfileError, match='reserve_usd must be a dollar amount or "new_taxable"'):
        P.load(d)
    (d / "profile.toml").write_text(PROFILE_TOML.replace("bonds = 0.15", "bonds = 0.15\nreserve_usd = -100"))
    with pytest.raises(P.ProfileError, match="reserve_usd is negative"):
        P.load(d)


def test_contribution_limits_file():
    lim = P.contribution_limits()
    assert lim["401k_employee"] == 24_500 and lim["ira"] == 7_500 and lim["401k_total_415c"] == 72_000
    assert P.contribution_limits(2027)["401k_employee"] == 25_500  # projected
