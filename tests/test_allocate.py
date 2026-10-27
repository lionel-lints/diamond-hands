"""Allocator on the fake repo. Accounts: taxable VTI 1000 / VXUS 500 / META 200 (frozen),
401(k) 400 and Roth 100 (movable), bank 300 (emergency fund 100 -> 200 new taxable money),
real estate 1000 and crypto 50 (satellites). Pool = 1000 + 500 + 400 + 100 + 200 = 2200."""
import pytest

import allocate as A
from conftest import holdings_total


@pytest.fixture
def state(fake_root):
    return A.account_state()


def test_account_state_capacities(state):
    assert state["capacity"] == {"401k": 400, "roth_ira": 100, "taxable_new": 200}
    assert state["emergency_fund"] == 100
    assert state["fixed_blocks"].to_dict() == {"CRYPTO": 50, "INTL": 500, "META": 200, "PRIVATE_RE": 1000, "TOTAL_US": 1000}


def test_underfunded_emergency_fund_warns(fake_root, capsys):
    """Bank cash below the emergency fund must warn, not silently shrink the fund."""
    import user_profile as P
    prof = fake_root / "profiles/test"
    p = (prof / "profile.toml")
    p.write_text(p.read_text().replace("emergency_fund = 100", "emergency_fund = 5000"))
    state = A.account_state(profile=P.load(prof))
    assert state["emergency_fund"] == 300  # all there is; the gap is reported, not hidden
    assert "below the $5,000 emergency fund" in capsys.readouterr().out


def test_feasible_target_is_hit_and_located(state):
    # want: bonds 330, scv 110, intl 550 (500 frozen), us 1210 (1000 frozen)
    fills, summary = A.allocate({"us": .55, "intl": .25, "scv": .05, "bonds": .15}, state)
    assert (summary.gap_usd.abs() < 1e-6).all()
    assert fills.usd.sum() == pytest.approx(700)  # every movable dollar placed exactly once
    by = fills.groupby(["account", "class"]).usd.sum()
    assert by["401k", "bonds"] == pytest.approx(330)       # bonds fill the 401(k) first...
    assert by["roth_ira", "scv"] == pytest.approx(100)      # ...SCV prefers the Roth, spills to 401(k)...
    assert by["401k", "scv"] == pytest.approx(10)
    assert by["taxable_new", "intl"] == pytest.approx(50)   # ...intl prefers taxable new money
    assert summary.target_usd.sum() == pytest.approx(2200)


def test_frozen_overweight_is_reported_not_sold(state):
    fills, summary = A.allocate({"us": .5, "intl": .2, "bonds": .3}, state)
    # Taxable VXUS (500) already exceeds the 440 intl target; nothing in taxable is sold
    assert summary.gap_usd["intl"] == pytest.approx(60)
    # Bonds overflow from the 401(k) into the Roth (500 of capacity) -> 160 short; leftover goes to US
    assert summary.gap_usd["bonds"] == pytest.approx(-160)
    assert summary.gap_usd["us"] == pytest.approx(100)
    assert fills.usd.sum() == pytest.approx(700)


def test_reserve_new_taxable_keeps_cash_where_it_is(state):
    fills, summary = A.allocate({"us": .6, "intl": .3, "bonds": .1, "reserve_usd": "new_taxable"}, state)
    res = fills[fills["class"] == "reserve"]
    assert res.account.tolist() == ["taxable_new"] and res.usd.sum() == pytest.approx(200)
    assert summary.target_usd["reserve"] == pytest.approx(200)


def test_short_tips_only_in_roth(state):
    fills, _ = A.allocate({"us": .6, "intl": .3, "tips_short": .1}, state)
    assert fills[fills["class"] == "tips_short"].account.unique().tolist() == ["roth_ira"]


def test_to_blocks_conserves_net_worth(state, fake_root):
    fills, _ = A.allocate({"us": .55, "intl": .25, "scv": .05, "bonds": .15}, state)
    b = A.to_blocks(fills, state)
    assert b.sum() == pytest.approx(holdings_total())
    assert b["CASH"] == pytest.approx(100)  # only the emergency fund stays cash


def test_rejects_targets_not_summing_to_one(state):
    with pytest.raises(AssertionError, match="sum to 1"):
        A.allocate({"us": .5, "intl": .2}, state)


def test_profile_candidates_are_valid(fake_root):
    import user_profile as P
    for name, t in P.current().candidates.items():
        shares = {k: v for k, v in t.items() if k != "reserve_usd"}
        assert set(shares) <= set(A.CLASSES), name
        assert sum(shares.values()) == pytest.approx(1.0), name


def test_every_account_fund_resolves_to_a_known_block(fake_root):
    import blocks as B
    import user_profile as P
    for acct, classes in P.current().account_funds().items():
        for cls, funds in classes.items():
            for fund, _ in funds:
                assert A.fund_block(fund) in B.KNOWN_BLOCKS, (acct, cls, fund)


def test_unknown_fund_in_accounts_is_a_clear_error(fake_root):
    import user_profile as P
    with pytest.raises(P.ProfileError, match="NOTAFUND"):
        A.fund_block("NOTAFUND")
