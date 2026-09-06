"""A research call must ask for the allowance its depth actually needs.

Allowances POOL rather than isolate — deliberately, because the pooled ceiling is what
protects the subscription from N concurrent runs. That makes the SIZE of each request the
thing that matters, and every tool used to ask for the same env default of 100.

Measured 2026-09-02: seven runs armed inside twelve minutes shared a ceiling of 171.
After the synthesis reserve that is ~22 calls each, and at the then-current 8-9 sessions
per node each of those runs researched a SINGLE node of the twenty it was asked for. The
reports say so themselves: "answered 1 of 5 questions ... (time/budget cutoff)".
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tiers import DEFAULT_TIER, TIERS, resolve  # noqa: E402


def test_every_tier_declares_the_whole_preset():
    required = {"max_nodes", "max_depth", "max_breadth", "max_calls",
                "time_budget_s", "node_concurrency", "verify"}
    for name, preset in TIERS.items():
        missing = required - set(preset)
        assert not missing, f"tier {name!r} is missing {sorted(missing)}"


def test_the_tiers_are_ordered_by_what_they_cost():
    light, standard, deep = (TIERS[n] for n in ("light", "standard", "deep"))
    for key in ("max_nodes", "max_calls", "time_budget_s"):
        assert light[key] <= standard[key] <= deep[key], (
            f"{key} is not monotonic across the tiers: "
            f"{light[key]} / {standard[key]} / {deep[key]}")


def test_an_allowance_covers_what_that_depth_measurably_costs():
    """Sized from the live by-site breakdown, not chosen.

    fixed 2 (classify + choose_agent) + 1.33/node + ~3 verify, with the merge ceiling
    taking MERGE_CALL_PCT (25%) of the allowance on top.
    """
    for name, preset in TIERS.items():
        need = (2 + 1.33 * preset["max_nodes"] + (3 if preset["verify"] else 0)) / 0.75
        assert preset["max_calls"] >= need, (
            f"tier {name!r} asks for {preset['max_calls']} sessions but its own "
            f"{preset['max_nodes']} node(s) need about {need:.0f}. Under-asking does not "
            f"save anything — the run just stops early and reports a partial answer")


def test_no_tier_asks_for_the_old_flat_default():
    """The defect: every tool requested 100 regardless of what it was about to do."""
    for name, preset in TIERS.items():
        assert preset["max_calls"] < 100, (
            f"tier {name!r} still asks for {preset['max_calls']}; seven concurrent calls "
            f"at that size are what shared one ceiling and each got a single node")


def test_seven_concurrent_standard_runs_ask_for_less_than_one_old_pool():
    """The property the tiers exist for, in the shape it was measured in."""
    seven = 7 * TIERS["standard"]["max_calls"]
    assert seven < 7 * 100, "tiering did not reduce what concurrent calls request"
    assert seven <= 200, (
        f"seven concurrent standard runs still request {seven} sessions; the 2026-09-02 "
        f"incident pooled to 171 and starved every one of them")


@pytest.mark.parametrize("given,expected", [
    (None, DEFAULT_TIER), ("", DEFAULT_TIER), ("  DEEP  ", "deep"),
    ("Light", "light"), ("standard", "standard"),
])
def test_resolve_normalises_what_a_caller_writes(given, expected):
    assert resolve(given)[0] == expected


def test_an_unknown_depth_falls_back_instead_of_raising():
    """A research call is not worth failing over a typo in a depth label."""
    name, preset = resolve("thorough")
    assert name == DEFAULT_TIER and preset == TIERS[DEFAULT_TIER]


def test_resolve_hands_back_a_copy():
    """A caller that adjusts its preset must not edit the table for everyone else."""
    _, preset = resolve("deep")
    preset["max_calls"] = 1
    assert TIERS["deep"]["max_calls"] != 1
