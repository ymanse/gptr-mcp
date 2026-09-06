"""How much a research call is allowed to want.

Every LLM call on the ``claude_agent`` provider spawns a `claude` CLI subprocess that
registers as its own session on the subscription, and ``begin_agent_run`` grants each
tool call an allowance. Those allowances POOL rather than isolate — deliberately, because
the ceiling is what protects the account from N concurrent runs.

Which makes the size of the request the thing that matters, and it was wrong. Every tool
asked for the env default of 100 regardless of what it was about to do. Measured
2026-09-02, seven runs armed inside twelve minutes shared a ceiling of 171; after the
synthesis reserve that is ~22 calls each, and at the then-current 8-9 calls per node each
of those runs researched a SINGLE node of the twenty it was asked for. The reports say so
themselves: "answered 1 of 5 questions ... (time/budget cutoff)".

So a call declares its depth and gets the allowance that depth actually needs. Seven
concurrent `standard` runs now request 175 between them where they used to request 700.

The numbers are derived from measurement, not chosen:

  fixed per run   classify 1 + choose_agent 1                 = 2
  per node        answer 1 + a shared expansion call          ~ 1.33
  synthesis       the merge ceiling is a share of the allowance (MERGE_CALL_PCT, 25%)
  verification    the faithfulness audit                      ~ 3

  allowance >= (2 + 1.33*nodes + 3) / (1 - 0.25), rounded up with room for retries

A tier is a REQUEST, not a guarantee: the pooled ceiling can still be exhausted by a
sibling run, and the tree degrades to a shallower report rather than an error when it is.
"""
from __future__ import annotations

TIERS: dict[str, dict] = {
    # One question, answered from one round of search. What "look it up" means.
    "light": {
        "max_nodes": 1,
        "max_depth": 0,
        "max_breadth": 0,
        "max_calls": 12,
        "time_budget_s": 90,
        "node_concurrency": 1,
        "verify": False,
    },
    # A question that has a few follow-ups, and no more. The default.
    "standard": {
        "max_nodes": 5,
        "max_depth": 1,
        "max_breadth": 4,
        "max_calls": 25,
        "time_budget_s": 240,
        "node_concurrency": 3,
        "verify": True,
    },
    # A topic that genuinely branches. The expensive one; ask for it on purpose.
    "deep": {
        "max_nodes": 20,
        "max_depth": 3,
        "max_breadth": 4,
        "max_calls": 50,
        "time_budget_s": 600,
        "node_concurrency": 3,
        "verify": True,
    },
}

DEFAULT_TIER = "standard"


def resolve(depth: str | None) -> tuple[str, dict]:
    """Return (name, preset). An unknown name falls back to the default rather than
    raising: a research call is not worth failing over a typo in a depth label."""
    name = (depth or DEFAULT_TIER).strip().lower()
    if name not in TIERS:
        name = DEFAULT_TIER
    return name, dict(TIERS[name])
