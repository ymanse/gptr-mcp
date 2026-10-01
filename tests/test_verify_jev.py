"""`verify` is a jev citation check first, and the old Sonnet audit only when jev cannot run.

Measured 2026-10-01: the faithfulness audit spent a `claude` session per verified run
(agent_purpose "verify") on a question that is a judgment, not a generation: does the page
a claim cites back it up. TypeSafe's citation-check cookbook is that job at ~$0.03 per
1000 passages. These tests pin the properties that make the swap safe:

  - a jev run spends NO LLM session, and maps its verdicts onto the existing output
    contract (unsupported_claims / contradictions / overall_confidence / notes /
    evidence_chars / evidence_truncated) so callers and the skill's report do not change;
  - a verdict below AUTO_ACCEPT (0.8) is escalated to `needs_review`, never shipped as a
    finding the caller will act on;
  - ABSENCE IS NOT DISAGREEMENT: `says_nothing` and a never-obtained page become
    "not-in-excerpt"/review, never a contradiction;
  - jev off, failing, or with nothing to judge runs the OLD LLM audit, and a failure there
    still raises, because "0 unsupported claims" is what a clean audit looks like;
  - the two quote-matching bugs the live check found stay fixed.

Hermetic: jev and the LLM are both replaced; nothing here touches the network.
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import verification  # noqa: E402
from gpt_researcher.utils import jev  # noqa: E402
import gpt_researcher.utils.llm as llm  # noqa: E402

A = "https://a.example.com/post"
B = "https://b.example.com/doc"
PAGE_A = ("Home | Blog | Pricing\n\n" + "Unrelated filler sentence. " * 120 + "\n\n"
          "Our study found 27.67% of AI-authored PRs hit merge conflicts. Prompt "
          "instructions are best-effort model guidance, not authorization controls.")
PAGE_B = "Merge queues validate each branch against trunk plus everything queued ahead."
SOURCES = [{"url": A, "title": "a", "raw_content": PAGE_A},
           {"url": B, "title": "b", "raw_content": PAGE_B}]
CONTEXT = "\n".join([
    f"27.67% of AI-authored PRs hit merge conflicts. [Source: {A}]",
    f"Merge queues validate against trunk plus the queue ahead. [Source: {B}]",
    "Source: https://a.example.com/post\nTitle: a\nContent: raw evidence, not a claim",
])
LLM_AUDIT = {"unsupported_claims": [{"claim": "x", "reason": "not-in-excerpt: y"}],
             "contradictions": [], "overall_confidence": 0.6, "notes": "llm"}


class _Cfg:
    smart_llm_model = "sonnet"
    smart_llm_provider = "claude_agent"
    llm_kwargs = {}


class _Researcher:
    cfg = _Cfg()


@pytest.fixture
def llm_calls(monkeypatch):
    """Record every Sonnet audit call; answer with a valid audit unless told otherwise."""
    calls = {"n": 0, "reply": json.dumps(LLM_AUDIT)}

    async def fake(**kwargs):
        calls["n"] += 1
        return calls["reply"]

    monkeypatch.setattr(llm, "create_chat_completion", fake)
    return calls


@pytest.fixture
def judge(monkeypatch):
    """jev switched on, answering from `answers` keyed by claim substring."""
    state = {"answers": {}, "states": [], "stats": {"calls": 0, "errors": 0,
                                                    "cost": 0.0, "input_tokens": 0}}

    async def ask_many(states, questions):
        assert list(questions) == ["relation"]
        out = []
        for st in states:
            state["states"].append(st)
            ans = next((v for k, v in state["answers"].items() if k in st["claim"]),
                       ("supports", 0.95))
            if ans is None:
                state["stats"]["errors"] += 1
                out.append(None)
                continue
            state["stats"]["calls"] += 1
            state["stats"]["cost"] += 0.00005
            out.append({"relation": ans[0], "relation__confidence": ans[1]})
        return out

    monkeypatch.delenv("GPTR_MCP_VERIFY_ENGINE", raising=False)
    monkeypatch.setattr(jev, "enabled", lambda: True)
    monkeypatch.setattr(jev, "ask_many", ask_many)
    monkeypatch.setattr(jev, "stats", lambda: dict(state["stats"]))
    return state


def _audit(context=CONTEXT, sources=SOURCES):
    return asyncio.run(verification.audit_faithfulness(_Researcher(), "q", context, sources))


def test_a_jev_audit_spends_no_session_and_keeps_the_output_contract(judge, llm_calls):
    out = _audit()

    assert llm_calls["n"] == 0, "the jev path still spent a Sonnet session"
    assert out["audit_engine"] == "jev"
    for key in ("unsupported_claims", "contradictions", "overall_confidence", "notes",
                "evidence_chars", "evidence_truncated"):
        assert key in out, f"output contract lost {key!r}"
    assert out["unsupported_claims"] == [] and out["contradictions"] == []
    assert out["overall_confidence"] == 1.0
    # one call per tagged claim; the raw evidence block is not a claim
    assert [s["claim"] for s in judge["states"]] == [
        "27.67% of AI-authored PRs hit merge conflicts.",
        "Merge queues validate against trunk plus the queue ahead."]
    assert "27.67%" in judge["states"][0]["section"], (
        "the judge was not shown the part of the cited page that carries the figure")
    assert out["jev"]["calls"] == 2 and out["jev"]["cost"] > 0, (
        f"jev spend is not reported next to the result: {out['jev']}")


def test_an_uncertain_verdict_is_escalated_not_acted_on(judge, llm_calls):
    judge["answers"] = {"27.67%": ("contradicts", 0.4), "Merge queues": ("says_nothing", 0.5)}

    out = _audit()

    assert out["contradictions"] == [] and out["unsupported_claims"] == [], (
        "a verdict below AUTO_ACCEPT reached the lists callers act on")
    assert {r["tentative"] for r in out["needs_review"]} == {"contradicted", "unsupported"}
    assert out["overall_confidence"] == 0.0, (
        "a run the judge could confirm nothing in must not read as confident")


def test_a_confident_contradiction_ships_in_the_existing_shape(judge, llm_calls):
    judge["answers"] = {"27.67%": ("contradicts", 0.97)}

    out = _audit()

    assert len(out["contradictions"]) == 1
    c = out["contradictions"][0]
    assert set(c) == {"topic", "a", "b", "sources"} and c["sources"] == [A]
    assert "27.67%" in c["a"] and "27.67%" in c["b"]


def test_says_nothing_is_absence_not_disagreement(judge, llm_calls):
    judge["answers"] = {"Merge queues": ("says_nothing", 0.93)}

    out = _audit()

    assert out["contradictions"] == []
    [u] = out["unsupported_claims"]
    assert u["reason"].startswith("not-in-excerpt:"), u["reason"]
    # and the question itself must steer cut evidence away from `contradicts`
    crit = verification._JEV_QUESTIONS["relation"]["criteria"]
    assert set(crit) == {"supports", "contradicts", "says_nothing"}
    assert "stops before" in crit["says_nothing"] and "explicitly" in crit["contradicts"]


def test_a_page_never_obtained_is_a_gap_not_a_finding(judge, llm_calls):
    ctx = CONTEXT + "\nTeams cap at five agents. [Source: https://never.example.com/x]"

    out = _audit(context=ctx)

    assert not any("five agents" in u["claim"] for u in out["unsupported_claims"])
    assert any(r["tentative"] == "unverifiable" and "five agents" in r["claim"]
               for r in out["needs_review"]), "the unchecked claim was silently dropped"
    assert out["overall_confidence"] < 1.0


def test_a_misquote_is_caught_before_any_model_call(judge, llm_calls):
    ctx = f'It says "agents must never ever merge into main directly" outright. [Source: {A}]'

    out = _audit(context=ctx + "\n" + CONTEXT)

    assert out["audit_engine"] == "jev"
    assert any(u["reason"].startswith("quote-not-in-source:") for u in out["unsupported_claims"])
    assert not any("never ever" in s["claim"] for s in judge["states"]), (
        "a quote absent from the page still cost a judge call")


def test_a_comma_inside_the_quote_marks_still_matches(judge, llm_calls):
    """Live 2026-10-01: this quote is on the page verbatim; the comma is US style."""
    ctx = (f'Docs note "prompt instructions are best-effort model guidance, not '
           f'authorization controls," in so many words. [Source: {A}]')

    out = _audit(context=ctx + "\n" + CONTEXT)

    assert out["audit_engine"] == "jev"
    assert out["unsupported_claims"] == [], out["unsupported_claims"]
    assert any("best-effort" in s["claim"] for s in judge["states"]), (
        "a verbatim quote was ruled not-in-source instead of reaching the judge")


def test_short_quotes_do_not_pair_into_a_fake_quote(judge, llm_calls):
    """Live 2026-10-01: the text BETWEEN "contains" and "invokes" was matched as a quote."""
    ctx = (f'Results use only "contains" edges (though "invokes" edges are shown to be '
           f'stronger in every language per Table 3) and "x" too. [Source: {A}]')

    out = _audit(context=ctx + "\n" + CONTEXT)

    assert out["audit_engine"] == "jev"
    assert not any(u["reason"].startswith("quote-not-in-source:")
                   for u in out["unsupported_claims"]), out["unsupported_claims"]
    assert any("Table 3" in s["claim"] for s in judge["states"]), (
        "the gap between two short quotes was treated as a quote")


@pytest.mark.parametrize("setup, reason_part", [
    ("disabled", "not configured"),
    ("all_fail", "judged none"),
    ("half_fail", "failed on"),
    ("no_claims", "no [Source"),
    ("forced_llm", "GPTR_MCP_VERIFY_ENGINE"),
    # quote misses alone are a string check, not an audit of the run
    ("only_quote_misses", "judged none"),
])
def test_when_jev_cannot_audit_the_old_llm_audit_runs(judge, llm_calls, monkeypatch,
                                                      setup, reason_part):
    ctx = CONTEXT
    if setup == "disabled":
        monkeypatch.setattr(jev, "enabled", lambda: False)
    elif setup == "all_fail":
        judge["answers"] = {"27.67%": None, "Merge queues": None}
    elif setup == "half_fail":
        judge["answers"] = {"27.67%": None}
    elif setup == "no_claims":
        ctx = "Source: https://a.example.com\nTitle: t\nContent: raw evidence only"
    elif setup == "forced_llm":
        monkeypatch.setenv("GPTR_MCP_VERIFY_ENGINE", "llm")
    elif setup == "only_quote_misses":
        ctx = f'It says "agents must never ever merge into main directly" outright. [Source: {A}]'

    out = _audit(context=ctx)

    assert llm_calls["n"] == 1, f"{setup}: the LLM audit did not run in jev's place"
    assert out["audit_engine"] == "llm" and reason_part in out["jev_fallback"], out
    assert out["unsupported_claims"] == LLM_AUDIT["unsupported_claims"]


def test_a_failed_fallback_still_fails_loud_and_tiering_still_ships(judge, llm_calls,
                                                                    monkeypatch):
    monkeypatch.setattr(jev, "enabled", lambda: False)
    llm_calls["reply"] = "I could not audit this."

    with pytest.raises(Exception):
        _audit()
    out = asyncio.run(verification.verify_research(_Researcher(), "q", CONTEXT, SOURCES))

    assert "unsupported_claims" not in out, "a failed audit shipped as a clean one"
    assert out["tier_summary"]["L3"] == 2, "tiering must ship even when the audit fails"
    assert "jev path not taken" in out["audit_error"], (
        f"audit_error does not say why jev did not run: {out['audit_error']!r}")


# ── A jev path that RAISES falls back too; it does not skip the audit ────────────────
# Review 2026-10-01: the endpoint answered HTTP 200 with a malformed body, the real client
# (or this module reading its answer) raised, the exception escaped audit_faithfulness,
# and verify_research recorded audit_error with llm_calls=0. The Sonnet audit that ran
# before jev existed was skipped outright -- the opposite of fail open. Reproduced with
# the REAL jev client and only `_post` patched, so the client's own parsing is in the path.
@pytest.fixture
def live_client(monkeypatch):
    """The real jev.ask/ask_many/stats, enabled, with the HTTP POST replaced."""
    monkeypatch.delenv("GPTR_MCP_VERIFY_ENGINE", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(jev, "MODEL", "typesafe/jev-test")
    monkeypatch.setattr(jev, "MAX_RETRIES", 0)
    monkeypatch.setattr(jev, "_retry_after", 0.0)
    monkeypatch.setattr(jev, "_consecutive_failures", 0)
    monkeypatch.setattr(jev, "_stats", {"calls": 0, "errors": 0, "cost": 0.0,
                                        "input_tokens": 0})
    body = {}
    monkeypatch.setattr(jev, "_post", lambda _b, _h: body["v"])
    return body


@pytest.mark.parametrize("reply", [
    {"answers": ["x"]},                                       # .items() on a list
    {"usage": "n/a"},                                         # _note_success on a str
    {"answers": {"relation": {"type": "choice", "choice": ["supports"]}}},  # unhashable
], ids=["answers_is_list", "usage_is_str", "choice_is_list"])
def test_a_malformed_200_from_jev_falls_back_to_the_llm_audit(live_client, llm_calls,
                                                              reply):
    live_client["v"] = reply

    out = asyncio.run(verification.verify_research(_Researcher(), "q", CONTEXT, SOURCES))

    assert "audit_error" not in out, f"the audit was skipped, not replaced: {out}"
    assert llm_calls["n"] == 1, "the LLM audit did not run in jev's place"
    assert out["audit_engine"] == "llm" and out["jev_fallback"], out
    assert out["unsupported_claims"] == LLM_AUDIT["unsupported_claims"]
    assert out["tier_summary"]["L3"] == 2


@pytest.mark.parametrize("boom", [AttributeError("'list' object has no attribute 'items'"),
                                  TypeError("unhashable type: 'list'"),
                                  RuntimeError("anything else")],
                         ids=lambda e: type(e).__name__)
def test_any_exception_on_the_jev_path_falls_back_and_says_so(judge, llm_calls,
                                                              monkeypatch, boom):
    async def ask_many(states, questions):
        raise boom

    monkeypatch.setattr(jev, "ask_many", ask_many)

    out = _audit()

    assert llm_calls["n"] == 1, "the LLM audit did not run in jev's place"
    assert out["audit_engine"] == "llm"
    # "the judge never ran" must stay distinguishable from "the judge found nothing"
    assert type(boom).__name__ in out["jev_fallback"] and str(boom) in out["jev_fallback"]


def test_a_raising_jev_path_and_a_failing_llm_audit_both_show_in_audit_error(
        judge, llm_calls, monkeypatch):
    async def ask_many(states, questions):
        raise AttributeError("'list' object has no attribute 'items'")

    monkeypatch.setattr(jev, "ask_many", ask_many)
    llm_calls["reply"] = "I could not audit this."

    out = asyncio.run(verification.verify_research(_Researcher(), "q", CONTEXT, SOURCES))

    assert "unsupported_claims" not in out, "a failed audit shipped as a clean one"
    assert llm_calls["n"] == 1
    assert "jev path raised AttributeError" in out["audit_error"], out["audit_error"]


def test_one_unhashable_label_among_good_ones_is_unjudged_not_fatal(judge, llm_calls,
                                                                     monkeypatch):
    """A single garbage answer is one judge failure, counted against VERIFY_JEV_MAX_FAIL_FRAC
    like a missing id or an unknown label -- not a crash that throws away every good verdict."""
    ctx = "\n".join([CONTEXT] + [f"Merge queues validate item {i}. [Source: {B}]"
                                 for i in range(4)])

    async def ask_many(states, questions):
        out = [{"relation": "supports", "relation__confidence": 0.95} for _ in states]
        out[0] = {"relation": ["supports"], "relation__confidence": 0.95}
        return out

    monkeypatch.setattr(jev, "ask_many", ask_many)

    out = _audit(context=ctx)

    assert llm_calls["n"] == 0 and out["audit_engine"] == "jev", out.get("jev_fallback")
    assert out["jev"]["verdicts"].get("unjudged") == 1
    assert any(r["tentative"] == "unjudged" for r in out["needs_review"])


# ── A confidence that is not a probability is no confidence at all ───────────────────
# Review 2026-10-01: the real client, only `_post` patched, every claim answered with
# `"confidence": true`. bool is an int in Python and True >= 0.8, so verify_research
# shipped audit_engine=jev, llm_calls=0, overall_confidence=1.0, verdicts {verified: 2},
# needs_review=[] -- a clean bill of health from a judge that returned no probability.
# With `contradicts`, both claims went into `contradictions`. 7 did the same.
_NOT_A_PROBABILITY = [True, False, 7, 1.5, -0.1, float("nan"), float("inf"), "0.95", None]


@pytest.mark.parametrize("label", ["supports", "contradicts"])
@pytest.mark.parametrize("conf", _NOT_A_PROBABILITY, ids=repr)
def test_a_non_probability_confidence_falls_back_instead_of_reading_as_a_verdict(
        live_client, llm_calls, label, conf):
    ans = {"type": "choice", "choice": label}
    if conf is not None:                                      # None: no confidence sent
        ans["confidence"] = conf
    live_client["v"] = {"answers": {"relation": ans},
                        "usage": {"cost": 0.00005, "input_tokens": 300}}

    out = asyncio.run(verification.verify_research(_Researcher(), "q", CONTEXT, SOURCES))

    assert "audit_error" not in out, out
    # every claim came back ungateable, so jev audited nothing and the LLM audit ran
    assert llm_calls["n"] == 1, f"the LLM audit did not run: {out.get('jev')}"
    assert out["audit_engine"] == "llm" and "judged none" in out["jev_fallback"], out
    assert out["contradictions"] == LLM_AUDIT["contradictions"], (
        "a verdict with no usable confidence reached the contradictions callers act on")
    assert out["overall_confidence"] == LLM_AUDIT["overall_confidence"]


def test_one_non_probability_confidence_is_unjudged_not_verified(judge, llm_calls):
    """One bad confidence among good ones is one judge failure: listed as unjudged, not
    counted as verified, and the good verdicts (int 1 and the 0.8 boundary included) stand."""
    ctx = "\n".join([CONTEXT] + [f"Merge queues validate item {i}. [Source: {B}]"
                                 for i in range(4)])
    judge["answers"] = {"27.67%": ("supports", True), "item 0": ("supports", 1),
                        "item 1": ("supports", 0.8)}

    out = _audit(context=ctx)

    assert llm_calls["n"] == 0 and out["audit_engine"] == "jev", out.get("jev_fallback")
    assert out["jev"]["verdicts"] == {"unjudged": 1, "verified": 5}
    assert out["jev"]["below_threshold"] == 0
    assert [r["tentative"] for r in out["needs_review"]] == ["unjudged"]
    assert "27.67%" in out["needs_review"][0]["claim"]
    assert out["overall_confidence"] == round(5 / 6, 2)


# ── s26: the audit checks the REPORT, with the context as its evidence ───────────────
# Live 2026-10-01, GPTR_MCP_VERIFY=true: s24's jev gate made the context raw passages in
# Source:/Title:/Content: blocks (0 `[Source:` tags on a real run), the audit ran BEFORE
# synthesis over that context, found nothing it could judge, spent a Sonnet session and
# shipped overall_confidence 0.9 with 0 unsupported / 0 contradictions. A scraped passage
# supports itself; the report written from the passages is what can drift.
STRIPE = "https://docs.stripe.com/api/idempotent_requests"
BLOG = "https://blog.example.com/outbox"
GATED_CONTEXT = "\n\n".join([
    f"Source: {STRIPE}\nTitle: Idempotent requests | Stripe\nContent: Stripe keeps "
    "idempotency keys for at least 24 hours, after which a reused key is treated as a new "
    "request. Keys may be up to 255 characters long.",
    f"Source: {BLOG}\nTitle: Outbox notes\nContent: The relay reads the outbox table and "
    "publishes each row to the broker at least once. " + "Filler about brokers. " * 60,
    "Source: https://other.example.com/x\nTitle: Unrelated\nContent: "
    + "Gardening tips for spring planting and soil care. " * 30,
])
REPORT = "\n".join([
    "> **Partial research.** This research stopped early because its time budget ran out.",
    "",
    "# Idempotency keys",
    "",
    "## Retention",
    "",
    f"Stripe keeps idempotency keys for at least 24 hours ([Stripe, 2024]({STRIPE})). "
    "What happens after that?",
    "",
    "This report synthesizes the findings below.",
    "",
    f"The outbox relay publishes each row to the broker at least once ([Blog]({BLOG})).",
    "",
    "| Property | Stripe |",
    "|---|---|",
    "| Max key length | 255 characters |",
    "",
    "## References",
    "",
    f"Stripe. (2024). Idempotent requests. {STRIPE}",
])
REPORT_CLAIMS = [
    "Stripe keeps idempotency keys for at least 24 hours.",
    "The outbox relay publishes each row to the broker at least once.",
    "Property: Max key length; Stripe: 255 characters",
]


def _report_audit(report=REPORT, context=GATED_CONTEXT, researcher=None):
    return asyncio.run(verification.audit_faithfulness(
        researcher or _Researcher(), "q", context, [], report=report))


def test_report_claims_are_split_deterministically():
    claims = verification.extract_report_claims(REPORT)

    # not claims: the banner, headings, a question, a sentence about the report itself,
    # the table's header and separator, everything under References
    assert [c["claim"] for c in claims] == REPORT_CLAIMS
    assert claims[0]["urls"] == [STRIPE] and claims[0]["kind"] == "number"
    assert claims[2]["kind"] == "table_row"


def test_a_quote_or_code_span_is_not_split_into_half_claims():
    """Measured on the 2026-10-01 reports: a two-sentence quotation and
    `INSERT ... ON CONFLICT` each came out as two fragments no passage could support."""
    claims = verification.extract_report_claims(
        'The guide warns: "the key can expire while the operation is still processing. A '
        'new request arrives and proceeds." It recommends `INSERT ... ON CONFLICT DO '
        'NOTHING` for the claim step in Postgres.')

    assert len(claims) == 2, [c["claim"] for c in claims]
    assert "A new request arrives" in claims[0]["claim"]
    assert "INSERT ... ON CONFLICT DO NOTHING" in claims[1]["claim"]


def test_a_gated_context_with_a_report_is_audited_by_jev_not_the_llm(judge, llm_calls):
    """The production defect: this exact shape fell back to the LLM with "no [Source:
    url]-tagged claims in the context to check"."""
    out = _report_audit()

    assert llm_calls["n"] == 0, f"spent a Sonnet session: {out.get('jev_fallback')}"
    assert out["audit_engine"] == "jev" and out["audit_target"] == "report"
    assert "jev_fallback" not in out
    assert [s["claim"] for s in judge["states"]] == REPORT_CLAIMS, (
        "the judge was not asked about the report's claims")
    assert out["jev"]["claims_by_scope"] == {"report": 3, "context": 0}
    for key in ("unsupported_claims", "contradictions", "overall_confidence", "notes",
                "evidence_chars", "evidence_truncated"):
        assert key in out, f"output contract lost {key!r}"


def test_each_report_claim_is_shown_the_passage_that_carries_its_figure(judge, llm_calls):
    """Evidence is RETRIEVED per claim from the context, not the context's head."""
    _report_audit()

    by_claim = {s["claim"]: s["section"] for s in judge["states"]}
    assert "at least 24 hours" in by_claim[REPORT_CLAIMS[0]]
    assert "publishes each row to the broker" in by_claim[REPORT_CLAIMS[1]]
    assert "255 characters" in by_claim[REPORT_CLAIMS[2]]
    assert not any(s.startswith("[Unrelated]") for s in by_claim.values()), (
        "an unrelated passage was ranked first")


def test_an_edited_number_in_the_report_lands_in_contradictions(judge, llm_calls):
    """Live 2026-10-01: six numbers edited in copies of two real reports all came back
    contradicts. Here: the plumbing from that verdict to the field callers act on."""
    bad = REPORT.replace("at least 24 hours", "at least 90 days")
    judge["answers"] = {"90 days": ("contradicts", 0.93)}

    out = _report_audit(report=bad)

    [c] = out["contradictions"]
    assert set(c) == {"topic", "a", "b", "sources"}, "the contradiction shape changed"
    assert "90 days" in c["a"] and "24 hours" in c["b"] and STRIPE in c["sources"]
    assert out["overall_confidence"] < 1.0


def test_a_report_quote_absent_from_the_context_needs_no_judge_call(judge, llm_calls):
    """Live 2026-10-01: 3 of 70 claims quoted words the context does not contain
    ("to avoid punishing a client" where the page says "would punish a client")."""
    bad = REPORT.replace("## References", 'Stripe says keys "must never be reused across '
                         'distinct payments".\n\n## References')

    out = _report_audit(report=bad)

    assert any(u["reason"].startswith("quote-not-in-source:") and "research context" in
               u["reason"] for u in out["unsupported_claims"])
    assert not any("never be reused" in s["claim"] for s in judge["states"])


class _Embeddings:
    """Deterministic stand-in: a claim and a passage sharing a marker word get close
    vectors. `fail` makes it raise like an unreachable embedder."""

    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.fail:
            raise ConnectionError("embedder unreachable")
        marks = ("24 hours", "broker", "255", "gardening")
        return [[1.0 if m in t.lower() else 0.0 for m in marks] + [0.01] for t in texts]


class _MemResearcher(_Researcher):
    def __init__(self, emb):
        self.memory = type("M", (), {"get_embeddings": lambda _s: emb})()


def test_retrieval_is_hybrid_with_embeddings_and_lexical_without(judge, llm_calls):
    emb = _Embeddings()
    out = _report_audit(researcher=_MemResearcher(emb))
    assert emb.calls == 1 and out["jev"]["retrieval"] == "hybrid"

    judge["states"].clear()
    out = _report_audit(researcher=_MemResearcher(_Embeddings(fail=True)))

    # fail open: a dead embedder degrades the ranking, it does not skip the audit
    assert out["audit_engine"] == "jev" and out["jev"]["retrieval"] == "lexical"
    assert llm_calls["n"] == 0 and len(judge["states"]) == 3


def test_old_shape_with_a_report_audits_both_hops(judge, llm_calls):
    """jev gate off: learnings carry [Source: url]. page -> learning and learning ->
    report are two separate places to drift, so both are checked."""
    out = asyncio.run(verification.audit_faithfulness(
        _Researcher(), "q", CONTEXT, SOURCES, report=REPORT))

    assert out["audit_engine"] == "jev" and out["audit_target"] == "report+context"
    assert out["jev"]["claims_by_scope"] == {"report": 3, "context": 2}
    asked = [s["claim"] for s in judge["states"]]
    assert REPORT_CLAIMS[0] in asked
    assert "27.67% of AI-authored PRs hit merge conflicts." in asked


def test_no_report_and_a_gated_context_falls_back_and_says_why(judge, llm_calls):
    """Synthesis failed: nothing jev can judge. The LLM audit runs (never skipped), and
    the response says the report was missing, so this is not read as a report audit."""
    out = _report_audit(report=None)

    assert llm_calls["n"] == 1
    assert out["audit_engine"] == "llm" and out["audit_target"] == "context"
    assert "no [Source" in out["jev_fallback"] and "no report" in out["jev_fallback"]


def test_a_report_with_no_checkable_claims_falls_back_and_says_why(judge, llm_calls):
    out = _report_audit(report="# Title\n\n## Section\n\nWhat next?")

    assert llm_calls["n"] == 1 and out["audit_engine"] == "llm"
    assert "no checkable claims in the report" in out["jev_fallback"]


def test_with_jev_off_the_llm_audits_the_report_not_the_context(judge, llm_calls,
                                                                 monkeypatch):
    seen = {}

    async def fake(**kwargs):
        seen.update(kwargs)
        llm_calls["n"] += 1
        return json.dumps(LLM_AUDIT)

    monkeypatch.setattr(llm, "create_chat_completion", fake)
    monkeypatch.setattr(jev, "enabled", lambda: False)

    out = _report_audit()

    assert out["audit_engine"] == "llm" and out["audit_target"] == "report"
    assert out["jev_fallback"] == "jev not configured or in failure cooldown"
    system, user = (m["content"] for m in seen["messages"])
    assert "REPORT" in system and "ABSENCE IS NOT DISAGREEMENT" in system
    assert "REPORT (the claims to audit)" in user and REPORT_CLAIMS[1][:30] in user


def test_when_both_engines_fail_jev_fallback_is_still_a_field(judge, llm_calls,
                                                              monkeypatch):
    """A broken judge must not look like a confident one: "could not audit" carries the
    same jev_fallback field as "audited by the LLM instead"."""
    monkeypatch.setattr(jev, "enabled", lambda: False)
    llm_calls["reply"] = "I could not audit this."

    out = asyncio.run(verification.verify_research(_Researcher(), "q", GATED_CONTEXT, [],
                                                   report=REPORT))

    assert "unsupported_claims" not in out and "overall_confidence" not in out
    assert out["audit_engine"] == "none" and out["audit_error"]
    assert out["jev_fallback"] == "jev not configured or in failure cooldown"
    assert out["audit_scopes"]["report"]["status"] == "could_not_audit"


# ── Per-scope accounting: a scope the judge broke on is never hidden by one it did not ──
# Review 2026-10-01 (verify-review-probe/failopen.py, "report scope all fail"): old shape,
# 40 [Source: url] learnings + a 3-claim report, jev failing on every report state and
# fine on every context state. Summed, 3 failed of 43 sat under the 25% line, and the
# response said audit_engine jev, 0 unsupported, 0 contradictions, overall_confidence
# 0.93 with the report never checked.
LEARNINGS = "\n".join(f"Fact number {i} about idempotency keys holds true. [Source: {A}]"
                      for i in range(40))
FACT_PAGE = [{"url": A, "title": "a",
              "raw_content": "Fact number holds true about idempotency keys. " * 20}]


def _old_shape_audit():
    return asyncio.run(verification.verify_research(
        _Researcher(), "q", LEARNINGS + "\n" + GATED_CONTEXT, FACT_PAGE, report=REPORT))


def test_a_broken_report_scope_is_not_hidden_by_a_good_context_scope(judge, llm_calls):
    judge["answers"] = {c[:25]: None for c in REPORT_CLAIMS}

    out = _old_shape_audit()

    # The report was audited, by the LLM pass: never "jev, 0 findings" over a report
    # nothing judged.
    assert llm_calls["n"] == 1, "the broken report scope did not fall back to the LLM"
    assert out["audit_engine"] == "llm" and out["audit_target"] == "report"
    assert out["unsupported_claims"] == LLM_AUDIT["unsupported_claims"]
    assert "judged none of 3 report claims" in out["jev_fallback"], out["jev_fallback"]
    assert "40 judged learnings were dropped" in out["jev_fallback"]
    assert out["audit_scopes"]["report"] == {"status": "audited", "engine": "llm",
                                             "reason": None}
    assert out["audit_scopes"]["context"]["status"] == "not_run"


def test_a_broken_context_scope_keeps_the_report_audit_and_says_so(judge, llm_calls):
    judge["answers"] = {"Fact number": None}

    out = _old_shape_audit()

    assert llm_calls["n"] == 0 and out["audit_engine"] == "jev"
    # audit_target and jev_fallback are the pre-existing fields: a caller that never reads
    # audit_scopes still sees that the context was not part of this audit.
    assert out["audit_target"] == "report"
    assert "context scope not audited" in out["jev_fallback"]
    ctx = out["audit_scopes"]["context"]
    assert ctx["status"] == "could_not_audit" and ctx["engine"] == "jev"
    assert (ctx["asked"], ctx["judged"], ctx["failed"]) == (40, 0, 40)
    rep = out["audit_scopes"]["report"]
    assert rep["status"] == "audited" and (rep["judged"], rep["failed"]) == (3, 0)
    # Nothing from the unaudited scope reaches the findings or the confidence: 3 report
    # claims verified of 3, not 3 of 43.
    assert out["jev"]["claims"] == 3 and out["overall_confidence"] == 1.0
    assert not any(r["scope"] == "context" for r in out["needs_review"])
    assert "NOT audited" in out["notes"]


def test_every_outcome_names_each_scope(judge, llm_calls):
    clean = _report_audit()
    assert "jev_fallback" not in clean
    assert clean["audit_scopes"]["report"]["status"] == "audited"
    assert clean["audit_scopes"]["context"]["status"] == "not_run"

    no_report = _report_audit(report=None)
    assert no_report["audit_scopes"]["report"] == {
        "status": "not_run", "engine": None, "reason": "no report was given to audit"}
    assert no_report["audit_scopes"]["context"]["engine"] == "llm"


def test_the_reviewers_transport_scenario_reproduced(llm_calls, monkeypatch):
    """The probe's exact shape, through the real jev client: HTTP 400 on every
    report-question state, a good answer on every context state."""
    import urllib.error

    def post(body, headers):
        st = json.loads(body)
        if "research report" in st["questions"]["relation"]["instructions"]:
            raise urllib.error.HTTPError("u", 400, "state too large", {}, None)
        return {"answers": {"relation": {"type": "choice", "choice": "supports",
                                         "confidence": 0.95}}, "usage": {"cost": 0.0001}}

    monkeypatch.delenv("GPTR_MCP_VERIFY_ENGINE", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setattr(jev, "MODEL", "typesafe/jev-test")
    monkeypatch.setattr(jev, "MAX_RETRIES", 0)
    monkeypatch.setattr(jev, "_BREAKER_AFTER", 10 ** 6)
    monkeypatch.setattr(jev, "_consecutive_failures", 0)
    monkeypatch.setattr(jev, "_retry_after", 0.0)
    monkeypatch.setattr(jev, "_post", post)

    out = _old_shape_audit()

    assert not (out["audit_engine"] == "jev" and out["unsupported_claims"] == []), (
        "a report nothing judged shipped as a clean jev audit")
    assert llm_calls["n"] == 1 and out["audit_engine"] == "llm"
    assert out["audit_scopes"]["report"]["status"] == "audited"
    assert out["audit_scopes"]["report"]["engine"] == "llm"
