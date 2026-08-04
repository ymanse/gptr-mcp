"""Post-research verification passes for the GPT Researcher MCP wrapper.

ORCHESTRATION layer that runs AFTER researcher.conduct_research(), complementing
(NOT duplicating) the engine's in-built MultiLLMReviewer:
  - MultiLLMReviewer -> recall/gaps      ("what's MISSING -> go research more").
  - This module      -> faithfulness     ("are claims SUPPORTED, any CONTRADICTIONS,
                                           how CREDIBLE are the sources").

Self-contained on purpose: depends only on the running researcher's config + the
gpt-researcher LLM helper, so it can later be lifted into a standalone OSS layer
that DEPENDS on gpt-researcher without modifying it.
"""
from __future__ import annotations

import json
import logging
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ── Source tiering (deterministic, no LLM, ~0 cost) ───────────────────────────
# ponytail: domain heuristic, not ground truth. Upgrade path: add an LLM refine
# pass or a maintained allowlist if tier accuracy ever matters more than speed.
_L1_SUFFIXES = (".gov", ".edu", ".mil", ".int")
_L1_DOMAINS = {
    "arxiv.org", "doi.org", "nature.com", "science.org", "ieee.org", "acm.org",
    "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov", "who.int", "w3.org",
    "ietf.org", "rfc-editor.org", "iso.org", "semanticscholar.org", "openalex.org",
}
_L4_DOMAINS = {
    "reddit.com", "stackoverflow.com", "quora.com", "news.ycombinator.com",
    "medium.com", "dev.to", "substack.com", "youtube.com", "x.com", "twitter.com",
    "facebook.com", "tiktok.com", "blogspot.com", "wordpress.com", "pinterest.com",
}
_L2_HINTS = ("docs.", "developer.", "blog.", ".readthedocs.io", "github.com", "gitlab.com")


def _host(url: str) -> str:
    try:
        net = urlparse(url if "//" in url else "//" + url).netloc.lower()
    except Exception:
        return ""
    return net[4:] if net.startswith("www.") else net


def _match(host: str, domains) -> bool:
    """host equals a listed domain or is a subdomain of one."""
    return any(host == d or host.endswith("." + d) for d in domains)


def tier_source(url: str) -> str:
    """Classify one URL: L1 (authoritative) .. L4 (community/UGC)."""
    host = _host(url)
    if not host or "." not in host:  # empty or no TLD -> not a real domain
        return "L4"
    if host.endswith(_L1_SUFFIXES) or _match(host, _L1_DOMAINS):
        return "L1"
    if _match(host, _L4_DOMAINS) or host.endswith(".stackexchange.com"):
        return "L4"
    if any(h in host for h in _L2_HINTS):
        return "L2"
    return "L3"


def tier_sources(urls):
    """Map urls -> tier, plus a per-tier count summary."""
    tiers = {u: tier_source(u) for u in (urls or [])}
    summary = {"L1": 0, "L2": 0, "L3": 0, "L4": 0}
    for t in tiers.values():
        summary[t] = summary.get(t, 0) + 1
    return tiers, summary


# ── Faithfulness audit (citation + contradiction) — single LLM pass ───────────
_VERIFY_SYSTEM = (
    "You audit gathered research for FAITHFULNESS. You do NOT add new facts. "
    "Given the QUERY, the gathered CONTEXT, and the SOURCE list, you "
    "(1) flag claims in the context that the evidence does not support, "
    "(2) flag CONTRADICTIONS, (3) give an overall confidence 0-1. "
    "Be specific; quote the claim.\n"
    # What the auditor can actually see. Stating it prevents two failures that the
    # earlier wording invited: calling a claim a hallucination because the excerpt was
    # cut before its support, and 'finding' disagreements between documents it was
    # never shown.
    "WHAT YOU CAN SEE: CONTEXT is an EXCERPT and may be truncated mid-evidence. "
    "SOURCES is a list of urls and titles ONLY - you are NOT given the text of those "
    "pages, and some of them may never have been read at all. "
    "Therefore: you cannot compare two sources against each other, and you cannot "
    "conclude that a source contradicts a claim. Report a contradiction ONLY when the "
    "CONTEXT itself states both sides.\n"
    "ABSENCE IS NOT DISAGREEMENT. If you cannot locate support for a claim, that may "
    "mean the evidence was cut, or that the page it came from was never retrieved - "
    "not that the claim is false. Say which you mean in `reason`, and start it with "
    "`not-in-excerpt:` when you simply cannot see the support, reserving "
    "`contradicted:` for a claim the CONTEXT actively refutes. A missing source is a "
    "hole in the evidence; treating it as evidence against the claim has caused "
    "correct findings to be discarded.\n"
    "Respond with ONLY a JSON object, no prose. `sources` is optional and may be left "
    "empty - you were not shown which page said what, so leave it out rather than "
    "guessing:\n"
    '{"unsupported_claims":[{"claim":"","reason":""}],'
    '"contradictions":[{"topic":"","a":"","b":"","sources":[""]}],'
    '"overall_confidence":0.0,"notes":""}'
)


def _compact_sources(sources, limit=40):
    return [
        {"url": s.get("url", ""), "title": s.get("title") or ""}
        for s in (sources or [])[:limit]
    ]


def _parse_json(text):
    """The audit's reply as a dict, or None if it did not give us one."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    try:
        direct = json.loads(text)
        if isinstance(direct, dict):
            return direct
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        try:
            fenced = json.loads(m.group(1))
            if isinstance(fenced, dict):
                return fenced
        except Exception:
            pass
    try:
        import json_repair
        repaired = json_repair.loads(text)
    except Exception:
        repaired = None
    # None, not {}: a dict is the ONLY usable answer, and every other outcome must stay
    # distinguishable from one. json_repair returns '' for input it cannot repair, and
    # json.loads returns a bare str for a JSON string literal — both then met
    # parsed.get(...) and raised "'str' object has no attribute 'get'", observed
    # 2026-08-03. Returning {} instead would have been worse than the crash: the audit
    # would have reported ZERO unsupported claims and zero contradictions, which reads
    # as a clean bill of health for a pass that never ran.
    return repaired if isinstance(repaired, dict) else None


async def audit_faithfulness(researcher, query, context, sources):
    """One LLM pass: unsupported-claim + contradiction audit. Reuses the researcher's
    SMART LLM (their claude_agent subscription -> no extra metered API cost)."""
    from gpt_researcher.utils.llm import create_chat_completion

    ctx = context if isinstance(context, str) else str(context)
    ctx = ctx[:12000]  # ponytail: bound the evidence fed into one call
    user = (
        f"QUERY:\n{query}\n\nCONTEXT (evidence):\n{ctx}\n\n"
        f"SOURCES:\n{json.dumps(_compact_sources(sources), ensure_ascii=False)}"
    )
    resp = await create_chat_completion(
        model=researcher.cfg.smart_llm_model,
        messages=[
            {"role": "system", "content": _VERIFY_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        llm_provider=researcher.cfg.smart_llm_provider,
        max_tokens=2000,
        llm_kwargs=researcher.cfg.llm_kwargs,
    )
    parsed = _parse_json(resp)
    # A dict that carries none of the audit's keys is not an audit either. Rejecting
    # only non-dicts would still let `{"error": "..."}` or `{}` through, and both then
    # report zero unsupported claims and zero contradictions — a clean bill of health
    # from a pass that produced nothing. Same hollow zero, one type down.
    if isinstance(parsed, dict) and not any(
        k in parsed for k in ("unsupported_claims", "contradictions", "overall_confidence")
    ):
        parsed = None
    if parsed is None:
        # Fail LOUD rather than empty. An audit that could not be read has found
        # nothing, and "found nothing" is exactly what a clean audit looks like — so
        # returning defaults here would ship "0 unsupported claims, 0 contradictions"
        # for a pass that never happened. verify_research turns this into audit_error,
        # which is the one field a caller can act on.
        raise ValueError(
            f"audit LLM did not return a JSON object (got {type(resp).__name__}, "
            f"{len(str(resp))} chars): {str(resp)[:200]}"
        )
    return {
        "unsupported_claims": parsed.get("unsupported_claims", []),
        "contradictions": parsed.get("contradictions", []),
        "overall_confidence": parsed.get("overall_confidence"),
        "notes": parsed.get("notes", ""),
        # what the audit was actually shown, so "unsupported" can be read for what it
        # is — see the evidence-bound note in _VERIFY_SYSTEM
        "evidence_chars": len(ctx),
        "evidence_truncated": len(context if isinstance(context, str) else str(context)) > len(ctx),
    }


async def verify_research(researcher, query, context, sources):
    """Full bundle: deterministic tiering + LLM faithfulness audit. Tiering always
    ships; audit failure degrades to tiers-only (best-effort, never blocks research)."""
    source_urls = [s.get("url", "") for s in (sources or []) if s.get("url")]
    tiers, tier_summary = tier_sources(source_urls)
    result = {"source_tiers": tiers, "tier_summary": tier_summary}
    try:
        result.update(await audit_faithfulness(researcher, query, context, sources))
    except Exception as e:  # audit is best-effort; tiering still ships
        logger.warning(f"faithfulness audit failed: {e}")
        result["audit_error"] = str(e)
    return result


def demo():
    """Runnable self-check for the deterministic tiering logic."""
    assert tier_source("https://www.nist.gov/x") == "L1"
    assert tier_source("https://arxiv.org/abs/1") == "L1"
    assert tier_source("https://old.reddit.com/r/x") == "L4"          # subdomain match
    assert tier_source("https://unix.stackexchange.com/q/1") == "L4"
    assert tier_source("https://docs.python.org/3/") == "L2"
    assert tier_source("https://github.com/a/b") == "L2"
    assert tier_source("https://some-news-site.com/a") == "L3"
    assert tier_source("garbage") == "L4"                            # unparseable -> lowest
    _, summ = tier_sources(["https://x.gov", "https://reddit.com/z", "https://foo.com"])
    assert summ == {"L1": 1, "L2": 0, "L3": 1, "L4": 1}, summ
    print("verification.py self-check OK")


if __name__ == "__main__":
    demo()
