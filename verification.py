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
    "Given the QUERY, the gathered CONTEXT (the only evidence), and the SOURCE list, "
    "you (1) flag claims in the context NOT clearly supported by the evidence "
    "(possible hallucination/overreach), (2) flag CONTRADICTIONS between sources or "
    "within the context, (3) give an overall confidence 0-1. Be specific; quote the claim. "
    "Respond with ONLY a JSON object, no prose:\n"
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
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    try:
        import json_repair
        return json_repair.loads(text)
    except Exception:
        return {}


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
    return {
        "unsupported_claims": parsed.get("unsupported_claims", []),
        "contradictions": parsed.get("contradictions", []),
        "overall_confidence": parsed.get("overall_confidence"),
        "notes": parsed.get("notes", ""),
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
