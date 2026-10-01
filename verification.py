"""Post-research verification passes for the GPT Researcher MCP wrapper.

ORCHESTRATION layer that runs AFTER researcher.conduct_research(), complementing
(NOT duplicating) the engine's in-built MultiLLMReviewer:
  - MultiLLMReviewer -> recall/gaps      ("what's MISSING -> go research more").
  - This module      -> faithfulness     ("are claims SUPPORTED, any CONTRADICTIONS,
                                           how CREDIBLE are the sources").

Self-contained on purpose: depends only on the running researcher's config + the
gpt-researcher LLM helper and jev client (gpt_researcher.utils.jev), so it can later be
lifted into a standalone OSS layer that DEPENDS on gpt-researcher without modifying it.
"""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
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


# ── Faithfulness audit (citation + contradiction) -- one LLM pass, the fallback ──
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


# ── Faithfulness audit, jev path: a citation check that costs no session ───────
# The LLM audit below spends a Sonnet session (agent_purpose "verify") on one free-text
# question, roughly three per verified run (2026-10-01 measurement, see
# D:\dev_ext\jev-research-redesign-2026-10.md P2). The question it asks is a judgment, not
# a generation: does the page this claim cites back it up. TypeSafe's citation-check
# cookbook is that exact job, so this follows it step for step:
#   1. exact substring match first (normalised): free, and catches a misquote outright;
#   2. ONE `choice` question per claim, supports / contradicts / says_nothing, with the
#      claim and a section of the cited page as state;
#   3. a verdict stands only at confidence >= AUTO_ACCEPT (the cookbook's 0.8); below it
#      the verdict is escalated to `needs_review` instead of being decided here.
#
# Claims come from deep_research's own format, `<learning> [Source: <url>]`, one per line
# (gpt_researcher/skills/deep_research.py builds it). That tag is what makes a claim
# checkable at all: it names the page whose scraped text we hold in `sources`. Lines
# without one are raw evidence blocks, not claims, and are not judged.

#: deep_research writes "[Source: u]"; a learning backed by two pages comes out as
#: "[Source: u1, u2]" or "u1 ; u2" (both seen in outputs/, 10 of 645 tags).
_CLAIM_TAG = re.compile(r"^(?P<claim>.*\S)\s*\[Sources?:\s*(?P<urls>[^\]]+)\]\s*$")
_URL_SPLIT = re.compile(r"\s*[,;]\s*(?=https?://)")
#: A quoted span this long is a claim to quote the page verbatim. Shorter quoted spans
#: are names ("Agentic drift", "Integration-Gate Flow") that a page can phrase three ways.
_QUOTE_MIN_WORDS = 6
#: Any quoted span, however short; length is filtered afterwards. A length floor inside the
#: regex mis-pairs straight quotes: live 2026-10-01, in '"contains" edges (though
#: "invokes" edges ...' the 8-char "contains" failed the floor, the scan restarted on its
#: CLOSING mark, and the text BETWEEN two quotes came back quote-not-in-source.
_QUOTED = re.compile(r'"([^"]*)"|\u201c([^\u201c\u201d]*)\u201d')
_STOP = frozenset(
    "the and for are was were that this with from into than then they them their there "
    "which while when where what who will would could should can not but its has have had "
    "been being also such more most only over under about after before between each per "
    "via use used using one two any all may might must does did our your".split())

#: The cookbook's choice question, with the two distinctions our LLM audit learned the
#: hard way carried into the criteria. The section is an EXCERPT of the page (we pick
#: it); a claim whose support sits outside the excerpt must land on says_nothing, never
#: on contradicts. ABSENCE IS NOT DISAGREEMENT -- treating a missing passage as evidence
#: against a claim is what discarded correct findings before (see _VERIFY_SYSTEM).
#: One judgment, one question: three labels of one relation, not three questions.
_JEV_QUESTIONS = {
    "relation": {
        "type": "choice",
        "instructions": (
            "How does `section` relate to `claim`? `section` is an excerpt cut out of the "
            "web page that `claim` cites as its source; the rest of that page is not shown."),
        "criteria": {
            "supports": (
                "The section states what the claim asserts, or directly implies it, with "
                "the same figures and names wherever the claim gives figures and names."),
            "contradicts": (
                "The section explicitly states something incompatible with the claim about "
                "the same subject, such as a different figure for the same measurement or "
                "the opposite conclusion about the same thing."),
            "says_nothing": (
                "The section does not state what the claim asserts either way: it covers a "
                "different detail of the topic, stops before reaching the claim's subject, "
                "or is navigation, a reference list or page boilerplate."),
        },
    },
}
_RELATION_TO_VERDICT = {"supports": "verified", "contradicts": "contradicted",
                        "says_nothing": "unsupported"}


def _env_float(name, default):
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _norm_url(url):
    u = (url or "").strip().rstrip(".,);]").strip()
    try:
        p = urlparse(u)
    except Exception:
        return u.lower()
    host = p.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return f"{host}{p.path.rstrip('/')}{'?' + p.query if p.query else ''}"


def _norm_text(text):
    """Fold the differences a scrape introduces that the quoting LLM never saw: markdown
    links and emphasis, curly quotes, dash variants, line wraps, case."""
    t = unicodedata.normalize("NFKC", text or "")
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)          # [text](url) -> text
    t = t.translate(str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'",
                                   "\u2019": "'", "\u2013": "-", "\u2014": "-"}))
    t = re.sub(r"[*_`\\]", "", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def extract_claims(context):
    """Deterministic claim split: one `[Source: ...]`-tagged line is one claim. No LLM."""
    claims, seen = [], set()
    for line in (context or "").splitlines():
        m = _CLAIM_TAG.match(line.strip())
        if not m:
            continue
        claim = m.group("claim").strip()
        urls = [u.strip() for u in _URL_SPLIT.split(m.group("urls").strip()) if u.strip()]
        if not urls or claim in seen:
            continue
        seen.add(claim)
        claims.append({"claim": claim, "urls": urls})
    return claims


def _source_texts(sources):
    """norm url -> scraped page text. The longest copy wins when a url repeats."""
    out = {}
    for s in sources or []:
        if not isinstance(s, dict):
            continue
        text = s.get("raw_content") or s.get("content") or ""
        key = _norm_url(s.get("url", ""))
        if key and isinstance(text, str) and len(text.strip()) > len(out.get(key, "")):
            out[key] = text.strip()
    return out


def _windows(text, size):
    """Paragraph-packed windows of at most `size` chars, in page order."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        while len(p) > size:                                   # one giant paragraph
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:size])
            p = p[size:]
        if cur and len(cur) + len(p) + 2 > size:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        out.append(cur)
    return out


def _terms(text):
    """Claim terms worth matching, numbers weighted 3x: a figure is the part of a claim
    most likely to be wrong and the easiest to find."""
    weights = {}
    for w in re.findall(r"\w+(?:[.,%]\w+)*%?", _norm_text(text)):
        if any(c.isdigit() for c in w):
            weights[w] = 3
        elif len(w) >= 3 and w not in _STOP:
            weights[w] = 1
    return weights


def _section_for(claim, pages, size):
    """The excerpt of the cited page(s) the judge reads, and whether a quote check failed.

    The cookbook's sources are RFCs with numbered sections, and a verbatim quote names
    the section. Ours are scraped pages with no structure and our claims paraphrase, so
    the section is the best-matching window(s) by claim-term overlap instead.
    ponytail: lexical window choice. A claim whose support uses different words than the
    claim gets the wrong window and comes back says_nothing (unsupported, never
    contradicts). Upgrade path: rank windows with the same embedding the s12 router uses.

    Returns (section, page_chars, missing_quote): missing_quote is the first long quoted
    span found in NO cited page, else None.
    """
    norm_pages = [_norm_text(p) for p in pages]
    missing = None
    quote_hit = None
    for q in (a or b for a, b in _QUOTED.findall(claim)):
        # A span that starts or ends in whitespace is the gap between two quotes, not a
        # quote: an odd mark (an inch sign, a stray quote) still shifts the pairing.
        if len(q.split()) < _QUOTE_MIN_WORDS or q != q.strip():
            continue
        # Punctuation at the quote's edges belongs to the claim's sentence, not the page
        # (US style puts the comma inside the marks). Live 2026-10-01: '"... not
        # authorization controls,"' was in the cited page word for word and still came
        # back quote-not-in-source because of that comma. An ellipsis marks an elision,
        # so each side of it must match on its own.
        parts = [_norm_text(f).strip(" .,;:!?'\"") for f in re.split(r"\.\.\.|…", q)]
        parts = [f for f in parts if f]
        if parts and all(any(f in np_ for np_ in norm_pages) for f in parts):
            quote_hit = quote_hit or max(parts, key=len)
        elif missing is None:
            missing = q
    terms = _terms(claim)
    scored = []
    for pi, page in enumerate(pages):
        for wi, win in enumerate(_windows(page, max(400, size // 2))):
            nw = _norm_text(win)
            score = sum(wt for t, wt in terms.items() if t in nw)
            if quote_hit and quote_hit in nw:
                score += 1000                                  # the quote names its section
            scored.append((score, pi, wi, win))
    if not scored:
        return "", sum(len(p) for p in pages), missing
    best = sorted(scored, key=lambda x: (-x[0], x[1], x[2]))[:2]
    best.sort(key=lambda x: (x[1], x[2]))                      # page order reads naturally
    section = "\n\n[...]\n\n".join(b[3] for b in best)[:size]
    return section, sum(len(p) for p in pages), missing


async def _audit_with_jev(context, sources):
    """The cookbook citation check over every tagged claim. Returns (result, None), or
    (None, reason) when the LLM audit has to run instead.

    Never returns a result that judged nothing: an audit with zero findings because it
    looked at zero claims is the hollow clean bill of health _parse_json warns about.
    """
    if os.getenv("GPTR_MCP_VERIFY_ENGINE", "auto").strip().lower() == "llm":
        return None, "GPTR_MCP_VERIFY_ENGINE=llm"
    try:
        from gpt_researcher.utils import jev
    except ImportError:                                        # pragma: no cover
        return None, "jev client not importable"
    if not jev.enabled():
        return None, "jev not configured or in failure cooldown"

    ctx = context if isinstance(context, str) else str(context)
    claims = extract_claims(ctx)
    if not claims:
        return None, "no [Source: url]-tagged claims in the context to check"
    # ponytail: first N claims in context order. jev is ~$0.03 per 1000 passages, so the
    # cap bounds latency, not money; raise it before reaching for sampling.
    cap = int(_env_float("VERIFY_JEV_MAX_CLAIMS", 120))
    dropped = max(0, len(claims) - cap)
    claims = claims[:cap]
    auto_accept = _env_float("VERIFY_JEV_AUTO_ACCEPT", 0.8)
    size = int(_env_float("VERIFY_JEV_SECTION_CHARS", 3000))
    texts = _source_texts(sources)

    rows, states = [], []
    for c in claims:
        pages = [texts[k] for k in (_norm_url(u) for u in c["urls"]) if k in texts]
        row = dict(c, verdict=None, confidence=None, auto=False, section="",
                   page_chars=0)
        if not pages:
            # The page was never obtained, or obtained empty. A hole in the evidence, not
            # evidence against the claim -- the `unretrieved` lesson in deep_research.
            row["verdict"] = "unverifiable"
        else:
            section, page_chars, missing = _section_for(c["claim"], pages, size)
            row.update(section=section, page_chars=page_chars)
            if missing is not None:
                # Cookbook step 1: a quote that is not in the source needs no model call.
                row.update(verdict="quote_not_in_source", auto=True, missing_quote=missing)
            elif not section.strip():
                row["verdict"] = "unverifiable"
            else:
                states.append({"claim": c["claim"], "section": section})
                row["_ask"] = len(states) - 1
        rows.append(row)

    before = jev.stats()
    answers = await jev.ask_many(states, _JEV_QUESTIONS) if states else []
    after = jev.stats()
    spent = {k: round(after.get(k, 0) - before.get(k, 0), 6)
             for k in ("calls", "errors", "cost", "input_tokens")}

    failed = 0
    for row in rows:
        i = row.pop("_ask", None)
        if i is None:
            continue
        ans = answers[i] if i < len(answers) else None
        ans = ans if isinstance(ans, dict) else {}
        rel = ans.get("relation")
        # A label that is not a string (review 2026-10-01: a 200 whose `choice` was a
        # list) is one judge failure like an unknown label, not a TypeError on the dict
        # lookup that would discard every good verdict in the run.
        if not isinstance(rel, str) or rel not in _RELATION_TO_VERDICT:
            failed += 1
            row["verdict"] = "unjudged"
            continue
        conf = ans.get("relation__confidence")
        # The confidence is what gates the verdict, so a label without a usable one is a
        # verdict we cannot gate: unjudged, like a bad label, and counted toward
        # VERIFY_JEV_MAX_FAIL_FRAC so a judge that does this on every claim falls back to
        # the LLM audit. Review 2026-10-01: `confidence: true` passed a bare isinstance
        # (bool is an int) and `True >= 0.8`, so every claim shipped as verified at 1.0
        # -- and with `contradicts`, straight into the contradictions callers act on.
        # 7 did the same. The range check also rejects NaN and inf; missing (None) is
        # rejected by the type check.
        if (isinstance(conf, bool) or not isinstance(conf, (int, float))
                or not 0.0 <= conf <= 1.0):
            failed += 1
            row["verdict"] = "unjudged"
            continue
        row.update(verdict=_RELATION_TO_VERDICT[rel], confidence=conf,
                   auto=conf >= auto_accept)

    judged = len(states) - failed
    if judged == 0:
        # Nothing reached a verdict from the judge. Quote misses alone are not an audit
        # of the run, and unverifiable claims are not findings at all.
        return None, (f"jev judged none of {len(claims)} claims "
                      f"({len(states)} asked, {failed} failed)")
    max_fail = _env_float("VERIFY_JEV_MAX_FAIL_FRAC", 0.25)
    if failed / len(states) > max_fail:
        # A judge that broke on a quarter of the claims has not audited the run. Partial
        # silence would read as "those claims were fine"; the LLM pass reads them all.
        return None, f"jev failed on {failed} of {len(states)} claims (> {max_fail:.0%})"

    return _jev_result(rows, auto_accept, dropped, spent), None


def _jev_result(rows, auto_accept, dropped, spent):
    """Map cookbook verdicts onto the audit's existing output contract."""
    unsupported, contradictions, review = [], [], []
    counts = {}
    for r in rows:
        v = r["verdict"]
        counts[v] = counts.get(v, 0) + 1
        conf = r["confidence"]
        if v == "quote_not_in_source":
            unsupported.append({"claim": r["claim"], "reason": (
                f'quote-not-in-source: the quoted words "{r["missing_quote"]}" do not '
                f"appear in the scraped text of {', '.join(r['urls'])}. Misquoted, or "
                f"paraphrased inside quote marks; the page may also be only partly "
                f"scraped, so this is not evidence the underlying claim is false.")})
        elif v in ("unverifiable", "unjudged") or not r["auto"]:
            # Below AUTO_ACCEPT the cookbook has a human confirm the verdict before
            # anything acts on it. Callers act on unsupported_claims/contradictions, so an
            # uncertain verdict must not land there; it is listed here instead, never
            # dropped.
            review.append({"claim": r["claim"], "tentative": v,
                           "confidence": conf, "sources": r["urls"]})
        elif v == "unsupported":
            unsupported.append({"claim": r["claim"], "reason": (
                f"not-in-excerpt: the {len(r['section'])}-char section of the cited page "
                f"shown to the judge (of {r['page_chars']} scraped chars) does not state "
                f"this (jev says_nothing, confidence {conf:.2f}). Support may sit "
                f"elsewhere on the page; absence is not disagreement.")})
        elif v == "contradicted":
            contradictions.append({
                "topic": r["claim"][:120],
                "a": r["claim"],
                "b": r["section"][:600],
                "sources": r["urls"]})
    below = sum(1 for r in rows if r["verdict"] in _RELATION_TO_VERDICT.values()
                and not r["auto"])
    verified = sum(1 for r in rows if r["verdict"] == "verified" and r["auto"])
    total = len(rows)
    # Confidence = the share of the context's claims positively confirmed at or above
    # AUTO_ACCEPT. Uncertain, unjudged and unverifiable claims count against it, so a run
    # where the judge could confirm nothing reads as low confidence, never as clean.
    overall = round(verified / total, 2) if total else 0.0
    shown = sum(len(r["section"]) for r in rows)
    excerpted = any(r["section"] and len(r["section"]) < r["page_chars"] for r in rows)
    notes = (
        f"jev citation check over {total} claims"
        + (f" ({dropped} more past VERIFY_JEV_MAX_CLAIMS not checked)" if dropped else "")
        + f": {verified} verified, {len(unsupported)} unsupported, {len(contradictions)} "
        f"contradicted at confidence >= {auto_accept}; {len(review)} in needs_review "
        f"({below} below the threshold, {counts.get('unverifiable', 0)} cite a page whose "
        f"text was never obtained, {counts.get('unjudged', 0)} the judge failed on). Each "
        f"claim was judged against an excerpt of the page it cites, not the whole page.")
    return {
        "unsupported_claims": unsupported,
        "contradictions": contradictions,
        "overall_confidence": overall,
        "notes": notes,
        # the judge reads one excerpt per claim, so this is the total excerpt text shown
        "evidence_chars": shown,
        "evidence_truncated": bool(dropped) or excerpted,
        "needs_review": review,
        "audit_engine": "jev",
        # What this audit cost, read next to agent_calls_by_site: the session it did not
        # spend on one side, fractions of a cent on the other.
        # ponytail: delta of process-wide counters, so a concurrent jev user (the P1 gate
        # in another run) leaks into it. Upgrade path: per-call usage from jev.ask.
        "jev": {**spent, "claims": total, "not_checked": dropped,
                "auto_accept": auto_accept, "verdicts": counts,
                "below_threshold": below},
    }


async def audit_faithfulness(researcher, query, context, sources):
    """Unsupported-claim + contradiction audit: jev citation check first, LLM pass second.

    The jev path costs no `claude` session; the LLM path costs one. When jev is off,
    erroring, or has nothing it can judge, this is exactly the old single LLM pass, and
    that fallback is the point: "fail open" here means "run the check we had", never
    "skip the check", because a skipped audit and a clean one look identical downstream.
    """
    try:
        jev_out, why_not = await _audit_with_jev(context, sources)
    except Exception as e:                                     # noqa: BLE001
        # Review 2026-10-01: jev answered HTTP 200 with a malformed body (`answers` a
        # list, `usage` a string), the client raised, and the exception escaped to
        # verify_research as audit_error -- so NO audit ran at all. Any raise on the jev
        # path is "jev could not audit", and that runs the LLM pass like every other
        # reason. Logged, and named in jev_fallback, so a judge that breaks this way
        # stays distinguishable from one that found nothing.
        logger.warning(f"jev citation check raised, falling back to the LLM audit: {e!r}")
        jev_out, why_not = None, f"jev path raised {type(e).__name__}: {e}"
    if jev_out is not None:
        return jev_out
    try:
        result = await _audit_with_llm(researcher, query, context, sources)
    except Exception as e:
        # Keep why jev did not run in the audit_error verify_research ships: "the LLM
        # audit failed" and "both audits failed" call for different fixes.
        raise RuntimeError(f"{e} [jev path not taken: {why_not}]") from e
    result["audit_engine"] = "llm"
    result["jev_fallback"] = why_not
    return result


async def _audit_with_llm(researcher, query, context, sources):
    """One LLM pass: unsupported-claim + contradiction audit. Reuses the researcher's
    SMART LLM (their claude_agent subscription -> no extra metered API cost)."""
    from gpt_researcher.utils.llm import create_chat_completion

    try:
        from gpt_researcher.utils.agent_purpose import agent_purpose
    except ImportError:  # pragma: no cover - a fork without the attribution
        from contextlib import nullcontext

        def agent_purpose(_site):
            return nullcontext()

    ctx = context if isinstance(context, str) else str(context)
    ctx = ctx[:12000]  # ponytail: bound the evidence fed into one call
    user = (
        f"QUERY:\n{query}\n\nCONTEXT (evidence):\n{ctx}\n\n"
        f"SOURCES:\n{json.dumps(_compact_sources(sources), ensure_ascii=False)}"
    )
    with agent_purpose("verify"):
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
    # claim split for the jev citation check: only [Source: ...]-tagged lines are claims
    claims = extract_claims(
        "Merge trains cut conflicts. [Source: https://a.dev/x, https://b.dev/y]\n"
        "Source: https://a.dev/x\nTitle: raw evidence block, not a claim\n"
        "Untagged learning line")
    assert claims == [{"claim": "Merge trains cut conflicts.",
                       "urls": ["https://a.dev/x", "https://b.dev/y"]}], claims
    page = ("Nav | Home | Blog\n\n" + "filler words here. " * 200
            + "\n\nWe measured 27.67% of AI PRs hit merge conflicts.")
    section, _, missing = _section_for("27.67% of AI PRs hit merge conflicts", [page], 3000)
    assert "27.67%" in section and missing is None, section[:200]
    _, _, missing = _section_for(
        'It says "agents never ever touch the main branch at all" plainly', [page], 3000)
    assert missing == "agents never ever touch the main branch at all", missing
    print("verification.py self-check OK")


if __name__ == "__main__":
    demo()
