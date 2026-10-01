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
# Two claim sources (see "Report audit" below for why the report is now the main one):
#   - the REPORT's sentences, judged against the passages of the context most similar to
#     each (retrieve-then-judge), whenever there is a report;
#   - context learnings in deep_research's pre-s24 format, `<learning> [Source: <url>]`,
#     one per line (only with the jev gate off). That tag names the page whose scraped
#     text we hold in `sources`, so each is judged against its page. Lines without one
#     are raw evidence blocks, not claims, and are not judged as claims.

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


def _missing_quote(claim, haystack_norm):
    """The first long quoted span of `claim` absent from the normalised haystack, else None.
    Same matching rules as _section_for (edge punctuation, ellipsis, quote pairing)."""
    for q in (a or b for a, b in _QUOTED.findall(claim)):
        if len(q.split()) < _QUOTE_MIN_WORDS or q != q.strip():
            continue
        parts = [_norm_text(f).strip(" .,;:!?'\"") for f in re.split(r"\.\.\.|…", q)]
        parts = [f for f in parts if f]
        if parts and not all(f in haystack_norm for f in parts):
            return q
    return None


# ── Report audit: the claims of the synthesis, checked against the research ──────────
# Live 2026-10-01, GPTR_MCP_VERIFY=true in production: audit_engine=llm, one `verify`
# session, jev_fallback "no [Source: url]-tagged claims in the context to check",
# overall_confidence 0.9, 0 unsupported, 0 contradictions. s24's jev gate had removed the
# per-sub-query learnings step that wrote those tags; the context is now raw selected
# passages in Source:/Title:/Content: blocks (`grep -c "\[Source:"` = 0 on a real run).
# Two streams, each right alone, incompatible together.
#
# Teaching extract_claims the block format would not have fixed it: the audit ran BEFORE
# synthesis, over the context, and a raw scraped passage always supports itself. What can
# drift in this architecture is the one report Sonnet writes at the end from those
# passages. So the claims now come from the REPORT and the evidence is the CONTEXT -- a
# claim against its evidence, which is what the citation-check cookbook does.

#: The server's own partial-research banner (server._partial_banner) is a disclosure we
#: wrote, not a claim the writer made.
_BANNER = re.compile(r"^>\s*\*\*Partial research\.\*\*")
#: Where the report stops making claims: the bibliography.
_REFS_HEADING = re.compile(
    r"^#{1,6}\s*(?:\d+[.)]?\s*)?(references|sources|bibliography|works cited|citations)\b",
    re.I)
_TABLE_SEP = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_LIST_MARK = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_MD_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")
#: "([Stripe, 2024](u))" or "([A](u); [B](v))": a citation group, not part of the assertion.
_CITE_GROUP = re.compile(
    r"\s*\(\s*\[[^\]]*\]\(https?://[^)\s]+\)(?:\s*[;,]\s*\[[^\]]*\]\(https?://[^)\s]+\))*"
    r"\s*\)")
#: Periods that do not end a sentence. Kept short on purpose: an unlisted abbreviation
#: splits one claim into two shorter ones, which still get judged.
_ABBREV = re.compile(r"\b(e\.g|i\.e|vs|etc|n\.d|et al|cf|approx|Fig|No|Dr|Mr|Ms|St|U\.S)\.",
                     re.I)
_SENT_END = re.compile(r"(?<=[.!?])[\"'”)\]*_]*\s+(?=[\"'“(\[*_`]?[A-Z0-9$])")
#: Openers that make a sentence a guess rather than an assertion.
_HEDGE = re.compile(
    r"^(perhaps|possibly|arguably|presumably|conceivably|it (is|seems|appears) "
    r"(possible|plausible|likely|unclear)|it (may|might|could)|one (may|might|could)|"
    r"this (may|might|could) )", re.I)
#: A sentence about the report itself ("This report synthesizes ...") asserts nothing
#: the research could support or refute.
_META = re.compile(r"\b(this|the following|the present) (report|section|analysis|document|"
                   r"synthesis|overview|summary|table)\b|\bthe (table|sections?) below\b",
                   re.I)
_CLAIM_MIN_WORDS = 5
#: "*Source: [se.dosibridge.com, n.d.](u)*" under a table: a citation, not a claim.
_CAPTION = re.compile(r"^(sources?|adapted from|via)\s*:", re.I)


def _clean_claim(text):
    """Claim text as the judge should read it, and the urls it cites."""
    urls = [u for _, u in _MD_LINK.findall(text)]
    t = _CITE_GROUP.sub("", text)
    t = _MD_LINK.sub(r"\1", t)
    t = re.sub(r"\*\*|__|(?<!\w)[*_](?!\s)|(?<!\s)[*_](?!\w)", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([.,;:!?])(?=\s|$)", r"\1", t)  # the gap a removed citation left
    return t, urls


def _claim_kind(text):
    """Why a sentence counts: a figure, a named thing, or a plain definite assertion."""
    if re.search(r"\d", text):
        return "number"
    words = text.split()
    if any(re.match(r"^[\"'(`]*(?:[A-Z][a-z]+[A-Z]\w*|[A-Z]{2,}\w*|[A-Z][a-z]+)", w)
           for w in words[1:]) or "`" in text:
        return "entity"
    return "assertion"


def _split_sentences(text):
    """Sentences, never split inside a quotation or a code span. Measured on the
    2026-10-01 reports: a quoted two-sentence passage ('"... still processing. A new
    request arrives ..."') and `INSERT ... ON CONFLICT` both came out as two half-claims,
    the second of which no passage could support on its own."""
    def hide(m):
        t = m.group(0)
        # US style ends the sentence inside the marks ('... proceeds." It ...'): the
        # punctuation right before a closing quote still ends a sentence.
        tail = t[-2:] if len(t) > 2 and t[-1] in "\"”" and t[-2] in ".?!" else ""
        body = t[:len(t) - len(tail)]
        return body.replace(".", "\x00").replace("?", "\x01").replace("!", "\x02") + tail

    protected = re.sub(r"`[^`]*`|\"[^\"]*\"|“[^“”]*”", hide, text)
    protected = _ABBREV.sub(hide, protected)
    return [s.replace("\x00", ".").replace("\x01", "?").replace("\x02", "!").strip()
            for s in _SENT_END.split(protected) if s.strip()]


def extract_report_claims(report):
    """Deterministic claim split of a markdown report. No LLM.

    A claim is a body sentence or a table row that asserts something: it carries a figure,
    a named entity, or is a definite declarative statement. Not claims: headings, the
    partial banner, code, lead-ins ending in ':', questions, hedged openers, sentences
    about the report itself, fragments under five words, and everything from the
    References heading on. Each claim keeps the urls it cites inline.
    """
    units = []                                   # (text, is_table_row)
    para, header, in_code = [], None, False

    def flush():
        if para:
            units.append((" ".join(para), False))
            para.clear()

    for raw in (report or "").splitlines():
        line = raw.strip()
        if line.startswith("```"):
            flush()
            in_code = not in_code
            continue
        if in_code:
            continue
        if _REFS_HEADING.match(line):
            break
        if not line or line.startswith("#") or re.fullmatch(r"[-*_]{3,}", line) \
                or _BANNER.match(line) or re.fullmatch(r"_[^_].*_", line):
            flush()
            header = None if not line.startswith("|") else header
            continue
        if line.startswith("|"):
            flush()
            cells = [c.strip() for c in line.strip("|").split("|")]
            if _TABLE_SEP.match(line):
                continue
            if header is None:
                header = cells
                continue
            # A row is a claim only with its column names: "| 409 | Conflict |" alone
            # says nothing a page could support.
            units.append(("; ".join(f"{h}: {c}" for h, c in zip(header, cells)
                                    if c and c not in ("-", "--")), True))
            continue
        header = None
        if _LIST_MARK.match(line):
            flush()
            line = _LIST_MARK.sub("", line)
        para.append(line.lstrip("> ").strip())
    flush()

    claims, seen = [], set()
    for text, is_row in units:
        sentences = [text] if is_row else _split_sentences(text)
        for s in sentences:
            claim, urls = _clean_claim(s)
            if not claim.strip(" .;:") or _CAPTION.match(claim):
                # a citation group that landed after the period belongs to the sentence
                # before it
                if urls and claims:
                    claims[-1]["urls"] += [u for u in urls if u not in claims[-1]["urls"]]
                continue
            if (claim.endswith("?") or claim.endswith(":") or _HEDGE.match(claim)
                    or (_META.search(claim) and not re.search(r"\d", claim))
                    or len(re.findall(r"\w+", claim)) < _CLAIM_MIN_WORDS):
                continue
            key = _norm_text(claim)
            if key in seen:
                continue
            seen.add(key)
            claims.append({"claim": claim, "urls": urls,
                           "kind": "table_row" if is_row else _claim_kind(claim)})
    return claims


#: The persisted context's block header (utils.format_context_with_sources / the jev gate).
_SOURCE_HDR = re.compile(r"^Source:\s*(\S+)\s*$", re.M)
_ALSO = re.compile(r"^\[Also reported by:[^\]]*\]\s*$", re.M)


def _context_passages(context, size):
    """The research context cut into retrievable passages: [{"url", "text"}], in order.

    Both shapes: `[Source: url]`-tagged learning lines (jev gate off) are one passage
    each; `Source:/Title:/Content:` blocks (gate on) are windowed to `size` chars, each
    window keeping its block's url and title.
    """
    ctx = context or ""
    out, rest = [], []
    for line in ctx.splitlines():
        m = _CLAIM_TAG.match(line.strip())
        if m:
            urls = [u.strip() for u in _URL_SPLIT.split(m.group("urls")) if u.strip()]
            out.append({"url": urls[0] if urls else "", "text": m.group("claim").strip()})
        else:
            rest.append(line)
    ctx = "\n".join(rest)
    hdrs = list(_SOURCE_HDR.finditer(ctx))
    blocks = [("", ctx[:hdrs[0].start()] if hdrs else ctx)]
    for i, m in enumerate(hdrs):
        end = hdrs[i + 1].start() if i + 1 < len(hdrs) else len(ctx)
        blocks.append((m.group(1), ctx[m.end():end]))
    for url, body in blocks:
        body = _ALSO.sub("", body)
        title = ""
        tm = re.match(r"\s*Title:\s*(.*)", body)
        if tm:
            title, body = tm.group(1).strip(), body[tm.end():]
        body = re.sub(r"^\s*Content:\s*", "", body)
        if len(body.strip()) < 40:                  # a header with nothing under it
            continue
        for win in _windows(body.strip(), size):
            out.append({"url": url, "text": f"[{title}]\n{win}" if title else win})
    return out


async def _embed_texts(embeddings, texts):
    """Vectors for `texts`, or None. Fail-open: without vectors retrieval goes lexical.

    Bounded because this runs after synthesis, inside the caller's wait. Measured
    2026-10-01 on the host: 230-240 texts in 17-37s normally, and once 411s when the
    local embedder was busy -- the run that hit the bound finished on lexical ranking.
    """
    if embeddings is None or not texts:
        return None
    import asyncio
    try:
        vecs = await asyncio.wait_for(
            asyncio.to_thread(embeddings.embed_documents, list(texts)),
            timeout=_env_float("VERIFY_EMBED_TIMEOUT_S", 120))
        if not isinstance(vecs, list) or len(vecs) != len(texts):
            raise ValueError(f"{len(vecs) if isinstance(vecs, list) else vecs!r} vectors "
                             f"for {len(texts)} texts")
        return vecs
    except Exception as exc:                                    # noqa: BLE001
        logger.warning(f"verify: evidence embeddings unavailable ({exc!r}); ranking "
                       f"passages lexically")
        return None


#: Reciprocal-rank-fusion constant (the usual 60): a passage's fused score is
#: sum(1 / (RRF_K + rank)) over the two rankings.
_RRF_K = 60


async def _rank_evidence(claims, passages, embeddings):
    """Per claim, every passage index ranked best-first, and how they were ranked.

    RANK, DO NOT THRESHOLD: the judge reads the top k whatever the similarity, so a run
    whose scores sit low still shows each claim its best evidence.

    Hybrid, because neither ranking alone held up on the 2026-10-01 reports. Embeddings
    alone never surfaced the passage behind "Replay window; API v1: At least 24 hours;
    API v2: 30 days" -- a markdown table row, which embeds poorly -- until k=12, so the
    judge said says_nothing at 0.99 on a claim the context states almost verbatim; the
    lexical ranking had it at rank 1. Lexical alone misses paraphrase. The two ranks are
    fused (reciprocal rank fusion), which is still a ranking, not a score cutoff.
    """
    norm = [_norm_text(p["text"]) for p in passages]
    lexical = []
    for c in claims:
        terms = _terms(c["claim"])
        score = [sum(w for t, w in terms.items() if t in n) for n in norm]
        lexical.append(sorted(range(len(passages)), key=lambda i: (-score[i], i)))
    vecs = await _embed_texts(embeddings, [c["claim"] for c in claims]
                              + [p["text"] for p in passages])
    if vecs is None:
        # ponytail: lexical only, claim-term overlap with figures weighted 3x. A
        # paraphrase finds the wrong passage and comes back says_nothing -- unsupported
        # or needs_review, never contradicts -- and the result says "lexical".
        return lexical, "lexical"
    import numpy as np
    m = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    m = m / norms
    sim = m[:len(claims)] @ m[len(claims):].T
    fused = []
    for row, lex in zip(sim, lexical):
        rrf = np.zeros(len(passages))
        rrf[np.argsort(-row, kind="stable")] += 1.0 / (_RRF_K + np.arange(len(passages)))
        rrf[np.asarray(lex)] += 1.0 / (_RRF_K + np.arange(len(passages)))
        fused.append([int(i) for i in np.argsort(-rrf, kind="stable")])
    return fused, "hybrid"


#: The report variant of the one relation question. Same three labels, same criteria;
#: what changes is what `section` is -- retrieved passages of the research the report was
#: written from, not the page a learning cites.
_JEV_REPORT_QUESTIONS = {
    "relation": {
        "type": "choice",
        "instructions": (
            "How does `section` relate to `claim`? `claim` is a sentence from a research "
            "report. `section` holds the passages of the research that report was written "
            "from that are most similar to `claim`, separated by [...]; the rest of the "
            "research is not shown."),
        "criteria": _JEV_QUESTIONS["relation"]["criteria"],
    },
}


def _report_rows(report_claims, ctx, passages, ranked, k):
    """Rows + jev states for the report's claims. Cookbook step 1 (a quote absent from
    the research needs no model call), then the top-k passages as the section."""
    rows, states = [], []
    ctx_norm = _norm_text(ctx)
    for ci, c in enumerate(report_claims):
        row = dict(claim=c["claim"], urls=c["urls"], scope="report", kind=c["kind"],
                   verdict=None, confidence=None, auto=False, section="",
                   page_chars=len(ctx), evidence_urls=[])
        missing = _missing_quote(c["claim"], ctx_norm)
        if missing is not None:
            row.update(verdict="quote_not_in_source", auto=True, missing_quote=missing)
        elif not passages:
            row["verdict"] = "unverifiable"
        else:
            # Best-ranked first: a contradiction's `b` shows the head of the section.
            top = [int(i) for i in ranked[ci][:k]]
            row["section"] = "\n\n[...]\n\n".join(passages[i]["text"] for i in top)
            row["evidence_urls"] = list(dict.fromkeys(
                passages[i]["url"] for i in top if passages[i]["url"]))
            states.append({"claim": c["claim"], "section": row["section"]})
            row["_ask"] = ("report", len(states) - 1)
        rows.append(row)
    return rows, states


def _context_rows(claims, sources, size):
    """Rows + jev states for `[Source: url]`-tagged learnings, each against the page it
    cites. The pre-report audit, unchanged."""
    texts = _source_texts(sources)
    rows, states = [], []
    for c in claims:
        pages = [texts[k] for k in (_norm_url(u) for u in c["urls"]) if k in texts]
        row = dict(c, scope="context", verdict=None, confidence=None, auto=False,
                   section="", page_chars=0)
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
                row["_ask"] = ("context", len(states) - 1)
        rows.append(row)
    return rows, states


async def _no_states():
    return []


async def _audit_with_jev(context, sources, report=None, embeddings=None):
    """The cookbook citation check. Returns (result, None), or (None, reason) when the
    LLM audit has to run instead.

    What it checks depends on what exists:
      - a report: every claim in it, against the top-k passages of the research context
        (either shape: Source:/Title:/Content: blocks or `[Source: url]` learnings);
      - `[Source: url]`-tagged learnings in the context (jev gate off): each against the
        page it cites, as before the report audit existed. With a report too, both run:
        page -> learning and learning -> report are two separate places to drift, and
        the report check alone would pass a report that faithfully repeats a learning
        that misread its page.

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
    report_claims = extract_report_claims(report) if report else []
    learnings = extract_claims(ctx)
    if not report_claims and not learnings:
        if report:
            return None, ("no checkable claims in the report, and no [Source: url]-tagged "
                          "claims in the context to check")
        # Kept word for word at its head: it is how a caller recognises this case.
        return None, ("no [Source: url]-tagged claims in the context to check, and no "
                      "report was given to audit")
    # ponytail: first N claims per scope, in order. jev is ~$0.03 per 1000 passages, so
    # the cap bounds latency, not money; raise it before reaching for sampling.
    cap = int(_env_float("VERIFY_JEV_MAX_CLAIMS", 120))
    dropped = max(0, len(report_claims) - cap) + max(0, len(learnings) - cap)
    report_claims, learnings = report_claims[:cap], learnings[:cap]
    auto_accept = _env_float("VERIFY_JEV_AUTO_ACCEPT", 0.8)
    size = int(_env_float("VERIFY_JEV_SECTION_CHARS", 3000))

    rows, rep_states, retrieval, k, n_passages = [], [], None, 0, 0
    if report_claims:
        # k sweep, 2026-10-01, two real gate-on reports (70 + 67 claims), ~1000-char
        # passages, hybrid ranking: claims changing bucket per step k=2->4: 2/4,
        # 4->6: 1/2, 6->8: 4/2, 8->12: 3/2, against 2/0 between two identical k=4 runs.
        # Verdicts stop moving at 4. Embedding-only ranking was still moving at 8.
        k = int(_env_float("VERIFY_REPORT_EVIDENCE_K", 4))
        passages = _context_passages(
            ctx, int(_env_float("VERIFY_REPORT_PASSAGE_CHARS", 1000)))
        n_passages = len(passages)
        ranked, retrieval = ((await _rank_evidence(report_claims, passages, embeddings))
                             if passages else ([], None))
        rows, rep_states = _report_rows(report_claims, ctx, passages, ranked, k)
    ctx_rows, ctx_states = _context_rows(learnings, sources, size)
    rows += ctx_rows
    asked = len(rep_states) + len(ctx_states)

    import asyncio
    before = jev.stats()
    rep_ans, ctx_ans = await asyncio.gather(
        jev.ask_many(rep_states, _JEV_REPORT_QUESTIONS) if rep_states else _no_states(),
        jev.ask_many(ctx_states, _JEV_QUESTIONS) if ctx_states else _no_states())
    after = jev.stats()
    spent = {key: round(after.get(key, 0) - before.get(key, 0), 6)
             for key in ("calls", "errors", "cost", "input_tokens")}
    answers = {"report": rep_ans or [], "context": ctx_ans or []}

    failed = {"report": 0, "context": 0}
    for row in rows:
        ask = row.pop("_ask", None)
        if ask is None:
            continue
        got = answers[ask[0]]
        ans = got[ask[1]] if ask[1] < len(got) else None
        ans = ans if isinstance(ans, dict) else {}
        rel = ans.get("relation")
        # A label that is not a string (review 2026-10-01: a 200 whose `choice` was a
        # list) is one judge failure like an unknown label, not a TypeError on the dict
        # lookup that would discard every good verdict in the run.
        if not isinstance(rel, str) or rel not in _RELATION_TO_VERDICT:
            failed[ask[0]] += 1
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
            failed[ask[0]] += 1
            row["verdict"] = "unjudged"
            continue
        row.update(verdict=_RELATION_TO_VERDICT[rel], confidence=conf,
                   auto=conf >= auto_accept)

    # Judged, failed and the fallback decision are PER SCOPE, never summed across both.
    # Review 2026-10-01 (verify-review-probe/failopen.py): old shape, a 3-claim report
    # plus 40 [Source: url] learnings, jev answering HTTP 400 on every report state and
    # fine on every context state. Summed, that was 3 failed of 43 asked, under the 25%
    # line, and the response said audit_engine jev, 0 unsupported, 0 contradictions,
    # overall_confidence 0.93 -- with the report, the one thing the caller reads, never
    # checked. Fourth time this rule broke here (merge_stats on a 200 with no answers,
    # `confidence: true`, the dedup judge's invisible timeouts): a broken judge must not
    # look like a confident one, and that has to hold per scope.
    max_fail = _env_float("VERIFY_JEV_MAX_FAIL_FRAC", 0.25)
    scopes, broken = {}, {}
    for s, n, a in (("report", len(report_claims), len(rep_states)),
                    ("context", len(learnings), len(ctx_states))):
        if not n:
            continue
        f = failed[s]
        scopes[s] = {"status": "audited", "engine": "jev", "reason": None,
                     "claims": n, "asked": a, "judged": a - f, "failed": f}
        if a - f == 0:
            # Nothing in this scope reached a verdict from the judge. Quote misses alone
            # are not an audit, and unverifiable claims are not findings at all.
            broken[s] = f"jev judged none of {n} {s} claims ({a} asked, {f} failed)"
        elif f / a > max_fail:
            # A judge that broke on a quarter of a scope has not audited it. Partial
            # silence would read as "those claims were fine".
            broken[s] = f"jev failed on {f} of {a} {s} claims (> {max_fail:.0%})"
        if s in broken:
            scopes[s].update(status="could_not_audit", reason=broken[s])

    # What a broken scope costs. The REPORT must be audited, so a broken report scope runs
    # the LLM pass over the report, as a wholly broken judge always did; a good context
    # result is dropped with it, because that pass audits the report only and one
    # overall_confidence cannot honestly mix the two engines' scales. A broken CONTEXT
    # scope beside a good report scope keeps the report audit: the LLM fallback would
    # audit exactly the report and nothing else, so falling back would trade a per-claim
    # audit of the same target for a 12k-char excerpt pass. The context scope then ships
    # as could_not_audit, outside audit_target, and is named in jev_fallback.
    if "report" in broken or len(broken) == len(scopes):
        why = "; ".join(broken.values())
        if "report" in broken and "context" in scopes and "context" not in broken:
            why += (f"; the context scope's {scopes['context']['judged']} judged learnings "
                    f"were dropped with it: the LLM fallback audits the report only")
        return None, why

    rows = [r for r in rows if r["scope"] not in broken]
    target = "+".join(s for s in ("report", "context") if s in scopes and s not in broken)
    out = _jev_result(rows, auto_accept, dropped, spent, target)
    out["jev"].update(retrieval=retrieval, evidence_k=k, passages=n_passages,
                      claims_by_scope={"report": len(report_claims),
                                       "context": len(learnings)})
    out["audit_scopes"] = scope_status(report, ctx, scopes)
    if broken:
        out["jev_fallback"] = (f"context scope not audited ({broken['context']}); the "
                               f"report scope was audited by jev and stands")
        out["notes"] += (f" The context scope was NOT audited ({broken['context']}); "
                         f"overall_confidence covers the report scope only.")
    return out, None


def scope_status(report, context, done):
    """Per scope, what the audit did, for the response's `audit_scopes`. Both scopes are
    always listed. `done` holds the entries for scopes an engine took on.

    status is one of:
      - "audited": an engine judged this scope, and its findings are in
        unsupported_claims / contradictions and count in overall_confidence;
      - "could_not_audit": an engine was asked and failed; nothing from this scope is in
        the findings;
      - "not_run": nothing in this scope was put to an engine; `reason` says why.
    Only "audited" means the scope was checked. A caller must not read the absence of
    findings for any other status as clean.
    """
    out = {}
    for s in ("report", "context"):
        if s in done:
            out[s] = done[s]
            continue
        if s == "report":
            why = ("no report was given to audit" if not report
                   else "the report had no checkable claims")
        elif not extract_claims(context if isinstance(context, str) else str(context)):
            why = ("no [Source: url]-tagged learnings in the context; with a report it is "
                   "the evidence, not a claim" if report else
                   "no [Source: url]-tagged learnings in the context")
        else:
            why = "the LLM fallback audits the report only, not the context's learnings"
        out[s] = {"status": "not_run", "engine": None, "reason": why}
    return out


def _jev_result(rows, auto_accept, dropped, spent, target="context"):
    """Map cookbook verdicts onto the audit's existing output contract.

    unsupported_claims and contradictions keep their exact pre-report entry shape (callers
    and the skill's report render them); which scope a finding came from shows in its
    reason text and in jev.verdicts_by_scope. needs_review entries carry `scope`.
    """
    unsupported, contradictions, review = [], [], []
    counts, by_scope = {}, {}
    for r in rows:
        v = r["verdict"]
        counts[v] = counts.get(v, 0) + 1
        sc = by_scope.setdefault(r["scope"], {})
        sc[v] = sc.get(v, 0) + 1
        conf = r["confidence"]
        on_report = r["scope"] == "report"
        if v == "quote_not_in_source":
            where = ("the research context the report was written from" if on_report
                     else f"the scraped text of {', '.join(r['urls'])}")
            unsupported.append({"claim": r["claim"], "reason": (
                f'quote-not-in-source: the quoted words "{r["missing_quote"]}" do not '
                f"appear in {where}. Misquoted, or paraphrased inside quote marks; "
                + ("" if on_report else "the page may also be only partly scraped, so ")
                + "this is not evidence the underlying claim is false.")})
        elif v in ("unverifiable", "unjudged") or not r["auto"]:
            # Below AUTO_ACCEPT the cookbook has a human confirm the verdict before
            # anything acts on it. Callers act on unsupported_claims/contradictions, so an
            # uncertain verdict must not land there; it is listed here instead, never
            # dropped.
            review.append({"claim": r["claim"], "scope": r["scope"], "tentative": v,
                           "confidence": conf,
                           "sources": r["urls"] or r.get("evidence_urls", [])})
        elif v == "unsupported":
            if on_report:
                reason = (
                    f"not-in-excerpt: the {len(r['section'])} chars of research passages "
                    f"most similar to this claim (of {r['page_chars']} context chars) do "
                    f"not state it (jev says_nothing, confidence {conf:.2f}). Support may "
                    f"sit in a passage not shown, or the sentence is the writer's own "
                    f"synthesis; absence is not disagreement.")
            else:
                reason = (
                    f"not-in-excerpt: the {len(r['section'])}-char section of the cited "
                    f"page shown to the judge (of {r['page_chars']} scraped chars) does not "
                    f"state this (jev says_nothing, confidence {conf:.2f}). Support may sit "
                    f"elsewhere on the page; absence is not disagreement.")
            unsupported.append({"claim": r["claim"], "reason": reason})
        elif v == "contradicted":
            contradictions.append({
                "topic": r["claim"][:120],
                "a": r["claim"],
                "b": r["section"][:600],
                "sources": r.get("evidence_urls") or r["urls"]})
    below = sum(1 for r in rows if r["verdict"] in _RELATION_TO_VERDICT.values()
                and not r["auto"])
    verified = sum(1 for r in rows if r["verdict"] == "verified" and r["auto"])
    total = len(rows)
    # Confidence = the share of the claims positively confirmed at or above AUTO_ACCEPT.
    # Uncertain, unjudged and unverifiable claims count against it, so a run where the
    # judge could confirm nothing reads as low confidence, never as clean.
    overall = round(verified / total, 2) if total else 0.0
    shown = sum(len(r["section"]) for r in rows)
    excerpted = any(r["section"] and len(r["section"]) < r["page_chars"] for r in rows)
    what = {"report": "report claims against the research passages most similar to each",
            "context": "context claims against the page each cites",
            "report+context": "report claims against the research passages most similar "
                              "to each, and context claims against the page each cites"}
    notes = (
        f"jev citation check over {total} claims ({what.get(target, target)})"
        + (f" ({dropped} more past VERIFY_JEV_MAX_CLAIMS not checked)" if dropped else "")
        + f": {verified} verified, {len(unsupported)} unsupported, {len(contradictions)} "
        f"contradicted at confidence >= {auto_accept}; {len(review)} in needs_review "
        f"({below} below the threshold, {counts.get('unverifiable', 0)} with no evidence "
        f"to check against, {counts.get('unjudged', 0)} the judge failed on). Each claim "
        f"was judged against an excerpt, not the whole research.")
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
        # What was audited: "report", "context", or "report+context".
        "audit_target": target,
        # What this audit cost, read next to agent_calls_by_site: the session it did not
        # spend on one side, fractions of a cent on the other.
        # ponytail: delta of process-wide counters, so a concurrent jev user (the P1 gate
        # in another run) leaks into it. Upgrade path: per-call usage from jev.ask.
        "jev": {**spent, "claims": total, "not_checked": dropped,
                "auto_accept": auto_accept, "verdicts": counts,
                "verdicts_by_scope": by_scope, "below_threshold": below},
    }


class _AuditFailed(RuntimeError):
    """Neither engine produced an audit. Carries why jev did not do the work, so the
    response can name it in jev_fallback like every other non-jev outcome."""

    def __init__(self, message, jev_fallback):
        super().__init__(message)
        self.jev_fallback = jev_fallback


def _embeddings_of(researcher):
    """The researcher's configured embedder (the one the jev gate and the s12 router
    use), or None. Never raises: no embedder means lexical retrieval, not no audit."""
    try:
        return researcher.memory.get_embeddings()
    except Exception:                                          # noqa: BLE001
        return None


async def audit_faithfulness(researcher, query, context, sources, report=None):
    """Unsupported-claim + contradiction audit: jev citation check first, LLM pass second.

    `report` is the synthesis written from `context`. Given one, its claims are what get
    audited; without one (synthesis failed, or a caller holding only research), the
    context is, as before. The jev path costs no `claude` session; the LLM path costs
    one. When jev is off, erroring, or has nothing it can judge, this is exactly one LLM
    pass, and that fallback is the point: "fail open" here means "run the check we had",
    never "skip the check", because a skipped audit and a clean one look identical
    downstream.
    """
    report = report if isinstance(report, str) and report.strip() else None
    try:
        jev_out, why_not = await _audit_with_jev(
            context, sources, report=report,
            embeddings=_embeddings_of(researcher) if report else None)
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
        result = await _audit_with_llm(researcher, query, context, sources, report=report)
    except Exception as e:
        # Keep why jev did not run in the audit_error verify_research ships: "the LLM
        # audit failed" and "both audits failed" call for different fixes.
        raise _AuditFailed(f"{e} [jev path not taken: {why_not}]", why_not) from e
    result["audit_engine"] = "llm"
    result["audit_target"] = target = "report" if report else "context"
    result["jev_fallback"] = why_not
    result["audit_scopes"] = scope_status(report, context, {
        target: {"status": "audited", "engine": "llm", "reason": None}})
    return result


#: The LLM fallback's question when there IS a report: the report's claims are audited
#: and the context is the evidence. The rules about what the auditor can see are the
#: context audit's own, carried over, with one addition: a report claim the context
#: refutes is a contradiction the auditor can actually see, so it may be reported.
_VERIFY_REPORT_SYSTEM = (
    "You audit a research REPORT for FAITHFULNESS to the research it was written from. "
    "You do NOT add new facts. Given the QUERY, the REPORT, the gathered CONTEXT it was "
    "written from, and the SOURCE list, you "
    "(1) flag claims in the REPORT that the CONTEXT does not support, "
    "(2) flag CONTRADICTIONS, (3) give an overall confidence 0-1. "
    "Be specific; quote the claim.\n"
    "WHAT YOU CAN SEE: CONTEXT is an EXCERPT of the research and may be truncated "
    "mid-evidence; REPORT may be cut short too. SOURCES is a list of urls and titles ONLY "
    "- you are NOT given the text of those pages. Report a contradiction ONLY when the "
    "CONTEXT itself states both sides, or when the REPORT states one side and the CONTEXT "
    "explicitly states the other.\n"
    + _VERIFY_SYSTEM[_VERIFY_SYSTEM.index("ABSENCE IS NOT DISAGREEMENT"):]
)


async def _audit_with_llm(researcher, query, context, sources, report=None):
    """One LLM pass: unsupported-claim + contradiction audit. Reuses the researcher's
    SMART LLM (their claude_agent subscription -> no extra metered API cost)."""
    from gpt_researcher.utils.llm import create_chat_completion

    try:
        from gpt_researcher.utils.agent_purpose import agent_purpose
    except ImportError:  # pragma: no cover - a fork without the attribution
        from contextlib import nullcontext

        def agent_purpose(_site):
            return nullcontext()

    full = context if isinstance(context, str) else str(context)
    ctx = full[:12000]  # ponytail: bound the evidence fed into one call
    srcs = json.dumps(_compact_sources(sources), ensure_ascii=False)
    if report:
        # ponytail: the first 24k chars of the report against the first 12k of the
        # context. Most of a 160k-char context is unseen here, so expect not-in-excerpt;
        # this is the fallback, and the jev path is the one that retrieves per claim.
        rep, system = report[:24000], _VERIFY_REPORT_SYSTEM
        user = (f"QUERY:\n{query}\n\nREPORT (the claims to audit):\n{rep}\n\n"
                f"CONTEXT (evidence the report was written from):\n{ctx}\n\n"
                f"SOURCES:\n{srcs}")
    else:
        rep, system = "", _VERIFY_SYSTEM
        user = f"QUERY:\n{query}\n\nCONTEXT (evidence):\n{ctx}\n\nSOURCES:\n{srcs}"
    with agent_purpose("verify"):
        resp = await create_chat_completion(
            model=researcher.cfg.smart_llm_model,
            messages=[
                {"role": "system", "content": system},
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
        "evidence_truncated": len(full) > len(ctx) or len(report or "") > len(rep),
    }


async def verify_research(researcher, query, context, sources, report=None):
    """Full bundle: deterministic tiering + faithfulness audit. Tiering always ships;
    audit failure degrades to tiers-only (best-effort, never blocks research).

    Pass `report`, the synthesis, whenever there is one: it is what the audit checks.
    """
    source_urls = [s.get("url", "") for s in (sources or []) if s.get("url")]
    tiers, tier_summary = tier_sources(source_urls)
    result = {"source_tiers": tiers, "tier_summary": tier_summary}
    try:
        result.update(await audit_faithfulness(researcher, query, context, sources,
                                               report=report))
    except Exception as e:  # audit is best-effort; tiering still ships
        logger.warning(f"faithfulness audit failed: {e}")
        # Three outcomes stay distinguishable: audited (audit_engine jev/llm, the finding
        # keys present), could not audit (this: audit_engine "none", audit_error, no
        # finding keys), never ran (verification is null). jev_fallback is set whenever
        # jev did not do the work. The same three hold per scope in audit_scopes.
        result.update(audit_error=str(e), audit_engine="none",
                      jev_fallback=getattr(e, "jev_fallback", None)
                      or f"the audit raised before an engine ran: {type(e).__name__}")
        target = "report" if isinstance(report, str) and report.strip() else "context"
        result["audit_scopes"] = scope_status(report, context, {target: {
            "status": "could_not_audit", "engine": None, "reason": str(e)}})
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
    # report claim split: headings, banner, questions, references are not claims; a table
    # row carries its column names; a citation is metadata, not claim text
    rc = extract_report_claims(
        "> **Partial research.** This research stopped early.\n\n# Title\n\n"
        "Stripe keeps keys for 24 hours ([Stripe, 2024](https://docs.stripe.com/x)). "
        "Why does this matter?\n\n| Property | v1 |\n|---|---|\n| Window | 24 hours |\n\n"
        "## References\n\nStripe. (2024). Docs. https://docs.stripe.com/x")
    assert [c["claim"] for c in rc] == ["Stripe keeps keys for 24 hours.",
                                        "Property: Window; v1: 24 hours"], rc
    assert rc[0]["urls"] == ["https://docs.stripe.com/x"], rc
    print("verification.py self-check OK")


if __name__ == "__main__":
    demo()
