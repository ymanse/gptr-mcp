"""
GPT Researcher MCP Server Utilities

This module provides utility functions and helpers for the GPT Researcher MCP Server.
"""

import os
import re
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

# Configure logging for console only (no file logging)
logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO"}])

# Research store to track ongoing research topics and contexts
research_store = {}

# Number of characters of context/report exposed inline as a preview.
PREVIEW_CHARS = int(os.getenv("GPTR_MCP_PREVIEW_CHARS", "1200"))

# API Response Utilities
def create_error_response(message: str) -> Dict[str, Any]:
    """Create a standardized error response"""
    return {"status": "error", "message": message}


def create_success_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a standardized success response"""
    return {"status": "success", **data}


def handle_exception(e: Exception, operation: str) -> Dict[str, Any]:
    """Handle exceptions in a consistent way"""
    error_message = str(e)
    logger.error(f"{operation} failed: {error_message}")
    return create_error_response(error_message)


def get_researcher_by_id(researchers_dict: Dict, research_id: str) -> Tuple[bool, Any, Dict[str, Any]]:
    """
    Helper function to retrieve a researcher by ID.
    
    Args:
        researchers_dict: Dictionary of research objects
        research_id: The ID of the research session
        
    Returns:
        Tuple containing (success, researcher_object, error_response)
    """
    if not researchers_dict or research_id not in researchers_dict:
        return False, None, create_error_response("Research ID not found. Please conduct research first.")
    return True, researchers_dict[research_id], {}


def format_sources_for_response(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format source information for API responses.
    
    Args:
        sources: List of source dictionaries
        
    Returns:
        Formatted source list for API responses
    """
    return [
        {
            "title": source.get("title", "Unknown"),
            "url": source.get("url", ""),
            "content_length": len(source.get("content") or source.get("raw_content") or "")
        }
        for source in sources
    ]


def format_context_with_sources(topic: str, context: str, sources: List[Dict[str, Any]]) -> str:
    """
    Format research context with sources for display.
    
    Args:
        topic: Research topic
        context: Research context
        sources: List of sources
        
    Returns:
        Formatted context string with sources
    """
    formatted_context = f"## Research: {topic}\n\n{context}\n\n"
    formatted_context += "## Sources:\n"
    for i, source in enumerate(sources):
        formatted_context += f"{i+1}. {source.get('title', 'Unknown')}: {source.get('url', '')}\n"
    return formatted_context


def store_research_results(topic: str, context: str, sources: List[Dict[str, Any]], 
                           source_urls: List[str], formatted_context: Optional[str] = None):
    """
    Store research results in the research store.
    
    Args:
        topic: Research topic
        context: Research context
        sources: List of sources
        source_urls: List of source URLs
        formatted_context: Optional pre-formatted context
    """
    research_store[topic] = {
        "context": formatted_context or context,
        "sources": sources,
        "source_urls": source_urls
    }


def create_research_prompt(topic: str, goal: str, report_format: str = "research_report") -> str:
    """
    Create a research query prompt for GPT Researcher.
    
    Args:
        topic: The topic to research
        goal: The goal or specific question to answer
        report_format: The format of the report to generate
        
    Returns:
        A formatted prompt for research
    """
    return f"""
    Please research the following topic: {topic}
    
    Goal: {goal}
    
    You have two methods to access web-sourced information:
    
    1. Use the "research://{topic}" resource to directly access context about this topic if it exists
       or if you want to get straight to the information without tracking a research ID.
       
    2. Use the deep_research tool to perform new research and get a research_id for later use.
       This tool also returns the context directly in its response, which you can use immediately.
    
    After getting context, you can:
    - Use it directly in your response
    - Use the write_report tool with a custom prompt to generate a structured {report_format}

    You can also use get_research_sources to view additional details about the information sources.
    """


# Artifact Persistence Utilities
#
# Large research results (context / report / sources-with-content) are written to
# disk and only a compact metadata + preview is returned inline. This keeps the MCP
# response small (no 70KB single-line JSON dumped into the model context) while the
# full output stays available as a clean, grep-able file the caller can open on demand.

def normalize_context(context_raw: Any) -> str:
    """
    Normalize a research context to a single string.

    get_research_context() may return a list (see gpt_researcher agent), which breaks
    FastMCP dict serialization ("'list' object is not an instance of 'str'"). Collapse
    any non-str shape into a stable string here.
    """
    if isinstance(context_raw, str):
        return context_raw
    if context_raw is None:
        return ""
    if isinstance(context_raw, list):
        return "\n\n".join(
            item if isinstance(item, str) else str(item) for item in context_raw
        )
    return str(context_raw)


def get_output_dir() -> Path:
    """
    Resolve (and create) the directory where research artifacts are written.

    Override with GPTR_MCP_OUTPUT_DIR; defaults to ``<server>/outputs`` next to this
    module so paths returned to the caller are absolute and stable.
    """
    base = os.getenv("GPTR_MCP_OUTPUT_DIR")
    out = Path(base) if base else (Path(__file__).resolve().parent / "outputs")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _to_host_path(filename: str, write_dir: Path) -> str:
    """
    Build the path the CALLER should open.

    When the server runs in a container, the real write dir (e.g. /app/outputs) is not
    readable from the host. GPTR_MCP_OUTPUT_DIR_HOST names the host-visible location that
    the write dir is bind-mounted to, so the returned path points there instead. Unset
    (native run) -> return the actual write path.
    """
    host_base = os.getenv("GPTR_MCP_OUTPUT_DIR_HOST")
    if not host_base:
        return str(write_dir / filename)
    host_base = host_base.rstrip("/\\")
    sep = "\\" if ("\\" in host_base or re.match(r"^[A-Za-z]:", host_base)) else "/"
    return f"{host_base}{sep}{filename}"


def _slugify(text: str, max_len: int = 40) -> str:
    """Filesystem-safe, human-readable slug for use in artifact filenames."""
    slug = re.sub(r"[^\w\s-]", "", text or "", flags=re.UNICODE).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug[:max_len].strip("-") or "research"


SNIPPET_CHARS = int(os.getenv("GPTR_MCP_SNIPPET_CHARS", "600") or 600)


def persist_search_results(
    query: str,
    search_results: Any,
    search_id: str,
) -> Dict[str, Any]:
    """Write full search results to disk; return snippets + the path.

    Same artifact pattern as persist_research_artifacts, for the same reason. A
    "quick" search is documented to return SNIPPETS, but the retrievers it fans out
    over do not agree on what a result is: engine retrievers return 170-280 character
    snippets while firecrawl returns the whole scraped page. Measured 2026-08-03 on a
    9-result search — 5 real snippets and 4 full pages (5,177 / 7,740 / 13,996 /
    20,757 chars), 48,771 characters of body that JSON escaping turned into 100,159.
    That exceeds the MCP client's output ceiling, so the ENTIRE result was spilled to
    a file the caller then had to read back in chunks; in practice it does not get
    read, and a research run silently proceeds on whatever the first chunk held.

    Bounding it here rather than at the client is the difference between paying for
    what you use and paying 100 KB to learn nine page titles. Nothing is lost: the
    untruncated bodies are on disk, and every truncated result says so and carries
    its href.

    600 chars is derived, not picked: it clears the longest engine snippet measured
    (284), so no real snippet is ever cut, and a scraped page still shows its opening.
    """
    out_dir = get_output_dir()
    stem = f"{_slugify(query)}-{search_id[:8]}"
    results = list(search_results or [])

    name = f"{stem}.search.json"
    (out_dir / name).write_text(
        json.dumps({"query": query, "search_id": search_id, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    snippets, truncated, body_chars = [], 0, 0
    for r in results:
        if not isinstance(r, dict):
            snippets.append({"body": str(r)[:SNIPPET_CHARS]})
            continue
        body = str(r.get("body") or r.get("content") or "")
        body_chars += len(body)
        row = {k: r.get(k) for k in ("title", "href", "url") if r.get(k)}
        row["body"] = body[:SNIPPET_CHARS]
        if len(body) > SNIPPET_CHARS:
            # per-result, not just in the summary: an agent reading one row has to be
            # able to see that this row is partial without correlating a global count
            row["body_truncated"] = True
            row["body_chars"] = len(body)
            truncated += 1
        snippets.append(row)

    meta: Dict[str, Any] = {
        "result_count": len(results),
        "search_results": snippets,
        "results_path": _to_host_path(name, out_dir),
        "body_chars_total": body_chars,
        "snippet_chars": SNIPPET_CHARS,
        "truncated_results": truncated,
    }
    # Escape hatch, mirroring GPTR_MCP_INLINE_CONTEXT: restore the old full-body reply.
    if os.getenv("GPTR_MCP_INLINE_SEARCH", "false").strip().lower() in ("1", "true", "yes"):
        meta["search_results"] = results
        meta["truncated_results"] = 0

    logger.info(
        f"Persisted search results: {name} ({len(results)} results, "
        f"{body_chars} body chars, {truncated} truncated inline)"
    )
    return meta


def build_report_markdown(
    query: str,
    context: str,
    sources: List[Dict[str, Any]],
    source_urls: List[str],
    research_id: str,
    costs: Any = None,
) -> str:
    """Render context + sources as a readable, grep-able Markdown report (L2)."""
    lines = [f"# Research: {query}", ""]
    lines.append(f"- research_id: `{research_id}`")
    lines.append(f"- generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- sources: {len(sources or [])}")
    if costs is not None:
        lines.append(f"- costs: {costs}")
    # The context goes in byte-for-byte. Do not "tidy" its scraped headings here. A demotion
    # of ATX headings with a ``` fence tracker was tried (2026-10-01) and removed: a context
    # is a run of arbitrary page slices, so a slice can start inside a code block whose
    # opening fence was never scraped, and then the tracker has the fences inverted. On
    # outputs/nautilustrader-built-in-reporting-and-an-a76fd491.md it rewrote Python comments
    # ("# Write a list of data objects") as headings and skipped a real one ("### [Writing
    # data]") as code. "# Write data" is the same bytes as a comment and as a heading, so no
    # regex can tell them apart. The outline it was cleaning is gone from the report anyway
    # (split=True). tests/test_report_evidence_split.py
    # ::test_the_evidence_holds_the_context_byte_for_byte fails if this is reintroduced.
    lines.extend(["", "## Context", "", context if context else "_(empty)_", "", "## Sources", ""])
    if sources:
        for i, s in enumerate(sources, 1):
            lines.append(f"{i}. [{s.get('title', 'Unknown')}]({s.get('url', '')})")
    else:
        for i, u in enumerate(source_urls or [], 1):
            lines.append(f"{i}. {u}")
    lines.append("")
    return "\n".join(lines)


def persist_research_artifacts(
    query: str,
    context: Any,
    sources: List[Dict[str, Any]],
    source_urls: List[str],
    research_id: str,
    *,
    report_text: Optional[str] = None,
    costs: Any = None,
    verification: Optional[Dict[str, Any]] = None,
    split: bool = False,
) -> Dict[str, Any]:
    """
    Write the full research output to disk and return compact inline metadata.

    Files written (under get_output_dir(), stem = ``<slug>-<short_id>``):
      - ``<stem>.md``           : the thing to read first                  [L2]
      - ``<stem>.context.md``   : the evidence -- context + sources        [L2, split only]
      - ``<stem>.sources.json`` : full sources incl. content, pretty JSON    [L3]

    split=False (the default, and what every caller got before): ONE ``<stem>.md`` --
    report_text verbatim when given, otherwise the context rendered by
    build_report_markdown.

    split=True: the evidence ALWAYS goes to ``<stem>.context.md``, and ``<stem>.md`` is
    written only when there is a report_text to put in it (with a last line naming the
    evidence file, so the link survives without the response). Why: deep_research used to
    glue synthesis and evidence into one file, and a caller told to "read report_path"
    reads all of it. Measured on empirical-evidence-on-llm-coding-agent-a-e59185e7
    (2026-10-01): 72,436 chars, of which banner + synthesis + references were 19,142
    (26.4%) and the evidence dump 53,423 (73.8%); 89-93% dump on the four larger reports
    re-split the same way (233k-375k chars). A
    separate stream measured the same summary from the synthesis alone at the same
    essential recall (12/12, 14/14) for 60-85% less. With no report_text, ``<stem>.md`` is
    left untouched -- get_research_context on a deep_research id used to overwrite the
    synthesis there with the dump.

    Returned dict (merged into the tool's success response) contains only paths,
    counts and a bounded preview -- never the full body, unless GPTR_MCP_INLINE_CONTEXT
    is truthy (escape hatch to restore the old inline behavior):
      - report_path  : read this first. The synthesis when there is one.
      - context_path : the markdown holding the evidence. Equal to report_path when the
                       two are one file (split=False with no report_text, or split=True
                       with no report_text); None when this call wrote no evidence
                       markdown (split=False with a report_text).
    """
    out_dir = get_output_dir()
    stem = f"{_slugify(query)}-{research_id[:8]}"
    context = normalize_context(context)

    report_name = f"{stem}.md"
    context_name = f"{stem}.context.md"
    sources_name = f"{stem}.sources.json"

    def evidence_md() -> str:
        return build_report_markdown(query, context, sources, source_urls, research_id,
                                     costs=costs)

    if split:
        (out_dir / context_name).write_text(evidence_md(), encoding="utf-8")
        written_context = context_name
        if report_text is not None:
            (out_dir / report_name).write_text(
                f"{report_text.rstrip()}\n\n---\n\n_The evidence this report was written "
                f"from: `{context_name}`, in the same folder._\n",
                encoding="utf-8")
            written_report = report_name
        else:
            written_report = context_name
    else:
        # L2: one file. For write_report-style callers report_text is already Markdown.
        report_md = report_text if report_text is not None else evidence_md()
        (out_dir / report_name).write_text(report_md, encoding="utf-8")
        written_report = report_name
        written_context = report_name if report_text is None else None

    # L3: full sources (with raw content) as pretty, multi-line JSON.
    sources_path = out_dir / sources_name
    sources_path.write_text(
        json.dumps(
            {
                "query": query,
                "research_id": research_id,
                "source_urls": source_urls,
                "sources": sources,
                "verification": verification,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    preview_source = report_text if report_text is not None else context
    meta: Dict[str, Any] = {
        "report_path": _to_host_path(written_report, out_dir),
        "context_path": _to_host_path(written_context, out_dir) if written_context else None,
        "sources_path": _to_host_path(sources_name, out_dir),
        "context_chars": len(context),
        "context_words": len(context.split()),
        "context_preview": preview_source[:PREVIEW_CHARS],
        "truncated": len(preview_source) > PREVIEW_CHARS,
    }

    # Escape hatch: GPTR_MCP_INLINE_CONTEXT=true restores full inline context.
    if os.getenv("GPTR_MCP_INLINE_CONTEXT", "false").strip().lower() in ("1", "true", "yes"):
        meta["context"] = context

    logger.info(
        f"Persisted research artifacts: {written_report}"
        + (f" + {written_context}" if written_context not in (None, written_report) else "")
        + " "
        f"({len(context)} chars, {len(sources or [])} sources)"
    )
    return meta