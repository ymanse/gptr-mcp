"""
GPT Researcher MCP Server

This script implements an MCP server for GPT Researcher, allowing AI assistants
to conduct web research and generate reports via the MCP protocol.
"""

import os
import sys
import uuid
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastmcp import FastMCP
from gpt_researcher import GPTResearcher

# Load environment variables
load_dotenv()

from utils import (
    research_store,
    create_success_response,
    handle_exception,
    get_researcher_by_id,
    format_sources_for_response,
    format_context_with_sources,
    store_research_results,
    create_research_prompt,
    normalize_context,
    persist_research_artifacts,
    build_report_markdown,
    persist_search_results,
)
from tiers import DEFAULT_TIER, TIERS, resolve as resolve_tier
from verification import verify_research


def _with_refusal(result: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the provider's own words when it refused.

    Two very different failures render as the same "Failed to get response from
    claude_agent API": this run spending its allowance, and the ACCOUNT hitting its
    session limit. On 2026-09-10 that ambiguity sent a debugging session into the code
    before anyone opened the container log, where the CLI had said plainly:
    "You've hit your session limit · resets 5:50am (UTC)". Say it here instead.
    """
    refused, reason = provider_refused()
    if refused:
        result["provider_refused"] = True
        result["provider_refusal_reason"] = reason
        result["hint"] = (
            "This is the Claude subscription's own limit, not a bug in the research "
            "server and not this call's budget: " + reason
        )
    return result

# Per-run cap on `claude` CLI sessions. With FAST/SMART/STRATEGIC on claude_agent every
# LLM call spawns a CLI subprocess that registers as its own Claude Code session, so a
# single tool call could run into the hundreds (measured 2026-08-05: 42 sessions in one
# 10-minute research round). Each tool arms the budget on entry — that call IS the "run".
# Guarded: on the OpenRouter rollback path the provider module need not be importable,
# and a research server that will not start is worse than an unbounded one.
try:
    from gpt_researcher.llm_provider.claude_agent._subscription import (
        agent_calls_by_site,
        agent_calls_spent,
        agent_calls_this_run,
        begin_agent_run,
        provider_refused,
    )
except ImportError:  # pragma: no cover - rollback path
    def begin_agent_run(limit=None) -> int:
        return 0

    def agent_calls_spent() -> int:
        return 0

    def agent_calls_this_run() -> int:
        return 0

    def agent_calls_by_site() -> dict:
        return {}

    def provider_refused() -> tuple:
        return False, ""

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] - %(message)s',
)

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(
    name="GPT Researcher"
)

# Initialize researchers dictionary
if not hasattr(mcp, "researchers"):
    mcp.researchers = {}


@mcp.resource("research://{topic}")
async def research_resource(topic: str) -> str:
    """
    Provide research context for a given topic directly as a resource.
    
    This allows LLMs to access web-sourced information without explicit function calls.
    
    Args:
        topic: The research topic or query
        
    Returns:
        String containing the research context with source information
    """
    # Check if we've already researched this topic
    if topic in research_store:
        logger.info(f"Returning cached research for topic: {topic}")
        return research_store[topic]["context"]
    
    # If not, conduct the research
    logger.info(f"Conducting new research for resource on topic: {topic}")
    
    # Initialize GPT Researcher
    researcher = GPTResearcher(topic)
    
    try:
        # Conduct the research
        await researcher.conduct_research()
        
        # Get the context and sources
        context = researcher.get_research_context()
        sources = researcher.get_research_sources()
        source_urls = researcher.get_source_urls()
        
        # Format with sources included
        formatted_context = format_context_with_sources(topic, context, sources)
        
        # Store for future use
        store_research_results(topic, context, sources, source_urls, formatted_context)
        
        return formatted_context
    except Exception as e:
        return f"Error conducting research on '{topic}': {str(e)}"


# The tree tool's parameter defaults predate tiers, so a caller who says nothing is
# indistinguishable from one who restates the old default. Comparing against those
# defaults is how the tier gets to fill in the rest without overriding a real choice.
_TREE_ARG_DEFAULTS = {"max_depth": 3, "max_breadth": 4, "max_nodes": 20,
                      "time_budget_s": 600}


def _explicit_tree_args(max_depth, max_breadth, max_nodes, time_budget_s) -> dict:
    """Only the tree arguments whose value differs from the pre-tier default."""
    given = {"max_depth": max_depth, "max_breadth": max_breadth,
             "max_nodes": max_nodes, "time_budget_s": time_budget_s}
    return {k: v for k, v in given.items() if v != _TREE_ARG_DEFAULTS[k]}


def _partial_banner(researcher) -> str:
    """A disclosure at the TOP of a report whose research was cut short, or "".

    Deterministic on purpose. The research context already ends in an "Incomplete
    Research" notice, and the report is written from that context -- but measured
    2026-09-21, a report written from a time-truncated 25k-word context never mentioned
    it: the notice sat at the end of the input and the writer did not relay it. A
    partial report that reads as complete is the failure the notice exists to prevent,
    so it is stated here, where no model can drop it.
    """
    skill = getattr(researcher, "deep_researcher", None)
    reasons = []
    if getattr(skill, "time_exhausted", False):
        reasons.append("its time budget ran out")
    if getattr(skill, "budget_exhausted", False):
        reasons.append("its LLM call budget ran out")
    if not reasons:
        return ""
    return ("> **Partial research.** This research stopped early because "
            + " and ".join(reasons)
            + ", before every planned question was investigated. The report below is "
              "written only from what was gathered: a topic it does not cover was not "
              "examined, which is not evidence that it does not matter.\n\n")


@mcp.tool()
async def deep_research(query: str, retriever: str = "smart", multi_llm_review: bool = False, verify: bool = False, scope: bool = False, time_budget_s: Optional[float] = None) -> Dict[str, Any]:
    """
    Conduct a web deep research on a given query using GPT Researcher.
    Use this tool when you need time-sensitive, real-time information like stock prices, news, people, specific knowledge, etc.

    Args:
        query: The research query or topic
        retriever: Search retriever to use (e.g. "smart", "tavily", "duckduckgo"). Defaults to "smart" which auto-selects optimal retrievers per query type.
        multi_llm_review: Enable multi-LLM consensus review (Gemini + ChatGPT + Claude review research for gaps and deeper exploration). Defaults to False.
        verify: Run a post-research verification pass — source tiering (L1-L4 credibility) + faithfulness audit (unsupported-claim + contradiction detection). Complements multi_llm_review (which fills gaps). Off by default; also enabled globally via GPTR_MCP_VERIFY=true.
        scope: Build a 1-round scope brief (clarification questions resolved into a scope statement) before researching, instead of auto-proceeding. Defaults to False.
        time_budget_s: Wall-clock ceiling for the whole run, in seconds. Omit to use
            DEEP_RESEARCH_TIME_BUDGET_S from the server .env (600). Exceeding it returns
            a shallower report marked partial (time_budget_exhausted=true), not an
            error. 0 = unbounded (the pre-2026-09-21 behaviour).

    Returns:
        Dict containing research status, ID, and the actual research context and sources
        that can be used directly by LLMs for context enrichment
    """
    logger.info(f"Conducting research on query: {query} (retriever={retriever}, multi_llm_review={multi_llm_review})...")
    # A linear deep run measured 43 CLI sessions (2026-09-02, two runs in one
    # window). Asking for the "deep" tier rather than the env default of 100 is what
    # stops concurrent calls starving each other out of one pooled ceiling.
    begin_agent_run(TIERS["deep"]["max_calls"])

    # Save and set environment variables (GPTResearcher reads config from env,
    # NOT as constructor kwargs — passing them as kwargs leaks into LLM API calls)
    prev_retriever = os.environ.get("RETRIEVER")
    prev_multi_llm = os.environ.get("MULTI_LLM_REVIEW_ENABLED")
    prev_time_budget = os.environ.get("DEEP_RESEARCH_TIME_BUDGET_S")
    # DeepResearchSkill reads this off cfg at GPTResearcher construction, and Config
    # takes it from the environment -- same reason RETRIEVER is set this way rather
    # than passed as a kwarg. Only when the caller asked: otherwise the .env value
    # stands, so the operator knob is not silently shadowed on every call.
    if time_budget_s is not None:
        os.environ["DEEP_RESEARCH_TIME_BUDGET_S"] = str(float(time_budget_s))

    if multi_llm_review:
        os.environ["MULTI_LLM_REVIEW_ENABLED"] = "true"
    if retriever:
        os.environ["RETRIEVER"] = retriever

    # Generate a unique ID for this research session
    research_id = str(uuid.uuid4())

    # Initialize GPT Researcher in deep mode: report_type="deep" routes
    # conduct_research() through DeepResearchSkill (iterative breadth/depth research
    # with learnings extraction). The default report_type only does flat web research.
    researcher = GPTResearcher(query, report_type="deep")

    # Restore previous env vars to avoid side effects between calls
    for key, prev_val in [("RETRIEVER", prev_retriever), ("MULTI_LLM_REVIEW_ENABLED", prev_multi_llm),
                          ("DEEP_RESEARCH_TIME_BUDGET_S", prev_time_budget)]:
        if prev_val is not None:
            os.environ[key] = prev_val
        elif key in os.environ:
            del os.environ[key]

    # Start research
    try:
        await researcher.conduct_research(scope=scope)
        mcp.researchers[research_id] = researcher
        logger.info(f"Research completed for ID: {research_id}")

        # Get the research context and sources. get_research_context() can return a
        # list (see gpt_researcher agent), which breaks FastMCP serialization — normalize.
        context = normalize_context(researcher.get_research_context())
        sources = researcher.get_research_sources()
        source_urls = researcher.get_source_urls()

        # Optional post-research verification (source tiering + faithfulness audit).
        # Off by default; enable per-call (verify=true) or globally (GPTR_MCP_VERIFY=true).
        verification = None
        if verify or os.getenv("GPTR_MCP_VERIFY", "false").strip().lower() in ("1", "true", "yes"):
            try:
                verification = await verify_research(researcher, query, context, sources)
            except Exception as e:
                logger.warning(f"Verification pass failed: {e}")

        # ALWAYS synthesize -- a time-truncated run above all. Measured 2026-09-21: a
        # run that spent its 600s budget came back as a 284KB context dump with no
        # report, because synthesis lived only in the separate write_report tool: a
        # second round trip the caller had to remember, keyed on an in-memory
        # researcher that a container restart loses. The research budget bounds
        # EXPLORATION; this runs after it, so a cut-short run still ends in a report
        # -- and the context it is written from carries the "Incomplete Research"
        # notice, so the report says it is partial rather than passing as complete.
        #
        # A synthesis failure must never cost the research: the context is still
        # persisted and returned below, and the failure is named in the response.
        # write_report stays available to re-synthesize with a custom prompt.
        report, report_error = None, None
        try:
            report = await researcher.write_report()
        except Exception as e:
            report_error = f"{type(e).__name__}: {e}"
            logger.warning(f"Synthesis failed, returning the research without a report: "
                           f"{report_error}")

        # Store in the research store for the resource API
        store_research_results(query, context, sources, source_urls)

        # Artifact pattern: write the full context + sources to disk and return only a
        # compact preview + file paths, so the (potentially 70KB+) context never gets
        # dumped into the model context as a single-line JSON blob. With a report, it
        # goes FIRST and the raw research follows it -- the preview is then the
        # synthesis, and the evidence it was written from is still in the same file.
        report_md = None
        if report:
            report_md = (_partial_banner(researcher) + f"{report}\n\n---\n\n"
                         + build_report_markdown(query, context, sources, source_urls,
                                                 research_id))
        artifacts = persist_research_artifacts(
            query, context, sources, source_urls, research_id,
            verification=verification, report_text=report_md,
        )

        return create_success_response({
            "research_id": research_id,
            "query": query,
            "source_count": len(sources),
            "sources": format_sources_for_response(sources),
            "source_urls": source_urls,
            "verification": verification,
            # agent_calls is the pooled process total; these two are what THIS
            # call spent and which of the seven sites spent it.
            "agent_calls": agent_calls_spent(),
            "agent_calls_this_run": agent_calls_this_run(),
            "agent_calls_by_site": agent_calls_by_site(),
            # The context carries the "Incomplete Research" notice too, but the
            # caller reads a preview; this is where it can tell without opening it.
            "time_budget_exhausted": bool(getattr(researcher.deep_researcher,
                                                  "time_exhausted", False)),
            # False means report_path holds the raw research only; report_error says why.
            "report_written": bool(report),
            "report_error": report_error,
            **artifacts,
        })
    except Exception as e:
        return _with_refusal(handle_exception(e, "Research"))


@mcp.tool()
async def deep_tree_research(
    query: str,
    max_depth: int = 3,
    max_breadth: int = 4,
    max_nodes: int = 20,
    token_budget: int = 300_000,
    credit_budget: float = 150,
    novelty_threshold: float = 0.20,
    expansion_policy: str = "best_first",
    stream: bool = False,
    time_budget_s: float = 600,
    depth: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Conduct tree-structured deep research: the answer to each research question spawns
    follow-up questions expanded best-first into an explicit persisted node tree, with
    novelty pruning, question-embedding dedup and post-order hierarchical synthesis.
    Use this for broad, multi-faceted topics where a single linear deep_research pass
    would miss follow-up threads.

    Args:
        query: The root research question
        max_depth: Maximum tree depth below the root (default 3)
        max_breadth: Maximum children accepted per node (default 4)
        max_nodes: Cap on researched nodes; leftover nodes stay pending (default 20).
            Lowered from 40 on 2026-07-28: benchmark runs were reaching 24-30 researched
            nodes, and each node costs a search plus its scrapes (~5,800 Firecrawl credits
            per 5-query benchmark round). 20 keeps the measured S4/S5 headroom while
            cutting roughly a third of the retrieval spend. Raise it per-call for a
            deliberately exhaustive run.
        token_budget: Approximate total token budget (default 300000)
        credit_budget: Search/scrape credit budget (default 150)
        novelty_threshold: Nodes with novelty below this are pruned, never expanded (default
            0.20, i.e. cosine similarity to already-covered ground must exceed 0.80 to prune).
            Deliberately stricter than TreeResearchSkill.run's own 0.30 library default: a
            root query that names several subtopics inflates cosine between the root and any
            child drilling into one of them, so 0.30 (cosine > 0.70) pruned live children whose
            content was genuinely new -- observed: a bun-rust-port child specifically about
            what unsafe patterns adversarial reviewers caught was pruned and its findings
            never reached the report, even though the root only mentioned reviewers in passing.
        expansion_policy: "best_first" (default), "bfs" or "dfs"
        stream: Reserved for streaming progress events (default False)
        time_budget_s: Wall-clock seconds for tree EXPANSION (default 600). Nodes are
            researched in concurrent batches (node_concurrency, ~45s per node), so this —
            not max_nodes — is what keeps a run inside the caller's MCP idle timeout;
            leftover nodes stay pending and the report is still synthesized. The roll-up
            afterwards is pure text assembly (no LLM), so total runtime tracks this value.
            Raise it only alongside a raised client-side idle timeout.
        depth: "light" | "standard" | "deep". Sets max_nodes/max_depth/max_breadth,
            time_budget_s, node_concurrency AND — the part that matters — how large an
            allowance of `claude` CLI sessions this call asks for. Allowances POOL across
            concurrent calls, so a call that asks for more than it needs starves its
            siblings: measured 2026-09-02, seven runs armed within twelve minutes shared
            one ceiling of 171 and each researched a SINGLE node of the twenty it wanted.
            Any explicit parameter above overrides the tier's value for that parameter.
            Omitted, the tier's own defaults apply and the tier is "standard".

    Returns:
        Dict with research status, stats, citation count and host paths of the persisted
        tree.json + final report markdown artifacts.
    """
    from gpt_researcher.skills.tree_research import TreeResearchSkill
    from utils import get_output_dir, _to_host_path

    # A tier is a REQUEST for an allowance, sized from what that depth measurably costs.
    # Explicit arguments still win: the tier fills in what the caller did not state, so a
    # caller who names max_nodes gets exactly that, with a budget scaled to the tier.
    tier_name, preset = resolve_tier(depth)
    explicit = _explicit_tree_args(max_depth, max_breadth, max_nodes, time_budget_s)
    max_depth = explicit.get("max_depth", preset["max_depth"])
    max_breadth = explicit.get("max_breadth", preset["max_breadth"])
    max_nodes = explicit.get("max_nodes", preset["max_nodes"])
    time_budget_s = explicit.get("time_budget_s", preset["time_budget_s"])

    logger.info(f"Starting deep_tree_research on: {query} "
                f"(tier={tier_name}, max_depth={max_depth}, max_nodes={max_nodes}, "
                f"allowance={preset['max_calls']})")
    # The tree checks this budget between node batches, so a spent allowance stops
    # expansion and still synthesizes — see stats.agent_calls_spent in the result.
    begin_agent_run(preset["max_calls"])
    research_id = str(uuid.uuid4())
    researcher = GPTResearcher(query)
    skill = TreeResearchSkill(researcher)

    try:
        out_dir = get_output_dir()
        result = await skill.run(
            query=query, max_depth=max_depth, max_breadth=max_breadth,
            max_nodes=max_nodes, token_budget=token_budget, credit_budget=credit_budget,
            node_concurrency=preset["node_concurrency"],
            novelty_threshold=novelty_threshold, expansion_policy=expansion_policy,
            stream=stream, outputs_dir=str(out_dir), time_budget_s=time_budget_s,
        )
        mcp.researchers[research_id] = researcher

        # Artifact pattern: tree + report are on disk; return host-visible paths only.
        artifacts = result.get("artifacts", {})
        tree_name = os.path.basename(artifacts.get("tree_json", "")) or None
        report_name = os.path.basename(artifacts.get("report_md", "")) or None
        return create_success_response({
            "research_id": research_id,
            "query": query,
            "tier": tier_name,
            "stats": result["stats"],
            "citation_count": len(result.get("citation_map", {})),
            "tree_json_path": _to_host_path(tree_name, out_dir) if tree_name else None,
            "report_path": _to_host_path(report_name, out_dir) if report_name else None,
        })
    except Exception as e:
        return _with_refusal(handle_exception(e, "Tree research"))


@mcp.tool()
async def quick_search(query: str) -> Dict[str, Any]:
    """
    Perform a quick web search on a given query and return search results with snippets.
    This optimizes for speed over quality and is useful when an LLM doesn't need in-depth
    information on a topic.

    Bodies are snippets (see snippet_chars); any result whose body was cut is flagged
    with body_truncated and its full length. The COMPLETE, untruncated results are
    written to results_path — open that file when a snippet is not enough, rather than
    re-running the search.

    Args:
        query: The search query

    Returns:
        Dict with search_results (snippets), results_path (full results on disk),
        result_count, truncated_results and body_chars_total
    """
    logger.info(f"Performing quick search on query: {query}...")
    begin_agent_run(TIERS["light"]["max_calls"])  # measured: 1 session

    # Generate a unique ID for this search session
    search_id = str(uuid.uuid4())
    
    # Initialize GPT Researcher
    researcher = GPTResearcher(query)
    
    try:
        # Perform quick search
        search_results = await researcher.quick_search(query=query)
        mcp.researchers[search_id] = researcher
        logger.info(f"Quick search completed for ID: {search_id}")
        
        # Artifact pattern, as in write_report: full results to disk, snippets inline.
        # Returning them raw exceeded the MCP client's output ceiling (measured: 9
        # results -> 100,159 chars), which spills the WHOLE reply to a file the caller
        # then has to read back — and a research run that does not read it proceeds on
        # a fraction of what it asked for, silently.
        return create_success_response({
            "search_id": search_id,
            "query": query,
            # agent_calls is the pooled process total; these two are what THIS
            # call spent and which of the seven sites spent it.
            "agent_calls": agent_calls_spent(),
            "agent_calls_this_run": agent_calls_this_run(),
            "agent_calls_by_site": agent_calls_by_site(),
            **persist_search_results(query, search_results, search_id),
        })
    except Exception as e:
        return _with_refusal(handle_exception(e, "Quick search"))


@mcp.tool()
async def write_report(research_id: str, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a report based on previously conducted research.
    
    Args:
        research_id: The ID of the research session from deep_research
        custom_prompt: Optional custom prompt for report generation
        
    Returns:
        Dict containing the report content and metadata
    """
    success, researcher, error = get_researcher_by_id(mcp.researchers, research_id)
    if not success:
        return error
    
    logger.info(f"Generating report for research ID: {research_id}")
    begin_agent_run(TIERS["standard"]["max_calls"])

    try:
        # Generate report
        report = await researcher.write_report(custom_prompt=custom_prompt)

        # Get additional information
        sources = researcher.get_research_sources()
        source_urls = researcher.get_source_urls()
        costs = researcher.get_costs()
        query = getattr(researcher, "query", research_id)

        # Artifact pattern: the report (already Markdown) is written to disk verbatim;
        # only a preview + paths are returned inline.
        artifacts = persist_research_artifacts(
            query, report, sources, source_urls, research_id,
            report_text=report, costs=costs,
        )

        return create_success_response({
            "research_id": research_id,
            "source_count": len(sources),
            "costs": costs,
            # agent_calls is the pooled process total; these two are what THIS
            # call spent and which of the seven sites spent it.
            "agent_calls": agent_calls_spent(),
            "agent_calls_this_run": agent_calls_this_run(),
            "agent_calls_by_site": agent_calls_by_site(),
            **artifacts,
        })
    except Exception as e:
        return _with_refusal(handle_exception(e, "Report generation"))


@mcp.tool()
async def get_research_sources(research_id: str) -> Dict[str, Any]:
    """
    Get the sources used in the research.
    
    Args:
        research_id: The ID of the research session
        
    Returns:
        Dict containing the research sources
    """
    success, researcher, error = get_researcher_by_id(mcp.researchers, research_id)
    if not success:
        return error
    
    sources = researcher.get_research_sources()
    source_urls = researcher.get_source_urls()
    
    return create_success_response({
        "sources": format_sources_for_response(sources),
        "source_urls": source_urls
    })


@mcp.tool()
async def get_research_context(research_id: str) -> Dict[str, Any]:
    """
    Get the full context of the research.
    
    Args:
        research_id: The ID of the research session
        
    Returns:
        Dict containing the research context
    """
    success, researcher, error = get_researcher_by_id(mcp.researchers, research_id)
    if not success:
        return error
    
    context = normalize_context(researcher.get_research_context())
    sources = researcher.get_research_sources()
    source_urls = researcher.get_source_urls()
    query = getattr(researcher, "query", research_id)

    # Artifact pattern: persist the full context and return a preview + paths instead of
    # dumping the whole context inline. Set GPTR_MCP_INLINE_CONTEXT=true to get it inline.
    artifacts = persist_research_artifacts(
        query, context, sources, source_urls, research_id
    )

    return create_success_response({
        "research_id": research_id,
        "source_count": len(sources),
        **artifacts,
    })


@mcp.prompt()
def research_query(topic: str, goal: str, report_format: str = "research_report") -> str:
    """
    Create a research query prompt for GPT Researcher.
    
    Args:
        topic: The topic to research
        goal: The goal or specific question to answer
        report_format: The format of the report to generate
        
    Returns:
        A formatted prompt for research
    """
    return create_research_prompt(topic, goal, report_format)

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    return JSONResponse({"status": "healthy", "service": "mcp-server"})

def run_server():
    """Run the MCP server using FastMCP's built-in event loop handling."""
    # Check if API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found. Please set it in your .env file.")
        return

    # Determine transport based on environment
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower().replace("_", "-")
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8000"))
    http_path = os.getenv("MCP_PATH")
    
    # Auto-detect Docker environment (only override if transport is default stdio)
    if transport == "stdio" and (os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER")):
        transport = "streamable-http"
        logger.info("Docker environment detected, using Streamable HTTP transport")

    if transport == "http":
        transport = "streamable-http"
    
    # Add startup message
    logger.info(f"Starting GPT Researcher MCP Server with {transport} transport...")
    print(f"🚀 GPT Researcher MCP Server starting with {transport} transport...")
    print("   Check researcher_mcp_server.log for details")

    # Let FastMCP handle the event loop
    try:
        if transport == "stdio":
            logger.info("Using STDIO transport (Claude Desktop compatible)")
            mcp.run(transport="stdio")
        elif transport == "sse":
            mcp.run(transport="sse", host=host, port=port, path=http_path or "/sse")
        elif transport == "streamable-http":
            mcp.run(transport="streamable-http", host=host, port=port, path=http_path or "/mcp")
        else:
            raise ValueError(f"Unsupported transport: {transport}")
            
        # Note: If we reach here, the server has stopped
        logger.info("MCP Server is running...")
        while True:
            pass  # Keep the process alive
    except Exception as e:
        logger.error(f"Error running MCP server: {str(e)}")
        print(f"❌ MCP Server error: {str(e)}")
        return
        
    print("✅ MCP Server stopped")


if __name__ == "__main__":
    # Use the non-async approach to avoid asyncio nesting issues
    run_server()
