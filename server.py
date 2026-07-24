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
)
from verification import verify_research

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


@mcp.tool()
async def deep_research(query: str, retriever: str = "smart", multi_llm_review: bool = False, verify: bool = False, scope: bool = False) -> Dict[str, Any]:
    """
    Conduct a web deep research on a given query using GPT Researcher.
    Use this tool when you need time-sensitive, real-time information like stock prices, news, people, specific knowledge, etc.

    Args:
        query: The research query or topic
        retriever: Search retriever to use (e.g. "smart", "tavily", "duckduckgo"). Defaults to "smart" which auto-selects optimal retrievers per query type.
        multi_llm_review: Enable multi-LLM consensus review (Gemini + ChatGPT + Claude review research for gaps and deeper exploration). Defaults to False.
        verify: Run a post-research verification pass — source tiering (L1-L4 credibility) + faithfulness audit (unsupported-claim + contradiction detection). Complements multi_llm_review (which fills gaps). Off by default; also enabled globally via GPTR_MCP_VERIFY=true.
        scope: Build a 1-round scope brief (clarification questions resolved into a scope statement) before researching, instead of auto-proceeding. Defaults to False.

    Returns:
        Dict containing research status, ID, and the actual research context and sources
        that can be used directly by LLMs for context enrichment
    """
    logger.info(f"Conducting research on query: {query} (retriever={retriever}, multi_llm_review={multi_llm_review})...")

    # Save and set environment variables (GPTResearcher reads config from env,
    # NOT as constructor kwargs — passing them as kwargs leaks into LLM API calls)
    prev_retriever = os.environ.get("RETRIEVER")
    prev_multi_llm = os.environ.get("MULTI_LLM_REVIEW_ENABLED")

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
    for key, prev_val in [("RETRIEVER", prev_retriever), ("MULTI_LLM_REVIEW_ENABLED", prev_multi_llm)]:
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

        # Store in the research store for the resource API
        store_research_results(query, context, sources, source_urls)

        # Artifact pattern: write the full context + sources to disk and return only a
        # compact preview + file paths, so the (potentially 70KB+) context never gets
        # dumped into the model context as a single-line JSON blob.
        artifacts = persist_research_artifacts(
            query, context, sources, source_urls, research_id, verification=verification
        )

        return create_success_response({
            "research_id": research_id,
            "query": query,
            "source_count": len(sources),
            "sources": format_sources_for_response(sources),
            "source_urls": source_urls,
            "verification": verification,
            **artifacts,
        })
    except Exception as e:
        return handle_exception(e, "Research")


@mcp.tool()
async def deep_tree_research(
    query: str,
    max_depth: int = 3,
    max_breadth: int = 4,
    max_nodes: int = 40,
    token_budget: int = 300_000,
    credit_budget: float = 150,
    novelty_threshold: float = 0.30,
    expansion_policy: str = "best_first",
    stream: bool = False,
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
        max_nodes: Cap on researched nodes; leftover nodes stay pending (default 40)
        token_budget: Approximate total token budget (default 300000)
        credit_budget: Search/scrape credit budget (default 150)
        novelty_threshold: Nodes with novelty below this are pruned, never expanded (default 0.30)
        expansion_policy: "best_first" (default), "bfs" or "dfs"
        stream: Reserved for streaming progress events (default False)

    Returns:
        Dict with research status, stats, citation count and host paths of the persisted
        tree.json + final report markdown artifacts.
    """
    from gpt_researcher.skills.tree_research import TreeResearchSkill
    from utils import get_output_dir, _to_host_path

    logger.info(f"Starting deep_tree_research on: {query} (max_depth={max_depth}, max_nodes={max_nodes})")
    research_id = str(uuid.uuid4())
    researcher = GPTResearcher(query)
    skill = TreeResearchSkill(researcher)

    try:
        out_dir = get_output_dir()
        result = await skill.run(
            query=query, max_depth=max_depth, max_breadth=max_breadth,
            max_nodes=max_nodes, token_budget=token_budget, credit_budget=credit_budget,
            novelty_threshold=novelty_threshold, expansion_policy=expansion_policy,
            stream=stream, outputs_dir=str(out_dir),
        )
        mcp.researchers[research_id] = researcher

        # Artifact pattern: tree + report are on disk; return host-visible paths only.
        artifacts = result.get("artifacts", {})
        tree_name = os.path.basename(artifacts.get("tree_json", "")) or None
        report_name = os.path.basename(artifacts.get("report_md", "")) or None
        return create_success_response({
            "research_id": research_id,
            "query": query,
            "stats": result["stats"],
            "citation_count": len(result.get("citation_map", {})),
            "tree_json_path": _to_host_path(tree_name, out_dir) if tree_name else None,
            "report_path": _to_host_path(report_name, out_dir) if report_name else None,
        })
    except Exception as e:
        return handle_exception(e, "Tree research")


@mcp.tool()
async def quick_search(query: str) -> Dict[str, Any]:
    """
    Perform a quick web search on a given query and return search results with snippets.
    This optimizes for speed over quality and is useful when an LLM doesn't need in-depth
    information on a topic.
    
    Args:
        query: The search query
        
    Returns:
        Dict containing search results and snippets
    """
    logger.info(f"Performing quick search on query: {query}...")
    
    # Generate a unique ID for this search session
    search_id = str(uuid.uuid4())
    
    # Initialize GPT Researcher
    researcher = GPTResearcher(query)
    
    try:
        # Perform quick search
        search_results = await researcher.quick_search(query=query)
        mcp.researchers[search_id] = researcher
        logger.info(f"Quick search completed for ID: {search_id}")
        
        return create_success_response({
            "search_id": search_id,
            "query": query,
            "result_count": len(search_results) if search_results else 0,
            "search_results": search_results
        })
    except Exception as e:
        return handle_exception(e, "Quick search")


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
            **artifacts,
        })
    except Exception as e:
        return handle_exception(e, "Report generation")


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
