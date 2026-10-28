"""deep_research / deep_tree_research run as MCP tasks when the caller asks for one.

Measured 2026-10-28 from session transcripts: deep_research p50 121s, max 770s, every
call past 60s. As one tools/call that is one HTTP request held for the whole run. With
the io.modelcontextprotocol/tasks extension (SEP-2663) the call returns a task handle at
once and the client polls tasks/get; the research result arrives in the final poll.

What these pin:

  - the two research tools are task-"optional", quick_search is not a task at all;
  - an opted-in call returns a handle while the research is still running, tasks/get
    names the phase it is in, and the finished result is the same response the
    synchronous call gives;
  - a caller that did NOT declare the extension (a Claude Code subagent, `claude -p`)
    still gets the synchronous answer. Mode "required" would refuse those callers.

Needs fastmcp-tasks, so it runs in the gptr-mcp image, not the host venv (FastMCP 2.x):

    docker run --rm -v D:/dev_ext/gptr-mcp:/src -w /src --entrypoint sh gptr-mcp-gptr-mcp \
        -c "pip install -q pytest && python -m pytest tests/test_research_tasks.py -q"

Hermetic: GPTResearcher is replaced by a stub, the artifacts go to tmp_path.
"""
import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastmcp_tasks")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402
from fastmcp import Client  # noqa: E402
from fastmcp_tasks import call_tool_task  # noqa: E402

QUERY = "What bounds outbox table growth?"
# Long enough to clear MIN_RESEARCH_CONTEXT_CHARS, so the run takes the report path.
CONTEXT = "Teams partition the outbox by day and drop old partitions. " * 30
REPORT = "# Outbox growth\n\nTeams partition by day."


class _GatedStub:
    """GPTResearcher reduced to what deep_research reads, held mid-research.

    `gate` keeps conduct_research from returning until the test has looked at the task
    while it is still running -- otherwise the stub finishes before the first poll and
    "returns a handle while working" is never actually observed.
    """

    gate: asyncio.Event | None = None

    def __init__(self, query, report_type=None, **kwargs):
        self.query = query
        self.deep_researcher = type("DR", (), {"time_exhausted": False,
                                               "budget_exhausted": False,
                                               "time_budget_s": 600.0})()

    async def conduct_research(self, scope=False):
        if self.gate is not None:
            await self.gate.wait()
        return CONTEXT

    def get_research_context(self):
        return CONTEXT

    def get_research_sources(self):
        return [{"url": "https://example.invalid/outbox", "title": "outbox",
                 "raw_content": "partition by day"}]

    def get_source_urls(self):
        return ["https://example.invalid/outbox"]

    async def write_report(self, custom_prompt=None):
        return REPORT


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    monkeypatch.setenv("GPTR_MCP_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("GPTR_MCP_OUTPUT_DIR_HOST", raising=False)
    monkeypatch.setattr(server, "GPTResearcher", _GatedStub)
    monkeypatch.setattr(server, "begin_agent_run", lambda *a, **k: None)
    monkeypatch.setattr(server.mcp, "researchers", {}, raising=False)
    monkeypatch.setattr(_GatedStub, "gate", None)
    return tmp_path


def _payload(result) -> dict:
    """The tool's dict, whichever way the client surfaced it."""
    data = result.structured_content or {}
    return data if "status" in data else data.get("result", data)


async def _wait_for_message(task, message: str, timeout: float = 15.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        status = await task.status()
        if status.status_message == message or status.status != "working":
            return status
        if asyncio.get_running_loop().time() > deadline:
            return status
        await asyncio.sleep(0.05)


def test_research_tools_are_task_optional():
    async def modes():
        return {name: (await server.mcp.get_tool(name)).task_config.mode
                for name in ("deep_research", "deep_tree_research", "quick_search")}

    assert asyncio.run(modes()) == {"deep_research": "optional",
                                    "deep_tree_research": "optional",
                                    "quick_search": "forbidden"}


def test_opted_in_call_returns_a_handle_then_the_same_result(stubbed):
    async def run():
        _GatedStub.gate = asyncio.Event()
        async with Client(server.mcp) as client:
            # Bounded: if the call is NOT tasked, call_tool_task waits for the
            # synchronous run, which waits on a gate only this test opens -- a hang
            # that pytest never reports as a failure. Measured on the mutant with
            # tasks switched off, before this bound existed.
            try:
                task = await asyncio.wait_for(
                    call_tool_task(client, "deep_research", {"query": QUERY}), 10)
            except asyncio.TimeoutError:
                _GatedStub.gate.set()
                pytest.fail("deep_research did not return a task handle: the call ran "
                            "synchronously")
            assert task.task_id

            # Still researching: the handle came back before the run finished.
            working = await _wait_for_message(task, "researching")
            assert working.status == "working"
            assert working.status_message == "researching"

            _GatedStub.gate.set()
            return await task.result()

    out = _payload(asyncio.run(run()))
    assert out["status"] == "success"
    assert out["report_written"] is True
    assert out["source_count"] == 1
    report = (stubbed / Path(out["report_path"]).name).read_text(encoding="utf-8")
    assert REPORT in report


def test_caller_without_the_extension_still_gets_a_synchronous_answer(stubbed, monkeypatch):
    """A client that never declared io.modelcontextprotocol/tasks.

    The registry is what folds the tasks extension into every FastMCP Client once
    fastmcp_tasks is imported. Emptied, the client advertises nothing -- and if the
    server tasked the call anyway, this client could not resolve the task result and
    the call would fail, so a plain success here IS the synchronous path.
    """
    from fastmcp.client import extension_hooks
    monkeypatch.setattr(extension_hooks, "_internal_client_extension_factories", [])

    async def run():
        async with Client(server.mcp) as client:
            return await asyncio.wait_for(
                client.call_tool("deep_research", {"query": QUERY}), 30)

    out = _payload(asyncio.run(run()))
    assert out["status"] == "success"
    assert out["report_written"] is True
