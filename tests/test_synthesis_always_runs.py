"""deep_research ends in a REPORT, including when its time budget cut it short.

Measured 2026-09-21: a linear run spent its 600s research budget and came back as a
284KB context dump with no synthesis at all. Nothing had failed. Synthesis simply lived
only in the separate `write_report` tool -- a second round trip the caller had to
remember, keyed on an in-memory researcher that any container restart loses.

So `deep_research` now writes the report itself, after the research budget:

  - a truncated run still gets one, written from a context that carries the
    "Incomplete Research" notice, so the report says it is partial;
  - the report goes FIRST in the artifact and the raw research follows it;
  - a synthesis failure never costs the research: the context is still persisted and
    returned, and the response names the failure.

Hermetic: GPTResearcher is replaced by a stub, the artifacts go to tmp_path.
"""
import asyncio
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402

QUERY = "What bounds outbox table growth?"
NOTICE = "## Incomplete Research\nThis research stopped early: the run's time budget ran out"
CONTEXT = "Teams partition the outbox by day and drop old partitions.\n\n" + NOTICE
REPORT = "# Outbox growth\n\nTeams partition by day. This report is PARTIAL: the time budget ran out."


class _Stub:
    """GPTResearcher reduced to what deep_research reads off it."""

    TRUNCATED = True

    def __init__(self, query, report_type=None, **kwargs):
        self.query = query
        self.deep_researcher = type("DR", (), {"time_exhausted": self.TRUNCATED,
                                               "budget_exhausted": False,
                                               "time_budget_s": 600.0})()
        self.report_calls = 0

    async def conduct_research(self, scope=False):
        return CONTEXT

    def get_research_context(self):
        return CONTEXT

    def get_research_sources(self):
        return [{"url": "https://example.invalid/outbox", "title": "outbox",
                 "raw_content": "partition by day"}]

    def get_source_urls(self):
        return ["https://example.invalid/outbox"]

    async def write_report(self, custom_prompt=None):
        self.report_calls += 1
        return REPORT


def _run(monkeypatch, tmp_path, stub_cls=_Stub):
    monkeypatch.setenv("GPTR_MCP_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("GPTR_MCP_OUTPUT_DIR_HOST", raising=False)
    monkeypatch.setattr(server, "GPTResearcher", stub_cls)
    monkeypatch.setattr(server, "begin_agent_run", lambda *a, **k: None)
    tool = getattr(server.deep_research, "fn", server.deep_research)
    return asyncio.run(tool(QUERY))


def _artifact(out, tmp_path) -> str:
    return (tmp_path / Path(out["report_path"]).name).read_text(encoding="utf-8")


def test_a_time_truncated_run_still_ends_in_a_report(monkeypatch, tmp_path):
    """The defect itself: time_budget_exhausted=true used to mean 'no report'."""
    out = _run(monkeypatch, tmp_path)

    assert out.get("status") == "success", out
    assert out.get("time_budget_exhausted") is True, "fixture precondition: a truncated run"
    assert out.get("report_written") is True and out.get("report_error") is None, (
        f"a run cut short by its time budget came back without a synthesis "
        f"(report_written={out.get('report_written')!r}, "
        f"report_error={out.get('report_error')!r}) -- the caller gets a raw dump")


def test_the_report_comes_first_and_the_research_is_still_there(monkeypatch, tmp_path):
    """The preview a caller reads is the synthesis; the evidence it was written from
    stays in the same file, below it."""
    out = _run(monkeypatch, tmp_path)
    body = _artifact(out, tmp_path)

    head, _, tail = body.partition(REPORT)
    assert tail, f"the report is missing from the artifact (head: {body[:160]!r})"
    assert "## Context" not in head, (
        "the raw research dump comes before the report -- the caller opens the file "
        "to the evidence, not the synthesis")
    assert "partition the outbox by day" in tail, (
        "the raw research is missing from the artifact once a report is written -- "
        "the evidence behind the synthesis was thrown away")
    assert REPORT.splitlines()[0] in out["context_preview"], (
        "the inline preview is not the synthesis, so a caller reading only the "
        "response still sees the dump")


def test_a_truncated_report_says_so_at_the_top_whatever_the_writer_did(monkeypatch, tmp_path):
    """The live defect: the context ended in the "Incomplete Research" notice and the
    report writer dropped it, so a partial report read as complete. The disclosure must
    not depend on the model relaying it -- this stub's report never mentions it."""
    class _Silent(_Stub):
        async def write_report(self, custom_prompt=None):
            return "# Outbox growth\n\nTeams partition by day."

    out = _run(monkeypatch, tmp_path, _Silent)
    body = _artifact(out, tmp_path)

    first = body.lstrip().splitlines()[0]
    assert "Partial research" in first and "time budget" in first, (
        f"a time-truncated run's report opens with {first!r} -- nothing at the top "
        "says the research stopped early, so it reads as a complete answer")


def test_a_complete_run_carries_no_partial_banner(monkeypatch, tmp_path):
    """Positive control: a banner on every report would teach readers to ignore it."""
    class _Complete(_Stub):
        TRUNCATED = False

    out = _run(monkeypatch, tmp_path, _Complete)

    assert "Partial research" not in _artifact(out, tmp_path), (
        "a run that finished its research was labelled partial")


def test_the_truncation_notice_reaches_the_report_writer(monkeypatch, tmp_path):
    """The report is written from the context, so the notice has to be in it -- that is
    how a partial run's report can say it is partial."""
    seen = {}

    class _Watching(_Stub):
        async def write_report(self, custom_prompt=None):
            seen["context"] = self.get_research_context()
            return REPORT

    _run(monkeypatch, tmp_path, _Watching)

    assert "Incomplete Research" in seen.get("context", ""), (
        "the report writer never saw the truncation notice, so a partial run's report "
        "reads as a complete answer")


def test_a_failed_synthesis_never_costs_the_research(monkeypatch, tmp_path):
    """Synthesis is the LAST step; a failure there must not discard what the budget
    paid for."""
    class _Broken(_Stub):
        async def write_report(self, custom_prompt=None):
            raise RuntimeError("report LLM unreachable")

    out = _run(monkeypatch, tmp_path, _Broken)

    assert out.get("status") == "success", (
        f"a synthesis failure turned the whole research call into an error: {out}")
    assert out.get("report_written") is False
    assert "report LLM unreachable" in (out.get("report_error") or ""), (
        f"the response does not say why there is no report: {out.get('report_error')!r}")
    assert "partition the outbox by day" in _artifact(out, tmp_path), (
        "the research was not persisted after the synthesis failed")


def test_a_run_that_gathered_nothing_writes_no_report_and_says_so(monkeypatch, tmp_path):
    """Measured 2026-09-21, live: a slow academic query's first round did not finish in
    600s, the run gathered 0 sources and an empty context, and the synthesis step wrote
    20KB from the model's prior knowledge anyway -- status "success", under a banner
    claiming it was "written only from what was gathered". A report from nothing is the
    fabrication this pipeline exists to prevent; with nothing gathered the call fails."""
    calls = {"report": 0}

    class _Empty(_Stub):
        async def conduct_research(self, scope=False):
            return ""

        def get_research_context(self):
            return ""

        def get_research_sources(self):
            return []

        def get_source_urls(self):
            return []

        async def write_report(self, custom_prompt=None):
            calls["report"] += 1
            return "# A report written from prior knowledge"

    out = _run(monkeypatch, tmp_path, _Empty)

    assert calls["report"] == 0, (
        "the report writer was called with an empty context -- whatever it returns can "
        "only come from the model's prior knowledge")
    assert out.get("status") == "error", (
        f"a run that gathered nothing reported status {out.get('status')!r}; a caller "
        "reading success does not go on to check source_count")
    assert out.get("report_written") is False
    message = out.get("message", "")
    assert "NO evidence" in message and "time budget" in message, (
        f"the failure does not say what happened or what to change: {message!r}")
