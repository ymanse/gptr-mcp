"""write_report refuses a research_id whose context is not research.

deep_research already refuses a context under MIN_RESEARCH_CONTEXT_CHARS, but it stores
the researcher and returns its research_id BEFORE refusing. write_report on that id
checked nothing: it handed the empty context to the report writer, the model wrote a
report from its prior knowledge, and the tool persisted it with status "success".
deep_tree_research's id is the same hole by construction -- the tree researches on
per-node GPTResearcher instances and the root researcher's context is never assigned,
so get_research_context() on it is always [].

The library now abstains on an empty context too (gpt_researcher writer.py, upstream
5d987288), but that guard returns a SENTENCE. Without this one, "I could not gather any
source material" is saved to report_path as a report with status success -- worse than
an error, because a caller reads the status. So the tool refuses before the writer is
reached, in the same shape deep_research refuses with, and arms no allowance.

Hermetic: GPTResearcher is replaced by a stub whose write_report is the LLM entry point
and counts every call; artifacts go to tmp_path.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import server  # noqa: E402

QUERY = "What bounds outbox table growth?"
ENOUGH = "Teams partition the outbox by day and drop old partitions. " * 30


class _Researcher:
    """What the tools read off a stored GPTResearcher. write_report is where the LLM
    would be spent, so it only counts."""

    def __init__(self, context, sources=()):
        self.query = QUERY
        self._context = context
        self._sources = list(sources)
        self.llm_calls = 0

    def get_research_context(self):
        return self._context

    def get_research_sources(self):
        return self._sources

    def get_source_urls(self):
        return [s["url"] for s in self._sources]

    def get_costs(self):
        return 0.0

    async def write_report(self, custom_prompt=None):
        self.llm_calls += 1
        return "# A report written from prior knowledge"


@pytest.fixture
def stored(monkeypatch, tmp_path):
    monkeypatch.setenv("GPTR_MCP_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("GPTR_MCP_OUTPUT_DIR_HOST", raising=False)
    armed = []
    monkeypatch.setattr(server, "begin_agent_run", lambda *a, **k: armed.append(a))
    monkeypatch.setattr(server.mcp, "researchers", {}, raising=False)

    def _store(researcher, research_id="0b7c1d2e-empty"):
        server.mcp.researchers[research_id] = researcher
        return research_id

    _store.armed = armed
    _store.out_dir = tmp_path
    return _store


def _write_report(research_id, custom_prompt=None):
    tool = getattr(server.write_report, "fn", server.write_report)
    return asyncio.run(tool(research_id, custom_prompt))


@pytest.mark.parametrize("context", ["", [], "[]\n" * 60, None],
                         ids=["empty-str", "tree-root-list", "crumbs", "none"])
def test_an_empty_context_is_refused_and_the_llm_is_never_called(stored, context):
    researcher = _Researcher(context)
    out = _write_report(stored(researcher))

    assert researcher.llm_calls == 0, (
        "the report writer was reached with nothing to write from -- one CLI session "
        "spent on a report that can only come from the model's prior knowledge")
    assert out.get("status") == "error", (
        f"write_report on an empty context returned {out.get('status')!r}; a caller "
        "reading success stores it as research")
    assert out.get("report_written") is False
    assert "NO usable evidence" in out.get("message", "")
    assert stored.armed == [], (
        "a refusal that spends nothing armed an allowance -- allowances pool, and arming "
        "re-baselines the per-run counters a concurrent call is reporting from")
    assert not list(stored.out_dir.iterdir()), (
        "the refusal still wrote an artifact; report_path must not open to a report "
        "that was never written")


def test_the_refusal_names_the_tree_case(stored):
    """A deep_tree_research id always lands here, and its report already exists."""
    out = _write_report(stored(_Researcher([])))
    assert "deep_tree_research" in out.get("message", "") and "report_path" in out["message"]


def test_the_refusal_names_the_quick_search_case(stored):
    """quick_search stores its researcher under search_id with context [], so that id
    always lands here too; "run the research again" does not describe it."""
    out = _write_report(stored(_Researcher([])))
    assert "quick_search" in out.get("message", ""), out.get("message")


def test_the_refusal_has_deep_researchs_shape(stored):
    """Same keys deep_research refuses with, so a caller needs no new branch."""
    rid = stored(_Researcher(""))
    out = _write_report(rid)

    assert out["status"] == "error" and out["research_id"] == rid
    assert out["source_count"] == 0 and out["report_written"] is False


def test_a_stale_provider_refusal_is_not_named_as_the_cause(stored, monkeypatch):
    """The recorded refusal is cleared only by begin_agent_run, which a refusal here
    never reaches, so whatever limit an earlier call hit is still recorded. Attached,
    it would tell the caller to wait for a reset that can never give this id evidence."""
    reason = "You've hit your session limit · resets 5:50am (UTC)"
    monkeypatch.setattr(server, "provider_refused", lambda: (True, reason))

    out = _write_report(stored(_Researcher([])))

    assert out["status"] == "error" and out["report_written"] is False
    assert "provider_refused" not in out and "hint" not in out, (
        f"an empty context was blamed on a provider refusal: {out.get('hint')!r}. The "
        "caller would wait for 5:50am and get the same refusal again")


def test_the_floor_is_the_one_deep_research_uses(stored):
    short = _Researcher("x" * (server.MIN_RESEARCH_CONTEXT_CHARS - 1))
    assert _write_report(stored(short, "short"))["status"] == "error"
    assert short.llm_calls == 0

    enough = _Researcher("x" * server.MIN_RESEARCH_CONTEXT_CHARS)
    assert _write_report(stored(enough, "enough"))["status"] == "success"
    assert enough.llm_calls == 1


def test_a_gathered_context_is_still_written(stored):
    researcher = _Researcher(ENOUGH, [{"url": "https://example.invalid/outbox",
                                       "title": "outbox", "raw_content": "by day"}])
    out = _write_report(stored(researcher))

    assert out["status"] == "success", out
    assert researcher.llm_calls == 1
    assert len(stored.armed) == 1, "a real report must still arm its allowance"
