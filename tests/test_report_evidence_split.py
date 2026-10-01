"""A deep_research report and the evidence it was written from are TWO files.

They used to be one: the synthesis, then `---`, then the whole research dump. The caller
is an agent told to "read report_path", and it read all of it. Measured 2026-10-01 on
outputs/empirical-evidence-on-llm-coding-agent-a-e59185e7.md (72,436 chars): report
19,142 (26.4%), evidence dump 53,423 (73.8%); four larger reports ran 89-93% dump.
Summarizing the synthesis alone kept the same essential recall (12/12, 14/14) for 60-85%
less. So:

  - with a synthesis, report_path is the synthesis and context_path the evidence;
  - without one, nothing changes: report_path is the dump (273 of 513 files in outputs/
    are that shape) and context_path names the same file;
  - get_research_context no longer overwrites the synthesis with the dump;
  - write_report points at the evidence too;
  - the context goes into the evidence file byte-for-byte. A heading demotion that rode
    along with the split rewrote Python comments as headings and was removed; see
    test_the_evidence_holds_the_context_byte_for_byte.

Hermetic: GPTResearcher is replaced by a stub, the artifacts go to tmp_path.
"""
import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import server  # noqa: E402
import utils  # noqa: E402

QUERY = "What bounds outbox table growth?"
# A scraped page's own outline, as it really arrives: an H1 prompt-template heading, an
# author's address as a heading, a no-space "#Word", and a shell comment inside a code
# fence. All of it must reach the evidence file exactly as scraped.
SCRAPED = ("# Task Instructions\n"
           "## Overview\n"
           "Teams partition the outbox by day and drop old partitions.\n"
           "#Instances\n"
           "```bash\n"
           "# drop yesterday's partition\n"
           "psql -c 'DROP TABLE outbox_2026_09_30'\n"
           "```\n"
           "# jeffrey.da@example.com\n")
CONTEXT = SCRAPED + "Partitioning by day bounds growth. " * 40
REPORT = "# Outbox growth\n\nTeams partition by day [1].\n\n## References\n\n[1] https://example.invalid/outbox"


class _Stub:
    """GPTResearcher reduced to what the tools read off it."""

    REPORT = REPORT

    def __init__(self, query, report_type=None, **kwargs):
        self.query = query
        self.deep_researcher = type("DR", (), {"time_exhausted": False,
                                               "budget_exhausted": False,
                                               "time_budget_s": 600.0})()

    async def conduct_research(self, scope=False):
        return CONTEXT

    def get_research_context(self):
        return CONTEXT

    def get_research_sources(self):
        return [{"url": "https://example.invalid/outbox", "title": "outbox",
                 "raw_content": "partition by day"}]

    def get_source_urls(self):
        return ["https://example.invalid/outbox"]

    def get_costs(self):
        return 0.0

    async def write_report(self, custom_prompt=None):
        if self.REPORT is None:
            raise RuntimeError("report LLM unreachable")
        return self.REPORT


def _setup(monkeypatch, tmp_path, stub_cls=_Stub):
    monkeypatch.setenv("GPTR_MCP_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("GPTR_MCP_OUTPUT_DIR_HOST", raising=False)
    monkeypatch.setattr(server, "GPTResearcher", stub_cls)
    monkeypatch.setattr(server, "begin_agent_run", lambda *a, **k: None)
    monkeypatch.setattr(server.mcp, "researchers", {}, raising=False)


def _call(name, *args):
    tool = getattr(server, name)
    return asyncio.run(getattr(tool, "fn", tool)(*args))


def _read(path, tmp_path) -> str:
    return (tmp_path / Path(path).name).read_text(encoding="utf-8")


# ── With a synthesis: two files ──────────────────────────────────────────────────────

def test_report_path_is_the_synthesis_and_context_path_the_evidence(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    out = _call("deep_research", QUERY)

    assert out["status"] == "success" and out["report_written"] is True, out
    report, evidence = _read(out["report_path"], tmp_path), _read(out["context_path"], tmp_path)
    assert out["report_path"] != out["context_path"], (
        "a run with a synthesis still has one file -- the reader of report_path reads "
        "the dump too")
    assert report.startswith(REPORT), f"report_path does not open to the synthesis: {report[:120]!r}"
    assert "# Research:" not in report and "## Context" not in report, (
        "the research dump is still glued under the synthesis in report_path")
    assert "Partitioning by day bounds growth." not in report
    assert evidence.startswith("# Research: "), evidence[:120]
    assert "## Context" in evidence and "Partitioning by day bounds growth." in evidence, (
        "the evidence file does not hold the research the report was written from")
    assert Path(out["context_path"]).name in report, (
        "the report does not name its evidence file -- opened without the response, "
        "nothing links the two")


def test_the_preview_is_still_the_synthesis(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    out = _call("deep_research", QUERY)
    assert out["context_preview"].startswith(REPORT.splitlines()[0]), out["context_preview"][:120]
    assert out["context_chars"] == len(CONTEXT)


def test_the_urls_in_the_synthesis_survive_into_report_path(monkeypatch, tmp_path):
    """s20 asserts a URL survives into report_path. With the dump gone from it, only
    the synthesis' own references carry one -- they must not be cut."""
    _setup(monkeypatch, tmp_path)
    out = _call("deep_research", QUERY)
    assert re.search(r"https?://", _read(out["report_path"], tmp_path))


# ── Without a synthesis: today's single file, unchanged ──────────────────────────────

def test_no_synthesis_leaves_the_dump_at_report_path(monkeypatch, tmp_path):
    class _Broken(_Stub):
        REPORT = None

    _setup(monkeypatch, tmp_path, _Broken)
    out = _call("deep_research", QUERY)

    assert out["report_written"] is False
    assert out["context_path"] == out["report_path"], (
        "with no synthesis there is one file; context_path must name it, not a missing one")
    assert _read(out["report_path"], tmp_path).startswith("# Research: ")
    assert not list(tmp_path.glob("*.context.md")), (
        "a no-synthesis run wrote a second copy of its dump")


# ── The other call sites ─────────────────────────────────────────────────────────────

def test_get_research_context_does_not_overwrite_the_synthesis(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    first = _call("deep_research", QUERY)
    ctx = _call("get_research_context", first["research_id"])

    assert _read(first["report_path"], tmp_path).startswith(REPORT), (
        "get_research_context replaced deep_research's synthesis with the dump")
    assert ctx["report_path"] == ctx["context_path"] == first["context_path"]
    assert "Partitioning by day bounds growth." in _read(ctx["report_path"], tmp_path)


def test_write_report_points_at_the_evidence(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    first = _call("deep_research", QUERY)
    out = _call("write_report", first["research_id"])

    assert out["status"] == "success", out
    assert _read(out["report_path"], tmp_path).startswith(REPORT)
    assert "Partitioning by day bounds growth." in _read(out["context_path"], tmp_path)
    assert out["context_chars"] == len(CONTEXT), (
        "context_chars measures the report again, not the research context")


def test_the_default_is_the_old_single_file(tmp_path, monkeypatch):
    """persist_research_artifacts without split= writes what it always wrote."""
    monkeypatch.setenv("GPTR_MCP_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("GPTR_MCP_OUTPUT_DIR_HOST", raising=False)
    meta = utils.persist_research_artifacts(QUERY, CONTEXT, [], [], "abcdef12-0000",
                                            report_text=REPORT)
    assert (tmp_path / Path(meta["report_path"]).name).read_text(encoding="utf-8") == REPORT
    assert meta["context_path"] is None and not list(tmp_path.glob("*.context.md"))


# ── The evidence is the context, byte for byte ──────────────────────────────────────

# A faithful reduction of outputs/nautilustrader-built-in-reporting-and-an-a76fd491.md,
# context lines 91-119, copied verbatim. The scraped chunk starts PART-WAY THROUGH a Python
# code block on the NautilusTrader data docs: its opening fence was on the page, not in the
# chunk. So "# With storage options" is a Python comment outside any visible fence, the first
# ``` is that block's CLOSER, "### [Writing data]" is a real doc heading, and the "# Write"
# and "# Skip" lines after the next ``` are Python comments.
WITNESS_CONTEXT = (
    "Source: https://nautilustrader.io/docs/latest/concepts/data/\n"
    "Title: \n"
    "Content: # S3 bucket\n"
    'catalog = ParquetDataCatalog.from_uri("s3://my-bucket/nautilus-data/")\n'
    "\n"
    "# With storage options\n"
    "catalog = ParquetDataCatalog.from_uri(\n"
    '    "s3://my-bucket/nautilus-data/",\n'
    '    fs_storage_options={"access_key_id": "your-key", "secret_access_key": "your-secret"},\n'
    ")\n"
    "```\n"
    "\n"
    "### [Writing data](https://nautilustrader.io/docs/latest/concepts/data/\\#writing-data)\n"
    "\n"
    "Use `write_data()` to store built‑in `Data` objects and registered custom `Data` subclasses.\n"
    "\n"
    "```\n"
    "# Write a list of data objects\n"
    "catalog.write_data(quote_ticks)\n"
    "\n"
    "# Write with custom timestamp range\n"
    "catalog.write_data(\n"
    "    trade_ticks,\n"
    "    start=1704067200000000000,  # Optional start timestamp override (UNIX nanoseconds)\n"
    "    end=1704153600000000000,  # Optional end timestamp override (UNIX nanoseconds)\n"
    ")\n"
    "\n"
    "# Skip disjoint check for overlapping data\n"
    "catalog.write_data(bars, skip_disjoint_check=True)\n"
    "```"
)


@pytest.mark.parametrize("split,report_text", [(True, REPORT), (True, None), (False, None)],
                         ids=["split-with-synthesis", "split-no-synthesis", "single-file"])
def test_the_evidence_holds_the_context_byte_for_byte(tmp_path, monkeypatch, split, report_text):
    """Whatever is handed to persist_research_artifacts as context is what the evidence
    file holds between `## Context` and `## Sources` -- not one byte rewritten.

    What went wrong (2026-10-01): a utils.demote_headings pushed every ATX heading in the
    context two levels down, tracking ``` fences so code comments would be left alone. On
    the witness file it did the opposite. A context is a concatenation of scraped chunks,
    and a chunk is an arbitrary slice of a page; this one begins inside a code block whose
    opening fence was never scraped. So "# With storage options" became a heading, the
    block's closing ``` was read as an OPENING fence, the real heading "### [Writing data]"
    was skipped as code, and the next real code block was taken for prose: its comments
    became "### Write a list of data objects" and so on. 65 lines of that file's context
    changed, Python comments among them. Evidence text was rewritten.

    Why no better regex or fence tracker can fix it: the Python comment "# Write data" and
    the Markdown heading "# Write data" are the same bytes, and whether a line is inside a
    fence depends on an opener that may lie outside the slice. The information is not in
    the input. What demotion was for -- scraped "# Task Instructions" in the outline of
    what a caller reads -- was in the REPORT, and the split already took the dump out of
    it. Tidier headings in the evidence file are cosmetics, not worth one corrupted line of
    what a reader checks a claim against.

    Asserts the invariant, not the symptom, so any rewrite of the context fails it.
    """
    monkeypatch.setenv("GPTR_MCP_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("GPTR_MCP_OUTPUT_DIR_HOST", raising=False)
    context = "The catalog persists data as Parquet.\n\n" + WITNESS_CONTEXT
    meta = utils.persist_research_artifacts(QUERY, context, [], ["https://nautilustrader.io/"],
                                            "a76fd491-0000", report_text=report_text,
                                            split=split)

    # The bytes on disk. write_text turns "\n" into os.linesep, so expect that and no more.
    raw = (tmp_path / Path(meta["context_path"]).name).read_bytes().decode("utf-8")
    nl = os.linesep
    _, sep, rest = raw.partition(f"{nl}## Context{nl}{nl}")
    assert sep, f"no `## Context` wrapper in the evidence file: {raw[:200]!r}"
    body, sep, _ = rest.rpartition(f"{nl}{nl}## Sources{nl}")
    assert sep, "no `## Sources` after the context"
    assert body == context.replace("\n", nl), (
        "the evidence file does not hold the context byte for byte -- something rewrote "
        "it on the way to disk. If that is heading demotion coming back, read this docstring.")
