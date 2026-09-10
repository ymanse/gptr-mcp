"""A subscription session limit must not look like a bug in this server.

Measured 2026-09-10: `deep_research` armed 50, spent 2, and died in 16 seconds. The MCP
response said only "Failed to get response from claude_agent API" — the same words a
spent budget produces, and the same words a genuine code defect would. The reason was in
the container log all along:

    You've hit your session limit · resets 5:50am (UTC)

Reading that log is what the caller should not have to do.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402


def test_a_failure_without_a_refusal_is_left_alone():
    out = server._with_refusal({"status": "error", "error": "boom"})
    assert "provider_refused" not in out and "hint" not in out, (
        f"an ordinary failure was labelled a provider refusal: {out}")


def test_a_refusal_is_named_with_the_providers_own_words(monkeypatch):
    reason = "You've hit your session limit · resets 5:50am (UTC)"
    monkeypatch.setattr(server, "provider_refused", lambda: (True, reason))

    out = server._with_refusal({"status": "error",
                                "error": "Failed to get response from claude_agent API"})

    assert out.get("provider_refused") is True
    assert out.get("provider_refusal_reason") == reason
    assert "5:50am" in out.get("hint", ""), (
        f"the response does not carry the reset time: {out.get('hint')!r}. Without it the "
        "caller cannot tell a subscription limit from a defect, which is exactly the "
        "confusion this exists to end")
    assert "not a bug" in out.get("hint", ""), (
        "the hint must say plainly that this is not a server defect — the identical "
        "wrapper text is what sent the 2026-09-10 investigation into the code first")
