"""Chat slash-command tests: completion matching and command dispatch.

The interactive loop needs a tty, but the registry, the matcher feeding the
completion menu, and the dispatcher are pure and shared with it.
"""

from __future__ import annotations

from src.cli import _CHAT_COMMANDS, _render_answer, _run_chat_command, _slash_matches


class FakeCitation:
    def __init__(self, marker: int, label: str):
        self.marker = marker
        self._label = label

    def label(self) -> str:
        return self._label


class FakeAnswer:
    def __init__(self, citations):
        self.citations = citations
        self.text = "An answer [1]."
        self.contexts = []
        self.retrieval_s = 0.05
        self.generation_s = 1.0
        self.completion_tokens = 10


def test_slash_matches_by_prefix():
    assert _slash_matches("/") == list(_CHAT_COMMANDS.items())
    assert _slash_matches("/so") == [("/sources", _CHAT_COMMANDS["/sources"])]
    assert _slash_matches("/sources") == [("/sources", _CHAT_COMMANDS["/sources"])]
    assert _slash_matches("/c") == [("/clear", _CHAT_COMMANDS["/clear"])]


def test_slash_matches_only_for_lone_leading_token():
    assert _slash_matches("") == []
    assert _slash_matches("hello") == []
    assert _slash_matches("what is /sources") == []
    assert _slash_matches("/sources now") == []  # args typed -> menu goes away
    assert _slash_matches("/nope") == []


def test_sources_command_renders_last_answer(capsys):
    ans = FakeAnswer([FakeCitation(1, "rev.pdf, p.3-4"), FakeCitation(2, "soc.pdf, p.9")])
    _run_chat_command("/sources", ans)
    out = capsys.readouterr().out
    assert "Sources" in out
    assert "rev.pdf, p.3-4" in out
    assert "soc.pdf, p.9" in out


def test_unique_prefix_dispatches(capsys):
    _run_chat_command("/so", FakeAnswer([FakeCitation(1, "doc.pdf, p.1")]))
    assert "doc.pdf, p.1" in capsys.readouterr().out


def test_sources_before_any_answer(capsys):
    _run_chat_command("/sources", None)
    assert "No answer yet" in capsys.readouterr().out


def test_sources_when_none_were_cited(capsys):
    _run_chat_command("/sources", FakeAnswer([]))
    assert "No sources cited" in capsys.readouterr().out


def test_sources_command_returns_last_ans_unchanged():
    ans = FakeAnswer([FakeCitation(1, "rev.pdf, p.3-4")])
    assert _run_chat_command("/sources", ans) is ans
    assert _run_chat_command("/bogus", ans) is ans  # unknown commands keep state too


def test_clear_command_resets_and_repaints():
    ans = FakeAnswer([FakeCitation(1, "rev.pdf, p.3-4")])
    repainted = []
    result = _run_chat_command("/clear", ans, on_clear=lambda: repainted.append(True))
    assert result is None  # /sources afterwards reports "No answer yet"
    assert repainted == [True]  # banner is repainted after the wipe


def test_chat_answers_suppress_sources_table(capsys):
    ans = FakeAnswer([FakeCitation(1, "rev.pdf, p.3-4")])
    _render_answer(ans, show_context=False, show_sources=False)  # the chat path
    out = capsys.readouterr().out
    assert "Answer" in out
    assert "rev.pdf" not in out  # sources only via /sources in chat


def test_query_answers_keep_sources_table(capsys):
    ans = FakeAnswer([FakeCitation(1, "rev.pdf, p.3-4")])
    _render_answer(ans, show_context=False)  # the one-shot `rag query` path
    assert "rev.pdf, p.3-4" in capsys.readouterr().out


def test_unknown_command_lists_available(capsys):
    _run_chat_command("/bogus", None)
    out = capsys.readouterr().out
    assert "Unknown command" in out
    assert "/sources" in out
