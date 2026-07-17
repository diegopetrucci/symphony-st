"""Tests for the OpenCode adapter module.

Only the JSON event parser is exercised here. We deliberately do not run
the real ``opencode`` binary in tests: it requires a live LLM, model
credentials, and is inherently non-deterministic. The parser is what we
own; everything else is OpenCode's problem.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from symphony_linear.opencode import (
    OpenCodeTimeout,
    _assemble_message,
    _extract_context_tokens,
    _parse_stream,
    run_initial,
    run_resume,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

# Ensure DEBUG logs are visible during test runs.
logging.basicConfig(level=logging.DEBUG)


# ---------------------------------------------------------------------------
# Unit: JSON event parser (uses fixture data)
# ---------------------------------------------------------------------------


class TestParseEventsFromFixture:
    """Parse the fixture NDJSON file and verify basic event structure."""

    def test_fixture_has_three_events(self) -> None:
        """The captured fixture should contain exactly three events."""
        events = _load_fixture_events()
        assert len(events) == 3

    def test_fixture_contains_session_id(self) -> None:
        """Every event should carry a sessionID."""
        events = _load_fixture_events()
        for evt in events:
            assert "sessionID" in evt
            assert isinstance(evt["sessionID"], str)
            assert evt["sessionID"].startswith("ses_")

    def test_fixture_event_types(self) -> None:
        """The three events should be step_start, text, step_finish."""
        events = _load_fixture_events()
        types = [evt["type"] for evt in events]
        assert types == ["step_start", "text", "step_finish"]

    def test_extract_text_and_session_id(self) -> None:
        """Simulate the parser logic: session_id from first event,
        text from text events, detect step_finish."""
        events = _load_fixture_events()

        session_id: str | None = None
        text_parts: list[str] = []
        finished = False

        for evt in events:
            if session_id is None:
                session_id = evt.get("sessionID")
            if evt.get("type") == "text":
                text_parts.append(evt["part"]["text"])
            if evt.get("type") == "step_finish":
                finished = True

        assert session_id == "ses_1e3790378ffecZySU3wIpFOoIz"
        assert "".join(text_parts) == "hi"
        assert finished is True

    def test_parse_corrupt_line_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        """A corrupt JSON line should be logged and skipped."""
        corrupt_line = "not valid json {{{"
        with caplog.at_level(logging.DEBUG):
            _parse_one_line(corrupt_line)
        # Should have a debug log about skipping.
        assert any("Skipping" in rec.message for rec in caplog.records), (
            f"Expected debug skip message in: {[r.message for r in caplog.records]}"
        )

    def test_parse_empty_line(self) -> None:
        """Empty lines should be skipped silently."""
        result = _parse_one_line("")
        assert result is None
        result = _parse_one_line("   ")
        assert result is None


# ---------------------------------------------------------------------------
# Unit: context token extraction from step_finish events
# ---------------------------------------------------------------------------


class TestExtractContextTokens:
    """Verify context-token computation from the last step_finish event."""

    def test_existing_fixture_context_tokens(self) -> None:
        """Fixture 1: input=6, cache.read=0, cache.write=23097 → 23103."""
        events = _load_fixture_events("opencode_events.jsonl")
        assert _extract_context_tokens(events) == 23103

    def test_tool_use_fixture_context_tokens(self) -> None:
        """Fixture 2: input=10, cache.read=0, cache.write=80 → 90."""
        events = _load_fixture_events("opencode_events_tool_use.jsonl")
        assert _extract_context_tokens(events) == 90

    def test_multi_step_fixture_last_wins(self) -> None:
        """Multiple step_finish events — last one's tokens are used.
        First: input=20+read=0+write=65=85. Last: input=30+read=0+write=145=175.
        """
        events = _load_fixture_events("opencode_events_multi_step.jsonl")
        assert _extract_context_tokens(events) == 175

    def test_no_step_finish_returns_none(self) -> None:
        """Events with no step_finish → context_tokens is None."""
        events = [
            _make_text("Just a text event, no step_finish."),
        ]
        assert _extract_context_tokens(events) is None

    def test_step_finish_no_cache_subdict(self) -> None:
        """Missing 'cache' key defaults to 0 for read and write."""
        event = {
            "type": "step_finish",
            "sessionID": "ses_test",
            "part": {
                "type": "step-finish",
                "tokens": {"input": 42, "output": 10},
            },
        }
        assert _extract_context_tokens([event]) == 42

    def test_step_finish_missing_input_defaults_zero(self) -> None:
        """Missing 'input' key defaults to 0."""
        event = {
            "type": "step_finish",
            "sessionID": "ses_test",
            "part": {
                "type": "step-finish",
                "tokens": {"output": 10, "cache": {"read": 5, "write": 7}},
            },
        }
        assert _extract_context_tokens([event]) == 12

    def test_step_finish_none_values_treated_as_zero(self) -> None:
        """None values for numeric fields are treated as 0."""
        event = {
            "type": "step_finish",
            "sessionID": "ses_test",
            "part": {
                "type": "step-finish",
                "tokens": {
                    "input": None,
                    "cache": {"read": None, "write": None},
                },
            },
        }
        assert _extract_context_tokens([event]) == 0

    def test_step_finish_part_is_none(self) -> None:
        """part = None (null in JSON) → treated as {} → returns 0."""
        event = {
            "type": "step_finish",
            "sessionID": "ses_test",
            "part": None,
        }
        assert _extract_context_tokens([event]) == 0

    def test_step_finish_no_tokens_key(self) -> None:
        """No 'tokens' key at all → defaults to 0."""
        event = {
            "type": "step_finish",
            "sessionID": "ses_test",
            "part": {"type": "step-finish"},
        }
        assert _extract_context_tokens([event]) == 0

    def test_step_finish_tokens_is_not_a_dict(self) -> None:
        """tokens is a string (not a dict) → treated as missing → returns 0."""
        event = {
            "type": "step_finish",
            "sessionID": "ses_test",
            "part": {
                "type": "step-finish",
                "tokens": "invalid",
            },
        }
        assert _extract_context_tokens([event]) == 0


# ---------------------------------------------------------------------------
# Unit: message assembly (text + tool_use segments)
# ---------------------------------------------------------------------------


class TestAssembleMessage:
    """Verify the segment-based message assembly logic."""

    def test_existing_fixture_still_yields_hi(self) -> None:
        """The original single-text-burst fixture must still produce 'hi'."""
        events = _load_fixture_events()
        assert _assemble_message(events) == "hi"

    def test_tool_use_with_title(self) -> None:
        """tool_use with a state.title produces *<title>* between text bursts."""
        events = [
            _make_text("Hello"),
            _make_tool_use(tool="bash", title="Running shell command"),
            _make_text("Done."),
        ]
        result = _assemble_message(events)
        assert result == "Hello\n\n*Running shell command*\n\nDone."

    def test_tool_use_with_no_title_falls_back_to_tool_name(self) -> None:
        """tool_use with no title but a tool name produces *<tool>*."""
        events = [
            _make_text("Before"),
            _make_tool_use(tool="read_file", title=""),
            _make_text("After"),
        ]
        result = _assemble_message(events)
        assert result == "Before\n\n*read_file*\n\nAfter"

    def test_tool_use_with_neither_title_nor_tool_is_skipped(self) -> None:
        """tool_use with no title and no tool name contributes no segment."""
        events = [
            _make_text("Only text"),
            _make_tool_use(tool="", title=""),
            _make_text("More text"),
        ]
        result = _assemble_message(events)
        assert result == "Only text\n\nMore text"

    def test_tool_use_fixture_full_sequence(self) -> None:
        """The tool_use fixture file produces the expected assembled message."""
        events = _load_fixture_events("opencode_events_tool_use.jsonl")
        result = _assemble_message(events)
        assert (
            result == "Let me check that for you.\n\n*Running shell command*\n\nDone."
        )

    def test_italics_use_single_asterisks(self) -> None:
        """Tool labels must use *foo* (single asterisk), not _foo_ or **foo**."""
        events = [_make_tool_use(tool="bash", title="My Tool")]
        result = _assemble_message(events)
        assert result == "*My Tool*"
        assert "_My Tool_" not in result
        assert "**My Tool**" not in result


# ---------------------------------------------------------------------------
# Unit: _parse_stream shared helper + OpenCodeTimeout
# ---------------------------------------------------------------------------


class TestParseStream:
    """Verify the lenient NDJSON stream parser (returns session_id + events only)."""

    def test_parse_stream_extracts_session_id_and_events(self) -> None:
        """A valid stream yields session_id and parsed events."""
        events = [
            {"type": "step_start", "sessionID": "ses_abc", "part": {}},
            {"type": "text", "sessionID": "ses_abc", "part": {"text": "Hello"}},
            {
                "type": "step_finish",
                "sessionID": "ses_abc",
                "part": {"tokens": {"input": 10}},
            },
        ]
        stdout = "\n".join(json.dumps(e) for e in events)
        session_id, parsed = _parse_stream(stdout)
        assert session_id == "ses_abc"
        assert len(parsed) == 3
        msg = _assemble_message(parsed)
        assert msg == "Hello"

    def test_partial_stream_with_truncated_last_line(self) -> None:
        """A mid-line-truncated last event is skipped without error."""
        stdout = (
            json.dumps({"type": "step_start", "sessionID": "ses_123", "part": {}})
            + "\n"
            + json.dumps(
                {
                    "type": "text",
                    "sessionID": "ses_123",
                    "part": {"text": "Partial output"},
                }
            )
            + "\n"
            + '{"type":"text","sessionID":"ses_123","part":{"text":"trunc'  # truncated
        )
        session_id, parsed = _parse_stream(stdout)
        assert session_id == "ses_123"
        assert len(parsed) == 2
        msg = _assemble_message(parsed)
        assert msg == "Partial output"

    def test_empty_stdout_yields_no_session_and_empty_events(self) -> None:
        """Empty stdout produces no session_id and empty events list."""
        session_id, parsed = _parse_stream("")
        assert session_id is None
        assert parsed == []

    def test_garbage_stdout_yields_no_session_and_empty_events(self) -> None:
        """Completely unparseable stdout is silently skipped."""
        session_id, parsed = _parse_stream("not json at all\n{garbage}\nbork")
        assert session_id is None
        assert parsed == []

    def test_blank_lines_are_skipped(self) -> None:
        """Blank lines between events are skipped silently."""
        events = [
            {"type": "text", "sessionID": "ses_x", "part": {"text": "A"}},
        ]
        stdout = "\n\n" + json.dumps(events[0]) + "\n\n"
        session_id, parsed = _parse_stream(stdout)
        assert session_id == "ses_x"
        assert len(parsed) == 1

    def test_tool_use_events_parsed_in_partial_stream(self) -> None:
        """tool_use events in a partial stream are captured as events."""
        events = [
            {"type": "step_start", "sessionID": "ses_t", "part": {}},
            {"type": "text", "sessionID": "ses_t", "part": {"text": "Running..."}},
            {
                "type": "tool_use",
                "sessionID": "ses_t",
                "part": {
                    "tool": "bash",
                    "state": {"title": "Installing deps", "status": "running"},
                },
            },
        ]
        stdout = "\n".join(json.dumps(e) for e in events)
        _, parsed = _parse_stream(stdout)
        msg = _assemble_message(parsed)
        assert msg == "Running...\n\n*Installing deps*"

    def test_step_finish_in_partial_stream(self) -> None:
        """step_finish events are parsed but contribute nothing to message."""
        events = [
            {
                "type": "step_finish",
                "sessionID": "ses_f",
                "part": {"reason": "stop", "tokens": {"input": 5}},
            },
        ]
        stdout = "\n".join(json.dumps(e) for e in events)
        _, parsed = _parse_stream(stdout)
        msg = _assemble_message(parsed)
        assert msg == ""

    def test_null_json_line_is_skipped(self) -> None:
        """A line containing valid JSON ``null`` is skipped (not a dict)."""
        stdout = (
            json.dumps({"type": "text", "sessionID": "ses_n", "part": {"text": "Hi"}})
            + "\nnull\n"
        )
        session_id, parsed = _parse_stream(stdout)
        assert session_id == "ses_n"
        assert len(parsed) == 1

    def test_nested_null_json_is_skipped(self) -> None:
        """A line containing a JSON array is skipped (not a dict)."""
        stdout = (
            json.dumps({"type": "text", "sessionID": "ses_arr", "part": {"text": "X"}})
            + "\n[]\n"
        )
        session_id, parsed = _parse_stream(stdout)
        assert session_id == "ses_arr"
        assert len(parsed) == 1

    def test_event_with_null_part_does_not_break_assembly(self) -> None:
        """An event with ``"part": null`` is parseable but assembly handles None."""
        event = {"type": "text", "sessionID": "ses_p", "part": None}
        stdout = json.dumps(event) + "\n"
        session_id, parsed = _parse_stream(stdout)
        assert session_id == "ses_p"
        assert len(parsed) == 1
        # _assemble_message must not raise on part=None.
        msg = _assemble_message(parsed)
        assert msg == ""


class TestExecuteTimeout:
    """Verify the real timeout path through :func:`_execute` (mocked Popen)."""

    @staticmethod
    def _make_timeout_proc(
        partial_stdout: bytes, exit_after_kill: int = -9
    ) -> MagicMock:
        """Return a mock Popen that simulates a timeout then kill."""
        proc = MagicMock(spec=subprocess.Popen)
        # First communicate() raises TimeoutExpired.
        proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd=["opencode"], timeout=30, output=b""),
            (partial_stdout, b""),  # second communicate after kill
        ]
        proc.returncode = exit_after_kill
        return proc

    def test_timeout_carries_partial_message_and_session(self) -> None:
        """Timeout via _execute salvages session_id and partial message."""
        stdout = (
            json.dumps({"type": "step_start", "sessionID": "ses_timeout", "part": {}})
            + "\n"
            + json.dumps(
                {
                    "type": "text",
                    "sessionID": "ses_timeout",
                    "part": {"text": "I was in the middle of..."},
                }
            )
            + "\n"
        ).encode()
        proc = self._make_timeout_proc(stdout)
        with patch(
            "symphony_linear.opencode.run_in_sandbox", return_value=proc
        ) as mock_sandbox:
            with pytest.raises(OpenCodeTimeout) as excinfo:
                run_initial(
                    workspace_path="/ws",
                    prompt="do it",
                    timeout_seconds=30,
                    on_subprocess=lambda p: None,
                )
        assert excinfo.value.session_id == "ses_timeout"
        assert excinfo.value.partial_message == "I was in the middle of..."
        # Ensure the sandbox was actually called (not bypassed).
        mock_sandbox.assert_called_once()

    def test_timeout_with_null_line_does_not_mask_timeout(self) -> None:
        """A ``null`` line in partial stdout is skipped; timeout still raised."""
        stdout = (
            json.dumps({"type": "step_start", "sessionID": "ses_nl", "part": {}})
            + "\nnull\n"
            + json.dumps(
                {
                    "type": "text",
                    "sessionID": "ses_nl",
                    "part": {"text": "Working..."},
                }
            )
            + "\n"
        ).encode()
        proc = self._make_timeout_proc(stdout)
        with patch("symphony_linear.opencode.run_in_sandbox", return_value=proc):
            with pytest.raises(OpenCodeTimeout) as excinfo:
                run_initial(
                    workspace_path="/ws",
                    prompt="do it",
                    timeout_seconds=30,
                    on_subprocess=lambda p: None,
                )
        assert excinfo.value.session_id == "ses_nl"
        assert excinfo.value.partial_message == "Working..."

    def test_timeout_with_null_part_does_not_mask_timeout(self) -> None:
        """An event with ``"part": null`` does not mask the timeout."""
        stdout = (
            json.dumps({"type": "step_start", "sessionID": "ses_np", "part": {}})
            + "\n"
            + json.dumps({"type": "text", "sessionID": "ses_np", "part": None})
            + "\n"
            + json.dumps(
                {
                    "type": "text",
                    "sessionID": "ses_np",
                    "part": {"text": "More text"},
                }
            )
            + "\n"
        ).encode()
        proc = self._make_timeout_proc(stdout)
        with patch("symphony_linear.opencode.run_in_sandbox", return_value=proc):
            with pytest.raises(OpenCodeTimeout) as excinfo:
                run_initial(
                    workspace_path="/ws",
                    prompt="do it",
                    timeout_seconds=30,
                    on_subprocess=lambda p: None,
                )
        assert excinfo.value.session_id == "ses_np"
        # null-part text event contributes nothing, "More text" does.
        assert excinfo.value.partial_message == "More text"

    def test_timeout_with_pure_garbage_stdout(self) -> None:
        """Completely unparseable partial stdout still raises timeout."""
        proc = self._make_timeout_proc(b"not json\n{garbage}")
        with patch("symphony_linear.opencode.run_in_sandbox", return_value=proc):
            with pytest.raises(OpenCodeTimeout) as excinfo:
                run_initial(
                    workspace_path="/ws",
                    prompt="do it",
                    timeout_seconds=30,
                    on_subprocess=lambda p: None,
                )
        assert excinfo.value.session_id is None
        assert excinfo.value.partial_message == ""

    def test_timeout_with_no_session_id(self) -> None:
        ("""Timeout with no sessionID in partial stdout still raises.""",)
        stdout = (
            json.dumps({"type": "text", "part": {"text": "No session here."}}) + "\n"
        ).encode()
        proc = self._make_timeout_proc(stdout)
        with patch("symphony_linear.opencode.run_in_sandbox", return_value=proc):
            with pytest.raises(OpenCodeTimeout) as excinfo:
                run_initial(
                    workspace_path="/ws",
                    prompt="do it",
                    timeout_seconds=30,
                    on_subprocess=lambda p: None,
                )
        assert excinfo.value.session_id is None
        assert excinfo.value.partial_message == "No session here."


class TestOpenCodeTimeoutAttributes:
    """Verify OpenCodeTimeout carries partial_message and session_id."""

    def test_timeout_with_partial_output(self) -> None:
        """Full partial output is captured as attributes."""
        exc = OpenCodeTimeout(
            "timed out",
            partial_message="Hello\n\n*Running command*",
            session_id="ses_partial",
        )
        assert exc.partial_message == "Hello\n\n*Running command*"
        assert exc.session_id == "ses_partial"
        assert "timed out" in str(exc)

    def test_timeout_with_no_partial_output(self) -> None:
        """When partial_message is empty (default), attributes are empty/None."""
        exc = OpenCodeTimeout("timed out")
        assert exc.partial_message == ""
        assert exc.session_id is None

    def test_timeout_with_message_only_no_session(self) -> None:
        """Partial text without a sessionID event yields a message but no session."""
        exc = OpenCodeTimeout(
            "timed out",
            partial_message="Partial text",
            session_id=None,
        )
        assert exc.partial_message == "Partial text"
        assert exc.session_id is None


# ---------------------------------------------------------------------------
# Unit: --file flag argv construction
# ---------------------------------------------------------------------------


class TestFilesArgv:
    """Verify --file flags are placed correctly in the constructed command."""

    def _make_fake_popen(self) -> MagicMock:
        """Return a mock Popen with valid NDJSON stdout and exit code 0."""
        events = [
            {"type": "step_start", "sessionID": "ses_test", "part": {}},
            {"type": "text", "sessionID": "ses_test", "part": {"text": "ok"}},
            {"type": "step_finish", "sessionID": "ses_test", "part": {}},
        ]
        stdout = "\n".join(json.dumps(e) for e in events).encode()
        proc = MagicMock(spec=subprocess.Popen)
        proc.returncode = 0
        proc.communicate.return_value = (stdout, b"")
        return proc

    def test_run_initial_no_files(self) -> None:
        """When files is None, no --file flags appear."""
        fake_proc = self._make_fake_popen()
        with patch(
            "symphony_linear.opencode.run_in_sandbox", return_value=fake_proc
        ) as mock_sandbox:
            run_initial(
                workspace_path="/ws",
                prompt="hello",
                timeout_seconds=60,
                on_subprocess=lambda p: None,
            )
        cmd = mock_sandbox.call_args.kwargs["cmd"]
        assert "--file" not in cmd

    def test_run_initial_empty_files(self) -> None:
        """When files is an empty list, no --file flags appear."""
        fake_proc = self._make_fake_popen()
        with patch(
            "symphony_linear.opencode.run_in_sandbox", return_value=fake_proc
        ) as mock_sandbox:
            run_initial(
                workspace_path="/ws",
                prompt="hello",
                timeout_seconds=60,
                on_subprocess=lambda p: None,
                files=[],
            )
        cmd = mock_sandbox.call_args.kwargs["cmd"]
        assert "--file" not in cmd

    def test_run_initial_single_file(self) -> None:
        """A single file emits one --file <path> pair."""
        fake_proc = self._make_fake_popen()
        with patch(
            "symphony_linear.opencode.run_in_sandbox", return_value=fake_proc
        ) as mock_sandbox:
            run_initial(
                workspace_path="/ws",
                prompt="hello",
                timeout_seconds=60,
                on_subprocess=lambda p: None,
                files=["/tmp/foo.txt"],
            )
        cmd = mock_sandbox.call_args.kwargs["cmd"]
        file_idx = cmd.index("--file")
        assert cmd[file_idx + 1] == "/tmp/foo.txt"

    def test_run_initial_multiple_files(self) -> None:
        """Multiple files emit --file pairs in order."""
        fake_proc = self._make_fake_popen()
        with patch(
            "symphony_linear.opencode.run_in_sandbox", return_value=fake_proc
        ) as mock_sandbox:
            run_initial(
                workspace_path="/ws",
                prompt="hello",
                timeout_seconds=60,
                on_subprocess=lambda p: None,
                files=["/a.txt", "/b.txt"],
            )
        cmd = mock_sandbox.call_args.kwargs["cmd"]
        # Find all --file positions
        file_positions = [i for i, a in enumerate(cmd) if a == "--file"]
        assert len(file_positions) == 2
        assert cmd[file_positions[0] + 1] == "/a.txt"
        assert cmd[file_positions[1] + 1] == "/b.txt"

    def test_run_initial_file_before_separator(self) -> None:
        """--file flags appear before the -- separator."""
        fake_proc = self._make_fake_popen()
        with patch(
            "symphony_linear.opencode.run_in_sandbox", return_value=fake_proc
        ) as mock_sandbox:
            run_initial(
                workspace_path="/ws",
                prompt="hello",
                timeout_seconds=60,
                on_subprocess=lambda p: None,
                files=["/x.txt"],
            )
        cmd = mock_sandbox.call_args.kwargs["cmd"]
        file_idx = cmd.index("--file")
        sep_idx = cmd.index("--")
        assert file_idx < sep_idx

    def test_run_resume_no_files(self) -> None:
        """When files is None, no --file flags appear in resume."""
        fake_proc = self._make_fake_popen()
        with patch(
            "symphony_linear.opencode.run_in_sandbox", return_value=fake_proc
        ) as mock_sandbox:
            run_resume(
                workspace_path="/ws",
                session_id="ses_x",
                message="continue",
                timeout_seconds=60,
                on_subprocess=lambda p: None,
            )
        cmd = mock_sandbox.call_args.kwargs["cmd"]
        assert "--file" not in cmd

    def test_run_resume_with_files(self) -> None:
        """Files are emitted as --file pairs in the resume command."""
        fake_proc = self._make_fake_popen()
        with patch(
            "symphony_linear.opencode.run_in_sandbox", return_value=fake_proc
        ) as mock_sandbox:
            run_resume(
                workspace_path="/ws",
                session_id="ses_x",
                message="continue",
                timeout_seconds=60,
                on_subprocess=lambda p: None,
                files=["/f1.txt", "/f2.txt"],
            )
        cmd = mock_sandbox.call_args.kwargs["cmd"]
        file_positions = [i for i, a in enumerate(cmd) if a == "--file"]
        assert len(file_positions) == 2
        assert cmd[file_positions[0] + 1] == "/f1.txt"
        assert cmd[file_positions[1] + 1] == "/f2.txt"
        # --file before --
        assert file_positions[1] < cmd.index("--")

    def test_run_resume_file_before_separator(self) -> None:
        """--file appears before -- in resume command."""
        fake_proc = self._make_fake_popen()
        with patch(
            "symphony_linear.opencode.run_in_sandbox", return_value=fake_proc
        ) as mock_sandbox:
            run_resume(
                workspace_path="/ws",
                session_id="ses_x",
                message="continue",
                timeout_seconds=60,
                on_subprocess=lambda p: None,
                files=["/f.txt"],
            )
        cmd = mock_sandbox.call_args.kwargs["cmd"]
        file_idx = cmd.index("--file")
        sep_idx = cmd.index("--")
        assert file_idx < sep_idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fixture_events(
    filename: str = "opencode_events.jsonl",
) -> list[dict[str, Any]]:
    """Load the recorded OpenCode JSON events from the fixture file."""
    fixture_path = FIXTURE_DIR / filename
    raw = fixture_path.read_text()
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped:
            events.append(json.loads(stripped))
    return events


def _parse_one_line(line: str) -> dict[str, Any] | None:
    """Simulate the parser logic on a single line (used by unit tests)."""
    stripped = line.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        logging.getLogger("symphony_linear.opencode").debug(
            "Skipping unparseable JSON line: %s", stripped[:200]
        )
        return None


def _make_text(text: str) -> dict[str, Any]:
    """Build a minimal ``text`` event dict."""
    return {
        "type": "text",
        "sessionID": "ses_test",
        "part": {"type": "text", "text": text},
    }


def _make_tool_use(tool: str, title: str) -> dict[str, Any]:
    """Build a minimal ``tool_use`` event dict."""
    return {
        "type": "tool_use",
        "sessionID": "ses_test",
        "part": {
            "type": "tool-use",
            "tool": tool,
            "state": {"title": title, "status": "running"},
        },
    }
