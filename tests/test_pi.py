"""Tests for the pi agent adapter.

The tests use a captured pi-family NDJSON stream and a mocked shared runner;
they never invoke the real ``pi`` binary or an LLM.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from symphony_linear import agent_runner, opencode, pi, pi_protocol
from symphony_linear.pi import (
    PiCancelled,
    PiError,
    PiTimeout,
    _agent_dir_rw_paths,
    _build_env,
    run_initial,
    run_resume,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "omp_events.jsonl"
ERROR_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "omp_error_events.jsonl"
)
# Real captured pi stream; detects pi format drift from OMP's shared parser.
PI_SUCCESS_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "pi_success_events.jsonl"
)


def _fixture_text(fixture_path: Path = FIXTURE_PATH) -> str:
    """Return a captured pi-family NDJSON stream for a completed simple turn."""
    return fixture_path.read_text()


def _error_fixture_text() -> str:
    """Return a pi-family stream whose retries end in a provider failure."""
    return ERROR_FIXTURE_PATH.read_text()


# This regression test catches pi format drift away from OMP's shared parser.
class TestRealPiStream:
    """Exercise pi's real wire format through the shared adapter."""

    def test_run_initial_extracts_real_pi_stream(self) -> None:
        stream = _fixture_text(PI_SUCCESS_FIXTURE_PATH)
        _, events = pi_protocol._parse_stream(stream)

        with patch(
            "symphony_linear.pi.agent_runner.run",
            return_value=(0, stream, "", None),
        ):
            session_id, message, context_tokens = pi.run_initial(
                workspace_path="/workspace",
                tmp_path="/workspace/tmp",
                prompt="Reply with exactly: OK",
                timeout_seconds=60,
                idle_timeout_seconds=30,
                on_subprocess=lambda process: None,
            )

        assert (session_id, message, context_tokens) == (
            "01a051fa-999d-7ff3-b776-5fe27cb1e592",
            "OK",
            1067,
        )
        assert pi_protocol._assemble_message(events) == "OK"


class TestCommandConstruction:
    """pi uses pi-family positional attachments and pi-specific flags."""

    def test_public_signatures_match_opencode(self) -> None:
        """pi exposes the opencode-compatible interface, plus an optional binary kwarg."""
        # Extract common parameters, excluding the pi-specific 'binary' kwarg.
        pi_initial = {
            k: v
            for k, v in inspect.signature(run_initial).parameters.items()
            if k != "binary"
        }
        oc_initial = dict(inspect.signature(opencode.run_initial).parameters)
        assert pi_initial == oc_initial

        pi_resume = {
            k: v
            for k, v in inspect.signature(run_resume).parameters.items()
            if k != "binary"
        }
        oc_resume = dict(inspect.signature(opencode.run_resume).parameters)
        assert pi_resume == oc_resume

    def test_initial_command_prefixes_prompt_and_uses_positional_attachments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Clear PI_CODING_AGENT_DIR so the test is deterministic regardless of
        # the host environment.
        monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)

        with patch(
            "symphony_linear.pi.agent_runner.run",
            return_value=(0, _fixture_text(), "", None),
        ) as runner:
            session_id, message, context_tokens = run_initial(
                workspace_path="/workspace",
                tmp_path="/workspace/tmp",
                prompt="@developer implement this",
                timeout_seconds=60,
                idle_timeout_seconds=30,
                on_subprocess=lambda process: None,
                hide_paths=["/secret"],
                extra_rw_paths=["/shared"],
                attachments_path="/workspace/attachments",
                dir_map=[("/host/mount", "/sandbox/mount")],
                files=["/workspace/attachments/brief.md", "notes.txt"],
                model="anthropic/claude-opus-5",
            )

        assert (session_id, message, context_tokens) == (
            "01a035ff-248a-735c-8173-f5ee428fe917",
            "hello",
            3479,
        )
        assert runner.call_args.kwargs["cmd"] == [
            "pi",
            "-p",
            "--mode",
            "json",
            "--no-approve",
            "--model",
            "anthropic/claude-opus-5",
            "@/workspace/attachments/brief.md",
            "@notes.txt",
            "\n@developer implement this",
        ]
        kwargs = runner.call_args.kwargs
        assert kwargs["env"] == {"HOME": str(Path.home())}
        assert kwargs["hide_paths"] == ["/secret"]
        assert kwargs["extra_rw_paths"] == ["/shared"]
        assert kwargs["attachments_path"] == "/workspace/attachments"
        assert kwargs["dir_map"] == [("/host/mount", "/sandbox/mount")]

    def test_resume_uses_session_and_prefixes_an_already_prefixed_newline(
        self,
    ) -> None:
        with patch(
            "symphony_linear.pi.agent_runner.run",
            return_value=(0, _fixture_text(), "", None),
        ) as runner:
            message, context_tokens = run_resume(
                workspace_path="/workspace",
                tmp_path="/workspace/tmp",
                session_id="pi-session",
                message="\ncontinue",
                timeout_seconds=60,
                idle_timeout_seconds=30,
                on_subprocess=lambda process: None,
                files=["/workspace/attachments/followup.txt"],
                model="anthropic/claude-opus-5",
            )

        assert (message, context_tokens) == ("hello", 3479)
        assert runner.call_args.kwargs["cmd"] == [
            "pi",
            "-p",
            "--mode",
            "json",
            "--no-approve",
            "--session",
            "pi-session",
            "--model",
            "anthropic/claude-opus-5",
            "@/workspace/attachments/followup.txt",
            "\n\ncontinue",
        ]

    def test_initial_binary_kwarg_overrides_argv0(self) -> None:
        with patch(
            "symphony_linear.pi.agent_runner.run",
            return_value=(0, _fixture_text(), "", None),
        ) as runner:
            run_initial(
                workspace_path="/workspace",
                tmp_path="/workspace/tmp",
                prompt="do it",
                timeout_seconds=60,
                idle_timeout_seconds=30,
                on_subprocess=lambda process: None,
                binary="tlh",
            )

        assert runner.call_args.kwargs["cmd"][0] == "tlh"

    def test_resume_binary_kwarg_overrides_argv0(self) -> None:
        with patch(
            "symphony_linear.pi.agent_runner.run",
            return_value=(0, _fixture_text(), "", None),
        ) as runner:
            run_resume(
                workspace_path="/workspace",
                tmp_path="/workspace/tmp",
                session_id="pi-session-abc",
                message="continue",
                timeout_seconds=60,
                idle_timeout_seconds=30,
                on_subprocess=lambda process: None,
                binary="tlh",
            )

        assert runner.call_args.kwargs["cmd"][0] == "tlh"

    @pytest.mark.parametrize("session_id", ["", None])
    def test_resume_rejects_empty_session_id(self, session_id: str | None) -> None:
        with patch("symphony_linear.pi.agent_runner.run") as runner:
            with pytest.raises(PiError, match="non-empty session_id"):
                run_resume(
                    workspace_path="/workspace",
                    tmp_path="/workspace/tmp",
                    session_id=session_id,
                    message="continue",
                    timeout_seconds=60,
                    idle_timeout_seconds=30,
                    on_subprocess=lambda process: None,
                )

        runner.assert_not_called()


class TestExitValidation:
    """The adapter rejects terminal protocol failures despite a zero exit."""

    def test_zero_exit_error_turn_raises_last_provider_error(self) -> None:
        with patch(
            "symphony_linear.pi.agent_runner.run",
            return_value=(0, _error_fixture_text(), "", None),
        ):
            with pytest.raises(PiError) as excinfo:
                run_initial(
                    workspace_path="/workspace",
                    tmp_path="/workspace/tmp",
                    prompt="implement this",
                    timeout_seconds=60,
                    idle_timeout_seconds=30,
                    on_subprocess=lambda process: None,
                )

        message = str(excinfo.value)
        assert "429 provider rate limit persisted after final retry" in message
        assert "429 retry attempt 3 failed" not in message


class TestSignalCancellation:
    """A killed pi turn stays a cancellation and salvages its session id."""

    def test_signal_exit_without_session_yields_none(self) -> None:
        with patch(
            "symphony_linear.pi.agent_runner.run",
            return_value=(-9, "not NDJSON", "", None),
        ):
            with pytest.raises(PiCancelled, match="killed by signal 9") as excinfo:
                run_initial(
                    workspace_path="/workspace",
                    tmp_path="/workspace/tmp",
                    prompt="implement this",
                    timeout_seconds=60,
                    idle_timeout_seconds=30,
                    on_subprocess=lambda process: None,
                )

        assert excinfo.value.session_id is None

    def test_signal_exit_carries_parsed_session_id(self) -> None:
        with patch(
            "symphony_linear.pi.agent_runner.run",
            return_value=(-9, _fixture_text(), "", None),
        ):
            with pytest.raises(PiCancelled, match="killed by signal 9") as excinfo:
                run_initial(
                    workspace_path="/workspace",
                    tmp_path="/workspace/tmp",
                    prompt="implement this",
                    timeout_seconds=60,
                    idle_timeout_seconds=30,
                    on_subprocess=lambda process: None,
                )

        assert excinfo.value.session_id == "01a035ff-248a-735c-8173-f5ee428fe917"


class TestExceptionAliases:
    """Adapter-facing exception names remain runner-compatible."""

    def test_exception_names_alias_agent_runner_exceptions(self) -> None:
        assert PiError is agent_runner.AgentError
        assert PiTimeout is agent_runner.AgentTimeout
        assert PiCancelled is agent_runner.AgentCancelled


# ---------------------------------------------------------------------------
# _build_env
# ---------------------------------------------------------------------------


class TestBuildEnv:
    """_build_env returns HOME always and PI_CODING_AGENT_DIR when set."""

    def test_home_always_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
        env = _build_env()
        assert env["HOME"] == str(Path.home())

    def test_pi_coding_agent_dir_forwarded_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PI_CODING_AGENT_DIR", "/custom/pi/agent")
        env = _build_env()
        assert env.get("PI_CODING_AGENT_DIR") == "/custom/pi/agent"

    def test_pi_coding_agent_dir_absent_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
        env = _build_env()
        assert "PI_CODING_AGENT_DIR" not in env

    def test_pi_coding_agent_session_dir_never_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PI_CODING_AGENT_SESSION_DIR", "/some/session/dir")
        env = _build_env()
        assert "PI_CODING_AGENT_SESSION_DIR" not in env


# ---------------------------------------------------------------------------
# _agent_dir_rw_paths
# ---------------------------------------------------------------------------


class TestAgentDirRwPaths:
    """_agent_dir_rw_paths appends PI_CODING_AGENT_DIR to extra_rw_paths."""

    def test_returns_unchanged_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
        assert _agent_dir_rw_paths(["/shared"]) == ["/shared"]

    def test_returns_empty_when_env_unset_and_no_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
        assert _agent_dir_rw_paths(None) == []

    def test_appends_when_env_set_and_dir_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        agent_dir = tmp_path / "pi_agent"
        agent_dir.mkdir()
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
        result = _agent_dir_rw_paths(None)
        assert str(agent_dir) in result

    def test_not_appended_when_dir_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        missing = tmp_path / "nonexistent"
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(missing))
        result = _agent_dir_rw_paths(None)
        assert str(missing) not in result

    def test_warning_logged_when_dir_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        missing = tmp_path / "nonexistent"
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(missing))
        with caplog.at_level(logging.WARNING, logger="symphony_linear.pi"):
            _agent_dir_rw_paths(None)
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert str(missing) in warnings[0]
        assert "EROFS" in warnings[0]

    def test_no_warning_when_env_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
        with caplog.at_level(logging.WARNING, logger="symphony_linear.pi"):
            _agent_dir_rw_paths(None)
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == []

    def test_no_warning_when_dir_exists(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        agent_dir = tmp_path / "pi_agent"
        agent_dir.mkdir()
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
        with caplog.at_level(logging.WARNING, logger="symphony_linear.pi"):
            _agent_dir_rw_paths(None)
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == []

    def test_tilde_expanded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Create a real dir and set env to a ~ path pointing at it so we can
        # check expansion without relying on the home dir layout.
        agent_dir = tmp_path / "pi_tilde_agent"
        agent_dir.mkdir()
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("PI_CODING_AGENT_DIR", "~/pi_tilde_agent")
        result = _agent_dir_rw_paths(None)
        # The appended path must not contain a literal tilde.
        assert any("pi_tilde_agent" in p and "~" not in p for p in result)

    def test_realpath_deduplication(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        agent_dir = tmp_path / "pi_agent"
        agent_dir.mkdir()
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
        # Pass the same path already in extra_rw_paths.
        result = _agent_dir_rw_paths([str(agent_dir)])
        assert result.count(str(agent_dir)) == 1

    def test_tilde_form_deduplication(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A tilde-form base entry dedupes against an expanded PI_CODING_AGENT_DIR."""
        agent_dir = tmp_path / "tlh_agent"
        agent_dir.mkdir()
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
        # Pass the tilde form; expanduser will resolve to agent_dir.
        result = _agent_dir_rw_paths(["~/tlh_agent"])
        # The agent dir must appear exactly once — no duplicate bind.
        paths_containing = [p for p in result if "tlh_agent" in p]
        assert len(paths_containing) == 1
