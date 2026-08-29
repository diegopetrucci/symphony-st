"""Tests for the tracker protocol utilities."""

from __future__ import annotations

import logging

import pytest

from symphony_linear.linear import Issue, Project
from symphony_linear.tracker import (
    is_bot_comment,
    model_for_issue,
    model_from_labels,
    model_label_name,
)


def _make_issue(labels: list[str], project_labels: list[str] | None) -> Issue:
    """Build a minimal Issue with *labels*, and a project unless None."""
    project = (
        None
        if project_labels is None
        else Project(id="proj-1", name="Test Project", labels=project_labels)
    )
    return Issue(
        id="ticket-1",
        identifier="TEAM-1",
        title="Test ticket",
        state="In Progress",
        labels=labels,
        project=project,
        updatedAt="2025-06-01T00:00:00Z",
    )


class TestIsBotComment:
    def test_positive_workspace_kind(self) -> None:
        """A body containing the '*Symphony · workspace*' footer returns True."""
        assert is_bot_comment("Some work done\n\n*Symphony · workspace*") is True

    def test_positive_context_tokens_variant(self) -> None:
        """A body containing the '*Symphony · context: ... tokens*' footer returns True."""
        assert is_bot_comment("Done.\n\n*Symphony · context: 37,074 tokens*") is True

    def test_negative_plain_human_comment(self) -> None:
        """A plain human comment with no marker returns False."""
        assert is_bot_comment("This is a regular human comment.") is False

    def test_negative_partial_match_no_dot(self) -> None:
        """A body containing 'Symphony' without the middle dot returns False."""
        assert is_bot_comment("I like Symphony music") is False

    def test_negative_only_marker_substring(self) -> None:
        """Only the '*Symphony · ' substring without the closing '*' returns True."""
        # The marker is '*Symphony · ', trailing content does not matter.
        assert is_bot_comment("Foo\n\n*Symphony · ") is True

    def test_negative_empty_body(self) -> None:
        """An empty string returns False."""
        assert is_bot_comment("") is False

    def test_negative_none_raises(self) -> None:
        """None raises TypeError (since str methods cannot be called on None)."""
        with pytest.raises(TypeError):
            is_bot_comment(None)  # type: ignore[arg-type]


class TestModelFromLabels:
    """Unit tests for the Model: <value> label resolver."""

    ALIASES = {"strong": "anthropic/claude-opus-5", "cheap": "openai/gpt-5-mini"}

    def test_no_matching_label_returns_none(self) -> None:
        assert model_from_labels(["Agent", "Bug"], self.ALIASES) is None

    def test_empty_labels_returns_none(self) -> None:
        assert model_from_labels([], self.ALIASES) is None

    def test_alias_hit_returns_mapped_id(self) -> None:
        assert (
            model_from_labels(["Agent", "Model: strong"], self.ALIASES)
            == "anthropic/claude-opus-5"
        )

    def test_prefix_is_case_insensitive(self) -> None:
        assert (
            model_from_labels(["MODEL: strong"], self.ALIASES)
            == "anthropic/claude-opus-5"
        )

    def test_alias_lookup_is_case_insensitive(self) -> None:
        assert (
            model_from_labels(["Model: STRONG"], self.ALIASES)
            == "anthropic/claude-opus-5"
        )

    def test_unknown_value_passed_through_verbatim(self) -> None:
        raw = "anthropic/claude-sonnet-4-6"
        assert model_from_labels([f"Model: {raw}"], self.ALIASES) == raw

    def test_value_is_stripped(self) -> None:
        assert (
            model_from_labels(["Model:   strong   "], self.ALIASES)
            == "anthropic/claude-opus-5"
        )

    def test_label_is_stripped(self) -> None:
        assert (
            model_from_labels(["  Model: strong  "], self.ALIASES)
            == "anthropic/claude-opus-5"
        )

    def test_empty_value_ignored_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            assert model_from_labels(["Model: "], self.ALIASES) is None
        assert any("empty" in rec.message for rec in caplog.records)

    def test_multiple_labels_first_in_sorted_order(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            result = model_from_labels(["Model: zeta", "Model: alpha"], self.ALIASES)
        assert result == "alpha"
        assert any("Multiple" in rec.message for rec in caplog.records)

    def test_multiple_labels_skip_empty_candidate(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            result = model_from_labels(["Model: ", "Model: strong"], self.ALIASES)
        assert result == "anthropic/claude-opus-5"
        # The multiple-match warning must name the label actually selected,
        # not the (skipped) empty candidate that sorts first.
        assert any("'Model: strong'" in rec.message for rec in caplog.records)

    def test_near_prefix_labels_do_not_match(self) -> None:
        assert model_from_labels(["Model", "modelish"], self.ALIASES) is None

    def test_empty_value_warning_names_the_ticket_by_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            assert model_from_labels(["Model: "], self.ALIASES) is None
        assert any("on ticket" in rec.message for rec in caplog.records)

    def test_multiple_labels_warning_names_the_ticket_by_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            model_from_labels(["Model: zeta", "Model: alpha"], self.ALIASES)
        assert any("labels on ticket" in rec.message for rec in caplog.records)

    def test_source_is_interpolated_into_both_warnings(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            result = model_from_labels(
                ["Model: ", "Model: strong"], self.ALIASES, source="project"
            )
        assert result == "anthropic/claude-opus-5"
        messages = [rec.message for rec in caplog.records]
        assert any("empty value" in m and "on project" in m for m in messages)
        assert any("labels on project" in m for m in messages)
        assert not any("ticket" in m for m in messages)


class TestModelForIssue:
    """Unit tests for the two-tier issue > project resolver."""

    ALIASES = {"strong": "anthropic/claude-opus-5", "cheap": "openai/gpt-5-mini"}

    def test_issue_label_wins_over_project_label(self) -> None:
        issue = _make_issue(["Agent", "Model: strong"], ["Model: cheap"])
        assert model_for_issue(issue, self.ALIASES) == "anthropic/claude-opus-5"

    def test_project_label_used_when_issue_has_none(self) -> None:
        issue = _make_issue(["Agent"], ["Team: core", "Model: cheap"])
        assert model_for_issue(issue, self.ALIASES) == "openai/gpt-5-mini"

    def test_no_label_on_either_tier_returns_none(self) -> None:
        issue = _make_issue(["Agent"], ["Team: core"])
        assert model_for_issue(issue, self.ALIASES) is None

    def test_no_project_returns_none(self) -> None:
        """A ticket with no project (e.g. the GitHub backend) has one tier."""
        issue = _make_issue(["Agent"], None)
        assert model_for_issue(issue, self.ALIASES) is None

    def test_no_project_still_honours_issue_label(self) -> None:
        issue = _make_issue(["Model: cheap"], None)
        assert model_for_issue(issue, self.ALIASES) == "openai/gpt-5-mini"

    def test_project_unknown_alias_passed_through_verbatim(self) -> None:
        raw = "anthropic/claude-sonnet-4-6"
        issue = _make_issue(["Agent"], [f"Model: {raw}"])
        assert model_for_issue(issue, self.ALIASES) == raw

    def test_empty_issue_label_falls_through_to_project(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unusable issue-tier label is not an override, so the project
        default still applies."""
        issue = _make_issue(["Model: "], ["Model: strong"])
        with caplog.at_level(logging.WARNING):
            assert model_for_issue(issue, self.ALIASES) == "anthropic/claude-opus-5"
        assert any("empty" in rec.message for rec in caplog.records)

    def test_empty_project_labels_returns_none(self) -> None:
        issue = _make_issue(["Agent"], [])
        assert model_for_issue(issue, self.ALIASES) is None

    def test_empty_project_label_warning_names_the_project(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A bad label on the project must not send the operator to the
        ticket, which carries no Model: label at all."""
        issue = _make_issue(["Agent"], ["Model: "])
        with caplog.at_level(logging.WARNING):
            assert model_for_issue(issue, self.ALIASES) is None
        messages = [rec.message for rec in caplog.records]
        assert any("empty value" in m and "on project" in m for m in messages)
        assert not any("ticket" in m for m in messages)

    def test_multiple_project_labels_warning_names_the_project(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        issue = _make_issue(["Agent"], ["Model: zeta", "Model: alpha"])
        with caplog.at_level(logging.WARNING):
            assert model_for_issue(issue, self.ALIASES) == "alpha"
        messages = [rec.message for rec in caplog.records]
        assert any("labels on project" in m for m in messages)
        assert not any("ticket" in m for m in messages)

    def test_empty_issue_label_warning_names_the_ticket(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The issue tier keeps the original wording."""
        issue = _make_issue(["Model: "], ["Model: strong"])
        with caplog.at_level(logging.WARNING):
            assert model_for_issue(issue, self.ALIASES) == "anthropic/claude-opus-5"
        assert any("on ticket" in rec.message for rec in caplog.records)
        assert not any("project" in rec.message for rec in caplog.records)


class TestModelLabelName:
    """Unit tests for the Model: <alias> label name builder."""

    def test_simple_alias(self) -> None:
        assert model_label_name("Strong") == "Model: Strong"

    def test_alias_with_spaces_kept_verbatim(self) -> None:
        assert model_label_name("my alias") == "Model: my alias"

    def test_roundtrip_resolves_via_model_from_labels(self) -> None:
        """A label built by model_label_name resolves through the parser."""
        assert (
            model_from_labels(
                [model_label_name("Cheap")], {"cheap": "openai/gpt-5-mini"}
            )
            == "openai/gpt-5-mini"
        )
