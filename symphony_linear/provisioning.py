"""Trigger- and model-label provisioning on daemon startup.

Ensures the configured ``linear.trigger_label`` and ``Model: <alias>`` labels
exist as workspace-wide labels in Linear, creating them if necessary.
"""

from __future__ import annotations

import logging

from symphony_linear.linear import LinearClient
from symphony_linear.state import StateManager

logger = logging.getLogger(__name__)


def provision_trigger_label(
    linear: LinearClient,
    state: StateManager,
    trigger_label: str,
) -> None:
    """Ensure *trigger_label* exists as a workspace-wide label in Linear.

    Compares against the last provisioned label name in state to avoid
    redundant API calls.  On any failure, logs a warning and continues
    startup – the daemon must not crash because of a label provisioning
    error.
    """
    # Already provisioned for this exact label name – nothing to do.
    if state.provisioned_label_name == trigger_label:
        return

    if _ensure_label(linear, trigger_label) is None:
        return

    # Success – record that this label name has been provisioned.
    try:
        state.set_provisioned_label_name(trigger_label)
    except Exception as exc:
        _warn(trigger_label, exc)
        return

    logger.info("Trigger label '%s' is provisioned.", trigger_label)


def provision_model_labels(linear: LinearClient, labels: list[str]) -> None:
    """Ensure each model alias label in *labels* exists in Linear.

    Unlike the trigger label there is no state caching: a handful of extra
    find queries once per startup is not worth a new state field.  On any
    failure, logs a warning and continues – a provisioning failure must not
    stop the daemon.
    """
    for name in labels:
        _ensure_label(linear, name)


def _ensure_label(linear: LinearClient, name: str) -> str | None:
    """Find (or create) a workspace-wide label named *name* in Linear.

    Race-tolerant: if the create fails because another caller created the
    label in between, one more lookup resolves it.  Returns the label id,
    or ``None`` on failure after logging a warning.  Never raises – a
    provisioning failure must not stop the daemon.
    """
    # Try to find an existing workspace-wide label with this name.
    try:
        label_id = linear.find_workspace_label(name)
    except Exception as exc:
        _warn(name, exc)
        return None

    if label_id is None:
        # Label doesn't exist yet – create it.
        try:
            label_id = linear.create_workspace_label(name)
        except Exception as exc:
            # Race: another caller may have created the label between our
            # find and create.  Try one more lookup.
            try:
                label_id = linear.find_workspace_label(name)
            except Exception as lookup_exc:
                _warn(name, lookup_exc)
                return None
            if label_id is None:
                _warn(name, exc)
                return None

    return label_id


def _warn(label_name: str, exc: Exception) -> None:
    """Log an actionable warning when provisioning fails."""
    logger.warning(
        "Failed to auto-provision Linear label '%s': %s. "
        "Create it manually in Linear if needed.",
        label_name,
        exc,
    )
