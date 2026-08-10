"""Per-ticket workspace lifecycle: clone, branch, setup, serve, remove.

Git operations run outside the sandbox using the daemon's credentials.
The ``.symphony/setup`` and ``.symphony/serve`` scripts run inside the sandbox
via the sandbox wrapper.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from symphony_linear.sandbox import run_in_sandbox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Character class for valid ticket identifier characters.
_VALID_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]")

# Default timeout for the setup script (5 minutes).
SETUP_TIMEOUT_SECONDS = 300

# Default branch name is derived from the ticket identifier (lowercased).
_DEFAULT_BRANCH_PREFIX = "symphony/"

# Subdirectories inside each per-ticket directory.
_REPO_DIR = "repo"
_ATTACHMENTS_DIR = "attachments"
_TMP_DIR = "tmp"


# ---------------------------------------------------------------------------
# Typed exceptions
# ---------------------------------------------------------------------------


class WorkspaceError(Exception):
    """Base exception for all workspace lifecycle errors."""


class CloneFailed(WorkspaceError):
    """Git clone operation failed."""


class BranchFailed(WorkspaceError):
    """Git branch switch / creation failed."""


class SetupFailed(WorkspaceError):
    """The ``.symphony/setup`` script exited with a non-zero code or timed out."""


class PathContainmentError(WorkspaceError):
    """Computed workspace path escapes the workspace root (security invariant)."""


class ServeScriptMissing(WorkspaceError):
    """The ``.symphony/serve`` script is absent or not executable."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sanitize_identifier(identifier: str) -> str:
    """Replace any character not in ``[A-Za-z0-9._-]`` with ``_``."""
    return _VALID_CHARS_RE.sub("_", identifier)


def compute_ticket_dir(ticket_identifier: str, workspace_root: str) -> str:
    """Return the per-ticket directory path for *ticket_identifier*.

    The path is ``<workspace_root>/<sanitized_identifier>/`` and holds all
    per-ticket state (``repo/``, ``attachments/``, ``tmp/``).
    This function does **not** create the directory or check containment.
    """
    workspace_key = _sanitize_identifier(ticket_identifier)
    return os.path.join(workspace_root, workspace_key)


def compute_workspace_path(ticket_identifier: str, workspace_root: str) -> str:
    """Return the workspace (git clone) directory path for *ticket_identifier*.

    The path is ``<workspace_root>/<sanitized_identifier>/repo/``.
    This function does **not** create the directory or check containment.
    """
    return os.path.join(
        compute_ticket_dir(ticket_identifier, workspace_root), _REPO_DIR
    )


def compute_attachments_path(ticket_identifier: str, workspace_root: str) -> str:
    """Return the per-ticket attachments directory path.

    The path is ``<workspace_root>/<sanitized_identifier>/attachments/``.
    This function does **not** create the directory or check containment.
    """
    return os.path.join(
        compute_ticket_dir(ticket_identifier, workspace_root), _ATTACHMENTS_DIR
    )


def compute_tmp_path(ticket_identifier: str, workspace_root: str) -> str:
    """Return the per-ticket tmp directory path.

    The path is ``<workspace_root>/<sanitized_identifier>/tmp/``.
    This function does **not** create the directory or check containment.
    """
    return os.path.join(compute_ticket_dir(ticket_identifier, workspace_root), _TMP_DIR)


def ensure_attachments_dir(ticket_identifier: str, workspace_root: str) -> str:
    """Create and return the per-ticket attachments directory, verified safe.

    1. Computes the path via :func:`compute_attachments_path`.
    2. Validates the path is contained within *workspace_root* (blocks symlink
       escapes).
    3. Creates the directory tree with mode ``0o700``.
    4. Returns the validated host path.

    Raises:
        PathContainmentError: If the computed path is not within
            *workspace_root* after realpath resolution.
    """
    attachments_dir = compute_attachments_path(ticket_identifier, workspace_root)
    _check_containment(attachments_dir, workspace_root)
    os.makedirs(attachments_dir, mode=0o700, exist_ok=True)
    return attachments_dir


def ensure_tmp_dir(ticket_identifier: str, workspace_root: str) -> str:
    """Create and return the per-ticket tmp directory, verified safe.

    1. Computes the path via :func:`compute_tmp_path`.
    2. Validates the path is contained within *workspace_root* (blocks symlink
       escapes).
    3. Creates the directory tree with mode ``0o700``.
    4. Returns the validated host path.

    Raises:
        PathContainmentError: If the computed path is not within
            *workspace_root* after realpath resolution.
    """
    tmp_dir = compute_tmp_path(ticket_identifier, workspace_root)
    _check_containment(tmp_dir, workspace_root)
    os.makedirs(tmp_dir, mode=0o700, exist_ok=True)
    return tmp_dir


def _check_containment(workspace_path: str, workspace_root: str) -> str:
    """Verify *workspace_path* resides within *workspace_root* after symlink
    resolution.

    Returns the real path of *workspace_path* on success.

    Raises:
        PathContainmentError: If the resolved workspace path is not a child of
            the resolved workspace root.
    """
    real_root = os.path.realpath(workspace_root)
    real_path = os.path.realpath(workspace_path)

    # Normalise to avoid trailing-slash mismatches.
    if not real_path.startswith(real_root + os.sep) and real_path != real_root:
        raise PathContainmentError(
            f"Workspace path '{workspace_path}' (resolved to '{real_path}') "
            f"is not contained within workspace root '{workspace_root}' "
            f"(resolved to '{real_root}')"
        )

    return real_path


def _run_git(
    args: list[str],
    cwd: str | None = None,
    *,
    description: str = "git operation",
) -> subprocess.CompletedProcess[str]:
    """Run a git command and return the completed process.

    Args:
        args: Git command arguments (without the leading ``git``).
        cwd: Working directory for the command.
        description: Human-readable label used in error messages.

    Returns:
        The completed process.

    Raises:
        CloneFailed / BranchFailed: Depending on context, if the process exits
            with a non-zero code.
    """
    cmd = ["git"] + args
    logger.debug("Running git: %s (cwd=%s)", " ".join(cmd), cwd)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr_tail = result.stderr.strip().splitlines()
        tail = "\n".join(stderr_tail[-5:]) if stderr_tail else "(no stderr)"
        logger.error("Git %s failed (rc=%d): %s", description, result.returncode, tail)

        # Distinguish clone from branch by looking at the sub-command.
        if args[0] == "clone":
            raise CloneFailed(f"git clone failed (rc={result.returncode}): {tail}")
        raise BranchFailed(f"git {args[0]} failed (rc={result.returncode}): {tail}")

    return result


def _run_setup_script(
    workspace_path: str,
    hide_paths: list[str],
    on_subprocess: Callable[[subprocess.Popen[bytes]], None] | None = None,
    extra_rw_paths: list[str] | None = None,
) -> None:
    """Run ``.symphony/setup`` inside the sandbox.

    Args:
        workspace_path: Host path to the workspace directory.
        hide_paths: Paths to conceal inside the sandbox.
        on_subprocess: Optional callback invoked with the Popen handle
            immediately after launch, for external cancellation.
        extra_rw_paths: Additional host paths to bind read-write inside the
            sandbox.

    Raises:
        SetupFailed: If the script exits with a non-zero code or times out.
    """
    setup_path = os.path.join(workspace_path, ".symphony", "setup")
    if not os.path.isfile(setup_path) or not os.access(setup_path, os.X_OK):
        logger.debug("No executable .symphony/setup found at %s – skipping", setup_path)
        return

    logger.info("Running .symphony/setup for workspace %s", workspace_path)

    proc = run_in_sandbox(
        cmd=["./.symphony/setup"],
        workspace_path=workspace_path,
        hide_paths=hide_paths,
        env={
            "HOME": os.environ.get("HOME", str(Path.home())),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        extra_rw_paths=extra_rw_paths or [],
    )

    if on_subprocess is not None:
        on_subprocess(proc)

    try:
        stdout_bytes, stderr_bytes = proc.communicate(timeout=SETUP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            _, stderr_bytes = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stderr_bytes = b"(timed out collecting stderr)"
        stderr_text = (
            stderr_bytes.decode(errors="replace")
            if isinstance(stderr_bytes, bytes)
            else str(stderr_bytes)
        )
        raise SetupFailed(
            f".symphony/setup timed out after {SETUP_TIMEOUT_SECONDS}s\n"
            f"stderr tail:\n{_tail(stderr_text)}"
        )

    stderr_text = stderr_bytes.decode(errors="replace") if stderr_bytes else ""

    if proc.returncode != 0:
        logger.error(
            ".symphony/setup failed (rc=%d) for workspace %s",
            proc.returncode,
            workspace_path,
        )
        raise SetupFailed(
            f".symphony/setup exited with code {proc.returncode}\n"
            f"stderr tail:\n{_tail(stderr_text)}"
        )

    logger.info(
        ".symphony/setup completed successfully for workspace %s", workspace_path
    )


def start_serve(
    workspace_path: str,
    hide_paths: list[str],
    extra_rw_paths: list[str] | None = None,
) -> subprocess.Popen[bytes]:
    """Launch ``.symphony/serve`` inside the sandbox and return the Popen handle.

    Unlike :func:`_run_setup_script`, this function does **not** wait for the
    process to finish — ``.symphony/serve`` is expected to be a long-running
    process.  The caller is responsible for managing the process lifetime
    (draining/closing pipes, killing, waiting).

    Args:
        workspace_path: Host path to the workspace directory.
        hide_paths: Paths to conceal inside the sandbox.
        extra_rw_paths: Additional host paths to bind read-write inside the
            sandbox.

    Returns:
        A :class:`~subprocess.Popen` instance for the sandboxed serve process.
        Both ``stdout`` and ``stderr`` are :data:`subprocess.PIPE` so the
        caller can capture stderr tail on early failure.

    Raises:
        ServeScriptMissing: If ``.symphony/serve`` is absent or not executable.
        FileNotFoundError: If ``bwrap`` is not available on ``$PATH``.
    """
    serve_path = os.path.join(workspace_path, ".symphony", "serve")
    if not os.path.isfile(serve_path) or not os.access(serve_path, os.X_OK):
        raise ServeScriptMissing(
            f".symphony/serve is missing or not executable at {serve_path}"
        )

    logger.info("Launching .symphony/serve for workspace %s", workspace_path)

    return run_in_sandbox(
        cmd=["./.symphony/serve"],
        workspace_path=workspace_path,
        hide_paths=hide_paths,
        env={
            "HOME": os.environ.get("HOME", str(Path.home())),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        extra_rw_paths=extra_rw_paths or [],
    )


def _tail(text: str, lines: int = 20) -> str:
    """Return the last *lines* lines of *text*."""
    if not text:
        return "(no stderr)"
    all_lines = text.strip().splitlines()
    return "\n".join(all_lines[-lines:]) if all_lines else "(no stderr)"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _git_switch_branch(
    branch_name: str,
    workspace_path: str,
) -> None:
    """Switch to *branch_name*, creating it if it does not exist.

    Tries ``git switch <branch>`` first (which auto-creates from a matching
    remote).  If that fails, falls back to ``git switch -c <branch>``.

    Raises:
        BranchFailed: If both attempts fail.
    """
    # Attempt 1: plain switch (works if branch exists locally or on remote).
    result = subprocess.run(
        ["git", "switch", branch_name],
        cwd=workspace_path,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.debug("Switched to existing branch '%s'", branch_name)
        return

    logger.debug(
        "git switch '%s' failed (rc=%d), trying -c: %s",
        branch_name,
        result.returncode,
        result.stderr.strip().splitlines()[-1]
        if result.stderr.strip()
        else "(no stderr)",
    )

    # Attempt 2: create a new branch from HEAD.
    result = subprocess.run(
        ["git", "switch", "-c", branch_name],
        cwd=workspace_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr_tail = result.stderr.strip().splitlines()
        tail = "\n".join(stderr_tail[-5:]) if stderr_tail else "(no stderr)"
        raise BranchFailed(
            f"git switch -c {branch_name} failed (rc={result.returncode}): {tail}"
        )

    logger.debug("Created and switched to new branch '%s'", branch_name)


def _git_failure_summary(command: str, stderr: str) -> str:
    """Build a short dirty summary for a failed *command*.

    Kept small — the summary is destined for a tracker comment.
    """
    stderr_tail = stderr.strip().splitlines()
    tail = "\n".join(stderr_tail[-3:]) if stderr_tail else "(no stderr)"
    return f"Could not verify workspace state (`{command}` failed): {tail}"


def dirty_summary(path: str) -> str | None:
    """Return a short Markdown summary of uncommitted work at *path*.

    Returns ``None`` when there is nothing to protect: *path* is not a
    directory, or the workspace is clean — ``git status --porcelain`` is
    empty **and** ``git rev-list --branches --not --remotes`` is empty (no
    local-only commits that aren't mirrored on any remote).

    Otherwise returns a short human-readable summary suitable for a tracker
    comment: counts of uncommitted files and commits not on any remote, plus
    up to 10 ``git status --porcelain`` lines in a fenced block.

    If any git command fails or raises (e.g. the path is not a repository),
    the workspace is treated as dirty and the summary says so — conservative,
    since the caller can decide to override.
    """
    if not os.path.isdir(path):
        return None

    # Check 1: working tree and index.  Pass --untracked-files=normal
    # explicitly so a status.showUntrackedFiles=no config (repo or global)
    # cannot hide untracked work — a false "clean" would let the workspace
    # be deleted.
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=path,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        logger.debug("git status failed during dirtiness check", exc_info=True)
        return f"Could not verify workspace state (`git status` errored: {exc})."
    if result.returncode != 0:
        return _git_failure_summary("git status", result.stderr)
    dirty_files = [line for line in result.stdout.splitlines() if line.strip()]

    # Check 2: local-only commits.  Check ALL local branches (--branches),
    # not just HEAD — the agent may have committed to a side branch while
    # HEAD sits on a remote-backed branch, and those commits would be the
    # only copy of the work.  Compare against ALL remote-tracking refs
    # (--not --remotes), not just the current branch's upstream (@{u}).
    # No --tags: a tag on a commit no remote branch holds is deliberate
    # release tooling, not unmirrored work.  In no-push setups local-only
    # commits are the only copy of the agent's work, so every local branch
    # must be considered before we call the workspace clean.
    try:
        result = subprocess.run(
            ["git", "rev-list", "--branches", "--not", "--remotes"],
            cwd=path,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        logger.debug("git rev-list failed during dirtiness check", exc_info=True)
        return f"Could not verify workspace state (`git rev-list` errored: {exc})."
    if result.returncode != 0:
        return _git_failure_summary("git rev-list", result.stderr)
    unpushed_commits = [line for line in result.stdout.splitlines() if line.strip()]

    if not dirty_files and not unpushed_commits:
        return None

    n_files = len(dirty_files)
    n_commits = len(unpushed_commits)
    summary = (
        f"{n_files} uncommitted file{'' if n_files == 1 else 's'}, "
        f"{n_commits} commit{'' if n_commits == 1 else 's'} not on any remote."
    )
    if dirty_files:
        block = "\n".join(dirty_files[:10])
        summary += f"\n\n```text\n{block}\n```"
    return summary


def clone_workspace(
    ticket_identifier: str,
    repo_url: str,
    workspace_root: str,
) -> tuple[str, bool]:
    """Clone or fetch the repository for *ticket_identifier*.

    1. Compute the workspace path (``<ticket_dir>/repo``, sanitizing the
       identifier) and verify it is within *workspace_root*.
    2. Clone the repository into ``repo/`` if it does not already exist,
       otherwise fetch to pick up new remote branches.
    3. On fetch failure: if the workspace is clean, nuke ``repo/`` and
       re-clone from scratch (no exception).  If it has local state to
       preserve, log a warning and return normally (no exception).

    .. note::
       Clone has no ``-b`` — it checks out the remote's default branch.
       Branch selection is performed later by :func:`finalize_workspace`.

    Args:
        ticket_identifier: Human-readable ticket ID (e.g. ``TEAM-42``).
        repo_url: Git clone URL (supports local paths for testing).
        workspace_root: Root directory under which all workspaces live.

    Returns:
        A ``(path, recovered)`` tuple where *path* is the real path to the
        cloned/existing workspace and *recovered* is ``True`` only when the
        workspace had to be nuked and re-cloned after a fetch failure.

    Raises:
        PathContainmentError: If the computed workspace path escapes
            *workspace_root*.
        CloneFailed: If ``git clone`` fails (initial clone or re-clone).
    """
    # 1. Compute and validate workspace path
    workspace_path = compute_workspace_path(ticket_identifier, workspace_root)
    real_path = _check_containment(workspace_path, workspace_root)

    # 3. Clone if the workspace does not exist; refresh if it does.
    if not os.path.isdir(real_path):
        logger.info("Cloning %s into %s", repo_url, real_path)
        _run_git(
            ["clone", repo_url, real_path],
            description="clone",
        )
        return real_path, False

    logger.info("Workspace %s already exists – reusing", real_path)
    result = subprocess.run(
        ["git", "fetch", "origin"],
        cwd=real_path,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return real_path, False

    # Fetch failed.  Decide whether we can safely nuke and re-clone.
    stderr_tail = result.stderr.strip().splitlines()
    tail = "\n".join(stderr_tail[-5:]) if stderr_tail else "(no stderr)"
    logger.debug("git fetch failed (rc=%d): %s", result.returncode, tail)

    if dirty_summary(real_path) is None:
        logger.info(
            "Workspace %s is clean — nuking and re-cloning after fetch failure",
            real_path,
        )
        shutil.rmtree(real_path, ignore_errors=False)
        _run_git(
            ["clone", repo_url, real_path],
            description="clone (recovery)",
        )
        return real_path, True

    logger.warning(
        "Workspace %s has local state — preserving after fetch failure: %s",
        real_path,
        tail,
    )
    return real_path, False


def finalize_workspace(
    workspace_path: str,
    ticket_identifier: str,
    branch_name: str | None,
    sandbox_hide_paths: list[str],
    on_subprocess: Callable[[subprocess.Popen[bytes]], None] | None = None,
    sandbox_extra_rw_paths: list[str] | None = None,
    auto_branch: bool = True,
) -> None:
    """Finalize a cloned workspace for *ticket_identifier*.

    1. Determine the target branch name (default: ``symphony/<id_lower>``).
    2. Switch to (or create) the target branch (skipped when ``auto_branch``
       is false — the workspace stays on whatever :func:`clone_workspace`
       checked out).
    3. Run ``.symphony/setup`` inside the sandbox if present and executable.

    This function is idempotent: re-calling it on an existing workspace will
    switch to the right branch (when ``auto_branch`` is true) and re-run setup.

    Args:
        workspace_path: Real path to the cloned workspace (from
            :func:`clone_workspace`).
        ticket_identifier: Human-readable ticket ID (e.g. ``TEAM-42``).
        branch_name: Target branch name.  If ``None``, defaults to
            ``symphony/<identifier_lower>``. Ignored when ``auto_branch`` is
            false.
        sandbox_hide_paths: Paths to conceal inside the sandbox when running
            the setup script.
        on_subprocess: Optional callback invoked with the Popen handle of the
            setup script (if any), for external cancellation.
        sandbox_extra_rw_paths: Additional host paths to bind read-write inside
            the sandbox when running the setup script.
        auto_branch: If true (default), switch to a per-ticket branch after
            clone/fetch. If false, skip the branch switch entirely and leave
            the workspace on the cloned default branch.

    Raises:
        BranchFailed: If ``git switch`` fails.
        SetupFailed: If ``.symphony/setup`` fails or times out.
    """
    # Determine the target branch name (only relevant when auto_branch is on).
    if branch_name is None:
        branch_name = f"{_DEFAULT_BRANCH_PREFIX}{ticket_identifier.lower()}"

    # 1. Switch to (or create) the target branch — unless disabled.
    if auto_branch:
        logger.info("Switching to branch '%s' in %s", branch_name, workspace_path)
        _git_switch_branch(branch_name, workspace_path)
    else:
        logger.info(
            "auto_branch disabled — staying on cloned default branch in %s",
            workspace_path,
        )

    # 2. Run setup script if present
    _run_setup_script(
        workspace_path,
        sandbox_hide_paths,
        on_subprocess=on_subprocess,
        extra_rw_paths=sandbox_extra_rw_paths,
    )


def prepare(
    ticket_identifier: str,
    repo_url: str,
    branch_name: str | None,
    workspace_root: str,
    sandbox_hide_paths: list[str],
    on_subprocess: Callable[[subprocess.Popen[bytes]], None] | None = None,
    sandbox_extra_rw_paths: list[str] | None = None,
    auto_branch: bool = True,
) -> str:
    """Prepare a workspace for *ticket_identifier*.

    Thin wrapper that calls :func:`clone_workspace` followed by
    :func:`finalize_workspace`.  See those functions for full documentation
    of each step.

    Also creates the per-ticket attachments and tmp directories at
    ``<workspace_root>/<sanitized_identifier>/{attachments,tmp}/`` with
    mode 0700.

    Returns:
        The real path to the prepared workspace.

    Raises:
        PathContainmentError: If the computed workspace path escapes
            *workspace_root*.
        CloneFailed: If ``git clone`` or ``git fetch`` fails.
        BranchFailed: If ``git switch`` fails.
        SetupFailed: If ``.symphony/setup`` fails or times out.
    """
    real_path, _ = clone_workspace(ticket_identifier, repo_url, workspace_root)
    finalize_workspace(
        workspace_path=real_path,
        ticket_identifier=ticket_identifier,
        branch_name=branch_name,
        sandbox_hide_paths=sandbox_hide_paths,
        on_subprocess=on_subprocess,
        sandbox_extra_rw_paths=sandbox_extra_rw_paths,
        auto_branch=auto_branch,
    )

    # Ensure the per-ticket attachments and tmp directories exist.
    ensure_attachments_dir(ticket_identifier, workspace_root)
    ensure_tmp_dir(ticket_identifier, workspace_root)

    return real_path


def remove(
    ticket_identifier: str,
    workspace_root: str,
) -> None:
    """Delete the per-ticket directory (repo, attachments, tmp) for
    *ticket_identifier*.

    Idempotent — no error if the ticket directory is already gone.

    Args:
        ticket_identifier: Human-readable ticket ID.
        workspace_root: Root directory for workspaces.

    Raises:
        PathContainmentError: If the computed path escapes *workspace_root*.
    """
    ticket_dir = compute_ticket_dir(ticket_identifier, workspace_root)

    # Verify containment before removing anything.
    _check_containment(ticket_dir, workspace_root)

    if os.path.isdir(ticket_dir):
        logger.info("Removing ticket directory %s", ticket_dir)
        shutil.rmtree(ticket_dir, ignore_errors=False)
        logger.info("Ticket directory %s removed", ticket_dir)
    else:
        logger.debug(
            "Ticket directory %s does not exist – nothing to remove", ticket_dir
        )
