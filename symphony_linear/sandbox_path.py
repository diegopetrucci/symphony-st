"""PATH resolution shared by sandbox execution and startup validation."""

from __future__ import annotations

import os
from collections.abc import Mapping


_FALLBACK_PATH = "/usr/local/bin:/usr/bin:/bin"


def resolve_sandbox_path(env: Mapping[str, str] | None = None) -> str:
    """Return the PATH that a sandboxed command will receive.

    An explicit ``PATH`` in *env* takes precedence over
    ``SYMPHONY_SANDBOX_PATH``, which in turn takes precedence over the
    daemon's own ``PATH``. The hard-coded fallback is used when none is set.
    """
    if env is not None and "PATH" in env:
        return env["PATH"]
    if "SYMPHONY_SANDBOX_PATH" in os.environ:
        return os.environ["SYMPHONY_SANDBOX_PATH"]
    if "PATH" in os.environ:
        return os.environ["PATH"]
    return _FALLBACK_PATH
