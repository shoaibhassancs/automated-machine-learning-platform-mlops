"""Which address a service may listen on (Phase 6, Step 33).

Outside a container a service listens on the loopback interface only. Inside a container the
loopback interface is unreachable from the host, so the image sets `AUTOML_IN_CONTAINER=1` and the
service may listen on all interfaces *of the container*; the Compose file then publishes the port on
the host's 127.0.0.1 only. Both conditions must hold (the variable and Docker's `/.dockerenv`
marker), so setting the variable by hand on a normal machine does not open the service to the
network. One implementation, shared by the control-plane and serving entry points.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

LOOPBACK = ("127.0.0.1", "::1", "localhost")
ALL_INTERFACES = ("0.0.0.0",)  # noqa: S104 (allowed only inside a container, see `in_container`)
CONTAINER_MARKER = Path("/.dockerenv")


def in_container(env: Mapping[str, str] | None = None, marker: Path | None = None) -> bool:
    source = os.environ if env is None else env
    return source.get("AUTOML_IN_CONTAINER") == "1" and (marker or CONTAINER_MARKER).exists()


def host_problem(
    host: str, env: Mapping[str, str] | None = None, marker: Path | None = None
) -> str | None:
    """Why `host` may not be used (None: allowed)."""
    if host in LOOPBACK:
        return None
    if host in ALL_INTERFACES and in_container(env, marker):
        return None
    allowed = ", ".join(LOOPBACK)
    return f"refusing to bind to {host!r}; only {allowed} (0.0.0.0 only inside the project's image)"
