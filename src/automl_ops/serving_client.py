"""The control plane's view of the serving app (P4-7): tell it to reload, ask how it is doing.

The serving app lives at a host and port taken from settings (never from a request), so this
cannot be pointed at anything a caller chooses. Outside a container the host is always 127.0.0.1.
Inside the project's image (see `bind.in_container`) it may instead be one bare Compose service name
such as `serving`, so the two containers stay reachable after either restarts; the request still
carries the loopback `Host` header, so the serving app's strict Host check is unchanged. It uses
only the standard library. Everything here is best-effort: a serving app that is stopped or slow
never makes a deployment change fail, because PostgreSQL already holds the change and the serving
app reads it from there when it starts or polls.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from automl_ops.bind import LOOPBACK, in_container

HOST = "127.0.0.1"
_SERVICE_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
IN_SYNC = "in_sync"
OUT_OF_SYNC = "out_of_sync"
UNREACHABLE = "unreachable"


def host_problem(host: str, env: dict[str, str] | None = None) -> str | None:
    """Why `host` may not be used as the serving app's address (None: allowed). Loopback always;
    a bare service name (no dots, no address) only inside the project's image."""
    if host in LOOPBACK:
        return None
    if _SERVICE_NAME.fullmatch(host) and in_container(env):
        return None
    return (
        f"AUTOML_SERVING_HOST {host!r} is not allowed: use 127.0.0.1, or a bare Compose service "
        "name inside the project's image"
    )


@dataclass(frozen=True)
class ServingReport:
    state: str  # in_sync | out_of_sync | unreachable
    loaded_deployment_id: int | None = None
    model_state: str | None = None
    error: str | None = None


class ServingClient:
    def __init__(self, port: int, timeout: float = 10.0, host: str = HOST) -> None:
        problem = host_problem(host)
        if problem is not None:
            raise ValueError(problem)
        self.port = port
        self.timeout = timeout
        self.host = host

    def _call(self, path: str, method: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"http://{self.host}:{self.port}{path}",
            data=b"" if method == "POST" else None,
            method=method,
            # the serving app accepts only loopback Host values, whatever name we connect by
            headers={"Host": f"{HOST}:{self.port}"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            body = json.loads(response.read(65536))
        if not isinstance(body, dict):
            raise ValueError("unexpected response")
        return body

    def report(self, desired_deployment_id: int | None, *, reload: bool) -> ServingReport:
        """Ask the serving app to re-read the active deployment (`reload`) or just report, and
        compare what it says it serves with `desired_deployment_id`."""
        try:
            body = self._call("/reload" if reload else "/health", "POST" if reload else "GET")
            loaded = body.get("loaded_deployment_id")
            model_state = str(body.get("model_state"))
            error = body.get("error")
        except (OSError, ValueError, urllib.error.URLError) as error_:
            return ServingReport(
                UNREACHABLE, error=f"the serving app did not answer ({type(error_).__name__})"
            )
        in_sync = (desired_deployment_id is None and model_state == "no_deployment") or (
            desired_deployment_id is not None
            and loaded == desired_deployment_id
            and model_state == "ready"
        )
        return ServingReport(
            IN_SYNC if in_sync else OUT_OF_SYNC,
            loaded if isinstance(loaded, int) else None,
            model_state,
            None if in_sync or error is None else str(error),
        )
