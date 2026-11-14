"""The control plane's view of the serving app (P4-7): tell it to reload, ask how it is doing.

The serving app lives on 127.0.0.1 at a port taken from settings (never from a request), so this
cannot be pointed anywhere else. It uses only the standard library. Everything here is best-effort:
a serving app that is stopped or slow never makes a deployment change fail, because PostgreSQL
already holds the change and the serving app reads it from there when it starts or polls.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

HOST = "127.0.0.1"
IN_SYNC = "in_sync"
OUT_OF_SYNC = "out_of_sync"
UNREACHABLE = "unreachable"


@dataclass(frozen=True)
class ServingReport:
    state: str  # in_sync | out_of_sync | unreachable
    loaded_deployment_id: int | None = None
    model_state: str | None = None
    error: str | None = None


class ServingClient:
    def __init__(self, port: int, timeout: float = 10.0) -> None:
        self.port = port
        self.timeout = timeout

    def _call(self, path: str, method: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"http://{HOST}:{self.port}{path}",
            data=b"" if method == "POST" else None,
            method=method,
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
