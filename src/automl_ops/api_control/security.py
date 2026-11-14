"""Local-only request rules, as one pure-ASGI middleware (D-036).

In order, for every HTTP request:

1. **Host**: the `Host` header must be exactly one of 127.0.0.1, localhost or [::1] with the API's
   port. This is what defeats DNS rebinding (a hostile page whose DNS name points at 127.0.0.1).
2. **CORS preflight**: answered only for the single configured UI origin; refused for any other.
3. **Origin on state-changing requests** (POST, PUT, PATCH, DELETE): if an `Origin` header is
   present it must equal the configured UI origin exactly (scheme, host and port compared as parsed
   values). `Origin: null`, malformed values, repeated headers and any other origin are refused.
   A request with no `Origin` (curl, scripts, the test client) is allowed: browsers always send one
   on cross-site POSTs, so this closes the "any web page can POST a form to localhost" hole. This
   is not authentication; it reduces the browser attack surface of an unauthenticated local API.
4. **Content type** of state-changing requests that have a body: `multipart/form-data` for the
   upload, otherwise `application/json`.
5. **Body size**, enforced while streaming: a declared `Content-Length` over the limit is refused
   before any byte is read; a chunked or understated body is cut off the moment the count passes
   the limit, so nothing larger than the limit is ever spooled to disk or memory.

Responses to the one configured UI origin get CORS headers; no other origin ever does, and
credentials are never allowed.
"""

from __future__ import annotations

from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from automl_ops.api_control.errors import envelope
from automl_ops.api_control.settings import JSON_BODY_LIMIT_BYTES, ApiSettings, origin_of

STATE_CHANGING = frozenset({"POST", "PUT", "PATCH", "DELETE"})
UPLOAD_PATH = "/datasets"
CORS_METHODS = ("GET", "POST")
CORS_HEADERS = frozenset({"content-type"})
PREFLIGHT_MAX_AGE = "600"


class _BodyTooLarge(Exception):
    pass


class _BodyState:
    def __init__(self) -> None:
        self.received = 0
        self.exceeded = False
        self.replaced = False


def _headers(scope: Scope) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for raw_name, raw_value in scope.get("headers", []):
        found.setdefault(raw_name.decode("latin-1").lower(), []).append(raw_value.decode("latin-1"))
    return found


class LocalSecurityMiddleware:
    def __init__(self, app: ASGIApp, settings: ApiSettings) -> None:
        self.app = app
        self.settings = settings

    def _body_limit(self, method: str, path: str) -> int:
        if method == "POST" and path == UPLOAD_PATH:
            return self.settings.upload_body_limit
        return JSON_BODY_LIMIT_BYTES

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            await self.app(scope, receive, send)
            return

        method: str = scope["method"]
        path: str = scope["path"]
        headers = _headers(scope)

        hosts = headers.get("host", [])
        if len(hosts) != 1 or hosts[0].lower() not in self.settings.allowed_hosts:
            await envelope(400, "host_not_allowed", "This host name is not allowed.")(
                scope, receive, send
            )
            return

        origins = headers.get("origin", [])
        origin_ok = len(origins) == 1 and origin_of(origins[0]) == self.settings.ui_origin_tuple

        async def refuse(status: int, code: str, message: str) -> None:
            await envelope(status, code, message)(scope, receive, self._cors_send(send, origin_ok))

        if method == "OPTIONS" and origins and "access-control-request-method" in headers:
            await self._preflight(scope, receive, send, headers, origin_ok)
            return

        if method in STATE_CHANGING:
            if origins and not origin_ok:
                await refuse(403, "origin_not_allowed", "This origin is not allowed.")
                return
            has_body = headers.get("content-length", ["0"])[0] not in ("", "0") or (
                "transfer-encoding" in headers
            )
            media = (headers.get("content-type", [""])[0].split(";")[0].strip()).lower()
            upload = method == "POST" and path == UPLOAD_PATH
            expected = "multipart/form-data" if upload else "application/json"
            # A request with no body cannot carry a form or a file; the router answers it
            # (405, 404 or a validation error for the missing body).
            if has_body and media != expected:
                await refuse(415, "unsupported_media_type", f"Send this request as {expected}.")
                return

        limit = self._body_limit(method, path)
        declared = headers.get("content-length", [None])[0]
        if declared is not None:
            if not declared.isdigit():
                await refuse(400, "bad_request", "The Content-Length header is not valid.")
                return
            if int(declared) > limit:
                await refuse(413, "body_too_large", self._too_large_message(limit))
                return

        state = _BodyState()

        async def counting_receive() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                state.received += len(message.get("body", b""))
                if state.received > limit:
                    state.exceeded = True
                    raise _BodyTooLarge
            return message

        cors_send = self._cors_send(send, origin_ok)

        async def guarded_send(message: Message) -> None:
            if state.replaced:
                return  # the original response was replaced by the 413 below
            if state.exceeded and message["type"] == "http.response.start":
                state.replaced = True
                await envelope(413, "body_too_large", self._too_large_message(limit))(
                    scope, receive, cors_send
                )
                return
            await cors_send(message)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except _BodyTooLarge:
            if not state.replaced:
                state.replaced = True
                await envelope(413, "body_too_large", self._too_large_message(limit))(
                    scope, receive, cors_send
                )

    @staticmethod
    def _too_large_message(limit: int) -> str:
        return f"The request body is larger than the {limit:,}-byte limit for this endpoint."

    def _cors_send(self, send: Send, allowed: bool) -> Send:
        if not allowed:
            return send

        async def wrapped(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers: list[Any] = list(message.get("headers", []))
                headers.append((b"access-control-allow-origin", self.settings.ui_origin.encode()))
                headers.append((b"vary", b"Origin"))
                message = {**message, "headers": headers}
            await send(message)

        return wrapped

    async def _preflight(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        headers: dict[str, list[str]],
        origin_ok: bool,
    ) -> None:
        wanted_method = headers["access-control-request-method"][0].strip().upper()
        wanted_headers = {
            h.strip().lower()
            for value in headers.get("access-control-request-headers", [])
            for h in value.split(",")
            if h.strip()
        }
        if not origin_ok or wanted_method not in CORS_METHODS or not wanted_headers <= CORS_HEADERS:
            await envelope(403, "cors_not_allowed", "This cross-origin request is not allowed.")(
                scope, receive, send
            )
            return
        await send(
            {
                "type": "http.response.start",
                "status": 204,
                "headers": [
                    (b"access-control-allow-origin", self.settings.ui_origin.encode()),
                    (b"access-control-allow-methods", ", ".join(CORS_METHODS).encode()),
                    (b"access-control-allow-headers", b"content-type"),
                    (b"access-control-max-age", PREFLIGHT_MAX_AGE.encode()),
                    (b"vary", b"Origin"),
                    (b"content-length", b"0"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": b""})
