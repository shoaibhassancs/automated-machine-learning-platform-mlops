"""One error format for every API response: `{"error": {"code", "message", "details"}}`.

Messages are written for people and never contain tracebacks, file contents, credentials or
filesystem paths. Framework errors (404, 405, validation, malformed bodies) are reshaped here too;
anything unexpected becomes a generic 500 and the real error goes to the server log only.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException as StarletteHTTPException

from automl_ops.dataset_reports import ReportError
from automl_ops.dataset_store import (
    DatasetExists,
    DatasetNotFound,
    DatasetStoreError,
    InvalidName,
    VersionNotFound,
)
from automl_ops.ingest import IngestError

log = logging.getLogger("automl_ops.api")


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error: ErrorBody


class ApiError(Exception):
    """An error raised on purpose by a route: status, stable code, human message."""

    def __init__(
        self, status: int, code: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details


def envelope(
    status: int, code: str, message: str, details: dict[str, Any] | None = None
) -> JSONResponse:
    body: dict[str, Any] = {"code": code, "message": message}
    if details:
        body["details"] = details
    return JSONResponse({"error": body}, status_code=status)


# IngestError codes with their own status; every other ingest rejection is "unprocessable".
_INGEST_STATUS = {"too_large": 413}

_HTTP_CODES = {
    400: ("bad_request", "The request could not be understood."),
    404: ("not_found", "No such endpoint."),
    405: ("method_not_allowed", "That method is not allowed for this endpoint."),
    413: ("body_too_large", "The request body is too large."),
    415: ("unsupported_media_type", "Unsupported content type."),
}


async def _api_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return envelope(exc.status, exc.code, exc.message, exc.details)


async def _ingest_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, IngestError)
    return envelope(_INGEST_STATUS.get(exc.code, 422), exc.code, exc.message)


async def _store_error(_: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, InvalidName):
        return envelope(422, "invalid_name", str(exc))
    if isinstance(exc, DatasetNotFound):
        return envelope(404, "dataset_not_found", str(exc))
    if isinstance(exc, VersionNotFound):
        return envelope(404, "version_not_found", str(exc))
    if isinstance(exc, DatasetExists):
        return envelope(409, "dataset_exists", str(exc))
    # Anything else (an unreadable record, a changed file) can carry a file path: log it, hide it.
    log.error("dataset store problem: %s", exc)
    return envelope(
        500,
        "store_error",
        "The stored dataset could not be read or failed its integrity check. "
        "Run `scripts/datasets.py verify` on the server.",
    )


async def _report_error(_: Request, exc: Exception) -> JSONResponse:
    return envelope(422, "invalid_task_config", str(exc))


async def _validation_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    problems = [
        {
            "where": ".".join(str(part) for part in error.get("loc", ())),
            "message": str(error.get("msg", "invalid")),
            "type": str(error.get("type", "")),
        }
        for error in exc.errors()  # never `input`: it may hold file or form contents
    ]
    return envelope(422, "validation_error", "The request is not valid.", {"problems": problems})


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code, message = _HTTP_CODES.get(exc.status_code, ("http_error", "The request failed."))
    response = envelope(exc.status_code, code, message)
    if exc.status_code == 405:
        # Starlette lists the methods of one matching route only; list every method of the path.
        methods: set[str] = set()
        for route in getattr(request.app.state, "api_routes", ()):
            if route.path_regex.match(request.scope["path"]):
                methods |= set(route.methods or ())
        if methods:
            response.headers["Allow"] = ", ".join(sorted(methods))
    return response


async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
    log.error("unexpected error", exc_info=exc)
    return envelope(500, "internal_error", "An internal error occurred.")


def install(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(IngestError, _ingest_error)
    app.add_exception_handler(DatasetStoreError, _store_error)
    app.add_exception_handler(ReportError, _report_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unexpected)
