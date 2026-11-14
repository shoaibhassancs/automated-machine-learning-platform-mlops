"""The serving app factory and the production wiring (P4-7)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from starlette.concurrency import run_in_threadpool

from automl_ops.api_control import errors
from automl_ops.api_control.errors import ApiError, ErrorEnvelope
from automl_ops.api_control.security import LocalSecurityMiddleware
from automl_ops.api_serving.inputs import InvalidInput
from automl_ops.api_serving.runtime import (
    Health,
    NotReady,
    PredictionFailed,
    ServingRuntime,
)
from automl_ops.api_serving.schemas import (
    DeploymentRef,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    InputColumn,
    ModelInfo,
    PredictionItem,
    PredictRequest,
    PredictResponse,
    label_model,
)
from automl_ops.api_serving.settings import MAX_ROWS, ServingSettings
from automl_ops.db.config import make_engine
from automl_ops.feedback import (
    FeedbackConflict,
    FeedbackError,
    FeedbackUnavailable,
    InvalidTrueValue,
    PredictionNotFound,
    ReferenceMismatch,
)

log = logging.getLogger("automl_ops.serving")
API_VERSION = "0.1.0"

_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorEnvelope, "description": text}
    for code, text in {
        400: "Bad request, or a Host the app does not accept",
        403: "Origin not allowed",
        413: "Request body too large",
        415: "Unsupported content type",
        404: "No such logged prediction",
        409: "The label conflicts with the prediction or with its existing label",
        422: "The input does not match the model's feature schema, or the true value is invalid",
        500: "Internal error",
        503: "No verified model is being served",
    }.items()
}


def _iso(value: Any) -> str | None:
    return None if value is None else str(value.isoformat())


def _health_body(h: Health) -> HealthResponse:
    return HealthResponse(
        status="ok",
        ready=h.ready,
        model_state=h.model_state,
        database=h.database,  # type: ignore[arg-type]
        desired_deployment_id=h.desired_deployment_id,
        loaded_deployment_id=h.loaded_deployment_id,
        error=h.error,
        checked_at=_iso(h.checked_at),
    )


def _not_ready(error: NotReady) -> ApiError:
    return ApiError(503, error.code, error.message)


def create_app(settings: ServingSettings, runtime: ServingRuntime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predict_slots = asyncio.Semaphore(settings.predict_concurrency)
        await asyncio.to_thread(runtime.refresh)  # serve the active deployment from the start
        poll = asyncio.create_task(_poll(runtime, settings.poll_seconds))
        try:
            yield
        finally:
            poll.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poll

    app = FastAPI(
        title="AutoML Ops serving app",
        version=API_VERSION,
        summary="Serves the one deployed model version. Local, single-user: bound to 127.0.0.1.",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.predict_slots = asyncio.Semaphore(settings.predict_concurrency)
    router = _routes(runtime)
    app.state.api_routes = list(router.routes)
    errors.install(app)
    app.add_exception_handler(NotReady, _handler_not_ready)
    app.add_exception_handler(InvalidInput, _handler_invalid)
    app.add_exception_handler(PredictionFailed, _handler_failed)
    app.add_exception_handler(FeedbackError, _handler_feedback)
    app.include_router(router)
    app.add_middleware(LocalSecurityMiddleware, settings=settings.http_rules())
    return app


async def _handler_not_ready(_: Any, exc: Exception) -> Any:
    assert isinstance(exc, NotReady)
    return errors.envelope(503, exc.code, exc.message)


async def _handler_invalid(_: Any, exc: Exception) -> Any:
    assert isinstance(exc, InvalidInput)
    return errors.envelope(
        422,
        "invalid_input",
        "The input does not match the model's feature schema (see GET /model).",
        {"problems": [p.to_dict() for p in exc.problems], "truncated": exc.truncated},
    )


async def _handler_failed(_: Any, exc: Exception) -> Any:
    log.error("prediction failed: %s", exc)
    return errors.envelope(500, "prediction_failed", "The model could not produce a prediction.")


_FEEDBACK_STATUS: dict[type[FeedbackError], int] = {
    PredictionNotFound: 404,
    InvalidTrueValue: 422,
    ReferenceMismatch: 409,
    FeedbackConflict: 409,
    FeedbackUnavailable: 503,
}


async def _handler_feedback(_: Any, exc: Exception) -> Any:
    assert isinstance(exc, FeedbackError)
    return errors.envelope(
        _FEEDBACK_STATUS.get(type(exc), 409), exc.code, str(exc), exc.details or None
    )


async def _poll(runtime: ServingRuntime, interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        with contextlib.suppress(Exception):
            await asyncio.to_thread(runtime.refresh)


def _routes(runtime: ServingRuntime) -> Any:
    router = APIRouter(responses=_ERRORS, tags=["serving"])

    @router.get("/health", response_model=HealthResponse, summary="Liveness and model state")
    async def health() -> HealthResponse:
        return _health_body(runtime.health())

    @router.get(
        "/ready",
        response_model=HealthResponse,
        summary="Readiness: 200 only while a verified model is being served",
    )
    async def ready() -> HealthResponse:
        h = runtime.health()
        if not h.ready:
            raise ApiError(
                503,
                "not_ready",
                h.error or "The model is not ready.",
                {"model_state": h.model_state},
            )
        return _health_body(h)

    @router.post(
        "/reload",
        response_model=HealthResponse,
        summary="Re-read the active deployment now",
        description=(
            "Takes no input: the app re-reads the active deployment from PostgreSQL and loads and "
            "verifies it. It cannot be told what to load. The control plane calls this after every "
            "deployment change; without it the app notices the change within its poll interval."
        ),
    )
    async def reload() -> HealthResponse:
        return _health_body(await run_in_threadpool(runtime.refresh, force=True))

    @router.get("/model", response_model=ModelInfo, summary="The model being served")
    async def model() -> ModelInfo:
        served = runtime.model()  # NotReady -> 503
        d = served.deployment
        return ModelInfo(
            deployment=DeploymentRef(
                deployment_id=d.id,
                model=d.model,
                version=d.version,
                package_id=d.package_id,
                task=d.task,  # type: ignore[arg-type]
            ),
            target=d.target,
            algorithm=d.algorithm,
            classes=served.package.classes,
            input_columns=[InputColumn(name=c.name, kind=c.kind) for c in served.columns],  # type: ignore[arg-type]
            deployed_at=str(_iso(d.deployed_at)),
            loaded_at=str(_iso(served.loaded_at)),
            registry_status=d.registry_status,
            max_rows=MAX_ROWS,
        )

    @router.post(
        "/predict",
        response_model=PredictResponse,
        summary="Predict for one or more rows",
        description=(
            "Rows are validated against the model's feature schema first (`GET /model`): exactly "
            "its input columns, clean numbers in numeric columns. Each row gets its own prediction "
            "id and is logged with its input, output, model version and latency; a logging failure "
            "never fails the prediction (`logged: false`)."
        ),
    )
    async def predict(body: PredictRequest, request: Request) -> PredictResponse:
        slots: asyncio.Semaphore = request.app.state.predict_slots
        async with slots:
            batch = await run_in_threadpool(runtime.predict, body.rows, simulated=body.simulated)
        d = batch.deployment
        return PredictResponse(
            deployment=DeploymentRef(
                deployment_id=d.id,
                model=d.model,
                version=d.version,
                package_id=d.package_id,
                task=d.task,  # type: ignore[arg-type]
            ),
            predictions=[
                PredictionItem(
                    prediction_id=i.prediction_id,
                    prediction=i.prediction,
                    probabilities=i.probabilities,
                )
                for i in batch.items
            ],
            latency_ms=batch.latency_ms,
            logged=batch.logged,
        )

    @router.post(
        "/feedback",
        response_model=FeedbackResponse,
        summary="Attach the true value to a logged prediction",
        description=(
            "Records the ground truth for one `prediction_id`. A label is append-only: the same "
            "label again is a no-op (`created: false`); a different label for an already labelled "
            "prediction is refused (409 `feedback_conflict`) and the first label stands. Works "
            "even if the prediction's deployment has ended or no model is loaded, because it "
            "needs only PostgreSQL. Live performance is read from the control plane "
            "(`GET /deployments/{id}/performance`)."
        ),
    )
    async def feedback(body: FeedbackRequest) -> FeedbackResponse:
        outcome = await run_in_threadpool(
            lambda: runtime.feedback.submit(
                body.prediction_id,
                body.true_value,
                deployment_id=body.deployment_id,
                model_version=body.model_version,
            )
        )
        return FeedbackResponse(label=label_model(outcome.label), created=outcome.created)

    return router


def build_app(settings: ServingSettings | None = None) -> FastAPI:
    """The real app: the application database (short timeouts) and the models folder."""
    settings = settings or ServingSettings.from_environment()
    engine = make_engine(connect_timeout=2, pool_timeout=2.0)
    return create_app(
        settings, ServingRuntime(engine, settings.models_root, retry_seconds=settings.retry_seconds)
    )
