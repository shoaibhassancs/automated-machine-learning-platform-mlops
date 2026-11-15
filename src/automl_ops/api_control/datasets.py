"""Dataset endpoints (BUILD_PLAN Step 18): thin routes over the existing dataset modules.

Datasets are addressed by name and version number, like the file store (`abalone@v1` is
`/datasets/abalone/versions/1`); database ids never appear. The file store is the source of truth
for every read; the PostgreSQL catalog is only written after an upload. Heavy work (anything that
reads a whole CSV) runs in a worker thread behind a small semaphore, so a few clients loading
panels at once cannot multiply the pandas memory use.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, Path, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from automl_ops.api_control.errors import ErrorEnvelope
from automl_ops.api_control.schemas import (
    DatabaseHealth,
    DatasetList,
    DatasetModel,
    ExploreReport,
    Health,
    IsolationReport,
    ProfileReport,
    TaskCheckRequest,
    TaskReportModel,
    UploadInfo,
    UploadResult,
    ValidationReportModel,
    VersionModel,
)
from automl_ops.api_control.settings import ApiSettings
from automl_ops.dataset_reports import (
    build_task_config,
    explore_payload,
    profile_payload,
    task_payload,
    validation_payload,
    version_payload,
)
from automl_ops.dataset_store import validate_name
from automl_ops.db.catalog import CatalogedDatasetStore
from automl_ops.db.migrate import DbStatus
from automl_ops.deployments import DeploymentRepository
from automl_ops.feedback import FeedbackRepository
from automl_ops.ingest import IngestError
from automl_ops.isolation import NOT_ASSESSED, Isolation
from automl_ops.jobs import JobRepository
from automl_ops.monitoring import MonitoringRepository
from automl_ops.profile import profile_frame as _profile_frame
from automl_ops.profile import profile_version, read_version_frame
from automl_ops.reference import ReferenceService
from automl_ops.registry import RegistryRepository
from automl_ops.registry_mirror import Mirror
from automl_ops.serving_client import ServingClient
from automl_ops.task import TaskType

log = logging.getLogger("automl_ops.api")


@dataclass
class Context:
    """What the routes share; created once by `create_app`."""

    settings: ApiSettings
    store: CatalogedDatasetStore
    db_status: Callable[[], DbStatus]
    heavy: Any  # asyncio.Semaphore, created inside the running loop
    upload_lock: threading.Lock
    jobs: JobRepository | None = None  # P4-4; None: the job endpoints answer 503
    registry: RegistryRepository | None = None  # P4-5; None: the registry endpoints answer 503
    mirror: Mirror | None = None  # P4-6; None: the MLflow mirror is off
    deployments: DeploymentRepository | None = (
        None  # P4-7; None: the deployment endpoints answer 503
    )
    feedback: FeedbackRepository | None = None  # P4-8; None: the performance endpoints answer 503
    monitoring: MonitoringRepository | None = None  # P4-9; None: monitoring endpoints answer 503
    references: ReferenceService | None = None  # P4-9
    serving: ServingClient | None = None  # P4-7; None: no serving app to notify
    worker_state: Callable[[], str] = lambda: "unknown"  # "running" | "stopped" | "unknown"
    tracking_problem: Callable[[], str | None] = lambda: None  # why track=true cannot be honoured
    isolation: Callable[[], Isolation] = lambda: NOT_ASSESSED  # throwaway-stack report (E2E safety)


def context(request: Request) -> Context:
    ctx: Context = request.app.state.ctx
    return ctx


Ctx = Annotated[Context, Depends(context)]
NameParam = Annotated[str, Path(description="Dataset name (a-z, 0-9, '_' and '-').")]
NumberParam = Annotated[int, Path(ge=1, le=9_999_999, description="Version number.")]
TargetQuery = Annotated[str | None, Query(min_length=1, max_length=1000)]

_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorEnvelope, "description": text}
    for code, text in {
        400: "Bad request, or a Host the API does not accept",
        403: "Origin not allowed",
        404: "No such dataset, version or endpoint",
        413: "Request body or file too large",
        415: "Unsupported content type",
        422: "Invalid input, or the CSV was rejected",
        500: "Internal or store error",
    }.items()
}
router = APIRouter(responses=_ERRORS)


async def _heavy[T](ctx: Context, work: Callable[[], T]) -> T:
    async with ctx.heavy:
        return await run_in_threadpool(work)


def _ref(name: str, number: int) -> str:
    validate_name(name)
    return f"{name}@v{number}"


@router.get(
    "/health",
    response_model=Health,
    tags=["health"],
    summary="Health",
    responses={
        503: {"model": Health, "description": "The database is unreachable or not at the head"}
    },
)
def health(response: Response, ctx: Ctx) -> Health:
    try:
        state = ctx.db_status()
        database = DatabaseHealth(
            reachable=True, at_head=state.up_to_date, revision=state.current, head=state.head
        )
    except Exception as error:  # never put the error text in the response (it can name a host)
        log.warning("health check: database not reachable (%s)", type(error).__name__)
        database = DatabaseHealth(reachable=False, at_head=None, revision=None, head=None)
    healthy = database.reachable and bool(database.at_head)
    if not healthy:
        response.status_code = 503
    try:
        worker = ctx.worker_state()
    except Exception:
        worker = "unknown"
    try:
        isolation = IsolationReport(**ctx.isolation().to_dict())
    except Exception:  # an unreadable setting must not break the health check
        isolation = IsolationReport(**NOT_ASSESSED.to_dict())
    return Health(
        status="ok" if healthy else "degraded",
        database=database,
        worker=worker,  # type: ignore[arg-type]
        isolation=isolation,
    )


@router.post(
    "/datasets",
    status_code=201,
    response_model=UploadResult,
    tags=["datasets"],
    summary="Upload a CSV as a new dataset or a new version",
    responses={200: {"model": UploadResult, "description": "Identical content already stored"}},
)
async def upload_dataset(
    response: Response,
    ctx: Ctx,
    name: Annotated[str, Form(max_length=200)],
    file: Annotated[UploadFile, File()],
    add_version: Annotated[bool, Form()] = False,
) -> UploadResult:
    limit = ctx.settings.limits.max_bytes
    try:
        data = await file.read(limit + 1)
    finally:
        await file.close()
    if len(data) > limit:
        raise IngestError("too_large", f"file is larger than the limit of {limit:,} bytes")
    filename = file.filename or "upload.csv"

    def work() -> UploadResult:
        with ctx.upload_lock:  # one upload at a time: the file store is single-writer
            before = len(ctx.store.unsynced)
            result = ctx.store.ingest(name, data, source_filename=filename, add_version=add_version)
            catalogued = len(ctx.store.unsynced) == before
        return UploadResult(
            created=result.created,
            catalogued=catalogued,
            version=VersionModel(**version_payload(result.version)),
            upload=UploadInfo(
                sha256=result.upload_sha256,
                encoding=result.upload_encoding,
                delimiter=result.upload_delimiter,
                reencoded=result.upload_reencoded,
            ),
        )

    outcome = await _heavy(ctx, work)
    response.status_code = 201 if outcome.created else 200
    return outcome


@router.get("/datasets", response_model=DatasetList, tags=["datasets"], summary="List datasets")
def list_datasets(ctx: Ctx) -> DatasetList:
    return DatasetList(
        datasets=[
            DatasetModel(
                name=name,
                versions=[
                    VersionModel(**version_payload(v)) for v in ctx.store.list_versions(name)
                ],
            )
            for name in ctx.store.list_datasets()
        ]
    )


@router.get(
    "/datasets/{name}", response_model=DatasetModel, tags=["datasets"], summary="One dataset"
)
def get_dataset(name: NameParam, ctx: Ctx) -> DatasetModel:
    validate_name(name)
    return DatasetModel(
        name=name,
        versions=[VersionModel(**version_payload(v)) for v in ctx.store.list_versions(name)],
    )


@router.get(
    "/datasets/{name}/versions/{number}",
    response_model=VersionModel,
    tags=["datasets"],
    summary="One version's record",
)
def get_version(name: NameParam, number: NumberParam, ctx: Ctx) -> VersionModel:
    return VersionModel(**version_payload(ctx.store.get_version(_ref(name, number))))


@router.get(
    "/datasets/{name}/versions/{number}/profile",
    response_model=ProfileReport,
    tags=["reports"],
    summary="Column profile of a version",
)
async def get_profile(name: NameParam, number: NumberParam, ctx: Ctx) -> ProfileReport:
    ref = _ref(name, number)
    payload = await _heavy(ctx, lambda: profile_payload(profile_version(ctx.store, ref)))
    return ProfileReport(**payload)


@router.get(
    "/datasets/{name}/versions/{number}/validation",
    response_model=ValidationReportModel,
    tags=["reports"],
    summary="Rule-based validation findings",
)
async def get_validation(
    name: NameParam, number: NumberParam, ctx: Ctx, target: TargetQuery = None
) -> ValidationReportModel:
    ref = _ref(name, number)

    def work() -> dict[str, Any]:
        vp = profile_version(ctx.store, ref)
        return validation_payload(vp.version, vp.profile, target)

    return ValidationReportModel(**await _heavy(ctx, work))


@router.get(
    "/datasets/{name}/versions/{number}/preview",
    response_model=ExploreReport,
    tags=["reports"],
    summary="Sample rows, histograms, correlation and class balance",
)
async def get_preview(
    name: NameParam,
    number: NumberParam,
    ctx: Ctx,
    target: TargetQuery = None,
    task: TaskType | None = None,
) -> ExploreReport:
    ref = _ref(name, number)

    def work() -> dict[str, Any]:
        version, frame = read_version_frame(ctx.store, ref)
        profile = _profile_frame(frame)
        config = build_task_config(profile, target, task, ()) if target else None
        return explore_payload(version, frame, profile, config)

    return ExploreReport(**await _heavy(ctx, work))


@router.post(
    "/datasets/{name}/versions/{number}/task-check",
    response_model=TaskReportModel,
    tags=["reports"],
    summary="Check a task configuration (stateless; nothing is stored)",
)
async def check_task(
    name: NameParam, number: NumberParam, body: TaskCheckRequest, ctx: Ctx
) -> TaskReportModel:
    ref = _ref(name, number)

    def work() -> dict[str, Any]:
        version, frame = read_version_frame(ctx.store, ref)
        profile = _profile_frame(frame)
        config = build_task_config(profile, body.target, body.task, tuple(body.excluded))
        return task_payload(version, frame, profile, config)

    return TaskReportModel(**await _heavy(ctx, work))
