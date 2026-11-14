"""Model-registry endpoints (P4-5): register a sealed package, promote, archive, roll back.

PostgreSQL decides every status and records every change as an event in the same transaction
(`automl_ops.registry`); nothing here talks to MLflow, and nothing reads the held-out test part: the
evaluation shown for a version is the summary stored from the sealed package's own test result.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Path, Query
from starlette.concurrency import run_in_threadpool

from automl_ops.api_control.datasets import Ctx, _heavy
from automl_ops.api_control.errors import ApiError, ErrorEnvelope
from automl_ops.api_control.schemas import (
    MirrorCheck,
    MirrorFinding,
    MirrorModelStatus,
    MirrorOverview,
    MirrorSummary,
    ModelName,
    ModelVersionDetail,
    ModelVersionList,
    ModelVersionSummary,
    PromoteRequest,
    ReasonRequest,
    ReconcileRequest,
    ReconcileResult,
    RegisteredModel,
    RegisteredModelList,
    RegisterRequest,
    RegistryEventList,
    RegistryEventModel,
    RollbackResult,
    SyncResult,
)
from automl_ops.registry import (
    EVENT_PAGE,
    InvalidMove,
    JobNotFoundForRegistry,
    JobNotRegisterable,
    ModelNotFound,
    NoPreviousProduction,
    PackageAlreadyRegistered,
    PackageNotRegisterable,
    PackageRequired,
    PackageUnverified,
    ProvenanceMismatch,
    RegistryError,
    RegistryRepository,
    VersionNotFound,
    VersionRow,
)
from automl_ops.registry_mirror import (
    Finding,
    InvalidReason,
    Mirror,
    MirrorDisabled,
    MirrorModelNotFound,
    MirrorStatus,
)

_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorEnvelope, "description": text}
    for code, text in {
        400: "Bad request, or a Host the API does not accept",
        403: "Origin not allowed",
        404: "No such model, version, job or endpoint",
        409: "The change conflicts with the registry's state or rules",
        413: "Request body too large",
        415: "Unsupported content type",
        422: "Invalid input, or the package cannot be registered",
        500: "Internal error",
        503: "The registry tables are not available",
    }.items()
}
router = APIRouter(responses=_ERRORS, tags=["registry"])

Name = Annotated[ModelName, Path(description="Model name (a-z, 0-9, '_' and '-').")]
Number = Annotated[int, Path(ge=1, le=9_999_999, description="Version number.")]

_STATUS: dict[type[RegistryError], int] = {
    ModelNotFound: 404,
    VersionNotFound: 404,
    JobNotFoundForRegistry: 404,
    PackageRequired: 422,
    ProvenanceMismatch: 422,
    JobNotRegisterable: 422,
    PackageNotRegisterable: 422,
    InvalidMove: 409,
    PackageAlreadyRegistered: 409,
    NoPreviousProduction: 409,
    PackageUnverified: 409,
}


def _repo(ctx: Any) -> RegistryRepository:
    if ctx.registry is None:
        raise ApiError(503, "registry_unavailable", "The registry is not available on this server.")
    repo: RegistryRepository = ctx.registry
    return repo


def _api_error(error: RegistryError) -> ApiError:
    status = _STATUS.get(type(error), 409)
    return ApiError(status, error.code, str(error), error.details or None)


def _iso(value: Any) -> str:
    return str(value.isoformat())


def mirror_summary(s: MirrorStatus) -> MirrorSummary:
    return MirrorSummary(
        state=s.state,
        desired_seq=s.desired_seq,
        synced_seq=s.synced_seq,
        pending_events=s.pending_events,
        attempts=s.attempts,
        last_error=s.last_error,
        last_error_kind=s.last_error_kind,  # type: ignore[arg-type]
        last_success_at=None if s.last_success_at is None else _iso(s.last_success_at),
        next_attempt_at=None if s.next_attempt_at is None else _iso(s.next_attempt_at),
    )


def _finding(f: Finding) -> MirrorFinding:
    return MirrorFinding(**f.to_dict(), blocking=f.blocking)


def _mirror(ctx: Any, name: str) -> Mirror:
    if ctx.mirror is None:
        raise ApiError(409, "mirror_disabled", "The MLflow mirror is not enabled on this server.")
    mirror: Mirror = ctx.mirror
    if not mirror.enabled:
        raise ApiError(
            409, "mirror_disabled", "The MLflow mirror is disabled (AUTOML_REGISTRY_MIRROR=on)."
        )
    return mirror


def _status_of(ctx: Any, name: str) -> MirrorSummary | None:
    """The mirror's standing for a model, or None if there is no mirror object at all."""
    if ctx.mirror is None:
        return None
    try:
        return mirror_summary(ctx.mirror.status(name))
    except MirrorModelNotFound:
        return None


async def _after_change(ctx: Any, name: str) -> MirrorSummary | None:
    """Best-effort inline sync after a successful registry change. Never fails the request: the
    registry change is already committed; the mirror just stays pending if MLflow is unavailable."""
    if ctx.mirror is None:
        return None
    status = await run_in_threadpool(ctx.mirror.sync_soon, name)
    return None if status is None else mirror_summary(status)


def summary(v: VersionRow) -> ModelVersionSummary:
    return ModelVersionSummary(
        model=v.model,
        version=v.version,
        status=v.status,  # type: ignore[arg-type]
        package_id=v.package_id,
        job_id=v.job_id,
        dataset=v.dataset,
        algorithm=v.algorithm,
        task=v.task,
        target=v.target,
        registered_at=_iso(v.registered_at),
        status_changed_at=_iso(v.status_changed_at),
    )


def detail(v: VersionRow) -> ModelVersionDetail:
    return ModelVersionDetail(
        **summary(v).model_dump(),
        selection_sha256=v.selection_sha256,
        evaluation=v.evaluation or {},
    )


@router.post(
    "/models",
    status_code=201,
    response_model=ModelVersionDetail,
    summary="Register a sealed model package as a new version",
    description=(
        "The package is verified deeply (seal hashes, library versions, prediction parity) before "
        "anything is stored, and a package can be registered only once. The new version starts in "
        "`development`. The evaluation shown is the summary stored from the package's own test "
        "result; it carries the standing limitation that repeated use of the test part across "
        "jobs can influence model choices."
    ),
)
async def register_model(body: RegisterRequest, ctx: Ctx) -> ModelVersionDetail:
    repo = _repo(ctx)

    def work() -> VersionRow:
        return repo.register(
            body.name, package_id=body.package_id, job_id=body.job_id, reason=body.reason
        )

    try:
        version = await _heavy(ctx, work)
    except RegistryError as error:
        raise _api_error(error) from None
    result = detail(version)
    result.mirror = await _after_change(ctx, version.model)
    return result


@router.get("/models", response_model=RegisteredModelList, summary="List registered models")
def list_models(ctx: Ctx) -> RegisteredModelList:
    return RegisteredModelList(
        models=[
            RegisteredModel(
                name=m.name,
                created_at=_iso(m.created_at),
                versions=m.versions,
                production_version=m.production_version,
                latest_version=m.latest_version,
            )
            for m in _repo(ctx).list_models()
        ]
    )


@router.get("/models/{name}", response_model=RegisteredModel, summary="One registered model")
def get_model(name: Name, ctx: Ctx) -> RegisteredModel:
    try:
        m = _repo(ctx).get_model(name)
    except RegistryError as error:
        raise _api_error(error) from None
    return RegisteredModel(
        name=m.name,
        created_at=_iso(m.created_at),
        versions=m.versions,
        production_version=m.production_version,
        latest_version=m.latest_version,
        mirror=_status_of(ctx, name),
    )


@router.get(
    "/models/{name}/versions", response_model=ModelVersionList, summary="A model's versions"
)
def list_versions(name: Name, ctx: Ctx) -> ModelVersionList:
    try:
        versions = _repo(ctx).list_versions(name)
    except RegistryError as error:
        raise _api_error(error) from None
    return ModelVersionList(model=name, versions=[summary(v) for v in versions])


@router.get(
    "/models/{name}/versions/{number}",
    response_model=ModelVersionDetail,
    summary="One version, with its stored evaluation summary",
)
def get_version(name: Name, number: Number, ctx: Ctx) -> ModelVersionDetail:
    try:
        result = detail(_repo(ctx).get_version(name, number))
    except RegistryError as error:
        raise _api_error(error) from None
    result.mirror = _status_of(ctx, name)
    return result


@router.post(
    "/models/{name}/versions/{number}/promote",
    response_model=ModelVersionDetail,
    summary="Promote a version to staging or production",
    description=(
        "development -> staging, staging -> production, archived -> staging. Promoting to "
        "production archives the current production version in the same transaction. There is no "
        "direct jump to production; a rollback is the only way an archived version returns to it."
    ),
)
async def promote_version(
    name: Name, number: Number, body: PromoteRequest, ctx: Ctx
) -> ModelVersionDetail:
    repo = _repo(ctx)
    try:
        version = await _heavy(ctx, lambda: repo.promote(name, number, body.to, body.reason))
    except RegistryError as error:
        raise _api_error(error) from None
    result = detail(version)
    result.mirror = await _after_change(ctx, name)
    return result


@router.post(
    "/models/{name}/versions/{number}/archive",
    response_model=ModelVersionDetail,
    summary="Archive a development or staging version",
)
async def archive_version(
    name: Name, number: Number, body: ReasonRequest, ctx: Ctx
) -> ModelVersionDetail:
    repo = _repo(ctx)
    try:
        version = await _heavy(ctx, lambda: repo.archive(name, number, body.reason))
    except RegistryError as error:
        raise _api_error(error) from None
    result = detail(version)
    result.mirror = await _after_change(ctx, name)
    return result


@router.post(
    "/models/{name}/rollback",
    response_model=RollbackResult,
    summary="Archive the production version and restore the one it replaced",
    description=(
        "Both changes and their events commit together. If the previous production version "
        "cannot be established from the history (or no longer verifies), the answer is 409 and "
        "nothing changes."
    ),
)
async def rollback_model(name: Name, body: ReasonRequest, ctx: Ctx) -> RollbackResult:
    repo = _repo(ctx)
    try:
        outcome = await _heavy(ctx, lambda: repo.rollback(name, body.reason))
    except RegistryError as error:
        raise _api_error(error) from None
    return RollbackResult(
        restored=detail(outcome.restored),
        archived=detail(outcome.archived),
        mirror=await _after_change(ctx, name),
    )


@router.get(
    "/models/{name}/events", response_model=RegistryEventList, summary="A model's event history"
)
def model_events(
    name: Name,
    ctx: Ctx,
    after_seq: Annotated[int, Query(ge=0, le=2_147_483_647)] = 0,
) -> RegistryEventList:
    try:
        events = _repo(ctx).events(name, after_seq, EVENT_PAGE)
    except RegistryError as error:
        raise _api_error(error) from None
    return RegistryEventList(
        model=name,
        events=[
            RegistryEventModel(
                seq=e.seq,
                version=e.version,
                action=e.action,  # type: ignore[arg-type]
                from_status=e.from_status,  # type: ignore[arg-type]
                to_status=e.to_status,  # type: ignore[arg-type]
                reason=e.reason,
                created_at=_iso(e.created_at),
            )
            for e in events
        ],
        last_seq=events[-1].seq if events else after_seq,
    )


# -- the MLflow mirror (P4-6) --------------------------------------------------------------------


@router.get(
    "/registry/sync",
    response_model=MirrorOverview,
    summary="The MLflow mirror's standing for every model",
)
def mirror_overview(ctx: Ctx) -> MirrorOverview:
    mirror: Mirror | None = ctx.mirror
    if mirror is None:
        return MirrorOverview(enabled=False, problem="the mirror is not configured", models=[])
    return MirrorOverview(
        enabled=mirror.enabled,
        problem=mirror.problem(),
        models=[
            MirrorModelStatus(**mirror_summary(s).model_dump(), model=s.model)
            for s in mirror.overview()
        ],
    )


@router.get(
    "/models/{name}/sync",
    response_model=MirrorSummary,
    summary="Where this model's MLflow mirror stands",
)
def mirror_status(name: Name, ctx: Ctx) -> MirrorSummary:
    try:
        _repo(ctx).get_model(name)
    except RegistryError as error:
        raise _api_error(error) from None
    standing = _status_of(ctx, name)
    if standing is None:
        raise ApiError(409, "mirror_disabled", "The MLflow mirror is not enabled on this server.")
    return standing


@router.post(
    "/models/{name}/sync",
    response_model=SyncResult,
    summary="Sync the MLflow mirror now",
    description=(
        "Makes MLflow agree with the registry if it does not already. A failure is a normal "
        "answer (`result: failed`, the registry is unaffected). Drift found in MLflow stops the "
        "sync (`result: drift`) until an explicit reconcile."
    ),
)
async def mirror_sync(name: Name, ctx: Ctx) -> SyncResult:
    try:
        _repo(ctx).get_model(name)
    except RegistryError as error:
        raise _api_error(error) from None
    mirror = _mirror(ctx, name)
    try:
        outcome = await _heavy(ctx, lambda: mirror.sync(name))
    except MirrorDisabled as error:
        raise ApiError(409, "mirror_disabled", str(error)) from None
    except MirrorModelNotFound as error:
        raise ApiError(404, "model_not_found", str(error)) from None
    return SyncResult(
        result=outcome.result,  # type: ignore[arg-type]
        mirror=mirror_summary(mirror.status(name)),
        actions=outcome.actions,
        findings=[_finding(f) for f in outcome.findings],
        error=outcome.error,
    )


@router.get(
    "/models/{name}/mirror/check",
    response_model=MirrorCheck,
    summary="Compare the registry with MLflow (read-only)",
)
async def mirror_check(name: Name, ctx: Ctx) -> MirrorCheck:
    try:
        _repo(ctx).get_model(name)
    except RegistryError as error:
        raise _api_error(error) from None
    mirror = _mirror(ctx, name)
    try:
        report = await _heavy(ctx, lambda: mirror.check(name))
    except MirrorDisabled as error:
        raise ApiError(409, "mirror_disabled", str(error)) from None
    except MirrorModelNotFound as error:
        raise ApiError(404, "model_not_found", str(error)) from None
    except Exception as error:
        raise ApiError(
            503, "mirror_unavailable", f"MLflow could not be read ({type(error).__name__})"
        ) from None
    return MirrorCheck(
        model=name,
        mirror=mirror_summary(mirror.status(name)),
        in_sync=not any(f.stops_sync or f.category == "pending" for f in report.findings),
        findings=[_finding(f) for f in report.findings],
    )


@router.post(
    "/models/{name}/reconcile",
    response_model=ReconcileResult,
    summary="Reset the platform-owned parts of MLflow to the registry (explicit repair)",
    description=(
        "Needs a reason. Repairs only platform-owned tags, versions and the `production` alias; "
        "foreign models, other aliases, extra or duplicate versions are reported, never changed "
        "(and a blocking conflict means nothing is changed). Nothing in MLflow is ever deleted. "
        "What was found, what was intended and what happened is written to the audit log."
    ),
)
async def mirror_reconcile(name: Name, body: ReconcileRequest, ctx: Ctx) -> ReconcileResult:
    try:
        _repo(ctx).get_model(name)
    except RegistryError as error:
        raise _api_error(error) from None
    mirror = _mirror(ctx, name)
    try:
        outcome = await _heavy(ctx, lambda: mirror.reconcile(name, body.reason))
    except MirrorDisabled as error:
        raise ApiError(409, "mirror_disabled", str(error)) from None
    except MirrorModelNotFound as error:
        raise ApiError(404, "model_not_found", str(error)) from None
    except InvalidReason as error:
        raise ApiError(422, "validation_error", str(error)) from None
    if outcome.outcome == "busy":
        raise ApiError(409, "mirror_busy", "Another sync of this model is running; try again.")
    return ReconcileResult(
        outcome=outcome.outcome,
        mirror=mirror_summary(mirror.status(name)),
        detected=[_finding(f) for f in outcome.detected],
        intended=outcome.intended,
        actions=outcome.actions,
        result=[_finding(f) for f in outcome.result],
        error=outcome.error,
    )
