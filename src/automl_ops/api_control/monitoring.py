"""Monitoring endpoints (P4-9): reference samples, drift reports, alerts.

Runs happen here, in the control plane, on demand; the serving app is never involved and never
imports any of this. Nothing here reads a dataset row except when a reference sample is made (the
training rows only; see `automl_ops.reference`), and nothing trains or changes a model.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Path, Query, Response
from starlette.concurrency import run_in_threadpool

from automl_ops.api_control.datasets import Ctx, _heavy
from automl_ops.api_control.errors import ApiError, ErrorEnvelope
from automl_ops.api_control.schemas import (
    AlertEventModel,
    AlertList,
    AlertModel,
    ModelName,
    MonitoringStatus,
    MonitorRequest,
    ReasonRequest,
    ReferenceModel,
    ReportList,
    ReportModel,
    TrafficScope,
)
from automl_ops.deployments import DeploymentNotFound
from automl_ops.monitor import MonitorError, Thresholds
from automl_ops.monitoring import (
    AlertNotFound,
    AlertRow,
    InvalidAlertMove,
    InvalidAlertReason,
    MonitoringBusy,
    MonitoringError,
    MonitoringRepository,
    MonitoringUnavailable,
    NoReference,
    ReportNotFound,
    ReportRow,
)
from automl_ops.reference import (
    ReferenceNotFound,
    ReferenceRow,
    ReferenceSampleError,
    ReferenceService,
    ReferenceStoreFailed,
    ReferenceUnavailable,
    SplitMismatch,
)
from automl_ops.registry import ModelNotFound, PackageUnverified, VersionNotFound

_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorEnvelope, "description": text}
    for code, text in {
        400: "Bad request, or a Host the API does not accept",
        403: "Origin not allowed",
        404: "No such deployment, model, version, reference, report or alert",
        409: "The request conflicts with the current state",
        413: "Request body too large",
        415: "Unsupported content type",
        422: "Invalid input",
        500: "Internal error",
        503: "The monitoring tables or the database are not available",
    }.items()
}
router = APIRouter(responses=_ERRORS, tags=["monitoring"])

BigId = Annotated[int, Path(ge=1, le=9_223_372_036_854_775_807)]
Name = Annotated[ModelName, Path(description="Model name (a-z, 0-9, '_' and '-').")]
Number = Annotated[int, Path(ge=1, le=9_999_999, description="Version number.")]

_STATUS: dict[type[Exception], int] = {
    DeploymentNotFound: 404,
    ModelNotFound: 404,
    VersionNotFound: 404,
    ReferenceNotFound: 404,
    ReportNotFound: 404,
    AlertNotFound: 404,
    InvalidAlertReason: 422,
    NoReference: 409,
    MonitoringBusy: 409,
    InvalidAlertMove: 409,
    ReferenceUnavailable: 409,
    SplitMismatch: 409,
    PackageUnverified: 409,
    MonitoringUnavailable: 503,
    ReferenceStoreFailed: 503,
}


def _api_error(error: Exception) -> ApiError:
    code = getattr(error, "code", "error")
    details = getattr(error, "details", None) or None
    return ApiError(_STATUS.get(type(error), 409), code, str(error), details)


def _iso(value: Any) -> str | None:
    return None if value is None else str(value.isoformat())


def _monitoring(ctx: Any) -> MonitoringRepository:
    if ctx.monitoring is None:
        raise ApiError(503, "monitoring_unavailable", "Monitoring is not available on this server.")
    repo: MonitoringRepository = ctx.monitoring
    return repo


def _references(ctx: Any) -> ReferenceService:
    if ctx.references is None:
        raise ApiError(503, "monitoring_unavailable", "Monitoring is not available on this server.")
    service: ReferenceService = ctx.references
    return service


def reference_model(row: ReferenceRow, created: bool | None = None) -> ReferenceModel:
    return ReferenceModel(**{**row.__dict__, "created_at": _iso(row.created_at)}, created=created)


def report_model(r: ReportRow) -> ReportModel:
    return ReportModel(
        id=r.id,
        deployment_id=r.deployment_id,
        reference_id=r.reference_id,
        scope=r.scope,  # type: ignore[arg-type]
        status=r.status,  # type: ignore[arg-type]
        window_rows=r.window_rows,
        window_start=_iso(r.window_start),
        window_end=_iso(r.window_end),
        thresholds=r.thresholds,
        summary=r.summary,
        error=r.error,
        started_at=str(_iso(r.started_at)),
        finished_at=str(_iso(r.finished_at)),
    )


def alert_model(a: AlertRow, *, events: bool = False) -> AlertModel:
    return AlertModel(
        id=a.id,
        deployment_id=a.deployment_id,
        rule=a.rule,  # type: ignore[arg-type]
        scope=a.scope,  # type: ignore[arg-type]
        severity=a.severity,  # type: ignore[arg-type]
        status=a.status,  # type: ignore[arg-type]
        details=a.details,
        first_report_id=a.first_report_id,
        last_report_id=a.last_report_id,
        occurrences=a.occurrences,
        created_at=str(_iso(a.created_at)),
        updated_at=str(_iso(a.updated_at)),
        resolved_at=_iso(a.resolved_at),
        events=[
            AlertEventModel(
                id=e.id,
                action=e.action,  # type: ignore[arg-type]
                report_id=e.report_id,
                reason=e.reason,
                created_at=str(_iso(e.created_at)),
            )
            for e in a.events
        ]
        if events
        else None,
    )


# -- reference samples --------------------------------------------------------------------


@router.post(
    "/models/{name}/versions/{number}/reference",
    response_model=ReferenceModel,
    status_code=201,
    summary="Create the drift reference sample of a model version",
    description=(
        "Draws a deterministic sample (at most 1,000 rows) of the version's **training** rows, "
        "stores it with the model's predictions on it, and records its source (dataset version "
        "and SHA-256, training-row fingerprint, package). The held-out test rows are never "
        "selected. A version has at most one reference sample and it is never replaced: calling "
        "this again returns the stored one (200, `created: false`)."
    ),
)
async def create_reference(
    name: Name, number: Number, ctx: Ctx, response: Response
) -> ReferenceModel:
    service = _references(ctx)
    try:
        row, created = await _heavy(ctx, lambda: service.create(name, number))
    except (ReferenceSampleError, ModelNotFound, VersionNotFound, PackageUnverified) as error:
        raise _api_error(error) from None
    response.status_code = 201 if created else 200
    return reference_model(row, created)


@router.get(
    "/models/{name}/versions/{number}/reference",
    response_model=ReferenceModel,
    summary="The reference sample's provenance (not its rows)",
)
def get_reference(name: Name, number: Number, ctx: Ctx) -> ReferenceModel:
    try:
        return reference_model(_references(ctx).get(name, number))
    except (ReferenceSampleError, ModelNotFound, VersionNotFound) as error:
        raise _api_error(error) from None


# -- monitoring runs and reports ----------------------------------------------------------


@router.post(
    "/deployments/{deployment_id}/monitor",
    response_model=ReportModel,
    status_code=201,
    summary="Run drift monitoring for a deployment now",
    description=(
        "Compares the most recent logged predictions of this deployment (one traffic scope: real "
        "by default, or simulated; never mixed) with its reference sample: per-feature drift, the "
        "dataset-drift share, prediction drift, and live performance against the stored offline "
        "value. Stores an immutable report and updates alerts. Below `min_rows` rows the report "
        "says `insufficient_data` and reaches no verdict; a run that breaks is stored as `failed`, "
        "never as a success; neither opens or resolves an alert. Drift means inputs or predictions "
        "differ from the training sample, not that the model is wrong. Needs a reference sample "
        "(409 `no_reference`). One run per deployment at a time (409 `monitoring_in_progress`)."
    ),
)
async def run_monitoring(deployment_id: BigId, body: MonitorRequest, ctx: Ctx) -> ReportModel:
    repo = _monitoring(ctx)
    overrides = {k: v for k, v in body.thresholds.model_dump().items() if v is not None}
    try:
        thresholds = Thresholds(**overrides)
        report = await run_in_threadpool(repo.run, deployment_id, body.scope, thresholds)
    except MonitorError as error:
        raise ApiError(422, "validation_error", str(error)) from None
    except (MonitoringError, DeploymentNotFound) as error:
        raise _api_error(error) from None
    return report_model(report)


@router.get(
    "/deployments/{deployment_id}/monitoring",
    response_model=MonitoringStatus,
    summary="Monitoring status: reference, latest reports, unresolved alerts",
)
async def monitoring_status(deployment_id: BigId, ctx: Ctx) -> MonitoringStatus:
    try:
        s = await run_in_threadpool(_monitoring(ctx).status, deployment_id)
    except (MonitoringError, DeploymentNotFound) as error:
        raise _api_error(error) from None
    return MonitoringStatus(
        deployment_id=s["deployment_id"],
        reference_id=s["reference_id"],
        latest_real=None if s["latest"]["real"] is None else report_model(s["latest"]["real"]),
        latest_simulated=None
        if s["latest"]["simulated"] is None
        else report_model(s["latest"]["simulated"]),
        unresolved_alerts=s["unresolved_alerts"],
    )


@router.get(
    "/deployments/{deployment_id}/reports",
    response_model=ReportList,
    summary="Monitoring report history, newest first (summaries omitted; read one by id)",
)
async def list_reports(
    deployment_id: BigId,
    ctx: Ctx,
    scope: Annotated[TrafficScope | None, Query()] = None,
    before_id: Annotated[int | None, Query(ge=1, le=9_223_372_036_854_775_807)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
) -> ReportList:
    try:
        rows = await run_in_threadpool(
            _monitoring(ctx).reports, deployment_id, scope, before_id, limit
        )
    except (MonitoringError, DeploymentNotFound) as error:
        raise _api_error(error) from None
    return ReportList(
        deployment_id=deployment_id,
        reports=[report_model(r) for r in rows],
        next_before_id=rows[-1].id if len(rows) == limit else None,
    )


@router.get("/reports/{report_id}", response_model=ReportModel, summary="One monitoring report")
async def get_report(report_id: BigId, ctx: Ctx) -> ReportModel:
    try:
        return report_model(await run_in_threadpool(_monitoring(ctx).get_report, report_id))
    except MonitoringError as error:
        raise _api_error(error) from None


# -- alerts -------------------------------------------------------------------------------


@router.get("/alerts", response_model=AlertList, summary="Alerts, newest first")
async def list_alerts(
    ctx: Ctx,
    status: Annotated[str | None, Query(pattern="^(open|acknowledged|resolved)$")] = None,
    deployment_id: Annotated[int | None, Query(ge=1, le=9_223_372_036_854_775_807)] = None,
    rule: Annotated[
        str | None, Query(pattern="^(drifted_feature_share|prediction_shift|performance_drop)$")
    ] = None,
    scope: Annotated[TrafficScope | None, Query()] = None,
    before_id: Annotated[int | None, Query(ge=1, le=9_223_372_036_854_775_807)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
) -> AlertList:
    try:
        rows = await run_in_threadpool(
            lambda: _monitoring(ctx).alerts(
                status=status,
                deployment_id=deployment_id,
                rule=rule,
                scope=scope,
                before_id=before_id,
                limit=limit,
            )
        )
    except MonitoringError as error:
        raise _api_error(error) from None
    return AlertList(
        alerts=[alert_model(a) for a in rows],
        next_before_id=rows[-1].id if len(rows) == limit else None,
    )


@router.get("/alerts/{alert_id}", response_model=AlertModel, summary="One alert with its history")
async def get_alert(alert_id: BigId, ctx: Ctx) -> AlertModel:
    try:
        return alert_model(
            await run_in_threadpool(_monitoring(ctx).get_alert, alert_id), events=True
        )
    except MonitoringError as error:
        raise _api_error(error) from None


@router.post(
    "/alerts/{alert_id}/acknowledge",
    response_model=AlertModel,
    summary="Acknowledge an open alert",
)
async def acknowledge_alert(alert_id: BigId, body: ReasonRequest, ctx: Ctx) -> AlertModel:
    try:
        row = await run_in_threadpool(_monitoring(ctx).acknowledge, alert_id, body.reason)
    except MonitoringError as error:
        raise _api_error(error) from None
    return alert_model(row, events=True)


@router.post(
    "/alerts/{alert_id}/resolve",
    response_model=AlertModel,
    summary="Resolve an alert by hand",
    description=(
        "A resolved alert never changes. If the condition still holds, the next monitoring run "
        "opens a new alert."
    ),
)
async def resolve_alert(alert_id: BigId, body: ReasonRequest, ctx: Ctx) -> AlertModel:
    try:
        row = await run_in_threadpool(_monitoring(ctx).resolve, alert_id, body.reason)
    except MonitoringError as error:
        raise _api_error(error) from None
    return alert_model(row, events=True)
