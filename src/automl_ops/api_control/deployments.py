"""Deployment endpoints (P4-7): deploy a registered version, switch, roll back, undeploy.

PostgreSQL decides what is live (`automl_ops.deployments`). After each change the control plane
asks the serving app to reload, best-effort: the change is already committed, so a stopped or slow
serving app never fails the request; it is reported in `serving` and the serving app converges on
its own from PostgreSQL. The model itself is never loaded here beyond the deep package check that
every deployment change makes.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Path, Query
from starlette.concurrency import run_in_threadpool

from automl_ops.api_control.datasets import Ctx, _heavy
from automl_ops.api_control.errors import ApiError, ErrorEnvelope
from automl_ops.api_control.schemas import (
    DeploymentDetail,
    DeploymentList,
    DeploymentModel,
    DeploymentRequest,
    LabelList,
    LivePerformance,
    ReasonRequest,
    ServingReportModel,
)
from automl_ops.api_serving.app import label_model
from automl_ops.deployments import (
    AlreadyDeployed,
    DeploymentActive,
    DeploymentError,
    DeploymentNotFound,
    DeploymentRefused,
    DeploymentRepository,
    DeploymentRow,
    InvalidDeploymentReason,
    NoPreviousDeployment,
    NotActive,
    NotDeployable,
)
from automl_ops.feedback import DEFAULT_MIN_LABELLED, FeedbackRepository, Scope
from automl_ops.registry import ModelNotFound, PackageUnverified, RegistryError, VersionNotFound
from automl_ops.serving_client import ServingReport

_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorEnvelope, "description": text}
    for code, text in {
        400: "Bad request, or a Host the API does not accept",
        403: "Origin not allowed",
        404: "No such deployment, model or version",
        409: "The change conflicts with the deployment or registry state",
        413: "Request body too large",
        415: "Unsupported content type",
        422: "Invalid input",
        500: "Internal error",
        503: "The deployment tables are not available",
    }.items()
}
router = APIRouter(responses=_ERRORS, tags=["deployments"])

DeploymentId = Annotated[int, Path(ge=1, le=9_223_372_036_854_775_807)]

_STATUS: dict[type[Exception], int] = {
    DeploymentNotFound: 404,
    ModelNotFound: 404,
    VersionNotFound: 404,
    InvalidDeploymentReason: 422,
    NotDeployable: 409,
    DeploymentActive: 409,
    NotActive: 409,
    AlreadyDeployed: 409,
    NoPreviousDeployment: 409,
    DeploymentRefused: 409,
    PackageUnverified: 409,
}


def _repo(ctx: Any) -> DeploymentRepository:
    if ctx.deployments is None:
        raise ApiError(
            503, "deployments_unavailable", "Deployments are not available on this server."
        )
    repo: DeploymentRepository = ctx.deployments
    return repo


def _api_error(error: DeploymentError | RegistryError) -> ApiError:
    status = _STATUS.get(type(error), 409)
    return ApiError(status, error.code, str(error), error.details or None)


def _iso(value: Any) -> str:
    return str(value.isoformat())


def model_of(d: DeploymentRow) -> DeploymentModel:
    return DeploymentModel(
        id=d.id,
        model=d.model,
        version=d.version,
        package_id=d.package_id,
        status=d.status,  # type: ignore[arg-type]
        action=d.action,  # type: ignore[arg-type]
        reason=d.reason,
        replaced_id=d.replaced_id,
        deployed_at=_iso(d.deployed_at),
        ended_at=None if d.ended_at is None else _iso(d.ended_at),
        end_reason=d.end_reason,
        registry_status=d.registry_status,  # type: ignore[arg-type]
        task=d.task,
        algorithm=d.algorithm,
    )


def _report(r: ServingReport) -> ServingReportModel:
    return ServingReportModel(
        state=r.state,  # type: ignore[arg-type]
        loaded_deployment_id=r.loaded_deployment_id,
        model_state=r.model_state,
        error=r.error,
    )


async def _serving(ctx: Any, desired: int | None, *, reload: bool) -> ServingReportModel | None:
    if ctx.serving is None:
        return None
    report = await run_in_threadpool(ctx.serving.report, desired, reload=reload)
    return _report(report)


async def _changed(ctx: Any, row: DeploymentRow, *, desired: int | None) -> DeploymentDetail:
    detail = DeploymentDetail(**model_of(row).model_dump())
    detail.serving = await _serving(ctx, desired, reload=True)
    return detail


@router.post(
    "/deployments",
    status_code=201,
    response_model=DeploymentDetail,
    summary="Deploy a registered model version",
    description=(
        "Makes a `staging` or `production` version the live model, after verifying its sealed "
        "package deeply. Only one deployment is active at a time: if one is, this is refused "
        "(409 `deployment_active`); use `POST /deployments/{id}/switch`. Archiving or rolling back "
        "the registry status of a deployed version does not undeploy it."
    ),
)
async def deploy(body: DeploymentRequest, ctx: Ctx) -> DeploymentDetail:
    repo = _repo(ctx)
    try:
        row = await _heavy(ctx, lambda: repo.deploy(body.model, body.version, body.reason))
    except (DeploymentError, RegistryError) as error:
        raise _api_error(error) from None
    return await _changed(ctx, row, desired=row.id)


@router.get(
    "/deployments",
    response_model=DeploymentList,
    summary="Deployment history, newest first",
)
def list_deployments(
    ctx: Ctx,
    status: Annotated[Literal["active", "ended"] | None, Query()] = None,
    before_id: Annotated[int | None, Query(ge=1, le=9_223_372_036_854_775_807)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
) -> DeploymentList:
    rows = _repo(ctx).list(status, before_id, limit)
    return DeploymentList(
        deployments=[model_of(r) for r in rows],
        next_before_id=rows[-1].id if len(rows) == limit else None,
    )


@router.get(
    "/deployments/active",
    response_model=DeploymentDetail,
    summary="The active deployment and the serving app's state",
)
async def active_deployment(ctx: Ctx) -> DeploymentDetail:
    row = await run_in_threadpool(_repo(ctx).active)
    if row is None:
        raise ApiError(404, "no_active_deployment", "No model is deployed.")
    detail = DeploymentDetail(**model_of(row).model_dump())
    detail.serving = await _serving(ctx, row.id, reload=False)
    return detail


@router.post(
    "/deployments/rollback",
    response_model=DeploymentDetail,
    summary="Undo the last switch",
    description=(
        "Makes the version that the active deployment replaced live again, as a new deployment "
        "with action `rollback`. Only a switch can be rolled back: a first deploy has nothing "
        "before it and a rollback is not undone by another rollback (409 "
        "`no_previous_deployment`)."
    ),
)
async def rollback(body: ReasonRequest, ctx: Ctx) -> DeploymentDetail:
    repo = _repo(ctx)
    try:
        row = await _heavy(ctx, lambda: repo.rollback(body.reason))
    except (DeploymentError, RegistryError) as error:
        raise _api_error(error) from None
    return await _changed(ctx, row, desired=row.id)


@router.get(
    "/deployments/{deployment_id}", response_model=DeploymentModel, summary="One deployment"
)
def get_deployment(deployment_id: DeploymentId, ctx: Ctx) -> DeploymentModel:
    try:
        return model_of(_repo(ctx).get(deployment_id))
    except DeploymentError as error:
        raise _api_error(error) from None


@router.post(
    "/deployments/{deployment_id}/switch",
    response_model=DeploymentDetail,
    summary="Switch the live model to another version",
    description=(
        "Ends the active deployment and makes another `staging` or `production` version live in "
        "one transaction. `deployment_id` must be the active deployment."
    ),
)
async def switch(
    deployment_id: DeploymentId, body: DeploymentRequest, ctx: Ctx
) -> DeploymentDetail:
    repo = _repo(ctx)
    try:
        row = await _heavy(
            ctx, lambda: repo.switch(deployment_id, body.model, body.version, body.reason)
        )
    except (DeploymentError, RegistryError) as error:
        raise _api_error(error) from None
    return await _changed(ctx, row, desired=row.id)


@router.post(
    "/deployments/{deployment_id}/undeploy",
    response_model=DeploymentDetail,
    summary="Stop serving: no model is live afterwards",
)
async def undeploy(deployment_id: DeploymentId, body: ReasonRequest, ctx: Ctx) -> DeploymentDetail:
    repo = _repo(ctx)
    try:
        row = await run_in_threadpool(repo.undeploy, deployment_id, body.reason)
    except (DeploymentError, RegistryError) as error:
        raise _api_error(error) from None
    return await _changed(ctx, row, desired=None)


def _feedback(ctx: Any) -> FeedbackRepository:
    if ctx.feedback is None:
        raise ApiError(503, "feedback_unavailable", "Feedback is not available on this server.")
    repo: FeedbackRepository = ctx.feedback
    return repo


@router.get(
    "/deployments/{deployment_id}/performance",
    response_model=LivePerformance,
    summary="Live performance of a deployment, from labelled predictions only",
    description=(
        "Computed on demand with the same metric code as the offline evaluation, but only over "
        "predictions that have a true value (`POST /feedback` on the serving app). Unlabelled "
        "predictions are counted and excluded. With fewer than `min_labelled` labelled "
        "predictions you get a status and the counts, not metrics. `scope` chooses real traffic "
        "(default), simulated traffic or both. The stored offline final-test summary is returned "
        "separately in `offline_reference`; it is a different measurement and is never mixed in. "
        "Nothing here reads a dataset or the held-out test part."
    ),
)
async def live_performance(
    deployment_id: DeploymentId,
    ctx: Ctx,
    scope: Annotated[Scope, Query()] = "real",
    min_labelled: Annotated[int, Query(ge=1, le=100_000)] = DEFAULT_MIN_LABELLED,
) -> LivePerformance:
    repo = _feedback(ctx)
    try:
        result = await run_in_threadpool(repo.performance, deployment_id, scope, min_labelled)
    except DeploymentError as error:
        raise _api_error(error) from None
    return LivePerformance(**result)


@router.get(
    "/deployments/{deployment_id}/feedback",
    response_model=LabelList,
    summary="The labels recorded for a deployment's predictions, newest first",
)
async def deployment_feedback(
    deployment_id: DeploymentId,
    ctx: Ctx,
    before_id: Annotated[int | None, Query(ge=1, le=9_223_372_036_854_775_807)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
) -> LabelList:
    repo = _feedback(ctx)
    try:
        rows = await run_in_threadpool(repo.labels, deployment_id, before_id, limit)
    except DeploymentError as error:
        raise _api_error(error) from None
    return LabelList(
        deployment_id=deployment_id,
        labels=[label_model(r) for r in rows],
        next_before_id=rows[-1].id if len(rows) == limit else None,
    )
