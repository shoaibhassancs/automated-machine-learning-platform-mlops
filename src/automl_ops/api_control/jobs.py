"""Training-job endpoints (BUILD_PLAN Step 19): submit, observe and cancel jobs.

The API never trains anything. `POST /jobs` checks the request against the dataset (the same task
checks as `POST .../task-check`), stores a queued job row and returns; a separate worker process
(`scripts/worker.py`) claims jobs one at a time. Everything that changes a job is one conditional
SQL statement in `automl_ops.jobs`, so the API and the worker cannot both win a race.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Path, Query, Response

from automl_ops.algorithms import AlgorithmError, algorithms_for, get_algorithm
from automl_ops.api_control.datasets import Ctx, _heavy, _ref
from automl_ops.api_control.errors import ApiError, ErrorEnvelope
from automl_ops.api_control.schemas import (
    AlgorithmList,
    AlgorithmModel,
    CancelResult,
    JobConfigModel,
    JobDetail,
    JobEventModel,
    JobFailure,
    JobList,
    JobProgress,
    JobRequest,
    JobResultModel,
    JobSummary,
    ParamModel,
)
from automl_ops.dataset_reports import build_task_config, class_counts_for
from automl_ops.dataset_store import validate_name
from automl_ops.jobs import (
    EVENT_PAGE,
    DatasetMismatch,
    DatasetNotCatalogued,
    Job,
    JobNotFound,
    JobRepository,
    QueueFull,
)
from automl_ops.profile import profile_frame, read_version_frame
from automl_ops.runner import RunSpec, resolve_requests, spec_to_config
from automl_ops.selection import ALLOWED_METRICS
from automl_ops.task import TASK_TYPES, TaskType, check_task_config
from automl_ops.train import AlgorithmRequest, TrainingError, TrainSettings

_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorEnvelope, "description": text}
    for code, text in {
        400: "Bad request, or a Host the API does not accept",
        403: "Origin not allowed",
        404: "No such job, dataset, version or endpoint",
        409: "The request conflicts with the current state (queue full, job too far along)",
        413: "Request body too large",
        415: "Unsupported content type",
        422: "Invalid input, or the task cannot be trained",
        500: "Internal error",
        503: "The job tables are not available",
    }.items()
}
router = APIRouter(responses=_ERRORS, tags=["jobs"])

JobId = Annotated[int, Path(ge=1, le=2_147_483_647, description="Job id.")]


def _repo(ctx: Any) -> JobRepository:
    if ctx.jobs is None:
        raise ApiError(503, "jobs_unavailable", "Training jobs are not available on this server.")
    repo: JobRepository = ctx.jobs
    return repo


def _iso(value: Any) -> str | None:
    return None if value is None else str(value.isoformat())


def summary(job: Job) -> JobSummary:
    return JobSummary(
        id=job.id,
        kind=job.kind,  # type: ignore[arg-type]
        status=job.status,  # type: ignore[arg-type]
        phase=job.phase,
        dataset=job.ref,
        dataset_sha256=job.dataset_sha256,
        track=job.track,
        tracking_state=job.tracking_state,  # type: ignore[arg-type]
        cancel_requested=job.cancel_requested,
        stale=job.stale,
        progress=None if job.progress is None else JobProgress(**job.progress),
        error=(
            None
            if job.error_code is None
            else JobFailure(code=job.error_code, message=job.error_message or "")
        ),
        created_at=str(_iso(job.created_at)),
        started_at=_iso(job.started_at),
        finished_at=_iso(job.finished_at),
    )


def detail(job: Job, repo: JobRepository, after_seq: int) -> JobDetail:
    events = repo.events(job.id, after_seq, EVENT_PAGE)
    return JobDetail(
        **summary(job).model_dump(),
        config=JobConfigModel(**job.config),
        confirm_final_test=job.confirm_final_test,
        package_id=job.package_id,
        tracking_job_key=job.tracking_job_key,
        tracking_error=job.tracking_error,
        result=None if job.result is None else JobResultModel(**job.result),
        events=[
            JobEventModel(
                seq=e.seq, kind=e.kind, payload=e.payload, created_at=str(_iso(e.created_at))
            )
            for e in events
        ],
        last_seq=events[-1].seq if events else after_seq,
    )


@router.get("/algorithms", response_model=AlgorithmList, summary="The algorithms a job can use")
def list_algorithms(task: TaskType | None = None) -> AlgorithmList:
    tasks = (task,) if task else TASK_TYPES
    return AlgorithmList(
        algorithms=[
            AlgorithmModel(
                key=a.key,
                label=a.label,
                task=a.task,
                scale_numeric=a.scale_numeric,
                params=[ParamModel(**p.to_dict()) for p in a.params],
            )
            for t in tasks
            for a in algorithms_for(t)
        ]
    )


@router.post(
    "/jobs",
    status_code=202,
    response_model=JobDetail,
    summary="Queue a training job",
    description=(
        "Checks the request against the dataset and queues the job; a worker runs it later. "
        "`kind: final` reads the held-out test part once, for the selected model, and needs "
        "`confirm_final_test: true`. Tracking to MLflow is off unless `track: true`."
    ),
)
async def submit_job(body: JobRequest, ctx: Ctx) -> JobDetail:
    repo = _repo(ctx)
    if body.kind == "final" and not body.confirm_final_test:
        raise ApiError(
            422,
            "confirmation_required",
            "A final job reads the held-out test part once. Set confirm_final_test to true.",
        )
    if body.track:
        problem = ctx.tracking_problem()
        if problem:
            raise ApiError(422, "tracking_not_available", problem)
    ref = _ref(body.dataset, body.version)

    def work() -> Job:
        version, frame = read_version_frame(ctx.store, ref)  # hash-checked read
        profile = profile_frame(frame)
        config = build_task_config(profile, body.target, body.task, tuple(body.excluded))
        report = check_task_config(profile, class_counts_for(frame, config), config)
        if not report.ok:
            errors = [f for f in report.to_dict()["findings"] if f["severity"] == "error"]
            raise ApiError(
                422,
                "task_not_trainable",
                "The task configuration has errors; see details.",
                {"findings": errors},
            )
        requests = _requests(config.task, body)
        settings = TrainSettings(test_fraction=body.test_fraction, seed=body.seed, folds=body.folds)
        try:
            settings.validate()
        except TrainingError as error:
            raise ApiError(422, "invalid_settings", str(error)) from None
        metric = body.selection_metric
        if metric is not None and metric not in ALLOWED_METRICS[config.task]:
            raise ApiError(
                422,
                "invalid_metric",
                f"selection_metric must be one of {list(ALLOWED_METRICS[config.task])}",
            )
        spec = RunSpec(body.kind, config, tuple(requests), settings, metric)
        try:
            return repo.enqueue(
                kind=body.kind,
                dataset_name=body.dataset,
                number=body.version,
                sha256=version.sha256,
                config=spec_to_config(spec),
                track=body.track,
                confirm_final_test=body.confirm_final_test,
            )
        except DatasetNotCatalogued:
            raise ApiError(
                409,
                "dataset_not_catalogued",
                "This version is not in the catalog yet; run `scripts/db.py sync`.",
            ) from None
        except DatasetMismatch:
            raise ApiError(
                409,
                "catalog_mismatch",
                "The catalog and the dataset store disagree; run `db sync`.",
            ) from None
        except QueueFull as error:
            raise ApiError(409, "queue_full", str(error)) from None

    job = await _heavy(ctx, work)
    return detail(job, repo, 0)


def _requests(task: TaskType, body: JobRequest) -> list[AlgorithmRequest]:
    if body.algorithms is None:
        return resolve_requests(task, None)
    keys = [a.key for a in body.algorithms]
    if len(set(keys)) != len(keys):
        raise ApiError(422, "duplicate_algorithm", "An algorithm was listed more than once.")
    requests = [AlgorithmRequest(a.key, a.params) for a in body.algorithms]
    if not requests:
        raise ApiError(422, "invalid_algorithm", "List at least one algorithm, or omit the field.")
    for request in requests:
        try:
            get_algorithm(task, request.key).validate_params(request.params)
        except AlgorithmError as error:
            raise ApiError(422, "invalid_algorithm", str(error)) from None
    return requests


@router.get("/jobs", response_model=JobList, summary="List jobs, newest first")
def list_jobs(
    ctx: Ctx,
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"] | None = None,
    kind: Literal["train", "final"] | None = None,
    dataset: Annotated[str | None, Query(max_length=64)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    before_id: Annotated[int | None, Query(ge=1, le=2_147_483_647)] = None,
) -> JobList:
    repo = _repo(ctx)
    if dataset is not None:
        validate_name(dataset)
    jobs = repo.list_jobs(
        status=status, kind=kind, dataset=dataset, limit=limit, before_id=before_id
    )
    return JobList(
        jobs=[summary(j) for j in jobs],
        next_before_id=jobs[-1].id if len(jobs) == limit else None,
    )


@router.get("/jobs/{job_id}", response_model=JobDetail, summary="One job, with progress events")
def get_job(
    job_id: JobId,
    ctx: Ctx,
    after_seq: Annotated[int, Query(ge=0, le=2_147_483_647)] = 0,
) -> JobDetail:
    repo = _repo(ctx)
    try:
        return detail(repo.get(job_id), repo, after_seq)
    except JobNotFound:
        raise ApiError(404, "job_not_found", f"No job {job_id}.") from None


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=CancelResult,
    summary="Cancel a job",
    responses={202: {"model": CancelResult, "description": "The worker will stop it shortly"}},
)
def cancel_job(job_id: JobId, ctx: Ctx, response: Response) -> CancelResult:
    repo = _repo(ctx)
    try:
        outcome = repo.request_cancel(job_id)
    except JobNotFound:
        raise ApiError(404, "job_not_found", f"No job {job_id}.") from None
    if outcome.result == "finished":
        raise ApiError(409, "job_finished", f"Job {job_id} already {outcome.status}.")
    if outcome.result == "refused":
        if outcome.phase == "finalizing":
            message = "The final test has started and cannot be interrupted."
        else:
            message = f"Job {job_id} is past the point where it can be cancelled ({outcome.phase})."
        raise ApiError(409, "cannot_cancel_now", message)
    response.status_code = 200 if outcome.result == "cancelled" else 202
    return CancelResult(outcome=outcome.result, job=summary(repo.get(job_id, with_result=False)))
