"""Response and request models. Every model forbids extra fields, so a library change that adds a
field to a report fails the contract tests instead of silently drifting from the API."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from automl_ops.api_serving.schemas import LabelModel
from automl_ops.profile import Kind
from automl_ops.task import TaskType
from automl_ops.validate import Severity


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# -- datasets ---------------------------------------------------------------------------------


class VersionModel(Strict):
    ref: str
    dataset: str
    number: int
    sha256: str
    original_sha256: str
    size_bytes: int
    original_size_bytes: int
    rows: int
    columns: list[str]
    source_filename: str
    original_encoding: str
    original_delimiter: str
    reencoded: bool
    trailing_blank_lines_ignored: int
    created_at: str
    store_format: int


class DatasetModel(Strict):
    name: str
    versions: list[VersionModel]


class DatasetList(Strict):
    datasets: list[DatasetModel]


class UploadInfo(Strict):
    sha256: str
    encoding: str
    delimiter: str
    reencoded: bool


class UploadResult(Strict):
    created: bool
    catalogued: bool
    version: VersionModel
    upload: UploadInfo


# -- health -----------------------------------------------------------------------------------


class DatabaseHealth(Strict):
    reachable: bool
    at_head: bool | None
    revision: str | None
    head: str | None


class Health(Strict):
    status: Literal["ok", "degraded"]
    database: DatabaseHealth
    worker: Literal["running", "stopped", "unknown"] = "unknown"  # is a job worker holding the lock


# -- profile ----------------------------------------------------------------------------------


class ValueCount(Strict):
    value: str
    count: int


class NumericStatsModel(Strict):
    min: float
    max: float
    mean: float
    std: float | None
    median: float
    p25: float
    p75: float


class ColumnProfileModel(Strict):
    name: str
    position: int
    kind: Kind
    count: int
    missing: int
    missing_pct: float
    unique: int
    top_values: list[ValueCount]
    numeric: NumericStatsModel | None
    numeric_share: float
    non_numeric_examples: list[ValueCount]
    na_like_count: int
    na_like_tokens: list[ValueCount]
    leading_zero_count: int
    leading_zero_examples: list[ValueCount]


class ProfileSettingsModel(Strict):
    max_categories: int
    mixed_min_numeric_share: float
    top_n: int


class ProfileReport(Strict):
    dataset_ref: str
    sha256: str
    profile_format: int
    settings: ProfileSettingsModel
    rows: int
    columns: int
    duplicate_rows: int
    rows_in_duplicate_groups: int
    rows_with_missing: int
    missing_cells: int
    column_profiles: list[ColumnProfileModel]


# -- validation and task ----------------------------------------------------------------------


class FindingModel(Strict):
    code: str
    severity: Severity
    column: str | None
    message: str
    details: dict[str, Any]  # shape depends on the finding code (docs/profiling.md)


class ThresholdsModel(Strict):
    min_rows_error: int
    min_rows_warning: int
    high_missing_pct: float
    id_like_min_rows: int
    id_like_unique_ratio: float


class ValidationReportModel(Strict):
    dataset_ref: str
    sha256: str
    ok: bool
    target: str | None
    errors: int
    warnings: int
    infos: int
    thresholds: ThresholdsModel
    findings: list[FindingModel]


class TaskConfigModel(Strict):
    target: str
    task: TaskType
    excluded: list[str]


class TaskSuggestionModel(Strict):
    task: TaskType
    reason: str


class TaskSettingsModel(Strict):
    whole_number_regression_min_distinct: int
    min_class_size_error: int
    min_class_size_warning: int
    max_classes_warning: int


class TaskReportModel(Strict):
    dataset_ref: str
    sha256: str
    ok: bool
    config: TaskConfigModel
    suggestion: TaskSuggestionModel | None
    errors: int
    warnings: int
    infos: int
    settings: TaskSettingsModel
    findings: list[FindingModel]
    feature_columns: list[str]


class TaskCheckRequest(Strict):
    """Stateless: checks a task configuration against a version and stores nothing."""

    target: str = Field(min_length=1, max_length=1000)
    task: TaskType | None = None
    excluded: list[str] = Field(default_factory=list, max_length=100)


# -- preview ----------------------------------------------------------------------------------


class ExploreSettingsModel(Strict):
    sample_rows: int
    histogram_bins: int


class HistogramModel(Strict):
    column: str
    count: int
    missing: int
    min: float
    max: float
    edges: list[float]
    counts: list[int]


class CorrelationPair(Strict):
    a: str
    b: str
    r: float


class CorrelationModel(Strict):
    columns: list[str]
    matrix: list[list[float | None]]
    top_pairs: list[CorrelationPair]


class ClassShare(Strict):
    value: str
    count: int
    share: float


class ClassBalanceModel(Strict):
    total_classes: int
    rows: int
    classes: list[ClassShare]
    classes_not_listed: int
    largest_class_share: float | None
    smallest_class_rows: int
    largest_to_smallest_ratio: float | None


class ExploreReport(Strict):
    dataset_ref: str
    sha256: str
    explore_format: int
    settings: ExploreSettingsModel
    rows: int
    columns: list[str]
    sample_rows: list[dict[str, str | None]]
    histograms: list[HistogramModel]
    correlation: CorrelationModel
    class_balance: ClassBalanceModel | None


# -- training jobs (P4-4) ---------------------------------------------------------------------


class ParamModel(Strict):
    name: str
    kind: str
    default: Any
    minimum: float | None
    maximum: float | None
    choices: list[str]
    description: str


class AlgorithmModel(Strict):
    key: str
    label: str
    task: TaskType
    scale_numeric: bool
    params: list[ParamModel]


class AlgorithmList(Strict):
    algorithms: list[AlgorithmModel]


class AlgorithmChoice(Strict):
    key: str = Field(min_length=1, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict, max_length=20)


class JobRequest(Strict):
    """A training job. `kind: train` is cross-validation only; `kind: final` also selects the best
    algorithm and tests it once on the held-out part (needs `confirm_final_test: true`)."""

    dataset: str = Field(min_length=1, max_length=64)
    version: int = Field(ge=1, le=9_999_999)
    kind: Literal["train", "final"] = "train"
    target: str = Field(min_length=1, max_length=1000)
    task: TaskType | None = None
    excluded: list[str] = Field(default_factory=list, max_length=100)
    algorithms: list[AlgorithmChoice] | None = Field(default=None, max_length=20)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    folds: int = Field(default=5, ge=1, le=100)
    test_fraction: float = Field(default=0.2, gt=0, lt=1)
    selection_metric: str | None = Field(default=None, min_length=1, max_length=32)
    confirm_final_test: bool = False
    track: bool = False  # opt-in MLflow logging


class JobProgress(Strict):
    event: str | None = None
    algorithm: str | None = None
    algorithm_index: int | None = None
    algorithm_count: int | None = None
    fold: int | None = None
    folds: int | None = None


class JobFailure(Strict):
    code: str
    message: str


class JobSettingsModel(Strict):
    test_fraction: float
    seed: int
    folds: int


class JobConfigModel(Strict):
    target: str
    task: TaskType
    excluded: list[str]
    algorithms: list[AlgorithmChoice]
    settings: JobSettingsModel
    selection_metric: str | None


class JobSummary(Strict):
    id: int
    kind: Literal["train", "final"]
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    phase: str
    dataset: str
    dataset_sha256: str
    track: bool
    tracking_state: Literal["off", "logged", "pending"]
    cancel_requested: bool
    stale: bool
    progress: JobProgress | None
    error: JobFailure | None
    created_at: str
    started_at: str | None
    finished_at: str | None


class JobEventModel(Strict):
    seq: int
    kind: str
    payload: dict[str, Any]
    created_at: str


class JobResultModel(Strict):
    """The same payloads as `scripts/datasets.py train|finalize --json` (the nested library
    reports are passed through as JSON, not described field by field)."""

    dataset_ref: str
    sha256: str
    training: dict[str, Any]
    selection: dict[str, Any] | None
    selection_sha256: str | None
    selection_note: str | None
    test_result: dict[str, Any] | None
    package_id: str | None


class JobDetail(JobSummary):
    config: JobConfigModel
    confirm_final_test: bool
    package_id: str | None
    tracking_job_key: str | None
    tracking_error: str | None
    result: JobResultModel | None
    events: list[JobEventModel]
    last_seq: int


class JobList(Strict):
    jobs: list[JobSummary]
    next_before_id: int | None


class CancelResult(Strict):
    outcome: Literal["cancelled", "requested"]
    job: JobSummary


# -- model registry (P4-5) --------------------------------------------------------------------

ModelStatus = Literal["development", "staging", "production", "archived"]
RegistryAction = Literal["register", "promote", "archive", "supersede", "rollback"]
MirrorState = Literal["disabled", "in_sync", "pending", "failed", "drifted", "conflict"]
ModelName = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]
Reason = Annotated[str, Field(min_length=1, max_length=500)]


class RegisterRequest(Strict):
    """Register a sealed package as the next version of a model. Give a `package_id`, a `job_id`
    (a succeeded final job), or both; if both, they must name the same package."""

    name: ModelName
    package_id: str | None = Field(
        default=None, pattern=r"^m-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$", max_length=64
    )
    job_id: int | None = Field(default=None, ge=1, le=2_147_483_647)
    reason: Reason | None = None


class PromoteRequest(Strict):
    to: Literal["staging", "production"]
    reason: Reason


class ReasonRequest(Strict):
    reason: Reason


class ModelVersionSummary(Strict):
    model: str
    version: int
    status: ModelStatus
    package_id: str
    job_id: int | None
    dataset: str | None
    algorithm: str
    task: str
    target: str
    registered_at: str
    status_changed_at: str


class MirrorSummary(Strict):
    """Where the MLflow mirror of a model stands (P4-6). PostgreSQL is authoritative; this only says
    whether MLflow has caught up. `disabled` unless `AUTOML_REGISTRY_MIRROR=on`."""

    state: MirrorState
    desired_seq: int
    synced_seq: int
    pending_events: int
    attempts: int
    last_error: str | None
    last_error_kind: Literal["error", "drift", "conflict"] | None
    last_success_at: str | None
    next_attempt_at: str | None


class ModelVersionDetail(ModelVersionSummary):
    selection_sha256: str
    # The stored summary of the sealed package's test result (metrics, confusion matrix, the
    # limitation text); curves and residual points stay in the package and are named in `omitted`.
    evaluation: dict[str, Any]
    mirror: MirrorSummary | None = None  # set on registry changes and detail reads (P4-6)


class RegisteredModel(Strict):
    name: str
    created_at: str
    versions: int
    production_version: int | None
    latest_version: int | None
    mirror: MirrorSummary | None = None


class RegisteredModelList(Strict):
    models: list[RegisteredModel]


class ModelVersionList(Strict):
    model: str
    versions: list[ModelVersionSummary]


class RegistryEventModel(Strict):
    seq: int
    version: int
    action: RegistryAction
    from_status: ModelStatus | None
    to_status: ModelStatus
    reason: str
    created_at: str


class RegistryEventList(Strict):
    model: str
    events: list[RegistryEventModel]
    last_seq: int


class RollbackResult(Strict):
    restored: ModelVersionDetail
    archived: ModelVersionDetail
    mirror: MirrorSummary | None = None


class MirrorFinding(Strict):
    """One difference between the registry and MLflow. `pending`: a sync will fix it. `drift`: an
    unrequested change to a platform-owned part; only a reconcile repairs it. `conflict`: not
    platform-owned or ambiguous; never changed."""

    kind: str
    category: Literal["pending", "drift", "conflict"]
    blocking: bool
    version: int | None
    expected: Any
    actual: Any
    message: str


class MirrorModelStatus(MirrorSummary):
    model: str


class MirrorOverview(Strict):
    enabled: bool
    problem: str | None
    models: list[MirrorModelStatus]


class MirrorCheck(Strict):
    model: str
    mirror: MirrorSummary
    in_sync: bool  # nothing pending, drifted or blocking; informational conflicts may remain
    findings: list[MirrorFinding]


class SyncResult(Strict):
    result: Literal["up_to_date", "synced", "failed", "drift", "conflict", "busy"]
    mirror: MirrorSummary
    actions: list[dict[str, Any]]
    findings: list[MirrorFinding]
    error: str | None


class ReconcileRequest(Strict):
    reason: Reason


class ReconcileResult(Strict):
    outcome: Literal["ok", "conflict", "failed", "partial"]
    mirror: MirrorSummary
    detected: list[MirrorFinding]
    intended: list[str]
    actions: list[dict[str, Any]]
    result: list[MirrorFinding]
    error: str | None


# --------------------------------------------------------------------------------------------
# Deployments (P4-7)
# --------------------------------------------------------------------------------------------


class DeploymentRequest(Strict):
    model: ModelName
    version: Annotated[int, Field(ge=1, le=9_999_999)]
    reason: Reason


class ServingReportModel(Strict):
    state: Literal["in_sync", "out_of_sync", "unreachable"] = Field(
        description=(
            "`in_sync`: the serving app serves exactly the active deployment (or none, if none is "
            "active). `out_of_sync`: it answered but is serving something else, or could not load "
            "the deployment (see `error`). `unreachable`: it did not answer; it will pick the "
            "deployment up from PostgreSQL when it starts or polls."
        )
    )
    loaded_deployment_id: int | None
    model_state: str | None
    error: str | None


class DeploymentModel(Strict):
    id: int
    model: str
    version: int
    package_id: str
    status: Literal["active", "ended"]
    action: Literal["deploy", "switch", "rollback"]
    reason: str
    replaced_id: int | None
    deployed_at: str
    ended_at: str | None
    end_reason: str | None
    registry_status: ModelStatus = Field(
        description="The version's registry status now; it can differ from when it was deployed."
    )
    task: str
    algorithm: str


class DeploymentDetail(DeploymentModel):
    serving: ServingReportModel | None = Field(
        default=None, description="The serving app's state right after this change or read."
    )


class DeploymentList(Strict):
    deployments: list[DeploymentModel]
    next_before_id: int | None = Field(description="Pass as `before_id` for the next page.")


# --------------------------------------------------------------------------------------------
# Feedback and live performance (P4-8)
# --------------------------------------------------------------------------------------------


class LabelList(Strict):
    deployment_id: int
    labels: list[LabelModel]
    next_before_id: int | None = Field(description="Pass as `before_id` for the next page.")


class TrafficCounts(Strict):
    predictions: int
    labelled: int


class LiveCounts(Strict):
    predictions: int = Field(description="Logged predictions in the chosen scope.")
    labelled: int = Field(
        description="Of those, how many have a true value. Only these are scored."
    )
    unlabelled: int = Field(description="Excluded from every metric.")


class LiveByTraffic(Strict):
    real: TrafficCounts
    simulated: TrafficCounts


class LiveMetrics(Strict):
    source: Literal["live_feedback"]
    window: dict[str, str]
    rows: int
    metrics: dict[str, float | None]
    undefined: dict[str, str]
    classes: list[str] | None = None
    confusion_matrix: list[list[int]] | None = None
    per_class: list[dict[str, Any]] | None = None


class OfflineReference(Strict):
    source: Literal["offline_final_test"]
    metrics: dict[str, float | None] | None
    rows: int | None
    undefined: dict[str, str] | None
    limitation: str | None
    note: str


class LivePerformance(Strict):
    deployment_id: int
    model: str
    version: int
    task: Literal["classification", "regression"]
    scope: Literal["real", "simulated", "all"]
    status: Literal[
        "no_predictions", "no_labels", "insufficient_labels", "ok", "not_computable"
    ] = Field(
        description=(
            "`ok`: `live` holds metrics. Otherwise `live` is null and `notes` says why; "
            "`insufficient_labels` means fewer than `min_labelled` labelled predictions."
        )
    )
    min_labelled: int
    counts: LiveCounts
    by_traffic: LiveByTraffic
    rows_scored: int
    truncated: bool
    live: LiveMetrics | None = Field(
        description="Metrics computed from labelled live predictions only."
    )
    offline_reference: OfflineReference = Field(
        description="The stored offline final-test summary: a different thing, shown separately."
    )
    notes: list[str]


# --------------------------------------------------------------------------------------------
# Monitoring: reference samples, drift reports, alerts (P4-9)
# --------------------------------------------------------------------------------------------

TrafficScope = Literal["real", "simulated"]


class ReferenceModel(Strict):
    id: int
    model: str
    version: int
    package_id: str
    dataset_ref: str = Field(description="The training dataset version the sample was drawn from.")
    dataset_sha256: str
    train_sha256: str = Field(
        description="SHA-256 of the training row positions (never test rows)."
    )
    seed: int
    train_rows: int
    sample_rows: int
    sample_sha256: str = Field(description="SHA-256 of the stored sample and its predictions.")
    created_at: str
    created: bool | None = Field(
        default=None, description="True: made by this call. False: it already existed."
    )


class ThresholdOverrides(Strict):
    alpha: float | None = Field(default=None, gt=0, le=1)
    js_distance: float | None = Field(default=None, gt=0, le=1)
    feature_share: float | None = Field(default=None, gt=0, le=1)
    performance_drop: float | None = Field(default=None, gt=0, le=1)
    min_rows: int | None = Field(default=None, ge=5, le=100_000)
    min_labelled: int | None = Field(default=None, ge=1, le=100_000)
    window: int | None = Field(default=None, ge=5, le=5000)


class MonitorRequest(Strict):
    scope: TrafficScope = Field(
        default="real",
        description="Which traffic to monitor. Real and simulated traffic are never mixed.",
    )
    thresholds: ThresholdOverrides = Field(default_factory=ThresholdOverrides)


class ReportModel(Strict):
    id: int
    deployment_id: int
    reference_id: int
    scope: TrafficScope
    status: Literal["ok", "insufficient_data", "failed"] = Field(
        description=(
            "`ok`: a drift verdict was reached. `insufficient_data`: too few rows, no verdict. "
            "`failed`: the run broke (see `error`); never a success."
        )
    )
    window_rows: int
    window_start: str | None
    window_end: str | None
    thresholds: dict[str, Any]
    summary: dict[str, Any] | None = Field(
        description="Feature drift, dataset-drift share, prediction drift, performance, rules."
    )
    error: str | None
    started_at: str
    finished_at: str


class ReportList(Strict):
    deployment_id: int
    reports: list[ReportModel]
    next_before_id: int | None


class MonitoringStatus(Strict):
    deployment_id: int
    reference_id: int | None = Field(description="Null: the version has no reference sample yet.")
    latest_real: ReportModel | None
    latest_simulated: ReportModel | None
    unresolved_alerts: list[dict[str, Any]]


class AlertEventModel(Strict):
    id: int
    action: Literal["opened", "reoccurred", "acknowledged", "resolved", "auto_resolved"]
    report_id: int | None
    reason: str | None
    created_at: str


class AlertModel(Strict):
    id: int
    deployment_id: int
    rule: Literal["drifted_feature_share", "prediction_shift", "performance_drop"]
    scope: TrafficScope
    severity: Literal["warning", "critical"]
    status: Literal["open", "acknowledged", "resolved"]
    details: dict[str, Any]
    first_report_id: int
    last_report_id: int
    occurrences: int
    created_at: str
    updated_at: str
    resolved_at: str | None
    events: list[AlertEventModel] | None = None


class AlertList(Strict):
    alerts: list[AlertModel]
    next_before_id: int | None
