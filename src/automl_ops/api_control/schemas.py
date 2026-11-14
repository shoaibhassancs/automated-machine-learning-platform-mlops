"""Response and request models. Every model forbids extra fields, so a library change that adds a
field to a report fails the contract tests instead of silently drifting from the API."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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
