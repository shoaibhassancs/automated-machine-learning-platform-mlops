"""Request and response models of the serving app."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

from automl_ops.api_serving.settings import MAX_ROWS
from automl_ops.feedback import LabelRow


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictRequest(Strict):
    rows: list[dict[str, Any]] = Field(
        min_length=1,
        max_length=MAX_ROWS,
        description=(
            "One JSON object per row, with exactly the model's input columns (see `GET /model`). "
            "`null` or an empty string is a missing value. Numeric columns take a number or a "
            "clean number as text; categorical columns take text."
        ),
    )
    simulated: bool = Field(
        default=False,
        description="Label these rows as simulated traffic (never real observations).",
    )


class PredictionItem(Strict):
    prediction_id: str = Field(description="Identifies this row's prediction (for feedback later).")
    prediction: str | float = Field(
        description="The predicted class (classification) or value (regression)."
    )
    probabilities: dict[str, float] | None = Field(
        description="Class probabilities in the model's class order (classification only)."
    )


class DeploymentRef(Strict):
    deployment_id: int
    model: str
    version: int
    package_id: str
    task: Literal["classification", "regression"]


class PredictResponse(Strict):
    deployment: DeploymentRef
    predictions: list[PredictionItem]
    latency_ms: float = Field(description="Model time for the whole request, in milliseconds.")
    logged: bool = Field(
        description="False if the prediction log could not be written; the predictions are valid."
    )


class InputColumn(Strict):
    name: str
    kind: Literal["numeric", "categorical"]


class ModelInfo(Strict):
    deployment: DeploymentRef
    target: str
    algorithm: str
    classes: list[str] | None
    input_columns: list[InputColumn]
    deployed_at: str
    loaded_at: str
    registry_status: str = Field(
        description="The version's registry status now (it can differ from when it was deployed)."
    )
    max_rows: int


class IsolationReport(Strict):
    """Whether this process is wired to a throwaway stack (E2E safety). Booleans, reason codes and
    two one-way identifiers only: no path, host, user, password or token."""

    isolated: bool
    checks: dict[str, bool]
    reasons: list[str]
    database_id: str | None
    run_id: str | None


class HealthResponse(Strict):
    status: Literal["ok"] = Field(description="The process is up.")
    ready: bool = Field(description="A verified model is being served.")
    model_state: str
    database: Literal["ok", "unavailable"]
    desired_deployment_id: int | None
    loaded_deployment_id: int | None
    error: str | None
    checked_at: str | None
    isolation: IsolationReport | None = None


class FeedbackRequest(Strict):
    prediction_id: str = Field(
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        description="A `prediction_id` returned by `POST /predict`.",
    )
    true_value: StrictStr | StrictInt | StrictFloat = Field(
        description=(
            "The true outcome: one of the model's classes as text (classification), or a finite "
            "number (regression)."
        )
    )
    deployment_id: int | None = Field(
        default=None,
        ge=1,
        description="Optional: refuse (409) unless the prediction belongs to this deployment.",
    )
    model_version: int | None = Field(
        default=None,
        ge=1,
        description="Optional: refuse (409) unless the prediction was made by this model version.",
    )


class LabelModel(Strict):
    id: int
    prediction_id: str
    deployment_id: int
    model_version: int
    true_value: str
    created_at: str
    prediction: str | float | None = Field(description="What the model predicted for this row.")
    is_simulated: bool | None


class FeedbackResponse(Strict):
    label: LabelModel
    created: bool = Field(
        description="False: this exact label had already been recorded; nothing was written."
    )


def label_model(row: LabelRow) -> LabelModel:
    return LabelModel(
        id=row.id,
        prediction_id=row.prediction_id,
        deployment_id=row.deployment_id,
        model_version=row.model_version,
        true_value=row.true_value,
        created_at=row.created_at.isoformat(),
        prediction=row.prediction,
        is_simulated=row.is_simulated,
    )
