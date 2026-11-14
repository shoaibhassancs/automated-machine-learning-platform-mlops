"""Feedback labels and live model performance (P4-8, BUILD_PLAN Step 23).

A **label** is the true value for one logged prediction. PostgreSQL is authoritative and the
table is append-only: a prediction has at most one label, and a label is never changed or
removed (migration 0006 enforces this, and that the label's deployment and model version are the
prediction's own).

**Submitting.** The prediction must exist; optional `deployment_id` / `model_version` in the request
must match it (else 409, nothing is stored), so a label can never be attached to a different model.
A classification label must be one of the model's classes (taken from the prediction's own
probabilities, so it works after the deployment ended); a regression label must be a finite number.
The same label again is an idempotent no-op (`created: false`); a *different* label for an already
labelled prediction is refused (409 `feedback_conflict`) and the first label stands.

**Live performance.** Computed on demand, per deployment, **only on predictions that have a label**,
with the same functions as the offline evaluation (`evaluate.py`, Step 10). It is never presented
as the offline test result: the offline numbers are returned separately and labelled as such. A
sample below `min_labelled` (default 30) gets a status and counts, not metrics. Simulated traffic
is separate: the default scope is real traffic only. Nothing here reads any dataset or the held-out
test part; the offline reference is the summary already stored with the registered version.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from automl_ops.deployments import DeploymentNotFound
from automl_ops.evaluate import (
    EvaluationError,
    evaluate_classification,
    evaluate_regression,
)
from automl_ops.profile import NUMBER_RE

DEFAULT_MIN_LABELLED = 30
MAX_SCORED = 50_000  # the most recent labelled rows scored at once; a larger set says so
PAGE = 100
STATEMENT_TIMEOUT_MS = 5000
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

Scope = Literal["real", "simulated", "all"]
OFFLINE_NOTE = (
    "Offline reference: the one final test made before this version was registered. It is not "
    "comparable to live numbers without care: different rows, a different time, and the live "
    "labels may be a biased sample."
)


class FeedbackError(Exception):
    code = "feedback_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class PredictionNotFound(FeedbackError):
    code = "prediction_not_found"


class InvalidTrueValue(FeedbackError):
    code = "invalid_true_value"


class ReferenceMismatch(FeedbackError):
    code = "prediction_mismatch"


class FeedbackConflict(FeedbackError):
    code = "feedback_conflict"


class FeedbackUnavailable(FeedbackError):
    code = "feedback_unavailable"


@dataclass(frozen=True)
class LabelRow:
    id: int
    prediction_id: str
    deployment_id: int
    model_version: int
    true_value: str
    created_at: datetime
    prediction: str | float | None = None
    is_simulated: bool | None = None


@dataclass(frozen=True)
class Submitted:
    label: LabelRow
    created: bool  # False: the same label had already been recorded (nothing was written)


def canonical_true_value(raw: Any, classes: list[str] | None) -> str:
    """The stored text of a label, or `InvalidTrueValue`."""
    if classes is not None:
        if not isinstance(raw, str) or raw not in classes:
            raise InvalidTrueValue(
                "the true value must be one of the model's classes, given as text",
                {"classes": classes},
            )
        return raw
    if isinstance(raw, bool) or not isinstance(raw, int | float | str):
        raise InvalidTrueValue("the true value must be a finite number")
    if isinstance(raw, str) and not NUMBER_RE.fullmatch(raw):
        raise InvalidTrueValue("the true value must be a clean number")
    value = float(raw)
    if not math.isfinite(value):
        raise InvalidTrueValue("the true value must be a finite number")
    return repr(value)


def _classes(output: dict[str, Any]) -> list[str] | None:
    probabilities = output.get("probabilities")
    return None if probabilities is None else sorted(probabilities)  # the platform's class order


class FeedbackRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # ------------------------------------------------------------------ submit

    def submit(
        self,
        prediction_id: str,
        true_value: Any,
        *,
        deployment_id: int | None = None,
        model_version: int | None = None,
    ) -> Submitted:
        try:
            return self._submit(prediction_id, true_value, deployment_id, model_version)
        except SQLAlchemyError as error:
            raise FeedbackUnavailable(
                f"the feedback could not be stored ({type(error).__name__}); nothing was changed"
            ) from None

    def _submit(
        self, prediction_id: str, true_value: Any, deployment_id: int | None, version: int | None
    ) -> Submitted:
        with self.engine.begin() as c:
            c.execute(text(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}"))
            p = c.execute(
                text(
                    "SELECT id, deployment_id, model_version, output FROM predictions WHERE id = :i"
                ),
                {"i": prediction_id},
            ).first()
            if p is None:
                raise PredictionNotFound(f"no logged prediction {prediction_id}")
            if deployment_id is not None and deployment_id != p.deployment_id:
                raise ReferenceMismatch(
                    "the prediction belongs to a different deployment",
                    {"prediction_deployment_id": p.deployment_id},
                )
            if version is not None and version != p.model_version:
                raise ReferenceMismatch(
                    "the prediction was made by a different model version",
                    {"prediction_model_version": p.model_version},
                )
            value = canonical_true_value(true_value, _classes(p.output))
            inserted = c.execute(
                text(
                    "INSERT INTO feedback (prediction_id, deployment_id, model_version, "
                    "true_value) VALUES (:i, :d, :v, :t) "
                    "ON CONFLICT (prediction_id) DO NOTHING RETURNING id"
                ),
                {"i": prediction_id, "d": p.deployment_id, "v": p.model_version, "t": value},
            ).first()
            row = c.execute(
                text(_LABEL + " WHERE f.prediction_id = :i"), {"i": prediction_id}
            ).one()
            label = _label(row)
        if inserted is not None:
            return Submitted(label, True)
        if label.true_value != value:
            raise FeedbackConflict(
                "this prediction already has a different label; the first label stands",
                {
                    "existing_true_value": label.true_value,
                    "recorded_at": label.created_at.isoformat(),
                },
            )
        return Submitted(label, False)

    # -------------------------------------------------------------------- reads

    def labels(
        self, deployment_id: int, before_id: int | None = None, limit: int = PAGE
    ) -> list[LabelRow]:
        """Labels of a deployment, newest first."""
        with self.engine.connect() as c:
            self._deployment(c, deployment_id)
            rows = c.execute(
                text(
                    _LABEL
                    + " WHERE f.deployment_id = :d AND (CAST(:b AS bigint) IS NULL OR f.id < :b)"
                    " ORDER BY f.id DESC LIMIT :n"
                ),
                {"d": deployment_id, "b": before_id, "n": min(max(limit, 1), PAGE)},
            ).all()
        return [_label(r) for r in rows]

    @staticmethod
    def _deployment(c: Any, deployment_id: int) -> Any:
        row = c.execute(
            text(
                "SELECT d.id, d.status, m.name AS model, v.version, v.task, v.evaluation "
                "FROM deployments d JOIN model_versions v ON v.id = d.version_id "
                "JOIN registered_models m ON m.id = d.model_id WHERE d.id = :i"
            ),
            {"i": deployment_id},
        ).first()
        if row is None:
            raise DeploymentNotFound(f"no deployment {deployment_id}")
        return row

    def performance(
        self,
        deployment_id: int,
        scope: Scope = "real",
        min_labelled: int = DEFAULT_MIN_LABELLED,
    ) -> dict[str, Any]:
        flags = {"real": [False], "simulated": [True], "all": [False, True]}[scope]
        with self.engine.connect() as c:
            # one snapshot for every query below, so labels arriving meanwhile cannot make the
            # counts and the scored rows disagree
            c.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
            d = self._deployment(c, deployment_id)
            counts = c.execute(
                text(
                    "SELECT p.is_simulated AS simulated, count(*) AS predictions, "
                    "count(f.id) AS labelled FROM predictions p "
                    "LEFT JOIN feedback f ON f.prediction_id = p.id "
                    "WHERE p.deployment_id = :d GROUP BY p.is_simulated"
                ),
                {"d": deployment_id},
            ).all()
            labelled = c.execute(
                text(
                    "SELECT p.output, f.true_value, f.created_at FROM feedback f "
                    "JOIN predictions p ON p.id = f.prediction_id "
                    "WHERE f.deployment_id = :d AND p.is_simulated = ANY(:s) "
                    "ORDER BY f.id DESC LIMIT :n"
                ),
                {"d": deployment_id, "s": flags, "n": MAX_SCORED},
            ).all()
        by = {bool(r.simulated): (int(r.predictions), int(r.labelled)) for r in counts}
        in_scope = [by.get(flag, (0, 0)) for flag in flags]
        predictions = sum(p for p, _ in in_scope)
        total_labelled = sum(n for _, n in in_scope)
        result: dict[str, Any] = {
            "deployment_id": d.id,
            "model": d.model,
            "version": d.version,
            "task": d.task,
            "scope": scope,
            "min_labelled": min_labelled,
            "counts": {
                "predictions": predictions,
                "labelled": total_labelled,
                "unlabelled": predictions - total_labelled,
            },
            "by_traffic": {
                "real": dict(zip(("predictions", "labelled"), by.get(False, (0, 0)), strict=True)),
                "simulated": dict(
                    zip(("predictions", "labelled"), by.get(True, (0, 0)), strict=True)
                ),
            },
            "rows_scored": 0,
            "truncated": False,
            "live": None,
            "offline_reference": _offline(d.evaluation),
            "notes": [],
        }
        notes: list[str] = result["notes"]
        if scope != "real" and by.get(True, (0, 0))[0]:
            notes.append("simulated traffic is included in or reported next to these numbers")
        if predictions == 0:
            result["status"] = "no_predictions"
        elif total_labelled == 0:
            result["status"] = "no_labels"
        elif total_labelled < min_labelled:
            result["status"] = "insufficient_labels"
            notes.append(
                f"{total_labelled} labelled predictions is fewer than the {min_labelled} needed; "
                "no metrics are shown rather than numbers that would be over-read"
            )
        else:
            result["rows_scored"] = len(labelled)
            result["truncated"] = total_labelled > len(labelled)
            if result["truncated"]:
                notes.append(f"the {len(labelled)} most recent labelled predictions were scored")
            try:
                result["live"] = _score(d.task, labelled)
                result["status"] = "ok"
            except EvaluationError as error:
                result["status"] = "not_computable"
                notes.append(str(error))
        return result


_LABEL = (
    "SELECT f.id, f.prediction_id, f.deployment_id, f.model_version, f.true_value, f.created_at, "
    "p.output->'prediction' AS prediction, p.is_simulated "
    "FROM feedback f JOIN predictions p ON p.id = f.prediction_id"
)


def _label(row: Any) -> LabelRow:
    return LabelRow(**dict(row._mapping))


def _offline(evaluation: dict[str, Any]) -> dict[str, Any]:
    inner = evaluation.get("evaluation", {})
    return {
        "source": "offline_final_test",
        "metrics": inner.get("metrics"),
        "rows": inner.get("rows"),
        "undefined": inner.get("undefined"),
        "limitation": evaluation.get("limitation"),
        "note": OFFLINE_NOTE,
    }


def _score(task: str, labelled: Sequence[Any]) -> dict[str, Any]:
    window = {
        "first_label_at": min(r.created_at for r in labelled).isoformat(),
        "last_label_at": max(r.created_at for r in labelled).isoformat(),
    }
    if task == "classification":
        classes = sorted(labelled[0].output["probabilities"])
        y_true = np.array([r.true_value for r in labelled])
        y_pred = np.array([str(r.output["prediction"]) for r in labelled])
        proba = np.array([[r.output["probabilities"][c] for c in classes] for r in labelled])
        scored = evaluate_classification(y_true, y_pred, proba, classes, curves=False)
    else:
        y_true = np.array([float(r.true_value) for r in labelled])
        y_pred = np.array([float(r.output["prediction"]) for r in labelled])
        scored = evaluate_regression(y_true, y_pred, np.arange(len(labelled)), residuals=False)
    return {"source": "live_feedback", "window": window, **scored}
