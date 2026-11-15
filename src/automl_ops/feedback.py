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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import numpy as np
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from automl_ops.deployments import DeploymentNotFound
from automl_ops.evaluate import (
    EvaluationError,
    classification_scalars,
    evaluate_classification,
    evaluate_regression,
    regression_scalars,
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

    def predictions(
        self,
        deployment_id: int,
        scope: Scope = "real",
        since: datetime | None = None,
        until: datetime | None = None,
        before: str | None = None,
        limit: int = PAGE,
    ) -> list[Any]:
        """Logged predictions, newest first: when, which version, what was answered, how fast and
        whether a true value is recorded. The input values are deliberately not returned. `before`
        is the cursor `<created_at microseconds>.<id>` of the last row of the previous page."""
        flags = {"real": [False], "simulated": [True], "all": [False, True]}[scope]
        cursor_time, cursor_id = _parse_cursor(before)
        with self.engine.connect() as c:
            self._deployment(c, deployment_id)
            return list(
                c.execute(
                    text(
                        "SELECT p.id, p.created_at, p.model_version, p.is_simulated, p.latency_ms, "
                        "p.output->'prediction' AS prediction, "
                        "CASE WHEN jsonb_typeof(p.output->'probabilities') = 'object' THEN "
                        "(SELECT max(v::float) FROM jsonb_each_text(p.output->'probabilities') "
                        "AS t(k, v)) END AS confidence, f.true_value, f.id AS label_id "
                        "FROM predictions p LEFT JOIN feedback f ON f.prediction_id = p.id "
                        "WHERE p.deployment_id = :d AND p.is_simulated = ANY(:s) "
                        "AND (CAST(:since AS timestamptz) IS NULL OR p.created_at >= :since) "
                        "AND (CAST(:until AS timestamptz) IS NULL OR p.created_at < :until) "
                        "AND (CAST(:ct AS timestamptz) IS NULL "
                        "OR (p.created_at, p.id) < (CAST(:ct AS timestamptz), :ci)) "
                        "ORDER BY p.created_at DESC, p.id DESC LIMIT :n"
                    ),
                    {
                        "d": deployment_id,
                        "s": flags,
                        "since": since,
                        "until": until,
                        "ct": cursor_time,
                        "ci": cursor_id,
                        "n": min(max(limit, 1), PAGE),
                    },
                ).all()
            )

    def activity(
        self,
        deployment_id: int,
        scope: Scope = "all",
        since: datetime | None = None,
        until: datetime | None = None,
        bucket: str | None = None,
    ) -> dict[str, Any]:
        """Prediction counts per time bucket (empty buckets included), split into real and
        simulated traffic, with how many are labelled, and overall latency. Bounded: at most
        `MAX_BUCKETS` buckets, else `ActivityRangeError`."""
        flags = {"real": [False], "simulated": [True], "all": [False, True]}[scope]
        with self.engine.connect() as c:
            c.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
            d = self._deployment(c, deployment_id)
            bounds = c.execute(
                text(
                    "SELECT min(created_at) AS first, max(created_at) AS last, count(*) AS n "
                    "FROM predictions WHERE deployment_id = :d AND is_simulated = ANY(:s)"
                ),
                {"d": deployment_id, "s": flags},
            ).one()
            end = until or datetime.now(UTC)
            start = since or bounds.first or (end - timedelta(hours=1))
            if start >= end:
                raise ActivityRangeError("`since` must be before `until`")
            step = bucket or _auto_bucket(end - start)
            span = BUCKETS[step]
            first = _floor(start, step)
            count = int((end - first) / span) + 1
            if count > MAX_BUCKETS:
                raise ActivityRangeError(
                    f"{count} {step} buckets is more than the {MAX_BUCKETS} allowed; "
                    "choose a shorter range or a larger bucket"
                )
            rows = c.execute(
                text(
                    "SELECT date_trunc(:u, p.created_at AT TIME ZONE 'UTC') AS b, "
                    "p.is_simulated AS sim, "
                    "count(*) AS predictions, count(f.id) AS labelled "
                    "FROM predictions p LEFT JOIN feedback f ON f.prediction_id = p.id "
                    "WHERE p.deployment_id = :d AND p.is_simulated = ANY(:s) "
                    "AND p.created_at >= :since AND p.created_at < :until GROUP BY 1, 2"
                ),
                {"u": step, "d": deployment_id, "s": flags, "since": first, "until": end},
            ).all()
            latency = c.execute(
                text(
                    "SELECT count(*) AS n, avg(latency_ms) AS mean, "
                    "percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95 "
                    "FROM predictions WHERE deployment_id = :d AND is_simulated = ANY(:s) "
                    "AND created_at >= :since AND created_at < :until"
                ),
                {"d": deployment_id, "s": flags, "since": first, "until": end},
            ).one()
        found = {
            (r.b.replace(tzinfo=UTC), bool(r.sim)): (int(r.predictions), int(r.labelled))
            for r in rows
        }
        buckets = []
        for i in range(count):
            at = first + i * span
            real, sim = found.get((at, False), (0, 0)), found.get((at, True), (0, 0))
            buckets.append(
                {
                    "start": at.isoformat(),
                    "real": {"predictions": real[0], "labelled": real[1]},
                    "simulated": {"predictions": sim[0], "labelled": sim[1]},
                }
            )
        return {
            "deployment_id": d.id,
            "scope": scope,
            "since": first.isoformat(),
            "until": end.isoformat(),
            "bucket": step,
            "buckets": buckets,
            "totals": {
                "real": _total(buckets, "real"),
                "simulated": _total(buckets, "simulated"),
            },
            "latency_ms": {
                "mean": None if latency.mean is None else round(float(latency.mean), 3),
                "p95": None if latency.p95 is None else round(float(latency.p95), 3),
                "predictions": int(latency.n),
            },
        }

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


# --------------------------------------------------------------------------------------------
# Windowed scoring (monitoring): the same scoring code over rows the caller already selected
# --------------------------------------------------------------------------------------------


def window_performance(
    task: str, labelled: Sequence[Any], min_labelled: int = DEFAULT_MIN_LABELLED
) -> dict[str, Any]:
    """Live metrics for exactly the labelled rows given (rows with `output`, `true_value` and
    `created_at`). Same fields and same scoring as `FeedbackRepository.performance`, but it never
    reads the database: the monitoring run chooses the rows, so the report and the drift verdict
    describe the same window. Below `min_labelled` no metrics are produced."""
    result: dict[str, Any] = {"labelled": len(labelled), "min_labelled": min_labelled, "live": None}
    if not labelled:
        result["status"] = "no_labels"
    elif len(labelled) < min_labelled:
        result["status"] = "insufficient_labels"
    else:
        try:
            result["live"] = _score(task, labelled)
            result["status"] = "ok"
        except EvaluationError:
            result["status"] = "not_computable"
    return result


def metric_sampler(
    task: str, labelled: Sequence[Any], metric: str
) -> Callable[[np.ndarray[Any, Any]], float | None]:
    """A function from row indices (a bootstrap resample) to the value of `metric` on those rows,
    or None where it is undefined for that resample. Uses the same scalar code as live scoring,
    so direction and definition cannot drift apart."""
    if task == "classification":
        classes = sorted(labelled[0].output["probabilities"])
        y_true = np.array([r.true_value for r in labelled])
        y_pred = np.array([str(r.output["prediction"]) for r in labelled])
        proba = np.array([[r.output["probabilities"][c] for c in classes] for r in labelled])

        def classification(index: np.ndarray[Any, Any]) -> float | None:
            try:
                scalars, _ = classification_scalars(
                    y_true[index], y_pred[index], proba[index], classes
                )
            except (EvaluationError, ValueError):
                return None
            return scalars.get(metric)

        return classification
    t = np.array([float(r.true_value) for r in labelled])
    p = np.array([float(r.output["prediction"]) for r in labelled])

    def regression(index: np.ndarray[Any, Any]) -> float | None:
        try:
            scalars, _ = regression_scalars(t[index], p[index])
        except (EvaluationError, ValueError):
            return None
        return scalars.get(metric)

    return regression


# --------------------------------------------------------------------------------------------
# Read-only history and activity (Phase 5)
# --------------------------------------------------------------------------------------------

MAX_BUCKETS = 500
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
BUCKETS = {
    "minute": timedelta(minutes=1),
    "hour": timedelta(hours=1),
    "day": timedelta(days=1),
}
CURSOR_RE = re.compile(r"^\d{1,20}\.[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class ActivityRangeError(ValueError):
    """The requested time range and bucket would give too many (or no) buckets."""


def _auto_bucket(span: timedelta) -> str:
    """The smallest bucket that keeps the chart to at most 200 points."""
    for name in ("minute", "hour", "day"):
        if span / BUCKETS[name] <= 200:
            return name
    return "day"


def _floor(moment: datetime, step: str) -> datetime:
    moment = moment.astimezone(UTC)
    if step == "minute":
        return moment.replace(second=0, microsecond=0)
    if step == "hour":
        return moment.replace(minute=0, second=0, microsecond=0)
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)


def _total(buckets: list[dict[str, Any]], key: str) -> dict[str, int]:
    return {
        "predictions": sum(b[key]["predictions"] for b in buckets),
        "labelled": sum(b[key]["labelled"] for b in buckets),
    }


def make_cursor(created_at: datetime, prediction_id: str) -> str:
    moment = created_at.astimezone(UTC)
    micros = (moment - EPOCH) // timedelta(microseconds=1)
    return f"{micros}.{prediction_id}"


def _parse_cursor(cursor: str | None) -> tuple[datetime | None, str | None]:
    if cursor is None:
        return None, None
    if not CURSOR_RE.match(cursor):
        raise ActivityRangeError("`before` is not a cursor from this endpoint")
    micros, _, uid = cursor.partition(".")
    return EPOCH + timedelta(microseconds=int(micros)), uid
