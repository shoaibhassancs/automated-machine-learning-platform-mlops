"""The serving runtime (P4-7): which model is live, loading it safely, predicting, logging.

PostgreSQL decides what is live: the one `active` row of `deployments`. The runtime serves exactly
that deployment, nothing else, and never anything it could not verify:

1. the sealed package is loaded through `package.load_package` (seal hashes, read-only files,
   library versions, the model file's SHA-256 *before* it is deserialized);
2. its package id, task and frozen-selection hash must equal the registry's record;
3. the loaded model must reproduce its saved reference predictions (parity).

A deployment that fails any step is **not served**: `/predict` answers 503 `model_not_ready`, and
`/health` says why. The model of an earlier deployment is not kept as a stand-in, so what answers
is always what PostgreSQL says is live. If PostgreSQL itself becomes unreachable, the last verified
deployment keeps serving (nothing can be deployed or undeployed while the database is down) and
`/health` reports `database: unavailable`. A change of deployment is picked up within
`poll_seconds`, or at once when the control plane calls `POST /reload`.

Prediction logging never fails a prediction: a failed write is reported as `logged: false`.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from automl_ops.api_serving.inputs import Column, InvalidInput, prepare_rows
from automl_ops.db import migrate
from automl_ops.deployments import DeploymentRepository, DeploymentRow
from automl_ops.feedback import FeedbackRepository
from automl_ops.package import (
    ModelPackage,
    PackageError,
    load_package,
    loaded_parity_problems,
)

log = logging.getLogger("automl_ops.serving")

LOG_TIMEOUT_MS = 2000

# `model_state` values
READY = "ready"
STARTING = "starting"
NO_DEPLOYMENT = "no_deployment"
LOAD_FAILED = "load_failed"
SCHEMA_NOT_CURRENT = "schema_not_current"


class LoadFailure(Exception):
    """The active deployment's package could not be loaded and verified."""


class NotReady(Exception):
    """No verified model is being served; `code` and `message` say why."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ServingModel:
    deployment: DeploymentRow
    package: ModelPackage
    columns: tuple[Column, ...]
    loaded_at: datetime


@dataclass(frozen=True)
class Health:
    model_state: str
    database: str  # "ok" | "unavailable"
    desired_deployment_id: int | None
    loaded_deployment_id: int | None
    error: str | None
    checked_at: datetime | None

    @property
    def ready(self) -> bool:
        return self.model_state == READY


@dataclass(frozen=True)
class Served:
    prediction_id: str
    prediction: str | float
    probabilities: dict[str, float] | None


@dataclass(frozen=True)
class PredictionBatch:
    deployment: DeploymentRow
    items: list[Served]
    latency_ms: float
    logged: bool


def _columns(package: ModelPackage) -> tuple[Column, ...]:
    return tuple(
        Column(str(c["name"]), str(c["kind"]))
        for c in package.manifest["features"]["input_columns"]
    )


class ServingRuntime:
    def __init__(
        self,
        engine: Engine,
        models_root: Path,
        *,
        retry_seconds: float = 30.0,
        clock: Any = time.monotonic,
    ) -> None:
        self.engine = engine
        self.models_root = models_root
        self.repository = DeploymentRepository(engine, models_root)
        self.feedback = FeedbackRepository(engine)
        self.retry_seconds = retry_seconds
        self._clock = clock
        self._refresh_lock = threading.Lock()
        self._model: ServingModel | None = None
        self._health = Health(STARTING, "unavailable", None, None, None, None)
        self._failed: tuple[int, float] | None = None  # (deployment id, when it failed)

    # ------------------------------------------------------------------ state

    def health(self) -> Health:
        return self._health

    def model(self) -> ServingModel:
        """The model being served, or `NotReady` saying why there is none."""
        health, model = self._health, self._model
        if model is None or health.model_state != READY:
            raise _not_ready(health)
        return model

    def refresh(self, *, force: bool = False) -> Health:
        """Re-read the active deployment and (re)load if it changed. Never raises."""
        with self._refresh_lock:
            try:
                self._refresh(force)
            except Exception as error:  # a bug here must not stop the poll loop
                log.error("refresh failed unexpectedly", exc_info=error)
                self._set(self._health.model_state, self._health.database, error=_safe(error))
            return self._health

    def _set(
        self,
        model_state: str,
        database: str,
        *,
        desired: int | None = None,
        error: str | None = None,
    ) -> None:
        model = self._model
        self._health = Health(
            model_state,
            database,
            desired,
            None if model is None else model.deployment.id,
            error,
            datetime.now(UTC),
        )

    def _refresh(self, force: bool) -> None:
        try:
            state = migrate.status(self.engine)
            if not state.up_to_date:
                self._model = None
                self._set(
                    SCHEMA_NOT_CURRENT,
                    "ok",
                    error=f"the database is at revision {state.current or 'none'}, "
                    f"the code expects {state.head}: run `uv run python scripts/db.py upgrade`",
                )
                return
            active = self.repository.active()
        except SQLAlchemyError as error:
            # Keep serving the last verified deployment; say that the database is unreachable.
            log.warning("database unreachable: %s", type(error).__name__)
            self._set(
                self._health.model_state,
                "unavailable",
                desired=self._health.desired_deployment_id,
                error=f"the database is unreachable ({type(error).__name__})",
            )
            return
        if active is None:
            self._model = None
            self._set(NO_DEPLOYMENT, "ok", error="no model is deployed")
            return
        current = self._model
        if current is not None and current.deployment.id == active.id:
            # Same deployment: only its registry status can have moved. No reload.
            self._model = replace(current, deployment=active)
            self._set(READY, "ok", desired=active.id)
            return
        if not force and self._failed and self._failed[0] == active.id:
            if self._clock() - self._failed[1] < self.retry_seconds:
                self._model = None
                self._set(LOAD_FAILED, "ok", desired=active.id, error=self._health.error)
                return
        try:
            loaded = self._load(active)
        except LoadFailure as error:
            self._model = None
            self._failed = (active.id, self._clock())
            self._set(LOAD_FAILED, "ok", desired=active.id, error=str(error))
            return
        self._failed = None
        self._model = loaded
        self._set(READY, "ok", desired=active.id)

    def _load(self, row: DeploymentRow) -> ServingModel:
        try:
            package = load_package(self.models_root, row.package_id)
        except PackageError as error:
            raise LoadFailure(_clean(str(error), self.models_root)) from None
        except Exception as error:
            log.error("loading %s failed", row.package_id, exc_info=error)
            raise LoadFailure(f"the model could not be loaded ({type(error).__name__})") from None
        problems: list[str] = []
        if package.package_id != row.package_id:
            problems.append("the package's id differs from the deployment's")
        if str(package.manifest.get("selection_sha256")) != row.selection_sha256:
            problems.append("the package's frozen selection differs from the registry's record")
        if str(package.manifest["task"]["task"]) != row.task:
            problems.append("the package's task differs from the registry's record")
        classes = package.classes
        if classes is not None and [str(c) for c in package.pipeline.classes_] != [
            str(c) for c in classes
        ]:
            problems.append("the model's class order differs from the manifest's")
        try:
            problems.extend(loaded_parity_problems(package))
        except Exception as error:
            log.error("parity check of %s failed", row.package_id, exc_info=error)
            problems.append(f"the parity check could not run ({type(error).__name__})")
        if problems:
            raise LoadFailure(_clean("; ".join(problems), self.models_root))
        return ServingModel(row, package, _columns(package), datetime.now(UTC))

    # --------------------------------------------------------------- prediction

    def predict(self, rows: list[dict[str, Any]], *, simulated: bool = False) -> PredictionBatch:
        """Validate, predict and log. Raises `NotReady` or `InvalidInput`."""
        model = self.model()
        frame = prepare_rows(list(model.columns), rows)  # InvalidInput
        started = time.perf_counter()
        package = model.package
        labels = package.predict(frame)
        probabilities = package.predict_proba(frame) if package.classes is not None else None
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        items: list[Served] = []
        for index in range(len(frame)):
            if package.classes is not None:
                assert probabilities is not None
                items.append(
                    Served(
                        str(uuid.uuid4()),
                        str(labels[index]),
                        {
                            str(name): float(p)
                            for name, p in zip(package.classes, probabilities[index], strict=True)
                        },
                    )
                )
            else:
                value = float(labels[index])
                if not np.isfinite(value):
                    raise PredictionFailed("the model produced a value that is not finite")
                items.append(Served(str(uuid.uuid4()), value, None))
        logged = self._log(model.deployment, rows, items, latency_ms, simulated)
        return PredictionBatch(model.deployment, items, latency_ms, logged)

    def _log(
        self,
        deployment: DeploymentRow,
        rows: list[dict[str, Any]],
        items: list[Served],
        latency_ms: float,
        simulated: bool,
    ) -> bool:
        try:
            records = [
                {
                    "id": item.prediction_id,
                    "d": deployment.id,
                    "v": deployment.version,
                    "i": json.dumps(row, allow_nan=False),
                    "o": json.dumps(
                        {"prediction": item.prediction, "probabilities": item.probabilities},
                        allow_nan=False,
                    ),
                    "l": latency_ms,
                    "s": simulated,
                }
                for row, item in zip(rows, items, strict=True)
            ]
            with self.engine.begin() as connection:
                connection.execute(text(f"SET LOCAL statement_timeout = {LOG_TIMEOUT_MS}"))
                connection.execute(
                    text(
                        "INSERT INTO predictions (id, deployment_id, model_version, inputs, "
                        "output, latency_ms, is_simulated) VALUES (:id, :d, :v, "
                        "CAST(:i AS jsonb), CAST(:o AS jsonb), :l, :s)"
                    ),
                    records,
                )
            return True
        except Exception as error:  # logging must never fail a prediction
            log.warning("prediction log failed: %s", type(error).__name__)
            return False


class PredictionFailed(Exception):
    """The model ran but produced an unusable result."""


def _not_ready(health: Health) -> NotReady:
    if health.model_state == NO_DEPLOYMENT:
        return NotReady("no_deployment", "No model is deployed.")
    return NotReady(
        "model_not_ready",
        f"The deployed model is not being served ({health.model_state}): "
        f"{health.error or 'it has not been loaded yet'}",
    )


def _clean(message: str, models_root: Path) -> str:
    """No filesystem path in anything shown to a caller."""
    return message.replace(str(models_root), "<models>")


def _safe(error: BaseException) -> str:
    return f"unexpected error ({type(error).__name__})"


__all__ = [
    "LOAD_FAILED",
    "NO_DEPLOYMENT",
    "READY",
    "SCHEMA_NOT_CURRENT",
    "STARTING",
    "Health",
    "InvalidInput",
    "LoadFailure",
    "NotReady",
    "PredictionBatch",
    "PredictionFailed",
    "Served",
    "ServingModel",
    "ServingRuntime",
]
