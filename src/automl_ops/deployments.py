"""Deployments in PostgreSQL (P4-7, BUILD_PLAN Step 22).

A **deployment** is one period in which one registered model version was the live model. PostgreSQL
is authoritative: the serving app reads the one `active` row and serves exactly that version, and
the control plane only ever changes deployments through this repository. MLflow is not involved.

**What can be deployed.** A version registered in the registry whose status is `staging` or
`production` (not `development`, not `archived`) and whose sealed package verifies *deeply*
(seal hashes, read-only files, library versions, a reload with prediction parity) at the moment of
the change. The registry's own rules are untouched: archiving a deployed staging version, or a
rollback that archives a deployed production version, is still allowed; the deployment keeps
serving until someone switches or undeploys it, and its `registry_status` shows the difference.

**Changes** (one transaction each, serialised by an advisory lock, so exactly one wins a race):

- `deploy`: nothing is live; the version becomes live.
- `switch`: the active deployment ends and a new one for another version becomes active.
- `rollback`: undoes the last *switch*, only: it makes the version that the active deployment
  replaced live again. A deployment made by a rollback (or a first deploy) has nothing to roll back
  to, so a rollback never undoes a rollback; switch explicitly instead.
- `undeploy`: the active deployment ends; nothing is live.

Rows are never deleted, never truncated, and an ended row never changes; the database enforces
this (migration 0005) and also refuses to deploy a version that is not staging or production.
Nothing here reads dataset rows or the held-out test part.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

from automl_ops.db.models import DEPLOYABLE_STATES
from automl_ops.package import verify_package
from automl_ops.registry import ModelNotFound, PackageUnverified, VersionNotFound

LOCK_CLASS = 7267  # advisory lock class for every deployment change (the mirror uses 7266)
REASON_MAX = 500
PAGE = 100


class DeploymentError(Exception):
    """Base class; `code` is the stable machine-readable name used by the API."""

    code = "deployment_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class DeploymentNotFound(DeploymentError):
    code = "deployment_not_found"


class InvalidDeploymentReason(DeploymentError):
    code = "validation_error"


class NotDeployable(DeploymentError):
    code = "version_not_deployable"


class DeploymentActive(DeploymentError):
    code = "deployment_active"


class NotActive(DeploymentError):
    code = "deployment_not_active"


class AlreadyDeployed(DeploymentError):
    code = "already_deployed"


class NoPreviousDeployment(DeploymentError):
    code = "no_previous_deployment"


class DeploymentRefused(DeploymentError):
    code = "deployment_refused"


@dataclass(frozen=True)
class DeploymentRow:
    id: int
    model: str
    version: int
    package_id: str
    status: str
    action: str
    reason: str
    replaced_id: int | None
    deployed_at: datetime
    ended_at: datetime | None
    end_reason: str | None
    registry_status: str  # the version's registry status *now*
    task: str
    target: str
    algorithm: str
    selection_sha256: str


_SELECT = """
    SELECT d.id, m.name AS model, v.version, d.package_id, d.status, d.action, d.reason,
           d.replaced_id, d.deployed_at, d.ended_at, d.end_reason, v.status AS registry_status,
           v.task, v.target, v.algorithm, v.selection_sha256
    FROM deployments d
    JOIN model_versions v ON v.id = d.version_id
    JOIN registered_models m ON m.id = d.model_id
"""


def _row(row: Any) -> DeploymentRow:
    return DeploymentRow(**dict(row._mapping))


def _reason(reason: str) -> str:
    cleaned = reason.strip()
    if not 1 <= len(cleaned) <= REASON_MAX:
        raise InvalidDeploymentReason(f"the reason must be 1 to {REASON_MAX} characters")
    return cleaned


@dataclass(frozen=True)
class _Target:
    version_id: int
    model_id: int
    model: str
    version: int
    package_id: str
    status: str


class DeploymentRepository:
    """All reads and writes of the deployment tables."""

    def __init__(self, engine: Engine, models_root: Path) -> None:
        self.engine = engine
        self.models_root = models_root

    # ------------------------------------------------------------------ reads

    def get(self, deployment_id: int) -> DeploymentRow:
        with self.engine.connect() as connection:
            row = connection.execute(
                text(_SELECT + " WHERE d.id = :i"), {"i": deployment_id}
            ).first()
        if row is None:
            raise DeploymentNotFound(f"no deployment {deployment_id}")
        return _row(row)

    def active(self) -> DeploymentRow | None:
        with self.engine.connect() as connection:
            return self.active_in(connection)

    @staticmethod
    def active_in(connection: Connection) -> DeploymentRow | None:
        row = connection.execute(text(_SELECT + " WHERE d.status = 'active'")).first()
        return None if row is None else _row(row)

    def list(
        self, status: str | None = None, before_id: int | None = None, limit: int = PAGE
    ) -> list[DeploymentRow]:
        """Newest first. `before_id` continues after the last id of the previous page."""
        clauses: list[str] = []
        params: dict[str, Any] = {"limit": min(max(limit, 1), PAGE)}
        if status is not None:
            clauses.append("d.status = :s")
            params["s"] = status
        if before_id is not None:
            clauses.append("d.id < :b")
            params["b"] = before_id
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self.engine.connect() as connection:
            rows = connection.execute(
                text(_SELECT + where + " ORDER BY d.id DESC LIMIT :limit"), params
            ).all()
        return [_row(r) for r in rows]

    # ------------------------------------------------------------------ changes

    def deploy(self, model: str, number: int, reason: str) -> DeploymentRow:
        return self._activate("deploy", model, number, _reason(reason), expect_active=None)

    def switch(self, deployment_id: int, model: str, number: int, reason: str) -> DeploymentRow:
        return self._activate("switch", model, number, _reason(reason), expect_active=deployment_id)

    def rollback(self, reason: str) -> DeploymentRow:
        """Make the version the active deployment replaced live again (undoes one switch)."""
        note = _reason(reason)
        current = self.active()
        if current is None:
            raise NotActive("nothing is deployed, so there is nothing to roll back")
        if current.action != "switch" or current.replaced_id is None:
            raise NoPreviousDeployment(
                f"deployment {current.id} was made by a {current.action}; only a switch can be "
                "rolled back (switch to the version you want instead)",
                {"deployment_id": current.id, "action": current.action},
            )
        previous = self.get(current.replaced_id)
        return self._activate(
            "rollback", previous.model, previous.version, note, expect_active=current.id
        )

    def undeploy(self, deployment_id: int, reason: str) -> DeploymentRow:
        note = _reason(reason)
        try:
            with self.engine.begin() as connection:
                _lock(connection)
                current = _lock_active(connection)
                if current is None or current != deployment_id:
                    raise NotActive(
                        f"deployment {deployment_id} is not the active deployment",
                        {"active_deployment_id": current},
                    )
                _end(connection, deployment_id, f"undeployed: {note}")
        except IntegrityError:
            raise DeploymentRefused(
                "the database refused this change; nothing was changed"
            ) from None
        return self.get(deployment_id)

    # ---------------------------------------------------------------- internals

    def _target(self, model: str, number: int) -> _Target:
        with self.engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT v.id, v.model_id, v.version, v.package_id, v.status "
                    "FROM model_versions v JOIN registered_models m ON m.id = v.model_id "
                    "WHERE m.name = :n AND v.version = :v"
                ),
                {"n": model, "v": number},
            ).first()
            if row is None:
                known = connection.execute(
                    text("SELECT 1 FROM registered_models WHERE name = :n"), {"n": model}
                ).first()
                if known is None:
                    raise ModelNotFound(f"no model named {model}")
                raise VersionNotFound(f"model {model} has no version {number}")
        return _Target(row.id, row.model_id, model, row.version, row.package_id, row.status)

    def _check_deployable(self, target: _Target) -> None:
        if target.status not in DEPLOYABLE_STATES:
            raise NotDeployable(
                f"{target.model} version {target.version} is {target.status}; only "
                f"{' or '.join(DEPLOYABLE_STATES)} versions can be deployed",
                {"status": target.status},
            )

    def _activate(
        self, action: str, model: str, number: int, reason: str, *, expect_active: int | None
    ) -> DeploymentRow:
        target = self._target(model, number)
        self._check_deployable(target)
        # Deep verification (a reload and a prediction-parity check) runs before any lock is
        # taken; sealed packages are read-only, and the transaction re-checks the version row.
        check = verify_package(self.models_root, target.package_id, deep=True)
        if not check.ok:
            raise PackageUnverified(
                f"package {target.package_id} no longer verifies ({check.state}); "
                "nothing was changed",
                {
                    "state": check.state,
                    "problems": [
                        p.replace(str(self.models_root), "<models>") for p in check.problems
                    ],
                },
            )
        try:
            with self.engine.begin() as connection:
                _lock(connection)
                version = connection.execute(
                    text("SELECT status, package_id FROM model_versions WHERE id = :i FOR SHARE"),
                    {"i": target.version_id},
                ).one()
                if version.package_id != target.package_id:
                    raise DeploymentRefused("the model version changed; nothing was changed")
                self._check_deployable(
                    _Target(
                        target.version_id,
                        target.model_id,
                        model,
                        number,
                        target.package_id,
                        version.status,
                    )
                )
                current = _lock_active(connection)
                replaced: int | None = None
                if expect_active is None:
                    if current is not None:
                        raise DeploymentActive(
                            f"deployment {current} is active; switch it instead",
                            {"active_deployment_id": current},
                        )
                else:
                    if current is None or current != expect_active:
                        raise NotActive(
                            f"deployment {expect_active} is not the active deployment",
                            {"active_deployment_id": current},
                        )
                    same = connection.execute(
                        text("SELECT version_id FROM deployments WHERE id = :i"), {"i": current}
                    ).scalar_one()
                    if same == target.version_id:
                        raise AlreadyDeployed(
                            f"{model} version {number} is already the active deployment",
                            {"deployment_id": current},
                        )
                    _end(connection, current, f"{action}: replaced by {model} version {number}")
                    replaced = current
                new_id = connection.execute(
                    text(
                        "INSERT INTO deployments (model_id, version_id, package_id, action, "
                        "reason, replaced_id) VALUES (:m, :v, :p, :a, :r, :x) RETURNING id"
                    ),
                    {
                        "m": target.model_id,
                        "v": target.version_id,
                        "p": target.package_id,
                        "a": action,
                        "r": reason,
                        "x": replaced,
                    },
                ).scalar_one()
        except IntegrityError:
            raise DeploymentRefused(
                "the database refused this change; nothing was changed"
            ) from None
        return self.get(int(new_id))


def _lock(connection: Connection) -> None:
    connection.execute(text("SELECT pg_advisory_xact_lock(:c, 0)"), {"c": LOCK_CLASS})


def _lock_active(connection: Connection) -> int | None:
    row = connection.execute(
        text("SELECT id FROM deployments WHERE status = 'active' FOR UPDATE")
    ).first()
    return None if row is None else int(row.id)


def _end(connection: Connection, deployment_id: int, why: str) -> None:
    connection.execute(
        text(
            "UPDATE deployments SET status = 'ended', ended_at = now(), end_reason = :r "
            "WHERE id = :i AND status = 'active'"
        ),
        {"i": deployment_id, "r": why[:REASON_MAX]},
    )
