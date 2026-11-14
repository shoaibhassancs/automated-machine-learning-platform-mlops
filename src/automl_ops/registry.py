"""The model registry in PostgreSQL (P4-5, BUILD_PLAN Step 20, first part).

A registered *model* is a name; each *version* is one sealed model package. A version has a status:
`development` (new), `staging`, `production` or `archived`. PostgreSQL is authoritative for status
and history; MLflow is not involved yet (the mirror is P4-6).

**Integrity.** A status can change only by inserting a `registry_events` row: a trigger (migration
0003) checks the move, allocates the event's `seq` under a per-model row lock, and applies the new
status in the same statement. The repository checks the same rules first to give clear errors, and
holds the model's row lock for the whole change, but the database refuses a bad move even from a
hand-written statement. Every change and its event commit together or not at all.

**Moves.** development -> staging -> production, and archived -> staging (never straight to
production). A promotion to production first archives the current production version
(`supersede`), so there is exactly one at most. `rollback` is the one exception that returns an
archived version straight to production: it archives the current production version and restores
the version it replaced, as two events in one transaction. The previous production version is read
from the event history; if it cannot be established the call fails and nothing changes.

**Test-set isolation.** Nothing here reads dataset rows or the held-out test part. Registration
reads only the sealed package's own `manifest.json` and `test_result.json` (after the package has
been verified), and keeps a bounded snapshot of the stored evaluation; curves and residual points
are left in the package and listed in `omitted`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import Engine, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import DBAPIError, IntegrityError

from automl_ops.db.models import NAME_REGEX
from automl_ops.package import (
    MANIFEST_FILE,
    TEST_RESULT_FILE,
    PackageError,
    package_path,
    verify_package,
)

EVENT_PAGE = 200
REASON_MAX = 500
BULKY = ("roc_curves", "pr_curves", "residual_points")  # stay in the package, listed as omitted
_NAME = re.compile(NAME_REGEX)

Status = Literal["development", "staging", "production", "archived"]
LEGAL_PROMOTIONS = {
    ("development", "staging"),
    ("staging", "production"),
    ("archived", "staging"),
}


class RegistryError(Exception):
    """Base class; `code` is the stable machine-readable name used by the API."""

    code = "registry_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ModelNotFound(RegistryError):
    code = "model_not_found"


class VersionNotFound(RegistryError):
    code = "version_not_found"


class JobNotFoundForRegistry(RegistryError):
    code = "job_not_found"


class PackageRequired(RegistryError):
    code = "package_required"


class ProvenanceMismatch(RegistryError):
    code = "provenance_mismatch"


class JobNotRegisterable(RegistryError):
    code = "job_not_registerable"


class PackageNotRegisterable(RegistryError):
    code = "package_not_registerable"


class PackageAlreadyRegistered(RegistryError):
    code = "package_already_registered"


class InvalidMove(RegistryError):
    code = "invalid_transition"


class NoPreviousProduction(RegistryError):
    code = "no_previous_production"


class PackageUnverified(RegistryError):
    code = "package_unverified"


@dataclass(frozen=True)
class VersionRow:
    model: str
    version: int
    status: str
    package_id: str
    job_id: int | None
    dataset: str | None
    registered_at: datetime
    status_changed_at: datetime
    algorithm: str
    task: str
    target: str
    selection_sha256: str
    evaluation: dict[str, Any] | None


@dataclass(frozen=True)
class ModelRow:
    name: str
    created_at: datetime
    versions: int
    production_version: int | None
    latest_version: int | None


@dataclass(frozen=True)
class EventRow:
    seq: int
    version: int
    action: str
    from_status: str | None
    to_status: str
    reason: str
    created_at: datetime


@dataclass(frozen=True)
class RollbackOutcome:
    restored: VersionRow
    archived: VersionRow


def valid_name(name: str) -> bool:
    return bool(_NAME.match(name))


def evaluation_snapshot(test_result: dict[str, Any]) -> dict[str, Any]:
    """The bounded, stored evaluation summary: everything in the sealed test result except the
    bulky per-point curve and residual data, which are named in `omitted` (never cut silently)."""
    evaluation = test_result.get("evaluation", {})
    kept = {k: v for k, v in evaluation.items() if k not in BULKY}
    return {
        "algorithm": test_result.get("algorithm"),
        "selection_metric": test_result.get("selection_metric"),
        "dataset": test_result.get("dataset"),
        "test_split": test_result.get("test_split"),
        "positive_class": test_result.get("positive_class"),
        "selection_sha256": test_result.get("selection_sha256"),
        "limitation": test_result.get("limitation"),
        "evaluation": kept,
        "omitted": [k for k in BULKY if k in evaluation],
    }


def _problems(models_root: Path, problems: tuple[str, ...]) -> list[str]:
    """Package problems for a response: the models folder's real path is never shown."""
    return [p.replace(str(models_root), "<models>") for p in problems]


def read_package_evidence(models_root: Path, package_id: str) -> dict[str, Any]:
    """Verify a package deeply, then read its stored manifest and test result (no data rows)."""
    check = verify_package(models_root, package_id, deep=True)
    if not check.ok:
        raise PackageNotRegisterable(
            f"package {package_id} cannot be registered ({check.state})",
            {"state": check.state, "problems": _problems(models_root, check.problems)},
        )
    try:
        path = package_path(models_root, package_id)
        manifest = json.loads((path / MANIFEST_FILE).read_text(encoding="utf-8"))
        result = json.loads((path / TEST_RESULT_FILE).read_text(encoding="utf-8"))
    except (PackageError, OSError, ValueError) as error:
        raise PackageNotRegisterable(
            f"package {package_id} could not be read: {type(error).__name__}"
        ) from None
    return {
        "algorithm": str(manifest["model"]["algorithm"]),
        "task": str(manifest["task"]["task"]),
        "target": str(manifest["task"]["target"]),
        "selection_sha256": str(manifest["selection_sha256"]),
        "dataset": manifest.get("dataset") or {},
        "evaluation": evaluation_snapshot(result),
    }


_VERSION_SELECT = """
    SELECT m.name AS model, v.version, v.status, v.package_id, v.job_id,
           CASE WHEN dv.id IS NULL THEN NULL ELSE d.name || '@v' || dv.number END AS dataset,
           v.registered_at, v.status_changed_at, v.algorithm, v.task, v.target,
           v.selection_sha256, {evaluation} AS evaluation
    FROM model_versions v
    JOIN registered_models m ON m.id = v.model_id
    LEFT JOIN dataset_versions dv ON dv.id = v.dataset_version_id
    LEFT JOIN datasets d ON d.id = dv.dataset_id
"""


def _select(with_evaluation: bool) -> str:
    return _VERSION_SELECT.format(evaluation="v.evaluation" if with_evaluation else "NULL")


def _version(row: Any) -> VersionRow:
    return VersionRow(**dict(row._mapping))


def _reason(reason: str) -> str:
    cleaned = reason.strip()
    if not 1 <= len(cleaned) <= REASON_MAX:
        raise InvalidMove(f"the reason must be 1 to {REASON_MAX} characters")
    return cleaned


class RegistryRepository:
    """All reads and writes of the registry tables."""

    def __init__(self, engine: Engine, models_root: Path) -> None:
        self.engine = engine
        self.models_root = models_root

    # ------------------------------------------------------------------ reads

    def list_models(self) -> list[ModelRow]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT m.name, m.created_at, count(v.id) AS versions, "
                    "max(v.version) FILTER (WHERE v.status = 'production') AS production_version, "
                    "max(v.version) AS latest_version "
                    "FROM registered_models m LEFT JOIN model_versions v ON v.model_id = m.id "
                    "GROUP BY m.id ORDER BY m.name"
                )
            ).all()
        return [ModelRow(**dict(r._mapping)) for r in rows]

    def get_model(self, name: str) -> ModelRow:
        for model in self.list_models():
            if model.name == name:
                return model
        raise ModelNotFound(f"no model named {name}")

    def list_versions(self, name: str) -> list[VersionRow]:
        self.get_model(name)
        with self.engine.connect() as connection:
            rows = connection.execute(
                text(_select(False) + " WHERE m.name = :n ORDER BY v.version DESC"), {"n": name}
            ).all()
        return [_version(r) for r in rows]

    def get_version(self, name: str, number: int) -> VersionRow:
        with self.engine.connect() as connection:
            row = connection.execute(
                text(_select(True) + " WHERE m.name = :n AND v.version = :v"),
                {"n": name, "v": number},
            ).first()
        if row is None:
            self.get_model(name)
            raise VersionNotFound(f"model {name} has no version {number}")
        return _version(row)

    def events(self, name: str, after_seq: int = 0, limit: int = EVENT_PAGE) -> list[EventRow]:
        self.get_model(name)
        with self.engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT e.seq, v.version, e.action, e.from_status, e.to_status, e.reason, "
                    "e.created_at FROM registry_events e "
                    "JOIN model_versions v ON v.id = e.version_id "
                    "JOIN registered_models m ON m.id = e.model_id "
                    "WHERE m.name = :n AND e.seq > :after ORDER BY e.seq LIMIT :limit"
                ),
                {"n": name, "after": after_seq, "limit": limit},
            ).all()
        return [EventRow(**dict(r._mapping)) for r in rows]

    # -------------------------------------------------------------- registration

    def register(
        self,
        name: str,
        *,
        package_id: str | None,
        job_id: int | None,
        reason: str | None = None,
    ) -> VersionRow:
        """Register a sealed package as the next version of `name` (created if new)."""
        if not valid_name(name):
            raise InvalidMove("the model name must be 1-64 characters of a-z, 0-9, '_' and '-'")
        note = _reason(reason) if reason else "registered"
        job = self._job_for_registration(job_id) if job_id is not None else None
        if job is not None:
            if package_id is not None and package_id != job["package_id"]:
                raise ProvenanceMismatch(
                    f"job {job_id} produced package {job['package_id']}, not {package_id}"
                )
            package_id = job["package_id"]
        if package_id is None:
            raise PackageRequired("give a package_id, a job_id, or both")
        evidence = read_package_evidence(self.models_root, package_id)  # deep check, no DB lock
        try:
            with self.engine.begin() as connection:
                if job is None:
                    job = self._job_by_package(connection, package_id)
                dataset_version_id = (
                    job["dataset_version_id"]
                    if job is not None
                    else self._dataset_version(connection, evidence["dataset"])
                )
                number = self._insert_version(
                    connection,
                    name,
                    package_id,
                    None if job is None else job["id"],
                    dataset_version_id,
                    evidence,
                    note,
                )
        except IntegrityError:
            where = self._registered_as(package_id)
            if where is not None:
                raise PackageAlreadyRegistered(
                    f"package {package_id} is already registered as {where[0]} version {where[1]}",
                    {"model": where[0], "version": where[1]},
                ) from None
            raise InvalidMove("the database refused this registration") from None
        return self.get_version(name, number)

    def _job_for_registration(self, job_id: int) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT id, kind, status, package_id, dataset_version_id "
                    "FROM training_jobs WHERE id = :id"
                ),
                {"id": job_id},
            ).first()
        if row is None:
            raise JobNotFoundForRegistry(f"no job {job_id}")
        if row.status != "succeeded":
            raise JobNotRegisterable(f"job {job_id} is {row.status}, not succeeded")
        if row.package_id is None:
            raise JobNotRegisterable(
                f"job {job_id} ({row.kind}) has no sealed package; only a final job makes one"
            )
        return dict(row._mapping)

    @staticmethod
    def _job_by_package(connection: Connection, package_id: str) -> dict[str, Any] | None:
        row = connection.execute(
            text(
                "SELECT id, dataset_version_id FROM training_jobs "
                "WHERE package_id = :p AND status = 'succeeded' ORDER BY id LIMIT 1"
            ),
            {"p": package_id},
        ).first()
        return None if row is None else dict(row._mapping)

    @staticmethod
    def _dataset_version(connection: Connection, dataset: dict[str, Any]) -> int | None:
        """The catalog row for the package's dataset, if the catalog has exactly that version."""
        name, number, sha = dataset.get("name"), dataset.get("version"), dataset.get("sha256")
        if not isinstance(name, str) or not isinstance(number, int) or not isinstance(sha, str):
            return None
        row = connection.execute(
            text(
                "SELECT v.id FROM dataset_versions v JOIN datasets d ON d.id = v.dataset_id "
                "WHERE d.name = :n AND v.number = :v AND v.sha256 = :s"
            ),
            {"n": name, "v": number, "s": sha},
        ).first()
        return None if row is None else int(row.id)

    @staticmethod
    def _insert_version(
        connection: Connection,
        name: str,
        package_id: str,
        job_id: int | None,
        dataset_version_id: int | None,
        evidence: dict[str, Any],
        reason: str,
    ) -> int:
        connection.execute(
            text("INSERT INTO registered_models (name) VALUES (:n) ON CONFLICT (name) DO NOTHING"),
            {"n": name},
        )
        model_id = connection.execute(
            text("SELECT id FROM registered_models WHERE name = :n FOR UPDATE"), {"n": name}
        ).scalar_one()
        number = connection.execute(
            text("SELECT COALESCE(max(version), 0) + 1 FROM model_versions WHERE model_id = :m"),
            {"m": model_id},
        ).scalar_one()
        version_id = connection.execute(
            text(
                "INSERT INTO model_versions (model_id, version, package_id, job_id, "
                "dataset_version_id, algorithm, task, target, selection_sha256, evaluation) "
                "VALUES (:m, :v, :p, :j, :d, :a, :t, :g, :s, CAST(:e AS jsonb)) RETURNING id"
            ),
            {
                "m": model_id,
                "v": number,
                "p": package_id,
                "j": job_id,
                "d": dataset_version_id,
                "a": evidence["algorithm"],
                "t": evidence["task"],
                "g": evidence["target"],
                "s": evidence["selection_sha256"],
                "e": json.dumps(evidence["evaluation"], allow_nan=False, sort_keys=True),
            },
        ).scalar_one()
        _event(connection, model_id, version_id, "register", None, "development", reason)
        return int(number)

    def _registered_as(self, package_id: str) -> tuple[str, int] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT m.name, v.version FROM model_versions v "
                    "JOIN registered_models m ON m.id = v.model_id WHERE v.package_id = :p"
                ),
                {"p": package_id},
            ).first()
        return None if row is None else (str(row.name), int(row.version))

    # ------------------------------------------------------------------ changes

    def promote(
        self, name: str, number: int, to: Literal["staging", "production"], reason: str
    ) -> VersionRow:
        note = _reason(reason)
        try:
            with self.engine.begin() as connection:
                model_id = _lock_model(connection, name)
                version = _lock_version(connection, model_id, name, number)
                if (version.status, to) not in LEGAL_PROMOTIONS:
                    raise InvalidMove(
                        f"version {number} is {version.status}; it cannot be promoted to {to}"
                        + (
                            " (use a rollback to restore an archived version)"
                            if to == "production"
                            else ""
                        )
                    )
                self._require_sealed(version.package_id)
                if to == "production":
                    current = connection.execute(
                        text(
                            "SELECT id, version FROM model_versions "
                            "WHERE model_id = :m AND status = 'production' FOR UPDATE"
                        ),
                        {"m": model_id},
                    ).first()
                    if current is not None:
                        _event(
                            connection,
                            model_id,
                            current.id,
                            "supersede",
                            "production",
                            "archived",
                            f"superseded by version {number}",
                        )
                _event(connection, model_id, version.id, "promote", version.status, to, note)
        except IntegrityError as error:
            raise _refused(error) from None
        return self.get_version(name, number)

    def archive(self, name: str, number: int, reason: str) -> VersionRow:
        note = _reason(reason)
        try:
            with self.engine.begin() as connection:
                model_id = _lock_model(connection, name)
                version = _lock_version(connection, model_id, name, number)
                if version.status == "production":
                    raise InvalidMove(
                        f"version {number} is in production; promote another version or roll "
                        "back instead of archiving it"
                    )
                if version.status == "archived":
                    raise InvalidMove(f"version {number} is already archived")
                _event(
                    connection, model_id, version.id, "archive", version.status, "archived", note
                )
        except IntegrityError as error:
            raise _refused(error) from None
        return self.get_version(name, number)

    def rollback(self, name: str, reason: str) -> RollbackOutcome:
        """Archive the current production version and restore the one it replaced, atomically."""
        note = _reason(reason)
        try:
            with self.engine.begin() as connection:
                model_id = _lock_model(connection, name)
                current = connection.execute(
                    text(
                        "SELECT id, version FROM model_versions "
                        "WHERE model_id = :m AND status = 'production' FOR UPDATE"
                    ),
                    {"m": model_id},
                ).first()
                if current is None:
                    raise NoPreviousProduction(f"model {name} has no production version")
                target = self._previous_production(connection, model_id, current)
                self._require_sealed(target.package_id)
                _event(connection, model_id, current.id, "rollback", "production", "archived", note)
                _event(connection, model_id, target.id, "rollback", "archived", "production", note)
                archived_number, restored_number = int(current.version), target.version
        except IntegrityError as error:
            raise _refused(error) from None
        return RollbackOutcome(
            restored=self.get_version(name, restored_number),
            archived=self.get_version(name, archived_number),
        )

    @staticmethod
    def _previous_production(connection: Connection, model_id: int, current: Any) -> _Locked:
        """The version the current production version replaced, from the event history only."""
        made = connection.execute(
            text(
                "SELECT seq, action FROM registry_events WHERE version_id = :v "
                "AND to_status = 'production' ORDER BY seq DESC LIMIT 1"
            ),
            {"v": current.id},
        ).first()
        if made is None or made.action != "promote":
            raise NoPreviousProduction(
                f"version {current.version} was not promoted over an earlier production version "
                "(it may already be a rollback); promote a version explicitly instead"
            )
        before = connection.execute(
            text("SELECT version_id, action FROM registry_events WHERE model_id = :m AND seq = :s"),
            {"m": model_id, "s": made.seq - 1},
        ).first()
        if before is None or before.action != "supersede" or before.version_id == current.id:
            raise NoPreviousProduction(
                f"version {current.version} replaced no earlier production version"
            )
        target = connection.execute(
            text(
                "SELECT id, version, status, package_id FROM model_versions "
                "WHERE id = :v FOR UPDATE"
            ),
            {"v": before.version_id},
        ).first()
        if target is None or target.status != "archived":
            raise NoPreviousProduction(
                "the version that production replaced is no longer archived; nothing was changed"
            )
        return _Locked(target.id, target.version, target.status, target.package_id)

    def _require_sealed(self, package_id: str) -> None:
        check = verify_package(self.models_root, package_id, deep=False)
        if not check.ok:
            raise PackageUnverified(
                f"package {package_id} no longer verifies ({check.state}); nothing was changed",
                {"state": check.state, "problems": _problems(self.models_root, check.problems)},
            )


@dataclass(frozen=True)
class _Locked:
    id: int
    version: int
    status: str
    package_id: str


def _lock_model(connection: Connection, name: str) -> int:
    row = connection.execute(
        text("SELECT id FROM registered_models WHERE name = :n FOR UPDATE"), {"n": name}
    ).first()
    if row is None:
        raise ModelNotFound(f"no model named {name}")
    return int(row.id)


def _lock_version(connection: Connection, model_id: int, name: str, number: int) -> _Locked:
    row = connection.execute(
        text(
            "SELECT id, version, status, package_id FROM model_versions "
            "WHERE model_id = :m AND version = :v FOR UPDATE"
        ),
        {"m": model_id, "v": number},
    ).first()
    if row is None:
        raise VersionNotFound(f"model {name} has no version {number}")
    return _Locked(row.id, row.version, row.status, row.package_id)


def _event(
    connection: Connection,
    model_id: int,
    version_id: int,
    action: str,
    from_status: str | None,
    to_status: str,
    reason: str,
) -> None:
    """Insert one event; the database trigger checks the move, numbers it and applies it."""
    connection.execute(
        text(
            "INSERT INTO registry_events (model_id, version_id, seq, action, from_status, "
            "to_status, reason) VALUES (:m, :v, 1, :a, :f, :t, :r)"
        ),
        {
            "m": model_id,
            "v": version_id,
            "a": action,
            "f": from_status,
            "t": to_status,
            "r": reason,
        },
    )


def _refused(error: DBAPIError) -> InvalidMove:
    """The database refused a change the repository's own checks did not catch."""
    return InvalidMove("the database refused this change; nothing was changed")
