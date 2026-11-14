"""The registry's MLflow mirror (P4-6, BUILD_PLAN Step 20, second part).

**PostgreSQL is the sole authority.** This module reads the registry (`registered_models`,
`model_versions`, the event numbers) and makes MLflow's model registry agree with it. It never
reads MLflow to decide registry state, and a failure here can never undo or block a registry
operation: the registry commit has already happened, and the mirror just stays *pending*.

**State-based and idempotent.** A pass snapshots the desired mirror state at event number N (every
version, its status, which one is in production), applies it step by step, reads MLflow back and
compares. Only if the *complete* desired state is verified does `synced_seq` advance to N (together
with the verified state). Events that arrive during the pass have a higher number, so they stay
pending. A failed, interrupted or partly applied pass changes nothing in the progress row.

**Drift is reported, never silently accepted.** Before a sync applies anything it compares MLflow
with what it expects: the last verified state, the state of the last attempted pass (an interrupted
pass may have gotten part of the way) or the new desired state. Anything else (a moved alias, an
edited tag, a deleted version) is *drift*: the sync stops, records it and waits for an explicit
`reconcile` with a reason, which resets the platform-owned parts of MLflow and writes an audit
row. Things the platform does not own (a foreign model of the same name, other aliases, extra
versions, duplicates, a `production` alias held by a foreign version) are *conflicts*: reported,
never changed, and a reconcile that meets a blocking conflict changes nothing at all.

**One syncer per model.** A PostgreSQL session advisory lock (class 7266, key = model id) is held
for the whole pass; a busy lock means another syncer is on it, so this one skips. The lock vanishes
with the connection, so a crashed process never leaves a model stuck.

Opt-in (`AUTOML_REGISTRY_MIRROR=on`) and subject to the same MLflow backend rules as job tracking
(D6): PostgreSQL MLflow, or SQLite only in `AUTOML_JOBS_MLFLOW_DIR`, never the archive.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import Engine, text
from sqlalchemy.engine import Connection

from automl_ops.db.config import setting
from automl_ops.job_tracking import TrackingNotAllowed, job_tracking_store
from automl_ops.tracking import (
    MIRROR_ALIAS,
    MirrorConflict,
    MirrorSnapshot,
    MirrorVersion,
    TrackingError,
    TrackingStore,
    mirror_ensure_model,
    mirror_ensure_version,
    mirror_package_run,
    mirror_set_production_alias,
    mirror_set_tags,
    mirror_snapshot,
    scrub,
)

MIRROR_SETTING = "AUTOML_REGISTRY_MIRROR"
LOCK_CLASS = 7266  # advisory lock classid of the mirror (7265 belongs to jobs)
BACKOFF_BASE_SECONDS = 30.0
BACKOFF_MAX_SECONDS = 900.0
RECHECK_SECONDS = 600.0  # drifted, conflicting and idle models are re-read this often
REASON_MAX = 500
LOG_FINDINGS_MAX = 200
STATUSES = ("development", "staging", "production", "archived")

Category = Literal["pending", "drift", "conflict"]
State = Literal["disabled", "in_sync", "pending", "failed", "drifted", "conflict"]


class MirrorSettingError(ValueError):
    """`AUTOML_REGISTRY_MIRROR` has a value that is neither on nor off."""


class MirrorDisabled(Exception):
    """The mirror is not enabled (`AUTOML_REGISTRY_MIRROR=on`)."""


class MirrorModelNotFound(Exception):
    """The registry has no model with that name."""


class InvalidReason(ValueError):
    """A reconcile needs a reason of 1 to 500 characters."""


def mirror_enabled(env: dict[str, str] | None = None) -> bool:
    value = str(setting(MIRROR_SETTING, "off", env) or "off").strip().lower()
    if value in ("on", "true", "1", "yes"):
        return True
    if value in ("off", "false", "0", "no", ""):
        return False
    raise MirrorSettingError(f"{MIRROR_SETTING} must be on or off, not {value!r}")


# ---------------------------------------------------------------------------------- data shapes


@dataclass(frozen=True)
class DesiredVersion:
    number: int
    package_id: str
    status: str
    selection_sha256: str


@dataclass(frozen=True)
class Desired:
    model_id: int
    name: str
    seq: int  # the registry event number this snapshot corresponds to
    versions: tuple[DesiredVersion, ...]

    @property
    def production(self) -> DesiredVersion | None:
        for v in self.versions:
            if v.status == "production":
                return v
        return None

    def state(self) -> dict[str, Any]:
        production = self.production
        return {
            "versions": {
                str(v.number): {"package_id": v.package_id, "status": v.status}
                for v in self.versions
            },
            "production": None if production is None else production.number,
        }


@dataclass(frozen=True)
class Finding:
    """One difference between the registry (desired) and MLflow (actual).

    `pending`: explained by registry changes not mirrored yet; a sync fixes it. `drift`: a change
    nobody asked for in a platform-owned part of MLflow; only an explicit reconcile repairs it.
    `conflict`: not platform-owned or ambiguous; never changed. A *blocking* conflict also stops a
    sync (it concerns something the mirror needs); the others are only reported."""

    kind: str
    category: Category
    blocking: bool
    version: int | None
    expected: Any
    actual: Any
    message: str

    @property
    def stops_sync(self) -> bool:
        return self.category == "drift" or (self.category == "conflict" and self.blocking)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "category": self.category,
            "version": self.version,
            "expected": self.expected,
            "actual": self.actual,
            "message": self.message,
        }


@dataclass(frozen=True)
class MirrorStatus:
    model: str
    enabled: bool
    state: State
    desired_seq: int
    synced_seq: int
    attempts: int
    last_error: str | None
    last_error_kind: str | None
    last_success_at: datetime | None
    next_attempt_at: datetime | None

    @property
    def pending_events(self) -> int:
        return max(self.desired_seq - self.synced_seq, 0)


@dataclass
class SyncOutcome:
    result: Literal["disabled", "up_to_date", "synced", "failed", "drift", "conflict", "busy"]
    seq: int = 0
    actions: list[dict[str, Any]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    error: str | None = None


@dataclass
class CheckReport:
    model: str
    desired_seq: int
    synced_seq: int
    findings: list[Finding]


@dataclass
class ReconcileOutcome:
    outcome: Literal["ok", "conflict", "failed", "partial", "busy"]
    seq: int
    detected: list[Finding]
    intended: list[str]
    actions: list[dict[str, Any]]
    result: list[Finding]
    error: str | None = None


# --------------------------------------------------------------------------------- the comparison


def classify(
    desired: Desired,
    last_state: dict[str, Any] | None,
    attempt_state: dict[str, Any] | None,
    snapshot: MirrorSnapshot,
) -> list[Finding]:
    """Compare the desired state with MLflow's actual state. Pure: reads nothing, writes nothing."""
    out: list[Finding] = []
    last_versions = (last_state or {}).get("versions", {})
    attempt_versions = (attempt_state or {}).get("versions", {})
    if not snapshot.exists:
        had_before = bool(last_versions)
        out.append(
            Finding(
                "model_missing",
                "drift" if had_before else "pending",
                True,
                None,
                "registered model present",
                "absent",
                "the registered model is not in MLflow"
                + (" (it was there when last synced)" if had_before else ""),
            )
        )
        return out
    if not snapshot.owned:
        return [
            Finding(
                "model_foreign",
                "conflict",
                True,
                None,
                "tag platform.managed=true",
                "not platform-owned",
                "MLflow has a registered model with this name that the platform does not own; "
                "it is never modified",
            )
        ]

    by_package: dict[str, list[MirrorVersion]] = {}
    for v in snapshot.versions:
        if v.package_id is not None:
            by_package.setdefault(v.package_id, []).append(v)
    wanted = {d.package_id for d in desired.versions}
    holder_of = {v.mlflow_version: v for v in snapshot.versions}
    for v in snapshot.versions:
        if v.package_id is None or v.package_id not in wanted:
            out.append(
                Finding(
                    "orphan_version",
                    "conflict",
                    False,
                    None,
                    None,
                    v.mlflow_version,
                    f"MLflow version {v.mlflow_version} is not a registry version; left alone",
                )
            )

    for d in desired.versions:
        found = by_package.get(d.package_id, [])
        known_before = str(d.number) in last_versions or str(d.number) in attempt_versions
        if len(found) > 1:
            out.append(
                Finding(
                    "version_duplicate",
                    "conflict",
                    True,
                    d.number,
                    1,
                    [v.mlflow_version for v in found],
                    f"MLflow has {len(found)} versions for registry version {d.number}; "
                    "none is deleted",
                )
            )
            continue
        if not found:
            was_synced = str(d.number) in last_versions
            out.append(
                Finding(
                    "version_missing",
                    "drift" if was_synced else "pending",
                    True,
                    d.number,
                    "present",
                    "absent",
                    f"registry version {d.number} has no MLflow version"
                    + (" (it was there when last synced)" if was_synced else ""),
                )
            )
            continue
        mv = found[0]
        if not mv.source_ok:
            out.append(
                Finding(
                    "source_missing",
                    "conflict",
                    True,
                    d.number,
                    "package files in the source run",
                    mv.source,
                    f"the source of MLflow version {mv.mlflow_version} has no package files; "
                    "versions and runs are never deleted or edited",
                )
            )
        expected_status_values = {d.status}
        for source in (last_versions, attempt_versions):
            previous = source.get(str(d.number), {}).get("status")
            if previous in STATUSES:
                expected_status_values.add(previous)
        if mv.status != d.status:
            explained = mv.status in expected_status_values or (
                not known_before and mv.status in STATUSES
            )
            out.append(
                Finding(
                    "tag_status",
                    "pending" if explained else "drift",
                    True,
                    d.number,
                    d.status,
                    mv.status,
                    f"tag platform.status of registry version {d.number} is {mv.status!r}, "
                    f"expected {d.status!r}",
                )
            )
        for key, expected, actual in (
            ("registry_version", str(d.number), mv.registry_version),
            ("selection_sha256", d.selection_sha256, mv.selection_sha256),
        ):
            if actual != expected:
                out.append(
                    Finding(
                        f"tag_{key}",
                        "drift",
                        True,
                        d.number,
                        expected,
                        actual,
                        f"tag platform.{key} of registry version {d.number} was changed",
                    )
                )

    actual_alias = snapshot.aliases.get(MIRROR_ALIAS)
    holder = holder_of.get(actual_alias) if actual_alias is not None else None
    expected_package = None if desired.production is None else desired.production.package_id
    if actual_alias is not None and (holder is None or not holder.owned):
        out.append(
            Finding(
                "alias_foreign_holder",
                "conflict",
                True,
                None,
                expected_package,
                actual_alias,
                f"the {MIRROR_ALIAS!r} alias points at MLflow version {actual_alias}, which the "
                "platform does not own; it is not moved",
            )
        )
    else:
        actual_package = None if holder is None else holder.package_id
        if actual_package != expected_package:
            explained_holders = {
                _production_package(last_state, last_versions),
                _production_package(attempt_state, attempt_versions),
            }
            explained = actual_package in explained_holders
            out.append(
                Finding(
                    "alias_stale" if explained else "alias_wrong",
                    "pending" if explained else "drift",
                    True,
                    None,
                    expected_package,
                    actual_package,
                    f"the {MIRROR_ALIAS!r} alias should "
                    + ("be absent" if expected_package is None else f"mirror {expected_package}")
                    + f" but mirrors {actual_package}",
                )
            )
    for alias, version in snapshot.aliases.items():
        if alias != MIRROR_ALIAS:
            out.append(
                Finding(
                    "alias_foreign",
                    "conflict",
                    False,
                    None,
                    None,
                    {alias: version},
                    f"MLflow has the alias {alias!r}, which the platform does not manage; "
                    "left alone",
                )
            )
    return out


def _production_package(state: dict[str, Any] | None, versions: dict[str, Any]) -> str | None:
    number = (state or {}).get("production")
    if number is None:
        return None
    package = versions.get(str(number), {}).get("package_id")
    return None if package is None else str(package)


def intended_repair(finding: Finding) -> str:
    """What a reconcile would do about one finding, in words (for the audit record)."""
    if finding.category == "conflict":
        return f"no change: {finding.kind} is reported only ({finding.message})"
    return {
        "model_missing": "create the platform-owned registered model",
        "version_missing": f"track the package and create the MLflow version for {finding.version}",
        "alias_stale": f"point the production alias at {finding.expected}",
        "alias_wrong": f"point the production alias at {finding.expected} (was {finding.actual})",
    }.get(
        finding.kind,
        f"set {finding.kind} of version {finding.version} to {finding.expected!r} "
        f"(was {finding.actual!r})",
    )


# -------------------------------------------------------------------------------------- the mirror


class Mirror:
    """Sync, check, reconcile and report for every registry model; safe to share between threads."""

    def __init__(
        self,
        engine: Engine,
        models_root: Path,
        *,
        enabled: Callable[[], bool] | None = None,
        store_factory: Callable[[], TrackingStore] | None = None,
    ) -> None:
        self.engine = engine
        self.models_root = models_root
        self._enabled = enabled or mirror_enabled
        self._store_factory = store_factory or job_tracking_store

    # ------------------------------------------------------------------------------ configuration

    @property
    def enabled(self) -> bool:
        try:
            return bool(self._enabled())
        except MirrorSettingError:
            return False

    def problem(self) -> str | None:
        """Why the mirror cannot run right now (setting or backend), or None."""
        try:
            if not self._enabled():
                return "the mirror is disabled"
        except MirrorSettingError as error:
            return str(error)
        try:
            self._store_factory()
        except TrackingNotAllowed as error:
            return str(error)
        except TrackingError as error:
            return scrub(str(error))
        return None

    def _store(self) -> TrackingStore:
        try:
            return self._store_factory()
        except TrackingNotAllowed as error:
            raise TrackingError(f"the mirror may not use this MLflow store: {error}") from None

    # ----------------------------------------------------------------------------------- reading

    def _model(self, name: str) -> tuple[int, int]:
        with self.engine.connect() as c:
            row = c.execute(
                text("SELECT id, last_event_seq FROM registered_models WHERE name = :n"),
                {"n": name},
            ).first()
        if row is None:
            raise MirrorModelNotFound(f"no model named {name}")
        return int(row.id), int(row.last_event_seq)

    def desired(self, name: str) -> Desired:
        """The registry's state at one event number (one repeatable-read snapshot)."""
        with self.engine.connect().execution_options(isolation_level="REPEATABLE READ") as c:
            model = c.execute(
                text("SELECT id, last_event_seq FROM registered_models WHERE name = :n"),
                {"n": name},
            ).first()
            if model is None:
                raise MirrorModelNotFound(f"no model named {name}")
            rows = c.execute(
                text(
                    "SELECT version, package_id, status, selection_sha256 FROM model_versions "
                    "WHERE model_id = :m ORDER BY version"
                ),
                {"m": model.id},
            ).all()
        return Desired(
            int(model.id),
            name,
            int(model.last_event_seq),
            tuple(
                DesiredVersion(
                    int(r.version), str(r.package_id), str(r.status), str(r.selection_sha256)
                )
                for r in rows
            ),
        )

    def _progress(self, model_id: int) -> dict[str, Any]:
        with self.engine.connect() as c:
            row = c.execute(
                text(
                    "SELECT synced_seq, synced_state, attempt_state, attempts FROM registry_sync "
                    "WHERE model_id = :m"
                ),
                {"m": model_id},
            ).first()
        if row is None:
            return {"synced_seq": 0, "synced_state": None, "attempt_state": None, "attempts": 0}
        return dict(row._mapping)

    def status(self, name: str) -> MirrorStatus:
        for item in self.overview():
            if item.model == name:
                return item
        raise MirrorModelNotFound(f"no model named {name}")

    def overview(self) -> list[MirrorStatus]:
        enabled = self.enabled
        with self.engine.connect() as c:
            rows = c.execute(
                text(
                    "SELECT m.name, m.last_event_seq, COALESCE(s.synced_seq, 0) AS synced_seq, "
                    "COALESCE(s.attempts, 0) AS attempts, s.last_error, s.last_error_kind, "
                    "s.last_success_at, s.next_attempt_at FROM registered_models m "
                    "LEFT JOIN registry_sync s ON s.model_id = m.id ORDER BY m.name"
                )
            ).all()
        return [_status(r, enabled) for r in rows]

    # -------------------------------------------------------------------------------------- locks

    def _try_lock(self, model_id: int) -> Connection | None:
        connection = self.engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        try:
            got = connection.execute(
                text("SELECT pg_try_advisory_lock(:c, :k)"), {"c": LOCK_CLASS, "k": model_id}
            ).scalar_one()
        except Exception:
            connection.close()
            raise
        if not got:
            connection.close()
            return None
        return connection

    @staticmethod
    def _unlock(connection: Connection, model_id: int) -> None:
        try:
            connection.execute(
                text("SELECT pg_advisory_unlock(:c, :k)"), {"c": LOCK_CLASS, "k": model_id}
            )
            connection.close()
        except Exception:
            connection.invalidate()  # never return a connection that may still hold the lock
            connection.close()

    # --------------------------------------------------------------------------------- bookkeeping

    def _record_failure(self, model_id: int, kind: str, message: str) -> None:
        recheck = kind in ("drift", "conflict")
        with self.engine.begin() as c:
            c.execute(
                text(
                    "INSERT INTO registry_sync (model_id, attempts, last_attempt_at, last_error, "
                    "last_error_kind, next_attempt_at) VALUES (:m, 1, now(), :e, :k, "
                    "now() + make_interval(secs => :s)) "
                    "ON CONFLICT (model_id) DO UPDATE SET "
                    "attempts = registry_sync.attempts + 1, last_attempt_at = now(), "
                    "last_error = :e, last_error_kind = :k, "
                    "next_attempt_at = now() + make_interval(secs => "
                    "CASE WHEN :recheck THEN :s ELSE LEAST(:base * power(2, "
                    "registry_sync.attempts), :cap) END)"
                ),
                {
                    "m": model_id,
                    "e": message[:2000],
                    "k": kind,
                    "s": RECHECK_SECONDS if recheck else BACKOFF_BASE_SECONDS,
                    "recheck": recheck,
                    "base": BACKOFF_BASE_SECONDS,
                    "cap": BACKOFF_MAX_SECONDS,
                },
            )

    def _record_attempt_state(self, model_id: int, state: dict[str, Any]) -> None:
        with self.engine.begin() as c:
            c.execute(
                text(
                    "INSERT INTO registry_sync (model_id, attempt_state, last_attempt_at) "
                    "VALUES (:m, CAST(:s AS jsonb), now()) ON CONFLICT (model_id) DO UPDATE SET "
                    "attempt_state = CAST(:s AS jsonb), last_attempt_at = now()"
                ),
                {"m": model_id, "s": _dump(state)},
            )

    def _record_success(self, desired: Desired) -> None:
        """Advance `synced_seq`: only called after the complete desired state was verified."""
        with self.engine.begin() as c:
            c.execute(
                text(
                    "INSERT INTO registry_sync (model_id, synced_seq, synced_state, attempts, "
                    "last_attempt_at, last_success_at) VALUES (:m, :n, CAST(:s AS jsonb), 0, "
                    "now(), now()) ON CONFLICT (model_id) DO UPDATE SET "
                    "synced_seq = GREATEST(registry_sync.synced_seq, :n), "
                    "synced_state = CASE WHEN registry_sync.synced_seq <= :n "
                    "THEN CAST(:s AS jsonb) ELSE registry_sync.synced_state END, "
                    "attempts = 0, last_error = NULL, last_error_kind = NULL, "
                    "last_attempt_at = now(), last_success_at = now(), next_attempt_at = NULL"
                ),
                {"m": desired.model_id, "n": desired.seq, "s": _dump(desired.state())},
            )

    def _log(
        self,
        model_id: int,
        kind: str,
        outcome: str,
        seq: int,
        details: dict[str, Any],
        reason: str | None = None,
    ) -> None:
        with self.engine.begin() as c:
            c.execute(
                text(
                    "INSERT INTO registry_sync_log (model_id, kind, outcome, seq, reason, details) "
                    "VALUES (:m, :k, :o, :n, :r, CAST(:d AS jsonb))"
                ),
                {
                    "m": model_id,
                    "k": kind,
                    "o": outcome,
                    "n": seq,
                    "r": reason,
                    "d": _dump(details),
                },
            )

    # ---------------------------------------------------------------------------------- applying

    def _apply(
        self,
        store: TrackingStore,
        desired: Desired,
        snapshot: MirrorSnapshot,
        actions: list[dict[str, Any]],
    ) -> None:
        """Make MLflow's platform-owned parts equal the desired state. Every step is idempotent;
        `actions` is filled as it goes, so a failure still shows what was done."""
        name = desired.name
        if mirror_ensure_model(store, name):
            actions.append({"action": "create_model"})
        mlflow_versions: dict[int, str] = {}
        for d in desired.versions:
            run_id = mirror_package_run(store, self.models_root, d.package_id)
            version, created = mirror_ensure_version(
                store,
                name,
                package_id=d.package_id,
                run_id=run_id,
                registry_version=d.number,
                status=d.status,
                selection_sha256=d.selection_sha256,
            )
            if created:
                actions.append({"action": "create_version", "version": d.number, "mlflow": version})
            changed = mirror_set_tags(
                store,
                name,
                version,
                {
                    "platform.status": d.status,
                    "platform.registry_version": str(d.number),
                    "platform.selection_sha256": d.selection_sha256,
                },
            )
            if changed:
                actions.append({"action": "set_tags", "version": d.number, "keys": changed})
            mlflow_versions[d.number] = version
        production = desired.production
        target = None if production is None else mlflow_versions[production.number]
        current = snapshot.aliases.get(MIRROR_ALIAS)
        if mirror_set_production_alias(store, name, target, current=current):
            actions.append(
                {"action": "clear_alias" if target is None else "set_alias", "mlflow": target}
            )

    def _pass(
        self, name: str, *, reconcile: bool, reason: str | None
    ) -> SyncOutcome | ReconcileOutcome:
        """One sync (or reconcile) pass for one model, under the model's advisory lock."""
        if not self.enabled:
            raise MirrorDisabled("the registry mirror is disabled (AUTOML_REGISTRY_MIRROR=on)")
        model_id, _ = self._model(name)
        lock = self._try_lock(model_id)
        if lock is None:
            return ReconcileOutcome("busy", 0, [], [], [], []) if reconcile else SyncOutcome("busy")
        kind = "reconcile" if reconcile else "sync"
        desired = None
        actions: list[dict[str, Any]] = []
        detected: list[Finding] = []
        try:
            desired = self.desired(name)
            progress = self._progress(model_id)
            if not reconcile and desired.seq <= progress["synced_seq"]:
                return SyncOutcome("up_to_date", desired.seq)
            store = self._store()
            snapshot = mirror_snapshot(store, name)
            detected = classify(
                desired, progress["synced_state"], progress["attempt_state"], snapshot
            )
            blockers = [f for f in detected if f.category == "conflict" and f.blocking]
            drift = [f for f in detected if f.category == "drift"]
            if blockers or (drift and not reconcile):
                category = "conflict" if blockers else "drift"
                self._record_failure(model_id, category, _summary(blockers or drift))
                outcome = "conflict" if blockers else "drift"
                self._log(
                    model_id,
                    kind,
                    outcome,
                    desired.seq,
                    {"findings": _findings(detected), "actions": []},
                    reason,
                )
                if reconcile:
                    return ReconcileOutcome(
                        "conflict",
                        desired.seq,
                        detected,
                        [intended_repair(f) for f in detected],
                        [],
                        detected,
                    )
                return SyncOutcome(outcome, desired.seq, [], detected)  # type: ignore[arg-type]
            self._record_attempt_state(model_id, desired.state())
            self._apply(store, desired, snapshot, actions)
            after = classify(
                desired, desired.state(), desired.state(), mirror_snapshot(store, name)
            )
            unresolved = [f for f in after if f.category != "conflict" or f.blocking]
            if unresolved:
                raise TrackingError("verification failed: " + _summary(unresolved))
            self._record_success(desired)
            if actions or reconcile:
                self._log(
                    model_id,
                    kind,
                    "ok",
                    desired.seq,
                    {"actions": actions, "findings_before": _findings(detected)}
                    if not reconcile
                    else {
                        "detected": _findings(detected),
                        "intended": [intended_repair(f) for f in detected],
                        "actions": actions,
                        "result": _findings(after),
                    },
                    reason,
                )
            if reconcile:
                return ReconcileOutcome(
                    "ok",
                    desired.seq,
                    detected,
                    [intended_repair(f) for f in detected],
                    actions,
                    after,
                )
            return SyncOutcome("synced", desired.seq, actions, detected)
        except MirrorDisabled:
            raise
        except MirrorModelNotFound:
            raise
        except Exception as error:
            message = (
                scrub(f"{type(error).__name__}: {error}")
                if not isinstance(error, TrackingError)
                else scrub(str(error))
            )
            category = "conflict" if isinstance(error, MirrorConflict) else "error"
            seq = 0 if desired is None else desired.seq
            try:
                self._record_failure(model_id, category, message)
                self._log(
                    model_id,
                    kind,
                    "conflict" if category == "conflict" else "failed",
                    seq,
                    {
                        "error": message,
                        "actions": actions,
                        **({"detected": _findings(detected)} if reconcile else {}),
                    },
                    reason,
                )
            except Exception:  # bookkeeping must never mask the real result
                pass
            if reconcile:
                return ReconcileOutcome(
                    "conflict" if category == "conflict" else ("partial" if actions else "failed"),
                    seq,
                    detected,
                    [intended_repair(f) for f in detected],
                    actions,
                    [],
                    message,
                )
            return SyncOutcome(
                "conflict" if category == "conflict" else "failed", seq, actions, [], message
            )
        finally:
            self._unlock(lock, model_id)

    # ------------------------------------------------------------------------------- public calls

    def sync(self, name: str) -> SyncOutcome:
        """Bring MLflow up to the registry's current state, if it is not already."""
        outcome = self._pass(name, reconcile=False, reason=None)
        assert isinstance(outcome, SyncOutcome)
        return outcome

    def check(self, name: str) -> CheckReport:
        """Read-only: what differs between the registry and MLflow. Writes nothing anywhere."""
        if not self.enabled:
            raise MirrorDisabled("the registry mirror is disabled (AUTOML_REGISTRY_MIRROR=on)")
        model_id, _ = self._model(name)
        desired = self.desired(name)
        progress = self._progress(model_id)
        findings = classify(
            desired,
            progress["synced_state"],
            progress["attempt_state"],
            mirror_snapshot(self._store(), name),
        )
        return CheckReport(name, desired.seq, int(progress["synced_seq"]), findings)

    def reconcile(self, name: str, reason: str) -> ReconcileOutcome:
        """Explicit repair: reset the platform-owned parts of MLflow to the registry's state and
        record what was found, what was intended and what happened. Conflicts are never changed."""
        cleaned = (reason or "").strip()
        if not 1 <= len(cleaned) <= REASON_MAX:
            raise InvalidReason(f"the reason must be 1 to {REASON_MAX} characters")
        outcome = self._pass(name, reconcile=True, reason=cleaned)
        assert isinstance(outcome, ReconcileOutcome)
        return outcome

    def sync_soon(self, name: str, timeout: float = 15.0) -> MirrorStatus | None:
        """Best-effort sync right after a registry change. Never raises and never waits longer
        than `timeout`: the registry change has already been committed."""
        if not self.enabled:
            return None
        done = threading.Event()

        def run() -> None:
            try:
                self.sync(name)
            except Exception:
                pass
            finally:
                done.set()

        threading.Thread(target=run, name=f"mirror-sync-{name}", daemon=True).start()
        done.wait(timeout)
        try:
            return self.status(name)
        except Exception:
            return None

    def verify(self, name: str) -> list[Finding]:
        """A read-only look at a model that is in sync: records drift (in PostgreSQL only) so it is
        shown, and clears it when it has been fixed. Never writes to MLflow."""
        model_id, _ = self._model(name)
        report = self.check(name)
        stoppers = [f for f in report.findings if f.stops_sync]
        row = self._progress(model_id)
        if stoppers:
            category = "conflict" if any(f.category == "conflict" for f in stoppers) else "drift"
            first = self._last_error_kind(model_id) != category
            self._record_failure(model_id, category, _summary(stoppers))
            if first:
                self._log(
                    model_id,
                    "sync",
                    category,
                    report.desired_seq,
                    {"findings": _findings(report.findings), "actions": []},
                )
        elif row["synced_seq"] == report.desired_seq:
            self._record_success(self.desired(name))
        return report.findings

    def _last_error_kind(self, model_id: int) -> str | None:
        with self.engine.connect() as c:
            value = c.execute(
                text("SELECT last_error_kind FROM registry_sync WHERE model_id = :m"),
                {"m": model_id},
            ).scalar_one_or_none()
        return None if value is None else str(value)

    def run_pending(self) -> dict[str, str]:
        """One round of the background retry: sync every model with pending (persisted) work whose
        retry time has come, and re-read drifted or idle models. Returns model -> result."""
        results: dict[str, str] = {}
        if not self.enabled:
            return results
        with self.engine.connect() as c:
            due = c.execute(
                text(
                    "SELECT m.name, m.last_event_seq > COALESCE(s.synced_seq, 0) AS pending, "
                    "s.last_error_kind AS kind FROM registered_models m "
                    "LEFT JOIN registry_sync s ON s.model_id = m.id WHERE "
                    "(s.next_attempt_at IS NULL OR s.next_attempt_at <= now()) AND ("
                    "m.last_event_seq > COALESCE(s.synced_seq, 0) OR s.last_error_kind IS NOT NULL "
                    "OR s.last_success_at IS NULL OR s.last_success_at < now() - "
                    "make_interval(secs => :idle)) ORDER BY m.name"
                ),
                {"idle": RECHECK_SECONDS},
            ).all()
        for row in due:
            name = str(row.name)
            try:
                if row.pending:
                    results[name] = self.sync(name).result
                else:
                    findings = self.verify(name)
                    results[name] = "drift" if any(f.stops_sync for f in findings) else "verified"
            except Exception as error:  # one model must not stop the others
                results[name] = f"error: {type(error).__name__}"
        return results


def _status(row: Any, enabled: bool) -> MirrorStatus:
    desired, synced = int(row.last_event_seq), int(row.synced_seq)
    kind = row.last_error_kind
    state: State
    if not enabled:
        state = "disabled"
    elif kind == "conflict":
        state = "conflict"
    elif kind == "drift":
        state = "drifted"
    elif kind == "error" and desired > synced:
        state = "failed"
    elif desired > synced:
        state = "pending"
    elif kind == "error":
        state = "failed"
    else:
        state = "in_sync"
    return MirrorStatus(
        str(row.name),
        enabled,
        state,
        desired,
        synced,
        int(row.attempts),
        row.last_error,
        kind,
        row.last_success_at,
        row.next_attempt_at,
    )


def _dump(value: Any) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, default=str)


def _findings(findings: list[Finding]) -> dict[str, Any]:
    return {
        "total": len(findings),
        "items": [f.to_dict() for f in findings[:LOG_FINDINGS_MAX]],
        "truncated": len(findings) > LOG_FINDINGS_MAX,
    }


def _summary(findings: list[Finding]) -> str:
    shown = "; ".join(f.message for f in findings[:3])
    more = f" (+{len(findings) - 3} more)" if len(findings) > 3 else ""
    return (shown + more)[:1500]
