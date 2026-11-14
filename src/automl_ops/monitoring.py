"""Monitoring runs, reports and alerts in PostgreSQL (P4-9, BUILD_PLAN Step 24).

A **run** compares the most recent logged predictions of one deployment, in one traffic scope
(`real` by default, or `simulated`; the two are never mixed), with the reference sample of the
deployed version (`reference.py`), using `monitor.py`. It runs in the control plane, on demand,
never inside the serving app, and only reads the `predictions` table, so a failing or slow run
cannot affect serving.

**Reports** are immutable. A run that cannot reach a verdict is recorded as such:
`insufficient_data` (a window or reference below `min_rows`) or `failed` (with the error); neither
ever looks like a successful drift check, and neither opens or resolves an alert. A run writes at
the very end in one transaction (report plus alert changes), so an interrupted run leaves nothing
behind. Runs for one deployment are serialised by a PostgreSQL advisory lock; a second concurrent
run is refused (409), not queued.

**Alerts** (`alerts`, `alert_events`): at most one *unresolved* alert per (deployment, rule,
scope), so a condition that keeps holding updates its alert (`occurrences`, a `reoccurred` event)
instead of piling up copies. An `ok` run that evaluates a rule as no longer triggered resolves the
alert automatically (`auto_resolved`); a rule that was not evaluated (for example too few labels
for `performance_drop`) changes nothing. People can `acknowledge` (open only) and `resolve` (open
or acknowledged) with a reason. A resolved alert never changes; a later recurrence opens a new
alert. Every state change is an append-only event, and the database refuses forbidden moves from a
hand-written statement.

Nothing here reads a dataset, a sealed package or the held-out test part, trains or changes a model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy import Engine, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError

from automl_ops.deployments import DeploymentNotFound
from automl_ops.feedback import FeedbackRepository
from automl_ops.monitor import (
    MonitorError,
    Thresholds,
    build_summary,
    evaluate_rules,
    performance_comparison,
)
from automl_ops.reference import ReferenceService

log = logging.getLogger("automl_ops.monitoring")

LOCK_CLASS = 7268
PAGE = 100
REASON_MAX = 500
Scope = Literal["real", "simulated"]


class MonitoringError(Exception):
    code = "monitoring_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class NoReference(MonitoringError):
    code = "no_reference"


class MonitoringBusy(MonitoringError):
    code = "monitoring_in_progress"


class ReportNotFound(MonitoringError):
    code = "report_not_found"


class AlertNotFound(MonitoringError):
    code = "alert_not_found"


class InvalidAlertMove(MonitoringError):
    code = "invalid_alert_transition"


class InvalidAlertReason(MonitoringError):
    code = "validation_error"


class MonitoringUnavailable(MonitoringError):
    code = "monitoring_unavailable"


@dataclass(frozen=True)
class ReportRow:
    id: int
    deployment_id: int
    reference_id: int
    scope: str
    status: str
    window_rows: int
    window_start: datetime | None
    window_end: datetime | None
    thresholds: dict[str, Any]
    summary: dict[str, Any] | None
    error: str | None
    started_at: datetime
    finished_at: datetime


@dataclass(frozen=True)
class AlertEventRow:
    id: int
    action: str
    report_id: int | None
    reason: str | None
    created_at: datetime


@dataclass(frozen=True)
class AlertRow:
    id: int
    deployment_id: int
    rule: str
    scope: str
    severity: str
    status: str
    details: dict[str, Any]
    first_report_id: int
    last_report_id: int
    occurrences: int
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None
    events: tuple[AlertEventRow, ...] = ()


_REPORT = (
    "SELECT id, deployment_id, reference_id, scope, status, window_rows, window_start, window_end, "
    "thresholds, {summary} AS summary, error, started_at, finished_at FROM monitoring_reports"
)
_ALERT = (
    "SELECT id, deployment_id, rule, scope, severity, status, details, first_report_id, "
    "last_report_id, occurrences, created_at, updated_at, resolved_at FROM alerts"
)


def _reason(reason: str) -> str:
    cleaned = reason.strip()
    if not 1 <= len(cleaned) <= REASON_MAX:
        raise InvalidAlertReason(f"the reason must be 1 to {REASON_MAX} characters")
    return cleaned


def _unavailable(error: SQLAlchemyError) -> MonitoringUnavailable:
    return MonitoringUnavailable(
        f"the database is unavailable ({type(error).__name__}); nothing was changed"
    )


class MonitoringRepository:
    def __init__(
        self, engine: Engine, references: ReferenceService, feedback: FeedbackRepository
    ) -> None:
        self.engine = engine
        self.references = references
        self.feedback = feedback

    # ---------------------------------------------------------------------- run

    def run(
        self,
        deployment_id: int,
        scope: Scope = "real",
        thresholds: Thresholds | None = None,
    ) -> ReportRow:
        th = thresholds or Thresholds()
        try:
            return self._run(deployment_id, scope, th)
        except SQLAlchemyError as error:
            raise _unavailable(error) from None

    def _run(self, deployment_id: int, scope: Scope, th: Thresholds) -> ReportRow:
        started = datetime.now(UTC)
        with self.engine.connect() as guard:
            if not guard.execute(
                text("SELECT pg_try_advisory_lock(:c, :d)"), {"c": LOCK_CLASS, "d": deployment_id}
            ).scalar_one():
                raise MonitoringBusy(f"a monitoring run for deployment {deployment_id} is running")
            try:
                return self._guarded(deployment_id, scope, th, started)
            finally:
                guard.rollback()
                guard.execute(
                    text("SELECT pg_advisory_unlock(:c, :d)"), {"c": LOCK_CLASS, "d": deployment_id}
                )
                guard.commit()

    def _guarded(
        self, deployment_id: int, scope: Scope, th: Thresholds, started: datetime
    ) -> ReportRow:
        with self.engine.connect() as c:
            dep = c.execute(
                text(
                    "SELECT d.id, d.version_id, v.task, v.evaluation FROM deployments d "
                    "JOIN model_versions v ON v.id = d.version_id WHERE d.id = :i"
                ),
                {"i": deployment_id},
            ).first()
            if dep is None:
                raise DeploymentNotFound(f"no deployment {deployment_id}")
            ref = c.execute(
                text("SELECT id FROM reference_samples WHERE version_id = :v"),
                {"v": dep.version_id},
            ).first()
        if ref is None:
            raise NoReference(
                "this deployment's model version has no reference sample; create it first "
                "(POST /models/{name}/versions/{n}/reference)"
            )
        status, summary, error = "ok", None, None
        window: list[Any] = []
        try:
            reference = self.references.data(ref.id)
            window = self._window(deployment_id, scope, th.window)
            context = {
                "reference": {
                    "id": reference.meta.id,
                    "model": reference.meta.model,
                    "version": reference.meta.version,
                    "package_id": reference.meta.package_id,
                    "dataset_ref": reference.meta.dataset_ref,
                    "sample_rows": reference.meta.sample_rows,
                    "sample_sha256": reference.meta.sample_sha256,
                },
                "traffic": {
                    "scope": scope,
                    "simulated": scope == "simulated",
                    "note": "simulated traffic: not real observations"
                    if scope == "simulated"
                    else "real traffic only (simulated predictions are excluded)",
                },
                "window": {"rows": len(window), "max_rows": th.window},
            }
            try:
                summary = self._summarise(dep, reference, window, scope, th, context)
            except MonitorError as short:
                status, summary = (
                    "insufficient_data",
                    {**context, "reason": str(short), "rules": []},
                )
        except Exception as problem:  # a failed run is recorded as failed, never as a success
            log.warning("monitoring run for deployment %s failed: %s", deployment_id, problem)
            status, summary = "failed", None
            error = f"the run failed ({type(problem).__name__}): {str(problem)[:300]}"
        return self._store(
            deployment_id, ref.id, scope, status, window, th, summary, error, started
        )

    def _window(self, deployment_id: int, scope: Scope, size: int) -> list[Any]:
        with self.engine.connect() as c:
            c.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
            return list(
                c.execute(
                    text(
                        "SELECT inputs, output->'prediction' AS prediction, created_at "
                        "FROM predictions WHERE deployment_id = :d AND is_simulated = :s "
                        "ORDER BY created_at DESC, id DESC LIMIT :n"
                    ),
                    {"d": deployment_id, "s": scope == "simulated", "n": size},
                ).all()
            )

    def _summarise(
        self,
        dep: Any,
        reference: Any,
        window: list[Any],
        scope: Scope,
        th: Thresholds,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        core = build_summary(
            reference.columns,
            reference.rows,
            reference.predictions,
            [r.inputs for r in window],
            [r.prediction for r in window],
            dep.task,
            th,
        )
        performance = self.feedback.performance(dep.id, scope, th.min_labelled)
        stored = dep.evaluation or {}
        comparison = performance_comparison(
            performance,
            (stored.get("evaluation") or {}).get("metrics"),
            stored.get("selection_metric"),
            th,
        )
        return {
            **context,
            **core,
            "performance": comparison,
            "rules": evaluate_rules(core, comparison),
        }

    def _store(
        self,
        deployment_id: int,
        reference_id: int,
        scope: Scope,
        status: str,
        window: list[Any],
        th: Thresholds,
        summary: dict[str, Any] | None,
        error: str | None,
        started: datetime,
    ) -> ReportRow:
        with self.engine.begin() as c:
            report_id = c.execute(
                text(
                    "INSERT INTO monitoring_reports (deployment_id, reference_id, scope, status, "
                    "window_rows, window_start, window_end, thresholds, summary, error, "
                    "started_at) "
                    "VALUES (:d, :r, :s, :st, :n, :ws, :we, CAST(:th AS jsonb), "
                    "CAST(:su AS jsonb), :e, :sa) RETURNING id"
                ),
                {
                    "d": deployment_id,
                    "r": reference_id,
                    "s": scope,
                    "st": status,
                    "n": len(window),
                    "ws": min((w.created_at for w in window), default=None),
                    "we": max((w.created_at for w in window), default=None),
                    "th": json.dumps(th.to_dict()),
                    "su": None if summary is None else json.dumps(summary, allow_nan=False),
                    "e": error,
                    "sa": started,
                },
            ).scalar_one()
            if status == "ok" and summary is not None:
                for rule in summary["rules"]:
                    self._apply_rule(c, deployment_id, scope, int(report_id), rule)
        return self.get_report(int(report_id))

    @staticmethod
    def _apply_rule(
        c: Connection, deployment_id: int, scope: str, report_id: int, rule: dict[str, Any]
    ) -> None:
        if not rule["evaluated"]:
            return  # no verdict: leave any alert exactly as it is
        current = c.execute(
            text(
                "SELECT id FROM alerts WHERE deployment_id = :d AND rule = :r AND scope = :s "
                "AND status <> 'resolved' FOR UPDATE"
            ),
            {"d": deployment_id, "r": rule["rule"], "s": scope},
        ).first()
        details = json.dumps(rule["details"], allow_nan=False)

        def event(alert_id: int, action: str, reason: str | None = None) -> None:
            c.execute(
                text(
                    "INSERT INTO alert_events (alert_id, action, report_id, reason) "
                    "VALUES (:a, :ac, :r, :re)"
                ),
                {"a": alert_id, "ac": action, "r": report_id, "re": reason},
            )

        if rule["triggered"]:
            if current is None:
                alert_id = c.execute(
                    text(
                        "INSERT INTO alerts (deployment_id, rule, scope, severity, details, "
                        "first_report_id, last_report_id) VALUES (:d, :r, :s, :sv, "
                        "CAST(:de AS jsonb), :rep, :rep) RETURNING id"
                    ),
                    {
                        "d": deployment_id,
                        "r": rule["rule"],
                        "s": scope,
                        "sv": rule["severity"],
                        "de": details,
                        "rep": report_id,
                    },
                ).scalar_one()
                event(int(alert_id), "opened")
            else:
                c.execute(
                    text(
                        "UPDATE alerts SET occurrences = occurrences + 1, last_report_id = :rep, "
                        "details = CAST(:de AS jsonb), updated_at = now() WHERE id = :i"
                    ),
                    {"rep": report_id, "de": details, "i": current.id},
                )
                event(int(current.id), "reoccurred")
        elif current is not None:
            c.execute(
                text(
                    "UPDATE alerts SET status = 'resolved', resolved_at = now(), "
                    "last_report_id = :rep, updated_at = now() WHERE id = :i"
                ),
                {"rep": report_id, "i": current.id},
            )
            event(int(current.id), "auto_resolved", "the condition no longer holds")

    # ------------------------------------------------------------------- reports

    def get_report(self, report_id: int) -> ReportRow:
        try:
            with self.engine.connect() as c:
                row = c.execute(
                    text(_REPORT.format(summary="summary") + " WHERE id = :i"), {"i": report_id}
                ).first()
        except SQLAlchemyError as error:
            raise _unavailable(error) from None
        if row is None:
            raise ReportNotFound(f"no monitoring report {report_id}")
        return ReportRow(**dict(row._mapping))

    def reports(
        self,
        deployment_id: int,
        scope: Scope | None = None,
        before_id: int | None = None,
        limit: int = PAGE,
    ) -> list[ReportRow]:
        """Reports of a deployment, newest first, without their (large) summaries."""
        try:
            with self.engine.connect() as c:
                self._deployment(c, deployment_id)
                rows = c.execute(
                    text(
                        _REPORT.format(summary="NULL")
                        + " WHERE deployment_id = :d AND (CAST(:s AS text) IS NULL OR scope = :s)"
                        " AND (CAST(:b AS bigint) IS NULL OR id < :b) ORDER BY id DESC LIMIT :n"
                    ),
                    {"d": deployment_id, "s": scope, "b": before_id, "n": min(max(limit, 1), PAGE)},
                ).all()
        except SQLAlchemyError as error:
            raise _unavailable(error) from None
        return [ReportRow(**dict(r._mapping)) for r in rows]

    @staticmethod
    def _deployment(c: Connection, deployment_id: int) -> None:
        if (
            c.execute(text("SELECT 1 FROM deployments WHERE id = :i"), {"i": deployment_id}).first()
            is None
        ):
            raise DeploymentNotFound(f"no deployment {deployment_id}")

    def status(self, deployment_id: int) -> dict[str, Any]:
        """Reference present?, the latest report per traffic scope, unresolved alerts."""
        try:
            with self.engine.connect() as c:
                c.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
                dep = c.execute(
                    text("SELECT version_id FROM deployments WHERE id = :i"), {"i": deployment_id}
                ).first()
                if dep is None:
                    raise DeploymentNotFound(f"no deployment {deployment_id}")
                ref = c.execute(
                    text("SELECT id FROM reference_samples WHERE version_id = :v"),
                    {"v": dep.version_id},
                ).first()
                latest = {
                    scope: c.execute(
                        text(
                            _REPORT.format(summary="NULL")
                            + " WHERE deployment_id = :d AND scope = :s ORDER BY id DESC LIMIT 1"
                        ),
                        {"d": deployment_id, "s": scope},
                    ).first()
                    for scope in ("real", "simulated")
                }
                alerts = c.execute(
                    text(
                        "SELECT rule, scope, severity, status, count(*) AS n FROM alerts "
                        "WHERE deployment_id = :d AND status <> 'resolved' "
                        "GROUP BY rule, scope, severity, status ORDER BY scope, rule"
                    ),
                    {"d": deployment_id},
                ).all()
        except SQLAlchemyError as error:
            raise _unavailable(error) from None
        return {
            "deployment_id": deployment_id,
            "reference_id": None if ref is None else int(ref.id),
            "latest": {
                scope: None if row is None else ReportRow(**dict(row._mapping))
                for scope, row in latest.items()
            },
            "unresolved_alerts": [dict(a._mapping) for a in alerts],
        }

    # -------------------------------------------------------------------- alerts

    def alerts(
        self,
        *,
        status: str | None = None,
        deployment_id: int | None = None,
        rule: str | None = None,
        scope: str | None = None,
        before_id: int | None = None,
        limit: int = PAGE,
    ) -> list[AlertRow]:
        try:
            with self.engine.connect() as c:
                rows = c.execute(
                    text(
                        _ALERT + " WHERE (CAST(:st AS text) IS NULL OR status = :st)"
                        " AND (CAST(:d AS bigint) IS NULL OR deployment_id = :d)"
                        " AND (CAST(:r AS text) IS NULL OR rule = :r)"
                        " AND (CAST(:s AS text) IS NULL OR scope = :s)"
                        " AND (CAST(:b AS bigint) IS NULL OR id < :b) ORDER BY id DESC LIMIT :n"
                    ),
                    {
                        "st": status,
                        "d": deployment_id,
                        "r": rule,
                        "s": scope,
                        "b": before_id,
                        "n": min(max(limit, 1), PAGE),
                    },
                ).all()
        except SQLAlchemyError as error:
            raise _unavailable(error) from None
        return [AlertRow(**dict(r._mapping)) for r in rows]

    def get_alert(self, alert_id: int) -> AlertRow:
        try:
            with self.engine.connect() as c:
                row = c.execute(text(_ALERT + " WHERE id = :i"), {"i": alert_id}).first()
                if row is None:
                    raise AlertNotFound(f"no alert {alert_id}")
                events = c.execute(
                    text(
                        "SELECT id, action, report_id, reason, created_at FROM alert_events "
                        "WHERE alert_id = :i ORDER BY id"
                    ),
                    {"i": alert_id},
                ).all()
        except SQLAlchemyError as error:
            raise _unavailable(error) from None
        return AlertRow(
            **dict(row._mapping), events=tuple(AlertEventRow(**dict(e._mapping)) for e in events)
        )

    def acknowledge(self, alert_id: int, reason: str) -> AlertRow:
        return self._move(alert_id, "acknowledged", ("open",), _reason(reason))

    def resolve(self, alert_id: int, reason: str) -> AlertRow:
        return self._move(alert_id, "resolved", ("open", "acknowledged"), _reason(reason))

    def _move(self, alert_id: int, to: str, allowed: tuple[str, ...], reason: str) -> AlertRow:
        try:
            with self.engine.begin() as c:
                row = c.execute(
                    text("SELECT status FROM alerts WHERE id = :i FOR UPDATE"), {"i": alert_id}
                ).first()
                if row is None:
                    raise AlertNotFound(f"no alert {alert_id}")
                if row.status not in allowed:
                    raise InvalidAlertMove(
                        f"alert {alert_id} is {row.status}; it cannot be moved to {to}",
                        {"status": row.status},
                    )
                stamp = ", resolved_at = now()" if to == "resolved" else ""
                c.execute(
                    text(
                        f"UPDATE alerts SET status = :to, updated_at = now(){stamp} WHERE id = :i"
                    ),
                    {"to": to, "i": alert_id},
                )
                c.execute(
                    text("INSERT INTO alert_events (alert_id, action, reason) VALUES (:i, :a, :r)"),
                    {"i": alert_id, "a": to, "r": reason},
                )
        except SQLAlchemyError as error:
            raise _unavailable(error) from None
        return self.get_alert(alert_id)
