# ruff: noqa: F811
"""P4-7: deployments on a real PostgreSQL server (throwaway `automl_test_*` databases).

Packages are real sealed ones made by the platform's own `finalize`. NOT RUN when the server is
unreachable. Nothing here touches `data/` or the real `automl_ops` database.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from automl_ops.db import migrate
from automl_ops.deployments import (
    AlreadyDeployed,
    DeploymentActive,
    DeploymentNotFound,
    DeploymentRepository,
    InvalidDeploymentReason,
    NoPreviousDeployment,
    NotActive,
    NotDeployable,
)
from automl_ops.registry import (
    ModelNotFound,
    PackageUnverified,
    RegistryRepository,
    VersionNotFound,
)
from conftest import JobEnv, Run
from test_registry_postgres import pid, pool, reg, sealed_copy, unlock  # noqa: F401

pytestmark = pytest.mark.postgres

NAME = "churn"


class Rig:
    """Three classifier versions of one model: v1 and v2 in staging, v3 in development."""

    def __init__(self, env: JobEnv, pool: list[Run]) -> None:
        self.engine = env.engine
        self.registry = RegistryRepository(env.engine, pool[0].root)
        self.deployments = DeploymentRepository(env.engine, pool[0].root)
        for run in pool[:3]:
            reg(self.registry, run, NAME)
        for number in (1, 2):
            self.registry.promote(NAME, number, "staging", "ready")


@pytest.fixture
def rig(job_env: JobEnv, pool: list[Run]) -> Rig:
    return Rig(job_env, pool)


def rows(engine: Engine, sql: str = "SELECT * FROM deployments ORDER BY id") -> list[Any]:
    with engine.connect() as c:
        return list(c.execute(text(sql)).all())


# -- what can be deployed -------------------------------------------------------------------------


def test_a_staging_version_can_be_deployed_and_is_active(rig: Rig) -> None:
    d = rig.deployments.deploy(NAME, 1, "first go-live")
    assert (d.model, d.version, d.status, d.action, d.replaced_id) == (
        NAME,
        1,
        "active",
        "deploy",
        None,
    )
    assert d.registry_status == "staging" and d.ended_at is None and d.task == "classification"
    assert rig.deployments.active() == d
    assert rig.deployments.get(d.id) == d


def test_a_production_version_can_be_deployed(rig: Rig) -> None:
    rig.registry.promote(NAME, 1, "production", "live")
    assert rig.deployments.deploy(NAME, 1, "serve it").registry_status == "production"


def test_development_and_archived_versions_are_refused_and_nothing_is_written(rig: Rig) -> None:
    with pytest.raises(NotDeployable, match="development"):
        rig.deployments.deploy(NAME, 3, "no")
    rig.registry.archive(NAME, 2, "retired")
    with pytest.raises(NotDeployable, match="archived"):
        rig.deployments.deploy(NAME, 2, "no")
    assert rows(rig.engine) == []


def test_unknown_models_versions_and_bad_reasons_are_clear_errors(rig: Rig) -> None:
    with pytest.raises(ModelNotFound):
        rig.deployments.deploy("ghost", 1, "x")
    with pytest.raises(VersionNotFound):
        rig.deployments.deploy(NAME, 99, "x")
    for bad in ("", "   ", "x" * 501):
        with pytest.raises(InvalidDeploymentReason):
            rig.deployments.deploy(NAME, 1, bad)
    with pytest.raises(DeploymentNotFound):
        rig.deployments.get(12345)
    assert rows(rig.engine) == []


def test_a_package_that_no_longer_verifies_is_refused_and_nothing_changes(
    job_env: JobEnv, pool: list[Run], tmp_path: Path
) -> None:
    root = sealed_copy(pool[0], tmp_path / "copy")
    registry = RegistryRepository(job_env.engine, root)
    reg(registry, pool[0], NAME)
    registry.promote(NAME, 1, "staging", "ready")
    unlock(root / pid(pool[0]) / "model.joblib").write_bytes(b"not the model")
    deployments = DeploymentRepository(job_env.engine, root)
    with pytest.raises(PackageUnverified) as raised:
        deployments.deploy(NAME, 1, "try")
    assert str(root) not in str(raised.value) + str(raised.value.details)
    assert rows(job_env.engine) == []


# -- one active deployment, switch, rollback, undeploy -------------------------------------------


def test_deploying_while_one_is_active_points_at_switch(rig: Rig) -> None:
    first = rig.deployments.deploy(NAME, 1, "a")
    with pytest.raises(DeploymentActive) as raised:
        rig.deployments.deploy(NAME, 2, "b")
    assert raised.value.details == {"active_deployment_id": first.id}
    assert len(rows(rig.engine)) == 1


def test_switch_ends_the_old_deployment_and_starts_the_new_one_together(rig: Rig) -> None:
    first = rig.deployments.deploy(NAME, 1, "a")
    second = rig.deployments.switch(first.id, NAME, 2, "better model")
    assert (second.version, second.action, second.replaced_id, second.status) == (
        2,
        "switch",
        first.id,
        "active",
    )
    ended = rig.deployments.get(first.id)
    assert ended.status == "ended" and ended.ended_at is not None
    assert ended.end_reason == f"switch: replaced by {NAME} version 2"
    assert rig.deployments.active() == second
    assert [d.id for d in rig.deployments.list()] == [second.id, first.id]
    assert [d.id for d in rig.deployments.list(status="active")] == [second.id]


def test_switch_needs_the_active_deployment_and_a_different_version(rig: Rig) -> None:
    first = rig.deployments.deploy(NAME, 1, "a")
    with pytest.raises(AlreadyDeployed):
        rig.deployments.switch(first.id, NAME, 1, "same")
    second = rig.deployments.switch(first.id, NAME, 2, "b")
    with pytest.raises(NotActive) as raised:
        rig.deployments.switch(first.id, NAME, 1, "stale id")
    assert raised.value.details == {"active_deployment_id": second.id}
    with pytest.raises(NotDeployable):
        rig.deployments.switch(second.id, NAME, 3, "development version")
    assert rig.deployments.active() == second


def test_rollback_undoes_a_switch_and_is_logged_as_a_rollback(rig: Rig) -> None:
    first = rig.deployments.deploy(NAME, 1, "a")
    second = rig.deployments.switch(first.id, NAME, 2, "b")
    back = rig.deployments.rollback("b misbehaves")
    assert (back.version, back.action, back.replaced_id) == (1, "rollback", second.id)
    assert rig.deployments.get(second.id).status == "ended"
    assert [(d.version, d.action) for d in rig.deployments.list()] == [
        (1, "rollback"),
        (2, "switch"),
        (1, "deploy"),
    ]


def test_a_rollback_is_not_undone_by_a_second_rollback_and_a_first_deploy_has_no_previous(
    rig: Rig,
) -> None:
    first = rig.deployments.deploy(NAME, 1, "a")
    with pytest.raises(NoPreviousDeployment):
        rig.deployments.rollback("nothing before")
    rig.deployments.switch(first.id, NAME, 2, "b")
    rig.deployments.rollback("back")
    with pytest.raises(NoPreviousDeployment, match="rollback"):
        rig.deployments.rollback("again")


def test_rollback_with_nothing_deployed_and_to_a_version_that_is_no_longer_deployable(
    rig: Rig,
) -> None:
    with pytest.raises(NotActive):
        rig.deployments.rollback("none")
    first = rig.deployments.deploy(NAME, 1, "a")
    rig.deployments.switch(first.id, NAME, 2, "b")
    rig.registry.archive(NAME, 1, "retired meanwhile")
    with pytest.raises(NotDeployable):
        rig.deployments.rollback("back")
    assert rig.deployments.active() is not None and rig.deployments.active().version == 2  # type: ignore[union-attr]


def test_undeploy_ends_the_deployment_and_leaves_nothing_active(rig: Rig) -> None:
    first = rig.deployments.deploy(NAME, 1, "a")
    ended = rig.deployments.undeploy(first.id, "maintenance")
    assert ended.status == "ended" and ended.end_reason == "undeployed: maintenance"
    assert rig.deployments.active() is None
    with pytest.raises(NotActive):
        rig.deployments.undeploy(first.id, "twice")
    again = rig.deployments.deploy(NAME, 2, "back online")  # a new deploy after an undeploy
    assert again.action == "deploy" and again.replaced_id is None


def test_archiving_a_deployed_staging_version_does_not_undeploy_it_but_shows_in_its_status(
    rig: Rig,
) -> None:
    d = rig.deployments.deploy(NAME, 1, "a")
    rig.registry.archive(NAME, 1, "retired")
    now = rig.deployments.get(d.id)
    assert now.status == "active" and now.registry_status == "archived"


# -- races ---------------------------------------------------------------------------------------


def run_all(calls: list[Any]) -> tuple[list[Any], list[BaseException]]:
    barrier = threading.Barrier(len(calls))
    wins: list[Any] = []
    losses: list[BaseException] = []

    def go(call: Any) -> None:
        barrier.wait()
        try:
            wins.append(call())
        except BaseException as error:  # noqa: BLE001
            losses.append(error)

    with ThreadPoolExecutor(len(calls)) as pool_:
        list(pool_.map(go, calls))
    return wins, losses


def test_simultaneous_deploys_have_exactly_one_winner(rig: Rig) -> None:
    calls = [lambda n=n: rig.deployments.deploy(NAME, 1 + n % 2, f"racer {n}") for n in range(6)]
    wins, losses = run_all(calls)
    assert len(wins) == 1 and all(isinstance(e, DeploymentActive) for e in losses)
    assert len(rows(rig.engine, "SELECT id FROM deployments WHERE status = 'active'")) == 1


def test_simultaneous_switches_of_the_same_deployment_have_exactly_one_winner(rig: Rig) -> None:
    first = rig.deployments.deploy(NAME, 1, "a")
    calls = [lambda n=n: rig.deployments.switch(first.id, NAME, 2, f"racer {n}") for n in range(5)]
    wins, losses = run_all(calls)
    assert len(wins) == 1 and all(isinstance(e, NotActive | AlreadyDeployed) for e in losses)
    assert len(rows(rig.engine)) == 2


def test_a_switch_racing_an_undeploy_leaves_a_consistent_history(rig: Rig) -> None:
    first = rig.deployments.deploy(NAME, 1, "a")
    wins, losses = run_all(
        [
            lambda: rig.deployments.switch(first.id, NAME, 2, "b"),
            lambda: rig.deployments.undeploy(first.id, "stop"),
        ]
    )
    assert len(wins) == 1 and len(losses) == 1 and isinstance(losses[0], NotActive)
    active = rows(rig.engine, "SELECT id FROM deployments WHERE status = 'active'")
    assert len(active) <= 1


# -- the database's own rules (hand-written statements) ----------------------------------------


def refused(engine: Engine, sql: str, **params: Any) -> str:
    with pytest.raises((IntegrityError, DBAPIError)) as raised, engine.begin() as c:
        c.execute(text(sql), params)
    return str(raised.value)


def test_the_database_refuses_to_deploy_a_development_version(rig: Rig) -> None:
    message = refused(
        rig.engine,
        "INSERT INTO deployments (model_id, version_id, package_id, action, reason) "
        "SELECT model_id, id, package_id, 'deploy', 'sneaky' FROM model_versions WHERE version = 3",
    )
    assert "development version cannot be deployed" in message
    assert rows(rig.engine) == []


def test_the_database_refuses_a_mismatched_package_and_a_second_active_row(rig: Rig) -> None:
    refused(
        rig.engine,
        "INSERT INTO deployments (model_id, version_id, package_id, action, reason) "
        "SELECT model_id, id, 'm-20260101T000000Z-0badc0de', 'deploy', 'x' "
        "FROM model_versions WHERE version = 1",
    )
    rig.deployments.deploy(NAME, 1, "a")
    refused(
        rig.engine,
        "INSERT INTO deployments (model_id, version_id, package_id, action, reason) "
        "SELECT model_id, id, package_id, 'deploy', 'second' FROM model_versions WHERE version = 2",
    )
    assert len(rows(rig.engine)) == 1


def test_the_database_keeps_deployments_immutable_and_never_deletes_them(rig: Rig) -> None:
    first = rig.deployments.deploy(NAME, 1, "a")
    for sql in (
        "UPDATE deployments SET reason = 'rewritten'",
        "UPDATE deployments SET version_id = (SELECT id FROM model_versions WHERE version = 2)",
        "UPDATE deployments SET package_id = 'm-20260101T000000Z-0badc0de'",
        "UPDATE deployments SET status = 'ended'",  # no end fields: the consistency check
        "DELETE FROM deployments",
        "TRUNCATE deployments CASCADE",
    ):
        refused(rig.engine, sql)
    rig.deployments.undeploy(first.id, "done")
    for sql in (
        "UPDATE deployments SET status = 'active', ended_at = NULL, end_reason = NULL",
        "UPDATE deployments SET end_reason = 'rewritten history'",
        "DELETE FROM deployments",
    ):
        refused(rig.engine, sql)
    assert rig.deployments.get(first.id).end_reason == "undeployed: done"


def test_action_and_replaced_id_must_agree(rig: Rig) -> None:
    refused(
        rig.engine,
        "INSERT INTO deployments (model_id, version_id, package_id, action, reason) "
        "SELECT model_id, id, package_id, 'switch', 'no predecessor' FROM model_versions "
        "WHERE version = 1",
    )
    first = rig.deployments.deploy(NAME, 1, "a")
    refused(
        rig.engine,
        "INSERT INTO deployments (model_id, version_id, package_id, action, reason, replaced_id) "
        "SELECT model_id, id, package_id, 'switch', 'predecessor still active', :r "
        "FROM model_versions WHERE version = 2",
        r=first.id,
    )


def test_a_logged_prediction_cannot_be_changed(rig: Rig) -> None:
    d = rig.deployments.deploy(NAME, 1, "a")
    with rig.engine.begin() as c:
        c.execute(
            text(
                "INSERT INTO predictions (id, deployment_id, model_version, inputs, output, "
                "latency_ms) VALUES ('p1', :d, 1, '{}', '{}', 1.5)"
            ),
            {"d": d.id},
        )
    refused(rig.engine, "UPDATE predictions SET output = '{\"x\": 1}'")
    # a prediction needs a real deployment and a sane latency
    refused(
        rig.engine,
        "INSERT INTO predictions (id, deployment_id, model_version, inputs, output, latency_ms) "
        "VALUES ('p2', 987654, 1, '{}', '{}', 1)",
    )
    refused(
        rig.engine,
        "INSERT INTO predictions (id, deployment_id, model_version, inputs, output, latency_ms) "
        f"VALUES ('p3', {d.id}, 1, '{{}}', '{{}}', -1)",
    )


def test_migration_0005_goes_down_and_up_again(job_env: JobEnv) -> None:
    assert migrate.downgrade(job_env.engine, "0004").current == "0004"
    with job_env.engine.connect() as c:
        functions = c.execute(
            text(
                "SELECT proname FROM pg_proc WHERE proname LIKE 'deployments%' "
                "OR proname LIKE 'predictions%'"
            )
        ).all()
        tables = c.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name IN ('deployments', 'predictions')"
            )
        ).all()
    assert functions == [] and tables == []
    assert migrate.upgrade(job_env.engine).current == "0007"
    assert migrate.check(job_env.engine).ok
