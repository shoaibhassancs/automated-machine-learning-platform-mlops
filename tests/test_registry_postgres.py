"""P4-5: the model registry on a real PostgreSQL server (throwaway `automl_test_*` databases).

Sealed packages are real ones, made by the platform's own `finalize` into scratch folders. NOT RUN
when the server is unreachable. Nothing here touches `data/` or the real `automl_ops` database.
"""

from __future__ import annotations

import ast
import json
import shutil
import threading
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from automl_ops import registry as registry_module
from automl_ops.jobs import JobRepository
from automl_ops.registry import (
    BULKY,
    InvalidMove,
    JobNotFoundForRegistry,
    JobNotRegisterable,
    ModelNotFound,
    NoPreviousProduction,
    PackageAlreadyRegistered,
    PackageNotRegisterable,
    PackageRequired,
    PackageUnverified,
    ProvenanceMismatch,
    RegistryRepository,
    VersionNotFound,
)
from conftest import JobEnv, Run, make_run
from test_worker_postgres import enqueue, make_spec, make_worker

pytestmark = pytest.mark.postgres

FAKE_PACKAGE = "m-20260101T000000Z-0badc0de"
FAKE_SHA = "a" * 64


@pytest.fixture(scope="module")
def pool(tmp_path_factory: pytest.TempPathFactory) -> list[Run]:
    """Four real sealed packages in one scratch models folder (three classifiers, one regressor)."""
    root = tmp_path_factory.mktemp("registry-models")
    return [make_run(root, "classification") for _ in range(3)] + [make_run(root, "regression")]


@pytest.fixture
def repo(job_env: JobEnv, pool: list[Run]) -> RegistryRepository:
    return RegistryRepository(job_env.engine, pool[0].root)


def sealed_copy(run: Run, root: Path) -> Path:
    """A copy of a sealed package that keeps its read-only modes (so it still verifies)."""
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run.result.path, root / run.result.package_id)
    return root


def unlock(path: Path) -> Path:
    """Make one file of a copied package writable so a test can tamper with it."""
    path.parent.chmod(0o755)
    path.chmod(0o644)
    return path


def pid(run: Run) -> str:
    return run.result.package_id


def reg(repo: RegistryRepository, run: Run, name: str = "churn", **kw: Any) -> Any:
    return repo.register(name, package_id=pid(run), job_id=None, **kw)


def to_production(repo: RegistryRepository, name: str, number: int) -> None:
    repo.promote(name, number, "staging", "to staging")
    repo.promote(name, number, "production", "to production")


def statuses(repo: RegistryRepository, name: str = "churn") -> dict[int, str]:
    return {v.version: v.status for v in repo.list_versions(name)}


def event_count(engine: Engine) -> int:
    with engine.connect() as c:
        return int(c.execute(text("SELECT count(*) FROM registry_events")).scalar_one())


def assert_consistent(engine: Engine) -> None:
    """Replay every model's events: contiguous numbers, every move starts where the last ended,
    the replayed statuses equal the stored ones, and at most one production version per model."""
    with engine.connect() as c:
        events = c.execute(
            text(
                "SELECT m.name, v.version, e.seq, e.action, e.from_status, e.to_status "
                "FROM registry_events e JOIN model_versions v ON v.id = e.version_id "
                "JOIN registered_models m ON m.id = e.model_id ORDER BY m.name, e.seq"
            )
        ).all()
        actual = {
            (r.name, r.version): r.status
            for r in c.execute(
                text(
                    "SELECT m.name, v.version, v.status FROM model_versions v "
                    "JOIN registered_models m ON m.id = v.model_id"
                )
            )
        }
        counters = {
            r.name: r.last_event_seq
            for r in c.execute(text("SELECT name, last_event_seq FROM registered_models"))
        }
    replayed: dict[tuple[str, int], str] = {}
    last: dict[str, int] = defaultdict(int)
    for e in events:
        assert e.seq == last[e.name] + 1, "event numbers must be contiguous per model"
        last[e.name] = e.seq
        assert e.from_status == replayed.get((e.name, e.version)), "a move starts where it ended"
        replayed[(e.name, e.version)] = e.to_status
    assert replayed == actual
    assert dict(last) == {k: v for k, v in counters.items() if v}
    per_model: dict[str, int] = defaultdict(int)
    for (name, _), status in actual.items():
        per_model[name] += status == "production"
    assert all(count <= 1 for count in per_model.values())


# -- registration ---------------------------------------------------------------------------------


def test_registering_a_sealed_package_creates_version_one_in_development(
    repo: RegistryRepository, pool: list[Run], job_env: JobEnv
) -> None:
    v = reg(repo, pool[0], reason="first candidate")
    assert (v.model, v.version, v.status, v.package_id) == ("churn", 1, "development", pid(pool[0]))
    assert (
        v.job_id is None and v.dataset == "toy@v1"
    )  # the catalog has exactly that dataset version
    assert v.algorithm and v.task == "classification" and v.target == "y"
    second = reg(repo, pool[1])
    assert second.version == 2
    other = reg(repo, pool[2], "other-model")
    assert other.version == 1  # numbers are per model
    events = repo.events("churn")
    assert [(e.seq, e.version, e.action, e.from_status, e.to_status) for e in events] == [
        (1, 1, "register", None, "development"),
        (2, 2, "register", None, "development"),
    ]
    assert events[0].reason == "first candidate" and events[1].reason == "registered"
    assert [m.name for m in repo.list_models()] == ["churn", "other-model"]
    assert_consistent(job_env.engine)


def test_a_package_without_a_catalogued_dataset_has_no_dataset_link(
    repo: RegistryRepository, pool: list[Run]
) -> None:
    v = reg(repo, pool[3])  # the regression package's data was never catalogued
    assert v.dataset is None and v.task == "regression"


def test_the_stored_evaluation_is_the_packages_own_test_result_without_bulky_data(
    repo: RegistryRepository, pool: list[Run]
) -> None:
    clf = repo.get_version("churn", reg(repo, pool[0]).version)
    stored = json.loads((pool[0].result.path / "test_result.json").read_text())
    assert clf.evaluation is not None
    assert clf.evaluation["evaluation"]["metrics"] == stored["evaluation"]["metrics"]
    assert clf.evaluation["selection_sha256"] == stored["selection_sha256"]
    assert clf.evaluation["limitation"] == stored["limitation"]
    assert "repeated" in clf.evaluation["limitation"] or "test" in clf.evaluation["limitation"]
    omitted = [k for k in BULKY if k in stored["evaluation"]]
    assert clf.evaluation["omitted"] == omitted
    assert not any(k in clf.evaluation["evaluation"] for k in BULKY)
    reg_v = repo.get_version("price", reg(repo, pool[3], "price").version)
    assert reg_v.evaluation is not None and "residual_points" in reg_v.evaluation["omitted"]


def test_registering_by_job_id_and_by_both_ids(job_env: JobEnv) -> None:
    job_id = enqueue(job_env, make_spec("final"))
    make_worker(job_env).run(once=True)
    package = JobRepositoryHelper.package_of(job_env, job_id)
    repo = RegistryRepository(job_env.engine, job_env.models)
    v = repo.register("churn", package_id=None, job_id=job_id)
    assert v.package_id == package and v.job_id == job_id and v.dataset == "toy@v1"
    with pytest.raises(PackageAlreadyRegistered):  # same package, same job, again
        repo.register("churn", package_id=package, job_id=job_id)
    assert_consistent(job_env.engine)


class JobRepositoryHelper:
    @staticmethod
    def package_of(env: JobEnv, job_id: int) -> str:
        with env.engine.connect() as c:
            value = c.execute(
                text("SELECT package_id FROM training_jobs WHERE id = :i"), {"i": job_id}
            ).scalar_one()
        return str(value)


def test_both_ids_must_name_the_same_package(job_env: JobEnv, pool: list[Run]) -> None:
    job_id = enqueue(job_env, make_spec("final"))
    make_worker(job_env).run(once=True)
    package = JobRepositoryHelper.package_of(job_env, job_id)
    shutil.copytree(pool[0].result.path, job_env.models / pid(pool[0]))
    repo = RegistryRepository(job_env.engine, job_env.models)
    with pytest.raises(ProvenanceMismatch):
        repo.register("churn", package_id=pid(pool[0]), job_id=job_id)
    assert repo.list_models() == []  # nothing was created
    ok = repo.register("churn", package_id=package, job_id=job_id)
    assert ok.package_id == package


def test_job_provenance_rules(job_env: JobEnv, repo: RegistryRepository) -> None:
    with pytest.raises(JobNotFoundForRegistry):
        repo.register("churn", package_id=None, job_id=9999)
    train_id = enqueue(job_env, make_spec("train"))
    queued_final = enqueue(job_env, make_spec("final"))
    with pytest.raises(JobNotRegisterable, match="queued"):
        repo.register("churn", package_id=None, job_id=queued_final)
    make_worker(job_env).run(once=True)  # runs both
    with pytest.raises(JobNotRegisterable, match="no sealed package"):
        repo.register("churn", package_id=None, job_id=train_id)
    with pytest.raises(PackageRequired):
        repo.register("churn", package_id=None, job_id=None)
    cancelled = enqueue(job_env, make_spec("final"))
    JobRepository(job_env.engine).request_cancel(cancelled)
    with pytest.raises(JobNotRegisterable, match="cancelled"):
        repo.register("churn", package_id=None, job_id=cancelled)
    assert repo.list_models() == []


def test_a_package_registers_once_even_under_another_name(
    repo: RegistryRepository, pool: list[Run], job_env: JobEnv
) -> None:
    reg(repo, pool[0])
    with pytest.raises(PackageAlreadyRegistered) as same:
        reg(repo, pool[0])
    assert same.value.details == {"model": "churn", "version": 1}
    with pytest.raises(PackageAlreadyRegistered):
        reg(repo, pool[0], "another-name")
    assert [m.name for m in repo.list_models()] == ["churn"]  # no empty model left behind
    assert event_count(job_env.engine) == 1
    assert_consistent(job_env.engine)


def test_bad_names_and_missing_packages_are_refused(
    repo: RegistryRepository, pool: list[Run]
) -> None:
    for bad in ("", "Upper", "-x", "a b", "a" * 65, "../x"):
        with pytest.raises(InvalidMove):
            reg(repo, pool[0], bad)
    with pytest.raises(PackageNotRegisterable) as missing:
        repo.register("churn", package_id=FAKE_PACKAGE, job_id=None)
    assert missing.value.details["state"] == "missing"
    with pytest.raises(PackageNotRegisterable):
        repo.register("churn", package_id="../../etc", job_id=None)
    with pytest.raises(InvalidMove):
        reg(repo, pool[0], reason="x" * 501)
    assert repo.list_models() == []


def test_open_spent_and_tampered_packages_are_refused(
    job_env: JobEnv, pool: list[Run], tmp_path: Path
) -> None:
    root = sealed_copy(pool[0], tmp_path / "w")
    repo = RegistryRepository(job_env.engine, root)
    package = root / pid(pool[0])
    model = unlock(package / "model.joblib")
    original = model.read_bytes()
    model.write_bytes(original + b"x")  # altered after sealing
    with pytest.raises(PackageNotRegisterable) as tampered:
        repo.register("churn", package_id=pid(pool[0]), job_id=None)
    assert tampered.value.details["problems"]
    model.chmod(0o444)  # read-only again: now only the content differs from the seal
    with pytest.raises(PackageNotRegisterable) as content:
        repo.register("churn", package_id=pid(pool[0]), job_id=None)
    assert any("sha" in p.lower() or "hash" in p.lower() for p in content.value.details["problems"])
    model.chmod(0o644)
    model.write_bytes(original)
    unlock(package / "seal.json").unlink()
    with pytest.raises(PackageNotRegisterable) as spent:
        repo.register("churn", package_id=pid(pool[0]), job_id=None)
    assert spent.value.details["state"] == "spent"
    unlock(package / "attempt.json").unlink()
    with pytest.raises(PackageNotRegisterable) as open_:
        repo.register("churn", package_id=pid(pool[0]), job_id=None)
    assert open_.value.details["state"] == "open"
    assert repo.list_models() == [] and event_count(job_env.engine) == 0


# -- status moves ---------------------------------------------------------------------------------


def test_every_legal_move_and_every_illegal_one(
    repo: RegistryRepository, pool: list[Run], job_env: JobEnv
) -> None:
    for run in pool[:3]:
        reg(repo, run)
    with pytest.raises(InvalidMove):  # no jump from development to production
        repo.promote("churn", 1, "production", "skip")
    repo.promote("churn", 1, "staging", "ok")
    with pytest.raises(InvalidMove):  # staging -> staging
        repo.promote("churn", 1, "staging", "again")
    repo.promote("churn", 1, "production", "ok")
    with pytest.raises(InvalidMove):  # production cannot be promoted or archived directly
        repo.promote("churn", 1, "production", "again")
    with pytest.raises(InvalidMove):
        repo.promote("churn", 1, "staging", "demote")
    with pytest.raises(InvalidMove):
        repo.archive("churn", 1, "no")
    repo.archive("churn", 2, "retire a development version")
    assert statuses(repo) == {1: "production", 2: "archived", 3: "development"}
    with pytest.raises(InvalidMove):  # an archived version cannot go straight to production
        repo.promote("churn", 2, "production", "back")
    with pytest.raises(InvalidMove):
        repo.archive("churn", 2, "twice")
    repo.promote("churn", 2, "staging", "revive")  # archived -> staging is allowed
    repo.promote("churn", 3, "staging", "ok")
    repo.archive("churn", 3, "retire a staging version")
    assert statuses(repo) == {1: "production", 2: "staging", 3: "archived"}
    with pytest.raises(VersionNotFound):
        repo.promote("churn", 99, "staging", "x")
    with pytest.raises(VersionNotFound):
        repo.archive("churn", 99, "x")
    with pytest.raises(ModelNotFound):
        repo.promote("nope", 1, "staging", "x")
    with pytest.raises(InvalidMove):
        repo.promote("churn", 2, "production", "")
    assert_consistent(job_env.engine)


def test_promoting_to_production_archives_the_previous_production_version(
    repo: RegistryRepository, pool: list[Run], job_env: JobEnv
) -> None:
    reg(repo, pool[0])
    reg(repo, pool[1])
    to_production(repo, "churn", 1)
    to_production(repo, "churn", 2)
    assert statuses(repo) == {1: "archived", 2: "production"}
    tail = [(e.seq, e.version, e.action, e.to_status) for e in repo.events("churn")][-2:]
    assert tail == [(6, 1, "supersede", "archived"), (7, 2, "promote", "production")]
    assert [e.reason for e in repo.events("churn") if e.action == "supersede"] == [
        "superseded by version 2"
    ]
    assert_consistent(job_env.engine)


def test_the_package_must_still_verify_to_be_promoted(
    job_env: JobEnv, pool: list[Run], tmp_path: Path
) -> None:
    root = sealed_copy(pool[0], tmp_path / "w")
    repo = RegistryRepository(job_env.engine, root)
    repo.register("churn", package_id=pid(pool[0]), job_id=None)
    before = event_count(job_env.engine)
    model = unlock(root / pid(pool[0]) / "model.joblib")
    model.write_bytes(model.read_bytes() + b"x")
    with pytest.raises(PackageUnverified):
        repo.promote("churn", 1, "staging", "tampered")
    (root / pid(pool[0])).chmod(0o755)
    shutil.rmtree(root / pid(pool[0]))
    with pytest.raises(PackageUnverified) as gone:
        repo.promote("churn", 1, "staging", "missing")
    assert gone.value.details["state"] == "missing"
    assert statuses(repo) == {1: "development"} and event_count(job_env.engine) == before
    assert repo.get_version("churn", 1).evaluation  # the stored summary needs no package file
    assert_consistent(job_env.engine)


# -- rollback -------------------------------------------------------------------------------------


def test_rollback_archives_the_current_version_and_restores_the_previous_one(
    repo: RegistryRepository, pool: list[Run], job_env: JobEnv
) -> None:
    reg(repo, pool[0])
    reg(repo, pool[1])
    to_production(repo, "churn", 1)
    to_production(repo, "churn", 2)
    outcome = repo.rollback("churn", "version 2 misbehaves")
    assert (outcome.restored.version, outcome.restored.status) == (1, "production")
    assert (outcome.archived.version, outcome.archived.status) == (2, "archived")
    tail = [(e.version, e.action, e.from_status, e.to_status) for e in repo.events("churn")][-2:]
    assert tail == [
        (2, "rollback", "production", "archived"),
        (1, "rollback", "archived", "production"),
    ]
    before = event_count(job_env.engine)
    with pytest.raises(NoPreviousProduction):  # version 1 was itself restored by a rollback
        repo.rollback("churn", "again")
    assert (
        statuses(repo) == {1: "production", 2: "archived"} and event_count(job_env.engine) == before
    )
    # an explicit promotion makes history again, and then it can be rolled back
    repo.promote("churn", 2, "staging", "fixed")
    repo.promote("churn", 2, "production", "fixed")
    assert statuses(repo) == {1: "archived", 2: "production"}
    assert repo.rollback("churn", "second rollback").restored.version == 1
    assert_consistent(job_env.engine)


def test_rollback_without_history_changes_nothing(
    repo: RegistryRepository, pool: list[Run], job_env: JobEnv
) -> None:
    with pytest.raises(ModelNotFound):
        repo.rollback("churn", "x")
    reg(repo, pool[0])
    with pytest.raises(NoPreviousProduction, match="no production"):
        repo.rollback("churn", "x")
    to_production(repo, "churn", 1)  # the first production version replaced nothing
    before = event_count(job_env.engine)
    with pytest.raises(NoPreviousProduction):
        repo.rollback("churn", "x")
    assert statuses(repo) == {1: "production"} and event_count(job_env.engine) == before


def test_rollback_refuses_when_the_previous_version_was_changed_or_cannot_verify(
    job_env: JobEnv, pool: list[Run], tmp_path: Path
) -> None:
    root = tmp_path / "w"
    for run in pool[:2]:
        sealed_copy(run, root)
    repo = RegistryRepository(job_env.engine, root)
    for run in pool[:2]:
        repo.register("churn", package_id=pid(run), job_id=None)
    to_production(repo, "churn", 1)
    to_production(repo, "churn", 2)
    before = event_count(job_env.engine)
    model = unlock(root / pid(pool[0]) / "model.joblib")
    model.write_bytes(model.read_bytes() + b"x")  # the version to restore no longer verifies
    with pytest.raises(PackageUnverified):
        repo.rollback("churn", "x")
    assert statuses(repo) == {1: "archived", 2: "production"}
    assert event_count(job_env.engine) == before
    assert_consistent(job_env.engine)


def test_rollback_refuses_when_the_previous_version_is_no_longer_archived(
    repo: RegistryRepository, pool: list[Run], job_env: JobEnv
) -> None:
    for run in pool[:3]:
        reg(repo, run)
    to_production(repo, "churn", 1)
    to_production(repo, "churn", 2)
    repo.promote("churn", 1, "staging", "revived")  # archived -> staging
    before = event_count(job_env.engine)
    with pytest.raises(NoPreviousProduction, match="no longer archived"):
        repo.rollback("churn", "x")
    assert statuses(repo) == {1: "staging", 2: "production", 3: "development"}
    assert event_count(job_env.engine) == before


# -- the database enforces the rules itself -------------------------------------------------------


@pytest.fixture
def three(repo: RegistryRepository, pool: list[Run]) -> RegistryRepository:
    for run in pool[:3]:
        reg(repo, run)
    return repo


def raw(engine: Engine, sql: str, **params: Any) -> Any:
    with engine.begin() as c:
        return c.execute(text(sql), params)


def version_id(engine: Engine, number: int) -> int:
    with engine.connect() as c:
        return int(
            c.execute(
                text("SELECT id FROM model_versions WHERE version = :v"), {"v": number}
            ).scalar_one()
        )


def raw_event(
    engine: Engine, number: int, action: str, old: str | None, new: str, seq: int = 1
) -> None:
    raw(
        engine,
        "INSERT INTO registry_events (model_id, version_id, seq, action, from_status, to_status, "
        "reason) SELECT model_id, id, :s, :a, :f, :t, 'raw' FROM model_versions WHERE version = :v",
        s=seq, a=action, f=old, t=new, v=number,
    )  # fmt: skip


def test_a_status_cannot_change_without_its_event(
    three: RegistryRepository, job_env: JobEnv
) -> None:
    for sql in (
        "UPDATE model_versions SET status = 'production' WHERE version = 1",
        "UPDATE model_versions SET status = 'archived'",
    ):
        with pytest.raises(IntegrityError, match="registry event"):
            raw(job_env.engine, sql)
    assert statuses(three) == {1: "development", 2: "development", 3: "development"}


def test_an_event_applies_its_own_move_and_the_number_is_allocated_by_the_database(
    three: RegistryRepository, job_env: JobEnv
) -> None:
    raw_event(job_env.engine, 1, "promote", "development", "staging", seq=999)
    assert statuses(three)[1] == "staging"
    assert [e.seq for e in three.events("churn")] == [1, 2, 3, 4]  # the supplied 999 was ignored
    assert_consistent(job_env.engine)


def test_an_illegal_event_changes_nothing_and_leaves_no_event(
    three: RegistryRepository, job_env: JobEnv
) -> None:
    before = event_count(job_env.engine)
    illegal = [
        (1, "promote", "development", "production"),  # skipping staging
        (1, "promote", "staging", "production"),  # wrong from-status
        (1, "archive", "development", "production"),  # archive means archived
        (1, "supersede", "development", "archived"),  # supersede needs production
        (1, "rollback", "development", "archived"),
        (1, "register", None, "development"),  # already registered
        (1, "promote", "development", "archived"),
    ]
    for number, action, old, new in illegal:
        with pytest.raises(IntegrityError):
            raw_event(job_env.engine, number, action, old, new)
    assert event_count(job_env.engine) == before
    assert statuses(three) == {1: "development", 2: "development", 3: "development"}
    assert_consistent(job_env.engine)


def test_the_database_allows_one_production_version_per_model(
    three: RegistryRepository, job_env: JobEnv
) -> None:
    to_production(three, "churn", 1)
    raw_event(job_env.engine, 2, "promote", "development", "staging")
    with pytest.raises(IntegrityError, match="uq_model_versions_one_production"):
        raw_event(job_env.engine, 2, "promote", "staging", "production")
    assert statuses(three) == {1: "production", 2: "staging", 3: "development"}
    assert_consistent(job_env.engine)


def test_a_raw_rollback_restore_needs_the_matching_archive_just_before_it(
    three: RegistryRepository, job_env: JobEnv
) -> None:
    to_production(three, "churn", 1)
    to_production(three, "churn", 2)  # version 1 is archived
    raw_event(job_env.engine, 2, "supersede", "production", "archived")
    with pytest.raises(IntegrityError, match="rollback"):
        raw_event(job_env.engine, 1, "rollback", "archived", "production")
    with pytest.raises(IntegrityError):  # an archived version never returns by a plain promotion
        raw_event(job_env.engine, 1, "promote", "archived", "production")
    assert statuses(three)[1] == "archived"


def test_events_are_append_only(three: RegistryRepository, job_env: JobEnv) -> None:
    for sql in (
        "UPDATE registry_events SET reason = 'edited'",
        "DELETE FROM registry_events",
        "TRUNCATE registry_events",
        "UPDATE registry_events SET seq = seq + 100",
    ):
        with pytest.raises(DBAPIError, match="append-only"):
            raw(job_env.engine, sql)
    assert event_count(job_env.engine) == 3


def test_a_version_cannot_exist_without_its_register_event(
    three: RegistryRepository, job_env: JobEnv
) -> None:
    insert = (
        "INSERT INTO model_versions (model_id, version, package_id, algorithm, task, target, "
        "selection_sha256, evaluation) SELECT id, 99, :p, 'a', 'classification', 'y', :s, "
        "'{}'::jsonb FROM registered_models WHERE name = 'churn'"
    )
    with pytest.raises(DBAPIError, match="no register event"):
        raw(job_env.engine, insert, p=FAKE_PACKAGE, s=FAKE_SHA)
    production = insert.replace("evaluation)", "evaluation, status)").replace(
        "'{}'::jsonb FROM", "'{}'::jsonb, 'production' FROM"
    )
    with pytest.raises(DBAPIError, match="starts in development"):
        raw(job_env.engine, production, p=FAKE_PACKAGE, s=FAKE_SHA)
    assert [v.version for v in three.list_versions("churn")] == [3, 2, 1]


def test_versions_and_models_are_immutable_and_never_deleted(
    three: RegistryRepository, job_env: JobEnv
) -> None:
    for sql, message in (
        ("UPDATE model_versions SET package_id = :p WHERE version = 1", "immutable"),
        ("UPDATE model_versions SET evaluation = '{}'::jsonb", "immutable"),
        ("UPDATE model_versions SET version = 50 WHERE version = 1", "immutable"),
        ("UPDATE model_versions SET algorithm = 'x'", "immutable"),
        ("DELETE FROM model_versions", "never deleted"),
        ("DELETE FROM registered_models", "never deleted"),
        ("UPDATE registered_models SET name = 'renamed'", "event number"),
        ("UPDATE registered_models SET last_event_seq = last_event_seq + 5", "event number"),
        ("UPDATE registered_models SET last_event_seq = 0", "event number"),
    ):
        with pytest.raises(DBAPIError, match=message):
            raw(job_env.engine, sql, p=FAKE_PACKAGE)
    assert event_count(job_env.engine) == 3
    assert_consistent(job_env.engine)


def test_a_failed_change_rolls_back_completely(
    three: RegistryRepository, job_env: JobEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    to_production(three, "churn", 1)
    three.promote("churn", 2, "staging", "ok")
    before = (statuses(three), event_count(job_env.engine))
    original = registry_module._event

    def failing(connection: Any, model_id: int, vid: int, action: str, *rest: Any) -> None:
        if action == "promote" and rest[1] == "production":
            raise RuntimeError("simulated failure after the supersede event")
        original(connection, model_id, vid, action, *rest)

    monkeypatch.setattr(registry_module, "_event", failing)
    with pytest.raises(RuntimeError):
        three.promote("churn", 2, "production", "x")  # the supersede event was written first
    monkeypatch.undo()
    assert (statuses(three), event_count(job_env.engine)) == before
    assert_consistent(job_env.engine)


def test_migration_0003_goes_down_and_up_again(job_env: JobEnv) -> None:
    from automl_ops.db import migrate

    down = migrate.downgrade(job_env.engine, "0002")
    assert down.current == "0002"
    with job_env.engine.connect() as c:
        left = {
            r[0]
            for r in c.execute(
                text(
                    "SELECT proname FROM pg_proc WHERE proname LIKE 'registry%' "
                    "OR proname LIKE 'model_versions%' OR proname LIKE 'registered%'"
                )
            )
        }
    assert left == set()
    assert migrate.upgrade(job_env.engine).current == "0007"
    assert migrate.check(job_env.engine).ok


# -- concurrency ----------------------------------------------------------------------------------


def run_all(tasks: Sequence[Callable[[], Any]]) -> list[Any]:
    barrier = threading.Barrier(len(tasks))

    def call(task: Callable[[], Any]) -> Any:
        barrier.wait(timeout=30)
        try:
            return task()
        except Exception as error:
            return error

    with ThreadPoolExecutor(len(tasks)) as executor:
        return list(executor.map(call, tasks))


def test_concurrent_promotions_never_leave_two_production_versions(
    three: RegistryRepository, job_env: JobEnv
) -> None:
    for n in (1, 2, 3):
        three.promote("churn", n, "staging", "ready")
    results = run_all(
        [partial(three.promote, "churn", n, "production", "go") for n in (1, 2, 3)] * 3
    )
    assert not [r for r in results if isinstance(r, Exception) and not isinstance(r, InvalidMove)]
    assert sum(1 for s in statuses(three).values() if s == "production") == 1
    assert_consistent(job_env.engine)


@pytest.mark.parametrize("attempt", range(5))
def test_a_rollback_racing_a_promotion_leaves_consistent_history(
    attempt: int, repo: RegistryRepository, pool: list[Run], job_env: JobEnv
) -> None:
    for run in pool[:3]:
        reg(repo, run)
    to_production(repo, "churn", 1)
    to_production(repo, "churn", 2)
    repo.promote("churn", 3, "staging", "ready")
    results = run_all(
        [
            lambda: repo.rollback("churn", "race"),
            lambda: repo.promote("churn", 3, "production", "race"),
            lambda: repo.rollback("churn", "race"),
        ]
    )
    expected = (InvalidMove, NoPreviousProduction)
    assert not [r for r in results if isinstance(r, Exception) and not isinstance(r, expected)]
    assert sum(1 for s in statuses(repo).values() if s == "production") <= 1
    assert_consistent(job_env.engine)


def test_concurrent_registration_of_one_package_succeeds_once(
    repo: RegistryRepository, pool: list[Run], job_env: JobEnv
) -> None:
    results = run_all([partial(reg, repo, pool[0], f"model-{i}") for i in range(6)])
    assert sum(1 for r in results if not isinstance(r, Exception)) == 1
    assert all(isinstance(r, PackageAlreadyRegistered) for r in results if isinstance(r, Exception))
    assert len(repo.list_models()) == 1  # the losers left no empty models
    assert_consistent(job_env.engine)


def test_concurrent_registrations_into_one_model_get_distinct_numbers_and_events(
    repo: RegistryRepository, pool: list[Run], job_env: JobEnv
) -> None:
    results = run_all([partial(reg, repo, run) for run in pool])
    assert not [r for r in results if isinstance(r, Exception)]
    assert sorted(v.version for v in results) == [1, 2, 3, 4]
    assert [e.seq for e in repo.events("churn")] == [1, 2, 3, 4]
    assert_consistent(job_env.engine)


# -- the held-out test part is never touched ------------------------------------------------------


def test_registry_code_does_not_import_anything_that_reads_data_or_the_test_part() -> None:
    forbidden = {
        "automl_ops.finalize", "automl_ops.train", "automl_ops.selection", "automl_ops.runner",
        "automl_ops.dataset_store", "automl_ops.profile", "automl_ops.tracking", "mlflow",
        "pandas", "numpy", "sklearn", "joblib",
    }  # fmt: skip
    for module in (registry_module, __import__("automl_ops.api_control.registry", fromlist=["x"])):
        tree = ast.parse(Path(module.__file__ or "").read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
        assert not names & forbidden, (module.__name__, names & forbidden)


def test_registering_and_promoting_never_reads_a_dataset_or_the_test_part(
    repo: RegistryRepository, pool: list[Run], monkeypatch: pytest.MonkeyPatch
) -> None:
    from automl_ops import dataset_store, finalize, profile

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the registry must never read dataset rows or the test part")

    monkeypatch.setattr(finalize, "_read_test_part", forbidden)
    monkeypatch.setattr(profile, "read_version_frame", forbidden)
    monkeypatch.setattr(dataset_store.FileDatasetStore, "read_bytes", forbidden, raising=False)
    reg(repo, pool[0])
    to_production(repo, "churn", 1)
    repo.get_version("churn", 1)
    repo.events("churn")
