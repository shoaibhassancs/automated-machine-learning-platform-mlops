# ruff: noqa: F811
"""P4-6: the MLflow mirror of the registry, on a real PostgreSQL server with real MLflow stores.

Throwaway `automl_test_*` databases (one for the registry, one cloned MLflow database per test) and
short scratch artifact folders. The original archive `data/mlflow` is only read (fingerprinted) to
prove it is never written. NOT RUN when the server is unreachable.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from mlflow import MlflowClient
from sqlalchemy import Engine, text
from sqlalchemy.exc import DBAPIError

from automl_ops import registry_mirror as mirror_module
from automl_ops import tracking as tracking_module
from automl_ops.job_tracking import archive_root
from automl_ops.registry import RegistryRepository
from automl_ops.registry_mirror import (
    InvalidReason,
    Mirror,
    MirrorDisabled,
    MirrorSettingError,
    classify,
    mirror_enabled,
)
from automl_ops.tracking import (
    MIRROR_PACKAGE_TAG,
    JobNotFoundError,
    MirrorError,
    TrackingStore,
    mirror_snapshot,
    run_ids,
)
from conftest import JobEnv, Run
from test_registry_postgres import pid, pool, sealed_copy, unlock  # noqa: F401
from test_tracking_postgres import mlflow_template, pg_clone, pg_store, server  # noqa: F401

pytestmark = pytest.mark.postgres

NAME = "churn"


@dataclass(frozen=True, eq=False)
class Rig:
    engine: Engine
    repo: RegistryRepository
    mirror: Mirror
    store: TrackingStore
    runs: list[Run]
    models: Path

    def client(self) -> MlflowClient:
        return MlflowClient(tracking_uri=self.store.tracking_uri)

    def register(self, number: int) -> None:
        self.repo.register(NAME, package_id=pid(self.runs[number]), job_id=None)

    def promote(self, index: int, to: str = "production") -> None:
        """Promote the version made from `runs[index]` (registry version `index + 1`)."""
        number = index + 1
        if to == "production":
            self.repo.promote(NAME, number, "staging", "ready")
        self.repo.promote(NAME, number, to, "go")  # type: ignore[arg-type]

    def state(self) -> Any:
        return mirror_snapshot(self.store, NAME)

    def mlflow_of(self, number: int) -> str:
        package = pid(self.runs[number])
        return str(next(v.mlflow_version for v in self.state().versions if v.package_id == package))

    def log(self) -> list[dict[str, Any]]:
        with self.engine.connect() as c:
            rows = c.execute(
                text(
                    "SELECT kind, outcome, seq, reason, details FROM registry_sync_log ORDER BY id"
                )
            ).all()
        return [dict(r._mapping) for r in rows]

    def progress(self) -> dict[str, Any]:
        with self.engine.connect() as c:
            row = c.execute(
                text(
                    "SELECT s.synced_seq, s.attempts, s.last_error_kind, m.last_event_seq "
                    "FROM registry_sync s JOIN registered_models m ON m.id = s.model_id"
                )
            ).first()
        return {} if row is None else dict(row._mapping)

    def clean(self) -> bool:
        findings = self.mirror.check(NAME).findings
        return not any(f.stops_sync or f.category == "pending" for f in findings)


@pytest.fixture
def rig(job_env: JobEnv, pool: list[Run], pg_store: TrackingStore) -> Rig:  # noqa: F811
    models = pool[0].root
    return Rig(
        job_env.engine,
        RegistryRepository(job_env.engine, models),
        Mirror(job_env.engine, models, enabled=lambda: True, store_factory=lambda: pg_store),
        pg_store,
        pool,
        models,
    )


def sha_of(path: Path) -> dict[str, str]:
    return {
        str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(path.rglob("*"))
        if f.is_file()
    }


def tree_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(f"{f.relative_to(root)}|{f.stat().st_size}|".encode())
        digest.update(hashlib.sha256(f.read_bytes()).digest())
    return digest.hexdigest()


# -- initial and incremental sync -----------------------------------------------------------------


def test_the_first_sync_mirrors_versions_tags_alias_and_package_files(rig: Rig) -> None:
    rig.register(0)
    rig.register(1)
    rig.promote(1)
    outcome = rig.mirror.sync(NAME)
    assert outcome.result == "synced" and outcome.error is None
    assert {a["action"] for a in outcome.actions} >= {"create_model", "create_version", "set_alias"}
    snapshot = rig.state()
    assert snapshot.exists and snapshot.owned and len(snapshot.versions) == 2
    by_number = {v.registry_version: v for v in snapshot.versions}
    assert by_number["1"].status == "development" and by_number["2"].status == "production"
    assert by_number["1"].package_id == pid(rig.runs[0])
    assert snapshot.aliases == {"production": by_number["2"].mlflow_version}
    assert all(v.source_ok and v.source.startswith("runs:/") for v in snapshot.versions)
    # the mirrored version's source holds exactly the sealed package files (hash for hash)
    run_id = by_number["2"].source.split("/")[1]
    downloaded = Path(
        rig.client().download_artifacts(run_id, "package", str(rig.models.parent / "dl"))
    )
    assert sha_of(downloaded) == sha_of(rig.models / pid(rig.runs[1]))
    status = rig.mirror.status(NAME)
    assert (status.state, status.pending_events) == ("in_sync", 0)
    assert rig.progress()["synced_seq"] == rig.progress()["last_event_seq"]
    assert rig.clean()
    assert [r["kind"] for r in rig.log()] == ["sync"] and rig.log()[0]["outcome"] == "ok"


def test_a_second_sync_with_nothing_new_makes_no_mlflow_call(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig.register(0)
    assert rig.mirror.sync(NAME).result == "synced"
    log_size = len(rig.log())

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("an up-to-date model must not touch MLflow")

    monkeypatch.setattr("automl_ops.registry_mirror.mirror_snapshot", forbidden)
    assert rig.mirror.sync(NAME).result == "up_to_date"
    assert len(rig.log()) == log_size  # a no-op pass is not logged


def test_incremental_sync_and_the_alias_follows_a_rollback(rig: Rig) -> None:
    rig.register(0)
    rig.register(1)
    rig.promote(0)
    assert rig.mirror.sync(NAME).result == "synced"
    first = rig.mlflow_of(0)
    assert rig.state().aliases == {"production": first}

    rig.promote(1)  # version 1 supersedes version 0
    assert rig.mirror.status(NAME).state == "pending"
    assert rig.mirror.sync(NAME).result == "synced"
    second = rig.mlflow_of(1)
    snapshot = rig.state()
    assert snapshot.aliases == {"production": second}
    tags = {v.registry_version: v.status for v in snapshot.versions}
    assert tags == {"1": "archived", "2": "production"}

    rig.repo.rollback(NAME, "version 2 misbehaves")  # one PostgreSQL transaction, one alias move
    assert rig.mirror.sync(NAME).result == "synced"
    snapshot = rig.state()
    assert snapshot.aliases == {"production": first}
    assert {v.registry_version: v.status for v in snapshot.versions} == {
        "1": "production",
        "2": "archived",
    }
    assert rig.clean() and rig.mirror.status(NAME).state == "in_sync"


# -- outage, interruption, events during a pass ---------------------------------------------------


def test_an_mlflow_outage_never_blocks_or_undoes_the_registry_and_recovery_catches_up(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig.register(0)
    rig.register(1)
    assert rig.mirror.sync(NAME).result == "synced"
    real = tracking_module.mirror_snapshot

    def down(*args: object, **kwargs: object) -> None:
        raise MirrorError("MLflow is unreachable (simulated)")

    monkeypatch.setattr("automl_ops.registry_mirror.mirror_snapshot", down)
    rig.promote(1)  # a registry operation succeeds while MLflow is down
    assert rig.repo.get_version(NAME, 2).status == "production"
    failed = rig.mirror.sync(NAME)
    assert failed.result == "failed" and "unreachable" in (failed.error or "")
    status = rig.mirror.status(NAME)
    assert (status.state, status.attempts, status.last_error_kind) == ("failed", 1, "error")
    assert status.pending_events == 2 and status.next_attempt_at is not None
    assert rig.progress()["synced_seq"] < rig.progress()["last_event_seq"]  # not advanced
    again = rig.mirror.sync(NAME)
    assert again.result == "failed" and rig.mirror.status(NAME).attempts == 2
    assert [r["outcome"] for r in rig.log()][-2:] == ["failed", "failed"]

    monkeypatch.setattr("automl_ops.registry_mirror.mirror_snapshot", real)  # MLflow is back
    with rig.engine.begin() as c:  # the retry time has come
        c.execute(text("UPDATE registry_sync SET next_attempt_at = now() - interval '1 second'"))
    assert rig.mirror.run_pending() == {NAME: "synced"}
    assert rig.mirror.status(NAME).state == "in_sync" and rig.clean()
    assert rig.mirror.status(NAME).attempts == 0 and rig.mirror.status(NAME).last_error is None


@pytest.mark.parametrize("break_at", ["mirror_set_tags", "mirror_set_production_alias"])
def test_an_interrupted_pass_resumes_without_duplicates(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, break_at: str
) -> None:
    rig.register(0)
    rig.register(1)
    rig.promote(0)
    real = getattr(mirror_module, break_at)
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise MirrorError("process died here (simulated)")
        return real(*args, **kwargs)

    monkeypatch.setattr(mirror_module, break_at, flaky)
    first = rig.mirror.sync(NAME)
    assert first.result == "failed"
    assert rig.progress()["synced_seq"] == 0  # nothing was claimed as synced
    assert rig.mirror.status(NAME).state == "failed"
    partial = rig.state()
    assert partial.exists and len(partial.versions) >= 1  # the interrupted pass got part-way
    monkeypatch.setattr(mirror_module, break_at, real)
    second = rig.mirror.sync(NAME)
    assert second.result == "synced"
    snapshot = rig.state()
    assert sorted(v.package_id for v in snapshot.versions) == sorted(
        [pid(rig.runs[0]), pid(rig.runs[1])]
    )  # exactly one MLflow version per package
    assert snapshot.aliases == {"production": rig.mlflow_of(0)}
    assert rig.clean()


def test_events_arriving_during_a_pass_stay_pending(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig.register(0)
    rig.register(1)
    rig.promote(0)
    real = tracking_module.mirror_set_production_alias
    fired = {"done": False}

    def with_a_change_first(*args: Any, **kwargs: Any) -> Any:
        if not fired["done"]:
            fired["done"] = True
            rig.promote(1)  # a new registry event lands in the middle of the pass
        return real(*args, **kwargs)

    monkeypatch.setattr(
        "automl_ops.registry_mirror.mirror_set_production_alias", with_a_change_first
    )
    outcome = rig.mirror.sync(NAME)
    assert outcome.result == "synced"
    progress = rig.progress()
    assert progress["synced_seq"] == outcome.seq < progress["last_event_seq"]
    status = rig.mirror.status(NAME)
    assert status.state == "pending" and status.pending_events == 3
    assert rig.state().aliases == {"production": rig.mlflow_of(0)}  # the state as of the pass
    monkeypatch.setattr("automl_ops.registry_mirror.mirror_set_production_alias", real)
    assert rig.mirror.sync(NAME).result == "synced"
    assert rig.state().aliases == {"production": rig.mlflow_of(1)}
    assert rig.mirror.status(NAME).state == "in_sync" and rig.clean()


def test_concurrent_syncers_never_interfere(rig: Rig) -> None:
    rig.register(0)
    rig.register(1)
    rig.promote(1)
    others = [
        Mirror(rig.engine, rig.models, enabled=lambda: True, store_factory=lambda: rig.store)
        for _ in range(5)
    ]
    mirrors = [rig.mirror, *others]
    barrier = threading.Barrier(len(mirrors))

    def go(m: Mirror) -> str:
        barrier.wait(timeout=30)
        return m.sync(NAME).result

    with ThreadPoolExecutor(len(mirrors)) as pool_:
        results = list(pool_.map(go, mirrors))
    assert set(results) <= {"synced", "busy", "up_to_date"} and "synced" in results
    snapshot = rig.state()
    assert len(snapshot.versions) == 2  # no duplicate versions
    assert snapshot.aliases == {"production": rig.mlflow_of(1)}
    assert rig.clean() and rig.mirror.status(NAME).state == "in_sync"


def test_the_advisory_lock_means_a_second_syncer_skips_and_a_crash_frees_it(rig: Rig) -> None:
    rig.register(0)
    with rig.engine.connect() as c:
        model_id = c.execute(text("SELECT id FROM registered_models")).scalar_one()
    holder = rig.engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    assert holder.execute(
        text("SELECT pg_try_advisory_lock(:c, :k)"),
        {"c": mirror_module.LOCK_CLASS, "k": model_id},
    ).scalar_one()
    try:
        assert rig.mirror.sync(NAME).result == "busy"
        assert rig.mirror.status(NAME).synced_seq == 0
    finally:
        holder.invalidate()  # the holder "crashes": its session ends and the lock goes with it
        holder.close()
    assert rig.mirror.sync(NAME).result == "synced"


def test_pending_work_survives_a_restart_and_is_found_by_a_new_instance(
    job_env: JobEnv,
    pool: list[Run],
    pg_store: TrackingStore,  # noqa: F811
) -> None:
    models = pool[0].root
    repo = RegistryRepository(job_env.engine, models)
    repo.register(NAME, package_id=pid(pool[0]), job_id=None)
    down = Mirror(job_env.engine, models, enabled=lambda: False)  # the first "API": mirror off
    assert down.run_pending() == {}
    fresh = Mirror(job_env.engine, models, enabled=lambda: True, store_factory=lambda: pg_store)
    assert fresh.status(NAME).pending_events == 1  # derived from persisted state only
    assert fresh.run_pending() == {NAME: "synced"}
    assert fresh.status(NAME).state == "in_sync"


# -- conflicts, drift and reconcile ---------------------------------------------------------------


def test_a_duplicate_version_is_reported_as_a_conflict_and_nothing_is_deleted(rig: Rig) -> None:
    rig.register(0)
    assert rig.mirror.sync(NAME).result == "synced"
    only = rig.state().versions[0]
    run_id = only.source.split("/")[1]
    rig.client().create_model_version(
        NAME, only.source, run_id=run_id, tags={MIRROR_PACKAGE_TAG: pid(rig.runs[0])}
    )
    rig.promote(0, "staging")
    report = rig.mirror.check(NAME)
    assert [f.kind for f in report.findings if f.category == "conflict"] == ["version_duplicate"]
    synced = rig.mirror.sync(NAME)
    assert synced.result == "conflict" and rig.mirror.status(NAME).state == "conflict"
    assert len(rig.state().versions) == 2  # neither was deleted or changed
    done = rig.mirror.reconcile(NAME, "try to repair a duplicate")
    assert done.outcome == "conflict" and done.actions == []
    assert len(rig.state().versions) == 2
    kinds = [(r["kind"], r["outcome"]) for r in rig.log()]
    assert kinds[-2:] == [("sync", "conflict"), ("reconcile", "conflict")]


def test_manual_drift_is_detected_then_repaired_only_by_an_explicit_reconcile(rig: Rig) -> None:
    rig.register(0)
    rig.register(1)
    rig.promote(0)
    assert rig.mirror.sync(NAME).result == "synced"
    one, two = rig.mlflow_of(0), rig.mlflow_of(1)
    client = rig.client()
    client.set_registered_model_alias(NAME, "production", two)  # someone moves the alias ...
    client.set_model_version_tag(NAME, two, "platform.status", "archived")  # ... and edits a tag
    before_check = (rig.progress(), len(rig.log()), rig.state())

    report = rig.mirror.check(NAME)
    drift = {f.kind for f in report.findings if f.category == "drift"}
    assert drift == {"alias_wrong", "tag_status"}
    assert (rig.progress(), len(rig.log()), rig.state()) == before_check  # check is read-only

    rig.promote(1)  # a registry change makes the model pending as well
    blocked = rig.mirror.sync(NAME)
    assert blocked.result == "drift" and rig.mirror.status(NAME).state == "drifted"
    assert rig.state().aliases == {"production": two}  # the sync changed nothing
    assert {v.mlflow_version: v.status for v in rig.state().versions}[two] == "archived"
    assert rig.log()[-1]["outcome"] == "drift"

    with pytest.raises(InvalidReason):
        rig.mirror.reconcile(NAME, "   ")
    done = rig.mirror.reconcile(NAME, "alias and tag edited by hand")
    assert done.outcome == "ok" and done.error is None
    assert {f.kind for f in done.detected} >= {"tag_status"}
    assert any("production alias" in line or "platform.status" in line or "tag_status" in line
               for line in done.intended)  # fmt: skip
    assert {a["action"] for a in done.actions} >= {"set_tags"}
    snapshot = rig.state()
    assert snapshot.aliases == {"production": two}
    assert {v.mlflow_version: v.status for v in snapshot.versions} == {
        one: "archived",
        two: "production",
    }
    assert rig.clean() and rig.mirror.status(NAME).state == "in_sync"
    audit = rig.log()[-1]
    assert (audit["kind"], audit["outcome"], audit["reason"]) == (
        "reconcile",
        "ok",
        "alias and tag edited by hand",
    )
    assert set(audit["details"]) == {"detected", "intended", "actions", "result"}
    assert audit["details"]["detected"]["total"] >= 1 and audit["details"]["actions"]


def test_a_reconcile_of_a_deleted_version_recreates_it_never_deletes(rig: Rig) -> None:
    rig.register(0)
    rig.register(1)
    assert rig.mirror.sync(NAME).result == "synced"
    rig.client().delete_model_version(NAME, rig.mlflow_of(1))  # someone deletes a version
    report = rig.mirror.check(NAME)
    assert [(f.kind, f.category) for f in report.findings] == [("version_missing", "drift")]
    assert rig.mirror.sync(NAME).result in (
        "up_to_date",
    )  # nothing pending: only check/verify see it
    assert rig.mirror.verify(NAME)
    assert rig.mirror.status(NAME).state == "drifted"
    assert rig.mirror.reconcile(NAME, "version was deleted by hand").outcome == "ok"
    assert len(rig.state().versions) == 2 and rig.mirror.status(NAME).state == "in_sync"


def test_foreign_models_aliases_and_versions_are_never_touched(rig: Rig) -> None:
    client = rig.client()
    client.create_registered_model(NAME)  # a foreign model that happens to have our name
    client.set_registered_model_tag(NAME, "owner", "somebody-else")
    client.create_registered_model("other-team", tags={"owner": "them"})
    rig.register(0)
    foreign_before = (dict(client.get_registered_model(NAME).tags), len(rig.state().versions))
    out = rig.mirror.sync(NAME)
    assert out.result == "conflict" and [f.kind for f in out.findings] == ["model_foreign"]
    assert rig.mirror.reconcile(NAME, "take over the foreign model?").outcome == "conflict"
    assert (
        dict(client.get_registered_model(NAME).tags),
        len(rig.state().versions),
    ) == foreign_before
    assert not rig.state().owned and rig.state().aliases == {}
    assert dict(client.get_registered_model("other-team").tags) == {"owner": "them"}


def test_unrelated_aliases_and_foreign_holders_are_reported_not_removed(rig: Rig) -> None:
    rig.register(0)
    rig.register(1)
    rig.promote(0)
    assert rig.mirror.sync(NAME).result == "synced"
    client = rig.client()
    client.set_registered_model_alias(NAME, "champion", rig.mlflow_of(1))  # not ours
    extra = client.create_model_version(
        NAME, "runs:/" + rig.state().versions[0].source.split("/")[1] + "/package"
    )
    report = rig.mirror.check(NAME)
    assert {(f.kind, f.blocking) for f in report.findings if f.category == "conflict"} == {
        ("alias_foreign", False),
        ("orphan_version", False),
    }
    rig.promote(1)
    assert rig.mirror.sync(NAME).result == "synced"  # informational conflicts do not stop a sync
    rig.mirror.reconcile(NAME, "routine")
    assert rig.state().aliases["champion"] == rig.mlflow_of(1)  # still there
    assert any(v.mlflow_version == str(extra.version) for v in rig.state().versions)
    # the production alias held by a version the platform does not own is never moved
    client.set_registered_model_alias(NAME, "production", str(extra.version))
    rig.repo.rollback(NAME, "back")
    blocked = rig.mirror.sync(NAME)
    assert blocked.result == "conflict"
    assert [f.kind for f in blocked.findings if f.stops_sync] == ["alias_foreign_holder"]
    assert rig.mirror.reconcile(NAME, "foreign holder").outcome == "conflict"
    assert rig.state().aliases["production"] == str(extra.version)


def test_mlflow_models_versions_and_runs_are_never_deleted(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(name: str) -> Callable[..., Any]:
        def blocked(*args: object, **kwargs: object) -> None:
            raise AssertionError(f"the mirror must never call MlflowClient.{name}")

        return blocked

    for forbidden in (
        "delete_registered_model",
        "delete_model_version",
        "delete_run",
        "delete_experiment",
        "delete_registered_model_tag",
        "delete_model_version_tag",
    ):
        monkeypatch.setattr(MlflowClient, forbidden, refuse(forbidden))
    real_delete = MlflowClient.delete_registered_model_alias
    deleted: list[str] = []

    def alias_delete(self: Any, name: str, alias: str) -> Any:
        deleted.append(alias)
        return real_delete(self, name, alias)

    monkeypatch.setattr(MlflowClient, "delete_registered_model_alias", alias_delete)
    for n in (0, 1):
        rig.register(n)
    rig.promote(0)
    rig.mirror.sync(NAME)
    rig.promote(1)
    rig.mirror.sync(NAME)
    rig.repo.rollback(NAME, "back")
    rig.mirror.sync(NAME)
    rig.mirror.reconcile(NAME, "routine")
    assert set(deleted) <= {"production"}


# -- packages and settings ------------------------------------------------------------------------


def test_the_package_is_tracked_automatically_and_a_broken_one_fails_only_the_mirror(
    job_env: JobEnv, pool: list[Run], pg_store: TrackingStore, tmp_path: Path
) -> None:
    root = sealed_copy(pool[0], tmp_path / "w")
    sealed_copy(pool[1], tmp_path / "w")
    repo = RegistryRepository(job_env.engine, root)
    mirror = Mirror(job_env.engine, root, enabled=lambda: True, store_factory=lambda: pg_store)
    repo.register(NAME, package_id=pid(pool[0]), job_id=None)
    with pytest.raises(JobNotFoundError):  # nothing is tracked yet
        run_ids(pid(pool[0]), pg_store)
    assert mirror.sync(NAME).result == "synced"
    assert run_ids(pid(pool[0]), pg_store)["state"] == "complete"  # tracked by the mirror itself

    repo.register(NAME, package_id=pid(pool[1]), job_id=None)  # a second, still untracked package
    model = unlock(root / pid(pool[1]) / "model.joblib")
    model.write_bytes(model.read_bytes() + b"x")  # it no longer verifies against its seal
    broken = mirror.sync(NAME)
    assert broken.result == "failed" and "failed verification" in (broken.error or "")
    assert root.name == "w" and repo.get_version(NAME, 2).status == "development"  # registry intact
    status = mirror.status(NAME)
    assert status.state == "failed" and status.pending_events == 1
    snapshot = mirror_snapshot(pg_store, NAME)
    assert [v.package_id for v in snapshot.versions] == [pid(pool[0])]  # nothing half-mirrored
    with pytest.raises(JobNotFoundError):
        run_ids(pid(pool[1]), pg_store)  # a failed package is never tracked


def test_a_sync_never_reads_the_test_part_a_model_or_dataset_rows(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from automl_ops import dataset_store, finalize, package, profile

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the mirror must not read data, the test part or unpickle a model")

    monkeypatch.setattr(finalize, "_read_test_part", forbidden)
    monkeypatch.setattr(profile, "read_version_frame", forbidden)
    monkeypatch.setattr(package, "load_package", forbidden)
    monkeypatch.setattr(package, "load_pipeline_for_test", forbidden)
    monkeypatch.setattr(dataset_store.FileDatasetStore, "read_bytes", forbidden, raising=False)
    rig.register(0)
    rig.promote(0)
    assert rig.mirror.sync(NAME).result == "synced" and rig.clean()


def test_the_mirror_is_opt_in_and_disabled_changes_nothing(
    job_env: JobEnv,
    pool: list[Run],
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    monkeypatch.delenv("AUTOML_REGISTRY_MIRROR", raising=False)
    monkeypatch.setenv("AUTOML_REGISTRY_MIRROR", "")
    assert mirror_enabled() is False
    models = pool[0].root
    mirror = Mirror(job_env.engine, models)
    repo = RegistryRepository(job_env.engine, models)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a disabled mirror must not touch MLflow")

    monkeypatch.setattr("automl_ops.registry_mirror.mirror_snapshot", forbidden)
    monkeypatch.setattr("automl_ops.registry_mirror.mirror_ensure_model", forbidden)
    version = repo.register(NAME, package_id=pid(pool[0]), job_id=None)  # P4-5 behaviour intact
    repo.promote(NAME, version.version, "staging", "ok")
    assert repo.get_version(NAME, 1).status == "staging"
    assert mirror.status(NAME).state == "disabled" and not mirror.enabled
    assert mirror.run_pending() == {} and mirror.sync_soon(NAME) is None
    for call in (lambda: mirror.sync(NAME), lambda: mirror.check(NAME)):
        with pytest.raises(MirrorDisabled):
            call()
    with pytest.raises(MirrorDisabled):
        mirror.reconcile(NAME, "x")
    with job_env.engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM registry_sync")).scalar_one() == 0
        assert c.execute(text("SELECT count(*) FROM registry_sync_log")).scalar_one() == 0
    monkeypatch.setenv("AUTOML_REGISTRY_MIRROR", "maybe")
    with pytest.raises(MirrorSettingError):
        mirror_enabled()
    for value, expected in (("on", True), ("TRUE", True), ("off", False), ("0", False)):
        monkeypatch.setenv("AUTOML_REGISTRY_MIRROR", value)
        assert mirror_enabled() is expected


def test_the_archive_is_refused_and_a_separate_sqlite_directory_works(
    job_env: JobEnv,
    pool: list[Run],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    archive = archive_root()
    before = tree_fingerprint(archive)
    monkeypatch.setenv("AUTOML_REGISTRY_MIRROR", "on")
    monkeypatch.setenv("AUTOML_MLFLOW_BACKEND", "sqlite")
    monkeypatch.delenv("AUTOML_JOBS_MLFLOW_DIR", raising=False)
    models = pool[0].root
    repo = RegistryRepository(job_env.engine, models)
    repo.register(NAME, package_id=pid(pool[0]), job_id=None)
    mirror = Mirror(job_env.engine, models)
    assert mirror.enabled and "AUTOML_JOBS_MLFLOW_DIR" in (mirror.problem() or "")
    refused = mirror.sync(NAME)
    assert refused.result == "failed" and "may not use" in (refused.error or "")
    for forbidden_dir in (archive, archive / "artifacts", archive / "sub"):
        monkeypatch.setenv("AUTOML_JOBS_MLFLOW_DIR", str(forbidden_dir))
        assert mirror.sync(NAME).result == "failed"
        assert "may not use" in (mirror.status(NAME).last_error or "")
    link = tmp_path / "link-to-archive"
    link.symlink_to(archive)
    monkeypatch.setenv("AUTOML_JOBS_MLFLOW_DIR", str(link))
    assert mirror.sync(NAME).result == "failed"
    assert tree_fingerprint(archive) == before  # not a byte written, not a file created

    separate = tmp_path / "separate-sqlite"
    monkeypatch.setenv("AUTOML_JOBS_MLFLOW_DIR", str(separate))
    assert mirror.problem() is None
    done = mirror.sync(NAME)
    assert done.result == "synced", done.error
    assert (separate / "mlflow.db").exists()
    client = MlflowClient(tracking_uri=f"sqlite:///{(separate / 'mlflow.db').as_posix()}")
    assert client.get_registered_model(NAME).tags["platform.managed"] == "true"
    assert tree_fingerprint(archive) == before


# -- the database objects -------------------------------------------------------------------------


def test_progress_and_log_rules_are_enforced_by_the_database(rig: Rig) -> None:
    rig.register(0)
    rig.promote(0)
    rig.mirror.sync(NAME)
    n = rig.progress()["last_event_seq"]
    bad = (
        "UPDATE registry_sync SET synced_seq = synced_seq - 1",
        f"UPDATE registry_sync SET synced_seq = {n + 1}",
        "DELETE FROM registry_sync",
        "UPDATE registry_sync SET last_error_kind = 'nonsense'",
        "UPDATE registry_sync_log SET outcome = 'ok'",
        "DELETE FROM registry_sync_log",
        "TRUNCATE registry_sync_log",
        "INSERT INTO registry_sync_log (model_id, kind, outcome, seq, reason, details) "
        "SELECT id, 'reconcile', 'ok', 1, '', '{}'::jsonb FROM registered_models",
        "INSERT INTO registry_sync_log (model_id, kind, outcome, seq, details) "
        "SELECT id, 'reconcile', 'ok', 1, '{}'::jsonb FROM registered_models",
        "INSERT INTO registry_sync_log (model_id, kind, outcome, seq, details) "
        "SELECT id, 'sync', 'invented', 1, '{}'::jsonb FROM registered_models",
    )
    for statement in bad:
        with pytest.raises(DBAPIError):
            with rig.engine.begin() as c:
                c.execute(text(statement))
    assert rig.progress()["synced_seq"] == n


def test_migration_0004_goes_down_and_up_again(job_env: JobEnv) -> None:
    from automl_ops.db import migrate

    assert migrate.downgrade(job_env.engine, "0003").current == "0003"
    with job_env.engine.connect() as c:
        left = c.execute(
            text("SELECT proname FROM pg_proc WHERE proname LIKE 'registry_sync%'")
        ).all()
        tables = c.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name LIKE 'registry_sync%'"
            )
        ).all()
    assert left == [] and tables == []
    assert migrate.upgrade(job_env.engine).current == "0007"
    assert migrate.check(job_env.engine).ok


def test_classify_is_pure_and_needs_no_mlflow() -> None:
    from automl_ops.registry_mirror import Desired, DesiredVersion
    from automl_ops.tracking import MirrorSnapshot

    desired = Desired(
        1, NAME, 3, (DesiredVersion(1, "m-20260101T000000Z-aaaaaaaa", "staging", "f" * 64),)
    )
    absent = classify(desired, None, None, MirrorSnapshot(False, False))
    assert [(f.kind, f.category) for f in absent] == [("model_missing", "pending")]
    had = classify(
        desired, {"versions": {"1": {"status": "development"}}}, None, MirrorSnapshot(False, False)
    )
    assert [(f.kind, f.category) for f in had] == [("model_missing", "drift")]
    foreign = classify(desired, None, None, MirrorSnapshot(True, False))
    assert [(f.kind, f.category, f.blocking) for f in foreign] == [
        ("model_foreign", "conflict", True)
    ]
