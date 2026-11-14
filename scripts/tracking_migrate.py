"""Move the tracking history from the SQLite archive to the PostgreSQL store (BUILD_PLAN Step 17,
chunk P4-2b). Nothing here writes to the archive; every command says what it did.

    uv run python scripts/tracking_migrate.py backup  [--archive data/mlflow] [--dest data/backups]
    uv run python scripts/tracking_migrate.py plan    [--archive data/mlflow]
    uv run python scripts/tracking_migrate.py relog   [--archive data/mlflow] [--models data/models]
    uv run python scripts/tracking_migrate.py rerun   [--archive data/mlflow] [--store data/store]
    uv run python scripts/tracking_migrate.py verify  [--archive data/mlflow] [--manifest PATH]
        [--json]

* `backup`: a consistent copy of the SQLite archive (SQLite's backup API on an immutable read-only
  connection, so the original is never written, checkpointed or locked) plus its artifact folder,
  with a SHA-256 manifest. It refuses while any process has the database open (stop the MLflow UI
  first), if a WAL/journal file is present, or if the destination exists.
* `relog`: the archive's finalized jobs are logged again from their **sealed packages** (manifest
  and test result JSON only) into the PostgreSQL store, under the same job key (the package id).
* `rerun`: the archive's cross-validation-only jobs are run again from their recorded
  configuration through the same training and tracking code as `train --track`, under **new** job
  keys. The run is saved before anything is logged, so a retry resumes the same job.
* Both add provenance tags (`platform.origin`, `platform.migration`, and `platform.source_store` or
  `platform.replaces_job_key`): these jobs are new records, never the original MLflow history.
* `verify` is read-only: archive and backup against the manifest, PostgreSQL against the archive,
  package hashes, comparison, reproduction, run-URI lengths, duplicates and "no writes".

The target is always the PostgreSQL backend (`scripts/db.py mlflow-init` first). Exit codes: 0 ok,
1 a check failed, 2 cannot run (a refusal or an error).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from automl_ops import tracking
from automl_ops.compare import compare_jobs
from automl_ops.dataset_store import FileDatasetStore
from automl_ops.finalize import DatasetInfo
from automl_ops.package import library_versions, verify_package
from automl_ops.profile import read_version_frame
from automl_ops.reproduce import (
    DEFAULT_REL_TOLERANCE,
    TIMING_KEYS,
    _Walk,
    derive_config,
    reproduce_job,
    sealed_manifest_selection_sha,
    sealed_test_result,
)
from automl_ops.selection import metric_direction  # noqa: F401  (documents the CV-only basis)
from automl_ops.tracking import (
    PACKAGE_ARTIFACT_DIR,
    PACKAGE_FILES,
    AlreadyTrackedError,
    LoggedJob,
    TrackingError,
    TrackingStore,
    artifact_file,
    list_jobs,
    log_job,
    read_job,
    record_from_package,
    record_from_training,
    run_ids,
    store_status,
)
from automl_ops.train import AlgorithmRequest, TrainSettings, run_training

MIGRATION = "P4-2b"
ORIGIN_RELOG = "relogged-from-package"
ORIGIN_RERUN = "rerun-from-archive"
SOURCE_STORE = "sqlite-archive:data/mlflow"
MAPPING_FILE = "mapping.json"
MIGRATION_TAGS = (
    "platform.origin",
    "platform.source_store",
    "platform.migration",
    "platform.replaces_job_key",
)


class MigrationError(Exception):
    """A refusal or problem (exit code 2)."""


# -- helpers -----------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_files(root: Path) -> dict[str, str]:
    """relative path -> SHA-256 of every file below `root`."""
    return {
        str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()
    }


def tree_digest(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(f"{name}\0{value}\n".encode())
    return digest.hexdigest()


def state_dir() -> Path:
    """Migration state (saved re-runs and the mapping): next to the PostgreSQL artifact folder,
    by default `data/mlflow-pg/migration/`."""
    return target_store().artifact_path.parent / "migration"


def archive_store(args: argparse.Namespace) -> TrackingStore:
    return TrackingStore(Path(args.archive))


def target_store() -> TrackingStore:
    return TrackingStore.from_settings(backend="postgresql")


def normalised(job: LoggedJob) -> dict[str, Any]:
    """A job as plain data minus what legitimately differs between stores (system tags, the
    migration provenance tags, timestamps)."""

    from dataclasses import asdict

    data = asdict(job)
    data.pop("start_ms", None)

    def clean(tags: dict[str, str]) -> dict[str, str]:
        return {
            k: v for k, v in tags.items() if not k.startswith("mlflow.") and k not in MIGRATION_TAGS
        }

    data["tags"] = clean(data["tags"])
    for algorithm in data["algorithms"].values():
        algorithm["tags"] = clean(algorithm["tags"])
    return data


def differences(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{path}/{key}: present on one side only")
            else:
                out += differences(a[key], b[key], f"{path}/{key}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in differences(x, y, f"{path}[{i}]")
        ]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def _test_keys(value: Any) -> set[str]:
    """Any field whose name mentions the test part (the split's `test_fraction` setting aside):
    a comparison is cross-validation only and must not contain such a field."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, inner in value.items():
            if "test" in str(key) and key != "test_fraction":
                found.add(str(key))
            found |= _test_keys(inner)
    elif isinstance(value, list):
        for inner in value:
            found |= _test_keys(inner)
    return found


def pg_row_counts() -> dict[str, int]:
    """Row count of every table of MLflow's PostgreSQL database (read-only)."""
    from sqlalchemy import create_engine, text

    from automl_ops.db.config import database_url

    store = target_store()
    engine = create_engine(database_url().set(database=store.database_name))
    try:
        with engine.connect() as connection:
            names = [
                r[0]
                for r in connection.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public' ORDER BY 1"
                    )
                )
            ]
            return {
                n: int(connection.execute(text(f'SELECT count(*) FROM "{n}"')).scalar_one())
                for n in names
            }
    finally:
        engine.dispose()


def archive_jobs(args: argparse.Namespace) -> list[tracking.JobSummary]:
    jobs = list_jobs(archive_store(args), state=tracking.COMPLETE)
    return sorted(jobs, key=lambda j: j.start_ms)


# -- backup ------------------------------------------------------------------------------------


def _open_handles(db: Path) -> list[str]:
    lsof = shutil.which("lsof")
    if lsof is None:
        raise MigrationError(
            "`lsof` is not available, so I cannot prove that no process has the archive open"
        )
    found: list[str] = []
    done = subprocess.run(
        [lsof, "-Fpc", "--", str(db)], capture_output=True, text=True, check=False
    )
    found += [line for line in done.stdout.splitlines() if line.startswith(("p", "c"))]
    ui = subprocess.run(
        [lsof, "-nP", "-iTCP:5001", "-sTCP:LISTEN", "-Fpc"],
        capture_output=True,
        text=True,
        check=False,
    )
    found += [
        f"UI port 5001: {line}" for line in ui.stdout.splitlines() if line.startswith(("p", "c"))
    ]
    return found


def _counts(connection: sqlite3.Connection) -> dict[str, int]:
    names = [
        r[0]
        for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1")
    ]
    return {n: connection.execute(f'SELECT count(*) FROM "{n}"').fetchone()[0] for n in names}


def cmd_backup(args: argparse.Namespace) -> int:
    archive = Path(args.archive).resolve()
    db = archive / "mlflow.db"
    artifacts = archive / "artifacts"
    if not db.is_file() or not artifacts.is_dir():
        raise MigrationError(f"{archive} is not a tracking store (needs mlflow.db and artifacts/)")
    leftovers = [p.name for p in archive.iterdir() if p.name.startswith("mlflow.db-")]
    if leftovers:
        raise MigrationError(
            f"{leftovers} present next to the archive database (a WAL or journal file means a "
            "write may be in progress or was interrupted); I will not touch it. Resolve that first."
        )
    handles = _open_handles(db)
    if handles:
        raise MigrationError(
            "a process has the archive database open or the MLflow UI is listening: "
            + "; ".join(handles)
            + ". Stop it first."
        )
    dest = Path(args.dest).resolve() / f"mlflow-sqlite-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    if dest.exists():
        raise MigrationError(f"{dest} already exists; backups are never overwritten")
    before_stat = db.stat()
    before: dict[str, Any] = {"db_sha256": sha256_file(db), "artifacts": tree_files(artifacts)}
    dest.mkdir(parents=True)
    # immutable=1: SQLite takes no locks and never creates -wal/-shm/-journal files next to the
    # original, and the backup API copies a consistent snapshot of it.
    source = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    target = sqlite3.connect(dest / "mlflow.db")
    try:
        source.backup(target)
        source_counts = _counts(source)
        backup_counts = _counts(target)
        integrity = target.execute("PRAGMA integrity_check").fetchone()[0]
        revision = target.execute("SELECT version_num FROM alembic_version").fetchone()[0]
    finally:
        target.close()
        source.close()
    shutil.copytree(artifacts, dest / "artifacts", copy_function=shutil.copy2)
    copied = tree_files(dest / "artifacts")
    after_stat = db.stat()
    after: dict[str, Any] = {"db_sha256": sha256_file(db), "artifacts": tree_files(artifacts)}
    problems: list[str] = []
    if integrity != "ok":
        problems.append(f"integrity_check on the backup: {integrity}")
    if source_counts != backup_counts:
        problems.append("row counts differ between the archive and the backup")
    if copied != before["artifacts"]:
        problems.append("the copied artifacts differ from the archive's")
    if before != after or (before_stat.st_mtime_ns, before_stat.st_size) != (
        after_stat.st_mtime_ns,
        after_stat.st_size,
    ):
        problems.append("THE ARCHIVE CHANGED DURING THE BACKUP")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "archive_path": str(archive),
        "archive": {
            "db_sha256": before["db_sha256"],
            "db_size": before_stat.st_size,
            "db_mtime_ns": before_stat.st_mtime_ns,
            "alembic_revision": revision,
            "artifacts_digest": tree_digest(before["artifacts"]),
            "artifact_files": len(before["artifacts"]),
        },
        "backup": {
            "db_sha256": sha256_file(dest / "mlflow.db"),
            "artifacts_digest": tree_digest(copied),
            "integrity_check": integrity,
        },
        "row_counts": source_counts,
        "problems": problems,
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"backup: {dest}")
    print(
        f"  database: integrity_check {integrity}, {sum(source_counts.values())} rows in "
        f"{len(source_counts)} tables, equal to the archive's; artifacts: {len(copied)} files, "
        "SHA-256 equal file by file"
    )
    print(
        f"  archive before/after: sha256 {before['db_sha256'][:16]} -> {after['db_sha256'][:16]}, "
        f"mtime and size {'unchanged' if not problems else 'CHECK PROBLEMS'}"
    )
    for problem in problems:
        print(f"  PROBLEM: {problem}")
    print(f"  manifest: {dest / 'MANIFEST.json'}")
    return 1 if problems else 0


# -- plan / relog / rerun ----------------------------------------------------------------------


def cmd_plan(args: argparse.Namespace) -> int:
    for job in archive_jobs(args):
        what = (
            f"relog from sealed package, same key ({job.package_id})"
            if job.kind == "final"
            else "rerun, new key (original stays archived)"
        )
        print(
            f"{job.job_key:<32} {job.kind:<5} {job.dataset_ref}  seed {job.seed} "
            f"folds {job.folds_used}  -> {what}"
        )
    return 0


def _write_mapping(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Original job -> new job, with run ids, from the stores and the saved re-run states."""
    archive, target = archive_store(args), target_store()
    entries: list[dict[str, Any]] = []
    for job in archive_jobs(args):
        entry: dict[str, Any] = {
            "original_job_key": job.job_key,
            "original_run_ids": run_ids(job.job_key, archive),
            "kind": job.kind,
        }
        if job.kind == "final":
            entry["origin"] = ORIGIN_RELOG
            new_key: str | None = job.job_key
        else:
            entry["origin"] = ORIGIN_RERUN
            state_file = state_dir() / f"rerun-{job.job_key}.json"
            new_key = json.loads(state_file.read_text())["job_key"] if state_file.exists() else None
        entry["new_job_key"] = new_key
        try:
            entry["new_run_ids"] = run_ids(new_key, target) if new_key else None
        except TrackingError:
            entry["new_run_ids"] = None
        entries.append(entry)
    state_dir().mkdir(parents=True, exist_ok=True)
    (state_dir() / MAPPING_FILE).write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    return entries


def _print_mapping(entries: list[dict[str, Any]]) -> None:
    print("\nmapping (original archive job -> PostgreSQL job):")
    for e in entries:
        new = e["new_run_ids"]
        print(f"  [{e['origin']}] {e['original_job_key']} -> {e['new_job_key']}")
        print(f"      original parent run {e['original_run_ids']['parent']}")
        print(f"      new      parent run {new['parent'] if new else '(not logged yet)'}")


def cmd_relog(args: argparse.Namespace) -> int:
    target = target_store()
    for job in [j for j in archive_jobs(args) if j.kind == "final"]:
        record = record_from_package(
            Path(args.models),
            str(job.package_id),
            extra_tags={
                "platform.origin": ORIGIN_RELOG,
                "platform.source_store": SOURCE_STORE,
                "platform.migration": MIGRATION,
            },
        )
        try:
            tracked = log_job(record, target)
            print(
                f"{job.job_key}: {'resumed and completed' if tracked.resumed else 'logged'} "
                f"({len(tracked.child_run_ids)} algorithm runs)"
            )
        except AlreadyTrackedError:
            print(f"{job.job_key}: already tracked in the PostgreSQL store (nothing to do)")
    _print_mapping(_write_mapping(args))
    return 0


def _rerun_state(args: argparse.Namespace, job: tracking.JobSummary) -> dict[str, Any]:
    """The saved re-run (created on first use, then reused): the training result is computed once
    and persisted *before* any MLflow write, so a retry logs exactly the same job."""
    state_file = state_dir() / f"rerun-{job.job_key}.json"
    if state_file.exists():
        return dict(json.loads(state_file.read_text()))
    archived = read_job(job.job_key, archive_store(args))
    store = FileDatasetStore(Path(args.store))
    version, frame = read_version_frame(store, str(archived.dataset["ref"]))
    if version.sha256 != archived.tags.get("platform.dataset_sha256"):
        raise MigrationError(f"{job.job_key}: the dataset version differs from the one it used")
    config = derive_config(archived, [str(c) for c in frame.columns])
    settings = TrainSettings(**archived.training["settings"])
    requests = [
        AlgorithmRequest(r["algorithm"], dict(r["params"])) for r in archived.training["results"]
    ]
    result = run_training(frame, config, requests, settings)
    state = {
        "original_job_key": job.job_key,
        "job_key": tracking.new_train_job_key(),
        "created_ms": int(datetime.now(UTC).timestamp() * 1000),
        "training": json.loads(json.dumps(result.to_dict())),
        "dataset": DatasetInfo.from_version(version).to_dict(),
        "libraries": library_versions(),
    }
    state_dir().mkdir(parents=True, exist_ok=True)
    with state_file.open("x", encoding="utf-8") as handle:  # never overwritten
        handle.write(json.dumps(state, sort_keys=True))
    return state


def cmd_rerun(args: argparse.Namespace) -> int:
    target = target_store()
    for job in [j for j in archive_jobs(args) if j.kind == "train"]:
        state = _rerun_state(args, job)
        record = record_from_training(
            state["training"],
            state["dataset"],
            state["libraries"],
            job_key=state["job_key"],
            created_ms=state["created_ms"],
            extra_tags={
                "platform.origin": ORIGIN_RERUN,
                "platform.replaces_job_key": job.job_key,
                "platform.migration": MIGRATION,
            },
        )
        try:
            tracked = log_job(record, target)
            print(
                f"{job.job_key} -> {state['job_key']}: "
                f"{'resumed and completed' if tracked.resumed else 'logged'}"
            )
        except AlreadyTrackedError:
            print(f"{job.job_key} -> {state['job_key']}: already tracked (nothing to do)")
    _print_mapping(_write_mapping(args))
    return 0


# -- verify ------------------------------------------------------------------------------------


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, name: str, status: str, detail: str) -> None:
        self.rows.append({"check": name, "status": status, "detail": detail})
        print(f"[{status}] {name}: {detail}")

    @property
    def failed(self) -> bool:
        return any(r["status"] == "FAIL" for r in self.rows)


def cmd_verify(args: argparse.Namespace) -> int:
    checks = Checks()
    archive, target = archive_store(args), target_store()
    archive_dir = Path(args.archive).resolve()
    counts_before = pg_row_counts()
    archive_before = tree_files(archive_dir / "artifacts"), sha256_file(archive_dir / "mlflow.db")

    # 1-2: archive and backup against the manifest
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        backup_dir = Path(args.manifest).parent
        same = (
            archive_before[1] == manifest["archive"]["db_sha256"]
            and tree_digest(archive_before[0]) == manifest["archive"]["artifacts_digest"]
        )
        checks.add(
            "archive equals the manifest",
            "PASS" if same else "FAIL",
            f"database sha256 {archive_before[1][:16]}, {len(archive_before[0])} artifact files",
        )
        intact = (
            sha256_file(backup_dir / "mlflow.db") == manifest["backup"]["db_sha256"]
            and tree_digest(tree_files(backup_dir / "artifacts"))
            == manifest["backup"]["artifacts_digest"]
            and not manifest["problems"]
        )
        checks.add("backup equals the manifest", "PASS" if intact else "FAIL", str(backup_dir))
    else:
        checks.add("archive and backup against the manifest", "NOT RUN", "no --manifest given")

    # 3: the archive is still readable with explicit SQLite selection
    jobs = archive_jobs(args)
    readable = [read_job(j.job_key, archive) for j in jobs]
    checks.add(
        "archive readable with the explicit SQLite backend",
        "PASS" if len(readable) >= 1 else "FAIL",
        f"{len(readable)} complete jobs: " + ", ".join(j.job_key for j in jobs),
    )

    # 4: the PostgreSQL store
    status = store_status(target)
    ok4 = bool(
        status.get("initialised")
        and status.get("location_matches_configuration")
        and status.get("run_uri_margin", -1) >= 0
    )
    checks.add(
        "PostgreSQL store initialised, location matches, URIs fit",
        "PASS" if ok4 else "FAIL",
        f"revision {status.get('schema_revision')}, {status.get('runs')} runs, longest run URI "
        f"{status.get('longest_run_uri')} of 200 (margin {status.get('run_uri_margin')})",
    )
    if not ok4:
        return 1

    pg_jobs = {
        j.job_key: read_job(j.job_key, target) for j in list_jobs(target, state=tracking.COMPLETE)
    }
    mapping = (
        json.loads((state_dir() / MAPPING_FILE).read_text())
        if (state_dir() / MAPPING_FILE).exists()
        else []
    )
    by_original = {e["original_job_key"]: e for e in mapping}
    models = Path(args.models)
    store = FileDatasetStore(Path(args.store))

    for summary, old in zip(jobs, readable, strict=True):
        entry = by_original.get(summary.job_key)
        new_key = entry["new_job_key"] if entry else None
        new = pg_jobs.get(new_key) if new_key else None
        if new is None:
            checks.add(f"{summary.job_key} present in PostgreSQL", "FAIL", f"new key {new_key}")
            continue
        origin = ORIGIN_RELOG if summary.kind == "final" else ORIGIN_RERUN
        tags_ok = (
            new.tags.get("platform.origin") == origin
            and new.tags.get("platform.migration") == MIGRATION
            and not any(t in old.tags for t in MIGRATION_TAGS)
            and (
                new.tags.get("platform.source_store") == SOURCE_STORE
                if summary.kind == "final"
                else new.tags.get("platform.replaces_job_key") == summary.job_key
            )
        )
        checks.add(
            f"{summary.job_key}: provenance tags (new record, original untouched)",
            "PASS" if tags_ok else "FAIL",
            origin,
        )
        if summary.kind == "final":
            diff = differences(normalised(old), normalised(new))
            checks.add(
                f"{summary.job_key}: PostgreSQL record equals the archive's",
                "PASS" if not diff else "FAIL",
                f"{len(diff)} differences (system tags, timestamps and migration tags excluded) "
                + "; ".join(diff[:3]),
            )
            selected = str(new.tags["platform.selected_algorithm"])
            bad = []
            for name in PACKAGE_FILES:
                logged = artifact_file(
                    new.job_key, selected, f"{PACKAGE_ARTIFACT_DIR}/{name}", target
                )
                if sha256_file(logged) != sha256_file(models / new.job_key / name):
                    bad.append(name)
            deep = verify_package(models, new.job_key)
            checks.add(
                f"{summary.job_key}: logged package files equal the sealed package",
                "PASS" if not bad and deep.ok else "FAIL",
                f"{len(PACKAGE_FILES)} files hashed; package-verify "
                f"{'ok' if deep.ok else deep.problems}; mismatches {bad}",
            )
        else:
            walk = _Walk(DEFAULT_REL_TOLERANCE)
            walk.walk("training", old.training, new.training, TIMING_KEYS)
            same_inputs = old.dataset == new.dataset and old.tags.get(
                "platform.dataset_sha256"
            ) == new.tags.get("platform.dataset_sha256")
            same_inputs = (
                same_inputs
                and old.params == new.params
                and old.libraries.keys() == new.libraries.keys()
            )
            checks.add(
                f"{summary.job_key}: re-run equals the recorded original configuration and results",
                "PASS" if not walk.mismatches and same_inputs else "FAIL",
                f"dataset/seed/folds/params equal: {same_inputs}; CV results within tolerance: "
                f"{not walk.mismatches} (largest difference {walk.max_deviation:.3g}, "
                f"{'bit-identical' if walk.identical else 'not bit-identical'})",
            )

    # duplicates: exactly one parent run per new job key and per replaced key
    api = tracking._Api(target, read_only=True)
    dupes = []
    for entry in mapping:
        key = entry["new_job_key"]
        if key:
            n = len(
                api.client.search_runs(
                    [api.experiment_id],
                    f"tags.`platform.job_key` = '{key}' and tags.`platform.role` = 'job'",
                    max_results=10,
                )
            )
            if n != 1:
                dupes.append(f"{key}: {n}")
        if entry["kind"] == "train":
            n = len(
                api.client.search_runs(
                    [api.experiment_id],
                    f"tags.`platform.replaces_job_key` = '{entry['original_job_key']}'",
                    max_results=10,
                )
            )
            if n != 1:
                dupes.append(f"replaces {entry['original_job_key']}: {n}")
    checks.add(
        "no duplicate runs",
        "PASS" if not dupes and mapping else "FAIL",
        f"{len(mapping)} mapped jobs; problems {dupes}",
    )

    # comparison (CV only) within each task/target group of the PostgreSQL store
    groups: dict[tuple[str | None, str | None], list[LoggedJob]] = {}
    for job in pg_jobs.values():
        groups.setdefault(
            (job.tags.get("platform.task"), job.tags.get("platform.target")), []
        ).append(job)
    compared = []
    for (task, target_name), members in groups.items():
        if len(members) >= 2:
            result = compare_jobs(members)
            leaks = sorted(_test_keys(result))
            compared.append(
                f"{task}/{target_name}: {len(members)} jobs, {len(result['rows'])} rows, "
                f"split note: {'yes' if result['split_note'] else 'no'}, "
                f"test fields in output: {leaks or 'none'}"
            )
            if leaks:
                checks.add("comparison uses CV numbers only", "FAIL", compared[-1])
    checks.add(
        "comparison runs on the PostgreSQL store",
        "PASS" if compared else "NOT RUN",
        "; ".join(compared) or "no two comparable jobs",
    )

    # reproduction of every job in the PostgreSQL store
    verdicts = []
    ok9 = True
    for job in pg_jobs.values():
        version, frame = read_version_frame(store, str(job.dataset["ref"]))
        package_id = job.tags.get("platform.package_id")
        have_package = job.kind == "final" and package_id and (models / package_id).is_dir()
        report = reproduce_job(
            job,
            frame,
            version.sha256,
            library_versions(),
            package_test_result=sealed_test_result(models, str(package_id))
            if have_package
            else None,
            package_selection_sha256=sealed_manifest_selection_sha(models, str(package_id))
            if have_package
            else None,
            package_checked=bool(have_package),
        )
        ok9 &= report.ok and report.bit_identical
        verdicts.append(
            f"{job.job_key}: {report.verdict}{' bit-identical' if report.bit_identical else ''}"
        )
    checks.add(
        "reproduce every PostgreSQL job",
        "PASS" if ok9 and verdicts else "FAIL",
        "; ".join(verdicts),
    )

    # nothing written by verification
    counts_after = pg_row_counts()
    archive_after = tree_files(archive_dir / "artifacts"), sha256_file(archive_dir / "mlflow.db")
    wrote = counts_before != counts_after or archive_before != archive_after
    checks.add(
        "verification wrote nothing (PostgreSQL row counts, archive hashes)",
        "FAIL" if wrote else "PASS",
        f"{sum(counts_after.values())} rows in {len(counts_after)} tables before and after",
    )
    if args.json:
        print(json.dumps(checks.rows, indent=2))
    return 1 if checks.failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("backup", "plan", "relog", "rerun", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--archive", default="data/mlflow", help="the SQLite archive folder")
        if name == "backup":
            p.add_argument("--dest", default="data/backups")
        if name in ("relog", "verify"):
            p.add_argument("--models", default="data/models")
        if name in ("rerun", "verify"):
            p.add_argument("--store", default="data/store")
        if name == "verify":
            p.add_argument("--manifest", default=None)
            p.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    commands = {
        "backup": cmd_backup,
        "plan": cmd_plan,
        "relog": cmd_relog,
        "rerun": cmd_rerun,
        "verify": cmd_verify,
    }
    try:
        return commands[args.command](args)
    except (MigrationError, TrackingError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    raise SystemExit(main())
