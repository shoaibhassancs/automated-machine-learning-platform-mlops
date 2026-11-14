"""The dataset catalog: an index of the file store's versions in PostgreSQL (P4-1, D-028).

**The file store stays authoritative** for bytes, hashes and version numbers. A catalog row is
written only *after* a version is stored (`CatalogedDatasetStore`), and `sync_catalog` re-creates
any missing row from the version records. Both are idempotent. The catalog never contains dataset
contents, only identity, shape, column names and provenance fields.

Sync never deletes or overwrites: a row with no file (orphan), a row whose hash disagrees with the
file (conflict) and a version whose files fail verification are *reported* for the owner, not
repaired silently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import Connection, Engine, select
from sqlalchemy.dialects.postgresql import insert

from automl_ops.dataset_store import (
    STORE_FORMAT,
    DatasetStore,
    DatasetStoreError,
    DatasetVersion,
    IngestResult,
    VerifyResult,
)
from automl_ops.db.models import Dataset, DatasetVersionRow

log = logging.getLogger(__name__)

Outcome = Literal["added", "present", "conflict"]


@dataclass(frozen=True)
class SyncResult:
    added: tuple[str, ...]  # refs inserted by this run
    present: int  # refs already catalogued and identical
    conflicts: tuple[str, ...]  # catalogued differently from the file store (not changed)
    orphans: tuple[str, ...]  # catalogued but absent from the file store (not deleted)
    unverified: tuple[str, ...]  # file store versions that fail verification (not catalogued)

    @property
    def ok(self) -> bool:
        return not (self.conflicts or self.orphans or self.unverified)

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": list(self.added),
            "present": self.present,
            "conflicts": list(self.conflicts),
            "orphans": list(self.orphans),
            "unverified": list(self.unverified),
            "ok": self.ok,
        }


def version_values(version: DatasetVersion) -> dict[str, Any]:
    """Column values for one catalog row, from the version's own record."""
    return {
        "number": version.number,
        "sha256": version.sha256,
        "original_sha256": version.original_sha256,
        "size_bytes": version.size_bytes,
        "original_size_bytes": version.original_size_bytes,
        "rows": version.rows,
        "column_names": list(version.columns),
        "source_filename": version.source_filename,
        "original_encoding": version.original_encoding,
        "original_delimiter": version.original_delimiter,
        "reencoded": version.reencoded,
        "trailing_blank_lines_ignored": version.trailing_blank_lines_ignored,
        "store_format": STORE_FORMAT,
        "created_at": datetime.fromisoformat(version.created_at),
    }


def record_version(connection: Connection, version: DatasetVersion) -> tuple[Outcome, str]:
    """Catalog one version if it is not there yet; never change an existing row."""
    connection.execute(insert(Dataset).values(name=version.dataset).on_conflict_do_nothing())
    dataset_id = connection.execute(
        select(Dataset.id).where(Dataset.name == version.dataset)
    ).scalar_one()
    inserted = connection.execute(
        insert(DatasetVersionRow)
        .values(dataset_id=dataset_id, **version_values(version))
        .on_conflict_do_nothing()
        .returning(DatasetVersionRow.id)
    ).scalar_one_or_none()
    if inserted is not None:
        return "added", version.ref
    existing = connection.execute(
        select(DatasetVersionRow.sha256).where(
            DatasetVersionRow.dataset_id == dataset_id, DatasetVersionRow.number == version.number
        )
    ).scalar_one_or_none()
    if existing == version.sha256:
        return "present", version.ref
    if existing is not None:
        return (
            "conflict",
            f"{version.ref}: the catalog has sha256 {existing[:12]}..., the file store has "
            f"{version.sha256[:12]}...",
        )
    other = connection.execute(
        select(DatasetVersionRow.number).where(
            DatasetVersionRow.dataset_id == dataset_id, DatasetVersionRow.sha256 == version.sha256
        )
    ).scalar_one_or_none()
    return "conflict", f"{version.ref}: identical content is already catalogued as v{other}"


def sync_catalog(engine: Engine, store: DatasetStore) -> SyncResult:
    """Make the catalog contain every verified version of the file store. Reads the store only."""
    added: list[str] = []
    conflicts: list[str] = []
    unverified: list[str] = []
    present = 0
    in_store: set[tuple[str, int]] = set()
    for name in store.list_datasets():
        try:
            versions = store.list_versions(name)
            checks: dict[str, VerifyResult] = {r.ref: r for r in store.verify(name)}
        except DatasetStoreError as error:
            unverified.append(f"{name}: {error}")
            continue
        for version in versions:
            in_store.add((version.dataset, version.number))
            check = checks.get(version.ref)
            if check is None or not check.ok:
                problems = "; ".join(check.problems) if check else "not verified"
                unverified.append(f"{version.ref}: {problems}")
                continue
            with engine.begin() as connection:
                outcome, detail = record_version(connection, version)
            if outcome == "added":
                added.append(detail)
            elif outcome == "present":
                present += 1
            else:
                conflicts.append(detail)
    with engine.connect() as connection:
        rows = connection.execute(
            select(Dataset.name, DatasetVersionRow.number).join(
                DatasetVersionRow, DatasetVersionRow.dataset_id == Dataset.id
            )
        ).all()
    orphans = sorted(f"{name}@v{number}" for name, number in rows if (name, number) not in in_store)
    return SyncResult(tuple(added), present, tuple(conflicts), tuple(orphans), tuple(unverified))


class CatalogedDatasetStore:
    """A `DatasetStore` that also catalogs each newly stored version, after the file write.

    If the catalog cannot be written the version is still stored (the bytes are safe and the file
    store is authoritative); the ref is kept in `unsynced` and `sync_catalog` repairs it later.
    """

    def __init__(self, store: DatasetStore, engine: Engine) -> None:
        self._store = store
        self._engine = engine
        self.unsynced: list[str] = []

    def ingest(
        self,
        name: str,
        data: bytes,
        *,
        source_filename: str = "upload.csv",
        add_version: bool = False,
    ) -> IngestResult:
        result = self._store.ingest(
            name, data, source_filename=source_filename, add_version=add_version
        )
        try:
            with self._engine.begin() as connection:
                outcome, detail = record_version(connection, result.version)
            if outcome == "conflict":
                self.unsynced.append(result.version.ref)
                log.warning("catalog conflict: %s", detail)
        except Exception as error:  # the version is stored; the catalog is repaired by sync
            self.unsynced.append(result.version.ref)
            log.warning(
                "%s stored but not catalogued (run `db sync`): %s", result.version.ref, error
            )
        return result

    def list_datasets(self) -> list[str]:
        return self._store.list_datasets()

    def list_versions(self, name: str) -> list[DatasetVersion]:
        return self._store.list_versions(name)

    def get_version(self, ref: str) -> DatasetVersion:
        return self._store.get_version(ref)

    def version_path(self, ref: str) -> Path:
        return self._store.version_path(ref)

    def verify(self, name: str | None = None) -> list[VerifyResult]:
        return self._store.verify(name)
