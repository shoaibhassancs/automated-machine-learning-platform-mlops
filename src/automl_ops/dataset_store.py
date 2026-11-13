"""Dataset versions on disk (BUILD_PLAN Step 3).

A *dataset* is a named collection of *versions*. A version is one canonical CSV file plus a small
JSON record, stored as `<root>/<dataset>/v0001.csv` and `v0001.json`. Versions are immutable:
nothing here edits or replaces a stored version, and the files are made read-only.

Identity is the SHA-256 of the canonical bytes (see `ingest.py`). The `DatasetStore` protocol is
the interface that a database-backed store replaces in Phase 4 (BUILD_PLAN Steps 17 and 18).
Single-user, single-machine design: new version numbers are claimed with an atomic exclusive file
create, but no other protection against several processes writing at once is claimed.
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Any, Protocol

from automl_ops.ingest import (
    DEFAULT_LIMITS,
    IngestLimits,
    PreparedCsv,
    prepare_csv,
    sha256_bytes,
)

STORE_FORMAT = 1
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
REF_PATTERN = re.compile(r"^([a-z0-9][a-z0-9_-]{0,63})@v([0-9]+)$")
FILE_PATTERN = re.compile(r"^v([0-9]{4,})\.(csv|json)$")


class DatasetStoreError(Exception):
    """Base class for store problems."""


class InvalidName(DatasetStoreError):
    pass


class DatasetExists(DatasetStoreError):
    pass


class DatasetNotFound(DatasetStoreError):
    pass


class VersionNotFound(DatasetStoreError):
    pass


@dataclass(frozen=True)
class DatasetVersion:
    """One stored version. Its fields mirror the future `dataset_versions` database columns."""

    dataset: str
    number: int
    sha256: str  # of the canonical stored bytes: the identity
    original_sha256: str  # of the first upload that created it: provenance only
    size_bytes: int
    original_size_bytes: int
    rows: int
    columns: tuple[str, ...]
    source_filename: str  # file name as uploaded, for information only; never used in a path
    original_encoding: str
    original_delimiter: str
    reencoded: bool
    trailing_blank_lines_ignored: int
    created_at: str  # UTC, ISO 8601

    @property
    def ref(self) -> str:
        return f"{self.dataset}@v{self.number}"

    @property
    def short_hash(self) -> str:
        return self.sha256[:8]

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["columns"] = list(self.columns)
        record["store_format"] = STORE_FORMAT
        return record

    @staticmethod
    def from_record(record: dict[str, Any]) -> DatasetVersion:
        fields = {k: v for k, v in record.items() if k != "store_format"}
        fields["columns"] = tuple(fields["columns"])
        return DatasetVersion(**fields)


@dataclass(frozen=True)
class IngestResult:
    version: DatasetVersion
    created: bool  # False: identical content was already stored; nothing was written
    upload_sha256: str  # about the file just uploaded (which may differ from the stored version)
    upload_encoding: str
    upload_delimiter: str
    upload_reencoded: bool


@dataclass(frozen=True)
class VerifyResult:
    ref: str
    ok: bool
    problems: tuple[str, ...]


class DatasetStore(Protocol):
    """What the rest of the platform may rely on. A database-backed store will implement this."""

    def ingest(
        self,
        name: str,
        data: bytes,
        *,
        source_filename: str = "upload.csv",
        add_version: bool = False,
    ) -> IngestResult: ...

    def list_datasets(self) -> list[str]: ...

    def list_versions(self, name: str) -> list[DatasetVersion]: ...

    def get_version(self, ref: str) -> DatasetVersion: ...

    def version_path(self, ref: str) -> Path: ...

    def verify(self, name: str | None = None) -> list[VerifyResult]: ...


def validate_name(name: str) -> str:
    if not NAME_PATTERN.fullmatch(name):
        raise InvalidName(
            f"invalid dataset name {name!r}: use 1-64 characters from a-z, 0-9, '_' and '-', "
            "starting with a letter or digit"
        )
    return name


def parse_ref(ref: str) -> tuple[str, int]:
    match = REF_PATTERN.fullmatch(ref)
    if not match:
        raise VersionNotFound(
            f"{ref!r} is not a version reference; expected name@vN, e.g. adult@v1"
        )
    return match.group(1), int(match.group(2))


def clean_source_filename(raw: str) -> str:
    """Keep only the base name, printable characters, at most 255 characters."""
    base = PurePath(raw.replace("\\", "/")).name
    base = "".join(ch for ch in base if ch.isprintable())[:255]
    return base or "upload.csv"


def _make_read_only(path: Path) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


class FileDatasetStore:
    """Versions stored as files under `root` (default for the CLI: `data/store`)."""

    def __init__(self, root: Path, limits: IngestLimits = DEFAULT_LIMITS) -> None:
        self.root = Path(root)
        self.limits = limits

    # -- paths ------------------------------------------------------------------------------

    def _dataset_dir(self, name: str) -> Path:
        return self.root / validate_name(name)

    @staticmethod
    def _file_stem(number: int) -> str:
        return f"v{number:04d}"

    def _csv_path(self, name: str, number: int) -> Path:
        return self._dataset_dir(name) / f"{self._file_stem(number)}.csv"

    def _json_path(self, name: str, number: int) -> Path:
        return self._dataset_dir(name) / f"{self._file_stem(number)}.json"

    # -- reading ----------------------------------------------------------------------------

    def list_datasets(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            entry.name
            for entry in self.root.iterdir()
            if entry.is_dir() and NAME_PATTERN.fullmatch(entry.name)
        )

    def list_versions(self, name: str) -> list[DatasetVersion]:
        directory = self._dataset_dir(name)
        if not directory.is_dir():
            raise DatasetNotFound(f"no dataset named {name!r}")
        versions: list[DatasetVersion] = []
        for path in directory.glob("v*.json"):
            match = FILE_PATTERN.fullmatch(path.name)
            if not match:
                continue
            versions.append(self._read_record(path))
        return sorted(versions, key=lambda v: v.number)

    @staticmethod
    def _read_record(path: Path) -> DatasetVersion:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            return DatasetVersion.from_record(record)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise DatasetStoreError(f"cannot read version record {path}: {error}") from error

    def get_version(self, ref: str) -> DatasetVersion:
        name, number = parse_ref(ref)
        path = self._json_path(name, number)
        if not path.is_file():
            raise VersionNotFound(f"no version {ref!r}")
        return self._read_record(path)

    def version_path(self, ref: str) -> Path:
        version = self.get_version(ref)
        return self._csv_path(version.dataset, version.number)

    # -- writing ----------------------------------------------------------------------------

    def ingest(
        self,
        name: str,
        data: bytes,
        *,
        source_filename: str = "upload.csv",
        add_version: bool = False,
    ) -> IngestResult:
        """Store `data` as a new version, or return the existing version with identical content.

        Rules: a new dataset name must not exist yet; adding a new version to an existing
        dataset needs `add_version=True`. Identical content is always safe to re-upload: it
        returns the existing version and writes nothing. Nothing is written if the CSV is
        rejected.
        """
        validate_name(name)
        prepared = prepare_csv(data, self.limits)
        directory = self._dataset_dir(name)

        def result(version: DatasetVersion, created: bool) -> IngestResult:
            return IngestResult(
                version=version,
                created=created,
                upload_sha256=prepared.original_sha256,
                upload_encoding=prepared.original_encoding,
                upload_delimiter=prepared.original_delimiter,
                upload_reencoded=prepared.reencoded,
            )

        created_directory = False
        if directory.is_dir():
            for existing in self.list_versions(name):
                if existing.sha256 == prepared.sha256:
                    return result(existing, False)
            if not add_version:
                raise DatasetExists(
                    f"dataset {name!r} already exists and this file has different content. "
                    "To store it as a new version, ask for a new version explicitly "
                    "(--add-version); to create a separate dataset, use another name."
                )
        else:
            if add_version:
                raise DatasetNotFound(
                    f"dataset {name!r} does not exist, so there is nothing to add a version to; "
                    "ingest it without --add-version to create it"
                )
            self.root.mkdir(parents=True, exist_ok=True)
            directory.mkdir(exist_ok=False)
            created_directory = True

        try:
            version = self._write_new_version(name, prepared, source_filename)
        except BaseException:
            if created_directory:
                try:
                    directory.rmdir()  # only succeeds if empty
                except OSError:
                    pass
            raise
        return result(version, True)

    def _next_number(self, name: str) -> int:
        numbers = [0]
        for path in self._dataset_dir(name).iterdir():
            match = FILE_PATTERN.fullmatch(path.name)
            if match:
                numbers.append(int(match.group(1)))
        return max(numbers) + 1

    def _write_new_version(
        self, name: str, prepared: PreparedCsv, source_filename: str
    ) -> DatasetVersion:
        for _attempt in range(10):
            number = self._next_number(name)
            csv_path = self._csv_path(name, number)
            try:
                handle = csv_path.open("xb")  # exclusive create: claims the number atomically
            except FileExistsError:
                continue
            committed = False
            try:
                with handle:
                    handle.write(prepared.canonical)
                version = DatasetVersion(
                    dataset=name,
                    number=number,
                    sha256=prepared.sha256,
                    original_sha256=prepared.original_sha256,
                    size_bytes=prepared.size_bytes,
                    original_size_bytes=prepared.original_size_bytes,
                    rows=prepared.rows,
                    columns=prepared.columns,
                    source_filename=clean_source_filename(source_filename),
                    original_encoding=prepared.original_encoding,
                    original_delimiter=prepared.original_delimiter,
                    reencoded=prepared.reencoded,
                    trailing_blank_lines_ignored=prepared.trailing_blank_lines_ignored,
                    created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                )
                json_path = self._json_path(name, number)
                temp_path = json_path.with_name(json_path.name + ".tmp")
                temp_path.write_text(
                    json.dumps(version.to_record(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(temp_path, json_path)  # the record is the commit marker
                committed = True
            finally:
                if not committed:
                    csv_path.unlink(missing_ok=True)
                    self._json_path(name, number).with_suffix(".json.tmp").unlink(missing_ok=True)
            _make_read_only(csv_path)
            _make_read_only(self._json_path(name, number))
            return version
        raise DatasetStoreError("could not claim a new version number; try again")

    # -- verification -----------------------------------------------------------------------

    def verify(self, name: str | None = None) -> list[VerifyResult]:
        """Re-check stored files against their records. Reads everything; changes nothing."""
        names = [name] if name is not None else self.list_datasets()
        results: list[VerifyResult] = []
        for dataset in names:
            directory = self._dataset_dir(dataset)
            if not directory.is_dir():
                raise DatasetNotFound(f"no dataset named {dataset!r}")
            claimed: set[tuple[int, str]] = set()
            for path in sorted(directory.iterdir()):
                match = FILE_PATTERN.fullmatch(path.name)
                if match:
                    claimed.add((int(match.group(1)), match.group(2)))
                elif path.name.endswith(".tmp"):
                    results.append(
                        VerifyResult(f"{dataset}/{path.name}", False, ("leftover temporary file",))
                    )
            for number in sorted({n for n, _ in claimed}):
                ref = f"{dataset}@v{number}"
                results.append(self._verify_one(dataset, number, ref, claimed))
        return results

    def _verify_one(
        self, name: str, number: int, ref: str, claimed: set[tuple[int, str]]
    ) -> VerifyResult:
        problems: list[str] = []
        if (number, "json") not in claimed:
            problems.append("data file has no record (incomplete or interrupted write)")
            return VerifyResult(ref, False, tuple(problems))
        try:
            version = self._read_record(self._json_path(name, number))
        except DatasetStoreError as error:
            return VerifyResult(ref, False, (str(error),))
        if (version.dataset, version.number) != (name, number):
            problems.append("record says it is a different dataset or version number")
        csv_path = self._csv_path(name, number)
        if not csv_path.is_file():
            problems.append("data file is missing")
            return VerifyResult(ref, False, tuple(problems))
        data = csv_path.read_bytes()
        if len(data) != version.size_bytes:
            problems.append(f"size is {len(data)} bytes, record says {version.size_bytes}")
        if sha256_bytes(data) != version.sha256:
            problems.append("SHA-256 does not match the record (the file was changed)")
        else:
            try:
                again = prepare_csv(data, self.limits)
            except ValueError as error:
                problems.append(f"stored file no longer parses: {error}")
            else:
                if again.canonical != data:
                    problems.append("stored file is not in canonical form")
                if again.rows != version.rows or again.columns != version.columns:
                    problems.append("row or column details do not match the record")
        return VerifyResult(ref, not problems, tuple(problems))
