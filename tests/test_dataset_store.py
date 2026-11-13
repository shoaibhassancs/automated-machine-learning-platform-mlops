"""Dataset store tests: versions, immutability, re-ingestion and verification (invented CSVs)."""

from __future__ import annotations

import codecs
import os
from pathlib import Path

import pytest

from automl_ops.dataset_store import (
    DatasetExists,
    DatasetNotFound,
    DatasetStoreError,
    FileDatasetStore,
    InvalidName,
    VersionNotFound,
    clean_source_filename,
)
from automl_ops.ingest import IngestError

CSV_V1 = b"id,name\n1,Ana\n2,Bo\n"
CSV_V2 = b"id,name\n1,Ana\n2,Bo\n3,Cy\n"


@pytest.fixture
def store(tmp_path: Path) -> FileDatasetStore:
    return FileDatasetStore(tmp_path / "store")


def files(store: FileDatasetStore, name: str) -> list[str]:
    return sorted(p.name for p in (store.root / name).iterdir())


def make_writable(path: Path) -> None:
    path.chmod(0o644)


# --- creating versions ---------------------------------------------------------------------


def test_first_ingest_creates_v1_with_files_and_record(store: FileDatasetStore) -> None:
    result = store.ingest("demo", CSV_V1, source_filename="people.csv")
    v = result.version
    assert result.created
    assert v.ref == "demo@v1" and v.number == 1
    assert (v.rows, v.columns) == (2, ("id", "name"))
    assert v.source_filename == "people.csv"
    assert files(store, "demo") == ["v0001.csv", "v0001.json"]
    assert store.version_path("demo@v1").read_bytes() == CSV_V1
    assert store.get_version("demo@v1") == v
    assert store.list_datasets() == ["demo"]
    assert store.list_versions("demo") == [v]


def test_stored_files_are_read_only(store: FileDatasetStore) -> None:
    store.ingest("demo", CSV_V1)
    for path in (store.version_path("demo@v1"), store.root / "demo" / "v0001.json"):
        assert path.stat().st_mode & 0o222 == 0
        if os.geteuid() != 0:  # root ignores permission bits
            with pytest.raises(PermissionError):
                path.open("ab")


def test_identity_is_the_hash_of_the_canonical_bytes(store: FileDatasetStore) -> None:
    semicolon_utf16 = codecs.BOM_UTF16_LE + "id;name\r\n1;Ana\r\n2;Bo\r\n".encode("utf-16-le")
    v = store.ingest("demo", semicolon_utf16, source_filename="odd.csv").version
    assert store.version_path("demo@v1").read_bytes() == CSV_V1  # converted form is stored
    assert v.sha256 != v.original_sha256  # provenance of the upload is kept separately
    assert (v.original_encoding, v.original_delimiter, v.reencoded) == ("utf-16", ";", True)
    assert v.original_size_bytes == len(semicolon_utf16)


def test_original_upload_bytes_are_not_retained(store: FileDatasetStore) -> None:
    store.ingest("demo", codecs.BOM_UTF8 + CSV_V1)
    assert files(store, "demo") == ["v0001.csv", "v0001.json"]
    assert not any(codecs.BOM_UTF8 in p.read_bytes() for p in (store.root / "demo").iterdir())


# --- re-ingestion and new versions ---------------------------------------------------------


@pytest.mark.parametrize("add_version", [False, True])
def test_identical_reingest_returns_the_same_version_and_writes_nothing(
    store: FileDatasetStore, add_version: bool
) -> None:
    first = store.ingest("demo", CSV_V1).version
    before = files(store, "demo")
    again = store.ingest("demo", CSV_V1, add_version=add_version)
    assert not again.created
    assert again.version == first
    assert files(store, "demo") == before


def test_same_content_in_other_packaging_maps_to_the_same_version(
    store: FileDatasetStore,
) -> None:
    first = store.ingest("demo", CSV_V1).version
    again = store.ingest("demo", CSV_V1.replace(b",", b";").replace(b"\n", b"\r\n"))
    assert not again.created and again.version == first
    assert again.upload_delimiter == ";" and again.upload_reencoded  # about this upload
    assert files(store, "demo") == ["v0001.csv", "v0001.json"]


def test_changed_content_needs_an_explicit_new_version_and_never_touches_v1(
    store: FileDatasetStore,
) -> None:
    v1 = store.ingest("demo", CSV_V1).version
    v1_bytes = store.version_path("demo@v1").read_bytes()
    with pytest.raises(DatasetExists, match="--add-version"):
        store.ingest("demo", CSV_V2)
    assert files(store, "demo") == ["v0001.csv", "v0001.json"]  # refusal wrote nothing

    v2 = store.ingest("demo", CSV_V2, add_version=True).version
    assert v2.ref == "demo@v2" and v2.sha256 != v1.sha256
    assert store.version_path("demo@v1").read_bytes() == v1_bytes
    assert store.get_version("demo@v1") == v1
    assert [v.ref for v in store.list_versions("demo")] == ["demo@v1", "demo@v2"]
    assert all(r.ok for r in store.verify())


def test_going_back_to_old_content_returns_the_old_version(store: FileDatasetStore) -> None:
    store.ingest("demo", CSV_V1)
    store.ingest("demo", CSV_V2, add_version=True)
    again = store.ingest("demo", CSV_V1, add_version=True)
    assert not again.created and again.version.ref == "demo@v1"


def test_add_version_to_a_missing_dataset_is_refused(store: FileDatasetStore) -> None:
    with pytest.raises(DatasetNotFound, match="without --add-version"):
        store.ingest("typo-name", CSV_V1, add_version=True)
    assert store.list_datasets() == []


def test_same_content_in_two_datasets_is_two_separate_versions(store: FileDatasetStore) -> None:
    a = store.ingest("one", CSV_V1).version
    b = store.ingest("two", CSV_V1).version
    assert a.sha256 == b.sha256 and a.ref != b.ref
    assert store.list_datasets() == ["one", "two"]


def test_rejected_csv_leaves_no_trace(store: FileDatasetStore) -> None:
    with pytest.raises(IngestError):
        store.ingest("demo", b"a,b\n")
    assert store.list_datasets() == []
    assert not store.root.exists() or list(store.root.iterdir()) == []


# --- names, references and file names ------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["", "A", "a b", "../x", "a/b", ".hidden", "-x", "a" * 65, "x.y", "é", "demo\n"]
)
def test_invalid_dataset_names_are_refused_and_create_nothing(
    store: FileDatasetStore, name: str
) -> None:
    with pytest.raises(InvalidName):
        store.ingest(name, CSV_V1)
    assert store.list_datasets() == []
    assert not store.root.exists()


@pytest.mark.parametrize("name", ["a", "adult", "adult_2", "my-data", "0abc", "a" * 64])
def test_valid_dataset_names_are_accepted(store: FileDatasetStore, name: str) -> None:
    assert store.ingest(name, CSV_V1).version.dataset == name


@pytest.mark.parametrize(
    "ref", ["demo", "demo@1", "demo@v", "demo@vx", "@v1", "Demo@v1", "../a@v1"]
)
def test_bad_version_references_are_refused(store: FileDatasetStore, ref: str) -> None:
    store.ingest("demo", CSV_V1)
    with pytest.raises(VersionNotFound):
        store.get_version(ref)


def test_unknown_version_and_dataset(store: FileDatasetStore) -> None:
    store.ingest("demo", CSV_V1)
    with pytest.raises(VersionNotFound):
        store.get_version("demo@v2")
    with pytest.raises(DatasetNotFound):
        store.list_versions("nothing")


def test_source_filename_is_information_only(store: FileDatasetStore) -> None:
    v = store.ingest("demo", CSV_V1, source_filename="../../etc/passwd").version
    assert v.source_filename == "passwd"
    assert files(store, "demo") == ["v0001.csv", "v0001.json"]  # nothing written elsewhere
    assert clean_source_filename("C:\\Users\\me\\data.csv") == "data.csv"
    assert clean_source_filename("") == "upload.csv"
    assert clean_source_filename("a\x00b\nc.csv") == "abc.csv"


# --- verification --------------------------------------------------------------------------


def test_verify_passes_on_untouched_store(store: FileDatasetStore) -> None:
    store.ingest("demo", CSV_V1)
    store.ingest("other", CSV_V2)
    results = store.verify()
    assert [r.ref for r in results] == ["demo@v1", "other@v1"]
    assert all(r.ok and not r.problems for r in results)


def test_verify_detects_a_changed_file(store: FileDatasetStore) -> None:
    store.ingest("demo", CSV_V1)
    path = store.version_path("demo@v1")
    make_writable(path)
    path.write_bytes(b"id,name\n1,Ana\n2,Eve\n")  # same size, different content
    (result,) = store.verify("demo")
    assert not result.ok
    assert any("SHA-256" in p for p in result.problems)


def test_verify_detects_a_size_change_and_a_missing_file(store: FileDatasetStore) -> None:
    store.ingest("demo", CSV_V1)
    path = store.version_path("demo@v1")
    make_writable(path)
    path.write_bytes(CSV_V1 + b"3,Cy\n")
    (changed,) = store.verify("demo")
    assert not changed.ok and any("size" in p for p in changed.problems)
    path.unlink()
    (missing,) = store.verify("demo")
    assert not missing.ok and any("missing" in p for p in missing.problems)


def test_verify_detects_a_wrong_record(store: FileDatasetStore) -> None:
    store.ingest("demo", CSV_V1)
    record = store.root / "demo" / "v0001.json"
    make_writable(record)
    record.write_text(record.read_text().replace('"rows": 2', '"rows": 99'))
    (result,) = store.verify("demo")
    assert not result.ok and any("row or column" in p for p in result.problems)


def test_verify_detects_a_corrupt_record(store: FileDatasetStore) -> None:
    store.ingest("demo", CSV_V1)
    record = store.root / "demo" / "v0001.json"
    make_writable(record)
    record.write_text("{not json")
    (result,) = store.verify("demo")
    assert not result.ok


def test_verify_reports_orphans_and_leftovers(store: FileDatasetStore) -> None:
    store.ingest("demo", CSV_V1)
    (store.root / "demo" / "v0002.csv").write_bytes(b"half written")
    (store.root / "demo" / "v0003.json.tmp").write_text("x")
    results = {r.ref: r for r in store.verify("demo")}
    assert results["demo@v1"].ok
    assert not results["demo@v2"].ok and "no record" in results["demo@v2"].problems[0]
    assert not results["demo/v0003.json.tmp"].ok


def test_an_orphan_data_file_is_never_overwritten_by_the_next_version(
    store: FileDatasetStore,
) -> None:
    store.ingest("demo", CSV_V1)
    orphan = store.root / "demo" / "v0002.csv"
    orphan.write_bytes(b"half written")
    v = store.ingest("demo", CSV_V2, add_version=True).version
    assert v.number == 3
    assert orphan.read_bytes() == b"half written"


def test_verify_unknown_dataset(store: FileDatasetStore) -> None:
    with pytest.raises(DatasetNotFound):
        store.verify("nothing")


def test_a_failed_write_does_not_leave_a_half_made_version(
    store: FileDatasetStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: object, **__: object) -> None:
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="simulated"):
        store.ingest("demo", CSV_V1)
    monkeypatch.undo()
    assert store.list_datasets() == []  # the empty dataset folder was removed


def test_corrupt_record_makes_listing_fail_loudly(store: FileDatasetStore) -> None:
    store.ingest("demo", CSV_V1)
    record = store.root / "demo" / "v0001.json"
    make_writable(record)
    record.write_text("{not json")
    with pytest.raises(DatasetStoreError, match="cannot read version record"):
        store.list_versions("demo")
