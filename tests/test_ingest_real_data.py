"""Ingest the prepared Adult and Abalone demo files (skipped until `make demo-data` has run).

These checks use the real files from Step 2 and exercise the store exactly as a user would.
They do not modify the demo files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from automl_ops.dataset_store import FileDatasetStore

DATA = Path(__file__).resolve().parents[1] / "data"
MANIFEST = DATA / "demo" / "manifest.json"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(), reason="demo data not prepared; run `make demo-data` first"
)


def demo_files() -> list[tuple[str, str, dict[str, object]]]:
    """(dataset key, role, manifest file entry) for every prepared file."""
    if not MANIFEST.exists():
        return []
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return [
        (key, role, meta)
        for key, entry in manifest["datasets"].items()
        for role, meta in entry["files"].items()
    ]


@pytest.mark.parametrize(("key", "role", "meta"), demo_files())
def test_demo_file_ingests_with_the_manifest_counts(
    tmp_path: Path, key: str, role: str, meta: dict[str, object]
) -> None:
    store = FileDatasetStore(tmp_path / "store")
    path = DATA / "demo" / str(meta["file"])
    before = path.read_bytes()
    name = f"{key}-{role.replace('_', '-')}"

    result = store.ingest(name, before, source_filename=path.name)
    v = result.version

    assert result.created and v.ref == f"{name}@v1"
    assert v.rows == meta["rows"]
    assert list(v.columns) == meta["columns"]
    # The prepared files are already canonical, so nothing should have been converted and the
    # version identity should equal the hash recorded in the Step 2 manifest.
    assert not v.reencoded
    assert v.sha256 == v.original_sha256 == meta["sha256"]
    assert store.version_path(v.ref).read_bytes() == before

    again = store.ingest(name, before, source_filename=path.name)
    assert not again.created and again.version == v
    assert path.read_bytes() == before  # the demo file itself is untouched
    assert all(r.ok for r in store.verify())


def test_all_six_demo_files_can_live_in_one_store(tmp_path: Path) -> None:
    store = FileDatasetStore(tmp_path / "store")
    for key, role, meta in demo_files():
        path = DATA / "demo" / str(meta["file"])
        store.ingest(f"{key}-{role.replace('_', '-')}", path.read_bytes())
    assert len(store.list_datasets()) == 6
    assert len(store.verify()) == 6 and all(r.ok for r in store.verify())
