"""Tests for scripts/datasets.py, run as a real subprocess against a temporary store."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "datasets.py"


def run(store: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--store", str(store), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


def test_cli_full_cycle(tmp_path: Path) -> None:
    store = tmp_path / "store"
    csv1 = tmp_path / "people.csv"
    csv1.write_text("id;name\r\n1;Ana\r\n2;Bo\r\n")
    csv2 = tmp_path / "people2.csv"
    csv2.write_text("id,name\n1,Ana\n2,Bo\n3,Cy\n")

    made = run(store, "ingest", "demo", str(csv1))
    assert made.returncode == 0 and "CREATED: demo@v1" in made.stdout
    assert "semicolon" in made.stdout and "converted" in made.stdout

    again = run(store, "ingest", "demo", str(csv1))
    assert again.returncode == 0 and "ALREADY EXISTS" in again.stdout

    refused = run(store, "ingest", "demo", str(csv2))
    assert refused.returncode == 2 and "--add-version" in refused.stderr

    added = run(store, "ingest", "demo", str(csv2), "--add-version")
    assert added.returncode == 0 and "CREATED: demo@v2" in added.stdout

    listing = run(store, "list")
    assert "demo: 2 version(s), latest demo@v2" in listing.stdout
    versions = run(store, "list", "demo")
    assert "demo@v1" in versions.stdout and "demo@v2" in versions.stdout

    shown = run(store, "show", "demo@v1")
    assert shown.returncode == 0 and '"rows": 2' in shown.stdout and "v0001.csv" in shown.stdout

    good = run(store, "verify")
    assert good.returncode == 0 and good.stdout.count("PASS") == 2

    (store / "demo" / "v0001.csv").chmod(0o644)
    (store / "demo" / "v0001.csv").write_bytes(b"id,name\n1,Ana\n2,Eve\n")
    bad = run(store, "verify", "demo")
    assert bad.returncode == 1 and "FAIL  demo@v1" in bad.stdout


def test_cli_reports_bad_input_without_a_traceback(tmp_path: Path) -> None:
    store = tmp_path / "store"
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    result = run(store, "ingest", "demo", str(empty))
    assert result.returncode == 2
    assert "ERROR: the file is empty" in result.stderr
    assert "Traceback" not in result.stderr
    assert not store.exists()

    missing = run(store, "ingest", "demo", str(tmp_path / "nope.csv"))
    assert missing.returncode == 2 and "Traceback" not in missing.stderr

    bad_name = run(store, "ingest", "Bad Name", str(empty))
    assert bad_name.returncode == 2 and "invalid dataset name" in bad_name.stderr


def _invented_csv(path: Path) -> None:
    rows = ["id,group,score,target"]
    for i in range(120):
        rows.append(f"{i},g{i % 4},{i % 10}.5,{i % 2}")
    path.write_text("\n".join(rows) + "\n")


def test_cli_profile_and_validate(tmp_path: Path) -> None:
    import json

    store = tmp_path / "store"
    source = tmp_path / "data.csv"
    _invented_csv(source)
    assert run(store, "ingest", "demo", str(source)).returncode == 0

    text = run(store, "profile", "demo@v1")
    assert text.returncode == 0
    assert "demo@v1" in text.stdout and "120 rows x 4 columns" in text.stdout
    assert "group" in text.stdout and "categorical" in text.stdout and "float" in text.stdout

    as_json = run(store, "profile", "demo@v1", "--json")
    payload = json.loads(as_json.stdout)
    assert payload["dataset_ref"] == "demo@v1" and payload["rows"] == 120
    assert [c["name"] for c in payload["column_profiles"]] == ["id", "group", "score", "target"]
    assert as_json.stdout == run(store, "profile", "demo@v1", "--json").stdout  # deterministic

    ok = run(store, "validate", "demo@v1", "--target", "target")
    assert ok.returncode == 0 and "id_like_integer" in ok.stdout  # a warning does not fail
    assert "0 error(s), 1 warning(s)" in ok.stdout

    bad = run(store, "validate", "demo@v1", "--target", "nope")
    assert bad.returncode == 1 and "unknown_target" in bad.stdout

    bad_json = run(store, "validate", "demo@v1", "--target", "nope", "--json")
    report = json.loads(bad_json.stdout)
    assert bad_json.returncode == 1 and report["ok"] is False and report["errors"] == 1
    assert report["dataset_ref"] == "demo@v1"


def test_cli_profile_leaves_the_store_untouched_and_refuses_tampered_files(tmp_path: Path) -> None:
    store = tmp_path / "store"
    source = tmp_path / "data.csv"
    _invented_csv(source)
    run(store, "ingest", "demo", str(source))
    listing = sorted(p.name for p in (store / "demo").iterdir())
    before = (store / "demo" / "v0001.csv").read_bytes()
    run(store, "profile", "demo@v1")
    run(store, "validate", "demo@v1")
    assert sorted(p.name for p in (store / "demo").iterdir()) == listing
    assert (store / "demo" / "v0001.csv").read_bytes() == before

    (store / "demo" / "v0001.csv").chmod(0o644)
    (store / "demo" / "v0001.csv").write_bytes(before.replace(b"g1", b"gX"))
    refused = run(store, "profile", "demo@v1")
    assert refused.returncode == 2 and "does not match its recorded SHA-256" in refused.stderr
    assert "Traceback" not in refused.stderr
    assert run(store, "profile", "demo@v9").returncode == 2


def _ingest_demo(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    source = tmp_path / "data.csv"
    _invented_csv(source)
    assert run(store, "ingest", "demo", str(source)).returncode == 0
    return store


def _snapshot(store: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(store)): p.read_bytes() for p in sorted(store.rglob("*")) if p.is_file()
    }


def test_cli_task_suggests_checks_and_reports_errors(tmp_path: Path) -> None:
    import json

    store = _ingest_demo(tmp_path)
    ok = run(store, "task", "demo@v1", "--target", "target")
    assert ok.returncode == 0 and "suggested task: classification" in ok.stdout
    assert "input columns (3): id, group, score" in ok.stdout

    as_json = json.loads(run(store, "task", "demo@v1", "--target", "target", "--json").stdout)
    assert as_json["ok"] is True and as_json["config"]["task"] == "classification"
    assert as_json["feature_columns"] == ["id", "group", "score"]

    override = run(store, "task", "demo@v1", "--target", "target", "--task", "regression")
    assert override.returncode == 0 and "task_differs_from_suggestion" in override.stdout

    no_target = run(store, "task", "demo@v1", "--target", "nope")
    assert no_target.returncode == 2 and "cannot suggest a task" in no_target.stderr
    bad_target = run(store, "task", "demo@v1", "--target", "nope", "--task", "classification")
    assert bad_target.returncode == 1 and "unknown_target" in bad_target.stdout

    no_features = run(store, "task", "demo@v1", "--target", "target", "--exclude", "id,group,score")
    assert no_features.returncode == 1 and "no_feature_columns" in no_features.stdout
    assert "Traceback" not in no_features.stderr


def test_cli_explore_text_and_json(tmp_path: Path) -> None:
    import json

    store = _ingest_demo(tmp_path)
    text = run(store, "explore", "demo@v1", "--target", "target")
    assert (
        text.returncode == 0 and "histograms" in text.stdout and "class balance: 2" in text.stdout
    )
    payload = json.loads(run(store, "explore", "demo@v1", "--json").stdout)
    assert payload["dataset_ref"] == "demo@v1" and len(payload["sample_rows"]) == 10
    assert payload["class_balance"] is None  # no target given
    assert all(len(h["counts"]) == 20 for h in payload["histograms"])


def test_cli_schema_separates_failures_from_observations(tmp_path: Path) -> None:
    import json

    store = _ingest_demo(tmp_path)
    same = run(store, "schema", "demo@v1", "--target", "target", "--check", "demo@v1")
    assert same.returncode == 0 and "STRUCTURALLY OK" in same.stdout
    assert "observations about the training data (0)" in same.stdout

    # drift-like data: well-formed, but outside what the training data contained
    drifted = tmp_path / "drifted.csv"
    rows = ["id,group,score,target"] + [f"{1000 + i},gNEW,{500 + i}.5,0" for i in range(60)]
    drifted.write_text("\n".join(rows) + "\n")
    assert run(store, "ingest", "demo", str(drifted), "--add-version").returncode == 0
    seen = run(store, "schema", "demo@v1", "--target", "target", "--check", "demo@v2")
    assert seen.returncode == 0 and "STRUCTURALLY OK" in seen.stdout
    assert "unseen_categories" in seen.stdout and "out_of_observed_range" in seen.stdout

    # malformed: a non-number in a numeric column is a structural failure
    broken = tmp_path / "broken.csv"
    rows = ["id,group,score,target"] + [f"{i},g0,abc,0" for i in range(60)]
    broken.write_text("\n".join(rows) + "\n")
    assert run(store, "ingest", "demo", str(broken), "--add-version").returncode == 0
    bad = run(store, "schema", "demo@v1", "--target", "target", "--check", "demo@v3")
    assert bad.returncode == 1 and "STRUCTURAL FAILURES" in bad.stdout
    assert "malformed_numeric" in bad.stdout

    payload = json.loads(
        run(store, "schema", "demo@v1", "--target", "target", "--check", "demo@v3", "--json").stdout
    )
    assert payload["schema"]["schema_format"] == 1
    assert payload["check"]["ok"] is False and payload["check"]["dataset_ref"] == "demo@v3"
    assert [c["name"] for c in payload["schema"]["columns"]] == ["id", "group", "score"]


def test_cli_schema_refuses_an_unusable_task_configuration(tmp_path: Path) -> None:
    store = _ingest_demo(tmp_path)
    refused = run(store, "schema", "demo@v1", "--target", "nope", "--task", "classification")
    assert refused.returncode == 1 and "unknown_target" in refused.stdout
    assert "fix the task configuration" in refused.stderr


def test_cli_task_explore_schema_leave_the_store_untouched_and_refuse_tampered_files(
    tmp_path: Path,
) -> None:
    store = _ingest_demo(tmp_path)
    before = _snapshot(store)
    run(store, "task", "demo@v1", "--target", "target")
    run(store, "explore", "demo@v1", "--target", "target")
    run(store, "schema", "demo@v1", "--target", "target", "--check", "demo@v1")
    assert _snapshot(store) == before

    path = store / "demo" / "v0001.csv"
    path.chmod(0o644)
    path.write_bytes(before["demo/v0001.csv"].replace(b"g1", b"gX"))
    for args in (
        ("task", "demo@v1", "--target", "target"),
        ("explore", "demo@v1"),
        ("schema", "demo@v1", "--target", "target"),
    ):
        refused = run(store, *args)
        assert refused.returncode == 2 and "does not match its recorded SHA-256" in refused.stderr
        assert "Traceback" not in refused.stderr
