"""Small command-line tool for the dataset store (BUILD_PLAN Step 3).

    uv run python scripts/datasets.py ingest NAME FILE [--add-version]
    uv run python scripts/datasets.py list [NAME]
    uv run python scripts/datasets.py show NAME@vN
    uv run python scripts/datasets.py verify [NAME]
    uv run python scripts/datasets.py profile NAME@vN [--json]
    uv run python scripts/datasets.py validate NAME@vN [--target COLUMN] [--json]
    uv run python scripts/datasets.py task NAME@vN --target COLUMN [--task T] [--exclude A,B]
        [--json]
    uv run python scripts/datasets.py explore NAME@vN [--target COLUMN] [--task T] [--json]
    uv run python scripts/datasets.py schema NAME@vN --target COLUMN [--task T] [--exclude A,B]
        [--check OTHER@vN] [--json]

`profile`, `validate`, `task`, `explore` and `schema` (Steps 4-6) compute their reports on demand
from the stored version and never change it. Exit codes: 0 ok, 1 validation errors / verify failure
/ task errors / structural schema failures, 2 bad input.

The store lives in data/store/ by default (git-ignored). The CLI and, later, the web API both use
`automl_ops.dataset_store`; this script contains no storage logic of its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from automl_ops.dataset_store import DatasetStoreError, FileDatasetStore
from automl_ops.explore import explore_frame
from automl_ops.ingest import IngestError, read_limited
from automl_ops.profile import (
    ColumnProfile,
    DatasetProfile,
    VersionProfile,
    profile_frame,
    profile_version,
    read_version_bytes,
    read_version_frame,
)
from automl_ops.schema import (
    SchemaError,
    SchemaSpec,
    build_schema,
    check_csv_bytes,
    profile_for_schema,
)
from automl_ops.task import (
    TASK_TYPES,
    TaskConfig,
    TaskReport,
    check_task_config,
    class_counts,
    suggest_task,
)
from automl_ops.validate import validate


class CliError(Exception):
    """Bad command-line input (reported as `ERROR:` with exit code 2)."""


def _describe_upload(result_encoding: str, delimiter: str, reencoded: bool) -> str:
    shown = {"\t": "tab", ",": "comma", ";": "semicolon", "|": "pipe"}.get(delimiter, delimiter)
    note = "converted to canonical UTF-8 comma-separated form" if reencoded else "already canonical"
    return f"upload was {result_encoding}, {shown}-separated ({note})"


def cmd_ingest(store: FileDatasetStore, args: argparse.Namespace) -> int:
    path = Path(args.file)
    data = read_limited(path, store.limits)
    result = store.ingest(args.name, data, source_filename=path.name, add_version=args.add_version)
    v = result.version
    status = "CREATED" if result.created else "ALREADY EXISTS (identical content; nothing written)"
    print(f"{status}: {v.ref}")
    print(f"  rows {v.rows:,}, columns {len(v.columns)}, sha256 {v.sha256}")
    upload = _describe_upload(
        result.upload_encoding, result.upload_delimiter, result.upload_reencoded
    )
    print(f"  {upload}")
    return 0


def cmd_list(store: FileDatasetStore, args: argparse.Namespace) -> int:
    if args.name:
        for v in store.list_versions(args.name):
            print(
                f"{v.ref}  rows {v.rows:,}  cols {len(v.columns)}  {v.short_hash}  {v.created_at}"
            )
        return 0
    names = store.list_datasets()
    if not names:
        print("no datasets stored")
    for name in names:
        versions = store.list_versions(name)
        print(f"{name}: {len(versions)} version(s), latest {versions[-1].ref if versions else '-'}")
    return 0


def cmd_show(store: FileDatasetStore, args: argparse.Namespace) -> int:
    version = store.get_version(args.ref)
    print(json.dumps(version.to_record(), indent=2, sort_keys=True))
    print(f"file: {store.version_path(args.ref)}")
    return 0


def cmd_verify(store: FileDatasetStore, args: argparse.Namespace) -> int:
    results = store.verify(args.name)
    if not results:
        print("nothing to verify")
    for r in results:
        print(
            f"{'PASS' if r.ok else 'FAIL'}  {r.ref}"
            + ("" if r.ok else f"  {'; '.join(r.problems)}")
        )
    return 0 if all(r.ok for r in results) else 1


def _summary(column: ColumnProfile) -> str:
    if column.numeric is not None:
        n = column.numeric
        return f"{n.minimum:g} .. {n.maximum:g}, mean {n.mean:.4g}"
    if column.kind == "empty":
        return "all values missing"
    shown = ", ".join(f"{value!r} ({count:,})" for value, count in column.top_values[:3])
    return f"top: {shown}" if shown else ""


def format_profile(vp: VersionProfile) -> str:
    p = vp.profile
    lines = [
        f"{vp.version.ref}  sha256 {vp.version.short_hash}  "
        f"{p.rows:,} rows x {p.column_count} columns",
        f"duplicate rows: {p.duplicate_rows:,} extra copies; rows with a missing value: "
        f"{p.rows_with_missing:,}; missing cells: {p.missing_cells:,}",
        "",
        f"{'#':>3}  {'column':<20} {'kind':<11} {'missing':>13} {'unique':>8}  summary",
    ]
    for c in p.columns:
        missing = f"{c.missing:,} ({c.missing_pct:.1f}%)" if c.missing else "0"
        name = c.name if len(c.name) <= 20 else c.name[:19] + "~"
        lines.append(
            f"{c.position:>3}  {name:<20} {c.kind:<11} {missing:>13} {c.unique:>8,}  {_summary(c)}"
        )
    return "\n".join(lines)


def cmd_profile(store: FileDatasetStore, args: argparse.Namespace) -> int:
    vp = profile_version(store, args.ref)
    if args.json:
        print(json.dumps(vp.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    else:
        print(format_profile(vp))
    return 0


def cmd_validate(store: FileDatasetStore, args: argparse.Namespace) -> int:
    vp = profile_version(store, args.ref)
    report = validate(vp.profile, target=args.target)
    if args.json:
        payload = {"dataset_ref": vp.version.ref, "sha256": vp.version.sha256, **report.to_dict()}
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0 if report.ok else 1
    target = f", target {args.target!r}" if args.target else ", no target given"
    print(f"{vp.version.ref}{target}")
    for f in report.findings:
        where = f"[{f.column}] " if f.column else ""
        print(f"  {f.severity.upper():<7} {f.code}: {where}{f.message}")
    if not report.findings:
        print("  no findings")
    print(f"{report.errors} error(s), {report.warnings} warning(s), {report.infos} info")
    return 0 if report.ok else 1


def _excluded(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(name.strip() for name in raw.split(",") if name.strip())


def _task_config(profile: DatasetProfile, args: argparse.Namespace) -> TaskConfig:
    """The configuration from the command line; without --task, the suggested task is used."""
    task = args.task
    if task is None:
        suggestion = suggest_task(profile, args.target)
        if suggestion is None:
            raise CliError(
                f"cannot suggest a task for target {args.target!r} (it does not exist or has no "
                "values); pass --task"
            )
        task = suggestion.task
    return TaskConfig(target=args.target, task=task, excluded=_excluded(args.exclude))


def _print_task_report(report: TaskReport, ref: str, features: tuple[str, ...]) -> None:
    c = report.config
    s = report.suggestion
    suggested = f"{s.task} - {s.reason}" if s else "none (no suggestion possible)"
    excluded = ", ".join(c.excluded) if c.excluded else "none"
    print(f"{ref}  target {c.target!r}, task {c.task}, excluded: {excluded}")
    print(f"  suggested task: {suggested}")
    print(f"  input columns ({len(features)}): {', '.join(features) or '-'}")
    for f in report.findings:
        where = f"[{f.column}] " if f.column else ""
        print(f"  {f.severity.upper():<7} {f.code}: {where}{f.message}")
    if not report.findings:
        print("  no findings")
    print(f"{report.errors} error(s), {report.warnings} warning(s), {report.infos} info")


def cmd_task(store: FileDatasetStore, args: argparse.Namespace) -> int:
    version, frame = read_version_frame(store, args.ref)
    profile = profile_frame(frame)
    config = _task_config(profile, args)
    report = check_task_config(profile, class_counts_for(frame, config), config)
    features = config.feature_columns(profile)
    if args.json:
        payload = {
            "dataset_ref": version.ref,
            "sha256": version.sha256,
            **report.to_dict(),
            "feature_columns": list(features),
        }
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        _print_task_report(report, version.ref, features)
    return 0 if report.ok else 1


def class_counts_for(frame: pd.DataFrame, config: TaskConfig) -> dict[str, int]:
    """Class counts for a classification config whose target exists; otherwise none."""
    if config.task == "classification" and config.target in frame.columns:
        return class_counts(frame, config.target)
    return {}


def _print_explore(data: dict[str, Any], ref: str) -> None:
    print(f"{ref}  {data['rows']:,} rows x {len(data['columns'])} columns")
    print(f"sample rows (first {len(data['sample_rows'])}):")
    for row in data["sample_rows"]:
        print("  " + " | ".join("" if v is None else str(v) for v in row.values()))
    print("histograms (counts per equal-width bin):")
    for h in data["histograms"]:
        print(f"  {h['column']}: {h['min']:g} .. {h['max']:g}  {h['counts']}")
    print("strongest numeric correlations (Pearson):")
    for pair in data["correlation"]["top_pairs"][:5]:
        print(f"  {pair['a']} ~ {pair['b']}: {pair['r']:+.3f}")
    balance = data["class_balance"]
    if balance:
        print(
            f"class balance: {balance['total_classes']} classes, "
            f"largest share {balance['largest_class_share']:.3f}"
        )
        for c in balance["classes"][:10]:
            print(f"  {c['value']!r}: {c['count']:,} ({c['share']:.3f})")


def cmd_explore(store: FileDatasetStore, args: argparse.Namespace) -> int:
    version, frame = read_version_frame(store, args.ref)
    profile = profile_frame(frame)
    config = _task_config(profile, args) if args.target else None
    data = explore_frame(frame, profile, config)
    if args.json:
        payload = {"dataset_ref": version.ref, "sha256": version.sha256, **data}
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        _print_explore(data, version.ref)
    return 0


def _print_spec(spec: SchemaSpec) -> None:
    print(
        f"schema from {spec.training_rows:,} training rows; {len(spec.columns)} input column(s); "
        f"target {spec.target!r}; excluded: {', '.join(spec.excluded) or 'none'}"
    )
    for c in spec.columns:
        if c.observed_min is not None:
            seen = f"seen {c.observed_min:g} .. {c.observed_max:g}"
        elif c.observed_categories is not None:
            shown = ", ".join(repr(v) for v in c.observed_categories[:6])
            more = len(c.observed_categories) - 6
            seen = f"{len(c.observed_categories)} categories: {shown}" + (
                f", +{more} more" if more > 0 else ""
            )
        else:
            seen = "any text"
        blanks = "blanks seen" if c.had_missing else "no blanks seen"
        print(f"  {c.position:>3}  {c.name:<20} {c.kind:<11} {seen}; {blanks}")


def cmd_schema(store: FileDatasetStore, args: argparse.Namespace) -> int:
    version, frame = read_version_frame(store, args.ref)
    config = _task_config(profile_frame(frame), args)
    profile = profile_for_schema(frame)
    task_report = check_task_config(profile, class_counts_for(frame, config), config)
    if not task_report.ok:
        _print_task_report(task_report, version.ref, config.feature_columns(profile))
        print("ERROR: fix the task configuration before building a schema", file=sys.stderr)
        return 1
    spec = build_schema(profile, config)

    checked = None
    if args.check:
        other, data = read_version_bytes(store, args.check)
        checked = (other, check_csv_bytes(spec, data))

    if args.json:
        payload: dict[str, object] = {
            "dataset_ref": version.ref,
            "sha256": version.sha256,
            "schema": spec.to_dict(),
        }
        if checked:
            payload["check"] = {
                "dataset_ref": checked[0].ref,
                "sha256": checked[0].sha256,
                **checked[1].to_dict(),
            }
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"{version.ref}")
        _print_spec(spec)
        if checked:
            other, report = checked
            print(f"checked {other.ref} ({report.rows:,} rows): ", end="")
            print("STRUCTURALLY OK" if report.ok else "STRUCTURAL FAILURES")
            for f in report.failures:
                where = f"[{f.column}] " if f.column else ""
                print(f"  ERROR   {f.code}: {where}{f.message}")
            print(f"  observations about the training data ({len(report.observations)}):")
            for f in report.observations:
                where = f"[{f.column}] " if f.column else ""
                print(f"  INFO    {f.code}: {where}{f.message}")
    return 0 if checked is None or checked[1].ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dataset store tool")
    parser.add_argument("--store", type=Path, default=Path("data/store"), help="store folder")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="store a CSV as a dataset version")
    p_ingest.add_argument("name")
    p_ingest.add_argument("file")
    p_ingest.add_argument("--add-version", action="store_true")
    p_list = sub.add_parser("list", help="list datasets, or the versions of one dataset")
    p_list.add_argument("name", nargs="?")
    p_show = sub.add_parser("show", help="show one version's record")
    p_show.add_argument("ref", help="e.g. adult@v1")
    p_verify = sub.add_parser("verify", help="re-check stored files against their records")
    p_verify.add_argument("name", nargs="?")
    p_profile = sub.add_parser("profile", help="describe each column of a stored version")
    p_profile.add_argument("ref", help="e.g. adult@v1")
    p_profile.add_argument("--json", action="store_true", help="machine-readable output")
    p_validate = sub.add_parser("validate", help="check a stored version against the rules")
    p_validate.add_argument("ref", help="e.g. adult@v1")
    p_validate.add_argument("--target", help="also check this column as the prediction target")
    p_validate.add_argument("--json", action="store_true", help="machine-readable output")

    def add_task_options(p: argparse.ArgumentParser, *, target_required: bool) -> None:
        p.add_argument("--target", required=target_required, help="the column to predict")
        p.add_argument(
            "--task",
            choices=TASK_TYPES,
            help="task type (default: the suggestion, which is advisory)",
        )
        p.add_argument("--exclude", help="comma-separated columns to leave out of the inputs")
        p.add_argument("--json", action="store_true", help="machine-readable output")

    p_task = sub.add_parser("task", help="suggest and check a task configuration")
    p_task.add_argument("ref", help="e.g. adult@v1")
    add_task_options(p_task, target_required=True)
    p_explore = sub.add_parser("explore", help="exploration numbers for the preview page")
    p_explore.add_argument("ref", help="e.g. adult@v1")
    add_task_options(p_explore, target_required=False)
    p_schema = sub.add_parser("schema", help="build the feature schema; optionally check a version")
    p_schema.add_argument("ref", help="the training version, e.g. adult@v1")
    p_schema.add_argument("--check", metavar="REF", help="another version to check against it")
    add_task_options(p_schema, target_required=True)

    args = parser.parse_args(argv)
    store = FileDatasetStore(args.store)
    commands = {
        "ingest": cmd_ingest,
        "list": cmd_list,
        "show": cmd_show,
        "verify": cmd_verify,
        "profile": cmd_profile,
        "validate": cmd_validate,
        "task": cmd_task,
        "explore": cmd_explore,
        "schema": cmd_schema,
    }
    try:
        return commands[args.command](store, args)
    except (IngestError, DatasetStoreError, SchemaError, CliError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
