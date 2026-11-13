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
    uv run python scripts/datasets.py train NAME@vN --target COLUMN [--task T] [--exclude A,B]
        [--algorithms a,b] [--folds 5] [--seed 42] [--test-fraction 0.2] [--track] [--json]
    uv run python scripts/datasets.py finalize NAME@vN --target COLUMN [--task T] [--exclude A,B]
        [--algorithms a,b] [--folds 5] [--seed 42] [--test-fraction 0.2]
        [--selection-metric M] --confirm [--track] [--json]
    uv run python scripts/datasets.py package-verify [PACKAGE_ID] [--json]
    uv run python scripts/datasets.py track-package PACKAGE_ID [--json]
    uv run python scripts/datasets.py jobs [--kind K] [--task T] [--dataset REF] [--algorithm A]
        [--state S] [--limit N] [--json]
    uv run python scripts/datasets.py job JOB_KEY [--json]
    uv run python scripts/datasets.py compare JOB_KEY JOB_KEY... [--metric M] [--json]
    uv run python scripts/datasets.py reproduce JOB_KEY [--tolerance X] [--json]

`profile`, `validate`, `task`, `explore` and `schema` (Steps 4-6) compute their reports on demand
from the stored version and never change it. Exit codes: 0 ok, 1 validation errors / verify failure
/ task errors / structural schema failures / an algorithm failed, 2 bad input or an infeasible
training run. `train` (Step 9) cross-validates on a training part only; the held-out test part is
reserved and never used, and no test-set score is produced. `finalize` (Steps 11-12) trains,
selects the best algorithm from the cross-validation results alone, freezes that choice, refits it
on the training part, scores it **once** on the held-out test part and writes a sealed model package
under `data/models/` (`--models`); it refuses to run without `--confirm`, and exit code 2 also
covers a refused or failed finalization. `package-verify` re-checks packages (exit 1 if any fails).
`--track` (Steps 13-14) logs the job to the local MLflow store (`data/mlflow`, `--mlflow-dir`)
through `automl_ops.tracking`: after the package is sealed for `finalize`, from the
cross-validation results for `train`. If tracking fails the package stays sealed and the exit code
is 3 ("tracking pending"); `track-package` then logs, or finishes logging, a sealed package without
retraining or reading any test row. See docs/tracking.md.
`jobs`, `job` and `compare` (Step 15) read the MLflow records and change nothing; `compare` uses
cross-validation numbers only. `reproduce` (Step 16) re-runs a job's cross-validation from its
record and compares; it never reads the held-out test part and writes nothing (exit 0 reproduced,
1 not reproduced, 2 cannot run). See docs/reproducibility.md.

The store lives in data/store/ by default (git-ignored). The CLI and, later, the web API both use
`automl_ops.dataset_store`; this script contains no storage logic of its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from automl_ops.algorithms import AlgorithmError, algorithms_for
from automl_ops.compare import CompareError, compare_jobs
from automl_ops.dataset_store import DatasetStoreError, FileDatasetStore
from automl_ops.explore import explore_frame
from automl_ops.finalize import DatasetInfo, FinalizeError, finalize
from automl_ops.ingest import IngestError, read_limited
from automl_ops.package import (
    PackageError,
    leftover_staging,
    library_versions,
    list_packages,
    verify_package,
)
from automl_ops.pipeline import PipelineError
from automl_ops.profile import (
    ColumnProfile,
    DatasetProfile,
    VersionProfile,
    profile_frame,
    profile_version,
    read_version_bytes,
    read_version_frame,
)
from automl_ops.reproduce import (
    DEFAULT_REL_TOLERANCE,
    ReproductionError,
    ReproductionReport,
    reproduce_job,
    sealed_manifest_selection_sha,
    sealed_test_result,
)
from automl_ops.schema import (
    SchemaError,
    SchemaSpec,
    build_schema,
    check_csv_bytes,
    profile_for_schema,
)
from automl_ops.selection import SelectionError, freeze_selection, select_model
from automl_ops.task import (
    TASK_TYPES,
    TaskConfig,
    TaskReport,
    check_task_config,
    class_counts,
    suggest_task,
)
from automl_ops.tracking import (
    AlreadyTrackedError,
    JobRecord,
    TrackedJob,
    TrackingError,
    TrackingStore,
    list_jobs,
    log_job,
    read_job,
    record_from_package,
    record_from_training,
)
from automl_ops.train import (
    AlgorithmRequest,
    ProgressEvent,
    TrainingError,
    TrainingResult,
    TrainSettings,
    run_training,
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


def _print_progress(event: ProgressEvent) -> None:
    head = f"  [{event.algorithm_index}/{event.algorithm_count}] {event.algorithm}"
    if event.kind == "fold_finished":
        print(f"{head} fold {event.fold}/{event.folds} ({event.seconds:.2f}s)", file=sys.stderr)
    elif event.kind == "algorithm_failed":
        print(f"{head} FAILED: {event.message}", file=sys.stderr)


def _score(summary: dict[str, dict[str, object]], name: str) -> str:
    entry = summary.get(name)
    if not entry or entry["mean"] is None:
        return "n/a"
    mean = float(str(entry["mean"]))
    return f"{mean:.4f}" + (
        f" +-{float(str(entry['std'])):.4f}" if entry["std"] is not None else ""
    )


def _print_training(result: TrainingResult, ref: str) -> None:
    split = result.split
    print(
        f"{ref}  target {result.target!r}, task {result.task}, seed {result.settings.seed}\n"
        f"rows {split.to_dict()['rows']:,}: training part {len(split.train_rows):,}, held-out test "
        f"part {len(split.test_rows):,} (reserved, not used); "
        f"{result.folds_used}-fold cross-validation"
    )
    print(f"input columns ({len(result.plan.used)}): {', '.join(result.plan.used)}")
    for d in result.plan.dropped:
        print(f"left out: {d.name!r} ({d.kind}): {d.reason}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    metrics = (
        ("accuracy", "f1_macro", "roc_auc")
        if result.task == "classification"
        else ("rmse", "mae", "r2")
    )
    print(f"\n{'algorithm':<26} {'status':<7} " + " ".join(f"{m:<17}" for m in metrics) + "seconds")
    for r in result.results:
        scores = " ".join(f"{_score(r.summary, m):<17}" for m in metrics) if r.summary else ""
        print(f"{r.key:<26} {r.status:<7} {scores or 'n/a':<53}{r.seconds:.1f}")
        if r.error:
            print(f"    {r.error['type']}: {r.error['message']}")
    print(
        "\nScores are mean +- standard deviation over folds, from cross-validation on the training "
        "part only (no test-set scores)."
    )


EXIT_TRACKING_PENDING = 3


def _track(record: JobRecord, args: argparse.Namespace) -> tuple[TrackedJob | None, str | None]:
    """Log a record; returns (tracked job, None) or (None, the reason it is pending)."""
    try:
        return log_job(record, TrackingStore(args.mlflow_dir)), None
    except TrackingError as error:
        return None, str(error)


def _tracking_payload(tracked: TrackedJob | None, problem: str | None) -> dict[str, Any]:
    if tracked is None:
        return {"state": "pending", "problem": problem}
    return {
        "state": "complete",
        "job_key": tracked.job_key,
        "parent_run_id": tracked.parent_run_id,
        "child_run_ids": tracked.child_run_ids,
        "resumed": tracked.resumed,
    }


def _print_tracking(
    tracked: TrackedJob | None, problem: str | None, args: argparse.Namespace
) -> None:
    store = TrackingStore(args.mlflow_dir)
    if tracked is None:
        print(f"\nTRACKING PENDING: {problem}", file=sys.stderr)
        return
    verb = "resumed and completed" if tracked.resumed else "logged"
    print(
        f"\ntracking: {verb} job {tracked.job_key} ({len(tracked.child_run_ids)} algorithm "
        f"runs) in {store.db_path}"
    )


def cmd_train(store: FileDatasetStore, args: argparse.Namespace) -> int:
    version, frame = read_version_frame(store, args.ref)
    config = _task_config(profile_frame(frame), args)
    keys = (
        [k.strip() for k in args.algorithms.split(",") if k.strip()]
        if args.algorithms
        else [a.key for a in algorithms_for(config.task)]
    )
    settings = TrainSettings(test_fraction=args.test_fraction, seed=args.seed, folds=args.folds)
    result = run_training(
        frame,
        config,
        [AlgorithmRequest(k) for k in keys],
        settings,
        progress=None if args.json else _print_progress,
    )
    tracked, problem = (
        _track(
            record_from_training(
                result.to_dict(), DatasetInfo.from_version(version).to_dict(), library_versions()
            ),
            args,
        )
        if args.track
        else (None, None)
    )
    if args.json:
        payload = {"dataset_ref": version.ref, "sha256": version.sha256, **result.to_dict()}
        if args.track:
            payload["tracking"] = _tracking_payload(tracked, problem)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        _print_training(result, version.ref)
        if args.track:
            _print_tracking(tracked, problem, args)
    if args.track and tracked is None:
        return EXIT_TRACKING_PENDING
    return 0 if result.ok else 1


def _print_finalize(report: dict[str, Any], result: Any, training: TrainingResult) -> None:
    print("\nSelection")
    for entry in report["ranking"]:
        std = entry["metric_std"]
        spread = f" +-{std:.4f}" if std is not None else ""
        print(
            f"  {entry['rank']}. {entry['algorithm']:<26} {report['metric']} "
            f"{entry['metric_mean']:.4f}{spread}  ({entry['decided_by']})"
        )
    for item in report["excluded"]:
        print(f"  excluded: {item['algorithm']}: {item['reason']}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    print(report["explanation"])
    if report["positive_class"] is not None:
        print(f"positive class: {report['positive_class']!r}")
    evaluation = result.test_result["evaluation"]
    scores = ", ".join(
        f"{name} {value:.4f}" for name, value in evaluation["metrics"].items() if value is not None
    )
    print(
        f"\nFinal test (held-out part, {evaluation['rows']:,} rows, read once, selected model "
        f"only): {scores}"
    )
    print(f"NOTE: {result.test_result['limitation']}")
    print(f"package {result.package_id} sealed at {result.path}")


def cmd_finalize(store: FileDatasetStore, args: argparse.Namespace) -> int:
    if not args.confirm:
        raise CliError(
            "finalize reads the held-out test part once for the selected model. "
            "Add --confirm to proceed."
        )
    version, frame = read_version_frame(store, args.ref)
    config = _task_config(profile_frame(frame), args)
    keys = (
        [k.strip() for k in args.algorithms.split(",") if k.strip()]
        if args.algorithms
        else [a.key for a in algorithms_for(config.task)]
    )
    settings = TrainSettings(test_fraction=args.test_fraction, seed=args.seed, folds=args.folds)
    # One training run feeds the selection, the freeze, the refit and the package.
    training = run_training(
        frame,
        config,
        [AlgorithmRequest(k) for k in keys],
        settings,
        progress=None if args.json else _print_progress,
    )
    report = select_model(training, args.selection_metric)
    frozen = freeze_selection(training, report)
    result = finalize(
        frame,
        config,
        training,
        frozen,
        DatasetInfo.from_version(version),
        args.models,
        confirm_final_test=True,
    )
    tracked, problem = (None, None)
    if args.track:
        # Only after the package is sealed: tracking can never change or spend a package.
        try:
            tracked, problem = _track(record_from_package(args.models, result.package_id), args)
        except TrackingError as error:
            problem = str(error)
    if args.json:
        payload = {
            "dataset_ref": version.ref,
            "package_id": result.package_id,
            "path": str(result.path),
            "selection": report.to_dict(),
            "selection_sha256": frozen.sha256,
            "test_result": result.test_result,
        }
        if args.track:
            payload["tracking"] = _tracking_payload(tracked, problem)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        _print_training(training, version.ref)
        _print_finalize(report.to_dict(), result, training)
        if args.track:
            _print_tracking(tracked, problem, args)
            if tracked is None:
                print(
                    f"The package is sealed and unaffected. Retry with: "
                    f"scripts/datasets.py track-package {result.package_id}",
                    file=sys.stderr,
                )
    if args.track and tracked is None:
        return EXIT_TRACKING_PENDING
    return 0


def cmd_track_package(store: FileDatasetStore, args: argparse.Namespace) -> int:
    """Log, or finish logging, a sealed package. Reads only the package's manifest and test result
    (JSON): no retraining, no data rows, no unpickling."""
    record = record_from_package(args.models, args.package_id)
    try:
        tracked = log_job(record, TrackingStore(args.mlflow_dir))
    except AlreadyTrackedError as error:
        raise CliError(str(error)) from error
    except TrackingError as error:
        print(f"TRACKING PENDING: {error}", file=sys.stderr)
        return EXIT_TRACKING_PENDING
    if args.json:
        print(json.dumps(_tracking_payload(tracked, None), indent=2, sort_keys=True))
    else:
        _print_tracking(tracked, None, args)
    return 0


def _when(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%d %H:%M")


def _num(value: Any, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def cmd_jobs(store: FileDatasetStore, args: argparse.Namespace) -> int:
    """History of tracked jobs, newest first. Read-only."""
    jobs = list_jobs(
        TrackingStore(args.mlflow_dir),
        kind=args.kind,
        task=args.task,
        dataset=args.dataset,
        algorithm=args.algorithm,
        state=args.state,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps([j.to_dict() for j in jobs], indent=2, sort_keys=True))
        return 0
    if not jobs:
        print("no tracked jobs match")
        return 0
    print(
        f"{'job':<30} {'kind':<5} {'state':<10} {'started (UTC)':<16} {'dataset':<24} "
        f"{'seed':>5} {'folds':>5} {'selected / CV score':<32} {'ok/failed':>9} test"
    )
    for j in jobs:
        picked = (
            f"{j.selected_algorithm} {j.selection_metric}={_num(j.selected_cv_mean)}"
            if j.selected_algorithm
            else "- (cv only)"
        )
        print(
            f"{j.job_key:<30} {j.kind:<5} {j.state:<10} {_when(j.start_ms):<16} "
            f"{str(j.dataset_ref):<24} {str(j.seed):>5} {str(j.folds_used):>5} {picked:<32} "
            f"{j.algorithms_ok:>4}/{j.algorithms_failed:<4} "
            f"{'recorded' if j.has_test_result else '-'}"
        )
    return 0


def cmd_job(store: FileDatasetStore, args: argparse.Namespace) -> int:
    """One job's full record. The recorded test result (finalized jobs only) is shown here, with
    its limitation text, and nowhere else."""
    job = read_job(args.job_key, TrackingStore(args.mlflow_dir))
    if args.json:
        payload = {
            "job_key": job.job_key,
            "state": job.state,
            "kind": job.kind,
            "tags": job.tags,
            "params": job.params,
            "metrics": job.metrics,
            "dataset": job.dataset,
            "libraries": job.libraries,
            "training": job.training,
            "selection": job.selection,
            "algorithms": {
                k: {
                    "run_status": a.run_status,
                    "tags": a.tags,
                    "params": a.params,
                    "metrics": a.metrics,
                    "fold_metrics": a.fold_metrics,
                    "test_result": a.test_result,
                }
                for k, a in job.algorithms.items()
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    t = job.training
    print(f"job {job.job_key}  [{job.kind}, {job.state}]  started {_when(job.start_ms)} UTC")
    print(
        f"dataset {job.tags.get('platform.dataset_ref')}  "
        f"sha256 {job.tags.get('platform.dataset_sha256')}"
    )
    print(f"task {t['task']}, target {t['target']}, seed {job.tags.get('platform.seed')}, "
          f"folds {job.params.get('folds_used')} of {job.params.get('folds')} requested, "
          f"test fraction {job.params.get('test_fraction')}")  # fmt: skip
    print("libraries: " + ", ".join(f"{k} {v}" for k, v in sorted(job.libraries.items())))
    metric = job.tags.get("platform.selection_metric")
    print(
        f"\nCross-validation on the training part "
        f"({'selection metric ' + metric if metric else 'cv only'}):"
    )
    for r in t["results"]:
        if r["status"] != "ok":
            err = r.get("error") or {}
            print(f"  {r['algorithm']:<22} FAILED  {err.get('type')}: {err.get('message')}")
            continue
        parts = []
        for name, entry in sorted(r["summary"].items()):
            if entry.get("mean") is not None:
                parts.append(f"{name} {_num(entry['mean'])}+-{_num(entry.get('std'))}")
        mark = " *" if r["algorithm"] == job.tags.get("platform.selected_algorithm") else ""
        print(f"  {r['algorithm']:<22}{mark:<2} " + "  ".join(parts[:4]))
    if job.selection:
        print(
            f"\nselected: {job.selection['selected']['algorithm']} - "
            f"{job.selection.get('explanation', '')}"
        )
    for a in job.algorithms.values():
        if a.test_result is not None:
            ev = a.test_result.get("evaluation", {}).get("metrics", {})
            print("\nRecorded single test result (selected model, read once, not used for any "
                  "choice):")  # fmt: skip
            print(
                "  " + "  ".join(f"{k} {_num(v)}" for k, v in sorted(ev.items()) if v is not None)
            )
            if a.test_result.get("limitation"):
                print(f"  limitation: {a.test_result['limitation']}")
    return 0


def cmd_compare(store: FileDatasetStore, args: argparse.Namespace) -> int:
    """Cross-validation comparison of two or more jobs. Read-only; no test result is used."""
    tracking = TrackingStore(args.mlflow_dir)
    try:
        result = compare_jobs([read_job(k, tracking) for k in args.job_keys], args.metric)
    except CompareError as error:
        raise CliError(str(error)) from error
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    print(f"{result['task']} / {result['target']}: {result['metric']} "
          f"({result['direction']} is better), {result['basis']}")  # fmt: skip
    if result["split_note"]:
        print(f"NOTE: {result['split_note']}")
    print(
        f"\n{'rank':>4}  {'job':<30} {'algorithm':<22} {'cv mean':>10} {'cv std':>9} "
        f"{'pooled':>9}  status"
    )
    for r in result["rows"]:
        flag = " (selected in job)" if r["selected_in_job"] else ""
        status = r["status"] if r["error"] is None else f"{r['status']} ({r['error']})"
        print(
            f"{str(r['rank'] or '-'):>4}  {r['job_key']:<30} {r['algorithm']:<22} "
            f"{_num(r['cv_mean']):>10} {_num(r['cv_std']):>9} {_num(r['oof']):>9}  "
            f"{status}{flag}"
        )
    for name, params in result["parameter_differences"].items():
        print(f"parameters differ for {name}: {', '.join(params)}")
    if result["library_differences"]:
        print("library versions differ: " + ", ".join(result["library_differences"]))
    return 0


def _print_report(report: ReproductionReport) -> None:
    print(f"\n{report.verdict}  ({report.job_key}, {report.kind})")
    for item in report.items:
        mark = {"pass": "PASS", "fail": "FAIL", "skipped": "SKIP"}[item.status]
        print(f"  {mark}  {item.name}: {item.detail}")
        for line in item.mismatches[:8]:
            print(f"        {line}")
        if len(item.mismatches) > 8:
            print(f"        ... and {len(item.mismatches) - 8} more")
    tol = report.tolerance
    print(f"\ntolerance: |difference| <= {tol['absolute']:g} + {tol['relative']:g} x |logged|; "
          f"largest difference {report.max_deviation:.3g}; "
          f"{'bit-identical' if report.bit_identical else 'not bit-identical'}")  # fmt: skip
    if report.environment_same:
        print("environment: library versions equal to those logged")
    else:
        print("environment differs from the logged one:")
        for line in report.environment_differences:
            print(f"  {line}")
    print(f"scope: {report.scope}")


def cmd_reproduce(store: FileDatasetStore, args: argparse.Namespace) -> int:
    """Re-run a tracked job's cross-validation from its record and compare. Writes nothing and
    never reads the held-out test part. Exit 0 reproduced, 1 not reproduced, 2 cannot run."""
    job = read_job(args.job_key, TrackingStore(args.mlflow_dir))
    ref = job.dataset.get("ref")
    if not ref:
        raise CliError("the job record names no dataset version")
    version, frame = read_version_frame(store, str(ref))  # verifies the stored file's SHA-256
    package_test = package_sha = None
    checked = False
    package_id = job.tags.get("platform.package_id")
    if job.kind == "final" and package_id and (args.models / package_id).is_dir():
        package_test = sealed_test_result(args.models, package_id)
        package_sha = sealed_manifest_selection_sha(args.models, package_id)
        checked = True
    report = reproduce_job(
        job,
        frame,
        version.sha256,
        library_versions(),
        package_test_result=package_test,
        package_selection_sha256=package_sha,
        package_checked=checked,
        rel_tolerance=args.tolerance,
        progress=None if args.json else _print_progress,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_report(report)
    return 0 if report.ok else 1


def cmd_package_verify(store: FileDatasetStore, args: argparse.Namespace) -> int:
    ids = [args.package_id] if args.package_id else list_packages(args.models)
    checks = [verify_package(args.models, i) for i in ids]
    leftovers = [] if args.package_id else leftover_staging(args.models)
    if args.json:
        payload = {
            "packages": [
                {"id": c.package_id, "state": c.state, "ok": c.ok, "problems": list(c.problems)}
                for c in checks
            ],
            "leftover_staging": leftovers,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for c in checks:
            print(f"{'PASS' if c.ok else 'FAIL'}  {c.package_id}  [{c.state}]")
            for problem in c.problems:
                print(f"    {problem}")
        for name in leftovers:
            print(f"NOTE  {name}: an interrupted creation (never usable)")
        if not checks:
            print("no packages")
    return 0 if all(c.ok for c in checks) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dataset store tool")
    parser.add_argument("--store", type=Path, default=Path("data/store"), help="store folder")
    parser.add_argument(
        "--models", type=Path, default=Path("data/models"), help="model package folder"
    )
    parser.add_argument(
        "--mlflow-dir",
        type=Path,
        default=Path("data/mlflow"),
        help="MLflow tracking store: SQLite database + artifacts (default data/mlflow)",
    )
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

    p_train = sub.add_parser("train", help="cross-validate algorithms on the training part")
    p_train.add_argument("ref", help="e.g. adult@v1")
    add_task_options(p_train, target_required=True)
    p_train.add_argument("--algorithms", help="comma-separated keys (default: all for the task)")
    p_train.add_argument("--folds", type=int, default=5, help="cross-validation folds (default 5)")
    p_train.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    p_train.add_argument(
        "--test-fraction", type=float, default=0.2, help="held-out share (default 0.2)"
    )
    p_train.add_argument(
        "--track", action="store_true", help="also log the cross-validation results to MLflow"
    )

    p_final = sub.add_parser(
        "finalize", help="select, test once on the held-out part, and package the best model"
    )
    p_final.add_argument("ref", help="e.g. adult@v1")
    add_task_options(p_final, target_required=True)
    p_final.add_argument("--algorithms", help="comma-separated keys (default: all for the task)")
    p_final.add_argument("--folds", type=int, default=5, help="cross-validation folds (default 5)")
    p_final.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    p_final.add_argument(
        "--test-fraction", type=float, default=0.2, help="held-out share (default 0.2)"
    )
    p_final.add_argument("--selection-metric", help="metric to rank by (default: f1_macro or rmse)")
    p_final.add_argument(
        "--confirm", action="store_true", help="confirm that the test part is read once"
    )
    p_final.add_argument(
        "--track", action="store_true", help="after sealing, log the job to MLflow"
    )
    p_pv = sub.add_parser("package-verify", help="re-check model packages")
    p_pv.add_argument("package_id", nargs="?", help="one package (default: all)")
    p_pv.add_argument("--json", action="store_true", help="machine-readable output")

    p_tp = sub.add_parser(
        "track-package", help="log (or finish logging) a sealed package to MLflow"
    )
    p_tp.add_argument("package_id")
    p_tp.add_argument("--json", action="store_true", help="machine-readable output")

    p_jobs = sub.add_parser("jobs", help="list tracked jobs (newest first)")
    p_jobs.add_argument("--kind", choices=("train", "final"))
    p_jobs.add_argument("--task", choices=TASK_TYPES)
    p_jobs.add_argument("--dataset", help="dataset reference (name@vN) or start of its SHA-256")
    p_jobs.add_argument("--algorithm", help="selected algorithm")
    p_jobs.add_argument("--state", choices=("complete", "incomplete"))
    p_jobs.add_argument("--limit", type=int)
    p_jobs.add_argument("--json", action="store_true", help="machine-readable output")
    p_job = sub.add_parser("job", help="show one tracked job")
    p_job.add_argument("job_key")
    p_job.add_argument("--json", action="store_true", help="machine-readable output")
    p_cmp = sub.add_parser("compare", help="compare tracked jobs by cross-validation score")
    p_cmp.add_argument("job_keys", nargs="+", metavar="JOB_KEY")
    p_cmp.add_argument("--metric", help="metric to rank by (default: the jobs' selection metric)")
    p_cmp.add_argument("--json", action="store_true", help="machine-readable output")
    p_rep = sub.add_parser("reproduce", help="re-run a tracked job's cross-validation and compare")
    p_rep.add_argument("job_key")
    p_rep.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_REL_TOLERANCE,
        help="relative tolerance; a value passes if |diff| <= 1e-9 + tolerance x |logged| "
        f"(default {DEFAULT_REL_TOLERANCE:g})",
    )
    p_rep.add_argument("--json", action="store_true", help="machine-readable output")

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
        "train": cmd_train,
        "finalize": cmd_finalize,
        "package-verify": cmd_package_verify,
        "track-package": cmd_track_package,
        "jobs": cmd_jobs,
        "job": cmd_job,
        "compare": cmd_compare,
        "reproduce": cmd_reproduce,
    }
    try:
        return commands[args.command](store, args)
    except (
        IngestError,
        DatasetStoreError,
        SchemaError,
        CliError,
        TrainingError,
        AlgorithmError,
        PipelineError,
        SelectionError,
        FinalizeError,
        PackageError,
        TrackingError,
        ReproductionError,
        OSError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
