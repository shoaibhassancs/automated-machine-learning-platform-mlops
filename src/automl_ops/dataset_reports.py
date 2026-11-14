"""The JSON payloads of the dataset reports, shared by the command line and the HTTP API (P4-3).

A thin response-shaping layer. The business logic stays in `profile`, `validate`, `task` and
`explore`; this module only assembles what each entry point returns (`dataset_ref` and `sha256`
next to the report, the task configuration rules) so the CLI's `--json` output and the API cannot
drift apart. Everything here is pure: nothing is read from or written to a store.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from automl_ops.dataset_store import DatasetVersion
from automl_ops.explore import explore_frame
from automl_ops.profile import DatasetProfile, VersionProfile
from automl_ops.task import (
    TASK_TYPES,
    TaskConfig,
    check_task_config,
    class_counts,
    suggest_task,
)
from automl_ops.validate import validate


class ReportError(ValueError):
    """The request for a report is unusable (no target, cannot suggest a task, unknown task)."""


def parse_excluded(raw: str | None) -> tuple[str, ...]:
    """`--exclude A,B` as a tuple of names (blank entries dropped)."""
    if not raw:
        return ()
    return tuple(name.strip() for name in raw.split(",") if name.strip())


def build_task_config(
    profile: DatasetProfile,
    target: str,
    task: str | None,
    excluded: tuple[str, ...],
    hint: str = "pass a task",
) -> TaskConfig:
    """The task configuration; without `task`, the suggested task is used. `hint` is the entry
    point's own wording of how to supply the task (the CLI says `pass --task`)."""
    if task is None:
        suggestion = suggest_task(profile, target)
        if suggestion is None:
            raise ReportError(
                f"cannot suggest a task for target {target!r} (it does not exist or has no "
                f"values); {hint}"
            )
        task = suggestion.task
    if task not in TASK_TYPES:
        raise ReportError(f"unknown task {task!r}; use one of {', '.join(TASK_TYPES)}")
    return TaskConfig(target=target, task=task, excluded=excluded)


def class_counts_for(frame: pd.DataFrame, config: TaskConfig) -> dict[str, int]:
    """Class counts for a classification config whose target exists; otherwise none."""
    if config.task == "classification" and config.target in frame.columns:
        return class_counts(frame, config.target)
    return {}


def version_payload(version: DatasetVersion) -> dict[str, Any]:
    """One version's record (the `show` output) plus its reference."""
    return {**version.to_record(), "ref": version.ref}


def profile_payload(vp: VersionProfile) -> dict[str, Any]:
    return vp.to_dict()


def validation_payload(
    version: DatasetVersion, profile: DatasetProfile, target: str | None
) -> dict[str, Any]:
    report = validate(profile, target=target)
    return {"dataset_ref": version.ref, "sha256": version.sha256, **report.to_dict()}


def task_payload(
    version: DatasetVersion, frame: pd.DataFrame, profile: DatasetProfile, config: TaskConfig
) -> dict[str, Any]:
    report = check_task_config(profile, class_counts_for(frame, config), config)
    return {
        "dataset_ref": version.ref,
        "sha256": version.sha256,
        **report.to_dict(),
        "feature_columns": list(config.feature_columns(profile)),
    }


def explore_payload(
    version: DatasetVersion,
    frame: pd.DataFrame,
    profile: DatasetProfile,
    config: TaskConfig | None,
) -> dict[str, Any]:
    return {
        "dataset_ref": version.ref,
        "sha256": version.sha256,
        **explore_frame(frame, profile, config),
    }
