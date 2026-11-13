"""Task configuration (BUILD_PLAN Step 5): target, task type and excluded columns.

A `TaskConfig` says what to predict (`target`), whether it is a `classification` or `regression`
task, and which columns are left out of the model inputs (`excluded`). `suggest_task` proposes a
task type from the target's profile; the suggestion is **advisory**, and the user can choose either
type.

`check_task_config` reports whether a chosen configuration is usable:

* **error**: the configuration cannot work (no such target, missing or constant target, no input
  columns left, a regression target that is not numeric, a classification class with fewer than
  2 rows, which the planned seeded stratified split of Step 9 cannot place in both parts);
* **warning**: it can work but deserves a look (many classes, a very small class);
* **info**: a fact worth knowing (decimal-looking class labels, a choice that differs from the
  suggestion, a regression target with few distinct values).

A high class count is **never** an automatic rejection: numeric labels can be classes, and many
classes are legitimate. Target *quality* (exists, no missing values, not constant) reuses the
Step 4A code in `validate.py`; this module adds the task-specific checks only.

Settings are provisional defaults (see docs/task-config.md). Nothing here converts or stores data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from automl_ops.profile import DatasetProfile
from automl_ops.validate import SEVERITY_RANK, Finding, target_findings

TaskType = Literal["classification", "regression"]
TASK_TYPES: tuple[TaskType, ...] = ("classification", "regression")


@dataclass(frozen=True)
class TaskSettings:
    """Provisional thresholds (docs/task-config.md)."""

    # Suggestion: a whole-number target with MORE distinct values than this suggests regression.
    whole_number_regression_min_distinct: int = 20
    min_class_size_error: int = 2  # needed by the planned stratified split (Step 9)
    min_class_size_warning: int = 5  # placeholder until Step 9 fixes the number of folds
    max_classes_warning: int = 50  # more classes than this: warning, never an error

    def to_dict(self) -> dict[str, Any]:
        return {
            "whole_number_regression_min_distinct": self.whole_number_regression_min_distinct,
            "min_class_size_error": self.min_class_size_error,
            "min_class_size_warning": self.min_class_size_warning,
            "max_classes_warning": self.max_classes_warning,
        }


DEFAULT_TASK_SETTINGS = TaskSettings()


@dataclass(frozen=True)
class TaskConfig:
    target: str
    task: TaskType
    excluded: tuple[str, ...] = ()

    def feature_columns(self, profile: DatasetProfile) -> tuple[str, ...]:
        """Model-input columns: every column except the target and the excluded ones."""
        left_out = {self.target, *self.excluded}
        return tuple(c.name for c in profile.columns if c.name not in left_out)

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "task": self.task, "excluded": list(self.excluded)}


@dataclass(frozen=True)
class TaskSuggestion:
    task: TaskType
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"task": self.task, "reason": self.reason}


@dataclass(frozen=True)
class TaskReport:
    config: TaskConfig
    findings: tuple[Finding, ...]
    suggestion: TaskSuggestion | None
    settings: TaskSettings

    @property
    def errors(self) -> int:
        return sum(1 for f in self.findings if f.severity == "error")

    @property
    def warnings(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warning")

    @property
    def infos(self) -> int:
        return sum(1 for f in self.findings if f.severity == "info")

    @property
    def ok(self) -> bool:
        """True when there are no errors (warnings and info do not block)."""
        return self.errors == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "config": self.config.to_dict(),
            "suggestion": self.suggestion.to_dict() if self.suggestion else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "infos": self.infos,
            "settings": self.settings.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
        }


def class_counts(frame: pd.DataFrame, target: str) -> dict[str, int]:
    """Rows per distinct non-missing target value (the exact text values as read)."""
    counted = frame[target].value_counts(dropna=True, sort=False)
    return {str(value): int(count) for value, count in counted.items()}


def suggest_task(
    profile: DatasetProfile,
    target: str,
    settings: TaskSettings = DEFAULT_TASK_SETTINGS,
) -> TaskSuggestion | None:
    """Suggest a task type from the target's profile; None if there is nothing to go on.

    Text-like target: classification. Decimal target: regression. Whole-number target: regression
    when it has more than `whole_number_regression_min_distinct` distinct values, otherwise
    classification. Advisory only.
    """
    column = profile.column(target)
    if column is None or column.kind == "empty":
        return None
    if column.kind == "float":
        return TaskSuggestion("regression", "the target holds decimal numbers")
    if column.kind == "integer":
        limit = settings.whole_number_regression_min_distinct
        if column.unique > limit:
            return TaskSuggestion(
                "regression",
                f"the target holds whole numbers with {column.unique} distinct values "
                f"(more than {limit})",
            )
        return TaskSuggestion(
            "classification",
            f"the target holds whole numbers with only {column.unique} distinct values "
            f"(at most {limit}), which look like class labels",
        )
    return TaskSuggestion(
        "classification", f"the target holds text values ({column.kind}), not numbers"
    )


def _task_findings(
    profile: DatasetProfile,
    config: TaskConfig,
    counts: Mapping[str, int],
    settings: TaskSettings,
) -> list[Finding]:
    found: list[Finding] = []
    target = config.target
    column = profile.column(target)
    names = {c.name for c in profile.columns}

    for name in config.excluded:
        if name not in names:
            found.append(
                Finding(
                    "unknown_excluded_column",
                    "error",
                    name,
                    f"the excluded column {name!r} does not exist",
                    {"columns": [c.name for c in profile.columns]},
                )
            )
    if target in config.excluded:
        found.append(
            Finding(
                "target_excluded",
                "error",
                target,
                f"the target {target!r} is also listed as excluded",
                {},
            )
        )
    if column is not None and not config.feature_columns(profile):
        found.append(
            Finding(
                "no_feature_columns",
                "error",
                None,
                "no input columns are left once the target and the excluded columns are removed",
                {"excluded": list(config.excluded)},
            )
        )
    if column is None:
        return found  # target_findings (unknown_target) already covers this

    if config.task == "regression":
        if column.kind not in ("integer", "float") and column.count > 0:
            found.append(
                Finding(
                    "target_not_numeric",
                    "error",
                    target,
                    f"regression needs a numeric target, but this column is {column.kind!r} "
                    "(not every value is a clean number)",
                    {"kind": column.kind, "numeric_share": column.numeric_share},
                )
            )
        elif column.unique <= settings.whole_number_regression_min_distinct and column.count > 0:
            found.append(
                Finding(
                    "few_distinct_values_for_regression",
                    "info",
                    target,
                    f"the target has only {column.unique} distinct values; if they are labels "
                    "rather than quantities, classification may fit better",
                    {"unique": column.unique},
                )
            )
        return found

    # classification
    if column.kind == "float" and column.count > 0:
        found.append(
            Finding(
                "decimal_class_labels",
                "info",
                target,
                f"the target holds decimal numbers; its {column.unique} distinct values will "
                "each be treated as a class (the exact text is kept, so 1 and 1.0 are different)",
                {"unique": column.unique},
            )
        )
    if counts:
        smallest_value, smallest = min(counts.items(), key=lambda item: (item[1], item[0]))
        too_small = sorted(v for v, c in counts.items() if c < settings.min_class_size_error)
        if too_small:
            found.append(
                Finding(
                    "class_too_small",
                    "error",
                    target,
                    f"{len(too_small)} class(es) have fewer than {settings.min_class_size_error} "
                    "rows (for example "
                    + ", ".join(repr(v) for v in too_small[:3])
                    + "); a stratified split needs at least "
                    f"{settings.min_class_size_error} rows per class",
                    {
                        "classes": too_small[:10],
                        "count": len(too_small),
                        "minimum": settings.min_class_size_error,
                    },
                )
            )
        elif smallest < settings.min_class_size_warning:
            found.append(
                Finding(
                    "small_class",
                    "warning",
                    target,
                    f"the smallest class {smallest_value!r} has only {smallest} rows (fewer than "
                    f"{settings.min_class_size_warning}); cross-validation folds will be thin "
                    "(provisional limit)",
                    {"smallest_class": smallest_value, "rows": smallest},
                )
            )
        if len(counts) > settings.max_classes_warning:
            found.append(
                Finding(
                    "many_classes",
                    "warning",
                    target,
                    f"{len(counts)} classes (more than {settings.max_classes_warning}); this is "
                    "allowed, but every class needs enough rows for the split and "
                    "cross-validation, per-class scores get noisy, and the confusion matrix "
                    "becomes very large. Check that this is intended",
                    {"classes": len(counts), "limit": settings.max_classes_warning},
                )
            )
    return found


def check_task_config(
    profile: DatasetProfile,
    counts: Mapping[str, int],
    config: TaskConfig,
    settings: TaskSettings = DEFAULT_TASK_SETTINGS,
) -> TaskReport:
    """Check a task configuration. `counts` are the class counts from `class_counts`.

    Pure: the same inputs always give the same findings, and nothing is changed.
    """
    findings = target_findings(profile, config.target)
    findings.extend(_task_findings(profile, config, counts, settings))

    suggestion = suggest_task(profile, config.target, settings)
    if suggestion is not None and suggestion.task != config.task:
        findings.append(
            Finding(
                "task_differs_from_suggestion",
                "info",
                config.target,
                f"you chose {config.task}; the suggestion was {suggestion.task} "
                f"({suggestion.reason}). The choice is yours",
                {"chosen": config.task, "suggested": suggestion.task},
            )
        )

    position = {c.name: c.position for c in profile.columns}
    findings.sort(
        key=lambda f: (
            SEVERITY_RANK[f.severity],
            position.get(f.column, 0) if f.column is not None else -1,
            f.code,
        )
    )
    return TaskReport(
        config=config, findings=tuple(findings), suggestion=suggestion, settings=settings
    )
