"""Rule-based validation of a dataset profile (BUILD_PLAN Step 4, chunk 4A).

`validate` is a pure function of a `DatasetProfile`: it returns structured findings and changes
nothing. Severities:

* **error**: training on this data (or with this target) would be invalid regardless of which
  columns are excluded;
* **warning**: review it, or exclude the column (Step 5 decides what an exclusion resolves);
* **info**: a fact worth knowing.

Task-suitability checks (task type, class sizes, regression constraints) are **not** here: they
need the prediction task, which is chosen in Step 5. With a target, this module checks only target
*quality*: that it exists, has no missing values, and is not constant.

Thresholds are provisional defaults (see docs/profiling.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from automl_ops.profile import ColumnProfile, DatasetProfile

Severity = Literal["error", "warning", "info"]
SEVERITY_RANK: dict[str, int] = {"error": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class Thresholds:
    min_rows_error: int = 30  # fewer rows than this: error
    min_rows_warning: int = 100  # fewer rows than this: warning
    high_missing_pct: float = 40.0  # more missing than this (but not all): warning
    id_like_min_rows: int = 50  # ID-like checks are skipped on tiny datasets
    id_like_unique_ratio: float = 0.9  # text/code: distinct values / non-missing at or above this

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_rows_error": self.min_rows_error,
            "min_rows_warning": self.min_rows_warning,
            "high_missing_pct": self.high_missing_pct,
            "id_like_min_rows": self.id_like_min_rows,
            "id_like_unique_ratio": self.id_like_unique_ratio,
        }


DEFAULT_THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    column: str | None  # None for dataset-level findings
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "column": self.column,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True)
class ValidationReport:
    findings: tuple[Finding, ...]
    target: str | None
    thresholds: Thresholds

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
            "target": self.target,
            "errors": self.errors,
            "warnings": self.warnings,
            "infos": self.infos,
            "thresholds": self.thresholds.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
        }


def _pairs(items: tuple[tuple[str, int], ...]) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in items]


def _dataset_findings(profile: DatasetProfile, t: Thresholds) -> list[Finding]:
    found: list[Finding] = []
    if profile.rows < t.min_rows_error:
        found.append(
            Finding(
                "too_few_rows",
                "error",
                None,
                f"only {profile.rows} rows; at least {t.min_rows_error} are needed to train and "
                "evaluate a model",
                {"rows": profile.rows, "minimum": t.min_rows_error},
            )
        )
    elif profile.rows < t.min_rows_warning:
        found.append(
            Finding(
                "few_rows",
                "warning",
                None,
                f"only {profile.rows} rows; results from so little data will be unreliable "
                f"(fewer than {t.min_rows_warning})",
                {"rows": profile.rows, "threshold": t.min_rows_warning},
            )
        )
    if profile.duplicate_rows > 0 and profile.rows > 0:
        share = 100.0 * profile.duplicate_rows / profile.rows
        found.append(
            Finding(
                "duplicate_rows",
                "info",
                None,
                f"{profile.duplicate_rows} extra copies of exact duplicate rows "
                f"({share:.2f}% of rows); duplicates are kept",
                {
                    "extra_copies": profile.duplicate_rows,
                    "rows_in_duplicate_groups": profile.rows_in_duplicate_groups,
                    "percent_of_rows": round(share, 4),
                },
            )
        )
    return found


def _column_findings(
    column: ColumnProfile, profile: DatasetProfile, t: Thresholds
) -> list[Finding]:
    name = column.name
    found: list[Finding] = []

    def add(code: str, severity: Severity, message: str, details: dict[str, Any]) -> None:
        found.append(Finding(code, severity, name, message, details))

    if column.kind == "empty":
        add("all_missing", "warning", "every value is missing; exclude this column", {})
        return found

    if column.missing_pct > t.high_missing_pct and column.missing > 0:
        add(
            "high_missing",
            "warning",
            f"{column.missing_pct:.1f}% of values are missing (more than {t.high_missing_pct:g}%)",
            {"missing": column.missing, "missing_pct": column.missing_pct},
        )
    if column.unique == 1:
        only = column.top_values[0][0]
        add(
            "constant_column",
            "warning",
            f"every non-missing value is {only!r}; the column carries no information",
            {"value": only},
        )
    if column.kind == "mixed":
        add(
            "mixed_types",
            "warning",
            f"{100 * column.numeric_share:.1f}% of values look like numbers but others do not "
            "(for example "
            + ", ".join(repr(v) for v, _ in column.non_numeric_examples[:3])
            + "); the column cannot be treated as a plain number",
            {
                "numeric_share": column.numeric_share,
                "non_numeric_examples": _pairs(column.non_numeric_examples),
            },
        )
    if column.na_like_tokens:
        total = column.na_like_count
        add(
            "na_like_values",
            "warning",
            f"{total} cells contain text that often means 'missing' "
            + ", ".join(repr(v) for v, _ in column.na_like_tokens[:3])
            + "; only empty cells count as missing, so these are kept as ordinary values",
            {"cells": total, "tokens": _pairs(column.na_like_tokens)},
        )
    if column.leading_zero_count:
        add(
            "leading_zero_values",
            "warning",
            f"{column.leading_zero_count} values start with a zero, such as "
            + ", ".join(repr(v) for v, _ in column.leading_zero_examples[:3])
            + "; they may be codes or identifiers, not numbers. The original text is kept and "
            "the column is not treated as numeric",
            {
                "cells": column.leading_zero_count,
                "examples": _pairs(column.leading_zero_examples),
                "kind": column.kind,
            },
        )
    if column.unique > 1 and column.count >= t.id_like_min_rows:
        ratio = column.unique / column.count
        if column.kind == "integer" and column.unique == column.count:
            add(
                "id_like_integer",
                "warning",
                "every value is different (a whole-number column like a row number or an ID); "
                "it is unlikely to help a model",
                {"unique": column.unique, "count": column.count},
            )
        elif column.kind in ("text", "code", "categorical") and ratio >= t.id_like_unique_ratio:
            add(
                "id_like_text",
                "warning",
                f"{100 * ratio:.1f}% of values are distinct; this looks like an identifier or "
                "free text",
                {"unique": column.unique, "count": column.count, "unique_ratio": round(ratio, 6)},
            )
        elif column.kind == "text":
            add(
                "many_categories",
                "warning",
                f"{column.unique} distinct text values "
                f"(more than {profile.settings.max_categories}); "
                "turning it into model inputs would create very many columns",
                {"unique": column.unique, "limit": profile.settings.max_categories},
            )
    return found


def target_findings(profile: DatasetProfile, target: str) -> list[Finding]:
    column = profile.column(target)
    if column is None:
        return [
            Finding(
                "unknown_target",
                "error",
                target,
                f"the target column {target!r} does not exist",
                {"columns": [c.name for c in profile.columns]},
            )
        ]
    found: list[Finding] = []
    if column.missing > 0:
        found.append(
            Finding(
                "target_missing",
                "error",
                target,
                f"the target has {column.missing} missing values ({column.missing_pct:.2f}%); "
                "every row needs a target value",
                {"missing": column.missing, "missing_pct": column.missing_pct},
            )
        )
    if column.unique <= 1 and column.count > 0:
        found.append(
            Finding(
                "target_constant",
                "error",
                target,
                "the target has a single value, so there is nothing to predict",
                {"value": column.top_values[0][0]},
            )
        )
    return found


def validate(
    profile: DatasetProfile,
    target: str | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> ValidationReport:
    """Check a profile against the rules. Pure: same profile in, same findings out."""
    findings = _dataset_findings(profile, thresholds)
    for column in profile.columns:
        findings.extend(_column_findings(column, profile, thresholds))
    if target is not None:
        findings.extend(target_findings(profile, target))

    position = {c.name: c.position for c in profile.columns}
    findings.sort(
        key=lambda f: (
            SEVERITY_RANK[f.severity],
            position.get(f.column, 0) if f.column is not None else -1,
            f.code,
        )
    )
    return ValidationReport(findings=tuple(findings), target=target, thresholds=thresholds)
