"""CSV intake (BUILD_PLAN Step 3): check one uploaded CSV and convert it to a canonical form.

Standard library only, and no storage: this module knows nothing about where files are kept.

Accepted files are converted to **UTF-8, comma-separated, "\\n" line endings, minimal quoting**.
Every cell is kept as the exact text that was uploaded (no type guessing, no missing-value
handling; that is Step 4). The SHA-256 of the canonical bytes is the identity of a dataset version.

Limits and format rules are provisional defaults (PROJECT_GUIDE assumption A4).
"""

from __future__ import annotations

import codecs
import csv
import hashlib
import io
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CANDIDATE_DELIMITERS = (",", ";", "\t", "|")
SAMPLE_RECORDS = 50


@dataclass(frozen=True)
class IngestLimits:
    """Upload limits. Provisional defaults; do not change silently."""

    max_bytes: int = 50 * 1024 * 1024
    max_rows: int = 100_000
    max_columns: int = 100


DEFAULT_LIMITS = IngestLimits()


class IngestError(ValueError):
    """The file cannot be accepted. `code` is a stable short name; the message is for people."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class PreparedCsv:
    canonical: bytes
    sha256: str  # of the canonical bytes: the version identity
    original_sha256: str  # of the uploaded bytes: provenance only
    original_size_bytes: int
    size_bytes: int  # of the canonical bytes
    original_encoding: str  # "utf-8", "utf-8-sig" (UTF-8 with BOM) or "utf-16"
    original_delimiter: str
    rows: int  # data rows, not counting the header
    columns: tuple[str, ...]
    trailing_blank_lines_ignored: int

    @property
    def reencoded(self) -> bool:
        """True if the canonical bytes differ from the uploaded bytes."""
        return self.sha256 != self.original_sha256


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_limited(path: Path, limits: IngestLimits = DEFAULT_LIMITS) -> bytes:
    """Read a file, refusing anything over the byte limit before it is fully read."""
    size = path.stat().st_size
    if size > limits.max_bytes:
        raise IngestError("too_large", _too_large_message(size, limits))
    with path.open("rb") as handle:
        data = handle.read(limits.max_bytes + 1)
    if len(data) > limits.max_bytes:
        raise IngestError("too_large", _too_large_message(len(data), limits))
    return data


def _too_large_message(size: int, limits: IngestLimits) -> str:
    return f"file is {size:,} bytes; the limit is {limits.max_bytes:,} bytes"


def _decode(data: bytes) -> tuple[str, str]:
    """Decode UTF-8 (with or without BOM) or BOM-marked UTF-16. Never guess other encodings."""
    try:
        if data.startswith(codecs.BOM_UTF8):
            return data.decode("utf-8-sig"), "utf-8-sig"
        if data.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
            raise IngestError("bad_encoding", "UTF-32 is not supported; re-save as UTF-8")
        if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            return data.decode("utf-16"), "utf-16"
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError as error:
        raise IngestError(
            "bad_encoding",
            "file is not valid UTF-8 or BOM-marked UTF-16 "
            f"(first problem at byte {error.start}). Re-save it as UTF-8 and upload again; "
            "the platform does not guess other encodings.",
        ) from error


def _sample_counts(text: str, delimiter: str) -> list[int]:
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    counts: list[int] = []
    for row in reader:
        if row:
            counts.append(len(row))
        if len(counts) >= SAMPLE_RECORDS:
            break
    return counts


def _detect_delimiter(text: str) -> str:
    counts = {d: _sample_counts(text, d) for d in CANDIDATE_DELIMITERS}
    consistent = [d for d, c in counts.items() if c and len(set(c)) == 1 and c[0] >= 2]
    if len(consistent) == 1:
        return consistent[0]
    if len(consistent) > 1:
        names = ", ".join(repr(d) for d in consistent)
        raise IngestError(
            "bad_delimiter",
            f"cannot tell the delimiter ({names} all give consistent columns). "
            "Re-save the file with commas.",
        )
    scored: list[tuple[float, str]] = []
    for d, c in counts.items():
        if not c:
            continue
        modal, freq = Counter(c).most_common(1)[0]
        if modal >= 2:
            scored.append((freq / len(c), d))
    if not scored:
        raise IngestError(
            "bad_delimiter",
            "could not find a delimiter (comma, semicolon, tab or pipe) that gives 2 or more "
            "columns. A dataset needs at least a feature column and a target column.",
        )
    scored.sort(reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        raise IngestError(
            "bad_delimiter", "cannot tell the delimiter; re-save the file with commas"
        )
    return scored[0][1]


def _check_header(row: list[str], limits: IngestLimits) -> tuple[str, ...]:
    if len(row) < 2:
        raise IngestError("too_few_columns", "the file needs at least 2 columns")
    if len(row) > limits.max_columns:
        raise IngestError(
            "too_many_columns", f"{len(row)} columns; the limit is {limits.max_columns}"
        )
    seen: dict[str, int] = {}
    for position, name in enumerate(row, start=1):
        key = name.strip().casefold()
        if not key:
            raise IngestError("bad_header", f"column {position} has an empty name")
        if key in seen:
            raise IngestError(
                "bad_header",
                f"columns {seen[key]} and {position} have the same name {name.strip()!r} "
                "(names are compared ignoring case and surrounding spaces)",
            )
        seen[key] = position
    return tuple(row)


def prepare_csv(data: bytes, limits: IngestLimits = DEFAULT_LIMITS) -> PreparedCsv:
    """Check an uploaded CSV and return its canonical form, or raise `IngestError`."""
    if len(data) > limits.max_bytes:
        raise IngestError("too_large", _too_large_message(len(data), limits))
    if not data:
        raise IngestError("empty_file", "the file is empty")
    text, encoding = _decode(data)
    if "\x00" in text:
        raise IngestError(
            "nul_byte",
            "the file contains NUL characters (it may be binary or an unsupported encoding)",
        )
    if not text.strip():
        raise IngestError("empty_file", "the file contains only blank lines")
    try:
        delimiter = _detect_delimiter(text)
    except csv.Error as error:
        raise IngestError(
            "csv_error", f"CSV syntax problem while reading the file: {error}"
        ) from error

    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
    out = io.StringIO(newline="")
    writer = csv.writer(
        out, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
    )
    columns: tuple[str, ...] | None = None
    rows = 0
    first_blank_line: int | None = None
    blank_records = 0
    try:
        for row in reader:
            if not row:
                if first_blank_line is None:
                    first_blank_line = reader.line_num
                blank_records += 1
                continue
            if first_blank_line is not None:
                raise IngestError(
                    "blank_line",
                    f"blank line at line {first_blank_line} is followed by more data "
                    "(only blank lines at the very end are allowed)",
                )
            if columns is None:
                columns = _check_header(row, limits)
                writer.writerow(row)
                continue
            if len(row) != len(columns):
                raise IngestError(
                    "ragged_row",
                    f"the record ending on line {reader.line_num} has {len(row)} fields; "
                    f"the header has {len(columns)}",
                )
            rows += 1
            if rows > limits.max_rows:
                raise IngestError("too_many_rows", f"more than {limits.max_rows:,} data rows")
            writer.writerow(row)
    except csv.Error as error:
        raise IngestError(
            "csv_error", f"CSV syntax problem near line {reader.line_num}: {error}"
        ) from error

    if columns is None:
        raise IngestError("empty_file", "the file has no header row")
    if rows == 0:
        raise IngestError("no_data_rows", "the file has a header but no data rows")

    canonical = out.getvalue().encode("utf-8")
    return PreparedCsv(
        canonical=canonical,
        sha256=sha256_bytes(canonical),
        original_sha256=sha256_bytes(data),
        original_size_bytes=len(data),
        size_bytes=len(canonical),
        original_encoding=encoding,
        original_delimiter=delimiter,
        rows=rows,
        columns=columns,
        trailing_blank_lines_ignored=blank_records,
    )
