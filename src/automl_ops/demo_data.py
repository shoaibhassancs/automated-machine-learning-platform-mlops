"""Demo data preparation (BUILD_PLAN Step 2).

Downloads two public UCI datasets, cleans them lightly, and writes three separate files per dataset:

* a **training pool**: the only file meant to be uploaded for training;
* a **monitoring stream**: rows never used for training, cross-validation, selection or the final
  test; used only by the traffic simulator and for feedback labels;
* a **shifted monitoring copy**: derived from the monitoring stream rows only, with a documented
  synthetic shift.

All of this is demo data. Nothing here makes a fairness claim, and the shift is synthetic: it says
nothing about real-world data drift.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd

ALLOWED_HOST = "archive.ics.uci.edu"
USER_AGENT = "automl-ops-demo-data/0.1"
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
MAX_UNZIPPED_BYTES = 50 * 1024 * 1024

# Facts below were read from the official UCI dataset pages on this date (see docs/data.md).
SOURCES_CHECKED_ON = "2026-11-13"

DEMO_NOTICE = (
    "Demo data for a portfolio project. Not for real decisions. No fairness or real-world drift "
    "claims are made. The shifted monitoring copy is synthetic."
)


@dataclass(frozen=True)
class NumericShift:
    """Add `std_multiple` x (standard deviation of the stream column) to a numeric column."""

    column: str
    std_multiple: float


@dataclass(frozen=True)
class CategoryShift:
    """Re-assign a seeded fraction of the non-missing rows to `category`."""

    column: str
    category: str
    fraction: float


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    task: str
    target: str
    page_url: str
    zip_url: str
    zip_name: str
    licence_name: str
    licence_url: str
    licence_statement: str
    citation: str
    doi: str
    documented_rows: int
    numeric_shifts: tuple[NumericShift, ...]
    category_shifts: tuple[CategoryShift, ...]
    stratify: bool
    notes: tuple[str, ...] = field(default_factory=tuple)


LICENCE_STATEMENT = (
    "This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) "
    "license. This allows for the sharing and adaptation of the datasets for any purpose, "
    "provided that the appropriate credit is given."
)
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/legalcode"

ADULT = DatasetSpec(
    key="adult",
    name="Adult (Census Income)",
    task="classification",
    target="income",
    page_url="https://archive.ics.uci.edu/dataset/2/adult",
    zip_url="https://archive.ics.uci.edu/static/public/2/adult.zip",
    zip_name="adult.zip",
    licence_name="CC BY 4.0",
    licence_url=LICENCE_URL,
    licence_statement=LICENCE_STATEMENT,
    citation=(
        "Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C5XW20."
    ),
    doi="10.24432/C5XW20",
    documented_rows=48842,
    numeric_shifts=(
        NumericShift("age", 1.0),
        NumericShift("hours-per-week", 1.0),
    ),
    category_shifts=(CategoryShift("occupation", "Exec-managerial", 0.25),),
    stratify=True,
    notes=(
        "Contains sex and race columns and 1994 census income labels. Demo only; no fairness "
        "claims.",
        "'?' missing-value markers are converted to empty values; no other value is changed.",
    ),
)

ABALONE = DatasetSpec(
    key="abalone",
    name="Abalone",
    task="regression",
    target="Rings",
    page_url="https://archive.ics.uci.edu/dataset/1/abalone",
    zip_url="https://archive.ics.uci.edu/static/public/1/abalone.zip",
    zip_name="abalone.zip",
    licence_name="CC BY 4.0",
    licence_url=LICENCE_URL,
    licence_statement=LICENCE_STATEMENT,
    citation=(
        "Nash, W., Sellers, T., Talbot, S., Cawthorn, A., & Ford, W. (1994). Abalone [Dataset]. "
        "UCI Machine Learning Repository. https://doi.org/10.24432/C55C7W."
    ),
    doi="10.24432/C55C7W",
    documented_rows=4177,
    numeric_shifts=(
        NumericShift("Whole_weight", 1.0),
        NumericShift("Shell_weight", 1.0),
    ),
    category_shifts=(CategoryShift("Sex", "I", 0.30),),
    stratify=False,
    notes=(
        "UCI states that continuous values were scaled for use with an ANN (divided by 200), and "
        "that rows with missing values were removed from the original data.",
    ),
)

SPECS: dict[str, DatasetSpec] = {ADULT.key: ADULT, ABALONE.key: ABALONE}

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
ABALONE_COLUMNS = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole_weight",
    "Shucked_weight",
    "Viscera_weight",
    "Shell_weight",
    "Rings",
]


# --------------------------------------------------------------------------------------------
# Download (HTTPS, one host, no redirects, size cap)
# --------------------------------------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_download_url(url: str, allowed_host: str = ALLOWED_HOST) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"only https URLs are allowed: {url}")
    if parsed.hostname != allowed_host:
        raise ValueError(f"host {parsed.hostname!r} is not allowed (only {allowed_host!r})")


def download(url: str, dest: Path, max_bytes: int = MAX_DOWNLOAD_BYTES) -> str:
    """Download `url` to `dest` and return its SHA-256. Refuses redirects and other hosts."""
    check_download_url(url)
    opener = urllib.request.build_opener(_NoRedirect())
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    try:
        with opener.open(request, timeout=60) as response:  # noqa: S310
            if response.status != 200:
                raise RuntimeError(f"unexpected HTTP status {response.status} for {url}")
            data: bytes = response.read(max_bytes + 1)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code} for {url} (redirects are refused)") from error
    if len(data) > max_bytes:
        raise RuntimeError(f"download larger than {max_bytes} bytes was refused: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return sha256_bytes(data)


def read_zip_members(zip_path: Path, wanted: list[str]) -> dict[str, bytes]:
    """Read named members into memory. Nothing is extracted to disk; unsafe names are refused."""
    out: dict[str, bytes] = {}
    with zipfile.ZipFile(zip_path) as archive:
        infos = {info.filename: info for info in archive.infolist()}
        for name, info in infos.items():
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"unsafe name in zip: {name!r}")
            if info.file_size > MAX_UNZIPPED_BYTES:
                raise ValueError(f"zip member too large: {name!r}")
        for name in wanted:
            if name not in infos:
                raise ValueError(f"{name!r} not found in {zip_path.name}; has {sorted(infos)}")
            out[name] = archive.read(name)
    return out


# --------------------------------------------------------------------------------------------
# Parsing and light cleaning
# --------------------------------------------------------------------------------------------


def _clean_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace in text columns and turn '?' markers into real missing values."""
    out = frame.copy()
    for column in out.columns:
        if not pd.api.types.is_numeric_dtype(out[column]):
            stripped = out[column].astype("string").str.strip()
            out[column] = stripped.mask(stripped == "?")
    return out


def normalise_adult_label(value: str) -> str:
    """Adult's test file writes labels with a trailing period ('<=50K.'); train does not."""
    cleaned = value.strip().rstrip(".")
    if cleaned not in {"<=50K", ">50K"}:
        raise ValueError(f"unexpected Adult label: {value!r}")
    return cleaned


def parse_adult_file(raw: bytes, *, skip_header_line: bool) -> pd.DataFrame:
    """Parse one of Adult's headerless comma-separated files (adult.data or adult.test)."""
    text = raw.decode("utf-8")
    if skip_header_line:
        first_line, text = text.split("\n", 1)
        if not first_line.startswith("|"):
            raise ValueError(f"expected a '|' comment line first, got {first_line[:40]!r}")
    frame = pd.read_csv(
        io.StringIO(text),
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
        na_filter=False,
        skip_blank_lines=True,
    )
    return frame


def prepare_adult(train_raw: bytes, test_raw: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge Adult's two files after checking and normalising their label formats."""
    train = parse_adult_file(train_raw, skip_header_line=False)
    test = parse_adult_file(test_raw, skip_header_line=True)
    report: dict[str, Any] = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_label_values_before": sorted(map(str, train["income"].unique())),
        "test_label_values_before": sorted(map(str, test["income"].unique())),
    }
    for part in (train, test):
        part["income"] = part["income"].map(normalise_adult_label)
    merged = pd.concat([train, test], ignore_index=True)
    merged = _clean_strings(merged)
    for column in ("age", "fnlwgt", "education-num", "capital-gain", "capital-loss"):
        merged[column] = merged[column].astype("int64")
    merged["hours-per-week"] = merged["hours-per-week"].astype("int64")
    report["missing_by_column"] = {
        column: int(count) for column, count in merged.isna().sum().items() if count
    }
    report["label_values_after"] = sorted(map(str, merged["income"].unique()))
    return merged, report


def prepare_abalone(raw: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(
        io.BytesIO(raw), header=None, names=ABALONE_COLUMNS, skipinitialspace=True, na_filter=False
    )
    frame = _clean_strings(frame)
    report: dict[str, Any] = {
        "rows": len(frame),
        "missing_by_column": {c: int(n) for c, n in frame.isna().sum().items() if n},
        "sex_values": sorted(map(str, frame["Sex"].dropna().unique())),
    }
    return frame, report


# --------------------------------------------------------------------------------------------
# Duplicates and the pool / stream partition
# --------------------------------------------------------------------------------------------


def duplicate_report(frame: pd.DataFrame) -> dict[str, int]:
    """Count exact duplicate rows (all columns equal, missing values treated as equal)."""
    keep_first = frame.duplicated(keep="first")
    keep_none = frame.duplicated(keep=False)
    group_ids = frame.groupby(list(frame.columns), dropna=False, sort=False).ngroup()
    sizes = group_ids.value_counts()
    return {
        "rows": len(frame),
        "extra_copies": int(keep_first.sum()),
        "rows_in_duplicate_groups": int(keep_none.sum()),
        "duplicate_groups": int((sizes > 1).sum()),
        "largest_group": int(sizes.max()) if len(sizes) else 0,
    }


def partition_rows(
    frame: pd.DataFrame,
    target: str,
    *,
    seed: int,
    stream_fraction: float,
    stratify: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Split source row positions into (pool, stream) so that no row is in both.

    Exact duplicate rows are kept together on the same side, so no content-identical row can sit
    in both the pool and the stream. Nothing is removed. With `stratify`, each target class is
    split separately so the two sides have close to the same class balance.
    """
    if not 0 < stream_fraction < 1:
        raise ValueError("stream_fraction must be between 0 and 1")
    group_ids = frame.groupby(list(frame.columns), dropna=False, sort=False).ngroup().to_numpy()
    n_rows = len(frame)
    stream_mask = np.zeros(n_rows, dtype=bool)
    rng = np.random.default_rng(seed)

    if stratify:
        strata = [
            np.flatnonzero((frame[target] == label).to_numpy())
            for label in sorted(frame[target].dropna().unique())
        ]
    else:
        strata = [np.arange(n_rows)]

    chosen_groups: list[int] = []
    for positions in strata:
        # np.unique returns sorted groups, so the seeded shuffle below is reproducible.
        groups, counts = np.unique(group_ids[positions], return_counts=True)
        size_of = dict(zip(groups.tolist(), counts.tolist(), strict=True))
        order = rng.permutation(groups)
        quota = round(stream_fraction * len(positions))
        taken = 0
        for g in order.tolist():
            if taken >= quota:
                break
            chosen_groups.append(g)
            taken += size_of[g]
    stream_mask[np.isin(group_ids, chosen_groups)] = True

    pool = np.flatnonzero(~stream_mask)
    stream = np.flatnonzero(stream_mask)
    return pool, stream


# --------------------------------------------------------------------------------------------
# Shifted monitoring copy (derived from stream rows only)
# --------------------------------------------------------------------------------------------


def make_shifted_copy(
    stream: pd.DataFrame,
    numeric_shifts: tuple[NumericShift, ...],
    category_shifts: tuple[CategoryShift, ...],
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a synthetic shifted copy of the stream and the parameters that were applied.

    Only feature columns are changed. The target column and the row order are untouched, so the
    shifted rows keep their true labels (which is what makes a real accuracy drop possible).
    """
    shifted = stream.copy()
    rng = np.random.default_rng(seed)
    applied: dict[str, Any] = {"numeric": [], "categorical": []}

    for shift in numeric_shifts:
        original = stream[shift.column]
        std = float(original.std())
        delta = shift.std_multiple * std
        is_integer = pd.api.types.is_integer_dtype(original)
        if is_integer:
            # Whole-number columns get a whole-number shift, so the documented amount is exact.
            delta = float(round(delta))
            shifted[shift.column] = original + int(delta)
        else:
            shifted[shift.column] = original + delta
        applied["numeric"].append(
            {
                "column": shift.column,
                "std_multiple": shift.std_multiple,
                "stream_std": round(std, 6),
                "added_amount": round(delta, 6),
                "rounded_to_integer": bool(is_integer),
            }
        )

    for cat in category_shifts:
        column = shifted[cat.column].astype("string")
        eligible = np.flatnonzero((column.notna() & (column != cat.category)).to_numpy())
        count = round(cat.fraction * len(eligible))
        chosen = (
            rng.choice(eligible, size=count, replace=False) if count else np.array([], dtype=int)
        )
        values = column.to_numpy(dtype=object, na_value=None).copy()
        values[chosen] = cat.category
        shifted[cat.column] = pd.array(values, dtype="string")
        applied["categorical"].append(
            {
                "column": cat.column,
                "category": cat.category,
                "fraction_of_eligible_rows": cat.fraction,
                "rows_changed": int(count),
            }
        )
    return shifted, applied


# --------------------------------------------------------------------------------------------
# Writing files and the manifest
# --------------------------------------------------------------------------------------------


def write_csv(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return sha256_file(path)


def index_digest(positions: np.ndarray) -> str:
    return sha256_bytes(np.asarray(positions, dtype="<i8").tobytes())


def class_counts(frame: pd.DataFrame, target: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in frame[target].value_counts().sort_index().items()}


@dataclass(frozen=True)
class PreparedDataset:
    spec: DatasetSpec
    source: pd.DataFrame
    pool: pd.DataFrame
    stream: pd.DataFrame
    shifted: pd.DataFrame
    pool_positions: np.ndarray
    stream_positions: np.ndarray
    duplicates: dict[str, int]
    shift_applied: dict[str, Any]


def build_dataset(
    spec: DatasetSpec, source: pd.DataFrame, *, seed: int, stream_fraction: float
) -> PreparedDataset:
    duplicates = duplicate_report(source)
    pool_pos, stream_pos = partition_rows(
        source, spec.target, seed=seed, stream_fraction=stream_fraction, stratify=spec.stratify
    )
    pool = source.iloc[pool_pos].reset_index(drop=True)
    stream = source.iloc[stream_pos].reset_index(drop=True)
    shifted, applied = make_shifted_copy(
        stream, spec.numeric_shifts, spec.category_shifts, seed=seed + 1
    )
    return PreparedDataset(
        spec, source, pool, stream, shifted, pool_pos, stream_pos, duplicates, applied
    )


def file_names(spec: DatasetSpec) -> dict[str, str]:
    return {
        "training_pool": f"{spec.key}_train_pool.csv",
        "monitoring_stream": f"{spec.key}_monitor_stream.csv",
        "monitoring_shifted": f"{spec.key}_monitor_shifted.csv",
    }


def write_dataset(prepared: PreparedDataset, out_dir: Path) -> dict[str, Any]:
    spec = prepared.spec
    names = file_names(spec)
    frames = {
        "training_pool": prepared.pool,
        "monitoring_stream": prepared.stream,
        "monitoring_shifted": prepared.shifted,
    }
    files: dict[str, Any] = {}
    for role, frame in frames.items():
        path = out_dir / names[role]
        files[role] = {
            "file": names[role],
            "rows": len(frame),
            "columns": list(frame.columns),
            "sha256": write_csv(frame, path),
        }
    entry: dict[str, Any] = {
        "name": spec.name,
        "task": spec.task,
        "target": spec.target,
        "source": {
            "page_url": spec.page_url,
            "download_url": spec.zip_url,
            "doi": spec.doi,
            "licence": spec.licence_name,
            "licence_url": spec.licence_url,
            "licence_statement": spec.licence_statement,
            "citation": spec.citation,
            "checked_on": SOURCES_CHECKED_ON,
            "documented_rows": spec.documented_rows,
        },
        "notes": list(spec.notes),
        "source_rows_after_cleaning": len(prepared.source),
        "duplicates_in_source": prepared.duplicates,
        "duplicate_handling": (
            "No rows removed. Exact duplicate rows are kept together on the same side of the "
            "pool/stream split."
        ),
        "partition": {
            "pool_rows": len(prepared.pool),
            "stream_rows": len(prepared.stream),
            "pool_source_positions_sha256": index_digest(prepared.pool_positions),
            "stream_source_positions_sha256": index_digest(prepared.stream_positions),
            "stratified_by_target": spec.stratify,
        },
        "files": files,
        "shift": prepared.shift_applied,
    }
    if spec.stratify:
        entry["target_counts"] = {
            "source": class_counts(prepared.source, spec.target),
            "training_pool": class_counts(prepared.pool, spec.target),
            "monitoring_stream": class_counts(prepared.stream, spec.target),
        }
    return entry


def build_manifest(
    entries: dict[str, dict[str, Any]],
    raw_files: dict[str, dict[str, Any]],
    *,
    seed: int,
    stream_fraction: float,
) -> dict[str, Any]:
    return {
        "notice": DEMO_NOTICE,
        "split_seed": seed,
        "shift_seed": seed + 1,
        "monitoring_stream_fraction": stream_fraction,
        "raw_files": raw_files,
        "datasets": entries,
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


# --------------------------------------------------------------------------------------------
# Verification of prepared files (used by the script and by tests)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _read_text_frame(path: Path) -> pd.DataFrame:
    """Read a CSV as raw text so comparisons are exact (empty cells stay empty strings)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _row_keys(frame: pd.DataFrame) -> set[str]:
    return {"\x1f".join(row) for row in frame.itertuples(index=False, name=None)}


def verify_demo_data(data_dir: Path, demo_dir: Path | None = None) -> list[CheckResult]:
    """Check the prepared files against their manifest and the separation rules."""
    demo = demo_dir if demo_dir is not None else data_dir / "demo"
    manifest = json.loads((demo / "manifest.json").read_text(encoding="utf-8"))
    results: list[CheckResult] = []

    def add(name: str, passed: bool, detail: str) -> None:
        results.append(CheckResult(name, bool(passed), detail))

    for zip_name, info in manifest["raw_files"].items():
        path = data_dir / "raw" / zip_name
        ok = path.exists() and sha256_file(path) == info["sha256"]
        add(f"raw file hash matches manifest: {zip_name}", ok, info["sha256"][:12])

    for key, entry in manifest["datasets"].items():
        files = entry["files"]
        frames = {role: _read_text_frame(demo / meta["file"]) for role, meta in files.items()}
        pool, stream, shifted = (
            frames["training_pool"],
            frames["monitoring_stream"],
            frames["monitoring_shifted"],
        )
        target = entry["target"]

        hashes_ok = all(sha256_file(demo / m["file"]) == m["sha256"] for m in files.values())
        add(f"[{key}] file hashes match manifest", hashes_ok, "3 files")
        add(
            f"[{key}] source rows equal UCI-documented count",
            entry["source_rows_after_cleaning"] == entry["source"]["documented_rows"],
            f"{entry['source_rows_after_cleaning']} vs {entry['source']['documented_rows']}",
        )
        add(
            f"[{key}] pool + stream equals source rows (nothing dropped)",
            len(pool) + len(stream) == entry["source_rows_after_cleaning"],
            f"{len(pool)} + {len(stream)}",
        )
        overlap = _row_keys(pool) & _row_keys(stream)
        add(
            f"[{key}] no row content shared by pool and stream",
            not overlap,
            f"{len(overlap)} shared",
        )
        add(
            f"[{key}] same columns in all three files",
            list(pool.columns) == list(stream.columns) == list(shifted.columns),
            f"{len(pool.columns)} columns",
        )
        question = sum(bool((frame == "?").to_numpy().any()) for frame in frames.values())
        add(f"[{key}] no '?' markers remain", question == 0, f"{question} files with '?'")
        add(
            f"[{key}] shifted copy has the stream's row count and unchanged target column",
            len(shifted) == len(stream) and bool((shifted[target] == stream[target]).all()),
            f"{len(shifted)} rows",
        )

        shifted_columns = {s["column"] for s in entry["shift"]["numeric"]} | {
            s["column"] for s in entry["shift"]["categorical"]
        }
        untouched = [c for c in stream.columns if c not in shifted_columns]
        same = all(bool((shifted[c] == stream[c]).all()) for c in untouched)
        add(
            f"[{key}] only the documented columns differ from the stream",
            same,
            f"shifted: {sorted(shifted_columns)}",
        )

        for shift in entry["shift"]["numeric"]:
            column = shift["column"]
            moved = shifted[column].astype(float).mean() - stream[column].astype(float).mean()
            add(
                f"[{key}] mean of {column} moved by the documented amount",
                abs(moved - shift["added_amount"]) <= 1e-6,
                f"moved {moved:.4f}, documented {shift['added_amount']}",
            )
        for shift in entry["shift"]["categorical"]:
            column = shift["column"]
            changed = int((shifted[column] != stream[column]).sum())
            add(
                f"[{key}] rows changed in {column} equals the documented count",
                changed == shift["rows_changed"],
                f"{changed} vs {shift['rows_changed']}",
            )

        if entry["partition"]["stratified_by_target"]:
            share_pool = float((pool[target] == pool[target].iloc[0]).mean())
            share_stream = float((stream[target] == pool[target].iloc[0]).mean())
            add(
                f"[{key}] class balance of pool and stream within 0.5 percentage points",
                abs(share_pool - share_stream) <= 0.005,
                f"{share_pool:.4f} vs {share_stream:.4f}",
            )
    return results
