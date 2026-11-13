"""Model packages: create, finalize, seal, load, verify (BUILD_PLAN Step 12).

A package is a directory `<models root>/<package id>/` (default root `data/models/`, git-ignored):

- `model.joblib` (at creation): the fitted preprocessing + estimator `Pipeline`, as one object.
- `manifest.json` (at creation): task, features, classes, parameters, seed, provenance, CV and
  selection results, library versions and the model file's SHA-256.
- `parity.json` (at creation): a small sample of training rows and the predictions made for them.
- `attempt.json`: the one-attempt marker, created exclusively before any test row is read.
- `test_result.json`: the single test evaluation, created exclusively.
- `seal.json` (last): SHA-256 of every file above and of the whole package.

**Lifecycle.** The package is built in `.staging-<id>/` (writable), checked by reloading it and
comparing predictions with the parity sample, then moved into place with one atomic rename. It is
now *open*: files are read-only but the folder still accepts the two finalization files. The
attempt marker is created exclusively (a second attempt fails), the single test is run, the test
result is created exclusively, and `seal_package` writes `seal.json` and makes the folder read-only.
States: `open` (no attempt yet), `spent` (attempt marker but no seal: a crashed or failed attempt;
**never retried**, never loadable) and `sealed` (usable).

**Trust.** Model files are pickle-based joblib files. `joblib.load` can run arbitrary code, so only
packages produced by this platform in its own store are ever loaded: a package id is resolved
inside the models root and nowhere else, and the SHA-256 checks run **before** the file is
deserialized. The hashes protect against accidental corruption and wrong files, **not** against
someone who can rewrite the whole folder (they could also rewrite the hashes). Never load a model
file that came from anywhere else.

`parity.json` contains up to 25 rows copied from the *training* part, so a package folder holds a
small piece of the data; it stays in the git-ignored models folder.
"""

from __future__ import annotations

import io
import json
import os
import platform
import re
import secrets
import shutil
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from automl_ops.hashing import canonical_json, sha256_bytes, sha256_file, sha256_text
from automl_ops.pipeline import PIPELINE_FORMAT
from automl_ops.train import TRAIN_FORMAT

Array = NDArray[Any]

PACKAGE_FORMAT = 1
MODEL_FILE = "model.joblib"
MANIFEST_FILE = "manifest.json"
PARITY_FILE = "parity.json"
ATTEMPT_FILE = "attempt.json"
TEST_RESULT_FILE = "test_result.json"
SEAL_FILE = "seal.json"
SEALED_FILES = (MODEL_FILE, MANIFEST_FILE, PARITY_FILE, ATTEMPT_FILE, TEST_RESULT_FILE)
PARITY_ROWS = 25
PARITY_TOLERANCE = 1e-12
LIBRARIES = ("scikit-learn", "pandas", "numpy", "scipy", "joblib")
ID_PATTERN = re.compile(r"^m-\d{8}T\d{6}Z-[0-9a-f]{8}$")


class PackageError(ValueError):
    """A package cannot be created, loaded or verified."""


class PackageSpentError(PackageError):
    """The package's one test attempt was already made (or begun)."""


def installed_version(name: str) -> str:
    return version(name)


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    versions.update({name: installed_version(name) for name in LIBRARIES})
    return versions


def new_package_id() -> str:
    return f"m-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"


def package_path(root: Path, package_id: str) -> Path:
    """The folder of a package id, only ever inside `root`."""
    if not ID_PATTERN.fullmatch(package_id):
        raise PackageError(
            f"{package_id!r} is not a package id (expected m-YYYYMMDDTHHMMSSZ-xxxxxxxx)"
        )
    path = root / package_id
    if path.is_symlink() or not path.is_dir():
        raise PackageError(f"no package {package_id!r} in {root}")
    return path


def list_packages(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(
        p.name
        for p in root.iterdir()
        if ID_PATTERN.fullmatch(p.name) and p.is_dir() and not p.is_symlink()
    )


def leftover_staging(root: Path) -> list[str]:
    """Interrupted creations (never usable, never loaded)."""
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.name.startswith(".staging-"))


def package_state(path: Path) -> str:
    if (path / SEAL_FILE).exists():
        return "sealed"
    return "spent" if (path / ATTEMPT_FILE).exists() else "open"


# --------------------------------------------------------------------------------------------
# Rows <-> frames, and prediction through a package
# --------------------------------------------------------------------------------------------


def _frame_from_rows(columns: list[str], rows: list[list[str | None]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns, dtype="string")


def parity_rows(x_train: pd.DataFrame) -> list[list[str | None]]:
    """The first rows of the training inputs as text, plus one row with every value missing."""
    sample = x_train.head(PARITY_ROWS)
    rows: list[list[str | None]] = [
        [None if pd.isna(v) else str(v) for v in row] for row in sample.itertuples(index=False)
    ]
    rows.append([None] * len(x_train.columns))
    return rows


def _predict(
    pipeline: Pipeline, columns: list[str], frame: pd.DataFrame
) -> tuple[Array, Array | None]:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise PackageError(f"the input is missing column(s) the model needs: {missing}")
    x = frame[columns]
    predicted: Array = pipeline.predict(x)
    proba: Array | None = pipeline.predict_proba(x) if hasattr(pipeline, "classes_") else None
    return predicted, proba


def _parity_problems(pipeline: Pipeline, parity: dict[str, Any]) -> list[str]:
    columns: list[str] = parity["columns"]
    frame = _frame_from_rows(columns, parity["rows"])
    predicted, proba = _predict(pipeline, columns, frame)
    problems: list[str] = []
    expected = parity["predictions"]
    if parity["probabilities"] is None:
        same = np.allclose(
            predicted.astype(float),
            np.array(expected, dtype=float),
            rtol=PARITY_TOLERANCE,
            atol=PARITY_TOLERANCE,
        )
    else:
        same = [str(v) for v in predicted] == [str(v) for v in expected]
    if not same:
        problems.append("predictions after reload differ from the saved reference predictions")
    if parity["probabilities"] is not None and (
        proba is None
        or not np.allclose(
            proba, np.array(parity["probabilities"]), rtol=PARITY_TOLERANCE, atol=PARITY_TOLERANCE
        )
    ):
        problems.append("probabilities after reload differ from the saved reference probabilities")
    return problems


# --------------------------------------------------------------------------------------------
# Loaded package
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ModelPackage:
    """A loaded, verified package with a thin prediction interface."""

    path: Path
    manifest: dict[str, Any]
    pipeline: Pipeline
    test_result: dict[str, Any] | None

    @property
    def package_id(self) -> str:
        return str(self.manifest["package_id"])

    @property
    def feature_columns(self) -> list[str]:
        return [c["name"] for c in self.manifest["features"]["input_columns"]]

    @property
    def classes(self) -> list[str] | None:
        classes: list[str] | None = self.manifest["classes"]
        return classes

    def predict(self, frame: pd.DataFrame) -> Array:
        """Predictions for a text frame; extra columns are ignored, missing ones are an error."""
        return _predict_only(self, frame, probabilities=False)

    def predict_proba(self, frame: pd.DataFrame) -> Array:
        """Class probabilities in `classes` order (classification only)."""
        return _predict_only(self, frame, probabilities=True)


def _predict_only(package: ModelPackage, frame: pd.DataFrame, *, probabilities: bool) -> Array:
    columns = package.feature_columns
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise PackageError(f"the input is missing column(s) the model needs: {missing}")
    if probabilities:
        if package.classes is None:
            raise PackageError("this is a regression model; it has no class probabilities")
        proba: Array = package.pipeline.predict_proba(frame[columns])
        return proba
    predicted: Array = package.pipeline.predict(frame[columns])
    return predicted


# --------------------------------------------------------------------------------------------
# Creating a package (staging -> open)
# --------------------------------------------------------------------------------------------


def _write_new(path: Path, text: str) -> None:
    """Create `path` exclusively (fails if it exists), flush it to disk."""
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def create_package(
    root: Path, pipeline: Pipeline, manifest: dict[str, Any], parity_input: list[list[str | None]]
) -> Path:
    """Write a new package in `.staging-<id>/`, verify it by reloading, then move it into place.

    `manifest` supplies everything about the model and its provenance; the package id, creation
    time, library versions and model-file hash are added here. Returns the folder of the new,
    *open* package. On any failure the staging folder is removed and nothing is left in `root`.
    """
    root.mkdir(parents=True, exist_ok=True)
    package_id = new_package_id()
    staging = root / f".staging-{package_id}"
    staging.mkdir()
    try:
        columns = [c["name"] for c in manifest["features"]["input_columns"]]
        model_path = staging / MODEL_FILE
        joblib.dump(pipeline, model_path)
        frame = _frame_from_rows(columns, parity_input)
        predicted, proba = _predict(pipeline, columns, frame)
        classification = manifest["classes"] is not None
        parity = {
            "parity_format": 1,
            "columns": columns,
            "rows": parity_input,
            "predictions": [str(v) for v in predicted]
            if classification
            else [float(v) for v in predicted],
            "probabilities": proba.tolist() if classification and proba is not None else None,
        }
        full = dict(manifest)
        full.update(
            {
                "package_format": PACKAGE_FORMAT,
                "package_id": package_id,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "libraries": library_versions(),
                "formats": {"pipeline": PIPELINE_FORMAT, "train": TRAIN_FORMAT},
                "model_file": {
                    "name": MODEL_FILE,
                    "sha256": sha256_file(model_path),
                    "bytes": model_path.stat().st_size,
                },
            }
        )
        _write_new(staging / PARITY_FILE, _json_text(parity))
        _write_new(staging / MANIFEST_FILE, _json_text(full))
        reloaded = _read_pipeline(staging, full)
        problems = _parity_problems(reloaded, parity)
        if problems:
            raise PackageError("the new package failed its parity check: " + "; ".join(problems))
        for name in (MODEL_FILE, MANIFEST_FILE, PARITY_FILE):
            (staging / name).chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        final = root / package_id
        os.rename(staging, final)  # atomic: the package appears whole or not at all
        return final
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


# --------------------------------------------------------------------------------------------
# Finalization files and the seal
# --------------------------------------------------------------------------------------------


def begin_test_attempt(path: Path, info: dict[str, Any]) -> Path:
    """Create the one-attempt marker. Raises `PackageSpentError` if it exists; call this before
    any test row is read."""
    if (path / SEAL_FILE).exists():
        raise PackageSpentError(f"{path.name}: already sealed; its test part was used")
    try:
        _write_new(path / ATTEMPT_FILE, _json_text(info))
    except FileExistsError:
        raise PackageSpentError(
            f"{path.name}: the test attempt was already made for this package; it is never "
            "retried. Run a new job to get a new package."
        ) from None
    return path / ATTEMPT_FILE


def write_test_result(path: Path, result: dict[str, Any]) -> None:
    """Create `test_result.json` exclusively; an existing result is never overwritten."""
    if not (path / ATTEMPT_FILE).exists():
        raise PackageError("a test result can only follow the attempt marker")
    try:
        _write_new(path / TEST_RESULT_FILE, _json_text(result))
    except FileExistsError:
        raise PackageSpentError(f"{path.name}: a test result already exists") from None


def seal_package(path: Path) -> dict[str, Any]:
    """Write `seal.json` (hashes of every file) and make the package read-only."""
    if package_state(path) != "spent":
        raise PackageError(f"{path.name}: only a package with a test attempt can be sealed")
    if not (path / TEST_RESULT_FILE).exists():
        raise PackageError(f"{path.name}: no test result to seal")
    json.loads((path / TEST_RESULT_FILE).read_text(encoding="utf-8"))  # must be complete JSON
    files = {name: sha256_file(path / name) for name in SEALED_FILES}
    seal = {"seal_format": 1, "files": files, "package_sha256": sha256_text(canonical_json(files))}
    _write_new(path / SEAL_FILE, _json_text(seal))
    read_only = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    for name in (*SEALED_FILES, SEAL_FILE):
        (path / name).chmod(read_only)
    path.chmod(read_only | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return seal


# --------------------------------------------------------------------------------------------
# Loading and verifying
# --------------------------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PackageError(f"cannot read {path.name}: {error}") from error
    if not isinstance(loaded, dict):
        raise PackageError(f"{path.name} is not a JSON object")
    return loaded


def _read_pipeline(path: Path, manifest: dict[str, Any]) -> Pipeline:
    """Read the model file, check its SHA-256 **before** deserializing, check what came out."""
    data = (path / MODEL_FILE).read_bytes()
    expected = manifest["model_file"]["sha256"]
    if sha256_bytes(data) != expected:
        raise PackageError(f"{MODEL_FILE} does not match the SHA-256 in the manifest")
    pipeline = joblib.load(io.BytesIO(data))
    if not isinstance(pipeline, Pipeline):
        raise PackageError(f"{MODEL_FILE} does not contain a scikit-learn Pipeline")
    expected_columns = [c["name"] for c in manifest["features"]["input_columns"]]
    if list(pipeline.feature_names_in_) != expected_columns:
        raise PackageError("the model's input columns differ from the manifest's feature order")
    return pipeline


def _seal_problems(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        seal = _read_json(path / SEAL_FILE)
    except PackageError as error:
        return [str(error)]
    files = seal.get("files", {})
    for name in SEALED_FILES:
        file = path / name
        if not file.exists():
            problems.append(f"{name} is missing")
        elif files.get(name) != sha256_file(file):
            problems.append(f"{name} does not match the SHA-256 in {SEAL_FILE}")
    if seal.get("package_sha256") != sha256_text(canonical_json(files)):
        problems.append("the package SHA-256 does not match its file hashes")
    for name in (*SEALED_FILES, SEAL_FILE):
        file = path / name
        if file.exists() and file.stat().st_mode & 0o222:
            problems.append(f"{name} is writable (a sealed package must be read-only)")
    return problems


def _version_problems(manifest: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if manifest.get("package_format") != PACKAGE_FORMAT:
        problems.append(
            f"package format {manifest.get('package_format')!r} is not supported "
            f"(this code reads format {PACKAGE_FORMAT})"
        )
    formats = manifest.get("formats", {})
    if formats.get("pipeline") != PIPELINE_FORMAT or formats.get("train") != TRAIN_FORMAT:
        problems.append("the pipeline or training format differs from this code's")
    saved = manifest.get("libraries", {}).get("scikit-learn")
    if saved != installed_version("scikit-learn"):
        problems.append(
            f"saved with scikit-learn {saved}, but {installed_version('scikit-learn')} is installed"
        )
    return problems


def load_package(
    root: Path, package_id: str, *, allow_version_mismatch: bool = False
) -> ModelPackage:
    """Load a **sealed** package from the platform's own store, after verifying it.

    Order: id and folder -> state -> seal hashes and read-only check -> manifest and format
    versions -> model-file SHA-256 -> only then `joblib.load` -> feature-order check.
    """
    path = package_path(root, package_id)
    state = package_state(path)
    if state != "sealed":
        raise PackageError(
            f"{package_id} is {state}, not sealed: "
            + (
                "its test attempt was not completed and is never retried"
                if state == "spent"
                else "it has not been finalized"
            )
        )
    problems = _seal_problems(path)
    if problems:
        raise PackageError(f"{package_id} failed verification: " + "; ".join(problems))
    manifest = _read_json(path / MANIFEST_FILE)
    version_problems = _version_problems(manifest)
    if version_problems and not allow_version_mismatch:
        raise PackageError(f"{package_id}: " + "; ".join(version_problems))
    pipeline = _read_pipeline(path, manifest)
    return ModelPackage(path, manifest, pipeline, _read_json(path / TEST_RESULT_FILE))


@dataclass(frozen=True)
class PackageCheck:
    package_id: str
    state: str
    ok: bool
    problems: tuple[str, ...]


def verify_package(root: Path, package_id: str, *, deep: bool = True) -> PackageCheck:
    """Check a package without raising: state, seal hashes, read-only files, versions and (deep)
    a reload with a prediction-parity check against the saved reference predictions."""
    try:
        path = package_path(root, package_id)
    except PackageError as error:
        return PackageCheck(package_id, "missing", False, (str(error),))
    state = package_state(path)
    if state == "open":
        return PackageCheck(package_id, state, False, ("not finalized: no test attempt was made",))
    if state == "spent":
        return PackageCheck(
            package_id,
            state,
            False,
            ("spent: a test attempt began but was not completed; it is never retried",),
        )
    problems = _seal_problems(path)
    if not problems:
        try:
            manifest = _read_json(path / MANIFEST_FILE)
            problems.extend(_version_problems(manifest))
            result = _read_json(path / TEST_RESULT_FILE)
            if result.get("selection_sha256") != manifest.get("selection_sha256"):
                problems.append("the test result refers to a different frozen selection")
            if deep and not problems:
                pipeline = _read_pipeline(path, manifest)
                problems.extend(_parity_problems(pipeline, _read_json(path / PARITY_FILE)))
        except PackageError as error:
            problems.append(str(error))
    return PackageCheck(package_id, state, not problems, tuple(problems))


def load_pipeline_for_test(path: Path) -> Pipeline:
    """Load the fitted pipeline of an *open* package for its single test (hash-checked)."""
    manifest = _read_json(path / MANIFEST_FILE)
    return _read_pipeline(path, manifest)
