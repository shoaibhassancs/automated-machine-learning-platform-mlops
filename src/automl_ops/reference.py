"""Drift reference samples (P4-9, BUILD_PLAN Step 24).

The **reference sample** of a model version is a small, deterministic sample of the rows the model
was *trained* on, together with the model's predictions on them. It is stored in PostgreSQL (table
`reference_samples`), separate from the dataset version and from the sealed package, and is
immutable: one per version, never replaced (re-creating it returns the stored one).

**How it is made** (explicit, `POST /models/{name}/versions/{n}/reference`): the sealed package is
loaded through `load_package` (so it must verify); the dataset version named in its manifest is
read with its SHA-256 check; the frozen holdout split is recomputed from the package's own seed and
settings and must equal the split fingerprints stored in the manifest (else 409, nothing stored).
Only rows at the **training** positions are selected (up to `MAX_SAMPLE` = 1,000, chosen with the
package's seed, so the same package always gives the same sample); the held-out test rows are never
selected, indexed or stored. (The dataset file is read as a whole, as the training code does, and
the target column of every row places rows in the split, as in training; no test-row feature value
is used.) Nothing is retrained and no package file is written.

Small data: if the training part has fewer than 1,000 rows, all of them are the sample. Monitoring
refuses to draw conclusions from fewer than `min_rows` reference rows (status `insufficient_data`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from automl_ops.dataset_store import FileDatasetStore
from automl_ops.hashing import canonical_json, sha256_text
from automl_ops.package import PackageError, load_package
from automl_ops.profile import read_version_frame
from automl_ops.registry import ModelNotFound, PackageUnverified, VersionNotFound
from automl_ops.task import TaskConfig
from automl_ops.train import TrainSettings, make_split, target_values

MAX_SAMPLE = 1000


class ReferenceSampleError(Exception):
    code = "reference_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ReferenceNotFound(ReferenceSampleError):
    code = "reference_not_found"


class ReferenceUnavailable(ReferenceSampleError):
    code = "reference_source_unavailable"


class SplitMismatch(ReferenceSampleError):
    code = "reference_split_mismatch"


class ReferenceStoreFailed(ReferenceSampleError):
    code = "reference_unavailable"


@dataclass(frozen=True)
class ReferenceRow:
    id: int
    model: str
    version: int
    package_id: str
    dataset_ref: str
    dataset_sha256: str
    train_sha256: str
    seed: int
    train_rows: int
    sample_rows: int
    sample_sha256: str
    created_at: datetime


@dataclass(frozen=True)
class ReferenceData:
    meta: ReferenceRow
    columns: list[dict[str, str]]
    rows: list[list[Any]]
    predictions: list[Any]


_SELECT = """
    SELECT r.id, m.name AS model, v.version, r.package_id, r.dataset_ref, r.dataset_sha256,
           r.train_sha256, r.seed, r.train_rows, r.sample_rows, r.sample_sha256, r.created_at
    FROM reference_samples r
    JOIN model_versions v ON v.id = r.version_id
    JOIN registered_models m ON m.id = v.model_id
"""


def _meta(row: Any) -> ReferenceRow:
    return ReferenceRow(**dict(row._mapping))


class ReferenceService:
    def __init__(self, engine: Engine, store_root: Path, models_root: Path) -> None:
        self.engine = engine
        self.store_root = store_root
        self.models_root = models_root

    # ------------------------------------------------------------------ reads

    def get(self, model: str, number: int) -> ReferenceRow:
        try:
            with self.engine.connect() as c:
                self._version(c, model, number)
                row = c.execute(
                    text(_SELECT + " WHERE m.name = :n AND v.version = :v"),
                    {"n": model, "v": number},
                ).first()
        except SQLAlchemyError as error:
            raise ReferenceStoreFailed(
                f"the database is unavailable ({type(error).__name__})"
            ) from None
        if row is None:
            raise ReferenceNotFound(f"{model} version {number} has no reference sample yet")
        return _meta(row)

    def data(self, reference_id: int) -> ReferenceData:
        with self.engine.connect() as c:
            row = c.execute(
                text(
                    _SELECT.replace(
                        "r.created_at", "r.created_at, r.columns, r.rows, r.predictions"
                    )
                    + " WHERE r.id = :i"
                ),
                {"i": reference_id},
            ).one()
        mapping = dict(row._mapping)
        columns, rows, predictions = (mapping.pop(k) for k in ("columns", "rows", "predictions"))
        return ReferenceData(ReferenceRow(**mapping), columns, rows, predictions)

    @staticmethod
    def _version(c: Any, model: str, number: int) -> Any:
        row = c.execute(
            text(
                "SELECT v.id, v.package_id, v.task FROM model_versions v "
                "JOIN registered_models m ON m.id = v.model_id WHERE m.name = :n AND v.version = :v"
            ),
            {"n": model, "v": number},
        ).first()
        if row is not None:
            return row
        if c.execute(text("SELECT 1 FROM registered_models WHERE name = :n"), {"n": model}).first():
            raise VersionNotFound(f"model {model} has no version {number}")
        raise ModelNotFound(f"no model named {model}")

    # ------------------------------------------------------------------ create

    def create(self, model: str, number: int) -> tuple[ReferenceRow, bool]:
        """The version's reference sample and whether this call made it (False: it existed)."""
        try:
            with self.engine.connect() as c:
                version = self._version(c, model, number)
                existing = c.execute(
                    text(_SELECT + " WHERE r.version_id = :v"), {"v": version.id}
                ).first()
        except SQLAlchemyError as error:
            raise ReferenceStoreFailed(
                f"the database is unavailable ({type(error).__name__})"
            ) from None
        if existing is not None:
            return _meta(existing), False
        sample = self._build(version.package_id)
        try:
            with self.engine.begin() as c:
                inserted = c.execute(
                    text(
                        "INSERT INTO reference_samples (version_id, package_id, dataset_ref, "
                        "dataset_sha256, train_sha256, seed, train_rows, sample_rows, columns, "
                        "rows, predictions, sample_sha256) VALUES (:v, :p, :r, :d, :t, :s, :tr, "
                        ":sr, CAST(:c AS jsonb), CAST(:ro AS jsonb), CAST(:pr AS jsonb), :h) "
                        "ON CONFLICT (version_id) DO NOTHING RETURNING id"
                    ),
                    {"v": version.id, **sample},
                ).first()
                row = c.execute(text(_SELECT + " WHERE r.version_id = :v"), {"v": version.id}).one()
        except SQLAlchemyError as error:
            raise ReferenceStoreFailed(
                f"the reference sample could not be stored ({type(error).__name__}); "
                "nothing was changed"
            ) from None
        return _meta(row), inserted is not None

    def _build(self, package_id: str) -> dict[str, Any]:
        import json

        try:
            package = load_package(self.models_root, package_id)
        except PackageError as error:
            raise PackageUnverified(
                f"package {package_id} does not verify; nothing was stored",
                {"problems": [str(error).replace(str(self.models_root), "<models>")]},
            ) from None
        manifest = package.manifest
        try:
            dataset = manifest["dataset"]
            version, frame = read_version_frame(FileDatasetStore(self.store_root), dataset["ref"])
        except Exception as error:
            raise ReferenceUnavailable(
                f"the package's training dataset could not be read ({type(error).__name__})"
            ) from None
        if version.sha256 != dataset["sha256"]:
            raise ReferenceUnavailable("the dataset version no longer matches the package's record")
        config = TaskConfig(
            manifest["task"]["target"],
            manifest["task"]["task"],
            tuple(manifest["task"]["excluded"]),
        )
        settings = TrainSettings(**manifest["settings"])
        split = make_split(target_values(frame, config), config.task, settings)
        if split.fingerprints() != manifest["split"]:
            raise SplitMismatch(
                "the frozen split recomputed from the package's settings differs from its record"
            )
        rng = np.random.default_rng(settings.seed)
        size = min(MAX_SAMPLE, len(split.train_rows))
        chosen = split.train_rows[np.sort(rng.choice(len(split.train_rows), size, replace=False))]
        columns = package.feature_columns
        sample = frame.iloc[chosen][columns].reset_index(drop=True)
        rows = [[None if pd.isna(v) else str(v) for v in r] for r in sample.itertuples(index=False)]
        predicted = package.predict(sample)
        predictions: list[Any] = (
            [str(v) for v in predicted]
            if config.task == "classification"
            else [float(v) for v in predicted]
        )
        kinds = {c["name"]: c["kind"] for c in manifest["features"]["input_columns"]}
        described = [{"name": c, "kind": kinds[c]} for c in columns]
        digest = sha256_text(
            canonical_json({"columns": described, "rows": rows, "predictions": predictions})
        )
        return {
            "p": package_id,
            "r": dataset["ref"],
            "d": dataset["sha256"],
            "t": split.fingerprints()["train_sha256"],
            "s": settings.seed,
            "tr": int(len(split.train_rows)),
            "sr": int(size),
            "c": json.dumps(described),
            "ro": json.dumps(rows),
            "pr": json.dumps(predictions, allow_nan=False),
            "h": digest,
        }
