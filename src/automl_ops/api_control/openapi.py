"""The OpenAPI document of the control plane, generated without a database or a store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from automl_ops.api_control.app import create_app
from automl_ops.api_control.settings import PROJECT_ROOT, ApiSettings
from automl_ops.dataset_store import FileDatasetStore
from automl_ops.db.catalog import CatalogedDatasetStore

OPENAPI_PATH = PROJECT_ROOT / "docs" / "openapi-control.json"


def generate_openapi() -> dict[str, Any]:
    """Routes and models only: the store and the database are never touched."""
    settings = ApiSettings(port=8100)
    store = CatalogedDatasetStore(FileDatasetStore(Path("/nonexistent")), cast(Any, None))

    def no_database() -> Any:
        raise RuntimeError("OpenAPI generation does not use a database")

    return create_app(settings, store, no_database).openapi()


def render_openapi() -> str:
    return json.dumps(generate_openapi(), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
