"""The OpenAPI document of the serving app, generated without a database or a model."""

from __future__ import annotations

import json
from typing import Any, cast

from automl_ops.api_control.settings import PROJECT_ROOT
from automl_ops.api_serving.app import create_app
from automl_ops.api_serving.runtime import ServingRuntime
from automl_ops.api_serving.settings import ServingSettings

OPENAPI_PATH = PROJECT_ROOT / "docs" / "openapi-serving.json"


def generate_openapi() -> dict[str, Any]:
    """Routes and models only: the runtime is never used."""
    return create_app(ServingSettings(port=8200), cast(ServingRuntime, None)).openapi()


def render_openapi() -> str:
    return json.dumps(generate_openapi(), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
