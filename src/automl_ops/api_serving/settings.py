"""Serving-app settings: the environment first, then the git-ignored `.env`, then the defaults.

Like the control plane, the serving app listens on the loopback interface only (`HOST` is a
constant). It reuses the control plane's local-request rules (Host check, Origin check on POST,
JSON content type, body limit) through `ApiSettings`, so there is one implementation of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from automl_ops.api_control.settings import (
    DEFAULT_UI_ORIGIN,
    PROJECT_ROOT,
    ApiSettings,
    SettingsError,
)
from automl_ops.db.config import setting

HOST = "127.0.0.1"
DEFAULT_PORT = 8200
MAX_ROWS = 100  # rows per /predict request


@dataclass(frozen=True)
class ServingSettings:
    port: int = DEFAULT_PORT
    ui_origin: str = DEFAULT_UI_ORIGIN
    models_root: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "models")
    poll_seconds: float = 5.0  # how often the serving app re-reads the active deployment
    retry_seconds: float = 30.0  # a model that failed to load is not retried sooner than this
    predict_concurrency: int = 4  # predictions computed at the same time

    def __post_init__(self) -> None:
        if not 1.0 <= self.poll_seconds <= 300.0:
            raise SettingsError("poll_seconds must be between 1 and 300")
        if not 1.0 <= self.retry_seconds <= 3600.0:
            raise SettingsError("retry_seconds must be between 1 and 3600")
        if self.predict_concurrency < 1:
            raise SettingsError("predict_concurrency must be at least 1")
        self.http_rules()  # validates the port and the UI origin

    def http_rules(self) -> ApiSettings:
        """The request rules of the control plane, applied to this app's port."""
        return ApiSettings(port=self.port, ui_origin=self.ui_origin)

    @staticmethod
    def from_environment(env: dict[str, str] | None = None) -> ServingSettings:
        def get(name: str, default: str | None = None) -> str | None:
            return setting(name, default, env=env)

        try:
            port = int(get("AUTOML_SERVING_PORT", str(DEFAULT_PORT)) or DEFAULT_PORT)
        except ValueError as error:
            raise SettingsError("AUTOML_SERVING_PORT must be a number") from error
        models = get("AUTOML_MODELS_DIR")
        return ServingSettings(
            port=port,
            ui_origin=get("AUTOML_UI_ORIGIN", DEFAULT_UI_ORIGIN) or DEFAULT_UI_ORIGIN,
            models_root=Path(models) if models else PROJECT_ROOT / "data" / "models",
        )
