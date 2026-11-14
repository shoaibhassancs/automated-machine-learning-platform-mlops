"""API settings: the environment first, then the git-ignored `.env`, then the defaults.

The API never listens on anything but the loopback interface: `HOST` is a constant, not a setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from automl_ops.db.config import setting
from automl_ops.ingest import DEFAULT_LIMITS, IngestLimits

HOST = "127.0.0.1"
DEFAULT_PORT = 8100
DEFAULT_SERVING_PORT = 8200  # the serving app (P4-7); the control plane only calls it
DEFAULT_UI_ORIGIN = "http://localhost:5173"  # the Vite dev server (Step 27)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# The request body may be larger than the CSV by this much (multipart boundaries, field values,
# file part headers). The file itself is held to `IngestLimits.max_bytes` separately.
MULTIPART_OVERHEAD_BYTES = 64 * 1024
JSON_BODY_LIMIT_BYTES = 1024 * 1024
HEAVY_CONCURRENCY = 2


class SettingsError(ValueError):
    """A setting is missing or unusable."""


def origin_of(value: str) -> tuple[str, str, int] | None:
    """Parse an `Origin` value into (scheme, host, port), or None if it is not a plain origin.

    Rejects `null`, anything with a path, query, fragment or credentials, other schemes, and
    anything malformed. Default ports are filled in so `http://a` equals `http://a:80`.
    """
    if not value or value != value.strip() or any(ord(ch) < 33 or ord(ch) > 126 for ch in value):
        return None
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    if parts.path or parts.query or parts.fragment or parts.username or parts.password:
        return None
    if value.endswith(("/", "?", "#")):
        return None
    return parts.scheme, parts.hostname.lower(), port or (443 if parts.scheme == "https" else 80)


@dataclass(frozen=True)
class ApiSettings:
    port: int = DEFAULT_PORT
    ui_origin: str = DEFAULT_UI_ORIGIN
    store_root: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "store")
    limits: IngestLimits = DEFAULT_LIMITS
    heavy_concurrency: int = HEAVY_CONCURRENCY
    mirror_retry_seconds: float = 30.0  # background retry of pending MLflow mirror work
    models_root: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "models")
    serving_port: int = DEFAULT_SERVING_PORT

    def __post_init__(self) -> None:
        if not 1024 <= self.port <= 65535:
            raise SettingsError("the API port must be between 1024 and 65535")
        if origin_of(self.ui_origin) is None:
            raise SettingsError(
                "AUTOML_UI_ORIGIN must be one origin like http://localhost:5173 "
                "(scheme, host and optional port; no path)"
            )
        if not 1024 <= self.serving_port <= 65535:
            raise SettingsError("the serving port must be between 1024 and 65535")
        if not 1.0 <= self.mirror_retry_seconds <= 3600.0:
            raise SettingsError("mirror_retry_seconds must be between 1 and 3600")
        if self.heavy_concurrency < 1:
            raise SettingsError("heavy_concurrency must be at least 1")

    @property
    def allowed_hosts(self) -> frozenset[str]:
        """Exact `Host` header values accepted (lower case, with the port)."""
        return frozenset({f"127.0.0.1:{self.port}", f"localhost:{self.port}", f"[::1]:{self.port}"})

    @property
    def ui_origin_tuple(self) -> tuple[str, str, int]:
        parsed = origin_of(self.ui_origin)
        assert parsed is not None  # checked in __post_init__
        return parsed

    @property
    def upload_body_limit(self) -> int:
        return self.limits.max_bytes + MULTIPART_OVERHEAD_BYTES

    @staticmethod
    def from_environment(env: dict[str, str] | None = None) -> ApiSettings:
        def get(name: str, default: str | None = None) -> str | None:
            return setting(name, default, env=env)

        try:
            port = int(get("AUTOML_API_PORT", str(DEFAULT_PORT)) or DEFAULT_PORT)
        except ValueError as error:
            raise SettingsError("AUTOML_API_PORT must be a number") from error
        try:
            serving_port = int(
                get("AUTOML_SERVING_PORT", str(DEFAULT_SERVING_PORT)) or DEFAULT_SERVING_PORT
            )
        except ValueError as error:
            raise SettingsError("AUTOML_SERVING_PORT must be a number") from error
        if serving_port == port:
            raise SettingsError("AUTOML_SERVING_PORT must differ from AUTOML_API_PORT")
        store = get("AUTOML_STORE_DIR")
        models = get("AUTOML_MODELS_DIR")
        return ApiSettings(
            port=port,
            ui_origin=get("AUTOML_UI_ORIGIN", DEFAULT_UI_ORIGIN) or DEFAULT_UI_ORIGIN,
            store_root=Path(store) if store else PROJECT_ROOT / "data" / "store",
            models_root=Path(models) if models else PROJECT_ROOT / "data" / "models",
            serving_port=serving_port,
        )
