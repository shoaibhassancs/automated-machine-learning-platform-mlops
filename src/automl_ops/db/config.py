"""Where the application database is and how to connect (no secrets in code or output).

The URL comes from `AUTOML_DATABASE_URL`, or is built from the same variables the Compose file
uses (`POSTGRES_PASSWORD` required; host 127.0.0.1, port 55432, user `automl`, database
`automl_ops` by default). If the password is not in the environment, a `POSTGRES_PASSWORD=` line in
the git-ignored `.env` file of the project root is used. The password is never printed.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL, make_url

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 55432
DEFAULT_USER = "automl"
DEFAULT_DATABASE = "automl_ops"
DRIVER = "postgresql+psycopg"  # psycopg 3
CONNECT_TIMEOUT_SECONDS = 5


class DatabaseConfigError(ValueError):
    """The database settings are missing or unusable."""


def read_env_file(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE reader for `.env` (comments and blank lines ignored, optional quotes)."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _project_env_file() -> Path:
    return Path(__file__).resolve().parents[3] / ".env"


def database_url(env: dict[str, str] | None = None, env_file: Path | None = None) -> URL:
    """The connection URL for the application database."""
    source = dict(os.environ if env is None else env)
    explicit = source.get("AUTOML_DATABASE_URL")
    if explicit:
        try:
            url = make_url(explicit)
        except Exception as error:  # sqlalchemy raises ArgumentError
            raise DatabaseConfigError("AUTOML_DATABASE_URL is not a valid URL") from error
        if url.get_backend_name() != "postgresql":
            raise DatabaseConfigError("AUTOML_DATABASE_URL must be a postgresql URL")
        return url.set(drivername=DRIVER)
    password = source.get("POSTGRES_PASSWORD") or read_env_file(
        env_file if env_file is not None else _project_env_file()
    ).get("POSTGRES_PASSWORD")
    if not password:
        raise DatabaseConfigError(
            "no database password: set POSTGRES_PASSWORD (environment or the git-ignored .env "
            "file, see .env.example) or AUTOML_DATABASE_URL"
        )
    try:
        port = int(source.get("AUTOML_DB_PORT", str(DEFAULT_PORT)))
    except ValueError as error:
        raise DatabaseConfigError("AUTOML_DB_PORT must be a number") from error
    return URL.create(
        DRIVER,
        username=source.get("AUTOML_DB_USER", DEFAULT_USER),
        password=password,
        host=source.get("AUTOML_DB_HOST", DEFAULT_HOST),
        port=port,
        database=source.get("AUTOML_DB_NAME", DEFAULT_DATABASE),
    )


def setting(
    name: str,
    default: str | None = None,
    env: dict[str, str] | None = None,
    env_file: Path | None = None,
) -> str | None:
    """One setting: the environment first, then the git-ignored `.env`, then `default`. When `env`
    is given explicitly (tests) the `.env` file is only read if `env_file` is also given."""
    source = os.environ if env is None else env
    value = source.get(name)
    if value:
        return value
    if env is not None and env_file is None:
        return default
    from_file = read_env_file(env_file if env_file is not None else _project_env_file()).get(name)
    return from_file or default


def describe(url: URL) -> str:
    """A printable form of the URL with the password hidden."""
    return url.render_as_string(hide_password=True)


def make_engine(
    url: URL | None = None,
    *,
    connect_timeout: int = CONNECT_TIMEOUT_SECONDS,
    pool_timeout: float = 30.0,
) -> Engine:
    """An engine with a short connect timeout; connections are checked before use. The serving app
    (P4-7) passes shorter timeouts so a slow database never stalls a prediction for long."""
    return create_engine(
        url if url is not None else database_url(),
        pool_pre_ping=True,
        pool_timeout=pool_timeout,
        connect_args={"connect_timeout": connect_timeout},
    )
