"""The control-plane app factory and the production wiring."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from automl_ops.api_control import errors
from automl_ops.api_control.datasets import Context, router
from automl_ops.api_control.jobs import router as jobs_router
from automl_ops.api_control.registry import router as registry_router
from automl_ops.api_control.security import LocalSecurityMiddleware
from automl_ops.api_control.settings import ApiSettings
from automl_ops.dataset_store import FileDatasetStore
from automl_ops.db import migrate
from automl_ops.db.catalog import CatalogedDatasetStore
from automl_ops.db.config import make_engine
from automl_ops.db.migrate import DbStatus
from automl_ops.job_tracking import tracking_problem
from automl_ops.jobs import JobRepository, worker_running
from automl_ops.registry import RegistryRepository
from automl_ops.registry_mirror import Mirror

API_VERSION = "0.1.0"


class StartupError(RuntimeError):
    """The API must not start (database unreachable, or not at the migration head)."""


def require_current_schema(db_status: Callable[[], DbStatus]) -> DbStatus:
    """The start-up policy (P4-1): refuse to run against a missing or out-of-date schema."""
    try:
        state = db_status()
    except Exception as error:
        raise StartupError(
            f"cannot reach the application database ({type(error).__name__}); "
            "start it with `make db-up`"
        ) from None
    if not state.up_to_date:
        raise StartupError(
            f"the database is at revision {state.current or 'none'} but the code expects "
            f"{state.head}: run `uv run python scripts/db.py upgrade`"
        )
    return state


def create_app(
    settings: ApiSettings,
    store: CatalogedDatasetStore,
    db_status: Callable[[], DbStatus],
    jobs: JobRepository | None = None,
    worker_state: Callable[[], str] | None = None,
    tracking_check: Callable[[], str | None] | None = None,
    registry: RegistryRepository | None = None,
    mirror: Mirror | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await asyncio.to_thread(require_current_schema, db_status)
        app.state.ctx.heavy = asyncio.Semaphore(settings.heavy_concurrency)
        # Restart-safe retry: pending mirror work lives in PostgreSQL, so the first round at
        # start-up already sees whatever an earlier (stopped) API left behind.
        retry = (
            asyncio.create_task(_mirror_retry(mirror, settings.mirror_retry_seconds))
            if mirror is not None
            else None
        )
        try:
            yield
        finally:
            if retry is not None:
                retry.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await retry

    app = FastAPI(
        title="AutoML Ops control plane",
        version=API_VERSION,
        summary="Local, single-user, no authentication: bound to 127.0.0.1.",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.ctx = Context(
        settings=settings,
        store=store,
        db_status=db_status,
        heavy=asyncio.Semaphore(settings.heavy_concurrency),
        upload_lock=threading.Lock(),
        jobs=jobs,
        registry=registry,
        mirror=mirror,
        worker_state=worker_state or (lambda: "unknown"),
        tracking_problem=tracking_check or (lambda: None),
    )
    app.state.api_routes = [  # for the Allow header of a 405
        *router.routes,
        *jobs_router.routes,
        *registry_router.routes,
    ]
    errors.install(app)
    app.include_router(router)
    app.include_router(jobs_router)
    app.include_router(registry_router)
    app.add_middleware(LocalSecurityMiddleware, settings=settings)
    return app


async def _mirror_retry(mirror: Mirror, interval: float) -> None:
    """Every `interval` seconds: sync pending models and re-read drifted or idle ones. Does nothing
    while the mirror is disabled. A round that fails is simply tried again later."""
    while True:
        with contextlib.suppress(Exception):
            await asyncio.to_thread(mirror.run_pending)
        await asyncio.sleep(interval)


def build_app(settings: ApiSettings | None = None) -> FastAPI:
    """The real app: the file store, the PostgreSQL catalog and the migration-head check."""
    settings = settings or ApiSettings.from_environment()
    engine = make_engine()
    store = CatalogedDatasetStore(FileDatasetStore(settings.store_root, settings.limits), engine)
    return create_app(
        settings,
        store,
        lambda: migrate.status(engine),
        jobs=JobRepository(engine),
        registry=RegistryRepository(engine, settings.models_root),
        mirror=Mirror(engine, settings.models_root),
        worker_state=lambda: "running" if worker_running(engine) else "stopped",
        tracking_check=tracking_problem,
    )
