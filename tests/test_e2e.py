# ruff: noqa: F811
"""P4-10 (Step 35, backend half): the whole lifecycle through both apps' HTTP interfaces.

A real worker, real sealed packages, real MLflow tracking (a temporary store), real PostgreSQL
(throwaway database) and the simulator. NOT RUN (skipped) when the server is unreachable. Nothing
here touches `data/` or the real database."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from api_support import BASE
from automl_ops import e2e
from automl_ops import simulator as sim
from automl_ops.api_control.app import create_app as create_control
from automl_ops.api_control.settings import ApiSettings
from automl_ops.api_serving.app import create_app as create_serving
from automl_ops.api_serving.runtime import ServingRuntime
from automl_ops.api_serving.settings import ServingSettings
from automl_ops.dataset_store import FileDatasetStore
from automl_ops.db import migrate
from automl_ops.db.catalog import CatalogedDatasetStore
from automl_ops.deployments import DeploymentRepository
from automl_ops.feedback import FeedbackRepository
from automl_ops.jobs import JobRepository
from automl_ops.monitoring import MonitoringRepository
from automl_ops.reference import ReferenceService
from automl_ops.registry import RegistryRepository
from automl_ops.simulator import Reply
from automl_ops.tracking import TrackingStore
from automl_ops.worker import Worker, WorkerSettings
from conftest import JobEnv, toy_bytes
from test_simulator import AppHttp, demo_dir

pytestmark = pytest.mark.postgres


class ControlHttp(AppHttp):
    def upload(
        self, path: str, fields: dict[str, str], filename: str, data: bytes, field: str = "file"
    ) -> Reply:
        r = self.client.post(path, data=fields, files={field: (filename, data, "text/csv")})
        return Reply(r.status_code, r.json() if r.content else None)


class Stack:
    """Both apps and the worker in process, over one throwaway database."""

    def __init__(self, env: JobEnv, tmp: Path, *, track: bool = True) -> None:
        self.env = env
        engine = env.engine
        settings = ApiSettings(store_root=env.store_root, models_root=env.models)
        feedback = FeedbackRepository(engine)
        references = ReferenceService(engine, env.store_root, env.models)
        self.deployments = DeploymentRepository(engine, env.models)
        control = create_control(
            settings,
            CatalogedDatasetStore(FileDatasetStore(env.store_root, settings.limits), engine),
            lambda: migrate.status(engine),
            jobs=JobRepository(engine),
            registry=RegistryRepository(engine, env.models),
            deployments=self.deployments,
            feedback=feedback,
            monitoring=MonitoringRepository(engine, references, feedback),
            references=references,
        )
        self.control = ControlHttp(
            TestClient(control, base_url=BASE, raise_server_exceptions=False)
        )
        self.runtime = ServingRuntime(engine, env.models, retry_seconds=1.0)
        serving = create_serving(
            ServingSettings(port=8200, models_root=env.models, poll_seconds=60), self.runtime
        )
        self.serving = AppHttp(
            TestClient(serving, base_url="http://127.0.0.1:8200", raise_server_exceptions=False)
        )
        mlflow = TrackingStore(tmp / "mlflow", "sqlite")
        self.worker = Worker(
            engine,
            FileDatasetStore(env.store_root),
            env.models,
            settings=WorkerSettings(poll_seconds=0.05, heartbeat_seconds=0.5),
            tracking_store=(lambda: mlflow) if track else _refuse,
        )
        self.ran = False

    def idle(self) -> None:
        if not self.ran:
            self.ran = True
            self.worker.run(once=True)
        self.runtime.refresh()

    def story(self, tmp: Path, **kw: Any) -> e2e.Report:
        root = demo_dir(tmp / "demo", 1100)
        return e2e.run_story(
            self.control,
            self.serving,
            train_csv=("toy.csv", toy_bytes("classification")),
            dataset="toye2e",
            target="y",
            task="classification",
            model="e2e-model",
            normal=sim.load_stream(root / "toy_monitor_stream.csv"),
            shifted=sim.load_stream(root / "toy_monitor_shifted.csv"),
            idle=self.idle,
            timeout=300,
            **kw,
        )


def _refuse() -> TrackingStore:
    from automl_ops.job_tracking import TrackingNotAllowed

    raise TrackingNotAllowed("tracking is not configured")


def stages(report: e2e.Report) -> dict[str, bool]:
    return {s["stage"]: s["ok"] for s in report.stages}


def test_the_whole_lifecycle_runs_through_the_apis(job_env: JobEnv, tmp_path: Path) -> None:
    stack = Stack(job_env, tmp_path)
    report = stack.story(tmp_path)
    assert report.passed, report.stages
    assert list(stages(report)) == [
        "upload",
        "train",
        "tracking",
        "register_promote",
        "deploy_serve",
        "traffic_feedback",
        "normal_no_alert",
        "drift_alert",
        "drift_resolved",
    ]
    with job_env.engine.connect() as c:
        counts = {
            t: c.execute(text(f"SELECT count(*) FROM {t}")).scalar_one()
            for t in (
                "training_jobs",
                "model_versions",
                "deployments",
                "predictions",
                "feedback",
                "reference_samples",
                "monitoring_reports",
                "alerts",
            )
        }
        real = c.execute(
            text("SELECT count(*) FROM predictions WHERE NOT is_simulated")
        ).scalar_one()
    assert counts["training_jobs"] == counts["model_versions"] == counts["deployments"] == 1
    assert counts["predictions"] == 3000 and counts["reference_samples"] == 1
    assert counts["monitoring_reports"] == 3 and counts["feedback"] == 900 and counts["alerts"] >= 2
    assert real == 0  # nothing the story sent is presented as real traffic
    json_report = report.to_dict()
    assert json_report["passed"] is True and len(json_report["phases"]) == 3


def test_it_fails_at_the_stage_that_was_broken_on_purpose(job_env: JobEnv, tmp_path: Path) -> None:
    # tracking not configured for the worker: the tracked job fails and the story stops there
    stack = Stack(job_env, tmp_path, track=False)
    report = stack.story(tmp_path)
    assert not report.passed and report.failed == "train"
    assert "tracking_not_allowed" in report.stages[-1]["detail"]
    assert stages(report) == {"upload": True, "train": False}
    assert report.phases == []
    with job_env.engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM predictions")).scalar_one() == 0


def test_data_that_does_not_drift_fails_the_alert_stage_honestly(
    job_env: JobEnv, tmp_path: Path
) -> None:
    import dataclasses

    stack = Stack(job_env, tmp_path)
    root = demo_dir(tmp_path / "demo", 1100)
    normal = sim.load_stream(root / "toy_monitor_stream.csv")
    unshifted = dataclasses.replace(
        sim.load_stream(root / "toy_monitor_shifted.csv"), frame=normal.frame
    )
    report = e2e.run_story(
        stack.control,
        stack.serving,
        train_csv=("toy.csv", toy_bytes("classification")),
        dataset="toye2e",
        target="y",
        task="classification",
        model="e2e-model",
        normal=normal,
        shifted=unshifted,
        idle=stack.idle,
        timeout=300,
    )
    assert report.failed == "drift_alert" and "no drift alert" in report.stages[-1]["detail"]
    assert stages(report)["normal_no_alert"] is True and len(report.phases) == 3


def test_a_failed_job_stops_the_story_without_a_model(job_env: JobEnv, tmp_path: Path) -> None:
    stack = Stack(job_env, tmp_path)
    root = demo_dir(tmp_path / "demo", 1100)
    report = e2e.run_story(
        stack.control,
        stack.serving,
        train_csv=("toy.csv", toy_bytes("classification")),
        dataset="toye2e",
        target="no_such_column",
        task="classification",
        model="e2e-model",
        normal=sim.load_stream(root / "toy_monitor_stream.csv"),
        shifted=sim.load_stream(root / "toy_monitor_shifted.csv"),
        idle=stack.idle,
        timeout=60,
    )
    assert not report.passed and report.failed in ("train",)
    with job_env.engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM model_versions")).scalar_one() == 0


def test_a_job_that_never_finishes_times_out(job_env: JobEnv, tmp_path: Path) -> None:
    stack = Stack(job_env, tmp_path)
    stack.ran = True  # nobody runs the worker
    ticks = iter(range(0, 10_000, 100))
    root = demo_dir(tmp_path / "demo", 1100)
    report = e2e.run_story(
        stack.control,
        stack.serving,
        train_csv=("toy.csv", toy_bytes("classification")),
        dataset="toye2e",
        target="y",
        task="classification",
        model="e2e-model",
        normal=sim.load_stream(root / "toy_monitor_stream.csv"),
        shifted=sim.load_stream(root / "toy_monitor_shifted.csv"),
        idle=lambda: None,
        timeout=250,
        clock=lambda: float(next(ticks)),
    )
    assert report.failed == "train" and "did not finish" in report.stages[-1]["detail"]
