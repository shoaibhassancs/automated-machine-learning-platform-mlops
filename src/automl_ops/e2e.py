"""The backend lifecycle in one story (P4-10, backend half of BUILD_PLAN Step 35).

Over HTTP only, against running services: upload a dataset, train a final job with experiment
tracking, check it was tracked, register and promote the model, deploy it, check the serving app
answers, make the reference sample, then run the simulator's normal / shifted / normal traffic with
feedback labels and a monitoring run after each phase. Each stage is recorded as passed or failed
with what was observed; the story stops at the first failed stage and never reports a stage it did
not run. It uses the default monitoring thresholds, so a stage that expects an alert can fail
honestly on data that does not trip one.

The job itself runs in the separate worker process (or, in tests, through the `idle` hook).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from automl_ops.simulator import Http, Phase, Reply, SimulatorError, Stream, demo

TERMINAL = ("succeeded", "failed", "cancelled")
ALGORITHMS: dict[str, list[dict[str, Any]]] = {
    "classification": [
        {"key": "logistic_regression"},
        {"key": "random_forest", "params": {"n_estimators": 10}},
    ],
    "regression": [{"key": "ridge"}, {"key": "random_forest", "params": {"n_estimators": 10}}],
}


class Transport(Http, Protocol):
    def upload(
        self, path: str, fields: dict[str, str], filename: str, data: bytes, field: str = "file"
    ) -> Reply: ...


class StageFailed(Exception):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"{stage}: {message}")
        self.stage, self.detail = stage, message


@dataclass
class Report:
    stages: list[dict[str, Any]] = field(default_factory=list)
    phases: list[Phase] = field(default_factory=list)
    failed: str | None = None

    @property
    def passed(self) -> bool:
        return self.failed is None and bool(self.stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed_stage": self.failed,
            "stages": self.stages,
            "phases": [
                {
                    "phase": p.name,
                    "traffic": p.traffic.to_dict(),
                    "monitoring": p.monitoring,
                    "alerts": p.alerts,
                }
                for p in self.phases
            ],
        }


def _ok(reply: Reply, stage: str, *codes: int) -> dict[str, Any]:
    if reply.status not in codes or reply.body is None:
        message = ""
        if reply.body and isinstance(reply.body.get("error"), dict):
            message = f": {reply.body['error'].get('code')}: {reply.body['error'].get('message')}"
        raise StageFailed(stage, f"unexpected answer, status {reply.status}{message}")
    return reply.body


def run_story(
    control: Transport,
    serving: Http,
    *,
    train_csv: tuple[str, bytes],
    dataset: str,
    target: str,
    task: str,
    model: str,
    normal: Stream,
    shifted: Stream,
    idle: Callable[[], None] = lambda: time.sleep(1.0),
    timeout: float = 1800.0,
    rows: int = 1000,
    seed: int = 0,
    clock: Callable[[], float] = time.monotonic,
) -> Report:
    report = Report()

    def stage(name: str, detail: str) -> None:
        report.stages.append({"stage": name, "ok": True, "detail": detail})

    try:
        # 1 upload
        filename, data = train_csv
        up = _ok(control.upload("/datasets", {"name": dataset}, filename, data), "upload", 200, 201)
        version = int(up["version"]["number"])
        stage("upload", f"{dataset} version {version}, {up['version']['rows']} rows")

        # 2-3 train, wait
        body = {
            "dataset": dataset,
            "version": version,
            "kind": "final",
            "target": target,
            "task": task,
            "algorithms": ALGORITHMS[task],
            "folds": 3,
            "confirm_final_test": True,
            "track": True,
        }
        job = _ok(control.request("POST", "/jobs", body), "train", 202)
        job_id = int(job["id"])
        deadline = clock() + timeout
        while True:
            idle()
            job = _ok(control.request("GET", f"/jobs/{job_id}"), "train", 200)
            if job["status"] in TERMINAL:
                break
            if clock() > deadline:
                raise StageFailed("train", f"job {job_id} did not finish within {timeout:.0f}s")
        if job["status"] != "succeeded":
            raise StageFailed("train", f"job {job_id} ended {job['status']}: {job.get('error')}")
        stage("train", f"job {job_id} succeeded, package {job['package_id']}")

        # 4 experiment tracking
        if job["tracking_state"] != "logged" or job.get("tracking_error"):
            raise StageFailed(
                "tracking",
                f"state {job['tracking_state']}, error {job.get('tracking_error')} "
                "(is AUTOML_JOBS_MLFLOW_DIR or the PostgreSQL backend set for the worker?)",
            )
        stage("tracking", f"logged as {job['tracking_job_key']}")

        # 5-6 registry
        registered = _ok(
            control.request("POST", "/models", {"name": model, "job_id": job_id}), "register", 201
        )
        number = int(registered["version"])
        _ok(
            control.request(
                "POST",
                f"/models/{model}/versions/{number}/promote",
                {"to": "staging", "reason": "end-to-end demo"},
            ),
            "promote",
            200,
        )
        stage("register_promote", f"{model} version {number} is in staging")

        # 7 deploy and serve
        deployed = _ok(
            control.request(
                "POST",
                "/deployments",
                {"model": model, "version": number, "reason": "end-to-end demo"},
            ),
            "deploy",
            201,
        )
        deployment_id = int(deployed["id"])
        for _ in range(60):
            info = serving.request("GET", "/model")
            if (
                info.status == 200
                and info.body
                and (info.body["deployment"]["deployment_id"] == deployment_id)
            ):
                break
            serving.request("POST", "/reload")
            idle()
        else:
            raise StageFailed("serve", "the serving app did not load the new deployment")
        stage("deploy_serve", f"deployment {deployment_id} is served")

        # 8-9 traffic, feedback, monitoring (reference sample is made inside the demo)
        report.phases = demo((normal, shifted), serving, control, rows=rows, seed=seed)
        first, second, third = report.phases
        stage(
            "traffic_feedback",
            "predictions "
            + "/".join(str(p.traffic.predictions) for p in report.phases)
            + ", labels "
            + "/".join(str(p.traffic.feedback_created) for p in report.phases),
        )
        if not all(p.traffic.clean for p in report.phases):
            raise StageFailed("traffic_feedback", "a traffic phase had failures")

        def unresolved(phase: Phase) -> set[str]:
            return {a["rule"] for a in phase.alerts if a["status"] != "resolved"}

        if first.monitoring["status"] != "ok":
            raise StageFailed("monitoring", f"normal phase status {first.monitoring['status']}")
        drift_rules = {"drifted_feature_share", "prediction_shift"}
        if unresolved(first):
            raise StageFailed(
                "false_alarm", f"normal traffic raised {sorted(unresolved(first))} (not expected)"
            )
        stage("normal_no_alert", "normal traffic raised no alert")
        if not unresolved(second) & drift_rules:
            raise StageFailed(
                "drift_alert",
                "the shifted copy raised no drift alert with the default thresholds "
                f"(drifted features {second.monitoring['drifted_features']}, share "
                f"{second.monitoring['drifted_feature_share']}, prediction statistic "
                f"{second.monitoring['prediction']['statistic']})",
            )
        stage("drift_alert", f"shifted traffic opened {sorted(unresolved(second))}")
        if unresolved(third) & drift_rules:
            raise StageFailed(
                "drift_resolved", f"drift alerts still open after recovery: {unresolved(third)}"
            )
        stage(
            "drift_resolved",
            "drift alerts auto-resolved after normal traffic; still open: "
            f"{sorted(unresolved(third)) or 'none'}",
        )
    except StageFailed as failure:
        report.stages.append({"stage": failure.stage, "ok": False, "detail": failure.detail})
        report.failed = failure.stage
    except SimulatorError as failure:
        report.stages.append({"stage": "simulation", "ok": False, "detail": str(failure)})
        report.failed = "simulation"
    return report
