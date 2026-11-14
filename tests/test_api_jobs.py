"""P4-4: the job endpoints, on a real PostgreSQL server (throwaway database, scratch stores)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from api_support import BASE, PORT, UI_ORIGIN, assert_envelope
from automl_ops.api_control.app import create_app
from automl_ops.api_control.schemas import JobDetail, JobSummary
from automl_ops.api_control.settings import ApiSettings
from automl_ops.dataset_store import FileDatasetStore
from automl_ops.db import migrate
from automl_ops.db.catalog import CatalogedDatasetStore
from automl_ops.jobs import JobRepository, worker_running
from automl_ops.worker import Worker, WorkerSettings
from conftest import JobEnv, toy_bytes

pytestmark = pytest.mark.postgres

FAST = [{"key": "logistic_regression"}, {"key": "random_forest", "params": {"n_estimators": 10}}]


def build(env: JobEnv, **over: Any) -> TestClient:
    settings = ApiSettings(port=PORT, store_root=env.store_root)
    store = CatalogedDatasetStore(FileDatasetStore(env.store_root, settings.limits), env.engine)
    kwargs: dict[str, Any] = {
        "jobs": JobRepository(env.engine),
        "worker_state": lambda: "running" if worker_running(env.engine) else "stopped",
        "tracking_check": lambda: None,
    }
    kwargs.update(over)
    app = create_app(settings, store, lambda: migrate.status(env.engine), **kwargs)
    return TestClient(app, base_url=BASE, raise_server_exceptions=False)


def body(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "dataset": "toy",
        "version": 1,
        "target": "y",
        "algorithms": FAST,
        "folds": 3,
    }
    base.update(over)
    return base


def run_worker(env: JobEnv) -> None:
    Worker(
        env.engine,
        env.store,
        env.models,
        settings=WorkerSettings(poll_seconds=0.05, heartbeat_seconds=0.2),
    ).run(once=True)


# -- the happy path ------------------------------------------------------------------------------


def test_submit_observe_and_read_the_result(job_env: JobEnv) -> None:
    c = build(job_env)
    submitted = c.post("/jobs", json=body())
    assert submitted.status_code == 202, submitted.text
    job = JobDetail(**submitted.json())
    assert (job.status, job.phase, job.kind, job.dataset) == ("queued", "queued", "train", "toy@v1")
    assert job.track is False and job.tracking_state == "off" and job.result is None
    assert job.config.settings.folds == 3 and [a.key for a in job.config.algorithms] == [
        "logistic_regression",
        "random_forest",
    ]
    assert c.get("/health").json()["worker"] == "stopped"

    run_worker(job_env)
    done = JobDetail(**c.get(f"/jobs/{job.id}").json())
    assert done.status == "succeeded" and done.result is not None and done.error is None
    assert done.result.test_result is None and done.result.selection is not None
    assert {r["algorithm"] for r in done.result.training["results"]} == {
        "logistic_regression",
        "random_forest",
    }
    first = c.get(f"/jobs/{job.id}?after_seq=0").json()
    later = c.get(f"/jobs/{job.id}?after_seq={first['last_seq'] - 1}").json()
    assert len(first["events"]) == first["last_seq"] and len(later["events"]) == 1
    assert later["events"][0]["kind"] == "state"
    listed = c.get("/jobs").json()
    assert [j["id"] for j in listed["jobs"]] == [job.id] and "result" not in listed["jobs"][0]
    JobSummary(**listed["jobs"][0])


def test_a_final_job_over_the_api_seals_a_package_with_one_test_result(job_env: JobEnv) -> None:
    c = build(job_env)
    ok = c.post("/jobs", json=body(kind="final", confirm_final_test=True))
    assert ok.status_code == 202 and ok.json()["kind"] == "final"
    assert ok.json()["confirm_final_test"] is True
    run_worker(job_env)
    done = c.get(f"/jobs/{ok.json()['id']}").json()
    assert done["status"] == "succeeded" and done["package_id"]
    assert done["result"]["test_result"]["evaluation"]["metrics"]
    assert "limitation" in done["result"]["test_result"]  # the reuse caveat is carried


def test_regression_jobs_and_defaults(job_env: JobEnv) -> None:
    c = build(job_env)
    job = c.post("/jobs", json={"dataset": "toyreg", "version": 1, "target": "y"}).json()
    assert job["config"]["task"] == "regression" and len(job["config"]["algorithms"]) >= 2
    assert job["config"]["settings"] == {"test_fraction": 0.2, "seed": 42, "folds": 5}


def test_listing_filters_and_paging(job_env: JobEnv) -> None:
    c = build(job_env)
    ids = [c.post("/jobs", json=body()).json()["id"] for _ in range(3)]
    c.post(f"/jobs/{ids[0]}/cancel")
    assert [j["id"] for j in c.get("/jobs?status=cancelled").json()["jobs"]] == [ids[0]]
    assert [j["id"] for j in c.get("/jobs?status=queued").json()["jobs"]] == [ids[2], ids[1]]
    page = c.get("/jobs?limit=2").json()
    assert page["next_before_id"] == ids[1]
    rest = c.get(f"/jobs?limit=2&before_id={page['next_before_id']}").json()
    assert [j["id"] for j in rest["jobs"]] == [ids[0]] and rest["next_before_id"] is None
    assert c.get("/jobs?dataset=toy&kind=train").json()["jobs"]
    assert c.get("/jobs?dataset=other").json()["jobs"] == []


def test_the_algorithm_catalogue(job_env: JobEnv) -> None:
    c = build(job_env)
    everything = c.get("/algorithms").json()["algorithms"]
    only = c.get("/algorithms?task=regression").json()["algorithms"]
    assert {a["task"] for a in only} == {"regression"} and len(everything) > len(only)
    forest = next(a for a in only if a["key"] == "random_forest")
    assert any(p["name"] == "n_estimators" for p in forest["params"])
    assert_envelope(c.get("/algorithms?task=weather"), 422, "validation_error")


# -- rejections: every one has the error envelope ------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "status", "code"),
    [
        ({"dataset": "nope"}, 404, "version_not_found"),
        ({"version": 9}, 404, "version_not_found"),
        ({"dataset": "Bad Name"}, 422, "invalid_name"),
        ({"target": "missing"}, 422, "invalid_task_config"),
        ({"task": "regression"}, 422, "task_not_trainable"),  # y is a text label
        ({"algorithms": [{"key": "nope"}]}, 422, "invalid_algorithm"),
        (
            {"algorithms": [{"key": "random_forest", "params": {"depth": 2}}]},
            422,
            "invalid_algorithm",
        ),
        ({"algorithms": [{"key": "ridge"}]}, 422, "invalid_algorithm"),  # a regression algorithm
        ({"algorithms": FAST[:1] * 2}, 422, "duplicate_algorithm"),
        ({"algorithms": []}, 422, "invalid_algorithm"),
        ({"folds": 50}, 422, "invalid_settings"),
        ({"test_fraction": 0.9}, 422, "invalid_settings"),
        ({"selection_metric": "rmse"}, 422, "invalid_metric"),
        ({"kind": "final"}, 422, "confirmation_required"),
        ({"kind": "other"}, 422, "validation_error"),
        ({"surprise": 1}, 422, "validation_error"),
        ({"folds": "five"}, 422, "validation_error"),
        ({"version": 0}, 422, "validation_error"),
        ({"track": "maybe"}, 422, "validation_error"),
    ],
)
def test_submit_rejections(
    job_env: JobEnv, changes: dict[str, Any], status: int, code: str
) -> None:
    c = build(job_env)
    response = c.post("/jobs", json=body(**changes))
    assert_envelope(response, status, code)
    assert JobRepository(job_env.engine).counts() == {}  # nothing was queued


def test_a_task_with_errors_lists_them(job_env: JobEnv) -> None:
    c = build(job_env)
    error = assert_envelope(
        c.post("/jobs", json=body(task="regression")), 422, "task_not_trainable"
    )
    findings = error["details"]["findings"]
    assert findings and all(f["severity"] == "error" for f in findings)


def test_malformed_and_missing_bodies(job_env: JobEnv) -> None:
    c = build(job_env)
    assert_envelope(
        c.post("/jobs", content=b"{not json", headers={"content-type": "application/json"}), 422
    )
    assert_envelope(c.post("/jobs", json={}), 422, "validation_error")
    assert_envelope(c.post("/jobs", data={"dataset": "toy"}), 415, "unsupported_media_type")
    assert_envelope(
        c.post(
            "/jobs", content=b"x" * (1024 * 1024 + 1), headers={"content-type": "application/json"}
        ),
        413,
    )


def test_the_queue_is_capped_and_a_missing_catalog_row_is_a_conflict(job_env: JobEnv) -> None:
    c = build(job_env)
    for _ in range(10):
        assert c.post("/jobs", json=body()).status_code == 202
    assert_envelope(c.post("/jobs", json=body()), 409, "queue_full")
    FileDatasetStore(job_env.store_root).ingest("late", toy_bytes(), source_filename="late.csv")
    with job_env.engine.begin() as connection:
        connection.execute(text("DELETE FROM training_jobs"))
    assert_envelope(c.post("/jobs", json=body(dataset="late")), 409, "dataset_not_catalogued")


def test_tracking_that_is_not_available_is_refused_before_queueing(job_env: JobEnv) -> None:
    c = build(job_env, tracking_check=lambda: "tracking is not configured for jobs")
    assert_envelope(c.post("/jobs", json=body(track=True)), 422, "tracking_not_available")
    accepted = c.post("/jobs", json=body())  # tracking is opt-in: off needs no setup
    assert accepted.status_code == 202 and accepted.json()["track"] is False
    assert JobRepository(job_env.engine).counts() == {"queued": 1}


def test_get_and_cancel_unknown_or_malformed_ids(job_env: JobEnv) -> None:
    c = build(job_env)
    assert_envelope(c.get("/jobs/99"), 404, "job_not_found")
    assert_envelope(c.post("/jobs/99/cancel"), 404, "job_not_found")
    for bad in ("0", "-1", "abc", "1.5", "99999999999"):
        assert_envelope(c.get(f"/jobs/{bad}"), 422, "validation_error")
        assert_envelope(c.post(f"/jobs/{bad}/cancel"), 422, "validation_error")
    assert_envelope(c.get("/jobs?limit=0"), 422, "validation_error")
    assert_envelope(c.get("/jobs?status=sleeping"), 422, "validation_error")
    assert_envelope(c.get("/jobs/1?after_seq=-1"), 422, "validation_error")


def test_methods_and_the_jobs_unavailable_case(job_env: JobEnv) -> None:
    c = build(job_env)
    jobs = assert_envelope(c.delete("/jobs"), 405, "method_not_allowed")
    assert jobs["code"] == "method_not_allowed"
    assert c.delete("/jobs").headers["allow"] == "GET, POST"
    assert c.delete("/jobs/1").headers["allow"] == "GET"
    assert c.get("/jobs/1/cancel").headers["allow"] == "POST"
    offline = build(job_env, jobs=None)
    assert_envelope(offline.get("/jobs"), 503, "jobs_unavailable")
    assert_envelope(offline.post("/jobs", json=body()), 503, "jobs_unavailable")
    assert offline.get("/health").json()["status"] == "ok"


# -- cancel ---------------------------------------------------------------------------------------


def test_cancel_by_state(job_env: JobEnv) -> None:
    c = build(job_env)
    repo = JobRepository(job_env.engine)
    queued = c.post("/jobs", json=body()).json()["id"]
    cancelled = c.post(f"/jobs/{queued}/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["outcome"] == "cancelled"
    assert cancelled.json()["job"]["status"] == "cancelled"
    assert_envelope(c.post(f"/jobs/{queued}/cancel"), 409, "job_finished")

    running = c.post("/jobs", json=body(kind="final", confirm_final_test=True)).json()["id"]
    assert repo.claim("w-api-test") is not None
    repo.set_phase(running, "w-api-test", "training")
    accepted = c.post(f"/jobs/{running}/cancel")
    assert accepted.status_code == 202 and accepted.json()["outcome"] == "requested"
    assert accepted.json()["job"]["cancel_requested"] is True
    repo.finish(running, "w-api-test", "cancelled")

    final = c.post("/jobs", json=body(kind="final", confirm_final_test=True)).json()["id"]
    assert repo.claim("w-api-test") is not None
    assert repo.enter_uncancellable(final, "w-api-test", "finalizing")
    error = assert_envelope(c.post(f"/jobs/{final}/cancel"), 409, "cannot_cancel_now")
    assert "final test" in error["message"]
    after = c.get(f"/jobs/{final}").json()
    assert after["status"] == "running" and after["phase"] == "finalizing"
    assert after["cancel_requested"] is False


def test_a_running_job_with_an_old_heartbeat_is_shown_as_stale(job_env: JobEnv) -> None:
    c = build(job_env)
    repo = JobRepository(job_env.engine)
    job_id = c.post("/jobs", json=body()).json()["id"]
    repo.claim("w-api-test")
    assert c.get(f"/jobs/{job_id}").json()["stale"] is False
    with job_env.engine.begin() as connection:
        connection.execute(
            text("UPDATE training_jobs SET heartbeat_at = now() - interval '120 seconds'")
        )
    assert c.get(f"/jobs/{job_id}").json()["stale"] is True


# -- security of the new routes -----------------------------------------------------------------


def test_the_security_rules_apply_to_the_job_routes(job_env: JobEnv) -> None:
    c = build(job_env)
    submit = {"json": body()}
    assert_envelope(c.post("/jobs", headers={"Origin": "http://evil.example"}, **submit), 403)
    assert_envelope(c.post("/jobs", headers={"Origin": "null"}, **submit), 403)
    assert_envelope(c.post("/jobs/1/cancel", headers={"Origin": "http://evil.example"}), 403)
    assert_envelope(c.get("/jobs", headers={"Host": "evil.example"}), 400)
    assert c.post("/jobs", headers={"Origin": UI_ORIGIN}, **submit).status_code == 202
    assert (
        c.get("/jobs", headers={"Origin": UI_ORIGIN}).headers["access-control-allow-origin"]
        == UI_ORIGIN
    )
    preflight = c.options(
        "/jobs",
        headers={"Origin": UI_ORIGIN, "Access-Control-Request-Method": "POST"},
    )
    assert preflight.status_code == 204
    for text_ in (c.get("/jobs").text, c.get("/jobs/1").text):
        assert "/Users/" not in text_ and "Traceback" not in text_
