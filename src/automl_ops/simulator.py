"""Traffic simulator (P4-10, BUILD_PLAN Step 25): demo traffic from one person, always labelled
simulated.

It sends rows of the separate **monitoring stream** (or its synthetic shifted copy) to the serving
app's `POST /predict` with `simulated: true`, and can send some true labels back through
`POST /feedback`. It talks to the two services over HTTP only and writes nothing to the database
itself, so everything it does is visible through the same APIs as real traffic and is flagged
simulated there.

Safety rules (each is tested):

- Only the **known demo monitoring files** are accepted: a file name of the form
  `<name>_monitor_stream.csv` / `<name>_monitor_shifted.csv` whose SHA-256 equals the one in the
  `manifest.json` next to it. There is no override.
- It refuses a file whose SHA-256 (or original-bytes SHA-256) equals any registered dataset version,
  checked through the control-plane API, so training data can never be replayed as traffic.
- Every request is marked `simulated: true`; the transports only ever reach `127.0.0.1`.
- Row choice and the labelled subset are functions of the seed alone (same seed, same traffic).
- A serving app that is down or failing makes the run stop with an explicit result; it never hangs
  and never reports success it did not have.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

HOST = "127.0.0.1"
MAX_BATCH = 100
MAX_ROWS = 50_000
MAX_CONSECUTIVE_FAILURES = 3
KINDS = {"monitor_stream": "normal", "monitor_shifted": "shifted"}


class SimulatorError(Exception):
    """A refusal or a setup problem (the run never started)."""

    def __init__(self, message: str, code: str = "simulator_error") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Reply:
    status: int  # 0: no answer (connection failed or timed out)
    body: dict[str, Any] | None


class Http(Protocol):
    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Reply: ...


class LocalHttp:
    """Plain HTTP to a service on 127.0.0.1 (the host is fixed; only the port is chosen)."""

    def __init__(self, port: int, timeout: float = 30.0) -> None:
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise SimulatorError("the port must be a whole number from 1 to 65535", "bad_port")
        self.port = port
        self.timeout = timeout

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Reply:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            f"http://{HOST}:{self.port}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                return Reply(response.status, _json(response.read(4_000_000)))
        except urllib.error.HTTPError as error:
            return Reply(error.code, _json(error.read(4_000_000)))
        except (OSError, urllib.error.URLError):
            return Reply(0, None)

    def upload(
        self, path: str, fields: dict[str, str], filename: str, data: bytes, field: str = "file"
    ) -> Reply:
        """A multipart form with one file (for the dataset upload)."""
        boundary = "----automl" + hashlib.sha256(data).hexdigest()[:24]
        parts = [
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
            for k, v in fields.items()
        ]
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; '
            f'filename="{filename}"\r\nContent-Type: text/csv\r\n\r\n'.encode()
            + data
            + f"\r\n--{boundary}--\r\n".encode()
        )
        request = urllib.request.Request(
            f"http://{HOST}:{self.port}{path}",
            data=b"".join(parts),
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                return Reply(response.status, _json(response.read(4_000_000)))
        except urllib.error.HTTPError as error:
            return Reply(error.code, _json(error.read(4_000_000)))
        except (OSError, urllib.error.URLError):
            return Reply(0, None)


def _json(raw: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


# --------------------------------------------------------------------------------------------
# The stream file
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Stream:
    path: Path
    dataset: str
    kind: str  # normal | shifted
    sha256: str
    frame: pd.DataFrame  # all cells as text, empty cell = missing


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stream(path: Path) -> Stream:
    """The known demo monitoring file at `path`, or a `SimulatorError`. There is no override."""
    path = Path(path)
    stem = path.name.removesuffix(".csv")
    kind = next((k for k in KINDS if stem.endswith("_" + k)), None)
    if not path.name.endswith(".csv") or kind is None or not path.is_file():
        raise SimulatorError(
            "only the demo monitoring files (<name>_monitor_stream.csv or "
            "<name>_monitor_shifted.csv from `make demo-data`) can be replayed",
            "unknown_file",
        )
    dataset = stem.removesuffix("_" + kind)
    manifest_path = path.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest["datasets"][dataset]["files"][
            "monitoring_" + kind.removeprefix("monitor_")
        ]["sha256"]
    except (OSError, ValueError, KeyError, TypeError):
        raise SimulatorError(
            "the demo manifest next to the file is missing or does not describe it", "unknown_file"
        ) from None
    digest = _sha256(path)
    if digest != expected:
        raise SimulatorError(
            "the file does not match the hash recorded in the demo manifest", "unknown_file"
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return Stream(path, dataset, KINDS[kind], digest, frame)


def check_not_training_data(control: Http, stream: Stream) -> None:
    """Refuse if the file is, byte for byte, a registered dataset version."""
    reply = control.request("GET", "/datasets")
    if reply.status != 200 or reply.body is None:
        raise SimulatorError(
            "the control plane could not list datasets, so the file could not be checked "
            f"against the training data (status {reply.status})",
            "control_unavailable",
        )
    for dataset in reply.body.get("datasets", []):
        for version in dataset.get("versions", []):
            if stream.sha256 in (version.get("sha256"), version.get("original_sha256")):
                raise SimulatorError(
                    f"the file is identical to registered dataset version {version.get('ref')}; "
                    "training data is never replayed as traffic",
                    "training_data",
                )


# --------------------------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------------------------


@dataclass
class RunResult:
    stream: str
    kind: str
    seed: int
    data_sha256: str
    deployment_id: int | None = None
    model: str | None = None
    version: int | None = None
    rows_selected: int = 0
    rows_sent: int = 0
    predictions: int = 0
    not_logged: int = 0
    feedback_planned: int = 0
    feedback_sent: int = 0
    feedback_created: int = 0
    feedback_failed: int = 0
    batches_failed: int = 0
    aborted: bool = False
    reason: str | None = None
    first_prediction_id: str | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def clean(self) -> bool:
        return (
            not self.aborted
            and self.batches_failed == 0
            and self.feedback_failed == 0
            and self.rows_sent == self.rows_selected
        )


def _cell(value: Any) -> str | None:
    return None if value is None or value == "" else str(value)


def select_rows(size: int, count: int, fraction: float, seed: int) -> tuple[list[int], set[int]]:
    """Row positions in send order, and the subset (by position) that will get a label. Functions
    of `seed` only."""
    if not 1 <= count <= min(size, MAX_ROWS):
        raise SimulatorError(f"rows must be from 1 to {min(size, MAX_ROWS)}", "bad_arguments")
    if not 0.0 <= fraction <= 1.0:
        raise SimulatorError("the feedback fraction must be from 0 to 1", "bad_arguments")
    order = [int(i) for i in np.random.default_rng(seed).permutation(size)[:count]]
    chosen = int(round(count * fraction))
    pick = np.random.default_rng([seed, 1]).permutation(count)[:chosen]
    return order, {order[int(i)] for i in pick}


def run(
    stream: Stream,
    serving: Http,
    control: Http,
    *,
    rows: int = 200,
    seed: int = 0,
    batch_size: int = 50,
    rate: float | None = None,
    feedback_fraction: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> RunResult:
    """Send `rows` rows of the stream as simulated traffic, then the planned labels."""
    if not 1 <= batch_size <= MAX_BATCH:
        raise SimulatorError(f"the batch size must be from 1 to {MAX_BATCH}", "bad_arguments")
    if rate is not None and (not math.isfinite(rate) or rate <= 0):
        raise SimulatorError(
            "the rate must be a positive number of rows per second", "bad_arguments"
        )
    order, labelled = select_rows(len(stream.frame), rows, feedback_fraction, seed)
    check_not_training_data(control, stream)

    result = RunResult(stream.dataset, stream.kind, seed, stream.sha256, rows_selected=len(order))
    info = serving.request("GET", "/model")
    if info.status != 200 or info.body is None:
        result.aborted, result.reason = True, f"no model is being served (status {info.status})"
        return result
    deployment = info.body["deployment"]
    result.deployment_id = int(deployment["deployment_id"])
    result.model, result.version = str(deployment["model"]), int(deployment["version"])
    task, target = deployment["task"], str(info.body["target"])
    columns = [str(c["name"]) for c in info.body["input_columns"]]
    missing = [c for c in [*columns, target] if c not in stream.frame.columns]
    if missing:
        raise SimulatorError(
            f"the deployed model does not fit this stream (columns not in the file: {missing})",
            "model_mismatch",
        )

    records: list[dict[str, Any]] = [
        {str(k): v for k, v in r.items()} for r in stream.frame.to_dict("records")
    ]
    failures = 0
    for start in range(0, len(order), batch_size):
        positions = order[start : start + batch_size]
        batch = [{c: _cell(records[p][c]) for c in columns} for p in positions]
        reply = serving.request("POST", "/predict", {"rows": batch, "simulated": True})
        if reply.status != 200 or reply.body is None:
            result.batches_failed += 1
            failures += 1
            if failures >= MAX_CONSECUTIVE_FAILURES:
                result.aborted = True
                result.reason = (
                    f"stopped after {failures} failed batches in a row (last status {reply.status})"
                )
                return result
            continue
        failures = 0
        items = reply.body["predictions"]
        result.rows_sent += len(positions)
        result.predictions += len(items)
        if not reply.body.get("logged", False):
            result.not_logged += len(items)
        if result.first_prediction_id is None and items:
            result.first_prediction_id = items[0]["prediction_id"]
        for position, item in zip(positions, items, strict=True):
            if position not in labelled:
                continue
            result.feedback_planned += 1
            truth = str(records[position][target])
            true_value: str | float = str(truth) if task == "classification" else float(truth)
            sent = serving.request(
                "POST",
                "/feedback",
                {
                    "prediction_id": item["prediction_id"],
                    "true_value": true_value,
                    "deployment_id": result.deployment_id,
                },
            )
            if sent.status == 200 and sent.body is not None:
                result.feedback_sent += 1
                result.feedback_created += bool(sent.body.get("created"))
            else:
                result.feedback_failed += 1
        if rate is not None and start + batch_size < len(order):
            sleep(len(positions) / rate)
    return result


# --------------------------------------------------------------------------------------------
# The drift demonstration
# --------------------------------------------------------------------------------------------


def _verdict(report: dict[str, Any]) -> dict[str, Any]:
    """The numbers that decided one monitoring run, flattened for a printed report."""
    summary = report.get("summary") or {}
    drift = summary.get("dataset_drift") or {}
    prediction = summary.get("prediction") or {}
    comparison = summary.get("performance") or {}
    features = summary.get("features") or []
    return {
        "report_id": report["id"],
        "status": report["status"],
        "window_rows": report["window_rows"],
        "drifted_feature_share": drift.get("share"),
        "drifted_features": [f["name"] for f in features if f.get("drifted")],
        "feature_statistics": {
            f["name"]: {"method": f["method"], "statistic": f["statistic"], "p": f["p_value"]}
            for f in features
        },
        "prediction": {
            k: prediction.get(k) for k in ("method", "statistic", "p_value", "threshold", "drifted")
        },
        "performance": comparison,
        "error": report.get("error"),
    }


@dataclass
class Phase:
    name: str
    traffic: RunResult
    monitoring: dict[str, Any]
    alerts: list[dict[str, Any]]


def demo(
    streams: tuple[Stream, Stream],
    serving: Http,
    control: Http,
    *,
    rows: int = 1000,
    seed: int = 0,
    batch_size: int = 100,
    feedback_fraction: float = 0.3,
) -> list[Phase]:
    """normal -> shifted -> normal again, each followed by one monitoring run on the simulated
    scope with the **default thresholds** (nothing is tuned). `rows` defaults to the monitoring
    window (1,000), so each run's window holds only that phase's traffic. Returns the observed
    result of every phase; it does not assert that an alert appears."""
    normal, shifted = streams
    if (normal.kind, shifted.kind) != ("normal", "shifted") or normal.dataset != shifted.dataset:
        raise SimulatorError(
            "the demo needs a dataset's stream and its shifted copy", "bad_arguments"
        )
    phases: list[Phase] = []
    ensured = False
    for index, (name, stream) in enumerate(
        (("normal", normal), ("shifted", shifted), ("recovery", normal))
    ):
        traffic = run(
            stream,
            serving,
            control,
            rows=rows,
            seed=seed + index,
            batch_size=batch_size,
            feedback_fraction=feedback_fraction,
        )
        if traffic.aborted or traffic.deployment_id is None:
            raise SimulatorError(f"the {name} phase could not run: {traffic.reason}", "run_failed")
        if not ensured:
            reference = control.request(
                "POST", f"/models/{traffic.model}/versions/{traffic.version}/reference"
            )
            if reference.status not in (200, 201):
                raise SimulatorError(
                    f"no reference sample could be made (status {reference.status})",
                    "no_reference",
                )
            ensured = True
        report = control.request(
            "POST", f"/deployments/{traffic.deployment_id}/monitor", {"scope": "simulated"}
        )
        if report.status not in (200, 201) or report.body is None:
            raise SimulatorError(
                f"monitoring the {name} phase failed (status {report.status})", "monitor_failed"
            )
        listed = control.request(
            "GET", f"/alerts?deployment_id={traffic.deployment_id}&scope=simulated&limit=100"
        )
        alerts = [
            {k: a[k] for k in ("id", "rule", "severity", "status", "occurrences")}
            for a in (listed.body or {}).get("alerts", [])
        ]
        phases.append(Phase(name, traffic, _verdict(report.body), alerts))
    return phases
