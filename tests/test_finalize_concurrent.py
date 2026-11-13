"""Two real processes make the one test attempt on the same package at nearly the same time."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import pytest

import automl_ops.finalize as finalize_module
from automl_ops.finalize import finalize
from automl_ops.package import list_packages, load_package, package_state, verify_package
from automl_ops.selection import freeze_selection, select_model
from automl_ops.train import run_training
from conftest import CLF, FAST, SETTINGS, toy_bytes, toy_dataset

WORKER = Path(__file__).parent / "concurrent_worker.py"
ROUNDS = 5


def test_two_processes_racing_for_one_package_produce_exactly_one_test_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, info = toy_dataset(toy_bytes())
    training = run_training(frame, CLF, FAST, SETTINGS)
    frozen = freeze_selection(training, select_model(training))
    state = tmp_path / "state.pickle"
    state.write_bytes(pickle.dumps((frame, CLF, training, frozen, info)))
    models = tmp_path / "models"

    for round_number in range(ROUNDS):
        # Build an *open* package: everything finalize does except the test and the seal.
        with monkeypatch.context() as patch:
            patch.setattr(finalize_module, "run_final_test", lambda *a, **k: {})
            patch.setattr(finalize_module, "seal_package", lambda *a, **k: {})
            built = finalize(frame, CLF, training, frozen, info, models, confirm_final_test=True)
        package = built.path
        assert package_state(package) == "open"

        start = time.time() + 6.0  # both interpreters finish importing before this moment
        outputs = [tmp_path / f"out-{round_number}-{i}.json" for i in range(2)]
        workers = [
            subprocess.Popen(
                [sys.executable, str(WORKER), str(state), str(package), str(start), str(out)]
            )
            for out in outputs
        ]
        assert [worker.wait(timeout=120) for worker in workers] == [0, 0]
        results = [json.loads(out.read_text(encoding="utf-8")) for out in outputs]

        won = [r for r in results if r["outcome"] == "won"]
        refused = [r for r in results if r["outcome"] == "refused"]
        assert len(won) == 1 and len(refused) == 1, results
        assert refused[0]["error"] == "PackageSpentError"
        # Only the winner ever read the test rows: exactly one test evaluation happened.
        assert won[0]["test_reads"] == 1 and refused[0]["test_reads"] == 0

        # The package ends consistent: one attempt, one result, sealed, verifiable, loadable.
        assert package_state(package) == "sealed"
        stored = json.loads((package / "test_result.json").read_text(encoding="utf-8"))
        assert stored["package_id"] == package.name and stored["evaluation"]
        check = verify_package(models, package.name)
        assert check.ok, check
        load_package(models, package.name)

    assert len(list_packages(models)) == ROUNDS
