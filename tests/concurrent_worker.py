"""Worker for the concurrent-finalization test: one process making the one test attempt.

Usage: python concurrent_worker.py <state.pickle> <package_dir> <start_epoch> <out.json>
It waits until `start_epoch` so two workers start the attempt at nearly the same moment, counts
how many times the test rows were read, and writes its outcome to `out.json`.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import automl_ops.finalize as finalize_module
from automl_ops.finalize import run_final_test
from automl_ops.package import seal_package


def main(state_file: str, package: str, start: str, out: str) -> None:
    with open(state_file, "rb") as handle:
        frame, config, training, frozen, dataset = pickle.load(handle)
    reads = 0
    original = finalize_module._read_test_part

    def counting(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal reads
        reads += 1
        return original(*args, **kwargs)

    finalize_module._read_test_part = counting
    while time.time() < float(start):
        pass
    outcome: dict[str, object]
    try:
        run_final_test(Path(package), frame, config, training, frozen, dataset)
        seal_package(Path(package))
        outcome = {"outcome": "won"}
    except Exception as error:
        outcome = {"outcome": "refused", "error": type(error).__name__, "message": str(error)}
    outcome["test_reads"] = reads
    Path(out).write_text(json.dumps(outcome), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:5])
