"""Smoke tests: the package imports and runs on the intended Python."""

import sys

import automl_ops


def test_package_imports_and_has_version() -> None:
    assert isinstance(automl_ops.__version__, str)
    assert automl_ops.__version__


def test_runs_on_python_3_12() -> None:
    assert sys.version_info[:2] == (3, 12)
