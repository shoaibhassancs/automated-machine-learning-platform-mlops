"""P4-7: validating prediction input against a model's feature schema (no database needed)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from automl_ops.api_control.settings import ApiSettings, SettingsError
from automl_ops.api_serving.inputs import MAX_PROBLEMS, Column, InvalidInput, prepare_rows
from automl_ops.api_serving.settings import ServingSettings

COLUMNS = [Column("x", "numeric"), Column("z", "numeric"), Column("color", "categorical")]


def ok(rows: list[dict[str, Any]]) -> list[list[Any]]:
    frame = prepare_rows(COLUMNS, rows)
    assert list(frame.columns) == ["x", "z", "color"]
    return [
        [None if v is None or v is __import__("pandas").NA else v for v in r]
        for r in frame.to_numpy()
    ]


def refused(rows: list[dict[str, Any]]) -> InvalidInput:
    with pytest.raises(InvalidInput) as raised:
        prepare_rows(COLUMNS, rows)
    return raised.value


def test_a_valid_row_becomes_text_in_the_models_column_order() -> None:
    assert ok([{"color": "red", "z": 2, "x": 1.5}]) == [["1.5", "2", "red"]]


def test_numbers_can_be_json_numbers_or_clean_text() -> None:
    assert ok([{"x": "1e-05", "z": -3, "color": "a"}]) == [["1e-05", "-3", "a"]]
    assert ok([{"x": 1e-5, "z": 10**20, "color": "a"}])[0][0] == "1e-05"


def test_null_and_empty_string_are_missing_values_not_errors() -> None:
    assert ok([{"x": None, "z": "", "color": None}]) == [[None, None, None]]


def test_missing_and_unknown_columns_are_both_refused_with_the_names() -> None:
    error = refused([{"x": 1, "z": 2}, {"x": 1, "z": 2, "color": "a", "bogus": 1}])
    messages = [(p.row, p.message) for p in error.problems]
    assert (0, "missing column(s): ['color']") in messages
    assert (1, "unknown column(s): ['bogus']") in messages


@pytest.mark.parametrize(
    "value",
    ["abc", "1_000", "nan", "inf", "0x10", " 1", "1,5", True, [1], {"a": 1}, math.inf, math.nan],
)
def test_a_bad_numeric_value_is_refused_never_turned_into_missing(value: Any) -> None:
    error = refused([{"x": value, "z": 1, "color": "a"}])
    assert [(p.row, p.column) for p in error.problems] == [(0, "x")]


def test_a_number_in_a_categorical_column_is_refused() -> None:
    error = refused([{"x": 1, "z": 1, "color": 7}])
    assert error.problems[0].column == "color" and "strings" in error.problems[0].message


def test_an_unseen_category_is_not_an_error() -> None:
    assert ok([{"x": 1, "z": 1, "color": "never-seen"}])[0][2] == "never-seen"


def test_leading_zero_text_keeps_its_text_in_a_categorical_column() -> None:
    assert ok([{"x": 1, "z": 1, "color": "007"}])[0][2] == "007"


def test_problems_are_capped_and_long_values_are_shortened() -> None:
    rows = [{"x": "q" * 500, "z": 1, "color": "a"} for _ in range(MAX_PROBLEMS + 5)]
    error = refused(rows)
    assert len(error.problems) == MAX_PROBLEMS and error.truncated
    assert all(len(p.message) < 120 for p in error.problems)


def test_serving_settings_validate_and_never_share_the_apis_port() -> None:
    assert ServingSettings.from_environment({}).port == 8200
    assert ServingSettings.from_environment({"AUTOML_SERVING_PORT": "8300"}).port == 8300
    for bad in ({"AUTOML_SERVING_PORT": "abc"}, {"AUTOML_SERVING_PORT": "80"}):
        with pytest.raises(SettingsError):
            ServingSettings.from_environment(bad)
    with pytest.raises(SettingsError, match="differ"):
        ApiSettings.from_environment({"AUTOML_API_PORT": "8200"})
    assert ApiSettings.from_environment({}).serving_port == 8200
    http = ServingSettings(port=8200).http_rules()
    assert "127.0.0.1:8200" in http.allowed_hosts
