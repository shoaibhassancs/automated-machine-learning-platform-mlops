"""Preprocessing pipeline tests with small invented data.

Each learned quantity (median, scaling, category list) is compared with a value computed by hand
or with NumPy from the rows the pipeline was fitted on, and the leak checks use values that exist
only in the rows that must NOT influence fitting.
"""

from __future__ import annotations

import csv
import io
import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from automl_ops.pipeline import (
    ColumnPlan,
    MalformedValueError,
    PipelineError,
    build_column_plan,
    build_pipeline,
    build_preprocessor,
    text_to_category,
    text_to_number,
)
from automl_ops.profile import profile_frame, read_text_frame
from automl_ops.task import TaskConfig


def frame_of(columns: dict[str, list[str]]) -> pd.DataFrame:
    names = list(columns)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(names)
    for i in range(len(columns[names[0]])):
        writer.writerow([columns[name][i] for name in names])
    return read_text_frame(out.getvalue().encode("utf-8"))


def plan_of(
    columns: dict[str, list[str]], target: str = "y", excluded: tuple[str, ...] = ()
) -> Any:
    frame = frame_of(columns)
    return build_column_plan(profile_frame(frame), TaskConfig(target, "classification", excluded))


def training_columns(n: int = 60) -> dict[str, list[str]]:
    return {
        "age": [str(20 + i % 30) for i in range(n)],
        "score": [f"{(i % 9) / 3:.3f}" for i in range(n)],
        "color": [("red", "green", "blue")[i % 3] for i in range(n)],
        "y": [("a", "b")[i % 2] for i in range(n)],
    }


# --- the column plan ----------------------------------------------------------------------


def test_each_kind_gets_the_documented_treatment_and_every_exclusion_is_reported() -> None:
    n = 80
    columns = {
        "int": [str(i % 7) for i in range(n)],
        "float": [f"{i % 5}.5" for i in range(n)],
        "cat": [("p", "q")[i % 2] for i in range(n)],
        "text": [f"free text {i}" for i in range(n)],  # > 50 distinct
        "code": [("007", "012", "5", "9")[i % 4] for i in range(n)],
        "mixed": [("1", "2", "3", "x")[i % 4] for i in range(n)],
        "empty": [""] * n,
        "y": [("a", "b")[i % 2] for i in range(n)],
    }
    plan = plan_of(columns)
    assert plan.numeric == ("int", "float") and plan.categorical == ("cat",)
    assert {d.name: d.kind for d in plan.dropped} == {
        "text": "text",
        "code": "code",
        "mixed": "mixed",
        "empty": "empty",
    }
    assert all(d.reason for d in plan.dropped)  # each exclusion has a reason
    assert "identifiers" in next(d for d in plan.dropped if d.name == "code").reason
    assert plan.to_dict()["dropped"][0].keys() == {"name", "kind", "reason"}


def test_target_and_explicitly_excluded_columns_are_not_inputs_and_not_reported_as_dropped() -> (
    None
):
    plan = plan_of(training_columns(), excluded=("score",))
    assert plan.used == ("age", "color") and plan.dropped == ()


def test_no_usable_columns_is_an_error_that_names_the_reasons() -> None:
    columns = {"free": [f"t{i}" for i in range(80)], "y": [("a", "b")[i % 2] for i in range(80)]}
    with pytest.raises(PipelineError, match=r"no usable input columns.*'free'"):
        plan_of(columns)


# --- strict number parsing ----------------------------------------------------------------


def test_empty_cells_become_missing_and_numbers_are_parsed() -> None:
    parsed = text_to_number(frame_of({"a": ["1", "", "-2.5", "1e2", ".5"]}))
    assert np.isnan(parsed[1, 0])
    assert parsed[[0, 2, 3, 4], 0].tolist() == [1.0, -2.5, 100.0, 0.5]


@pytest.mark.parametrize("bad", ["abc", "1_000", "nan", "inf", "1e999", " 5", "1,5", "0x1F"])
def test_a_malformed_value_is_an_error_not_a_silent_missing_value(bad: str) -> None:
    with pytest.raises(MalformedValueError, match="not clean numbers"):
        text_to_number(frame_of({"a": ["1", bad, "3"]}))


def test_category_text_is_kept_and_empty_cells_become_nan() -> None:
    out = text_to_category(frame_of({"c": ["NA", "", "007", "x"]}))
    assert out[0, 0] == "NA" and out[2, 0] == "007" and out[3, 0] == "x"  # original text
    assert isinstance(out[1, 0], float) and np.isnan(out[1, 0])


# --- the fitted pipeline: missing values and unseen categories --------------------------------


@pytest.mark.parametrize("scale", [True, False])
def test_a_fitted_pipeline_scores_a_row_with_a_missing_value_and_an_unseen_category(
    scale: bool,
) -> None:
    columns = training_columns()
    frame = frame_of(columns)
    plan = build_column_plan(profile_frame(frame), TaskConfig("y", "classification"))
    pipe = build_pipeline(plan, LogisticRegression(max_iter=200), scale_numeric=scale)
    pipe.fit(frame[list(plan.used)], frame["y"].astype(str))
    new = pd.DataFrame(
        {"age": [None, "33"], "score": ["1.0", None], "color": ["NEVER-SEEN", ""]}, dtype="string"
    )
    predicted = pipe.predict(new)
    proba = pipe.predict_proba(new)
    assert list(predicted) and proba.shape == (2, 2) and np.allclose(proba.sum(axis=1), 1.0)


def test_regression_pipeline_also_scores_missing_and_unseen() -> None:
    columns = training_columns() | {"y": [f"{i % 11}.5" for i in range(60)]}
    frame = frame_of(columns)
    plan = build_column_plan(profile_frame(frame), TaskConfig("y", "regression"))
    pipe = build_pipeline(plan, Ridge(), scale_numeric=True)
    pipe.fit(frame[list(plan.used)], text_to_number(frame[["y"]])[:, 0])
    new = pd.DataFrame({"age": [None], "score": [None], "color": ["zzz"]}, dtype="string")
    assert np.isfinite(pipe.predict(new)).all()


def test_an_unseen_category_encodes_as_all_zeros_and_a_missing_one_as_the_fitted_mode() -> None:
    frame = frame_of({"c": ["a", "a", "a", "b", "b", "c"], "x": ["1", "2", "3", "4", "5", "6"]})
    prep = build_preprocessor(ColumnPlan(("x",), ("c",), ()), scale_numeric=False)
    prep.fit(frame[["x", "c"]])
    names = list(prep.get_feature_names_out())
    encoded = prep.transform(pd.DataFrame({"x": ["1", "1"], "c": ["???", None]}, dtype="string"))
    cat_cols = [i for i, n in enumerate(names) if n.startswith("categorical__")]
    assert encoded[0, cat_cols].tolist() == [0.0, 0.0, 0.0]  # unseen -> all zeros
    mode_col = names.index("categorical__c_a")
    assert encoded[1, mode_col] == 1.0  # missing -> most frequent ("a")


def test_scaling_applies_only_when_requested() -> None:
    columns = {"a": [str(v) for v in [1, 2, 3, 4, 100]], "y": ["p", "q", "p", "q", "p"]}
    frame = frame_of(columns)
    plan = plan_of(columns)
    scaled = build_preprocessor(plan, scale_numeric=True).fit_transform(frame[["a"]])
    raw = build_preprocessor(plan, scale_numeric=False).fit_transform(frame[["a"]])
    assert raw[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 100.0]
    assert abs(scaled[:, 0].mean()) < 1e-12 and abs(scaled[:, 0].std() - 1.0) < 1e-12


def test_a_malformed_value_at_inference_is_rejected_by_the_pipeline() -> None:
    frame = frame_of(training_columns())
    plan = build_column_plan(profile_frame(frame), TaskConfig("y", "classification"))
    pipe = build_pipeline(plan, LogisticRegression(max_iter=200), scale_numeric=True)
    pipe.fit(frame[list(plan.used)], frame["y"].astype(str))
    bad = pd.DataFrame({"age": ["abc"], "score": ["1"], "color": ["red"]}, dtype="string")
    with pytest.raises(MalformedValueError):
        pipe.predict(bad)


def test_extra_columns_at_inference_are_ignored_and_the_pipeline_can_be_saved_and_loaded() -> None:
    frame = frame_of(training_columns())
    plan = build_column_plan(profile_frame(frame), TaskConfig("y", "classification"))
    pipe = build_pipeline(plan, LogisticRegression(max_iter=200), scale_numeric=True)
    pipe.fit(frame[list(plan.used)], frame["y"].astype(str))
    restored = pickle.loads(pickle.dumps(pipe))  # module-level steps make this possible
    with_extra = frame.head(10).drop(columns=["y"]).assign(surprise="x")
    assert list(restored.predict(with_extra)) == list(pipe.predict(with_extra))


# --- fitting learns only from the fitted rows (the no-leakage property) ---------------------


def test_preprocessing_statistics_come_only_from_the_rows_it_was_fitted_on() -> None:
    columns = {
        "a": ["1", "2", "3", "", "5", "1000", "2000", "3000"],  # last three: "validation" rows
        "c": ["x", "x", "y", "y", "x", "only-in-validation", "only-in-validation", "z"],
        "y": ["p", "q"] * 4,
    }
    frame = frame_of(columns)
    plan = plan_of(columns)
    fit_rows = frame.iloc[:5][["a", "c"]]  # the fold's training rows
    pipe = build_pipeline(plan, LogisticRegression(), scale_numeric=True)
    prep = pipe.named_steps["prep"]
    prep.fit(fit_rows)
    imputer = prep.named_transformers_["numeric"].named_steps["impute"]
    scaler = prep.named_transformers_["numeric"].named_steps["scale"]
    encoder = prep.named_transformers_["categorical"].named_steps["encode"]
    fitted = np.array([1.0, 2.0, 3.0, 5.0])  # the non-missing values among the fitted rows
    assert imputer.statistics_[0] == np.median(fitted)  # not influenced by 1000, 2000, 3000
    imputed = np.array([1.0, 2.0, 3.0, np.median(fitted), 5.0])
    assert scaler.mean_[0] == pytest.approx(imputed.mean())
    assert scaler.scale_[0] == pytest.approx(imputed.std())
    assert list(encoder.categories_[0]) == ["x", "y"]  # no category from the validation rows
    # sensitivity control: fitting on everything WOULD have given different numbers
    everything = build_pipeline(plan, LogisticRegression(), scale_numeric=True)
    everything.named_steps["prep"].fit(frame[["a", "c"]])
    all_stats = everything.named_steps["prep"].named_transformers_["numeric"].named_steps["impute"]
    assert all_stats.statistics_[0] != imputer.statistics_[0]
