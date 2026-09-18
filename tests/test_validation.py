from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import bioconverge as bc
from bioconverge.validation import align_features


@pytest.mark.parametrize("bad", [None, [], {}, 4, np.ones((3, 2))])
def test_invalid_score_types(bad):
    assert not bc.validate(bad).valid


@pytest.mark.parametrize("change", ["empty", "id_missing", "id_null", "id_duplicate", "id_blank", "nonnumeric",
                                     "infinity", "negative_infinity", "missing_column", "missing_row", "constant", "duplicate_column"])
def test_invalid_scores(scores, change):
    frame = scores.copy()
    if change == "empty": frame = frame.iloc[:0]
    if change == "id_missing": frame = frame.drop(columns="patient_id")
    if change == "id_null": frame.loc[0, "patient_id"] = None
    if change == "id_duplicate": frame.loc[1, "patient_id"] = frame.loc[0, "patient_id"]
    if change == "id_blank": frame.loc[0, "patient_id"] = " "
    if change == "nonnumeric": frame["a"] = "bad"
    if change == "infinity": frame.loc[0, "a"] = np.inf
    if change == "negative_infinity": frame.loc[0, "a"] = -np.inf
    if change == "missing_column": frame["a"] = np.nan
    if change == "missing_row": frame.loc[0, ["a", "b", "c"]] = np.nan
    if change == "constant": frame[["a", "b", "c"]] = 1.0
    if change == "duplicate_column": frame.columns = ["patient_id", "a", "a", "c"]
    report = bc.validate(frame)
    assert not report.valid
    with pytest.raises(bc.InputValidationError):
        bc.integrate(frame)


@pytest.mark.parametrize("parameter,value", [("n_archetypes", 0), ("n_archetypes", 25), ("n_archetypes", 2.5),
    ("n_archetypes", True), ("n_bootstrap", 0), ("n_bootstrap", -1), ("n_bootstrap", 2.5),
    ("random_state", None), ("random_state", -1), ("random_state", 2**32), ("random_state", True),
    ("on_error", "ignore"), ("disease_context", ""), ("replay_only", 1), ("target_class", -1)])
def test_invalid_parameters(scores, parameter, value):
    assert not bc.validate(scores, **{parameter: value}).valid


@pytest.mark.parametrize("metadata", [{}, {"absent": {}}, {"a": "bad"}, {"a": {"genes": "TP53"}},
                                      {"a": {"genes": [None]}}, {"a": {"process": ""}}])
def test_metadata_validation(scores, metadata):
    assert not bc.validate(scores, score_metadata=metadata).valid


def test_partial_missing_and_constant_reported(scores):
    scores.loc[0, "a"] = np.nan
    scores["c"] = 1.0
    result = bc.validate(scores)
    assert result.valid and len(result.warnings) == 2
    assert result.details["missing_scores"]["a"] == 1


def test_path_and_no_mutation(scores, tmp_path):
    path = tmp_path / "scores.csv"
    scores.to_csv(path, index=False)
    before = set(tmp_path.iterdir())
    assert bc.validate(path).valid
    assert set(tmp_path.iterdir()) == before


def test_feature_alignment(scores):
    features = scores.set_index("patient_id")[["a", "b"]]
    actual = align_features(features.iloc[::-1], scores.patient_id)
    pd.testing.assert_frame_equal(actual, features)
    with pytest.raises(bc.InputValidationError):
        align_features(features.reset_index(drop=True), scores.patient_id)


@pytest.mark.parametrize("shape", [(23, 2), (24, 0), (24,)])
def test_feature_shape(scores, shape):
    model = LinearRegression().fit(np.ones((24, 2)), np.arange(24))
    assert not bc.validate(scores, models=model, feature_matrix=np.ones(shape)).valid


def test_partial_models_and_feature_order(scores):
    features = scores.set_index("patient_id")[["a", "b"]]
    model = LinearRegression().fit(features, scores.c)
    assert not bc.validate(scores, models=model).valid
    assert not bc.validate(scores, feature_matrix=features).valid
    assert not bc.validate(scores, models={}, feature_matrix=features).valid
    assert not bc.validate(scores, models=model, feature_matrix=features[["b", "a"]]).valid
    assert bc.validate(scores, models=model, feature_matrix=features.iloc[::-1]).valid


def test_missing_and_ambiguous_datasets(scores, tmp_path):
    assert not bc.validate(scores, dataset_dir=tmp_path / "missing").valid
    for name in ["a_clinical.tar.gz", "b_clinical.tar.gz"]:
        (tmp_path / name).touch()
    assert not bc.validate(scores, dataset_dir=tmp_path).valid


@pytest.mark.parametrize("n", [0, 1, 2])
def test_small_cohorts_rejected_with_default_clusters(scores, n):
    assert not bc.validate(scores.iloc[:n]).valid


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_features_rejected(scores, value):
    features = scores[["a", "b"]].to_numpy()
    model = LinearRegression().fit(features, scores.c)
    features[0, 0] = value
    assert not bc.validate(scores, models=model, feature_matrix=features).valid


def test_missing_pathway_file_rejected(scores, tmp_path):
    features = scores[["a", "b"]].to_numpy()
    model = LinearRegression().fit(features, scores.c)
    report = bc.validate(scores, models=model, feature_matrix=features, pathway_constraints=tmp_path / "missing.gmt")
    assert not report.valid
