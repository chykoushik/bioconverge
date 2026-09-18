import sys
from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from bioconverge.layer2 import FragilityAnalyzer, _get_predictions, _class_index
from bioconverge.validation import InputValidationError


class LinearModel:
    def __init__(self, slope=1): self.slope = slope
    def predict(self, X): return self.slope * X[:, 0]


def test_exact_gradient_and_cancellation():
    X = np.arange(12, dtype=float).reshape(6, 2)
    analyzer = FragilityAnalyzer({"positive": LinearModel(), "negative": LinearModel(-1)}, X)
    analyzer._compute_fragility()
    result = analyzer.fragility()
    assert result.fragility_score.to_numpy() == pytest.approx(np.zeros(6))
    assert result.mean_individual_fragility.to_numpy() == pytest.approx(np.ones(6))
    assert result.gradient_cancellation.to_numpy() == pytest.approx(np.ones(6))


def test_zero_trajectory_and_small_topology():
    analyzer = FragilityAnalyzer(LinearModel(0), np.ones((2, 3))).fit()
    assert analyzer.trajectory().trajectory_norm.eq(0).all()
    assert analyzer.trajectory().top_features.tolist() == [[], []]
    assert analyzer.topology().umap_x.isna().all()
    assert analyzer.statuses["topology"]["state"] == "unavailable"


def test_topology_failure_not_artificial_cluster(monkeypatch):
    def fail(**kwargs): raise RuntimeError("topology failed")
    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=fail))
    monkeypatch.setitem(sys.modules, "hdbscan", SimpleNamespace())
    analyzer = FragilityAnalyzer(LinearModel(), np.ones((5, 2)))
    analyzer._mean_grad = np.arange(10).reshape(5, 2).astype(float)
    with pytest.warns(RuntimeWarning, match="topology unavailable"):
        analyzer._compute_topology()
    assert analyzer.topology().umap_x.isna().all()
    assert analyzer.topology().fragility_cluster.eq(-1).all()


@pytest.mark.parametrize("missing", ["umap", "hdbscan"])
def test_missing_topology_dependency_preserves_core(monkeypatch, missing):
    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace())
    monkeypatch.setitem(sys.modules, missing, None)
    class Model:
        def predict(self, X): return np.sum(X ** 2, axis=1)
    analyzer = FragilityAnalyzer(Model(), np.arange(10, dtype=float).reshape(5, 2)).fit()
    assert np.isfinite(analyzer.fragility().fragility_score).all()
    assert analyzer.statuses["topology"]["state"] == "unavailable"
    assert missing in analyzer.statuses["topology"]["reason"]
    assert analyzer.topology().umap_x.isna().all()
    assert analyzer.topology().fragility_cluster.eq(-1).all()


def test_broken_topology_dependency_warns(monkeypatch):
    def fail(**kwargs): raise ModuleNotFoundError("missing transitive dependency", name="broken_dependency")
    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=fail))
    monkeypatch.setitem(sys.modules, "hdbscan", SimpleNamespace())
    analyzer = FragilityAnalyzer(LinearModel(), np.ones((5, 2)))
    analyzer._mean_grad = np.arange(10, dtype=float).reshape(5, 2)
    with pytest.warns(RuntimeWarning, match="missing transitive dependency"):
        analyzer._compute_topology()


def test_installed_topology():
    for name in ("umap", "hdbscan"):
        if find_spec(name) is None:
            pytest.skip(f"optional topology dependency unavailable: {name}")
    rng = np.random.default_rng(18)
    analyzer = FragilityAnalyzer(LinearModel(), rng.normal(size=(30, 5)))
    analyzer._mean_grad = rng.normal(size=(30, 5))
    analyzer._compute_topology()
    assert analyzer.statuses["topology"]["state"] == "completed"
    assert np.isfinite(analyzer.topology()[["umap_x", "umap_y"]]).all().all()


def test_named_model_features_preserved():
    features = pd.DataFrame({"a": np.arange(8), "b": np.arange(8) ** 2}, dtype=float)
    model = LinearRegression().fit(features, 2 * features.a + 3 * features.b)
    analyzer = FragilityAnalyzer(model, features)
    analyzer._compute_fragility()
    assert analyzer._mean_grad == pytest.approx(np.tile([2, 3], (8, 1)), abs=1e-6)


@pytest.mark.parametrize("kind", ["nan", "infinity", "wrong_shape"])
def test_invalid_prediction_rejected(kind):
    class BadModel:
        def predict(self, X):
            return np.full(len(X), np.nan if kind == "nan" else np.inf) if kind != "wrong_shape" else np.ones((len(X), 2))
    with pytest.raises(InputValidationError):
        FragilityAnalyzer(BadModel(), np.ones((3, 2)))._compute_fragility()


@pytest.mark.parametrize("width,index", [(1, None), (3, None), (3, -1), (3, 3)])
def test_target_validation(width, index):
    with pytest.raises(InputValidationError): _class_index(width, index)


def test_multiclass_explicit_target():
    class Model:
        def predict_proba(self, X): return np.tile([0.2, 0.3, 0.5], (len(X), 1))
    assert _get_predictions(Model(), np.ones((4, 2)), "sklearn", 2).tolist() == [0.5] * 4


def test_feature_metadata_lengths():
    with pytest.raises(InputValidationError): FragilityAnalyzer(LinearModel(), np.ones((3, 2)), patient_ids=[1, 2])
    with pytest.raises(InputValidationError): FragilityAnalyzer(LinearModel(), np.ones((3, 2)), feature_names=["a"])


def test_no_pathway_overlap_reported():
    analyzer = FragilityAnalyzer(LinearModel(), np.ones((3, 2)), gene_sets={"path": ["absent"]}).fit()
    assert analyzer.fragility_pathways().empty
    assert analyzer.statuses["pathways"]["state"] == "unavailable"
