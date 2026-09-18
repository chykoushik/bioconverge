import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bioconverge.layer2 import FragilityAnalyzer, _detect_backend, _torch_output


def test_custom_predictor_does_not_import_optional_frameworks(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    class Model:
        def predict(self, X): return X[:, 0]
    assert _detect_backend(Model()) == "sklearn"
    analyzer = FragilityAnalyzer(Model(), np.ones((3, 2)))
    analyzer._compute_fragility()
    assert np.isfinite(analyzer.fragility().fragility_score).all()


def test_topology_seed_propagates(monkeypatch):
    seen = []
    class UMAP:
        def __init__(self, **kwargs): seen.append(kwargs)
        def fit_transform(self, X): return X
    class Cluster:
        def __init__(self, **kwargs): pass
        def fit_predict(self, X): return np.zeros(len(X), dtype=int)
    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=UMAP))
    monkeypatch.setitem(sys.modules, "hdbscan", SimpleNamespace(HDBSCAN=Cluster))
    class Model:
        def predict(self, X): return X[:, 0]
    analyzer = FragilityAnalyzer(Model(), np.ones((5, 2)), random_state=71)
    analyzer._mean_grad = np.arange(10, dtype=float).reshape(5, 2)
    analyzer._compute_topology()
    assert seen[0]["random_state"] == 71
    assert analyzer.statuses["topology"]["state"] == "completed"
