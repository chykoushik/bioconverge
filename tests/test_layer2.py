import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from bioconverge.layer2 import FragilityAnalyzer


def _make_model_and_data(n=40, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = (X[:, 0] + rng.standard_normal(n) * 0.1 > 0).astype(int)
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    return model, X


def test_fragility_runs():
    model, X = _make_model_and_data()
    fa = FragilityAnalyzer({"m": model}, X)
    fa.fit()
    assert fa.fragility() is not None
    assert len(fa.fragility()) == len(X)


def test_fragility_scores_positive():
    model, X = _make_model_and_data()
    fa = FragilityAnalyzer({"m": model}, X)
    fa.fit()
    scores = fa.fragility()["fragility_score"].values
    assert (scores >= 0).all()


def test_trajectory_shape():
    model, X = _make_model_and_data(n=30, d=5)
    fa = FragilityAnalyzer({"m": model}, X, feature_names=[f"g{i}" for i in range(5)])
    fa.fit()
    traj = fa.trajectory()
    assert len(traj) == 30
    assert "top_features" in traj.columns


def test_consistency_two_models():
    model1, X = _make_model_and_data(seed=0)
    model2, _ = _make_model_and_data(seed=1)
    fa = FragilityAnalyzer({"m1": model1, "m2": model2}, X)
    fa.fit()
    cons = fa.consistency()
    assert "mean_rho" in cons
    assert "pairwise" in cons


def test_single_model_no_consistency():
    model, X = _make_model_and_data()
    fa = FragilityAnalyzer({"m": model}, X)
    fa.fit()
    cons = fa.consistency()
    assert "message" in cons


def test_pathway_fragility_with_gene_sets():
    model, X = _make_model_and_data(n=30, d=8)
    feat_names = [f"GENE{i}" for i in range(8)]
    gene_sets = {"PATHWAY_A": feat_names[:4], "PATHWAY_B": feat_names[4:]}
    fa = FragilityAnalyzer({"m": model}, X, feature_names=feat_names, gene_sets=gene_sets)
    fa.fit()
    pf = fa.fragility_pathways()
    assert pf is not None
    assert len(pf) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
