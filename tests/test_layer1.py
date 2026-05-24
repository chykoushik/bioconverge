import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from bioconverge.layer1 import ConcordanceAnalyzer


def _make_df(n=50, n_scores=3, seed=0):
    rng = np.random.default_rng(seed)
    data = {"patient_id": [f"P{i}" for i in range(n)]}
    for s in range(n_scores):
        data[f"score_{s}"] = rng.uniform(0, 1, n)
    return pd.DataFrame(data)


def test_fit_runs():
    df = _make_df()
    ca = ConcordanceAnalyzer(df, "patient_id")
    ca.fit(n_archetypes=3, n_bootstrap=50)
    assert ca.concordance() is not None
    assert ca.convergence() is not None
    assert ca.archetypes() is not None
    assert ca.stability() is not None


def test_concordance_shape():
    df = _make_df(n_scores=4)
    ca = ConcordanceAnalyzer(df, "patient_id").fit(n_bootstrap=20)
    conc = ca.concordance()
    # 4 scores -> 6 pairs
    assert len(conc) == 6
    assert "spearman_rho" in conc.columns


def test_convergence_strata():
    df = _make_df()
    ca = ConcordanceAnalyzer(df, "patient_id").fit(n_bootstrap=20)
    conv = ca.convergence()
    assert set(conv["stratum"].unique()).issubset(
        {"convergent_high", "convergent_mid", "convergent_low", "unknown"}
    )


def test_archetypes_count():
    df = _make_df(n=60)
    ca = ConcordanceAnalyzer(df, "patient_id").fit(n_archetypes=3, n_bootstrap=20)
    arch = ca.archetypes()
    assert len(arch["archetype"].unique()) == 3


def test_discordance_subset():
    df = _make_df(n=60)
    ca = ConcordanceAnalyzer(df, "patient_id").fit(n_bootstrap=20)
    disc = ca.discordance()
    assert disc is not None
    assert len(disc) <= len(df)


def test_single_patient_handled():
    df = _make_df(n=10, n_scores=2)
    ca = ConcordanceAnalyzer(df, "patient_id").fit(n_archetypes=2, n_bootstrap=10)
    assert ca.concordance() is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
