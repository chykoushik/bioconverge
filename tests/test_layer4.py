import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from bioconverge.layer4 import ValidationEngine, LEHMANN_SIGNATURES


def _make_hypotheses(n=12):
    rng = np.random.default_rng(0)
    processes = ["T-cell exhaustion", "DNA repair", "immune activation"] * (n // 3) + ["T-cell exhaustion"] * (n % 3)
    return pd.DataFrame({
        "archetype": rng.integers(0, 3, n),
        "score": [f"score_{i}" for i in range(n)],
        "process": processes,
        "modality": ["transcriptomic"] * n,
        "hypothesis": [f"hyp {i}" for i in range(n)],
        "db_support": rng.integers(0, 5, n),
        "reactome_pathway": ["pathway"] * n,
        "reactome_url": [""] * n,
        "enrichr_term": ["hallmark"] * n,
        "pubmed_count": rng.integers(0, 500, n),
        "pubmed_flag": ["moderate_support"] * n,
        "n_patients_archetype": [20] * n,
    })


def _make_archetypes(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "patient_id": [f"TCGA-XX-{i:04d}" for i in range(n)],
        "archetype": rng.integers(0, 3, n),
    })


def test_tier_assignment_no_data():
    ve = ValidationEngine()
    hyp = _make_hypotheses()
    arch = _make_archetypes()
    score_df = pd.DataFrame({"patient_id": arch["patient_id"], "s1": np.random.rand(len(arch)), "s2": np.random.rand(len(arch))})
    meta = {"s1": {"process": "immune activation"}, "s2": {"process": "DNA repair"}}
    ve.validate(hyp, arch, score_df, "patient_id", meta)
    tiered = ve.tiered_hypotheses()
    assert tiered is not None
    assert "confidence_tier" in tiered.columns
    assert set(tiered["confidence_tier"].unique()).issubset({"A", "B", "C"})


def test_tier_c_when_no_validation():
    ve = ValidationEngine()
    hyp = _make_hypotheses()
    arch = _make_archetypes()
    score_df = pd.DataFrame({"patient_id": arch["patient_id"], "s1": np.random.rand(len(arch)), "s2": np.random.rand(len(arch))})
    meta = {"s1": {"process": "some obscure process"}}
    ve.validate(hyp, arch, score_df, "patient_id", meta)
    tiered = ve.tiered_hypotheses()
    assert "C" in tiered["confidence_tier"].values


def test_replication_rate_no_metabric():
    ve = ValidationEngine()
    hyp = _make_hypotheses()
    rate = ve.compute_replication_rate(hyp)
    assert rate is None


def test_lehmann_signatures_defined():
    assert "BL1" in LEHMANN_SIGNATURES
    assert "BL2" in LEHMANN_SIGNATURES
    assert "M" in LEHMANN_SIGNATURES
    assert "IM" in LEHMANN_SIGNATURES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
