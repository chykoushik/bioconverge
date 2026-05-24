import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from bioconverge.layer3 import HypothesisGenerator, _pubmed_flag


def _make_archetypes(n=30, n_arch=3, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(n)],
        "archetype": rng.integers(0, n_arch, size=n),
    })


def _mock_api(*args, **kwargs):
    class FakeResp:
        def json(self):
            return {}
        def raise_for_status(self):
            pass
    return FakeResp()


def test_pubmed_flag():
    assert _pubmed_flag(200) == "convergent_with_prior_knowledge"
    assert _pubmed_flag(50) == "moderate_support"
    assert _pubmed_flag(5) == "exploratory"


@patch("bioconverge.layer3.api_get", side_effect=_mock_api)
def test_generate_runs(mock_get):
    arch = _make_archetypes()
    meta = {
        "immune_score": {"process": "T-cell exhaustion", "modality": "transcriptomic"},
        "genomic_score": {"process": "DNA repair", "modality": "genomic"},
    }
    gen = HypothesisGenerator(meta, arch)
    gen.generate()
    hyp = gen.hypotheses()
    assert hyp is not None
    assert len(hyp) > 0


@patch("bioconverge.layer3.api_get", side_effect=_mock_api)
def test_cross_support_shape(mock_get):
    arch = _make_archetypes()
    meta = {"score_a": {"process": "immune", "modality": "transcriptomic"}}
    gen = HypothesisGenerator(meta, arch)
    gen.generate()
    cs = gen.cross_support()
    assert "archetype" in cs.columns


@patch("bioconverge.layer3.api_get", side_effect=_mock_api)
def test_repro_log_not_empty(mock_get):
    arch = _make_archetypes()
    meta = {"s": {"process": "inflammation", "modality": "proteomic"}}
    gen = HypothesisGenerator(meta, arch)
    gen.generate()
    log = gen.reproducibility_log()
    assert len(log) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
