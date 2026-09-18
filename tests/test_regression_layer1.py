import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from bioconverge.layer1 import ConcordanceAnalyzer


def test_repeat_seed_is_independent_of_global_rng(scores):
    np.random.seed(1)
    first = ConcordanceAnalyzer(scores, "patient_id").fit(n_bootstrap=15, random_state=42)
    np.random.seed(892)
    second = ConcordanceAnalyzer(scores, "patient_id").fit(n_bootstrap=15, random_state=42)
    assert first.stability() == second.stability()
    pd.testing.assert_frame_equal(first.archetypes(), second.archetypes())
    pd.testing.assert_frame_equal(first.concordance(), second.concordance())


def test_cross_process_seed_independent_of_python_hash():
    program = """import json, numpy as np, pandas as pd
from bioconverge.layer1 import ConcordanceAnalyzer
r=np.random.default_rng(3)
f=pd.DataFrame({'patient_id':list(range(15)),'a':r.normal(size=15),'b':r.normal(size=15)})
a=ConcordanceAnalyzer(f,'patient_id').fit(n_bootstrap=8,random_state=4)
print(json.dumps(a.stability(),sort_keys=True))
"""
    outputs = []
    for seed in ["1", "2"]:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        outputs.append(subprocess.check_output([sys.executable, "-c", program], text=True, env=env))
    assert outputs[0] == outputs[1]


def test_degenerate_bootstrap_draws_keep_finite_ci():
    data = pd.DataFrame({"patient_id": list("abcde"), "a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})
    analyzer = ConcordanceAnalyzer(data, "patient_id")
    analyzer._compute_concordance(1000, 42)
    row = analyzer.concordance().iloc[0]
    assert row.ci_lo == pytest.approx(1)
    assert 0 < row.n_bootstrap_valid < 1000


def test_constant_score_is_undefined(scores):
    scores["c"] = 1.0
    analyzer = ConcordanceAnalyzer(scores, "patient_id").fit(n_bootstrap=3)
    rows = analyzer.concordance().query("score_b == 'c'")
    assert rows.spearman_rho.isna().all()
    assert rows.reason.eq("constant score").all()


def test_degenerate_strata_not_high():
    data = pd.DataFrame({"patient_id": list("abcdef"), "a": [1, -1] * 3, "b": [-1, 1] * 3})
    analyzer = ConcordanceAnalyzer(data, "patient_id")
    analyzer._compute_convergence()
    assert analyzer.convergence().convergence_index.eq(-1).all()
    assert analyzer.convergence().stratum.eq("unknown").all()
    assert analyzer.statuses["convergence_strata"]["state"] == "unavailable"


def test_cap_and_valid_counts(scores):
    analyzer = ConcordanceAnalyzer(scores, "patient_id")
    analyzer._compute_archetypes(3, 42)
    analyzer._compute_stability(3, 201, 42)
    assert analyzer.stability()["requested_bootstrap"] == 201
    assert analyzer.stability()["attempted_bootstrap"] == 200


def test_pairwise_missing_and_small_samples(scores):
    scores.loc[4:, "a"] = np.nan
    analyzer = ConcordanceAnalyzer(scores, "patient_id").fit(n_bootstrap=3)
    assert analyzer.concordance().query("score_a == 'a'").spearman_rho.isna().all()
    assert len(analyzer.archetypes()) == len(scores)


def test_small_bootstrap_fits_report_degeneracy():
    frame = pd.DataFrame({"patient_id": ["a", "b", "c"], "x": [1, 2, 3], "y": [3, 1, 2]})
    analyzer = ConcordanceAnalyzer(frame, "patient_id").fit(n_archetypes=3, n_bootstrap=10)
    assert analyzer.stability()["failed_bootstrap"] > 0
    assert analyzer.stability()["n_bootstrap"] + analyzer.stability()["failed_bootstrap"] == 10
