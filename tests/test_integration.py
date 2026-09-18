import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import bioconverge as bc
from bioconverge.layer4 import ValidationEngine


def test_public_pipeline_statuses_and_report(scores, tmp_path):
    result = bc.integrate(scores, n_bootstrap=3)
    assert result.statuses()["layer1"]["state"] == "completed"
    assert result.statuses()["layer2"]["state"] == "skipped"
    directory = tmp_path / "report"
    result.report(directory)
    metadata = json.loads((directory / "run_metadata.json").read_text())
    assert metadata["random_state"] == 42 and metadata["score_sha256"]
    assert (directory / "manifest.json").is_file()
    with pytest.raises(FileExistsError): result.report(directory)


def test_outcome_without_metadata(scores):
    outcomes = pd.DataFrame({"patient_id": scores.patient_id, "OS_days": np.arange(1, 25), "OS_event": [0, 1] * 12})
    result = bc.integrate(scores, outcome=outcomes, n_archetypes=2, n_bootstrap=3)
    assert result.survival_analysis()["state"] == "completed"
    assert result.statuses()["layer3"]["state"] == "skipped"


def test_feature_reordering_produces_identical_fragility(scores, monkeypatch):
    monkeypatch.setattr(bc.FragilityAnalyzer, "_compute_topology", lambda self, output_dir=None: None)
    features = scores.set_index("patient_id")[["a", "b"]]
    model = LinearRegression().fit(features, scores.c)
    a = bc.integrate(scores, models=model, feature_matrix=features, n_bootstrap=2)
    b = bc.integrate(scores, models=model, feature_matrix=features.iloc[::-1], n_bootstrap=2)
    pd.testing.assert_frame_equal(a.fragility(), b.fragility())


def test_layer_failure_policy(scores, monkeypatch):
    def fail(*args, **kwargs): raise RuntimeError("model failure")
    monkeypatch.setattr(bc.FragilityAnalyzer, "fit", fail)
    features = scores[["a", "b"]].to_numpy()
    model = LinearRegression().fit(features, scores.c)
    result = bc.integrate(scores, models=model, feature_matrix=features, n_bootstrap=2, on_error="record")
    assert result.statuses()["layer2"]["state"] == "failed"
    with pytest.raises(bc.AnalysisError) as error:
        bc.integrate(scores, models=model, feature_matrix=features, n_bootstrap=2)
    assert error.value.statuses["layer2"]["state"] == "failed"


def test_replay_missing_evidence_never_promoted(scores):
    result = bc.integrate(scores, n_bootstrap=2, score_metadata={"a": {"process": "immune"}}, replay_only=True)
    hypotheses = result.hypotheses()
    assert hypotheses.confidence_tier.eq("C").all()
    assert hypotheses.validation_status.eq("unavailable").all()
    assert hypotheses.db_support.eq(0).all()
    assert result.statuses()["layer4"]["state"] == "unavailable"


def test_metabric_requires_pr_and_no_numeric_fallback(tmp_path):
    path = tmp_path / "metabric.tsv"
    pd.DataFrame({"Patient ID": ["a", "b"], "ER status measured by IHC": ["Negative"] * 2,
                  "PR Status": ["Positive", "Negative"], "HER2 Status": ["Negative"] * 2,
                  "arbitrary": [1, 2], "outcome": [9, 8]}).to_csv(path, sep="\t", index=False)
    engine = ValidationEngine(metabric_path=path)
    engine._run_metabric({})
    result = engine.metabric_results()
    assert result["n_tnbc"] == 1 and result["state"] == "unavailable"
    assert result["replication_rate"] is None
    assert engine.compute_replication_rate(pd.DataFrame({"process": ["arbitrary"]})) is None
    frame = pd.read_csv(path, sep="\t").drop(columns="PR Status")
    frame.to_csv(path, sep="\t", index=False)
    engine._run_metabric({})
    assert engine.metabric_results()["state"] == "failed"


def test_benchmark_never_manufactures_patients(tmp_path):
    path = tmp_path / "cohort.tsv"
    pd.DataFrame({"anything": [1, 2, 3]}).to_csv(path, sep="\t", index=False)
    engine = ValidationEngine(tcga_brca_path=path)
    engine._run_benchmark(None, None, {"s": {"process": "immune activation"}}, None)
    result = engine.benchmark_results()
    assert result["state"] == "unavailable" and result["empirical"] is False
    assert result["mean_precision"] is None and result["mean_recall"] is None


def test_single_score_report_has_javascript(scores, tmp_path):
    pytest.importorskip("plotly")
    result = bc.integrate(scores[["patient_id", "a"]], n_bootstrap=2)
    result.report(tmp_path / "report")
    text = (tmp_path / "report" / "summary.html").read_text(encoding="utf-8")
    assert "Plotly.newPlot" in text and "plotly.js" in text


def test_missing_correlation_stays_missing(scores, tmp_path):
    scores["c"] = 1.0
    result = bc.integrate(scores, n_bootstrap=2)
    result.report(tmp_path / "report")
    data = pd.read_csv(tmp_path / "report" / "concordance.csv")
    assert data.loc[data.score_b == "c", "spearman_rho"].isna().all()


def test_default_report_destination_is_deferred(scores, tmp_path):
    target = tmp_path / "default_report"
    result = bc.integrate(scores, n_bootstrap=2, output_dir=target)
    assert not target.exists()
    assert Path(result.report()) == target
    assert (target / "manifest.json").is_file()
