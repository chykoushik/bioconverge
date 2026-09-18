import hashlib
import importlib.metadata
import os
import platform
import pickle
from pathlib import Path

import pandas as pd

from .layer1 import ConcordanceAnalyzer
from .layer2 import FragilityAnalyzer
from .layer3 import HypothesisGenerator
from .layer4 import ValidationEngine
from .report import BioConvergeResult
from .utils import parse_gmt
from .validation import (validate, ValidationReport, InputValidationError, read_scores,
                         align_features, discover_data, status)

__version__ = "0.2.0"


class AnalysisError(RuntimeError):
    def __init__(self, layer, cause, statuses):
        super().__init__(f"{layer} failed: {cause}")
        self.layer = layer
        self.statuses = statuses.copy()


def _fingerprint(frame):
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    return hashlib.sha256(frame.to_json(orient="split", double_precision=15).encode()).hexdigest()


def _input_fingerprint(value):
    if isinstance(value, pd.DataFrame):
        return {"kind": "dataframe", "sha256": _fingerprint(value)}
    path = Path(value)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def _model_fingerprint(model):
    result = {"type": f"{type(model).__module__}.{type(model).__qualname__}"}
    try:
        result["sha256"] = hashlib.sha256(pickle.dumps(model, protocol=5)).hexdigest()
        result["serialization"] = "pickle-protocol-5"
    except Exception as e:
        result["sha256"] = None
        result["reason"] = f"model serialization unavailable: {type(e).__name__}: {e}"
    return result


def integrate(scores, patient_col="patient_id", score_metadata=None, models=None,
              feature_matrix=None, pathway_constraints=None, outcome=None, time_col=None,
              event_col=None, dataset_dir=None, n_archetypes=3, n_bootstrap=1000,
              random_state=42, output_dir="output", time_unit=None, event_mapping=None,
              disease_context=None, on_error="raise", api_cache=None, replay_only=False,
              target_class=None):
    validation = validate(scores, patient_col, score_metadata, models, feature_matrix,
                          pathway_constraints, outcome, dataset_dir, n_archetypes, n_bootstrap,
                          random_state, time_col, event_col, time_unit, event_mapping,
                          disease_context, output_dir, on_error, api_cache, replay_only, target_class)
    validation.raise_for_errors()
    score_df = read_scores(scores)
    statuses = {}

    def execute(name, operation):
        try:
            analyzer = operation()
            statuses[name] = status("completed", analyses=getattr(analyzer, "statuses", {}).copy())
            return analyzer
        except Exception as e:
            statuses[name] = status("failed", f"{type(e).__name__}: {e}")
            if on_error == "raise":
                raise AnalysisError(name, e, statuses) from e
            return None

    l1 = execute("layer1", lambda: ConcordanceAnalyzer(score_df, patient_col).fit(
        n_archetypes=n_archetypes, n_bootstrap=n_bootstrap, random_state=random_state))
    l2 = None
    features = None
    if models is not None:
        features = align_features(feature_matrix, score_df[patient_col])
        gene_sets = None
        if isinstance(pathway_constraints, dict):
            gene_sets = pathway_constraints
        elif pathway_constraints is not None:
            gmt = Path(dataset_dir or ".") / "h.all.v2023.2.Hs.symbols.gmt" if str(pathway_constraints) == "hallmark" else pathway_constraints
            gene_sets = parse_gmt(gmt)
        l2 = execute("layer2", lambda: FragilityAnalyzer(models, features, score_df[patient_col].to_numpy(),
                     gene_sets=gene_sets, random_state=random_state, target_class=target_class).fit())
    else:
        statuses["layer2"] = status("skipped", "models and feature_matrix not supplied")
    l3 = None
    if score_metadata is None:
        statuses["layer3"] = status("skipped", "score_metadata not supplied")
    elif l1 is None:
        statuses["layer3"] = status("unavailable", "Layer 1 did not complete")
    else:
        l3 = execute("layer3", lambda: HypothesisGenerator(score_metadata, l1.archetypes(),
                     fragility_df=l2.fragility() if l2 else None, disease_context=disease_context,
                     api_cache=api_cache, replay_only=replay_only).generate())
    paths = discover_data(dataset_dir)
    if outcome is not None:
        paths["clinical_tar_path"] = outcome
    l4 = None
    if l1 is None:
        statuses["layer4"] = status("unavailable", "Layer 1 did not complete")
    elif l3 is None and not any(value is not None for value in paths.values()):
        statuses["layer4"] = status("skipped", "no validation inputs or annotations")
    else:
        l4 = execute("layer4", lambda: ValidationEngine(**paths, patient_col=patient_col,
                     time_col=time_col, event_col=event_col, time_unit=time_unit, event_mapping=event_mapping,
                     random_state=random_state, disease_context=disease_context).validate(
                     l3.hypotheses() if l3 else pd.DataFrame(), l1.archetypes(), score_df,
                     patient_col, score_metadata or {}))
        if l4:
            components = list(l4.statuses.values())
            if any(v["state"] == "failed" for v in components):
                statuses["layer4"] = status("failed", "one or more validation analyses failed", analyses=l4.statuses.copy())
                if on_error == "raise":
                    reasons = [v["reason"] for v in components if v["state"] == "failed"]
                    raise AnalysisError("layer4", "; ".join(reasons), statuses)
            elif not any(v["state"] == "completed" for v in components):
                statuses["layer4"] = status("unavailable", "no empirical validation completed", analyses=l4.statuses.copy())
    versions = {}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "lifelines", "umap-learn", "hdbscan", "requests", "matplotlib", "plotly"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    input_files = {name: _input_fingerprint(value) for name, value in paths.items() if value is not None}
    metadata = {"output_dir": str(output_dir) if output_dir is not None else None,
                "input_files": input_files, "event_mapping": event_mapping,
                "version": __version__, "python": platform.python_version(), "platform": platform.platform(),
                "dependencies": versions, "random_state": int(random_state), "n_archetypes": int(n_archetypes),
                "n_bootstrap": int(n_bootstrap), "score_sha256": _fingerprint(score_df),
                "feature_sha256": _fingerprint(features) if features is not None else None,
                "score_metadata": score_metadata, "disease_context": disease_context,
                "validation": validation.to_dict(), "replay_only": replay_only,
                "time_col": time_col, "event_col": event_col, "time_unit": time_unit,
                "models": {k: _model_fingerprint(v) for k, v in
                                (models if isinstance(models, dict) else {"model": models} if models else {}).items()},
                "threads": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]}}
    return BioConvergeResult(l1=l1, l2=l2, l3=l3, l4=l4, statuses=statuses, metadata=metadata)
