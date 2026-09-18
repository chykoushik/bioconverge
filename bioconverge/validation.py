from dataclasses import dataclass, field
from numbers import Integral
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd


class InputValidationError(ValueError):
    pass


@dataclass
class ValidationReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def valid(self):
        return not self.errors

    def raise_for_errors(self):
        if self.errors:
            raise InputValidationError("; ".join(self.errors))
        return self

    def to_dict(self):
        return {"valid": self.valid, "errors": self.errors.copy(),
                "warnings": self.warnings.copy(), "details": self.details.copy()}


def status(state, reason=None, **details):
    if state not in {"completed", "skipped", "failed", "unavailable"}:
        raise ValueError("invalid analysis state")
    return {"state": state, "reason": reason, **details}


def validate_ids(values, name="patient IDs"):
    s = pd.Series(values)
    if s.empty or s.isna().any() or s.astype(str).str.strip().eq("").any():
        raise InputValidationError(f"{name} must be nonempty and nonmissing")
    if s.duplicated().any():
        raise InputValidationError(f"{name} must be unique")
    kinds = {type(v) for v in s.tolist()}
    if len(kinds) > 1:
        raise InputValidationError(f"{name} must use a consistent type")


def validate_seed(random_state):
    if isinstance(random_state, bool) or not isinstance(random_state, Integral):
        raise InputValidationError("random_state must be an integer")
    if not 0 <= random_state < 2 ** 32:
        raise InputValidationError("random_state must be between 0 and 2**32 - 1")


def read_scores(scores):
    if isinstance(scores, (str, PathLike)):
        return pd.read_csv(scores)
    if not isinstance(scores, pd.DataFrame):
        raise InputValidationError("scores must be a DataFrame or CSV path")
    return scores.copy(deep=True)


def check_scores(df, patient_col="patient_id", n_archetypes=None):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise InputValidationError("scores must contain patients")
    if not df.columns.is_unique or not all(isinstance(c, str) and c.strip() for c in df.columns):
        raise InputValidationError("score column names must be unique nonempty strings")
    if patient_col not in df:
        raise InputValidationError(f"missing patient column: {patient_col}")
    validate_ids(df[patient_col])
    cols = [c for c in df if c != patient_col]
    if not cols:
        raise InputValidationError("no score columns found")
    try:
        numeric = df[cols].apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as e:
        raise InputValidationError("scores must be numeric or missing") from e
    if np.isinf(numeric.to_numpy()).any():
        raise InputValidationError("scores cannot contain infinite values")
    if numeric.isna().all(axis=0).any() or numeric.isna().all(axis=1).any():
        raise InputValidationError("entirely missing score rows or columns are not supported")
    constants = numeric.columns[numeric.nunique() <= 1].tolist()
    if len(constants) == len(cols):
        raise InputValidationError("at least one score must vary")
    if n_archetypes is not None:
        if isinstance(n_archetypes, bool) or not isinstance(n_archetypes, Integral) or n_archetypes < 1:
            raise InputValidationError("n_archetypes must be a positive integer")
        filled = numeric.fillna(numeric.median())
        if len(filled.drop_duplicates()) < n_archetypes:
            raise InputValidationError("n_archetypes exceeds distinct usable patient profiles")
    out = df.copy(deep=True)
    out[cols] = numeric
    return out, constants


def check_metadata(metadata, score_cols=None):
    if metadata is None:
        return
    if not isinstance(metadata, dict) or not metadata:
        raise InputValidationError("score_metadata must be a nonempty dictionary")
    for name, meta in metadata.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(meta, dict):
            raise InputValidationError("score metadata entries must be named dictionaries")
        if score_cols is not None and name not in score_cols:
            raise InputValidationError(f"metadata references absent score: {name}")
        for key in ("process", "modality"):
            if key in meta and (not isinstance(meta[key], str) or not meta[key].strip()):
                raise InputValidationError(f"invalid {key} for {name}")
        genes = meta.get("genes", [])
        if not isinstance(genes, (list, tuple)) or any(not isinstance(g, str) or not g.strip() for g in genes):
            raise InputValidationError(f"genes for {name} must be a list of nonempty strings")


def align_features(features, patient_ids):
    if isinstance(features, pd.DataFrame):
        if not features.columns.is_unique or not features.index.is_unique:
            raise InputValidationError("feature index and columns must be unique")
        if set(features.index) != set(patient_ids):
            raise InputValidationError("feature DataFrame index must match patient IDs exactly")
        features = features.loc[list(patient_ids)].copy()
    try:
        values = np.asarray(features, dtype=float)
    except (TypeError, ValueError) as e:
        raise InputValidationError("features must be numeric") from e
    if values.ndim != 2 or values.shape[0] != len(patient_ids) or values.shape[1] == 0:
        raise InputValidationError("feature matrix must have one row per patient and at least one feature")
    if not np.isfinite(values).all():
        raise InputValidationError("features must contain only finite values")
    return features.copy() if isinstance(features, pd.DataFrame) else values.copy()


def check_models(models, features):
    mapping = models if isinstance(models, dict) else {"model": models}
    if not mapping or any(not isinstance(k, str) or not k.strip() for k in mapping):
        raise InputValidationError("models must have nonempty string names")
    if "score" in mapping:
        raise InputValidationError("model name score is reserved for aggregate fragility")
    for model in mapping.values():
        if model is None or not (callable(model) or callable(getattr(model, "predict", None))):
            raise InputValidationError("each model must be callable or provide predict")
        if hasattr(model, "n_features_in_") and model.n_features_in_ != features.shape[1]:
            raise InputValidationError("model and feature dimensions differ")
        if hasattr(model, "feature_names_in_"):
            if not isinstance(features, pd.DataFrame) or list(model.feature_names_in_) != list(features.columns):
                raise InputValidationError("feature columns must match model training order")
    return mapping


def discover_data(dataset_dir):
    paths = {}
    if dataset_dir is None:
        return paths
    rules = {"clinical_tar_path": lambda n: "clinical" in n and n.endswith(".tar.gz"),
             "mut_tar_path": lambda n: "Mutation_Packager" in n and n.endswith(".tar.gz"),
             "metabric_path": lambda n: "metabric" in n and n.endswith(".tsv"),
             "tcga_brca_path": lambda n: "brca_tcga" in n and n.endswith(".tsv")}
    for key, rule in rules.items():
        matches = [p for p in sorted(Path(dataset_dir).iterdir()) if p.is_file() and rule(p.name)]
        if len(matches) > 1:
            raise InputValidationError(f"ambiguous dataset selection for {key}: {[p.name for p in matches]}")
        paths[key] = matches[0] if matches else None
    return paths


def validate(scores, patient_col="patient_id", score_metadata=None, models=None,
             feature_matrix=None, pathway_constraints=None, outcome=None,
             dataset_dir=None, n_archetypes=3, n_bootstrap=1000, random_state=42,
             time_col=None, event_col=None, time_unit=None, event_mapping=None,
             disease_context=None, output_dir=None, on_error="raise", api_cache=None, replay_only=False, target_class=None, **kwargs):
    report = ValidationReport()
    try:
        if kwargs:
            raise InputValidationError(f"unknown parameters: {sorted(kwargs)}")
        validate_seed(random_state)
        if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, Integral) or n_bootstrap < 1:
            raise InputValidationError("n_bootstrap must be a positive integer")
        df, constants = check_scores(read_scores(scores), patient_col, n_archetypes)
        cols = [c for c in df if c != patient_col]
        check_metadata(score_metadata, cols)
        report.details = {"n_patients": len(df), "score_columns": cols,
                          "missing_scores": df[cols].isna().sum().to_dict(),
                          "constant_columns": constants}
        if constants:
            report.warnings.append(f"constant scores retained: {constants}")
        if df[cols].isna().any().any():
            report.warnings.append("clustering median-imputes missing scores; concordance uses pairwise observations")
        if (models is None) != (feature_matrix is None):
            raise InputValidationError("models and feature_matrix must be provided together")
        if models is not None:
            features = align_features(feature_matrix, df[patient_col])
            check_models(models, features)
        if dataset_dir is not None and not Path(dataset_dir).is_dir():
            raise InputValidationError("dataset_dir must be an existing directory")
        discover_data(dataset_dir)
        if api_cache is not None and not isinstance(api_cache, dict):
            raise InputValidationError("api_cache must be a dictionary")
        if not isinstance(replay_only, bool):
            raise InputValidationError("replay_only must be boolean")
        if target_class is not None and (isinstance(target_class, bool) or not isinstance(target_class, Integral) or target_class < 0):
            raise InputValidationError("target_class must be a nonnegative integer")
        if pathway_constraints is not None:
            if models is None:
                raise InputValidationError("pathway constraints require models and features")
            from .utils import parse_gmt
            if isinstance(pathway_constraints, dict):
                if not pathway_constraints or any(not isinstance(g, (list, tuple)) or not g or
                        any(not isinstance(v, str) or not v.strip() for v in g) for g in pathway_constraints.values()):
                    raise InputValidationError("invalid pathway gene sets")
            else:
                path = Path(dataset_dir or ".") / "h.all.v2023.2.Hs.symbols.gmt" if str(pathway_constraints) == "hallmark" else Path(pathway_constraints)
                parse_gmt(path)
        if outcome is not None:
            from .utils import load_survival
            clinical = load_survival(outcome, time_col, event_col, patient_col,
                                     time_unit=time_unit, event_mapping=event_mapping)
            if not df[patient_col].isin(clinical.patient_id).any():
                raise InputValidationError("survival and score patient IDs do not overlap")
        if disease_context is not None and (not isinstance(disease_context, str) or not disease_context.strip()):
            raise InputValidationError("disease_context must be a nonempty string")
        if output_dir is not None and Path(output_dir).exists() and not Path(output_dir).is_dir():
            raise InputValidationError("output_dir must be a directory")
        if on_error not in {"raise", "record"}:
            raise InputValidationError("on_error must be raise or record")
    except (ValueError, TypeError, OSError) as e:
        report.errors.append(str(e))
    return report
