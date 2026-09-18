import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .validation import InputValidationError, validate_ids, validate_seed, check_models, align_features, status


def _class_index(width, target_class):
    if width < 2:
        raise InputValidationError("classifier must expose at least two classes")
    index = 1 if target_class is None and width == 2 else target_class
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < width:
        raise InputValidationError("multiclass models require a valid target_class column index")
    return index


def _checked_predictions(values, n):
    values = np.asarray(values, dtype=float)
    if values.shape == (n, 1):
        values = values[:, 0]
    if values.shape != (n,) or not np.isfinite(values).all():
        raise InputValidationError("model must return one finite target per patient")
    return values


def _torch_output(model, X, target_class=None, gradient=False):
    import torch
    states = [(m, m.training) for m in model.modules()]
    anchor = next(model.parameters(), None)
    if anchor is None:
        anchor = next(model.buffers(), torch.tensor(0.0))
    try:
        model.eval()
        t = torch.tensor(np.asarray(X), dtype=anchor.dtype, device=anchor.device, requires_grad=gradient)
        with torch.set_grad_enabled(gradient):
            out = model(t)
            if out.ndim == 2 and out.shape[1] > 1:
                out = torch.softmax(out, dim=1)[:, _class_index(out.shape[1], target_class)]
            if out.shape not in [(len(X),), (len(X), 1)]:
                raise InputValidationError("model target shape is invalid")
            if gradient:
                result = torch.autograd.grad(out.sum(), t)[0]
            else:
                result = out
        return result.detach().cpu().numpy()
    finally:
        for module, training in states:
            module.training = training


def _tf_output(model, X, target_class=None, gradient=False):
    import tensorflow as tf
    t = tf.convert_to_tensor(np.asarray(X), dtype=getattr(model, "compute_dtype", "float32"))
    with tf.GradientTape() as tape:
        tape.watch(t)
        out = model(t, training=False)
        if len(out.shape) == 2 and out.shape[1] > 1:
            out = tf.nn.softmax(out, axis=1)[:, _class_index(out.shape[1], target_class)]
        if tuple(out.shape) not in [(len(X),), (len(X), 1)]:
            raise InputValidationError("model target shape is invalid")
        loss = tf.reduce_sum(out)
    result = tape.gradient(loss, t) if gradient else out
    if result is None:
        raise InputValidationError("model input gradient is unavailable")
    return result.numpy()


def _get_predictions(model, X, backend, target_class=None):
    if backend == "sklearn":
        data = X
        if hasattr(model, "feature_names_in_") and not isinstance(X, pd.DataFrame):
            data = pd.DataFrame(X, columns=model.feature_names_in_)
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(data))
            if probabilities.ndim != 2:
                raise InputValidationError("multioutput classifiers are unsupported")
            values = probabilities[:, _class_index(probabilities.shape[1], target_class)]
        else:
            if getattr(model, "_estimator_type", None) == "classifier":
                raise InputValidationError("classifiers require predict_proba for fragility")
            values = model.predict(data)
    elif backend == "torch":
        values = _torch_output(model, X, target_class)
    elif backend == "tensorflow":
        values = _tf_output(model, X, target_class)
    else:
        raise InputValidationError(f"unknown backend: {backend}")
    return _checked_predictions(values, len(X))


def _predict_proba_sklearn(model, X):
    return _get_predictions(model, X, "sklearn")


def _finite_diff_gradient(model, X, eps=1e-4, backend="sklearn", target_class=None):
    n, d = X.shape
    grads = np.zeros((n, d))
    base = _get_predictions(model, X, backend, target_class)
    for j in range(d):
        shifted = X.copy()
        shifted[:, j] += eps
        grads[:, j] = (_get_predictions(model, shifted, backend, target_class) - base) / eps
    return grads


def _torch_gradient(model, X):
    return _torch_output(model, X, gradient=True)


def _tf_gradient(model, X):
    return _tf_output(model, X, gradient=True)


def _detect_backend(model):
    modules = [cls.__module__ for cls in type(model).__mro__]
    if any(m.startswith("torch.") for m in modules):
        return "torch"
    if any(m.startswith(("tensorflow.", "keras.")) for m in modules):
        return "tensorflow"
    return "sklearn"


class FragilityAnalyzer:
    def __init__(self, models, X, patient_ids=None, feature_names=None, gene_sets=None, random_state=42, target_class=None):
        validate_seed(random_state)
        self.random_state = int(random_state)
        self.target_class = target_class
        self.statuses = {}
        if isinstance(X, pd.DataFrame) and feature_names is None:
            feature_names = list(X.columns)
        if isinstance(X, pd.DataFrame):
            if patient_ids is None:
                patient_ids = X.index.to_numpy()
            X = align_features(X, patient_ids)
        if isinstance(models, dict):
            self.models = models
        else:
            self.models = {"model": models}
        self.X = np.array(X, dtype=float)
        if self.X.ndim != 2 or min(self.X.shape) < 1 or not np.isfinite(self.X).all():
            raise InputValidationError("features must be a nonempty finite matrix")
        check_models(models, X if isinstance(X, pd.DataFrame) else self.X)
        self.patient_ids = np.asarray(patient_ids if patient_ids is not None else np.arange(len(self.X)))
        validate_ids(self.patient_ids)
        if len(self.patient_ids) != len(self.X):
            raise InputValidationError("patient and feature row counts differ")
        self.feature_names = feature_names if feature_names is not None else [f"f{i}" for i in range(self.X.shape[1])]
        if len(self.feature_names) != self.X.shape[1] or len(set(self.feature_names)) != len(self.feature_names):
            raise InputValidationError("feature names must be unique and match matrix columns")
        if gene_sets is not None and (not isinstance(gene_sets, dict) or not gene_sets or any(not isinstance(g, (list, tuple)) or not g for g in gene_sets.values())):
            raise InputValidationError("gene_sets must map pathways to nonempty gene lists")
        self.gene_sets = gene_sets
        self._fragility_df = None
        self._pathway_df = None
        self._trajectory_df = None
        self._topology_df = None
        self._consistency_dict = None

    def fit(self, output_dir=None):
        self.statuses = {}
        self._compute_fragility()
        if self.gene_sets:
            self._compute_pathway_fragility()
        else:
            self.statuses["pathways"] = status("skipped", "no pathway constraints")
        self._compute_trajectory()
        self._compute_topology(output_dir=output_dir)
        if len(self.models) >= 2:
            self._compute_consistency()
        self.statuses["fragility"] = status("completed", aggregation="norm_of_mean_gradient", perturbation_step=1e-4)
        return self

    def _compute_gradients_single(self, model):
        backend = _detect_backend(model)
        if backend == "torch":
            gradients = _torch_output(model, self.X, self.target_class, gradient=True)
        elif backend == "tensorflow":
            gradients = _tf_output(model, self.X, self.target_class, gradient=True)
        else:
            gradients = _finite_diff_gradient(model, self.X, backend=backend, target_class=self.target_class)
        if gradients.shape != self.X.shape or not np.isfinite(gradients).all():
            raise InputValidationError("gradients must be finite and match the feature matrix")
        return gradients

    def _compute_fragility(self):
        all_grads = {}
        for name, model in self.models.items():
            grads = self._compute_gradients_single(model)
            all_grads[name] = grads
        mean_grad = np.mean([g for g in all_grads.values()], axis=0)
        fragility_scores = np.linalg.norm(mean_grad, axis=1)
        self._all_grads = all_grads
        self._mean_grad = mean_grad
        self._fragility_df = pd.DataFrame({
            "patient_id": self.patient_ids,
            "fragility_score": fragility_scores,
        })
        individual = np.mean([np.linalg.norm(g, axis=1) for g in all_grads.values()], axis=0)
        self._fragility_df["mean_individual_fragility"] = individual
        self._fragility_df["gradient_cancellation"] = individual - fragility_scores
        for name, grads in all_grads.items():
            self._fragility_df[f"fragility_{name}"] = np.linalg.norm(grads, axis=1)

    def _compute_pathway_fragility(self):
        rows = []
        for pathway, genes in self.gene_sets.items():
            idxs = [i for i, f in enumerate(self.feature_names) if f in genes]
            if not idxs:
                continue
            pathway_grads = self._mean_grad[:, idxs]
            pathway_fragility = np.linalg.norm(pathway_grads, axis=1)
            rows.append({
                "pathway": pathway,
                "mean_fragility": float(np.mean(pathway_fragility)),
                "max_fragility": float(np.max(pathway_fragility)),
                "n_features": len(idxs),
            })
        if rows:
            self._pathway_df = pd.DataFrame(rows).sort_values("mean_fragility", ascending=False).reset_index(drop=True)
        else:
            self._pathway_df = pd.DataFrame(columns=["pathway", "mean_fragility", "max_fragility", "n_features"])
        self.statuses["pathways"] = status("completed" if rows else "unavailable", None if rows else "no pathway genes match features")

    def _compute_trajectory(self):
        directions = self._mean_grad.copy()
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        unit_dirs = np.divide(directions, norms, out=np.zeros_like(directions), where=norms != 0)
        rows = []
        for i, pid in enumerate(self.patient_ids):
            top_idx = np.argsort(np.abs(unit_dirs[i]), kind="stable")[::-1][:5] if norms[i, 0] > 0 else []
            top_feats = [self.feature_names[j] for j in top_idx]
            top_vals = [float(unit_dirs[i, j]) for j in top_idx]
            rows.append({
                "patient_id": pid,
                "top_features": top_feats,
                "top_directions": top_vals,
                "trajectory_norm": float(norms[i, 0]),
            })
        self._trajectory_df = pd.DataFrame(rows)

    def _compute_topology(self, output_dir=None):
        n_samples = len(self.X)
        embedding = np.full((n_samples, 2), np.nan)
        labels = np.full(n_samples, -1, dtype=int)
        if n_samples < 4 or len(np.unique(self._mean_grad, axis=0)) < 3:
            self.statuses["topology"] = status("unavailable", "at least four patients and three distinct gradients required")
        else:
            try:
                import umap
                import hdbscan
                norms = np.linalg.norm(self._mean_grad, axis=1, keepdims=True)
                normalized = np.divide(self._mean_grad, norms, out=np.zeros_like(self._mean_grad), where=norms != 0)
                reducer = umap.UMAP(n_neighbors=min(15, n_samples - 1), n_components=2,
                                    random_state=self.random_state, n_jobs=1)
                candidate = reducer.fit_transform(normalized)
                clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, n_samples // 20))
                candidate_labels = clusterer.fit_predict(candidate)
                if not np.isfinite(candidate).all():
                    raise ValueError("nonfinite embedding")
                embedding, labels = candidate, candidate_labels
                self.statuses["topology"] = status("completed")
            except Exception as e:
                self.statuses["topology"] = status("unavailable", f"{type(e).__name__}: {e}")
                if not isinstance(e, ModuleNotFoundError) or e.name not in {"umap", "hdbscan"}:
                    warnings.warn(f"topology unavailable: {e}", RuntimeWarning, stacklevel=2)
        self._topology_df = pd.DataFrame({"patient_id": self.patient_ids,
                                         "umap_x": embedding[:, 0], "umap_y": embedding[:, 1],
                                         "fragility_cluster": labels})
        if output_dir and self.statuses["topology"]["state"] == "completed":
            os.makedirs(output_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            try:
                scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=20, alpha=0.8)
                ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title="fragility topology")
                plt.colorbar(scatter, ax=ax, label="cluster")
                fig.savefig(os.path.join(output_dir, "fragility_topology.png"), dpi=150, bbox_inches="tight")
            finally:
                plt.close(fig)

    def _compute_consistency(self):
        model_names = list(self.models.keys())
        scores = {}
        for name in model_names:
            scores[name] = np.linalg.norm(self._all_grads[name], axis=1)
        from scipy.stats import spearmanr
        pairs = []
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                na, nb = model_names[i], model_names[j]
                rho, pval = (np.nan, np.nan) if np.ptp(scores[na]) == 0 or np.ptp(scores[nb]) == 0 else spearmanr(scores[na], scores[nb])
                pairs.append({"model_a": na, "model_b": nb, "spearman_rho": float(rho), "pvalue": float(pval)})
        mean_rho = np.mean([p["spearman_rho"] for p in pairs]) if pairs else np.nan
        self._consistency_dict = {
            "pairwise": pairs,
            "mean_rho": float(mean_rho),
            "consistent": bool(mean_rho > 0.5) if not np.isnan(mean_rho) else False,
        }

    def fragility(self):
        return self._fragility_df

    def fragility_pathways(self):
        return self._pathway_df

    def trajectory(self):
        return self._trajectory_df

    def topology(self):
        return self._topology_df

    def consistency(self):
        if self._consistency_dict is None:
            return {"message": "single model provided, no consistency check"}
        return self._consistency_dict
