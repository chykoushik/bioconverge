import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _predict_proba_sklearn(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X).ravel()


def _finite_diff_gradient(model, X, eps=1e-4, backend="sklearn"):
    n, d = X.shape
    grads = np.zeros((n, d))
    base = _get_predictions(model, X, backend)
    for j in range(d):
        Xp = X.copy()
        Xp[:, j] += eps
        pred_p = _get_predictions(model, Xp, backend)
        grads[:, j] = (pred_p - base) / eps
    return grads


def _get_predictions(model, X, backend):
    if backend == "sklearn":
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return model.predict(X).ravel()
    if backend == "torch":
        import torch
        model.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            out = model(t)
            if out.dim() > 1 and out.shape[1] > 1:
                out = torch.softmax(out, dim=1)[:, 1]
            return out.numpy().ravel()
    if backend == "tensorflow":
        import tensorflow as tf
        t = tf.constant(X, dtype=tf.float32)
        out = model(t)
        return out.numpy().ravel()
    raise ValueError(f"unknown backend: {backend}")


def _torch_gradient(model, X):
    import torch
    model.eval()
    t = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    out = model(t)
    if out.dim() > 1 and out.shape[1] > 1:
        out = torch.softmax(out, dim=1)[:, 1]
    loss = out.sum()
    loss.backward()
    return t.grad.detach().numpy()


def _tf_gradient(model, X):
    import tensorflow as tf
    t = tf.constant(X, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(t)
        out = model(t)
        if len(out.shape) > 1 and out.shape[1] > 1:
            out = tf.nn.softmax(out)[:, 1]
        loss = tf.reduce_sum(out)
    return tape.gradient(loss, t).numpy()


def _detect_backend(model):
    cls = type(model).__module__
    if "sklearn" in cls:
        return "sklearn"
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            return "torch"
    except ImportError:
        pass
    try:
        import tensorflow as tf
        if isinstance(model, tf.Module):
            return "tensorflow"
    except ImportError:
        pass
    return "sklearn"


class FragilityAnalyzer:
    def __init__(self, models, X, patient_ids=None, feature_names=None, gene_sets=None):
        if isinstance(models, dict):
            self.models = models
        else:
            self.models = {"model": models}
        self.X = np.array(X, dtype=float)
        self.patient_ids = patient_ids if patient_ids is not None else np.arange(len(self.X))
        self.feature_names = feature_names if feature_names is not None else [f"f{i}" for i in range(self.X.shape[1])]
        self.gene_sets = gene_sets
        self._fragility_df = None
        self._pathway_df = None
        self._trajectory_df = None
        self._topology_df = None
        self._consistency_dict = None

    def fit(self, output_dir=None):
        print("fitting fragility")
        self._compute_fragility()
        if self.gene_sets:
            self._compute_pathway_fragility()
        self._compute_trajectory()
        self._compute_topology(output_dir=output_dir)
        if len(self.models) >= 2:
            self._compute_consistency()
        print("layer2 done")
        return self

    def _compute_gradients_single(self, model):
        backend = _detect_backend(model)
        if backend == "torch":
            try:
                return _torch_gradient(model, self.X)
            except Exception:
                return _finite_diff_gradient(model, self.X, backend=backend)
        if backend == "tensorflow":
            try:
                return _tf_gradient(model, self.X)
            except Exception:
                return _finite_diff_gradient(model, self.X, backend=backend)
        return _finite_diff_gradient(model, self.X, backend=backend)

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
        for name, grads in all_grads.items():
            self._fragility_df[f"fragility_{name}"] = np.linalg.norm(grads, axis=1)

    def _compute_pathway_fragility(self):
        rows = []
        feat_set = set(self.feature_names)
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

    def _compute_trajectory(self):
        directions = self._mean_grad.copy()
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit_dirs = directions / norms
        rows = []
        for i, pid in enumerate(self.patient_ids):
            top_idx = np.argsort(np.abs(unit_dirs[i]))[::-1][:5]
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
        try:
            import umap
            import hdbscan
        except ImportError:
            self._topology_df = pd.DataFrame({
                "patient_id": self.patient_ids,
                "umap_x": np.nan,
                "umap_y": np.nan,
                "fragility_cluster": -1,
            })
            return
        fragility_mat = self._mean_grad.copy()
        norms = np.linalg.norm(fragility_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        fragility_norm = fragility_mat / norms
        n_samples = len(fragility_norm)
        n_neighbors = min(15, n_samples - 1)
        try:
            reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, random_state=42)
            embedding = reducer.fit_transform(fragility_norm)
            min_cluster = max(2, n_samples // 20)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster)
            labels = clusterer.fit_predict(embedding)
        except Exception as e:
            print(f"topology fallback: {e}")
            embedding = np.zeros((n_samples, 2))
            labels = np.zeros(n_samples, dtype=int)
        self._topology_df = pd.DataFrame({
            "patient_id": self.patient_ids,
            "umap_x": embedding[:, 0],
            "umap_y": embedding[:, 1],
            "fragility_cluster": labels,
        })
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=20, alpha=0.8)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_title("fragility topology")
            plt.colorbar(scatter, ax=ax, label="cluster")
            fig.savefig(os.path.join(output_dir, "fragility_topology.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("saved topology")

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
                rho, pval = spearmanr(scores[na], scores[nb])
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
