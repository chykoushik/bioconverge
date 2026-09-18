import copy
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _json_safe(value):
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict("records"))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    return value


class BioConvergeResult:
    def __init__(self, l1=None, l2=None, l3=None, l4=None, skipped=None, statuses=None, metadata=None):
        self._l1 = l1
        self._l2 = l2
        self._l3 = l3
        self._l4 = l4
        self._skipped = skipped or []
        self._statuses = copy.deepcopy(statuses or {})
        self._metadata = copy.deepcopy(metadata or {})

    def concordance(self):
        if self._l1 is None:
            return None
        return self._l1.concordance()

    def convergence(self):
        if self._l1 is None:
            return None
        return self._l1.convergence()

    def discordance(self):
        if self._l1 is None:
            return None
        return self._l1.discordance()

    def archetypes(self):
        if self._l1 is None:
            return None
        return self._l1.archetypes()

    def stability(self):
        if self._l1 is None:
            return None
        return self._l1.stability()

    def fragility(self):
        if self._l2 is None:
            return None
        return self._l2.fragility()

    def fragility_pathways(self):
        if self._l2 is None:
            return None
        return self._l2.fragility_pathways()

    def hypotheses(self):
        if self._l4 is not None and self._l4.tiered_hypotheses() is not None:
            return self._l4.tiered_hypotheses()
        if self._l3 is None:
            return None
        return self._l3.hypotheses()

    def survival_analysis(self):
        if self._l4 is None:
            return None
        return self._l4.survival_results()

    def compare_layers(self):
        if self._l1 is None or self._l2 is None:
            return None
        arch_df = self._l1.archetypes()
        conv_df = self._l1.convergence()
        frag_df = self._l2.fragility()
        merged = arch_df.merge(conv_df, on="patient_id", validate="one_to_one").merge(frag_df, on="patient_id", validate="one_to_one")
        frag_median = merged["fragility_score"].median()
        disc_mask = merged["stratum"] == "convergent_low"
        frag_mask = merged["fragility_score"] > frag_median
        double_flagged = merged[disc_mask & frag_mask].copy()
        double_flagged["flag_source"] = "both_l1_l2"
        return double_flagged.reset_index(drop=True)

    def statuses(self):
        return copy.deepcopy(self._statuses)

    def metadata(self):
        return copy.deepcopy(self._metadata)

    def report(self, output_dir=None):
        directory = Path(output_dir or self._metadata.get("output_dir") or "output")
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("report directory must be empty to prevent stale or overwritten results")
        directory.mkdir(parents=True, exist_ok=True)
        frames = []
        if self._l1 is not None:
            frames.append(self._l1.convergence().merge(self._l1.archetypes(), on="patient_id", validate="one_to_one"))
            self._l1.concordance().to_csv(directory / "concordance.csv", index=False)
        if frames:
            patients = frames[0]
            if self._l2 is not None:
                patients = patients.merge(self._l2.fragility(), on="patient_id", how="left", validate="one_to_one")
            patients.to_csv(directory / "per_patient_scores.csv", index=False)
        hypotheses = self.hypotheses()
        if hypotheses is not None:
            hypotheses.to_csv(directory / "hypotheses_ranked.csv", index=False)
        if self._l4 is not None:
            self._l4.plot_survival(directory / "kaplan_meier")
        if self._l2 is not None:
            self._write_fragility_topology(directory)
        self._write_summary_html(directory)
        evidence = self._l3.api_cache if self._l3 is not None else {}
        validation = {}
        if self._l4 is not None:
            validation = {"survival": self._l4.survival_results(), "metabric": self._l4.metabric_results(),
                          "benchmark": self._l4.benchmark_results(), "literature": self._l4.literature_results()}
        payloads = {"run_metadata.json": self._metadata, "analysis_statuses.json": self._statuses,
                    "api_responses.json": evidence, "validation_results.json": validation,
                    "reproducibility_log.json": self._l3.reproducibility_log() if self._l3 is not None else []}
        for name, data in payloads.items():
            (directory / name).write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        manifest = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(directory.rglob("*")) if p.is_file()}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return str(directory)

    def _write_summary_html(self, output_dir):
        parts = ["<html><head><meta charset='utf-8'><title>bioconverge report</title></head><body>",
                 "<h1>bioconverge</h1><p>Metadata annotations are exploratory; empirical hypothesis validation is unavailable.</p>",
                 "<pre>" + html.escape(json.dumps(_json_safe(self._statuses), indent=2)) + "</pre>"]
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            go = None
        include_js = True
        if self._l1 is not None:
            concordance = self._l1.concordance()
            if concordance is not None and not concordance.empty:
                if go is None:
                    parts.append(concordance.to_html(index=False, escape=True))
                else:
                    names = self._l1.score_cols
                    matrix = np.full((len(names), len(names)), np.nan)
                    indices = {name: i for i, name in enumerate(names)}
                    for _, row in concordance.iterrows():
                        i, j = indices[row.score_a], indices[row.score_b]
                        matrix[i, j] = matrix[j, i] = row.spearman_rho
                    for i, name in enumerate(names):
                        if name not in self._l1.diagnostics["constant_columns"]:
                            matrix[i, i] = 1.0
                    figure = go.Figure(go.Heatmap(z=matrix, x=names, y=names, colorscale="RdBu", zmin=-1, zmax=1))
                    figure.update_layout(title="concordance; blank cells are undefined")
                    parts.append(figure.to_html(full_html=False, include_plotlyjs=include_js))
                    include_js = False
            convergence = self._l1.convergence()
            if go is not None and convergence is not None and not convergence.empty:
                figure = px.histogram(convergence, x="convergence_index", color="stratum", title="convergence index")
                parts.append(figure.to_html(full_html=False, include_plotlyjs=include_js))
        hypotheses = self.hypotheses()
        if hypotheses is not None and not hypotheses.empty:
            columns = [c for c in ["archetype", "process", "evidence_scope", "db_support", "confidence_tier", "validation_status"] if c in hypotheses]
            parts.append(hypotheses[columns].to_html(index=False, escape=True))
        parts.append("</body></html>")
        (Path(output_dir) / "summary.html").write_text("\n".join(parts), encoding="utf-8")

    def _write_fragility_topology(self, output_dir):
        topology = self._l2.topology()
        if topology is None or topology.empty or not np.isfinite(topology[["umap_x", "umap_y"]].to_numpy()).all():
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        try:
            scatter = ax.scatter(topology.umap_x, topology.umap_y, c=topology.fragility_cluster, cmap="tab10", s=20, alpha=0.8)
            ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title="fragility topology")
            plt.colorbar(scatter, ax=ax, label="cluster")
            fig.savefig(Path(output_dir) / "fragility_topology.png", dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)
