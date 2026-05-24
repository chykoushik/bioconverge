import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class BioConvergeResult:
    def __init__(self, l1=None, l2=None, l3=None, l4=None, skipped=None):
        self._l1 = l1
        self._l2 = l2
        self._l3 = l3
        self._l4 = l4
        self._skipped = skipped or []

    def concordance(self):
        if self._l1 is None:
            print("layer1 not run")
            return None
        return self._l1.concordance()

    def convergence(self):
        if self._l1 is None:
            print("layer1 not run")
            return None
        return self._l1.convergence()

    def discordance(self):
        if self._l1 is None:
            print("layer1 not run")
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
            print("layer2 not run")
            return None
        return self._l2.fragility()

    def fragility_pathways(self):
        if self._l2 is None:
            print("layer2 not run")
            return None
        return self._l2.fragility_pathways()

    def hypotheses(self):
        if self._l4 is not None and self._l4.tiered_hypotheses() is not None:
            return self._l4.tiered_hypotheses()
        if self._l3 is None:
            print("layer3 not run")
            return None
        return self._l3.hypotheses()

    def survival_analysis(self):
        if self._l4 is None:
            print("layer4 not run")
            return None
        return self._l4.survival_results()

    def compare_layers(self):
        if self._l1 is None or self._l2 is None:
            print("need layer1 and layer2")
            return None
        arch_df = self._l1.archetypes()
        conv_df = self._l1.convergence()
        frag_df = self._l2.fragility()
        merged = arch_df.merge(conv_df, on="patient_id").merge(frag_df, on="patient_id")
        frag_median = merged["fragility_score"].median()
        disc_mask = merged["stratum"] == "convergent_low"
        frag_mask = merged["fragility_score"] > frag_median
        double_flagged = merged[disc_mask & frag_mask].copy()
        double_flagged["flag_source"] = "both_l1_l2"
        return double_flagged.reset_index(drop=True)

    def report(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        km_dir = os.path.join(output_dir, "kaplan_meier")
        os.makedirs(km_dir, exist_ok=True)
        self._write_per_patient_csv(output_dir)
        self._write_hypotheses_csv(output_dir)
        self._write_reproducibility_log(output_dir)
        self._write_summary_html(output_dir)
        if self._l2 is not None:
            self._write_fragility_topology(output_dir)
        print("report done")
        return output_dir

    def _write_per_patient_csv(self, output_dir):
        frames = []
        if self._l1 is not None:
            conv = self._l1.convergence()
            arch = self._l1.archetypes()
            if conv is not None and arch is not None:
                merged = conv.merge(arch, on="patient_id")
                frames.append(merged)
        if self._l2 is not None and frames:
            frag = self._l2.fragility()
            if frag is not None:
                frames[0] = frames[0].merge(frag, on="patient_id", how="left")
        if frames:
            out = frames[0]
            path = os.path.join(output_dir, "per_patient_scores.csv")
            out.to_csv(path, index=False)
            print("saved per_patient_scores")

    def _write_hypotheses_csv(self, output_dir):
        hyp = self.hypotheses()
        if hyp is not None and not hyp.empty:
            path = os.path.join(output_dir, "hypotheses_ranked.csv")
            hyp.to_csv(path, index=False)
            print("saved hypotheses")

    def _write_reproducibility_log(self, output_dir):
        lines = [f"bioconverge report generated {datetime.utcnow().isoformat()}\n"]
        if self._skipped:
            lines.append(f"skipped: {', '.join(self._skipped)}\n")
        if self._l3 is not None:
            log_df = self._l3.reproducibility_log()
            if log_df is not None and not log_df.empty:
                lines.append("database queries:\n")
                for _, row in log_df.iterrows():
                    lines.append(f"  {row['timestamp']}  {row['query']}\n")
        path = os.path.join(output_dir, "reproducibility_log.txt")
        with open(path, "w") as f:
            f.writelines(lines)
        print("saved repro log")

    def _write_summary_html(self, output_dir):
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.express as px
        except ImportError:
            self._write_summary_html_fallback(output_dir)
            return
        figs = []
        if self._l1 is not None:
            conc = self._l1.concordance()
            if conc is not None and not conc.empty:
                score_names = sorted(set(conc["score_a"].tolist() + conc["score_b"].tolist()))
                n = len(score_names)
                mat = np.zeros((n, n))
                idx_map = {s: i for i, s in enumerate(score_names)}
                for _, row in conc.iterrows():
                    i = idx_map[row["score_a"]]
                    j = idx_map[row["score_b"]]
                    mat[i, j] = row["spearman_rho"] if not np.isnan(row["spearman_rho"]) else 0
                    mat[j, i] = mat[i, j]
                np.fill_diagonal(mat, 1.0)
                fig_conc = go.Figure(data=go.Heatmap(
                    z=mat,
                    x=score_names,
                    y=score_names,
                    colorscale="RdBu",
                    zmin=-1,
                    zmax=1,
                ))
                fig_conc.update_layout(title="concordance matrix")
                figs.append(fig_conc.to_html(full_html=False, include_plotlyjs="cdn"))
            conv = self._l1.convergence()
            if conv is not None and not conv.empty:
                fig_conv = px.histogram(conv, x="convergence_index", color="stratum", title="convergence index")
                figs.append(fig_conv.to_html(full_html=False, include_plotlyjs=False))
        hyp = self.hypotheses()
        if hyp is not None and not hyp.empty:
            cols = [c for c in ["archetype", "process", "db_support", "confidence_tier", "pubmed_count"] if c in hyp.columns]
            fig_hyp = go.Figure(data=go.Table(
                header=dict(values=cols),
                cells=dict(values=[hyp[c].tolist() for c in cols]),
            ))
            fig_hyp.update_layout(title="top hypotheses")
            figs.append(fig_hyp.to_html(full_html=False, include_plotlyjs=False))
        html_content = "<html><head><title>bioconverge report</title></head><body>\n"
        html_content += f"<h1>bioconverge</h1><p>generated {datetime.utcnow().isoformat()}</p>\n"
        if self._skipped:
            html_content += f"<p>skipped: {', '.join(self._skipped)}</p>\n"
        for fig_html in figs:
            html_content += f"<div>{fig_html}</div>\n"
        html_content += "</body></html>"
        path = os.path.join(output_dir, "summary.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print("saved summary.html")

    def _write_summary_html_fallback(self, output_dir):
        html_content = "<html><body><h1>bioconverge report</h1>"
        html_content += f"<p>generated {datetime.utcnow().isoformat()}</p>"
        if self._skipped:
            html_content += f"<p>skipped: {', '.join(self._skipped)}</p>"
        html_content += "<p>install plotly for interactive plots</p></body></html>"
        path = os.path.join(output_dir, "summary.html")
        with open(path, "w") as f:
            f.write(html_content)

    def _write_fragility_topology(self, output_dir):
        topo = self._l2.topology()
        if topo is None or topo.empty:
            return
        if topo["umap_x"].isna().all():
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            topo["umap_x"],
            topo["umap_y"],
            c=topo["fragility_cluster"],
            cmap="tab10",
            s=20,
            alpha=0.8,
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("fragility topology")
        plt.colorbar(scatter, ax=ax, label="cluster")
        path = os.path.join(output_dir, "fragility_topology.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("saved topology")
