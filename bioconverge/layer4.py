import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import load_survival, load_maf_tp53, load_metabric, load_tcga_brca
from .validation import InputValidationError, validate_ids, validate_seed, status

LEHMANN_SIGNATURES = {
    "BL1": ["DNA damage response", "BRCA", "cell cycle", "checkpoint"],
    "BL2": ["growth factor signaling", "IGF1R", "EGFR", "PI3K"],
    "M": ["epithelial-mesenchymal transition", "EMT", "TGF", "WNT"],
    "IM": ["immune activation", "interferon", "cytokine", "T cell"],
}


def _result(state, reason=None, **details):
    return {**status(state, reason), "skipped": state != "completed", **details}


def _adjust_pvalues(values):
    values = np.asarray(values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    indices = np.flatnonzero(np.isfinite(values))
    order = indices[np.argsort(values[indices])]
    if len(order):
        ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
        adjusted[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return adjusted


def _km_plot(durations, events, groups, group_labels, title, save_path):
    from lifelines import KaplanMeierFitter
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        for group in sorted(set(groups)):
            mask = np.asarray(groups) == group
            fitter = KaplanMeierFitter()
            fitter.fit(np.asarray(durations)[mask], np.asarray(events)[mask], label=group_labels.get(group, str(group)))
            fitter.plot_survival_function(ax=ax)
        ax.set(xlabel="days", ylabel="survival", title=title)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)


class ValidationEngine:
    def __init__(self, clinical_tar_path=None, mut_tar_path=None, metabric_path=None,
                 tcga_brca_path=None, dataset_dir=None, time_col=None, event_col=None,
                 patient_col=None, random_state=42, time_unit=None, event_mapping=None,
                 disease_context=None):
        validate_seed(random_state)
        self.random_state = int(random_state)
        self.clinical_tar_path = clinical_tar_path
        self.mut_tar_path = mut_tar_path
        self.metabric_path = metabric_path
        self.tcga_brca_path = tcga_brca_path
        self.dataset_dir = dataset_dir
        self.time_col = time_col
        self.event_col = event_col
        self.patient_col = patient_col
        self.time_unit = time_unit
        self.event_mapping = event_mapping
        self.disease_context = disease_context
        self._survival_results = None
        self._metabric_results = None
        self._benchmark_results = None
        self._literature_results = None
        self._tiered_hypotheses = None
        self.statuses = {}

    def validate(self, hypotheses_df, archetypes_df, score_df, patient_col, score_metadata, km_output_dir=None):
        if not isinstance(archetypes_df, pd.DataFrame) or not {"patient_id", "archetype"}.issubset(archetypes_df):
            raise InputValidationError("archetypes require patient_id and archetype")
        validate_ids(archetypes_df.patient_id)
        if archetypes_df.archetype.isna().any():
            raise InputValidationError("archetype labels must be nonmissing")
        if not isinstance(score_df, pd.DataFrame) or patient_col not in score_df:
            raise InputValidationError("score patient column is missing")
        validate_ids(score_df[patient_col])
        if set(score_df[patient_col]) != set(archetypes_df.patient_id):
            raise InputValidationError("score and archetype patient IDs must match")
        if hypotheses_df is not None and not hypotheses_df.empty and not {"archetype", "process", "db_support"}.issubset(hypotheses_df):
            raise InputValidationError("invalid hypothesis schema")
        self._run_survival(archetypes_df, km_output_dir)
        self._run_metabric(score_metadata)
        self._run_benchmark(score_df, patient_col, score_metadata, archetypes_df)
        self._run_literature(hypotheses_df, score_metadata)
        self._assign_tiers(hypotheses_df)
        for name, result in [("survival", self._survival_results), ("metabric", self._metabric_results),
                             ("benchmark", self._benchmark_results), ("literature", self._literature_results)]:
            self.statuses[name] = status(result["state"], result["reason"])
        self.statuses["empirical_validation"] = status("unavailable", "metadata annotations are not empirically tested hypotheses")
        return self

    def _run_survival(self, archetypes_df, km_output_dir=None):
        if self.clinical_tar_path is None:
            self._survival_results = _result("skipped", "no clinical data")
            return
        try:
            clinical = load_survival(self.clinical_tar_path, self.time_col, self.event_col, self.patient_col,
                                     time_unit=self.time_unit, event_mapping=self.event_mapping)
            merged = archetypes_df.merge(clinical, on="patient_id", how="inner", validate="one_to_one")
            diagnostics = {**clinical.attrs, "n_matched": len(merged),
                           "n_unmatched": len(archetypes_df) - len(merged)}
            if len(merged) < 6 or not merged.OS_event.any():
                self._survival_results = _result("unavailable", "insufficient matched patients or no observed events", **diagnostics)
                return
            from lifelines.statistics import logrank_test
            mutants = set()
            assessed = set()
            mutation_status = status("skipped", "no mutation data")
            if self.mut_tar_path is not None:
                try:
                    mutations = load_maf_tp53(self.mut_tar_path)
                    mutants = set(mutations.patient_id)
                    assessed = set(mutations.attrs.get("assessed_patients", []))
                    mutation_status = status("completed")
                except Exception as e:
                    mutation_status = status("failed", f"{type(e).__name__}: {e}")
            rows = []
            for arch in sorted(merged.archetype.unique()):
                selected = merged.archetype == arch
                if selected.sum() < 3 or (~selected).sum() < 3:
                    rows.append({"archetype": arch, "state": "unavailable", "reason": "fewer than three patients in a comparison group",
                                 "logrank_pvalue": np.nan, "n_archetype": int(selected.sum()), "n_rest": int((~selected).sum())})
                    continue
                test = logrank_test(merged.loc[selected, "OS_days"], merged.loc[~selected, "OS_days"],
                                    merged.loc[selected, "OS_event"], merged.loc[~selected, "OS_event"])
                pvalue = float(test.p_value)
                if not np.isfinite(pvalue):
                    rows.append({"archetype": arch, "state": "unavailable", "reason": "log-rank test is undefined",
                                 "logrank_pvalue": np.nan, "n_archetype": int(selected.sum()), "n_rest": int((~selected).sum())})
                    continue
                row = {"archetype": arch, "state": "completed", "reason": None,
                       "n_archetype": int(selected.sum()), "n_rest": int((~selected).sum()), "logrank_pvalue": pvalue}
                for label, mask in [("archetype", selected), ("rest", ~selected)]:
                    ids = set(merged.loc[mask, "patient_id"])
                    row[f"n_tp53_assessed_{label}"] = len(ids & assessed)
                    row[f"n_tp53_unknown_{label}"] = len(ids - assessed)
                    row[f"n_tp53_mutant_{label}"] = len(ids & mutants) if ids & assessed else np.nan
                rows.append(row)
            results = pd.DataFrame(rows)
            results["logrank_adjusted_pvalue"] = _adjust_pvalues(results.logrank_pvalue)
            results["survival_signal"] = results.logrank_adjusted_pvalue < 0.05
            available = results.state.eq("completed").any()
            self._survival_results = _result("completed" if available else "unavailable",
                                            None if available else "no valid survival comparisons", results=results,
                                            mutation_status=mutation_status, evidence_scope="archetype_association",
                                            adjustment="Benjamini-Hochberg", **diagnostics)
            self._survival_data = merged.copy()
            if km_output_dir and available:
                self.plot_survival(km_output_dir)
        except Exception as e:
            self._survival_results = _result("failed", f"{type(e).__name__}: {e}")

    def plot_survival(self, output_dir):
        if not self._survival_results or self._survival_results["state"] != "completed":
            return
        os.makedirs(output_dir, exist_ok=True)
        merged = self._survival_data
        for _, row in self._survival_results["results"].iterrows():
            if row["state"] != "completed":
                continue
            arch = row["archetype"]
            groups = np.where(merged.archetype == arch, "archetype", "rest")
            _km_plot(merged.OS_days, merged.OS_event, groups, {"archetype": f"archetype {arch}", "rest": "other"},
                     f"archetype {arch} vs rest (adjusted p={row['logrank_adjusted_pvalue']:.3g})",
                     os.path.join(output_dir, f"km_archetype_{arch}.png"))

    def _run_metabric(self, score_metadata):
        if self.metabric_path is None:
            self._metabric_results = _result("skipped", "no METABRIC data", replication_rate=None)
            return
        try:
            frame = load_metabric(self.metabric_path)
            required = ["Patient ID", "ER status measured by IHC", "PR Status", "HER2 Status"]
            if not set(required).issubset(frame):
                raise InputValidationError("METABRIC requires Patient ID and ER, PR, HER2 status columns")
            validate_ids(frame["Patient ID"], "METABRIC patient IDs")
            receptors = frame[required[1:]].apply(lambda c: c.astype("string").str.strip().str.lower())
            eligible = receptors.eq("negative").fillna(False).all(axis=1)
            self._metabric_results = _result("unavailable", "clinical proxies and process-name overlap are not empirical replication",
                                            n_tnbc=int(eligible.sum()), n_patients=len(frame), replication_rate=None,
                                            n_receptor_unknown=int((~receptors.isin(["negative", "positive"])).any(axis=1).sum()),
                                            empirical=False)
        except Exception as e:
            self._metabric_results = _result("failed", f"{type(e).__name__}: {e}", replication_rate=None, empirical=False)

    def compute_replication_rate(self, tcga_hypotheses):
        return None

    def _run_benchmark(self, score_df, patient_col, score_metadata, archetypes_df):
        if self.tcga_brca_path is None:
            self._benchmark_results = _result("skipped", "no empirical benchmark supplied", empirical=False)
            return
        try:
            frame = load_tcga_brca(self.tcga_brca_path)
            if frame.empty:
                raise InputValidationError("benchmark table is empty")
            self._benchmark_results = _result("unavailable", "a clinical table alone cannot validate subtype recovery; no empirical benchmark adapter is available",
                                             empirical=False, n_records=len(frame), mean_precision=None, mean_recall=None)
        except Exception as e:
            self._benchmark_results = _result("failed", f"{type(e).__name__}: {e}", empirical=False)

    def _run_literature(self, hypotheses_df, score_metadata):
        if hypotheses_df is None or hypotheses_df.empty:
            self._literature_results = _result("skipped", "no hypotheses")
            return
        columns = [c for c in ["archetype", "process", "pubmed_query", "pubmed_count", "pubmed_flag", "source_status"] if c in hypotheses_df]
        self._literature_results = _result("unavailable", "publication counts annotate prior knowledge; they do not validate hypotheses",
                                          results=hypotheses_df[columns].copy(), empirical=False)

    def _assign_tiers(self, hypotheses_df):
        if hypotheses_df is None or hypotheses_df.empty:
            self._tiered_hypotheses = hypotheses_df
            return
        tiered = hypotheses_df.copy(deep=True)
        tiered["confidence_tier"] = "C"
        tiered["validation_score"] = 0
        tiered["validation_status"] = "unavailable"
        tiered["validation_reason"] = "no hypothesis-specific empirical validation; Tier C is exploratory only"
        tiered["empirically_validated"] = False
        self._tiered_hypotheses = tiered.sort_values("db_support", ascending=False, kind="stable").reset_index(drop=True)

    def survival_results(self):
        return self._survival_results

    def metabric_results(self):
        return self._metabric_results

    def benchmark_results(self):
        return self._benchmark_results

    def literature_results(self):
        return self._literature_results

    def tiered_hypotheses(self):
        return self._tiered_hypotheses
