import os

import pandas as pd

from .layer1 import ConcordanceAnalyzer
from .layer2 import FragilityAnalyzer
from .layer3 import HypothesisGenerator
from .layer4 import ValidationEngine
from .report import BioConvergeResult
from .utils import parse_gmt


def integrate(
    scores,
    patient_col="patient_id",
    score_metadata=None,
    models=None,
    feature_matrix=None,
    pathway_constraints=None,
    outcome=None,
    time_col="OS_days",
    event_col="OS_event",
    dataset_dir=None,
    n_archetypes=3,
    n_bootstrap=1000,
    random_state=42,
    output_dir="output",
):
    skipped = []
    print("bioconverge starting")

    # load scores
    if isinstance(scores, str):
        if not os.path.exists(scores):
            raise FileNotFoundError(f"scores file not found: {scores}")
        score_df = pd.read_csv(scores)
    else:
        score_df = scores.copy()

    score_cols = [c for c in score_df.columns if c != patient_col]
    if len(score_cols) == 0:
        raise ValueError("no score columns found")

    # layer 1
    print("layer1")
    l1 = ConcordanceAnalyzer(score_df, patient_col)
    l1.fit(n_archetypes=n_archetypes, n_bootstrap=n_bootstrap, random_state=random_state)

    # layer 2
    l2 = None
    if models is not None and feature_matrix is not None:
        print("layer2")
        gene_sets = None
        if pathway_constraints is not None:
            if isinstance(pathway_constraints, str) and os.path.exists(pathway_constraints):
                gene_sets = parse_gmt(pathway_constraints)
            elif pathway_constraints == "hallmark" and dataset_dir:
                gmt_path = os.path.join(dataset_dir, "h.all.v2023.2.Hs.symbols.gmt")
                if os.path.exists(gmt_path):
                    gene_sets = parse_gmt(gmt_path)
        import numpy as np
        X = np.array(feature_matrix)
        patient_ids = score_df[patient_col].values
        feat_names = None
        if hasattr(feature_matrix, "columns"):
            feat_names = list(feature_matrix.columns)
        l2 = FragilityAnalyzer(
            models=models,
            X=X,
            patient_ids=patient_ids,
            feature_names=feat_names,
            gene_sets=gene_sets,
        )
        l2.fit(output_dir=output_dir)
    else:
        skipped.append("layer2")
        print("layer2 skipped")

    # layer 3
    l3 = None
    if score_metadata is not None:
        print("layer3")
        archetypes_df = l1.archetypes()
        frag_df = l2.fragility() if l2 is not None else None
        l3 = HypothesisGenerator(score_metadata, archetypes_df, fragility_df=frag_df)
        l3.generate()
    else:
        skipped.append("layer3")
        print("layer3 skipped")

    # layer 4
    l4 = None
    if l3 is not None:
        print("layer4")
        clinical_tar = None
        mut_tar = None
        metabric_path = None
        tcga_brca_path = None
        if dataset_dir:
            for fname in os.listdir(dataset_dir):
                fpath = os.path.join(dataset_dir, fname)
                if "clinical" in fname and fname.endswith(".tar.gz"):
                    clinical_tar = fpath
                elif "Mutation_Packager" in fname and fname.endswith(".tar.gz"):
                    mut_tar = fpath
                elif "metabric" in fname and fname.endswith(".tsv"):
                    metabric_path = fpath
                elif "brca_tcga" in fname and fname.endswith(".tsv"):
                    tcga_brca_path = fpath
        if outcome is not None:
            clinical_tar = outcome
        l4 = ValidationEngine(
            clinical_tar_path=clinical_tar,
            mut_tar_path=mut_tar,
            metabric_path=metabric_path,
            tcga_brca_path=tcga_brca_path,
            dataset_dir=dataset_dir,
            time_col=time_col,
            event_col=event_col,
            patient_col=patient_col,
        )
        km_dir = os.path.join(output_dir, "kaplan_meier")
        l4.validate(
            hypotheses_df=l3.hypotheses(),
            archetypes_df=l1.archetypes(),
            score_df=score_df,
            patient_col=patient_col,
            score_metadata=score_metadata,
            km_output_dir=km_dir,
        )
        hyp = l3.hypotheses()
        if hyp is not None and not hyp.empty:
            l4.compute_replication_rate(hyp)
    else:
        skipped.append("layer4")
        print("layer4 skipped")

    if skipped:
        print(f"skipped: {', '.join(skipped)}")

    result = BioConvergeResult(l1=l1, l2=l2, l3=l3, l4=l4, skipped=skipped)
    print("done")
    return result
