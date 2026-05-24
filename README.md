# bioconverge

Multi-layer biological score integration and hypothesis validation tool for cancer research.

## What it does

bioconverge takes multiple biological scores per patient (transcriptomic, genomic, imaging, clinical) and runs them through four layers:

- **Layer 1** — finds which scores agree and which conflict per patient, assigns convergence strata and discordance archetypes
- **Layer 2** — computes per-patient biological fragility using model gradient perturbation
- **Layer 3** — generates ranked hypotheses for each archetype by querying Reactome, Enrichr, STRING, GWAS Catalog, and PubMed
- **Layer 4** — validates each hypothesis across four independent sources (survival, replication cohort, known-biology benchmark, literature) and assigns Tier A / B / C confidence

## Install

```bash
pip install bioconverge
```

## Quick start

```python
import bioconverge as bc

result = bc.integrate(
    scores="my_scores.csv",
    patient_col="patient_id",
    score_metadata={
        "immune_score": {"process": "T-cell exhaustion", "modality": "transcriptomic", "genes": ["CD8A", "CD8B", "GZMA", "PRF1"]},
        "genomic_score": {"process": "DNA repair instability", "modality": "genomic", "genes": ["BRCA1", "BRCA2", "ATM"]}
    },
    outcome="survival.csv",
    time_col="OS_days",
    event_col="OS_event"
)

result.concordance()
result.hypotheses()
result.report("output/")
```

## TNBC demonstration

The full demonstration on 116 TCGA-BRCA triple-negative breast cancer patients is in `notebooks/01_TNBC_demo.ipynb`.

Datasets required:

| Dataset | Source |
|---|---|
| TCGA-BRCA clinical | https://portal.gdc.cancer.gov/projects/TCGA-BRCA |
| TCGA-BRCA mutations | http://gdac.broadinstitute.org |
| TCGA-BRCA RNA-seq | http://gdac.broadinstitute.org |
| METABRIC clinical | https://www.cbioportal.org/study/summary?id=brca_metabric |
| Lehmann subtypes | https://www.cbioportal.org/study/summary?id=brca_tcga |
| MSigDB Hallmark | https://www.gsea-msigdb.org/gsea/msigdb |

## Output

Running `result.report("output/")` produces:

- `summary.html` — interactive concordance matrix and patient strata
- `per_patient_scores.csv` — convergence index, archetype, fragility per patient
- `hypotheses_ranked.csv` — full hypothesis table with Tier A/B/C labels
- `reproducibility_log.txt` — all API queries with timestamps
- `kaplan_meier/` — survival plots per archetype
- `fragility_topology.png` — UMAP of patient fragility clusters

## Confidence tiers

| Tier | Meaning | Validation sources passed |
|---|---|---|
| A | finding | 3 or 4 |
| B | supported hypothesis | 2 |
| C | exploratory | 0 or 1 |

## Requirements

```
numpy, pandas, scipy, scikit-learn, matplotlib, seaborn,
lifelines, umap-learn, hdbscan, requests, plotly, torch,
tensorflow, jupyter
```

## License

MIT
