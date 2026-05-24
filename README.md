# bioconverge

Multi-layer biological score integration and hypothesis validation tool for cancer research.

[![PyPI version](https://badge.fury.io/py/bioconverge.svg)](https://pypi.org/project/bioconverge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What it does

When you have multiple biological scores per patient — from transcriptomics, genomics, imaging, or clinical data — they often disagree. One patient might have high immune activity but low genomic instability. Another might show the opposite. bioconverge asks: what biology explains these conflicts, and how confident can we be?

It runs four layers:

- **Layer 1** — finds which scores agree and which conflict per patient, assigns convergence strata and discordance archetypes
- **Layer 2** — computes per-patient biological fragility using model gradient perturbation
- **Layer 3** — generates ranked hypotheses for each archetype by querying Reactome, Enrichr, STRING, GWAS Catalog, and PubMed
- **Layer 4** — validates each hypothesis across four independent sources and assigns Tier A / B / C confidence

## Install

```bash
pip install bioconverge
```

## What data you need

A CSV file with patients as rows and scores as columns:

```
patient_id, immune_score, genomic_score, proliferation_score
TCGA-A1-001, 0.82, 0.31, 0.74
TCGA-A1-002, 0.45, 0.88, 0.22
TCGA-A1-003, 0.61, 0.55, 0.60
```

bioconverge works with any domain where multiple scores exist per sample:

| Domain | Example scores |
|---|---|
| Cancer multi-omics | immune infiltration, TMB, CNA burden, methylation, proliferation |
| Drug response | IC50 from multiple assays on same cell lines |
| Clinical trial | PET scan score, blood biomarker, genomic risk score |
| Single cell | pathway activity scores across multiple pathways per cell |

## Quick start

```python
import bioconverge as bc

result = bc.integrate(
    scores="my_scores.csv",
    patient_col="patient_id",
    score_metadata={
        "immune_score": {
            "process": "T-cell exhaustion",
            "modality": "transcriptomic",
            "genes": ["CD8A", "CD8B", "GZMA", "PRF1"]
        },
        "genomic_score": {
            "process": "DNA repair instability",
            "modality": "genomic",
            "genes": ["BRCA1", "BRCA2", "ATM", "TP53"]
        },
        "proliferation_score": {
            "process": "cell cycle proliferation",
            "modality": "transcriptomic",
            "genes": ["MKI67", "PCNA", "TOP2A", "CDK1"]
        }
    }
)

result.report("output/")
```

## With survival data

```python
result = bc.integrate(
    scores="my_scores.csv",
    patient_col="patient_id",
    score_metadata={...},
    outcome="survival.csv",
    time_col="OS_days",
    event_col="OS_event"
)

result.survival_analysis()
```

## What you get back

```python
result.concordance()       # pairwise score agreement with confidence intervals
result.convergence()       # per-patient convergence index and strata
result.discordance()       # patients with conflicting scores
result.archetypes()        # patient archetype assignments
result.hypotheses()        # ranked hypotheses with Tier A/B/C labels
result.survival_analysis() # KM curves per archetype (if survival data provided)
result.compare_layers()    # patients flagged by both Layer 1 and Layer 2
result.report("output/")   # full HTML report and CSV outputs
```

## Confidence tiers

Each hypothesis is validated across four independent sources:

1. Survival consistency — does this archetype show a survival difference?
2. Replication — does the same hypothesis appear in an independent cohort?
3. Known-biology benchmark — does it match established pathway signatures?
4. Literature — how many PubMed records support this process?

| Tier | Validation sources passed | Interpretation |
|---|---|---|
| A | 3 or 4 | finding — state as result |
| B | 2 | supported hypothesis — state as supported |
| C | 0 or 1 | exploratory — requires experimental follow-up |

## Output files

Running `result.report("output/")` produces:

| File | Contents |
|---|---|
| `summary.html` | interactive concordance matrix and patient strata |
| `per_patient_scores.csv` | convergence index, archetype, fragility per patient |
| `hypotheses_ranked.csv` | full hypothesis table with Tier A/B/C labels and database links |
| `reproducibility_log.txt` | all API queries with timestamps for reproducibility |
| `kaplan_meier/` | survival plots per archetype as PNG |
| `fragility_topology.png` | UMAP of patient fragility clusters |

## TNBC demonstration

The full demonstration on 116 TCGA-BRCA triple-negative breast cancer patients is in `notebooks/01_TNBC_demo.ipynb`.

Scores used:

| Score | Source | Genes |
|---|---|---|
| proliferation_score | MKI67 RNA-seq expression | MKI67 |
| immune_score | cytotoxic T-cell signature | CD8A, CD8B, GZMA, PRF1 |
| emt_score | EMT hallmark gene set mean | 200 genes |
| mutation_score | TMB from MAF files | — |
| genomic_score | Fraction Genome Altered | — |
| size_score | Longest Dimension (clinical) | — |

Results: 6 Tier A findings, 9 Tier B supported hypotheses, 3 Tier C exploratory.

Datasets required:

| Dataset | Source |
|---|---|
| TCGA-BRCA clinical | https://portal.gdc.cancer.gov/projects/TCGA-BRCA |
| TCGA-BRCA mutations | http://gdac.broadinstitute.org |
| TCGA-BRCA RNA-seq | http://gdac.broadinstitute.org |
| METABRIC clinical | https://www.cbioportal.org/study/summary?id=brca_metabric |
| Lehmann subtypes | https://www.cbioportal.org/study/summary?id=brca_tcga |
| MSigDB Hallmark | https://www.gsea-msigdb.org/gsea/msigdb |

## Parameters

| Parameter | Required | Description |
|---|---|---|
| `scores` | yes | path to CSV with patient scores |
| `patient_col` | yes | name of patient ID column |
| `score_metadata` | yes | dict with process, modality, genes per score |
| `models` | no | trained sklearn/PyTorch/TensorFlow models for Layer 2 |
| `feature_matrix` | no | feature matrix X for Layer 2 fragility |
| `pathway_constraints` | no | "hallmark" to use MSigDB Hallmark gene sets |
| `outcome` | no | path to survival CSV |
| `time_col` | no | name of time column in survival CSV |
| `event_col` | no | name of event column in survival CSV |

If a parameter is not provided, that layer is skipped gracefully and noted in the report.

## Requirements

```
numpy, pandas, scipy, scikit-learn, matplotlib, seaborn,
lifelines, umap-learn, hdbscan, requests, plotly, torch,
tensorflow, jupyter
```

## License

MIT

## Citation

If you use bioconverge in your research, please cite:

```
Koushik, C. (2026). bioconverge: Multi-layer biological score integration
and hypothesis validation for cancer research.
https://github.com/chykoushik/bioconverge
```
