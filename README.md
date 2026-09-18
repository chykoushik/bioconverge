# BioConverge

[![PyPI version](https://img.shields.io/pypi/v/bioconverge.svg)](https://pypi.org/project/bioconverge/)
[![Python versions](https://img.shields.io/pypi/pyversions/bioconverge.svg)](https://pypi.org/project/bioconverge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

BioConverge is a Python framework for patient-level biological score integration,
concordance analysis, archetype discovery, hypothesis annotation, and model
fragility analysis when trained models and their inputs are available. It supports
validation-aware computational analysis with explicit evidence and failure states.

For bioinformatics and computational biology research, including cancer informatics
and multi-omics score analysis, BioConverge helps explore patient heterogeneity and
patient stratification. It operates on supplied scores, not a general raw multi-omics
processing pipeline. Hypothesis generation is exploratory metadata annotation,
not empirical validation or biomarker discovery.

## Installation

Requires **Python 3.10 or newer**.

```bash
pip install bioconverge
```

| Optional extra | Installation | Purpose |
|---|---|---|
| `topology` | `pip install "bioconverge[topology]"` | UMAP and HDBSCAN for model-gradient topology |
| `report` | `pip install "bioconverge[report]"` | Interactive Plotly reports |
| `torch` | `pip install "bioconverge[torch]"` | PyTorch model analysis |
| `tensorflow` | `pip install "bioconverge[tensorflow]"` | TensorFlow model analysis |
| `notebook` | `pip install "bioconverge[notebook]"` | JupyterLab and notebook/report dependencies |
| `test` | `pip install "bioconverge[test]"` | Tests, builds, and metadata checks |

Extras can be combined: `pip install "bioconverge[topology,report]"`.
Survival dependencies are included in the core installation.

## Quick start

These illustrative values demonstrate the API; they are not research results.
Replace them with measured scores for your study.

```python
import pandas as pd
import bioconverge as bc

scores = pd.DataFrame({
    "patient_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
    "immune_score": [0.8, 0.7, 0.2, 0.3, 0.6, 0.1],
    "genomic_score": [0.2, 0.4, 0.9, 0.7, 0.5, 0.8],
})

result = bc.integrate(scores, n_archetypes=2, n_bootstrap=100, random_state=42)
print(result.concordance())
print(result.archetypes())
print(result.statuses())
result.report("bioconverge_report")
```

This runs Layer 1 without external requests. Reports require a new or empty
directory. `integrate()` itself does not write report files.

## Input data and validation

Supply a pandas DataFrame or CSV path with one row per patient, a unique patient
identifier, and numeric score columns. For example:

| patient_id | immune_score | genomic_score |
|---|---:|---:|
| P1 | 0.8 | 0.2 |
| P2 | 0.7 | 0.4 |
| P3 | 0.2 | 0.9 |

Choose scientifically meaningful scores and record their derivation. Validate
the `scores` DataFrame above before execution:

```python
checks = bc.validate(scores, n_archetypes=2, random_state=42)
print(checks.to_dict())
checks.raise_for_errors()
```

`validate()` performs local preflight checks without API requests or output writes.
Its report exposes `valid`, `errors`, `warnings`, and `details`. Invalid integration
inputs raise `InputValidationError`.

- IDs must be unique, nonmissing, and consistently typed; score names must be unique.
- Partial missing scores are allowed. Concordance uses paired observations;
  clustering median-imputes each score. Convergence uses observed scores and is
  undefined with fewer than two available scores for a patient.
- Nonnumeric or infinite scores and entirely missing rows or columns are rejected.
  Constant columns are reported and retained; all-constant inputs are rejected.
- Correlations with fewer than five paired observations or a constant score are
  undefined, with reasons recorded. Undefined values are not evidence of disagreement.
- `n_archetypes` cannot exceed the number of distinct usable patient profiles.
- Model features must be finite. DataFrame indices must match patient IDs exactly
  and are reordered by ID; arrays must follow score-row order. Feature columns
  must follow the model's training order.

## Four analysis layers

| Layer | Current behavior | Required inputs |
|---|---|---|
| 1. Concordance and archetypes | Spearman correlations with bootstrap intervals, convergence strata, standardized KMeans archetypes, and bootstrap stability | Patient score matrix |
| 2. Model fragility | Input perturbation/gradient sensitivity, trajectories, optional pathway aggregation, and optional UMAP/HDBSCAN topology | Trained `models` and aligned `feature_matrix`; matching backend dependencies |
| 3. Hypothesis annotation | Metadata-based process annotations using Reactome, Enrichr, STRING, GWAS Catalog, and PubMed | `score_metadata`; explicit `disease_context` for literature queries |
| 4. Validation-aware analysis | Archetype-versus-rest survival association, evidence availability, and exploratory annotation tiers | Survival `outcome` and/or relevant validation inputs |

Optional analyses execute only when their required inputs and dependencies are
available. Missing topology dependencies do not prevent core fragility analysis.
Missing inputs are reported; invalid supplied inputs are not silently ignored.

Layer 2 supports compatible scikit-learn predictors and optional neural backends.
Scikit-learn classifiers must provide probabilities; multiclass outputs require
an explicit `target_class` column index. Fragility is local model sensitivity,
not patient risk or distance to a decision boundary.

Layer 4 reports raw log-rank and Benjamini-Hochberg-adjusted p-values. Survival
association does not validate every annotated process. Version 0.2.0 does not
implement empirical METABRIC replication or Lehmann subtype recovery. Tier C
means exploratory, unvalidated annotation; Tier A/B promotion is disabled.
See `CHANGES.md` in the source distribution for changes affecting earlier analyses.

## Custom data

For a CSV with a `sample_id` identifier and an `immune_score` column, alongside
other numeric scores:

```python
custom = bc.integrate(
    "my_scores.csv",
    patient_col="sample_id",
    score_metadata={
        "immune_score": {
            "process": "immune activation",
            "modality": "transcriptomic",
            "genes": ["CD8A", "CD8B", "GZMA", "PRF1"],
        }
    },
    disease_context="glioblastoma",
    random_state=42,
    replay_only=True,
)
print(custom.hypotheses())
```

Use metadata and a disease context appropriate to your study. This example runs
offline: uncached external annotations are unavailable. Set `replay_only=False`
to query current external resources.

For survival association, pass `outcome="survival.csv"`, `time_col`, `event_col`,
and `time_unit="days"`, `"months"`, or `"years"`. Supply `event_mapping` for
nonstandard labels. Common column names can also be detected. Missing outcomes
are excluded with counts; identical endpoint records are deduplicated. Conflicting
records, invalid events, and negative or infinite durations are rejected.
Missing mutation assessment remains unknown.

## Results and analysis states

`integrate()` returns a `BioConvergeResult`:

| API | Contents |
|---|---|
| `concordance()` | Score-pair correlations, intervals, counts, and reasons |
| `convergence()`, `discordance()` | Convergence indices, strata, and low-convergence patients |
| `archetypes()`, `stability()` | Cluster assignments and bootstrap diagnostics |
| `fragility()`, `fragility_pathways()` | Available model-sensitivity results |
| `hypotheses()` | Available exploratory metadata annotations |
| `survival_analysis()` | Survival results and availability details |
| `compare_layers()` | Patients flagged by both low convergence and high fragility |
| `statuses()`, `metadata()` | Layer/component states, reasons, inputs, parameters, and environment |
| `report(path)` | HTML, CSV, JSON, and available figures |

Inspect component states even when their parent layer completed:

| State | Meaning |
|---|---|
| `completed` | Operation ran; this does not establish biological validity |
| `skipped` | Operation was not requested or required inputs were not supplied |
| `failed` | Execution or supplied data caused an error |
| `unavailable` | No usable result or supported validation method is available |

Execution failures normally raise `AnalysisError` with layer statuses. Use
`on_error="record"` to retain partial results and failure reasons. Component
failures can also appear within returned results; inspect them explicitly.

Reports include `summary.html`, `concordance.csv`, `per_patient_scores.csv`,
available `hypotheses_ranked.csv`, and optional figures. JSON files preserve run
metadata, statuses, validation results, API responses, query logs, and an artifact
SHA-256 manifest. Unavailable analyses do not produce invented outputs.

## External resources and reproducibility

Online annotation sends process terms and supplied gene lists to external
resources. STRING evidence requires returned protein interactions; Enrichr uses
adjusted-significant enrichment terms. Requests have timeouts and bounded retries.
Failed, unavailable, and successful empty responses are distinguished. An
unavailable external service is **not negative biological evidence**.

Pass `api_cache={}` to retain responses, or reload a report's `api_responses.json`
and supply it with `replay_only=True` for offline evidence replay. Live databases
can change independently of the random seed.

`random_state` must be an integer from 0 through `2**32 - 1`. It controls package
clustering, bootstrap sampling, bootstrap KMeans seeds, and optional topology.
Stability uses at most 200 bootstrap fits and reports actual counts; concordance
uses the requested `n_bootstrap`. Reproducible research also requires unchanged
data, preprocessing, trained models, dependencies, and numerical runtime settings.
The seed does not control external model training or guarantee agreement across
hardware and library versions. Record your environment with `pip freeze`.

## Citation, license, and support

When using BioConverge, cite the software as **Koushik C. BioConverge, version
0.2.0**, with the [repository URL](https://github.com/chykoushik/bioconverge).
Record the version actually used and cite the underlying data resources separately.
No journal citation or DOI is specified for this release.

BioConverge is distributed under the **MIT License**; see the bundled `LICENSE`.
Report reproducible problems through the
[issue tracker](https://github.com/chykoushik/bioconverge/issues).
