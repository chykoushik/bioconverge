# Input data

## Score tables and patient IDs

Pass a pandas DataFrame or a CSV path as `scores`.
Each row represents one patient. Every column except the patient ID is a score.
Select score columns before calling BioConverge if your file also holds other data.

| patient_id | immune_score | genomic_score |
|---|---:|---:|
| P1 | 0.8 | 0.2 |
| P2 | 0.7 | 0.4 |
| P3 | 0.2 | 0.9 |

These values are illustrative. Three patients are too few for pairwise
concordance, which needs at least five paired observations.

The ID column defaults to `patient_id`. Set `patient_col` for another name.
IDs must be unique, nonmissing, and consistently typed. Empty or whitespace-only
IDs are rejected. Score column names must be unique, nonempty strings.
Result tables use `patient_id` even when the input uses another name.

Duplicate score IDs are errors. BioConverge does not choose a row or combine
patients for you. Resolve repeated measurements using a rule suitable for your
study. Keep that rule with your input records.

CSV input uses pandas type inference. Load the file yourself with an explicit
ID dtype if leading zeros or text IDs must be preserved.

## Numeric and missing values

Scores must be numeric or convertible to numbers. They need not be between zero
and one. Negative scores are allowed. Choose score definitions and directions
that fit your analysis.

| Input condition | Behavior |
|---|---|
| Some missing scores | Allowed and recorded in the validation report |
| Entirely missing row or column | Rejected |
| Positive or negative infinity | Rejected |
| Nonnumeric score | Rejected |
| One constant column among varying scores | Retained and reported |
| All score columns constant | Rejected |

Clustering fills missing scores with each column's median, then standardizes
the columns. Concordance uses paired observed values without this imputation.
A correlation involving a constant score is undefined.

Convergence uses each patient's observed scores. It needs at least two scores
for that patient. A tie between the two cohort stratum thresholds produces
`unknown` strata. This is not evidence that patients are biologically identical.

`n_archetypes` defaults to 3. It must be a positive integer no larger than the
number of distinct patient profiles after median imputation.
`n_bootstrap` must be a positive integer.

## Check invalid inputs

This example reports errors without starting an analysis.

```python
import pandas as pd
import bioconverge as bc

scores = pd.DataFrame({
    "patient_id": ["P1", "P2", "P3"],
    "score_a": [0.1, 0.5, 0.9],
    "score_b": [0.8, 0.4, 0.2],
})

duplicate = scores.copy()
duplicate.loc[1, "patient_id"] = "P1"
infinite = scores.copy()
infinite.loc[0, "score_a"] = float("inf")
missing_column = scores.copy()
missing_column["score_a"] = float("nan")

for invalid in [duplicate, infinite, missing_column]:
    checks = bc.validate(invalid, n_archetypes=2)
    assert not checks.valid
    print(checks.errors)
```

## Models and features

Supply `models` and `feature_matrix` together. A model can be a supported trained
predictor or a dictionary of named predictors. Dictionary names must be nonempty
strings. The name `score` is reserved.

A feature matrix must be two dimensional, numeric, and finite. It must have one
row per patient and at least one feature. Missing model features are rejected.

For a DataFrame, the index must contain exactly the score patient IDs.
BioConverge reorders feature rows by ID. Indices and column names must be unique.
For an array, row order must already match the score table.
Feature columns must follow the training order. When a model exposes
`feature_names_in_`, a DataFrame with those exact ordered names is required.
Known model feature dimensions are checked.

Scikit-learn classifiers need `predict_proba`. Binary classifiers use output
column 1 by default. Multiclass models require `target_class` as a column index.
Neural models with multiple output columns are treated as logits and passed
through softmax. Use a compatible output contract. A single output must give one
finite target per patient. Model training is outside the integration workflow.

`pathway_constraints` accepts a dictionary of gene lists or a GMT path.
It requires models and features. Feature names must overlap pathway genes to
produce pathway results. The value `"hallmark"` uses
`h.all.v2023.2.Hs.symbols.gmt` under `dataset_dir`, or the current directory when
`dataset_dir` is omitted.

## Clinical data

`outcome` accepts a DataFrame or a supported clinical file.
CSV, TSV, and text tables are supported. A clinical tar archive must contain
exactly one matching `clinical.tsv` table.

| patient_id | OS_days | OS_event |
|---|---:|---:|
| P1 | 120 | 1 |
| P2 | 240 | 0 |
| P3 | 360 | 1 |

These endpoint values are illustrative. For overall survival, 1 marks death and
0 marks censoring. Use `time_col` and `event_col` for other column names.
Use `event_mapping` for labels that are not recognized. Common labels include
`Alive`, `Dead`, `0:LIVING`, and `1:DECEASED`.
With a custom `patient_col`, the clinical table must use that ID column name too.

Set `time_unit` to `days`, `months`, or `years` when needed.
Detected month columns use months. Other combined duration columns default to
days. Returned times are in days, using 30.44 days per month and 365.25 per year.
When separate death and follow-up columns are detected, the event selects the
appropriate duration.

Missing outcomes are excluded with counts. Identical endpoint rows are
deduplicated. Conflicting endpoint rows for one patient are rejected.
Patient IDs must be present. Unrecognized events and negative or infinite times
are errors. At least one score patient must match the valid outcome IDs during
preflight. More observations are needed to run survival comparisons.

The clinical table is checked before it is matched to the score cohort.
An invalid record outside the analysis cohort can therefore stop validation.
If your analysis uses a subset, supply its clinical rows under a documented
cohort rule. Do not alter endpoint values to make a test pass.

## Annotation metadata

`score_metadata` is a nonempty dictionary keyed by existing score columns.
Each entry is a dictionary. Optional `process` and `modality` values must be
nonempty strings. `genes` must be a list or tuple of nonempty gene names.
Metadata need not cover every score.

`dataset_dir` supports discovery of specific TCGA and METABRIC files.
It is not a general dataset importer. Ambiguous filename matches are rejected.
Prefer an explicit `outcome` for your own clinical table.

[Examples](examples.md) | [Validation](validation.md) | [README](../README.md)
