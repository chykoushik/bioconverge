# Troubleshooting

## Installation problems

BioConverge 0.2.0 requires Python 3.10 or newer.
Check that pip belongs to the Python environment running your analysis.

```bash
python --version
python -m pip --version
python -m pip show bioconverge
python -m pip check
```

If dependencies conflict, use a separate environment and install BioConverge
there. Do not use `--no-deps` for a normal installation.
Core dependencies include lifelines, so a missing lifelines import means the
environment is incomplete or a different Python interpreter is being used.

## Optional dependency problems

Install `bioconverge[topology]` for UMAP and HDBSCAN.
Install `bioconverge[report]` for interactive Plotly output.
Use `bioconverge[torch]` or `bioconverge[tensorflow]` for the matching neural model.
The available extras are listed in [getting started](getting-started.md).

Missing topology packages leave topology unavailable without blocking core
fragility. A broken import or topology runtime failure can emit a warning.
Do not suppress that warning without checking its cause.
Without Plotly, reports use table output where interactive plots would appear.

## Invalid input

Call `validate()` with the same parameters you plan to pass to `integrate()`.
Read `errors`, `warnings`, and `details`. Fix the reported error and check again.
Unknown parameter names are errors. Input validation can stop at the first error.

Do not pass text annotations as score columns. Select only the ID and numeric
score columns. Use `patient_col` if your identifier has another name.

## Duplicate patient IDs

Score IDs must be unique. Decide whether duplicates are errors or repeated
measurements that need a study-specific aggregation rule.
Do not remove rows just to pass validation.
Clinical endpoints have separate rules for identical and conflicting duplicates.
See [input data](input-data.md).

## NaN or infinite values

Some missing scores are allowed. Entirely missing rows or columns are not.
Model features must have no missing values.
Infinity is rejected in scores, features, and survival times.
Check the calculation that produced an infinite value before deciding how to
handle it. Replacing it with zero can change the meaning of the analysis.

A NaN result can mean the calculation was undefined. For example, correlations
need five paired observations and nonconstant scores.
Read the result's reason rather than interpreting NaN as no association.

## Feature mismatch

A feature DataFrame needs patient IDs as its index, not a row counter.
Its IDs must match the score table exactly. Rows can be reordered by BioConverge,
but columns must remain in training order. Arrays must already follow patient
order. Check both the number of features and their names.

## Layer 2 is skipped, failed, or partly unavailable

Supply both `models` and `feature_matrix` to request Layer 2.
The package does not train a model from your scores automatically.
Check model outputs, target selection, and backend dependencies if it fails.
Scikit-learn classifiers need probabilities, not only class labels.

An unavailable topology component does not mean fragility failed.
Topology needs at least four patients and three distinct gradients.
Pathway summaries need gene names that match feature names.

## External services are unavailable

Inspect Layer 3 component states and each annotation's `source_status`.
HTTP errors, timeouts, or malformed responses can fail a query.
Uncached queries are unavailable in replay mode.
No supplied genes or disease context can also limit some sources.

Check network access and service status before requesting fresh results.
Preserve previous response caches for analyses that used them.
Unavailable evidence is not negative biological evidence.

## Validation is unavailable

METABRIC replication and empirical Lehmann recovery are not implemented in 0.2.0.
Supplying a clinical table does not enable those methods.
Tier C annotations remain unvalidated even when database queries succeed.

For survival, inspect patient matching, event counts, and group sizes.
Each comparison needs at least three patients in the archetype and three outside
it. Invalid times or conflicting endpoints are errors.
Validation checks the supplied clinical table before selecting matched patients.

## Reports and support

If `report()` raises `FileExistsError`, choose a new or empty directory.
Do not overwrite a prior analysis to hide missing or failed components.

For help, use the [issue tracker](https://github.com/chykoushik/bioconverge/issues).
Include the package version, environment, error message, and a small example
without private patient data or credentials.

[Validation](validation.md) | [README](../README.md)
