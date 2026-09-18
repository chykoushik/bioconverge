# Getting started

BioConverge 0.2.0 works with biological scores supplied by the user.
Start with a score table. Add models, metadata, or survival data when available.

## Install

Use Python 3.10 or newer.

```bash
pip install bioconverge
```

The core installation includes pandas, NumPy, SciPy, scikit-learn, Matplotlib,
lifelines, and requests. Install extras only for the features you need.

| Command | Adds |
|---|---|
| `pip install "bioconverge[topology]"` | UMAP and HDBSCAN |
| `pip install "bioconverge[report]"` | Plotly reports |
| `pip install "bioconverge[torch]"` | PyTorch |
| `pip install "bioconverge[tensorflow]"` | TensorFlow |
| `pip install "bioconverge[notebook]"` | JupyterLab, nbformat, and Plotly |
| `pip install "bioconverge[test]"` | pytest, build, and twine |

You can combine extras with `pip install "bioconverge[topology,report]"`.
Extras install dependencies. They do not supply data or trained models.

## Run an analysis

These values are illustrative. They are not biological findings.
The example runs without network requests.

```python
import pandas as pd
import bioconverge as bc

scores = pd.DataFrame({
    "patient_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
    "immune_score": [0.8, 0.7, 0.2, 0.3, 0.6, 0.1],
    "genomic_score": [0.2, 0.4, 0.9, 0.7, 0.5, 0.8],
})

checks = bc.validate(scores, n_archetypes=2, random_state=42)
print(checks.to_dict())
checks.raise_for_errors()

result = bc.integrate(
    scores, n_archetypes=2, n_bootstrap=100, random_state=42
)
print(result.statuses())
print(result.archetypes())
print(result.concordance())
```

`validate()` checks inputs without running the analysis or querying databases.
`integrate()` validates again before running. Here only Layer 1 runs.
The other layers report why they did not run.

## Read and save results

Continue with `result` from the example above.

```python
print(result.convergence())
print(result.stability())
print(result.metadata()["version"])
result.report("first_report")
```

`archetypes()` and `concordance()` return DataFrames. `stability()` returns a
dictionary with bootstrap counts, mean ARI, and other diagnostics.
`statuses()` and `metadata()` return dictionaries.
An unavailable optional result can be `None` or have an unavailable state.
See [the layers guide](layers.md) for each result type.

`report()` returns the output directory as a string.
The directory must be new or empty. `integrate()` does not write report files.
Its `output_dir` parameter sets the default destination for a later `report()` call.

Reports include an HTML summary, available CSV tables, and optional figures.
They also include JSON files for metadata, states, API responses, query logs,
validation results, and file hashes. A report does not mean all layers succeeded.

## Next steps

- Read [input data](input-data.md) before using your own scores.
- Use [examples](examples.md) for models, survival data, and annotations.
- Read [validation](validation.md) to interpret states and evidence.
- Follow [reproducibility](reproducibility.md) when saving a research analysis.
- Use [troubleshooting](troubleshooting.md) when an input or layer fails.

[Back to the README](../README.md)
