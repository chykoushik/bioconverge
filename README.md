# BioConverge

[![PyPI version](https://img.shields.io/pypi/v/bioconverge.svg)](https://pypi.org/project/bioconverge/)
[![Python versions](https://img.shields.io/pypi/pyversions/bioconverge.svg)](https://pypi.org/project/bioconverge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

BioConverge is a Python framework for patient-level biological score integration.
It measures concordance, finds score archetypes, and adds hypothesis annotations.
It also analyzes model fragility when trained models and their inputs are supplied.
Analysis states make missing evidence and failed operations visible.

BioConverge supports bioinformatics and computational biology research, including
cancer informatics and multi-omics score analysis. It works with scores prepared
by the user. Its annotations are exploratory, not empirically validated findings.

## Installation

Requires **Python 3.10 or newer**.

```bash
pip install bioconverge
```

Optional features have separate extras. For example,
`pip install "bioconverge[topology,report]"` adds gradient topology and interactive
reports. See [installation options](docs/getting-started.md).

## Quick start

These values are illustrative. They are not research results.
The example runs without external requests.

```python
import pandas as pd
import bioconverge as bc

scores = pd.DataFrame({
    "patient_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
    "immune_score": [0.8, 0.7, 0.2, 0.3, 0.6, 0.1],
    "genomic_score": [0.2, 0.4, 0.9, 0.7, 0.5, 0.8],
})
checks = bc.validate(scores, n_archetypes=2, random_state=42)
checks.raise_for_errors()
result = bc.integrate(scores, n_archetypes=2, n_bootstrap=100, random_state=42)
print(result.concordance())
print(result.archetypes())
print(result.statuses())
result.report("bioconverge_report")
```

Use one row per patient and one numeric column per score. Patient IDs must be
unique. Reports need a new or empty directory.
See [input requirements](docs/input-data.md) and [working examples](docs/examples.md).

## Four layers

| Layer | What it does | Needed inputs |
|---|---|---|
| 1. Concordance and archetypes | Score correlations, convergence, KMeans clusters, and bootstrap stability | Patient scores |
| 2. Model fragility | Local model sensitivity and optional gradient topology | Trained models and aligned features |
| 3. Hypothesis annotation | Process metadata with database evidence | Score metadata |
| 4. Validation-aware analysis | Survival association and evidence availability | Outcomes or other validation inputs |

Optional analyses depend on their inputs and dependencies.
Missing topology packages do not block core model fragility.
Read [the layer guide](docs/layers.md) for outputs and limits.

## Validation and evidence

`validate()` checks inputs without running analyses or querying databases.
`result.statuses()` records `completed`, `skipped`, `failed`, and `unavailable`
states. Inspect component reasons as well as layer states.

STRING, Enrichr, and other external resources can fail or return no hits.
Unavailable evidence is not negative biological evidence.
Database annotation is not empirical validation. Survival association does not
validate every annotated process. Tier C means exploratory and unvalidated.
Version 0.2.0 does not provide Tier A/B promotion, empirical METABRIC replication,
or an empirical Lehmann benchmark.

See [validation](docs/validation.md) and [reproducibility](docs/reproducibility.md).

## Documentation

- [Getting started](docs/getting-started.md)
- [Input data](docs/input-data.md)
- [Analysis layers](docs/layers.md)
- [Validation and states](docs/validation.md)
- [Examples](docs/examples.md)
- [Reproducibility](docs/reproducibility.md)
- [Troubleshooting](docs/troubleshooting.md)

## Citation

Cite the software as **Koushik C. BioConverge, version 0.2.0**, with the
[repository URL](https://github.com/chykoushik/bioconverge).
Record the version actually used and cite the data resources separately.
No journal citation or DOI is specified for this release.

## License and support

BioConverge uses the [MIT License](LICENSE).
Report reproducible problems through the
[issue tracker](https://github.com/chykoushik/bioconverge/issues).
