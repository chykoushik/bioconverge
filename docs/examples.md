# Examples

All values below are illustrative. They demonstrate the API and are not cancer
findings or biological results. Run the setup once in a Python session, then run
any example below in that session. No example needs an external dataset or API.

## Setup

```python
import pandas as pd
import bioconverge as bc

scores = pd.DataFrame({
    "patient_id": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
    "immune_score": [0.1, 0.2, 0.15, 0.25, 0.8, 0.9, 0.85, 0.95],
    "genomic_score": [0.2, 0.3, 0.25, 0.35, 0.7, 0.8, 0.75, 0.85],
})
```

## Basic integration

```python
result = bc.integrate(
    scores, n_archetypes=2, n_bootstrap=20, random_state=42
)
print(result.concordance())
print(result.archetypes())
```

## Validate before analysis

```python
options = {"n_archetypes": 2, "n_bootstrap": 20, "random_state": 42}
checks = bc.validate(scores, **options)
print(checks.to_dict())
checks.raise_for_errors()
checked_result = bc.integrate(scores, **options)
```

## Repeat an analysis

This checks numerical outputs in the same environment. Report timestamps are
not expected to match across separate executions.

```python
first = bc.integrate(scores, n_archetypes=2, n_bootstrap=20, random_state=42)
second = bc.integrate(scores, n_archetypes=2, n_bootstrap=20, random_state=42)
pd.testing.assert_frame_equal(first.archetypes(), second.archetypes())
pd.testing.assert_frame_equal(first.concordance(), second.concordance())
assert first.stability() == second.stability()
```

## Inspect states

```python
partial = bc.integrate(
    scores, n_archetypes=2, n_bootstrap=20, on_error="record"
)
for layer, entry in partial.statuses().items():
    print(layer, entry["state"], entry["reason"])
    for component, state in entry.get("analyses", {}).items():
        print(component, state["state"], state["reason"])

if partial.fragility() is None:
    print("No model fragility result")
```

This example has no trained model, so Layer 2 is skipped.
`on_error="record"` does not bypass preflight input errors.

## Use a custom patient column and CSV

The example writes its illustrative CSV in a temporary directory.
For your analysis, pass the path to your own score file.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

custom_scores = scores.rename(columns={"patient_id": "sample_id"})
with TemporaryDirectory() as directory:
    path = Path(directory) / "scores.csv"
    custom_scores.to_csv(path, index=False)
    custom = bc.integrate(
        path, patient_col="sample_id", n_archetypes=2,
        n_bootstrap=20, random_state=42
    )
    print(custom.archetypes())
```

## Add model fragility

This small example trains a regression model only to show the input contract.
It does not assess predictive performance. Three patients are intentionally too
few for topology. Core fragility still runs without optional topology packages.

```python
from sklearn.linear_model import LinearRegression

model_scores = scores.iloc[:3].copy()
features = model_scores.set_index("patient_id")[["immune_score", "genomic_score"]]
targets = [0.2, 0.5, 0.3]
model = LinearRegression().fit(features, targets)

model_result = bc.integrate(
    model_scores,
    models={"regression": model},
    feature_matrix=features.iloc[::-1],
    n_archetypes=2,
    n_bootstrap=20,
    random_state=42,
)
print(model_result.fragility())
print(model_result.statuses()["layer2"])
```

Feature rows are reversed here to show alignment by patient ID.
Feature columns retain their training order.

## Add survival data

The illustrative endpoints below are not observed clinical data.
Survival uses the core installation and does not require score metadata.

```python
outcomes = pd.DataFrame({
    "patient_id": scores["patient_id"],
    "OS_days": [120, 200, 300, 400, 150, 250, 350, 450],
    "OS_event": [1, 0, 1, 0, 1, 0, 1, 0],
})
survival_result = bc.integrate(
    scores, outcome=outcomes, time_col="OS_days", event_col="OS_event",
    time_unit="days", n_archetypes=2, n_bootstrap=20, random_state=42
)
survival = survival_result.survival_analysis()
print(survival["state"], survival["reason"])
if survival["state"] == "completed":
    print(survival["results"])
```

## Add offline metadata annotations

No cached evidence is supplied here. The annotations are created, but database
evidence is unavailable. This is not evidence against the named process.

```python
annotated = bc.integrate(
    scores,
    score_metadata={
        "immune_score": {
            "process": "immune activation",
            "modality": "transcriptomic",
            "genes": ["CD8A", "CD8B", "GZMA", "PRF1"],
        }
    },
    disease_context="glioblastoma",
    replay_only=True,
    n_archetypes=2,
    n_bootstrap=20,
    random_state=42,
)
print(annotated.hypotheses()[
    ["archetype", "process", "evidence_scope", "confidence_tier"]
])
print(annotated.statuses()["layer3"])
```

Use a process and disease context that fit your study.
To request current database evidence, set `replay_only=False`.
See [reproducibility](reproducibility.md) for saving and reusing responses.

[Getting started](getting-started.md) | [Input data](input-data.md) | [README](../README.md)
