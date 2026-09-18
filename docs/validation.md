# Validation and analysis states

BioConverge separates input checks, execution states, and scientific evidence.
A valid input or completed operation does not establish a biological conclusion.

## Check inputs with validate()

This example uses illustrative scores.

```python
import pandas as pd
import bioconverge as bc

scores = pd.DataFrame({
    "patient_id": ["P1", "P2", "P3"],
    "score_a": [0.1, 0.5, 0.9],
    "score_b": [0.8, 0.4, 0.2],
})
checks = bc.validate(scores, n_archetypes=2, n_bootstrap=20, random_state=42)
print(checks.valid)
print(checks.errors)
print(checks.warnings)
print(checks.details)
checks.raise_for_errors()
```

`validate()` returns a `ValidationReport`. `to_dict()` returns its four fields.
`raise_for_errors()` raises `InputValidationError` when errors exist.
Otherwise it returns the report itself.

Checks cover score values, IDs, cluster and seed parameters, supplied model
features, metadata, pathway files, and explicit survival inputs.
Pass the same input options to `validate()` and `integrate()`.
The report can stop at the first detected error. Fix it and validate again.

Preflight reads supplied files but does not fit models, query APIs, or write
outputs. It cannot guarantee that a model's predictions, a dependency, or an
external service will work later.

Warnings in this report are strings, such as notices about partial missingness
or constant scores. They do not make `valid` false by themselves.

## Read execution states

`result.statuses()` returns a dictionary keyed by layer.
Each entry has `state` and `reason`. Layer entries can also contain an `analyses`
dictionary of component states.

| State | Meaning |
|---|---|
| `completed` | The operation ran and returned its output |
| `skipped` | Required inputs were not supplied or the component was not requested |
| `failed` | Data or execution caused an error |
| `unavailable` | No usable result or supported method is available |

Always inspect component states. Layer 3 can complete annotation while a database
query fails. Layer 2 can complete fragility while topology is unavailable.
Some detailed states also live in result dictionaries, such as survival mutation
status, or in a table's `reason` column.

Input errors raise `InputValidationError` before execution, even with
`on_error="record"`. Layer execution errors normally raise `AnalysisError`, which
exposes `layer` and `statuses`. `on_error="record"` allows partial results with
failed layer states. Component errors can be recorded without raising a layer
exception. Topology runtime problems can also issue `RuntimeWarning`.

See [examples](examples.md) for state inspection code.

## External service behavior

Requests have timeouts and bounded retries. HTTP errors, timeouts, invalid
response formats, and other query errors are recorded as failed evidence.
Missing inputs or uncached requests in replay mode can be unavailable.
A successful empty response is recorded as completed with zero results.

Failed or unavailable evidence does not increase database support.
An unavailable PubMed count remains missing rather than becoming zero.
Unavailable evidence is not negative biological evidence. Even a successful
empty search only describes that query at that time.

`api_cache` retains responses and their states. It can retain a failed response
as well as a successful one. Use a new cache for a deliberate fresh query.
Save the old cache if it belongs to a recorded analysis.

## Scientific scope

Input validation checks whether data meet the API contract.
Layer 4 evaluates available survival association and reports the limits of other
validation inputs. It does not turn annotations into tested biological hypotheses.

Database hits, publication counts, and score clusters are descriptive evidence.
They do not establish clinical utility or biomarker validity.
METABRIC replication and empirical Lehmann benchmarks are unavailable in 0.2.0.
Tier C marks an exploratory annotation. Tier A/B promotion is disabled.

[Layer details](layers.md) | [Troubleshooting](troubleshooting.md) | [README](../README.md)
