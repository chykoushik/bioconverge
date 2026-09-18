# Reproducibility

The normal installation command is unchanged.

```bash
pip install bioconverge
```

For reproducing work that specifically used version 0.2.0, you can pin that version.

```bash
pip install bioconverge==0.2.0
```

Pinning BioConverge alone does not fix every dependency version.
Save the environment used for the analysis.

```bash
python --version
python -m pip freeze > requirements.txt
python -m pip check
```

Choose a new destination if a requirements file already exists.
The source distribution also includes `constraints-py310-windows.txt` for the
recorded Python 3.10 Windows dependency set. It is not a universal environment lock.

## Record versions

```python
import platform
import bioconverge as bc
from importlib.metadata import version

print(bc.__version__)
print(platform.python_version())
for package in ["numpy", "pandas", "scipy", "scikit-learn", "lifelines"]:
    print(package, version(package))
```

`result.metadata()` records the package version, Python version, platform,
selected dependencies, parameters, input fingerprints, and thread settings.
Use a full environment record as well, especially when using a neural backend.

## Random state

`random_state` defaults to 42. It must be an integer from 0 through `2**32 - 1`.
It controls score clustering, bootstrap sampling, bootstrap KMeans seeds, and
optional gradient topology. It does not train or seed user models.

Concordance attempts the requested `n_bootstrap` draws and records the number
of valid correlations. Stability attempts at most 200 fits and reports successful
and failed fits. The two counts measure different operations.

Use the same inputs, score and feature order, preprocessing, model, seed,
dependency versions, and numerical runtime settings for a repeated analysis.
Hardware and library changes can affect numerical results.
See [examples](examples.md) for a repeatability check within one environment.

## Input provenance

Keep the original data release identifiers and the code used to derive scores.
Record patient inclusion rules, missing values, score scales, feature order,
model training, target selection, and any exclusions.

Reports contain hashes of the score table and supplied features, fingerprints
of validation inputs, and model fingerprints when serialization is available.
A model fingerprint can be unavailable with a reason.
These hashes do not replace source data, model files, or preprocessing records.
They do not make unlike scores biologically comparable.

## External databases

Database results can change while the random seed stays fixed.
Online annotation sends process terms and gene lists to external resources.

Supply a dictionary as `api_cache` to retain query responses and states.
`report()` writes that cache to `api_responses.json` and writes a query log to
`reproducibility_log.json`. Reload the JSON as a dictionary and pass it as
`api_cache` with `replay_only=True` for offline replay.

An absent replay entry is unavailable. A saved failure remains a failure when
replayed. Keep the original cache when making a separate fresh query.
Report which analysis used live queries and which used replay.

## Files to retain

`report()` saves `run_metadata.json`, `analysis_statuses.json`,
`validation_results.json`, `api_responses.json`, `reproducibility_log.json`,
and `manifest.json`, along with available tables and figures.
The manifest contains SHA-256 hashes of the report files.
Keep warnings and exception records from your execution as well.
Reports do not automatically capture every Python warning or save every input.

When reporting research, state the version and environment, input sources,
score definitions, cohort rules, model details, seed, cluster count, and bootstrap
counts. Report failed, skipped, and unavailable components with their reasons.
Describe metadata annotation separately from survival association or any
independent validation you performed outside BioConverge.

[Validation](validation.md) | [README](../README.md)
