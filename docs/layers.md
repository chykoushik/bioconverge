# Analysis layers

`integrate()` always attempts Layer 1 after preflight succeeds.
Other layers depend on the supplied inputs. Inspect both layer and component
states through `result.statuses()`.

## Layer 1

Layer 1 uses a patient score table. It computes pairwise Spearman correlations,
bootstrap intervals, patient convergence indices, and KMeans archetypes.
Clustering uses median imputation and standardized score columns.
Stability compares bootstrap cluster fits using the adjusted Rand index.

No optional dependencies are needed. Invalid scores stop preflight.
Too few paired observations or constant scores produce undefined correlations.
Convergence strata can be unavailable when their thresholds have no variation.
Bootstrap samples with too few distinct profiles are counted as failed fits.

| Result method | Return value |
|---|---|
| `concordance()` | DataFrame of score pairs, rho, intervals, patient counts, valid bootstrap counts, and reasons |
| `convergence()` | DataFrame with `patient_id`, `convergence_index`, and `stratum` |
| `archetypes()` | DataFrame with `patient_id` and `archetype` |
| `discordance()` | DataFrame of patients in `convergent_low`, with archetype and convergence data |
| `stability()` | Dictionary of bootstrap counts, failures, ARI summaries, and interpretation |

Archetypes are clusters in the supplied scores. They are not established disease
subtypes. Stability describes the clustering procedure, not biological validity.
Cluster numbers have no fixed meaning across different runs or cohorts.

## Layer 2

Layer 2 runs when both trained `models` and `feature_matrix` are supplied.
It measures local sensitivity of model output to feature changes.
Scikit-learn predictors use finite differences with a step of `1e-4`.
Supported neural models use input gradients.

The aggregate fragility score is the norm of the mean model gradient.
The output also reports individual model sensitivity and gradient cancellation.
Gradients, features, and target scales need a meaningful common interpretation.
Opposing model gradients can cancel.

`fragility()` returns a DataFrame of patient scores.
`fragility_pathways()` returns a pathway summary DataFrame when constraints are
supplied. It returns `None` without pathway constraints.
No gene overlap produces an empty table and an unavailable pathway state.
Both methods return `None` if Layer 2 did not run.

The public `FragilityAnalyzer` class also exposes `trajectory()`, `topology()`,
and `consistency()` after `fit()`. These are not methods on `BioConvergeResult`.
Trajectory and topology results are DataFrames. Consistency is a dictionary.
A single model has no between-model consistency check.

Core predictors use scikit-learn. Neural backends need the `torch` or `tensorflow`
extra. Gradient topology needs `topology`. Without UMAP or HDBSCAN, core fragility
can still complete. Topology needs at least four patients and three distinct
gradients. Missing topology coordinates remain NaN with labels set to -1.
These conditions do not guarantee that a topology fit will succeed.

If neither models nor features are supplied, Layer 2 is skipped.
Supplying just one is an input error. Invalid model output or gradients can fail
execution. Check component reasons for dependency or topology problems.

Fragility is not a risk score, treatment recommendation, or distance to a model
decision boundary. A zero local derivative from a tree model does not establish
resistance to larger changes.

## Layer 3

Layer 3 runs with `score_metadata` after Layer 1 succeeds.
It attaches process annotations and database evidence to score metadata for each
archetype. It does not test whether a process distinguishes that archetype.

No optional package extra is needed. Online queries use Reactome, Enrichr,
STRING, GWAS Catalog, and PubMed. PubMed queries need `disease_context`.
Gene-based sources need suitable gene lists. `replay_only=True` prevents network
requests and uses only the supplied cache.

Without metadata this layer is skipped. When Layer 1 fails it is unavailable.
Individual services can fail while annotation rows are still returned.
Enrichr support uses adjusted-significant terms. STRING support requires actual
returned interactions. A successful query with no hits differs from a failure.

`hypotheses()` returns a DataFrame when annotations exist.
It includes the score, process, archetype, evidence scope, source states,
database support, and available annotation fields.
Layer 4 adds exploratory tier fields. Without Layer 3 it normally returns `None`.
An outcome-only run can instead return an empty hypothesis DataFrame.

Metadata annotations and publication counts are not empirical validation.
Database support does not establish a biomarker or an archetype-specific mechanism.

## Layer 4

Layer 4 runs after Layer 1 when annotations or validation inputs are available.
An explicit `outcome` supports survival association even without Layer 3.
The core dependency lifelines supplies survival methods.

Each valid survival test compares an archetype with the remaining patients.
The output includes raw log-rank p-values and Benjamini-Hochberg-adjusted p-values.
The `survival_signal` field uses adjusted p-values below 0.05.
At least six matched patients, an observed event, and three patients in each
comparison group are needed. Undefined tests remain unavailable.

`survival_analysis()` returns a dictionary when Layer 4 ran.
It includes a state and reason. Completed survival output contains a `results`
DataFrame, matching counts, and other diagnostics. It returns `None` when Layer 4
did not run. Available Kaplan-Meier plots are saved by `report()`.

METABRIC clinical input supports cohort inspection. It does not provide empirical
replication. A clinical table alone cannot validate Lehmann subtype recovery.
Those validation methods are unavailable in 0.2.0. Missing inputs are skipped.
Malformed supplied clinical inputs can fail validation.

All annotation tiers are C, meaning exploratory and unvalidated.
No Tier A/B promotion or synthetic benchmark is performed.
Survival association does not validate every process attached to an archetype.

## Combining results

`compare_layers()` returns patients who are in `convergent_low` and have fragility
above the cohort median. It returns `None` if Layer 1 or Layer 2 is absent.
This selection is not an independent biological validation test.

[Input data](input-data.md) | [Validation](validation.md) | [README](../README.md)
