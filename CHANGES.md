# 0.2.0

This release changes unsafe behavior explicitly. Existing manuscript results must
be rerun and reviewed before reuse. Existing output/ artifacts are not overwritten.

| Change | Scientific effect | Manuscript impact |
|---|---|---|
| Seed every bootstrap KMeans fit; propagate seeds | Resampling formula and KMeans settings unchanged | Stability values may change |
| Reject degenerate bootstrap fits; keep stability cap at 200 and expose counts | No hidden increase in resampling | Correct bootstrap sensitivity labels |
| Exclude undefined bootstrap correlations from CI calculation | Valid draws contribute; valid count reported | Confidence intervals may change |
| Degenerate convergence thresholds return unknown strata | Convergence formula and clipping unchanged | Discordance membership may change |
| Validate missingness, finiteness, unique IDs and matrix alignment | Reject invalid analyses before fitting | Previously accepted invalid inputs stop |
| Preserve IDs; deduplicate identical clinical endpoints; reject conflicts | Prevent repeated-patient inference | Survival sample sizes and p-values change |
| Use follow-up for censored patients and explicit time/event mapping | Correct survival endpoint construction | Survival results and interpretation change |
| Report adjusted survival p-values | Raw log-rank tests retained; BH family correction added | Significance decisions may change |
| No synthetic external benchmark or lexical replication | Unsupported empirical claims removed | Prior validation scores and tiers invalid |
| No arbitrary METABRIC numeric fallback; require PR negativity | Cohort inspection only; empirical replication unavailable | Cohort counts and validation claims change |
| Metadata annotations replace untested archetype assertions | No new biological inference algorithm | Hypothesis interpretation changes |
| No Tier A/B promotion from descriptive evidence | Tier C marks unvalidated annotations | Prior tier totals cannot be reused |
| Correct API protocols and schemas; explicit failure states | Failures never count as absence or novelty | Annotations and support counts change |
| Count only adjusted-significant Enrichr terms and actual STRING edges | Reject unsupported evidence credit | Database support counts may change |
| Remove hard-coded breast cancer queries | Disease context required | GBM and other literature annotations change |
| Retain gradient aggregation and perturbation step; expose cancellation | Core fragility method unchanged | New diagnostics; invalid targets now rejected |
| Consistent backend targets, device/dtype handling and state restoration | Correct prediction/gradient contract | Affected neural-model outputs may change |
| Undefined topology stays missing; zero trajectory norm stays zero | No artificial clusters or directions | Affected topology/trajectory outputs change |
| Tumor-only notebook expression; missing assays remain missing | No random proxies or tumor/normal averaging | Example scores and all derived results may change |
| Empirical examples and benchmark limitations documented | Remove circular subtype recovery and invalid subgroup tests | Historical benchmark claims withdrawn |

API changes: Python >=3.10; DataFrame features indexed by patient ID; integer
random_state required; explicit disease_context for literature; report directories
must be empty; integrate defers artifact writes until report(); survival column defaults are automatic detection; network evidence
can be replayed using api_cache and replay_only. Dependency ranges are bounded;
optional topology, reporting and neural backends have separate extras.
