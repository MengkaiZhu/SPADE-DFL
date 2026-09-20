# Implementation identity

The package preserves the implementations associated with the saved experiments. Main-comparison SPADE retains estimator memory across rounds and uses zero Gaussian release noise. Sensitivity MNIST/Fashion uses `full_refresh`, while Alzheimer/Brain uses `persistent_saga`. Historical private trainers clip the complete round-level model displacement. The non-private communication diagnostic initializes memory each round.

These are historical experiment variants rather than a line-by-line private execution of Algorithm 1. Classification and sensitivity batches sample with replacement, whereas Algorithm 1 specifies uniform subsets without replacement. Repeated draws and memory initialization or refresh queries count toward the query budget. Classification schedules also differ from the communication horizon schedule. See the [experiment protocols](docs/EXPERIMENT_PROTOCOLS.md) and [known limitations](docs/KNOWN_LIMITATIONS.md).

Privacy accounting is conditional on fixed prepared inputs and client partitions; it does not cover the complete raw-data pipeline or data-dependent logs. Sensitivity metadata retains `validation_only=true`, `formal_authorized=false`, `previous_test_exposure=true`, and `end_to_end_privacy_certified=false`. `formal_authorized` is a historical compatibility field: `false` means the exploratory sensitivity output was not designated for the frozen formal-result suite. It is not a software permission or legal-authorization flag.

## Exact numerical exceptions

ULP means the spacing between adjacent representable floating-point values. It is not an adjustable tolerance. Historical and current heterogeneity calculations agree exactly except for the checked adjacent float64 values below, shown as historical/current. LT-ADMM-VR is the reference.

- `dir_0.6 / spade_dfl / round 34`: p `0x1.28e316b1cf5f2p-17` / `0x1.28e316b1cf5f1p-17`; q `0x1.084a893023b6bp-15` / `0x1.084a893023b6ap-15`.
- `pathological / spade_dfl / round 12`: p `0x1.918f055507013p-13` / `0x1.918f055507014p-13`; q `0x1.dc28127709516p-12` / `0x1.dc28127709517p-12`.

All 1,200 identities, paired differences, test statistics, and classifications agree; classification counts are 424/410/366. These exceptions do not imply bitwise equality across all numerical libraries.

Communication has no hypothesis tests. Four `stationarity_ci95_low` values differ by one adjacent float64 value, again shown as historical/current:

- `T=1000, tau=10`: `0x1.cf7c490912a28p-17` / `0x1.cf7c490912a29p-17`.
- `T=10648, tau=22`: `0x1.c702be51097e0p-17` / `0x1.c702be51097e1p-17`.
- `T=29791, tau=1`: `0x1.b81e72e884775p-17` / `0x1.b81e72e884776p-17`.
- `T=29791, tau=31`: `0x1.b7030cdd6e819p-17` / `0x1.b7030cdd6e81ap-17`.

## Dependencies and provenance

Nine package paths contain `tuning` in their names. Direct import checks show that the Alzheimer protocol and v3 modules are required by frozen-configuration and runtime utilities; the associated JSON is a frozen configuration input. Other sensitivity tuning-named files are retained as hash-addressed members of the sensitivity source manifest. None is a documented parameter-search entry point.

Historical-source hashes and current artifact hashes serve different purposes and are recorded separately. Historical hashes identify archived source evidence. Current hashes identify the delivered files, including documented publication adaptations in `configs/publication_adaptations.json`. Do not use one set as a byte-identity claim for the other. The [source provenance guide](docs/SOURCE_PROVENANCE.md) explains the two layers. Missing historical command, revision, and environment metadata remains `unknown`; public path aliases preserve provenance without requiring an author's directory layout.

Author-held archive checks require an explicit source index and are separate from public saved-result verification. No source archive is bundled; setup and skip/failure behavior are documented in [optional source verification](docs/OPTIONAL_SOURCES.md).

Historical same-environment comparisons do not establish bitwise reproduction of every archived result. PCA recomputes a numerical decomposition. MRI archive-to-release training equivalence and the complete raw-image-to-training chain remain unverified; the exact raw MRI archive located in retained materials is not included.
