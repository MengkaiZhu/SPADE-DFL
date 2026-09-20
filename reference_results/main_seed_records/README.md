# Main-comparison run records

These are selected fields from all 6,720 historical runs supporting Figures 3–6 and Table II, not new training results. `histories/` contains all 342,720 accuracy observations (rounds 0–50). No run was selected by accuracy. MNIST, Fashion-MNIST and Brain each have 840 trajectories; Alzheimer's disease has 4,200 fold–seed trajectories.

`run_index.csv` identifies dataset, partition, method, seed, fold, configuration, input hashes and the history file. Within a history CSV, select rows by `run_id`. Accuracy is a fraction, not a percentage. Summary endpoints and available query/client/communication counts are retained. Unavailable evaluation counts, original commands, code revisions and sample-membership fingerprints are explicitly `unknown`; they are not reconstructed guesses. A logical partition identity is not a patient/sample membership list.

`configurations.json` and `environments.json` are content-keyed records. Configuration scientific values and parameter-source explanations are retained; historical input/output locations are opaque aliases, so these evidence records are not directly executable configurations. Use the original `configs/` and documented launchers to run experiments. Environment records contain only recorded Python, library and platform fields; their presence does not mean those environments were tested during release preparation.

Archive/member hashes identify the historical sources. All such hashes, and hashes of newly selected public records, were computed during candidate preparation, not presented as original runtime measurements. `MANIFEST.json` hashes the record files and excludes itself. The release-level manifest additionally covers this manifest and README.

From the release root:

```text
python experiments/05_recompute_main.py --verify-existing
python experiments/05_recompute_main.py --output ../recomputed-main
```

The second command requires a new directory outside the release. Both commands check complete identities, all rounds, endpoints, source-record integrity, and exact agreement with frozen means and population standard deviations. Alzheimer's disease uses 150 fold–seed trajectories per cell without pooling folds within seed. Decimal mean accumulation and population SD follow the existing reference importer unchanged. Neither command imports a trainer or dataset loader.

To redraw from the regenerated main statistics, create a new copy of `reference_results` outside the release, replace only its two main aggregate CSVs with these verified outputs, then use `experiments/make_paper_outputs.py --reference-root <that-copy> --output <new-figure-directory>`. Existing source hashes must still match; the check does not change them to accept different numbers. The default drawing command can also use the unchanged bundled aggregates after the recomputation check.

See [known limitations](../../docs/KNOWN_LIMITATIONS.md): completing this saved-record chain does not remove non-nested model selection, establish original-data privacy or certify full training reproduction.
