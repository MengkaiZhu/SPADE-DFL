# Reproducibility and evidence scope

Three evidence levels are used here. L1 checks saved identities, values, and products. L2 reconstructs statistics or figures from saved runs. L3 starts from original inputs and executes preprocessing and training, or reruns the synthetic optimizer. L1/L2 agreement does not establish L3 or remove evaluation-design limitations.

## Evidence summary

| Evidence | Date | Scope | Status |
| --- | --- | --- | --- |
| Current release artifact | 2026-09-18 | Installation, imports, integrity, saved-result checks, and result-only reconstruction | Exact commands and outcomes are recorded in the accompanying release report; they are not training evidence. |
| Historical audit | 2026-09-18 | L1/L2 for all four experiment families, one MNIST main replay, one sensitivity configuration, and one communication configuration | Historical evidence retained; limited reruns do not establish complete L3. |
| Earlier archived execution | Historical, exact date unknown | Saved outputs and hashes for the full 240-job communication experiment | Output provenance evidence; not a new rerun of the delivered package. |
| Full public reproduction | Unverified | All raw-data preparation, all training grids, complete MRI extraction, patient independence, and cross-platform execution | Exact inputs or validation evidence remain incomplete. |

The historical audit reported 301 passed and 23 skipped tests. Those counts describe that audit only. Any test totals for the current artifact belong to the accompanying release report.

## Saved-result reconstruction

`experiments/05_recompute_main.py` reads [run_index.csv](../reference_results/main_seed_records/run_index.csv), four CSV files in [histories](../reference_results/main_seed_records/histories), and their configuration, environment, and manifest records. The packaged evidence contains 6,720 runs and 342,720 round points and reconstructs 112 final and 5,712 roundwise rows under the frozen aggregation rules. Use `--verify-existing` to compare with packaged aggregates or `--output NEW_DIR` to write a separate reconstruction.

Sensitivity verification reconstructs `correct / eval_count` and the seed-level pooled Alzheimer estimator. Heterogeneity verification checks saved trajectories, distributions, paired tests, and Benjamini–Hochberg adjustment. Communication verification reconstructs eight summaries from 240 records and checks communication and query formulas without running optimization. Statistical definitions are in the [experiment protocols](EXPERIMENT_PROTOCOLS.md).

`experiments/make_paper_outputs.py --verify` checks three TeX tables, nine source CSVs, and 40 PDF/SVG figures: 52 products plus a manifest. It also renders figures independently in temporary storage. `--output NEW_DIR` writes regenerated products to a separate directory. Both paths use saved aggregates.

## Historical training evidence

The 2026-09-18 audit recorded the following limited reruns:

- Main: one MNIST IID SPADE seed ran for 50 rounds from raw pixels through PCA10 and matched all 51 accuracy points. Historical L1/L2 also covered all 112 final and 5,712 roundwise rows.
- Sensitivity: one MNIST `N=100, epsilon=32, seed=0` configuration ran for 10 rounds from prepared inputs with positive release noise. Historical L1/L2 reconstructed 360 count records into 36 unique cells.
- Heterogeneity: historical L1/L2 covered 50,490 roundwise rows, 255 distribution rows, 150,000 client-label proportions, and 1,200 Wilcoxon/BH records. No complete training replay was performed.
- Communication: one formal synthetic configuration was rerun. L1/L2 covered 240 records and eight summaries; the earlier full 240-job execution is supported by saved outputs and hashes.

These targeted checks do not extend to the full training grids, raw sensitivity preprocessing, or complete MRI workflow.

## Manuscript and source versions

The current reference PDF has SHA-256 `73b433c67cba991a67b3173f5b640bb787f873321cfb8218ac2d2185fc5d4afb`. Historical `spade_dfl_v1.tex` has SHA-256 `4eafcd1bfd5ebb609f37d056fe0705589495d83c5abefcbeec097737dab7a9ab`. The original code ZIP has SHA-256 `5acbf065176bfdba6e4d7f7da970a91828c6fa1e1473553afdaf0a5db17c11a2`. They are distinct artifacts; details are in [manuscript version metadata](../configs/manuscript_versions.json).

The retained [current-PDF comparison record](../reference_results/paper_values/current_pdf_comparison.json) preserves the preceding check and its recorded differences. It is historical comparison evidence, outside the acceptance scope of this publication cleanup.

Historical-source hashes and current delivered-file hashes are separate provenance layers; see the [implementation note](../IMPLEMENTATION_NOTE.md). Missing original commands, code revisions, and environment fields remain `unknown` rather than reconstructed.

## Environment and integrity

The historical Windows audit used Python 3.9.0, NumPy 2.0.2, SciPy 1.13.1, Matplotlib 3.9.4, mpmath 1.4.1, Pillow 11.3.0, and pytest 8.4.2. CPU torch/torchvision was not installed. The non-MRI versions used for the current artifact are listed in [requirements-verified.txt](../requirements-verified.txt); optional MRI pins describe prerequisites, not a validated MRI replay. Cross-platform equivalence is unverified.

Finite p/q and confidence-interval exceptions are listed in the [implementation note](../IMPLEMENTATION_NOTE.md). `MANIFEST.json` excludes the two integrity files from its payload inventory; `SHA256SUMS.txt` excludes itself and includes `MANIFEST.json`. Hashes detect changed bytes. The [data guide](../data/README.md) describes exact-input gaps; the [known limitations](KNOWN_LIMITATIONS.md) defines the interpretation boundaries.
