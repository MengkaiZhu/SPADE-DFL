# SPADE-DFL

Code and reproduction materials for the anonymous submission on derivative-free decentralized federated learning. The experiments cover the main classification comparison (Figures 3–6 and Table 1), sensitivity analysis (Table 2), heterogeneity (Figures 7–9), and communication (Figure 10). The table filenames in the package are mapped below. Run commands from the extracted package root.

## Install

Create and activate a Python 3.9 environment. To use the versions checked on Windows, run:

```text
python -m pip install --upgrade pip
python -m pip install -r requirements-verified.txt
python -m pip install --no-deps -e .
```

## Data preparation

The saved-result checks use bundled records. Classification reruns require the exact inputs listed in the [data guide](data/README.md) and the [sensitivity preparation instructions](data/README.md#sensitivity-inputs-and-recovered-producer). For MRI feature extraction, install the dependencies in [requirements-resnet18-cpu.txt](requirements-resnet18-cpu.txt).

## Verify saved results

```text
python experiments/01_main_comparison.py --verify-existing
python experiments/02_sensitivity.py --verify-existing
python experiments/03_heterogeneity.py --verify-existing
python experiments/04_communication.py --verify-existing
python experiments/make_paper_outputs.py --verify
```

The figure check verifies data series, table cells, and regenerated PDF and SVG files. Export uses Times New Roman when installed, with DejaVu Serif as the fallback. PDF bytes depend on the fonts and rendering libraries.

## Recompute statistics and figures

```text
python experiments/05_recompute_main.py --verify-existing
python experiments/05_recompute_main.py --output ../main-statistics-check
python experiments/make_paper_outputs.py --output ../paper-output-check
```

The main statistics are reconstructed from the [run index](reference_results/main_seed_records/run_index.csv) and its linked accuracy trajectories. Use fresh output directories.

## Training entry points

```text
python experiments/01_main_comparison.py --rerun --output ../runs-main
python experiments/02_sensitivity.py --rerun --output ../runs-sensitivity
python experiments/03_heterogeneity.py --rerun --output ../runs-heterogeneity
python experiments/04_communication.py --rerun --output ../runs-communication
```

The communication experiment uses synthetic data. See the [known limitations](docs/KNOWN_LIMITATIONS.md) for the scope of classification and privacy results.

## Package guide

The package renders submission Table 1 as `paper_outputs/tables/table_II.tex` and the two halves of submission Table 2 as `table_III.tex` and `table_IV.tex` in the same directory. Figures 3–10 use the submission numbering.

- [Experiment protocols](docs/EXPERIMENT_PROTOCOLS.md) gives the settings and aggregation rules.
- [Implementation note](IMPLEMENTATION_NOTE.md) records numerical differences between environments.
- [Notice](NOTICE.md) lists third-party sources.

[Source file hashes](configs/publication_adaptations.json) record the original and packaged versions of adapted vendor files. The verifiers check the packaged bytes and use the original hashes for compatibility checks.

`SHA256SUMS.txt` verifies the archive contents.
