# SPADE-DFL

SPADE-DFL is a communication-efficient, derivative-free method for decentralized federated learning, using single-point updates and linearized ADMM coordination. The paper evaluates it through the main comparison (Figures 3–6 and Table II), sensitivity analysis (Tables III–IV), heterogeneity analysis (Figures 7–9), and communication study (Figure 10).

This package provides the paper's accompanying code and reproduction materials as an anonymous ICLR 2027 submission attachment. It contains the frozen experiment implementations, saved results, and scripts for verification, reconstruction, and full reruns. Run commands from the `submission_code` directory. Start with saved-result verification; it does not train models or require external datasets.

Interpret results together with the [known limitations](docs/KNOWN_LIMITATIONS.md), especially the non-nested Alzheimer evaluation, exploratory sensitivity analysis, conditional privacy scope, and unresolved MRI reproduction.

## Install

Create an environment outside the package. For Windows PowerShell:

```powershell
python -m venv ../spade-dfl-env
../spade-dfl-env/Scripts/python.exe -m pip install --upgrade "pip>=21.3"
../spade-dfl-env/Scripts/python.exe -m pip install -r requirements.txt
../spade-dfl-env/Scripts/python.exe -m pip install --no-deps -e .
```

In the commands below, `python` means the interpreter from this environment. On Linux or macOS, the equivalent path is `../spade-dfl-env/bin/python`; those platforms have not been validated. Python 3.9 or later is declared in `pyproject.toml`. Use [requirements-verified.txt](requirements-verified.txt) instead of `requirements.txt` when the recorded non-MRI versions are required. Permitted dependency ranges are broader than the versions used for verification; see [reproducibility](docs/REPRODUCIBILITY.md).

## Data preparation

Saved-result verification and figure reconstruction use files already included in the package. Full reruns require the exact external inputs listed in [data/README.md](data/README.md). The 23 listed placements cover four dataset families; they are not 23 independent datasets. Raw datasets, MRI images, pretrained weights, feature caches, and prepared arrays are not distributed.

Sensitivity preprocessing has additional prerequisites and a recovered preparation entry point. Follow the [sensitivity input instructions](data/README.md#sensitivity-inputs-and-recovered-producer) and verify every input hash before use. The Brain workflow also requires `requirements-resnet18-cpu.txt` and torchvision's `ResNet18_Weights.IMAGENET1K_V1` weights. The documented full Brain launcher expects the raw MRI archive even when feature caches are present.

## Verify saved results

```powershell
python experiments/01_main_comparison.py --verify-existing
python experiments/02_sensitivity.py --verify-existing
python experiments/03_heterogeneity.py --verify-existing
python experiments/04_communication.py --verify-existing
python experiments/make_paper_outputs.py --verify
```

These commands check saved identities, numerical summaries, statistics, and paper products. Figure verification renders to temporary storage and does not replace packaged figures. It does not train models or validate claims that require raw-data reproduction.

## Recompute statistics and figures

```powershell
python experiments/05_recompute_main.py --verify-existing
python experiments/05_recompute_main.py --output ../main-statistics-check
python experiments/make_paper_outputs.py --output ../paper-output-check
```

Use new output directories. Main reconstruction reads all 6,720 saved runs and 342,720 round points and applies the frozen aggregation rules to 112 final and 5,712 roundwise rows. Run identities are in [run_index.csv](reference_results/main_seed_records/run_index.csv). Figure generation uses packaged aggregates. Family-specific standard deviations, confidence intervals, and tests are defined in the [experiment protocols](docs/EXPERIMENT_PROTOCOLS.md).

## Training entry points

The following commands run training or synthetic optimization and require the exact inputs and a new output directory:

```powershell
python -m pip install -r requirements-resnet18-cpu.txt
python experiments/01_main_comparison.py --rerun --output ../runs-main
python experiments/02_sensitivity.py --rerun --output ../runs-sensitivity
python experiments/03_heterogeneity.py --rerun --output ../runs-heterogeneity
python experiments/04_communication.py --rerun --output ../runs-communication
```

End-to-end runtime and peak memory have not been benchmarked. Some tests can invoke training or optimization, so use the explicit verification commands above when only saved results should be checked.

## Package guide

- [Experiment protocols](docs/EXPERIMENT_PROTOCOLS.md): models, partitions, schedules, and statistical rules.
- [Reproducibility](docs/REPRODUCIBILITY.md): current package checks, historical evidence, and unresolved reproduction levels.
- [Implementation note](IMPLEMENTATION_NOTE.md): implementation identity, numerical exceptions, and provenance.
- [Source provenance](docs/SOURCE_PROVENANCE.md): historical identities, current publication adaptations, and hash interpretation.
- [Known limitations](docs/KNOWN_LIMITATIONS.md): ten interpretation and evidence boundaries.
- [Data guide](data/README.md): exact-input checklist and preparation workflow.
- [Optional source verification](docs/OPTIONAL_SOURCES.md): explicit source-index setup for author-held archives; no source archive is bundled.
- [Notice](NOTICE.md): code-license status and third-party sources.

`src`, `configs`, `experiments`, and `tests` contain the implementation and verification structure. `reference_results` contains saved evidence; `paper_outputs` contains generated tables and figures. `MANIFEST.json` and `SHA256SUMS.txt` record file identities for package-integrity checks.
