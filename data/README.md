# Data preparation

[INPUT_PREPARATION.json](INPUT_PREPARATION.json) is the exact-input checklist. It lists 23 runtime placements with their purpose, filename, package-relative destination, byte count, SHA-256 hash, source status, and conversion script. These placements belong to four dataset families; they are not 23 independent datasets. The payloads are not included.

The cited source pages are:

- [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset): digits 6 and 7.
- [Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist): labels 0 (T-shirt/top) and 1 (Trouser).
- [Alzheimer disease](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset): binary diagnosis with the frozen encoding and fold protocol.
- [Brain Tumor MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset): glioma, meningioma, no-tumor, and pituitary classes, preserving the official Training/Testing split.

These pages identify the datasets cited by the paper, but not necessarily the exact historical file versions. Exact release identifiers, acquisition routes for prepared files, and redistribution rights remain unresolved. Obtain each input lawfully and compare its SHA-256 before placing it in the package tree. In PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 <file>
```

A different hash means different bytes and must not be substituted silently.

## Existing results

Saved-result verification and statistics/figure reconstruction use the included reference results. They do not require external datasets, image preprocessing, or model training. Passing those checks does not establish reproduction from raw data. See [reproducibility](../docs/REPRODUCIBILITY.md) and [known limitations](../docs/KNOWN_LIMITATIONS.md).

<a id="main-inputs"></a>
## Main comparison inputs

Place files at the exact destinations recorded in the checklist. The two Brain feature caches and the original MRI ZIP are separate inputs. The raw archive was hash-matched in retained historical materials but is not distributed; the Brain launcher expects it even when feature caches are available.

Brain feature extraction uses torchvision `ResNet18_Weights.IMAGENET1K_V1` and its `weights.transforms()`, removes the final fully connected layer, sets evaluation mode, and extracts 512 features per image with gradients disabled. PCA is fitted on training features and reduces them to 10 dimensions. The four-class classifier therefore has 40 parameters. The official test features are transformed with the training-fitted PCA.

Install the optional versions in `requirements-resnet18-cpu.txt` for this workflow. The pretrained weights are not bundled and torchvision may download them when they are missing from its local cache. End-to-end feature extraction and raw MRI training have not been validated, and patient-level independence remains unverified.

<a id="sensitivity-inputs-and-recovered-producer"></a>
## Sensitivity inputs and recovered producer

Sensitivity uses 16 runtime inputs: `data.npz` and `metadata.json` for MNIST, Fashion, Brain, and five Alzheimer folds. Their destinations are under `src/spade_dfl_release/vendor/sensitivity/table34_preprocessing_working/prepared/<group>/` and their identities are fixed by the checklist hashes.

[preprocessing/prepare_sensitivity.py](preprocessing/prepare_sensitivity.py) exposes the recovered preparation functions. It preserves the historical splits, hash assertions, scaling, and fitting calculations; only imports, paths, and the preparation entry point were adapted for the package. [SOURCE_LINEAGE.json](preprocessing/SOURCE_LINEAGE.json) records source hashes, line intervals, and abstract syntax tree (AST) comparisons. An AST is the parsed structure of the Python code.

Preparation requires eight additional outer-training caches listed in [UPSTREAM_INPUTS.json](preprocessing/UPSTREAM_INPUTS.json). These are prerequisites beyond the 23 runtime placements. Their payloads and exact public acquisition routes are not included.

The Brain sensitivity cache `resnet18_v1_training_512d.npz` has SHA-256 `03c357ea6f7e3aff70e2537fc0a723fcd3659b02757a14d3327f701e1cc03cac`. The main-comparison cache with the same filename has SHA-256 `9aaaccdba667743dc65d38110785e0a4e062830216c66eba7621d742a49be985`. They are different files and must not be interchanged.

Inspect the interface without processing data:

```powershell
python -B data/preprocessing/prepare_sensitivity.py --help
python -B data/preprocessing/prepare_sensitivity.py --group mnist_fold0 --check-code
```

To prepare a group after obtaining every exact prerequisite:

1. Place the eight caches at the destinations in `UPSTREAM_INPUTS.json`.
2. Place `mnist.npz`, `fashion-mnist.npz`, and `2、Alzheimer disease dataset.zip` under `data/preprocessing/inputs/raw/`, using the corresponding main-input hashes in the checklist. Brain preparation consumes its distinct 512-feature training cache and does not extract images.
3. Run one group per process and write outside the package, for example: `python -B data/preprocessing/prepare_sensitivity.py --group mnist_fold0 --output ../reconstructed_mnist`. The eight group names are listed by `--help`.
4. Compare reconstructed arrays with [EXPECTED_ARRAY_HASHES.json](preprocessing/EXPECTED_ARRAY_HASHES.json) before training. Keep reconstructed output separate from the canonical runtime destinations.

The producer fits preprocessing on inner-training data and writes candidate representations. Later frozen sensitivity processing uses the selected representation; Alzheimer uses five principal components plus an intercept, unlike the 39-dimensional main-comparison representation. Historical test exposure and selection boundaries still apply.

Raw-to-prepared numerical and bitwise equality has not been established. Reconstructed metadata contains relocated paths and therefore differs from historical path-bearing metadata. Fixed-input privacy accounting does not certify preprocessing or the full raw-data pipeline.

## Heterogeneity input

Heterogeneity reuses the same MNIST bytes as the main comparison at another package-relative location. The extra placement is recorded separately because the frozen code expects that path; it is not a fifth dataset.

## Communication input

The communication experiment uses a frozen synthetic configuration and no external classification dataset. Verifying its saved summaries does not rerun the optimizer simulation.
