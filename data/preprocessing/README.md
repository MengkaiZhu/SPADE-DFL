# Recovered sensitivity preparation

This directory contains the historical preparation functions and the frozen configuration fields they consume. External datasets and caches are not included. Read the parent [data guide](../README.md) before using this entry point.

## Inspect the interface

Run from the package root:

```powershell
python -B data/preprocessing/prepare_sensitivity.py --help
python -B data/preprocessing/prepare_sensitivity.py --group mnist_fold0 --check-code
```

`--check-code` checks the preserved scientific module identities without preparing arrays.

## Prepare one group

After obtaining and hash-checking every prerequisite, use a new output directory outside the package:

```powershell
python -B data/preprocessing/prepare_sensitivity.py --group mnist_fold0 --output ../reconstructed_mnist
```

The command fits preprocessing and writes one group. Run each group in a separate process; `--help` lists all eight group names. Keep reconstructed output separate from the frozen runtime destinations.

## File roles

- `prepare_sensitivity.py` provides the preparation entry point and recovered historical functions; `preprocessing.py` contains the scaling calculations.
- `configs/`, `data/data_manifest.json`, and `prepared/*/bundle_metadata.json` provide the required historical configuration and descriptive metadata. The `prepared/` directory contains no sample arrays.
- [UPSTREAM_INPUTS.json](UPSTREAM_INPUTS.json) lists eight outer-training cache prerequisites beyond the [23 runtime inputs](../INPUT_PREPARATION.json). It also gives their required destinations.
- [SOURCE_LINEAGE.json](SOURCE_LINEAGE.json) records source and function hashes, source intervals, and AST comparison results. [EXPECTED_ARRAY_HASHES.json](EXPECTED_ARRAY_HASHES.json) records historical array fingerprints for later comparison.

Of the 14 recovered functions, 11 have identical Python syntax trees. Three have path adaptations: `verify_modules` locates packaged vendor modules, `build_group` resolves relocated inputs and caches, and `prepare_group` writes to the selected output root. The entry point omits the historical training and selection dispatcher. The preprocessing calculations and numerical configuration are unchanged.

Brain sensitivity requires the cache with SHA-256 `03c357ea6f7e3aff70e2537fc0a723fcd3659b02757a14d3327f701e1cc03cac`. The main-comparison cache has the same filename but SHA-256 `9aaaccdba667743dc65d38110785e0a4e062830216c66eba7621d742a49be985`; do not interchange them.

Raw-to-prepared numerical and bitwise equality has not been established. Relocated paths also make reconstructed metadata byte-different from historical path-bearing metadata. MRI feature extraction, training, and patient-level independence remain outside the verified scope; see the [known limitations](../../docs/KNOWN_LIMITATIONS.md).
