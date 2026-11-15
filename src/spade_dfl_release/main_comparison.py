"""Exact historical main-comparison runtimes, isolated by subprocess."""
from __future__ import annotations

import importlib
import itertools
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import numpy as np

from .paths import release_root
from .publication import verify_publication_file
from .reference import METHODS, PARTITIONS, csv_rows, require_keys, verify_hash

DATASETS = ('mnist67', 'fashion01', 'alzheimer', 'brain_tumor_mri')
CSV_DATASETS = dict(zip(DATASETS, ('mnist', 'fashion', 'alzheimer', 'brain')))


def load_manifest():
    root = release_root()
    manifest = json.loads((root/'configs/main_comparison/manifest.json').read_text(encoding='utf-8'))
    require_keys(manifest['datasets'], DATASETS)
    require_keys([row['packaged_path'] for row in manifest['files']], {row['packaged_path'] for row in manifest['files']})
    for row in manifest['files']:
        path = (root/row['packaged_path']).resolve()
        if root.resolve() not in path.parents:
            raise ValueError('invalid vendored source path/size')
        verify_publication_file(path, row['sha256'], row['bytes'])
    return manifest


def verify(reference_root: Path):
    root = Path(reference_root)
    manifest = load_manifest()
    verification = json.loads((root/'paper_values/verification.json').read_text(encoding='utf-8'))
    if verification['status'] != 'verified' or any(verification[k] != 0 for k in ('difference_count', 'missing_cells', 'duplicate_cells')):
        raise ValueError('paper reference verification is not successful')
    loaded = {}
    for filename in ('main_final_accuracy.csv', 'main_roundwise.csv'):
        path = root/'aggregates'/filename
        verify_hash(path, verification['normalized_sha256'][filename])
        loaded[filename] = csv_rows(path.read_bytes())
    final = loaded['main_final_accuracy.csv']
    curves = loaded['main_roundwise.csv']
    keys = set(itertools.product(CSV_DATASETS.values(), PARTITIONS, METHODS))
    require_keys([(r['dataset'], r['partition'], r['method']) for r in final], keys)
    require_keys([(r['dataset'], r['partition'], r['method'], int(r['round'])) for r in curves], {key+(r,) for key in keys for r in range(51)})
    seed_evidence = {}
    for dataset, spec in manifest['datasets'].items():
        evidence = spec['run_identity_evidence']
        require_keys(evidence['seeds'], range(30))
        require_keys(evidence['folds'], range(5 if dataset == 'alzheimer' else 1))
        require_keys(evidence['methods'], METHODS)
        require_keys(evidence['partitions'], PARTITIONS)
        count = len(evidence['seeds']) * len(evidence['folds'])
        if evidence['unique_run_count'] != count * len(METHODS) * len(PARTITIONS):
            raise ValueError('frozen raw run identity count mismatch')
        seed_evidence[CSV_DATASETS[dataset]] = count
    for row in final + curves:
        if int(row['n_trajectories']) != seed_evidence[row['dataset']] or int(row['ddof']) != 0:
            raise ValueError('main trajectory count or population-SD convention mismatch')
        mean, sd = float(row['mean_percent']), float(row['sd_percent'])
        if not np.isfinite(mean) or not np.isfinite(sd) or not 0 <= mean <= 100 or sd < 0:
            raise ValueError('invalid main statistics')
    final_index = {(r['dataset'], r['partition'], r['method']):r for r in final}
    for row in final:
        if int(row['round']) != 50:
            raise ValueError('main final round must be 50')
    for row in curves:
        if int(row['round']) == 50:
            saved = final_index[(row['dataset'], row['partition'], row['method'])]
            if any(float(row[field]) != float(saved[field]) for field in ('mean_percent','sd_percent')):
                raise ValueError('main final accuracy differs from curve endpoint')
    return dict(status='verified', datasets=4, methods=7, partitions=4, rounds=50,
                final_rows=len(final), curve_rows=len(curves), missing=[], duplicates=[],
                trajectory_counts=seed_evidence,
                seed_fold_evidence='Hash-frozen archive member identities recorded during packaging; aggregate CSVs independently checked for counts and all 51 points.',
                paper_values='Checksum-verified final main-comparison values; manuscript result constants are not embedded.')


def _require_inputs(spec, vendor_root):
    for item in spec['required_inputs']:
        path = Path(vendor_root)/item['path']
        if not path.is_file():
            raise FileNotFoundError('missing frozen input: '+item['path'])
        verify_hash(path, item['sha256'])


def _stage_runtime(vendor_root, output, required_inputs):
    vendor_root, output = Path(vendor_root).resolve(), Path(output).resolve()
    runtime = output/'runtime'
    destinations = [(vendor_root/item['path'], (runtime/item['path']).resolve()) for item in required_inputs]
    if any(output not in destination.parents for _, destination in destinations):
        raise ValueError('frozen input staging would escape the new output directory')
    shutil.copytree(vendor_root, runtime, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    for source, destination in destinations:
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    return runtime


def run_main(dataset: str, output: Path, config: Path, vendor_root: Path) -> Path:
    if dataset not in DATASETS:
        raise ValueError('unsupported paper dataset: '+dataset)
    output, config, vendor_root = Path(output).resolve(), Path(config).resolve(), Path(vendor_root).resolve()
    if output.exists():
        raise ValueError('main rerun requires a new directory')
    spec = load_manifest()['datasets'][dataset]
    if vendor_root != (release_root()/spec['vendor_root']).resolve() or config != (release_root()/spec['config']).resolve():
        raise ValueError('only the packaged frozen main configuration/runtime is accepted')
    _require_inputs(spec, vendor_root)
    output.mkdir(parents=True, exist_ok=False)
    runtime = _stage_runtime(vendor_root, output, spec['required_inputs'])
    config_relative = config.relative_to(vendor_root)
    results = runtime/spec['package_relative']/'results'/'release_run'
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['PYTHONPATH'] = str(release_root()/'src')
    subprocess.run([sys.executable, '-B', str(runtime/'run_exact.py'), '--config', str(runtime/config_relative), '--output', str(results)], cwd=runtime/spec['package_relative'], env=environment, check=True)
    return results


def run(output: Path):
    output = Path(output).resolve()
    references = (release_root()/'reference_results').resolve()
    if output.exists() or output == references or references in output.parents:
        raise ValueError('main rerun requires a new directory outside reference_results')
    manifest = load_manifest()
    # Complete preflight before launching any of the four formal suites.
    for dataset in DATASETS:
        spec = manifest['datasets'][dataset]
        _require_inputs(spec, release_root()/spec['vendor_root'])
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for dataset in DATASETS:
        spec = manifest['datasets'][dataset]
        result = run_main(dataset, output/dataset, release_root()/spec['config'], release_root()/spec['vendor_root'])
        results[dataset] = result.relative_to(output).as_posix()
    return dict(status='completed', datasets=results, configuration='packaged frozen configurations; no numerical overrides')


def _run_vendor(dataset):
    """Private child entry; arguments contain locations only, never parameters."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package = args.config.resolve().parent.parent
    sys.path.insert(0, str(package/'src'))
    if dataset == 'brain_tumor_mri':
        # Keep the formal ResNet18 launcher intact, including extractor/cache
        # checks. Its default CPU and feature batch are historical defaults.
        sys.argv = [str(package/'scripts/run_resnet18_server_suite.py'), '--profile', 'server', '--output', str(args.output)]
        runpy.run_path(sys.argv[0], run_name='__main__')
    elif dataset == 'alzheimer':
        module = importlib.import_module('spade_dfl.alzheimer_v4_cross_validation')
        module.run_v4_cross_validation(args.config, args.output)
    else:
        module = importlib.import_module('spade_dfl.paper_suite')
        module.run_paper_suite(args.config, args.output)
