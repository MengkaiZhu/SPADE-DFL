"""Historical eight-configuration pilot adapter with verified publication sources."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
import time
import traceback
import zipfile
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str(value.dtype).encode('ascii'))
    h.update(json.dumps(list(value.shape)).encode('ascii'))
    h.update(value.tobytes(order='C'))
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def save_npz(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def read_training(path, feature_key, label_key):
    if (feature_key, label_key) not in {('train_x', 'train_y'), ('features', 'labels')}:
        raise ValueError('Only training arrays are allowed')
    with np.load(path, allow_pickle=False) as data:
        return data[feature_key], data[label_key]


def new_target(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if target == root or root not in target.parents:
        raise ValueError('Target outside isolated workspace')
    if target.exists():
        raise FileExistsError(target)
    return target


def split_training(labels, fraction, seed):
    # S18 calibration.py::split_training; independently cross-checked on preparation.
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    target = int(math.floor(len(labels) * fraction + .5))
    ideal = counts * fraction
    allocation = np.clip(np.floor(ideal).astype(int), 1, counts - 1)
    while allocation.sum() < target:
        eligible = allocation < counts - 1
        index = int(np.argmax(np.where(eligible, ideal - allocation, -np.inf)))
        allocation[index] += 1
    while allocation.sum() > target:
        eligible = allocation > 1
        index = int(np.argmax(np.where(eligible, allocation - ideal, -np.inf)))
        allocation[index] -= 1
    rng = np.random.default_rng(seed)
    inner, validation = [], []
    for label, count in zip(classes, allocation):
        shuffled = rng.permutation(np.flatnonzero(labels == label))
        validation.extend(shuffled[:count])
        inner.extend(shuffled[count:])
    return np.sort(np.array(inner, dtype=np.int64)), np.sort(np.array(validation, dtype=np.int64))


def validate_fields(job):
    expected = dict(job['original_config_fields'])
    expected.update(job['override_fields'])
    if expected != job['optimizer_config_fields']:
        raise ValueError('Unapproved optimizer changes')
    config = job['optimizer_config_fields']
    for key, value in dict(n_clients=30, target_epsilon=8, target_delta=1e-5,
                           n_rounds=50, seeds=[0], stage='spade_dfl_private').items():
        if config[key] != value:
            raise ValueError('Outside first eight authorized pilot runs')


def check_budget(clip, sigma, delta, epsilon):
    clip, sigma = np.asarray(clip), np.asarray(sigma)
    if not (np.isfinite(clip).all() and np.isfinite(sigma).all()
            and (clip > 0).all() and (sigma > 0).all()):
        raise ValueError('Clipping and noise must be finite and positive')
    rho = np.sum(2 * clip ** 2 / sigma ** 2)
    actual = float(rho + 2 * np.sqrt(rho * np.log(1 / delta)))
    if not np.isclose(actual, epsilon, rtol=1e-12, atol=1e-12):
        raise ValueError('Actual epsilon does not match approved epsilon')
    return actual


def load_job(relative):
    manifest = json.loads((ROOT / 'pilot_manifest.json').read_text(encoding='utf-8'))
    if relative not in manifest['config_files']:
        raise ValueError('Job not in first eight manifest')
    auth = json.loads((ROOT / 'EXECUTION_AUTHORIZATION.json').read_text(encoding='utf-8'))
    if auth['pilot_runs_authorized'] != 8 or auth['formal_training_authorized']:
        raise ValueError('Invalid pilot authorization')
    path = ROOT / relative
    if sha(path) != auth['config_sha256'][relative]:
        raise ValueError('Frozen configuration hash changed')
    job = json.loads(path.read_text(encoding='utf-8'))
    validate_fields(job)
    return job


def verify_modules(job):
    from spade_dfl_release.publication import historical_module_hashes
    package = ROOT / 'vendor' / job['dataset'] / 'src' / 'spade_dfl'
    hashes = historical_module_hashes(package)
    actual = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if actual != job['algorithm_source']['all_python_module_hash_map_sha256']:
        raise ValueError('Original source module hash mismatch')
    return package


def extract_sources():
    manifest = json.loads((ROOT / 'pilot_manifest.json').read_text(encoding='utf-8'))
    jobs = [load_job(p) for p in manifest['config_files']]
    planned, seen = [], set()
    archives = {}
    try:
        for job in jobs:
            ds = job['dataset']
            if ds in seen:
                continue
            seen.add(ds)
            archive, member = job['algorithm_source']['trainer_path'].split('::')
            if archive not in archives:
                archives[archive] = zipfile.ZipFile(archive)
            z = archives[archive]
            prefix = member.rsplit('/', 1)[0] + '/'
            selected = [i for i in z.infolist() if i.filename.startswith(prefix) and i.filename.endswith('.py')]
            names = [i.filename for i in selected]
            if len(names) != len(set(names)):
                raise ValueError('Duplicate source archive members')
            hashes = {i.filename[len(prefix):]: hashlib.sha256(z.read(i)).hexdigest() for i in selected}
            if hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest() != job['algorithm_source']['all_python_module_hash_map_sha256']:
                raise ValueError('Archive source changed')
            for info in selected:
                relative = info.filename[len(prefix):]
                target = new_target(ROOT, Path('vendor') / ds / 'src' / 'spade_dfl' / relative)
                planned.append((z, info, target, hashes[relative]))
        free = shutil.disk_usage(ROOT).free
        needed = sum(info.file_size for _, info, _, _ in planned)
        if free < needed:
            raise OSError('Insufficient disk space')
        save_json(ROOT / 'SOURCE_EXTRACTION_PREFLIGHT.json', {
            'free_bytes': free, 'selected_bytes': needed,
            'members': [{'archive': z.filename, 'member': i.filename, 'size': i.file_size,
                         'target': str(p), 'sha256': h} for z, i, p, h in planned]})
        for z, info, target, digest in planned:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(z.read(info))
            if sha(target) != digest:
                raise ValueError('Source copy failed integrity check')
        print(json.dumps({'extracted_modules': len(planned), 'bytes': needed}), flush=True)
    finally:
        for archive in archives.values():
            archive.close()


def original_split(job, labels):
    path = Path(job['holdout']['source'].split('::')[0])
    if sha(path) != job['holdout']['source_sha256']:
        raise ValueError('Holdout source changed')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'split_training')
    isolated = ast.Module(body=[function], type_ignores=[])
    scope = {'np': np, 'math': math}
    exec(compile(isolated, str(path), 'exec'), scope)
    return scope['split_training'](labels, job['holdout']['fraction'], job['holdout']['seed'])


def config_and_imports(job):
    package = verify_modules(job)
    sys.path.insert(0, str(package.parent))
    from spade_dfl.config import ExperimentConfig
    return ExperimentConfig.from_dict(job['optimizer_config_fields'])


def make_partition(config, labels, seed):
    from spade_dfl.partitioning import partition_labels
    # Identical SeedSequence child construction to each original run_experiment default.
    children = np.random.SeedSequence(seed).spawn(config.n_clients + 1)
    return partition_labels(labels, config.n_clients, config.partition_mode,
                            np.random.default_rng(children[0]), alpha=config.dirichlet_alpha,
                            pathological_minor_fraction=config.pathological_minor_fraction)


def prepare(relative):
    job = load_job(relative)
    config = config_and_imports(job)
    from spade_dfl.data import PCAState
    from spade_dfl.graph import build_graph, derive_midpoint_beta
    from spade_dfl.privacy import spade_private_schedule
    out = new_target(ROOT, Path('prepared') / job['job_id'])
    source = job['training_data']
    if sha(source['path']) != source['sha256']:
        raise ValueError('Training source file changed')
    x, y = read_training(source['path'], source['feature_key'], source['label_key'])
    assert array_hash(x) == source['training_features_sha256']
    assert array_hash(y) == source['training_labels_sha256']
    inner, valid = split_training(y, job['holdout']['fraction'], job['holdout']['seed'])
    original_inner, original_valid = original_split(job, y)
    assert np.array_equal(inner, original_inner) and np.array_equal(valid, original_valid)
    assert array_hash(inner) == source['inner_indices_sha256']
    assert array_hash(valid) == source['validation_indices_sha256']
    if job['dataset'] == 'brain':
        from spade_dfl.brain_tumor_resnet18_data import fit_training_pca, apply_training_pca
        from spade_dfl.brain_tumor_data import BRAIN_TUMOR_CLASS_NAMES, PCA_SEED, PCA_OVERSAMPLES, PCA_POWER_ITERATIONS
        pca = fit_training_pca(x[inner], config.feature_dimension)
        train_x, validation_x = apply_training_pca(x[inner], pca), apply_training_pca(x[valid], pca)
        assert np.allclose(pca.mean, x[inner].astype(np.float64).mean(axis=0))
        assert np.allclose(train_x.std(axis=0), 1)
        metadata = dict(dataset_name=config.dataset_name, original_feature_dimension=x.shape[1],
                        pca_dimension=config.feature_dimension, task_type=config.task_type,
                        n_classes=config.n_classes, class_names=list(BRAIN_TUMOR_CLASS_NAMES), model_dimension=config.dimension)
        pca_details = dict(fit_scope='inner_training_only', seed=PCA_SEED, oversamples=PCA_OVERSAMPLES,
                           power_iterations=PCA_POWER_ITERATIONS)
    else:
        with np.load(source['path'], allow_pickle=False) as data:
            pca = PCAState(data['pca_mean'], data['pca_components'], data['pca_scale'])
            metadata = json.loads(str(data['bundle_metadata_json']))
        train_x, validation_x = x[inner], x[valid]
        pca_details = dict(fit_scope='frozen_outer_training_features_exploratory_not_nested')
    metadata.update(source_path=str(source['path']), sha256=source['source_dataset_sha256'],
                    original_train_count=len(inner), filtered_train_count=len(inner),
                    original_test_count=len(valid), filtered_test_count=len(valid))
    partition = make_partition(config, y[inner], job['algorithm_seed'])
    assert np.array_equal(np.sort(np.concatenate(partition.indices)), np.arange(len(inner)))
    assert min(map(len, partition.indices)) > 0
    graph = build_graph(config.n_clients, topology=config.graph_topology,
                        probability=config.graph_probability, seed=job['algorithm_seed'])
    beta = derive_midpoint_beta(graph, config.tau, config.rho)
    rounds = np.arange(1., config.n_rounds + 1)
    clip, sigma, rho = spade_private_schedule(config.n_rounds, config.clip_norm, config.clip_exponent,
        config.target_delta, config.target_epsilon,
        utility_weights=rounds ** -(config.eta_exponent + config.mu_exponent), allocation=config.privacy_allocation)
    for key, actual in [('clip', clip), ('sigma', sigma), ('rho_zcdp', rho)]:
        assert np.allclose(actual, job['privacy_schedule_preview'][key], rtol=1e-12, atol=1e-15)
    epsilon = check_budget(clip, sigma, config.target_delta, config.target_epsilon)
    out.mkdir(parents=True)
    arrays = dict(train_x=train_x, train_y=y[inner], validation_x=validation_x, validation_y=y[valid],
                  inner_indices=inner, validation_indices=valid, pca_mean=pca.mean,
                  pca_components=pca.components, pca_scale=pca.scale, adjacency=graph.adjacency,
                  label_histograms=partition.label_histograms)
    arrays.update({f'client_{i}': ids for i, ids in enumerate(partition.indices)})
    save_npz(out / 'data.npz', **arrays)
    save_json(out / 'bundle_metadata.json', metadata)
    save_json(out / 'preflight.json', dict(job_id=job['job_id'], data_sha256=sha(out / 'data.npz'),
        array_sha256={k: array_hash(v) for k, v in arrays.items()}, pca=pca_details,
        train_count=len(inner), validation_count=len(valid), official_test_arrays_loaded=False,
        partition_seed_rule='Original trainer: SeedSequence(algorithm_seed).spawn(N+1)[0]',
        partition_mode=partition.mode, client_sample_counts=[len(i) for i in partition.indices],
        graph_seed=job['algorithm_seed'], directed_edges=int(graph.adjacency.sum()), beta=beta,
        graph_lambda_max=float(graph.laplacian_eigenvalues[-1]),
        clip=clip.tolist(), sigma=sigma.tolist(), rho_zcdp=rho.tolist(), epsilon_recalculated=epsilon))
    print(json.dumps({'prepared': job['job_id'], 'train': len(inner), 'validation': len(valid),
                      'client_min': min(map(len, partition.indices)), 'client_max': max(map(len, partition.indices))}), flush=True)


def run(relative):
    job = load_job(relative)
    config = config_and_imports(job)
    from spade_dfl.data import DatasetBundle, PCAState
    from spade_dfl.trainer import run_experiment
    from spade_dfl.objectives import build_objective
    prepared = ROOT / 'prepared' / job['job_id']
    preflight = json.loads((prepared / 'preflight.json').read_text(encoding='utf-8'))
    if sha(prepared / 'data.npz') != preflight['data_sha256']:
        raise ValueError('Prepared data changed')
    out = new_target(ROOT, Path(job['output_dir']))
    out.mkdir(parents=True)
    started = time.perf_counter()
    save_json(out / 'STARTED.json', dict(job_id=job['job_id'], utc=datetime.now(timezone.utc).isoformat(),
        config_sha256=sha(ROOT / relative), runtime_sha256=sha(__file__), data_sha256=preflight['data_sha256'],
        python=sys.version, numpy=np.__version__, platform=platform.platform(), training_runs=1))
    try:
        with np.load(prepared / 'data.npz', allow_pickle=False) as data:
            for k, h in preflight['array_sha256'].items():
                assert array_hash(data[k]) == h
            metadata = json.loads((prepared / 'bundle_metadata.json').read_text(encoding='utf-8'))
            accepted = {f.name for f in fields(DatasetBundle)}
            metadata = {k: v for k, v in metadata.items() if k in accepted}
            bundle = DatasetBundle(train_x=data['train_x'], train_y=data['train_y'],
                test_x=data['validation_x'], test_y=data['validation_y'],
                pca=PCAState(data['pca_mean'], data['pca_components'], data['pca_scale']), **metadata)
            partition = make_partition(config, bundle.train_y, job['algorithm_seed'])
            for i, ids in enumerate(partition.indices):
                assert np.array_equal(ids, data[f'client_{i}'])
        # Passing the identical default partition makes the saved partition auditable;
        # run_experiment still creates the original client RNG streams and graph.
        result = run_experiment(config, bundle, job['algorithm_seed'], partition=partition)
        save_npz(out / 'raw_result.npz', states=result.states, **result.history)
        save_json(out / 'original_metadata.json', result.metadata)
        for values in result.history.values():
            assert np.isfinite(values).all()
        assert np.isfinite(result.states).all()
        assert result.history['accuracy'].shape == (config.n_rounds + 1,)
        assert np.allclose(result.metadata['release_noise_std_schedule'], preflight['sigma'])
        assert np.allclose(result.metadata['clip_norm_schedule'], preflight['clip'])
        epsilon = check_budget(result.metadata['clip_norm_schedule'], result.metadata['release_noise_std_schedule'],
                               config.target_delta, config.target_epsilon)
        assert np.isclose(result.metadata['beta'], preflight['beta'])
        assert result.metadata['client_sample_counts'] == preflight['client_sample_counts']
        kwargs = {} if job['dataset'] in {'mnist', 'fashion'} else dict(feature_dimension=config.input_dimension, n_classes=config.n_classes)
        objective = build_objective(config.objective_model, config.regularization, **kwargs)
        recomputed = float(objective.accuracy(result.states.mean(axis=0), bundle.test_x, bundle.test_y))
        assert np.isclose(recomputed, result.history['accuracy'][-1], rtol=0, atol=1e-14)
        assert np.allclose(result.history['released_proxy_accuracy'], result.history['accuracy'][1:])
        verify_modules(job)
        rows = []
        for k in range(config.n_rounds + 1):
            rows.append(dict(round=k, validation_accuracy=float(result.history['accuracy'][k]),
                train_accuracy=float(result.history['train_accuracy'][k]), train_loss=float(result.history['loss'][k]),
                clip_fraction=None if k == 0 else result.metadata['clip_fraction'][k-1],
                unclipped_update_norm_mean=None if k == 0 else result.metadata['unclipped_update_norm_mean'][k-1],
                sigma=None if k == 0 else preflight['sigma'][k-1], clip=None if k == 0 else preflight['clip'][k-1]))
        with (out / 'history.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        summary = dict(job_id=job['job_id'], status='COMPLETED', dataset=job['dataset'], fold=job['fold'],
            seed=job['algorithm_seed'], n_clients=config.n_clients, epsilon=epsilon,
            evaluation='training_holdout_validation_released_state_final_round',
            final_accuracy=recomputed, initial_accuracy=float(result.history['accuracy'][0]),
            best_accuracy=float(max(result.history['accuracy'])), best_round=int(np.argmax(result.history['accuracy'])),
            initial_train_loss=float(result.history['loss'][0]), final_train_loss=float(result.history['loss'][-1]),
            final_train_accuracy=float(result.history['train_accuracy'][-1]),
            clip_fraction_mean=float(np.mean(result.metadata['clip_fraction'])),
            clip_fraction_final=float(result.metadata['clip_fraction'][-1]),
            mean_unclipped_update_norm=float(np.mean(result.metadata['unclipped_update_norm_mean'])),
            sigma_first=preflight['sigma'][0], sigma_last=preflight['sigma'][-1],
            query_count=result.query_count, source_counter_transmitted_bytes=result.transmitted_bytes,
            trainer_wall_seconds=result.wall_time, worker_wall_seconds=time.perf_counter()-started,
            single_seed=True, final_table_eligible=False, end_to_end_privacy_certified=False,
            raw_result_sha256=sha(out / 'raw_result.npz'), history_sha256=sha(out / 'history.csv'))
        save_json(out / 'summary.json', summary)
        print(json.dumps(summary), flush=True)
    except Exception:
        save_json(out / 'FAILED.json', dict(job_id=job['job_id'], traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter()-started, automatically_retried=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['extract', 'prepare', 'run'])
    parser.add_argument('--config')
    args = parser.parse_args()
    if args.action == 'extract':
        extract_sources()
    else:
        {'prepare': prepare, 'run': run}[args.action](args.config)
