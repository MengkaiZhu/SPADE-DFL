"""Historical 16-run paired-control adapter."""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import fields
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'table34_n30_eps8_working'
sys.path.insert(0, str(PREVIOUS))
import pilot_runtime as previous


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def arm_fields(base, arm):
    if arm not in {'A', 'B'}:
        raise ValueError('Only the two approved arms are supported')
    result = copy.deepcopy(base)
    if arm == 'A':
        result.update(stage='spade_dfl', target_epsilon=0, target_delta=0, dp_noise_std=0)
    return result


def validate_arm(base, arm, actual):
    if actual != arm_fields(base, arm):
        raise ValueError('Unapproved configuration change')


def suite_partition(dataset, config, labels, seed):
    from spade_dfl.partitioning import partition_labels
    if dataset in {'mnist', 'fashion'}:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0, 20260710]))
    elif dataset in {'brain', 'alzheimer'}:
        rng = np.random.default_rng(seed)
    else:
        raise ValueError('Dataset not approved')
    return partition_labels(labels, config['n_clients'], config['partition_mode'], rng,
                            alpha=config['dirichlet_alpha'],
                            pathological_minor_fraction=config['pathological_minor_fraction'])


class StateObserver:
    """Observe metric-boundary states only: no RNG, oracle calls, or state changes."""
    def __init__(self, recorder):
        self.recorder = recorder
        self.states = []

    def __call__(self, history, states, *args, **kwargs):
        self.states.append(np.array(states, copy=True))
        return self.recorder(history, states, *args, **kwargs)


def check_authorization():
    auth = read_json(ROOT / 'AUTHORIZATION.json')
    if auth['authorized_runs'] != 16 or auth['formal_training_authorized']:
        raise ValueError('Not authorized for this scope')
    for path, digest in auth['source_sha256'].items():
        if previous.sha(path) != digest:
            raise ValueError('Authorized source changed: ' + path)
    return auth


def freeze():
    check_authorization()
    manifest_path = previous.new_target(ROOT, 'MANIFEST.json')
    jobs = []
    for relative in read_json(PREVIOUS / 'pilot_manifest.json')['config_files']:
        base = previous.load_job(relative)
        for arm in ['A', 'B']:
            job = dict(pair_id=base['job_id'], arm=arm, dataset=base['dataset'], fold=base['fold'],
                seed=0, base_config_file=relative, base_config_sha256=previous.sha(PREVIOUS / relative),
                fields=arm_fields(base['optimizer_config_fields'], arm),
                evaluation='training_holdout_only_final_round', formal_table_eligible=False)
            path = ROOT / 'configs' / (base['job_id'] + '_' + arm + '.json')
            previous.save_json(path, job)
            jobs.append(dict(file=path.relative_to(ROOT).as_posix(), sha256=previous.sha(path)))
    previous.save_json(manifest_path, dict(authorized_runs=16, jobs=jobs,
        runtime_sha256=previous.sha(__file__), tests_sha256=previous.sha(ROOT / 'test_control_runtime.py'),
        all_original_sources_sha256={
            str(p.relative_to(PREVIOUS)): previous.sha(p) for p in (PREVIOUS / 'vendor').rglob('*.py')},
        previous_prepared_files_sha256={
            str(p.relative_to(PREVIOUS)): previous.sha(p) for p in (PREVIOUS / 'prepared').rglob('*') if p.is_file()},
        pause_after_16=True, parameter_search=False))
    print('FROZEN 16 configs; training not started', flush=True)


def load_job(relative):
    check_authorization()
    manifest = read_json(ROOT / 'MANIFEST.json')
    if previous.sha(__file__) != manifest['runtime_sha256']:
        raise ValueError('Runtime changed after freeze')
    selected = [item for item in manifest['jobs'] if item['file'] == relative]
    if len(selected) != 1 or previous.sha(ROOT / relative) != selected[0]['sha256']:
        raise ValueError('Job not frozen or changed')
    job = read_json(ROOT / relative)
    base = previous.load_job(job['base_config_file'])
    if previous.sha(PREVIOUS / job['base_config_file']) != job['base_config_sha256']:
        raise ValueError('Base config changed')
    validate_arm(base['optimizer_config_fields'], job['arm'], job['fields'])
    package = previous.verify_modules(base)
    sys.path.insert(0, str(package.parent))
    from spade_dfl.config import ExperimentConfig
    config = ExperimentConfig.from_dict(job['fields'])
    effective = json.loads(json.dumps(config.to_dict()))
    if any(effective[k] != value for k, value in job['fields'].items()):
        raise ValueError('Effective optimizer config does not match frozen fields')
    return job, base, config


def source_data(job):
    directory = PREVIOUS / 'prepared' / job['pair_id']
    manifest = read_json(ROOT / 'MANIFEST.json')
    for name in ['data.npz', 'bundle_metadata.json', 'preflight.json']:
        path = directory / name
        if previous.sha(path) != manifest['previous_prepared_files_sha256'][str(path.relative_to(PREVIOUS))]:
            raise ValueError('Prepared source changed')
    return directory


def prepare(relative):
    job, base, config = load_job(relative)
    if job['arm'] != 'A':
        raise ValueError('Prepare shared pair inputs once, using A configuration')
    out = previous.new_target(ROOT, Path('pairs') / job['pair_id'])
    source = source_data(job)
    prior = read_json(source / 'preflight.json')
    audit = next(a for a in read_json(PREVIOUS / 'OFFLINE_PARTITION_RULE_AUDIT.json') if a['job_id'] == job['pair_id'])
    if previous.sha(audit['source']) != audit['source_sha256']:
        raise ValueError('Suite rule source changed')
    from spade_dfl.graph import build_graph, derive_midpoint_beta
    from spade_dfl.initialization import initialize_model_states
    with np.load(source / 'data.npz', allow_pickle=False) as data:
        for key, digest in prior['array_sha256'].items():
            assert previous.array_hash(data[key]) == digest
        partition = suite_partition(job['dataset'], job['fields'], data['train_y'], job['seed'])
        assert [previous.array_hash(ids) for ids in partition.indices] == audit['partition_array_sha256']
        assert np.array_equal(np.sort(np.concatenate(partition.indices)), np.arange(len(data['train_y'])))
        assert min(map(len, partition.indices)) > 0
        graph = build_graph(config.n_clients, topology=config.graph_topology,
                            probability=config.graph_probability, seed=job['seed'])
        assert np.array_equal(graph.adjacency, data['adjacency'])
        labels, counts = np.unique(data['validation_y'], return_counts=True)
        train_labels, train_counts = np.unique(data['train_y'], return_counts=True)
        majority_label = train_labels[np.argmax(train_counts)]
        train_majority_accuracy = float(np.mean(data['validation_y'] == majority_label))
        preflight = dict(pair_id=job['pair_id'], data_path=str(source / 'data.npz'),
            data_sha256=previous.sha(source / 'data.npz'), suite_rule=audit,
            client_sample_counts=[len(ids) for ids in partition.indices],
            beta=derive_midpoint_beta(graph, config.tau, config.rho),
            adjacency_sha256=previous.array_hash(graph.adjacency),
            initialization_sha256=previous.array_hash(initialize_model_states(config, job['seed'], config.n_clients)),
            validation_count=int(counts.sum()), validation_class_counts=dict(zip(map(str, labels), map(int, counts))),
            validation_majority_fraction=float(counts.max()/counts.sum()),
            train_majority_label=float(majority_label), train_majority_validation_accuracy=train_majority_accuracy,
            validation_role_not_official_test=True, pca_refitted=False, source_array_hashes=prior['array_sha256'])
    out.mkdir(parents=True)
    previous.save_npz(out / 'partition.npz', **{f'client_{i}': ids for i, ids in enumerate(partition.indices)},
                      label_histograms=partition.label_histograms, adjacency=graph.adjacency)
    preflight['partition_file_sha256'] = previous.sha(out / 'partition.npz')
    previous.save_json(out / 'preflight.json', preflight)
    print('PREPARED ' + job['pair_id'], flush=True)


def run(relative):
    job, base, config = load_job(relative)
    source = source_data(job)
    pair = ROOT / 'pairs' / job['pair_id']
    preflight = read_json(pair / 'preflight.json')
    assert previous.sha(pair / 'partition.npz') == preflight['partition_file_sha256']
    out = previous.new_target(ROOT, Path('results') / job['pair_id'] / job['arm'])
    from spade_dfl.data import DatasetBundle, PCAState
    from spade_dfl.initialization import initialize_model_states
    import spade_dfl.trainer as trainer
    with np.load(source / 'data.npz', allow_pickle=False) as data, np.load(pair / 'partition.npz', allow_pickle=False) as partdata:
        metadata = read_json(source / 'bundle_metadata.json')
        accepted = {f.name for f in fields(DatasetBundle)}
        bundle = DatasetBundle(train_x=data['train_x'], train_y=data['train_y'],
            test_x=data['validation_x'], test_y=data['validation_y'],
            pca=PCAState(data['pca_mean'], data['pca_components'], data['pca_scale']),
            **{k:v for k,v in metadata.items() if k in accepted})
        partition = suite_partition(job['dataset'], job['fields'], bundle.train_y, job['seed'])
        for i, ids in enumerate(partition.indices):
            assert np.array_equal(ids, partdata[f'client_{i}'])
        assert previous.array_hash(initialize_model_states(config, 0, 30)) == preflight['initialization_sha256']
    out.mkdir(parents=True)
    start = time.perf_counter()
    previous.save_json(out / 'STARTED.json', dict(job_id=job['pair_id']+'_'+job['arm'],
        utc=datetime.now(timezone.utc).isoformat(), config_sha256=previous.sha(ROOT / relative),
        runtime_sha256=previous.sha(__file__), data_sha256=preflight['data_sha256'],
        partition_sha256=preflight['partition_file_sha256'], python=sys.version,
        numpy=np.__version__, platform=platform.platform(), training_calls_authorized=1))
    original_recorder = trainer.record_metrics
    observer = StateObserver(original_recorder)
    try:
        trainer.record_metrics = observer
        result = trainer.run_experiment(config, bundle, job['seed'], partition=partition)
        trajectory = np.stack(observer.states)
        previous.save_npz(out / 'raw_result.npz', states=result.states, states_by_round=trajectory, **result.history)
        previous.save_json(out / 'original_metadata.json', result.metadata)
        assert trajectory.shape == (51, 30, config.dimension)
        assert np.array_equal(trajectory[-1], result.states)
        assert all(np.isfinite(value).all() for value in result.history.values()) and np.isfinite(trajectory).all()
        assert result.metadata['client_sample_counts'] == preflight['client_sample_counts']
        assert np.isclose(result.metadata['beta'], preflight['beta'])
        if job['arm'] == 'B':
            previous.check_budget(result.metadata['clip_norm_schedule'], result.metadata['release_noise_std_schedule'],1e-5,8)
            assert np.allclose(result.metadata['clip_norm_schedule'], base['privacy_schedule_preview']['clip'])
            assert np.allclose(result.metadata['release_noise_std_schedule'], base['privacy_schedule_preview']['sigma'])
            assert np.array_equal(result.history['released_proxy_accuracy'], result.history['accuracy'][1:])
        else:
            assert result.metadata['epsilon'] is None and result.metadata['privacy_scope'] == 'none'
            assert 'release_noise_std_schedule' not in result.metadata and 'clip_fraction' not in result.metadata
        previous.verify_modules(base)
        step_norm = np.linalg.norm(np.diff(trajectory, axis=0), axis=2).mean(axis=1)
        rows=[]
        for k in range(51):
            rows.append(dict(round=k, validation_accuracy=float(result.history['accuracy'][k]),
                train_accuracy=float(result.history['train_accuracy'][k]), training_loss=float(result.history['loss'][k]),
                executed_state_change_norm_mean=None if k==0 else float(step_norm[k-1]),
                proposal_update_norm_mean=None if k==0 or job['arm']=='A' else result.metadata['unclipped_update_norm_mean'][k-1],
                clip_fraction=None if k==0 or job['arm']=='A' else result.metadata['clip_fraction'][k-1],
                release_sigma=None if k==0 or job['arm']=='A' else result.metadata['release_noise_std_schedule'][k-1]))
        with (out / 'history.csv').open('x', encoding='utf-8', newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        summary=dict(pair_id=job['pair_id'], arm=job['arm'], dataset=job['dataset'], fold=job['fold'], seed=0,
            n_clients=30, rounds=50, epsilon=result.metadata['epsilon'], status='COMPLETED',
            final_accuracy=float(result.history['accuracy'][-1]), initial_accuracy=float(result.history['accuracy'][0]),
            best_accuracy=float(np.max(result.history['accuracy'])), best_round=int(np.argmax(result.history['accuracy'])),
            initial_loss=float(result.history['loss'][0]), final_loss=float(result.history['loss'][-1]),
            validation_majority_fraction=preflight['validation_majority_fraction'],
            train_majority_validation_accuracy=preflight['train_majority_validation_accuracy'],
            mean_clip_fraction=None if job['arm']=='A' else float(np.mean(result.metadata['clip_fraction'])),
            query_count=result.query_count, source_transmitted_bytes=result.transmitted_bytes,
            trainer_wall_seconds=result.wall_time, worker_wall_seconds=time.perf_counter()-start,
            source_config_file=relative, source_config_sha256=previous.sha(ROOT/relative),
            shared_data_sha256=preflight['data_sha256'], shared_partition_sha256=preflight['partition_file_sha256'],
            shared_initialization_sha256=preflight['initialization_sha256'],
            raw_sha256=previous.sha(out/'raw_result.npz'), history_sha256=previous.sha(out/'history.csv'),
            single_seed_validation_only=True, final_table_eligible=False, end_to_end_privacy_certified=False)
        previous.save_json(out / 'summary.json', summary)
        print(json.dumps(summary), flush=True)
    except Exception:
        previous.save_json(out / 'FAILED.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
    finally:
        trainer.record_metrics = original_recorder


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['freeze','prepare','run'])
    parser.add_argument('--config')
    args=parser.parse_args()
    if args.action=='freeze':freeze()
    else: {'prepare':prepare,'run':run}[args.action](args.config)
