"""Hash-locked, selective import and independent verification of frozen results.

No experiment or dataset loader is called. Local source locations are supplied
only by the non-distributed source index; all exported provenance is relative.
"""
from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import os
import re
import tempfile
import zipfile
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
from scipy import stats

from .paths import release_root
from .statistics import mean_sd, mean_ci95, pool_fold_counts, benjamini_hochberg

METHODS = ('zo2_lt_admm_vr', 'zo2_lt_admm', 'zo2_dp_sgd', 'zo2_dp_fedavg', 'one_point_dsg', 'one_point_dsgt', 'spade_dfl')
PARTITIONS = ('iid', 'dir_0.6', 'dir_0.3', 'pathological')
DATASETS = ('mnist', 'fashion', 'alzheimer', 'brain')
MAIN_ROOTS = {
    'mnist': ('server_archive', 'SPADE-DFL/S7/S7_run_20260717_215502/SPADE-DFL-S7/results/server_50round_30seed/'),
    'fashion': ('server_archive', 'SPADE-DFL/S8/S8_run_20260720_120726/S8/results/fashion_server_50round_30seed/'),
    'alzheimer': ('server_archive', 'SPADE-DFL/S72/S72_run/S72/results/server_5fold_30seed_50round/'),
    'brain': ('brain_s0904_archive', 'spade_dfl_brain_tumor_resnet18_pca10/results/server/'),
}
S15_ROOT = 'SPADE-DFL/S15/threefig/results/mnist67_formal_three_figures/'
SOURCE_INDEX_ENV = 'SPADE_DFL_SOURCE_INDEX'


def resolve_source_index(source_index=None, *, required=False):
    """Resolve an explicitly configured, non-distributed source index."""
    configured = source_index if source_index is not None else os.environ.get(SOURCE_INDEX_ENV)
    if not configured:
        if required:
            raise FileNotFoundError(
                f'author source index unavailable; pass --source-index or set {SOURCE_INDEX_ENV}'
            )
        return None
    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'configured author source index does not exist: {path}')
    return path


def require_keys(keys, expected):
    keys = list(keys)
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate logical keys')
    if set(keys) != set(expected):
        raise ValueError(f'missing/unexpected logical keys: missing={len(set(expected)-set(keys))}, extra={len(set(keys)-set(expected))}')


def verify_hash(path, expected):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    actual = digest.hexdigest()
    if actual.lower() != expected.lower():
        raise ValueError('source hash mismatch: ' + Path(path).name)
    return actual


def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))


def read_history(data, rounds=50):
    rows = csv_rows(data)
    if not rows:
        raise ValueError('empty history')
    axis = 'round' if 'round' in rows[0] else 'iteration'
    if [int(r[axis]) for r in rows] != list(range(rounds + 1)):
        raise ValueError('history must contain exactly ordered rounds 0..50')
    values = np.asarray([float(r['accuracy']) for r in rows])
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError('invalid accuracy')
    return values


def assert_close(actual, expected, label):
    if not np.allclose(actual, expected, rtol=1e-10, atol=1e-10):
        raise ValueError('independent source comparison differs: ' + label)


class Sources:
    def __init__(self, source_index, manifest):
        source_index = resolve_source_index(source_index, required=True)
        specs = json.loads(source_index.read_text(encoding='utf-8'))['sources']
        # Relative source roots travel with the private index, independent of
        # the release location or the caller's current working directory.
        self.paths = {s['id']: (source_index.parent / s['root']).resolve() for s in specs}
        entries = json.loads(Path(manifest).read_text(encoding='utf-8'))['files']
        require_keys([(e['source_id'], e['path']) for e in entries], {(e['source_id'], e['path']) for e in entries})
        self.hashes = {(e['source_id'], e['path']): e['sha256'] for e in entries}
        self.archives = {}
        self.used = {}

    def file(self, source, relative=None):
        base = self.paths[source]
        path = base / relative if relative else base
        key = (source, relative if relative else base.name)
        if key not in self.hashes:
            raise ValueError('missing source hash: ' + repr(key))
        self.used['/'.join(key)] = verify_hash(path, self.hashes[key])
        return path.read_bytes()

    def archive(self, source):
        if source not in self.archives:
            path = self.paths[source]
            key = (source, path.name)
            if key not in self.hashes:
                raise ValueError('missing archive hash')
            self.used['/'.join(key)] = verify_hash(path, self.hashes[key])
            archive = zipfile.ZipFile(path)
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('duplicate archive members')
            self.archives[source] = archive
        return self.archives[source]

    def close(self):
        for archive in self.archives.values():
            archive.close()


def validate_suite_manifest(manifest, partitions, methods, folds):
    if manifest['status'] != 'completed':
        raise ValueError('formal suite not completed')
    config = manifest.get('suite_config', manifest)
    require_keys(config['seeds'], range(30))
    require_keys(config['methods'], methods)
    require_keys(config.get('partitions', config.get('panels', [])), partitions)
    if 'rounds' in manifest and manifest['rounds'] != 50:
        raise ValueError('formal manifest round mismatch')
    if 'folds' in manifest and manifest['folds'] != folds:
        raise ValueError('formal manifest fold mismatch')
    expected_count = len(partitions) * len(methods) * folds * 30
    for field in ('expected_run_count', 'completed_run_count'):
        if field in manifest and manifest[field] != expected_count:
            raise ValueError('formal manifest run count mismatch')
    if 'runs' in manifest:
        require_keys([(r['partition'],r['method'],int(r['seed'])) for r in manifest['runs']], itertools.product(partitions, methods, range(30)))


def load_suite(archive, prefix, partitions, methods, folds=1):
    manifest_name = prefix + ('suite_manifest.json' if prefix+'suite_manifest.json' in archive.namelist() else 'run_manifest.json')
    manifest = json.loads(archive.read(manifest_name))
    validate_suite_manifest(manifest, partitions, methods, folds)
    histories = sorted(n for n in archive.namelist() if n.startswith(prefix) and n.endswith('/history.csv'))
    runs, summaries = {}, {}
    for name in histories:
        relative = name[len(prefix):]
        match = re.fullmatch(r'(?:runs|raw)/(?:fold_(\d+)/)?([^/]+)/([^/]+)/(?:[^/]+/)?seed_(\d+)/history.csv', relative)
        if not match:
            raise ValueError('unrecognized formal history path: ' + relative)
        fold, partition, method, seed = match.groups()
        key = (partition, method, int(fold or 0), int(seed))
        if key in runs:
            raise ValueError('duplicate formal run')
        runs[key] = read_history(archive.read(name))
        summary = json.loads(archive.read(name.replace('history.csv', 'summary.json')))
        assert_close(runs[key][-1], summary['final_accuracy'], 'history final versus summary')
        if int(summary['metadata']['seed']) != key[-1]:
            raise ValueError('summary seed mismatch')
        central = method == 'zo2_dp_sgd'
        if int(summary['communication_rounds']) != (0 if central else 50):
            raise ValueError('communication round semantics mismatch')
        if central and int(summary['metadata']['n_clients']) != 1:
            raise ValueError('DP-SGD requires N=1')
        summaries[key] = summary
    require_keys(runs, itertools.product(partitions, methods, range(folds), range(30)))
    return runs, summaries


def aggregate_main_curves(matrix):
    # Decimal accumulation of the CSV decimal values avoids binary summation
    # error changing a display-rounding tie; no epsilon or result correction.
    mean = np.asarray([float(sum(Decimal(str(v)) for v in column) * 100 / len(column)) for column in matrix.T])
    return mean, matrix.std(axis=0, ddof=0) * 100


def format_display(value, decimals):
    return str(Decimal(str(value)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))


def import_main(sources):
    final, curves = [], []
    for dataset in DATASETS:
        source, prefix = MAIN_ROOTS[dataset]
        archive = sources.archive(source)
        folds = 5 if dataset == 'alzheimer' else 1
        runs, _ = load_suite(archive, prefix, PARTITIONS, METHODS, folds)
        for partition, method in itertools.product(PARTITIONS, METHODS):
            matrix = np.asarray([runs[(partition, method, f, s)] for f in range(folds) for s in range(30)])
            mean, sd = aggregate_main_curves(matrix)
            base = dict(dataset=dataset, partition=partition, method=method, n_trajectories=len(matrix), ddof=0, source_id=source, source_member_prefix=prefix)
            final.append(dict(base, round=50, mean_percent=float(mean[-1]), sd_percent=float(sd[-1])))
            for round_index in range(51):
                curves.append(dict(base, round=round_index, mean_percent=float(mean[round_index]), sd_percent=float(sd[round_index])))
    return final, curves


def import_sensitivity(sources):
    rows = csv_rows(sources.file('sensitivity_tables34', 'RUNS.csv'))
    frozen = csv_rows(sources.file('sensitivity_tables34', 'CELLS.csv'))
    conditions = {(n, 32) for n in (30, 40, 50, 75, 100)} | {(100, e) for e in (4, 8, 16, 24, 32)}
    groups = defaultdict(list)
    keys = []
    for row in rows:
        dataset = row['dataset']
        key = (dataset, int(row['n']), int(float(row['epsilon'])), int(row['seed']), int(row['fold']))
        keys.append(key)
        if row['status'] != 'completed' or int(row['rounds']) != (25 if dataset == 'alzheimer' else 10):
            raise ValueError('invalid validation run status/round count')
        groups[key[:3]].append(row)
    expected = {(d, n, e, s, f) for d in DATASETS for n, e in conditions for s in range(5) for f in range(5 if d == 'alzheimer' else 1)}
    require_keys(keys, expected)
    require_keys([(r['dataset'], int(r['n']), int(float(r['epsilon']))) for r in frozen], groups)
    frozen_index = {(r['dataset'], int(r['n']), int(float(r['epsilon']))): r for r in frozen}
    result = []
    for key, cell in sorted(groups.items()):
        dataset, n, epsilon = key
        values = list(pool_fold_counts(cell).values()) if dataset == 'alzheimer' else [float(r['accuracy']) for r in sorted(cell, key=lambda r: int(r['seed']))]
        mean, sd = mean_sd(np.asarray(values) * 100, ddof=1, expected_count=5)
        saved = frozen_index[key]
        assert_close([mean, sd], [float(saved['mean_percent']), float(saved['sd_percent'])], 'validation CELLS')
        origins = {r['origin'] for r in cell}
        if origins != {saved['origin']}:
            raise ValueError('validation origin mismatch')
        result.append(dict(dataset=dataset, n=n, epsilon=epsilon, mean_percent=mean, sd_percent=sd, n_seed=5, n_fold_seed=len(cell), ddof=1, rounds=int(cell[0]['rounds']), origin=saved['origin'], validation_only=True, formal_authorized=False, previous_test_exposure=True))
    return result


def import_communication(sources):
    raw = csv_rows(sources.file('communication_figure10', 'results/communication_cost_per_seed.csv'))
    saved = csv_rows(sources.file('communication_figure10', 'results/communication_cost_summary.csv'))
    config = json.loads(sources.file('communication_figure10', 'results/experiment_metadata.json'))['config']
    n_clients = int(config['n_clients'])
    dimension = int(config['dimension'])
    samples_per_client = int(config['samples_per_client'])
    batch_size = int(config['batch_size'])
    for record in raw:
        rounds, total_steps = int(record['rounds']), int(record['T'])
        expected_accounting = {
            'query_count': n_clients * (rounds * samples_per_client + total_steps * batch_size),
            'transmitted_scalars': 2 * n_clients * dimension * rounds,
        }
        for field, expected in expected_accounting.items():
            if int(record[field]) != expected:
                raise ValueError('communication accounting formula mismatch: ' + field)
    conditions = {(t, tau) for t, local_tau in ((1000, 10), (3375, 15), (10648, 22), (29791, 31)) for tau in (1, local_tau)}
    require_keys([(int(r['T']), int(r['tau']), int(r['seed'])) for r in raw], {(t, tau, s) for t, tau in conditions for s in range(30)})
    require_keys([(int(r['T']), int(r['tau'])) for r in saved], conditions)
    frozen = {(int(r['T']), int(r['tau'])): r for r in saved}
    output = []
    for t, tau in sorted(conditions):
        rows = sorted([r for r in raw if (int(r['T']), int(r['tau'])) == (t, tau)], key=lambda r: int(r['seed']))
        row = dict(T=t, tau=tau, n_seeds=30, diagnostic='non_private', paired_by='seed')
        for name in ('schedule', 'rounds', 'local_steps', 'query_count', 'transmitted_scalars'):
            values = {r[name] for r in rows}
            if len(values) != 1:
                raise ValueError('nonconstant communication accounting')
            row[name] = next(iter(values)) if name == 'schedule' else int(next(iter(values)))
        if row['rounds'] * tau != t or row['local_steps'] != t:
            raise ValueError('communication budget mismatch')
        row['communication_reduction_vs_tau1'] = t / row['rounds']
        for metric in ('criterion', 'stationarity', 'consensus'):
            mean, low, high = mean_ci95([float(r[metric]) for r in rows])
            row.update({metric+'_mean': mean, metric+'_ci95_low': low, metric+'_ci95_high': high})
        for name, value in frozen[(t, tau)].items():
            if name != 'schedule':
                assert_close(float(row[name]), float(value), 'communication ' + name)
        output.append(row)
    return output


def import_s15(sources):
    archive = sources.archive('server_archive')
    partitions = PARTITIONS + ('dir_0.1',)
    dist, dist_summaries = load_suite(archive, S15_ROOT+'distribution_suite/', partitions, ('spade_dfl_private',))
    methods, _ = load_suite(archive, S15_ROOT+'method_comparison_suite/', PARTITIONS, METHODS)
    roundwise = []
    for suite, runs in (('distribution', dist), ('method', methods)):
        for (partition, method, fold, seed), values in sorted(runs.items()):
            for round_index, accuracy in enumerate(values):
                central = method == 'zo2_dp_sgd'
                roundwise.append(dict(suite=suite, partition=partition, method=method, seed=seed, round=round_index, accuracy=float(accuracy), axis_semantics='centralized_optimizer_step' if central else 'communication_round', communication_round=0 if central else round_index, n_clients=1 if central else (1000 if suite == 'distribution' else 31)))
    figure1 = json.loads(archive.read(S15_ROOT+'paper_figures/figure1_distribution_accuracy.json'))
    figure2 = json.loads(archive.read(S15_ROOT+'paper_figures/figure2_client_label_heterogeneity.json'))
    clients, distribution = [], []
    require_keys(figure1['source_data'], partitions)
    require_keys(figure2['source_data'], partitions)
    for partition in partitions:
        matrix = np.asarray([dist[(partition, 'spade_dfl_private', 0, s)] for s in range(30)]) * 100
        saved = figure1['source_data'][partition]
        assert_close(matrix, saved['accuracy_by_seed_percent'], 'S15 Figure 7 trajectories')
        stats_by_round = np.asarray([mean_ci95(matrix[:, r]) for r in range(51)])
        for index, name in enumerate(('mean_percent', 'confidence_interval_95_lower_percent', 'confidence_interval_95_upper_percent')):
            assert_close(stats_by_round[:, index], saved[name], 'S15 Figure 7 ' + name)
        for r in range(51):
            distribution.append(dict(partition=partition, round=r, mean_percent=stats_by_round[r, 0], sd_percent=matrix[:, r].std(ddof=0), ci95_low=stats_by_round[r, 1], ci95_high=stats_by_round[r, 2]))
        proportions = []
        for seed in range(30):
            summary = dist_summaries[(partition, 'spade_dfl_private', 0, seed)]
            hist = np.asarray(summary['metadata']['label_histograms'])
            if hist.shape != (1000, 2) or np.any(hist < 0) or np.any(hist.sum(axis=1) <= 0):
                raise ValueError('invalid Figure 8 client histograms')
            p = hist[:, 1] / hist.sum(axis=1)
            proportions.extend(p)
            clients.extend(dict(partition=partition, seed=seed, client=client, p_y7=float(v)) for client, v in enumerate(p))
        assert_close(proportions, figure2['source_data'][partition]['p_y7'], 'Figure 8 client proportions')
        p = np.asarray(proportions)
        entropy = stats.entropy(np.stack([p, 1-p]), base=2, axis=0).mean()
        assert_close(entropy, figure2['source_data'][partition]['mean_binary_entropy'], 'Figure 8 entropy')
    saved_tests = json.loads(archive.read(S15_ROOT+'paper_figures/figure3_method_volcano.json'))['source_data']
    expected = set(itertools.product(PARTITIONS, METHODS[1:], range(1, 51)))
    require_keys([(r['partition'], r['method'], r['round']) for r in saved_tests], expected)
    tests = []
    for saved in saved_tests:
        partition, method, r = saved['partition'], saved['method'], saved['round']
        # The registered source computes percentage-point differences after percent conversion.
        differences = np.asarray([methods[(partition, method, 0, s)][r]*100 - methods[(partition, METHODS[0], 0, s)][r]*100 for s in range(30)])
        assert_close(differences, saved['paired_differences_pp'], 'paired same-seed differences')
        statistic, p = (0., 1.) if np.all(differences == 0) else stats.wilcoxon(differences, alternative='two-sided', zero_method='wilcox', method='auto')
        tests.append(dict(partition=partition, method=method, round=r, n_pairs=30, mean_paired_accuracy_difference_pp=float(differences.mean()), wilcoxon_statistic=float(statistic), raw_p_value=float(p), paired_differences_pp=json.dumps(differences.tolist(), separators=(',', ':'))))
    q_values = benjamini_hochberg([r['raw_p_value'] for r in tests])
    for row, q, saved in zip(tests, q_values, saved_tests):
        row['adjusted_q_value'] = float(q)
        row['minus_log10_q'] = float(-np.log10(max(q, np.finfo(float).tiny)))
        row['classification'] = 'unresolved' if q >= .05 or row['mean_paired_accuracy_difference_pp'] == 0 else ('better' if row['mean_paired_accuracy_difference_pp'] > 0 else 'worse')
        for name in ('mean_paired_accuracy_difference_pp', 'wilcoxon_statistic', 'raw_p_value', 'adjusted_q_value', 'minus_log10_q'):
            assert_close(row[name], saved[name], 'S15 Wilcoxon/BH ' + name)
        if row['classification'] != saved['classification']:
            raise ValueError('S15 classification differs')
    return roundwise, distribution, clients, tests


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def paper_table(text, label):
    marker = '\\label{' + label + '}'
    if text.count(marker) != 1:
        raise ValueError('missing/duplicate manuscript table ' + label)
    after = text.split(marker, 1)[1]
    return after.split('\\bottomrule', 1)[0]


def parse_communication_prose(text):
    marker = r'\subsection{Communication--Optimization Trade-off}'
    if text.count(marker) != 1:
        raise ValueError('missing/duplicate Figure 10 section')
    section = text.split(marker)[1].split(r'\subsection', 1)[0]
    means = re.search(r'mean value was \$(\d+\.\d+)\$ under the local-update schedule and \$(\d+\.\d+)\$ under \$\\tau=1\$', section)
    volume = re.search(r'At \$T=([\d{},]+)\$, the transmitted volume decreased from \$([\d{},]+)\$ to \$([\d{},]+)\$ model scalars, corresponding to a reduction of approximately \$(\d+\.\d+)\\%\$', section)
    if not means or not volume:
        raise ValueError('Figure 10 manuscript prose schema mismatch')
    numbers = [re.sub(r'[{},]', '', value) for value in volume.groups()]
    return dict(local_mean=means[1], baseline_mean=means[2], budget=numbers[0], baseline_scalars=numbers[1], local_scalars=numbers[2], reduction_percent=numbers[3])


def publish_outputs(root, outputs, report_text):
    root = Path(root)
    destinations = {root/'aggregates'/name:data for name,data in outputs.items()}
    destinations[root/'paper_values/verification.json'] = report_text.encode('utf-8')
    for destination,data in destinations.items():
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError('refusing to overwrite different normalized output: '+destination.name)
    for destination,data in destinations.items():
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)


def verify_current_paper(reference_root, manuscript):
    root = Path(reference_root)
    report = json.loads((root/'paper_values/verification.json').read_text(encoding='utf-8'))
    expected_files = {'main_final_accuracy.csv', 'main_roundwise.csv', 'sensitivity_cells.csv', 'communication_summary.csv', 's15_roundwise.csv', 's15_distribution_summary.csv', 's15_client_proportions.csv', 's15_wilcoxon_bh.csv'}
    require_keys(report.get('normalized_sha256', {}), expected_files)
    for name, digest in report['normalized_sha256'].items():
        verify_hash(root/'aggregates'/name, digest)
    current = _verify_paper_values(root, manuscript)
    return dict(report, **current)


def _verify_paper_values(reference_root, manuscript):
    root = Path(reference_root)
    main = csv_rows((root/'aggregates/main_final_accuracy.csv').read_bytes())
    sensitivity = csv_rows((root/'aggregates/sensitivity_cells.csv').read_bytes())
    communication = csv_rows((root/'aggregates/communication_summary.csv').read_bytes())
    require_keys([(r['dataset'], r['partition'], r['method']) for r in main], itertools.product(DATASETS, PARTITIONS, METHODS))
    conditions = {(n, 32) for n in (30, 40, 50, 75, 100)} | {(100, e) for e in (4, 8, 16, 24, 32)}
    require_keys([(r['dataset'], int(r['n']), int(r['epsilon'])) for r in sensitivity], {(d,n,e) for d in DATASETS for n,e in conditions})
    text = Path(manuscript).read_text(encoding='utf-8-sig')
    pair_pattern = r'(\d+\.\d{2})\{?\\pm\}?(\d+\.\d{2})'
    table = paper_table(text, 'tab:distribution_final_accuracy')
    pairs = re.findall(pair_pattern, table)
    deltas = re.findall(r'\$([+]\d+\.\d{2})\$', table)
    if len(pairs) != 112 or len(deltas) != 16:
        raise ValueError('manuscript main table schema mismatch')
    index = {(r['dataset'],r['partition'],r['method']): r for r in main}
    differences = []
    fields = 0
    def compare(value, displayed, label):
        nonlocal fields
        fields += 1
        decimals = len(displayed.split('.')[-1]) if '.' in displayed else 0
        actual = format_display(value, decimals)
        if actual != displayed.lstrip('+'):
            differences.append(dict(cell=label, calculated=actual, manuscript=displayed))
    for key, pair in zip(itertools.product(DATASETS, PARTITIONS, METHODS), pairs):
        for name, displayed in zip(('mean_percent','sd_percent'),pair):
            compare(index[key][name], displayed, '/'.join(key)+'/'+name)
    for (d,p), displayed in zip(itertools.product(DATASETS, PARTITIONS), deltas):
        delta = float(index[(d,p,'spade_dfl')]['mean_percent']) - max(float(index[(d,p,m)]['mean_percent']) for m in METHODS[:-1])
        compare(delta, displayed, d+'/'+p+'/delta_best')
    sensitivity_index = {(r['dataset'],int(r['n']),int(r['epsilon'])):r for r in sensitivity}
    for label, axis in (('tab:client_count_accuracy', [(n,32) for n in (30,40,50,75,100)]), ('tab:privacy_budget_accuracy', [(100,e) for e in (4,8,16,24,32)])):
        pairs = re.findall(pair_pattern,paper_table(text,label))
        if len(pairs) != 20:
            raise ValueError('manuscript sensitivity schema mismatch')
        for (d,(n,e)),pair in zip(itertools.product(DATASETS,axis),pairs):
            for name, displayed in zip(('mean_percent','sd_percent'),pair):
                compare(sensitivity_index[(d,n,e)][name],displayed,f'{label}/{d}/{n}/{e}/{name}')
    displayed = parse_communication_prose(text)
    terminal = sorted([r for r in communication if int(r['T']) == max(int(x['T']) for x in communication)], key=lambda r:int(r['tau']))
    if len(terminal) != 2:
        raise ValueError('Figure 10 manuscript criterion schema mismatch')
    for row, prefix in zip(terminal, ('baseline','local')):
        compare(row['criterion_mean'], displayed[prefix+'_mean'], 'figure10/criterion/tau='+row['tau'])
        compare(row['transmitted_scalars'], displayed[prefix+'_scalars'], 'figure10/scalars/tau='+row['tau'])
    compare(terminal[0]['T'], displayed['budget'], 'figure10/largest_budget')
    compare((1-float(terminal[1]['transmitted_scalars'])/float(terminal[0]['transmitted_scalars']))*100, displayed['reduction_percent'], 'figure10/reduction_percent')
    result = dict(schema_version=1, missing_cells=0, duplicate_cells=0, displayed_numeric_fields=fields, differences=differences, difference_count=len(differences), status='verified' if not differences else 'blocked')
    if differences:
        raise ValueError('manuscript display precision mismatch: '+json.dumps(differences))
    return result


def import_reference_results(source_index=None, output_root=None):
    release = release_root()
    source_index = resolve_source_index(source_index, required=True)
    output_root = Path(output_root) if output_root else release/'reference_results'
    sources = Sources(source_index, release/'reference_results/raw_manifests/source_manifest.json')
    try:
        manuscript_bytes = sources.file('paper_tex')
        main, curves = import_main(sources)
        sensitivity = import_sensitivity(sources)
        communication = import_communication(sources)
        s15, distribution, clients, tests = import_s15(sources)
        outputs = {'main_final_accuracy.csv':main, 'main_roundwise.csv':curves, 'sensitivity_cells.csv':sensitivity, 'communication_summary.csv':communication, 's15_roundwise.csv':s15, 's15_distribution_summary.csv':distribution, 's15_client_proportions.csv':clients, 's15_wilcoxon_bh.csv':tests}
        # Every family is validated before any published output is touched.
        with tempfile.TemporaryDirectory(prefix='spade-reference-') as temporary:
            stage = Path(temporary)
            for name, rows in outputs.items():
                write_csv(stage/'aggregates'/name, rows)
            manuscript = stage/'paper.tex'
            manuscript.write_bytes(manuscript_bytes)
            verification = _verify_paper_values(stage, manuscript)
            verification.update(row_counts={name:len(rows) for name,rows in outputs.items()}, source_hashes=dict(sorted(sources.used.items())), independent_checks=['main raw histories and final summaries', 'validation RUNS versus CELLS', 'Figure 7 raw histories and Student-t intervals', 'Figure 8 raw client label histograms', 'Figure 9 paired Wilcoxon and BH over 1200 tests', 'Figure 10 240 raw runs versus eight summaries'], semantics={'main_sd_ddof':0,'sensitivity_sd_ddof':1,'alzheimer_main':'150 direct fold-seed trajectories','alzheimer_sensitivity':'pool correct/eval_count within each seed','dp_sgd':'N=1; zero communication; horizontal index is centralized optimizer step','sensitivity':'validation_only; formal_authorized=false; previous_test_exposure=true'})
            verification['normalized_sha256'] = {name:hashlib.sha256((stage/'aggregates'/name).read_bytes()).hexdigest() for name in sorted(outputs)}
            encoded = json.dumps(verification, indent=2, sort_keys=True, allow_nan=False)+'\n'
            publish_outputs(output_root, {name:(stage/'aggregates'/name).read_bytes() for name in outputs}, encoded)
            return verification
    finally:
        sources.close()
