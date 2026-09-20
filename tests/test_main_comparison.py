import csv
import hashlib
import io
import json
import shutil

import pytest


def test_main_dataset_identity_is_exact():
    from spade_dfl_release.main_comparison import DATASETS
    assert DATASETS == ('mnist67', 'fashion01', 'alzheimer', 'brain_tumor_mri')


def test_main_reference_has_seven_methods_four_partitions_and_fifty_rounds(reference_root):
    from spade_dfl_release.main_comparison import verify
    report = verify(reference_root)
    assert (report['datasets'], report['methods'], report['partitions'], report['rounds']) == (4, 7, 4, 50)
    assert report['missing'] == []
    assert report['final_rows'] == 112
    assert report['curve_rows'] == 5712


@pytest.mark.parametrize('corruption', ['duplicate', 'missing', 'wrong_round', 'wrong_count', 'wrong_final'])
def test_main_verify_independently_rejects_corrupt_curves(tmp_path, reference_root, corruption):
    from spade_dfl_release.main_comparison import verify
    shutil.copytree(reference_root / 'aggregates', tmp_path / 'aggregates')
    shutil.copytree(reference_root / 'paper_values', tmp_path / 'paper_values')
    path = tmp_path / 'aggregates/main_roundwise.csv'
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8'))))
    if corruption == 'duplicate':
        rows.append(dict(rows[0]))
    elif corruption == 'missing':
        rows.pop()
    elif corruption == 'wrong_round':
        rows[-1]['round'] = '51'
    elif corruption == 'wrong_count':
        rows[-1]['n_trajectories'] = '29'
    else:
        rows[-1]['mean_percent'] = str(float(rows[-1]['mean_percent']) + 1)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    # Keep the local checksum internally consistent: structural and final-point
    # checks must reject the corrupted curve independently of the checksum.
    report_path = tmp_path / 'paper_values/verification.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    report['normalized_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError):
        verify(tmp_path)


def test_vendor_manifest_hashes_and_source_identities(release_root):
    from spade_dfl_release.main_comparison import load_manifest
    from spade_dfl_release.publication import publication_records, verify_publication_file
    manifest = load_manifest()
    assert set(manifest['datasets']) == {'mnist67', 'fashion01', 'alzheimer', 'brain_tumor_mri'}
    assert manifest['files']
    for row in manifest['files']:
        path = release_root / row['packaged_path']
        adaptation = publication_records().get(row['packaged_path'])
        expected = adaptation['published_sha256'] if adaptation else row['sha256']
        size = adaptation['published_bytes'] if adaptation else row['bytes']
        assert path.stat().st_size == size
        assert hashlib.sha256(path.read_bytes()).hexdigest().lower() == expected.lower()
        assert verify_publication_file(path, row['sha256'], row['bytes']) == expected
        assert row['source_id']
        assert ':' not in row['source_path']
        assert not row['source_path'].startswith('/')


def test_run_rejects_existing_output_without_launch(tmp_path):
    from spade_dfl_release.main_comparison import run
    with pytest.raises(ValueError, match='new directory'):
        run(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_run_main_rejects_unknown_dataset_without_writes(tmp_path):
    from spade_dfl_release.main_comparison import run_main
    output = tmp_path / 'run'
    with pytest.raises(ValueError, match='unsupported'):
        run_main('smoke', output, tmp_path / 'missing.json', tmp_path)
    assert not output.exists()


def test_runtime_staging_preserves_required_parent_relative_input(tmp_path):
    from spade_dfl_release.main_comparison import _stage_runtime
    vendor = tmp_path / 'source/package'
    vendor.mkdir(parents=True)
    (vendor / 'entry.py').write_bytes(b'# frozen code\n')
    (vendor.parent / 'dataset.zip').write_bytes(b'frozen dataset bytes')
    output = tmp_path / 'new-run'
    output.mkdir()
    runtime = _stage_runtime(vendor, output, [{'path': '../dataset.zip'}])
    assert (runtime / 'entry.py').read_bytes() == b'# frozen code\n'
    assert (runtime.parent / 'dataset.zip').read_bytes() == b'frozen dataset bytes'
    assert (vendor.parent / 'dataset.zip').read_bytes() == b'frozen dataset bytes'


@pytest.mark.parametrize('dataset', ['mnist67', 'fashion01', 'alzheimer', 'brain_tumor_mri'])
def test_runtime_imports_stay_in_the_selected_vendor(release_root, dataset):
    import subprocess
    import sys
    from pathlib import Path
    from spade_dfl_release.main_comparison import load_manifest
    spec = load_manifest()['datasets'][dataset]
    source = (release_root/spec['vendor_root']/spec['package_relative']/'src').resolve()
    module, function = spec['entrypoint'].rsplit('.', 1)
    code = """
import importlib,json,sys
sys.dont_write_bytecode=True
sys.path.insert(0,sys.argv[1])
module=importlib.import_module(sys.argv[2])
assert callable(getattr(module,sys.argv[3]))
if sys.argv[2].endswith('brain_tumor_suite'):
    module.runtime_protocol_paths()
print(json.dumps([str(x.__file__) for n,x in sys.modules.items() if n=='spade_dfl' or n.startswith('spade_dfl.')]))
"""
    completed = subprocess.run([sys.executable, '-B', '-c', code, str(source), module, function],
                               check=True, capture_output=True, text=True)
    imported = json.loads(completed.stdout)
    assert imported
    for path in imported:
        assert source in Path(path).resolve().parents


@pytest.mark.parametrize('dataset', ['mnist67', 'fashion01', 'alzheimer', 'brain_tumor_mri'])
def test_golden_historical_vs_release_seed0_iid(tmp_path, dataset):
    import os
    import subprocess
    import sys
    from pathlib import Path
    import numpy as np
    from spade_dfl_release.main_comparison import load_manifest
    from spade_dfl_release.paths import release_root
    from spade_dfl_release.reference import Sources, resolve_source_index
    if os.environ.get('SPADE_DFL_RUN_GOLDEN') != '1':
        pytest.skip('Set SPADE_DFL_RUN_GOLDEN=1 for author-workspace frozen-input golden runs')
    if dataset == 'brain_tumor_mri':
        pytest.skip('Brain archive-to-release equivalence has not been validated. The formal launcher requires the original MRI archive and CPU torch/torchvision dependencies; cached features alone do not satisfy these prerequisites.')
    release = release_root()
    manifest = load_manifest()
    spec = manifest['datasets'][dataset]
    source_index = resolve_source_index()
    if source_index is None:
        pytest.skip('Author source index is not configured; see docs/OPTIONAL_SOURCES.md')
    sources = Sources(source_index, release/'reference_results/raw_manifests/source_manifest.json')
    historical = tmp_path/'historical'
    packaged = tmp_path/'packaged'
    vendor = release/spec['vendor_root']
    shutil.copytree(vendor, packaged)
    try:
        archive = sources.archive(spec['source_id'])
        for row in manifest['files']:
            if row['dataset'] != dataset:
                continue
            relative = (release/row['packaged_path']).relative_to(vendor)
            destination = historical/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(row['source_path'])
            assert hashlib.sha256(data).hexdigest() == row['sha256']
            destination.write_bytes(data)
        for item in spec['required_inputs']:
            data = archive.read(item['source_path'])
            assert hashlib.sha256(data).hexdigest() == item['sha256']
            destination = historical/item['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    finally:
        sources.close()
    config = historical/(release/spec['config']).relative_to(vendor)
    assert config.read_bytes() == (release/spec['config']).read_bytes()
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    for label, runtime in (('historical', historical), ('packaged', packaged)):
        subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), dataset,
                        str(runtime/spec['package_relative']/'src'), str(config), str(tmp_path/label)],
                       env=environment, check=True)
    left = np.load(tmp_path/'historical.golden.npz')
    right = np.load(tmp_path/'packaged.golden.npz')
    assert set(left.files) == set(right.files)
    for key in left.files:
        np.testing.assert_array_equal(left[key], right[key], err_msg=dataset+'/'+key)
    a = json.loads((tmp_path/'historical.golden.json').read_text())
    b = json.loads((tmp_path/'packaged.golden.json').read_text())
    assert a == b
    assert a['seed'] == 0 and a['partition'] == 'iid'
    assert len(left['accuracy']) == 51
    assert left['final_model'].shape == (a['config']['n_clients'], a['config']['dimension'])


def _golden_worker():
    """Execute the frozen SPADE seed-0/IID case using each historical API.

    Both isolated processes read the same byte-identical frozen config/data;
    only their imported historical module tree differs. No fixture models or
    label histograms substitute for actual client indices or final states.
    """
    import ast
    from dataclasses import replace
    import inspect
    import os
    from pathlib import Path
    import sys
    import numpy as np
    dataset_name, source, config_path, output = sys.argv[1:]
    sys.path.insert(0, source)
    config_path, output = Path(config_path), Path(output)
    os.chdir(config_path.parent.parent)
    method, panel, seed = 'spade_dfl', 'iid', 0
    if dataset_name == 'alzheimer':
        from spade_dfl import alzheimer_v4_cross_validation as suite
        contract = suite._load_contract(config_path)
        tuning = suite._load_v3_tuning_config(contract.tuning_config_path)
        base = replace(tuning.experiment, n_rounds=contract.rounds, seeds=contract.seeds,
                       output_dir=str(output.parent/'declared-output'))
        candidate = suite._selected_candidates(contract)[method]
        split = suite.build_stratified_alzheimer_fold(contract.data_path, n_folds=contract.n_folds,
                                                     fold_index=0, split_seed=contract.split_seed)
        dataset = split.bundle
        # Evaluate the n_clients expression from the actual formal entrypoint.
        tree = ast.parse(inspect.getsource(suite.run_v4_cross_validation))
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_build_partition')
        clients = ast.literal_eval(next(k.value for k in call.keywords if k.arg == 'n_clients'))
        partition = suite._build_partition(dataset.train_y, n_clients=clients, partition=panel, seed=seed)
        config = suite._record_beta(suite._final_method_config(base, method, candidate, panel), candidate, seed)
    else:
        from spade_dfl import paper_suite as suite
        raw = json.loads(config_path.read_text(encoding='utf-8'))
        base = suite.load_config(config_path.parent/raw['base_config'])
        seeds = tuple(raw['seeds'])
        if dataset_name == 'mnist67':
            dataset = suite.load_mnist_binary(Path(base.data_path), dimension=base.dimension,
                                             download_url=base.download_url or None)
        else:
            dataset = suite.load_dataset_for_config(base)
        panel_index = raw['partitions'].index(panel)
        mode, alpha = suite._partition_spec(panel)
        # Execute the exact historical RNG expression; retain the IID panel's
        # original position among all four partitions, even in a one-case test.
        tree = ast.parse(inspect.getsource(suite.run_paper_suite))
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'partition_rng' for t in n.targets))
        partition_rng = eval(compile(ast.Expression(assignment.value), '<frozen partition RNG>', 'eval'), {'np':np,'seed':seed,'panel_index':panel_index})
        partition = suite.partition_labels(dataset.train_y, base.n_clients, mode, partition_rng,
                                           alpha=alpha, pathological_minor_fraction=base.pathological_minor_fraction)
        config = suite.method_config(base, method, seeds, mode, alpha)
        parameters = suite.resolve_method_parameters(raw, config_path)
        if method in parameters:
            config = suite.apply_method_parameters(config, method, parameters[method])
    result = suite.run_method(method, config, dataset, partition, seed)
    arrays = {'accuracy':result.history['accuracy'], 'final_model':result.states}
    arrays.update({'client_'+str(index):indices for index,indices in enumerate(partition.indices)})
    np.savez(str(output)+'.golden.npz', **arrays)
    metadata = dict(config=config.to_dict(), seed=seed, partition=panel,
                    query_count=result.query_count, communication_rounds=result.metadata['communication_rounds'])
    Path(str(output)+'.golden.json').write_text(json.dumps(metadata, sort_keys=True), encoding='utf-8')


if __name__ == '__main__':
    _golden_worker()
