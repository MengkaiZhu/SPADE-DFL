import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest

from spade_dfl_release import heterogeneity as h


def test_comparison_count_is_tests_not_seeds():
    assert h.expected_comparison_count(6,4,50)==1200
    assert h.expected_comparison_count(2,3,7)==42


@pytest.mark.parametrize('values',[[],[[.2,.5]],[np.nan],[np.inf],[-.1],[1.1]])
def test_bh_rejects_invalid_inputs(values):
    with pytest.raises(ValueError):h.benjamini_hochberg(values)


def test_bh_general_rank_ties_zero_one():
    p=np.array([1.,0.,.04,.04,.01])
    q=h.benjamini_hochberg(p)
    assert np.array_equal(q,np.array([1.,0.,.05,.05,.025]))
    assert np.all(np.diff(q[np.argsort(p)])>=0)
    assert np.all((q>=p)&(q<=1))


def test_cross_environment_comparison_allows_only_one_ulp():
    value=np.float64(.1);neighbor=np.nextafter(value,np.inf)
    assert h._same_or_adjacent(value,value)
    assert h._same_or_adjacent(value,neighbor)
    assert not h._same_or_adjacent(value,np.nextafter(neighbor,np.inf))


def test_byte_provenance_and_isolated_imports(release_root):
    from spade_dfl_release.publication import publication_records, verify_publication_file
    manifest=h.load_manifest()
    for row in manifest['files']:
        path=release_root/row['packaged_path']
        if not path.is_file():
            assert row['packaged_path'].endswith('/data/mnist.npz')
            continue
        data=path.read_bytes()
        adaptation=publication_records().get(row['packaged_path'])
        expected=adaptation['published_sha256'] if adaptation else row['sha256']
        size=adaptation['published_bytes'] if adaptation else row['bytes']
        assert hashlib.sha256(data).hexdigest()==expected
        assert len(data)==size
        assert verify_publication_file(path,row['sha256'],row['bytes'])==expected
        assert not Path(row['member']).is_absolute()
    imports=h._check_imports()
    assert imports and all(p.startswith('src/spade_dfl/') for p in imports)


def test_read_only_manifest_allows_absent_external_input_but_checks_present_bytes(tmp_path):
    relative='src/spade_dfl_release/vendor/heterogeneity/data/mnist.npz'
    size=11490434;digest='731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1'
    manifest={
        'files':[{'packaged_path':relative,'bytes':size,'sha256':digest}],
        'required_inputs':[{'path':'data/mnist.npz','bytes':size,'sha256':digest}],
    }
    h._verify_packaged_files(tmp_path,manifest)
    target=tmp_path/relative;target.parent.mkdir(parents=True);target.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='vendor path/size|hash mismatch'):
        h._verify_packaged_files(tmp_path,manifest)


@pytest.mark.parametrize('mutation',['path','bytes','sha256','extra'])
def test_external_input_identity_is_independently_anchored(mutation):
    import copy
    manifest=copy.deepcopy(h.load_manifest())
    if mutation=='extra':
        manifest['required_inputs'].append(dict(manifest['required_inputs'][0],path='src/spade_dfl/__init__.py'))
    else:
        manifest['required_inputs'][0][mutation]={'path':'src/spade_dfl/__init__.py','bytes':1,'sha256':'0'*64}[mutation]
    with pytest.raises(ValueError,match='external-input'):
        h._verify_packaged_files(h.release_root(),manifest)


def test_closure_has_independently_expected_36_paths():
    modules=('__init__ artifacts baselines config data distribution_study dp_baselines dsg dsgt evaluation first_order_admm graph initialization methods metrics mnist67_method_volcano mnist67_paper_evidence mnist67_paper_protocol mnist67_paper_runner mnist67_selected_three_figures mnist67_spade_distribution objectives paper_suite partitioning privacy trainer two_point vr zo').split()
    configs=('mnist67_paper_base mnist67_spade_distribution_50round_30seed mnist67_spade_distribution_parameters spade_mnist_50round_base spade_mnist_50round_server').split()
    expected={f'src/spade_dfl/{name}.py' for name in modules}|{f'configs/{name}.json' for name in configs}|{'scripts/run_mnist67_selected_three_figures.py','data/mnist.npz'}
    records=h.load_manifest()['files']
    assert len(records)==len(expected)==36
    assert {(r['member'],r['packaged_path']) for r in records}=={('SPADE-DFL/S15/threefig/'+p,'src/spade_dfl_release/vendor/heterogeneity/'+p) for p in expected}


@pytest.mark.parametrize('mutation',['removed','source_path'])
def test_closure_lock_rejects_self_consistent_manifest_mutation(monkeypatch,mutation):
    original=json.loads
    def read(text,*args,**kwargs):
        value=original(text,*args,**kwargs)
        if isinstance(value,dict) and 'archive_prefix' in value:
            if mutation=='removed':value['files'].pop()
            else:value['files'][0]['member']='SPADE-DFL/S15/threefig/unrelated.json'
        return value
    monkeypatch.setattr(h.json,'loads',read)
    with pytest.raises(ValueError,match='closure'):h.load_manifest()


@pytest.mark.skipif(os.environ.get('SPADE_DFL_AUTHOR_ARCHIVE_QA')!='1',reason='opt-in read-only author archive QA; no training')
def test_author_archive_qa_all_runtime_and_990_run_evidence(release_root):
    from spade_dfl_release.publication import verify_publication_file
    from spade_dfl_release.reference import Sources,resolve_source_index
    spec=resolve_source_index()
    if spec is None:pytest.skip('author source index is not configured; see docs/OPTIONAL_SOURCES.md')
    source=Sources(spec,release_root/'reference_results/raw_manifests/source_manifest.json')
    if not source.paths['server_archive'].is_file():pytest.skip('hash-frozen server archive unavailable')
    try:
        archive=source.archive('server_archive');manifest=h.load_manifest()
        assert len(manifest['files'])==36 and len(manifest['run_evidence'])==990
        for row in manifest['files']:
            data=archive.read(row['member'])
            assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
            verify_publication_file(release_root/row['packaged_path'],row['sha256'],row['bytes'])
        members=[]
        for row in manifest['run_evidence']:
            for kind in ('history','summary'):
                member=row[kind+'_member'];members.append(member)
                assert hashlib.sha256(archive.read(member)).hexdigest()==row[kind+'_sha256']
        assert len(members)==len(set(members))==1980
    finally:source.close()


def test_all_1200_bh_records_equal_original_helper(reference_root):
    from spade_dfl_release.reference import csv_rows
    records=csv_rows((reference_root/'aggregates/s15_wilcoxon_bh.csv').read_bytes())
    p=[float(r['raw_p_value']) for r in records]
    assert len(p)==1200
    assert np.array_equal(h.benjamini_hochberg(p),h._historical_bh(p))


def test_verify(reference_root):
    result=h.verify(reference_root)
    assert result['distribution_runs']==150
    assert result['method_runs']==840
    assert result['comparisons']==1200 and result['differences']==0
    assert result['client_proportions']==150000


@pytest.mark.parametrize('mutation',['curve_missing','client_duplicate','paired_sample_size','dp_axis','method_n','distribution_n'])
def test_structural_mutations_fail_after_checksum_stage(reference_root,monkeypatch,mutation):
    original=h.csv_rows
    def read(data):
        rows=original(data)
        if 'suite' in rows[0]:
            if mutation=='curve_missing':rows.pop()
            elif mutation=='dp_axis':next(r for r in rows if r['method']=='zo2_dp_sgd')['axis_semantics']='communication_round'
            elif mutation=='method_n':next(r for r in rows if r['suite']=='method' and r['method']!='zo2_dp_sgd')['n_clients']='1000'
            elif mutation=='distribution_n':next(r for r in rows if r['suite']=='distribution')['n_clients']='31'
        elif 'p_y7' in rows[0] and mutation=='client_duplicate':rows[1]=dict(rows[0])
        elif 'n_pairs' in rows[0] and mutation=='paired_sample_size':rows[0]['n_pairs']='1200'
        return rows
    monkeypatch.setattr(h,'csv_rows',read)
    with pytest.raises(ValueError):h.verify(reference_root)


def test_two_ulp_archive_mutation_is_rejected(reference_root,monkeypatch):
    manifest=h.load_manifest();record=manifest['statistical_evidence']['adjacent_float64'][0]
    value=float.fromhex(record['current_p_hex'])
    record['historical_p_hex']=float(np.nextafter(np.nextafter(value,np.inf),np.inf)).hex()
    monkeypatch.setattr(h,'load_manifest',lambda:manifest)
    with pytest.raises(ValueError):h.verify(reference_root)


def test_archived_paired_vector_fingerprint_is_checked(reference_root,monkeypatch):
    manifest=h.load_manifest()
    manifest['statistical_evidence']['paired_differences_sha256']='0'*64
    monkeypatch.setattr(h,'load_manifest',lambda:manifest)
    with pytest.raises(ValueError,match='archived paired'):h.verify(reference_root)


def test_manifest_contains_fingerprints_not_complete_result_arrays(release_root):
    manifest=h.load_manifest()
    assert 'historical_test_evidence' not in manifest
    evidence=manifest['statistical_evidence']
    assert len(evidence['adjacent_float64'])==2
    assert len(json.dumps(evidence))<4000
    assert (release_root/'configs/heterogeneity/manifest.json').stat().st_size<1000000


@pytest.mark.parametrize('field',['identities_sha256','statistic_sha256','historical_p_sha256','historical_q_sha256','classification_sha256'])
def test_statistical_vector_hash_mutations_fail(reference_root,monkeypatch,field):
    manifest=h.load_manifest();manifest['statistical_evidence'][field]='0'*64
    monkeypatch.setattr(h,'load_manifest',lambda:manifest)
    with pytest.raises(ValueError):h.verify(reference_root)


def test_run_no_overrides_rejects_existing(tmp_path):
    assert tuple(inspect.signature(h.run).parameters)==('output',)
    with pytest.raises(ValueError,match='new directory'):h.run(tmp_path)


def _fake_frozen_execution(tmp_path,monkeypatch,failure=None):
    from types import SimpleNamespace
    manifest=h.load_manifest();vendor=tmp_path/'vendor';vendor.mkdir();events=[]
    def event(name):
        events.append(name)
        if failure==name:raise RuntimeError('injected '+name)
    def train(config,output):
        suite='distribution' if config.name=='mnist67_spade_distribution_50round_30seed.json' else 'method'
        assert config.name=={'distribution':'mnist67_spade_distribution_50round_30seed.json','method':'spade_mnist_50round_server.json'}[suite]
        event(suite);output.mkdir()
        count={'distribution':150,'method':840}[suite]-(failure==suite+'_count')
        (output/'suite_manifest.json').write_text(json.dumps({'runs':[None]*count}),encoding='utf-8')
    def audit(distribution,method):
        event('audit');assert distribution.is_dir() and method.is_dir()
    def figures(distribution,method,output):
        event('figures');output.mkdir();(output/'figure.json').write_text('{}',encoding='utf-8')
    def curves(method):event('curves');return {},list(range(30))
    def records(curves,seeds):
        event('records')
        return [SimpleNamespace(raw_p_value=1.,adjusted_q_value=0. if failure=='bh' else 1.) for _ in range(1199 if failure=='comparisons_count' else 1200)]
    loaded={'paper_suite':SimpleNamespace(run_paper_suite=train),'mnist67_selected_three_figures':SimpleNamespace(audit_formal_results=audit,build_selected_three_figures=figures),'mnist67_method_volcano':SimpleNamespace(load_server_accuracy=curves,build_method_volcano_records=records)}
    monkeypatch.setattr(h,'load_manifest',lambda:manifest);monkeypatch.setattr(h,'_vendor_root',lambda:vendor);monkeypatch.setattr(h,'_require_inputs',lambda m:None)
    def child(action,payload,*,root):
        assert action=='run'
        return h._execute_frozen_suites(root,Path(payload['output']),loaded,manifest)
    monkeypatch.setattr(h,'_vendor_call',child)
    return events


def test_public_run_executes_exact_two_suites_audits_and_publishes_manifest(tmp_path,monkeypatch):
    events=_fake_frozen_execution(tmp_path,monkeypatch);output=tmp_path/'output'
    result=h.run(output)
    assert events==['distribution','method','audit','figures','curves','records']
    assert (result['distribution_runs'],result['method_runs'],result['comparisons'])==(150,840,1200)
    assert result['status']=='completed'
    assert json.loads((output/'run_manifest.json').read_text(encoding='utf-8'))==result
    assert set(result['files'])=={'distribution_suite/suite_manifest.json','method_suite/suite_manifest.json','paper_figures/figure.json'}
    for relative,digest in result['files'].items():assert hashlib.sha256((output/relative).read_bytes()).hexdigest()==digest


@pytest.mark.parametrize('failure',['distribution','method','audit','figures','curves','records','bh','distribution_count','method_count','comparisons_count'])
def test_public_run_stage_failure_never_publishes_completed(tmp_path,monkeypatch,failure):
    events=_fake_frozen_execution(tmp_path,monkeypatch,failure);output=tmp_path/'output'
    with pytest.raises((ValueError,RuntimeError)):h.run(output)
    assert not (output/'run_manifest.json').exists()
    assert events.count('distribution')==1 and events.count('method')<=1
    if failure in ('distribution','method'):assert 'audit' not in events and 'figures' not in events


def test_input_resolution_ignores_callers_cwd(tmp_path,monkeypatch):
    root=tmp_path/'vendor';(root/'data').mkdir(parents=True);(root/'configs').mkdir()
    raw=b'external substitute';(root/'data/mnist.npz').write_bytes(raw)
    (root/'configs/base.json').write_text(json.dumps({'data_path':'data/mnist.npz'}),encoding='utf-8')
    for name in ('distribution','method'):
        (root/f'configs/{name}.json').write_text(json.dumps({'base_config':'base.json'}),encoding='utf-8')
    digest=hashlib.sha256(raw).hexdigest()
    manifest={'required_inputs':[{'path':'data/mnist.npz','bytes':len(raw),'sha256':digest}],
              'suites':{name:{'config':f'configs/{name}.json'} for name in ('distribution','method')}}
    caller=tmp_path/'unrelated';caller.mkdir();monkeypatch.chdir(caller)
    result=h._resolved_inputs(root,manifest)
    assert result=={'distribution':{'path':'data/mnist.npz','sha256':digest},'method':{'path':'data/mnist.npz','sha256':digest}}


@pytest.mark.parametrize('mutation',['missing','hash','size'])
def test_bad_input_fails_before_subprocess_or_downloader(tmp_path,monkeypatch,mutation):
    import copy
    manifest=copy.deepcopy(h.load_manifest());payload=b'external substitute'
    vendor=tmp_path/'vendor';target=vendor/manifest['required_inputs'][0]['path'];target.parent.mkdir(parents=True)
    if mutation!='missing':
        target.write_bytes(payload)
    manifest['required_inputs'][0]['bytes']=len(payload)+(mutation=='size')
    manifest['required_inputs'][0]['sha256']=hashlib.sha256(
        b'different substitute' if mutation=='hash' else payload
    ).hexdigest()
    calls=[];monkeypatch.setattr(h,'load_manifest',lambda:manifest);monkeypatch.setattr(h,'_vendor_root',lambda:vendor)
    monkeypatch.setattr(h,'_vendor_call',lambda *a,**k:calls.append((a,k)) or {})
    output=tmp_path/'output'
    with pytest.raises((ValueError,FileNotFoundError),match={'missing':'missing','hash':'hash mismatch','size':'size mismatch'}[mutation]):h.run(output)
    assert calls==[] and not output.exists()


@pytest.mark.parametrize('suite',['distribution','method'])
@pytest.mark.skipif(os.environ.get('SPADE_DFL_RUN_GOLDEN')!='1',reason='opt-in frozen S15 training replay')
def test_golden(suite,tmp_path):
    result=_golden(suite,tmp_path)
    assert result['equal'] is True


def _golden(suite,tmp_path):
    from spade_dfl_release.reference import Sources,resolve_source_index
    root=h.release_root();manifest=h.load_manifest()
    source_index=resolve_source_index()
    if source_index is None:pytest.skip('author source index is not configured; see docs/OPTIONAL_SOURCES.md')
    sources=Sources(source_index,root/'reference_results/raw_manifests/source_manifest.json')
    archive=sources.archive('server_archive');original=tmp_path/'original'
    for row in manifest['files']:
        data=archive.read(row['member']);assert hashlib.sha256(data).hexdigest()==row['sha256']
        path=original/row['member'][len(manifest['archive_prefix']):];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    first=h._vendor_call('golden',dict(suite=suite,output=str(tmp_path/'first')),root=original)
    second=h._vendor_call('golden',dict(suite=suite,output=str(tmp_path/'second')))
    assert first==second
    for name in ('result.npz','partition.npz','preprocessing.npz'):
        with np.load(tmp_path/'first'/name) as a,np.load(tmp_path/'second'/name) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:assert np.array_equal(a[key],b[key]),(suite,name,key)
    for name in ('config.json','metadata.json'):
        assert json.loads((tmp_path/'first'/name).read_text(encoding='utf-8'))==json.loads((tmp_path/'second'/name).read_text(encoding='utf-8'))
    method='spade_dfl_private' if suite=='distribution' else 'spade_dfl'
    evidence=next(r for r in manifest['run_evidence'] if (r['suite'],r['partition'],r['method'],r['seed'])==(suite,'iid',method,0))
    saved=json.loads(archive.read(evidence['history_member'].rsplit('/',1)[0]+'/config.json'))
    assert saved==json.loads((tmp_path/'first/config.json').read_text(encoding='utf-8'))
    return dict(equal=True)
