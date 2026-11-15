import copy
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest

from spade_dfl_release import sensitivity as s


@pytest.fixture
def rows():
    return [dict(dataset=d, n=n, epsilon=e, seed=seed, fold=fold,
                 status='completed', origin='previous_validation',
                 correct=10+seed+fold, eval_count=20+2*fold,
                 accuracy=(10+seed+fold)/(20+2*fold))
            for d in ('mnist','fashion','alzheimer','brain')
            for n,e in s.sensitivity_keys() for seed in range(5)
            for fold in range(5 if d=='alzheimer' else 1)]


def test_grid_single_shared_anchor():
    keys=s.sensitivity_keys()
    assert len(keys)==len(set(keys))==9
    assert keys.count((100,32))==1
    assert set(keys)=={(n,32) for n in (30,40,50,75,100)}|{(100,e) for e in (4,8,16,24,32)}


def test_recompute_pools_within_seed_then_sample_sd(rows):
    cells=s.aggregate_sensitivity(rows)
    assert len(cells)==36
    for cell in cells:
        selected=[r for r in rows if (r['dataset'],r['n'],r['epsilon'])==(cell['dataset'],cell['n'],cell['epsilon'])]
        values=[100*sum(r['correct'] for r in selected if r['seed']==seed)/sum(r['eval_count'] for r in selected if r['seed']==seed) if cell['dataset']=='alzheimer' else 100*next(r['accuracy'] for r in selected if r['seed']==seed) for seed in range(5)]
        assert cell['mean_percent']==float(np.mean(values))
        assert cell['sd_percent']==float(np.std(values,ddof=1))
        assert cell['sd_percent']!=float(np.std(values,ddof=0))
        assert cell['ddof']==1
        assert cell['validation_only'] is True and cell['formal_authorized'] is False


@pytest.mark.parametrize('mutation',['missing','duplicate','wrong_seed','wrong_fold','formal','formal_flag','ddof','anchor_duplicate','anchor_nonidentical'])
def test_reject_corrupted_identity_or_semantics(rows,mutation):
    bad=copy.deepcopy(rows)
    if mutation=='missing':bad.pop()
    elif mutation=='duplicate':bad.append(copy.deepcopy(bad[0]))
    elif mutation=='wrong_seed':bad[0]['seed']=5
    elif mutation=='wrong_fold':bad[0]['fold']=1
    elif mutation=='formal':bad[0]['phase']='formal'
    elif mutation=='formal_flag':bad[0]['formal_authorized']=True
    elif mutation=='ddof':bad[0]['ddof']=0
    else:
        row=copy.deepcopy(next(r for r in bad if (r['n'],r['epsilon'])==(100,32)))
        if mutation=='anchor_nonidentical':row['correct']+=1
        bad.append(row)
    with pytest.raises(ValueError):s.aggregate_sensitivity(bad)


def test_composite_plan_and_byte_provenance(release_root):
    from spade_dfl_release.publication import publication_records, verify_publication_file
    plan=s.load_plan()
    assert [p['job_count'] for p in plan['sources']]==[200,160]
    assert len(plan['jobs'])==360
    manifest=s.load_manifest()
    for row in manifest['files']:
        payload=(release_root/row['packaged_path']).read_bytes()
        adaptation=publication_records().get(row['packaged_path'])
        expected=adaptation['published_sha256'] if adaptation else row['sha256']
        size=adaptation['published_bytes'] if adaptation else row['bytes']
        assert len(payload)==size
        assert hashlib.sha256(payload).hexdigest()==expected
        assert verify_publication_file(release_root/row['packaged_path'],row['sha256'],row['bytes'])==expected
        assert not Path(row['relative_path']).is_absolute()
        assert ':/' not in row['relative_path'] and ':\\' not in row['relative_path']
    encoded=json.dumps(plan)+json.dumps(manifest)
    assert 'D:/' not in encoded and 'D:\\\\' not in encoded


def test_all_original_dependencies_have_direct_source_manifest_entries(release_root):
    manifest=s.load_manifest()
    frozen=json.loads((release_root/'reference_results/raw_manifests/source_manifest.json').read_text(encoding='utf-8'))
    indexed={(r['source_id'],r['path']):r for r in frozen['files']}
    for row in manifest['files']+manifest['required_inputs']:
        key=(row['source_id'],row['relative_path'])
        assert key in indexed,('missing direct source entry',key)
        assert indexed[key]['sha256'].lower()==row.get('original_sha256',row['sha256']).lower()
    required={r['relative_path'] for r in manifest['files']+manifest['required_inputs'] if r['source_id']=='sensitivity_lineage'}
    assert {r['path'] for r in frozen['files'] if r['source_id']=='sensitivity_lineage'}==required
    from spade_dfl_release.reference import resolve_source_index
    spec_path=resolve_source_index()
    if spec_path is not None:
        specs=json.loads(spec_path.read_text(encoding='utf-8'))['sources']
        lineage=next(r for r in specs if r['id']=='sensitivity_lineage')
        assert lineage['root'] and not Path(lineage['root']).is_absolute() and lineage['required'] is True
        assert len(required)==151 and set(lineage['include'])==required and len(lineage['include'])==151


def test_verify(reference_root):
    result=s.verify(reference_root)
    assert result['status']=='verified'
    assert result['raw_runs']==360 and result['cells']==36
    assert result['ddof']==1 and result['differences']==0


@pytest.mark.parametrize('field',['summary_sha256','correct','eval_count','accuracy','origin','seed','fold'])
def test_anchor_fingerprint_mutation_fails(reference_root,monkeypatch,field):
    manifest=s.load_manifest()
    manifest['anchor_identity']['mnist'][0][field]='tampered'
    monkeypatch.setattr(s,'load_manifest',lambda:manifest)
    with pytest.raises(ValueError,match='anchor fingerprint'):s.verify(reference_root)


def test_evidence_source_hash_mutation_fails(reference_root,monkeypatch):
    manifest=s.load_manifest();manifest['source_evidence'][0]['sha256']='0'*64
    monkeypatch.setattr(s,'load_manifest',lambda:manifest)
    with pytest.raises(ValueError,match='source evidence hash'):s.verify(reference_root)


def test_public_run_preflight_missing_data_does_not_create_output(tmp_path):
    output=tmp_path/'new'
    with pytest.raises(FileNotFoundError,match='missing frozen sensitivity input'):s.run(output)
    assert not output.exists()


def _fake_training_boundary(monkeypatch,rows,mutation=None):
    plan=s.load_plan();trained=[];rechecked=[]
    index={(r['dataset'],r['n'],r['epsilon'],r['seed'],r['fold']):r for r in rows}
    names=('raw.npz','config.json','summary.json','metadata.json','STARTED.json','TRANSFORM.json','partition.npz')
    monkeypatch.setattr(s,'_require_inputs',lambda manifest:None)
    def execute(entry,output,manifest,*,audit=False):
        out=output/entry['job']['out'];job=entry['job'];group=plan['groups'][job['group']]
        if not audit:
            trained.append(copy.deepcopy(entry));out.mkdir(parents=True)
            row=dict(index[(group['dataset'],job['cfg']['n_clients'],job['cfg']['target_epsilon'],job['seed'],group['fold'])],rounds=job['cfg']['n_rounds'],queries=job['budget']['actual'])
            for name in names:(out/name).write_bytes(json.dumps(row,sort_keys=True).encode())
            snap=dict(verified=True,row=row,source_index=entry['source_index'],source_job_index=entry['source_job_index'],artifact_hashes={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in names})
            (out/'RELEASE_VERIFIED.json').write_text(json.dumps(snap),encoding='utf-8')
        else:
            assert len(trained)==360,'recheck began before all training finished'
            rechecked.append(copy.deepcopy(entry))
            snap=json.loads((out/'RELEASE_VERIFIED.json').read_text(encoding='utf-8'))
            if mutation=='row':snap['row']['correct']+=1
            elif mutation=='hash':snap['artifact_hashes']['raw.npz']='0'*64
            elif mutation=='artifact':(out/'raw.npz').write_bytes(b'tampered')
            elif mutation=='failure':raise RuntimeError('independent verification failed')
            (out/'RELEASE_RECHECKED.json').write_text(json.dumps(snap),encoding='utf-8')
        return out
    monkeypatch.setattr(s,'_execute',execute)
    return plan,trained,rechecked


def test_public_run_dispatches_exact_complete_plan(tmp_path,monkeypatch,rows):
    plan,trained,rechecked=_fake_training_boundary(monkeypatch,rows)
    output=tmp_path/'complete';result=s.run(output)
    assert trained==plan['jobs'] and len(trained)==360
    assert rechecked==plan['jobs'] and len(rechecked)==360
    assert result==json.loads((output/'run_manifest.json').read_text(encoding='utf-8'))
    assert result['runs']==360 and result['cells']==36 and result['rechecked_runs']==360
    assert result['status']=='completed' and result['formal_authorized'] is False
    assert result['composite_plan_sha256']==hashlib.sha256((s.release_root()/'configs/sensitivity/plan.json').read_bytes()).hexdigest()
    from spade_dfl_release.reference import csv_rows
    for name,count in [('RUNS.csv',360),('CELLS.csv',36)]:
        data=(output/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==result['files'][name]['sha256']
        assert len(csv_rows(data))==count
        assert str(output).encode() not in data


@pytest.mark.parametrize('mutation',['row','hash','artifact','failure'])
def test_failed_recheck_never_publishes_completed_manifest(tmp_path,monkeypatch,rows,mutation):
    _fake_training_boundary(monkeypatch,rows,mutation)
    output=tmp_path/'failed'
    with pytest.raises((ValueError,RuntimeError)):s.run(output)
    assert not (output/'run_manifest.json').exists()


@pytest.mark.parametrize('field,value',[('ddof','0'),('rounds','50'),('n_seed','30'),('validation_only','False'),('formal_authorized','True'),('previous_test_exposure','False'),('origin','new_validation')])
def test_normalized_mutation_rejected_independently(reference_root,tmp_path,field,value):
    import csv,io
    source=reference_root/'aggregates/sensitivity_cells.csv'
    reader=csv.DictReader(io.StringIO(source.read_text(encoding='utf-8')))
    fields=reader.fieldnames;rows=list(reader);rows[0][field]=value
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    data=stream.getvalue().encode('utf-8')
    (tmp_path/'aggregates').mkdir();(tmp_path/'paper_values').mkdir()
    (tmp_path/'aggregates'/source.name).write_bytes(data)
    verification=json.loads((reference_root/'paper_values/verification.json').read_text(encoding='utf-8'))
    verification['normalized_sha256'][source.name]=hashlib.sha256(data).hexdigest()
    (tmp_path/'paper_values/verification.json').write_text(json.dumps(verification),encoding='utf-8')
    with pytest.raises(ValueError):s.verify(tmp_path)


def test_all_scientific_jobs_match_hash_frozen_sources():
    from spade_dfl_release.reference import Sources,resolve_source_index
    spec_path=resolve_source_index()
    if spec_path is None:pytest.skip('author-only source index is not configured')
    sources=Sources(spec_path,s.release_root()/'reference_results/raw_manifests/source_manifest.json')
    plan=s.load_plan()
    for spec in plan['sources']:
        raw=sources.file(spec['source_id'],spec['relative_path'])
        assert hashlib.sha256(raw).hexdigest()==spec['sha256'].lower()
        original=json.loads(raw)
        for entry in [r for r in plan['jobs'] if r['source_index']==spec['order']]:
            job=original['jobs'][entry['source_job_index']]
            assert {k:v for k,v in job.items() if k!='out'}=={k:v for k,v in entry['job'].items() if k!='out'}
    evidence=s.load_manifest()['run_evidence']
    from spade_dfl_release.reference import csv_rows
    raw_rows=csv_rows(sources.file('sensitivity_tables34','RUNS.csv'))
    for row in evidence:
        source=raw_rows[row['source_row_index']]
        assert all(source[k]==v for k,v in row.items() if k!='source_row_index')


def test_run_rejects_existing_and_has_no_subset_overrides(tmp_path):
    assert tuple(inspect.signature(s.run).parameters)==('output',)
    with pytest.raises(ValueError,match='new directory'):s.run(tmp_path)


@pytest.mark.parametrize('dataset',['mnist','fashion','alzheimer','brain'])
@pytest.mark.skipif(os.environ.get('SPADE_DFL_RUN_GOLDEN')!='1',reason='opt-in exact historical training replay')
def test_golden(dataset,tmp_path):
    result=_golden(dataset,tmp_path)
    assert result['equal'] is True
    assert result['folds']==list(range(5 if dataset=='alzheimer' else 1))


def _golden(dataset,tmp_path):
    from spade_dfl_release.reference import Sources,resolve_source_index,verify_hash
    source_index=resolve_source_index()
    if source_index is None:pytest.skip('author-only source index is not configured')
    sources=Sources(source_index,s.release_root()/'reference_results/raw_manifests/source_manifest.json')
    source_root=sources.paths['sensitivity_runtime'].parent
    manifest=s.load_manifest();plan=s.load_plan()
    for spec in plan['sources']:
        original=json.loads(sources.file(spec['source_id'],spec['relative_path']))
        for entry in [e for e in plan['jobs'] if e['source_index']==spec['order']]:
            job=original['jobs'][entry['source_job_index']]
            assert hashlib.sha256(json.dumps(job,sort_keys=True,separators=(',',':')).encode()).hexdigest()==entry['original_job_sha256']
            assert {k:v for k,v in job.items() if k!='out'}=={k:v for k,v in entry['job'].items() if k!='out'}
    for row in manifest['files']:verify_hash(sources.paths[row['source_id']]/row['relative_path'],row['original_sha256'])
    s._require_inputs(manifest,source_root)
    selected=[e for e in plan['jobs'] if plan['groups'][e['job']['group']]['dataset']==dataset and e['job']['cfg']['n_clients']==100 and e['job']['cfg']['target_epsilon']==32 and e['job']['seed']==0]
    folds=[]
    for entry in selected:
        assert entry['source_index']==0
        first=s._execute(entry,tmp_path/'historical',manifest,vendor_root=source_root,input_root=source_root)
        second=s._execute(entry,tmp_path/'packaged',manifest,input_root=source_root)
        for out,root,vendor in [(first,tmp_path/'historical',source_root),(second,tmp_path/'packaged',None)]:
            before={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('raw.npz','config.json','summary.json','metadata.json','STARTED.json','TRANSFORM.json','partition.npz')}
            s._execute(entry,root,manifest,vendor_root=vendor,input_root=source_root,audit=True)
            initial=json.loads((out/'RELEASE_VERIFIED.json').read_text(encoding='utf-8'))
            checked=json.loads((out/'RELEASE_RECHECKED.json').read_text(encoding='utf-8'))
            assert initial==checked and checked['artifact_hashes']==before
            assert before=={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in before}
        for name in ('raw.npz','partition.npz'):
            with np.load(first/name) as a,np.load(second/name) as b:
                assert set(a.files)==set(b.files)
                for key in a.files:assert np.array_equal(a[key],b[key]),(dataset,name,key)
        for name in ('config.json','metadata.json','TRANSFORM.json'):
            assert json.loads((first/name).read_text(encoding='utf-8'))==json.loads((second/name).read_text(encoding='utf-8'))
        a_start=json.loads((first/'STARTED.json').read_text(encoding='utf-8'));b_start=json.loads((second/'STARTED.json').read_text(encoding='utf-8'))
        assert a_start['inputs']==b_start['inputs']
        a_audit=json.loads((first/'RELEASE_VERIFIED.json').read_text(encoding='utf-8'))['row']
        b_audit=json.loads((second/'RELEASE_VERIFIED.json').read_text(encoding='utf-8'))['row']
        assert {k:v for k,v in a_audit.items() if k not in ('trainer_seconds','summary_sha256')}=={k:v for k,v in b_audit.items() if k not in ('trainer_seconds','summary_sha256')}
        a=json.loads((first/'summary.json').read_text(encoding='utf-8'));b=json.loads((second/'summary.json').read_text(encoding='utf-8'))
        provenance={'identity','trainer_seconds','raw_sha256','metadata_sha256','config_sha256','path'}
        assert {k:v for k,v in a.items() if k not in provenance}=={k:v for k,v in b.items() if k not in provenance}
        for k in ('phase','group','candidate','seed','n_clients','epsilon','evaluation'):assert a['identity'][k]==b['identity'][k]
        folds.append(plan['groups'][entry['job']['group']]['fold'])
    assert len(selected)==(5 if dataset=='alzheimer' else 1)
    return dict(equal=True,folds=folds)
