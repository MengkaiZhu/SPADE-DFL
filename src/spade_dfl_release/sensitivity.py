"""Frozen, validation-only Tables III–IV; no scientific overrides."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .paths import release_root
from .publication import verify_publication_file
from .reference import csv_rows, require_keys, verify_hash, format_display

CLIENT_COUNTS = (30, 40, 50, 75, 100)
EPSILONS = (4, 8, 16, 24, 32)
DATASETS = ('mnist', 'fashion', 'alzheimer', 'brain')
# Exact artifact set from the frozen accuracy-priority study.worker.
ARTIFACTS = ('raw.npz','config.json','summary.json','metadata.json','STARTED.json','TRANSFORM.json','partition.npz')


def sensitivity_keys():
    return tuple(dict.fromkeys([(n, 32) for n in CLIENT_COUNTS]+[(100, e) for e in EPSILONS]))


def _integer(value):
    number=float(value)
    if not np.isfinite(number) or number != int(number):
        raise ValueError('noninteger identity/count')
    return int(number)


def _semantics(row):
    for field, expected in [('validation_only',True),('formal_authorized',False),('previous_test_exposure',True)]:
        if field in row and str(row[field]).lower()!=str(expected).lower():
            raise ValueError('validation-only boundary violated: '+field)
    if row.get('phase','validation')!='validation' or row.get('evaluation','training_holdout')!='training_holdout':
        raise ValueError('formal test contamination')
    if row.get('origin') not in ('previous_validation','new_validation'):
        raise ValueError('missing validation source identity')
    if row.get('status')!='completed' or _integer(row.get('ddof',1))!=1:
        raise ValueError('invalid status or sample-SD convention')


def aggregate_sensitivity(rows):
    groups=defaultdict(list); keys=[]
    for raw in rows:
        row=dict(raw); _semantics(row)
        for field in ('n','epsilon','seed','fold','correct','eval_count'):row[field]=_integer(row[field])
        if not 0<=row['correct']<=row['eval_count'] or row['eval_count']<=0:
            raise ValueError('invalid correct/eval_count')
        accuracy=float(row['accuracy'])
        if not np.isfinite(accuracy) or accuracy!=row['correct']/row['eval_count']:
            raise ValueError('accuracy differs from exact counts')
        row['accuracy']=accuracy
        key=tuple(row[k] for k in ('dataset','n','epsilon','seed','fold'))
        keys.append(key);groups[key[:3]].append(row)
    expected={(d,n,e,s,f) for d in DATASETS for n,e in sensitivity_keys() for s in range(5) for f in range(5 if d=='alzheimer' else 1)}
    require_keys(keys,expected)
    result=[]
    for (dataset,n,epsilon),cell in sorted(groups.items()):
        values=[]
        for seed in range(5):
            rs=[r for r in cell if r['seed']==seed]
            values.append(100*sum(r['correct'] for r in rs)/sum(r['eval_count'] for r in rs) if dataset=='alzheimer' else 100*rs[0]['accuracy'])
        origins={r['origin'] for r in cell}
        if len(origins)!=1:raise ValueError('nonidentical reused/shared cell source')
        result.append(dict(dataset=dataset,n=n,epsilon=epsilon,mean_percent=float(np.mean(values)),sd_percent=float(np.std(values,ddof=1)),
                           ddof=1,n_seed=5,n_fold_seed=len(cell),origin=next(iter(origins)),
                           validation_only=True,formal_authorized=False,previous_test_exposure=True))
    return result


def _read_config(name):
    return json.loads((release_root()/'configs/sensitivity'/name).read_text(encoding='utf-8'))


def load_manifest():
    manifest=_read_config('manifest.json');root=release_root().resolve()
    paths=[row['packaged_path'] for row in manifest['files']]
    require_keys(paths,set(paths))
    for row in manifest['files']:
        path=(root/row['packaged_path']).resolve()
        if root not in path.parents:raise ValueError('invalid vendor path/size')
        verify_publication_file(path,row['sha256'],row['bytes'])
    return manifest


def load_plan():
    plan=_read_config('plan.json');manifest=load_manifest()
    payload=(release_root()/'configs/sensitivity/plan.json').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=manifest['plan_sha256']:raise ValueError('composite plan hash mismatch')
    if plan['validation_only'] is not True or plan['formal_authorized'] is not False or plan['previous_test_exposure'] is not True:
        raise ValueError('plan validation boundary changed')
    keys=[]
    for entry in plan['jobs']:
        j=entry['job']; group=plan['groups'][j['group']];cfg=j['cfg']
        if cfg['seeds']!=[j['seed']] or j['reference'] is not None:raise ValueError('unfrozen seed/reference')
        if cfg['stage']!='spade_dfl_private':raise ValueError('only frozen SPADE-DFL stage')
        source=plan['sources'][entry['source_index']]
        if not 0<=entry['source_job_index']<source['job_count']:raise ValueError('invalid source job identity')
        keys.append((group['dataset'],cfg['n_clients'],cfg['target_epsilon'],j['seed'],group['fold']))
    expected={(d,n,e,s,f) for d in DATASETS for n,e in sensitivity_keys() for s in range(5) for f in range(5 if d=='alzheimer' else 1)}
    require_keys(keys,expected)
    require_keys([(r['source_index'],r['source_job_index']) for r in plan['jobs']], {(i,j) for i,src in enumerate(plan['sources']) for j in range(src['job_count'])})
    return plan


def verify(reference_root: Path):
    plan=load_plan();manifest=load_manifest();root=Path(reference_root)
    checked=json.loads((root/'paper_values/verification.json').read_text(encoding='utf-8'))
    if checked['status']!='verified' or any(checked[k] for k in ('difference_count','missing_cells','duplicate_cells')):
        raise ValueError('Manuscript-value verification failed')
    for source in manifest['source_evidence']:
        key=source['source_id']+'/'+source['relative_path']
        if checked['source_hashes'].get(key,'').lower()!=source['sha256'].lower():
            raise ValueError('frozen source evidence hash mismatch')
    path=root/'aggregates/sensitivity_cells.csv'
    verify_hash(path,checked['normalized_sha256'][path.name])
    evidence=manifest['run_evidence'];cells=aggregate_sensitivity(evidence)
    for dataset in DATASETS:
        anchor=[r for r in evidence if r['dataset']==dataset and _integer(r['n'])==100 and _integer(r['epsilon'])==32]
        if anchor!=manifest['anchor_identity'][dataset] or any(r['origin']!='previous_validation' for r in anchor):
            raise ValueError('shared anchor fingerprint differs')
    saved=csv_rows(path.read_bytes())
    key=lambda r:(r['dataset'],_integer(r['n']),_integer(r['epsilon']))
    require_keys([key(r) for r in saved],{key(r) for r in cells})
    jobs={(plan['groups'][x['job']['group']]['dataset'],x['job']['cfg']['n_clients'],x['job']['cfg']['target_epsilon'],x['job']['seed'],plan['groups'][x['job']['group']]['fold']):x for x in plan['jobs']}
    for row in evidence:
        j=jobs[(row['dataset'],int(row['n']),int(float(row['epsilon'])),int(row['seed']),int(row['fold']))]['job']
        if int(row['rounds'])!=j['cfg']['n_rounds'] or int(row['queries'])!=j['budget']['actual']:
            raise ValueError('raw evidence differs from frozen scientific job')
    index={key(r):r for r in saved}
    for cell in cells:
        other=index[key(cell)];_semantics(dict(other,status='completed'))
        if other['origin']!=cell['origin']:raise ValueError('normalized validation source origin mismatch')
        for field in ('mean_percent','sd_percent'):
            if format_display(cell[field],2)!=format_display(other[field],2):raise ValueError('sensitivity display precision mismatch')
            rs=[r for r in evidence if (r['dataset'],int(r['n']),int(float(r['epsilon'])))==key(cell)]
            values=[sum(int(r['correct']) for r in rs if int(r['seed'])==seed)/sum(int(r['eval_count']) for r in rs if int(r['seed'])==seed) for seed in range(5)]
            # The reference aggregation converts pooled fractions to percentages after division.
            # Check that exact floating-point operation order too; no tolerance.
            normalized=dict(mean_percent=float(np.mean(np.asarray(values)*100)),sd_percent=float(np.std(np.asarray(values)*100,ddof=1)))
            if normalized[field]!=float(other[field]):
                raise ValueError('independent sensitivity recomputation differs')
        for field in ('n_seed','n_fold_seed','ddof'):
            if int(other[field])!=cell[field]:raise ValueError('normalized count/ddof mismatch')
        expected_rounds={int(r['rounds']) for r in evidence if (r['dataset'],int(r['n']),int(float(r['epsilon'])))==key(cell)}
        if expected_rounds!={int(other['rounds'])}:raise ValueError('normalized round count differs from frozen jobs')
    return dict(status='verified',raw_runs=len(evidence),cells=len(cells),ddof=1,differences=0,
                validation_only=True,formal_authorized=False,previous_test_exposure=True,
                provenance='Hash-frozen RUNS evidence, exact composite job identities, and manuscript display checks')


def run(output: Path):
    output=Path(output).resolve();references=(release_root()/'reference_results').resolve()
    if output.exists() or output==references or references in output.parents:
        raise ValueError('sensitivity rerun requires a new directory outside reference_results')
    plan=load_plan();manifest=load_manifest()
    _require_inputs(manifest)
    output.mkdir(parents=True,exist_ok=False)
    for entry in plan['jobs']:_execute(entry,output,manifest)
    rows=[];audits=[]
    for entry in plan['jobs']:
        out=_execute(entry,output,manifest,audit=True)
        first=json.loads((out/'RELEASE_VERIFIED.json').read_text(encoding='utf-8'))
        checked=json.loads((out/'RELEASE_RECHECKED.json').read_text(encoding='utf-8'))
        _compare_audits(first,checked,out,entry)
        row=dict(checked['row'])
        row.update(origin={'sensitivity_runtime':'previous_validation','sensitivity_tables34':'new_validation'}[plan['sources'][entry['source_index']]['source_id']],
                   validation_only=True,formal_authorized=False,previous_test_exposure=True)
        rows.append(row)
        path=out/'RELEASE_RECHECKED.json'
        audits.append(dict(path=path.relative_to(output).as_posix(),sha256=_hash(path)))
    rows.sort(key=lambda r:(r['dataset'],_integer(r['n']),_integer(r['epsilon']),_integer(r['seed']),_integer(r['fold'])))
    cells=aggregate_sensitivity(rows)
    files={}
    for name,records in [('RUNS.csv',rows),('CELLS.csv',cells)]:
        path=output/name
        _write_csv(path,records)
        files[name]=dict(sha256=_hash(path),rows=len(records))
    result=dict(status='completed',runs=len(rows),rechecked_runs=len(audits),cells=len(cells),
                composite_plan_sha256=manifest['plan_sha256'],files=files,audit_files=audits,
                manifest_path='run_manifest.json',validation_only=True,formal_authorized=False,previous_test_exposure=True)
    with (output/'run_manifest.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
    return result


def _hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _artifact_hashes(out):
    return {name:_hash(Path(out)/name) for name in ARTIFACTS}


def _compare_audits(first,checked,out,entry):
    if first!=checked or checked.get('verified') is not True:
        raise ValueError('independent recheck differs from first verification')
    if any(checked.get(k)!=entry[k] for k in ('source_index','source_job_index')):
        raise ValueError('independent recheck job identity mismatch')
    if checked.get('artifact_hashes')!=_artifact_hashes(out):
        raise ValueError('scientific artifact hashes changed after verification')


def _write_csv(path,rows):
    fields=sorted({key for row in rows for key in row})
    with Path(path).open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False) if isinstance(value,(dict,list)) else value for key,value in row.items()})


def _require_inputs(manifest, input_root=None):
    base=Path(input_root) if input_root is not None else release_root()/'src/spade_dfl_release/vendor/sensitivity'
    for row in manifest['required_inputs']:
        path=base/row['path']
        if not path.is_file():raise FileNotFoundError('missing frozen sensitivity input: '+row['path'])
        verify_hash(path,row['sha256'])
    return base


def _execute(entry, output, manifest, *, vendor_root=None, input_root=None, audit=False):
    """Private worker: complete public run owns the immutable job selection."""
    vendor_root=Path(vendor_root) if vendor_root is not None else release_root()/'src/spade_dfl_release/vendor/sensitivity'
    input_root=_require_inputs(manifest,input_root)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    request=output/f"request_{entry['source_index']}_{entry['source_job_index']}{'_recheck' if audit else ''}.json"
    payload=dict(entry=entry,vendor_root=str(vendor_root.resolve()),input_root=str(input_root.resolve()),output=str(output),audit=audit)
    with request.open('x',encoding='utf-8') as stream:json.dump(payload,stream,ensure_ascii=False)
    environment=os.environ.copy();environment.pop('PYTHONUTF8',None)
    environment.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(release_root()/'src'))
    environment.update({k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    subprocess.run([sys.executable,'-B','-m','spade_dfl_release.sensitivity',str(request)],env=environment,check=True)
    return output/entry['job']['out']


def _child(request):
    import importlib.util
    config=json.loads(Path(request).read_text(encoding='utf-8'));root=Path(config['vendor_root'])
    entry=config['entry'];job=dict(entry['job']);output=Path(config['output']);audit=config.get('audit',False)
    phase_root=output/f"phase_{entry['source_index']}";phase_root.mkdir(exist_ok=True)
    plan_path=phase_root/'PLAN.json'
    if not plan_path.exists():
        if audit:raise ValueError('read-only recheck requires the existing phase plan')
        plan=load_plan()
        phase=dict(source=plan['sources'][entry['source_index']],jobs=[r['job'] for r in plan['jobs'] if r['source_index']==entry['source_index']],validation_only=True,formal_authorized=False,previous_test_exposure=True)
        with plan_path.open('x',encoding='utf-8') as stream:json.dump(phase,stream,sort_keys=True)
    job['out']=str(output/job['out'])
    runtime=root/'table34_accuracy_priority_working'
    sys.path.insert(0,str(runtime))
    spec=importlib.util.spec_from_file_location('frozen_sensitivity_entry',runtime/'study.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    e=module.e;e.s.PRE=Path(config['input_root'])/'table34_preprocessing_working'
    e.s.release.ROOT=phase_root
    e.check_schema_diff()
    bundle,base,state=e.prepare(job['group'],job['retained'],False)
    out=Path(job['out'])
    if not audit:
        e.s.release.run_one(job,bundle,base,'validation')
        e.save(out/'TRANSFORM.json',dict(state=state,train_sha256=e.array_hash(bundle.train_x),eval_sha256=e.array_hash(bundle.test_x),actual_model_dimension=job['cfg']['dimension'],labels_unchanged=True))
    row=e.verify(job,bundle,base,state,'validation',{})
    # Result files retain the historical scientific schema. This new audit has
    # portable source identity rather than the historical absolute result path.
    for key in ('source_summary','source_config','source_raw'):
        row[key]=Path(row[key]).relative_to(output).as_posix()
    snapshot=dict(verified=True,row=row,source_index=entry['source_index'],source_job_index=entry['source_job_index'],artifact_hashes=_artifact_hashes(out))
    if audit:
        first=json.loads((out/'RELEASE_VERIFIED.json').read_text(encoding='utf-8'))
        _compare_audits(first,snapshot,out,entry)
    e.save(out/('RELEASE_RECHECKED.json' if audit else 'RELEASE_VERIFIED.json'),snapshot)


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('private worker requires one generated request file')
    _child(sys.argv[1])
