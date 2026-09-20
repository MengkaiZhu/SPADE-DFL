"""Strict adapter for the frozen, non-private synthetic Figure 10 diagnostic."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

from ..paths import release_root
from ..reference import csv_rows, require_keys, verify_hash

_CLOSURE_SHA256='4f2617ba4a4489c44045ee7f2f6b50a79612e7dba950716e8f77b818ccadf61d'
# Independent complete parsed-manifest anchor, including semantic declarations,
# seeds, config, CI exceptions, plotting source identities and line provenance.
_MANIFEST_SHA256='b09ea24ef0618d45411d0056109d76b8701d0363112a22e96e85797485177046'
_ADJACENT_LOW={(1000,10),(10648,22),(29791,1),(29791,31)}
_INTEGER_FIELDS={'seed','T','tau','rounds','local_steps','query_count','transmitted_scalars','n_seeds'}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest():
    root=release_root();manifest=json.loads((root/'configs/communication/manifest.json').read_text(encoding='utf-8'))
    rows=manifest['files'];closure=sorted((r['source_id'],r['source_path'],r['packaged_path'],r['bytes'],r['sha256']) for r in rows)
    digest=hashlib.sha256(json.dumps(closure,separators=(',',':')).encode('ascii')).hexdigest()
    if len(rows)!=5 or digest!=_CLOSURE_SHA256:raise ValueError('communication source closure mismatch')
    canonical=json.dumps(manifest,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')
    if hashlib.sha256(canonical).hexdigest()!=_MANIFEST_SHA256:raise ValueError('communication manifest semantic/provenance anchor mismatch')
    source_rows=json.loads((root/'reference_results/raw_manifests/source_manifest.json').read_text(encoding='utf-8'))['files']
    source={(r['source_id'],r['path']):r for r in source_rows}
    for row in rows+manifest['plotting_sources']:
        if source[(row['source_id'],row['source_path'])]['sha256'].lower()!=row['sha256']:raise ValueError('Communication source-provenance mismatch')
    for row in rows:
        path=(root/row['packaged_path']).resolve()
        if root.resolve() not in path.parents or path.stat().st_size!=row['bytes']:raise ValueError('communication source path/size mismatch')
        verify_hash(path,row['sha256'])
    return manifest


def _runtime():
    load_manifest()
    modules=tuple(importlib.import_module(__package__+'._frozen.'+name) for name in ('spade_simulation','run_experiment'))
    root=Path(__file__).resolve().parent/'_frozen'
    if any(root not in Path(module.__file__).resolve().parents for module in modules):raise ValueError('communication scientific import escaped runtime')
    return modules


def budget_schedule():
    runtime,_=_runtime()
    return [dict(T=r['T'],tau=r['tau_local'],rounds=r['K_local']) for r in runtime.build_budget_schedule()]


def transmitted_scalars(n_clients,dimension,rounds):
    values=(n_clients,dimension,rounds)
    if any(isinstance(x,bool) or not isinstance(x,int) or x<=0 for x in values) or n_clients<3:
        raise ValueError('cycle scalar arguments must be positive integers with at least three clients')
    return 2*n_clients*dimension*rounds


def _read_csv(path):
    return [{k:(int(v) if k in _INTEGER_FIELDS else v if k in ('schedule','diagnostic','paired_by') else float(v)) for k,v in row.items()} for row in csv_rows(Path(path).read_bytes())]


def _fixtures():
    load_manifest();root=release_root()/'reference_results/source_data/communication'
    return (_read_csv(root/'communication_cost_per_seed.csv'),_read_csv(root/'communication_cost_summary.csv'),json.loads((root/'experiment_metadata.json').read_text(encoding='utf-8')))


def _condition(row):return (row['T'],row['tau'])


def _independent_summary(rows):
    output=[]
    for total,tau in sorted({_condition(r) for r in rows}):
        group=sorted([r for r in rows if _condition(r)==(total,tau)],key=lambda r:r['seed']);first=group[0]
        base=next(r for r in rows if r['T']==total and r['tau']==1 and r['seed']==first['seed'])
        row={k:first[k] for k in ('schedule','T','tau','rounds','query_count','transmitted_scalars')}
        row.update(n_seeds=len(group),communication_reduction_vs_tau1=base['transmitted_scalars']/first['transmitted_scalars'])
        for metric in ('criterion','stationarity','consensus'):
            values=np.asarray([r[metric] for r in group]);mean=float(values.mean())
            half=float(stats.t.ppf(.975,len(values)-1))*float(values.std(ddof=1))/np.sqrt(len(values))
            row.update({metric+'_mean':mean,metric+'_ci95_low':mean-half,metric+'_ci95_high':mean+half})
        output.append(row)
    return output


def _cross_chain(historical,current):
    adjacent=[]
    for old,new in zip(historical,current):
        if _condition(old)!=_condition(new):raise ValueError('summary identities changed')
        for field,value in old.items():
            other=new[field]
            if value==other:continue
            if field!='stationarity_ci95_low' or _condition(old) not in _ADJACENT_LOW or not np.isfinite(value) or np.nextafter(np.float64(value),np.float64(other))!=other:
                raise ValueError('summary cross-chain difference exceeds exact/one-ULP contract')
            adjacent.append(dict(T=old['T'],tau=old['tau'],field=field,historical_hex=float(value).hex(),current_hex=float(other).hex()))
    return adjacent


def _verify_records(rows,summary,metadata):
    runtime,runner=_runtime();manifest=load_manifest();config=runtime.SimulationConfig().__dict__
    if metadata['config']!=config or manifest['config']!=config:raise ValueError('frozen communication config changed')
    if metadata['budgets']!=runtime.build_budget_schedule() or manifest['budgets']!=metadata['budgets']:raise ValueError('frozen communication budgets changed')
    if metadata['n_seeds']!=30 or any(metadata[k]!=0 for k in ('query_noise','release_noise','clipping_error')) or metadata['hypothesis_tests']!='None':raise ValueError('non-private/no-hypothesis-test contract changed')
    _,_,frozen_metadata=_fixtures()
    if {k:v for k,v in metadata.items() if k!='elapsed_seconds'}!={k:v for k,v in frozen_metadata.items() if k!='elapsed_seconds'}:raise ValueError('communication metadata semantics changed')
    if not np.isfinite(metadata['elapsed_seconds']) or metadata['elapsed_seconds']<0:raise ValueError('invalid elapsed time')
    conditions={(r['T'],tau) for r in runtime.build_budget_schedule() for tau in (1,r['tau_local'])}
    require_keys([(r['T'],r['tau'],r['seed']) for r in rows],{(t,tau,s) for t,tau in conditions for s in range(30)})
    require_keys([_condition(r) for r in summary],conditions)
    for row in rows:
        if set(row)!=set(runner.RAW_FIELDS):raise ValueError('raw scientific fields changed')
        if any(isinstance(row[k],bool) or not isinstance(row[k],int) for k in _INTEGER_FIELDS if k in row):raise ValueError('noninteger raw identity/accounting')
        if any(not np.isfinite(v) or v<0 for k,v in row.items() if k!='schedule'):raise ValueError('nonfinite/negative communication data')
        total,tau=row['T'],row['tau'];rounds=total//tau
        if row['rounds']!=rounds or row['local_steps']!=total or rounds*tau!=total:raise ValueError('T=K*tau budget mismatch')
        if row['schedule']!=runner._schedule_label(total,tau):raise ValueError('schedule identity changed')
        queries=config['n_clients']*(rounds*config['samples_per_client']+total*config['batch_size'])
        if row['query_count']!=queries or row['transmitted_scalars']!=transmitted_scalars(config['n_clients'],config['dimension'],rounds):raise ValueError('independent query/scalar accounting mismatch')
        if row['criterion']!=row['stationarity']+row['consensus']:raise ValueError('criterion decomposition mismatch')
        eta=config['eta0']*total**(-.5);mu=config['mu0']*total**(-1./6.)
        eigenvalues=2.-2.*np.cos(2.*np.pi*np.arange(config['n_clients'],dtype=np.float64)/config['n_clients']);a0=1./float(np.max(eigenvalues))
        if (row['eta'],row['mu'],row['a0'],row['beta'])!=(eta,mu,a0,a0/(tau*eta*mu)):raise ValueError('frozen optimizer schedule mismatch')
    ordered=sorted(rows,key=lambda r:(r['T'],r['tau'],r['seed']))
    old=runner.aggregate_records(ordered);saved=sorted(summary,key=_condition)
    if old!=saved:raise ValueError('historical raw/helper/summary exact chain changed')
    current=_independent_summary(ordered);adjacent=_cross_chain(old,current)
    frozen_raw,_,_=_fixtures()
    if ordered!=frozen_raw:raise ValueError('raw scientific fields differ from frozen source')
    return current,adjacent


def verify(reference_root: Path):
    rows,saved,metadata=_fixtures();current,adjacent=_verify_records(rows,saved,metadata)
    root=Path(reference_root);verification=json.loads((root/'paper_values/verification.json').read_text(encoding='utf-8'))
    if verification['status']!='verified' or verification['difference_count']!=0:raise ValueError('Frozen reference verification failed')
    path=root/'aggregates/communication_summary.csv';verify_hash(path,verification['normalized_sha256'][path.name]);normalized=_read_csv(path)
    require_keys([_condition(r) for r in normalized],{_condition(r) for r in current})
    for actual,expected in zip(current,sorted(normalized,key=_condition)):
        if any(expected[k]!=v for k,v in actual.items()) or expected['diagnostic']!='non_private' or expected['paired_by']!='seed' or expected['local_steps']!=actual['T']:raise ValueError('independent current CI/normalized exact chain changed')
    if adjacent!=load_manifest()['adjacent_float64']:raise ValueError('adjacent CI evidence changed')
    return dict(status='verified',records=len(rows),summaries=len(current),paired_dataset_seeds=metadata['n_seeds'],differences=0,adjacent_float64=adjacent,communication_unit='transmitted_model_scalars',query_matched=False,full_common_random_numbers=False,saving_region='deterministic communication saved, not a confidence interval')


def run(output: Path):
    output=Path(output).resolve();references=(release_root()/'reference_results').resolve()
    if output.exists() or output==references or references in output.parents:raise ValueError('communication run requires a new directory outside reference_results')
    load_manifest();output.mkdir(parents=True,exist_ok=False)
    result=_execute(output)
    result.update(status='completed',manifest_path='run_manifest.json',source_manifest_sha256=_hash(release_root()/'configs/communication/manifest.json'))
    with (output/'run_manifest.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
    return result


def _execute(output):
    import contextlib
    import sys
    from . import plotting
    _,runner=_runtime()
    if any(output.iterdir()):raise ValueError('complete communication rerun requires an empty new directory')
    with contextlib.redirect_stdout(sys.stderr):rows,summary=runner.run_all(output)
    if len(rows)!=240 or len(summary)!=8:raise ValueError('actual rerun must contain 240 records and eight summaries')
    if _read_csv(output/'communication_cost_per_seed.csv')!=rows or _read_csv(output/'communication_cost_summary.csv')!=summary:raise ValueError('rerun disk records differ from actual returned results')
    metadata=json.loads((output/'experiment_metadata.json').read_text(encoding='utf-8'))
    current,adjacent=_verify_records(rows,summary,metadata)
    cache=output/'cache';expected={'seed_{:02d}_T_{}_tau_{}.json'.format(r['seed'],r['T'],r['tau']):r for r in rows}
    require_keys([p.name for p in cache.glob('*.json')],set(expected))
    for name,row in expected.items():
        result=json.loads((cache/name).read_text(encoding='utf-8'))
        if result!={**row,'config':metadata['config']}:raise ValueError('actual scientific execution record/config mismatch')
    figure=plotting.build_figure(summary)
    try:
        checked=plotting.verify_figure(figure,summary)
        exported=plotting.export_figure(figure,output,summary)
        if {p.name for p in exported}!={plotting.OUTPUT_STEM+s for s in ('.pdf','.svg','.png','.tiff')} or any(not p.is_file() or not p.stat().st_size for p in exported):raise ValueError('required Figure 10 output missing')
    finally:plotting.plt.close(figure)
    files={p.relative_to(output).as_posix():_hash(p) for p in sorted(output.rglob('*')) if p.is_file()}
    return dict(records=len(rows),summaries=len(current),executed_jobs=len(expected),figure=checked,files=files,adjacent_float64=adjacent,paired_dataset_seeds=30,query_matched=False,full_common_random_numbers=False,communication_unit='transmitted_model_scalars')
