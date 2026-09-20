"""Original-release experiments with bounded query allocation. Never edit vendor code."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'table34_final_working'
sys.path.insert(0,str(PREV))
import pipeline as previous
from pipeline import read,save,sha,array_hash,verify_modules,save_npz,suite_partition,predict,check_budget
sys.path.insert(0,str(previous.CAL))
from audit_results import prediction,csv_text,aggregate,save_check,tex_cell

def condition_grid():
    return [(10,8.),(30,8.),(50,8.),(100,8.),(30,.5),(30,2.),(30,32.)]

def budget_config(base,k,n,eps,seed,m):
    if k not in (10,25,50) or n not in (10,30,50,100) or eps not in (.5,2.,8.,32.):
        raise ValueError('Outside frozen grid')
    target=50*30*base['tau']*base['batch_size']+m*(50 if base['vr_mode']=='full_refresh' else 1)
    memory=m*(k if base['vr_mode']=='full_refresh' else 1)
    denominator=k*n*base['tau']
    batch=(target-memory)//denominator
    if batch<1:raise ValueError('Cannot allocate one batch within query budget')
    actual=memory+denominator*batch
    return dict(base,n_rounds=k,n_clients=n,target_epsilon=float(eps),seeds=[seed],batch_size=batch),dict(
        target=target,actual=actual,unused=target-actual,rounding_bound=denominator)

def choose_rounds(rows,groups,candidates):
    expected={(g,s) for g in groups for s in range(5)};ranked=[]
    for c in candidates:
        rs=[r for r in rows if r['candidate']==c and r['group'] in groups]
        if len(rs)!=len(expected) or {(r['group'],r['seed']) for r in rs}!=expected:
            raise ValueError('Incomplete candidate: '+c)
        if any(r['status']!='completed' for r in rs):raise ValueError('Failed candidate: '+c)
        x=np.array([[r['balanced'],r['accuracy'],r['loss']] for r in rs])
        if not np.isfinite(x).all():raise ValueError('Nonfinite selection metric')
        b,a,l=x.mean(axis=0)
        ranked.append(dict(candidate=c,balanced=float(b),accuracy=float(a),loss=float(l)))
    return sorted(ranked,key=lambda r:(-r['balanced'],-r['accuracy'],r['loss'],-int(r['candidate'][1:])))[0]

def plan(formal=False):
    p=read(ROOT/'PLAN.json')
    for path,h in p['sources'].items():assert sha(path)==h,path
    if not formal:return p
    f=read(ROOT/'FORMAL_PLAN.json')
    assert f['plan_sha256']==sha(ROOT/'PLAN.json')
    assert f['selection_sha256']==sha(ROOT/'SELECTIONS.json')
    return f

def freeze():
    previous.plan();groups={};jobs=[]
    sources=[Path(__file__),ROOT/'PROTOCOL_CN.md',PREV/'pipeline.py',PREV/'PLAN.json',
             PREV/'FORMAL_PLAN.json',PREV/'FORMAL_VERIFICATION.json',PREV/'VALIDATION_VERIFICATION.json',
             previous.CAL/'study_runtime.py',previous.CAL/'audit_results.py']
    selected=read(PREV/'SELECTED_CONFIGS.json')
    for group,base in selected.items():
        s=read(PREV/'selections'/f'{group}.json')
        assert s['fields']==base and not s['official_test_used']
        sources.append(PREV/'selections'/f'{group}.json')
        g=read(PREV/'PLAN.json')['groups'][group]
        with np.load(g['data']) as d:m=len(d['train_y'])
        groups[group]=dict(dataset=g['dataset'],fold=g['fold'],base=base,validation_m=m)
        for k in [50,25,10]:
            for seed in range(5):
                cfg,budget=budget_config(base,k,30,8.,seed,m);ref=None
                if k==50:
                    old=next(x for x in s['rows'] if x['seed']==seed and x['candidate']==s['selected']['candidate'])
                    ref=old['path'];assert read(Path(ref)/'config.json')==cfg
                    sources.append(Path(ref)/'summary.json')
                jobs.append(dict(group=group,candidate=f'k{k}',seed=seed,cfg=cfg,budget=budget,reference=ref,
                    out=str(ROOT/'validation'/group/f'k{k}'/f'seed{seed}')))
    save(ROOT/'PLAN.json',dict(groups=groups,jobs=jobs,sources={str(x):sha(x) for x in sources},
        calibration_uses_outer_test=False,previous_outer_test_exposure=True,formal_conditions=condition_grid()))
    print('FROZEN',len(jobs),'validation records;',sum(j['reference'] is None for j in jobs),'new',flush=True)

def identity(j,phase):
    return dict(phase=phase,group=j['group'],candidate=j['candidate'],seed=j['seed'],
        n_clients=j['cfg']['n_clients'],epsilon=j['cfg']['target_epsilon'],
        plan_sha256=sha(ROOT/('FORMAL_PLAN.json' if phase=='formal' else 'PLAN.json')),
        evaluation='outer_test' if phase=='formal' else 'training_holdout')

def run_one(j,bundle,base,phase):
    if j['reference']:
        path=Path(j['reference'])
        assert read(path/'config.json')==j['cfg']
        return
    from spade_dfl.config import ExperimentConfig
    from spade_dfl.graph import build_graph
    from spade_dfl.trainer import run_experiment
    out=Path(j['out'])
    if out.exists():raise FileExistsError('No overwrite/retry: '+str(out))
    cfg=ExperimentConfig.from_dict(j['cfg'])
    effective=json.loads(json.dumps(cfg.to_dict()))
    assert all(effective[k]==v for k,v in j['cfg'].items())
    out.mkdir(parents=True);save(out/'config.json',j['cfg'])
    ident=identity(j,phase)
    save(out/'STARTED.json',dict(identity=ident,utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=sha(__file__),config_sha256=sha(out/'config.json'),
        inputs={k:array_hash(v) for k,v in dict(train_x=bundle.train_x,train_y=bundle.train_y,
            eval_x=bundle.test_x,eval_y=bundle.test_y).items()}))
    try:
        part=suite_partition(base['dataset'],j['cfg'],bundle.train_y,j['seed'])
        ids=part.indices
        assert min(map(len,ids))>0 and np.array_equal(np.sort(np.concatenate(ids)),np.arange(len(bundle.train_y)))
        graph=build_graph(cfg.n_clients,topology=cfg.graph_topology,probability=cfg.graph_probability,seed=j['seed'])
        save_npz(out/'partition.npz',**{f'client_{i}':v for i,v in enumerate(ids)},adjacency=graph.adjacency)
        result=run_experiment(cfg,bundle,j['seed'],partition=part)
        save_npz(out/'raw.npz',states=result.states,**result.history);save(out/'metadata.json',result.metadata)
        assert len(result.history['accuracy'])==cfg.n_rounds+1
        assert np.isfinite(result.states).all() and all(np.isfinite(v).all() for v in result.history.values())
        eps=check_budget(result.metadata['clip_norm_schedule'],result.metadata['release_noise_std_schedule'],cfg.target_delta,cfg.target_epsilon)
        pred=predict(j['cfg'],result.states,bundle.test_x)
        acc=float(np.mean(pred==bundle.test_y))
        bal=float(np.mean([np.mean(pred[bundle.test_y==v]==v) for v in np.unique(bundle.test_y)]))
        assert acc==result.history['accuracy'][-1]
        assert result.query_count==j['budget']['actual']
        row=dict(identity=ident,candidate=j['candidate'],seed=j['seed'],status='completed',accuracy=acc,balanced=bal,
            loss=float(result.history['loss'][-1]),epsilon=eps,validation_correct=int(np.sum(pred==bundle.test_y)),eval_count=len(pred),
            queries=result.query_count,trainer_seconds=result.wall_time,mean_clip_fraction=float(np.mean(result.metadata['clip_fraction'])),
            raw_sha256=sha(out/'raw.npz'),metadata_sha256=sha(out/'metadata.json'),config_sha256=sha(out/'config.json'),path=str(out))
        verify_modules(base);save(out/'summary.json',row)
        print(phase,j['group'],j['candidate'],j['seed'],f'{acc:.6f}',flush=True)
    except Exception:
        save(out/'FAILED.json',dict(error=traceback.format_exc(),automatic_retry=False));raise

def verify_job(j,bundle,base,phase,cache):
    from spade_dfl.graph import build_graph,derive_midpoint_beta
    path=Path(j['reference'] or j['out']);cfg=j['cfg'];k=cfg['n_rounds']
    r=read(path/'summary.json');start=read(path/'STARTED.json');meta=read(path/'metadata.json')
    assert r['status']=='completed' and not (path/'FAILED.json').exists()
    assert read(path/'config.json')==cfg and r['identity']==start['identity']
    assert r['seed']==j['seed'] and r['identity']['group']==j['group']
    if j['reference']:
        assert start['code_sha256']==sha(PREV/'pipeline.py') and start['plan_sha256']==sha(PREV/'PLAN.json')
        assert r['identity']['evaluation']==('outer_test_final_round50' if phase=='formal' else 'training_holdout')
    else:
        assert start['code_sha256']==sha(__file__) and r['identity']==identity(j,phase)
    for name,key in [('raw.npz','raw_sha256'),('config.json','config_sha256'),('metadata.json','metadata_sha256')]:
        assert sha(path/name)==r[key]
    assert start['config_sha256']==r['config_sha256']
    for key,a in dict(train_x=bundle.train_x,train_y=bundle.train_y,eval_x=bundle.test_x,eval_y=bundle.test_y).items():
        assert array_hash(a)==(start[key+'_sha256'] if j['reference'] else start['inputs'][key])
    cachekey=(cfg['n_clients'],j['seed'],cfg['partition_mode'])
    if cachekey not in cache:
        part=suite_partition(base['dataset'],cfg,bundle.train_y,j['seed'])
        graph=build_graph(cfg['n_clients'],topology=cfg['graph_topology'],probability=cfg['graph_probability'],seed=j['seed'])
        cache[cachekey]=(part,graph)
    part,graph=cache[cachekey]
    with np.load(path/'partition.npz') as ps,np.load(path/'raw.npz') as raw:
        ids=[ps[f'client_{i}'] for i in range(cfg['n_clients'])]
        assert all(np.array_equal(x,y) for x,y in zip(ids,part.indices))
        assert np.array_equal(ps['adjacency'],graph.adjacency)
        assert min(map(len,ids))>0 and np.array_equal(np.sort(np.concatenate(ids)),np.arange(len(bundle.train_y)))
        assert meta['client_sample_counts']==list(map(len,ids))
        assert all(np.isfinite(raw[a]).all() for a in raw.files)
        assert raw['states'].shape==(cfg['n_clients'],cfg['dimension']) and len(raw['accuracy'])==k+1
        pred=prediction(cfg,raw['states'],bundle.test_x);acc=float(np.mean(pred==bundle.test_y))
        bal=float(np.mean([np.mean(pred[bundle.test_y==v]==v) for v in np.unique(bundle.test_y)]))
        assert acc==r['accuracy']==raw['accuracy'][-1] and bal==r['balanced']
        assert int(np.sum(pred==bundle.test_y))==r['validation_correct'] and len(pred)==r['eval_count']
        assert r['loss']==raw['loss'][-1] and np.array_equal(raw['accuracy'][1:],raw['released_proxy_accuracy'])
    clips=np.asarray(meta['clip_norm_schedule']);sigmas=np.asarray(meta['release_noise_std_schedule'])
    assert len(clips)==len(sigmas)==k and np.all(sigmas>0) and np.all(clips>0)
    assert np.allclose(clips,cfg['clip_norm']*np.arange(1.,k+1)**(-cfg['clip_exponent']),rtol=1e-14,atol=0)
    rho=float(np.sum(2*clips**2/sigmas**2));eps=rho+2*np.sqrt(rho*np.log(1/cfg['target_delta']))
    assert np.isclose(eps,cfg['target_epsilon'],rtol=1e-10) and np.isclose(eps,meta['epsilon']) and np.isclose(eps,r['epsilon'])
    assert meta['stage']==cfg['stage']=='spade_dfl_private'
    assert np.isclose(meta['beta'],derive_midpoint_beta(graph,cfg['tau'],cfg['rho']))
    expected=k*cfg['n_clients']*cfg['tau']*cfg['batch_size']+len(bundle.train_y)*(k if cfg['vr_mode']=='full_refresh' else 1)
    assert expected==r['queries']==j['budget']['actual']
    assert 0<=j['budget']['unused']<k*cfg['n_clients']*cfg['tau']
    return dict(group=j['group'],dataset=base['dataset'],fold=base['fold'],candidate=j['candidate'],seed=j['seed'],status='completed',
        n=cfg['n_clients'],epsilon=cfg['target_epsilon'],rounds=k,accuracy=acc,balanced=bal,loss=r['loss'],
        queries=expected,target_queries=j['budget']['target'],unused_queries=j['budget']['unused'],
        trainer_seconds=r['trainer_seconds'],reused=bool(j['reference']),source_summary=str(path/'summary.json'),
        source_config=str(path/'config.json'),source_raw=str(path/'raw.npz'),summary_sha256=sha(path/'summary.json'))

def worker(phase,task):
    p=plan(phase=='formal')
    if phase=='validation':jobs=[j for j in p['jobs'] if j['group']==task]
    else:jobs=[j for j in p['jobs'] if j['task']==task]
    group=jobs[0]['group'];bundle,base=previous.bundle_for(group,phase=='formal');rows=[];cache={}
    for j in jobs:
        run_one(j,bundle,base,phase);rows.append(verify_job(j,bundle,base,phase,cache))
    save(ROOT/f'audit_{phase}'/f'{task}.json',dict(rows=rows,verified=True))

def dispatch(phase):
    p=plan(phase=='formal')
    tasks=list(p['groups']) if phase=='validation' else list(dict.fromkeys(j['task'] for j in p['jobs']))
    def child(task):
        env=os.environ.copy();env.pop('PYTHONUTF8',None);env['PYTHONIOENCODING']='utf-8'
        env.update({a:'1' for a in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
        log=ROOT/f'logs_{phase}'/f'{task}.log';log.parent.mkdir(exist_ok=True)
        with log.open('x',encoding='utf-8') as f:
            x=subprocess.run([sys.executable,__file__,'worker','--phase',phase,'--task',task],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        return dict(task=task,exit_code=x.returncode,log=str(log))
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(child,t) for t in tasks]):
            x=future.result();results.append(x);print(len(results),'/',len(tasks),x,flush=True)
    save(ROOT/f'DISPATCH_{phase}.json',results)
    assert not any(x['exit_code'] for x in results),'Failures preserved; do not continue'

def collect(phase):
    p=plan(phase=='formal');rows=[]
    tasks=list(p['groups']) if phase=='validation' else list(dict.fromkeys(j['task'] for j in p['jobs']))
    for t in tasks:
        a=read(ROOT/f'audit_{phase}'/f'{t}.json');assert a['verified'];rows.extend(a['rows'])
    assert len(rows)==len(p['jobs']) and not list((ROOT/phase).rglob('FAILED.json'))
    for r in rows:assert sha(r['source_summary'])==r['summary_sha256']
    return rows

def select():
    p=plan();rows=collect('validation');selections={}
    for ds in ['mnist','fashion','brain','alzheimer']:
        gs=[g for g,v in p['groups'].items() if v['dataset']==ds]
        selections[ds]=choose_rounds(rows,gs,['k50','k25','k10'])
    save(ROOT/'SELECTIONS.json',dict(selections=selections,official_test_used=False,plan_sha256=sha(ROOT/'PLAN.json')))
    save_check(ROOT/'VALIDATION_RESULTS.csv',csv_text(rows))
    groups=p['groups'];jobs=[]
    for group,g in groups.items():
        k=int(selections[g['dataset']]['candidate'][1:])
        data=read(previous.CAL/'FORMAL_DATA.json')['entries'][group]
        assert sha(data['path'])==data['sha256']
        with np.load(data['path']) as d:m=len(d['train_y'])
        for n,eps in condition_grid():
            for seed in range(30):
                cfg,budget=budget_config(g['base'],k,n,eps,seed,m)
                ref=PREV/'formal'/f'{group}_n{n}_eps{eps:g}'/f'seed{seed}'
                reference=str(ref) if ref.exists() and read(ref/'config.json')==cfg else None
                jobs.append(dict(group=group,candidate=f'k{k}',seed=seed,cfg=cfg,budget=budget,reference=reference,
                    out=str(ROOT/'formal'/f'{group}_n{n}_eps{eps:g}'/f'seed{seed}'),task=f'{group}_n{n}_eps{eps:g}_b{seed//5}'))
    assert len(jobs)==1680
    save(ROOT/'FORMAL_PLAN.json',dict(groups=groups,jobs=jobs,plan_sha256=sha(ROOT/'PLAN.json'),
        selection_sha256=sha(ROOT/'SELECTIONS.json'),selection_frozen_before_formal=True))
    print(json.dumps(selections,indent=2));print('Formal new',sum(j['reference'] is None for j in jobs),flush=True)

def audit():
    rows=collect('formal');save_check(ROOT/'FORMAL_RESULTS.csv',csv_text(rows))
    for j in plan(True)['jobs']:
        path=Path(j['reference'] or j['out']);r=read(path/'summary.json')
        for f,key in [('raw.npz','raw_sha256'),('config.json','config_sha256'),('metadata.json','metadata_sha256')]:assert sha(path/f)==r[key]
    save(ROOT/'FORMAL_VERIFICATION.json',dict(status='VERIFIED',records=len(rows),new_runs=sum(not r['reused'] for r in rows),
        reused_runs=sum(r['reused'] for r in rows),script_sha256=sha(__file__),formal_plan_sha256=sha(ROOT/'FORMAL_PLAN.json'),
        queries=sum(r['queries'] for r in rows),new_trainer_seconds=sum(r['trainer_seconds'] for r in rows if not r['reused']),
        all_endpoints_partitions_graphs_accountants_checked=True))
    print('VERIFIED',len(rows),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','validation','select','formal','audit','worker'])
    ap.add_argument('--phase');ap.add_argument('--task');a=ap.parse_args()
    if a.action=='freeze':freeze()
    elif a.action in ['validation','formal']:dispatch(a.action)
    elif a.action=='select':select()
    elif a.action=='audit':audit()
    else:worker(a.phase,a.task)
