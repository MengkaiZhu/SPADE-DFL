"""New frozen ring-grid experiments. Original trainer and release code are read-only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
W=ROOT.parent
P=W.parent
OLD=W/'table34_n30_eps8_working'
PRE=W/'table34_preprocessing_working'
DEL=W/'table34_delivery_working'
CAL=W/'table34_calibrated_working'
RELEASE=W/'table34_release_budget_working'
sys.path.insert(0,str(RELEASE))
import study as release
from study import read,save,sha,array_hash,verify_modules,save_npz,csv_text,save_check
NAMES={'mnist':'MNIST','fashion':'Fashion-MNIST','brain':'Brain Tumor MRI','alzheimer':'Alzheimer disease'}
OLD_PREFIX='unavailable-historical-root'

def relocate(path):
    s=str(path).replace('\\','/')
    if s==OLD_PREFIX or s.startswith(OLD_PREFIX+'/'):
        return P/Path(s[len(OLD_PREFIX):].lstrip('/'))
    return Path(s)

def conditions():
    return [(10,8.),(20,8.),(30,8.),(50,8.),(30,2.),(30,4.),(30,16.)]

def make_config(base,k,n,epsilon,seed,m,topology='ring'):
    if k not in (10,25,50) or n not in (10,20,30,50) or epsilon not in (2.,4.,8.,16.) or seed not in range(30):
        raise ValueError('Outside frozen ranges')
    if topology not in ('ring','erdos_renyi') or base['vr_mode'] not in ('full_refresh','persistent_saga'):
        raise ValueError('Unsupported mode')
    target=50*30*base['tau']*base['batch_size']+m*(50 if base['vr_mode']=='full_refresh' else 1)
    memory=m*(k if base['vr_mode']=='full_refresh' else 1)
    denom=k*n*base['tau'];batch=(target-memory)//denom
    if batch<1:raise ValueError('Budget cannot support a batch')
    return dict(base,n_rounds=k,n_clients=n,target_epsilon=float(epsilon),seeds=[seed],
                batch_size=batch,graph_topology=topology),dict(
        target=target,actual=memory+denom*batch,unused=target-memory-denom*batch,rounding_bound=denom)

def summarize(rows,folds,seeds):
    expected={(f,s) for f in folds for s in seeds}
    if len(rows)!=len(expected) or {(r['fold'],r['seed']) for r in rows}!=expected or len(seeds)<2:
        raise ValueError('Incomplete/duplicate fold-seed grid or fewer than two seeds')
    if any(not 0<=r['correct']<=r['eval_count'] or r['eval_count']<=0 for r in rows):
        raise ValueError('Invalid counts')
    values=[]
    for seed in seeds:
        rs=[r for r in rows if r['seed']==seed]
        values.append(100*sum(r['correct'] for r in rs)/sum(r['eval_count'] for r in rs))
    return dict(mean_percent=float(np.mean(values)),std_percent=float(np.std(values,ddof=1)),
                seed_accuracies_percent=values,seeds=list(seeds),n_seeds=len(seeds),n_fold_seed_runs=len(rows))

def data_for(group,formal=False):
    base=read(OLD/'configs'/f'{group}_n30_eps8_seed0.json')
    package=verify_modules(base);sys.path.insert(0,str(package.parent))
    from spade_dfl.data import DatasetBundle,PCAState
    if formal:
        entry=read(CAL/'FORMAL_DATA.json')['entries'][group]
        path=relocate(entry['path']);assert sha(path)==entry['sha256']
        with np.load(path,allow_pickle=False) as d:
            meta=json.loads(str(d['bundle_metadata_json']))
            values={k:d[k] for k in ('train_x','train_y','test_x','test_y')}
            for k,v in values.items():assert array_hash(v)==entry['array_sha256'][k]
            pca=PCAState(d['pca_mean'],d['pca_components'],d['pca_scale'])
        if base['dataset']!='alzheimer':
            weights=pca.scale/np.sqrt(np.mean(pca.scale**2))
            values['train_x']=values['train_x']*weights;values['test_x']=values['test_x']*weights
        # Independently compare preprocessing output to the prior frozen full-data run.
        start=read(DEL/'formal'/f'{group}_n30_eps8'/'seed0'/'STARTED.json')
        for k,v in values.items():
            name=k.replace('test_','eval_')
            assert array_hash(v)==start['inputs'][name],(group,k)
    else:
        path=PRE/'prepared'/group/'data.npz';info=read(path.with_name('metadata.json'))
        assert sha(path)==info['data_sha256'] and info['fit_scope']=='inner_training_only'
        candidate='baseline' if base['dataset']=='alzheimer' else 'common_scale'
        with np.load(path,allow_pickle=False) as d:
            for k in d.files:assert array_hash(d[k])==info['arrays'][k]
            values=dict(train_x=d['train_x_'+candidate],train_y=d['train_y'],
                        test_x=d['validation_x_'+candidate],test_y=d['validation_y'])
            pca=PCAState(d['pca_mean'],d['pca_components'],d['pca_scale'])
        meta=info['bundle_metadata']
    accepted={f.name for f in fields(DatasetBundle)}
    return DatasetBundle(**values,pca=pca,**{k:v for k,v in meta.items() if k in accepted}),base

def freeze():
    profiles=read(DEL/'SELECTED.json')['profiles']
    groups=read(PRE/'PLAN.json')['groups']
    files=[Path(__file__),ROOT/'PROTOCOL_CN.md',ROOT/'test_study.py',DEL/'SELECTED.json',PRE/'PLAN.json',
           CAL/'FORMAL_DATA.json',RELEASE/'study.py',W/'table34_final_working'/'pipeline.py',
           CAL/'study_runtime.py',CAL/'audit_results.py',OLD/'pilot_runtime.py',
           W/'table34_paired_control16_working'/'control_runtime.py',P/'AGENTS.md',P/'HANDOFF_SPADE-DFL.md',
           P/'SPADE_DFL.pdf',P/'communication_efficient_DFL——0907版本 (4).pdf']
    pilot=[];formal=[];identity={}
    formal_data=read(CAL/'FORMAL_DATA.json')['entries']
    for group,g in groups.items():
        bpath=OLD/'configs'/f'{group}_n30_eps8_seed0.json';base=read(bpath)
        package=verify_modules(base);files.extend(package.rglob('*.py'));files.append(bpath)
        dp=PRE/'prepared'/group/'data.npz';mp=dp.with_name('metadata.json')
        info=read(mp);assert sha(dp)==info['data_sha256'];files.extend([dp,mp])
        source=relocate(formal_data[group]['path']);assert sha(source)==formal_data[group]['sha256'];files.append(source)
        with np.load(dp) as d:inner_m=len(d['train_y'])
        with np.load(source) as d:outer_m=len(d['train_y'])
        identity[group]=dict(dataset=base['dataset'],fold=base['fold'])
        profile=profiles[base['dataset']]
        for topology in ('erdos_renyi','ring'):
            for seed in range(5):
                cfg,q=make_config(profile['base'],profile['k'],30,8.,seed,inner_m,topology)
                ref=W/'table34_lowdim_working'/'runs'/group/'baseline'/f'seed{seed}'
                if topology=='erdos_renyi':
                    assert read(ref/'config.json')==cfg,(group,'control configuration mismatch')
                    files.extend(ref/f for f in ('config.json','raw.npz','summary.json'))
                pilot.append(dict(group=group,candidate=topology,seed=seed,cfg=cfg,budget=q,reference=None,
                                  control_reference=str(ref) if topology=='erdos_renyi' else None,
                                  out=str(ROOT/'pilot'/group/topology/f'seed{seed}'),task=group))
        for n,eps in conditions():
            for seed in range(30):
                cfg,q=make_config(profile['base'],profile['k'],n,eps,seed,outer_m)
                formal.append(dict(group=group,candidate='ring_selected_profile',seed=seed,cfg=cfg,budget=q,reference=None,
                                   out=str(ROOT/'formal'/f'{group}_n{n}_eps{eps:g}'/f'seed{seed}'),
                                   task=f'{group}_n{n}_eps{eps:g}_b{seed//5}'))
        start=DEL/'formal'/f'{group}_n30_eps8'/'seed0'/'STARTED.json';files.append(start)
    assert len(pilot)==80 and len(formal)==1680
    save(ROOT/'PLAN.json',dict(groups=identity,profiles=profiles,pilot=pilot,formal=formal,
        sources={str(f):sha(f) for f in files},formal_seeds=list(range(30)),old_prefix=OLD_PREFIX,
        new_prefix=str(P),new_range_authorized=True,trend_selection=False))
    print('FROZEN 80 pilot + 1680 formal; no training',flush=True)

def plan():
    p=read(ROOT/'PLAN.json')
    for path,digest in p['sources'].items():assert sha(path)==digest,path
    return p

def diagnostics(j,bundle):
    with np.load(Path(j['out'])/'partition.npz') as d:
        labels=np.unique(bundle.train_y)
        counts=np.array([[np.count_nonzero(bundle.train_y[d[f'client_{i}']]==y) for y in labels]
                         for i in range(j['cfg']['n_clients'])])
        adj=d['adjacency']
    global_p=np.array([np.mean(bundle.train_y==y) for y in labels])
    sizes=counts.sum(axis=1);tv=.5*np.abs(counts/sizes[:,None]-global_p).sum(axis=1)
    meta=read(Path(j['out'])/'metadata.json')
    return dict(group=j['group'],seed=j['seed'],n=j['cfg']['n_clients'],epsilon=j['cfg']['target_epsilon'],
        topology=j['cfg']['graph_topology'],edges=int(adj.sum()//2),
        degree_min=int(adj.sum(axis=1).min()),degree_max=int(adj.sum(axis=1).max()),
        sample_min=int(sizes.min()),sample_max=int(sizes.max()),missing_class_clients=int(np.any(counts==0,axis=1).sum()),
        mean_tv=float(tv.mean()),sample_weighted_tv=float(np.average(tv,weights=sizes)),beta=float(meta['beta']),
        mean_noise_update_upperbound_ratio=float(np.mean(np.asarray(meta['release_noise_std_schedule'])/
             np.asarray(meta['clip_norm_schedule'])*np.sqrt(j['cfg']['dimension']/j['cfg']['n_clients']))))

def preflight(group):
    p=plan();results={}
    for phase in ('pilot','formal'):
        bundle,base=data_for(group,phase=='formal')
        from spade_dfl.config import ExperimentConfig
        from spade_dfl.graph import build_graph
        checked=set()
        for j in [x for x in p[phase] if x['group']==group]:
            cfg=ExperimentConfig.from_dict(j['cfg'])
            actual=json.loads(json.dumps(cfg.to_dict()))
            assert all(actual[k]==v for k,v in j['cfg'].items())
            assert bundle.train_x.shape[1]==(j['cfg'].get('feature_dimension',cfg.dimension))
            key=(cfg.n_clients,j['seed'],cfg.graph_topology)
            if key in checked:continue
            part=release.suite_partition(base['dataset'],j['cfg'],bundle.train_y,j['seed'])
            assert min(map(len,part.indices))>0
            assert np.array_equal(np.sort(np.concatenate(part.indices)),np.arange(len(bundle.train_y)))
            graph=build_graph(cfg.n_clients,topology=cfg.graph_topology,probability=cfg.graph_probability,seed=j['seed'])
            if cfg.graph_topology=='ring':assert np.all(graph.adjacency.sum(axis=1)==2)
            checked.add(key)
        results[phase]=len(checked)
    assert results==dict(pilot=10,formal=120)
    save(ROOT/'preflight'/f'{group}.json',dict(verified=True,counts=results,plan_sha256=sha(ROOT/'PLAN.json')))
    print('PREFLIGHT',group,results,flush=True)

def worker(phase,task,audit=False):
    p=plan()
    if phase=='formal':
        f=read(ROOT/'FORMAL_PLAN.json')
        assert f['plan_sha256']==sha(ROOT/'PLAN.json')
        assert f['pilot_verification_sha256']==sha(ROOT/'PILOT_VERIFICATION.json')
    jobs=[j for j in p[phase] if j['group']==task] if audit else [j for j in p[phase] if j['task']==task]
    group=jobs[0]['group'];b,base=data_for(group,phase=='formal')
    release.ROOT=ROOT
    rows=[];cache={};diag=[];replays=0
    for j in jobs:
        if not audit:release.run_one(j,b,base,phase)
        row=release.verify_job(j,b,base,phase,cache)
        s=read(Path(j['out'])/'summary.json')
        row.update(correct=s['validation_correct'],eval_count=s['eval_count'],topology=j['cfg']['graph_topology'])
        if phase=='pilot' and j['control_reference']:
            ref=Path(j['control_reference'])
            with np.load(ref/'raw.npz') as old,np.load(Path(j['out'])/'raw.npz') as new:
                assert set(old.files)==set(new.files)
                assert all(np.array_equal(old[k],new[k]) for k in old.files),('Exact replay failed',group,j['seed'])
            replays+=1
        rows.append(row);diag.append(diagnostics(j,b))
    save(ROOT/(('independent_' if audit else 'audit_')+phase)/f'{task}.json',
         dict(verified=True,rows=rows,diagnostics=diag,replays=replays))
    print('VERIFIED',phase,task,len(rows),flush=True)

def tasks_for(p,phase,audit=False):
    if phase=='preflight' or audit:return list(p['groups'])
    return list(dict.fromkeys(j['task'] for j in p[phase]))

def dispatch(phase,audit=False):
    p=plan()
    if phase!='preflight':
        for group in p['groups']:
            assert read(ROOT/'preflight'/f'{group}.json')['plan_sha256']==sha(ROOT/'PLAN.json')
    tag=('recheck_' if audit else '')+phase
    tasks=tasks_for(p,phase,audit)
    def child(task):
        env=os.environ.copy();env.pop('PYTHONUTF8',None);env['PYTHONIOENCODING']='utf-8';env['PYTHONDONTWRITEBYTECODE']='1'
        env.update({k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
        path=ROOT/('logs_'+tag)/(task+'.log');path.parent.mkdir(exist_ok=True)
        args=[sys.executable,__file__,'worker','--phase',phase,'--task',task]+(['--audit'] if audit else [])
        with path.open('x',encoding='utf-8') as f:r=subprocess.run(args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        return dict(task=task,exit_code=r.returncode,log=str(path))
    outcomes=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fut in as_completed([pool.submit(child,t) for t in tasks]):
            row=fut.result();outcomes.append(row);print(len(outcomes),'/',len(tasks),row,flush=True)
    save(ROOT/('DISPATCH_'+tag+'.json'),outcomes)
    assert all(r['exit_code']==0 for r in outcomes),'Failure retained; no automatic retry'

def collect(phase,independent=False):
    p=plan();rows=[];diag=[];replays=0
    for task in tasks_for(p,phase,independent):
        a=read(ROOT/(('independent_' if independent else 'audit_')+phase)/f'{task}.json')
        assert a['verified'];rows+=a['rows'];diag+=a['diagnostics'];replays+=a['replays']
    assert len(rows)==len(p[phase]) and not list((ROOT/phase).rglob('FAILED.json'))
    for r in rows:assert sha(r['source_summary'])==r['summary_sha256']
    return rows,diag,replays

def pilot_gate():
    rows,diag,replays=collect('pilot');rs,ds,rp=collect('pilot',True)
    key=lambda r:(r['group'],r['candidate'],r['seed'])
    assert sorted(rows,key=key)==sorted(rs,key=key) and replays==rp==40
    save_check(ROOT/'PILOT_RESULTS.csv',csv_text(rows));save_check(ROOT/'PILOT_DIAGNOSTICS.csv',csv_text(diag))
    stats=[]
    for dataset in NAMES:
        for topology in ('erdos_renyi','ring'):
            selected=[r for r in rows if r['dataset']==dataset and r['topology']==topology]
            stats.append(dict(dataset=dataset,topology=topology,**summarize(selected,list(range(5 if dataset=='alzheimer' else 1)),list(range(5)))))
    save(ROOT/'PILOT_CELLS.json',stats)
    save(ROOT/'PILOT_VERIFICATION.json',dict(status='VERIFIED',records=80,exact_controls=40,independent_recheck=True,
        queries=sum(r['queries'] for r in rows),trainer_seconds=sum(r['trainer_seconds'] for r in rows),
        end_to_end_privacy_certified=False,trend_selected=False))
    save(ROOT/'FORMAL_PLAN.json',dict(plan_sha256=sha(ROOT/'PLAN.json'),pilot_verification_sha256=sha(ROOT/'PILOT_VERIFICATION.json'),
        jobs=plan()['formal'],profile_selection='frozen before new tests',formal_runs=1680,all_new=True))
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','preflight','pilot','recheck_pilot','pilot_gate','formal','recheck_formal','worker'])
    ap.add_argument('--phase');ap.add_argument('--task');ap.add_argument('--audit',action='store_true');a=ap.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='pilot_gate':pilot_gate()
    elif a.action=='worker':
        if a.phase=='preflight':preflight(a.task)
        else:worker(a.phase,a.task,a.audit)
    else:dispatch(a.action.removeprefix('recheck_'),a.action.startswith('recheck_'))

