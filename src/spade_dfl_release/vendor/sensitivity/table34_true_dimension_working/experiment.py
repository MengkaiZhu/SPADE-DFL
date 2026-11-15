"""Frozen true-dimensional PCA study; original optimizers are read-only."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from adapter import adapt,transform_bundle,activate_brain_config

ROOT=Path(__file__).resolve().parent
RING=ROOT.parent/'table34_ring_moderate_working'
sys.path.insert(0,str(RING))
import study_ring as s
from study_ring import read,save,sha,array_hash,csv_text,save_check

def candidates(ds):
    ks=[1,3,5,10,38] if ds=='alzheimer' else ([3,5] if ds=='brain' else [1,3,5])
    return [('baseline',None)]+[(f'pc{k}',k) for k in ks]

def pooled(rows,folds,seeds):
    result=s.summarize(rows,folds,seeds);balanced=[];loss=[]
    for seed in seeds:
        rr=[r for r in rows if r['seed']==seed]
        counts=np.sum([r['class_count'] for r in rr],axis=0)
        correct=np.sum([r['class_correct'] for r in rr],axis=0)
        if np.any(counts<=0) or np.any(correct>counts) or np.any(correct<0):raise ValueError('Invalid class counts')
        balanced.append(100*float(np.mean(correct/counts)))
        loss.append(float(np.average([r['validation_loss'] for r in rr],weights=[r['eval_count'] for r in rr])))
    return dict(result,balanced_percent=float(np.mean(balanced)),validation_loss=float(np.mean(loss)))

def check_schema_diff():
    original=ROOT.parent/'table34_n30_eps8_working/vendor/brain/src/spade_dfl/config.py'
    old=original.read_text(encoding='utf-8');new=(ROOT/'brain_config_working.py').read_text(encoding='utf-8')
    expected=old.replace('if self.feature_dimension != 10:', 'if self.feature_dimension <= 0:').replace(
        'brain_tumor_mri requires 10 PCA features','brain_tumor_mri requires positive PCA feature dimension')
    assert new==expected,'Unauthorized schema-copy change'

def freeze():
    check_schema_diff();prior=s.plan();sources=dict(prior['sources']);jobs=[]
    for name in ['experiment.py','adapter.py','brain_config_working.py','test_adapter.py','test_statistics.py','PROTOCOL_CN.md']:
        sources[str(ROOT/name)]=sha(ROOT/name)
    source=ROOT.parent/'table34_lowdim_working/lowdim.py';sources[str(source)]=sha(source)
    for group,g in prior['groups'].items():
        anchorpath=RING/'pilot'/group/'ring/seed0/config.json';anchor=read(anchorpath)
        sources[str(anchorpath)]=sha(anchorpath)
        dp=s.PRE/'prepared'/group/'data.npz'
        with np.load(dp) as data:m=len(data['train_y'])
        for candidate,k in candidates(g['dataset']):
            for seed in range(5):
                cfg=adapt(dict(anchor,seeds=[seed]),g['dataset'],k)
                q=cfg['n_rounds']*30*cfg['tau']*cfg['batch_size']+m*(cfg['n_rounds'] if cfg['vr_mode']=='full_refresh' else 1)
                jobs.append(dict(group=group,candidate=candidate,retained=k,seed=seed,cfg=cfg,
                    budget=dict(actual=q,target=q,unused=0),reference=None,task=group,
                    out=str(ROOT/'runs'/group/candidate/f'seed{seed}')))
                if k is None:
                    for f in ('config.json','raw.npz','summary.json'):
                        path=RING/'pilot'/group/'ring'/f'seed{seed}'/f;sources[str(path)]=sha(path)
    assert len(jobs)==205
    save(ROOT/'PLAN.json',dict(groups=prior['groups'],jobs=jobs,sources=sources,phase='validation',
        seeds=list(range(5)),end_to_end_privacy_certified=False,selection_uses_test=False,previous_test_exposure=True))
    print('FROZEN',len(jobs),flush=True)

def plan(phase):
    p=read(ROOT/'PLAN.json')
    for f,h in p['sources'].items():assert sha(f)==h,f
    check_schema_diff()
    if phase=='formal':
        f=read(ROOT/'FORMAL_PLAN.json')
        assert f['validation_plan_sha256']==sha(ROOT/'PLAN.json')
        assert f['selection_sha256']==sha(ROOT/'SELECTIONS.json')
        return f
    return p

def prepare(group,k,formal):
    original,base=s.data_for(group,formal)
    bundle,state=transform_bundle(original,base['dataset'],k)
    if state.get('kind')=='clinical':state['fit_scope']='outer_training_only' if formal else 'inner_training_only'
    if base['dataset']=='brain':activate_brain_config()
    return bundle,base,state

def verify(j,b,base,state,phase,cache):
    out=Path(j['out']);saved=read(out/'TRANSFORM.json')
    assert saved==dict(state=state,train_sha256=array_hash(b.train_x),eval_sha256=array_hash(b.test_x),
        actual_model_dimension=j['cfg']['dimension'],labels_unchanged=True)
    row=s.release.verify_job(j,b,base,phase,cache)
    summary=read(out/'summary.json')
    from spade_dfl.objectives import build_objective
    cfg=j['cfg'];kwargs=dict(feature_dimension=cfg['feature_dimension'],n_classes=cfg['n_classes']) if 'feature_dimension' in cfg else {}
    obj=build_objective(cfg['objective_model'],cfg['regularization'],**kwargs)
    with np.load(out/'raw.npz') as raw:
        pred=s.release.predict(cfg,raw['states'],b.test_x)
        loss=float(obj.value(raw['states'].mean(axis=0),b.test_x,b.test_y,u=1.))
        if phase=='validation' and j['retained'] is None:
            old=RING/'pilot'/j['group']/'ring'/f"seed{j['seed']}"
            assert read(old/'config.json')==cfg
            with np.load(old/'raw.npz') as control:
                assert set(control.files)==set(raw.files)
                assert all(np.array_equal(control[k],raw[k]) for k in control.files),'Baseline not bitwise equal'
    labels=np.unique(b.test_y)
    row.update(correct=summary['validation_correct'],eval_count=summary['eval_count'],retained=j['retained'],
        model_dimension=cfg['dimension'],validation_loss=loss,
        class_correct=[int(np.sum((pred==v)&(b.test_y==v))) for v in labels],
        class_count=[int(np.sum(b.test_y==v)) for v in labels],class_labels=labels.tolist(),
        transform_sha256=sha(out/'TRANSFORM.json'))
    return row

def worker(phase,task,audit=False):
    p=plan(phase);jobs=[j for j in p['jobs'] if j['group']==task] if audit else [j for j in p['jobs'] if j['task']==task]
    group=jobs[0]['group'];s.release.ROOT=ROOT;rows=[];cache={};prepared={}
    for j in jobs:
        k=j['retained']
        if k not in prepared:prepared[k]=prepare(group,k,phase=='formal')
        b,base,state=prepared[k];out=Path(j['out'])
        if not audit:
            s.release.run_one(j,b,base,phase)
            save(out/'TRANSFORM.json',dict(state=state,train_sha256=array_hash(b.train_x),eval_sha256=array_hash(b.test_x),
                actual_model_dimension=j['cfg']['dimension'],labels_unchanged=True))
        rows.append(verify(j,b,base,state,phase,cache))
    if phase=='validation':
        for seed in range(5):
            js=[j for j in jobs if j['seed']==seed]
            assert len({sha(Path(j['out'])/'partition.npz') for j in js})==1
            schedules=[read(Path(j['out'])/'metadata.json')['release_noise_std_schedule'] for j in js]
            assert all(x==schedules[0] for x in schedules)
    save(ROOT/(('independent_' if audit else 'audit_')+phase)/f'{task}.json',dict(verified=True,rows=rows))
    print('VERIFIED',phase,task,len(rows),flush=True)

def dispatch(phase,audit=False):
    p=plan(phase);tasks=list(p['groups']) if audit else list(dict.fromkeys(j['task'] for j in p['jobs']))
    pending=iter(tasks);outcomes=[];failed=False
    def child(task):
        tag=phase+('_recheck' if audit else '_run');log=ROOT/('logs_'+tag)/(task+'.log');log.parent.mkdir(exist_ok=True)
        env=os.environ.copy();env.pop('PYTHONUTF8',None)
        env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
        env.update({k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
        args=[sys.executable,'-B',__file__,'worker','--phase',phase,'--task',task]+(['--audit'] if audit else [])
        with log.open('x',encoding='utf-8') as f:r=subprocess.run(args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        return dict(task=task,exit_code=r.returncode,log=str(log))
    with ThreadPoolExecutor(max_workers=4) as pool:
        active={pool.submit(child,next(pending)) for _ in range(min(4,len(tasks)))}
        while active:
            done,active=wait(active,return_when=FIRST_COMPLETED)
            completed=[f.result() for f in done];outcomes+=completed
            failed=failed or any(r['exit_code'] for r in completed)
            for r in completed:print(r,flush=True)
            if not failed:
                for _ in completed:
                    task=next(pending,None)
                    if task is not None:active.add(pool.submit(child,task))
    save(ROOT/f'DISPATCH_{phase}_{"recheck" if audit else "run"}.json',outcomes)
    assert not failed and len(outcomes)==len(tasks),'Failed work retained, no automatic retry'

def verified_rows(phase):
    p=plan(phase);rows=[];first=[]
    for path in (ROOT/('audit_'+phase)).glob('*.json'):first+=read(path)['rows']
    for group in p['groups']:rows+=read(ROOT/('independent_'+phase)/(group+'.json'))['rows']
    key=lambda r:(r['group'],r['candidate'],r['n'],r['epsilon'],r['seed'])
    assert sorted(rows,key=key)==sorted(first,key=key)
    expected={(j['group'],j['candidate'],j['cfg']['n_clients'],j['cfg']['target_epsilon'],j['seed']) for j in p['jobs']}
    assert len(rows)==len(expected) and {key(r) for r in rows}==expected
    save_check(ROOT/(phase.upper()+'_RUNS.csv'),csv_text(rows))
    return rows

def select():
    p=plan('validation');rows=verified_rows('validation');stats=[];selected={}
    for ds in s.NAMES:
        folds=[g['fold'] for g in p['groups'].values() if g['dataset']==ds];choices=[]
        for order,(candidate,k) in enumerate(candidates(ds)):
            rr=[r for r in rows if r['dataset']==ds and r['candidate']==candidate]
            st=dict(dataset=ds,candidate=candidate,retained=k,dimension=rr[0]['model_dimension'],
                    order=order,**pooled(rr,folds,range(5)))
            stats.append(st);choices.append(st)
        selected[ds]=max(choices,key=lambda x:(x['balanced_percent'],x['mean_percent'],-x['validation_loss'],-x['order']))
    save(ROOT/'SELECTIONS.json',dict(selected=selected,all_candidates=stats,validation_only=True,
        criterion='balanced_accuracy_then_accuracy_then_validation_loss_then_order',trend_selection=False))
    print(json.dumps(selected,ensure_ascii=False,indent=2),flush=True)

def freeze_formal():
    p=plan('validation');verified_rows('validation');selections=read(ROOT/'SELECTIONS.json');jobs=[]
    for group,g in p['groups'].items():
        anchor=read(RING/'pilot'/group/'ring/seed0/config.json');choice=selections['selected'][g['dataset']]
        entry=read(s.CAL/'FORMAL_DATA.json')['entries'][group]
        with np.load(s.relocate(entry['path'])) as data:m=len(data['train_y'])
        for n,eps in s.conditions():
            for seed in range(30):
                cfg=adapt(dict(anchor,n_clients=n,target_epsilon=eps,seeds=[seed]),g['dataset'],choice['retained'])
                q=cfg['n_rounds']*n*cfg['tau']*cfg['batch_size']+m*(cfg['n_rounds'] if cfg['vr_mode']=='full_refresh' else 1)
                task=f'{group}_n{n}_eps{eps:g}'
                jobs.append(dict(group=group,candidate=choice['candidate'],retained=choice['retained'],seed=seed,cfg=cfg,
                    budget=dict(actual=q,target=q,unused=0),reference=None,task=task,out=str(ROOT/'formal'/task/f'seed{seed}')))
    assert len(jobs)==1680
    save(ROOT/'FORMAL_PLAN.json',dict(groups=p['groups'],jobs=jobs,seeds=list(range(30)),
        validation_plan_sha256=sha(ROOT/'PLAN.json'),selection_sha256=sha(ROOT/'SELECTIONS.json'),
        fixed_local_batch=True,end_to_end_privacy_certified=False,previous_test_exposure=True))
    print('FROZEN 1680 formal jobs',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run','recheck','worker','select','freeze_formal'])
    ap.add_argument('--phase',choices=['validation','formal'],default='validation');ap.add_argument('--task');ap.add_argument('--audit',action='store_true');a=ap.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='select':select()
    elif a.action=='freeze_formal':freeze_formal()
    elif a.action=='worker':worker(a.phase,a.task,a.audit)
    else:dispatch(a.phase,a.action=='recheck')
