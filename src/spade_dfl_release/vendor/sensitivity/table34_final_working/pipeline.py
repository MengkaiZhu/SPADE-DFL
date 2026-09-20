"""Frozen validation and final sensitivity runs; original algorithms are read-only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parent
W=ROOT.parent
CAL=W/'table34_calibrated_working'
OLD=W/'table34_n30_eps8_working'
D=W/'table34_heterogeneity_diagnostic'
RECAL=W/'table34_clip_recalibration_working'
LOCAL=W/'table34_local_compute_working'
sys.path.insert(0,str(CAL))
from study_runtime import read,save,sha,array_hash,verify_modules,check_budget,save_npz,suite_partition,select_candidate,predict


def query_budget(cfg,ntrain):
    assert cfg['vr_mode'] in ['persistent_saga','full_refresh']
    return cfg['n_rounds']*cfg['n_clients']*cfg['tau']*cfg['batch_size']+ntrain*(cfg['n_rounds'] if cfg['vr_mode']=='full_refresh' else 1)


def conditions():
    return [(10,8.),(30,8.),(50,8.),(100,8.),(30,.5),(30,2.),(30,32.)]


def candidates(cfg,small):
    variants=[cfg]
    for eta in [cfg['eta0'],1.5/cfg['dimension']]:
        variants.append(dict(cfg,eta0=eta))
        variants.extend(dict(cfg,eta0=eta,clip_norm=small,clip_exponent=p) for p in [0.,.51])
        variants.append(dict(cfg,eta0=eta,vr_mode='full_refresh'))
    unique=[]
    for v in variants:
        if v not in unique:unique.append(v)
    return [dict(id=f'v{i}',fields=v) for i,v in enumerate(unique)]


def selected_config(path):
    s=read(path)
    assert not s['official_test_used'] and select_candidate(s['rows'])==s['selected']
    row=next(r for r in s['rows'] if r['candidate']==s['selected']['candidate'])
    return read(Path(row['path'])/'config.json')


def freeze():
    dp=read(D/'PLAN.json');groups={};jobs=[]
    sources=[Path(__file__),ROOT/'PROTOCOL_CN.md',D/'PLAN.json',CAL/'FORMAL_DATA.json',
             RECAL/'VERIFICATION.json',LOCAL/'BATCH_VERIFICATION.json',CAL/'study_runtime.py',
             OLD/'pilot_runtime.py',W/'table34_paired_control16_working/control_runtime.py']
    for f in ['calibration.py','adapters.py']:
        sources.append(next(W.parent.glob('*/S18/SPADE_DFL_optimization_20260831/'+f)))
    for group,g in dp['groups'].items():
        rp=RECAL/'selections'/f'{group}.json';lp=LOCAL/'batch_selections'/f'{group}.json';sources.extend([rp,lp])
        cfg=selected_config(rp);small=selected_config(lp)['clip_norm']
        cfg.update(seeds=[0],n_clients=30,target_epsilon=8.)
        if g['dataset']=='alzheimer':cfg['partition_mode']='dirichlet_balanced'
        base=read(OLD/g['base']);verify_modules(base);sources.extend([OLD/g['base'],Path(g['data']),Path(g['metadata'])])
        assert sha(g['data'])==g['data_sha256'] and sha(g['metadata'])==g['metadata_sha256']
        cs=candidates(cfg,small);groups[group]=dict(g,candidates=cs)
        for c in cs:
            for seed in range(5):
                jobs.append(dict(group=group,candidate=c['id'],seed=seed,fields=dict(c['fields'],seeds=[seed])))
    save(ROOT/'PLAN.json',dict(groups=groups,jobs=jobs,source_hashes={str(f):sha(f) for f in sources},
        selection='complete five-seed mean balanced validation accuracy, accuracy, negative loss, ID',
        all_candidates_complete=True,official_test_used=False,formal_runs=1680))
    print('FROZEN calibration',len(jobs),'formal 1680',flush=True)


def plan():
    p=read(ROOT/'PLAN.json')
    for f,h in p['source_hashes'].items():assert sha(f)==h,f
    return p


def bundle_for(group,formal=False):
    p=plan();g=p['groups'][group];base=read(OLD/g['base']);package=verify_modules(base)
    sys.path.insert(0,str(package.parent))
    from spade_dfl.data import DatasetBundle,PCAState
    if formal:
        data=read(CAL/'FORMAL_DATA.json')['entries'][group];path=data['path'];assert sha(path)==data['sha256']
        with np.load(path) as d:
            meta=json.loads(str(d['bundle_metadata_json']));values={k:d[k] for k in ['train_x','train_y','test_x','test_y']}
            pca=PCAState(d['pca_mean'],d['pca_components'],d['pca_scale'])
    else:
        assert sha(g['data'])==g['data_sha256'];meta=read(g['metadata'])
        with np.load(g['data']) as d:
            values=dict(train_x=d['train_x'],train_y=d['train_y'],test_x=d['validation_x'],test_y=d['validation_y'])
            pca=PCAState(d['pca_mean'],d['pca_components'],d['pca_scale'])
    accepted={f.name for f in fields(DatasetBundle)}
    return DatasetBundle(**values,pca=pca,**{k:v for k,v in meta.items() if k in accepted}),base


def run_one(out,cfg_fields,base,bundle,seed,identity):
    if out.exists():raise FileExistsError('No automatic retry or overwrite: '+str(out))
    from spade_dfl.config import ExperimentConfig
    from spade_dfl.graph import build_graph,derive_midpoint_beta
    from spade_dfl.trainer import run_experiment
    cfg=ExperimentConfig.from_dict(cfg_fields)
    effective=json.loads(json.dumps(cfg.to_dict()));assert all(effective[k]==v for k,v in cfg_fields.items())
    out.mkdir(parents=True);save(out/'config.json',cfg_fields)
    save(out/'STARTED.json',dict(identity=identity,utc=datetime.now(timezone.utc).isoformat(),
        config_sha256=sha(out/'config.json'),code_sha256=sha(__file__),plan_sha256=sha(ROOT/'PLAN.json'),
        train_x_sha256=array_hash(bundle.train_x),train_y_sha256=array_hash(bundle.train_y),
        eval_x_sha256=array_hash(bundle.test_x),eval_y_sha256=array_hash(bundle.test_y),
        original_module_map_sha256=base['algorithm_source']['all_python_module_hash_map_sha256'],python=sys.version,numpy=np.__version__))
    try:
        partition=suite_partition(base['dataset'],cfg_fields,bundle.train_y,seed);ids=partition.indices
        assert min(map(len,ids))>0 and np.array_equal(np.sort(np.concatenate(ids)),np.arange(len(bundle.train_y)))
        graph=build_graph(cfg.n_clients,topology=cfg.graph_topology,probability=cfg.graph_probability,seed=seed)
        save_npz(out/'partition.npz',**{f'client_{i}':v for i,v in enumerate(ids)},adjacency=graph.adjacency)
        result=run_experiment(cfg,bundle,seed,partition=partition)
        save_npz(out/'raw.npz',states=result.states,**result.history);save(out/'metadata.json',result.metadata)
        assert np.isfinite(result.states).all() and all(np.isfinite(v).all() for v in result.history.values())
        assert len(result.history['accuracy'])==51
        eps=check_budget(result.metadata['clip_norm_schedule'],result.metadata['release_noise_std_schedule'],cfg.target_delta,cfg.target_epsilon)
        assert np.isclose(result.metadata['beta'],derive_midpoint_beta(graph,cfg.tau,cfg.rho))
        assert result.metadata['client_sample_counts']==[len(i) for i in ids]
        assert np.array_equal(result.history['accuracy'][1:],result.history['released_proxy_accuracy'])
        pred=predict(cfg_fields,result.states,bundle.test_x);accuracy=float(np.mean(pred==bundle.test_y))
        balanced=float(np.mean([np.mean(pred[bundle.test_y==v]==v) for v in np.unique(bundle.test_y)]))
        assert accuracy==result.history['accuracy'][-1]
        assert result.query_count==query_budget(cfg_fields,len(bundle.train_y)),(result.query_count,query_budget(cfg_fields,len(bundle.train_y)))
        row=dict(identity=identity,candidate=identity['candidate'],seed=seed,status='completed',accuracy=accuracy,balanced=balanced,
            loss=float(result.history['loss'][-1]),epsilon=eps,validation_correct=int(np.sum(pred==bundle.test_y)),eval_count=len(pred),
            queries=result.query_count,trainer_seconds=result.wall_time,mean_clip_fraction=float(np.mean(result.metadata['clip_fraction'])),
            raw_sha256=sha(out/'raw.npz'),metadata_sha256=sha(out/'metadata.json'),config_sha256=sha(out/'config.json'),path=str(out),
            end_to_end_privacy_certified=False)
        verify_modules(base);save(out/'summary.json',row);return row
    except Exception:
        save(out/'FAILED.json',dict(identity=identity,error=traceback.format_exc(),automatic_retry=False));raise


def calibration(group):
    p=plan();bundle,base=bundle_for(group);rows=[]
    for j in [j for j in p['jobs'] if j['group']==group]:
        identity=dict(phase='final_validation',group=group,candidate=j['candidate'],seed=j['seed'],n_clients=30,epsilon=8.,
            plan_sha256=sha(ROOT/'PLAN.json'),evaluation='training_holdout',formal_table_eligible=False)
        row=run_one(ROOT/'validation'/group/j['candidate']/f"seed{j['seed']}",j['fields'],base,bundle,j['seed'],identity);rows.append(row)
        print(group,j['candidate'],j['seed'],row['accuracy'],flush=True)
    assert len(rows)==5*len(p['groups'][group]['candidates'])
    selected=select_candidate(rows);cfg=next(c['fields'] for c in p['groups'][group]['candidates'] if c['id']==selected['candidate'])
    save(ROOT/'selections'/f'{group}.json',dict(selected=selected,fields=cfg,rows=rows,official_test_used=False))
    print('SELECTED',group,selected,flush=True)


def freeze_formal():
    p=plan();groups={};jobs=[]
    # Only check feasibility here, in isolated child processes, before reading formal outcomes.
    for group,g in p['groups'].items():
        path=ROOT/'selections'/f'{group}.json';s=read(path);assert select_candidate(s['rows'])==s['selected'] and not s['official_test_used']
        groups[group]=dict(selection=str(path),selection_sha256=sha(path),fields=s['fields'],candidate=s['selected']['candidate'])
        for n,eps in conditions():
            for seed in range(30):
                task=f'{group}_n{n}_eps{eps:g}_block{seed//5}'
                jobs.append(dict(task=task,group=group,n=n,epsilon=eps,seed=seed,fields=dict(s['fields'],n_clients=n,target_epsilon=eps,seeds=[seed])))
    assert len(jobs)==1680 and len({(j['group'],j['n'],j['epsilon'],j['seed']) for j in jobs})==1680
    save(ROOT/'FORMAL_PLAN.json',dict(groups=groups,jobs=jobs,source_plan_sha256=sha(ROOT/'PLAN.json'),code_sha256=sha(__file__),
        selection_frozen_before_testing=True,previous_outer_test_exposure=True,statistic='final50 ordinary accuracy; population SD ddof0'))
    print('FROZEN',len(jobs),'formal runs',flush=True)


def formal_plan():
    plan();p=read(ROOT/'FORMAL_PLAN.json');assert sha(__file__)==p['code_sha256'] and sha(ROOT/'PLAN.json')==p['source_plan_sha256']
    for g in p['groups'].values():assert sha(g['selection'])==g['selection_sha256']
    return p


def formal_task(task):
    p=formal_plan();jobs=[j for j in p['jobs'] if j['task']==task];assert len(jobs)==5
    group=jobs[0]['group'];bundle,base=bundle_for(group,True)
    for j in jobs:
        identity=dict(phase='formal',group=group,candidate=p['groups'][group]['candidate'],seed=j['seed'],n_clients=j['n'],epsilon=j['epsilon'],
            formal_plan_sha256=sha(ROOT/'FORMAL_PLAN.json'),evaluation='outer_test_final_round50',formal_table_eligible=True)
        out=ROOT/'formal'/f"{group}_n{j['n']}_eps{j['epsilon']:g}"/f"seed{j['seed']}"
        row=run_one(out,j['fields'],base,bundle,j['seed'],identity)
        print(task,j['seed'],row['accuracy'],flush=True)


def preflight(group):
    p=formal_plan();bundle,base=bundle_for(group,True);checked=0
    for n,_ in conditions()[:4]:
        for seed in range(30):
            cfg=dict(p['groups'][group]['fields'],n_clients=n,seeds=[seed]);part=suite_partition(base['dataset'],cfg,bundle.train_y,seed)
            ids=part.indices;assert min(map(len,ids))>0 and np.array_equal(np.sort(np.concatenate(ids)),np.arange(len(bundle.train_y)));checked+=1
    save(ROOT/'preflight'/f'{group}.json',dict(checked=checked,status='complete_no_training',dataset=base['dataset']))


def dispatch(mode):
    p=plan() if mode=='calibration' else formal_plan()
    if mode in ['calibration','preflight']:tasks=sorted(p['groups'],key=lambda g:(not g.startswith('brain'),g))
    else:tasks=list(dict.fromkeys(j['task'] for j in sorted(p['jobs'],key=lambda j:(not j['group'].startswith('brain'),-j['n'],j['group']))))
    if mode=='formal':
        for g in p['groups']:assert read(ROOT/'preflight'/f'{g}.json')['checked']==120
    def child(task):
        env=os.environ.copy();env.pop('PYTHONUTF8',None);env['PYTHONIOENCODING']='utf-8'
        env.update({k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
        log=ROOT/f'logs_{mode}'/f'{task}.log';log.parent.mkdir(exist_ok=True)
        with log.open('x',encoding='utf-8') as f:
            r=subprocess.run([sys.executable,__file__,mode,'--task',task],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        return dict(task=task,exit_code=r.returncode,log=str(log))
    outcomes=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(child,t) for t in tasks]):
            result=f.result();outcomes.append(result);print(len(outcomes),'/',len(tasks),result,flush=True)
    save(ROOT/f'DISPATCH_{mode}.json',outcomes)
    if any(r['exit_code'] for r in outcomes):raise RuntimeError('Failures preserved; no automatic retries')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run_calibration','calibration','freeze_formal','run_formal','formal','run_preflight','preflight']);ap.add_argument('--task');a=ap.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='freeze_formal':freeze_formal()
    elif a.action.startswith('run_'):dispatch(a.action[4:])
    elif a.action=='calibration':calibration(a.task)
    elif a.action=='formal':formal_task(a.task)
    else:preflight(a.task)
