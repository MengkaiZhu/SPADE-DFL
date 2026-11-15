"""Bounded original-release calibration; original vendor code stays read-only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'table34_n30_eps8_working'
PAIRED = ROOT.parent/'table34_paired_control16_working'
sys.path.insert(0, str(OLD))
from pilot_runtime import sha, array_hash, verify_modules, check_budget, save_npz
sys.path.insert(0, str(PAIRED))
from control_runtime import suite_partition


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)


def conditions():
    return [(10,8.),(30,8.),(50,8.),(100,8.),(30,.5),(30,2.),(30,32.)]


def variant(base, clip, exponent, n, epsilon, seed):
    if clip not in [.01,.03,.1] or exponent not in [0.,.51,1.02] or n not in [10,30,50,100] or epsilon not in [.5,2.,8.,32.] or seed not in range(30):
        raise ValueError('Outside frozen candidate/condition ranges')
    return dict(base, clip_norm=clip, clip_exponent=exponent, n_clients=n, target_epsilon=float(epsilon), seeds=[seed])


def select_candidate(rows):
    eligible=[]
    for candidate in sorted({r['candidate'] for r in rows}):
        selected=[r for r in rows if r['candidate']==candidate]
        if len(selected)!=5 or {r['seed'] for r in selected}!=set(range(5)):
            continue
        if any(r['status']!='completed' for r in selected):
            continue
        values=np.array([[r[k] for k in ['balanced','accuracy','loss']] for r in selected])
        if not np.isfinite(values).all():
            continue
        b,a,l=values.mean(axis=0)
        eligible.append(dict(candidate=candidate,balanced=float(b),accuracy=float(a),loss=float(l)))
    if not eligible:
        raise ValueError('No complete finite five-seed candidate')
    return sorted(eligible,key=lambda r:(-r['balanced'],-r['accuracy'],r['loss'],r['candidate']))[0]


def freeze():
    groups={}
    source_hashes={str(ROOT/'PROTOCOL_CN.md'):sha(ROOT/'PROTOCOL_CN.md'),
        str(ROOT/'study_runtime.py'):sha(ROOT/'study_runtime.py'),
        str(OLD/'pilot_runtime.py'):sha(OLD/'pilot_runtime.py'),
        str(PAIRED/'control_runtime.py'):sha(PAIRED/'control_runtime.py')}
    for name in read(OLD/'pilot_manifest.json')['config_files']:
        base=read(OLD/name)
        verify_modules(base)
        group=f"{base['dataset']}_fold{base['fold']}"
        prepared=OLD/'prepared'/base['job_id']
        pre=read(prepared/'preflight.json')
        assert sha(prepared/'data.npz')==pre['data_sha256']
        source_hashes[str(OLD/name)]=sha(OLD/name)
        source_hashes[str(prepared/'data.npz')]=sha(prepared/'data.npz')
        source_hashes[str(prepared/'bundle_metadata.json')]=sha(prepared/'bundle_metadata.json')
        calibration=Path(base['holdout']['source'].split('::')[0])
        assert sha(calibration)==base['holdout']['source_sha256']
        source_hashes[str(calibration)]=sha(calibration)
        groups[group]=dict(base_config=name,prepared=str(prepared),dataset=base['dataset'],fold=base['fold'])
    candidates=[dict(id=f'c{i}',clip=c,exponent=p) for i,(c,p) in enumerate(itertools.product([.01,.03,.1],[0.,.51,1.02]))]
    save(ROOT/'PLAN.json',dict(groups=groups,candidates=candidates,calibration_seeds=list(range(5)),
        conditions=conditions(),formal_seeds=list(range(30)),source_hashes=source_hashes,
        current_user_authorized_tuning_and_tables=True,calibration_runs=360,formal_runs_planned=1680,
        criterion='mean balanced validation accuracy; accuracy; lower training loss; candidate ID',
        privacy_formula_unchanged=True,trend_not_a_selection_criterion=True))
    print('FROZEN: 360 calibration runs; no training in freeze',flush=True)


def plan():
    p=read(ROOT/'PLAN.json')
    for path,digest in p['source_hashes'].items():
        assert sha(path)==digest, path
    return p


def predict(cfg, states, x):
    theta=states.mean(axis=0)
    if cfg['objective_model']=='paper_sigmoid':
        return np.where(x@theta>0,-1.,1.)
    if cfg['objective_model']=='lt_admm_nonconvex_logistic':
        return np.where(x@theta>=0,1.,-1.)
    assert cfg['objective_model']=='multiclass_softmax'
    return np.argmax(x@theta.reshape(cfg['feature_dimension'],cfg['n_classes']),axis=1)


def run_one(out, cfg_fields, base, bundle, seed, identity):
    # Per-run identity is written before training. Completed runs can only be verified/reused.
    if out.exists():
        row=read(out/'summary.json')
        assert row['identity']==identity and read(out/'config.json')==cfg_fields
        assert sha(out/'raw.npz')==row['raw_sha256'] and row['status']=='completed'
        return row
    from spade_dfl.config import ExperimentConfig
    from spade_dfl.graph import build_graph, derive_midpoint_beta
    from spade_dfl.trainer import run_experiment
    cfg=ExperimentConfig.from_dict(cfg_fields)
    effective=json.loads(json.dumps(cfg.to_dict()))
    assert all(effective[k]==v for k,v in cfg_fields.items())
    partition=suite_partition(base['dataset'],cfg_fields,bundle.train_y,seed)
    ids=partition.indices
    assert min(map(len,ids))>0
    assert np.array_equal(np.sort(np.concatenate(ids)),np.arange(len(bundle.train_y)))
    graph=build_graph(cfg.n_clients,topology=cfg.graph_topology,probability=cfg.graph_probability,seed=seed)
    out.mkdir(parents=True)
    save(out/'config.json',cfg_fields)
    save(out/'STARTED.json',dict(identity=identity,utc=datetime.now(timezone.utc).isoformat(),
        config_sha256=sha(out/'config.json'),code_sha256=sha(__file__),plan_sha256=sha(ROOT/'PLAN.json'),
        train_x_sha256=array_hash(bundle.train_x),train_y_sha256=array_hash(bundle.train_y),
        eval_x_sha256=array_hash(bundle.test_x),eval_y_sha256=array_hash(bundle.test_y),
        partition_hashes=[array_hash(i) for i in ids],adjacency_sha256=array_hash(graph.adjacency),
        original_module_map_sha256=base['algorithm_source']['all_python_module_hash_map_sha256'],
        python=sys.version,numpy=np.__version__))
    save_npz(out/'partition.npz',**{f'client_{i}':v for i,v in enumerate(ids)},adjacency=graph.adjacency)
    try:
        result=run_experiment(cfg,bundle,seed,partition=partition)
        save_npz(out/'raw.npz',states=result.states,**result.history)
        save(out/'metadata.json',result.metadata)
        assert np.isfinite(result.states).all() and all(np.isfinite(v).all() for v in result.history.values())
        assert len(result.history['accuracy'])==51
        eps=check_budget(result.metadata['clip_norm_schedule'],result.metadata['release_noise_std_schedule'],cfg.target_delta,cfg.target_epsilon)
        assert np.isclose(result.metadata['beta'],derive_midpoint_beta(graph,cfg.tau,cfg.rho))
        assert result.metadata['client_sample_counts']==[len(i) for i in ids]
        assert np.array_equal(result.history['accuracy'][1:],result.history['released_proxy_accuracy'])
        pred=predict(cfg_fields,result.states,bundle.test_x)
        accuracy=float(np.mean(pred==bundle.test_y))
        balanced=float(np.mean([np.mean(pred[bundle.test_y==v]==v) for v in np.unique(bundle.test_y)]))
        assert accuracy==result.history['accuracy'][-1]
        assert result.query_count==len(bundle.train_y)+cfg.n_clients*cfg.tau*cfg.batch_size*cfg.n_rounds
        row=dict(identity=identity,candidate=identity['candidate'],seed=seed,status='completed',
            accuracy=accuracy,balanced=balanced,loss=float(result.history['loss'][-1]),
            epsilon=eps,validation_correct=int(np.sum(pred==bundle.test_y)),eval_count=len(pred),
            queries=result.query_count,trainer_seconds=result.wall_time,
            mean_clip_fraction=float(np.mean(result.metadata['clip_fraction'])),
            raw_sha256=sha(out/'raw.npz'),metadata_sha256=sha(out/'metadata.json'),
            config_sha256=sha(out/'config.json'),path=str(out),end_to_end_privacy_certified=False)
        verify_modules(base)
        save(out/'summary.json',row)
        return row
    except Exception:
        save(out/'FAILED.json',dict(identity=identity,error=traceback.format_exc(),automatic_retry=False))
        raise


def group_calibration(group):
    p=plan();g=p['groups'][group];base=read(OLD/g['base_config'])
    package=verify_modules(base);sys.path.insert(0,str(package.parent))
    from spade_dfl.data import DatasetBundle,PCAState
    prepared=Path(g['prepared'])
    with np.load(prepared/'data.npz') as data:
        meta=read(prepared/'bundle_metadata.json');accepted={f.name for f in fields(DatasetBundle)}
        bundle=DatasetBundle(train_x=data['train_x'],train_y=data['train_y'],test_x=data['validation_x'],test_y=data['validation_y'],
            pca=PCAState(data['pca_mean'],data['pca_components'],data['pca_scale']),**{k:v for k,v in meta.items() if k in accepted})
    rows=[]
    for c,seed in itertools.product(p['candidates'],p['calibration_seeds']):
        cfg=variant(base['optimizer_config_fields'],c['clip'],c['exponent'],30,8.,seed)
        identity=dict(phase='calibration',group=group,candidate=c['id'],n_clients=30,epsilon=8.,seed=seed,
            plan_sha256=sha(ROOT/'PLAN.json'),evaluation='training_holdout')
        row=run_one(ROOT/'calibration'/group/c['id']/f'seed{seed}',cfg,base,bundle,seed,identity)
        rows.append(row)
        print(f"{group} {c['id']} seed{seed}: {100*row['accuracy']:.2f}%",flush=True)
    selected=select_candidate(rows)
    selected['fields']=next(c for c in p['candidates'] if c['id']==selected['candidate'])
    save(ROOT/'calibration'/group/'selection.json',dict(selected=selected,rows=rows,test_used=False,plan_sha256=sha(ROOT/'PLAN.json')))
    print(json.dumps(dict(group=group,selected=selected)),flush=True)


def dispatch():
    p=plan()
    def child(group):
        env=os.environ.copy();env.update({k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']});env['PYTHONUTF8']='1'
        log=ROOT/'logs'/f'{group}.log';log.parent.mkdir(exist_ok=True)
        with log.open('x',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,__file__,'group','--group',group],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{group} failed; preserve outputs; see {log}')
        return group
    with ThreadPoolExecutor(max_workers=4) as pool:
        for done in as_completed([pool.submit(child,g) for g in p['groups']]):
            print('COMPLETE '+done.result(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','calibrate','group']);parser.add_argument('--group')
    args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='calibrate':dispatch()
    else:group_calibration(args.group)
